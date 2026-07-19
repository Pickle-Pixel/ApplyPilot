#!/usr/bin/env python3
"""Verify and atomically execute a prepared fleet PostgreSQL rollback."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import stat
import tempfile
import time

import psycopg
from psycopg.rows import dict_row

from applypilot.fleet.pg_roles import authenticate_evidence_receipt, verify_evidence_receipt


_CONTAINMENT_MARKER_RELATION = "public.applypilot_rollback_commit_markers"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--rollback-sql", required=True, type=Path)
    parser.add_argument("--restore-hba", action="store_true")
    parser.add_argument("--partial-state-receipt", type=Path)
    return parser


def _hba_restore_supported(*, platform_name: str | None = None) -> bool:
    """HBA replacement requires descriptor-relative filesystem operations."""
    effective_platform = os.name if platform_name is None else platform_name
    return (
        effective_platform != "nt"
        and os.open in os.supports_dir_fd
        and os.replace in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
    )


def _read_bound_regular_file(path: Path, *, label: str) -> tuple[bytes, os.stat_result]:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise SystemExit(f"{label} path must be absolute, canonical, and contain no symlinks")
    parent = os.lstat(path.parent)
    before = os.lstat(path)
    if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode):
        raise SystemExit(f"{label} parent must be a real directory")
    if os.name != "nt" and stat.S_IMODE(parent.st_mode) & 0o022:
        raise SystemExit(f"{label} parent directory must not be group/world writable")
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise SystemExit(f"{label} must be a regular non-symlink file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise SystemExit(f"{label} changed while being opened")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    if (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino):
        raise SystemExit(f"{label} changed while being read")
    return payload, opened


def _verified_inputs(
    receipt_path: Path,
    rollback_path: Path,
    *,
    authentication_key: bytes,
    expected_key_id: str,
) -> tuple[dict, str]:
    receipt_bytes, _receipt_stat = _read_bound_regular_file(receipt_path, label="receipt")
    receipt = json.loads(receipt_bytes.decode("utf-8-sig"))
    try:
        verify_evidence_receipt(
            receipt,
            authentication_key=authentication_key,
            expected_key_id=expected_key_id,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from None
    rollback_mode = receipt.get("rollback_mode") or receipt.get("inventory", {}).get("rollback_mode")
    if rollback_mode not in {"topology_exact", "forward_v7_containment"}:
        raise SystemExit("receipt does not identify a supported rollback mode")
    signed_path = Path(receipt.get("rollback_sql_path", ""))
    if not signed_path.is_absolute() or signed_path.resolve(strict=True) != rollback_path:
        raise SystemExit("rollback SQL path does not match the authenticated receipt")
    rollback_bytes, _rollback_stat = _read_bound_regular_file(rollback_path, label="rollback SQL")
    expected = receipt.get("rollback_sql_sha256", "")
    actual = hashlib.sha256(rollback_bytes).hexdigest()
    if not expected or not hmac.compare_digest(expected, actual):
        raise SystemExit("rollback SQL SHA-256 does not match the durable receipt")
    return receipt, rollback_bytes.decode("utf-8")


def _break_glass_roles(receipt: dict) -> set[str]:
    inventory = receipt.get("inventory", {})
    topology = receipt.get("topology", {}) or inventory.get("topology", {})
    return set(
        inventory.get("infrastructure_superuser_roles", ()) or topology.get("infrastructure_superuser_roles", ())
    )


def _protected_containment_roles(receipt: dict) -> set[str]:
    inventory = receipt.get("inventory", {})
    topology = receipt.get("topology", {}) or inventory.get("topology", {})
    protected = _break_glass_roles(receipt)
    controller = topology.get("controller_role")
    if controller:
        protected.add(controller)
    return protected


def _expected_target_identity(receipt: dict) -> tuple[str | None, str | None, int | None]:
    inventory = receipt.get("inventory", {})
    expected_database = receipt.get("database_name") or inventory.get("database_name")
    expected_system_identifier = receipt.get("system_identifier") or inventory.get("system_identifier")
    expected_database_oid = receipt.get("database_oid") or inventory.get("database_oid")
    if receipt.get("rollback_mode") == "forward_v7_containment" and (
        not expected_database or not str(expected_system_identifier or "").isdigit()
        or type(expected_database_oid) is not int or expected_database_oid <= 0
    ):
        raise SystemExit("forward V7 rollback receipt is missing exact target identity pins")
    return (
        expected_database,
        str(expected_system_identifier) if expected_system_identifier is not None else None,
        expected_database_oid if type(expected_database_oid) is int else None,
    )


def _pin_rollback_search_path(conn) -> None:
    observed = conn.execute(
        "SELECT pg_catalog.set_config('search_path','pg_catalog,pg_temp',false) AS search_path"
    ).fetchone()["search_path"]
    if observed != "pg_catalog,pg_temp":
        raise RuntimeError("rollback session search_path could not be pinned")


def _validated_target_identity(conn, receipt: dict) -> dict[str, str]:
    identity = conn.execute(
        "SELECT session_user,current_user,role.rolsuper,"
        "pg_catalog.current_database() AS database_name,"
        "(pg_catalog.pg_control_system()).system_identifier::pg_catalog.text AS system_identifier,"
        "(SELECT database.oid FROM pg_catalog.pg_database database "
        "WHERE database.datname=pg_catalog.current_database()) AS database_oid "
        "FROM pg_catalog.pg_roles role WHERE role.rolname=session_user"
    ).fetchone()
    expected_database, expected_system_identifier, expected_database_oid = (
        _expected_target_identity(receipt)
    )
    if (
        identity["session_user"] != identity["current_user"]
        or not identity["rolsuper"]
        or identity["session_user"] not in _break_glass_roles(receipt)
    ):
        raise SystemExit("rollback session must be an explicitly receipted break-glass superuser")
    if expected_database is not None and identity["database_name"] != expected_database:
        raise SystemExit("rollback database does not match the authenticated target identity")
    if (
        expected_system_identifier is not None
        and identity["system_identifier"] != expected_system_identifier
    ):
        raise SystemExit("rollback cluster does not match the authenticated target identity")
    if expected_database_oid is not None and identity["database_oid"] != expected_database_oid:
        raise SystemExit("rollback database OID does not match the authenticated target identity")
    return {
        "database_name": identity["database_name"],
        "system_identifier": identity["system_identifier"],
        "database_oid": int(identity["database_oid"]),
        "session_user": identity["session_user"],
    }


def _containment_marker(receipt: dict, rollback_sql: str, target_identity: dict[str, str]) -> dict:
    rollback_sql_sha256 = hashlib.sha256(rollback_sql.encode("utf-8")).hexdigest()
    source_receipt_digest = receipt.get("receipt_authentication", {}).get("digest", "")
    marker_binding = {
        "database_name": target_identity["database_name"],
        "system_identifier": target_identity["system_identifier"],
        "database_oid": target_identity["database_oid"],
        "rollback_sql_sha256": rollback_sql_sha256,
        "source_receipt_digest": source_receipt_digest,
    }
    marker_id = hashlib.sha256(
        json.dumps(marker_binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "marker_id": marker_id,
        **marker_binding,
        "executed_by": target_identity["session_user"],
        "affected_role_names": [],
        "affected_role_oids": [],
        "completed_phases": [],
    }


def _execute_database_containment_sql(conn, rollback_sql: str) -> None:
    conn.execute(rollback_sql)


def _write_database_containment_marker(
    conn,
    *,
    marker: dict,
    phase: str = "phase1",
    approved_break_glass_roles: set[str] | None = None,
) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS public.applypilot_rollback_commit_markers("
        "marker_id text PRIMARY KEY,database_name text NOT NULL,system_identifier text NOT NULL,"
        "database_oid oid NOT NULL,"
        "executed_by text NOT NULL,affected_role_names text[] NOT NULL,affected_role_oids oid[] NOT NULL,"
        "completed_phases text[] NOT NULL,"
        "rollback_sql_sha256 text NOT NULL,source_receipt_digest text NOT NULL,"
        "committed_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp())"
    )
    relation = conn.execute(
        "SELECT relation.relkind,relation.relpersistence,relation.relrowsecurity,"
        "relation.relforcerowsecurity,owner.rolname AS owner_name,"
        "(SELECT count(*) FROM pg_catalog.pg_trigger trigger WHERE trigger.tgrelid=relation.oid "
        "AND NOT trigger.tgisinternal) AS trigger_count,"
        "(SELECT count(*) FROM pg_catalog.pg_policy policy "
        "WHERE policy.polrelid=relation.oid) AS policy_count,"
        "(SELECT count(*) FROM pg_catalog.aclexplode(COALESCE(relation.relacl,"
        "pg_catalog.acldefault('r',relation.relowner))) acl "
        "WHERE acl.grantee<>relation.relowner) AS nonowner_acl_count "
        "FROM pg_catalog.pg_class relation JOIN pg_catalog.pg_namespace namespace "
        "ON namespace.oid=relation.relnamespace "
        "JOIN pg_catalog.pg_roles owner ON owner.oid=relation.relowner "
        "WHERE namespace.nspname='public' AND relation.relname='applypilot_rollback_commit_markers'"
    ).fetchone()
    approved_owners = approved_break_glass_roles or {marker["executed_by"]}
    expected_relation = {
        "relkind": "r",
        "relpersistence": "p",
        "relrowsecurity": False,
        "relforcerowsecurity": False,
        "trigger_count": 0,
        "policy_count": 0,
        "nonowner_acl_count": 0,
    }
    if (
        relation is None
        or relation.get("owner_name") not in approved_owners
        or any(relation.get(key) != value for key, value in expected_relation.items())
    ):
        raise RuntimeError(f"database containment marker relation is unsafe: {relation!r}")
    columns = conn.execute(
        "SELECT attribute.attname,pg_catalog.format_type(attribute.atttypid,attribute.atttypmod) "
        "AS data_type,attribute.attnotnull FROM pg_catalog.pg_attribute attribute "
        "WHERE attribute.attrelid='public.applypilot_rollback_commit_markers'::pg_catalog.regclass "
        "AND attribute.attnum>0 AND NOT attribute.attisdropped ORDER BY attribute.attnum"
    ).fetchall()
    expected_columns = [
        {"attname": "marker_id", "data_type": "text", "attnotnull": True},
        {"attname": "database_name", "data_type": "text", "attnotnull": True},
        {"attname": "system_identifier", "data_type": "text", "attnotnull": True},
        {"attname": "database_oid", "data_type": "oid", "attnotnull": True},
        {"attname": "executed_by", "data_type": "text", "attnotnull": True},
        {"attname": "affected_role_names", "data_type": "text[]", "attnotnull": True},
        {"attname": "affected_role_oids", "data_type": "oid[]", "attnotnull": True},
        {"attname": "completed_phases", "data_type": "text[]", "attnotnull": True},
        {"attname": "rollback_sql_sha256", "data_type": "text", "attnotnull": True},
        {"attname": "source_receipt_digest", "data_type": "text", "attnotnull": True},
        {"attname": "committed_at", "data_type": "timestamp with time zone", "attnotnull": True},
    ]
    if columns != expected_columns:
        raise RuntimeError(f"database containment marker column contract mismatch: {columns!r}")
    conn.execute(
        "INSERT INTO public.applypilot_rollback_commit_markers("
        "marker_id,database_name,system_identifier,database_oid,executed_by,affected_role_names,affected_role_oids,completed_phases,"
        "rollback_sql_sha256,source_receipt_digest) "
        "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(marker_id) DO UPDATE SET "
        "affected_role_names=COALESCE((SELECT pg_catalog.array_agg(DISTINCT role_name ORDER BY role_name) "
        "FROM pg_catalog.unnest(public.applypilot_rollback_commit_markers.affected_role_names "
        "|| EXCLUDED.affected_role_names) role_name),ARRAY[]::pg_catalog.text[]),"
        "affected_role_oids=COALESCE((SELECT pg_catalog.array_agg(DISTINCT role_oid ORDER BY role_oid) "
        "FROM pg_catalog.unnest(public.applypilot_rollback_commit_markers.affected_role_oids "
        "|| EXCLUDED.affected_role_oids) role_oid),ARRAY[]::pg_catalog.oid[]),"
        "completed_phases=COALESCE((SELECT pg_catalog.array_agg(DISTINCT completed_phase "
        "ORDER BY completed_phase) FROM pg_catalog.unnest("
        "public.applypilot_rollback_commit_markers.completed_phases "
        "|| EXCLUDED.completed_phases) completed_phase),ARRAY[]::pg_catalog.text[])",
        (
            marker["marker_id"],
            marker["database_name"],
            marker["system_identifier"],
            marker["database_oid"],
            marker["executed_by"],
            marker["affected_role_names"],
            marker["affected_role_oids"],
            [phase],
            marker["rollback_sql_sha256"],
            marker["source_receipt_digest"],
        ),
    )


def _commit_database_containment_transaction(conn) -> None:
    conn.commit()


class ContainmentCommitOutcomeUnknown(RuntimeError):
    """The server may have committed containment although COMMIT acknowledgement failed."""

    def __init__(self, commit_error: BaseException, rollback_error: BaseException | None):
        super().__init__("database containment COMMIT outcome is unknown")
        self.commit_error = commit_error
        self.rollback_error = rollback_error


class ContainmentCommitUnreconciled(RuntimeError):
    """Commit acknowledgement was lost and independent marker observation failed."""

    def __init__(self, message: str, *, evidence_errors: BaseExceptionGroup):
        super().__init__(message)
        self.evidence_errors = evidence_errors


class ContainmentCommitReconciledConnectionLost(RuntimeError):
    """Containment committed, but the executor connection cannot safely continue."""

    def __init__(self, marker: dict, *, evidence_errors: BaseExceptionGroup):
        super().__init__("database containment committed but its executor connection was lost")
        self.marker = marker
        self.evidence_errors = evidence_errors


def _affected_authority_roles(conn, *, protected_roles: set[str]) -> tuple[list[str], list[int]]:
    if conn.execute(
        "SELECT pg_catalog.to_regclass('pg_temp.applypilot_authority_containment_roles') "
        "AS relation"
    ).fetchone()["relation"] is None:
        return [], []
    rows = conn.execute(
        "SELECT role.oid,role.rolname FROM pg_temp.applypilot_authority_containment_roles contained "
        "JOIN pg_catalog.pg_roles role ON role.oid=contained.role_oid "
        "WHERE NOT role.rolname=ANY(%s) ORDER BY role.rolname",
        (sorted(protected_roles),),
    ).fetchall()
    return [row["rolname"] for row in rows], [int(row["oid"]) for row in rows]


def _assert_database_containment_marker(
    conn,
    *,
    marker: dict,
    expected_phase: str | None = None,
) -> dict:
    observed = conn.execute(
        "SELECT marker_id,database_name,system_identifier,database_oid,executed_by,affected_role_names,affected_role_oids,completed_phases,rollback_sql_sha256,"
        "source_receipt_digest,committed_at FROM public.applypilot_rollback_commit_markers "
        "WHERE marker_id=%s",
        (marker["marker_id"],),
    ).fetchone()
    conn.commit()
    stable_keys = (
        "marker_id",
        "database_name",
        "system_identifier",
        "database_oid",
        "rollback_sql_sha256",
        "source_receipt_digest",
    )
    if (
        observed is None
        or any(observed[key] != marker[key] for key in stable_keys)
        or (expected_phase is not None and expected_phase not in observed["completed_phases"])
    ):
        raise RuntimeError("database containment committed but its durable marker binding is invalid")
    committed_at = observed["committed_at"]
    return {
        **marker,
        "executed_by": observed["executed_by"],
        "affected_role_names": list(observed["affected_role_names"]),
        "affected_role_oids": [int(oid) for oid in observed["affected_role_oids"]],
        "completed_phases": list(observed["completed_phases"]),
        "relation": _CONTAINMENT_MARKER_RELATION,
        "visible": True,
        "committed_at": committed_at.isoformat() if hasattr(committed_at, "isoformat") else str(committed_at),
    }


def _commit_database_containment(
    conn,
    rollback_sql: str,
    *,
    marker: dict,
    protected_roles: set[str] | None = None,
    approved_break_glass_roles: set[str] | None = None,
    phase: str = "phase1",
) -> dict:
    """Commit rollback SQL and its idempotent database marker atomically."""

    try:
        _execute_database_containment_sql(conn, rollback_sql)
        affected_names, affected_oids = _affected_authority_roles(
            conn, protected_roles=protected_roles or set()
        )
        marker = {
            **marker,
            "affected_role_names": affected_names,
            "affected_role_oids": affected_oids,
        }
        _write_database_containment_marker(
            conn,
            marker=marker,
            phase=phase,
            approved_break_glass_roles=approved_break_glass_roles,
        )
    except BaseException as operation_error:
        try:
            conn.rollback()
        except BaseException as rollback_error:
            raise RuntimeError(
                "database containment failed before commit and rollback reporting also failed"
            ) from BaseExceptionGroup(
                "containment operation and rollback failures",
                [operation_error, rollback_error],
            )
        raise
    try:
        _commit_database_containment_transaction(conn)
    except BaseException as commit_error:
        rollback_error = None
        try:
            conn.rollback()
        except BaseException as error:
            rollback_error = error
        raise ContainmentCommitOutcomeUnknown(commit_error, rollback_error) from commit_error
    return _assert_database_containment_marker(
        conn,
        marker=marker,
        expected_phase=phase,
    )


def _terminate_affected_sessions(
    conn,
    *,
    affected_role_names: list[str],
    affected_role_oids: list[int] | None = None,
    protected_roles: set[str],
) -> int:
    targets = sorted(set(affected_role_names) - protected_roles)
    target_oids = sorted(set(affected_role_oids or []))
    if not targets and not target_oids:
        return 0
    signaled = conn.execute(
        "WITH targets(pid) AS MATERIALIZED ("
        "SELECT activity.pid FROM pg_catalog.pg_stat_activity activity "
        "WHERE activity.pid<>pg_catalog.pg_backend_pid() "
        "AND (activity.usename=ANY(%s) OR activity.usesysid=ANY(%s))) "
        "SELECT count(*) AS signaled FROM targets "
        "WHERE pg_catalog.pg_terminate_backend(targets.pid)",
        (targets, target_oids),
    ).fetchone()
    conn.commit()
    deadline = time.monotonic() + 5.0
    while True:
        remaining = conn.execute(
            "SELECT activity.pid,activity.usename FROM pg_catalog.pg_stat_activity activity "
            "WHERE activity.pid<>pg_catalog.pg_backend_pid() "
            "AND (activity.usename=ANY(%s) OR activity.usesysid=ANY(%s)) "
            "ORDER BY activity.pid",
            (targets, target_oids),
        ).fetchall()
        conn.commit()
        if not remaining or time.monotonic() >= deadline:
            break
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    if remaining:
        raise RuntimeError(f"affected authority sessions survived termination: {remaining!r}")
    return int(signaled["signaled"])


def _commit_with_authenticated_reconciliation(
    conn,
    rollback_sql: str,
    *,
    marker: dict,
    protected_roles: set[str],
    admin_dsn: str,
    receipt: dict,
    phase: str = "phase1",
) -> dict:
    try:
        return _commit_database_containment(
            conn,
            rollback_sql,
            marker=marker,
            protected_roles=protected_roles,
            approved_break_glass_roles=_break_glass_roles(receipt),
            phase=phase,
        )
    except ContainmentCommitOutcomeUnknown as unknown:
        try:
            with psycopg.connect(admin_dsn, row_factory=dict_row) as observer:
                _pin_rollback_search_path(observer)
                _validated_target_identity(observer, receipt)
                observer.commit()
                observed_marker = _assert_database_containment_marker(
                    observer,
                    marker=marker,
                    expected_phase=phase,
                )
                errors: list[BaseException] = [unknown.commit_error]
                if unknown.rollback_error is not None:
                    errors.append(unknown.rollback_error)
                evidence_errors = BaseExceptionGroup(
                    "reconciled containment commit acknowledgement loss", errors
                )
                raise ContainmentCommitReconciledConnectionLost(
                    observed_marker,
                    evidence_errors=evidence_errors,
                ) from evidence_errors
        except ContainmentCommitReconciledConnectionLost:
            raise
        except BaseException as observation_error:
            errors: list[BaseException] = [unknown.commit_error]
            if unknown.rollback_error is not None:
                errors.append(unknown.rollback_error)
            errors.append(observation_error)
            evidence_errors = BaseExceptionGroup("containment commit outcome evidence", errors)
            raise ContainmentCommitUnreconciled(
                "database containment COMMIT acknowledgement was lost and the authenticated "
                "durable marker could not be observed",
                evidence_errors=evidence_errors,
            ) from evidence_errors


def _assert_candidate_artifact_containment(
    conn,
    *,
    break_glass_roles: set[str],
    affected_role_names: list[str] | None = None,
    affected_role_oids: list[int] | None = None,
) -> dict[str, int]:
    function_routes = conn.execute(
        "SELECT count(*) FILTER (WHERE function.proname LIKE 'brain_publish_v%_candidate') "
        "AS candidate_execute_routes,"
        "count(*) FILTER (WHERE function.proname LIKE "
        "'brain_register_authoritative_artifact_manifest%') AS artifact_execute_routes "
        "FROM pg_catalog.pg_proc function JOIN pg_catalog.pg_namespace namespace "
        "ON namespace.oid=function.pronamespace "
        "CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(function.proacl,"
        "pg_catalog.acldefault('f',function.proowner))) acl "
        "WHERE namespace.nspname='public' AND acl.privilege_type='EXECUTE' "
        "AND acl.grantee<>function.proowner"
    ).fetchone()
    membership_routes = conn.execute(
        "WITH RECURSIVE roots(role_oid) AS ("
        "SELECT oid FROM pg_catalog.pg_roles WHERE rolname LIKE 'brain_candidate_%' "
        "OR rolname LIKE 'brain_artifact_authority_%'), closure(role_oid) AS ("
        "SELECT role_oid FROM roots UNION "
        "SELECT membership.member FROM closure "
        "JOIN pg_catalog.pg_auth_members membership ON membership.roleid=closure.role_oid "
        ") "
        "SELECT count(*) AS set_role_routes FROM pg_catalog.pg_auth_members membership "
        "WHERE membership.roleid IN (SELECT role_oid FROM closure)"
    ).fetchone()
    protected_roles = sorted(break_glass_roles)
    affected_roles = sorted(set(affected_role_names or []) - break_glass_roles)
    affected_oids = sorted(set(affected_role_oids or []))
    authority = conn.execute(
        "SELECT "
        "count(*) FILTER (WHERE role.rolcanlogin) AS affected_logins,"
        "count(*) FILTER (WHERE activity.pid IS NOT NULL) AS affected_sessions "
        "FROM pg_catalog.pg_roles role LEFT JOIN pg_catalog.pg_stat_activity activity "
        "ON (activity.usename=role.rolname OR activity.usesysid=ANY(%s)) "
        "AND activity.pid<>pg_catalog.pg_backend_pid() "
        "WHERE (role.rolname=ANY(%s) OR role.oid=ANY(%s)) AND NOT role.rolname=ANY(%s)",
        (affected_oids, affected_roles, affected_oids, protected_roles),
    ).fetchone()
    residual = conn.execute(
        "SELECT "
        "count(*) FILTER (WHERE pg_catalog.has_database_privilege(role.rolname,"
        "pg_catalog.current_database(),'CONNECT')) AS affected_connect_routes,"
        "count(*) FILTER (WHERE database.datdba=role.oid) AS affected_database_owners "
        "FROM pg_catalog.pg_roles role "
        "LEFT JOIN pg_catalog.pg_database database "
        "ON database.datname=pg_catalog.current_database() "
        "WHERE role.rolname=ANY(%s) AND NOT role.rolname=ANY(%s)",
        (affected_roles, protected_roles),
    ).fetchone()
    object_owners = conn.execute(
        "SELECT "
        "(SELECT count(*) FROM pg_catalog.pg_namespace namespace JOIN pg_catalog.pg_roles role "
        "ON role.oid=namespace.nspowner WHERE (role.rolname=ANY(%s) OR role.oid=ANY(%s)) AND "
        "(namespace.nspname IN ('public','brain_archive') OR left(namespace.nspname,6)='brain_')) "
        "AS affected_schema_owners,"
        "(SELECT count(*) FROM pg_catalog.pg_proc executable JOIN pg_catalog.pg_namespace namespace "
        "ON namespace.oid=executable.pronamespace JOIN pg_catalog.pg_roles role "
        "ON role.oid=executable.proowner WHERE (role.rolname=ANY(%s) OR role.oid=ANY(%s)) AND "
        "((namespace.nspname='public' AND left(executable.proname,6)='brain_') "
        "OR namespace.nspname='brain_archive')) AS affected_executable_owners,"
        "(SELECT count(*) FROM pg_catalog.pg_class relation JOIN pg_catalog.pg_namespace namespace "
        "ON namespace.oid=relation.relnamespace JOIN pg_catalog.pg_roles role "
        "ON role.oid=relation.relowner WHERE (role.rolname=ANY(%s) OR role.oid=ANY(%s)) AND "
        "((namespace.nspname='public' AND left(relation.relname,6)='brain_') "
        "OR namespace.nspname='brain_archive')) AS affected_relation_owners,"
        "(SELECT count(*) FROM pg_catalog.pg_type type JOIN pg_catalog.pg_namespace namespace "
        "ON namespace.oid=type.typnamespace JOIN pg_catalog.pg_roles role ON role.oid=type.typowner "
        "WHERE (role.rolname=ANY(%s) OR role.oid=ANY(%s)) AND type.typrelid=0 "
        "AND type.typelem=0 AND "
        "((namespace.nspname='public' AND left(type.typname,6)='brain_') "
        "OR namespace.nspname='brain_archive')) AS affected_type_owners",
        (
            affected_roles,
            affected_oids,
            affected_roles,
            affected_oids,
            affected_roles,
            affected_oids,
            affected_roles,
            affected_oids,
        ),
    ).fetchone()
    if affected_roles or affected_oids:
        broad_routes = conn.execute(
        "SELECT "
        "(SELECT count(*) FROM pg_catalog.pg_proc executable "
        "JOIN pg_catalog.pg_namespace namespace ON namespace.oid=executable.pronamespace "
        "CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(executable.proacl,"
        "pg_catalog.acldefault('f',executable.proowner))) acl WHERE "
        "((namespace.nspname='public' AND left(executable.proname,6)='brain_') "
        "OR namespace.nspname='brain_archive') AND acl.grantee<>executable.proowner) "
        "AS executable_acl_routes,"
        "(SELECT count(*) FROM pg_catalog.pg_class relation "
        "JOIN pg_catalog.pg_namespace namespace ON namespace.oid=relation.relnamespace "
        "CROSS JOIN LATERAL pg_catalog.aclexplode(relation.relacl) acl WHERE "
        "((namespace.nspname='public' AND left(relation.relname,6)='brain_') "
        "OR namespace.nspname='brain_archive') AND acl.grantee<>relation.relowner) "
        "AS relation_acl_routes,"
        "(SELECT count(*) FROM pg_catalog.pg_attribute attribute "
        "JOIN pg_catalog.pg_class relation ON relation.oid=attribute.attrelid "
        "JOIN pg_catalog.pg_namespace namespace ON namespace.oid=relation.relnamespace "
        "CROSS JOIN LATERAL pg_catalog.aclexplode(attribute.attacl) acl WHERE "
        "((namespace.nspname='public' AND left(relation.relname,6)='brain_') "
        "OR namespace.nspname='brain_archive') AND acl.grantee<>relation.relowner) "
        "AS column_acl_routes,"
        "(SELECT count(*) FROM pg_catalog.pg_type type "
        "JOIN pg_catalog.pg_namespace namespace ON namespace.oid=type.typnamespace "
        "CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(type.typacl,"
        "pg_catalog.acldefault('T',type.typowner))) acl WHERE "
        "((namespace.nspname='public' AND left(type.typname,6)='brain_') "
        "OR namespace.nspname='brain_archive') AND type.typelem=0 "
        "AND acl.grantee<>type.typowner) "
        "AS type_acl_routes,"
        "(SELECT count(*) FROM pg_catalog.pg_namespace namespace "
        "CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(namespace.nspacl,"
        "pg_catalog.acldefault('n',namespace.nspowner))) acl WHERE "
        "(namespace.nspname IN ('public','brain_archive') OR left(namespace.nspname,6)='brain_') "
        "AND acl.grantee<>namespace.nspowner) AS schema_acl_routes,"
        "(SELECT count(*) FROM pg_catalog.pg_default_acl defaults "
        "CROSS JOIN LATERAL pg_catalog.aclexplode(defaults.defaclacl) acl WHERE "
        "(defaults.defaclrole=ANY(%s) OR acl.grantee=ANY(%s)) "
        "AND acl.grantee<>defaults.defaclrole) AS default_acl_routes,"
        "(SELECT count(*) FROM pg_catalog.pg_auth_members membership WHERE "
        "membership.roleid=ANY(%s) OR membership.member=ANY(%s)) AS affected_membership_routes",
            (affected_oids, affected_oids, affected_oids, affected_oids),
        ).fetchone()
    else:
        broad_routes = {
            "executable_acl_routes": 0,
            "relation_acl_routes": 0,
            "column_acl_routes": 0,
            "type_acl_routes": 0,
            "schema_acl_routes": 0,
            "default_acl_routes": 0,
            "affected_membership_routes": 0,
        }
    conn.commit()
    result = {
        "candidate_execute_routes": int(function_routes["candidate_execute_routes"]),
        "artifact_execute_routes": int(function_routes["artifact_execute_routes"]),
        "set_role_routes": int(membership_routes["set_role_routes"]),
        "affected_logins": int(authority["affected_logins"]),
        "affected_sessions": int(authority["affected_sessions"]),
        "affected_connect_routes": int(residual["affected_connect_routes"]),
        "affected_database_owners": int(residual["affected_database_owners"]),
        "affected_schema_owners": int(object_owners["affected_schema_owners"]),
        "affected_executable_owners": int(object_owners["affected_executable_owners"]),
        "affected_relation_owners": int(object_owners["affected_relation_owners"]),
        "affected_type_owners": int(object_owners["affected_type_owners"]),
        **{key: int(value) for key, value in broad_routes.items()},
    }
    if any(result.values()):
        raise RuntimeError(f"database containment postcondition failed: {result!r}")
    return result


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _partial_state_parent_identity(parent_path: Path) -> tuple[str, int, int]:
    parent = os.lstat(parent_path)
    return str(parent_path), int(parent.st_dev), int(parent.st_ino)


def _validate_partial_state_parent(path: Path) -> tuple[str, int, int]:
    if not path.is_absolute() or path.parent.resolve(strict=True) != path.parent:
        raise RuntimeError("partial-state receipt path must have an absolute canonical parent")
    parent = os.lstat(path.parent)
    if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode):
        raise RuntimeError("partial-state receipt parent must be a real directory")
    if os.name != "nt" and stat.S_IMODE(parent.st_mode) & 0o022:
        raise RuntimeError("partial-state receipt parent must not be group/world writable")
    return str(path.parent), int(parent.st_dev), int(parent.st_ino)


def _write_exclusive_durable(
    path: Path,
    payload: bytes,
    *,
    expected_parent_identity: tuple[str, int, int],
) -> None:
    if _partial_state_parent_identity(path.parent) != expected_parent_identity:
        raise RuntimeError("partial-state receipt parent directory changed after preflight")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    parent_descriptor = -1
    created = False
    try:
        if os.open in os.supports_dir_fd:
            parent_descriptor = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            opened_parent = os.fstat(parent_descriptor)
            if (str(path.parent), int(opened_parent.st_dev), int(opened_parent.st_ino)) != (
                expected_parent_identity
            ):
                raise RuntimeError("partial-state receipt parent directory changed after preflight")
            descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_descriptor)
        else:
            if _partial_state_parent_identity(path.parent) != expected_parent_identity:
                raise RuntimeError("partial-state receipt parent directory changed after preflight")
            descriptor = os.open(path, flags, 0o600)
        created = True
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if parent_descriptor >= 0:
            os.fsync(parent_descriptor)
        else:
            _fsync_parent(path)
    except BaseException:
        if created:
            try:
                if parent_descriptor >= 0 and os.unlink in os.supports_dir_fd:
                    os.unlink(path.name, dir_fd=parent_descriptor)
                else:
                    os.unlink(path)
            except FileNotFoundError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def _partial_state_destination(
    requested: Path,
    *,
    authentication_key: bytes,
    expected_key_id: str,
) -> tuple[Path, dict | None]:
    candidate = requested.resolve()
    predecessor: dict | None = None
    sequence = 1
    while candidate.exists():
        payload, _metadata = _read_bound_regular_file(candidate, label="partial-state predecessor")
        decoded = json.loads(payload.decode("utf-8-sig"))
        try:
            verify_evidence_receipt(
                decoded,
                authentication_key=authentication_key,
                expected_key_id=expected_key_id,
            )
        except ValueError as error:
            raise RuntimeError("existing partial-state predecessor is not authenticated") from error
        predecessor = {
            "path": str(candidate),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "status": decoded.get("status"),
            "marker_id": decoded.get("database_containment", {}).get("marker_id"),
            "receipt_authentication": decoded.get("receipt_authentication"),
            "target_identity": decoded.get("target_identity"),
            "source_receipt_authentication": decoded.get("evidence", {}).get(
                "source_receipt_authentication"
            ),
        }
        sequence += 1
        candidate = requested.with_name(f"{requested.name}.{sequence}").resolve()
    return candidate, predecessor


def _source_receipt_evidence(source_receipt_path: Path, source_receipt: dict) -> dict:
    source_bytes, _source_metadata = _read_bound_regular_file(
        source_receipt_path, label="source rollback receipt"
    )
    if json.loads(source_bytes.decode("utf-8-sig")) != source_receipt:
        raise RuntimeError("source rollback receipt changed after authentication")
    return {
        "source_receipt_path": str(source_receipt_path),
        "source_receipt_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_receipt_authentication": source_receipt.get("receipt_authentication"),
    }


def _observe_hba_state(receipt: dict, *, validated=None) -> dict:
    metadata = receipt.get("hba_restore", {})
    path = Path(metadata.get("live_hba_path", ""))
    try:
        payload, observed = _read_bound_regular_file(path, label="live HBA after failed restoration")
        return {
            "live_hba_path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
            "mode": stat.S_IMODE(observed.st_mode),
            "owner": {"uid": observed.st_uid, "gid": observed.st_gid},
            "preflight_target_sha256": (
                hashlib.sha256(validated[1]).hexdigest()
                if validated is not None and len(validated) > 1 and isinstance(validated[1], bytes)
                else metadata.get("expected_target_sha256")
            ),
        }
    except BaseException as error:
        return {
            "live_hba_path": str(path),
            "observation_error": {"type": type(error).__name__, "message": str(error)},
        }


def _emit_partial_state_receipt(
    requested_path: Path,
    *,
    source_receipt: dict,
    source_receipt_evidence: dict,
    rollback_sql_path: Path,
    rollback_sql: str,
    target_identity: dict[str, str],
    database_containment: dict,
    containment_postcondition: dict[str, int],
    hba_observed: dict,
    failure: BaseException,
    authentication_key: bytes,
    authentication_key_id: str,
    partial_parent_identity: tuple[str, int, int],
) -> Path:
    destination, predecessor = _partial_state_destination(
        requested_path,
        authentication_key=authentication_key,
        expected_key_id=authentication_key_id,
    )
    _require_matching_partial_state_predecessor(
        predecessor,
        marker_id=database_containment["marker_id"],
        target_identity=target_identity,
        source_receipt_authentication=source_receipt.get("receipt_authentication"),
    )
    partial = {
        "format": "applypilot-database-containment-hba-partial-state-v1",
        "status": "database_containment_committed_hba_restore_failed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "target_identity": target_identity,
        "database_containment": database_containment,
        "containment_postcondition": containment_postcondition,
        "rollback_sql": {
            "path": str(rollback_sql_path),
            "sha256": hashlib.sha256(rollback_sql.encode("utf-8")).hexdigest(),
        },
        "hba": {
            "expected": source_receipt.get("hba_restore"),
            "observed": hba_observed,
        },
        "failure": {"type": type(failure).__name__, "message": str(failure)},
        "evidence": {
            **source_receipt_evidence,
            "predecessor_partial_state_receipt": predecessor,
        },
    }
    authenticated = authenticate_evidence_receipt(
        partial,
        authentication_key=authentication_key,
        authentication_key_id=authentication_key_id,
    )
    _write_exclusive_durable(
        destination,
        (json.dumps(authenticated, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        expected_parent_identity=partial_parent_identity,
    )
    return destination


def _emit_commit_unknown_receipt(
    requested_path: Path,
    *,
    source_receipt: dict,
    source_receipt_evidence: dict,
    rollback_sql_path: Path,
    rollback_sql: str,
    target_identity: dict,
    marker: dict,
    failure: ContainmentCommitUnreconciled,
    authentication_key: bytes,
    authentication_key_id: str,
    partial_parent_identity: tuple[str, int, int],
) -> Path:
    destination, predecessor = _partial_state_destination(
        requested_path,
        authentication_key=authentication_key,
        expected_key_id=authentication_key_id,
    )
    _require_matching_partial_state_predecessor(
        predecessor,
        marker_id=marker["marker_id"],
        target_identity=target_identity,
        source_receipt_authentication=source_receipt.get("receipt_authentication"),
    )
    errors = [
        {"type": type(error).__name__, "message": str(error)}
        for error in failure.evidence_errors.exceptions
    ]
    partial = {
        "format": "applypilot-database-containment-commit-unknown-v1",
        "status": "database_containment_commit_outcome_unknown",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "target_identity": target_identity,
        "database_containment": {**marker, "visible": False, "commit_outcome": "unknown"},
        "rollback_sql": {
            "path": str(rollback_sql_path),
            "sha256": hashlib.sha256(rollback_sql.encode("utf-8")).hexdigest(),
        },
        "failure": {"type": type(failure).__name__, "message": str(failure), "errors": errors},
        "evidence": {
            **source_receipt_evidence,
            "predecessor_partial_state_receipt": predecessor,
        },
    }
    authenticated = authenticate_evidence_receipt(
        partial,
        authentication_key=authentication_key,
        authentication_key_id=authentication_key_id,
    )
    _write_exclusive_durable(
        destination,
        (json.dumps(authenticated, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        expected_parent_identity=partial_parent_identity,
    )
    return destination


def _emit_commit_reconciled_receipt(
    requested_path: Path,
    *,
    source_receipt: dict,
    source_receipt_evidence: dict,
    rollback_sql_path: Path,
    rollback_sql: str,
    target_identity: dict,
    failure: ContainmentCommitReconciledConnectionLost,
    authentication_key: bytes,
    authentication_key_id: str,
    partial_parent_identity: tuple[str, int, int],
) -> Path:
    destination, predecessor = _partial_state_destination(
        requested_path,
        authentication_key=authentication_key,
        expected_key_id=authentication_key_id,
    )
    _require_matching_partial_state_predecessor(
        predecessor,
        marker_id=failure.marker["marker_id"],
        target_identity=target_identity,
        source_receipt_authentication=source_receipt.get("receipt_authentication"),
    )
    errors = [
        {"type": type(error).__name__, "message": str(error)}
        for error in failure.evidence_errors.exceptions
    ]
    partial = {
        "format": "applypilot-database-containment-commit-reconciled-v1",
        "status": "database_containment_committed_executor_connection_lost",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "target_identity": target_identity,
        "database_containment": {
            **failure.marker,
            "visible": True,
            "session_fence_complete": False,
            "retry_required": True,
        },
        "rollback_sql": {
            "path": str(rollback_sql_path),
            "sha256": hashlib.sha256(rollback_sql.encode("utf-8")).hexdigest(),
        },
        "failure": {"type": type(failure).__name__, "message": str(failure), "errors": errors},
        "evidence": {
            **source_receipt_evidence,
            "predecessor_partial_state_receipt": predecessor,
        },
    }
    authenticated = authenticate_evidence_receipt(
        partial,
        authentication_key=authentication_key,
        authentication_key_id=authentication_key_id,
    )
    _write_exclusive_durable(
        destination,
        (json.dumps(authenticated, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        expected_parent_identity=partial_parent_identity,
    )
    return destination


def _require_matching_partial_state_predecessor(
    predecessor: dict | None,
    *,
    marker_id: str,
    target_identity: dict[str, str],
    source_receipt_authentication: dict | None,
) -> None:
    stable_target_keys = ("database_name", "system_identifier", "database_oid")
    predecessor_target = (predecessor or {}).get("target_identity") or {}
    same_target = all(
        predecessor_target.get(key) == target_identity.get(key) for key in stable_target_keys
    )
    if predecessor is not None and (
        predecessor["marker_id"] != marker_id
        or not same_target
        or predecessor["source_receipt_authentication"] != source_receipt_authentication
    ):
        raise RuntimeError("partial-state predecessor does not bind the same target and rollback evidence")


def _replace_atomic(
    path: Path,
    payload: bytes,
    *,
    expected_identity: tuple[int, int],
    expected_content: bytes,
    expected_sha256: str,
    replacement_mode: int,
    replacement_owner: tuple[int, int],
) -> None:
    parent_before = os.lstat(path.parent)
    target_before = os.lstat(path)
    if not stat.S_ISDIR(parent_before.st_mode) or stat.S_ISLNK(parent_before.st_mode):
        raise RuntimeError("HBA parent must be a real directory")
    if os.name != "nt" and stat.S_IMODE(parent_before.st_mode) & 0o022:
        raise RuntimeError("HBA parent directory must not be group/world writable")
    if not stat.S_ISREG(target_before.st_mode) or stat.S_ISLNK(target_before.st_mode):
        raise RuntimeError("HBA target must be a regular non-symlink file")
    if (target_before.st_dev, target_before.st_ino) != expected_identity:
        raise RuntimeError("HBA target changed after validation")
    parent_descriptor: int | None = None
    temporary: Path
    if os.open in os.supports_dir_fd and os.replace in os.supports_dir_fd:
        parent_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_parent = os.fstat(parent_descriptor)
        if (opened_parent.st_dev, opened_parent.st_ino) != (parent_before.st_dev, parent_before.st_ino):
            os.close(parent_descriptor)
            raise RuntimeError("HBA parent changed while being opened")
        temporary = Path(f".applypilot-hba-rollback-{secrets.token_hex(16)}")
        descriptor = os.open(
            temporary.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
    else:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".applypilot-hba-rollback-", dir=path.parent)
        temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), replacement_mode)
            if hasattr(os, "fchown"):
                os.fchown(stream.fileno(), *replacement_owner)
            os.fsync(stream.fileno())
        parent_after = os.lstat(path.parent)
        if (parent_after.st_dev, parent_after.st_ino) != (parent_before.st_dev, parent_before.st_ino):
            raise RuntimeError("HBA parent changed before replacement")
        if parent_descriptor is not None:
            target_descriptor = os.open(
                path.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
        else:
            target_descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened_target = os.fstat(target_descriptor)
            if (opened_target.st_dev, opened_target.st_ino) != expected_identity:
                raise RuntimeError("HBA target changed before replacement")
            with os.fdopen(target_descriptor, "rb", closefd=False) as target_stream:
                observed_content = target_stream.read()
            if (
                observed_content != expected_content
                or not hmac.compare_digest(
                    hashlib.sha256(observed_content).hexdigest(), expected_sha256
                )
            ):
                raise RuntimeError("HBA target content changed after authenticated validation")
            opened_target_after_read = os.fstat(target_descriptor)
            if (
                opened_target_after_read.st_dev,
                opened_target_after_read.st_ino,
            ) != expected_identity:
                raise RuntimeError("HBA target changed while authenticated content was read")
            if parent_descriptor is not None:
                directory_target = os.stat(
                    path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            else:
                directory_target = os.lstat(path)
            if (directory_target.st_dev, directory_target.st_ino) != expected_identity:
                raise RuntimeError("HBA directory entry changed before replacement")
            if parent_descriptor is not None:
                os.replace(
                    temporary.name,
                    path.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
                os.fsync(parent_descriptor)
            else:
                os.replace(temporary, path)
                _fsync_parent(path)
        finally:
            os.close(target_descriptor)
        replaced = os.lstat(path)
        if stat.S_IMODE(replaced.st_mode) != replacement_mode:
            raise RuntimeError("HBA replacement mode verification failed")
        if (replaced.st_uid, replaced.st_gid) != replacement_owner:
            raise RuntimeError("HBA replacement ownership verification failed")
    finally:
        if parent_descriptor is not None:
            try:
                os.unlink(temporary.name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
            os.close(parent_descriptor)
        else:
            temporary.unlink(missing_ok=True)


def _validated_hba_restore(
    conn, receipt: dict
) -> tuple[Path, bytes, bytes, tuple[int, int], int, int, tuple[int, int]]:
    metadata = receipt.get("hba_restore")
    if not isinstance(metadata, dict) or metadata.get("format") != "applypilot-hba-restore-v2":
        raise SystemExit("authenticated receipt has no supported HBA restoration record")
    live_setting = Path(conn.execute("SHOW hba_file").fetchone()["hba_file"])
    hba_path = Path(metadata.get("live_hba_path", ""))
    backup_path = Path(metadata.get("backup_path", ""))
    if not hba_path.is_absolute() or not backup_path.is_absolute():
        raise SystemExit("HBA restoration paths must be absolute")
    if hba_path.resolve(strict=True) != hba_path or backup_path.resolve(strict=True) != backup_path:
        raise SystemExit("HBA restoration paths must be canonical and contain no symlinks")
    if live_setting.resolve(strict=True) != hba_path:
        raise SystemExit("receipt HBA target does not match SHOW hba_file")
    if backup_path.parent != hba_path.parent or backup_path == hba_path:
        raise SystemExit("HBA backup must be a distinct file in the live HBA directory")
    current, hba_stat = _read_bound_regular_file(hba_path, label="live HBA")
    backup, backup_stat = _read_bound_regular_file(backup_path, label="HBA backup")
    expected_owner = metadata.get("owner")
    if expected_owner != {"uid": hba_stat.st_uid, "gid": hba_stat.st_gid}:
        raise SystemExit("live HBA ownership does not match authenticated receipt")
    if (backup_stat.st_uid, backup_stat.st_gid) != (hba_stat.st_uid, hba_stat.st_gid):
        raise SystemExit("HBA backup ownership differs from live HBA")
    checks = (
        ("expected_target_sha256", hashlib.sha256(current).hexdigest()),
        ("backup_sha256", hashlib.sha256(backup).hexdigest()),
        ("backup_size", len(backup)),
        ("backup_mode", stat.S_IMODE(backup_stat.st_mode)),
    )
    for field, actual in checks:
        if metadata.get(field) != actual:
            raise SystemExit(f"HBA {field} does not match authenticated receipt")
    return (
        hba_path,
        current,
        backup,
        (hba_stat.st_dev, hba_stat.st_ino),
        stat.S_IMODE(hba_stat.st_mode),
        stat.S_IMODE(backup_stat.st_mode),
        (hba_stat.st_uid, hba_stat.st_gid),
    )


def _restore_hba_and_reload(conn, receipt: dict, *, validated=None) -> None:
    if validated is None:
        validated = _validated_hba_restore(conn, receipt)
    hba_path, current, backup, identity, live_mode, backup_mode, owner = validated
    _replace_atomic(
        hba_path,
        backup,
        expected_identity=identity,
        expected_content=current,
        expected_sha256=hashlib.sha256(current).hexdigest(),
        replacement_mode=backup_mode,
        replacement_owner=owner,
    )
    try:
        with conn.transaction():
            if not conn.execute(
                "SELECT pg_catalog.pg_reload_conf() AS reloaded"
            ).fetchone()["reloaded"]:
                raise RuntimeError("pg_reload_conf returned false after HBA restoration")
    except BaseException:
        replacement = os.lstat(hba_path)
        _replace_atomic(
            hba_path,
            current,
            expected_identity=(replacement.st_dev, replacement.st_ino),
            expected_content=backup,
            expected_sha256=hashlib.sha256(backup).hexdigest(),
            replacement_mode=live_mode,
            replacement_owner=owner,
        )
        restore_reloaded = False
        try:
            with conn.transaction():
                restore_reloaded = bool(
                    conn.execute(
                        "SELECT pg_catalog.pg_reload_conf() AS reloaded"
                    ).fetchone()["reloaded"]
                )
        except BaseException:
            restore_reloaded = False
        if not restore_reloaded:
            raise RuntimeError(
                "HBA reload failed; original HBA bytes were restored on disk but active rules are uncertain"
            ) from None
        raise RuntimeError("HBA reload failed; original HBA bytes and active rules were restored") from None


def main() -> int:
    args = _parser().parse_args()
    if args.restore_hba and not _hba_restore_supported():
        raise SystemExit(
            "secure HBA restoration is unsupported on this platform; "
            "run database rollback without --restore-hba and restore HBA through a protected host procedure"
        )
    authentication_key_hex = os.environ.pop("APPLYPILOT_ROLLBACK_HMAC_KEY_HEX", "")
    expected_key_id = os.environ.pop("APPLYPILOT_ROLLBACK_HMAC_KEY_ID", "")
    try:
        authentication_key = bytes.fromhex(authentication_key_hex)
    except ValueError:
        authentication_key = b""
    if len(authentication_key) < 32 or not expected_key_id:
        raise SystemExit("trusted rollback HMAC key and key id are required")
    receipt_path = args.receipt.resolve()
    rollback_path = args.rollback_sql.resolve()
    partial_state_path = (
        args.partial_state_receipt.resolve()
        if args.partial_state_receipt is not None
        else receipt_path.with_name(f"{receipt_path.name}.partial-state.json")
    )
    if partial_state_path in {receipt_path, rollback_path}:
        raise SystemExit("partial-state receipt path must be distinct from input evidence")
    receipt, rollback_sql = _verified_inputs(
        receipt_path,
        rollback_path,
        authentication_key=authentication_key,
        expected_key_id=expected_key_id,
    )
    source_receipt_evidence = _source_receipt_evidence(receipt_path, receipt)
    partial_parent_identity = _validate_partial_state_parent(partial_state_path)
    _partial_destination, partial_state_predecessor = _partial_state_destination(
        partial_state_path,
        authentication_key=authentication_key,
        expected_key_id=expected_key_id,
    )
    admin_dsn = os.environ.pop("APPLYPILOT_ADMIN_PG_DSN", "")
    if not admin_dsn:
        raise SystemExit("APPLYPILOT_ADMIN_PG_DSN is required only for break-glass rollback")

    # The authenticated break-glass connection is deliberately established
    # before HBA replacement/reload so rollback authority cannot be locked out.
    with psycopg.connect(admin_dsn, row_factory=dict_row) as conn:
        _pin_rollback_search_path(conn)
        target_identity = _validated_target_identity(conn, receipt)
        conn.commit()
        hba_preflight = _validated_hba_restore(conn, receipt) if args.restore_hba else None
        if hba_preflight is not None:
            # SHOW hba_file starts an implicit psycopg transaction. Complete
            # that read-only preflight before opening the atomic DB rollback.
            conn.commit()
            if conn.info.transaction_status.name != "IDLE":
                raise RuntimeError("HBA preflight transaction did not return to idle state")
        marker = _containment_marker(receipt, rollback_sql, target_identity)
        protected_roles = _protected_containment_roles(receipt)
        _require_matching_partial_state_predecessor(
            partial_state_predecessor,
            marker_id=marker["marker_id"],
            target_identity=target_identity,
            source_receipt_authentication=receipt.get("receipt_authentication"),
        )
        def commit_phase(phase: str) -> dict:
            try:
                return _commit_with_authenticated_reconciliation(
                    conn,
                    rollback_sql,
                    marker=marker,
                    protected_roles=protected_roles,
                    admin_dsn=admin_dsn,
                    receipt=receipt,
                    phase=phase,
                )
            except ContainmentCommitReconciledConnectionLost as error:
                try:
                    partial_receipt = _emit_commit_reconciled_receipt(
                        partial_state_path,
                        source_receipt=receipt,
                        source_receipt_evidence=source_receipt_evidence,
                        rollback_sql_path=rollback_path,
                        rollback_sql=rollback_sql,
                        target_identity=target_identity,
                        failure=error,
                        authentication_key=authentication_key,
                        authentication_key_id=expected_key_id,
                        partial_parent_identity=partial_parent_identity,
                    )
                except BaseException as receipt_error:
                    raise RuntimeError(
                        "database containment commit was reconciled, the executor connection was "
                        "lost, and authenticated checkpoint creation failed"
                    ) from BaseExceptionGroup(
                        "reconciled commit and checkpoint failures", [error, receipt_error]
                    )
                raise RuntimeError(
                    "database containment commit was reconciled but the executor connection was "
                    f"lost; retry required from authenticated checkpoint={partial_receipt}"
                ) from error
            except ContainmentCommitUnreconciled as error:
                try:
                    partial_receipt = _emit_commit_unknown_receipt(
                        partial_state_path,
                        source_receipt=receipt,
                        source_receipt_evidence=source_receipt_evidence,
                        rollback_sql_path=rollback_path,
                        rollback_sql=rollback_sql,
                        target_identity=target_identity,
                        marker=marker,
                        failure=error,
                        authentication_key=authentication_key,
                        authentication_key_id=expected_key_id,
                        partial_parent_identity=partial_parent_identity,
                    )
                except BaseException as receipt_error:
                    raise RuntimeError(
                        "database containment COMMIT outcome is unknown and authenticated "
                        "checkpoint creation also failed"
                    ) from BaseExceptionGroup(
                        "commit ambiguity and checkpoint failures", [error, receipt_error]
                    )
                raise RuntimeError(
                    "database containment COMMIT outcome is unknown; authenticated checkpoint="
                    f"{partial_receipt}"
                ) from error

        database_containment = commit_phase("phase1")
        print("database_containment_phase1_committed")
        terminated_sessions = _terminate_affected_sessions(
            conn,
            affected_role_names=database_containment["affected_role_names"],
            affected_role_oids=database_containment["affected_role_oids"],
            protected_roles=protected_roles,
        )
        database_containment = commit_phase("phase2")
        database_containment["terminated_sessions"] = terminated_sessions
        print("database_rollback_committed")
        containment_postcondition = _assert_candidate_artifact_containment(
            conn,
            break_glass_roles=protected_roles,
            affected_role_names=database_containment["affected_role_names"],
            affected_role_oids=database_containment["affected_role_oids"],
        )
        if args.restore_hba:
            print("hba_restore_started")
            try:
                _restore_hba_and_reload(conn, receipt, validated=hba_preflight)
            except BaseException as error:
                hba_observed = _observe_hba_state(receipt, validated=hba_preflight)
                try:
                    partial_receipt = _emit_partial_state_receipt(
                        partial_state_path,
                        source_receipt=receipt,
                        source_receipt_evidence=source_receipt_evidence,
                        rollback_sql_path=rollback_path,
                        rollback_sql=rollback_sql,
                        target_identity=target_identity,
                        database_containment=database_containment,
                        containment_postcondition=containment_postcondition,
                        hba_observed=hba_observed,
                        failure=error,
                        authentication_key=authentication_key,
                        authentication_key_id=expected_key_id,
                        partial_parent_identity=partial_parent_identity,
                    )
                except BaseException as receipt_error:
                    failures = BaseExceptionGroup(
                        "HBA restoration and durable partial-state receipt creation failed",
                        [error, receipt_error],
                    )
                    raise RuntimeError(
                        "database_containment_committed=true;"
                        "hba_state=uncertain_not_restored;"
                        "partial_receipt_created=false;"
                        "recovery=manual_containment_required"
                    ) from failures
                raise RuntimeError(
                    "database rollback committed, but HBA restoration failed; "
                    "partial state: database containment is durable and HBA admission state is uncertain; "
                    f"authenticated partial-state receipt={partial_receipt}"
                ) from error
    print("authenticated database rollback committed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
