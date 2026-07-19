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


def _validated_target_identity(conn, receipt: dict) -> dict[str, str]:
    identity = conn.execute(
        "SELECT session_user,current_user,rolsuper,current_database() AS database_name,"
        "(pg_control_system()).system_identifier::text AS system_identifier,"
        "(SELECT oid FROM pg_database WHERE datname=current_database()) AS database_oid "
        "FROM pg_roles WHERE rolname=session_user"
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


def _containment_marker(receipt: dict, rollback_sql: str, target_identity: dict[str, str]) -> dict[str, str]:
    rollback_sql_sha256 = hashlib.sha256(rollback_sql.encode("utf-8")).hexdigest()
    source_receipt_digest = receipt.get("receipt_authentication", {}).get("digest", "")
    marker_binding = {
        "database_name": target_identity["database_name"],
        "system_identifier": target_identity["system_identifier"],
        "database_oid": target_identity["database_oid"],
        "executed_by": target_identity["session_user"],
        "rollback_sql_sha256": rollback_sql_sha256,
        "source_receipt_digest": source_receipt_digest,
    }
    marker_id = hashlib.sha256(
        json.dumps(marker_binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"marker_id": marker_id, **marker_binding}


def _execute_database_containment_sql(conn, rollback_sql: str) -> None:
    conn.execute(rollback_sql)


def _write_database_containment_marker(conn, *, marker: dict) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS public.applypilot_rollback_commit_markers("
        "marker_id text PRIMARY KEY,database_name text NOT NULL,system_identifier text NOT NULL,"
        "database_oid oid NOT NULL,"
        "executed_by text NOT NULL,rollback_sql_sha256 text NOT NULL,source_receipt_digest text NOT NULL,"
        "committed_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp())"
    )
    relation = conn.execute(
        "SELECT relation.relkind,relation.relpersistence,relation.relrowsecurity,"
        "relation.relforcerowsecurity,owner.rolname AS owner_name,"
        "(SELECT count(*) FROM pg_trigger trigger WHERE trigger.tgrelid=relation.oid "
        "AND NOT trigger.tgisinternal) AS trigger_count,"
        "(SELECT count(*) FROM pg_policy policy WHERE policy.polrelid=relation.oid) AS policy_count,"
        "(SELECT count(*) FROM aclexplode(COALESCE(relation.relacl,acldefault('r',relation.relowner))) acl "
        "WHERE acl.grantee<>relation.relowner) AS nonowner_acl_count "
        "FROM pg_class relation JOIN pg_namespace namespace ON namespace.oid=relation.relnamespace "
        "JOIN pg_roles owner ON owner.oid=relation.relowner "
        "WHERE namespace.nspname='public' AND relation.relname='applypilot_rollback_commit_markers'"
    ).fetchone()
    expected_relation = {
        "relkind": "r",
        "relpersistence": "p",
        "relrowsecurity": False,
        "relforcerowsecurity": False,
        "owner_name": marker["executed_by"],
        "trigger_count": 0,
        "policy_count": 0,
        "nonowner_acl_count": 0,
    }
    if relation != expected_relation:
        raise RuntimeError(f"database containment marker relation is unsafe: {relation!r}")
    columns = conn.execute(
        "SELECT attribute.attname,format_type(attribute.atttypid,attribute.atttypmod) AS data_type,"
        "attribute.attnotnull FROM pg_attribute attribute "
        "WHERE attribute.attrelid='public.applypilot_rollback_commit_markers'::regclass "
        "AND attribute.attnum>0 AND NOT attribute.attisdropped ORDER BY attribute.attnum"
    ).fetchall()
    expected_columns = [
        {"attname": "marker_id", "data_type": "text", "attnotnull": True},
        {"attname": "database_name", "data_type": "text", "attnotnull": True},
        {"attname": "system_identifier", "data_type": "text", "attnotnull": True},
        {"attname": "database_oid", "data_type": "oid", "attnotnull": True},
        {"attname": "executed_by", "data_type": "text", "attnotnull": True},
        {"attname": "rollback_sql_sha256", "data_type": "text", "attnotnull": True},
        {"attname": "source_receipt_digest", "data_type": "text", "attnotnull": True},
        {"attname": "committed_at", "data_type": "timestamp with time zone", "attnotnull": True},
    ]
    if columns != expected_columns:
        raise RuntimeError(f"database containment marker column contract mismatch: {columns!r}")
    conn.execute(
        "INSERT INTO public.applypilot_rollback_commit_markers("
        "marker_id,database_name,system_identifier,database_oid,executed_by,"
        "rollback_sql_sha256,source_receipt_digest) "
        "VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(marker_id) DO NOTHING",
        (
            marker["marker_id"],
            marker["database_name"],
            marker["system_identifier"],
            marker["database_oid"],
            marker["executed_by"],
            marker["rollback_sql_sha256"],
            marker["source_receipt_digest"],
        ),
    )


def _commit_database_containment_transaction(conn) -> None:
    conn.commit()


def _assert_database_containment_marker(conn, *, marker: dict) -> dict:
    observed = conn.execute(
        "SELECT marker_id,database_name,system_identifier,database_oid,executed_by,rollback_sql_sha256,"
        "source_receipt_digest,committed_at FROM public.applypilot_rollback_commit_markers "
        "WHERE marker_id=%s",
        (marker["marker_id"],),
    ).fetchone()
    conn.commit()
    if observed is None or any(observed[key] != value for key, value in marker.items()):
        raise RuntimeError("database containment committed but its durable marker binding is invalid")
    committed_at = observed["committed_at"]
    return {
        **marker,
        "relation": _CONTAINMENT_MARKER_RELATION,
        "visible": True,
        "committed_at": committed_at.isoformat() if hasattr(committed_at, "isoformat") else str(committed_at),
    }


def _commit_database_containment(conn, rollback_sql: str, *, marker: dict) -> dict:
    """Commit rollback SQL and its idempotent database marker atomically."""

    try:
        _execute_database_containment_sql(conn, rollback_sql)
        _write_database_containment_marker(conn, marker=marker)
        _commit_database_containment_transaction(conn)
    except BaseException:
        conn.rollback()
        raise
    return _assert_database_containment_marker(conn, marker=marker)


def _assert_candidate_artifact_containment(conn, *, break_glass_roles: set[str]) -> dict[str, int]:
    del break_glass_roles  # Explicit break-glass superusers are inherently outside role ACL containment.
    function_routes = conn.execute(
        "SELECT count(*) FILTER (WHERE function.proname LIKE 'brain_publish_v%_candidate') "
        "AS candidate_execute_routes,"
        "count(*) FILTER (WHERE function.proname LIKE "
        "'brain_register_authoritative_artifact_manifest%') AS artifact_execute_routes "
        "FROM pg_proc function JOIN pg_namespace namespace ON namespace.oid=function.pronamespace "
        "CROSS JOIN LATERAL aclexplode(COALESCE(function.proacl,acldefault('f',function.proowner))) acl "
        "WHERE namespace.nspname='public' AND acl.privilege_type='EXECUTE' "
        "AND acl.grantee<>function.proowner"
    ).fetchone()
    membership_routes = conn.execute(
        "WITH RECURSIVE roots(role_oid) AS ("
        "SELECT oid FROM pg_roles WHERE rolname LIKE 'brain_candidate_%' "
        "OR rolname LIKE 'brain_artifact_authority_%'), closure(role_oid,path) AS ("
        "SELECT role_oid,ARRAY[role_oid] FROM roots UNION ALL "
        "SELECT membership.member,closure.path||membership.member FROM closure "
        "JOIN pg_auth_members membership ON membership.roleid=closure.role_oid "
        "WHERE NOT membership.member=ANY(closure.path)) "
        "SELECT count(*) AS set_role_routes FROM pg_auth_members membership "
        "WHERE membership.roleid IN (SELECT role_oid FROM closure)"
    ).fetchone()
    conn.commit()
    result = {
        "candidate_execute_routes": int(function_routes["candidate_execute_routes"]),
        "artifact_execute_routes": int(function_routes["artifact_execute_routes"]),
        "set_role_routes": int(membership_routes["set_role_routes"]),
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


def _require_matching_partial_state_predecessor(
    predecessor: dict | None,
    *,
    marker_id: str,
    target_identity: dict[str, str],
    source_receipt_authentication: dict | None,
) -> None:
    if predecessor is not None and (
        predecessor["marker_id"] != marker_id
        or predecessor["target_identity"] != target_identity
        or predecessor["source_receipt_authentication"] != source_receipt_authentication
    ):
        raise RuntimeError("partial-state predecessor does not bind the same target and rollback evidence")


def _replace_atomic(
    path: Path,
    payload: bytes,
    *,
    expected_identity: tuple[int, int],
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
        target_after = os.lstat(path)
        if (parent_after.st_dev, parent_after.st_ino) != (parent_before.st_dev, parent_before.st_ino):
            raise RuntimeError("HBA parent changed before replacement")
        if (target_after.st_dev, target_after.st_ino) != expected_identity:
            raise RuntimeError("HBA target changed before replacement")
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
        replacement_mode=backup_mode,
        replacement_owner=owner,
    )
    try:
        with conn.transaction():
            if not conn.execute("SELECT pg_reload_conf() AS reloaded").fetchone()["reloaded"]:
                raise RuntimeError("pg_reload_conf returned false after HBA restoration")
    except BaseException:
        replacement = os.lstat(hba_path)
        _replace_atomic(
            hba_path,
            current,
            expected_identity=(replacement.st_dev, replacement.st_ino),
            replacement_mode=live_mode,
            replacement_owner=owner,
        )
        restore_reloaded = False
        try:
            with conn.transaction():
                restore_reloaded = bool(
                    conn.execute("SELECT pg_reload_conf() AS reloaded").fetchone()["reloaded"]
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
    partial_state_predecessor = None
    partial_parent_identity = None
    if args.restore_hba:
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
        _require_matching_partial_state_predecessor(
            partial_state_predecessor,
            marker_id=marker["marker_id"],
            target_identity=target_identity,
            source_receipt_authentication=receipt.get("receipt_authentication"),
        )
        database_containment = _commit_database_containment(
            conn,
            rollback_sql,
            marker=marker,
        )
        print("database_rollback_committed")
        containment_postcondition = _assert_candidate_artifact_containment(
            conn,
            break_glass_roles=_break_glass_roles(receipt),
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
