#!/usr/bin/env python3
"""Operate the canonical SQLite-to-PostgreSQL brain import with audit receipts."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import sys
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import psycopg
from psycopg.pq import TransactionStatus
from psycopg.rows import dict_row


_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(_SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIRECTORY))

from release_evidence_common import (  # noqa: E402
    assert_separated_keys,
    atomic_write_no_overwrite,
    canonical_receipt_payload,
    reject_symlink_components,
    regular_file,
    stable_read_bytes,
    strict_json_loads,
    validate_release_binding,
    verify_receipt,
)

from applypilot.brain.importer import (  # noqa: E402
    ONLINE_BACKUP_MODE,
    SOURCE_TABLES,
    SealedSnapshotReceipt,
    SourceFileAudit,
    seal_sqlite_snapshot,
)
from applypilot.brain.sqlite_to_postgres import (  # noqa: E402
    ImportSummary,
    finalize_sqlite_to_postgres_import,
    import_sqlite_to_postgres,
    recover_finalized_sqlite_to_postgres_import,
)


COMMAND_VERSION = "2.0.0"
FREEZE_SCHEMA = "applypilot.writer-freeze.v2"
MAX_FREEZE_AGE_SECONDS = 300
MAX_FREEZE_VALIDITY_SECONDS = 900
_MODES = ("bulk-import", "final-delta-finalize")
_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_IMPORT_PARENT_ROLES = {
    "brain_schema_migrator",
    "brain_schema_verifier",
}
_ATTESTATION_ENV = {
    "writer-freeze": "APPLYPILOT_WRITER_FREEZE_ATTESTATION",
    "brain-import": "APPLYPILOT_BRAIN_IMPORT_ATTESTATION",
}
_connect_postgres = psycopg.connect


class CLIError(RuntimeError):
    """An operator-safe failure whose message contains no connection secrets."""


class _PathAnchor(NamedTuple):
    path: Path
    parent: Path
    parent_identity: tuple[int, int]


def _path_identity(path: Path) -> tuple[int, int]:
    metadata = path.stat()
    return (metadata.st_dev, metadata.st_ino)


def _capture_path_anchor(path: Path, label: str) -> _PathAnchor:
    try:
        reject_symlink_components(path.parent)
        parent = path.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CLIError(f"{label} parent path is not anchored without aliases or reparse points") from exc
    return _PathAnchor(path=path, parent=parent, parent_identity=_path_identity(parent))


def _revalidate_path_anchor(anchor: _PathAnchor, label: str) -> None:
    try:
        reject_symlink_components(anchor.path.parent)
        parent = anchor.path.parent.resolve(strict=True)
        identity = _path_identity(parent)
    except (OSError, RuntimeError) as exc:
        raise CLIError(f"{label} parent path changed or became a reparse point") from exc
    if parent != anchor.parent or identity != anchor.parent_identity:
        raise CLIError(f"{label} parent path was replaced")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_existing_file(raw_path: str, label: str) -> Path:
    requested = Path(raw_path)
    if not requested.is_absolute():
        raise CLIError(f"{label} must be an explicit absolute path")
    try:
        reject_symlink_components(requested)
        resolved = requested.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CLIError(f"{label} does not exist: {requested}") from exc
    if requested != resolved:
        raise CLIError(f"{label} must be a canonical absolute path without aliases: {requested}")
    try:
        regular_file(resolved, label)
    except RuntimeError as exc:
        raise CLIError(str(exc)) from exc
    return resolved


def _canonical_output_path(raw_path: str, label: str) -> Path:
    requested = Path(raw_path)
    if not requested.is_absolute():
        raise CLIError(f"{label} must be an explicit absolute path")
    try:
        reject_symlink_components(requested.parent)
        parent = requested.parent.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CLIError(f"{label} parent directory does not exist: {requested.parent}") from exc
    canonical = parent / requested.name
    if requested != canonical:
        raise CLIError(f"{label} must be a canonical absolute path without aliases: {requested}")
    return canonical


def _load_dsn(environment_name: str) -> str:
    if _ENVIRONMENT_NAME_RE.fullmatch(environment_name) is None:
        raise CLIError("PostgreSQL DSN environment variable name is invalid")
    dsn = os.environ.get(environment_name)
    if not isinstance(dsn, str) or not dsn.strip():
        raise CLIError(f"PostgreSQL DSN environment variable is unset or empty: {environment_name}")
    return dsn


def _rollback_identity_preflight(pg) -> None:
    try:
        pg.rollback()
    except Exception:
        raise CLIError(
            "canonical import identity preflight could not restore an idle connection; "
            "connection details withheld"
        ) from None


def _require_canonical_import_identity(pg) -> None:
    """Require the exact bootstrap-provisioned database-owner import identity.

    This gate runs before any scoped ``SET ROLE``, durable identity read, import
    lock, output artifact, or database mutation. It rejects a provider-admin
    connection even though PostgreSQL superusers could otherwise activate the
    migrator role directly.
    """
    transaction_status = getattr(getattr(pg, "info", None), "transaction_status", None)
    if transaction_status is not None and transaction_status != TransactionStatus.IDLE:
        raise CLIError("canonical import identity preflight requires an idle dedicated connection")
    try:
        identity = pg.execute(
            "SELECT session_user AS canonical_session_user,"
            "current_user AS canonical_current_user,"
            "owner.rolname AS database_owner_role,"
            "login.rolcanlogin,login.rolinherit,login.rolsuper,login.rolcreatedb,"
            "login.rolcreaterole,login.rolreplication,login.rolbypassrls "
            "FROM pg_catalog.pg_database database "
            "JOIN pg_catalog.pg_roles owner ON owner.oid=database.datdba "
            "JOIN pg_catalog.pg_roles login ON login.rolname=session_user "
            "WHERE database.datname=pg_catalog.current_database()"
        ).fetchone()
        if not isinstance(identity, Mapping):
            raise RuntimeError("canonical identity row unavailable")
        session_user = identity.get("canonical_session_user")
        memberships = pg.execute(
            "SELECT parent.rolname AS canonical_parent_role,"
            "member.rolname AS canonical_member_role,"
            "grantor.oid AS grantor_oid,grantor.rolsuper AS grantor_is_superuser,"
            "membership.admin_option,membership.inherit_option,membership.set_option "
            "FROM pg_catalog.pg_auth_members membership "
            "JOIN pg_catalog.pg_roles parent ON parent.oid=membership.roleid "
            "JOIN pg_catalog.pg_roles member ON member.oid=membership.member "
            "JOIN pg_catalog.pg_roles grantor ON grantor.oid=membership.grantor "
            "WHERE member.rolname=%s OR parent.rolname=%s "
            "ORDER BY parent.rolname,member.rolname,grantor.oid",
            (session_user, session_user),
        ).fetchall()
    except CLIError:
        _rollback_identity_preflight(pg)
        raise
    except Exception:
        _rollback_identity_preflight(pg)
        raise CLIError(
            "canonical import identity preflight could not be verified; connection details withheld"
        ) from None

    expected_attributes = {
        "rolcanlogin": True,
        "rolinherit": False,
        "rolsuper": False,
        "rolcreatedb": False,
        "rolcreaterole": False,
        "rolreplication": False,
        "rolbypassrls": False,
    }
    attributes_match = all(
        identity.get(name) is expected for name, expected in expected_attributes.items()
    )
    memberships_match = (
        len(memberships) == 2
        and {
            row.get("canonical_parent_role")
            for row in memberships
            if isinstance(row, Mapping)
        }
        == _CANONICAL_IMPORT_PARENT_ROLES
        and all(
            isinstance(row, Mapping)
            and row.get("canonical_member_role") == session_user
            and row.get("grantor_oid") == 10
            and row.get("grantor_is_superuser") is True
            and row.get("admin_option") is False
            and row.get("inherit_option") is False
            and row.get("set_option") is True
            for row in memberships
        )
    )
    identity_matches = (
        isinstance(session_user, str)
        and bool(session_user)
        and identity.get("canonical_current_user") == session_user
        and identity.get("database_owner_role") == session_user
        and attributes_match
        and memberships_match
    )
    _rollback_identity_preflight(pg)
    if not identity_matches:
        raise CLIError(
            "PostgreSQL DSN must authenticate as the fixed database-owner canonical import "
            "login with exact SET-only memberships"
        )


def _preflight_canonical_import_dsn(dsn: str) -> None:
    try:
        with _connect_postgres(dsn, row_factory=dict_row) as pg:
            _require_canonical_import_identity(pg)
    except CLIError:
        raise
    except Exception:
        raise CLIError(
            "PostgreSQL canonical import identity preflight failed; connection details withheld"
        ) from None


def _attestation_key(purpose: str) -> tuple[bytes, str]:
    prefix = _ATTESTATION_ENV[purpose]
    encoded = os.environ.get(f"{prefix}_KEY_B64")
    key_id = os.environ.get(f"{prefix}_KEY_ID")
    if not isinstance(encoded, str) or not encoded or not isinstance(key_id, str) or not key_id.strip():
        raise CLIError(f"{purpose} attestation key and key ID environment variables are required")
    try:
        key = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CLIError(f"{purpose} attestation key must be valid base64") from exc
    if len(key) < 32:
        raise CLIError(f"{purpose} attestation key must decode to at least 32 bytes")
    return key, key_id


def _signed_receipt(document: dict[str, Any], purpose: str) -> dict[str, Any]:
    key, key_id = _attestation_key(purpose)
    signature = base64.b64encode(
        hmac.digest(key, canonical_receipt_payload(document), hashlib.sha256)
    ).decode("ascii")
    return {
        **document,
        "authentication": {"algorithm": "HMAC-SHA256", "keyId": key_id, "signature": signature},
    }


def _utc_datetime(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise CLIError(f"{label} must be a non-empty UTC timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise CLIError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise CLIError(f"{label} must include the UTC offset")
    return parsed.astimezone(timezone.utc)


def _parse_utc_timestamp(value: Any, label: str) -> str:
    _utc_datetime(value, label)
    return value


def _validate_freeze_time_window(document: Mapping[str, Any], *, now: datetime | None = None) -> None:
    frozen = _utc_datetime(document.get("frozenAt"), "writer-freeze marker frozenAt")
    expires = _utc_datetime(document.get("expiresAt"), "writer-freeze marker expiresAt")
    observed_now = _utc_now() if now is None else now
    if frozen > observed_now or expires <= observed_now or expires <= frozen:
        raise CLIError("writer-freeze marker freshness window is expired or invalid")
    if (observed_now - frozen).total_seconds() > MAX_FREEZE_AGE_SECONDS:
        raise CLIError(f"writer-freeze marker exceeds maximum age of {MAX_FREEZE_AGE_SECONDS} seconds")
    if (expires - frozen).total_seconds() > MAX_FREEZE_VALIDITY_SECONDS:
        raise CLIError(
            f"writer-freeze marker validity duration exceeds {MAX_FREEZE_VALIDITY_SECONDS} seconds"
        )


def _load_freeze_marker(
    marker_path: Path,
    *,
    anchor: _PathAnchor,
    release_id: str,
    release_nonce: str,
    source: Path,
) -> tuple[dict[str, Any], bytes]:
    try:
        content = _stable_read_anchored(anchor, "writer-freeze marker")
        document = strict_json_loads(content, "writer-freeze marker")
    except RuntimeError as exc:
        raise CLIError(str(exc)) from exc
    if not isinstance(document, dict):
        raise CLIError("writer-freeze marker must be a JSON object")
    freeze_key, freeze_key_id = _attestation_key("writer-freeze")
    try:
        verify_receipt(
            document,
            key=freeze_key,
            expected_key_id=freeze_key_id,
            label="writer-freeze marker",
        )
    except RuntimeError as exc:
        raise CLIError(str(exc)) from exc
    expected = {
        "schema": FREEZE_SCHEMA,
        "purpose": "writer-freeze",
        "releaseId": release_id,
        "releaseNonce": release_nonce,
        "sourcePath": str(source),
    }
    for name, value in expected.items():
        if document.get(name) != value:
            raise CLIError(f"writer-freeze marker {name} does not match this operation")
    if document.get("writerProcessCount") != 0 or document.get("activeWriterLeaseCount") != 0:
        raise CLIError("writer-freeze marker writer and lease counts must both be zero")
    _validate_freeze_time_window(document)
    source_state = document.get("sourceState")
    if not isinstance(source_state, dict) or set(source_state) != {"database", "wal", "shm"}:
        raise CLIError("writer-freeze marker sourceState must bind database, wal, and shm")
    return document, content


def _audit_final_state(audit: SourceFileAudit) -> dict[str, Any]:
    return {
        "exists": audit.after_exists,
        "sha256": audit.after_sha256,
        "size": audit.after_size,
        "mtime_ns": audit.after_mtime_ns,
        "stat_identity": audit.after_stat_identity,
    }


def _capture_file_state(
    path: Path,
    *,
    anchor: _PathAnchor | None = None,
    label: str = "evidence path",
) -> dict[str, Any]:
    if anchor is not None:
        _revalidate_path_anchor(anchor, label)
    else:
        reject_symlink_components(path.parent)
    try:
        before = path.stat()
    except FileNotFoundError:
        return {"exists": False, "sha256": None, "size": None, "identity": None}
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
    except (FileNotFoundError, OSError) as exc:
        raise CLIError(f"canonical SQLite DB/WAL state changed while reading: {path}") from exc
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if before_identity != after_identity:
        raise CLIError(f"canonical SQLite DB/WAL state changed while reading: {path}")
    return {
        "exists": True,
        "sha256": digest.hexdigest(),
        "size": after.st_size,
        "identity": [str(item) for item in after_identity],
    }


def _sealed_state(audit: SourceFileAudit) -> dict[str, Any]:
    return {
        "exists": audit.after_exists,
        "sha256": audit.after_sha256,
        "size": audit.after_size,
        "identity": None
        if audit.after_stat_identity is None
        else [str(item) for item in audit.after_stat_identity],
    }


def _assert_live_source_unchanged(
    source: Path,
    sealed: SealedSnapshotReceipt,
    anchors: Mapping[str, _PathAnchor],
) -> dict[str, Any]:
    observed = {
        "database": _capture_file_state(source, anchor=anchors["database"], label="SQLite database"),
        "wal": _capture_file_state(Path(f"{source}-wal"), anchor=anchors["wal"], label="SQLite WAL"),
        "shm": _capture_file_state(Path(f"{source}-shm"), anchor=anchors["shm"], label="SQLite SHM"),
    }
    expected = {
        "database": _sealed_state(sealed.source_db_audit),
        "wal": _sealed_state(sealed.source_wal_audit),
        "shm": _sealed_state(sealed.source_shm_audit),
    }
    if observed != expected:
        raise CLIError("canonical SQLite DB/WAL state changed across the final delta")
    return observed


def _validate_sealed_snapshot(
    sealed: SealedSnapshotReceipt,
    *,
    source: Path,
    snapshot: Path,
) -> None:
    if Path(sealed.source_path) != source or Path(sealed.path) != snapshot:
        raise CLIError("controlled snapshot receipt does not bind the requested source and destination")
    if sealed.source_mode != ONLINE_BACKUP_MODE:
        raise CLIError("controlled snapshot did not include SQLite WAL state through the online backup API")
    if sealed.source_changed_during_backup:
        raise CLIError("canonical SQLite source changed while the controlled snapshot was sealed")
    for audit, label in (
        (sealed.source_db_audit, "database"),
        (sealed.source_wal_audit, "WAL"),
    ):
        if not audit.observation_complete or audit.changed:
            raise CLIError(f"canonical SQLite {label} state was not stable while sealing")
    if not sealed.source_db_audit.after_exists or sealed.source_db_audit.after_sha256 is None:
        raise CLIError("controlled snapshot lacks complete canonical SQLite database evidence")


def _validate_freeze_binding(document: dict[str, Any], sealed: SealedSnapshotReceipt) -> None:
    expected = {
        "database": _sealed_state(sealed.source_db_audit),
        "wal": _sealed_state(sealed.source_wal_audit),
        "shm": _sealed_state(sealed.source_shm_audit),
    }
    if document["sourceState"] != expected:
        raise CLIError("writer-freeze marker does not bind the sealed DB/WAL/SHM state")


def _audit_as_dict(audit: SourceFileAudit) -> dict[str, Any]:
    return {
        "path": audit.path,
        "before_exists": audit.before_exists,
        "after_exists": audit.after_exists,
        "before_sha256": audit.before_sha256,
        "after_sha256": audit.after_sha256,
        "before_size": audit.before_size,
        "after_size": audit.after_size,
        "before_mtime_ns": audit.before_mtime_ns,
        "after_mtime_ns": audit.after_mtime_ns,
        "before_stat_identity": audit.before_stat_identity,
        "after_stat_identity": audit.after_stat_identity,
        "changed": audit.changed,
        "observation_complete": audit.observation_complete,
        "ephemeral": audit.ephemeral,
    }


def _source_evidence(sealed: SealedSnapshotReceipt) -> dict[str, Any]:
    return {
        "canonical_path": sealed.source_path,
        "sealed_snapshot_path": sealed.path,
        "sealed_snapshot_sha256": sealed.sha256,
        "sealed_snapshot_bytes": sealed.size,
        "quick_check": sealed.quick_check,
        "source_mode": sealed.source_mode,
        "source_changed_during_backup": sealed.source_changed_during_backup,
        "database_audit": _audit_as_dict(sealed.source_db_audit),
        "wal_audit": _audit_as_dict(sealed.source_wal_audit),
        "shm_audit": _audit_as_dict(sealed.source_shm_audit),
    }


def _is_canonical_uuid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value
    except (ValueError, AttributeError):
        return False


# The durable identity read is a pure SELECT, so it runs under the least-privileged role that
# V8 grants that SELECT to: ``brain_schema_verifier`` (schema_v8.sql grants it SELECT on the
# singleton), *not* the table owner ``brain_schema_migrator``. Reading through the owner would
# assume a role that can also create/alter/drop the identity table; the verifier role can only
# read it. Deployment must therefore grant the canonical import LOGIN membership in
# ``brain_schema_verifier`` WITH SET TRUE (the same non-inheriting SET channel it already holds
# for the migrator), which confers no standing authority -- only the ability to assume the
# read-only role for the duration of this scoped SELECT.
_DURABLE_IDENTITY_READ_ROLE = "brain_schema_verifier"


@contextmanager
def _durable_identity_role(pg):
    """Run a durable-identity read under the least-privileged canonical role.

    The canonical import LOGIN is ``NOINHERIT`` and is never granted direct SELECT on the V8
    ``public.applypilot_database_identity`` singleton. V8 grants that SELECT to the read-only
    ``brain_schema_verifier`` role, so the read executes under an explicit
    ``SET ROLE brain_schema_verifier`` -- the least-privileged role that can read it, never the
    table owner ``brain_schema_migrator`` -- with a hardened ``search_path`` of
    ``pg_catalog, public`` and nothing else. The role and search_path are always torn down --
    the read-only transaction is rolled back and the role/search_path reset -- on success, on
    exception, and after an ambiguous commit/recovery, so neither can leak onto a pooled or
    reused connection. No table grant is widened and no superuser/GUC/environment/argument
    fallback is consulted.
    """
    try:
        pg.execute(f"SET ROLE {_DURABLE_IDENTITY_READ_ROLE}")
        pg.execute("SET search_path TO pg_catalog, public")
        active = pg.execute("SELECT current_user AS current_user").fetchone()
        if not isinstance(active, Mapping) or active.get("current_user") != _DURABLE_IDENTITY_READ_ROLE:
            raise CLIError("least-privileged durable database identity role is not active")
        yield
    finally:
        _restore_login_role(pg)


def _restore_login_role(pg) -> None:
    """Fail-closed restore of the canonical login role and default search_path.

    Rolling back discards the read-only identity transaction and -- because ``SET`` is
    transactional -- reverts the scoped role and search_path while returning the connection to
    the idle state the importer's own boundary guard requires. The explicit ``RESET`` afterwards
    is defense-in-depth for an autocommit session where the ``SET`` persisted at session scope.
    A failure to reset or commit the restore is *surfaced* as a ``CLIError`` (never swallowed):
    a connection that might still be wearing the elevated role or hardened search_path must not
    be silently returned to a pool as if the teardown had succeeded.
    """
    rollback = getattr(pg, "rollback", None)
    commit = getattr(pg, "commit", None)
    if callable(rollback):
        try:
            rollback()
        except Exception:
            pass
    try:
        pg.execute("RESET ROLE")
        pg.execute("RESET search_path")
        if callable(commit):
            commit()
    except Exception as exc:
        if callable(rollback):
            try:
                rollback()
            except Exception:
                pass
        raise CLIError(
            "failed to restore the canonical login role after the durable identity read"
        ) from exc


def _read_database_incarnation(pg) -> str:
    """Read the durable incarnation from the ``applypilot_database_identity`` singleton itself.

    The ambient ``applypilot.database_incarnation_id`` GUC is deliberately never consulted, so a
    spoofed session setting has no effect. A missing, empty, ambiguous (multiple), or malformed
    singleton fails closed rather than silently importing into an unbound destination. The
    privileged singleton SELECT runs under a tightly scoped least-privilege role (see
    ``_durable_identity_role``) so the canonical NOINHERIT import LOGIN never needs a direct
    table grant.
    """
    with _durable_identity_role(pg):
        row = pg.execute(
            "SELECT count(*) AS identity_row_count,"
            " min(database_incarnation_id::text) AS incarnation_low,"
            " max(database_incarnation_id::text) AS incarnation_high"
            " FROM public.applypilot_database_identity WHERE singleton_id=1"
        ).fetchone()
    if not isinstance(row, Mapping):
        raise CLIError("connected PostgreSQL database identity singleton is unavailable")
    count = row.get("identity_row_count")
    low, high = row.get("incarnation_low"), row.get("incarnation_high")
    if count != 1 or low is None or low != high:
        raise CLIError("connected PostgreSQL database identity singleton is missing or ambiguous")
    if not _is_canonical_uuid(low):
        raise CLIError("connected PostgreSQL database incarnation is malformed")
    return low


def _destination_identity(pg) -> dict[str, Any]:
    row = pg.execute(
        """SELECT current_database() AS database,
                  (pg_control_system()).system_identifier::text AS system_identifier,
                  (SELECT oid::text FROM pg_database WHERE datname=current_database()) AS database_oid,
                  to_regclass('public.applypilot_database_identity') IS NOT NULL AS identity_present,
                  current_setting('server_version_num') AS server_version_num"""
    ).fetchone()
    if not isinstance(row, Mapping):
        raise CLIError("connected PostgreSQL database identity is unavailable")
    if row.get("identity_present") is not True:
        raise CLIError("connected PostgreSQL is missing the durable database identity singleton")
    return {
        "database": row.get("database"),
        "systemIdentifier": row.get("system_identifier"),
        "databaseOid": row.get("database_oid"),
        "databaseIncarnationId": _read_database_incarnation(pg),
        "serverVersionNum": row.get("server_version_num"),
    }


def _assert_destination_binding(
    destination: dict[str, Any],
    *,
    expected_database: str,
    expected_system_identifier: str,
    expected_database_oid: str,
    expected_database_incarnation_id: str | None,
) -> None:
    if destination["database"] != expected_database:
        raise CLIError("connected PostgreSQL database name does not match --expected-database")
    if str(destination["systemIdentifier"]) != expected_system_identifier:
        raise CLIError("connected PostgreSQL system identifier does not match --expected-system-identifier")
    if str(destination["databaseOid"]) != expected_database_oid:
        raise CLIError("connected PostgreSQL database OID does not match --expected-database-oid")
    if not _is_canonical_uuid(expected_database_incarnation_id):
        raise CLIError(
            "a canonical --expected-database-incarnation-id is required for a write-capable import"
        )
    incarnation = destination["databaseIncarnationId"]
    if not _is_canonical_uuid(incarnation):
        raise CLIError("connected PostgreSQL database incarnation is malformed or absent")
    if not hmac.compare_digest(str(incarnation), str(expected_database_incarnation_id)):
        raise CLIError(
            "connected PostgreSQL database incarnation does not match --expected-database-incarnation-id"
        )


def _run_key(
    release_id: str,
    release_nonce: str,
    mode: str,
    destination: Mapping[str, Any],
) -> str:
    binding = json.dumps(
        [release_id, release_nonce, mode, dict(destination)],
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    return f"sqlite-pg-cli-v1:{mode}:{_sha256_bytes(binding)}"


def _durable_destination_binding(destination: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: destination[name]
        for name in ("database", "systemIdentifier", "databaseOid", "databaseIncarnationId")
    }


def _parity_is_complete_and_clean(parity: Mapping[str, Mapping[str, Any]]) -> bool:
    required = {
        "source_count",
        "target_count",
        "source_hash",
        "target_hash",
        "mismatch_count",
        "unresolved_count",
        "passed",
    }
    if set(parity) != {table.name for table in SOURCE_TABLES}:
        return False
    for record in parity.values():
        if not isinstance(record, Mapping) or set(record) != required:
            return False
        counts = (
            record.get("source_count"),
            record.get("target_count"),
            record.get("mismatch_count"),
            record.get("unresolved_count"),
        )
        if (
            any(type(value) is not int or value < 0 for value in counts)
            or not isinstance(record.get("source_hash"), str)
            or not isinstance(record.get("target_hash"), str)
            or record.get("passed") is not True
            or record.get("source_count") != record.get("target_count")
            or record.get("source_hash") != record.get("target_hash")
            or record.get("mismatch_count") != 0
            or record.get("unresolved_count") != 0
            or _SHA256_RE.fullmatch(str(record.get("source_hash"))) is None
        ):
            return False
    return True


def _publication_commit_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(f"{receipt_path.name}.commit.json")


def _stable_read_anchored(anchor: _PathAnchor, label: str) -> bytes:
    _revalidate_path_anchor(anchor, label)
    content = stable_read_bytes(anchor.path, label)
    _revalidate_path_anchor(anchor, label)
    return content


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise CLIError(f"{label} schema keys are invalid")
    return value


def _validate_receipt_semantics(receipt: dict[str, Any], receipt_path: Path, commit_path: Path) -> None:
    _require_exact_keys(
        receipt,
        {
            "receiptSchema", "purpose", "promotable", "promotionEligible", "publicationProtocol",
            "publicationCommitPath", "command", "release", "source", "destination", "writer_freeze",
            "result", "timestamps", "authentication",
        },
        "brain import receipt",
    )
    if receipt["receiptSchema"] != "applypilot.sqlite-to-postgres-import-receipt.v2":
        raise CLIError("brain import receipt schema is unsupported")
    if receipt["purpose"] != "brain-import":
        raise CLIError("brain import receipt purpose is invalid")
    if receipt["promotable"] is not False or receipt["promotionEligible"] is not True:
        raise CLIError("brain import receipt is not eligible for publication commit promotion")
    if receipt["publicationProtocol"] != "authenticated-two-phase-commit-v1":
        raise CLIError("brain import receipt publication protocol is invalid")
    if receipt["publicationCommitPath"] != str(commit_path):
        raise CLIError("brain import receipt publication commit path is invalid")

    command = _require_exact_keys(
        receipt["command"],
        {"name", "version", "mode", "dryRun", "batchSize", "dsnEnvironmentVariable", "runKey"},
        "brain import receipt command",
    )
    if command["name"] != "brain-sqlite-to-postgres" or command["mode"] != "final-delta-finalize":
        raise CLIError("brain import receipt command mode is not final-delta-finalize")
    if not isinstance(command["version"], str) or not command["version"]:
        raise CLIError("brain import receipt command version is invalid")
    if command["dryRun"] is not False:
        raise CLIError("brain import receipt command dryRun must be false")
    if type(command["batchSize"]) is not int or command["batchSize"] <= 0:
        raise CLIError("brain import receipt command batch size is invalid")
    if not isinstance(command["runKey"], str) or not command["runKey"]:
        raise CLIError("brain import receipt command run key is invalid")
    if (
        not isinstance(command["dsnEnvironmentVariable"], str)
        or _ENVIRONMENT_NAME_RE.fullmatch(command["dsnEnvironmentVariable"]) is None
    ):
        raise CLIError("brain import receipt DSN environment variable name is invalid")

    release = _require_exact_keys(receipt["release"], {"id", "nonce"}, "brain import receipt release")
    try:
        validate_release_binding(release["id"], release["nonce"])
    except RuntimeError as exc:
        raise CLIError(f"brain import receipt release binding is invalid: {exc}") from exc

    destination = _require_exact_keys(
        receipt["destination"],
        {"database", "systemIdentifier", "databaseOid", "databaseIncarnationId", "serverVersionNum"},
        "brain import receipt destination",
    )
    durable_destination = _durable_destination_binding(destination)
    for name in ("database", "systemIdentifier", "databaseOid"):
        if not isinstance(durable_destination[name], str) or not durable_destination[name]:
            raise CLIError(f"brain import receipt destination {name} is invalid")
    incarnation = durable_destination["databaseIncarnationId"]
    if not _is_canonical_uuid(incarnation):
        raise CLIError("brain import receipt destination incarnation must be a canonical UUID")
    if not str(destination["serverVersionNum"]).isdigit():
        raise CLIError("brain import receipt destination server version is invalid")
    expected_run_key = _run_key(
        release["id"],
        release["nonce"],
        "final-delta-finalize",
        durable_destination,
    )
    if not hmac.compare_digest(command["runKey"], expected_run_key):
        raise CLIError("brain import receipt run key does not match release and destination binding")

    source = _require_exact_keys(
        receipt["source"],
        {
            "canonical_path", "sealed_snapshot_path", "sealed_snapshot_sha256", "sealed_snapshot_bytes",
            "quick_check", "source_mode", "source_changed_during_backup", "database_audit", "wal_audit",
            "shm_audit",
        },
        "brain import receipt source",
    )
    if _SHA256_RE.fullmatch(str(source["sealed_snapshot_sha256"])) is None:
        raise CLIError("brain import receipt source snapshot hash is invalid")
    if source["quick_check"] != "ok" or source["source_mode"] != ONLINE_BACKUP_MODE:
        raise CLIError("brain import receipt source snapshot contract is invalid")
    if not all(
        isinstance(source[name], str) and Path(source[name]).is_absolute()
        for name in ("canonical_path", "sealed_snapshot_path")
    ):
        raise CLIError("brain import receipt source paths are invalid")
    if type(source["sealed_snapshot_bytes"]) is not int or source["sealed_snapshot_bytes"] <= 0:
        raise CLIError("brain import receipt source snapshot size is invalid")
    if source["source_changed_during_backup"] is not False:
        raise CLIError("brain import receipt source changed during backup")
    audit_keys = {
        "path", "before_exists", "after_exists", "before_sha256", "after_sha256", "before_size",
        "after_size", "before_mtime_ns", "after_mtime_ns", "before_stat_identity", "after_stat_identity",
        "changed", "observation_complete", "ephemeral",
    }
    for name in ("database_audit", "wal_audit", "shm_audit"):
        audit = _require_exact_keys(source[name], audit_keys, f"brain import receipt {name}")
        if audit["observation_complete"] is not True or audit["changed"] is not False:
            raise CLIError(f"brain import receipt {name} is not stable and complete")

    freeze = _require_exact_keys(
        receipt["writer_freeze"],
        {"markerPath", "markerSha256", "frozenAt", "expiresAt", "finalLiveSourceState"},
        "brain import receipt writer freeze",
    )
    if _SHA256_RE.fullmatch(str(freeze["markerSha256"])) is None:
        raise CLIError("brain import receipt writer freeze marker hash is invalid")
    freeze_started = _utc_datetime(freeze["frozenAt"], "brain import receipt writer freeze frozenAt")
    freeze_expires = _utc_datetime(freeze["expiresAt"], "brain import receipt writer freeze expiresAt")
    if freeze_started >= freeze_expires:
        raise CLIError("brain import receipt writer freeze interval is reversed or empty")
    if (freeze_expires - freeze_started).total_seconds() > MAX_FREEZE_VALIDITY_SECONDS:
        raise CLIError("brain import receipt writer freeze validity duration exceeds the maximum")
    live = _require_exact_keys(
        freeze["finalLiveSourceState"], {"database", "wal", "shm"}, "brain import receipt live source"
    )
    state_keys = {"exists", "sha256", "size", "identity"}
    for name in ("database", "wal", "shm"):
        _require_exact_keys(live[name], state_keys, f"brain import receipt live {name}")
    expected_live = {
        name: {
            "exists": source[f"{name}_audit"]["after_exists"],
            "sha256": source[f"{name}_audit"]["after_sha256"],
            "size": source[f"{name}_audit"]["after_size"],
            "identity": None
            if source[f"{name}_audit"]["after_stat_identity"] is None
            else [str(item) for item in source[f"{name}_audit"]["after_stat_identity"]],
        }
        for name in ("database", "wal", "shm")
    }
    if live != expected_live:
        raise CLIError("brain import receipt live source state does not bind source audits")

    result = _require_exact_keys(
        receipt["result"],
        {"status", "bulkImport", "importedCounts", "finalizedCounts", "parity", "terminalEventId", "recovered"},
        "brain import receipt result",
    )
    if result["status"] != "finalized":
        raise CLIError("brain import receipt result status is not finalized")
    terminal_event_id = result["terminalEventId"]
    if type(terminal_event_id) is not int or terminal_event_id <= 0:
        raise CLIError("brain import receipt terminal event ID is invalid")
    if not _parity_is_complete_and_clean(result["parity"]):
        raise CLIError("brain import receipt parity is incomplete or not clean")
    for name in ("importedCounts", "finalizedCounts"):
        counts = result[name]
        if not isinstance(counts, dict) or any(
            not isinstance(key, str) or type(value) is not int or value < 0 for key, value in counts.items()
        ):
            raise CLIError(f"brain import receipt {name} is invalid")
    expected_finalized = {name: record["target_count"] for name, record in result["parity"].items()}
    if result["finalizedCounts"] != expected_finalized:
        raise CLIError("brain import receipt finalized counts do not match parity")
    bulk = _require_exact_keys(
        result["bulkImport"],
        {"migration_run_id", "source_sha256", "imported", "quarantined", "finalized", "parity", "terminal_event_id", "recovered"},
        "brain import receipt bulk result",
    )
    if result["recovered"] is not False and result["recovered"] is not True:
        raise CLIError("brain import receipt recovered flag is invalid")
    if bulk["source_sha256"] != source["sealed_snapshot_sha256"]:
        raise CLIError("brain import receipt bulk result source hash does not match")
    if (
        type(bulk["migration_run_id"]) is not int
        or bulk["migration_run_id"] < 0
        or (bulk["migration_run_id"] == 0 and result["recovered"] is not True)
    ):
        raise CLIError("brain import receipt bulk migration run ID is invalid")
    timestamps = _require_exact_keys(
        receipt["timestamps"], {"startedAt", "completedAt"}, "brain import receipt timestamps"
    )
    started_at = _utc_datetime(timestamps["startedAt"], "brain import receipt startedAt")
    completed_at = _utc_datetime(timestamps["completedAt"], "brain import receipt completedAt")
    if started_at > completed_at:
        raise CLIError("brain import receipt timestamp order is invalid")


def _validate_commit_semantics(commit: dict[str, Any]) -> None:
    _require_exact_keys(
        commit,
        {
            "receiptSchema", "purpose", "promotable", "receiptPath", "receiptSha256", "release", "runKey",
            "destination", "terminalEventId", "committedAt", "authentication",
        },
        "publication commit",
    )
    if commit["receiptSchema"] != "applypilot.sqlite-to-postgres-publication-commit.v1":
        raise CLIError("publication commit schema is unsupported")
    if commit["purpose"] != "brain-import-publication-commit":
        raise CLIError("publication commit purpose is invalid")
    if commit["promotable"] is not True:
        raise CLIError("publication commit is not promotable")
    if type(commit["terminalEventId"]) is not int or commit["terminalEventId"] <= 0:
        raise CLIError("publication commit terminal event ID is invalid")
    if _SHA256_RE.fullmatch(str(commit["receiptSha256"])) is None:
        raise CLIError("publication commit receipt hash is invalid")
    _parse_utc_timestamp(commit["committedAt"], "publication commit committedAt")


def verify_consumable_import_receipt(receipt_path: Path) -> dict[str, Any]:
    """Verify that a promotable receipt has its authenticated publication commit."""
    receipt_path = _canonical_existing_file(str(receipt_path), "brain import receipt")
    commit_path = _canonical_existing_file(str(_publication_commit_path(receipt_path)), "publication commit")
    receipt_anchor = _capture_path_anchor(receipt_path, "brain import receipt")
    commit_anchor = _capture_path_anchor(commit_path, "publication commit")
    try:
        receipt_content = _stable_read_anchored(receipt_anchor, "brain import receipt")
        receipt = strict_json_loads(receipt_content, "brain import receipt")
        commit = strict_json_loads(
            _stable_read_anchored(commit_anchor, "publication commit"),
            "publication commit",
        )
    except (OSError, RuntimeError) as exc:
        raise CLIError(f"publication commit is missing or invalid: {commit_path}") from exc
    if not isinstance(receipt, dict) or not isinstance(commit, dict):
        raise CLIError("brain import receipt and publication commit must be JSON objects")
    key, key_id = _attestation_key("brain-import")
    try:
        verify_receipt(receipt, key=key, expected_key_id=key_id, label="brain import receipt")
        verify_receipt(commit, key=key, expected_key_id=key_id, label="publication commit")
    except RuntimeError as exc:
        raise CLIError(str(exc)) from exc
    _validate_receipt_semantics(receipt, receipt_path, commit_path)
    _validate_commit_semantics(commit)
    expected_commit = {
        "receiptSchema": "applypilot.sqlite-to-postgres-publication-commit.v1",
        "purpose": "brain-import-publication-commit",
        "receiptPath": str(receipt_path),
        "receiptSha256": _sha256_bytes(receipt_content),
        "release": receipt.get("release"),
        "runKey": receipt.get("command", {}).get("runKey") if isinstance(receipt.get("command"), dict) else None,
        "destination": receipt.get("destination"),
        "terminalEventId": receipt.get("result", {}).get("terminalEventId")
        if isinstance(receipt.get("result"), dict)
        else None,
    }
    for name, expected in expected_commit.items():
        if commit.get(name) != expected:
            raise CLIError(f"publication commit {name} does not bind the receipt")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Canonical absolute path to the live SQLite authority")
    parser.add_argument("--sealed-snapshot", required=True, help="Absolute no-overwrite path for the controlled backup")
    parser.add_argument("--postgres-dsn-env", required=True, help="Name of the environment variable containing the DSN")
    parser.add_argument("--expected-database", required=True)
    parser.add_argument("--expected-system-identifier", required=True)
    parser.add_argument("--expected-database-oid", required=True)
    parser.add_argument(
        "--expected-database-incarnation-id",
        required=True,
        help="Exact UUID of the destination's durable database incarnation singleton",
    )
    parser.add_argument("--mode", required=True, choices=_MODES)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--release-nonce", required=True)
    parser.add_argument("--output", required=True, help="Absolute no-overwrite JSON receipt path")
    parser.add_argument("--writer-freeze-marker", help="Required external evidence file for final-delta-finalize")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true", help="Use the canonical bulk import dry-run path")
    parser.add_argument("--recover-terminal", action="store_true")
    return parser


def execute(argv: list[str] | None = None) -> Path:
    arguments = _parser().parse_args(argv)
    try:
        release_id, release_nonce = validate_release_binding(arguments.release_id, arguments.release_nonce)
    except RuntimeError as exc:
        raise CLIError(str(exc)) from exc
    if arguments.batch_size <= 0:
        raise CLIError("batch size must be positive")
    if arguments.dry_run and arguments.mode != "bulk-import":
        raise CLIError("dry-run is supported only for bulk-import by the canonical API")
    if arguments.mode == "final-delta-finalize" and not arguments.writer_freeze_marker:
        raise CLIError("an explicit writer-freeze marker is required for final-delta-finalize")
    if arguments.mode == "bulk-import" and arguments.writer_freeze_marker:
        raise CLIError("writer-freeze marker is valid only for final-delta-finalize")
    if arguments.recover_terminal and arguments.mode != "final-delta-finalize":
        raise CLIError("terminal recovery is valid only for final-delta-finalize")

    freeze_key = _attestation_key("writer-freeze")
    import_key = _attestation_key("brain-import")
    try:
        assert_separated_keys({"writer-freeze": freeze_key, "brain-import": import_key})
    except RuntimeError as exc:
        raise CLIError(str(exc).replace("release receipt purposes", "writer-freeze and brain-import purposes")) from exc

    source = _canonical_existing_file(arguments.source, "canonical SQLite source")
    snapshot = _canonical_output_path(arguments.sealed_snapshot, "sealed snapshot")
    output = _canonical_output_path(arguments.output, "output receipt")
    commit_path = _canonical_output_path(str(_publication_commit_path(output)), "publication commit")
    dsn = _load_dsn(arguments.postgres_dsn_env)
    _preflight_canonical_import_dsn(dsn)
    source_anchors = {
        "database": _capture_path_anchor(source, "SQLite database"),
        "wal": _capture_path_anchor(Path(f"{source}-wal"), "SQLite WAL"),
        "shm": _capture_path_anchor(Path(f"{source}-shm"), "SQLite SHM"),
    }

    freeze_document: dict[str, Any] | None = None
    freeze_content: bytes | None = None
    freeze_path: Path | None = None
    freeze_anchor: _PathAnchor | None = None
    if arguments.writer_freeze_marker:
        freeze_path = _canonical_existing_file(arguments.writer_freeze_marker, "writer-freeze marker")
        freeze_anchor = _capture_path_anchor(freeze_path, "writer-freeze marker")
        _revalidate_path_anchor(freeze_anchor, "writer-freeze marker")
        freeze_document, freeze_content = _load_freeze_marker(
            freeze_path,
            anchor=freeze_anchor,
            release_id=release_id,
            release_nonce=release_nonce,
            source=source,
        )

    paths = [source, snapshot, output, commit_path, *([] if freeze_path is None else [freeze_path])]
    normalized_paths = {os.path.normcase(os.path.abspath(path)) for path in paths}
    if len(normalized_paths) != len(paths):
        raise CLIError("source, sealed snapshot, output receipt, and freeze marker must be pairwise distinct")
    for path, label in ((output, "output receipt"), (commit_path, "publication commit")):
        if os.path.lexists(path):
            raise CLIError(f"{label} already exists; refusing to overwrite: {path}")
    freeze_identity = (
        None
        if freeze_path is None or freeze_anchor is None
        else _capture_file_state(freeze_path, anchor=freeze_anchor, label="writer-freeze marker")["identity"]
    )

    started_at = _utc_now()
    try:
        sealed = seal_sqlite_snapshot(source, snapshot, source_mode=ONLINE_BACKUP_MODE)
    except Exception as exc:
        raise CLIError(f"controlled SQLite snapshot failed ({type(exc).__name__}): {exc}") from exc
    _validate_sealed_snapshot(sealed, source=source, snapshot=snapshot)
    if freeze_document is not None:
        _validate_freeze_binding(freeze_document, sealed)
        live_state = _assert_live_source_unchanged(source, sealed, source_anchors)
    else:
        live_state = None

    final_summary: ImportSummary | None = None
    try:
        with _connect_postgres(dsn, row_factory=dict_row) as pg:
            _require_canonical_import_identity(pg)
            destination = _destination_identity(pg)
            _assert_destination_binding(
                destination,
                expected_database=arguments.expected_database,
                expected_system_identifier=arguments.expected_system_identifier,
                expected_database_oid=arguments.expected_database_oid,
                expected_database_incarnation_id=arguments.expected_database_incarnation_id,
            )
            durable_destination = _durable_destination_binding(destination)
            run_key = _run_key(release_id, release_nonce, arguments.mode, durable_destination)
            if arguments.recover_terminal:
                bulk_summary = ImportSummary(0, sealed.sha256, {}, {})
                final_summary = recover_finalized_sqlite_to_postgres_import(
                    pg,
                    expected_sha256=sealed.sha256,
                    run_key=run_key,
                    expected_destination_binding=durable_destination,
                )
            else:
                bulk_summary = import_sqlite_to_postgres(
                    pg,
                    snapshot,
                    expected_sha256=sealed.sha256,
                    run_key=run_key,
                    batch_size=arguments.batch_size,
                    dry_run=arguments.dry_run,
                )
            if arguments.mode == "final-delta-finalize" and not arguments.recover_terminal:
                live_state = _assert_live_source_unchanged(source, sealed, source_anchors)
                if freeze_document is None:
                    raise CLIError("writer-freeze marker unavailable before terminal finalization")
                _validate_freeze_time_window(freeze_document)
                final_summary = finalize_sqlite_to_postgres_import(
                    pg,
                    snapshot,
                    expected_sha256=sealed.sha256,
                    run_key=run_key,
                    destination_binding=durable_destination,
                )
                live_state = _assert_live_source_unchanged(source, sealed, source_anchors)
            destination_after = _destination_identity(pg)
            if destination_after != destination:
                raise CLIError("connected PostgreSQL destination identity changed during import")
    except CLIError:
        raise
    except Exception as exc:
        raise CLIError(f"PostgreSQL operation failed ({type(exc).__name__}); connection details withheld") from None

    if freeze_path is not None and freeze_content is not None and freeze_anchor is not None:
        try:
            if _capture_file_state(
                freeze_path, anchor=freeze_anchor, label="writer-freeze marker"
            )["identity"] != freeze_identity:
                raise CLIError("writer-freeze marker was replaced during final-delta-finalize")
            if _stable_read_anchored(freeze_anchor, "writer-freeze marker") != freeze_content:
                raise CLIError("writer-freeze marker changed during final-delta-finalize")
        except RuntimeError as exc:
            raise CLIError(str(exc)) from exc

    completed_at = _utc_now()
    if final_summary is not None and final_summary.finalized and not _parity_is_complete_and_clean(final_summary.parity):
        raise CLIError("finalization lacks complete clean durable per-table parity metadata")
    promotable = bool(
        arguments.mode == "final-delta-finalize"
        and not arguments.dry_run
        and final_summary is not None
        and final_summary.finalized
        and _parity_is_complete_and_clean(final_summary.parity)
    )
    receipt = {
        "receiptSchema": "applypilot.sqlite-to-postgres-import-receipt.v2",
        "purpose": "brain-import",
        "promotable": False,
        "promotionEligible": promotable,
        "publicationProtocol": "authenticated-two-phase-commit-v1" if promotable else None,
        "publicationCommitPath": str(commit_path) if promotable else None,
        "command": {
            "name": "brain-sqlite-to-postgres",
            "version": COMMAND_VERSION,
            "mode": arguments.mode,
            "dryRun": arguments.dry_run,
            "batchSize": arguments.batch_size,
            "dsnEnvironmentVariable": arguments.postgres_dsn_env,
            "runKey": run_key,
        },
        "release": {"id": release_id, "nonce": release_nonce},
        "source": _source_evidence(sealed),
        "destination": destination,
        "writer_freeze": None
        if freeze_document is None or freeze_content is None or freeze_path is None
        else {
            "markerPath": str(freeze_path),
            "markerSha256": _sha256_bytes(freeze_content),
            "frozenAt": freeze_document["frozenAt"],
            "expiresAt": freeze_document["expiresAt"],
            "finalLiveSourceState": live_state,
        },
        "result": {
            "status": "dry-run-non-promotable" if arguments.dry_run else ("finalized" if promotable else "imported-non-promotable"),
            "bulkImport": bulk_summary.as_dict(),
            "importedCounts": dict(bulk_summary.imported),
            "finalizedCounts": {} if final_summary is None else dict(final_summary.imported),
            "parity": {} if final_summary is None else final_summary.parity,
            "terminalEventId": None if final_summary is None else final_summary.terminal_event_id,
            "recovered": bool(final_summary is not None and final_summary.recovered),
        },
        "timestamps": {"startedAt": _timestamp(started_at), "completedAt": _timestamp(completed_at)},
    }
    signed = _signed_receipt(receipt, "brain-import")
    encoded = (json.dumps(signed, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    try:
        atomic_write_no_overwrite(output, encoded)
    except FileExistsError as exc:
        raise CLIError(f"output receipt already exists; refusing to overwrite: {output}") from exc
    except Exception as exc:
        if final_summary is not None and final_summary.finalized:
            raise CLIError(
                "terminal event committed but receipt publication failed; rerun with --recover-terminal"
            ) from exc
        raise CLIError(f"receipt publication failed ({type(exc).__name__})") from exc
    if arguments.mode == "final-delta-finalize":
        try:
            _assert_live_source_unchanged(source, sealed, source_anchors)
            if freeze_path is None or freeze_content is None or freeze_anchor is None:
                raise CLIError("writer-freeze marker unavailable during publication verification")
            if _capture_file_state(
                freeze_path, anchor=freeze_anchor, label="writer-freeze marker"
            )["identity"] != freeze_identity:
                raise CLIError("writer-freeze marker was replaced during receipt publication")
            if _stable_read_anchored(freeze_anchor, "writer-freeze marker") != freeze_content:
                raise CLIError("writer-freeze marker changed during receipt publication")
        except Exception as exc:
            if isinstance(exc, CLIError):
                raise
            raise CLIError(str(exc)) from exc
        commit = _signed_receipt(
            {
                "receiptSchema": "applypilot.sqlite-to-postgres-publication-commit.v1",
                "purpose": "brain-import-publication-commit",
                "promotable": True,
                "receiptPath": str(output),
                "receiptSha256": _sha256_bytes(encoded),
                "release": signed["release"],
                "runKey": run_key,
                "destination": destination,
                "terminalEventId": final_summary.terminal_event_id if final_summary is not None else None,
                "committedAt": _timestamp(_utc_now()),
            },
            "brain-import",
        )
        commit_encoded = (json.dumps(commit, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
        if freeze_document is None:
            raise CLIError("writer-freeze marker unavailable before promotion commit publication")
        _validate_freeze_time_window(freeze_document)
        try:
            atomic_write_no_overwrite(commit_path, commit_encoded)
        except Exception as exc:
            if os.path.lexists(commit_path):
                verify_consumable_import_receipt(output)
                return output
            raise CLIError(
                "receipt is non-consumable because publication commit failed; rerun with --recover-terminal"
            ) from exc
        verify_consumable_import_receipt(output)
    return output


def main(argv: list[str] | None = None) -> int:
    try:
        output = execute(argv)
    except CLIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"receipt: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
