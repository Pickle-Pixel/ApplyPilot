"""Least-privilege PostgreSQL role for remote fleet workers.

This reconciles privileges only inside the connection's current database. Whether
the role can connect to any other database is a deployment/pg_hba concern and must
be controlled separately.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from psycopg import sql

DEFAULT_ROLE = "fleet_worker"
_ROLE_LOCK_KEY = "applypilot:fleet-worker-role:v2"
_CANDIDATE_ROLE_LOCK_KEY = "applypilot:brain-candidate-roles:v1"
_ARTIFACT_AUTHORITY_ROLE_LOCK_KEY = "applypilot:brain-artifact-authority-roles:v1"


class CrossDatabaseInventoryDriftError(RuntimeError):
    """Post-fence cluster drift was forced back to a closed state."""


BRAIN_CANDIDATE_READER_ROLE = "brain_candidate_reader"
BRAIN_CANDIDATE_WRITER_ROLE = "brain_candidate_writer"
BRAIN_ARTIFACT_AUTHORITY_OWNER_ROLE = "brain_artifact_authority_owner"
BRAIN_ARTIFACT_AUTHORITY_WRITER_ROLE = "brain_artifact_authority_writer"
_CANDIDATE_READ_RELATIONS = (
    "brain_authority_scope_state",
    "brain_v4_candidate_decisions",
    "brain_v4_decision_envelopes",
    "brain_immutable_artifact_references",
)
_CANDIDATE_ALL_RELATIONS = (
    "brain_authority_scope_state",
    "brain_authority_transition_events",
    "brain_graph_approval_receipts",
    "brain_v4_candidate_decisions",
    "brain_v4_decision_envelopes",
    "brain_graph_approval_consumptions",
    "brain_immutable_artifact_references",
)
_CANDIDATE_V4_PUBLISH_SIGNATURE = (
    "public.brain_publish_v4_candidate(text,text,text,text,text,bigint,uuid,text,text,text,text,text,bigint)"
)
_CANDIDATE_V5_PUBLISH_SIGNATURE = (
    "public.brain_publish_v5_candidate(text,text,text,text,text,bigint,uuid,text,text,text,text,text,bigint)"
)
@dataclass(frozen=True)
class AclRegrant:
    """One parsed, non-PUBLIC ACL dependency to preserve during hardening."""

    object_kind: str
    qualified_name: str
    privileges: tuple[str, ...]
    grantee: str


@dataclass(frozen=True)
class RegrantManifest:
    """Operator-approved principals and structured grants for database hardening."""

    database_owner_role: str
    controller_roles: tuple[str, ...]
    verifier_roles: tuple[str, ...]
    retired_admin_roles: tuple[str, ...]
    infrastructure_superuser_roles: tuple[str, ...] = ()
    expected_service_roles: tuple[str, ...] = ()
    regrants: tuple[AclRegrant, ...] = ()


@dataclass(frozen=True)
class RoleReconciliationReceipt:
    role_name: str
    worker_id: str
    contract: str
    connect_allowlist: tuple[str, ...]
    effective_connect_grantees: tuple[Mapping[str, Any], ...]
    inventory: Mapping[str, Any]
    rollback_sql: str
    credential_forward_reconcile_required: bool
    reconciled_at: str


@dataclass(frozen=True)
class CandidateRoleReconciliationReceipt:
    reader_role: str
    writer_role: str
    reconciled_at: str


@dataclass(frozen=True)
class DurableEvidencePaths:
    """Prevalidated exclusive paths for internally written preparation evidence."""

    preparation_receipt_path: Path
    rollback_sql_path: Path
    authentication_key: bytes
    authentication_key_id: str
    _parent_identities: tuple[tuple[str, int, int], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        concrete_path_type = type(Path())
        for field_name in ("preparation_receipt_path", "rollback_sql_path"):
            value = getattr(self, field_name)
            if type(value) is not concrete_path_type:
                raise TypeError(f"{field_name} must be a concrete pathlib.Path")
            if not value.is_absolute():
                raise ValueError(f"{field_name} must be absolute")
        if self.preparation_receipt_path == self.rollback_sql_path:
            raise ValueError("preparation receipt and rollback SQL paths must be distinct")
        if type(self.authentication_key) is not bytes or len(self.authentication_key) < 32:
            raise ValueError("evidence authentication_key must be at least 32 bytes")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", self.authentication_key_id):
            raise ValueError("evidence authentication_key_id is invalid")
        identities = tuple(
            _evidence_parent_identity(path.parent)
            for path in (self.preparation_receipt_path, self.rollback_sql_path)
        )
        object.__setattr__(self, "_parent_identities", identities)


def _evidence_parent_identity(parent: Path) -> tuple[str, int, int]:
    metadata = os.lstat(parent)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"durable evidence parent must be a real directory: {parent}")
    resolved = parent.resolve(strict=True)
    if resolved != parent:
        raise RuntimeError(f"durable evidence parent path must be canonical and contain no symlinks: {parent}")
    return (str(resolved), metadata.st_dev, metadata.st_ino)


def _fsync_parent_directory(parent: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_authentication"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def authenticate_evidence_receipt(
    receipt: Mapping[str, Any],
    *,
    authentication_key: bytes,
    authentication_key_id: str,
) -> dict[str, Any]:
    """Return a receipt authenticated by a separately trusted operator key."""
    if type(authentication_key) is not bytes or len(authentication_key) < 32:
        raise ValueError("evidence authentication key must be at least 32 bytes")
    signed = dict(receipt)
    signed["receipt_authentication"] = {
        "algorithm": "hmac-sha256",
        "key_id": authentication_key_id,
        "digest": hmac.new(authentication_key, _canonical_receipt_bytes(signed), hashlib.sha256).hexdigest(),
    }
    return signed


def verify_evidence_receipt(
    receipt: Mapping[str, Any],
    *,
    authentication_key: bytes,
    expected_key_id: str | None = None,
) -> None:
    authentication = receipt.get("receipt_authentication")
    if not isinstance(authentication, Mapping):
        raise ValueError("legacy or unsigned evidence receipt is not accepted")
    if authentication.get("algorithm") != "hmac-sha256":
        raise ValueError("unsupported evidence receipt authentication algorithm")
    key_id = authentication.get("key_id")
    digest = authentication.get("digest")
    if not isinstance(key_id, str) or not isinstance(digest, str):
        raise ValueError("evidence receipt authentication metadata is malformed")
    if expected_key_id is not None and key_id != expected_key_id:
        raise ValueError("evidence receipt key id does not match the trusted operator key")
    expected = hmac.new(authentication_key, _canonical_receipt_bytes(receipt), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, digest):
        raise ValueError("evidence receipt authentication failed")


def _write_exclusive_fsync(
    path: Path,
    payload: bytes,
    *,
    expected_parent_identity: tuple[str, int, int],
) -> tuple[int, int]:
    if _evidence_parent_identity(path.parent) != expected_parent_identity:
        raise RuntimeError(f"durable evidence parent directory changed after validation: {path.parent}")
    if os.path.lexists(path):
        raise FileExistsError(f"durable evidence path already exists: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor: int | None = None
    if os.open in os.supports_dir_fd:
        parent_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        parent_metadata = os.fstat(parent_descriptor)
        if (parent_metadata.st_dev, parent_metadata.st_ino) != expected_parent_identity[1:]:
            os.close(parent_descriptor)
            raise RuntimeError("durable evidence parent changed while being opened")
        descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_descriptor)
    else:
        descriptor = os.open(path, flags, 0o600)
    metadata = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if parent_descriptor is not None:
            os.fsync(parent_descriptor)
        else:
            _fsync_parent_directory(path.parent)
    except BaseException:
        os.close(descriptor)
        _remove_created_evidence(
            path,
            expected_identity=(metadata.st_dev, metadata.st_ino),
            expected_parent_identity=expected_parent_identity,
        )
        raise
    else:
        os.close(descriptor)
    finally:
        if parent_descriptor is not None:
            os.close(parent_descriptor)
    return metadata.st_dev, metadata.st_ino


def _remove_created_evidence(
    path: Path,
    *,
    expected_identity: tuple[int, int],
    expected_parent_identity: tuple[str, int, int],
) -> None:
    if _evidence_parent_identity(path.parent) != expected_parent_identity:
        raise RuntimeError("cannot safely clean partial evidence after parent substitution")
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or (metadata.st_dev, metadata.st_ino) != expected_identity:
        raise RuntimeError("cannot safely clean partial evidence after leaf substitution")
    if os.unlink in os.supports_dir_fd:
        descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            parent_metadata = os.fstat(descriptor)
            if (parent_metadata.st_dev, parent_metadata.st_ino) != expected_parent_identity[1:]:
                raise RuntimeError("cannot safely clean partial evidence after parent substitution")
            os.unlink(path.name, dir_fd=descriptor)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    else:
        path.unlink()
        _fsync_parent_directory(path.parent)


def _validate_evidence_paths(evidence_paths: DurableEvidencePaths) -> None:
    if type(evidence_paths) is not DurableEvidencePaths:
        raise TypeError("evidence_paths must be an exact DurableEvidencePaths instance")
    collisions = [
        path
        for path in (
            evidence_paths.preparation_receipt_path,
            evidence_paths.rollback_sql_path,
        )
        if os.path.lexists(path)
    ]
    if collisions:
        raise FileExistsError("durable evidence paths already exist: " + ", ".join(map(str, collisions)))
    missing_parents = sorted(
        {
            path.parent
            for path in (
                evidence_paths.preparation_receipt_path,
                evidence_paths.rollback_sql_path,
            )
            if not path.parent.is_dir()
        },
        key=str,
    )
    if missing_parents:
        raise RuntimeError("durable evidence parent directories do not exist: " + ", ".join(map(str, missing_parents)))


def _write_preparation_evidence(
    evidence_paths: DurableEvidencePaths,
    *,
    inventory: Mapping[str, Any],
    rollback_sql: str,
) -> None:
    rollback_bytes = rollback_sql.encode("utf-8")
    receipt = {
        "status": "prepared_before_database_mutation",
        "prepared_at": inventory["prepared_at"],
        "database_name": inventory["database_name"],
        "inventory": inventory,
        "rollback_sql_path": str(evidence_paths.rollback_sql_path),
        "rollback_sql_sha256": hashlib.sha256(rollback_bytes).hexdigest(),
        "in_doubt": True,
        "escalation_required": True,
    }
    for key in (
        "atomic_bootstrap",
        "automatic_rollback_supported",
        "commit_outcome_on_interruption",
        "legacy_rollback_sql_recovers_v1_v4",
        "legacy_rollback_sql_recovers_v1_v5",
        "rollback_mode",
        "topology",
    ):
        if key in inventory:
            receipt[key] = inventory[key]
    receipt = authenticate_evidence_receipt(
        receipt,
        authentication_key=evidence_paths.authentication_key,
        authentication_key_id=evidence_paths.authentication_key_id,
    )
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    rollback_identity = _write_exclusive_fsync(
        evidence_paths.rollback_sql_path,
        rollback_bytes,
        expected_parent_identity=evidence_paths._parent_identities[1],
    )
    try:
        _write_exclusive_fsync(
            evidence_paths.preparation_receipt_path,
            receipt_bytes,
            expected_parent_identity=evidence_paths._parent_identities[0],
        )
    except BaseException:
        _remove_created_evidence(
            evidence_paths.rollback_sql_path,
            expected_identity=rollback_identity,
            expected_parent_identity=evidence_paths._parent_identities[1],
        )
        raise


@dataclass(frozen=True)
class RuntimePrincipal:
    session_user: str
    current_user: str
    worker_id: str
    contract: str


@dataclass(frozen=True)
class BootstrapTopology:
    """One-time provider-admin handoff to permanent non-superuser roles."""

    database_owner_role: str
    controller_role: str
    verifier_role: str
    migrator_role: str
    retired_admin_roles: tuple[str, ...]
    expected_service_roles: tuple[str, ...] = ()
    infrastructure_superuser_roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class BootstrapReceipt:
    database_name: str
    session_user: str
    topology: Mapping[str, Any]
    inventory: Mapping[str, Any]
    effective_connect_grantees: tuple[Mapping[str, Any], ...]
    rollback_sql: str
    escalation_required: bool
    bootstrapped_at: str


_COMMON_FUNCTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fleet_worker_identity", ()),
    ("fleet_worker_admission_snapshot", ()),
    ("fleet_worker_schema_contract", ()),
    ("fleet_worker_heartbeat", ("JSONB",)),
    ("fleet_worker_runtime_state", ("TEXT",)),
    ("fleet_worker_version_status", ("TEXT", "TEXT")),
    ("fleet_worker_ack_command", ("BIGINT", "TEXT")),
    ("fleet_worker_agent_blocks", ()),
)
_TRANSITION_FUNCTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fleet_worker_requeue", ("TEXT", "TEXT", "TEXT", "TEXT")),
    ("fleet_worker_park_infrastructure", ("TEXT", "TEXT", "TEXT")),
    ("fleet_worker_terminalize", ("TEXT", "TEXT", "TEXT", "TEXT", "TEXT", "TEXT", "JSONB")),
    ("fleet_worker_park", ("TEXT", "TEXT", "TEXT", "TEXT", "INTEGER", "JSONB")),
    ("fleet_worker_mark_browser_interaction", ()),
    ("fleet_worker_attempt_create", ("TEXT", "TEXT", "JSONB")),
    ("fleet_worker_attempt_transition", ("UUID", "TEXT", "TEXT", "JSONB")),
)
_APPLY_FUNCTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fleet_worker_lease_ats", ("TEXT", "TEXT", "INTEGER", "TEXT", "INTEGER")),
    *_TRANSITION_FUNCTIONS,
    ("fleet_worker_claim_liveness", ()),
    ("fleet_worker_write_liveness", ("TEXT", "TEXT", "TEXT")),
    ("fleet_worker_record_agent_wall", ("TEXT", "TIMESTAMPTZ")),
    ("fleet_worker_evaluate_agent_budget", ("JSONB", "INTEGER", "INTEGER")),
    ("fleet_worker_otp_request", ("TEXT", "TEXT", "TEXT", "INTEGER")),
    ("fleet_worker_otp_wait", ("BIGINT",)),
    ("fleet_worker_otp_consume", ("BIGINT",)),
)
_CONTRACT_FUNCTIONS = {
    "apply": _APPLY_FUNCTIONS,
    "linkedin": (
        ("fleet_worker_lease_linkedin", ("TEXT", "TEXT", "TEXT", "INTEGER", "TEXT")),
        *_TRANSITION_FUNCTIONS,
        ("fleet_worker_record_agent_wall", ("TEXT", "TIMESTAMPTZ")),
        ("fleet_worker_otp_request", ("TEXT", "TEXT", "TEXT", "INTEGER")),
        ("fleet_worker_otp_wait", ("BIGINT",)),
        ("fleet_worker_otp_consume", ("BIGINT",)),
    ),
    "compute": (
        ("fleet_worker_lease_compute", ()),
        ("fleet_worker_complete_compute", ("TEXT", "TEXT", "TEXT", "JSONB", "JSONB")),
    ),
    "discovery": (
        ("fleet_worker_lease_search", ()),
        ("fleet_worker_complete_search", ("TEXT", "JSONB", "TEXT")),
    ),
}
_CONTRACT_TYPES = {
    "apply": ("apply_queue", "_apply_queue"),
    "linkedin": ("linkedin_queue", "_linkedin_queue"),
    "compute": (),
    "discovery": (),
}
_REQUIRED_FUNCTION_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "fleet_config": (
        "id",
        "paused",
        "ats_paused",
        "ats_apply_mode",
        "canary_enabled",
        "canary_remaining",
        "linkedin_apply_mode",
        "linkedin_canary_enabled",
        "linkedin_canary_remaining",
        "ats_policy_version",
        "linkedin_policy_version",
    ),
    "apply_queue": (
        "url",
        "lane",
        "status",
        "lease_owner",
        "attempts",
        "approved_batch",
        "decision_id",
        "policy_version",
        "decision_action",
        "qualification_verdict",
        "qualification_score",
        "qualification_floor",
        "decision_expires_at",
        "score",
        "final_score",
        "worker_lease_id",
    ),
    "linkedin_queue": (
        "url",
        "lane",
        "status",
        "lease_owner",
        "attempts",
        "approved_batch",
        "decision_id",
        "policy_version",
        "decision_action",
        "qualification_verdict",
        "qualification_score",
        "qualification_floor",
        "decision_expires_at",
        "score",
        "final_score",
        "worker_lease_id",
    ),
    "fleet_worker_lease_ledger": (
        "lease_id",
        "lane",
        "url",
        "worker_id",
        "queue_attempt",
        "policy_version",
        "home_ip",
        "target_host",
        "canary_charged",
        "canary_capacity_before",
        "canary_exhausted",
        "refunded_at",
        "state",
    ),
}


def _row_value(row, key: str, index: int = 0):
    return row[key] if isinstance(row, Mapping) else row[index]


def _identifiers(names: Iterable[str]) -> sql.Composed:
    return sql.SQL(", ").join(sql.Identifier(name) for name in names)


def _function_identifier(name: str, argument_types: tuple[str, ...]) -> sql.Composed:
    return sql.SQL("{}.{}({})").format(
        sql.Identifier("public"),
        sql.Identifier(name),
        sql.SQL(", ").join(sql.SQL(argument_type) for argument_type in argument_types),
    )


def _user_schemas(cur) -> tuple[str, ...]:
    cur.execute(
        "SELECT nspname FROM pg_namespace WHERE nspname <> 'information_schema' AND nspname !~ '^pg_' ORDER BY nspname"
    )
    return tuple(_row_value(row, "nspname") for row in cur.fetchall())


def _transfer_application_ownership(cur, *, new_owner_role: str, apply: bool = True) -> tuple[dict[str, Any], ...]:
    """Transfer only application objects, never cluster/system-owned objects."""
    new_owner = sql.Identifier(new_owner_role)
    transferred: list[dict[str, Any]] = []
    schemas = _user_schemas(cur)

    cur.execute(
        "SELECT n.nspname,c.relname,c.relkind,owner.rolname AS owner_name "
        "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "JOIN pg_roles owner ON owner.oid=c.relowner "
        "WHERE n.nspname=ANY(%s) AND c.relkind=ANY(%s) "
        "ORDER BY CASE c.relkind WHEN 'r' THEN 0 WHEN 'p' THEN 0 WHEN 'v' THEN 1 "
        "WHEN 'm' THEN 2 WHEN 'f' THEN 3 WHEN 'S' THEN 4 ELSE 5 END,n.nspname,c.relname",
        (list(schemas), ["r", "p", "v", "m", "f", "S"]),
    )
    relation_kinds = {
        "r": "TABLE",
        "p": "TABLE",
        "v": "VIEW",
        "m": "MATERIALIZED VIEW",
        "f": "FOREIGN TABLE",
        "S": "SEQUENCE",
    }
    for row in cur.fetchall():
        owner_name = _row_value(row, "owner_name", 3)
        if owner_name == new_owner_role:
            continue
        object_kind = relation_kinds[_row_value(row, "relkind", 2)]
        namespace = _row_value(row, "nspname")
        object_name = _row_value(row, "relname", 1)
        if apply:
            cur.execute(
                sql.SQL("ALTER {} {} OWNER TO {}").format(
                    sql.SQL(object_kind), sql.Identifier(namespace, object_name), new_owner
                )
            )
        transferred.append(
            {
                "object_kind": object_kind.lower(),
                "qualified_name": f"{namespace}.{object_name}",
                "owner_before": owner_name,
                "schema_name": namespace,
                "object_name": object_name,
            }
        )

    cur.execute(
        "SELECT n.nspname,p.proname,p.prokind,pg_get_function_identity_arguments(p.oid) AS args,"
        "owner.rolname AS owner_name FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "JOIN pg_roles owner ON owner.oid=p.proowner WHERE n.nspname=ANY(%s) "
        "ORDER BY n.nspname,p.proname,args",
        (list(schemas),),
    )
    for row in cur.fetchall():
        owner_name = _row_value(row, "owner_name", 4)
        if owner_name == new_owner_role:
            continue
        prokind = _row_value(row, "prokind", 2)
        if prokind == "a":
            raise RuntimeError("bootstrap cannot transfer user-defined aggregate ownership safely")
        routine_kind = "PROCEDURE" if prokind == "p" else "FUNCTION"
        namespace = _row_value(row, "nspname")
        routine_name = _row_value(row, "proname", 1)
        arguments = _row_value(row, "args", 3)
        identity = sql.SQL("{}.{}({})").format(
            sql.Identifier(namespace), sql.Identifier(routine_name), sql.SQL(arguments)
        )
        if apply:
            cur.execute(sql.SQL("ALTER {} {} OWNER TO {}").format(sql.SQL(routine_kind), identity, new_owner))
        transferred.append(
            {
                "object_kind": routine_kind.lower(),
                "qualified_name": f"{namespace}.{routine_name}({arguments})",
                "owner_before": owner_name,
                "schema_name": namespace,
                "object_name": routine_name,
                "arguments": arguments,
            }
        )

    cur.execute(
        "SELECT n.nspname,t.typname,t.typtype,owner.rolname AS owner_name "
        "FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace "
        "JOIN pg_roles owner ON owner.oid=t.typowner "
        "WHERE n.nspname=ANY(%s) AND t.typrelid=0 AND t.typelem=0 "
        "AND t.typtype IN ('c','d','e','r','m') ORDER BY n.nspname,t.typname",
        (list(schemas),),
    )
    for row in cur.fetchall():
        owner_name = _row_value(row, "owner_name", 3)
        if owner_name == new_owner_role:
            continue
        object_kind = "DOMAIN" if _row_value(row, "typtype", 2) == "d" else "TYPE"
        namespace = _row_value(row, "nspname")
        type_name = _row_value(row, "typname", 1)
        if apply:
            cur.execute(
                sql.SQL("ALTER {} {} OWNER TO {}").format(
                    sql.SQL(object_kind), sql.Identifier(namespace, type_name), new_owner
                )
            )
        transferred.append(
            {
                "object_kind": object_kind.lower(),
                "qualified_name": f"{namespace}.{type_name}",
                "owner_before": owner_name,
                "schema_name": namespace,
                "object_name": type_name,
            }
        )

    cur.execute(
        "SELECT n.nspname,o.oprname FROM pg_operator o JOIN pg_namespace n ON n.oid=o.oprnamespace "
        "WHERE n.nspname=ANY(%s) ORDER BY n.nspname,o.oprname",
        (list(schemas),),
    )
    unsupported_operators = [f"{_row_value(row, 'nspname')}.{_row_value(row, 'oprname', 1)}" for row in cur.fetchall()]
    if unsupported_operators:
        raise RuntimeError("bootstrap cannot transfer user-defined operators: " + ", ".join(unsupported_operators))

    cur.execute(
        "SELECT n.nspname,owner.rolname AS owner_name FROM pg_namespace n "
        "JOIN pg_roles owner ON owner.oid=n.nspowner WHERE n.nspname=ANY(%s) ORDER BY n.nspname",
        (list(schemas),),
    )
    for row in cur.fetchall():
        owner_name = _row_value(row, "owner_name", 1)
        schema_name = _row_value(row, "nspname")
        schema = sql.Identifier(schema_name)
        if apply:
            if owner_name != new_owner_role:
                cur.execute(sql.SQL("ALTER SCHEMA {} OWNER TO {}").format(schema, new_owner))
            # Ownership privileges are not inherited by members of the owner role.
            cur.execute(sql.SQL("GRANT USAGE, CREATE ON SCHEMA {} TO {}").format(schema, new_owner))
        if owner_name == new_owner_role:
            continue
        transferred.append(
            {
                "object_kind": "schema",
                "qualified_name": schema_name,
                "owner_before": owner_name,
                "schema_name": schema_name,
            }
        )
    return tuple(transferred)


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$.-]{0,62}$")
_REGRANT_PRIVILEGES = {
    "schema": {"USAGE", "CREATE"},
    "table": {"SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER"},
    "sequence": {"USAGE", "SELECT", "UPDATE"},
    "function": {"EXECUTE"},
    "type": {"USAGE"},
    "default_functions": {"EXECUTE"},
    "default_types": {"USAGE"},
}


def _reject_unsafe_text(value: str, *, field: str) -> None:
    if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{field} contains empty or control-character data")
    if any(marker in value for marker in (";", "--", "/*", "*/", "#", "\\")):
        raise ValueError(f"{field} contains SQL/comment syntax")


def _qualified_identifiers(value: str, *, count: tuple[int, ...]) -> tuple[str, ...]:
    _reject_unsafe_text(value, field="qualified_name")
    parts = tuple(value.split("."))
    if len(parts) not in count or any(not _IDENTIFIER.fullmatch(part) for part in parts):
        raise ValueError(f"invalid structured qualified_name: {value!r}")
    return parts


def _validate_regrants(cur, manifest: RegrantManifest) -> tuple[dict[str, Any], ...]:
    allowed_grantees = set(
        (*manifest.controller_roles, *manifest.verifier_roles, *manifest.expected_service_roles)
    ) - set(manifest.retired_admin_roles)
    normalized: list[dict[str, Any]] = []
    for grant in manifest.regrants:
        kind = grant.object_kind.lower()
        if kind not in _REGRANT_PRIVILEGES:
            raise ValueError(f"unsupported regrant object_kind: {grant.object_kind!r}")
        _reject_unsafe_text(grant.grantee, field="grantee")
        if grant.grantee.upper() == "PUBLIC":
            raise ValueError("structured regrants to PUBLIC are forbidden")
        if grant.grantee not in allowed_grantees or not _IDENTIFIER.fullmatch(grant.grantee):
            raise ValueError(f"regrant grantee is not an approved service role: {grant.grantee!r}")
        privileges = tuple(dict.fromkeys(privilege.upper() for privilege in grant.privileges))
        if not privileges or not set(privileges) <= _REGRANT_PRIVILEGES[kind]:
            raise ValueError(f"invalid privileges for {kind}: {grant.privileges!r}")
        if kind == "schema":
            _qualified_identifiers(grant.qualified_name, count=(1,))
        elif kind in {"table", "sequence", "type"}:
            _qualified_identifiers(grant.qualified_name, count=(2,))
        elif kind == "function":
            _reject_unsafe_text(grant.qualified_name, field="qualified_name")
            if not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_$.-]*\.[A-Za-z_][A-Za-z0-9_$.-]*\([A-Za-z0-9_ ,.\[\]]*\)", grant.qualified_name
            ):
                raise ValueError(f"invalid function qualified_name: {grant.qualified_name!r}")
            cur.execute("SELECT to_regprocedure(%s) AS oid", (grant.qualified_name,))
            if _row_value(cur.fetchone(), "oid") is None:
                raise RuntimeError(f"regrant function does not exist: {grant.qualified_name}")
        else:
            _qualified_identifiers(grant.qualified_name, count=(1, 2))
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (grant.grantee,))
        if cur.fetchone() is None:
            raise RuntimeError(f"regrant grantee role does not exist: {grant.grantee}")
        normalized.append(
            {
                "object_kind": kind,
                "qualified_name": grant.qualified_name,
                "privileges": privileges,
                "grantee": grant.grantee,
            }
        )
    return tuple(normalized)


def _apply_regrants(cur, regrants: tuple[dict[str, Any], ...]) -> None:
    for grant in regrants:
        kind = grant["object_kind"]
        privileges = sql.SQL(", ").join(sql.SQL(value) for value in grant["privileges"])
        grantee = sql.Identifier(grant["grantee"])
        if kind == "schema":
            target = sql.Identifier(*grant["qualified_name"].split("."))
            cur.execute(sql.SQL("GRANT {} ON SCHEMA {} TO {}").format(privileges, target, grantee))
        elif kind in {"table", "sequence", "type"}:
            target = sql.Identifier(*grant["qualified_name"].split("."))
            cur.execute(sql.SQL("GRANT {} ON {} {} TO {}").format(privileges, sql.SQL(kind.upper()), target, grantee))
        elif kind == "function":
            cur.execute(
                "SELECT to_regprocedure(%s)::regprocedure::text AS identity",
                (grant["qualified_name"],),
            )
            identity = _row_value(cur.fetchone(), "identity")
            cur.execute(sql.SQL("GRANT {} ON FUNCTION {} TO {}").format(privileges, sql.SQL(identity), grantee))
        else:
            owner, *schema_name = grant["qualified_name"].split(".")
            object_type = "FUNCTIONS" if kind == "default_functions" else "TYPES"
            statement = sql.SQL("ALTER DEFAULT PRIVILEGES FOR ROLE {} ").format(sql.Identifier(owner))
            if schema_name:
                statement += sql.SQL("IN SCHEMA {} ").format(sql.Identifier(schema_name[0]))
            statement += sql.SQL("GRANT {} ON {} TO {}").format(privileges, sql.SQL(object_type), grantee)
            cur.execute(statement)


def _database_inventory(cur, *, role_name: str, manifest: RegrantManifest) -> dict[str, Any]:
    controller_roles = tuple(dict.fromkeys(manifest.controller_roles))
    verifier_roles = tuple(dict.fromkeys(manifest.verifier_roles))
    retired_admin_roles = tuple(dict.fromkeys(manifest.retired_admin_roles))
    break_glass_roles = tuple(dict.fromkeys(manifest.infrastructure_superuser_roles))
    if not controller_roles or not verifier_roles:
        raise RuntimeError("explicit regrant manifest requires controller and verifier roles")
    if not break_glass_roles:
        raise RuntimeError("explicit regrant manifest requires at least one break-glass superuser role")
    if not _IDENTIFIER.fullmatch(manifest.database_owner_role):
        raise ValueError("database_owner_role must be one plain role identifier")
    if set(retired_admin_roles) & set(
        (manifest.database_owner_role, *controller_roles, *verifier_roles, *break_glass_roles)
    ):
        raise RuntimeError("retired admin roles must be disjoint from owner/controller/verifier/break-glass roles")
    if set(break_glass_roles) & set((manifest.database_owner_role, *controller_roles, *verifier_roles)):
        raise RuntimeError("break-glass roles must be disjoint from owner/controller/verifier roles")
    cur.execute(
        "SELECT session_user AS session_name,current_user AS current_name,current_database() AS database_name,"
        "owner.rolname AS owner_name "
        "FROM pg_database d JOIN pg_roles owner ON owner.oid=d.datdba "
        "WHERE d.datname=current_database()"
    )
    database_row = cur.fetchone()
    migration_session = _row_value(database_row, "session_name")
    current_user = _row_value(database_row, "current_name", 1)
    prior_owner = _row_value(database_row, "owner_name", 3)
    if migration_session != current_user:
        raise RuntimeError("offline role provisioning forbids SET ROLE")
    if migration_session not in break_glass_roles:
        raise RuntimeError("fleet worker role reconciliation requires an explicit offline provider-admin identity")
    cur.execute(
        "SELECT rolcanlogin,rolsuper,rolcreaterole FROM pg_roles WHERE rolname=%s",
        (migration_session,),
    )
    provisioner = cur.fetchone()
    if provisioner is None or not all(
        _row_value(provisioner, name, index) for index, name in enumerate(("rolcanlogin", "rolsuper", "rolcreaterole"))
    ):
        raise RuntimeError("offline provider-admin must be LOGIN SUPERUSER CREATEROLE")
    if prior_owner != manifest.database_owner_role and prior_owner not in retired_admin_roles:
        raise RuntimeError(f"current database owner {prior_owner!r} must be the dedicated owner or explicitly retired")

    cur.execute(
        "SELECT role_name FROM public.fleet_worker_principals WHERE role_name<>%s ORDER BY role_name",
        (role_name,),
    )
    mapped_roles = tuple(_row_value(row, "role_name") for row in cur.fetchall())
    connect_allowlist = tuple(
        dict.fromkeys(
            (
                manifest.database_owner_role,
                *controller_roles,
                *verifier_roles,
                *break_glass_roles,
                *mapped_roles,
                role_name,
            )
        )
    )
    if any(name in {"PUBLIC", "fleet_worker"} for name in connect_allowlist):
        raise RuntimeError("CONNECT allowlist forbids PUBLIC and the shared fleet_worker role")

    cur.execute(
        "SELECT rolname FROM pg_catalog.pg_roles WHERE rolname=ANY(%s) ORDER BY rolname",
        (list(connect_allowlist),),
    )
    existing = {_row_value(row, "rolname") for row in cur.fetchall()}
    # The target worker role is allowed to be created by this transaction.
    missing = sorted(set(connect_allowlist) - existing - {role_name})
    if missing:
        raise RuntimeError("regrant manifest references missing roles: " + ", ".join(missing))
    cur.execute(
        "SELECT rolname,rolcanlogin,rolsuper FROM pg_roles WHERE rolname=ANY(%s) ORDER BY rolname",
        (list(connect_allowlist),),
    )
    allowlist_attributes = {
        _row_value(row, "rolname"): (
            _row_value(row, "rolcanlogin", 1),
            _row_value(row, "rolsuper", 2),
        )
        for row in cur.fetchall()
    }
    owner_attributes = allowlist_attributes.get(manifest.database_owner_role)
    if owner_attributes != (True, False):
        raise RuntimeError("dedicated database owner role must be LOGIN and NOSUPERUSER")
    if not any(allowlist_attributes.get(name, (False, False))[0] for name in controller_roles):
        raise RuntimeError("at least one explicit controller role must be LOGIN")
    invalid_controllers = sorted(name for name in controller_roles if allowlist_attributes.get(name, (False, False))[1])
    if invalid_controllers:
        raise RuntimeError("controller roles must be NOSUPERUSER: " + ", ".join(invalid_controllers))
    invalid_break_glass = sorted(name for name in break_glass_roles if allowlist_attributes.get(name) != (True, True))
    if invalid_break_glass:
        raise RuntimeError("break-glass roles must remain LOGIN SUPERUSER: " + ", ".join(invalid_break_glass))
    cur.execute(
        "SELECT rolname FROM pg_roles WHERE rolname=ANY(%s) ORDER BY rolname",
        (list(retired_admin_roles),),
    )
    existing_retired = {_row_value(row, "rolname") for row in cur.fetchall()}
    missing_retired = sorted(set(retired_admin_roles) - existing_retired)
    if missing_retired:
        raise RuntimeError("retired admin roles do not exist: " + ", ".join(missing_retired))
    structured_regrants = _validate_regrants(cur, manifest)

    cur.execute(
        "SELECT DISTINCT usename FROM pg_catalog.pg_stat_activity "
        "WHERE datname=current_database() AND pid<>pg_backend_pid() AND usename IS NOT NULL "
        "ORDER BY usename"
    )
    active_services = tuple(_row_value(row, "usename") for row in cur.fetchall())
    expected_services = set(manifest.expected_service_roles)
    unknown_services = sorted(
        set(active_services) - expected_services - set(connect_allowlist) - set(retired_admin_roles)
    )
    if unknown_services:
        raise RuntimeError("unknown database service principals are active: " + ", ".join(unknown_services))
    missing_services = sorted(expected_services - set(active_services))
    if missing_services:
        raise RuntimeError("expected database service principals are not active: " + ", ".join(missing_services))

    cur.execute(
        "SELECT DISTINCT grantee.rolname FROM pg_catalog.pg_database d "
        "CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(d.datacl,acldefault('d',d.datdba))) acl "
        "JOIN pg_catalog.pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE d.datname=current_database() AND acl.privilege_type='CONNECT' "
        "AND acl.grantee<>d.datdba "
        "ORDER BY grantee.rolname"
    )
    explicit_connect_roles = tuple(_row_value(row, "rolname") for row in cur.fetchall())
    unknown_connect = sorted(set(explicit_connect_roles) - set(connect_allowlist) - set(retired_admin_roles))
    if unknown_connect:
        raise RuntimeError(
            "unknown explicit CONNECT dependencies require manifest resolution: " + ", ".join(unknown_connect)
        )

    cur.execute(
        "SELECT r.rolname AS owner, COALESCE(n.nspname,'*') AS schema_name, "
        "d.defaclobjtype,CASE WHEN acl.grantee=0 THEN 'PUBLIC' ELSE grantee.rolname END AS grantee,"
        "acl.privilege_type,acl.is_grantable "
        "FROM pg_catalog.pg_default_acl d JOIN pg_catalog.pg_roles r ON r.oid=d.defaclrole "
        "LEFT JOIN pg_catalog.pg_namespace n ON n.oid=d.defaclnamespace "
        "CROSS JOIN LATERAL pg_catalog.aclexplode(d.defaclacl) acl "
        "LEFT JOIN pg_catalog.pg_roles grantee ON grantee.oid=acl.grantee "
        "ORDER BY r.rolname,schema_name,d.defaclobjtype,grantee,acl.privilege_type"
    )
    default_acls = tuple(
        {
            "owner": _row_value(row, "owner"),
            "schema": _row_value(row, "schema_name", 1),
            "object_type": _row_value(row, "defaclobjtype", 2),
            "grantee": _row_value(row, "grantee", 3),
            "privilege": _row_value(row, "privilege_type", 4),
            "grantable": _row_value(row, "is_grantable", 5),
        }
        for row in cur.fetchall()
    )
    return {
        "connect_allowlist": connect_allowlist,
        "database_name": _row_value(database_row, "database_name", 1),
        "database_owner_before": prior_owner,
        "database_owner_after": manifest.database_owner_role,
        "retired_admin_roles": retired_admin_roles,
        "infrastructure_superuser_roles": break_glass_roles,
        "active_service_roles": active_services,
        "explicit_connect_roles": explicit_connect_roles,
        "default_acls": default_acls,
        "regrants": structured_regrants,
    }


def _target_role_snapshot(cur, *, role_name: str, role_exists: bool) -> dict[str, Any]:
    """Capture only secret-free state needed to reverse mapped-role reconciliation."""
    attributes = None
    memberships: tuple[dict[str, Any], ...] = ()
    granted_memberships: tuple[dict[str, Any], ...] = ()
    principal = None
    if role_exists:
        cur.execute(
            "SELECT rolcanlogin,rolinherit,rolsuper,rolcreatedb,rolcreaterole,rolreplication,"
            "rolbypassrls,rolconnlimit,rolvaliduntil::text AS rolvaliduntil "
            "FROM pg_catalog.pg_roles WHERE rolname=%s",
            (role_name,),
        )
        row = cur.fetchone()
        attributes = {
            name: _row_value(row, name, index)
            for index, name in enumerate(
                (
                    "rolcanlogin",
                    "rolinherit",
                    "rolsuper",
                    "rolcreatedb",
                    "rolcreaterole",
                    "rolreplication",
                    "rolbypassrls",
                    "rolconnlimit",
                    "rolvaliduntil",
                )
            )
        }
        cur.execute(
            "SELECT parent.rolname AS parent_role,grantor.rolname AS grantor_role,"
            "m.admin_option,m.inherit_option,m.set_option "
            "FROM pg_catalog.pg_auth_members m "
            "JOIN pg_catalog.pg_roles parent ON parent.oid=m.roleid "
            "JOIN pg_catalog.pg_roles member ON member.oid=m.member "
            "JOIN pg_catalog.pg_roles grantor ON grantor.oid=m.grantor "
            "WHERE member.rolname=%s ORDER BY parent.rolname,grantor.rolname",
            (role_name,),
        )
        memberships = tuple(
            {
                "parent_role": _row_value(row, "parent_role"),
                "grantor_role": _row_value(row, "grantor_role", 1),
                "admin_option": _row_value(row, "admin_option", 2),
                "inherit_option": _row_value(row, "inherit_option", 3),
                "set_option": _row_value(row, "set_option", 4),
            }
            for row in cur.fetchall()
        )
        cur.execute(
            "SELECT member.rolname AS member_role,grantor.rolname AS grantor_role,"
            "m.admin_option,m.inherit_option,m.set_option "
            "FROM pg_catalog.pg_auth_members m "
            "JOIN pg_catalog.pg_roles parent ON parent.oid=m.roleid "
            "JOIN pg_catalog.pg_roles member ON member.oid=m.member "
            "JOIN pg_catalog.pg_roles grantor ON grantor.oid=m.grantor "
            "WHERE parent.rolname=%s ORDER BY member.rolname,grantor.rolname",
            (role_name,),
        )
        granted_memberships = tuple(
            {
                "member_role": _row_value(row, "member_role"),
                "grantor_role": _row_value(row, "grantor_role", 1),
                "admin_option": _row_value(row, "admin_option", 2),
                "inherit_option": _row_value(row, "inherit_option", 3),
                "set_option": _row_value(row, "set_option", 4),
            }
            for row in cur.fetchall()
        )

    cur.execute(
        "SELECT worker_id,contract,created_at::text AS created_at "
        "FROM public.fleet_worker_principals WHERE role_name=%s",
        (role_name,),
    )
    principal_row = cur.fetchone()
    if principal_row is not None:
        principal = {
            "worker_id": _row_value(principal_row, "worker_id"),
            "contract": _row_value(principal_row, "contract", 1),
            "created_at": _row_value(principal_row, "created_at", 2),
        }

    direct_acls = _direct_acl_snapshot(cur, grantee=role_name) if role_exists else ()
    default_acls, default_scopes = _default_acl_snapshot(cur, grantee=role_name)
    return {
        "existed": role_exists,
        "attributes": attributes,
        "memberships": memberships,
        "granted_memberships": granted_memberships,
        "principal_mapping": principal,
        "direct_acls": direct_acls,
        "default_acls": default_acls,
        "default_acl_scopes": default_scopes,
    }


def _direct_acl_snapshot(cur, *, grantee: str) -> tuple[dict[str, Any], ...]:
    queries = (
        (
            "database",
            "SELECT d.datname AS schema_name,NULL::text AS object_name,NULL::text AS arguments,"
            "acl.privilege_type,acl.is_grantable FROM pg_catalog.pg_database d "
            "CROSS JOIN LATERAL pg_catalog.aclexplode(d.datacl) acl "
            "WHERE d.datname=current_database() AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s)",
        ),
        (
            "schema",
            "SELECT n.nspname AS schema_name,NULL::text AS object_name,NULL::text AS arguments,"
            "acl.privilege_type,acl.is_grantable FROM pg_catalog.pg_namespace n "
            "CROSS JOIN LATERAL pg_catalog.aclexplode(n.nspacl) acl "
            "WHERE n.nspname<>'information_schema' AND n.nspname!~'^pg_' "
            "AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s)",
        ),
        (
            "relation",
            "SELECT n.nspname AS schema_name,c.relname AS object_name,c.relkind::text AS arguments,"
            "acl.privilege_type,acl.is_grantable FROM pg_catalog.pg_class c "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "CROSS JOIN LATERAL pg_catalog.aclexplode(c.relacl) acl "
            "WHERE c.relkind=ANY(ARRAY['r','p','v','m','f','S']::char[]) "
            "AND n.nspname<>'information_schema' AND n.nspname!~'^pg_' "
            "AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s)",
        ),
        (
            "routine",
            "SELECT n.nspname AS schema_name,p.proname AS object_name,"
            "pg_catalog.pg_get_function_identity_arguments(p.oid) AS arguments,"
            "acl.privilege_type,acl.is_grantable,p.prokind::text AS prokind "
            "FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace "
            "CROSS JOIN LATERAL pg_catalog.aclexplode(p.proacl) acl "
            "WHERE n.nspname<>'information_schema' AND n.nspname!~'^pg_' "
            "AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s)",
        ),
        (
            "type",
            "SELECT n.nspname AS schema_name,t.typname AS object_name,NULL::text AS arguments,"
            "acl.privilege_type,acl.is_grantable FROM pg_catalog.pg_type t "
            "JOIN pg_catalog.pg_namespace n ON n.oid=t.typnamespace "
            "CROSS JOIN LATERAL pg_catalog.aclexplode(t.typacl) acl "
            "WHERE t.typtype IN ('c','d','e','r','m') "
            "AND n.nspname<>'information_schema' AND n.nspname!~'^pg_' "
            "AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s)",
        ),
    )
    snapshot: list[dict[str, Any]] = []
    for catalog_kind, query in queries:
        cur.execute(query, (grantee,))
        for row in cur.fetchall():
            object_kind = catalog_kind
            relation_kind = _row_value(row, "arguments", 2)
            if catalog_kind == "relation":
                object_kind = "sequence" if relation_kind == "S" else "table"
            elif catalog_kind == "routine":
                object_kind = "procedure" if _row_value(row, "prokind", 5) == "p" else "function"
            snapshot.append(
                {
                    "object_kind": object_kind,
                    "schema_name": _row_value(row, "schema_name"),
                    "object_name": _row_value(row, "object_name", 1),
                    "arguments": None if catalog_kind == "relation" else _row_value(row, "arguments", 2),
                    "privilege": _row_value(row, "privilege_type", 3),
                    "grantable": _row_value(row, "is_grantable", 4),
                }
            )
    return tuple(
        sorted(
            snapshot,
            key=lambda item: (
                item["object_kind"],
                item["schema_name"] or "",
                item["object_name"] or "",
                item["arguments"] or "",
                item["privilege"],
            ),
        )
    )


def _default_acl_snapshot(cur, *, grantee: str) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    cur.execute(
        "SELECT owner.rolname AS owner_name,n.nspname AS schema_name,d.defaclobjtype,"
        "acl.privilege_type,acl.is_grantable FROM pg_catalog.pg_default_acl d "
        "JOIN pg_catalog.pg_roles owner ON owner.oid=d.defaclrole "
        "LEFT JOIN pg_catalog.pg_namespace n ON n.oid=d.defaclnamespace "
        "CROSS JOIN LATERAL pg_catalog.aclexplode(d.defaclacl) acl "
        "WHERE acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s) "
        "ORDER BY owner.rolname,n.nspname,d.defaclobjtype,acl.privilege_type",
        (grantee,),
    )
    entries = tuple(
        {
            "owner": _row_value(row, "owner_name"),
            "schema": _row_value(row, "schema_name", 1),
            "object_type": _row_value(row, "defaclobjtype", 2),
            "privilege": _row_value(row, "privilege_type", 3),
            "grantable": _row_value(row, "is_grantable", 4),
        }
        for row in cur.fetchall()
    )
    cur.execute(
        "SELECT DISTINCT owner_name,schema_name,object_type FROM ("
        "SELECT owner.rolname AS owner_name,NULL::text AS schema_name,kind.object_type "
        "FROM pg_catalog.pg_roles owner CROSS JOIN unnest(ARRAY['r','S','f','T']) kind(object_type) "
        "WHERE owner.rolname=current_user OR EXISTS(SELECT 1 FROM pg_default_acl d WHERE d.defaclrole=owner.oid) "
        "UNION ALL SELECT owner.rolname,n.nspname,kind.object_type FROM pg_catalog.pg_roles owner "
        "CROSS JOIN pg_catalog.pg_namespace n CROSS JOIN unnest(ARRAY['r','S','f','T']) kind(object_type) "
        "WHERE n.nspname<>'information_schema' AND n.nspname!~'^pg_' "
        "AND (owner.rolname=current_user OR EXISTS(SELECT 1 FROM pg_default_acl d WHERE d.defaclrole=owner.oid))"
        ") scopes ORDER BY owner_name,schema_name,object_type"
    )
    scopes = tuple(
        {
            "owner": _row_value(row, "owner_name"),
            "schema": _row_value(row, "schema_name", 1),
            "object_type": _row_value(row, "object_type", 2),
        }
        for row in cur.fetchall()
    )
    return entries, scopes


def _structured_regrant_snapshot(cur, regrants: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    snapshots: list[dict[str, Any]] = []
    for grant in regrants:
        kind = grant["object_kind"]
        qualified_name = grant["qualified_name"]
        grantee = grant["grantee"]
        target: dict[str, Any] = {
            "object_kind": kind,
            "qualified_name": qualified_name,
            "grantee": grantee,
        }
        if kind.startswith("default_"):
            owner, *schema_name = qualified_name.split(".")
            object_code = "f" if kind == "default_functions" else "T"
            cur.execute(
                "SELECT acl.privilege_type,acl.is_grantable FROM pg_catalog.pg_default_acl d "
                "LEFT JOIN pg_catalog.pg_namespace n ON n.oid=d.defaclnamespace "
                "CROSS JOIN LATERAL pg_catalog.aclexplode(d.defaclacl) acl "
                "WHERE d.defaclrole=(SELECT oid FROM pg_roles WHERE rolname=%s) "
                "AND d.defaclobjtype=%s AND n.nspname IS NOT DISTINCT FROM %s "
                "AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s) "
                "ORDER BY acl.privilege_type",
                (owner, object_code, schema_name[0] if schema_name else None, grantee),
            )
            target.update(owner=owner, schema=schema_name[0] if schema_name else None, object_code=object_code)
        elif kind == "schema":
            cur.execute(
                "SELECT acl.privilege_type,acl.is_grantable FROM pg_namespace n "
                "CROSS JOIN LATERAL aclexplode(n.nspacl) acl WHERE n.nspname=%s "
                "AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s) ORDER BY acl.privilege_type",
                (qualified_name, grantee),
            )
            target.update(schema_name=qualified_name, object_name=None, arguments=None)
        elif kind in {"table", "sequence"}:
            schema_name, object_name = qualified_name.split(".")
            cur.execute(
                "SELECT acl.privilege_type,acl.is_grantable FROM pg_class c "
                "JOIN pg_namespace n ON n.oid=c.relnamespace CROSS JOIN LATERAL aclexplode(c.relacl) acl "
                "WHERE n.nspname=%s AND c.relname=%s "
                "AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s) ORDER BY acl.privilege_type",
                (schema_name, object_name, grantee),
            )
            target.update(schema_name=schema_name, object_name=object_name, arguments=None)
        elif kind == "function":
            cur.execute(
                "SELECT n.nspname,p.proname,pg_get_function_identity_arguments(p.oid) AS arguments,"
                "acl.privilege_type,acl.is_grantable FROM pg_proc p "
                "JOIN pg_namespace n ON n.oid=p.pronamespace CROSS JOIN LATERAL aclexplode(p.proacl) acl "
                "WHERE p.oid=to_regprocedure(%s) "
                "AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s) ORDER BY acl.privilege_type",
                (qualified_name, grantee),
            )
            rows = cur.fetchall()
            if rows:
                target.update(
                    schema_name=_row_value(rows[0], "nspname"),
                    object_name=_row_value(rows[0], "proname", 1),
                    arguments=_row_value(rows[0], "arguments", 2),
                )
            else:
                cur.execute(
                    "SELECT n.nspname,p.proname,pg_get_function_identity_arguments(p.oid) AS arguments "
                    "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE p.oid=to_regprocedure(%s)",
                    (qualified_name,),
                )
                identity = cur.fetchone()
                target.update(
                    schema_name=_row_value(identity, "nspname"),
                    object_name=_row_value(identity, "proname", 1),
                    arguments=_row_value(identity, "arguments", 2),
                )
                rows = ()
            target["prior_acl"] = tuple(
                {
                    "privilege": _row_value(row, "privilege_type", 3),
                    "grantable": _row_value(row, "is_grantable", 4),
                }
                for row in rows
            )
            snapshots.append(target)
            continue
        else:
            schema_name, object_name = qualified_name.split(".")
            cur.execute(
                "SELECT acl.privilege_type,acl.is_grantable FROM pg_type t "
                "JOIN pg_namespace n ON n.oid=t.typnamespace CROSS JOIN LATERAL aclexplode(t.typacl) acl "
                "WHERE n.nspname=%s AND t.typname=%s "
                "AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s) ORDER BY acl.privilege_type",
                (schema_name, object_name, grantee),
            )
            target.update(schema_name=schema_name, object_name=object_name, arguments=None)
        target["prior_acl"] = tuple(
            {
                "privilege": _row_value(row, "privilege_type"),
                "grantable": _row_value(row, "is_grantable", 1),
            }
            for row in cur.fetchall()
        )
        snapshots.append(target)
    return tuple(snapshots)


def _acl_target_sql(cur, target: Mapping[str, Any]) -> str:
    kind = target["object_kind"]
    if kind == "database":
        return sql.Identifier(target["schema_name"]).as_string(cur.connection)
    if kind == "schema":
        return sql.Identifier(target["schema_name"]).as_string(cur.connection)
    if kind in {"table", "sequence", "type"}:
        return sql.Identifier(target["schema_name"], target["object_name"]).as_string(cur.connection)
    if kind in {"function", "procedure"}:
        return (
            sql.SQL("{}.{}({})")
            .format(
                sql.Identifier(target["schema_name"]),
                sql.Identifier(target["object_name"]),
                sql.SQL(target["arguments"]),
            )
            .as_string(cur.connection)
        )
    raise RuntimeError(f"unsupported rollback ACL object kind: {kind}")


def _grant_acl_sql(cur, *, target: Mapping[str, Any], grantee: str, acl: Iterable[Mapping[str, Any]]) -> list[str]:
    statements: list[str] = []
    object_kind = target["object_kind"].upper()
    identity = _acl_target_sql(cur, target)
    quoted_grantee = sql.Identifier(grantee).as_string(cur.connection)
    for grantable in (False, True):
        privileges = sorted(item["privilege"] for item in acl if bool(item["grantable"]) is grantable)
        if privileges:
            suffix = " WITH GRANT OPTION" if grantable else ""
            statements.append(f"GRANT {', '.join(privileges)} ON {object_kind} {identity} TO {quoted_grantee}{suffix};")
    return statements


def _default_acl_prefix(cur, scope: Mapping[str, Any]) -> tuple[str, str]:
    owner = sql.Identifier(scope["owner"]).as_string(cur.connection)
    schema = scope.get("schema")
    prefix = f"ALTER DEFAULT PRIVILEGES FOR ROLE {owner}"
    if schema:
        prefix += f" IN SCHEMA {sql.Identifier(schema).as_string(cur.connection)}"
    object_name = {"r": "TABLES", "S": "SEQUENCES", "f": "FUNCTIONS", "T": "TYPES"}[scope["object_type"]]
    return prefix, object_name


def _all_direct_acl_targets(cur) -> tuple[dict[str, Any], ...]:
    cur.execute(
        "SELECT 'database' AS object_kind,current_database() AS schema_name,NULL::text AS object_name,"
        "NULL::text AS arguments UNION ALL "
        "SELECT 'schema',n.nspname,NULL,NULL FROM pg_namespace n "
        "WHERE n.nspname<>'information_schema' AND n.nspname!~'^pg_' UNION ALL "
        "SELECT CASE WHEN c.relkind='S' THEN 'sequence' ELSE 'table' END,n.nspname,c.relname,NULL "
        "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE c.relkind=ANY(ARRAY['r','p','v','m','f','S']::char[]) "
        "AND n.nspname<>'information_schema' AND n.nspname!~'^pg_' UNION ALL "
        "SELECT CASE WHEN p.prokind='p' THEN 'procedure' ELSE 'function' END,n.nspname,p.proname,"
        "pg_get_function_identity_arguments(p.oid) FROM pg_proc p "
        "JOIN pg_namespace n ON n.oid=p.pronamespace "
        "WHERE n.nspname<>'information_schema' AND n.nspname!~'^pg_' UNION ALL "
        "SELECT 'type',n.nspname,t.typname,NULL FROM pg_type t "
        "JOIN pg_namespace n ON n.oid=t.typnamespace WHERE t.typtype IN ('c','d','e','r','m') "
        "AND n.nspname<>'information_schema' AND n.nspname!~'^pg_' "
        "ORDER BY object_kind,schema_name,object_name,arguments"
    )
    return tuple(
        {
            "object_kind": _row_value(row, "object_kind"),
            "schema_name": _row_value(row, "schema_name", 1),
            "object_name": _row_value(row, "object_name", 2),
            "arguments": _row_value(row, "arguments", 3),
        }
        for row in cur.fetchall()
    )


def _mapped_role_rollback_sql(
    cur,
    *,
    role_name: str,
    snapshot: Mapping[str, Any],
    regrant_snapshots: tuple[dict[str, Any], ...],
) -> str:
    quoted_role = sql.Identifier(role_name).as_string(cur.connection)
    statements = [
        f"DELETE FROM public.fleet_worker_principals WHERE role_name={sql.Literal(role_name).as_string(cur.connection)};"
    ]

    for regrant in regrant_snapshots:
        if regrant["object_kind"].startswith("default_"):
            scope = {
                "owner": regrant["owner"],
                "schema": regrant["schema"],
                "object_type": regrant["object_code"],
            }
            prefix, object_name = _default_acl_prefix(cur, scope)
            grantee = sql.Identifier(regrant["grantee"]).as_string(cur.connection)
            statements.append(f"{prefix} REVOKE ALL PRIVILEGES ON {object_name} FROM {grantee};")
            for grantable in (False, True):
                privileges = sorted(
                    item["privilege"] for item in regrant["prior_acl"] if bool(item["grantable"]) is grantable
                )
                if privileges:
                    suffix = " WITH GRANT OPTION" if grantable else ""
                    statements.append(f"{prefix} GRANT {', '.join(privileges)} ON {object_name} TO {grantee}{suffix};")
        else:
            object_kind = regrant["object_kind"].upper()
            identity = _acl_target_sql(cur, regrant)
            grantee = sql.Identifier(regrant["grantee"]).as_string(cur.connection)
            statements.append(f"REVOKE ALL PRIVILEGES ON {object_kind} {identity} FROM {grantee};")
            statements.extend(
                _grant_acl_sql(
                    cur,
                    target=regrant,
                    grantee=regrant["grantee"],
                    acl=regrant["prior_acl"],
                )
            )

    if not snapshot["existed"]:
        statements.extend((f"DROP OWNED BY {quoted_role};", f"DROP ROLE {quoted_role};"))
        return "\n".join(statements) + "\n"

    for target in _all_direct_acl_targets(cur):
        statements.append(
            f"REVOKE ALL PRIVILEGES ON {target['object_kind'].upper()} {_acl_target_sql(cur, target)} "
            f"FROM {quoted_role};"
        )
    for scope in snapshot["default_acl_scopes"]:
        prefix, object_name = _default_acl_prefix(cur, scope)
        statements.append(f"{prefix} REVOKE ALL PRIVILEGES ON {object_name} FROM {quoted_role};")

    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    targets: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for acl in snapshot["direct_acls"]:
        key = (
            acl["object_kind"],
            acl["schema_name"] or "",
            acl["object_name"] or "",
            acl["arguments"] or "",
        )
        grouped.setdefault(key, []).append(acl)
        targets[key] = acl
    for key in sorted(grouped):
        statements.extend(_grant_acl_sql(cur, target=targets[key], grantee=role_name, acl=grouped[key]))

    default_grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for acl in snapshot["default_acls"]:
        key = (acl["owner"], acl["schema"] or "", acl["object_type"])
        default_grouped.setdefault(key, []).append(acl)
    for key in sorted(default_grouped):
        acl = default_grouped[key]
        prefix, object_name = _default_acl_prefix(
            cur, {"owner": key[0], "schema": key[1] or None, "object_type": key[2]}
        )
        for grantable in (False, True):
            privileges = sorted(item["privilege"] for item in acl if bool(item["grantable"]) is grantable)
            if privileges:
                suffix = " WITH GRANT OPTION" if grantable else ""
                statements.append(f"{prefix} GRANT {', '.join(privileges)} ON {object_name} TO {quoted_role}{suffix};")

    def restore_membership(parent_role: str, member_role: str, membership: Mapping[str, Any]) -> None:
        parent = sql.Identifier(parent_role).as_string(cur.connection)
        member = sql.Identifier(member_role).as_string(cur.connection)
        grantor = sql.Identifier(membership["grantor_role"]).as_string(cur.connection)
        statements.append(f"SET LOCAL ROLE {grantor};")
        statements.append(
            f"GRANT {parent} TO {member} WITH "
            f"ADMIN {'TRUE' if membership['admin_option'] else 'FALSE'}, "
            f"INHERIT {'TRUE' if membership['inherit_option'] else 'FALSE'}, "
            f"SET {'TRUE' if membership['set_option'] else 'FALSE'} GRANTED BY {grantor};"
        )
        statements.append("RESET ROLE;")

    for membership in snapshot["memberships"]:
        restore_membership(membership["parent_role"], role_name, membership)
    for membership in snapshot["granted_memberships"]:
        restore_membership(role_name, membership["member_role"], membership)
    principal = snapshot["principal_mapping"]
    if principal is not None:
        statements.append(
            "INSERT INTO public.fleet_worker_principals(role_name,worker_id,contract,created_at) VALUES("
            f"{sql.Literal(role_name).as_string(cur.connection)},"
            f"{sql.Literal(principal['worker_id']).as_string(cur.connection)},"
            f"{sql.Literal(principal['contract']).as_string(cur.connection)},"
            f"{sql.Literal(principal['created_at']).as_string(cur.connection)});"
        )
    attributes = snapshot["attributes"]
    role_attributes = (
        "NOLOGIN",
        "INHERIT" if attributes["rolinherit"] else "NOINHERIT",
        "SUPERUSER" if attributes["rolsuper"] else "NOSUPERUSER",
        "CREATEDB" if attributes["rolcreatedb"] else "NOCREATEDB",
        "CREATEROLE" if attributes["rolcreaterole"] else "NOCREATEROLE",
        "REPLICATION" if attributes["rolreplication"] else "NOREPLICATION",
        "BYPASSRLS" if attributes["rolbypassrls"] else "NOBYPASSRLS",
        f"CONNECTION LIMIT {attributes['rolconnlimit']}",
    )
    statements.append(f"ALTER ROLE {quoted_role} {' '.join(role_attributes)};")
    return "\n".join(statements) + "\n"


def _public_acl_rollback_sql(
    cur,
    *,
    connect_allowlist: tuple[str, ...],
    database_owner_before: str,
    database_owner_after: str,
    retired_admin_roles: tuple[str, ...],
) -> str:
    cur.execute("SELECT current_database() AS database_name")
    database_name = _row_value(cur.fetchone(), "database_name")
    quoted_database = f'"{database_name.replace(chr(34), chr(34) * 2)}"'
    statements: list[str] = []
    if database_owner_before != database_owner_after:
        cur.execute("SELECT format('%%I',%s::text) AS owner_name", (database_owner_before,))
        statements.append(f"ALTER DATABASE {quoted_database} OWNER TO {_row_value(cur.fetchone(), 'owner_name')};")
    if retired_admin_roles:
        cur.execute(
            "SELECT rolname,rolcanlogin,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
            "FROM pg_roles WHERE rolname=ANY(%s) ORDER BY rolname",
            (list(retired_admin_roles),),
        )
        for row in cur.fetchall():
            cur.execute(
                "SELECT format('%%I',%s::text) AS role_name",
                (_row_value(row, "rolname"),),
            )
            quoted_role = _row_value(cur.fetchone(), "role_name")
            attributes = (
                "LOGIN" if _row_value(row, "rolcanlogin", 1) else "NOLOGIN",
                "SUPERUSER" if _row_value(row, "rolsuper", 2) else "NOSUPERUSER",
                "CREATEDB" if _row_value(row, "rolcreatedb", 3) else "NOCREATEDB",
                "CREATEROLE" if _row_value(row, "rolcreaterole", 4) else "NOCREATEROLE",
                "REPLICATION" if _row_value(row, "rolreplication", 5) else "NOREPLICATION",
                "BYPASSRLS" if _row_value(row, "rolbypassrls", 6) else "NOBYPASSRLS",
            )
            statements.append(f"ALTER ROLE {quoted_role} {' '.join(attributes)};")
    for privilege in ("CONNECT", "CREATE", "TEMPORARY"):
        cur.execute(
            "SELECT has_database_privilege('public',current_database(),%s) AS allowed",
            (privilege,),
        )
        if _row_value(cur.fetchone(), "allowed"):
            statements.append(f"GRANT {privilege} ON DATABASE {quoted_database} TO PUBLIC;")
    cur.execute(
        "SELECT grantee.rolname FROM pg_catalog.pg_database d "
        "CROSS JOIN LATERAL pg_catalog.aclexplode(d.datacl) acl "
        "JOIN pg_catalog.pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE d.datname=current_database() AND acl.privilege_type='CONNECT' "
        "ORDER BY grantee.rolname"
    )
    prior_direct_connect = {_row_value(row, "rolname") for row in cur.fetchall()}
    for role_name in connect_allowlist:
        if role_name not in prior_direct_connect:
            cur.execute("SELECT format('%%I',%s::text) AS role_name", (role_name,))
            quoted_role = _row_value(cur.fetchone(), "role_name")
            statements.append(
                _conditional_role_statement(
                    cur,
                    role_name=role_name,
                    statement=f"REVOKE CONNECT ON DATABASE {quoted_database} FROM {quoted_role}",
                )
            )
    for role_name in sorted(set(retired_admin_roles) & prior_direct_connect):
        cur.execute("SELECT format('%%I',%s::text) AS role_name", (role_name,))
        statements.append(f"GRANT CONNECT ON DATABASE {quoted_database} TO {_row_value(cur.fetchone(), 'role_name')};")

    queries = (
        (
            "SCHEMA",
            ("USAGE", "CREATE"),
            "SELECT format('%%I',n.nspname) AS identity FROM pg_namespace n "
            "WHERE n.nspname<>'information_schema' AND n.nspname!~'^pg_' "
            "AND has_schema_privilege('public',n.oid,%s) ORDER BY n.nspname",
        ),
        (
            "TABLE",
            ("SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER"),
            "SELECT format('%%I.%%I',n.nspname,c.relname) AS identity FROM pg_class c "
            "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.relkind IN ('r','p','v','m','f') "
            "AND n.nspname<>'information_schema' AND n.nspname!~'^pg_' "
            "AND has_table_privilege('public',c.oid,%s) ORDER BY n.nspname,c.relname",
        ),
        (
            "SEQUENCE",
            ("USAGE", "SELECT", "UPDATE"),
            "SELECT format('%%I.%%I',n.nspname,c.relname) AS identity FROM pg_class c "
            "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.relkind='S' "
            "AND n.nspname<>'information_schema' AND n.nspname!~'^pg_' "
            "AND CASE WHEN c.relkind='S' THEN has_sequence_privilege('public',c.oid,%s) "
            "ELSE FALSE END ORDER BY n.nspname,c.relname",
        ),
        (
            "FUNCTION",
            ("EXECUTE",),
            "SELECT p.oid::regprocedure::text AS identity FROM pg_proc p "
            "JOIN pg_namespace n ON n.oid=p.pronamespace "
            "WHERE n.nspname<>'information_schema' AND n.nspname!~'^pg_' "
            "AND has_function_privilege('public',p.oid,%s) ORDER BY p.oid::regprocedure::text",
        ),
        (
            "TYPE",
            ("USAGE",),
            "SELECT format('%%I.%%I',n.nspname,t.typname) AS identity FROM pg_type t "
            "JOIN pg_namespace n ON n.oid=t.typnamespace "
            "WHERE n.nspname<>'information_schema' AND n.nspname!~'^pg_' "
            "AND t.typtype IN ('c','d','e','r','m') "
            "AND has_type_privilege('public',t.oid,%s) ORDER BY n.nspname,t.typname",
        ),
    )
    for object_kind, privileges, query in queries:
        for privilege in privileges:
            cur.execute(query, (privilege,))
            statements.extend(
                f"GRANT {privilege} ON {object_kind} {_row_value(row, 'identity')} TO PUBLIC;" for row in cur.fetchall()
            )
    cur.execute(
        "SELECT DISTINCT owner_name FROM (SELECT current_user AS owner_name UNION ALL "
        "SELECT r.rolname FROM pg_default_acl d JOIN pg_roles r ON r.oid=d.defaclrole) owners "
        "ORDER BY owner_name"
    )
    default_owners = tuple(_row_value(row, "owner_name") for row in cur.fetchall())
    for owner_name in default_owners:
        cur.execute("SELECT format('%%I',%s::text) AS owner_name", (owner_name,))
        quoted_owner = _row_value(cur.fetchone(), "owner_name")
        for object_type, acl_code, privilege in (
            ("FUNCTIONS", "f", "EXECUTE"),
            ("TYPES", "T", "USAGE"),
        ):
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_default_acl d "
                "CROSS JOIN LATERAL aclexplode(d.defaclacl) acl "
                "WHERE d.defaclrole=(SELECT oid FROM pg_roles WHERE rolname=%s) "
                "AND d.defaclnamespace=0 AND d.defaclobjtype=%s "
                "AND acl.grantee=0 AND acl.privilege_type=%s) AS explicit, "
                "EXISTS(SELECT 1 FROM aclexplode(acldefault(%s,(SELECT oid FROM pg_roles WHERE rolname=%s))) acl "
                "WHERE acl.grantee=0 AND acl.privilege_type=%s) AS implicit",
                (owner_name, acl_code, privilege, acl_code, owner_name, privilege),
            )
            default_row = cur.fetchone()
            if _row_value(default_row, "explicit") or _row_value(default_row, "implicit", 1):
                statements.append(
                    f"ALTER DEFAULT PRIVILEGES FOR ROLE {quoted_owner} GRANT {privilege} ON {object_type} TO PUBLIC;"
                )
            cur.execute(
                "SELECT format('%%I',n.nspname) AS schema_name FROM pg_default_acl d "
                "JOIN pg_namespace n ON n.oid=d.defaclnamespace "
                "CROSS JOIN LATERAL aclexplode(d.defaclacl) acl "
                "WHERE d.defaclrole=(SELECT oid FROM pg_roles WHERE rolname=%s) "
                "AND d.defaclobjtype=%s AND acl.grantee=0 AND acl.privilege_type=%s "
                "ORDER BY n.nspname",
                (owner_name, acl_code, privilege),
            )
            statements.extend(
                "ALTER DEFAULT PRIVILEGES FOR ROLE "
                f"{quoted_owner} IN SCHEMA {_row_value(row, 'schema_name')} "
                f"GRANT {privilege} ON {object_type} TO PUBLIC;"
                for row in cur.fetchall()
            )
    cur.execute("SELECT to_regprocedure('public.fleet_worker_identity()') AS identity_oid")
    if _row_value(cur.fetchone(), "identity_oid") is None:
        statements.append("DROP FUNCTION IF EXISTS public.fleet_worker_identity();")
    return "\n".join(statements) + "\n"


def _install_identity_function(cur) -> None:
    cur.execute(
        "CREATE OR REPLACE FUNCTION public.fleet_worker_identity() RETURNS JSONB "
        "LANGUAGE SQL STABLE SECURITY DEFINER SET search_path=pg_catalog "
        "AS $$ SELECT pg_catalog.jsonb_build_object("
        "'role_name',p.role_name,'worker_id',p.worker_id,'contract',p.contract) "
        "FROM public.fleet_worker_principals p WHERE p.role_name=session_user $$"
    )
    cur.execute("REVOKE ALL PRIVILEGES ON FUNCTION public.fleet_worker_identity() FROM PUBLIC")
    cur.execute(
        "SELECT owner.rolname FROM pg_class relation "
        "JOIN pg_roles owner ON owner.oid=relation.relowner "
        "WHERE relation.oid='public.fleet_worker_principals'::regclass"
    )
    principal_owner = _row_value(cur.fetchone(), "rolname")
    cur.execute(
        sql.SQL("ALTER FUNCTION public.fleet_worker_identity() OWNER TO {}").format(sql.Identifier(principal_owner))
    )


def _harden_database_connect(
    cur,
    *,
    database: sql.Identifier,
    database_owner_role: str,
    retired_admin_roles: tuple[str, ...],
) -> None:
    cur.execute(
        "SELECT owner.rolname FROM pg_database d JOIN pg_roles owner ON owner.oid=d.datdba "
        "WHERE d.datname=current_database()"
    )
    if _row_value(cur.fetchone(), "rolname") != database_owner_role:
        cur.execute(sql.SQL("ALTER DATABASE {} OWNER TO {}").format(database, sql.Identifier(database_owner_role)))
    cur.execute(sql.SQL("REVOKE CONNECT, CREATE, TEMPORARY ON DATABASE {} FROM PUBLIC").format(database))
    for role_name in retired_admin_roles:
        role = sql.Identifier(role_name)
        cur.execute(sql.SQL("REVOKE CONNECT ON DATABASE {} FROM {}").format(database, role))
        cur.execute(
            "SELECT rolcanlogin,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
            "FROM pg_roles WHERE rolname=%s",
            (role_name,),
        )
        attributes = cur.fetchone()
        if attributes is not None and any(
            _row_value(attributes, name, index)
            for index, name in enumerate(
                ("rolcanlogin", "rolsuper", "rolcreatedb", "rolcreaterole", "rolreplication", "rolbypassrls")
            )
        ):
            cur.execute(
                sql.SQL("ALTER ROLE {} NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS").format(
                    role
                )
            )


def _effective_connect_grantees(cur) -> tuple[dict[str, Any], ...]:
    cur.execute(
        "SELECT r.rolname,r.rolcanlogin,r.rolsuper,(r.oid=d.datdba) AS database_owner,"
        "has_database_privilege(r.oid,d.oid,'CONNECT') AS effective_connect "
        "FROM pg_roles r CROSS JOIN pg_database d WHERE d.datname=current_database() "
        "AND has_database_privilege(r.oid,d.oid,'CONNECT') ORDER BY r.rolname"
    )
    return tuple(
        {
            "role_name": _row_value(row, "rolname"),
            "can_login": _row_value(row, "rolcanlogin", 1),
            "superuser": _row_value(row, "rolsuper", 2),
            "database_owner": _row_value(row, "database_owner", 3),
            "effective_connect": _row_value(row, "effective_connect", 4),
            "reconnect_capable": _row_value(row, "rolcanlogin", 1) and _row_value(row, "effective_connect", 4),
        }
        for row in cur.fetchall()
    )


def _revoke_legacy_privileges(cur, *, role: sql.Identifier, database: sql.Identifier) -> None:
    cur.execute(sql.SQL("REVOKE ALL PRIVILEGES ON DATABASE {} FROM {}").format(database, role))
    cur.execute(sql.SQL("REVOKE CONNECT, CREATE, TEMPORARY ON DATABASE {} FROM PUBLIC").format(database))
    schemas = _user_schemas(cur)
    for schema_name in schemas:
        schema = sql.Identifier(schema_name)
        cur.execute(sql.SQL("REVOKE ALL PRIVILEGES ON SCHEMA {} FROM {}").format(schema, role))
        cur.execute(sql.SQL("REVOKE ALL PRIVILEGES ON SCHEMA {} FROM PUBLIC").format(schema))
        cur.execute(sql.SQL("REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA {} FROM {}").format(schema, role))
        cur.execute(sql.SQL("REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA {} FROM PUBLIC").format(schema))
        cur.execute(sql.SQL("REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA {} FROM {}").format(schema, role))
        cur.execute(sql.SQL("REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA {} FROM PUBLIC").format(schema))
        cur.execute(sql.SQL("REVOKE ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA {} FROM {}").format(schema, role))
        cur.execute(
            "SELECT t.typname FROM pg_catalog.pg_type t "
            "JOIN pg_catalog.pg_namespace n ON n.oid=t.typnamespace "
            "WHERE n.nspname=%s AND t.typtype IN ('c','d','e','r','m') ORDER BY t.typname",
            (schema_name,),
        )
        for type_row in cur.fetchall():
            cur.execute(
                sql.SQL("REVOKE ALL PRIVILEGES ON TYPE {}.{} FROM {}, PUBLIC").format(
                    schema,
                    sql.Identifier(_row_value(type_row, "typname")),
                    role,
                )
            )
        # Functions are executable by PUBLIC by default, which would make every
        # authority helper callable by fleet_worker regardless of its own ACL.
        cur.execute(sql.SQL("REVOKE EXECUTE ON ALL FUNCTIONS IN SCHEMA {} FROM PUBLIC").format(schema))

    cur.execute(
        "SELECT DISTINCT owner_name FROM ("
        "SELECT current_user AS owner_name "
        "UNION ALL "
        "SELECT r.rolname FROM pg_default_acl d "
        "JOIN pg_roles r ON r.oid=d.defaclrole "
        "WHERE pg_has_role(current_user,r.oid,'USAGE')"
        ") owners"
    )
    for row in cur.fetchall():
        owner = sql.Identifier(_row_value(row, "owner_name"))
        for object_type in ("TABLES", "SEQUENCES", "FUNCTIONS", "TYPES"):
            cur.execute(
                sql.SQL("ALTER DEFAULT PRIVILEGES FOR ROLE {} REVOKE ALL PRIVILEGES ON {} FROM {}").format(
                    owner, sql.SQL(object_type), role
                )
            )
        cur.execute(
            sql.SQL("ALTER DEFAULT PRIVILEGES FOR ROLE {} REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC").format(owner)
        )
        cur.execute(sql.SQL("ALTER DEFAULT PRIVILEGES FOR ROLE {} REVOKE ALL ON TYPES FROM PUBLIC").format(owner))
        for schema_name in schemas:
            for object_type in ("TABLES", "SEQUENCES", "FUNCTIONS", "TYPES"):
                cur.execute(
                    sql.SQL(
                        "ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA {} REVOKE ALL PRIVILEGES ON {} FROM {}"
                    ).format(
                        owner,
                        sql.Identifier(schema_name),
                        sql.SQL(object_type),
                        role,
                    )
                )
            cur.execute(
                sql.SQL(
                    "ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA {} REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC"
                ).format(owner, sql.Identifier(schema_name))
            )
            cur.execute(
                sql.SQL("ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA {} REVOKE ALL ON TYPES FROM PUBLIC").format(
                    owner, sql.Identifier(schema_name)
                )
            )


def _reconcile_role_memberships(
    cur,
    *,
    role_name: str,
    approved_parent_roles: tuple[str, ...] = (),
    approved_grantee_roles: tuple[str, ...] = (),
    validate_approved_grantee_descendants: bool = True,
) -> None:
    approved_parents = set(approved_parent_roles)
    approved_grantees = set(approved_grantee_roles)
    cur.execute(
        "SELECT parent.rolname FROM pg_auth_members membership "
        "JOIN pg_roles parent ON parent.oid=membership.roleid "
        "JOIN pg_roles member ON member.oid=membership.member "
        "WHERE member.rolname=%s ORDER BY parent.rolname",
        (role_name,),
    )
    for row in cur.fetchall():
        parent = _row_value(row, "rolname")
        if parent not in approved_parents:
            cur.execute(
                sql.SQL("REVOKE {} FROM {}").format(sql.Identifier(parent), sql.Identifier(role_name))
            )

    cur.execute(
        "SELECT member.rolname FROM pg_auth_members membership "
        "JOIN pg_roles parent ON parent.oid=membership.roleid "
        "JOIN pg_roles member ON member.oid=membership.member "
        "WHERE parent.rolname=%s ORDER BY member.rolname",
        (role_name,),
    )
    for row in cur.fetchall():
        member = _row_value(row, "rolname")
        revoke = member not in approved_grantees
        if not revoke and validate_approved_grantee_descendants:
            cur.execute(
                "WITH RECURSIVE descendants(role_oid) AS ("
                "SELECT oid FROM pg_roles WHERE rolname=%s UNION "
                "SELECT membership.member FROM pg_auth_members membership "
                "JOIN descendants ON descendants.role_oid=membership.roleid) "
                "SELECT rolname FROM descendants JOIN pg_roles ON pg_roles.oid=descendants.role_oid "
                "WHERE rolname<>ALL(%s) ORDER BY rolname",
                (member, list(approved_grantees)),
            )
            revoke = cur.fetchone() is not None
        if revoke:
            cur.execute(
                sql.SQL("REVOKE {} FROM {}").format(sql.Identifier(role_name), sql.Identifier(member))
            )

    cur.execute(
        "SELECT parent.rolname AS parent_role,member.rolname AS member_role "
        "FROM pg_auth_members membership JOIN pg_roles parent ON parent.oid=membership.roleid "
        "JOIN pg_roles member ON member.oid=membership.member "
        "WHERE (member.rolname=%s AND parent.rolname<>ALL(%s)) "
        "OR (parent.rolname=%s AND member.rolname<>ALL(%s))",
        (role_name, list(approved_parents), role_name, list(approved_grantees)),
    )
    violation = cur.fetchone()
    if violation is not None:
        raise RuntimeError(
            f"role membership boundary invalid for {role_name}: "
            f"{violation['parent_role']} -> {violation['member_role']}"
        )


def _owned_objects(cur, role_name: str) -> list[str]:
    cur.execute("SELECT oid FROM pg_roles WHERE rolname=%s", (role_name,))
    role_oid = _row_value(cur.fetchone(), "oid")
    owned: list[str] = []

    cur.execute(
        "SELECT datname FROM pg_database WHERE datname=current_database() AND datdba=%s",
        (role_oid,),
    )
    owned.extend(f"database:{_row_value(row, 'datname')}" for row in cur.fetchall())

    cur.execute(
        "SELECT nspname FROM pg_namespace WHERE nspowner=%s "
        "AND nspname <> 'information_schema' AND nspname !~ '^pg_' ORDER BY nspname",
        (role_oid,),
    )
    owned.extend(f"schema:{_row_value(row, 'nspname')}" for row in cur.fetchall())

    cur.execute(
        "SELECT n.nspname,c.relname,c.relkind FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE c.relowner=%s AND c.relkind=ANY(%s) "
        "ORDER BY CASE WHEN c.relkind='S' THEN 1 ELSE 0 END,n.nspname,c.relname",
        (role_oid, ["r", "p", "v", "m", "f", "S"]),
    )
    for row in cur.fetchall():
        kind = "sequence" if _row_value(row, "relkind", 2) == "S" else "table"
        owned.append(f"{kind}:{_row_value(row, 'nspname')}.{_row_value(row, 'relname', 1)}")

    cur.execute(
        "SELECT n.nspname,p.proname,pg_get_function_identity_arguments(p.oid) AS args "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "WHERE p.proowner=%s ORDER BY n.nspname,p.proname,args",
        (role_oid,),
    )
    owned.extend(
        f"function:{_row_value(row, 'nspname')}.{_row_value(row, 'proname', 1)}({_row_value(row, 'args', 2)})"
        for row in cur.fetchall()
    )

    cur.execute(
        "SELECT n.nspname,t.typname,t.typtype FROM pg_type t "
        "JOIN pg_namespace n ON n.oid=t.typnamespace "
        "WHERE t.typowner=%s AND n.nspname <> 'information_schema' "
        "AND n.nspname !~ '^pg_' AND t.typtype IN ('c','d','e','r','m') "
        "ORDER BY n.nspname,t.typname",
        (role_oid,),
    )
    for row in cur.fetchall():
        kind = "domain" if _row_value(row, "typtype", 2) == "d" else "type"
        owned.append(f"{kind}:{_row_value(row, 'nspname')}.{_row_value(row, 'typname', 1)}")

    cur.execute(
        "SELECT n.nspname,o.oprname FROM pg_operator o "
        "JOIN pg_namespace n ON n.oid=o.oprnamespace "
        "WHERE o.oprowner=%s AND n.nspname <> 'information_schema' "
        "AND n.nspname !~ '^pg_' ORDER BY n.nspname,o.oprname",
        (role_oid,),
    )
    owned.extend(f"operator:{_row_value(row, 'nspname')}.{_row_value(row, 'oprname', 1)}" for row in cur.fetchall())
    return owned


def _reject_pg_temp_shadow(cur) -> None:
    cur.execute(
        "SELECT kind,name FROM ("
        "SELECT 'relation' AS kind,c.relname AS name FROM pg_catalog.pg_class c "
        "WHERE c.relnamespace=pg_catalog.pg_my_temp_schema() "
        "UNION ALL SELECT 'function',p.proname FROM pg_catalog.pg_proc p "
        "WHERE p.pronamespace=pg_catalog.pg_my_temp_schema() "
        "UNION ALL SELECT CASE WHEN t.typtype='d' THEN 'domain' ELSE 'type' END,t.typname "
        "FROM pg_catalog.pg_type t WHERE t.typnamespace=pg_catalog.pg_my_temp_schema() "
        "UNION ALL SELECT 'operator',o.oprname FROM pg_catalog.pg_operator o "
        "WHERE o.oprnamespace=pg_catalog.pg_my_temp_schema()"
        ") shadow ORDER BY kind,name"
    )
    shadows = [f"{_row_value(row, 'kind')}:{_row_value(row, 'name', 1)}" for row in cur.fetchall()]
    if shadows:
        raise RuntimeError("fleet worker reconciliation rejects pg_temp object shadowing: " + ", ".join(shadows))


def _functions_for_contract(contract: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    try:
        return (*_COMMON_FUNCTIONS, *_CONTRACT_FUNCTIONS[contract])
    except KeyError:
        raise ValueError(f"unsupported fleet worker contract: {contract}") from None


def _validate_objects(cur, functions) -> None:
    expected_columns = {table: set(columns) for table, columns in _REQUIRED_FUNCTION_COLUMNS.items()}
    expected_columns["fleet_worker_principals"] = {"role_name", "worker_id", "contract"}
    expected_columns["workers"] = {"worker_id"}
    all_required_tables = set(expected_columns)
    cur.execute(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name=ANY(%s)",
        (list(all_required_tables),),
    )
    actual_columns: dict[str, set[str]] = {}
    for row in cur.fetchall():
        table = _row_value(row, "table_name", 0)
        actual_columns.setdefault(table, set()).add(_row_value(row, "column_name", 1))

    missing_tables = sorted(all_required_tables - actual_columns.keys())
    if missing_tables:
        raise RuntimeError("fleet_worker required tables missing: " + ", ".join(missing_tables))

    missing_columns = {
        table: sorted(columns - actual_columns[table])
        for table, columns in expected_columns.items()
        if columns - actual_columns[table]
    }
    if missing_columns:
        detail = "; ".join(f"{table}: {', '.join(columns)}" for table, columns in sorted(missing_columns.items()))
        raise RuntimeError("fleet_worker required columns missing: " + detail)

    missing_functions: list[str] = []
    for name, argument_types in functions:
        if name == "fleet_worker_identity":
            continue
        signature = f"public.{name}({','.join(argument_type.lower() for argument_type in argument_types)})"
        cur.execute("SELECT to_regprocedure(%s) AS oid", (signature,))
        if _row_value(cur.fetchone(), "oid") is None:
            missing_functions.append(signature)
    if missing_functions:
        raise RuntimeError("fleet_worker required functions missing: " + ", ".join(missing_functions))

    return None


def _grant_columns(cur, privilege: str, grants: Mapping[str, tuple[str, ...]], role) -> None:
    for table, columns in grants.items():
        cur.execute(
            sql.SQL("GRANT {} ({}) ON TABLE {} TO {}").format(
                sql.SQL(privilege),
                _identifiers(columns),
                sql.Identifier(table),
                role,
            )
        )


def _validate_effective_boundary(cur, *, role_name: str, functions, allowed_types: tuple[str, ...] = ()) -> None:
    cur.execute(
        "SELECT n.nspname,c.relname FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname <> 'information_schema' AND n.nspname !~ '^pg_' "
        "AND c.relkind IN ('r','p','v','m','f') AND ("
        "has_table_privilege(%s,c.oid,'INSERT') OR has_table_privilege(%s,c.oid,'UPDATE') OR "
        "has_table_privilege(%s,c.oid,'DELETE') OR has_table_privilege(%s,c.oid,'TRUNCATE') OR "
        "has_table_privilege(%s,c.oid,'REFERENCES') OR has_table_privilege(%s,c.oid,'TRIGGER') OR "
        "has_table_privilege(%s,c.oid,'SELECT')) "
        "ORDER BY n.nspname,c.relname",
        (role_name, role_name, role_name, role_name, role_name, role_name, role_name),
    )
    forbidden = [f"table:{_row_value(row, 'nspname')}.{_row_value(row, 'relname', 1)}" for row in cur.fetchall()]

    cur.execute(
        "SELECT n.nspname,c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname <> 'information_schema' AND n.nspname !~ '^pg_' AND c.relkind='S' "
        "AND (has_sequence_privilege(%s,c.oid,'USAGE') "
        "OR has_sequence_privilege(%s,c.oid,'SELECT') OR has_sequence_privilege(%s,c.oid,'UPDATE')) "
        "ORDER BY n.nspname,c.relname",
        (role_name, role_name, role_name),
    )
    forbidden.extend(f"sequence:{_row_value(row, 'nspname')}.{_row_value(row, 'relname', 1)}" for row in cur.fetchall())

    cur.execute(
        "SELECT n.nspname,t.typname FROM pg_catalog.pg_type t "
        "JOIN pg_catalog.pg_namespace n ON n.oid=t.typnamespace "
        "WHERE n.nspname <> 'information_schema' AND n.nspname !~ '^pg_' "
        "AND pg_catalog.has_type_privilege(%s,t.oid,'USAGE') "
        "AND NOT (n.nspname='public' AND t.typname=ANY(%s)) "
        "ORDER BY n.nspname,t.typname",
        (role_name, list(allowed_types)),
    )
    forbidden.extend(f"type:{_row_value(row, 'nspname')}.{_row_value(row, 'typname', 1)}" for row in cur.fetchall())

    allowed_function_oids: list[int] = []
    for name, argument_types in functions:
        signature = f"public.{name}({','.join(argument_types).lower()})"
        cur.execute("SELECT to_regprocedure(%s)::oid AS oid", (signature,))
        allowed_function_oids.append(_row_value(cur.fetchone(), "oid"))
    cur.execute(
        "SELECT n.nspname,p.proname,pg_get_function_identity_arguments(p.oid) AS args "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "WHERE n.nspname <> 'information_schema' AND n.nspname !~ '^pg_' "
        "AND has_function_privilege(%s,p.oid,'EXECUTE') "
        "AND p.oid <> ALL(%s) "
        "ORDER BY n.nspname,p.proname,args",
        (role_name, allowed_function_oids),
    )
    forbidden.extend(
        f"function:{_row_value(row, 'nspname')}.{_row_value(row, 'proname', 1)}({_row_value(row, 'args', 2)})"
        for row in cur.fetchall()
    )
    if forbidden:
        raise RuntimeError(
            "fleet_worker retains forbidden effective privileges after reconciliation: " + ", ".join(forbidden)
        )


def _client_scram_verifier(conn, *, role_name: str, password: str) -> str:
    try:
        verifier = conn.pgconn.encrypt_password(
            password.encode("utf-8"),
            role_name.encode("utf-8"),
            algorithm=b"scram-sha-256",
        ).decode("ascii")
    except Exception:
        raise RuntimeError("could not generate fleet worker password verifier") from None
    if not verifier.startswith("SCRAM-SHA-256$"):
        raise RuntimeError("could not generate fleet worker password verifier")
    return verifier


def _validate_post_password_boundary(
    cur,
    *,
    role_name: str,
    functions,
    allowed_types: tuple[str, ...],
    connect_allowlist: tuple[str, ...],
    database_owner_role: str,
    retired_admin_roles: tuple[str, ...],
    break_glass_roles: tuple[str, ...],
    approved_grantee_roles: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    cur.execute(
        "SELECT rolcanlogin, rolinherit, rolsuper, rolcreatedb, rolcreaterole, "
        "rolreplication, rolbypassrls FROM pg_catalog.pg_roles WHERE rolname=%s",
        (role_name,),
    )
    row = cur.fetchone()
    attributes = (
        None
        if row is None
        else tuple(
            _row_value(row, name, index)
            for index, name in enumerate(
                (
                    "rolcanlogin",
                    "rolinherit",
                    "rolsuper",
                    "rolcreatedb",
                    "rolcreaterole",
                    "rolreplication",
                    "rolbypassrls",
                )
            )
        )
    )
    if attributes != (True, False, False, False, False, False, False):
        raise RuntimeError("fleet worker role attributes drifted during password reconciliation")

    cur.execute(
        "SELECT 1 FROM pg_catalog.pg_auth_members membership "
        "JOIN pg_catalog.pg_roles member ON member.oid=membership.member "
        "JOIN pg_catalog.pg_roles parent ON parent.oid=membership.roleid "
        "WHERE member.rolname=%s OR (parent.rolname=%s AND member.rolname<>ALL(%s)) LIMIT 1",
        (role_name, role_name, list(approved_grantee_roles)),
    )
    if cur.fetchone() is not None:
        raise RuntimeError("fleet worker role membership drifted during password reconciliation")

    if _owned_objects(cur, role_name):
        raise RuntimeError("fleet worker role ownership drifted during password reconciliation")
    _validate_effective_boundary(cur, role_name=role_name, functions=functions, allowed_types=allowed_types)
    cur.execute(
        "SELECT pg_catalog.has_database_privilege(%s,pg_catalog.current_database(),'CONNECT') AS db_ok, "
        "pg_catalog.has_schema_privilege(%s,'public','USAGE') AS schema_ok",
        (role_name, role_name),
    )
    positive = cur.fetchone()
    if not _row_value(positive, "db_ok") or not _row_value(positive, "schema_ok", 1):
        raise RuntimeError("fleet worker required database/schema privileges are missing")
    cur.execute(
        "SELECT CASE WHEN acl.grantee=0 THEN 'PUBLIC' ELSE grantee.rolname END AS grantee "
        "FROM pg_catalog.pg_database d "
        "CROSS JOIN LATERAL pg_catalog.aclexplode(d.datacl) acl "
        "LEFT JOIN pg_catalog.pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE d.datname=current_database() AND acl.privilege_type='CONNECT' "
        "ORDER BY grantee"
    )
    actual_connect = tuple(_row_value(row, "grantee") for row in cur.fetchall())
    if set(actual_connect) != set(connect_allowlist):
        raise RuntimeError(
            "database CONNECT ACL does not match controller/verifier/mapped role allowlist: "
            f"expected={sorted(connect_allowlist)!r}, actual={sorted(actual_connect)!r}"
        )
    cur.execute(
        "SELECT owner.rolname,owner.rolcanlogin,owner.rolsuper "
        "FROM pg_database d JOIN pg_roles owner ON owner.oid=d.datdba "
        "WHERE d.datname=current_database()"
    )
    owner = cur.fetchone()
    if (
        _row_value(owner, "rolname") != database_owner_role
        or not _row_value(owner, "rolcanlogin", 1)
        or _row_value(owner, "rolsuper", 2)
    ):
        raise RuntimeError("database owner must be the dedicated LOGIN NOSUPERUSER role")
    if retired_admin_roles:
        cur.execute(
            "SELECT rolname,rolcanlogin,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
            "FROM pg_roles WHERE rolname=ANY(%s) ORDER BY rolname",
            (list(retired_admin_roles),),
        )
        elevated_retired = []
        for retired in cur.fetchall():
            attributes = tuple(
                _row_value(retired, name, index)
                for index, name in enumerate(
                    (
                        "rolcanlogin",
                        "rolsuper",
                        "rolcreatedb",
                        "rolcreaterole",
                        "rolreplication",
                        "rolbypassrls",
                    ),
                    start=1,
                )
            )
            if any(attributes):
                elevated_retired.append(_row_value(retired, "rolname"))
        if elevated_retired:
            raise RuntimeError(
                "retired admin roles retain LOGIN or elevated attributes: " + ", ".join(elevated_retired)
            )
    cur.execute(
        "SELECT rolname,rolcanlogin,rolsuper FROM pg_roles WHERE rolname=ANY(%s) ORDER BY rolname",
        (list(break_glass_roles),),
    )
    invalid_break_glass = [
        _row_value(row, "rolname")
        for row in cur.fetchall()
        if not _row_value(row, "rolcanlogin", 1) or not _row_value(row, "rolsuper", 2)
    ]
    if invalid_break_glass:
        raise RuntimeError("break-glass roles must remain LOGIN SUPERUSER: " + ", ".join(invalid_break_glass))
    effective = _effective_connect_grantees(cur)
    reconnect_roles = {row["role_name"] for row in effective if row["reconnect_capable"]}
    cur.execute(
        "SELECT rolname FROM pg_roles WHERE rolname=ANY(%s) AND rolcanlogin ORDER BY rolname",
        (list(connect_allowlist),),
    )
    expected_reconnect_roles = {_row_value(row, "rolname") for row in cur.fetchall()}
    if reconnect_roles != expected_reconnect_roles:
        raise RuntimeError(
            "effective reconnect roles do not match exact allowlist after inspecting all roles: "
            f"expected={sorted(expected_reconnect_roles)!r}, actual={sorted(reconnect_roles)!r}"
        )
    for name, argument_types in functions:
        signature = f"public.{name}({','.join(argument_types).lower()})"
        cur.execute(
            "SELECT pg_catalog.has_function_privilege(%s,%s,'EXECUTE') AS allowed",
            (role_name, signature),
        )
        if not _row_value(cur.fetchone(), "allowed"):
            raise RuntimeError(f"fleet worker required function privilege is missing: {signature}")
    return effective


def validate_runtime_principal(conn, *, worker_id: str, contract: str) -> RuntimePrincipal:
    """Fail closed unless this login is the exact mapped per-node principal."""
    if not worker_id or contract not in _CONTRACT_FUNCTIONS:
        raise RuntimeError("worker identity and supported contract are required")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT session_user AS session_name,current_user AS current_name,"
            "public.fleet_worker_identity() AS principal"
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("runtime database login has no fleet principal mapping")
    principal = _row_value(row, "principal", 2)
    if principal is None:
        raise RuntimeError("runtime database login has no fleet principal mapping")
    identity = RuntimePrincipal(
        session_user=_row_value(row, "session_name"),
        current_user=_row_value(row, "current_name", 1),
        worker_id=principal["worker_id"],
        contract=principal["contract"],
    )
    if identity.session_user != identity.current_user:
        raise RuntimeError("runtime database role switching is forbidden")
    if identity.session_user in {"postgres", "fleet_worker"}:
        raise RuntimeError("runtime database login must be a unique mapped per-node role")
    if identity.worker_id != worker_id:
        raise RuntimeError(f"worker identity mismatch: mapped={identity.worker_id!r}, configured={worker_id!r}")
    if identity.contract != contract:
        raise RuntimeError(f"worker contract mismatch: mapped={identity.contract!r}, configured={contract!r}")
    return identity


def _require_pg18_authority_catalog(cur) -> None:
    from applypilot.brain.schema import require_pg18_authority_catalog

    require_pg18_authority_catalog(cur)


def ensure_brain_candidate_roles_in_transaction(
    cur,
    *,
    reader_role: str = BRAIN_CANDIDATE_READER_ROLE,
    writer_role: str = BRAIN_CANDIDATE_WRITER_ROLE,
    reader_approved_grantees: tuple[str, ...] = (),
    writer_approved_grantees: tuple[str, ...] = (),
) -> CandidateRoleReconciliationReceipt:
    """Reconcile V5 candidate capabilities using the caller's active transaction."""
    _require_pg18_authority_catalog(cur)
    if reader_role == writer_role:
        raise ValueError("candidate reader and writer roles must be distinct")
    for role_name in (reader_role, writer_role):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]{0,62}", role_name):
            raise ValueError("candidate role names must be plain PostgreSQL identifiers")
    cur.execute("SET LOCAL search_path=pg_catalog")
    cur.execute("SELECT rolsuper,rolcreaterole FROM pg_roles WHERE rolname=current_user")
    row = cur.fetchone()
    if row is None or not (row["rolsuper"] or row["rolcreaterole"]):
        raise RuntimeError("candidate role reconciliation requires CREATEROLE or provider-admin authority")
    is_superuser = bool(row["rolsuper"])
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (_CANDIDATE_ROLE_LOCK_KEY,))
    for role_name in (reader_role, writer_role):
        identifier = sql.Identifier(role_name)
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,))
        if cur.fetchone() is None:
            cur.execute(sql.SQL("CREATE ROLE {}").format(identifier))
        cur.execute(
            "SELECT rolcanlogin,rolinherit,rolsuper,rolcreatedb,rolcreaterole,"
            "rolreplication,rolbypassrls FROM pg_roles WHERE rolname=%s",
            (role_name,),
        )
        attributes = cur.fetchone()
        if attributes is None or any(
            attributes[name] for name in ("rolsuper", "rolreplication", "rolbypassrls")
        ):
            raise RuntimeError(f"unsafe privileged capability role requires superuser repair: {role_name}")
        attribute_repairs = [
            sql.SQL(clause)
            for enabled, clause in (
                (attributes["rolcanlogin"], "NOLOGIN"),
                (attributes["rolinherit"], "NOINHERIT"),
                (attributes["rolcreatedb"], "NOCREATEDB"),
                (attributes["rolcreaterole"], "NOCREATEROLE"),
            )
            if enabled
        ]
        if attribute_repairs:
            cur.execute(
                sql.SQL("ALTER ROLE {} ").format(identifier)
                + sql.SQL(" ").join(attribute_repairs)
            )
        cur.execute(sql.SQL("REVOKE CREATE ON SCHEMA public FROM {}").format(identifier))
        cur.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(identifier))

    _reconcile_role_memberships(
        cur,
        role_name=reader_role,
        approved_grantee_roles=reader_approved_grantees,
    )
    _reconcile_role_memberships(
        cur,
        role_name=writer_role,
        approved_grantee_roles=writer_approved_grantees,
    )
    cur.execute(
        "SELECT nspname FROM pg_namespace WHERE nspname NOT LIKE 'pg\\_%' ESCAPE '\\' "
        "AND nspname <> 'information_schema'"
    )
    namespaces = [row["nspname"] for row in cur.fetchall()]
    cur.execute(
        "SELECT n.nspname,t.typname FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace "
        "WHERE n.nspname=ANY(%s) AND t.typtype IN ('c','d','e','r','m') "
        "ORDER BY n.nspname,t.typname",
        (namespaces,),
    )
    grantable_types = [(row["nspname"], row["typname"]) for row in cur.fetchall()]
    cur.execute(
        "SELECT object_kind,object_name FROM ("
        "SELECT 'relation' AS object_kind,format('%%I.%%I',n.nspname,c.relname) AS object_name "
        "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE c.relowner IN (SELECT oid FROM pg_roles WHERE rolname=ANY(%s)) "
        "UNION ALL SELECT 'function',format('%%I.%%I',n.nspname,p.proname) "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "WHERE p.proowner IN (SELECT oid FROM pg_roles WHERE rolname=ANY(%s)) "
        "UNION ALL SELECT 'schema',n.nspname FROM pg_namespace n "
        "WHERE n.nspowner IN (SELECT oid FROM pg_roles WHERE rolname=ANY(%s)) "
        "UNION ALL SELECT 'type',format('%%I.%%I',n.nspname,t.typname) "
        "FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace "
        "WHERE t.typowner IN (SELECT oid FROM pg_roles WHERE rolname=ANY(%s)) "
        "UNION ALL SELECT 'database',d.datname FROM pg_database d "
        "WHERE d.datdba IN (SELECT oid FROM pg_roles WHERE rolname=ANY(%s))"
        ") owned ORDER BY object_kind,object_name",
        ([reader_role, writer_role],) * 5,
    )
    owned_objects = [f"{row['object_kind']}:{row['object_name']}" for row in cur.fetchall()]
    if owned_objects:
        raise RuntimeError("candidate roles must not own database objects: " + ", ".join(owned_objects))
    cur.execute("SELECT current_database() AS database_name")
    database = sql.Identifier(cur.fetchone()["database_name"])
    if is_superuser:
        for role_name in (reader_role, writer_role):
            identifier = sql.Identifier(role_name)
            cur.execute(sql.SQL("REVOKE ALL PRIVILEGES ON DATABASE {} FROM {}").format(database, identifier))
            for namespace in namespaces:
                schema_name = sql.Identifier(namespace)
                cur.execute(
                    sql.SQL("REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA {} FROM {}").format(schema_name, identifier)
                )
                cur.execute(
                    sql.SQL("REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA {} FROM {}").format(schema_name, identifier)
                )
                cur.execute(
                    sql.SQL("REVOKE ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA {} FROM {}").format(schema_name, identifier)
                )
                cur.execute(sql.SQL("REVOKE ALL PRIVILEGES ON SCHEMA {} FROM {}").format(schema_name, identifier))
            for namespace, type_name in grantable_types:
                cur.execute(
                    sql.SQL("REVOKE ALL PRIVILEGES ON TYPE {}.{} FROM {}").format(
                        sql.Identifier(namespace), sql.Identifier(type_name), identifier
                    )
                )
        cur.execute(
            "SELECT owner.rolname AS owner_role,n.nspname AS schema_name,da.defaclobjtype "
            "FROM pg_default_acl da JOIN pg_roles owner ON owner.oid=da.defaclrole "
            "LEFT JOIN pg_namespace n ON n.oid=da.defaclnamespace CROSS JOIN LATERAL aclexplode(da.defaclacl) acl "
            "WHERE acl.grantee IN (SELECT oid FROM pg_roles WHERE rolname=ANY(%s))",
            ([reader_role, writer_role],),
        )
        default_acl_kinds = {"r": "TABLES", "S": "SEQUENCES", "f": "FUNCTIONS", "T": "TYPES", "n": "SCHEMAS"}
        for default_acl in cur.fetchall():
            for role_name in (reader_role, writer_role):
                cur.execute(
                    sql.SQL("ALTER DEFAULT PRIVILEGES FOR ROLE {} {} REVOKE ALL ON {} FROM {}").format(
                        sql.Identifier(default_acl["owner_role"]),
                        sql.SQL("")
                        if default_acl["schema_name"] is None
                        else sql.SQL("IN SCHEMA {} ").format(sql.Identifier(default_acl["schema_name"])),
                        sql.SQL(default_acl_kinds[default_acl["defaclobjtype"]]),
                        sql.Identifier(role_name),
                    )
                )
    for role_name in (reader_role, writer_role):
        cur.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(role_name)))

    if not is_superuser:
        cur.execute(
            "SELECT n.nspname,c.relname,grantee.rolname AS grantee,acl.privilege_type "
            "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "CROSS JOIN LATERAL aclexplode(c.relacl) acl "
            "JOIN pg_roles grantee ON grantee.oid=acl.grantee "
            "WHERE grantee.rolname=ANY(%s) AND NOT ("
            "n.nspname='public' AND c.relname=ANY(%s)) LIMIT 1",
            ([reader_role, writer_role], list(_CANDIDATE_ALL_RELATIONS)),
        )
        if cur.fetchone() is not None:
            raise RuntimeError("non-superuser candidate reconciliation found cross-owner ACL leakage")
        cur.execute(
            "SELECT 1 FROM pg_default_acl defaults "
            "CROSS JOIN LATERAL aclexplode(defaults.defaclacl) acl "
            "WHERE acl.grantee IN (SELECT oid FROM pg_roles WHERE rolname=ANY(%s)) LIMIT 1",
            ([reader_role, writer_role],),
        )
        if cur.fetchone() is not None:
            raise RuntimeError("non-superuser candidate reconciliation found default ACL leakage")
        cur.execute("SET LOCAL ROLE brain_schema_migrator")
    for relation in _CANDIDATE_ALL_RELATIONS:
        cur.execute("SELECT to_regclass(%s) AS relation", (f"public.{relation}",))
        if cur.fetchone()["relation"] is None:
            continue
        table = sql.SQL("public.{}").format(sql.Identifier(relation))
        cur.execute(
            sql.SQL("REVOKE ALL PRIVILEGES ON TABLE {} FROM {}, {}").format(
                table, sql.Identifier(reader_role), sql.Identifier(writer_role)
            )
        )
        if relation in _CANDIDATE_READ_RELATIONS:
            cur.execute(sql.SQL("GRANT SELECT ON TABLE {} TO {}").format(table, sql.Identifier(reader_role)))
    selected_publish_signature = None
    for signature in (_CANDIDATE_V5_PUBLISH_SIGNATURE, _CANDIDATE_V4_PUBLISH_SIGNATURE):
        cur.execute("SELECT to_regprocedure(%s) AS procedure", (signature,))
        if cur.fetchone()["procedure"] is not None:
            procedure = sql.SQL(signature)
            cur.execute(
                sql.SQL("REVOKE ALL PRIVILEGES ON FUNCTION {} FROM {}, {}").format(
                    procedure, sql.Identifier(reader_role), sql.Identifier(writer_role)
                )
            )
            if selected_publish_signature is None:
                selected_publish_signature = signature
    if selected_publish_signature is not None:
        procedure = sql.SQL(selected_publish_signature)
        cur.execute(
            sql.SQL("REVOKE ALL PRIVILEGES ON FUNCTION {} FROM {}, {}").format(
                procedure, sql.Identifier(reader_role), sql.Identifier(writer_role)
            )
        )
        cur.execute(sql.SQL("GRANT EXECUTE ON FUNCTION {} TO {}").format(procedure, sql.Identifier(writer_role)))
    cur.execute(
        "SELECT rolname,rolcanlogin,rolinherit,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
        "FROM pg_roles WHERE rolname=ANY(%s)",
        ([reader_role, writer_role],),
    )
    roles = {role["rolname"]: role for role in cur.fetchall()}
    for role_name in (reader_role, writer_role):
        role = roles.get(role_name)
        if role is None or any(
            role[key]
            for key in (
                "rolcanlogin",
                "rolinherit",
                "rolsuper",
                "rolcreatedb",
                "rolcreaterole",
                "rolreplication",
                "rolbypassrls",
            )
        ):
            raise RuntimeError(f"candidate role attributes invalid for {role_name}")
    cur.execute("SELECT rolcreaterole FROM pg_roles WHERE rolname='brain_schema_migrator'")
    migrator = cur.fetchone()
    if migrator is not None and migrator["rolcreaterole"]:
        raise RuntimeError("brain_schema_migrator must not retain CREATEROLE")
    if not is_superuser:
        cur.execute("RESET ROLE")
    return CandidateRoleReconciliationReceipt(
        reader_role=reader_role,
        writer_role=writer_role,
        reconciled_at=datetime.now(timezone.utc).isoformat(),
    )


def ensure_brain_candidate_roles(
    conn,
    *,
    reader_role: str = BRAIN_CANDIDATE_READER_ROLE,
    writer_role: str = BRAIN_CANDIDATE_WRITER_ROLE,
    reader_approved_grantees: tuple[str, ...] = (),
    writer_approved_grantees: tuple[str, ...] = (),
) -> CandidateRoleReconciliationReceipt:
    """Reconcile NOLOGIN V5 candidate capabilities in one standalone transaction."""
    if conn.info.transaction_status.name != "IDLE":
        raise RuntimeError("candidate role reconciliation requires an idle connection")
    try:
        with conn.transaction():
            with conn.cursor() as cur:
                return ensure_brain_candidate_roles_in_transaction(
                    cur,
                    reader_role=reader_role,
                    writer_role=writer_role,
                    reader_approved_grantees=reader_approved_grantees,
                    writer_approved_grantees=writer_approved_grantees,
                )
    except BaseException:
        conn.rollback()
        raise


def _verify_artifact_authority_membership_closure(
    cur,
    *,
    allowed_session_descendant: str | None,
) -> None:
    cur.execute(
        "WITH RECURSIVE closure(root_oid,member_oid,path) AS ("
        "SELECT root.oid,membership.member,ARRAY[root.oid,membership.member] "
        "FROM pg_catalog.pg_roles root "
        "JOIN pg_catalog.pg_auth_members membership ON membership.roleid=root.oid "
        "WHERE root.rolname=ANY(%s) "
        "UNION ALL "
        "SELECT closure.root_oid,membership.member,closure.path||membership.member "
        "FROM closure JOIN pg_catalog.pg_auth_members membership "
        "ON membership.roleid=closure.member_oid "
        "WHERE NOT membership.member=ANY(closure.path)) "
        "SELECT root.rolname AS root_role,member.rolname AS member_role "
        "FROM closure JOIN pg_catalog.pg_roles root ON root.oid=closure.root_oid "
        "JOIN pg_catalog.pg_roles member ON member.oid=closure.member_oid "
        "ORDER BY root.rolname,member.rolname",
        ([BRAIN_ARTIFACT_AUTHORITY_OWNER_ROLE, BRAIN_ARTIFACT_AUTHORITY_WRITER_ROLE],),
    )
    allowed = {
        (BRAIN_ARTIFACT_AUTHORITY_OWNER_ROLE, "brain_schema_migrator"),
    }
    if allowed_session_descendant is not None:
        allowed.add((BRAIN_ARTIFACT_AUTHORITY_OWNER_ROLE, allowed_session_descendant))
    reachable = {
        (_row_value(row, "root_role"), _row_value(row, "member_role", 1))
        for row in cur.fetchall()
    }
    unauthorized = sorted(reachable - allowed)
    if unauthorized:
        rendered = ", ".join(f"{root}->{member}" for root, member in unauthorized)
        raise RuntimeError(
            "artifact authority roles have unauthorized direct or transitive descendants: " + rendered
        )
    missing = {(BRAIN_ARTIFACT_AUTHORITY_OWNER_ROLE, "brain_schema_migrator")} - reachable
    if missing:
        raise RuntimeError("artifact authority owner is not delegated to the fixed schema migrator")


def _controller_membership_rows(cur) -> list[dict[str, Any]]:
    cur.execute(
        "SELECT member.oid AS member_oid,member.rolname AS member_role,"
        "member.rolcanlogin,member.rolinherit,member.rolsuper,member.rolcreatedb,"
        "member.rolcreaterole,member.rolreplication,member.rolbypassrls,"
        "grantor.oid AS grantor_oid,"
        "grantor.rolname AS grantor_role,grantor.rolsuper AS grantor_is_superuser,"
        "membership.admin_option,membership.inherit_option,membership.set_option "
        "FROM pg_auth_members membership "
        "JOIN pg_roles parent ON parent.oid=membership.roleid "
        "JOIN pg_roles member ON member.oid=membership.member "
        "JOIN pg_roles grantor ON grantor.oid=membership.grantor "
        "WHERE parent.rolname='brain_policy_controller' "
        "ORDER BY member.rolname,grantor.rolname"
    )
    return cur.fetchall()


def _controller_membership_is_exact(
    rows: list[dict[str, Any]],
    *,
    controller_role: str,
) -> bool:
    return (
        len(rows) == 1
        and _row_value(rows[0], "member_role") == controller_role
        and bool(_row_value(rows[0], "rolcanlogin"))
        and not bool(_row_value(rows[0], "rolinherit"))
        and not bool(_row_value(rows[0], "rolsuper"))
        and not bool(_row_value(rows[0], "rolcreatedb"))
        and not bool(_row_value(rows[0], "rolcreaterole"))
        and not bool(_row_value(rows[0], "rolreplication"))
        and not bool(_row_value(rows[0], "rolbypassrls"))
        and _row_value(rows[0], "grantor_oid") == 10
        and bool(_row_value(rows[0], "grantor_is_superuser"))
        and not bool(_row_value(rows[0], "admin_option"))
        and not bool(_row_value(rows[0], "inherit_option"))
        and bool(_row_value(rows[0], "set_option"))
    )


def _controller_membership_closure(cur) -> list[int]:
    cur.execute(
        "WITH RECURSIVE closure(member_oid,path) AS ("
        "SELECT membership.member,ARRAY[parent.oid,membership.member] "
        "FROM pg_roles parent JOIN pg_auth_members membership "
        "ON membership.roleid=parent.oid WHERE parent.rolname='brain_policy_controller' "
        "UNION ALL SELECT membership.member,closure.path||membership.member "
        "FROM closure JOIN pg_auth_members membership "
        "ON membership.roleid=closure.member_oid "
        "WHERE NOT membership.member=ANY(closure.path)) "
        "SELECT DISTINCT member_oid FROM closure ORDER BY member_oid"
    )
    return [_row_value(row, "member_oid") for row in cur.fetchall()]


def _require_exact_controller_membership(cur, *, controller_role: str) -> None:
    rows = _controller_membership_rows(cur)
    exact = _controller_membership_is_exact(rows, controller_role=controller_role)
    closure = _controller_membership_closure(cur)
    if not exact or closure != [_row_value(rows[0], "member_oid") if exact else None]:
        raise RuntimeError(
            "exact brain_policy_controller membership contract mismatch: "
            f"expected sole safe OID 10-granted {controller_role!r}, "
            f"got rows={rows!r}, closure={closure!r}"
        )


def _require_persisted_controller_identity_if_v7(cur, *, controller_role: str) -> None:
    cur.execute("SELECT to_regclass('public.brain_v7_topology_contract') AS relation")
    if _row_value(cur.fetchone(), "relation") is None:
        return
    cur.execute("SELECT oid FROM pg_roles WHERE rolname=%s", (controller_role,))
    expected = cur.fetchone()
    if expected is None:
        raise RuntimeError(
            f"persisted brain schema v7 controller identity mismatch: missing {controller_role!r}"
        )
    expected_oid = _row_value(expected, "oid")
    cur.execute(
        "SELECT singleton_id,controller_role_oid "
        "FROM public.brain_v7_topology_contract ORDER BY singleton_id"
    )
    rows = cur.fetchall()
    if not (
        len(rows) == 1
        and _row_value(rows[0], "singleton_id") == 1
        and _row_value(rows[0], "controller_role_oid", 1) == expected_oid
    ):
        raise RuntimeError(
            "persisted brain schema v7 controller identity mismatch: "
            f"expected {controller_role!r} OID {expected_oid}, got {rows!r}"
        )


def _reconcile_exact_controller_membership(cur, *, controller_role: str) -> None:
    cur.execute("SELECT rolname,rolsuper FROM pg_roles WHERE oid=10")
    bootstrap_role = cur.fetchone()
    if bootstrap_role is None or not bool(_row_value(bootstrap_role, "rolsuper", 1)):
        raise RuntimeError("controller membership grantor OID 10 must remain a superuser")
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (controller_role,))
    if cur.fetchone() is None:
        raise RuntimeError(f"controller role does not exist: {controller_role!r}")
    cur.execute(
        sql.SQL(
            "ALTER ROLE {} LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE "
            "NOREPLICATION NOBYPASSRLS"
        ).format(sql.Identifier(controller_role))
    )
    rows = _controller_membership_rows(cur)
    if _controller_membership_is_exact(rows, controller_role=controller_role):
        return
    for row in rows:
        cur.execute(
            sql.SQL("REVOKE brain_policy_controller FROM {} GRANTED BY {} RESTRICT").format(
                sql.Identifier(_row_value(row, "member_role")),
                sql.Identifier(_row_value(row, "grantor_role", 2)),
            )
        )
    cur.execute(
        sql.SQL(
            "GRANT brain_policy_controller TO {} "
            "WITH ADMIN FALSE, INHERIT FALSE, SET TRUE GRANTED BY {}"
        ).format(
            sql.Identifier(controller_role),
            sql.Identifier(_row_value(bootstrap_role, "rolname")),
        )
    )
    _require_exact_controller_membership(cur, controller_role=controller_role)


def ensure_brain_artifact_authority_roles_in_transaction(
    cur,
    *,
    migrator_role: str = "brain_schema_migrator",
) -> tuple[str, str]:
    """Reconcile the fixed NOLOGIN v6 owner/writer capability roles."""
    _require_pg18_authority_catalog(cur)
    if migrator_role != "brain_schema_migrator":
        raise RuntimeError("artifact authority roles require fixed brain_schema_migrator")
    cur.execute("SET LOCAL search_path=pg_catalog")
    cur.execute("SELECT rolsuper OR rolcreaterole AS allowed FROM pg_roles WHERE rolname=current_user")
    row = cur.fetchone()
    if row is None or not _row_value(row, "allowed"):
        raise RuntimeError("artifact authority role reconciliation requires CREATEROLE or provider-admin authority")
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (migrator_role,))
    if cur.fetchone() is None:
        raise RuntimeError("artifact authority role reconciliation requires brain_schema_migrator")
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (_ARTIFACT_AUTHORITY_ROLE_LOCK_KEY,))
    for role_name in (BRAIN_ARTIFACT_AUTHORITY_OWNER_ROLE, BRAIN_ARTIFACT_AUTHORITY_WRITER_ROLE):
        identifier = sql.Identifier(role_name)
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,))
        if cur.fetchone() is None:
            cur.execute(sql.SQL("CREATE ROLE {}").format(identifier))
        cur.execute(
            "SELECT rolcanlogin,rolinherit,rolsuper,rolcreatedb,rolcreaterole,"
            "rolreplication,rolbypassrls FROM pg_roles WHERE rolname=%s",
            (role_name,),
        )
        attributes = cur.fetchone()
        if attributes is None or any(
            attributes[name] for name in ("rolsuper", "rolreplication", "rolbypassrls")
        ):
            raise RuntimeError(f"unsafe privileged capability role requires superuser repair: {role_name}")
        attribute_repairs = [
            sql.SQL(clause)
            for enabled, clause in (
                (attributes["rolcanlogin"], "NOLOGIN"),
                (attributes["rolinherit"], "NOINHERIT"),
                (attributes["rolcreatedb"], "NOCREATEDB"),
                (attributes["rolcreaterole"], "NOCREATEROLE"),
            )
            if enabled
        ]
        if attribute_repairs:
            cur.execute(
                sql.SQL("ALTER ROLE {} ").format(identifier)
                + sql.SQL(" ").join(attribute_repairs)
            )
        cur.execute(sql.SQL("REVOKE CREATE ON SCHEMA public FROM {}").format(identifier))
        cur.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(identifier))
    cur.execute(
        "SELECT grantor.oid AS grantor_oid,grantor.rolname AS grantor_role,"
        "grantor.rolsuper AS grantor_is_superuser,"
        "membership.admin_option,"
        "membership.inherit_option,membership.set_option "
        "FROM pg_auth_members membership "
        "JOIN pg_roles parent ON parent.oid=membership.roleid "
        "JOIN pg_roles member ON member.oid=membership.member "
        "JOIN pg_roles grantor ON grantor.oid=membership.grantor "
        "WHERE parent.rolname=%s AND member.rolname=%s "
        "ORDER BY grantor.rolname",
        (BRAIN_ARTIFACT_AUTHORITY_OWNER_ROLE, migrator_role),
    )
    owner_memberships = cur.fetchall()
    exact_owner_membership = (
        len(owner_memberships) == 1
        and _row_value(owner_memberships[0], "grantor_oid") == 10
        and bool(_row_value(owner_memberships[0], "grantor_is_superuser", 2))
        and not bool(_row_value(owner_memberships[0], "admin_option", 3))
        and not bool(_row_value(owner_memberships[0], "inherit_option", 4))
        and bool(_row_value(owner_memberships[0], "set_option", 5))
    )
    if not exact_owner_membership:
        cur.execute(
            sql.SQL("GRANT {} TO {} WITH INHERIT FALSE, SET TRUE").format(
                sql.Identifier(BRAIN_ARTIFACT_AUTHORITY_OWNER_ROLE), sql.Identifier(migrator_role)
            )
        )
    _reconcile_role_memberships(
        cur,
        role_name=BRAIN_ARTIFACT_AUTHORITY_OWNER_ROLE,
        approved_grantee_roles=(migrator_role,),
        validate_approved_grantee_descendants=False,
    )
    _reconcile_role_memberships(cur, role_name=BRAIN_ARTIFACT_AUTHORITY_WRITER_ROLE)
    return BRAIN_ARTIFACT_AUTHORITY_OWNER_ROLE, BRAIN_ARTIFACT_AUTHORITY_WRITER_ROLE


def _install_brain_authority_in_transaction(
    cur,
    *,
    topology: BootstrapTopology,
) -> CandidateRoleReconciliationReceipt:
    """Install the fixed authority schema; callers cannot inject bootstrap SQL."""
    _require_pg18_authority_catalog(cur)
    if topology.migrator_role != "brain_schema_migrator":
        raise RuntimeError("atomic authority installation requires fixed brain_schema_migrator role")
    if topology.verifier_role != "brain_schema_verifier":
        raise RuntimeError("atomic authority installation requires fixed brain_schema_verifier role")

    # Import locally to keep the fleet role module independent during ordinary runtime use.
    from applypilot.brain.schema import (
        ensure_brain_schema_v1_in_transaction,
        ensure_brain_schema_v7_in_transaction,
        verify_brain_schema_v7_in_transaction,
    )

    cur.execute(
        "SELECT session_user AS session_name,current_database() AS database_name,"
        "(SELECT rolsuper FROM pg_roles WHERE rolname=session_user) AS session_is_superuser"
    )
    identity = cur.fetchone()
    session_name = _row_value(identity, "session_name")
    database = sql.Identifier(_row_value(identity, "database_name", 1))
    migrator = sql.Identifier(topology.migrator_role)
    session_is_superuser = bool(_row_value(identity, "session_is_superuser", 2))
    database_owner_descendant = None

    _require_persisted_controller_identity_if_v7(
        cur,
        controller_role=topology.controller_role,
    )

    if session_is_superuser:
        cur.execute(
            "SELECT owner.oid,owner.rolname FROM pg_database database "
            "JOIN pg_roles owner ON owner.oid=database.datdba "
            "WHERE database.datname=current_database()"
        )
        database_owner_role = cur.fetchone()
        if _row_value(database_owner_role, "oid") == 10:
            cur.execute(
                "SELECT grantor.rolname AS grantor_role FROM pg_auth_members membership "
                "JOIN pg_roles parent ON parent.oid=membership.roleid "
                "JOIN pg_roles member ON member.oid=membership.member "
                "JOIN pg_roles grantor ON grantor.oid=membership.grantor "
                "WHERE parent.rolname=%s AND member.oid=10",
                (topology.migrator_role,),
            )
            for bootstrap_edge in cur.fetchall():
                cur.execute(
                    sql.SQL("REVOKE {} FROM {} GRANTED BY {} RESTRICT").format(
                        migrator,
                        sql.Identifier(_row_value(database_owner_role, "rolname", 1)),
                        sql.Identifier(_row_value(bootstrap_edge, "grantor_role")),
                    )
                )
        else:
            database_owner_descendant = _row_value(database_owner_role, "rolname", 1)

    if not session_is_superuser:
        _require_exact_controller_membership(
            cur,
            controller_role=topology.controller_role,
        )
        cur.execute(
            "SELECT owner.rolname AS owner_name FROM pg_database database "
            "JOIN pg_roles owner ON owner.oid=database.datdba "
            "WHERE database.datname=current_database()"
        )
        actual_database_owner = _row_value(cur.fetchone(), "owner_name")
        cur.execute(
            "SELECT rolcanlogin,rolinherit,rolsuper,rolcreatedb,rolcreaterole,"
            "rolreplication,rolbypassrls FROM pg_roles WHERE rolname=%s",
            (session_name,),
        )
        provider_attributes = cur.fetchone()
        provider_exact = (
            actual_database_owner == session_name
            and provider_attributes is not None
            and bool(_row_value(provider_attributes, "rolcanlogin"))
            and not any(
                bool(_row_value(provider_attributes, attribute))
                for attribute in (
                    "rolinherit",
                    "rolsuper",
                    "rolcreatedb",
                    "rolcreaterole",
                    "rolreplication",
                    "rolbypassrls",
                )
            )
        )
        cur.execute(
            "SELECT rolname,rolcanlogin,rolinherit,rolsuper,rolcreatedb,rolcreaterole,"
            "rolreplication,rolbypassrls FROM pg_roles "
            "WHERE rolname=ANY(%s) ORDER BY rolname",
            ([BRAIN_ARTIFACT_AUTHORITY_OWNER_ROLE, BRAIN_ARTIFACT_AUTHORITY_WRITER_ROLE],),
        )
        provisioned_roles = cur.fetchall()
        roles_exact = len(provisioned_roles) == 2 and all(
            not bool(row[attribute])
            for row in provisioned_roles
            for attribute in (
                "rolcanlogin",
                "rolinherit",
                "rolsuper",
                "rolcreatedb",
                "rolcreaterole",
                "rolreplication",
                "rolbypassrls",
            )
        )
        cur.execute(
            "SELECT parent.rolname AS parent_role,member.rolname AS member_role,"
            "grantor.oid AS grantor_oid,grantor.rolsuper AS grantor_is_superuser,"
            "membership.admin_option,"
            "membership.inherit_option,membership.set_option "
            "FROM pg_auth_members membership "
            "JOIN pg_roles parent ON parent.oid=membership.roleid "
            "JOIN pg_roles member ON member.oid=membership.member "
            "JOIN pg_roles grantor ON grantor.oid=membership.grantor "
            "WHERE parent.rolname=ANY(%s) OR member.rolname=ANY(%s)",
            (
                [BRAIN_ARTIFACT_AUTHORITY_OWNER_ROLE, BRAIN_ARTIFACT_AUTHORITY_WRITER_ROLE],
                [BRAIN_ARTIFACT_AUTHORITY_OWNER_ROLE, BRAIN_ARTIFACT_AUTHORITY_WRITER_ROLE],
            ),
        )
        provisioned_memberships = cur.fetchall()
        membership_exact = (
            len(provisioned_memberships) == 1
            and _row_value(provisioned_memberships[0], "parent_role")
            == BRAIN_ARTIFACT_AUTHORITY_OWNER_ROLE
            and _row_value(provisioned_memberships[0], "member_role", 1) == topology.migrator_role
            and _row_value(provisioned_memberships[0], "grantor_oid", 2) == 10
            and bool(_row_value(provisioned_memberships[0], "grantor_is_superuser", 3))
            and not bool(_row_value(provisioned_memberships[0], "admin_option", 4))
            and not bool(_row_value(provisioned_memberships[0], "inherit_option", 5))
            and bool(_row_value(provisioned_memberships[0], "set_option", 6))
        )
        cur.execute(
            "SELECT parent.rolname AS parent_role,grantor.oid AS grantor_oid,"
            "grantor.rolsuper AS grantor_is_superuser,membership.admin_option,"
            "membership.inherit_option,membership.set_option "
            "FROM pg_auth_members membership "
            "JOIN pg_roles parent ON parent.oid=membership.roleid "
            "JOIN pg_roles member ON member.oid=membership.member "
            "JOIN pg_roles grantor ON grantor.oid=membership.grantor "
            "WHERE member.rolname=%s",
            (session_name,),
        )
        provider_memberships = cur.fetchall()
        provider_membership_exact = (
            len(provider_memberships) == 1
            and _row_value(provider_memberships[0], "parent_role") == topology.migrator_role
            and _row_value(provider_memberships[0], "grantor_oid", 1) == 10
            and bool(_row_value(provider_memberships[0], "grantor_is_superuser", 2))
            and not bool(_row_value(provider_memberships[0], "admin_option", 3))
            and not bool(_row_value(provider_memberships[0], "inherit_option", 4))
            and bool(_row_value(provider_memberships[0], "set_option", 5))
        )
        if not provider_exact or not roles_exact or not membership_exact or not provider_membership_exact:
            raise RuntimeError(
                "non-superuser authority installation requires privileged pre-provisioning "
                "of exact owner/writer roles and OID 10-granted permanent memberships"
            )
    if session_is_superuser:
        candidate_roles = ensure_brain_candidate_roles_in_transaction(cur)
        ensure_brain_artifact_authority_roles_in_transaction(
            cur, migrator_role=topology.migrator_role
        )
    else:
        candidate_roles = CandidateRoleReconciliationReceipt(
            reader_role=BRAIN_CANDIDATE_READER_ROLE,
            writer_role=BRAIN_CANDIDATE_WRITER_ROLE,
            reconciled_at=datetime.now(timezone.utc).isoformat(),
        )

    cur.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_database database "
        "CROSS JOIN LATERAL aclexplode(database.datacl) acl "
        "WHERE database.datname=current_database() "
        "AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s) "
        "AND acl.privilege_type='CREATE') AS present",
        (topology.migrator_role,),
    )
    create_preexisted = bool(_row_value(cur.fetchone(), "present"))
    if not create_preexisted:
        cur.execute(sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(database, migrator))
    cur.execute(
        "SELECT owner.rolname AS owner_name FROM pg_namespace namespace "
        "JOIN pg_roles owner ON owner.oid=namespace.nspowner "
        "WHERE namespace.nspname='public'"
    )
    public_owner_before = _row_value(cur.fetchone(), "owner_name")
    # Dedicated non-superuser providers preserve public's pg_database_owner
    # ownership. PostgreSQL forbids explicit members of that predefined role,
    # so the permanent migrator SET edge plus schema grant options are used.
    public_owner_changed = session_is_superuser and public_owner_before != topology.migrator_role
    if public_owner_changed:
        cur.execute(sql.SQL("ALTER SCHEMA public OWNER TO {}").format(migrator))
    else:
        cur.execute(
            sql.SQL(
                "GRANT USAGE, CREATE ON SCHEMA public TO {} WITH GRANT OPTION"
            ).format(migrator)
        )
    if session_is_superuser:
        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM pg_roles WHERE rolname='brain_policy_controller') AS present"
        )
        if not bool(_row_value(cur.fetchone(), "present")):
            cur.execute(sql.SQL("SET LOCAL ROLE {}").format(migrator))
            ensure_brain_schema_v1_in_transaction(cur)
            cur.execute("RESET ROLE")
        _reconcile_exact_controller_membership(
            cur,
            controller_role=topology.controller_role,
        )
    else:
        _require_exact_controller_membership(
            cur,
            controller_role=topology.controller_role,
        )
    cur.execute(sql.SQL("SET LOCAL ROLE {}").format(migrator))
    ensure_brain_schema_v7_in_transaction(
        cur,
        expected_controller_role=topology.controller_role,
    )
    cur.execute("RESET ROLE")
    if public_owner_changed:
        cur.execute(
            sql.SQL("ALTER SCHEMA public OWNER TO {}").format(sql.Identifier(public_owner_before))
        )
    # ALTER SCHEMA OWNER normalizes away a redundant grant to the temporary
    # owner. Restore verifier access only after returning ownership, without
    # retaining CREATE on the schema.
    cur.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(migrator))
    cur.execute(
        "SELECT pg_get_userbyid(nspowner) AS owner_name "
        "FROM pg_namespace WHERE nspname='public'"
    )
    if _row_value(cur.fetchone(), "owner_name") != public_owner_before:
        raise RuntimeError("public schema owner was not restored after fixed authority installation")
    cur.execute(
        "SELECT has_schema_privilege("
        "'brain_artifact_authority_owner','public','CREATE') AS allowed"
    )
    if bool(_row_value(cur.fetchone(), "allowed")):
        raise RuntimeError(
            "brain_artifact_authority_owner retained CREATE on public after fixed authority installation"
        )
    if session_is_superuser:
        candidate_roles = ensure_brain_candidate_roles_in_transaction(cur)
        ensure_brain_artifact_authority_roles_in_transaction(
            cur, migrator_role=topology.migrator_role
        )
    if not create_preexisted:
        cur.execute(sql.SQL("REVOKE CREATE ON DATABASE {} FROM {}").format(database, migrator))
    _verify_artifact_authority_membership_closure(
        cur,
        allowed_session_descendant=(
            database_owner_descendant if session_is_superuser else session_name
        ),
    )
    cur.execute(sql.SQL("SET LOCAL ROLE {}").format(migrator))
    verify_brain_schema_v7_in_transaction(cur)
    cur.execute("RESET ROLE")
    _require_exact_controller_membership(
        cur,
        controller_role=topology.controller_role,
    )
    return candidate_roles


def _bootstrap_object_rollback_sql(cur, ownership: tuple[dict[str, Any], ...]) -> str:
    statements: list[str] = []
    for item in ownership:
        object_kind = item["object_kind"]
        owner = sql.Identifier(item["owner_before"])
        if object_kind == "schema":
            statement = sql.SQL("ALTER SCHEMA {} OWNER TO {}").format(sql.Identifier(item["schema_name"]), owner)
        elif object_kind in {"function", "procedure"}:
            identity = sql.SQL("{}.{}({})").format(
                sql.Identifier(item["schema_name"]),
                sql.Identifier(item["object_name"]),
                sql.SQL(item["arguments"]),
            )
            statement = sql.SQL("ALTER {} {} OWNER TO {}").format(sql.SQL(object_kind.upper()), identity, owner)
        else:
            statement = sql.SQL("ALTER {} {} OWNER TO {}").format(
                sql.SQL(object_kind.upper()),
                sql.Identifier(item["schema_name"], item["object_name"]),
                owner,
            )
        statements.append(statement.as_string(cur.connection) + ";")
    return "\n".join(statements) + ("\n" if statements else "")


def _conditional_role_statement(cur, *, role_name: str, statement: str) -> str:
    """Execute a rollback statement only when its referenced role exists."""
    return (
        sql.SQL(
            "DO $applypilot_rollback$ BEGIN "
            "IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname={}) THEN "
            "EXECUTE {}; END IF; END $applypilot_rollback$;"
        )
        .format(sql.Literal(role_name), sql.Literal(statement))
        .as_string(cur.connection)
    )


def _conditional_role_pair_statement(
    cur,
    *,
    first_role: str,
    second_role: str,
    statement: str,
) -> str:
    return (
        sql.SQL(
            "DO $applypilot_rollback$ BEGIN "
            "IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname={}) "
            "AND EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname={}) THEN "
            "EXECUTE {}; END IF; END $applypilot_rollback$;"
        )
        .format(sql.Literal(first_role), sql.Literal(second_role), sql.Literal(statement))
        .as_string(cur.connection)
    )


def _catalog_derived_authority_containment_sql(
    cur,
    *,
    database_name: str,
    infrastructure_superuser_roles: tuple[str, ...],
) -> tuple[str, str]:
    """Return catalog-derived containment plus a postcondition assertion.

    The V4-V7 verifier authenticates the complete ``brain_*`` catalog before a
    forward rollback receipt can be issued.  At rollback execution time this
    SQL snapshots every catalog owner and explicit ACL grantee, recursively
    follows all membership descendants, and contains that snapshot.  This
    deliberately avoids a function-name or two-edge allowlist that can fall
    behind an additive authority schema.
    """

    if infrastructure_superuser_roles:
        break_glass = sql.SQL("ARRAY[{}]::text[]").format(
            sql.SQL(",").join(sql.Literal(role) for role in infrastructure_superuser_roles)
        )
    else:
        break_glass = sql.SQL("ARRAY[]::text[]")
    database = sql.Literal(database_name)
    inventory_and_containment = sql.SQL(
        "DROP TABLE IF EXISTS pg_temp.applypilot_authority_containment_roles;\n"
        "CREATE TEMPORARY TABLE applypilot_authority_containment_roles ON COMMIT DROP AS\n"
        "WITH RECURSIVE authority_roots(role_oid) AS (\n"
        "  SELECT function.proowner FROM pg_catalog.pg_proc function\n"
        "  JOIN pg_catalog.pg_namespace namespace ON namespace.oid=function.pronamespace\n"
        "  WHERE namespace.nspname='public' AND left(function.proname,6)='brain_'\n"
        "  UNION\n"
        "  SELECT acl.grantee FROM pg_catalog.pg_proc function\n"
        "  JOIN pg_catalog.pg_namespace namespace ON namespace.oid=function.pronamespace\n"
        "  CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(function.proacl,"
        "pg_catalog.acldefault('f',function.proowner))) acl\n"
        "  WHERE namespace.nspname='public' AND left(function.proname,6)='brain_' "
        "AND acl.grantee<>0\n"
        "  UNION\n"
        "  SELECT relation.relowner FROM pg_catalog.pg_class relation\n"
        "  JOIN pg_catalog.pg_namespace namespace ON namespace.oid=relation.relnamespace\n"
        "  WHERE namespace.nspname='public' AND left(relation.relname,6)='brain_'\n"
        "  UNION\n"
        "  SELECT acl.grantee FROM pg_catalog.pg_class relation\n"
        "  JOIN pg_catalog.pg_namespace namespace ON namespace.oid=relation.relnamespace\n"
        "  CROSS JOIN LATERAL pg_catalog.aclexplode(relation.relacl) acl\n"
        "  WHERE namespace.nspname='public' AND left(relation.relname,6)='brain_' "
        "AND acl.grantee<>0\n"
        "  UNION\n"
        "  SELECT acl.grantee FROM pg_catalog.pg_namespace namespace\n"
        "  CROSS JOIN LATERAL pg_catalog.aclexplode(namespace.nspacl) acl\n"
        "  WHERE namespace.nspname IN ('public','brain_archive') AND acl.grantee<>0\n"
        "), authority_closure(role_oid,path) AS (\n"
        "  SELECT role_oid,ARRAY[role_oid] FROM authority_roots\n"
        "  UNION ALL\n"
        "  SELECT membership.member,closure.path||membership.member\n"
        "  FROM authority_closure closure\n"
        "  JOIN pg_catalog.pg_auth_members membership ON membership.roleid=closure.role_oid\n"
        "  WHERE NOT membership.member=ANY(closure.path)\n"
        ") SELECT DISTINCT role_oid FROM authority_closure;\n"
        "DO $applypilot_containment$\n"
        "DECLARE affected record; edge record; function_acl record;\n"
        "BEGIN\n"
        "  FOR affected IN\n"
        "    SELECT role.rolname FROM pg_temp.applypilot_authority_containment_roles contained\n"
        "    JOIN pg_catalog.pg_roles role ON role.oid=contained.role_oid\n"
        "    WHERE role.rolcanlogin AND NOT role.rolname=ANY({})\n"
        "    ORDER BY role.rolname\n"
        "  LOOP\n"
        "    EXECUTE pg_catalog.format("
        "'ALTER ROLE %I NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS',"
        "affected.rolname);\n"
        "    EXECUTE pg_catalog.format("
        "'REVOKE CONNECT, CREATE, TEMPORARY ON DATABASE %I FROM %I',{},affected.rolname);\n"
        "  END LOOP;\n"
        "  FOR edge IN\n"
        "    SELECT parent.rolname AS parent_role,member.rolname AS member_role\n"
        "    FROM pg_catalog.pg_auth_members membership\n"
        "    JOIN pg_catalog.pg_roles parent ON parent.oid=membership.roleid\n"
        "    JOIN pg_catalog.pg_roles member ON member.oid=membership.member\n"
        "    WHERE membership.roleid IN ("
        "SELECT role_oid FROM pg_temp.applypilot_authority_containment_roles)\n"
        "    ORDER BY parent.rolname,member.rolname\n"
        "  LOOP\n"
        "    EXECUTE pg_catalog.format('REVOKE %I FROM %I CASCADE',edge.parent_role,edge.member_role);\n"
        "  END LOOP;\n"
        "  FOR function_acl IN\n"
        "    SELECT pg_catalog.format('%I.%I(%s)',namespace.nspname,function.proname,"
        "pg_catalog.pg_get_function_identity_arguments(function.oid)) AS signature,\n"
        "           CASE WHEN acl.grantee=0 THEN 'PUBLIC' ELSE grantee.rolname END AS grantee\n"
        "    FROM pg_catalog.pg_proc function\n"
        "    JOIN pg_catalog.pg_namespace namespace ON namespace.oid=function.pronamespace\n"
        "    CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(function.proacl,"
        "pg_catalog.acldefault('f',function.proowner))) acl\n"
        "    LEFT JOIN pg_catalog.pg_roles grantee ON grantee.oid=acl.grantee\n"
        "    WHERE namespace.nspname='public' AND left(function.proname,6)='brain_'\n"
        "      AND function.prokind='f' AND acl.privilege_type='EXECUTE' "
        "AND acl.grantee<>function.proowner\n"
        "    ORDER BY function.oid,acl.grantee\n"
        "  LOOP\n"
        "    IF function_acl.grantee='PUBLIC' THEN\n"
        "      EXECUTE pg_catalog.format("
        "'REVOKE ALL PRIVILEGES ON FUNCTION %s FROM PUBLIC',function_acl.signature);\n"
        "    ELSE\n"
        "      EXECUTE pg_catalog.format("
        "'REVOKE ALL PRIVILEGES ON FUNCTION %s FROM %I',"
        "function_acl.signature,function_acl.grantee);\n"
        "    END IF;\n"
        "  END LOOP;\n"
        "END $applypilot_containment$;"
    ).format(break_glass, database).as_string(cur.connection)
    assertion = sql.SQL(
        "DO $applypilot_containment_assert$\n"
        "BEGIN\n"
        "  IF EXISTS (\n"
        "    SELECT 1 FROM pg_catalog.pg_proc function\n"
        "    JOIN pg_catalog.pg_namespace namespace ON namespace.oid=function.pronamespace\n"
        "    CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(function.proacl,"
        "pg_catalog.acldefault('f',function.proowner))) acl\n"
        "    WHERE namespace.nspname='public' AND left(function.proname,6)='brain_'\n"
        "      AND function.prokind='f' AND acl.privilege_type='EXECUTE' "
        "AND acl.grantee<>function.proowner\n"
        "  ) THEN RAISE EXCEPTION 'authority containment left a non-owner brain function EXECUTE route';\n"
        "  END IF;\n"
        "  IF EXISTS (\n"
        "    SELECT 1 FROM pg_catalog.pg_auth_members membership\n"
        "    WHERE membership.roleid IN ("
        "SELECT role_oid FROM pg_temp.applypilot_authority_containment_roles)\n"
        "  ) THEN RAISE EXCEPTION 'authority containment left a recursive SET ROLE route';\n"
        "  END IF;\n"
        "  IF EXISTS (\n"
        "    SELECT 1 FROM pg_temp.applypilot_authority_containment_roles contained\n"
        "    JOIN pg_catalog.pg_roles role ON role.oid=contained.role_oid\n"
        "    WHERE role.rolcanlogin AND NOT role.rolname=ANY({})\n"
        "  ) THEN RAISE EXCEPTION 'authority containment left an affected LOGIN provider';\n"
        "  END IF;\n"
        "END $applypilot_containment_assert$;"
    ).format(break_glass).as_string(cur.connection)
    return inventory_and_containment, assertion


def _forward_v7_containment_sql(
    cur,
    *,
    topology: BootstrapTopology,
    database_name: str,
    database_owner_before: str,
    ownership: tuple[dict[str, Any], ...],
    other_databases: tuple[dict[str, Any], ...],
    retired_memberships: tuple[dict[str, Any], ...],
) -> str:
    """Contain V4-V7 authority capabilities while preserving additive evidence data."""
    database = sql.Identifier(database_name).as_string(cur.connection)
    containment_sql, containment_assertion_sql = _catalog_derived_authority_containment_sql(
        cur,
        database_name=database_name,
        infrastructure_superuser_roles=topology.infrastructure_superuser_roles,
    )
    statements = [
        containment_sql,
        f"REVOKE CONNECT, CREATE, TEMPORARY ON DATABASE {database} FROM PUBLIC;",
    ]
    statements.append(
        sql.SQL("ALTER DATABASE {} OWNER TO {};")
        .format(sql.Identifier(database_name), sql.Identifier(database_owner_before))
        .as_string(cur.connection)
    )
    statements.append(
        _other_database_connect_rollback_sql(
            cur,
            databases=other_databases,
            infrastructure_superuser_roles=topology.infrastructure_superuser_roles,
        ).rstrip()
    )
    statements.append(_bootstrap_object_rollback_sql(cur, ownership).rstrip())
    # Forward containment must not recreate retired or provider-admin edges.
    # Their pre-upgrade shape remains authenticated in the receipt inventory.
    _ = retired_memberships
    statements.append(containment_assertion_sql)
    return "\n".join(statement for statement in statements if statement) + "\n"


# Kept as a private compatibility alias for callers preparing older topology-only
# packets; all newly authenticated authority receipts use V7 containment mode.
_forward_v5_deactivation_sql = _forward_v7_containment_sql


def _other_database_inventory(
    cur,
    *,
    infrastructure_superuser_roles: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    cur.execute(
        "SELECT database.datname AS database_name,owner.rolname AS owner_role,database.datconnlimit "
        "FROM pg_database database JOIN pg_roles owner ON owner.oid=database.datdba "
        "WHERE database.datallowconn AND NOT database.datistemplate "
        "AND database.datname<>current_database() ORDER BY database.datname"
    )
    databases = [dict(row) for row in cur.fetchall()]
    if not databases:
        return ()
    database_names = [row["database_name"] for row in databases]
    cur.execute(
        "SELECT datname AS database_name,usename AS role_name,count(*) AS session_count "
        "FROM pg_stat_activity WHERE datname=ANY(%s) AND pid<>pg_backend_pid() "
        "AND usename IS NOT NULL GROUP BY datname,usename ORDER BY datname,usename",
        (database_names,),
    )
    sessions_by_database: dict[str, list[dict[str, Any]]] = {name: [] for name in database_names}
    for row in cur.fetchall():
        session = {
            "role_name": _row_value(row, "role_name", 1),
            "session_count": _row_value(row, "session_count", 2),
        }
        database_name = _row_value(row, "database_name")
        sessions_by_database[database_name].append(session)

    cur.execute(
        "SELECT database.datname AS database_name,"
        "CASE WHEN acl.grantee=0 THEN 'PUBLIC' ELSE grantee.rolname END AS grantee,"
        "acl.privilege_type,acl.is_grantable "
        "FROM pg_database database "
        "CROSS JOIN LATERAL aclexplode(COALESCE(database.datacl,acldefault('d',database.datdba))) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE database.datname=ANY(%s) AND acl.privilege_type=ANY(%s) "
        "ORDER BY database.datname,grantee,acl.privilege_type,acl.is_grantable",
        (database_names, ["CONNECT", "CREATE", "TEMPORARY"]),
    )
    grants_by_database: dict[str, list[dict[str, Any]]] = {name: [] for name in database_names}
    for row in cur.fetchall():
        grants_by_database[_row_value(row, "database_name")].append(
            {
                "grantee": _row_value(row, "grantee", 1),
                "privilege": _row_value(row, "privilege_type", 2),
                "grantable": _row_value(row, "is_grantable", 3),
            }
        )
    return tuple(
        {
            "database_name": row["database_name"],
            "owner_role": row["owner_role"],
            "connection_limit_before": row["datconnlimit"],
            "acl_grants_before": tuple(grants_by_database[row["database_name"]]),
            "connect_grants_before": tuple(
                grant for grant in grants_by_database[row["database_name"]] if grant["privilege"] == "CONNECT"
            ),
            "active_sessions": tuple(sessions_by_database[row["database_name"]]),
        }
        for row in databases
    )


def _other_database_connect_rollback_sql(
    cur,
    *,
    databases: tuple[dict[str, Any], ...],
    infrastructure_superuser_roles: tuple[str, ...],
) -> str:
    statements: list[str] = []
    for database in databases:
        database_identifier = sql.Identifier(database["database_name"])
        statements.append(
            sql.SQL("ALTER DATABASE {} CONNECTION LIMIT {};")
            .format(database_identifier, sql.Literal(database["connection_limit_before"]))
            .as_string(cur.connection)
        )
        touched_roles = {
            database["owner_role"],
            *infrastructure_superuser_roles,
            *(grant["grantee"] for grant in database["acl_grants_before"] if grant["grantee"] != "PUBLIC"),
        }
        statements.append(
            sql.SQL("REVOKE CONNECT, CREATE, TEMPORARY ON DATABASE {} FROM PUBLIC;")
            .format(database_identifier)
            .as_string(cur.connection)
        )
        for role_name in sorted(touched_roles):
            statements.append(
                sql.SQL("REVOKE CONNECT, CREATE, TEMPORARY ON DATABASE {} FROM {};")
                .format(database_identifier, sql.Identifier(role_name))
                .as_string(cur.connection)
            )
        for grant in database["acl_grants_before"]:
            grantee = sql.SQL("PUBLIC") if grant["grantee"] == "PUBLIC" else sql.Identifier(grant["grantee"])
            statement = sql.SQL("GRANT {} ON DATABASE {} TO {}").format(
                sql.SQL(grant["privilege"]), database_identifier, grantee
            )
            if grant["grantable"]:
                statement += sql.SQL(" WITH GRANT OPTION")
            statements.append(statement.as_string(cur.connection) + ";")
    return "\n".join(statements) + ("\n" if statements else "")


def _retired_membership_inventory(
    cur,
    *,
    retired_admin_roles: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    cur.execute(
        "SELECT attname FROM pg_catalog.pg_attribute "
        "WHERE attrelid='pg_catalog.pg_auth_members'::regclass "
        "AND attname=ANY(%s) AND NOT attisdropped",
        (["inherit_option", "set_option"],),
    )
    option_columns = {_row_value(row, "attname") for row in cur.fetchall()}
    inherit_projection = (
        "membership.inherit_option" if "inherit_option" in option_columns else "NULL::boolean AS inherit_option"
    )
    set_projection = "membership.set_option" if "set_option" in option_columns else "NULL::boolean AS set_option"
    cur.execute(
        "SELECT parent.rolname AS parent_role,member.rolname AS member_role,"
        "grantor.rolname AS grantor_role,membership.admin_option,"
        f"{inherit_projection},{set_projection} "
        "FROM pg_catalog.pg_auth_members membership "
        "JOIN pg_catalog.pg_roles parent ON parent.oid=membership.roleid "
        "JOIN pg_catalog.pg_roles member ON member.oid=membership.member "
        "JOIN pg_catalog.pg_roles grantor ON grantor.oid=membership.grantor "
        "WHERE parent.rolname=ANY(%s) OR member.rolname=ANY(%s) "
        "ORDER BY parent.rolname,member.rolname,grantor.rolname",
        (list(retired_admin_roles), list(retired_admin_roles)),
    )
    return tuple(
        {
            "parent_role": _row_value(row, "parent_role"),
            "member_role": _row_value(row, "member_role", 1),
            "grantor_role": _row_value(row, "grantor_role", 2),
            "admin_option": bool(_row_value(row, "admin_option", 3)),
            "inherit_option": _row_value(row, "inherit_option", 4),
            "set_option": _row_value(row, "set_option", 5),
        }
        for row in cur.fetchall()
    )


def _retired_membership_rollback_sql(
    cur,
    *,
    memberships: tuple[dict[str, Any], ...],
) -> str:
    statements: list[str] = []
    for edge in memberships:
        if edge["inherit_option"] is None and edge["set_option"] is None:
            base = sql.SQL("GRANT {} TO {}{} GRANTED BY {}").format(
                sql.Identifier(edge["parent_role"]),
                sql.Identifier(edge["member_role"]),
                sql.SQL(" WITH ADMIN OPTION" if edge["admin_option"] else ""),
                sql.Identifier(edge["grantor_role"]),
            )
        else:
            base = sql.SQL("GRANT {} TO {} WITH ADMIN {} GRANTED BY {}").format(
                sql.Identifier(edge["parent_role"]),
                sql.Identifier(edge["member_role"]),
                sql.SQL("TRUE" if edge["admin_option"] else "FALSE"),
                sql.Identifier(edge["grantor_role"]),
            )
        statements.append(base.as_string(cur.connection) + ";")
        if edge["inherit_option"] is not None:
            statements.append(
                sql.SQL("GRANT {} TO {} WITH INHERIT {} GRANTED BY {};")
                .format(
                    sql.Identifier(edge["parent_role"]),
                    sql.Identifier(edge["member_role"]),
                    sql.SQL("TRUE" if edge["inherit_option"] else "FALSE"),
                    sql.Identifier(edge["grantor_role"]),
                )
                .as_string(cur.connection)
            )
        if edge["set_option"] is not None:
            statements.append(
                sql.SQL("GRANT {} TO {} WITH SET {} GRANTED BY {};")
                .format(
                    sql.Identifier(edge["parent_role"]),
                    sql.Identifier(edge["member_role"]),
                    sql.SQL("TRUE" if edge["set_option"] else "FALSE"),
                    sql.Identifier(edge["grantor_role"]),
                )
                .as_string(cur.connection)
            )
    return "\n".join(statements) + ("\n" if statements else "")


def _close_retired_admin_memberships(
    cur,
    *,
    retired_admin_roles: tuple[str, ...],
) -> None:
    cur.execute(
        "SELECT parent.rolname AS parent_role,member.rolname AS member_role,grantor.rolname AS grantor_role "
        "FROM pg_catalog.pg_auth_members membership "
        "JOIN pg_catalog.pg_roles parent ON parent.oid=membership.roleid "
        "JOIN pg_catalog.pg_roles member ON member.oid=membership.member "
        "JOIN pg_catalog.pg_roles grantor ON grantor.oid=membership.grantor "
        "WHERE grantor.rolname=ANY(%s) ORDER BY parent.rolname,member.rolname,grantor.rolname",
        (list(retired_admin_roles),),
    )
    dependent_grants = [
        f"{_row_value(row, 'parent_role')}->{_row_value(row, 'member_role', 1)} "
        f"granted-by {_row_value(row, 'grantor_role', 2)}"
        for row in cur.fetchall()
    ]
    if dependent_grants:
        raise RuntimeError(
            "retired provider-admin membership dependencies prevent exact non-cascading closure: "
            + ", ".join(dependent_grants)
        )
    for edge in _retired_membership_inventory(cur, retired_admin_roles=retired_admin_roles):
        cur.execute(
            sql.SQL("REVOKE {} FROM {} GRANTED BY {} RESTRICT").format(
                sql.Identifier(edge["parent_role"]),
                sql.Identifier(edge["member_role"]),
                sql.Identifier(edge["grantor_role"]),
            )
        )


def _validate_retired_admin_memberships_closed(
    cur,
    *,
    retired_admin_roles: tuple[str, ...],
) -> None:
    remaining = _retired_membership_inventory(cur, retired_admin_roles=retired_admin_roles)
    if remaining:
        rendered = [f"{edge['parent_role']}->{edge['member_role']}" for edge in remaining]
        raise RuntimeError("retired provider-admin role memberships remain: " + ", ".join(rendered))


def _lock_cluster_security_catalogs(cur) -> None:
    # SHARE blocks external database ACL/topology and role-membership writes while
    # allowing this transaction to upgrade its own lock for the handoff.
    cur.execute(
        "LOCK TABLE pg_catalog.pg_database,pg_catalog.pg_authid,pg_catalog.pg_auth_members IN SHARE MODE"
    )


def _validate_other_database_inventory_unchanged(
    cur,
    *,
    baseline: tuple[dict[str, Any], ...],
    baseline_database_names: tuple[str, ...],
    infrastructure_superuser_roles: tuple[str, ...],
) -> None:
    current = _other_database_inventory(
        cur,
        infrastructure_superuser_roles=infrastructure_superuser_roles,
    )
    normalized_current = tuple(
        {
            **database,
            "connection_limit_before": next(
                (
                    original["connection_limit_before"]
                    for original in baseline
                    if original["database_name"] == database["database_name"]
                    and database["connection_limit_before"] == 0
                ),
                database["connection_limit_before"],
            ),
            "active_sessions": (),
        }
        for database in current
    )
    normalized_baseline = tuple({**database, "active_sessions": ()} for database in baseline)
    baseline_by_name = {database["database_name"]: database for database in baseline}
    current_by_name = {database["database_name"]: database for database in current}
    cur.execute(
        "SELECT datname,datallowconn FROM pg_catalog.pg_database "
        "WHERE NOT datistemplate AND datname<>current_database() ORDER BY datname"
    )
    catalog = {_row_value(row, "datname"): bool(_row_value(row, "datallowconn", 1)) for row in cur.fetchall()}
    if normalized_current == normalized_baseline and set(catalog) == set(baseline_database_names):
        return

    new_database_names = sorted(set(catalog) - set(baseline_database_names))
    for database_name in new_database_names:
        cur.execute(
            sql.SQL("ALTER DATABASE {} CONNECTION LIMIT 0 ALLOW_CONNECTIONS false").format(
                sql.Identifier(database_name)
            )
        )
    for database_name, original in baseline_by_name.items():
        if database_name not in catalog:
            continue
        if not catalog[database_name]:
            cur.execute(
                sql.SQL("ALTER DATABASE {} ALLOW_CONNECTIONS true CONNECTION LIMIT 0").format(
                    sql.Identifier(database_name)
                )
            )
        current_database = current_by_name.get(database_name)
        if current_database is None or (
            current_database["owner_role"] != original["owner_role"]
            or current_database["acl_grants_before"] != original["acl_grants_before"]
        ):
            cur.execute(
                sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                    sql.Identifier(database_name),
                    sql.Identifier(original["owner_role"]),
                )
            )
            cur.execute(
                "SELECT DISTINCT CASE WHEN acl.grantee=0 THEN 'PUBLIC' ELSE grantee.rolname END AS grantee "
                "FROM pg_catalog.pg_database database "
                "CROSS JOIN LATERAL pg_catalog.aclexplode("
                "COALESCE(database.datacl,pg_catalog.acldefault('d',database.datdba))) acl "
                "LEFT JOIN pg_catalog.pg_roles grantee ON grantee.oid=acl.grantee "
                "WHERE database.datname=%s AND acl.privilege_type=ANY(%s) ORDER BY grantee",
                (database_name, ["CONNECT", "CREATE", "TEMPORARY"]),
            )
            for row in cur.fetchall():
                grantee_name = _row_value(row, "grantee")
                grantee = sql.SQL("PUBLIC") if grantee_name == "PUBLIC" else sql.Identifier(grantee_name)
                cur.execute(
                    sql.SQL("REVOKE CONNECT, CREATE, TEMPORARY ON DATABASE {} FROM {}").format(
                        sql.Identifier(database_name), grantee
                    )
                )
            for grant in original["acl_grants_before"]:
                grantee = sql.SQL("PUBLIC") if grant["grantee"] == "PUBLIC" else sql.Identifier(grant["grantee"])
                statement = sql.SQL("GRANT {} ON DATABASE {} TO {}").format(
                    sql.SQL(grant["privilege"]), sql.Identifier(database_name), grantee
                )
                if grant["grantable"]:
                    statement += sql.SQL(" WITH GRANT OPTION")
                cur.execute(statement)
    _terminate_and_validate_other_database_sessions(
        cur,
        databases=tuple({"database_name": database_name} for database_name in sorted(catalog)),
        infrastructure_superuser_roles=infrastructure_superuser_roles,
    )
    raise CrossDatabaseInventoryDriftError(
        "cross-database topology or ACL inventory changed after the admission fence committed; "
        "the reconciled admission fence remains closed"
    )


def _isolate_other_databases(
    cur,
    *,
    databases: tuple[dict[str, Any], ...],
    infrastructure_superuser_roles: tuple[str, ...],
) -> None:
    for database in databases:
        database_identifier = sql.Identifier(database["database_name"])
        allowed = {database["owner_role"], *infrastructure_superuser_roles}
        cur.execute(sql.SQL("REVOKE CONNECT, CREATE, TEMPORARY ON DATABASE {} FROM PUBLIC").format(database_identifier))
        for grant in database["acl_grants_before"]:
            grantee = grant["grantee"]
            if grantee != "PUBLIC" and grantee not in allowed:
                cur.execute(
                    sql.SQL("REVOKE {} ON DATABASE {} FROM {}").format(
                        sql.SQL(grant["privilege"]), database_identifier, sql.Identifier(grantee)
                    )
                )
        for role_name in sorted(allowed):
            cur.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(database_identifier, sql.Identifier(role_name))
            )


def _set_other_database_admission_fence(cur, *, databases: tuple[dict[str, Any], ...]) -> None:
    for database in databases:
        cur.execute(
            sql.SQL("ALTER DATABASE {} CONNECTION LIMIT 0").format(sql.Identifier(database["database_name"]))
        )


def _restore_other_database_admission_fence(cur, *, databases: tuple[dict[str, Any], ...]) -> None:
    for database in databases:
        cur.execute(
            sql.SQL("ALTER DATABASE {} CONNECTION LIMIT {}").format(
                sql.Identifier(database["database_name"]),
                sql.Literal(database["connection_limit_before"]),
            )
        )


def _terminate_and_validate_other_database_sessions(
    cur,
    *,
    databases: tuple[dict[str, Any], ...],
    infrastructure_superuser_roles: tuple[str, ...],
) -> None:
    if not databases:
        return
    database_names = [database["database_name"] for database in databases]
    allowed = list(infrastructure_superuser_roles)
    cur.execute(
        "SELECT pid,datname,usename,pg_terminate_backend(pid) AS terminated "
        "FROM pg_stat_activity WHERE datname=ANY(%s) AND pid<>pg_backend_pid() "
        "AND usename IS NOT NULL AND usename<>ALL(%s) ORDER BY datname,usename,pid",
        (database_names, allowed),
    )
    failed = [
        f"{row['datname']}:{row['usename']}:{row['pid']}" for row in cur.fetchall() if not row["terminated"]
    ]
    survivors: list[str] = []
    for attempt in range(41):
        cur.execute("SELECT pg_stat_clear_snapshot()")
        cur.execute(
            "SELECT datname,usename,count(*) AS session_count FROM pg_stat_activity "
            "WHERE datname=ANY(%s) AND pid<>pg_backend_pid() AND usename IS NOT NULL "
            "AND usename<>ALL(%s) GROUP BY datname,usename ORDER BY datname,usename",
            (database_names, allowed),
        )
        survivors = [f"{row['datname']}:{row['usename']}:{row['session_count']}" for row in cur.fetchall()]
        if not survivors or attempt == 40:
            break
        cur.execute("SELECT pg_sleep(0.05)")
    if failed or survivors:
        raise RuntimeError(
            "cross-database admission fence could not clear non-breakglass sessions: "
            f"failed={failed!r}, survivors={survivors!r}"
        )


def _validate_other_database_fence(cur, *, databases: tuple[dict[str, Any], ...]) -> None:
    if not databases:
        return
    cur.execute(
        "SELECT datname,datconnlimit FROM pg_database WHERE datname=ANY(%s) AND datconnlimit<>0 ORDER BY datname",
        ([database["database_name"] for database in databases],),
    )
    violations = [f"{row['datname']}:{row['datconnlimit']}" for row in cur.fetchall()]
    if violations:
        raise RuntimeError("cross-database admission fence changed before bootstrap commit: " + ", ".join(violations))


def _validate_other_database_isolation(
    cur,
    *,
    databases: tuple[dict[str, Any], ...],
    topology: BootstrapTopology,
) -> None:
    violations: list[str] = []
    for database in databases:
        cur.execute(
            "SELECT acl.privilege_type FROM pg_database database "
            "CROSS JOIN LATERAL aclexplode(COALESCE(database.datacl,acldefault('d',database.datdba))) acl "
            "WHERE database.datname=%s AND acl.grantee=0 AND acl.privilege_type=ANY(%s)",
            (database["database_name"], ["CONNECT", "CREATE", "TEMPORARY"]),
        )
        violations.extend(
            f"{database['database_name']}:PUBLIC:{row['privilege_type']}" for row in cur.fetchall()
        )
        cur.execute(
            "SELECT role.rolname,role.rolsuper,"
            "pg_catalog.has_database_privilege(role.oid,database.oid,'CONNECT') AS allowed "
            "FROM pg_catalog.pg_roles role CROSS JOIN pg_catalog.pg_database database "
            "WHERE role.rolcanlogin AND database.datname=%s ORDER BY role.rolname",
            (database["database_name"],),
        )
        allowed_roles = {database["owner_role"], *topology.infrastructure_superuser_roles}
        for row in cur.fetchall():
            role_name = _row_value(row, "rolname")
            if _row_value(row, "allowed", 2) and role_name not in allowed_roles:
                violations.append(f"{database['database_name']}:{role_name}")
            if _row_value(row, "rolsuper", 1) and role_name not in topology.infrastructure_superuser_roles:
                violations.append(f"{database['database_name']}:{role_name}:unapproved-superuser")
    if violations:
        raise RuntimeError(
            "login role retains effective CONNECT to another database: " + ", ".join(violations)
        )


def _bootstrap_inventory(cur, *, topology: BootstrapTopology) -> dict[str, Any]:
    role_names = (
        topology.database_owner_role,
        topology.controller_role,
        topology.verifier_role,
        topology.migrator_role,
        *topology.retired_admin_roles,
        *topology.infrastructure_superuser_roles,
    )
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]{0,62}", name) for name in role_names):
        raise ValueError("bootstrap role names must be plain PostgreSQL identifiers")
    permanent_roles = {
        topology.database_owner_role,
        topology.controller_role,
        topology.verifier_role,
        topology.migrator_role,
    }
    if len(permanent_roles) != 4:
        raise RuntimeError("bootstrap owner/controller/verifier/migrator roles must be distinct")
    if not topology.retired_admin_roles:
        raise RuntimeError("bootstrap requires at least one retired provider-admin role")
    if permanent_roles & set(topology.retired_admin_roles):
        raise RuntimeError("permanent topology roles must be disjoint from retired admins")
    if set(topology.infrastructure_superuser_roles) & (permanent_roles | set(topology.retired_admin_roles)):
        raise RuntimeError("infrastructure superusers must be disjoint from managed topology roles")

    cur.execute(
        "SELECT session_user AS session_name,current_user AS current_name,current_database() AS database_name,"
        "owner.rolname AS database_owner FROM pg_database d JOIN pg_roles owner ON owner.oid=d.datdba "
        "WHERE d.datname=current_database()"
    )
    database = cur.fetchone()
    session_user = _row_value(database, "session_name")
    if session_user != _row_value(database, "current_name", 1):
        raise RuntimeError("bootstrap forbids SET ROLE and requires the provider-admin session identity")
    if session_user not in topology.infrastructure_superuser_roles:
        raise RuntimeError("bootstrap session_user must be explicitly listed as an infrastructure break-glass role")

    cur.execute(
        "SELECT rolname,rolcanlogin,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
        "FROM pg_roles ORDER BY rolname"
    )
    roles = tuple(
        {
            "role_name": _row_value(row, "rolname"),
            "can_login": _row_value(row, "rolcanlogin", 1),
            "superuser": _row_value(row, "rolsuper", 2),
            "createdb": _row_value(row, "rolcreatedb", 3),
            "createrole": _row_value(row, "rolcreaterole", 4),
            "replication": _row_value(row, "rolreplication", 5),
            "bypassrls": _row_value(row, "rolbypassrls", 6),
        }
        for row in cur.fetchall()
    )
    role_index = {item["role_name"]: item for item in roles}
    admin = role_index.get(session_user)
    if not admin or not admin["can_login"] or not admin["superuser"]:
        raise RuntimeError("bootstrap requires a LOGIN SUPERUSER provider-admin session")
    existing_permanent = sorted(permanent_roles & role_index.keys())
    if existing_permanent:
        raise RuntimeError("one-time bootstrap topology roles already exist: " + ", ".join(existing_permanent))
    missing_retired = sorted(set(topology.retired_admin_roles) - role_index.keys())
    if missing_retired:
        raise RuntimeError("retired provider-admin roles do not exist: " + ", ".join(missing_retired))

    infrastructure = set(topology.infrastructure_superuser_roles)
    missing_infrastructure = sorted(infrastructure - role_index.keys())
    if missing_infrastructure:
        raise RuntimeError("infrastructure superuser roles do not exist: " + ", ".join(missing_infrastructure))
    unsafe_infrastructure = sorted(
        name for name in infrastructure if not role_index[name]["can_login"] or not role_index[name]["superuser"]
    )
    if unsafe_infrastructure:
        raise RuntimeError(
            "infrastructure break-glass roles must be LOGIN SUPERUSER roles: " + ", ".join(unsafe_infrastructure)
        )
    unknown_superusers = sorted(
        item["role_name"]
        for item in roles
        if item["superuser"]
        and item["role_name"] not in set(topology.retired_admin_roles)
        and item["role_name"] not in infrastructure
    )
    if unknown_superusers:
        raise RuntimeError("unknown superuser dependencies block bootstrap: " + ", ".join(unknown_superusers))

    cur.execute(
        "SELECT DISTINCT usename FROM pg_stat_activity WHERE datname=current_database() "
        "AND pid<>pg_backend_pid() AND usename IS NOT NULL ORDER BY usename"
    )
    active_services = tuple(_row_value(row, "usename") for row in cur.fetchall())
    expected_services = set(topology.expected_service_roles)
    unknown_services = sorted(set(active_services) - expected_services)
    missing_services = sorted(expected_services - set(active_services))
    if unknown_services or missing_services:
        raise RuntimeError(
            f"bootstrap active service inventory mismatch: unknown={unknown_services!r}, missing={missing_services!r}"
        )

    cur.execute(
        "SELECT CASE WHEN acl.grantee=0 THEN 'PUBLIC' ELSE grantee.rolname END AS grantee "
        "FROM pg_database d CROSS JOIN LATERAL aclexplode(COALESCE(d.datacl,acldefault('d',d.datdba))) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE d.datname=current_database() AND acl.privilege_type='CONNECT' "
        "AND (acl.grantee=0 OR acl.grantee<>d.datdba) ORDER BY grantee"
    )
    direct_connect = tuple(_row_value(row, "grantee") for row in cur.fetchall())
    unknown_connect = sorted(set(direct_connect) - {"PUBLIC"} - set(topology.retired_admin_roles) - infrastructure)
    if unknown_connect:
        raise RuntimeError("unknown explicit CONNECT dependencies block bootstrap: " + ", ".join(unknown_connect))

    ownership = _transfer_application_ownership(cur, new_owner_role=topology.migrator_role, apply=False)
    retired_memberships = _retired_membership_inventory(
        cur,
        retired_admin_roles=topology.retired_admin_roles,
    )
    other_databases = _other_database_inventory(
        cur,
        infrastructure_superuser_roles=topology.infrastructure_superuser_roles,
    )
    cur.execute(
        "SELECT datname FROM pg_catalog.pg_database "
        "WHERE NOT datistemplate AND datname<>current_database() ORDER BY datname"
    )
    cluster_database_names = tuple(_row_value(row, "datname") for row in cur.fetchall())
    return {
        "session_user": session_user,
        "database_name": _row_value(database, "database_name", 2),
        "database_owner_before": _row_value(database, "database_owner", 3),
        "roles": roles,
        "active_services": active_services,
        "direct_connect_before": direct_connect,
        "application_ownership": ownership,
        "retired_admin_memberships_before": retired_memberships,
        "other_connectable_databases": other_databases,
        "cluster_database_names_before": cluster_database_names,
    }


def _validate_bootstrap_boundary(cur, *, topology: BootstrapTopology) -> tuple[dict[str, Any], ...]:
    expected_attributes = {
        topology.database_owner_role: (True, False, False, False, False, False, False),
        topology.migrator_role: (False, False, False, False, False, False, False),
        topology.verifier_role: (False, False, False, False, False, False, False),
        topology.controller_role: (True, False, False, False, False, False, False),
    }
    for role_name, expected in expected_attributes.items():
        cur.execute(
            "SELECT rolcanlogin,rolinherit,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
            "FROM pg_roles WHERE rolname=%s",
            (role_name,),
        )
        row = cur.fetchone()
        actual = tuple(
            _row_value(row, name, index)
            for index, name in enumerate(
                (
                    "rolcanlogin",
                    "rolinherit",
                    "rolsuper",
                    "rolcreatedb",
                    "rolcreaterole",
                    "rolreplication",
                    "rolbypassrls",
                )
            )
        )
        if actual != expected:
            raise RuntimeError(f"bootstrap role attributes invalid for {role_name}: {actual!r}")
    cur.execute("SELECT to_regclass('public.brain_schema_versions') AS relation")
    if _row_value(cur.fetchone(), "relation") is not None:
        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM public.brain_schema_versions WHERE version=7) AS installed"
        )
        if bool(_row_value(cur.fetchone(), "installed")):
            _require_persisted_controller_identity_if_v7(
                cur,
                controller_role=topology.controller_role,
            )
            _require_exact_controller_membership(
                cur,
                controller_role=topology.controller_role,
            )
    cur.execute(
        "SELECT role_name FROM unnest(%s::text[]) role_name "
        "WHERE pg_has_role(%s,role_name,'MEMBER') ORDER BY role_name",
        (
            [topology.database_owner_role, topology.migrator_role],
            topology.controller_role,
        ),
    )
    forbidden_memberships = [_row_value(row, "role_name") for row in cur.fetchall()]
    if forbidden_memberships:
        raise RuntimeError(
            "persistent controller retains owner or migrator membership: " + ", ".join(forbidden_memberships)
        )
    for role_name in topology.retired_admin_roles:
        cur.execute(
            "SELECT rolcanlogin,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
            "FROM pg_roles WHERE rolname=%s",
            (role_name,),
        )
        row = cur.fetchone()
        if row is None or any(
            _row_value(row, name, index)
            for index, name in enumerate(
                ("rolcanlogin", "rolsuper", "rolcreatedb", "rolcreaterole", "rolreplication", "rolbypassrls")
            )
        ):
            raise RuntimeError(f"retired provider admin remains elevated: {role_name}")
    for role_name in topology.infrastructure_superuser_roles:
        cur.execute(
            "SELECT rolcanlogin,rolsuper FROM pg_roles WHERE rolname=%s",
            (role_name,),
        )
        row = cur.fetchone()
        if row is None or not _row_value(row, "rolcanlogin") or not _row_value(row, "rolsuper", 1):
            raise RuntimeError(f"infrastructure break-glass role was not preserved: {role_name}")

    cur.execute(
        "SELECT owner.rolname FROM pg_database d JOIN pg_roles owner ON owner.oid=d.datdba "
        "WHERE d.datname=current_database()"
    )
    if _row_value(cur.fetchone(), "rolname") != topology.database_owner_role:
        raise RuntimeError("bootstrap database ownership transfer failed")
    cur.execute(
        "SELECT CASE WHEN acl.grantee=0 THEN 'PUBLIC' ELSE grantee.rolname END AS grantee "
        "FROM pg_database d CROSS JOIN LATERAL aclexplode(d.datacl) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE d.datname=current_database() AND acl.privilege_type='CONNECT' ORDER BY grantee"
    )
    actual_connect = {_row_value(row, "grantee") for row in cur.fetchall()}
    expected_connect = {
        topology.database_owner_role,
        topology.controller_role,
        topology.verifier_role,
        *topology.infrastructure_superuser_roles,
    }
    if actual_connect != expected_connect:
        raise RuntimeError(
            f"bootstrap CONNECT ACL mismatch: expected={sorted(expected_connect)!r}, actual={sorted(actual_connect)!r}"
        )
    cur.execute(
        "SELECT acl.privilege_type FROM pg_database d "
        "CROSS JOIN LATERAL aclexplode(COALESCE(d.datacl,acldefault('d',d.datdba))) acl "
        "WHERE d.datname=current_database() AND acl.grantee=0 "
        "AND acl.privilege_type=ANY(%s) ORDER BY acl.privilege_type",
        (["CONNECT", "CREATE", "TEMPORARY"],),
    )
    public_privileges = [_row_value(row, "privilege_type") for row in cur.fetchall()]
    if public_privileges:
        raise RuntimeError("bootstrap PUBLIC database privileges remain: " + ", ".join(public_privileges))
    effective = _effective_connect_grantees(cur)
    reconnect = {row["role_name"] for row in effective if row["reconnect_capable"]}
    expected_reconnect = {
        topology.database_owner_role,
        topology.controller_role,
        *topology.infrastructure_superuser_roles,
    }
    if reconnect != expected_reconnect:
        raise RuntimeError(
            "bootstrap reconnect principals are not controller plus break-glass only: "
            f"expected={sorted(expected_reconnect)!r}, actual={sorted(reconnect)!r}"
        )
    return effective


def bootstrap_database_roles(
    conn,
    controller_password: str,
    *,
    topology: BootstrapTopology,
    evidence_paths: DurableEvidencePaths,
    install_brain_authority: bool = False,
) -> BootstrapReceipt:
    """Perform a fenced role handoff and optional fixed authority install."""
    if conn.autocommit:
        raise RuntimeError("atomic bootstrap requires autocommit to be disabled")
    if conn.info.transaction_status.name != "IDLE":
        raise RuntimeError("bootstrap requires an idle provider-admin connection")
    with conn.cursor() as preflight_cur:
        _require_pg18_authority_catalog(preflight_cur)
    _validate_evidence_paths(evidence_paths)
    verifier = _client_scram_verifier(conn, role_name=topology.controller_role, password=controller_password)
    fence_active = False
    preserve_fence_on_failure = False
    lock_acquired = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_lock(pg_catalog.hashtext(%s))",
                ("applypilot:fleet-role-bootstrap:v1",),
            )
            lock_acquired = True
            cur.execute("SET LOCAL search_path=pg_catalog")
            _lock_cluster_security_catalogs(cur)
            inventory = _bootstrap_inventory(cur, topology=topology)
            connect_allowlist = (
                topology.database_owner_role,
                topology.controller_role,
                topology.verifier_role,
                *topology.infrastructure_superuser_roles,
            )
            rollback_mode = "forward_v7_containment" if install_brain_authority else "topology_exact"
            if install_brain_authority:
                rollback_sql = _forward_v7_containment_sql(
                    cur,
                    topology=topology,
                    database_name=inventory["database_name"],
                    database_owner_before=inventory["database_owner_before"],
                    ownership=inventory["application_ownership"],
                    other_databases=inventory["other_connectable_databases"],
                    retired_memberships=inventory["retired_admin_memberships_before"],
                )
            else:
                rollback_sql = _public_acl_rollback_sql(
                    cur,
                    connect_allowlist=connect_allowlist,
                    database_owner_before=inventory["database_owner_before"],
                    database_owner_after=topology.database_owner_role,
                    retired_admin_roles=topology.retired_admin_roles,
                )
                rollback_sql += _other_database_connect_rollback_sql(
                    cur,
                    databases=inventory["other_connectable_databases"],
                    infrastructure_superuser_roles=topology.infrastructure_superuser_roles,
                )
                rollback_sql += _bootstrap_object_rollback_sql(cur, inventory["application_ownership"])
                rollback_sql += _retired_membership_rollback_sql(
                    cur,
                    memberships=inventory["retired_admin_memberships_before"],
                )
                for role_name in (
                    topology.controller_role,
                    topology.verifier_role,
                    topology.migrator_role,
                    topology.database_owner_role,
                ):
                    rollback_sql += (
                        sql.SQL("DROP ROLE IF EXISTS {};\n")
                        .format(sql.Identifier(role_name))
                        .as_string(conn)
                    )

            prepared_at = datetime.now(timezone.utc).isoformat()
            prepared_inventory = {
                **inventory,
                "atomic_bootstrap": True,
                "automatic_rollback_supported": True,
                "rollback_mode": rollback_mode,
                "commit_outcome_on_interruption": "unknown",
                "legacy_rollback_sql_recovers_v1_v4": False,
                "legacy_rollback_sql_recovers_v1_v5": False,
                "topology": {
                    "database_owner_role": topology.database_owner_role,
                    "controller_role": topology.controller_role,
                    "verifier_role": topology.verifier_role,
                    "migrator_role": topology.migrator_role,
                    "retired_admin_roles": topology.retired_admin_roles,
                    "infrastructure_superuser_roles": topology.infrastructure_superuser_roles,
                },
                "prepared_at": prepared_at,
                "authority_install_requested": install_brain_authority,
                "escalation_required": install_brain_authority,
            }
            _write_preparation_evidence(
                evidence_paths,
                inventory=prepared_inventory,
                rollback_sql=rollback_sql,
            )
            _set_other_database_admission_fence(cur, databases=inventory["other_connectable_databases"])
        conn.commit()
        fence_active = bool(inventory["other_connectable_databases"])

        with conn.cursor() as cur:
            cur.execute("SET LOCAL search_path=pg_catalog")
            _lock_cluster_security_catalogs(cur)
            try:
                _validate_other_database_inventory_unchanged(
                    cur,
                    baseline=inventory["other_connectable_databases"],
                    baseline_database_names=inventory["cluster_database_names_before"],
                    infrastructure_superuser_roles=topology.infrastructure_superuser_roles,
                )
            except CrossDatabaseInventoryDriftError:
                conn.commit()
                preserve_fence_on_failure = True
                raise
            _terminate_and_validate_other_database_sessions(
                cur,
                databases=inventory["other_connectable_databases"],
                infrastructure_superuser_roles=topology.infrastructure_superuser_roles,
            )
            _isolate_other_databases(
                cur,
                databases=inventory["other_connectable_databases"],
                infrastructure_superuser_roles=topology.infrastructure_superuser_roles,
            )

            cur.execute(
                sql.SQL(
                    "CREATE ROLE {} LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
                ).format(sql.Identifier(topology.database_owner_role))
            )
            cur.execute(
                sql.SQL(
                    "CREATE ROLE {} NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
                ).format(sql.Identifier(topology.migrator_role))
            )
            cur.execute(
                sql.SQL(
                    "CREATE ROLE {} NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
                ).format(sql.Identifier(topology.verifier_role))
            )
            cur.execute(
                sql.SQL(
                    "CREATE ROLE {} LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION "
                    "NOBYPASSRLS PASSWORD {}"
                ).format(sql.Identifier(topology.controller_role), sql.Literal(verifier))
            )
            _transfer_application_ownership(cur, new_owner_role=topology.migrator_role)
            database = sql.Identifier(inventory["database_name"])
            cur.execute(
                sql.SQL("ALTER DATABASE {} OWNER TO {}").format(database, sql.Identifier(topology.database_owner_role))
            )
            cur.execute(
                sql.SQL(
                    "GRANT {} TO {} WITH ADMIN FALSE, INHERIT FALSE, SET TRUE"
                ).format(
                    sql.Identifier(topology.migrator_role),
                    sql.Identifier(topology.database_owner_role),
                )
            )
            cur.execute(sql.SQL("REVOKE CONNECT, CREATE, TEMPORARY ON DATABASE {} FROM PUBLIC").format(database))
            for role_name in topology.retired_admin_roles:
                cur.execute(
                    sql.SQL("REVOKE CONNECT ON DATABASE {} FROM {}").format(database, sql.Identifier(role_name))
                )
            for role_name in connect_allowlist:
                cur.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(database, sql.Identifier(role_name)))
            for role_name in topology.retired_admin_roles:
                cur.execute(
                    sql.SQL(
                        "ALTER ROLE {} NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
                    ).format(sql.Identifier(role_name))
                )
            _close_retired_admin_memberships(
                cur,
                retired_admin_roles=topology.retired_admin_roles,
            )
            if install_brain_authority:
                _install_brain_authority_in_transaction(cur, topology=topology)
            _validate_other_database_isolation(
                cur,
                databases=inventory["other_connectable_databases"],
                topology=topology,
            )
            effective = _validate_bootstrap_boundary(cur, topology=topology)
            _validate_retired_admin_memberships_closed(
                cur,
                retired_admin_roles=topology.retired_admin_roles,
            )
            _validate_other_database_fence(cur, databases=inventory["other_connectable_databases"])
            _terminate_and_validate_other_database_sessions(
                cur,
                databases=inventory["other_connectable_databases"],
                infrastructure_superuser_roles=topology.infrastructure_superuser_roles,
            )
            _restore_other_database_admission_fence(cur, databases=inventory["other_connectable_databases"])
        conn.commit()
        fence_active = False
        return BootstrapReceipt(
            database_name=inventory["database_name"],
            session_user=inventory["session_user"],
            topology={
                "database_owner_role": topology.database_owner_role,
                "controller_role": topology.controller_role,
                "verifier_role": topology.verifier_role,
                "migrator_role": topology.migrator_role,
                "retired_admin_roles": topology.retired_admin_roles,
                "infrastructure_superuser_roles": topology.infrastructure_superuser_roles,
            },
            inventory=prepared_inventory,
            effective_connect_grantees=effective,
            rollback_sql=rollback_sql,
            escalation_required=install_brain_authority,
            bootstrapped_at=prepared_at,
        )
    except BaseException:
        conn.rollback()
        if fence_active and not preserve_fence_on_failure:
            try:
                with conn.transaction():
                    with conn.cursor() as cur:
                        _restore_other_database_admission_fence(
                            cur, databases=inventory["other_connectable_databases"]
                        )
                fence_active = False
            except BaseException as recovery_error:
                raise RuntimeError(
                    "bootstrap failed and the cross-database admission fence could not be restored; "
                    "execute the authenticated rollback receipt"
                ) from recovery_error
        raise
    finally:
        if lock_acquired:
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_advisory_unlock(pg_catalog.hashtext(%s))",
                    ("applypilot:fleet-role-bootstrap:v1",),
                )
            conn.commit()


def upgrade_brain_authority_existing_topology(
    conn,
    *,
    topology: BootstrapTopology,
    evidence_paths: DurableEvidencePaths,
    expected_database_name: str,
    expected_system_identifier: str,
) -> BootstrapReceipt:
    """Atomically install fixed brain authority on an existing role topology.

    This is intentionally narrower than :func:`bootstrap_database_roles`: it
    neither creates/replaces the permanent fleet roles nor changes database
    ownership or admission ACLs.  It exists for databases that completed the
    earlier role handoff before the fixed authority schema was introduced.
    """
    if conn.autocommit:
        raise RuntimeError("atomic authority upgrade requires autocommit to be disabled")
    if conn.info.transaction_status.name != "IDLE":
        raise RuntimeError("authority upgrade requires an idle provider-admin connection")
    _validate_evidence_paths(evidence_paths)
    lock_acquired = False
    try:
        with conn.cursor() as cur:
            _require_pg18_authority_catalog(cur)
            cur.execute(
                "SELECT pg_advisory_lock(pg_catalog.hashtext(%s))",
                ("applypilot:brain-authority-existing-topology:v1",),
            )
            lock_acquired = True
            cur.execute("SET LOCAL search_path=pg_catalog")
            _lock_cluster_security_catalogs(cur)
            _lock_existing_authority_runtime_state(cur)
            inventory = _existing_authority_topology_inventory(
                cur,
                topology=topology,
                expected_database_name=expected_database_name,
                expected_system_identifier=expected_system_identifier,
            )
            rollback_sql = _forward_v7_containment_sql(
                cur,
                topology=topology,
                database_name=inventory["database_name"],
                database_owner_before=inventory["database_owner_before"],
                ownership=(),
                other_databases=(),
                retired_memberships=(),
            )
            prepared_at = datetime.now(timezone.utc).isoformat()
            prepared_inventory = {
                **inventory,
                "atomic_bootstrap": False,
                "atomic_existing_topology_upgrade": True,
                "automatic_rollback_supported": True,
                "rollback_mode": "forward_v7_containment",
                "commit_outcome_on_interruption": "unknown",
                "legacy_rollback_sql_recovers_v1_v4": False,
                "legacy_rollback_sql_recovers_v1_v5": False,
                "topology": {
                    "database_owner_role": topology.database_owner_role,
                    "controller_role": topology.controller_role,
                    "verifier_role": topology.verifier_role,
                    "migrator_role": topology.migrator_role,
                    "retired_admin_roles": topology.retired_admin_roles,
                    "infrastructure_superuser_roles": topology.infrastructure_superuser_roles,
                },
                "prepared_at": prepared_at,
                "authority_install_requested": True,
                "escalation_required": True,
            }
            _write_preparation_evidence(
                evidence_paths,
                inventory=prepared_inventory,
                rollback_sql=rollback_sql,
            )
            _transition_existing_authority_topology(cur, topology=topology)
            candidate_roles = _install_brain_authority_in_transaction(cur, topology=topology)
            _existing_authority_topology_inventory(
                cur,
                topology=topology,
                expected_database_name=expected_database_name,
                expected_system_identifier=expected_system_identifier,
                require_v7=True,
            )
        conn.commit()
        return BootstrapReceipt(
            database_name=inventory["database_name"],
            session_user=inventory["session_user"],
            topology=prepared_inventory["topology"],
            inventory={
                **prepared_inventory,
                "candidate_roles": {
                    "reader_role": candidate_roles.reader_role,
                    "writer_role": candidate_roles.writer_role,
                    "reconciled_at": candidate_roles.reconciled_at,
                },
            },
            effective_connect_grantees=(),
            rollback_sql=rollback_sql,
            escalation_required=False,
            bootstrapped_at=prepared_at,
        )
    except BaseException:
        conn.rollback()
        raise
    finally:
        if lock_acquired:
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_advisory_unlock(pg_catalog.hashtext(%s))",
                    ("applypilot:brain-authority-existing-topology:v1",),
                )
            conn.commit()


def _existing_authority_topology_inventory(
    cur,
    *,
    topology: BootstrapTopology,
    expected_database_name: str,
    expected_system_identifier: str,
    require_v7: bool = False,
) -> dict[str, Any]:
    role_names = (
        topology.database_owner_role,
        topology.controller_role,
        topology.verifier_role,
        topology.migrator_role,
        *topology.retired_admin_roles,
        *topology.infrastructure_superuser_roles,
    )
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]{0,62}", name) for name in role_names):
        raise ValueError("authority upgrade role names must be plain PostgreSQL identifiers")
    permanent_roles = {
        topology.database_owner_role,
        topology.controller_role,
        topology.verifier_role,
        topology.migrator_role,
    }
    if len(permanent_roles) != 4:
        raise RuntimeError("authority upgrade owner/controller/verifier/migrator roles must be distinct")
    if not topology.retired_admin_roles:
        raise RuntimeError("authority upgrade requires at least one retired provider-admin role")
    if permanent_roles & set(topology.retired_admin_roles):
        raise RuntimeError("permanent topology roles must be disjoint from retired admins")
    if set(topology.infrastructure_superuser_roles) & (permanent_roles | set(topology.retired_admin_roles)):
        raise RuntimeError("infrastructure superusers must be disjoint from managed topology roles")

    if not expected_database_name or not expected_system_identifier.isdigit():
        raise ValueError("authority upgrade requires exact database and numeric system-identifier pins")
    cur.execute(
        "SELECT session_user AS session_name,current_user AS current_name,current_database() AS database_name,"
        "(pg_control_system()).system_identifier::text AS system_identifier,"
        "owner.rolname AS database_owner FROM pg_database database "
        "JOIN pg_roles owner ON owner.oid=database.datdba WHERE database.datname=current_database()"
    )
    identity = cur.fetchone()
    session_user = _row_value(identity, "session_name")
    if session_user != _row_value(identity, "current_name", 1):
        raise RuntimeError("authority upgrade forbids SET ROLE and requires provider-admin identity")
    if session_user not in topology.infrastructure_superuser_roles:
        raise RuntimeError("authority upgrade session_user must be an explicit infrastructure break-glass role")
    database_name = _row_value(identity, "database_name", 2)
    system_identifier = _row_value(identity, "system_identifier", 3)
    if database_name != expected_database_name or system_identifier != expected_system_identifier:
        raise RuntimeError(
            "authority upgrade target identity mismatch: "
            f"expected={expected_system_identifier}/{expected_database_name}, "
            f"actual={system_identifier}/{database_name}"
        )
    if _row_value(identity, "database_owner", 4) != topology.database_owner_role:
        raise RuntimeError("authority upgrade database owner does not match the declared existing topology")

    cur.execute(
        "SELECT rolname,rolcanlogin,rolinherit,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
        "FROM pg_roles WHERE rolname=ANY(%s) ORDER BY rolname",
        (list(role_names),),
    )
    roles = { _row_value(row, "rolname"): row for row in cur.fetchall() }
    missing = sorted(set(role_names) - set(roles))
    if missing:
        raise RuntimeError("authority upgrade topology roles do not exist: " + ", ".join(missing))
    admin = roles[session_user]
    if not bool(_row_value(admin, "rolcanlogin", 1)) or not bool(_row_value(admin, "rolsuper", 3)):
        raise RuntimeError("authority upgrade requires a LOGIN SUPERUSER provider-admin session")
    expected_attributes = {
        topology.database_owner_role: (None, None, False, False, False, False, False),
        topology.controller_role: (True, None, False, False, False, False, False),
        topology.verifier_role: (False, None, False, False, False, False, False),
        topology.migrator_role: (False, None, False, False, False, False, False),
    }
    attribute_names = (
        "rolcanlogin", "rolinherit", "rolsuper", "rolcreatedb", "rolcreaterole", "rolreplication", "rolbypassrls"
    )
    for role_name, expected in expected_attributes.items():
        actual = tuple(bool(_row_value(roles[role_name], name, index + 1)) for index, name in enumerate(attribute_names))
        if any(want is not None and got != want for got, want in zip(actual, expected, strict=True)):
            raise RuntimeError(f"authority upgrade role attributes invalid for {role_name}: {actual!r}")
    for role_name in topology.retired_admin_roles:
        row = roles[role_name]
        # LOGIN/CREATEDB/CREATEROLE are repairable only for roles explicitly
        # declared retired. Catalog locks, exact membership/ownership checks,
        # and the atomic NOLOGIN/NOCREATEDB/NOCREATEROLE transition fence the
        # legacy capability before the authority roles are created.
        if any(bool(_row_value(row, name, index + 1)) for index, name in enumerate(attribute_names)
               if name in {"rolsuper", "rolreplication", "rolbypassrls"}):
            raise RuntimeError(f"retired provider admin remains elevated: {role_name}")
    for role_name in topology.infrastructure_superuser_roles:
        row = roles[role_name]
        if not bool(_row_value(row, "rolcanlogin", 1)) or not bool(_row_value(row, "rolsuper", 3)):
            raise RuntimeError(f"infrastructure break-glass role is not LOGIN SUPERUSER: {role_name}")
    cur.execute("SELECT rolname FROM pg_roles WHERE rolsuper ORDER BY rolname")
    unknown_superusers = sorted(
        _row_value(row, "rolname") for row in cur.fetchall()
        if _row_value(row, "rolname") not in topology.infrastructure_superuser_roles
    )
    if unknown_superusers:
        raise RuntimeError("unknown superuser dependencies block authority upgrade: " + ", ".join(unknown_superusers))

    cur.execute(
        "SELECT role_name FROM unnest(%s::text[]) role_name "
        "WHERE pg_has_role(%s,role_name,'MEMBER') ORDER BY role_name",
        ([topology.database_owner_role, topology.migrator_role], topology.controller_role),
    )
    forbidden_controller_memberships = [
        _row_value(row, "role_name") for row in cur.fetchall()
    ]
    if forbidden_controller_memberships:
        raise RuntimeError(
            "authority upgrade controller retains owner or migrator membership: "
            + ", ".join(forbidden_controller_memberships)
        )
    cur.execute(
        "SELECT privilege_type FROM pg_database database "
        "CROSS JOIN LATERAL aclexplode(COALESCE(database.datacl,acldefault('d',database.datdba))) acl "
        "WHERE database.datname=current_database() AND acl.grantee=0 "
        "AND privilege_type=ANY(%s) ORDER BY privilege_type",
        (["CONNECT", "CREATE", "TEMPORARY"],),
    )
    public_database_privileges = [
        _row_value(row, "privilege_type") for row in cur.fetchall()
    ]
    if public_database_privileges:
        raise RuntimeError(
            "authority upgrade PUBLIC database privileges remain: "
            + ", ".join(public_database_privileges)
        )
    cur.execute(
        "SELECT usename,state FROM pg_stat_activity WHERE datname=current_database() "
        "AND pid<>pg_backend_pid() AND state='idle in transaction' ORDER BY usename"
    )
    idle_transactions = cur.fetchall()
    if idle_transactions:
        raise RuntimeError(
            f"authority upgrade is blocked by idle-in-transaction sessions: {idle_transactions!r}"
        )

    _verify_existing_authority_ledgers_and_gates(cur, require_v7=require_v7)
    memberships = _retired_membership_inventory(
        cur, retired_admin_roles=topology.retired_admin_roles
    )
    default_acl_dependencies = _retired_default_acl_dependencies(
        cur, retired_admin_roles=topology.retired_admin_roles
    )
    legacy_capability_memberships = (
        _existing_capability_memberships(cur, topology=topology)
        if not require_v7
        else ()
    )
    if not require_v7:
        _reject_retired_role_ownership(cur, retired_admin_roles=topology.retired_admin_roles)
        _validate_existing_retired_memberships(
            cur,
            topology=topology,
            memberships=memberships,
        )
    if require_v7:
        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM public.brain_schema_versions WHERE version=7) AS installed"
        )
        if not bool(_row_value(cur.fetchone(), "installed")):
            raise RuntimeError("authority upgrade did not install brain schema v7")
        _require_persisted_controller_identity_if_v7(cur, controller_role=topology.controller_role)
        _require_exact_controller_membership(cur, controller_role=topology.controller_role)
        _validate_retired_admin_memberships_closed(
            cur, retired_admin_roles=topology.retired_admin_roles
        )
        _verify_existing_provider_membership(cur, topology=topology)
        expected_final_attributes = {
            topology.database_owner_role: (True, False, False, False, False, False, False),
            topology.controller_role: (True, False, False, False, False, False, False),
            topology.verifier_role: (False, False, False, False, False, False, False),
            topology.migrator_role: (False, False, False, False, False, False, False),
        }
        for role_name, expected in expected_final_attributes.items():
            actual = tuple(
                bool(_row_value(roles[role_name], name, index + 1))
                for index, name in enumerate(attribute_names)
            )
            if actual != expected:
                raise RuntimeError(f"authority upgrade final role attributes invalid for {role_name}: {actual!r}")
    return {
        "session_user": session_user,
        "database_name": database_name,
        "system_identifier": system_identifier,
        "database_owner_before": _row_value(identity, "database_owner", 4),
        "retired_admin_memberships_before": memberships,
        "retired_admin_default_acl_dependencies": default_acl_dependencies,
        "legacy_capability_memberships_before": legacy_capability_memberships,
        "database_owner_password_present": _role_password_present(
            cur, topology.database_owner_role
        ),
        "roles": tuple(
            {
                "role_name": name,
                **{
                    key.removeprefix("rol"): bool(_row_value(row, key, index + 1))
                    for index, key in enumerate(attribute_names)
                },
            }
            for name, row in sorted(roles.items())
        ),
    }


def _role_password_present(cur, role_name: str) -> bool:
    cur.execute("SELECT rolpassword IS NOT NULL AS present FROM pg_authid WHERE rolname=%s", (role_name,))
    row = cur.fetchone()
    return bool(_row_value(row, "present"))


def _lock_existing_authority_runtime_state(cur) -> None:
    # Fleet lease functions read fleet_config before leasing from a queue.  Use
    # the same order and hold write-excluding locks through final verification
    # and commit so a paused/zero-live-lease preflight cannot race a worker.
    cur.execute("LOCK TABLE public.fleet_config IN SHARE ROW EXCLUSIVE MODE")
    cur.execute("SELECT id FROM public.fleet_config WHERE id=1 FOR UPDATE")
    if cur.fetchone() is None:
        raise RuntimeError("authority upgrade requires the fleet_config singleton")
    for table_name in ("apply_queue", "compute_queue", "linkedin_queue"):
        cur.execute(
            sql.SQL("LOCK TABLE public.{} IN SHARE ROW EXCLUSIVE MODE").format(
                sql.Identifier(table_name)
            )
        )
    cur.execute(
        "LOCK TABLE public.applypilot_fleet_schema_migrations IN SHARE MODE"
    )


def _verify_existing_authority_ledgers_and_gates(cur, *, require_v7: bool) -> None:
    from applypilot.brain.schema import verify_brain_schema_v1_in_transaction, verify_brain_schema_v7_in_transaction

    cur.execute(
        "SELECT version FROM public.brain_schema_versions ORDER BY version"
    )
    versions = [_row_value(row, "version") for row in cur.fetchall()]
    expected = list(range(1, 8)) if require_v7 else [1, 2, 3]
    if versions != expected:
        raise RuntimeError(f"authority upgrade requires exact brain ledger prefix {expected!r}, got {versions!r}")
    if require_v7:
        verify_brain_schema_v7_in_transaction(cur)
    else:
        verify_brain_schema_v1_in_transaction(cur)
    cur.execute(
        "SELECT migration_id,migration_sha256 FROM public.applypilot_fleet_schema_migrations "
        "ORDER BY migration_id"
    )
    fleet_rows = [tuple(row.values()) if isinstance(row, Mapping) else tuple(row) for row in cur.fetchall()]
    expected_fleet = [
        ("20260715_001_application_authority", "65d10d1a16e9829a7995f48776697e3200daf7a5074147a9a3ed8c0293f56bea"),
        ("20260717_002_lane_specific_canary_pins", "1ee176c419508cd5a9c7004460798d14fd6749addbf293892fd2d859a1b6dd1d"),
    ]
    if fleet_rows not in (expected_fleet, [*expected_fleet, ("20260719_003_harden_application_authority_functions", "f256bfff5aab2a9a2a6c7aaaa72f6ae2021fa3d21d40c9945298aca9d7defa03")]):
        raise RuntimeError(f"authority upgrade fleet migration ledger mismatch: {fleet_rows!r}")
    cur.execute(
        "SELECT paused,ats_paused,ats_apply_mode,linkedin_apply_mode,"
        "canary_enabled,linkedin_canary_enabled FROM public.fleet_config WHERE id=1"
    )
    gate = cur.fetchone()
    if gate is None or not bool(_row_value(gate, "paused")) or not bool(_row_value(gate, "ats_paused", 1)) \
            or _row_value(gate, "ats_apply_mode", 2) != "stopped" \
            or _row_value(gate, "linkedin_apply_mode", 3) != "stopped" \
            or bool(_row_value(gate, "canary_enabled", 4)) \
            or bool(_row_value(gate, "linkedin_canary_enabled", 5)):
        raise RuntimeError("authority upgrade requires paused, stopped, canary-disabled fleet gates")
    for table_name in ("apply_queue", "compute_queue", "linkedin_queue"):
        cur.execute(
            sql.SQL(
                "SELECT count(*) AS count FROM public.{} WHERE status='leased' "
                "AND lease_owner IS NOT NULL AND lease_expires_at>pg_catalog.now()"
            ).format(sql.Identifier(table_name))
        )
        if int(_row_value(cur.fetchone(), "count")) != 0:
            raise RuntimeError(f"authority upgrade requires zero active unexpired leases in {table_name}")


def _validate_existing_retired_memberships(
    cur,
    *,
    topology: BootstrapTopology,
    memberships: tuple[dict[str, Any], ...],
) -> None:
    cur.execute("SELECT role_name FROM public.fleet_worker_principals ORDER BY role_name")
    worker_roles = {_row_value(row, "role_name") for row in cur.fetchall()}
    approved_parents = {topology.database_owner_role, *worker_roles}
    for edge in memberships:
        if edge["member_role"] not in topology.retired_admin_roles \
                or edge["parent_role"] not in approved_parents \
                or edge["grantor_role"] not in topology.infrastructure_superuser_roles:
            raise RuntimeError(f"unexpected retired-admin membership blocks authority upgrade: {edge!r}")


def _reject_retired_role_ownership(cur, *, retired_admin_roles: tuple[str, ...]) -> None:
    cur.execute(
        "SELECT role.rolname,count(*) AS owned_count FROM pg_shdepend dependency "
        "JOIN pg_roles role ON role.oid=dependency.refobjid "
        "WHERE dependency.refclassid='pg_authid'::regclass "
        "AND dependency.deptype='o' AND dependency.classid<>'pg_default_acl'::regclass "
        "AND role.rolname=ANY(%s) "
        "GROUP BY role.rolname ORDER BY role.rolname",
        (list(retired_admin_roles),),
    )
    owned = [
        (_row_value(row, "rolname"), int(_row_value(row, "owned_count", 1)))
        for row in cur.fetchall()
    ]
    if owned:
        raise RuntimeError(f"retired provider admins still own database objects: {owned!r}")
    cur.execute(
        "SELECT grantee,object_kind,object_name FROM ("
        "SELECT grantee,'table'::text AS object_kind,table_schema||'.'||table_name AS object_name "
        "FROM information_schema.table_privileges UNION ALL "
        "SELECT grantee,'routine',routine_schema||'.'||routine_name FROM information_schema.routine_privileges "
        "UNION ALL SELECT grantee,'column',table_schema||'.'||table_name||'.'||column_name "
        "FROM information_schema.column_privileges UNION ALL "
        "SELECT grantee,'usage',object_schema||'.'||object_name FROM information_schema.usage_privileges"
        ") grants WHERE grantee=ANY(%s) ORDER BY grantee,object_kind,object_name",
        (list(retired_admin_roles),),
    )
    grants = cur.fetchall()
    if grants:
        raise RuntimeError(f"retired provider admins retain application ACLs: {grants!r}")


def _retired_default_acl_dependencies(
    cur,
    *,
    retired_admin_roles: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    """Inventory inert cross-database default ACLs owned by retiring roles.

    A pg_default_acl row affects only objects created later by its owning role.
    The same atomic transition makes every declared retired role NOLOGIN and
    NOCREATEROLE and removes all membership paths, so these dependencies are
    preserved as evidence instead of being mutated across another database.
    All other owned-object classes remain a hard preflight failure.
    """
    cur.execute(
        "SELECT role.rolname AS role_name,database.datname AS database_name,"
        "dependency.objid,dependency.objsubid "
        "FROM pg_shdepend dependency JOIN pg_roles role ON role.oid=dependency.refobjid "
        "LEFT JOIN pg_database database ON database.oid=dependency.dbid "
        "WHERE dependency.refclassid='pg_authid'::regclass "
        "AND dependency.deptype='o' AND dependency.classid='pg_default_acl'::regclass "
        "AND role.rolname=ANY(%s) ORDER BY role.rolname,database.datname,dependency.objid",
        (list(retired_admin_roles),),
    )
    return tuple(dict(row) for row in cur.fetchall())


def _verify_existing_provider_membership(cur, *, topology: BootstrapTopology) -> None:
    cur.execute(
        "SELECT parent.rolname AS parent_role,grantor.oid AS grantor_oid,"
        "membership.admin_option,membership.inherit_option,membership.set_option "
        "FROM pg_auth_members membership JOIN pg_roles parent ON parent.oid=membership.roleid "
        "JOIN pg_roles member ON member.oid=membership.member "
        "JOIN pg_roles grantor ON grantor.oid=membership.grantor WHERE member.rolname=%s",
        (topology.database_owner_role,),
    )
    rows = cur.fetchall()
    if not (
        len(rows) == 1
        and _row_value(rows[0], "parent_role") == topology.migrator_role
        and _row_value(rows[0], "grantor_oid", 1) == 10
        and not bool(_row_value(rows[0], "admin_option", 2))
        and not bool(_row_value(rows[0], "inherit_option", 3))
        and bool(_row_value(rows[0], "set_option", 4))
    ):
        raise RuntimeError(f"authority upgrade provider membership mismatch: {rows!r}")


def _existing_capability_memberships(
    cur,
    *,
    topology: BootstrapTopology,
) -> tuple[dict[str, Any], ...]:
    cur.execute(
        "SELECT parent.rolname AS parent_role,member.rolname AS member_role,"
        "grantor.rolname AS grantor_role,membership.admin_option,"
        "membership.inherit_option,membership.set_option "
        "FROM pg_auth_members membership JOIN pg_roles parent ON parent.oid=membership.roleid "
        "JOIN pg_roles member ON member.oid=membership.member "
        "JOIN pg_roles grantor ON grantor.oid=membership.grantor "
        "WHERE parent.rolname=ANY(%s) OR member.rolname=ANY(%s) "
        "ORDER BY parent.rolname,member.rolname,grantor.rolname",
        (
            [topology.migrator_role, topology.verifier_role],
            [topology.migrator_role, topology.verifier_role],
        ),
    )
    rows = tuple(dict(row) for row in cur.fetchall())
    infrastructure = set(topology.infrastructure_superuser_roles)
    unexpected = [
        row
        for row in rows
        if row["parent_role"] not in {topology.migrator_role, topology.verifier_role}
        or row["member_role"] not in infrastructure
        or row["grantor_role"] not in infrastructure
        or bool(row["admin_option"])
    ]
    if unexpected:
        raise RuntimeError(
            f"unexpected legacy schema capability memberships block authority upgrade: {unexpected!r}"
        )
    return rows


def _transition_existing_authority_topology(cur, *, topology: BootstrapTopology) -> None:
    # Close the known legacy admin graph first.  Forward rollback intentionally
    # never restores these memberships or elevated/login attributes.
    _close_retired_admin_memberships(cur, retired_admin_roles=topology.retired_admin_roles)
    database = sql.Identifier(cur.connection.info.dbname)
    for role_name in topology.retired_admin_roles:
        cur.execute(sql.SQL("REVOKE CONNECT,CREATE,TEMPORARY ON DATABASE {} FROM {}").format(
            database, sql.Identifier(role_name)
        ))
        cur.execute(sql.SQL(
            "ALTER ROLE {} NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
        ).format(sql.Identifier(role_name)))
    _validate_retired_admin_memberships_closed(cur, retired_admin_roles=topology.retired_admin_roles)

    for role_name, login in (
        (topology.database_owner_role, True),
        (topology.migrator_role, False),
        (topology.verifier_role, False),
    ):
        cur.execute(sql.SQL(
            "ALTER ROLE {} {} NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
        ).format(sql.Identifier(role_name), sql.SQL("LOGIN" if login else "NOLOGIN")))
    _reconcile_role_memberships(cur, role_name=topology.migrator_role)
    _reconcile_role_memberships(cur, role_name=topology.verifier_role)
    _reconcile_role_memberships(cur, role_name=topology.database_owner_role)
    _reconcile_exact_controller_membership(cur, controller_role=topology.controller_role)
    cur.execute(sql.SQL(
        "GRANT {} TO {} WITH ADMIN FALSE, INHERIT FALSE, SET TRUE"
    ).format(sql.Identifier(topology.migrator_role), sql.Identifier(topology.database_owner_role)))
    _reconcile_role_memberships(
        cur,
        role_name=topology.migrator_role,
        approved_grantee_roles=(topology.database_owner_role,),
    )
    _reconcile_role_memberships(
        cur,
        role_name=topology.database_owner_role,
        approved_parent_roles=(topology.migrator_role,),
    )
    _verify_existing_provider_membership(cur, topology=topology)


def ensure_fleet_worker_role(
    conn,
    password: str,
    *,
    role: str = DEFAULT_ROLE,
    worker_id: str | None = None,
    contract: str | None = None,
    regrant_manifest: RegrantManifest | None = None,
    evidence_paths: DurableEvidencePaths | None = None,
    approved_grantee_roles: tuple[str, ...] = (),
) -> RoleReconciliationReceipt:
    """Create or reconcile the remote-worker role on the current database.

    A locked, secret-free inventory and rollback SQL are handed to the required
    durable evidence writer before any persistent mutation. All database changes
    and final validation then commit in one transaction.
    """
    if contract is not None and contract not in _CONTRACT_FUNCTIONS:
        raise ValueError(f"unsupported fleet worker contract: {contract}")
    if regrant_manifest is None:
        raise RuntimeError("explicit regrant manifest is required before database-wide ACL changes")
    if evidence_paths is None:
        raise RuntimeError("durable evidence paths are required before database hardening")
    _validate_evidence_paths(evidence_paths)
    if conn.info.transaction_status.name != "IDLE":
        raise RuntimeError("role reconciliation requires an idle connection for its single transaction")
    password_verifier = _client_scram_verifier(conn, role_name=role, password=password)
    role_identifier = sql.Identifier(role)
    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL search_path=pg_catalog")
            cur.execute("SELECT pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext(%s))", (_ROLE_LOCK_KEY,))
            cur.execute("SELECT 1 FROM pg_catalog.pg_roles WHERE rolname=%s", (role,))
            role_exists = cur.fetchone() is not None
            if role_exists:
                cur.execute(
                    "SELECT rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
                    "FROM pg_catalog.pg_roles WHERE rolname=%s",
                    (role,),
                )
                elevated = cur.fetchone()
                if any(
                    _row_value(elevated, name, index)
                    for index, name in enumerate(
                        ("rolsuper", "rolcreatedb", "rolcreaterole", "rolreplication", "rolbypassrls")
                    )
                ):
                    raise RuntimeError(
                        "elevated worker role requires one-time provider-admin cleanup before reconciliation"
                    )

            cur.execute("SELECT current_database() AS current_database")
            database = sql.Identifier(_row_value(cur.fetchone(), "current_database"))
            inventory = _database_inventory(cur, role_name=role, manifest=regrant_manifest)
            _reject_pg_temp_shadow(cur)
            owned = _owned_objects(cur, role) if role_exists else []
            if owned:
                raise RuntimeError(
                    "fleet_worker owns database objects that bypass ACL hardening; "
                    "ownership must be repaired explicitly: " + ", ".join(owned)
                )

            target_snapshot = _target_role_snapshot(cur, role_name=role, role_exists=role_exists)
            existing_principal = target_snapshot["principal_mapping"]
            if existing_principal is not None:
                mapped_worker = existing_principal["worker_id"]
                mapped_contract = existing_principal["contract"]
                if worker_id is not None and worker_id != mapped_worker:
                    raise RuntimeError("fleet worker role is already mapped to a different node identity")
                if contract is not None and contract != mapped_contract:
                    raise RuntimeError("fleet worker role is already mapped to a different contract")
            effective_worker = worker_id or (existing_principal["worker_id"] if existing_principal else None)
            effective_contract = contract or (existing_principal["contract"] if existing_principal else None)
            if not effective_worker or not effective_contract:
                raise RuntimeError("fleet worker principal mapping is required before LOGIN can be enabled")
            functions = _functions_for_contract(effective_contract)
            allowed_types = _CONTRACT_TYPES[effective_contract]
            _validate_objects(cur, functions)
            regrant_snapshots = _structured_regrant_snapshot(cur, inventory["regrants"])
            cur.execute(
                "SELECT 1 FROM public.workers WHERE worker_id=%s",
                (effective_worker,),
            )
            if cur.fetchone() is None:
                raise RuntimeError(f"fleet worker enrollment is missing for worker_id {effective_worker!r}")

            rollback_sql = _public_acl_rollback_sql(
                cur,
                connect_allowlist=inventory["connect_allowlist"],
                database_owner_before=inventory["database_owner_before"],
                database_owner_after=inventory["database_owner_after"],
                retired_admin_roles=inventory["retired_admin_roles"],
            )
            rollback_sql += _mapped_role_rollback_sql(
                cur,
                role_name=role,
                snapshot=target_snapshot,
                regrant_snapshots=regrant_snapshots,
            )
            prepared_at = datetime.now(timezone.utc).isoformat()
            evidence_inventory = {
                **inventory,
                "target_role": role,
                "target_role_existed": role_exists,
                "target_role_snapshot": target_snapshot,
                "structured_regrant_acl_snapshots": regrant_snapshots,
                "credential_forward_reconcile_required": role_exists,
                "worker_id": effective_worker,
                "contract": effective_contract,
                "prepared_at": prepared_at,
            }
            _write_preparation_evidence(
                evidence_paths,
                inventory=evidence_inventory,
                rollback_sql=rollback_sql,
            )

            if not role_exists:
                cur.execute(sql.SQL("CREATE ROLE {}").format(role_identifier))
            _install_identity_function(cur)
            _harden_database_connect(
                cur,
                database=database,
                database_owner_role=regrant_manifest.database_owner_role,
                retired_admin_roles=regrant_manifest.retired_admin_roles,
            )
            _revoke_legacy_privileges(cur, role=role_identifier, database=database)
            _reconcile_role_memberships(
                cur,
                role_name=role,
                approved_grantee_roles=approved_grantee_roles,
            )
            cur.execute(sql.SQL("ALTER ROLE {} NOLOGIN NOINHERIT").format(role_identifier))
            cur.execute(
                "INSERT INTO public.fleet_worker_principals(role_name,worker_id,contract) "
                "VALUES(%s,%s,%s) ON CONFLICT(role_name) DO UPDATE SET "
                "worker_id=EXCLUDED.worker_id,contract=EXCLUDED.contract",
                (role, effective_worker, effective_contract),
            )
            cur.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(database, role_identifier))
            for allowed_role in inventory["connect_allowlist"]:
                cur.execute(
                    sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(database, sql.Identifier(allowed_role))
                )
            cur.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier("public"), role_identifier))
            for name, argument_types in functions:
                function = _function_identifier(name, argument_types)
                cur.execute(sql.SQL("REVOKE ALL PRIVILEGES ON FUNCTION {} FROM PUBLIC").format(function))
                cur.execute(sql.SQL("GRANT EXECUTE ON FUNCTION {} TO {}").format(function, role_identifier))
            for type_name in allowed_types:
                if type_name.startswith("_"):
                    continue
                cur.execute(
                    sql.SQL("GRANT USAGE ON TYPE {}.{} TO {}").format(
                        sql.Identifier("public"), sql.Identifier(type_name), role_identifier
                    )
                )
            _apply_regrants(cur, inventory["regrants"])
            _validate_effective_boundary(cur, role_name=role, functions=functions, allowed_types=allowed_types)
            try:
                cur.execute(
                    sql.SQL("ALTER ROLE {} LOGIN NOINHERIT PASSWORD {}").format(
                        role_identifier, sql.Literal(password_verifier)
                    )
                )
            except Exception:
                raise RuntimeError("fleet worker password update failed") from None
            effective_connect_grantees = _validate_post_password_boundary(
                cur,
                role_name=role,
                functions=functions,
                allowed_types=allowed_types,
                connect_allowlist=inventory["connect_allowlist"],
                database_owner_role=regrant_manifest.database_owner_role,
                retired_admin_roles=regrant_manifest.retired_admin_roles,
                break_glass_roles=regrant_manifest.infrastructure_superuser_roles,
                approved_grantee_roles=approved_grantee_roles,
            )
        conn.commit()
        return RoleReconciliationReceipt(
            role_name=role,
            worker_id=effective_worker,
            contract=effective_contract,
            connect_allowlist=inventory["connect_allowlist"],
            effective_connect_grantees=effective_connect_grantees,
            inventory=evidence_inventory,
            rollback_sql=rollback_sql,
            credential_forward_reconcile_required=role_exists,
            reconciled_at=prepared_at,
        )
    except BaseException:
        conn.rollback()
        raise
