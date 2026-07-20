#!/usr/bin/env python3
"""One-time provider-admin bootstrap for canonical fleet PostgreSQL roles."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import tempfile

import psycopg
from psycopg.rows import dict_row

from applypilot.fleet.pg_roles import (
    BRAIN_CANDIDATE_READER_ROLE,
    BRAIN_CANDIDATE_WRITER_ROLE,
    BootstrapTopology,
    DurableEvidencePaths,
    authenticate_evidence_receipt,
    bootstrap_database_roles,
)


def _safe_path(value: str) -> Path:
    if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise argparse.ArgumentTypeError("artifact path contains empty or control-character data")
    return Path(value).resolve()


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_durable(path: Path, payload: bytes) -> None:
    parent_before = os.lstat(path.parent)
    if not stat.S_ISDIR(parent_before.st_mode) or stat.S_ISLNK(parent_before.st_mode):
        raise RuntimeError("receipt parent must be a real directory")
    if os.name != "nt" and stat.S_IMODE(parent_before.st_mode) & 0o022:
        raise RuntimeError("receipt parent directory must not be group/world writable")
    target_before = os.lstat(path)
    if not stat.S_ISREG(target_before.st_mode) or stat.S_ISLNK(target_before.st_mode):
        raise RuntimeError("prepared receipt must be a regular non-symlink file")
    parent_descriptor: int | None = None
    if os.open in os.supports_dir_fd and os.replace in os.supports_dir_fd:
        parent_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_parent = os.fstat(parent_descriptor)
        if (opened_parent.st_dev, opened_parent.st_ino) != (parent_before.st_dev, parent_before.st_ino):
            os.close(parent_descriptor)
            raise RuntimeError("receipt parent changed while being opened")
        temporary = Path(f".applypilot-bootstrap-{secrets.token_hex(16)}")
        descriptor = os.open(
            temporary.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
    else:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".applypilot-bootstrap-", dir=path.parent)
        temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        parent_after = os.lstat(path.parent)
        target_after = os.lstat(path)
        if (parent_after.st_dev, parent_after.st_ino) != (parent_before.st_dev, parent_before.st_ino):
            raise RuntimeError("receipt parent changed before durable replacement")
        if (target_after.st_dev, target_after.st_ino) != (target_before.st_dev, target_before.st_ino):
            raise RuntimeError("prepared receipt changed before durable replacement")
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
        if parent_descriptor is not None:
            try:
                os.unlink(temporary.name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
            os.close(parent_descriptor)
        else:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-owner-role", required=True)
    parser.add_argument("--controller-role", required=True)
    parser.add_argument("--verifier-role", required=True, choices=("brain_schema_verifier",))
    parser.add_argument("--migrator-role", required=True, choices=("brain_schema_migrator",))
    parser.add_argument("--retired-admin-role", action="append", required=True)
    parser.add_argument("--expected-service-role", action="append", default=[])
    parser.add_argument("--infrastructure-superuser-role", action="append", default=[])
    parser.add_argument("--receipt-path", required=True, type=_safe_path)
    parser.add_argument("--rollback-sql", required=True, type=_safe_path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.receipt_path == args.rollback_sql:
        raise SystemExit("bootstrap receipt and rollback SQL paths must be distinct")
    collisions = [path for path in (args.receipt_path, args.rollback_sql) if path.exists()]
    if collisions:
        raise SystemExit("bootstrap refuses to overwrite evidence: " + ", ".join(map(str, collisions)))

    admin_dsn = os.environ.pop("APPLYPILOT_ADMIN_PG_DSN", "")
    controller_password = os.environ.pop("APPLYPILOT_CONTROLLER_PG_PASSWORD", "")
    backup_password = os.environ.pop("APPLYPILOT_BACKUP_PG_PASSWORD", "")
    database_owner_password = os.environ.pop(
        "APPLYPILOT_DATABASE_OWNER_PG_PASSWORD", ""
    )
    fleet_migration_password = os.environ.pop(
        "APPLYPILOT_FLEET_MIGRATOR_PG_PASSWORD", ""
    )
    authentication_key_hex = os.environ.pop("APPLYPILOT_ROLLBACK_HMAC_KEY_HEX", "")
    authentication_key_id = os.environ.pop("APPLYPILOT_ROLLBACK_HMAC_KEY_ID", "")
    try:
        authentication_key = bytes.fromhex(authentication_key_hex)
    except ValueError:
        authentication_key = b""
    if (
        not admin_dsn
        or not controller_password
        or not backup_password
        or not database_owner_password.strip()
        or not fleet_migration_password.strip()
        or len(authentication_key) < 32
        or not authentication_key_id
    ):
        raise SystemExit(
            "bootstrap requires APPLYPILOT_ADMIN_PG_DSN, APPLYPILOT_CONTROLLER_PG_PASSWORD, "
            "APPLYPILOT_BACKUP_PG_PASSWORD, APPLYPILOT_DATABASE_OWNER_PG_PASSWORD, "
            "APPLYPILOT_FLEET_MIGRATOR_PG_PASSWORD, "
            "APPLYPILOT_ROLLBACK_HMAC_KEY_HEX (at least 32 bytes), and APPLYPILOT_ROLLBACK_HMAC_KEY_ID"
        )
    topology = BootstrapTopology(
        database_owner_role=args.database_owner_role,
        controller_role=args.controller_role,
        verifier_role=args.verifier_role,
        migrator_role=args.migrator_role,
        retired_admin_roles=tuple(args.retired_admin_role),
        expected_service_roles=tuple(args.expected_service_role),
        infrastructure_superuser_roles=tuple(args.infrastructure_superuser_role),
    )
    for parent in {args.receipt_path.parent, args.rollback_sql.parent}:
        parent.mkdir(parents=True, exist_ok=True)
    evidence_paths = DurableEvidencePaths(
        preparation_receipt_path=args.receipt_path,
        rollback_sql_path=args.rollback_sql,
        authentication_key=authentication_key,
        authentication_key_id=authentication_key_id,
    )

    with psycopg.connect(admin_dsn, row_factory=dict_row) as conn:
        receipt = bootstrap_database_roles(
            conn,
            controller_password,
            backup_password,
            database_owner_password=database_owner_password,
            fleet_migration_password=fleet_migration_password,
            topology=topology,
            evidence_paths=evidence_paths,
            install_brain_authority=True,
        )
    result = asdict(receipt)
    rollback_sql = result.pop("rollback_sql")
    result.update(
        atomic_bootstrap=True,
        automatic_rollback_supported=True,
        commit_outcome_on_interruption="known_committed",
        escalation_required=False,
        in_doubt=False,
        legacy_rollback_sql_recovers_v1_v4=False,
        legacy_rollback_sql_recovers_v1_v5=False,
        rollback_mode=receipt.inventory["rollback_mode"],
        status="atomic_bootstrap_committed",
        rollback_sql_path=str(args.rollback_sql),
        rollback_sql_sha256=hashlib.sha256(rollback_sql.encode("utf-8")).hexdigest(),
        candidate_roles={
            "reader_role": BRAIN_CANDIDATE_READER_ROLE,
            "writer_role": BRAIN_CANDIDATE_WRITER_ROLE,
            "reconciled_at": receipt.bootstrapped_at,
        },
    )
    result = authenticate_evidence_receipt(
        result,
        authentication_key=authentication_key,
        authentication_key_id=authentication_key_id,
    )
    _replace_durable(
        args.receipt_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(
        "atomic bootstrap committed; "
        f"receipt={args.receipt_path}; authenticated forward rollback={args.rollback_sql}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
