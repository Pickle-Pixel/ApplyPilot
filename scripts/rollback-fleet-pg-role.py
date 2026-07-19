#!/usr/bin/env python3
"""Verify and atomically execute a prepared fleet PostgreSQL rollback."""

from __future__ import annotations

import argparse
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

from applypilot.fleet.pg_roles import verify_evidence_receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--rollback-sql", required=True, type=Path)
    parser.add_argument("--restore-hba", action="store_true")
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


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    receipt, rollback_sql = _verified_inputs(
        args.receipt.resolve(),
        args.rollback_sql.resolve(),
        authentication_key=authentication_key,
        expected_key_id=expected_key_id,
    )
    admin_dsn = os.environ.pop("APPLYPILOT_ADMIN_PG_DSN", "")
    if not admin_dsn:
        raise SystemExit("APPLYPILOT_ADMIN_PG_DSN is required only for break-glass rollback")

    # The authenticated break-glass connection is deliberately established
    # before HBA replacement/reload so rollback authority cannot be locked out.
    with psycopg.connect(admin_dsn, row_factory=dict_row) as conn:
        identity = conn.execute(
            "SELECT session_user,current_user,rolsuper FROM pg_roles WHERE rolname=session_user"
        ).fetchone()
        conn.commit()
        if (
            identity["session_user"] != identity["current_user"]
            or not identity["rolsuper"]
            or identity["session_user"] not in _break_glass_roles(receipt)
        ):
            raise SystemExit("rollback session must be an explicitly receipted break-glass superuser")
        hba_preflight = _validated_hba_restore(conn, receipt) if args.restore_hba else None
        if hba_preflight is not None:
            # SHOW hba_file starts an implicit psycopg transaction. Complete
            # that read-only preflight before opening the atomic DB rollback.
            conn.commit()
            if conn.info.transaction_status.name != "IDLE":
                raise RuntimeError("HBA preflight transaction did not return to idle state")
        # psycopg's transaction context is the executable equivalent of
        # psql --single-transaction --set=ON_ERROR_STOP=on.
        with conn.transaction():
            conn.execute(rollback_sql)
        print("database_rollback_committed")
        if args.restore_hba:
            print("hba_restore_started")
            try:
                _restore_hba_and_reload(conn, receipt, validated=hba_preflight)
            except BaseException as error:
                raise RuntimeError(
                    "database rollback committed, but HBA restoration failed; "
                    "partial state: database containment is durable and HBA admission state is uncertain"
                ) from error
    print("authenticated database rollback committed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
