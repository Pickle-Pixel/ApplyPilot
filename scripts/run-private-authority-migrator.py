#!/usr/bin/env python3
"""Private, fail-closed wrapper that applies the fleet migration manifest.

This is the *only* supported way to run the append-only fleet authority
migrations against a private (Tailscale / ``*.railway.internal``) target from the
``authority-migrator`` deployment. It is deliberately hostile to ambiguity:

- There is **no hardcoded target identity or default**. The operator must name
  the environment variable that carries the target DSN and must state the exact
  database name and PostgreSQL ``system_identifier`` the migration is allowed to
  touch. A mismatch fails closed *before* any migration runs, so a wrapper
  pointed at the wrong (for example, staging) cluster refuses to proceed.
- The target DSN is read from the named environment variable, never from argv,
  so the credential is never captured in process listings, shell history, or CI
  logs. Every diagnostic redacts the DSN to ``user@host/dbname``.
- Hostile target strings (control characters, newlines, shell metacharacters)
  are rejected: the DSN is parsed as a libpq conninfo and never handed to a
  shell.
- The wrapper verifies its own package manifest first -- the three deployment
  wrapper hashes plus the fleet migration manifest -- and then verifies the
  predecessor bytes and every migration checksum through ``migrator`` before a
  single statement executes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = (_REPOSITORY_ROOT / "src").resolve()
sys.path.insert(0, str(_SOURCE_ROOT))

import psycopg  # noqa: E402
from psycopg.conninfo import conninfo_to_dict  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402

from applypilot.fleet import migrator as _migrator  # noqa: E402

if not Path(_migrator.__file__).resolve().is_relative_to(_SOURCE_ROOT):
    raise RuntimeError("authority migrator refused a non-candidate ApplyPilot import")

_DEFAULT_PACKAGE_MANIFEST = _REPOSITORY_ROOT / "deploy" / "authority-migrator" / "package-manifest.json"
_MANIFEST_PATH = _REPOSITORY_ROOT / "src" / "applypilot" / "fleet" / "migrations" / "manifest-v1.json"
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_SYSTEM_IDENTIFIER = re.compile(r"^[0-9]{1,20}$")
_DATABASE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class TargetError(RuntimeError):
    """The declared target is missing, ambiguous, or hostile."""


class PackageError(RuntimeError):
    """The deployment package manifest is missing or does not match on disk."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def redact_dsn(dsn: str) -> str:
    """A log-safe rendering of a DSN: ``user@host/dbname``, never the password."""
    try:
        parts = conninfo_to_dict(dsn)
    except psycopg.Error:
        return "<unparseable-target>"
    user = parts.get("user") or "?"
    host = parts.get("host") or parts.get("hostaddr") or "?"
    dbname = parts.get("dbname") or "?"
    return f"{user}@{host}/{dbname}"


def validate_target(dsn: str) -> dict[str, str]:
    """Parse and defensively validate a target DSN, returning its conninfo dict.

    Raises ``TargetError`` (never leaking the raw DSN) for empty, control-char,
    newline, or otherwise hostile strings, and for a DSN with no host to bind to.
    The DSN is only ever parsed as a libpq conninfo -- it is never handed to a
    shell -- so quoting/metacharacter injection has nothing to inject into.
    """
    if not dsn:
        raise TargetError("target DSN is empty")
    if any(ord(char) < 32 or ord(char) == 127 for char in dsn):
        raise TargetError("target DSN contains control characters")
    try:
        parts = conninfo_to_dict(dsn)
    except psycopg.Error:
        raise TargetError("target DSN is not a valid libpq conninfo") from None
    if not (parts.get("host") or parts.get("hostaddr")):
        raise TargetError("target DSN must name an explicit private host")
    return parts


def verify_package_manifest(repository_root: Path, package_manifest: Path) -> dict:
    """Bind the three deployment wrappers and the fleet migration manifest by hash."""
    if not package_manifest.is_file():
        raise PackageError(f"package manifest is missing: {package_manifest.name}")
    payload = json.loads(package_manifest.read_bytes().decode("utf-8"))
    if payload.get("manifest_id") != "applypilot-authority-migrator-package-v1":
        raise PackageError("unexpected package manifest identity")
    wrappers = payload.get("wrappers")
    if not isinstance(wrappers, dict) or not wrappers:
        raise PackageError("package manifest binds no wrapper hashes")
    for relative, expected in sorted(wrappers.items()):
        if not isinstance(expected, str) or not _HEX64.fullmatch(expected):
            raise PackageError(f"package manifest has an invalid wrapper hash for {relative}")
        target = (repository_root / relative).resolve()
        if not target.is_relative_to(repository_root) or not target.is_file():
            raise PackageError(f"bound wrapper is missing: {relative}")
        actual = _sha256_file(target)
        if actual != expected:
            raise PackageError(f"wrapper hash drift for {relative}")
    migration_manifest = payload.get("migration_manifest")
    if not isinstance(migration_manifest, dict):
        raise PackageError("package manifest does not bind the fleet migration manifest")
    manifest_path = migration_manifest.get("path")
    manifest_sha = migration_manifest.get("sha256")
    if manifest_path != "src/applypilot/fleet/migrations/manifest-v1.json":
        raise PackageError("package manifest binds an unexpected fleet migration manifest path")
    if not isinstance(manifest_sha, str) or not _HEX64.fullmatch(manifest_sha):
        raise PackageError("package manifest has an invalid migration manifest hash")
    if _sha256_file(repository_root / manifest_path) != manifest_sha:
        raise PackageError("fleet migration manifest hash drift")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run-private-authority-migrator",
        description="Apply the append-only fleet authority migrations to an explicit private target.",
    )
    # No defaults: every target-identifying parameter must be stated explicitly.
    parser.add_argument(
        "--target-dsn-env",
        required=True,
        help="Name of the environment variable that holds the target DSN (never passed on argv).",
    )
    parser.add_argument("--expected-database-name", required=True)
    parser.add_argument("--expected-system-identifier", required=True)
    parser.add_argument(
        "--repository-root",
        default=str(_REPOSITORY_ROOT),
        help="Repository root that carries the migrations and predecessor artifacts.",
    )
    parser.add_argument(
        "--package-manifest",
        default=str(_DEFAULT_PACKAGE_MANIFEST),
        help="Path to the authority-migrator package manifest.",
    )
    return parser


def _resolve_target_dsn(env_name: str) -> str:
    if not _ENV_NAME.fullmatch(env_name):
        raise TargetError("target DSN env var name is not a safe identifier")
    dsn = os.environ.get(env_name)
    if dsn is None:
        raise TargetError(f"target DSN environment variable {env_name} is not set")
    validate_target(dsn)
    return dsn


def run(args: argparse.Namespace) -> int:
    if not _DATABASE_NAME.fullmatch(args.expected_database_name):
        raise TargetError("expected database name is not a safe identifier")
    if not _SYSTEM_IDENTIFIER.fullmatch(args.expected_system_identifier):
        raise TargetError("expected system identifier must be the numeric pg_control value")

    repository_root = Path(args.repository_root).resolve()
    verify_package_manifest(repository_root, Path(args.package_manifest).resolve())

    dsn = _resolve_target_dsn(args.target_dsn_env)
    manifest = _migrator.load_manifest(_MANIFEST_PATH)
    _migrator.verify_predecessor(manifest, repository_root)
    _migrator.verify_migration_files(manifest, repository_root)

    try:
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            identity = conn.execute(
                "SELECT current_database() AS database_name,"
                "control.system_identifier::text AS system_identifier "
                "FROM pg_control_system() AS control"
            ).fetchone()
            if identity["database_name"] != args.expected_database_name:
                raise TargetError("connected database name does not match the declared target")
            if identity["system_identifier"] != args.expected_system_identifier:
                raise TargetError("connected system identifier does not match the declared target")
            conn.execute("SET ROLE %s" % _quote_role(manifest.migration_role))
            result = _migrator.apply_manifest(conn, manifest, repository_root)
    except psycopg.Error:
        # Never surface libpq's connection string (host/user/password) in a
        # traceback. Redact to user@host/dbname and drop the driver chain.
        raise TargetError(f"target connection or migration failed for {redact_dsn(dsn)}") from None

    print(
        "authority migration complete: "
        f"target={redact_dsn(dsn)} applied={len(result.applied)} "
        f"already_applied={len(result.already_applied)}"
    )
    return 0


def _quote_role(role: str) -> str:
    # migration_role is already constrained by the manifest parser to
    # ``applypilot_fleet_migrator``; quote defensively regardless.
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", role):
        raise TargetError("refusing to assume a non-canonical migration role")
    return f'"{role}"'


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return run(args)
    except (TargetError, PackageError, _migrator.ManifestError, _migrator.MigrationError) as exc:
        # Diagnostics never include the raw DSN.
        print(f"authority migration refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
