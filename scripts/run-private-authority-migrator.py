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
from psycopg import sql  # noqa: E402
from psycopg.conninfo import conninfo_to_dict  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402

from applypilot.apply import pgqueue as _pgqueue  # noqa: E402
from applypilot.fleet import migrator as _migrator  # noqa: E402
from applypilot.fleet.pg_roles import (  # noqa: E402
    FLEET_MIGRATION_LOGIN_ROLE,
    FLEET_MIGRATION_ROLE,
    validate_fleet_migration_principal,
)

if not Path(_migrator.__file__).resolve().is_relative_to(_SOURCE_ROOT):
    raise RuntimeError("authority migrator refused a non-candidate ApplyPilot import")

_DEFAULT_PACKAGE_MANIFEST = _REPOSITORY_ROOT / "deploy" / "authority-migrator" / "package-manifest.json"
_MANIFEST_PATH = _REPOSITORY_ROOT / "src" / "applypilot" / "fleet" / "migrations" / "manifest-v1.json"
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_SYSTEM_IDENTIFIER = re.compile(r"^[0-9]{1,20}$")
_DATABASE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_HOST = re.compile(
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"(?:railway\.internal|ts\.net)"
)
_DISPOSABLE_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
# The exact deployment wrapper set this migrator package must bind. Asserting the required
# set -- rather than trusting whatever keys the manifest happens to list -- prevents a tampered
# manifest from silently dropping a wrapper (e.g. the Dockerfile) out of hash coverage.
_REQUIRED_WRAPPERS = frozenset(
    {
        "Dockerfile.authority-migrator",
        "railway.toml",
        "scripts/run-private-authority-migrator.py",
    }
)


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


def _normalized_expected_host(
    expected_private_host: str,
    *,
    allow_disposable_loopback: bool,
) -> str:
    if not isinstance(expected_private_host, str) or not expected_private_host:
        raise TargetError("expected private host is empty")
    if any(ord(char) < 33 or ord(char) == 127 for char in expected_private_host):
        raise TargetError("expected private host has invalid syntax")
    normalized = expected_private_host.rstrip(".").casefold()
    if allow_disposable_loopback and normalized in _DISPOSABLE_LOOPBACK_HOSTS:
        return normalized
    if _PRIVATE_HOST.fullmatch(normalized) is None:
        raise TargetError("expected host must be a full private Railway or Tailscale hostname")
    labels = normalized.split(".")
    if normalized.endswith(".ts.net") and len(labels) < 4:
        raise TargetError("expected Tailscale host must be a full MagicDNS hostname")
    return normalized


def validate_target(
    dsn: str,
    *,
    expected_private_host: str,
    allow_disposable_loopback: bool = False,
) -> str:
    """Return one hardened conninfo bound to the fixed login and exact host.

    The only non-TLS exception is an explicitly opted-in loopback address used
    by disposable PostgreSQL tests. Production callers cannot use an arbitrary
    IP, public DNS name, ``hostaddr``, ambient libpq option, or indirect service
    configuration.
    """
    if not dsn:
        raise TargetError("target DSN is empty")
    if any(ord(char) < 32 or ord(char) == 127 for char in dsn):
        raise TargetError("target DSN contains control characters")
    expected_host = _normalized_expected_host(
        expected_private_host,
        allow_disposable_loopback=allow_disposable_loopback,
    )
    try:
        if allow_disposable_loopback and expected_host in _DISPOSABLE_LOOPBACK_HOSTS:
            # Reuse the shared hardened parser for explicit password/host,
            # hostaddr/indirection, and ambient-libpq rejection. The production
            # validator below still owns every non-loopback invocation.
            parts, targets = _pgqueue._validate_explicit_conninfo(dsn)
            if parts.get("options"):
                raise RuntimeError("disposable target must not carry libpq options")
            if parts.get("user") != FLEET_MIGRATION_LOGIN_ROLE:
                raise RuntimeError("disposable target principal mismatch")
            if parts.get("sslmode", "").casefold() != "disable":
                raise RuntimeError("disposable loopback target must explicitly disable TLS")
            conninfo = _pgqueue._render_conninfo(dsn, parts)
        else:
            validated = _pgqueue.validate_runtime_conninfo(
                dsn,
                expected_principal=FLEET_MIGRATION_LOGIN_ROLE,
                require_expected_principal=True,
            )
            parts = conninfo_to_dict(validated.conninfo)
            targets = parts["host"].split(",")
            conninfo = validated.conninfo
    except (psycopg.Error, RuntimeError, ValueError):
        raise TargetError("target DSN violates the private migration transport contract") from None

    normalized_targets = tuple(target.strip().rstrip(".").casefold() for target in targets)
    if normalized_targets != (expected_host,):
        raise TargetError("target DSN host does not match the explicitly approved private host")
    return conninfo


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
    if frozenset(wrappers) != _REQUIRED_WRAPPERS:
        raise PackageError(
            "package manifest wrapper set does not match the exact required deployment wrappers"
        )
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
    parser.add_argument(
        "--expected-private-host",
        required=True,
        help="Exact private Railway or Tailscale hostname expected in the target DSN.",
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


def _resolve_target_dsn(
    env_name: str,
    *,
    expected_private_host: str,
    allow_disposable_loopback: bool = False,
) -> str:
    if not _ENV_NAME.fullmatch(env_name):
        raise TargetError("target DSN env var name is not a safe identifier")
    dsn = os.environ.get(env_name)
    if dsn is None:
        raise TargetError(f"target DSN environment variable {env_name} is not set")
    return validate_target(
        dsn,
        expected_private_host=expected_private_host,
        allow_disposable_loopback=allow_disposable_loopback,
    )


def run(args: argparse.Namespace) -> int:
    if not _DATABASE_NAME.fullmatch(args.expected_database_name):
        raise TargetError("expected database name is not a safe identifier")
    if not _SYSTEM_IDENTIFIER.fullmatch(args.expected_system_identifier):
        raise TargetError("expected system identifier must be the numeric pg_control value")

    repository_root = Path(args.repository_root).resolve()
    verify_package_manifest(repository_root, Path(args.package_manifest).resolve())

    dsn = _resolve_target_dsn(
        args.target_dsn_env,
        expected_private_host=args.expected_private_host,
        allow_disposable_loopback=getattr(args, "allow_disposable_loopback", False),
    )
    manifest = _migrator.load_manifest(_MANIFEST_PATH)
    if manifest.migration_role != FLEET_MIGRATION_ROLE:
        raise _migrator.ManifestError("manifest does not bind the fixed fleet migration capability")
    _migrator.verify_predecessor(manifest, repository_root)
    _migrator.verify_migration_files(manifest, repository_root)

    try:
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            # This must precede target reads, SET ROLE, advisory locks, and DDL.
            # A provider admin can assume the migration capability but cannot
            # impersonate this authenticated session-user boundary.
            validate_fleet_migration_principal(conn)
            identity = conn.execute(
                "SELECT pg_catalog.current_database() AS database_name,"
                "control.system_identifier::text AS system_identifier "
                "FROM pg_catalog.pg_control_system() AS control"
            ).fetchone()
            if identity["database_name"] != args.expected_database_name:
                raise TargetError("connected database name does not match the declared target")
            if identity["system_identifier"] != args.expected_system_identifier:
                raise TargetError("connected system identifier does not match the declared target")
            conn.commit()
            conn.execute(
                sql.SQL("SET ROLE {}").format(sql.Identifier(FLEET_MIGRATION_ROLE))
            )
            elevated = conn.execute(
                "SELECT session_user AS session_name,current_user AS current_name"
            ).fetchone()
            if (
                elevated["session_name"] != FLEET_MIGRATION_LOGIN_ROLE
                or elevated["current_name"] != FLEET_MIGRATION_ROLE
            ):
                raise TargetError("fleet migration SET ROLE boundary did not activate exactly")
            # pg_catalog remains implicitly first for object lookup, while
            # unqualified CREATE statements target the intended application schema.
            conn.execute("SET search_path TO public")
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


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return run(args)
    except (
        TargetError,
        PackageError,
        RuntimeError,
        _migrator.ManifestError,
        _migrator.MigrationError,
    ) as exc:
        # Diagnostics never include the raw DSN.
        print(f"authority migration refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
