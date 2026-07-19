"""Strict, append-only fleet migration manifest execution."""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


_MANIFEST_FIELDS = frozenset({"schema_version", "manifest_id", "migration_role", "predecessor", "migrations"})
_PREDECESSOR_FIELDS = frozenset({"runtime_commit", "runtime_tree", "files"})
_PREDECESSOR_FILE_FIELDS = frozenset({"path", "git_blob"})
_MIGRATION_FIELDS = frozenset(
    {
        "id",
        "path",
        "sha256",
        "predecessor_id",
        "predecessor_sha256",
        "transaction_mode",
        "minimum_schema_contract",
        "maximum_schema_contract",
        "forward_recovery_command",
    }
)
_MIGRATION_ID = re.compile(r"^20[0-9]{6}_[0-9]{3}_[a-z][a-z0-9_]{0,63}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LOCK_KEY = "applypilot:fleet:migrations:v1"
_DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0
_DEFAULT_STATEMENT_TIMEOUT_MS = 300_000
_LEDGER_TABLE = "applypilot_fleet_schema_migrations"
# The published, canonical predecessor tuples. A release archive strips `.git`,
# so it cannot appeal to the object store to prove its predecessor bytes. The
# only artifact-safe boundary is a pinned (commit, tree, files) tuple *plus* the
# exact predecessor bytes carried in the archive: the tuple alone is never
# enough (see verify_predecessor / _verify_predecessor_archive).
_PINNED_PREDECESSORS = {
    "2b3a7c83118df840dda60c9b728f29e3dc0c1b9d": {
        "runtime_tree": "b160cec1ef3c2bd35989bf7127ca0554269b45a0",
        "files": {
            "src/applypilot/apply/fleet_schema.sql": "6eb4a84dcc05568233ee32551b28c75f13b0a17f",
            "src/applypilot/fleet/schema.py": "5637001b6457f9fc8f0c22f4220fe2e1249ff9c0",
            "src/applypilot/fleet/schema_v3.sql": "0741a6e675d2ea42a3bb0d785fd4c0c444e96b3d",
        },
    }
}


class ManifestError(ValueError):
    """The manifest or one of its byte-pinned inputs is invalid."""


class MigrationError(RuntimeError):
    """The database migration state is unsafe or inconsistent."""


@dataclass(frozen=True)
class PredecessorFile:
    path: str
    git_blob: str


@dataclass(frozen=True)
class Predecessor:
    runtime_commit: str
    runtime_tree: str
    files: tuple[PredecessorFile, ...]


@dataclass(frozen=True)
class Migration:
    id: str
    path: str
    sha256: str
    predecessor_id: str | None
    predecessor_sha256: str | None
    transaction_mode: str
    minimum_schema_contract: str
    maximum_schema_contract: str
    forward_recovery_command: str


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    manifest_id: str
    migration_role: str
    predecessor: Predecessor
    migrations: tuple[Migration, ...]


@dataclass(frozen=True)
class ApplyResult:
    applied: tuple[str, ...]
    already_applied: tuple[str, ...]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _closed(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be an object")
    actual = frozenset(value)
    if actual != expected:
        raise ManifestError(f"{label} fields differ: expected={sorted(expected)} actual={sorted(actual)}")
    return value


def _safe_path(value: Any, *, sql: bool = False) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ManifestError("manifest path must be a nonempty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ManifestError(f"unsafe manifest path: {value!r}")
    normalized = path.as_posix()
    if normalized != value:
        raise ManifestError(f"noncanonical manifest path: {value!r}")
    if sql and (not normalized.startswith("src/applypilot/fleet/migrations/") or not normalized.endswith(".sql")):
        raise ManifestError(f"migration path is outside the closed migration root: {value!r}")
    return normalized


def _bounded_text(value: Any, label: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise ManifestError(f"invalid {label}")
    return value


def _parse_migration(value: Any) -> Migration:
    item = _closed(value, _MIGRATION_FIELDS, "migration")
    migration_id = _bounded_text(item["id"], "migration id", maximum=96)
    if not _MIGRATION_ID.fullmatch(migration_id):
        raise ManifestError(f"invalid migration id: {migration_id!r}")
    sha256 = item["sha256"]
    if not isinstance(sha256, str) or not _HEX64.fullmatch(sha256):
        raise ManifestError(f"invalid migration checksum for {migration_id}")
    predecessor_id = item["predecessor_id"]
    predecessor_sha256 = item["predecessor_sha256"]
    if predecessor_id is not None and (not isinstance(predecessor_id, str) or not _MIGRATION_ID.fullmatch(predecessor_id)):
        raise ManifestError(f"invalid predecessor id for {migration_id}")
    if predecessor_sha256 is not None and (
        not isinstance(predecessor_sha256, str) or not _HEX64.fullmatch(predecessor_sha256)
    ):
        raise ManifestError(f"invalid predecessor checksum for {migration_id}")
    if item["transaction_mode"] != "transactional":
        raise ManifestError(f"unsupported transaction mode for {migration_id}")
    return Migration(
        id=migration_id,
        path=_safe_path(item["path"], sql=True),
        sha256=sha256,
        predecessor_id=predecessor_id,
        predecessor_sha256=predecessor_sha256,
        transaction_mode="transactional",
        minimum_schema_contract=_bounded_text(item["minimum_schema_contract"], "minimum schema contract"),
        maximum_schema_contract=_bounded_text(item["maximum_schema_contract"], "maximum schema contract"),
        forward_recovery_command=_bounded_text(item["forward_recovery_command"], "forward recovery command", maximum=512),
    )


def load_manifest(path: str | Path) -> Manifest:
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw or not raw.endswith(b"\n"):
        raise ManifestError("manifest must be UTF-8 without BOM, LF-only, and final-LF terminated")
    try:
        payload = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=_strict_object)
    except ManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"invalid manifest JSON: {exc}") from exc
    root = _closed(payload, _MANIFEST_FIELDS, "manifest")
    if root["schema_version"] != 1 or root["manifest_id"] != "applypilot-fleet-migrations-v1":
        raise ManifestError("unsupported manifest identity")
    role = _bounded_text(root["migration_role"], "migration role", maximum=63)
    if role != "applypilot_fleet_migrator":
        raise ManifestError("unexpected migration role")
    predecessor_payload = _closed(root["predecessor"], _PREDECESSOR_FIELDS, "predecessor")
    runtime_commit = predecessor_payload["runtime_commit"]
    if not isinstance(runtime_commit, str) or not _HEX40.fullmatch(runtime_commit):
        raise ManifestError("invalid predecessor runtime commit")
    runtime_tree = predecessor_payload["runtime_tree"]
    if not isinstance(runtime_tree, str) or not _HEX40.fullmatch(runtime_tree):
        raise ManifestError("invalid predecessor runtime tree")
    files_payload = predecessor_payload["files"]
    if not isinstance(files_payload, list) or not files_payload:
        raise ManifestError("predecessor files must be a nonempty array")
    predecessor_files: list[PredecessorFile] = []
    seen_paths: set[str] = set()
    for value in files_payload:
        item = _closed(value, _PREDECESSOR_FILE_FIELDS, "predecessor file")
        file_path = _safe_path(item["path"])
        git_blob = item["git_blob"]
        if file_path in seen_paths:
            raise ManifestError(f"duplicate predecessor path: {file_path}")
        if not isinstance(git_blob, str) or not _HEX40.fullmatch(git_blob):
            raise ManifestError(f"invalid predecessor blob for {file_path}")
        seen_paths.add(file_path)
        predecessor_files.append(PredecessorFile(path=file_path, git_blob=git_blob))
    # Canonical order so a reordered predecessor manifest fails closed rather
    # than presenting the same closed set of blobs under a shuffled tree.
    ordered_paths = [item.path for item in predecessor_files]
    if ordered_paths != sorted(ordered_paths):
        raise ManifestError("predecessor files must be in canonical sorted order")
    migrations_payload = root["migrations"]
    if not isinstance(migrations_payload, list):
        raise ManifestError("migrations must be an array")
    migrations = tuple(_parse_migration(value) for value in migrations_payload)
    if len({migration.id for migration in migrations}) != len(migrations):
        raise ManifestError("duplicate migration id")
    if len({migration.path for migration in migrations}) != len(migrations):
        raise ManifestError("duplicate migration path")
    if tuple(sorted(migration.id for migration in migrations)) != tuple(migration.id for migration in migrations):
        raise ManifestError("migration ids must be strictly increasing")
    for index, migration in enumerate(migrations):
        if index == 0:
            if migration.predecessor_id is not None or migration.predecessor_sha256 is not None:
                raise ManifestError("first migration predecessor must be null")
        else:
            prior = migrations[index - 1]
            if migration.predecessor_id != prior.id or migration.predecessor_sha256 != prior.sha256:
                raise ManifestError(f"migration predecessor mismatch for {migration.id}")
    return Manifest(
        schema_version=1,
        manifest_id="applypilot-fleet-migrations-v1",
        migration_role=role,
        predecessor=Predecessor(
            runtime_commit=runtime_commit,
            runtime_tree=runtime_tree,
            files=tuple(predecessor_files),
        ),
        migrations=migrations,
    )


def _blob_id(data: bytes) -> str:
    header = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(header + data).hexdigest()


def _git_predecessor_blob(root: Path, runtime_commit: str, path: str) -> str | None:
    try:
        merge_base = subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", runtime_commit, "HEAD"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if merge_base.returncode != 0:
            return None
        result = subprocess.run(
            ["git", "-C", str(root), "ls-tree", runtime_commit, "--", path],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    fields = result.stdout.strip().split()
    if len(fields) < 3 or fields[1] != "blob" or not _HEX40.fullmatch(fields[2]):
        return None
    return fields[2]


def _git_commit_tree(root: Path, runtime_commit: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "--quiet", f"{runtime_commit}^{{tree}}"],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    tree = result.stdout.strip()
    return tree if _HEX40.fullmatch(tree) else None


def _predecessor_bundle_dir(root: Path) -> Path:
    return root / "src" / "applypilot" / "fleet" / "migrations" / "predecessor"


def _verify_predecessor_tuple_with_git(manifest: Manifest, root: Path) -> None:
    # With a real object store, bind the declared tree to the declared commit and
    # prove each declared blob is exactly what that commit's tree carried. This
    # verifies provenance against history rather than against the (current) live
    # working tree, which legitimately carries newer bytes than the predecessor.
    actual_tree = _git_commit_tree(root, manifest.predecessor.runtime_commit)
    if actual_tree is not None and actual_tree != manifest.predecessor.runtime_tree:
        raise ManifestError(
            f"predecessor tree mismatch for {manifest.predecessor.runtime_commit}: "
            f"expected {manifest.predecessor.runtime_tree}, got {actual_tree}"
        )
    for item in manifest.predecessor.files:
        git_blob = _git_predecessor_blob(root, manifest.predecessor.runtime_commit, item.path)
        if git_blob != item.git_blob:
            raise ManifestError(
                f"predecessor blob mismatch for {item.path}: expected {item.git_blob}, got {git_blob}"
            )


def _verify_predecessor_tuple_pinned(manifest: Manifest) -> None:
    # A release archive omits .git, so it cannot appeal to the object store. The
    # declared (commit, tree, files) tuple must instead match a pinned, published
    # predecessor exactly. Tuple identity alone is never sufficient -- the exact
    # bytes are still checked against the carried bundle below.
    pinned = _PINNED_PREDECESSORS.get(manifest.predecessor.runtime_commit)
    if pinned is None:
        raise ManifestError(
            f"predecessor tuple is not a pinned release predecessor: {manifest.predecessor.runtime_commit}"
        )
    if manifest.predecessor.runtime_tree != pinned["runtime_tree"]:
        raise ManifestError(
            f"predecessor tree mismatch for {manifest.predecessor.runtime_commit}: "
            f"expected {pinned['runtime_tree']}, got {manifest.predecessor.runtime_tree}"
        )
    declared = {item.path: item.git_blob for item in manifest.predecessor.files}
    if declared != pinned["files"]:
        raise ManifestError("predecessor file manifest does not match the pinned release tuple")


def _verify_predecessor_bundle(manifest: Manifest, bundle: Path) -> None:
    # Every release (with or without .git) carries the exact predecessor bytes as
    # content-addressed provenance artifacts. This is the artifact-safe substitute
    # for the object store: prefix-only (short id would not name a file),
    # reconstructed/normalized (blob id differs), and missing artifacts all fail
    # closed here -- there is no tuple-only escape hatch.
    for item in manifest.predecessor.files:
        artifact = (bundle / item.git_blob).resolve()
        if not artifact.is_relative_to(bundle.resolve()) or not artifact.is_file():
            raise ManifestError(
                f"missing carried predecessor byte artifact for {item.path}: {item.git_blob}"
            )
        data = artifact.read_bytes()
        actual = _blob_id(data)
        # Exact bytes only -- no CRLF/normalization leniency for carried bytes.
        if actual != item.git_blob:
            raise ManifestError(
                f"predecessor blob mismatch for {item.path}: expected {item.git_blob}, got {actual}"
            )


def verify_predecessor(manifest: Manifest, repository_root: str | Path) -> None:
    root = Path(repository_root).resolve()
    if (root / ".git").exists():
        _verify_predecessor_tuple_with_git(manifest, root)
    else:
        _verify_predecessor_tuple_pinned(manifest)
    _verify_predecessor_bundle(manifest, _predecessor_bundle_dir(root))


def verify_migration_files(manifest: Manifest, repository_root: str | Path) -> None:
    root = Path(repository_root).resolve()
    for migration in manifest.migrations:
        path = (root / Path(*PurePosixPath(migration.path).parts)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ManifestError(f"migration file is missing: {migration.path}")
        data = path.read_bytes()
        if data.startswith(b"\xef\xbb\xbf") or b"\r" in data or not data.endswith(b"\n"):
            raise ManifestError(f"migration file must be UTF-8 without BOM and LF-only: {migration.path}")
        try:
            data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ManifestError(f"migration file is not UTF-8: {migration.path}") from exc
        actual = hashlib.sha256(data).hexdigest()
        if actual != migration.sha256:
            raise ManifestError(f"migration checksum mismatch for {migration.id}: expected {migration.sha256}, got {actual}")


def _bounded_timeout(value: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) and parsed >= 0 else default


def _acquire_lock(conn, timeout_seconds: float) -> None:
    deadline = time.monotonic() + _bounded_timeout(timeout_seconds, _DEFAULT_LOCK_TIMEOUT_SECONDS)
    while True:
        row = conn.execute("SELECT pg_try_advisory_lock(hashtext(%s)) AS acquired", (_LOCK_KEY,)).fetchone()
        acquired = row["acquired"] if hasattr(row, "keys") else row[0]
        if bool(acquired):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for the fleet migration lock")
        time.sleep(0.1)


def _bootstrap_ledger(conn) -> None:
    existing = conn.execute("SELECT to_regclass('public.applypilot_fleet_schema_migrations') AS oid").fetchone()
    existing_oid = existing["oid"] if hasattr(existing, "keys") else existing[0]
    if existing_oid is not None:
        _verify_ledger_contract(conn)
        return
    existing_function = conn.execute(
        "SELECT to_regprocedure('public.applypilot_reject_migration_ledger_mutation()') AS oid"
    ).fetchone()
    function_oid = existing_function["oid"] if hasattr(existing_function, "keys") else existing_function[0]
    if function_oid is not None:
        raise MigrationError("migration ledger contract has an orphan mutation function")
    conn.execute(
        "CREATE TABLE applypilot_fleet_schema_migrations ("
        "migration_id text PRIMARY KEY, manifest_schema_version integer NOT NULL CHECK (manifest_schema_version=1),"
        "migration_path text NOT NULL UNIQUE, migration_sha256 text NOT NULL CHECK (migration_sha256 ~ '^[0-9a-f]{64}$'),"
        "predecessor_id text, predecessor_sha256 text, transaction_mode text NOT NULL CHECK (transaction_mode='transactional'),"
        "minimum_schema_contract text NOT NULL, maximum_schema_contract text NOT NULL,"
        "forward_recovery_command text NOT NULL, applied_at timestamptz NOT NULL DEFAULT clock_timestamp(),"
        "applied_by text NOT NULL CHECK (applied_by='applypilot_fleet_migrator'))"
    )
    conn.execute(
        "CREATE FUNCTION applypilot_reject_migration_ledger_mutation() RETURNS trigger "
        "LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'fleet migration ledger is append-only'; END $$"
    )
    conn.execute(
        "CREATE TRIGGER applypilot_fleet_schema_migrations_append_only BEFORE UPDATE OR DELETE "
        "ON applypilot_fleet_schema_migrations FOR EACH ROW EXECUTE FUNCTION applypilot_reject_migration_ledger_mutation()"
    )
    conn.execute(
        "CREATE TRIGGER applypilot_fleet_schema_migrations_truncate BEFORE TRUNCATE "
        "ON applypilot_fleet_schema_migrations FOR EACH STATEMENT "
        "EXECUTE FUNCTION applypilot_reject_migration_ledger_mutation()"
    )
    conn.execute("REVOKE ALL ON applypilot_fleet_schema_migrations FROM PUBLIC")
    conn.execute("REVOKE ALL ON FUNCTION applypilot_reject_migration_ledger_mutation() FROM PUBLIC")
    _verify_ledger_contract(conn)


def _verify_ledger_contract(conn) -> None:
    expected_columns = (
        ("migration_id", "text", "NO"),
        ("manifest_schema_version", "integer", "NO"),
        ("migration_path", "text", "NO"),
        ("migration_sha256", "text", "NO"),
        ("predecessor_id", "text", "YES"),
        ("predecessor_sha256", "text", "YES"),
        ("transaction_mode", "text", "NO"),
        ("minimum_schema_contract", "text", "NO"),
        ("maximum_schema_contract", "text", "NO"),
        ("forward_recovery_command", "text", "NO"),
        ("applied_at", "timestamp with time zone", "NO"),
        ("applied_by", "text", "NO"),
    )
    columns = conn.execute(
        "SELECT column_name,data_type,is_nullable FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name='applypilot_fleet_schema_migrations' ORDER BY ordinal_position"
    ).fetchall()
    actual_columns = tuple((row["column_name"], row["data_type"], row["is_nullable"]) for row in columns)
    if actual_columns != expected_columns:
        raise MigrationError(f"migration ledger contract column mismatch: {actual_columns!r}")
    metadata = conn.execute(
        "SELECT owner.rolname AS owner_name FROM pg_class c JOIN pg_roles owner ON owner.oid=c.relowner "
        "WHERE c.oid='public.applypilot_fleet_schema_migrations'::regclass"
    ).fetchone()
    if metadata["owner_name"] != "applypilot_fleet_migrator":
        raise MigrationError("migration ledger contract owner mismatch")
    constraints = conn.execute(
        "SELECT contype,pg_get_constraintdef(oid,true) AS definition FROM pg_constraint "
        "WHERE conrelid='public.applypilot_fleet_schema_migrations'::regclass"
    ).fetchall()
    definitions = {(row["contype"], row["definition"]) for row in constraints}
    required_fragments = (
        ("p", "PRIMARY KEY (migration_id)"),
        ("u", "UNIQUE (migration_path)"),
        ("c", "manifest_schema_version = 1"),
        ("c", "migration_sha256 ~ '^[0-9a-f]{64}$'::text"),
        ("c", "transaction_mode = 'transactional'::text"),
        ("c", "applied_by = 'applypilot_fleet_migrator'::text"),
    )
    for constraint_type, fragment in required_fragments:
        if not any(kind == constraint_type and fragment in definition for kind, definition in definitions):
            raise MigrationError(f"migration ledger contract missing constraint: {fragment}")
    triggers = conn.execute(
        "SELECT tgname,tgenabled,pg_get_triggerdef(oid,true) AS definition FROM pg_trigger "
        "WHERE tgrelid='public.applypilot_fleet_schema_migrations'::regclass AND NOT tgisinternal"
    ).fetchall()
    trigger_map = {row["tgname"]: row for row in triggers}
    if set(trigger_map) != {
        "applypilot_fleet_schema_migrations_append_only",
        "applypilot_fleet_schema_migrations_truncate",
    } or any(row["tgenabled"] != "O" for row in trigger_map.values()):
        raise MigrationError("migration ledger contract trigger mismatch")
    function = conn.execute(
        "SELECT owner.rolname AS owner_name,p.prosrc,p.prosecdef FROM pg_proc p "
        "JOIN pg_roles owner ON owner.oid=p.proowner "
        "WHERE p.oid='public.applypilot_reject_migration_ledger_mutation()'::regprocedure"
    ).fetchone()
    if (
        function is None
        or function["owner_name"] != "applypilot_fleet_migrator"
        or function["prosecdef"]
        or function["prosrc"].strip() != "BEGIN RAISE EXCEPTION 'fleet migration ledger is append-only'; END"
    ):
        raise MigrationError("migration ledger contract mutation function mismatch")


def _ledger_prefix(conn, manifest: Manifest) -> tuple[str, ...]:
    rows = conn.execute(
        "SELECT migration_id,migration_path,migration_sha256,predecessor_id,predecessor_sha256,transaction_mode,"
        "minimum_schema_contract,maximum_schema_contract,forward_recovery_command,applied_by "
        "FROM applypilot_fleet_schema_migrations ORDER BY migration_id"
    ).fetchall()
    if len(rows) > len(manifest.migrations):
        raise MigrationError("migration ledger contains entries absent from the manifest")
    for index, row in enumerate(rows):
        migration = manifest.migrations[index]
        get = row.__getitem__
        expected = {
            "migration_id": migration.id,
            "migration_path": migration.path,
            "migration_sha256": migration.sha256,
            "predecessor_id": migration.predecessor_id,
            "predecessor_sha256": migration.predecessor_sha256,
            "transaction_mode": migration.transaction_mode,
            "minimum_schema_contract": migration.minimum_schema_contract,
            "maximum_schema_contract": migration.maximum_schema_contract,
            "forward_recovery_command": migration.forward_recovery_command,
            "applied_by": manifest.migration_role,
        }
        if any(get(key) != value for key, value in expected.items()):
            raise MigrationError(f"migration ledger mismatch at {migration.id}")
    return tuple(manifest.migrations[index].id for index in range(len(rows)))


def apply_manifest(
    conn,
    manifest: Manifest,
    repository_root: str | Path,
    *,
    lock_timeout_seconds: float = _DEFAULT_LOCK_TIMEOUT_SECONDS,
    statement_timeout_ms: int = _DEFAULT_STATEMENT_TIMEOUT_MS,
) -> ApplyResult:
    verify_predecessor(manifest, repository_root)
    verify_migration_files(manifest, repository_root)
    current = conn.execute("SELECT current_user AS current_user").fetchone()
    current_user = current["current_user"] if hasattr(current, "keys") else current[0]
    if current_user != manifest.migration_role:
        raise MigrationError(f"migration role mismatch: expected {manifest.migration_role}, got {current_user}")
    _acquire_lock(conn, lock_timeout_seconds)
    applied: list[str] = []
    try:
        _bootstrap_ledger(conn)
        conn.commit()
        already_applied = _ledger_prefix(conn, manifest)
        conn.commit()
        root = Path(repository_root).resolve()
        for migration in manifest.migrations[len(already_applied) :]:
            path = (root / Path(*PurePosixPath(migration.path).parts)).resolve()
            sql = path.read_text(encoding="utf-8", errors="strict")
            with conn.transaction():
                conn.execute(
                    "SELECT set_config('lock_timeout', %s, true)",
                    (f"{max(1, int(lock_timeout_seconds * 1000))}ms",),
                )
                conn.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (f"{max(1, int(statement_timeout_ms))}ms",),
                )
                conn.execute(sql)
                conn.execute(
                    "INSERT INTO applypilot_fleet_schema_migrations "
                    "(migration_id,manifest_schema_version,migration_path,migration_sha256,predecessor_id,"
                    "predecessor_sha256,transaction_mode,minimum_schema_contract,maximum_schema_contract,"
                    "forward_recovery_command,applied_by) VALUES (%s,1,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        migration.id,
                        migration.path,
                        migration.sha256,
                        migration.predecessor_id,
                        migration.predecessor_sha256,
                        migration.transaction_mode,
                        migration.minimum_schema_contract,
                        migration.maximum_schema_contract,
                        migration.forward_recovery_command,
                        manifest.migration_role,
                    ),
                )
            applied.append(migration.id)
        final_prefix = _ledger_prefix(conn, manifest)
        conn.commit()
        return ApplyResult(applied=tuple(applied), already_applied=final_prefix[: len(already_applied)])
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            conn.execute("SELECT pg_advisory_unlock(hashtext(%s))", (_LOCK_KEY,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
