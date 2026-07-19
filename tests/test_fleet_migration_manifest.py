from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from applypilot.fleet import migrator
from applypilot.fleet import pg_roles
from applypilot.fleet import schema as fleet_schema


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "src" / "applypilot" / "fleet" / "migrations" / "manifest-v1.json"

# The canonical, published predecessor tuple these tests build archives around.
PINNED_COMMIT = "2b3a7c83118df840dda60c9b728f29e3dc0c1b9d"
PINNED_TREE = "b160cec1ef3c2bd35989bf7127ca0554269b45a0"
# Canonical (sorted) predecessor file -> exact committed blob id.
PINNED_FILES = {
    "src/applypilot/apply/fleet_schema.sql": "6eb4a84dcc05568233ee32551b28c75f13b0a17f",
    "src/applypilot/fleet/schema.py": "5637001b6457f9fc8f0c22f4220fe2e1249ff9c0",
    "src/applypilot/fleet/schema_v3.sql": "0741a6e675d2ea42a3bb0d785fd4c0c444e96b3d",
}


def _git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _cat_blob(git_blob: str) -> bytes:
    """The exact committed predecessor bytes, as a real release archive carries them."""
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "cat-file", "blob", git_blob],
        check=True,
        capture_output=True,
    ).stdout


def _write_predecessor_bundle(
    root: Path,
    files: dict[str, str],
    *,
    overrides: dict[str, bytes] | None = None,
) -> list[dict[str, str]]:
    """Write the content-addressed predecessor byte bundle a release archive carries."""
    overrides = overrides or {}
    bundle = root / "src" / "applypilot" / "fleet" / "migrations" / "predecessor"
    bundle.mkdir(parents=True, exist_ok=True)
    predecessor_files: list[dict[str, str]] = []
    for relative in sorted(files):
        git_blob = files[relative]
        (bundle / git_blob).write_bytes(overrides.get(git_blob, _cat_blob(git_blob)))
        predecessor_files.append({"path": relative, "git_blob": git_blob})
    return predecessor_files


def _write_manifest(
    root: Path,
    migrations: list[dict[str, object]],
    *,
    runtime_commit: str = PINNED_COMMIT,
    runtime_tree: str = PINNED_TREE,
    files: dict[str, str] | None = None,
    bundle_overrides: dict[str, bytes] | None = None,
) -> Path:
    predecessor_files = _write_predecessor_bundle(
        root, files if files is not None else PINNED_FILES, overrides=bundle_overrides
    )
    payload = {
        "schema_version": 1,
        "manifest_id": "applypilot-fleet-migrations-v1",
        "migration_role": "applypilot_fleet_migrator",
        "predecessor": {
            "runtime_commit": runtime_commit,
            "runtime_tree": runtime_tree,
            "files": predecessor_files,
        },
        "migrations": migrations,
    }
    path = root / "manifest-v1.json"
    path.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _migration(root: Path, migration_id: str, predecessor_id: str | None, predecessor_sha256: str | None) -> dict[str, object]:
    relative = f"src/applypilot/fleet/migrations/{migration_id}.sql"
    sql = f"CREATE TABLE migration_probe_{migration_id.split('_')[1]} (id integer PRIMARY KEY);\n"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sql, encoding="utf-8", newline="\n")
    return {
        "id": migration_id,
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "predecessor_id": predecessor_id,
        "predecessor_sha256": predecessor_sha256,
        "transaction_mode": "transactional",
        "minimum_schema_contract": "fleet-v3",
        "maximum_schema_contract": "fleet-v3",
        "forward_recovery_command": f"applypilot-fleet-migrate --through {migration_id}",
    }


@pytest.fixture
def migration_pg(fleet_pg):
    with psycopg.connect(fleet_pg, autocommit=True, row_factory=dict_row) as conn:
        fleet_schema.ensure_schema_v3(conn)
        original_owner = conn.execute("SELECT current_user AS role_name").fetchone()["role_name"]
        conn.execute("DROP TABLE IF EXISTS applypilot_fleet_schema_migrations CASCADE")
        conn.execute("DROP FUNCTION IF EXISTS applypilot_reject_migration_ledger_mutation() CASCADE")
        if conn.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='applypilot_fleet_migrator') AS present"
        ).fetchone()["present"]:
            conn.execute("DROP OWNED BY applypilot_fleet_migrator CASCADE")
            conn.execute("DROP ROLE applypilot_fleet_migrator")
        conn.execute("CREATE ROLE applypilot_fleet_migrator NOLOGIN")
        conn.execute("GRANT USAGE, CREATE ON SCHEMA public TO applypilot_fleet_migrator")
        with conn.cursor() as cur:
            pg_roles._transfer_application_ownership(
                cur,
                new_owner_role="applypilot_fleet_migrator",
            )
    try:
        yield fleet_pg
    finally:
        with psycopg.connect(fleet_pg, autocommit=True, row_factory=dict_row) as conn:
            conn.execute("DROP TABLE IF EXISTS applypilot_fleet_schema_migrations CASCADE")
            conn.execute("DROP FUNCTION IF EXISTS applypilot_reject_migration_ledger_mutation() CASCADE")
            if conn.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='applypilot_fleet_migrator') AS present"
            ).fetchone()["present"]:
                conn.execute(
                    sql.SQL("REASSIGN OWNED BY applypilot_fleet_migrator TO {}").format(
                        sql.Identifier(original_owner)
                    )
                )
                conn.execute("DROP OWNED BY applypilot_fleet_migrator CASCADE")
                conn.execute("DROP ROLE applypilot_fleet_migrator")


def test_repository_manifest_pins_the_exact_legacy_predecessor():
    manifest = migrator.load_manifest(MANIFEST_PATH)
    assert manifest.predecessor.runtime_commit == "2b3a7c83118df840dda60c9b728f29e3dc0c1b9d"
    assert manifest.predecessor.runtime_tree == "b160cec1ef3c2bd35989bf7127ca0554269b45a0"
    assert {item.path: item.git_blob for item in manifest.predecessor.files} == {
        "src/applypilot/fleet/schema_v3.sql": "0741a6e675d2ea42a3bb0d785fd4c0c444e96b3d",
        "src/applypilot/apply/fleet_schema.sql": "6eb4a84dcc05568233ee32551b28c75f13b0a17f",
        "src/applypilot/fleet/schema.py": "5637001b6457f9fc8f0c22f4220fe2e1249ff9c0",
    }
    # The predecessor blobs are the historical bytes, not today's HEAD bytes.
    assert manifest.predecessor.runtime_tree == PINNED_TREE
    assert {item.path: item.git_blob for item in manifest.predecessor.files} == PINNED_FILES
    assert _git_blob_id((REPO_ROOT / "src/applypilot/fleet/schema_v3.sql").read_bytes()) != (
        "0741a6e675d2ea42a3bb0d785fd4c0c444e96b3d"
    )
    assert tuple(item.id for item in manifest.migrations) == (
        "20260715_001_application_authority",
        "20260717_002_lane_specific_canary_pins",
        "20260719_003_harden_application_authority_functions",
    )
    migrator.verify_predecessor(manifest, REPO_ROOT)


def test_repository_manifest_rejects_wrong_predecessor_commit(tmp_path):
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["predecessor"]["runtime_commit"] = "0" * 40
    path = tmp_path / "manifest-v1.json"
    path.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    manifest = migrator.load_manifest(path)
    with pytest.raises(migrator.ManifestError, match="predecessor blob mismatch"):
        migrator.verify_predecessor(manifest, REPO_ROOT)


def test_archive_without_git_requires_exact_predecessor_bytes(tmp_path):
    # A faithful release archive omits .git and carries the exact predecessor bytes.
    manifest = migrator.load_manifest(_write_manifest(tmp_path, []))
    assert not (tmp_path / ".git").exists()
    migrator.verify_predecessor(manifest, tmp_path)

    # The current (HEAD) schema_v3 bytes hash to a different blob than the pinned
    # predecessor. Carrying them in the bundle under the matching pinned (commit,
    # tree, files) tuple is exactly the "wrong bytes over a matching tuple" attack
    # and must fail closed -- the tuple alone is never sufficient.
    schema_v3_blob = PINNED_FILES["src/applypilot/fleet/schema_v3.sql"]
    current = (REPO_ROOT / "src/applypilot/fleet/schema_v3.sql").read_bytes()
    assert migrator._blob_id(current) != schema_v3_blob
    migrator.load_manifest(
        _write_manifest(tmp_path, [], bundle_overrides={schema_v3_blob: current})
    )
    with pytest.raises(migrator.ManifestError, match="predecessor blob mismatch"):
        migrator.verify_predecessor(manifest, tmp_path)


def test_archive_rejects_unpinned_commit_and_tree_drift(tmp_path):
    # An unknown predecessor commit is not a pinned release predecessor.
    manifest = migrator.load_manifest(_write_manifest(tmp_path, [], runtime_commit="0" * 40))
    with pytest.raises(migrator.ManifestError, match="not a pinned release predecessor"):
        migrator.verify_predecessor(manifest, tmp_path)

    # The right commit under a forged tree fails the tree binding.
    manifest = migrator.load_manifest(_write_manifest(tmp_path, [], runtime_tree="1" * 40))
    with pytest.raises(migrator.ManifestError, match="predecessor tree mismatch"):
        migrator.verify_predecessor(manifest, tmp_path)


def test_archive_rejects_missing_and_extra_predecessor_files(tmp_path):
    # A missing pinned file is a closed-set mismatch.
    subset = dict(PINNED_FILES)
    subset.pop("src/applypilot/fleet/schema.py")
    manifest = migrator.load_manifest(_write_manifest(tmp_path, [], files=subset))
    with pytest.raises(migrator.ManifestError, match="does not match the pinned release tuple"):
        migrator.verify_predecessor(manifest, tmp_path)

    # An extra file (even one carrying a real blob) is also a closed-set mismatch.
    extra = dict(PINNED_FILES)
    extra["src/applypilot/apply/extra_probe.sql"] = PINNED_FILES[
        "src/applypilot/apply/fleet_schema.sql"
    ]
    manifest = migrator.load_manifest(_write_manifest(tmp_path, [], files=extra))
    with pytest.raises(migrator.ManifestError, match="does not match the pinned release tuple"):
        migrator.verify_predecessor(manifest, tmp_path)


def test_manifest_rejects_reordered_predecessor_files(tmp_path):
    path = _write_manifest(tmp_path, [])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["predecessor"]["files"].reverse()
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(migrator.ManifestError, match="canonical sorted order"):
        migrator.load_manifest(path)


def test_manifest_rejects_duplicate_keys_unknown_fields_and_unsafe_paths(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8", newline="\n")
    with pytest.raises(migrator.ManifestError, match="duplicate"):
        migrator.load_manifest(duplicate)

    valid = json.loads(_write_manifest(tmp_path, []).read_text(encoding="utf-8"))
    valid["unexpected"] = True
    (tmp_path / "unknown.json").write_text(json.dumps(valid) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(migrator.ManifestError, match="fields"):
        migrator.load_manifest(tmp_path / "unknown.json")

    migration = _migration(tmp_path, "20260715_001_probe", None, None)
    migration["path"] = "../escape.sql"
    with pytest.raises(migrator.ManifestError, match="path"):
        migrator.load_manifest(_write_manifest(tmp_path, [migration]))


def test_manifest_rejects_reordered_duplicate_and_broken_chains(tmp_path):
    first = _migration(tmp_path, "20260715_001_probe", None, None)
    second = _migration(tmp_path, "20260715_002_probe", first["id"], first["sha256"])
    migrator.load_manifest(_write_manifest(tmp_path, [first, second]))

    with pytest.raises(migrator.ManifestError, match="strictly increasing"):
        migrator.load_manifest(_write_manifest(tmp_path, [second, first]))
    with pytest.raises(migrator.ManifestError, match="duplicate"):
        migrator.load_manifest(_write_manifest(tmp_path, [first, first]))
    broken = dict(second)
    broken["predecessor_sha256"] = "0" * 64
    with pytest.raises(migrator.ManifestError, match="predecessor"):
        migrator.load_manifest(_write_manifest(tmp_path, [first, broken]))


def test_predecessor_and_sql_checksums_are_verified_from_bytes(tmp_path):
    manifest_path = _write_manifest(tmp_path, [])
    manifest = migrator.load_manifest(manifest_path)
    schema_py_blob = PINNED_FILES["src/applypilot/fleet/schema.py"]
    (tmp_path / "src/applypilot/fleet/migrations/predecessor" / schema_py_blob).write_bytes(b"tampered\n")
    with pytest.raises(migrator.ManifestError, match="predecessor blob"):
        migrator.verify_predecessor(manifest, tmp_path)

    first = _migration(tmp_path, "20260715_001_probe", None, None)
    manifest = migrator.load_manifest(_write_manifest(tmp_path, [first]))
    (tmp_path / str(first["path"])).write_text("SELECT 2;\n", encoding="utf-8", newline="\n")
    with pytest.raises(migrator.ManifestError, match="checksum"):
        migrator.verify_migration_files(manifest, tmp_path)


def test_apply_requires_exact_role_and_creates_immutable_ledger(migration_pg, tmp_path):
    manifest = migrator.load_manifest(_write_manifest(tmp_path, []))
    with psycopg.connect(migration_pg, row_factory=dict_row) as conn:
        with pytest.raises(migrator.MigrationError, match="migration role"):
            migrator.apply_manifest(conn, manifest, tmp_path)
        conn.rollback()
        conn.execute("SET ROLE applypilot_fleet_migrator")
        result = migrator.apply_manifest(conn, manifest, tmp_path)
        assert result.applied == ()
        assert result.already_applied == ()
        with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
            conn.execute("TRUNCATE applypilot_fleet_schema_migrations")


def test_apply_is_atomic_resumable_and_rejects_ledger_drift(migration_pg, tmp_path):
    first = _migration(tmp_path, "20260715_001_probe", None, None)
    second = _migration(tmp_path, "20260715_002_probe", first["id"], first["sha256"])
    manifest = migrator.load_manifest(_write_manifest(tmp_path, [first, second]))
    with psycopg.connect(migration_pg, row_factory=dict_row) as conn:
        conn.execute("SET ROLE applypilot_fleet_migrator")
        result = migrator.apply_manifest(conn, manifest, tmp_path)
        assert result.applied == (first["id"], second["id"])
        result = migrator.apply_manifest(conn, manifest, tmp_path)
        assert result.applied == ()
        assert result.already_applied == (first["id"], second["id"])
        conn.execute("RESET ROLE")
        conn.execute("ALTER TABLE applypilot_fleet_schema_migrations DISABLE TRIGGER USER")
        conn.execute("UPDATE applypilot_fleet_schema_migrations SET migration_sha256=%s WHERE migration_id=%s", ("0" * 64, first["id"]))
        conn.execute("ALTER TABLE applypilot_fleet_schema_migrations ENABLE TRIGGER USER")
        conn.commit()
        conn.execute("SET ROLE applypilot_fleet_migrator")
        with pytest.raises(migrator.MigrationError, match="ledger mismatch"):
            migrator.apply_manifest(conn, manifest, tmp_path)


def test_existing_ledger_contract_is_verified_not_repaired(migration_pg, tmp_path):
    manifest = migrator.load_manifest(_write_manifest(tmp_path, []))
    with psycopg.connect(migration_pg, row_factory=dict_row) as conn:
        conn.execute("SET ROLE applypilot_fleet_migrator")
        migrator.apply_manifest(conn, manifest, tmp_path)
        conn.execute("RESET ROLE")
        conn.execute("ALTER TABLE applypilot_fleet_schema_migrations ADD COLUMN injected text")
        conn.commit()
        conn.execute("SET ROLE applypilot_fleet_migrator")
        with pytest.raises(migrator.MigrationError, match="ledger contract"):
            migrator.apply_manifest(conn, manifest, tmp_path)


def test_failed_migration_rolls_back_sql_and_ledger(migration_pg, tmp_path):
    first = _migration(tmp_path, "20260715_001_probe", None, None)
    path = tmp_path / str(first["path"])
    path.write_text("CREATE TABLE must_rollback (id integer);\nSELECT 1 / 0;\n", encoding="utf-8", newline="\n")
    first["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = migrator.load_manifest(_write_manifest(tmp_path, [first]))
    with psycopg.connect(migration_pg, row_factory=dict_row) as conn:
        conn.execute("SET ROLE applypilot_fleet_migrator")
        with pytest.raises(psycopg.errors.DivisionByZero):
            migrator.apply_manifest(conn, manifest, tmp_path)
        conn.rollback()
        assert conn.execute("SELECT to_regclass('public.must_rollback') AS oid").fetchone()["oid"] is None
        assert conn.execute("SELECT count(*) AS n FROM applypilot_fleet_schema_migrations").fetchone()["n"] == 0


def test_apply_rebinds_executed_bytes_against_hostile_reread(migration_pg, tmp_path, monkeypatch):
    # A migration file honestly passes the up-front verify_migration_files hash, then an attacker
    # swaps its bytes in the window before apply_manifest re-reads the file to execute it. The
    # execution-time re-read MUST rehash the exact bytes and refuse the swap, so neither the hostile
    # SQL runs nor a ledger row (which binds the manifest hash) is minted for unrehashed bytes.
    first = _migration(tmp_path, "20260715_001_probe", None, None)
    manifest = migrator.load_manifest(_write_manifest(tmp_path, [first]))
    migration_path = tmp_path / str(first["path"])
    honest_sha = first["sha256"]

    real_verify = migrator.verify_migration_files

    def hostile_verify(bound_manifest, root):
        # The honest bytes pass the up-front verification exactly as in production...
        real_verify(bound_manifest, root)
        # ...then the file is swapped in the TOCTOU window before the execution re-read.
        migration_path.write_text(
            "DROP TABLE IF EXISTS applypilot_fleet_schema_migrations;\n",
            encoding="utf-8",
            newline="\n",
        )
        assert hashlib.sha256(migration_path.read_bytes()).hexdigest() != honest_sha

    monkeypatch.setattr(migrator, "verify_migration_files", hostile_verify)
    with psycopg.connect(migration_pg, row_factory=dict_row) as conn:
        conn.execute("SET ROLE applypilot_fleet_migrator")
        with pytest.raises(migrator.ManifestError, match="checksum mismatch"):
            migrator.apply_manifest(conn, manifest, tmp_path)
        conn.rollback()
        # The swapped SQL never executed and no ledger row bound the manifest hash to it.
        conn.execute("SET ROLE applypilot_fleet_migrator")
        assert conn.execute(
            "SELECT count(*) AS n FROM applypilot_fleet_schema_migrations"
        ).fetchone()["n"] == 0


def test_application_authority_is_single_owner_and_crash_conservative(migration_pg):
    signatures = (
        "public.fleet_worker_authorize_lease(text,text,text,text,uuid,text,integer)",
        "public.fleet_worker_mark_browser_interaction(text,text,bigint)",
        "public.fleet_worker_terminalize(text,text,bigint,text,jsonb)",
        "public.fleet_worker_requeue(text,text,bigint)",
        "public.fleet_worker_expire_authority()",
    )
    manifest = migrator.load_manifest(MANIFEST_PATH)
    with psycopg.connect(migration_pg, autocommit=True, row_factory=dict_row) as provider:
        created_worker_role = not provider.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='fleet_worker') AS present"
        ).fetchone()["present"]
        if created_worker_role:
            provider.execute("CREATE ROLE fleet_worker NOLOGIN")
    try:
        with psycopg.connect(migration_pg, row_factory=dict_row) as conn:
            conn.execute("SET ROLE applypilot_fleet_migrator")
            legacy_manifest = replace(manifest, migrations=manifest.migrations[:2])
            legacy_result = migrator.apply_manifest(conn, legacy_manifest, REPO_ROOT)
            assert legacy_result.applied == (
                "20260715_001_application_authority",
                "20260717_002_lane_specific_canary_pins",
            )
            result = migrator.apply_manifest(conn, manifest, REPO_ROOT)
            assert result.already_applied == legacy_result.applied
            assert result.applied == ("20260719_003_harden_application_authority_functions",)
            for signature in signatures:
                row = conn.execute(
                    "SELECT p.oid::regprocedure::text AS signature,p.prosecdef,p.proconfig,"
                    "COALESCE(bool_or(acl.grantee=0 AND acl.privilege_type='EXECUTE'),FALSE) "
                    "AS public_execute,"
                    "COALESCE(bool_or(grantee.rolname='fleet_worker' AND acl.privilege_type='EXECUTE'),FALSE) "
                    "AS worker_execute "
                    "FROM pg_proc p "
                    "LEFT JOIN LATERAL aclexplode(COALESCE(p.proacl,acldefault('f',p.proowner))) acl ON TRUE "
                    "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
                    "WHERE p.oid=to_regprocedure(%s) GROUP BY p.oid,p.prosecdef,p.proconfig",
                    (signature,),
                ).fetchone()
                assert row is not None, signature
                assert row["prosecdef"] is True, signature
                assert row["proconfig"] == ["search_path=pg_catalog, public"], signature
                assert row["public_execute"] is False, signature
                assert row["worker_execute"] is True, signature
            op = "00000000-0000-0000-0000-000000000001"
            first = conn.execute(
                "SELECT * FROM fleet_worker_authorize_lease(%s,%s,%s,%s,%s::uuid,%s,%s)",
                ("app-1", "https://jobs.example/a", "worker-a", "ats", op, "hash-a", 60),
            ).fetchone()
            assert first["authority_epoch"] == 1
            assert conn.execute(
                "SELECT fleet_worker_mark_browser_interaction(%s,%s,%s)",
                ("app-1", "worker-a", 1),
            ).fetchone()["fleet_worker_mark_browser_interaction"]
            conn.commit()
            with pytest.raises(psycopg.errors.RaiseException, match="not claimable"):
                conn.execute(
                    "SELECT * FROM fleet_worker_authorize_lease(%s,%s,%s,%s,%s::uuid,%s,%s)",
                    ("app-1", "https://jobs.example/a?alias=1", "worker-b", "linkedin", op, "hash-a", 60),
                )
            conn.rollback()
            assert not conn.execute(
                "SELECT fleet_worker_requeue(%s,%s,%s)", ("app-1", "worker-a", 1)
            ).fetchone()["fleet_worker_requeue"]
            assert conn.execute(
                "SELECT fleet_worker_terminalize(%s,%s,%s,%s,%s::jsonb)",
                ("app-1", "worker-a", 1, "applied", '{"receipt":"r1"}'),
            ).fetchone()["fleet_worker_terminalize"]
            row = conn.execute(
                "SELECT state, terminal_status, terminal_evidence->>'receipt' AS receipt "
                "FROM fleet_application_authority WHERE canonical_application_id='app-1'"
            ).fetchone()
            assert (row["state"], row["terminal_status"], row["receipt"]) == ("terminal", "applied", "r1")
    finally:
        if created_worker_role:
            with psycopg.connect(migration_pg, autocommit=True) as provider:
                provider.execute("DROP OWNED BY fleet_worker")
                provider.execute("DROP ROLE fleet_worker")


def _load_authority_migrator_wrapper():
    path = REPO_ROOT / "scripts" / "run-private-authority-migrator.py"
    spec = importlib.util.spec_from_file_location("authority_migrator_wrapper_it", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_private_wrapper_applies_to_explicit_target_and_never_leaks(migration_pg, capsys, monkeypatch):
    wrapper = _load_authority_migrator_wrapper()
    # Start from a clean authority slate: the migration_pg fixture rebuilds the
    # ledger per test but leaves migration-created relations behind, and the
    # first migration's CREATE TABLE is not idempotent.
    with psycopg.connect(migration_pg, autocommit=True) as admin:
        admin.execute("DROP TABLE IF EXISTS fleet_application_alias CASCADE")
        admin.execute("DROP TABLE IF EXISTS fleet_application_authority CASCADE")
    with psycopg.connect(migration_pg, row_factory=dict_row) as conn:
        identity = conn.execute(
            "SELECT current_database() AS db,"
            "control.system_identifier::text AS sysid FROM pg_control_system() AS control"
        ).fetchone()
    password = psycopg.conninfo.conninfo_to_dict(migration_pg).get("password")
    assert password, "the disposable test DSN is expected to carry a password to redact"

    monkeypatch.setenv("AUTHORITY_MIGRATOR_IT_DSN", migration_pg)
    base = dict(
        target_dsn_env="AUTHORITY_MIGRATOR_IT_DSN",
        expected_database_name=identity["db"],
        expected_system_identifier=identity["sysid"],
        repository_root=str(REPO_ROOT),
        package_manifest=str(REPO_ROOT / "deploy" / "authority-migrator" / "package-manifest.json"),
    )

    # The explicit, matching target applies exactly the three real migrations.
    assert wrapper.run(argparse.Namespace(**base)) == 0
    out = capsys.readouterr().out
    assert "applied=3" in out
    assert password not in out
    assert "AUTHORITY_MIGRATOR_IT_DSN" not in out  # the value, not the name, stays private

    # A mismatched declared database fails closed before touching the schema, and
    # still never surfaces the DSN/password.
    wrong = argparse.Namespace(**{**base, "expected_database_name": "not_the_declared_target"})
    with pytest.raises(wrapper.TargetError, match="database name does not match"):
        wrapper.run(wrong)
    leaked = capsys.readouterr()
    assert password not in leaked.out
    assert password not in leaked.err
