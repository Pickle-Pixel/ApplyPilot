"""Static + unit contract for the private authority-migrator deployment.

These tests never run `docker build`, never touch the network, and never open a
live database. They parse the three deployment wrappers
(`Dockerfile.authority-migrator`, `railway.toml`,
`scripts/run-private-authority-migrator.py`), the package manifest that binds
them by hash, and the wrapper's pure validation helpers, asserting the Task 6
deployment-provenance contract:

- all three wrappers are tracked, the package manifest binds their exact hashes
  plus the fleet migration manifest, and wrapper hash drift / a missing bound
  wrapper / migration-manifest drift all fail closed;
- the Dockerfile build context carries only intended tracked files and excludes
  secrets/untracked payloads;
- neither the Dockerfile nor railway.toml bakes a target identity, credential,
  or default DSN;
- an absent target, a hostile target string, and a credential/DSN never leak.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_SCRIPT = REPO_ROOT / "scripts" / "run-private-authority-migrator.py"
DOCKERFILE = REPO_ROOT / "Dockerfile.authority-migrator"
RAILWAY_TOML = REPO_ROOT / "railway.toml"
PACKAGE_MANIFEST = REPO_ROOT / "deploy" / "authority-migrator" / "package-manifest.json"
WRAPPERS = (
    "Dockerfile.authority-migrator",
    "railway.toml",
    "scripts/run-private-authority-migrator.py",
)


def _load_wrapper_module():
    spec = importlib.util.spec_from_file_location("authority_migrator_wrapper", WRAPPER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wrapper = _load_wrapper_module()


def _stage_package(root: Path, *, extra: dict[str, bytes] | None = None) -> None:
    """Copy the package (wrappers + manifests) into a git-less tree."""
    for relative in (*WRAPPERS, "deploy/authority-migrator/package-manifest.json",
                     "src/applypilot/fleet/migrations/manifest-v1.json"):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO_ROOT / relative).read_bytes())
    for relative, data in (extra or {}).items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)


# --- Tracking + package-manifest binding -------------------------------------

def test_deployment_wrappers_are_git_tracked() -> None:
    for relative in WRAPPERS:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", relative],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{relative} is not tracked: {result.stderr}"


def test_package_manifest_binds_all_three_wrappers_and_migration_manifest() -> None:
    payload = wrapper.verify_package_manifest(REPO_ROOT, PACKAGE_MANIFEST)
    assert set(payload["wrappers"]) == set(WRAPPERS)
    assert payload["migration_manifest"]["path"] == "src/applypilot/fleet/migrations/manifest-v1.json"


def test_package_manifest_rejects_wrapper_hash_drift(tmp_path) -> None:
    _stage_package(tmp_path)
    drifted = tmp_path / "railway.toml"
    drifted.write_bytes(drifted.read_bytes() + b"\n# drift\n")
    with pytest.raises(wrapper.PackageError, match="wrapper hash drift"):
        wrapper.verify_package_manifest(tmp_path, tmp_path / "deploy/authority-migrator/package-manifest.json")


def test_package_manifest_rejects_missing_bound_wrapper(tmp_path) -> None:
    _stage_package(tmp_path)
    (tmp_path / "Dockerfile.authority-migrator").unlink()
    with pytest.raises(wrapper.PackageError, match="bound wrapper is missing"):
        wrapper.verify_package_manifest(tmp_path, tmp_path / "deploy/authority-migrator/package-manifest.json")


def test_package_manifest_rejects_dropped_wrapper_from_its_own_list(tmp_path) -> None:
    # A tampered manifest that simply omits a required wrapper (here the Dockerfile) from its own
    # ``wrappers`` map must fail closed: the verifier asserts the exact required set rather than
    # trusting whatever keys the manifest lists, so a dropped wrapper cannot escape hash coverage.
    manifest_relative = "deploy/authority-migrator/package-manifest.json"
    payload = json.loads((REPO_ROOT / manifest_relative).read_text(encoding="utf-8"))
    payload["wrappers"].pop("Dockerfile.authority-migrator")
    _stage_package(tmp_path)
    (tmp_path / manifest_relative).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(wrapper.PackageError, match="wrapper set does not match"):
        wrapper.verify_package_manifest(tmp_path, tmp_path / manifest_relative)


def test_package_manifest_rejects_migration_manifest_drift(tmp_path) -> None:
    _stage_package(tmp_path)
    manifest = tmp_path / "src/applypilot/fleet/migrations/manifest-v1.json"
    manifest.write_bytes(manifest.read_bytes().replace(b"applypilot-fleet-migrations-v1", b"tampered"))
    with pytest.raises(wrapper.PackageError, match="fleet migration manifest hash drift"):
        wrapper.verify_package_manifest(tmp_path, tmp_path / "deploy/authority-migrator/package-manifest.json")


# --- Absent / hostile target, no leakage -------------------------------------

def test_absent_target_arguments_fail_closed(capsys) -> None:
    # A missing required target parameter aborts at argument parsing.
    with pytest.raises(SystemExit) as excinfo:
        wrapper.main(["--expected-database-name", "appdb", "--expected-system-identifier", "123"])
    assert excinfo.value.code == 2

    # A named-but-unset target env var fails closed before any connection.
    env_name = "AUTHORITY_MIGRATOR_TEST_ABSENT_DSN"
    os.environ.pop(env_name, None)
    code = wrapper.main([
        "--target-dsn-env", env_name,
        "--expected-database-name", "appdb",
        "--expected-system-identifier", "123",
    ])
    assert code == 1
    assert "is not set" in capsys.readouterr().err


def test_hostile_target_strings_fail_closed() -> None:
    for hostile in (
        "postgresql://u:p@h/db\nDROP DATABASE app",  # newline injection
        "postgresql://u:p@h/db\x00",  # embedded NUL
        "host='x'; rm -rf / #",  # shell metacharacters (never shelled out)
        "postgresql:///no-host",  # no explicit host
        "",  # empty
    ):
        with pytest.raises(wrapper.TargetError):
            wrapper.validate_target(hostile)

    for bad_env in ("lower_case", "1LEADING_DIGIT", "HAS-DASH", ""):
        with pytest.raises(wrapper.TargetError):
            wrapper._resolve_target_dsn(bad_env)


def test_hostile_target_env_value_fails_closed_without_leaking(capsys) -> None:
    env_name = "AUTHORITY_MIGRATOR_TEST_HOSTILE_DSN"
    secret = "sup3r-s3cret-password"
    os.environ[env_name] = f"postgresql://u:{secret}@h/db\nDROP"
    try:
        code = wrapper.main([
            "--target-dsn-env", env_name,
            "--expected-database-name", "appdb",
            "--expected-system-identifier", "123",
        ])
    finally:
        os.environ.pop(env_name, None)
    assert code == 1
    captured = capsys.readouterr()
    assert secret not in captured.err
    assert secret not in captured.out


def test_redact_dsn_never_exposes_the_password() -> None:
    redacted = wrapper.redact_dsn("postgresql://alice:hunter2@db.railway.internal:5432/appdb")
    assert redacted == "alice@db.railway.internal/appdb"
    assert "hunter2" not in redacted


# --- Dockerfile build context ------------------------------------------------

def test_dockerfile_copies_only_tracked_intended_files() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    copied = [
        line.split(None, 2)[1]
        for line in text.splitlines()
        if line.startswith("COPY ")
    ]
    assert copied, "Dockerfile.authority-migrator must copy an explicit file set"
    # No broad `COPY . /app` that would drag untracked payloads into the image.
    assert "." not in copied
    tracked = set(
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    )
    for source in copied:
        path = REPO_ROOT / source
        if path.is_dir():
            listed = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", source],
                capture_output=True,
                text=True,
                check=False,
            )
            assert listed.returncode == 0, f"COPY source {source} is not tracked"
        else:
            assert source in tracked, f"COPY source {source} is not a tracked file"


def test_dockerfile_bakes_no_target_identity_or_secret() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert 'ENTRYPOINT ["python3", "/app/scripts/run-private-authority-migrator.py"]' in text
    # No baked DSN, credential, or default target.
    for forbidden in ("postgres://", "postgresql://", "PGPASSWORD", "DATABASE_URL", "password="):
        assert forbidden not in text, f"Dockerfile must not bake {forbidden!r}"


def test_dockerignore_excludes_secrets_and_untracked_payloads() -> None:
    lines = {line.strip() for line in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()}
    for required in (".git", ".env", ".env.*", "tests"):
        assert required in lines, f".dockerignore must exclude {required}"


# --- railway.toml -------------------------------------------------------------

def test_railway_config_targets_the_migrator_dockerfile() -> None:
    config = tomllib.loads(RAILWAY_TOML.read_text(encoding="utf-8"))
    assert config["build"]["dockerfilePath"] == "Dockerfile.authority-migrator"
    assert config["deploy"]["restartPolicyType"] == "never"
    start = config["deploy"]["startCommand"]
    # The three explicit target parameters are named, not baked.
    assert "--target-dsn-env AUTHORITY_MIGRATOR_TARGET_DSN" in start
    assert "--expected-database-name ${AUTHORITY_MIGRATOR_EXPECTED_DB}" in start
    assert "--expected-system-identifier ${AUTHORITY_MIGRATOR_EXPECTED_SYSID}" in start


def test_railway_config_bakes_no_literal_target_or_credential() -> None:
    text = RAILWAY_TOML.read_text(encoding="utf-8")
    for forbidden in ("postgres://", "postgresql://", "PGPASSWORD", "password=", ".railway.internal"):
        assert forbidden not in text, f"railway.toml must not bake {forbidden!r}"
