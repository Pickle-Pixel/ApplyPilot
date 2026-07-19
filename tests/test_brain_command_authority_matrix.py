"""Tests for the Runtime brain command-authority discovery / matrix / receipt.

These exercise the fail-closed gates against both the *real* production
entrypoints (the checked-in matrix must have zero unclassified surfaces) and
injected synthetic repo facts (new/odd entrypoints, collisions, drift, forged
receipts, dirty trees, secrets).  Nothing here touches a live DB, remote, or
service.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import brain_command_authority as authority  # noqa: E402

MATRIX_PATH = REPO_ROOT / "config" / "brain-command-authority-matrix.runtime.json"
INVENTORY_CLI = SCRIPTS_DIR / "inventory-brain-command-authority.py"
VERIFY_CLI = SCRIPTS_DIR / "verify-brain-command-authority.py"


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real_discovered() -> list[dict]:
    return authority.discover(REPO_ROOT)


@pytest.fixture(scope="module")
def real_matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def _valid_matrix(real_matrix: dict) -> dict:
    return copy.deepcopy(real_matrix)


def _init_git_repo(root: Path) -> None:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "t@example.com"], check=True, env=env
    )
    subprocess.run(["git", "-C", str(root), "config", "user.name", "t"], check=True, env=env)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-q", "-m", "init"], check=True, env=env
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _minimal_repo(root: Path) -> None:
    _write(
        root / "pyproject.toml",
        '[project]\nname = "demo"\n\n[project.scripts]\n'
        'demo-run = "demo.cli:main"\n',
    )
    _write(root / "src" / "demo" / "__init__.py", "")
    _write(
        root / "src" / "demo" / "cli.py",
        'def main():\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n',
    )


def _complete_fixture_repo(root: Path) -> None:
    """Build one small repo containing every supported structural kind."""
    _minimal_repo(root)
    _write(root / "src" / "demo" / "__main__.py", "print('package runner')\n")
    _write(
        root / "src" / "demo" / "commands.py",
        """from mcp.server.fastmcp import FastMCP
import typer

app = typer.Typer()
admin = typer.Typer()
app.add_typer(admin, name="admin")
server = FastMCP("demo")

@app.command()
def status():
    return None

@admin.command("repair")
def repair_command():
    return None

@server.tool(name="renamed-tool")
def original_tool():
    return None

def configure():
    @server.tool()
    def nested_tool():
        return None

def registered_callable():
    return None

server.add_tool(registered_callable, name="registered-tool")
""",
    )
    _write(
        root / "src" / "demo" / "parser.py",
        """import argparse
parser = argparse.ArgumentParser()
subcommands = parser.add_subparsers()
subcommands.add_parser("inspect")
subcommands.add_parser("repair")
""",
    )
    _write(
        root / "src" / "demo" / "web.py",
        """from fastapi import FastAPI
api = FastAPI()

@api.post("/jobs")
def create_job():
    return None
""",
    )
    _write(root / "tools" / "nested" / "generate.report.py", "print('report')\n")
    _write(
        root / "ops" / "tasks.ps1",
        "$action = New-ScheduledTaskAction -Execute 'python'\n"
        "Register-ScheduledTask -TaskName 'demo' -Action $action\n"
        "$wrapper = 'wrapper.ps1'\nSet-Content -Path $wrapper -Value 'python -m demo'\n",
    )
    _write(root / "ops" / "module.psm1", "function Invoke-Demo { return 0 }\n")
    _write(root / "ops" / "launch.sh", "#!/bin/sh\npython -m demo\n")
    _write(root / "ops" / "launch.cmd", "@python -m demo\n")
    _write(root / "ops" / "demo.service", "[Service]\nExecStart=python -m demo\n")
    _write(root / "ops" / "demo.plist", "<plist><dict></dict></plist>\n")
    _write(root / "Dockerfile", 'FROM python:3.13\nENTRYPOINT ["python"]\nCMD ["-m", "demo"]\n')


def _write_fixture_matrix(root: Path) -> Path:
    rows = []
    for surface in authority.discover(root):
        rows.append(
            {
                **surface,
                "owner": "operator",
                "readWriteClass": "control",
                "resourcesRead": ["repository-source"],
                "resourcesWritten": ["repository-source"],
                "employerMutation": True,
                "authoritySource": "none",
                "capabilities": [],
                "pauseScopes": ["none"],
                "liveEligibility": False,
                "failClosedBehavior": "deny-not-live-writer",
            }
        )
    matrix = {
        "schemaVersion": authority.MATRIX_SCHEMA_VERSION,
        "runtimeKind": authority.RUNTIME_KIND,
        "repository": authority.REPOSITORY,
        "noGlobalAuthorityClaim": True,
        "scope": {
            "includedSurfaceKinds": list(authority.INCLUDED_SURFACE_KINDS),
            "deferredSurfaceKinds": [],
            "deferralReason": "Complete fixture inventory; no kinds deferred.",
        },
        "pauseScopeVocabulary": list(authority.PAUSE_SCOPE_VOCABULARY),
        "rows": sorted(rows, key=lambda row: row["id"]),
    }
    path = root / "config" / "brain-command-authority-matrix.runtime.json"
    _write(path, json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Acceptance: the checked-in matrix classifies every real surface, zero unclassified
# --------------------------------------------------------------------------- #


def test_checked_in_matrix_has_zero_unclassified(real_discovered, real_matrix):
    rows = authority.validate_matrix(real_matrix)
    authority.check_coverage(real_discovered, rows)
    assert len(rows) == len(real_discovered)
    # Every discovered surface is classified (bijection) and nothing is live.
    assert all(row["liveEligibility"] is False for row in rows)
    assert real_matrix["noGlobalAuthorityClaim"] is True


def test_matrix_uses_real_production_entrypoints(real_matrix):
    ids = {row["id"] for row in real_matrix["rows"]}
    # Console script, runnable module, MCP tool, stand-alone script, and the
    # tooling's own verifier -- all real, all present.
    assert "console-script:applypilot" in ids
    assert "console-script:applypilot-fleet-migrate-schema" in ids
    assert "python-module:applypilot.fleet.schema_main" in ids
    assert "mcp-tool:applypilot.fleet.codex_bridge:fleet_status:fleet_status" in ids
    assert "python-module:applypilot" in ids
    assert "python-file:applypilot-webdriver-cost-research/generate_report.py" in ids
    assert "python-file:scripts/verify-brain-command-authority.py" in ids


def test_employer_mutating_surfaces_are_flagged(real_matrix):
    rows = {row["id"]: row for row in real_matrix["rows"]}
    assert rows["console-script:applypilot-fleet-apply"]["employerMutation"] is True
    assert rows["python-module:applypilot.apply.container_worker"]["employerMutation"] is True
    # Read-only inventory/verify tooling never mutates an employer.
    assert rows["python-file:scripts/verify-brain-command-authority.py"]["employerMutation"] is False


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_discovery_is_deterministic(real_discovered):
    again = authority.discover(REPO_ROOT)
    assert authority.canonical_json(real_discovered) == authority.canonical_json(again)


def test_receipt_is_byte_identical_across_runs(real_discovered, real_matrix):
    rows = authority.validate_matrix(real_matrix)
    kwargs = dict(
        head="0" * 40,
        tree="1" * 40,
        matrix=real_matrix,
        discovered=real_discovered,
        rows=rows,
        matrix_path="config/brain-command-authority-matrix.runtime.json",
        clean_state=True,
    )
    first = authority.build_receipt(**kwargs)
    second = authority.build_receipt(**kwargs)
    assert authority.canonical_json(first) == authority.canonical_json(second)


# --------------------------------------------------------------------------- #
# Discovery of new / odd / nested / hyphen-dot entrypoints
# --------------------------------------------------------------------------- #


def test_discovers_new_nested_odd_module(tmp_path):
    _minimal_repo(tmp_path)
    nested = tmp_path / "src" / "demo" / "a" / "b" / "odd_worker_main.py"
    _write(tmp_path / "src" / "demo" / "a" / "__init__.py", "")
    _write(tmp_path / "src" / "demo" / "a" / "b" / "__init__.py", "")
    _write(
        nested,
        'if __name__ == "__main__":\n    print("odd")\n',
    )
    surfaces = {s["id"]: s for s in authority.discover(tmp_path)}
    assert "python-module:demo.a.b.odd_worker_main" in surfaces
    surface = surfaces["python-module:demo.a.b.odd_worker_main"]
    assert surface["sourcePath"] == "src/demo/a/b/odd_worker_main.py"


def test_discovers_new_pyproject_console_script(tmp_path):
    _minimal_repo(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8") + 'demo-extra = "demo.cli:extra"\n',
        encoding="utf-8",
    )
    surfaces = {s["id"]: s for s in authority.discover(tmp_path)}
    assert "console-script:demo-extra" in surfaces
    assert surfaces["console-script:demo-extra"]["canonicalTarget"] == "demo.cli:extra"


def test_alias_forms_python_m_direct_and_command(real_discovered):
    by_id = {s["id"]: s for s in real_discovered}
    module = by_id["python-module:applypilot.fleet.schema_main"]
    # python -m and direct-path forms are both retained and normalized to POSIX.
    assert "python -m applypilot.fleet.schema_main" in module["aliases"]
    assert "python src/applypilot/fleet/schema_main.py" in module["aliases"]
    console = by_id["console-script:applypilot-fleet-migrate-schema"]
    assert console["aliases"] == ["applypilot-fleet-migrate-schema"]


def test_hyphen_and_dot_names_not_dropped(real_discovered):
    ids = {s["id"] for s in real_discovered}
    # A hyphen+dot scripts filename must survive verbatim, not be normalized away.
    assert "python-file:scripts/brain-sqlite-to-postgres.py" in ids


def test_mcp_tool_registry_discovered(real_discovered):
    tools = {s["symbol"] for s in real_discovered if s["surfaceKind"] == "mcp-tool"}
    assert {"fleet_status", "restart_worker", "pause_scope", "quarantine_job"} <= tools


def test_library_modules_are_not_surfaces(real_discovered):
    paths = {s["sourcePath"] for s in real_discovered}
    # Pure libraries (no main guard, no top-level action) are excluded.
    assert "scripts/release_evidence_common.py" not in paths
    assert "scripts/brain_command_authority.py" not in paths


def test_runnable_file_without_main_guard_is_discovered(real_discovered):
    ids = {s["id"] for s in real_discovered}
    # fleet-status.py has top-level side effects but no __main__ guard.
    assert "python-file:fleet-status.py" in ids


def test_complete_structural_fixture_discovers_every_supported_kind(tmp_path):
    _complete_fixture_repo(tmp_path)
    discovered = authority.discover(tmp_path)
    kinds = {surface["surfaceKind"] for surface in discovered}
    assert kinds == set(authority.INCLUDED_SURFACE_KINDS)
    by_id = {surface["id"]: surface for surface in discovered}
    assert by_id["python-module:demo"]["aliases"] == [
        "python -m demo",
        "python src/demo/__main__.py",
    ]
    assert "python-file:tools/nested/generate.report.py" in by_id
    assert any("admin repair" in surface["canonicalTarget"] for surface in discovered)
    assert any("renamed-tool" in surface["canonicalTarget"] for surface in discovered)
    assert any("nested_tool" in surface["canonicalTarget"] for surface in discovered)
    assert any("registered-tool" in surface["canonicalTarget"] for surface in discovered)


# --------------------------------------------------------------------------- #
# Complete scope declaration
# --------------------------------------------------------------------------- #


def test_no_runtime_surface_kind_is_deferred(real_matrix):
    scope = real_matrix["scope"]
    assert scope["includedSurfaceKinds"] == list(authority.INCLUDED_SURFACE_KINDS)
    assert scope["deferredSurfaceKinds"] == []
    assert scope["deferralReason"].strip()
    assert real_matrix["noGlobalAuthorityClaim"] is True


# --------------------------------------------------------------------------- #
# Coverage: missing / stale / mixed-mode drift
# --------------------------------------------------------------------------- #


def test_missing_row_fails_closed(real_discovered, real_matrix):
    rows = [r for r in real_matrix["rows"] if r["id"] != "console-script:applypilot"]
    with pytest.raises(authority.AuthorityError, match="unclassified"):
        authority.check_coverage(real_discovered, rows)


def test_stale_row_fails_closed(real_discovered, real_matrix):
    rows = copy.deepcopy(real_matrix["rows"])
    ghost = copy.deepcopy(rows[0])
    ghost["id"] = "console-script:ghost-removed-entrypoint"
    ghost["canonicalTarget"] = "demo.gone:main"
    ghost["aliases"] = ["ghost-removed-entrypoint"]
    rows.append(ghost)
    with pytest.raises(authority.AuthorityError, match="stale"):
        authority.check_coverage(real_discovered, rows)


def test_mixed_mode_structural_drift_fails_closed(real_discovered, real_matrix):
    rows = copy.deepcopy(real_matrix["rows"])
    for row in rows:
        if row["id"] == "console-script:applypilot-fleet-migrate-schema":
            row["sourcePath"] = "src/applypilot/fleet/other_module.py"
    with pytest.raises(authority.AuthorityError, match="drifted"):
        authority.check_coverage(real_discovered, rows)


# --------------------------------------------------------------------------- #
# Matrix validation: duplicates / shape / vocabulary
# --------------------------------------------------------------------------- #


def test_duplicate_id_fails_closed(real_matrix):
    matrix = _valid_matrix(real_matrix)
    matrix["rows"].append(copy.deepcopy(matrix["rows"][0]))
    with pytest.raises(authority.AuthorityError, match="duplicate"):
        authority.validate_matrix(matrix)


def test_duplicate_alias_fails_closed(real_matrix):
    matrix = _valid_matrix(real_matrix)
    shared = matrix["rows"][0]["aliases"][0]
    matrix["rows"][1]["aliases"] = sorted(set(matrix["rows"][1]["aliases"]) | {shared})
    with pytest.raises(authority.AuthorityError, match="alias"):
        authority.validate_matrix(matrix)


def test_duplicate_target_fails_closed(real_matrix):
    matrix = _valid_matrix(real_matrix)
    donor = matrix["rows"][0]
    victim = matrix["rows"][1]
    victim["surfaceKind"] = donor["surfaceKind"]
    victim["canonicalTarget"] = donor["canonicalTarget"]
    victim["sourcePath"] = donor["sourcePath"]
    with pytest.raises(authority.AuthorityError, match="duplicate canonical targets"):
        authority.validate_matrix(matrix)


def test_extra_row_key_fails_closed(real_matrix):
    matrix = _valid_matrix(real_matrix)
    matrix["rows"][0]["surprise"] = "unexpected"
    with pytest.raises(authority.AuthorityError, match="unexpected or missing keys"):
        authority.validate_matrix(matrix)


def test_missing_row_key_fails_closed(real_matrix):
    matrix = _valid_matrix(real_matrix)
    del matrix["rows"][0]["owner"]
    with pytest.raises(authority.AuthorityError, match="unexpected or missing keys"):
        authority.validate_matrix(matrix)


def test_unknown_read_write_class_fails_closed(real_matrix):
    matrix = _valid_matrix(real_matrix)
    matrix["rows"][0]["readWriteClass"] = "wizardry"
    with pytest.raises(authority.AuthorityError, match="readWriteClass"):
        authority.validate_matrix(matrix)


def test_unsorted_list_fails_closed(real_matrix):
    matrix = _valid_matrix(real_matrix)
    row = next(r for r in matrix["rows"] if len(r["resourcesRead"]) >= 2)
    row["resourcesRead"] = list(reversed(row["resourcesRead"]))
    with pytest.raises(authority.AuthorityError, match="sorted"):
        authority.validate_matrix(matrix)


def test_pause_scope_outside_vocabulary_fails_closed(real_matrix):
    matrix = _valid_matrix(real_matrix)
    matrix["rows"][0]["pauseScopes"] = ["not-a-real-scope"]
    with pytest.raises(authority.AuthorityError, match="pause-scope vocabulary"):
        authority.validate_matrix(matrix)


def test_rows_must_be_sorted(real_matrix):
    matrix = _valid_matrix(real_matrix)
    matrix["rows"] = list(reversed(matrix["rows"]))
    with pytest.raises(authority.AuthorityError, match="sorted by id"):
        authority.validate_matrix(matrix)


def test_extra_top_level_key_fails_closed(real_matrix):
    matrix = _valid_matrix(real_matrix)
    matrix["extra"] = True
    with pytest.raises(authority.AuthorityError, match="top-level keys"):
        authority.validate_matrix(matrix)


# --------------------------------------------------------------------------- #
# Zero-live authority and classification rules
# --------------------------------------------------------------------------- #


def _live_writer_row() -> dict:
    return {
        "id": "console-script:demo-live",
        "repository": "applypilot",
        "sourcePath": "src/demo/live.py",
        "symbol": "main",
        "surfaceKind": "console-script",
        "canonicalTarget": "demo.live:main",
        "aliases": ["demo-live"],
        "installed": True,
        "directInvocation": True,
        "owner": "fleet-worker",
        "readWriteClass": "write",
        "resourcesRead": ["fleet-postgres"],
        "resourcesWritten": ["fleet-postgres"],
        "employerMutation": True,
        "authoritySource": "fleet-apply-role",
        "capabilities": ["lease-apply"],
        "pauseScopes": ["apply"],
        "liveEligibility": True,
        "failClosedBehavior": "checkpoint-required-before-write",
    }


def test_any_live_runtime_authority_fails_closed():
    with pytest.raises(authority.AuthorityError, match="attempts to grant live"):
        authority._validate_authority_rules(_live_writer_row(), 0)


def test_non_live_row_claiming_authority_fails_closed():
    row = _live_writer_row()
    row["liveEligibility"] = False
    row["capabilities"] = []
    # authoritySource still non-none while not live -> reject.
    with pytest.raises(authority.AuthorityError, match="claims an authoritySource"):
        authority._validate_authority_rules(row, 0)


def test_non_live_writer_requires_explicit_fail_closed_rule():
    row = _live_writer_row()
    row["liveEligibility"] = False
    row["authoritySource"] = "none"
    row["capabilities"] = []
    row["failClosedBehavior"] = "deny-not-live"  # missing the explicit writer marker
    with pytest.raises(authority.AuthorityError, match="fail-closed rule is not explicit"):
        authority._validate_authority_rules(row, 0)


def test_unknown_authority_source_fails_closed(real_matrix):
    matrix = _valid_matrix(real_matrix)
    matrix["rows"][0]["authoritySource"] = "invented-role"
    with pytest.raises(authority.AuthorityError, match="authoritySource"):
        authority.validate_matrix(matrix)


def test_structurally_mutating_command_cannot_be_marked_read_only(real_matrix):
    matrix = _valid_matrix(real_matrix)
    row = next(row for row in matrix["rows"] if "rollback" in row["id"])
    row["readWriteClass"] = "read"
    row["resourcesWritten"] = []
    row["failClosedBehavior"] = "deny-not-live"
    with pytest.raises(authority.AuthorityError, match="structurally names a mutation"):
        authority.validate_matrix(matrix)


# --------------------------------------------------------------------------- #
# Forbidden-content scan
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value",
    [
        {"dsn": "postgresql://user@host/db"},
        {"leak": "password=hunter2"},
        {"leak": "dbname=applypilot_fleet"},
        {"path": "C:\\\\Users\\\\alice\\\\secret"},
        {"path": "/home/alice/keys"},
        {"stamp": "2026-07-19T12:34:56"},
        {"token": "api_key=abcdef"},
    ],
)
def test_forbidden_content_scan_fails_closed(value):
    with pytest.raises(authority.AuthorityError):
        authority.scan_forbidden_content(value, "sample")


def test_real_matrix_has_no_forbidden_content(real_matrix):
    authority.scan_forbidden_content(real_matrix, "matrix")  # must not raise


# --------------------------------------------------------------------------- #
# Receipt integrity / forgery
# --------------------------------------------------------------------------- #


def _receipt_repo(root: Path) -> tuple[Path, dict, bytes]:
    _minimal_repo(root)
    matrix_path = _write_fixture_matrix(root)
    _init_git_repo(root)
    receipt = authority.evaluate(root, matrix_path)
    return matrix_path, receipt, authority.canonical_json(receipt)


def _rehash(receipt: dict) -> None:
    body = {key: value for key, value in receipt.items() if key != "selfHash"}
    receipt["selfHash"] = authority.sha256_hex(authority.canonical_json(body))


def test_receipt_self_hash_and_repository_rebinding_verify(tmp_path):
    matrix_path, receipt, raw = _receipt_repo(tmp_path)
    assert authority.verify_receipt(raw, repo_root=tmp_path, matrix_path=matrix_path) == receipt


def test_rehashed_bogus_head_fails_repository_rebinding(tmp_path):
    matrix_path, receipt, _raw = _receipt_repo(tmp_path)
    receipt["head"] = "f" * 40
    _rehash(receipt)
    with pytest.raises(authority.AuthorityError, match="exact current repository state"):
        authority.verify_receipt(
            authority.canonical_json(receipt), repo_root=tmp_path, matrix_path=matrix_path
        )


def test_zero_identity_and_digest_fail_closed_even_when_rehashed(tmp_path):
    matrix_path, receipt, _raw = _receipt_repo(tmp_path)
    receipt["tree"] = "0" * 40
    receipt["matrixDigest"] = "0" * 64
    _rehash(receipt)
    with pytest.raises(authority.AuthorityError, match="resolved tree"):
        authority.verify_receipt(
            authority.canonical_json(receipt), repo_root=tmp_path, matrix_path=matrix_path
        )


def test_receipt_duplicate_key_fails_closed(tmp_path):
    matrix_path, receipt, raw = _receipt_repo(tmp_path)
    duplicate = b'{"head":"' + receipt["head"].encode("ascii") + b'",' + raw[1:]
    with pytest.raises(authority.AuthorityError, match="duplicate object key"):
        authority.verify_receipt(duplicate, repo_root=tmp_path, matrix_path=matrix_path)


def test_receipt_reordered_or_pretty_json_fails_closed(tmp_path):
    matrix_path, receipt, raw = _receipt_repo(tmp_path)
    noncanonical = json.dumps(receipt, indent=2).encode("ascii")
    assert noncanonical != raw
    with pytest.raises(authority.AuthorityError, match="reordered or not canonical"):
        authority.verify_receipt(noncanonical, repo_root=tmp_path, matrix_path=matrix_path)


def test_receipt_rejects_dirty_repository_after_issuance(tmp_path):
    matrix_path, _receipt, raw = _receipt_repo(tmp_path)
    _write(tmp_path / "src" / "demo" / "cli.py", "print('changed')\n")
    with pytest.raises(authority.AuthorityError, match="dirty tree"):
        authority.verify_receipt(raw, repo_root=tmp_path, matrix_path=matrix_path)


def test_receipt_binds_sorted_discovery_digest(real_discovered):
    shuffled = list(reversed(real_discovered))
    assert authority.discovery_digest(real_discovered) == authority.discovery_digest(shuffled)


# --------------------------------------------------------------------------- #
# Dirty tree
# --------------------------------------------------------------------------- #


def test_clean_tree_then_dirty_tree(tmp_path):
    _minimal_repo(tmp_path)
    _init_git_repo(tmp_path)
    assert authority.tracked_tree_is_clean(tmp_path) is True
    authority.assert_clean(tmp_path)  # must not raise
    # Modify a tracked file -> dirty.
    (tmp_path / "src" / "demo" / "cli.py").write_text("x = 1\n", encoding="utf-8")
    assert authority.tracked_tree_is_clean(tmp_path) is False
    with pytest.raises(authority.AuthorityError, match="dirty tree"):
        authority.assert_clean(tmp_path)


def test_untracked_scratch_does_not_dirty_tree(tmp_path):
    _minimal_repo(tmp_path)
    _init_git_repo(tmp_path)
    (tmp_path / ".agent-run").mkdir()
    (tmp_path / ".agent-run" / "scratch.txt").write_text("scratch", encoding="utf-8")
    # Untracked scratch is ignored: the receipt binds committed content.
    assert authority.tracked_tree_is_clean(tmp_path) is True


# --------------------------------------------------------------------------- #
# CLI exit codes
# --------------------------------------------------------------------------- #


def _run_cli(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_inventory_cli_emits_canonical_json():
    result = _run_cli(INVENTORY_CLI, "--repo-root", str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, list) and parsed
    # canonical: compact, sorted -> re-serialization is byte-identical.
    assert authority.canonical_json(parsed).decode("ascii") + "\n" == result.stdout


def test_inventory_cli_check_passes_on_real_matrix():
    result = _run_cli(
        INVENTORY_CLI, "--repo-root", str(REPO_ROOT), "--check", "--matrix", str(MATRIX_PATH)
    )
    assert result.returncode == 0, result.stderr


def test_verify_cli_exits_nonzero_on_bad_matrix(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schemaVersion": "wrong"}), encoding="utf-8")
    result = _run_cli(VERIFY_CLI, "--repo-root", str(REPO_ROOT), "--matrix", str(bad))
    assert result.returncode == 1
    assert "verification failed" in result.stderr


def test_verify_cli_exits_nonzero_on_forged_matrix_row(tmp_path, real_matrix):
    forged = _valid_matrix(real_matrix)
    forged["rows"][0]["liveEligibility"] = True  # incomplete live writer
    forged_path = tmp_path / "forged.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")
    result = _run_cli(VERIFY_CLI, "--repo-root", str(REPO_ROOT), "--matrix", str(forged_path))
    assert result.returncode == 1


@pytest.mark.skipif(
    not authority.tracked_tree_is_clean(REPO_ROOT),
    reason="tracked tree is dirty; end-to-end receipt requires a clean tree",
)
def test_verify_cli_emits_receipt_on_clean_tree():
    result = _run_cli(VERIFY_CLI, "--repo-root", str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    authority.verify_receipt(
        authority.canonical_json(receipt), repo_root=REPO_ROOT, matrix_path=MATRIX_PATH
    )
    assert receipt["counts"]["unclassified"] == 0
    assert receipt["runtimeKind"] == authority.RUNTIME_KIND


@pytest.mark.skipif(
    not authority.tracked_tree_is_clean(REPO_ROOT),
    reason="tracked tree is dirty; end-to-end evaluate requires a clean tree",
)
def test_evaluate_end_to_end_on_real_repo():
    receipt = authority.evaluate(REPO_ROOT, MATRIX_PATH)
    authority.verify_receipt(
        authority.canonical_json(receipt), repo_root=REPO_ROOT, matrix_path=MATRIX_PATH
    )
    assert receipt["head"]
    assert receipt["counts"]["discovered"] == receipt["counts"]["rows"]
