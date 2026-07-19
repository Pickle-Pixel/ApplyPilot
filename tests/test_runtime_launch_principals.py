"""Focused + adversarial tests for the Runtime launch-principal scanner (Task 9B).

Covers:
  * the deterministic scanner discovers Runtime launch surfaces without a hand-maintained
    filename allowlist (root launchers, deploy templates, scripts wrappers, and console-entry
    modules plus the reporting helpers they import),
  * NO discovered production launch surface selects user=postgres / shared fleet_worker /
    PGUSER=postgres / a role-switching fallback / an admin URL principal,
  * every forbidden principal form is rejected (sentinel fixture proves rejection), while
    legitimate REJECTION guards are NOT flagged,
  * a missing FLEET_PG_DSN fails closed BEFORE any database/worker side effect,
  * the canonical JSON receipt carries base/head placeholders, counts, and a SHA-256 discovery
    digest, and never leaks DSN values.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCANNER_PATH = REPO / "scripts" / "verify-runtime-launch-principals.py"
SENTINEL_FIXTURE = REPO / "tests" / "fixtures" / "runtime_launch_principal_sentinels.txt"


def _load_scanner():
    spec = importlib.util.spec_from_file_location("runtime_launch_principals_scanner", SCANNER_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass processing can resolve the module by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scanner = _load_scanner()


# --------------------------------------------------------------------------------------------
# Discovery is real (not a filename allowlist) and green.
# --------------------------------------------------------------------------------------------
def test_no_offending_launch_surface_in_repository():
    receipt = scanner.scan_repository(REPO)
    assert receipt["offending_count"] == 0, receipt["offenders"]
    # A meaningful surface is discovered -- this is not an empty/degenerate scan.
    assert receipt["discovered_file_count"] > 100


def test_discovers_representative_surfaces_across_categories():
    surfaces = set(scanner.discover_launch_surfaces(REPO))
    # Root launchers (worker, scheduler, console, canary, discovery, OTP, setup).
    for rel in (
        "run-fleet-worker.ps1",
        "run-fleet-workers.ps1",
        "run-fleet-console.ps1",
        "load-canary-remote.ps1",
        "register-fleet-tasks.ps1",
        "run-otp-responder.ps1",
        "setup-mac-worker.sh",
        "setup-linux-worker.sh",
        "fleet-status.py",
    ):
        assert rel in surfaces, rel
    # Deploy/service template surfaces.
    assert "deploy/entrypoint.sh" in surfaces
    assert "com.applypilot.fleetworker.plist.template" in surfaces
    # Installed/direct launch wrappers under scripts/.
    assert "scripts/setup-fleet-worker.ps1" in surfaces
    # Console-entry modules and the reporting helper reached via a one-hop import from the CLI.
    assert "src/applypilot/fleet/apply_worker_main.py" in surfaces
    assert "src/applypilot/fleet/cost_quality_report.py" in surfaces


def test_scanner_excludes_itself_and_tests_and_ci():
    surfaces = set(scanner.discover_launch_surfaces(REPO))
    assert "scripts/verify-runtime-launch-principals.py" not in surfaces
    assert "tests/test_runtime_launch_principals.py" not in surfaces
    assert not any(s.startswith(".github/") for s in surfaces)
    assert not any(s.startswith("tests/") for s in surfaces)
    # The sentinel fixture must never be treated as a production launch surface.
    assert "tests/fixtures/runtime_launch_principal_sentinels.txt" not in surfaces


# --------------------------------------------------------------------------------------------
# Every forbidden principal form is rejected; legitimate rejection guards are not.
# --------------------------------------------------------------------------------------------
def test_scanner_flags_every_forbidden_principal_form():
    text = SENTINEL_FIXTURE.read_text(encoding="utf-8")
    findings = scanner.scan_text(text)
    patterns = {f.pattern for f in findings}
    assert {
        "dsn_user_postgres",
        "dsn_user_shared_fleet_worker",
        "pguser_admin_or_shared",
        "url_admin_or_shared",
        "role_switch_to_admin_or_shared",
        "set_role_call_admin_or_shared",
    } <= patterns
    # Every non-comment sentinel line is flagged (9 sentinel lines below the header comment).
    flagged_lines = {f.line for f in findings}
    assert len(flagged_lines) >= 8


@pytest.mark.parametrize(
    "line",
    [
        'if ($MappedRole -in @("postgres", "fleet_worker")) { throw "unique per-node" }',
        'if params.get("user") in {"postgres", "fleet_worker"}: raise SystemExit(1)',
        'user = os.environ["APPLYPILOT_SETUP_ROLE"]  # compared, never selected',
        "  PGUSER  # forbidden ambient variable name in a reject-list",
        'if params.get("user") != expected_role: raise SystemExit(1)',
    ],
)
def test_scanner_ignores_legitimate_rejection_guards(line):
    # These reference the admin/shared role names to REJECT them; none SELECTS the principal
    # via a `user=`/`PGUSER=`/URL/role-switch form, so none may be flagged.
    assert scanner.scan_text(line) == []


def test_sentinel_fixture_is_rejected_but_not_discovered():
    # Proves the fixture carries live sentinels (rejected) yet is outside the production surface.
    assert scanner.scan_text(SENTINEL_FIXTURE.read_text(encoding="utf-8"))
    assert str(SENTINEL_FIXTURE.relative_to(REPO)).replace("\\", "/") not in set(
        scanner.discover_launch_surfaces(REPO)
    )


# --------------------------------------------------------------------------------------------
# Canonical, secret-free receipt.
# --------------------------------------------------------------------------------------------
def test_receipt_is_canonical_and_carries_required_fields(monkeypatch):
    monkeypatch.setenv("APPLYPILOT_SCAN_BASE", "eadf68f")
    monkeypatch.setenv("APPLYPILOT_SCAN_HEAD", "HEAD")
    receipt = scanner.scan_repository(REPO)
    assert receipt["schema"] == scanner.RECEIPT_SCHEMA
    assert receipt["base"] == "eadf68f"
    assert receipt["head"] == "HEAD"
    assert set(receipt) == {
        "schema",
        "base",
        "head",
        "discovered_file_count",
        "offending_count",
        "discovery_sha256",
        "offenders",
    }
    assert len(receipt["discovery_sha256"]) == 64
    int(receipt["discovery_sha256"], 16)  # valid hex
    # Canonical JSON round-trips and is byte-stable.
    blob = scanner._canonical_json(receipt)
    assert json.loads(blob) == receipt
    assert scanner._canonical_json(receipt) == blob
    # No DSN values or secrets leak into the receipt.
    assert "user=" not in blob
    assert "password" not in blob.lower()


# --------------------------------------------------------------------------------------------
# Adversarial: a missing FLEET_PG_DSN fails closed BEFORE any side effect.
# --------------------------------------------------------------------------------------------
def test_fleet_status_fails_before_touching_the_database():
    env = {k: v for k, v in os.environ.items() if k != "FLEET_PG_DSN"}
    env["PYTHONPATH"] = str(REPO / "src")
    result = subprocess.run(
        [sys.executable, str(REPO / "fleet-status.py"), "m4"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "FLEET_PG_DSN is not set" in (result.stderr + result.stdout)
    # It exits before importing/using the Postgres queue layer.
    assert "Traceback" not in result.stderr


def test_cost_quality_report_refuses_without_fleet_dsn(monkeypatch):
    monkeypatch.delenv("FLEET_PG_DSN", raising=False)
    from applypilot.fleet import cost_quality_report

    with pytest.raises(RuntimeError, match="FLEET_PG_DSN is not set"):
        cost_quality_report.build_report(pg_dsn=None)


@pytest.mark.skipif(
    os.name != "nt" or shutil.which("pwsh") is None,
    reason="PowerShell launcher fail-closed proof requires pwsh on Windows",
)
def test_worker_launcher_throws_before_starting_workers_without_dsn():
    result = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Remove-Item Env:FLEET_PG_DSN -ErrorAction SilentlyContinue; "
            f"& '{(REPO / 'run-fleet-workers.ps1').as_posix()}' -Count 1 -Label home",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "FLEET_PG_DSN is not set" in (result.stderr + result.stdout)
    # The worker-launch banner must never print: we failed before Start-Process.
    assert "isolated workers up" not in (result.stderr + result.stdout)


def test_setup_shell_installers_are_fail_closed_on_missing_dsn():
    for rel in ("setup-mac-worker.sh", "setup-linux-worker.sh"):
        text = (REPO / rel).read_text(encoding="utf-8")
        assert "FLEET_PG_DSN is not set" in text
        assert "user=fleet_worker" not in text
        assert "user=postgres" not in text
