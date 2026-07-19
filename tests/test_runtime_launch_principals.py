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
        "url_encoded_admin_or_shared",  # percent-encoded URL identity (e.g. %70ostgres)
        "role_switch_to_admin_or_shared",
        "set_role_call_admin_or_shared",
        "generic_dsn_env_fallback",  # DATABASE_URL / APPLYPILOT_FLEET_DSN fallback source
        "generic_dsn_help_default",  # help text advertising the retired generic fallbacks
    } <= patterns
    # EVERY non-comment sentinel line is flagged -- no forbidden form silently slips through.
    non_comment_lines = [
        i for i, line in enumerate(text.splitlines(), start=1) if line.strip() and not line.startswith("#")
    ]
    flagged_lines = {f.line for f in findings}
    assert set(non_comment_lines) == flagged_lines, sorted(set(non_comment_lines) - flagged_lines)


@pytest.mark.parametrize(
    "line",
    [
        # mixed case / quoted / spaced keyword principals
        'user = "POSTGRES"',
        "user='fleet_worker'",
        # URL DSN, query-string identity, and percent-encoded identity
        "postgresql://postgres:pw@h/db",
        "postgresql://h/db?user=fleet_worker",
        "postgres://%70ostgres:pw@10.0.0.1/db",
        "postgresql://%66leet_worker:pw@10.0.0.1/db",
        # ambient PGUSER, incl. $env: mixed case
        "$env:PGUSER = 'Postgres'",
        # generic DSN fallback source + help advertisement
        'dsn = os.environ.get("FLEET_PG_DSN") or os.environ.get("APPLYPILOT_FLEET_DSN")',
        'help="default: DATABASE_URL / APPLYPILOT_FLEET_DSN env"',
    ],
)
def test_scanner_flags_extended_forbidden_forms(line):
    assert scanner.scan_text(line), line


@pytest.mark.parametrize(
    "line",
    [
        'if ($MappedRole -in @("postgres", "fleet_worker")) { throw "unique per-node" }',
        'if params.get("user") in {"postgres", "fleet_worker"}: raise SystemExit(1)',
        'user = os.environ["APPLYPILOT_SETUP_ROLE"]  # compared, never selected',
        "  PGUSER  # forbidden ambient variable name in a reject-list",
        'if params.get("user") != expected_role: raise SystemExit(1)',
        # PowerShell param named $PgUser (a script parameter, NOT the ambient libpq PGUSER var).
        '[string]$PgUser = "postgres"',
        # Redaction/scrub read of the retired alias -- the VALUE is passed to a sanitizer, not
        # used as a connection source (no `or`/`default=` fallback idiom).
        '_sanitize(text, os.environ.get("APPLYPILOT_FLEET_DSN", ""))',
        # A reject-list membership tuple naming the forbidden ambient vars.
        'FORBIDDEN_DATABASE_ENV_VARS = ("APPLYPILOT_FLEET_DSN", "DATABASE_URL")',
        # Aliasing the validated FLEET_PG_DSN under the legacy name (a write, not a source).
        "$env:APPLYPILOT_FLEET_DSN = $env:FLEET_PG_DSN",
    ],
)
def test_scanner_ignores_legitimate_rejection_guards(line):
    # These reference the admin/shared role names to REJECT them, redact them, or alias the
    # validated DSN; none SELECTS the principal via a `user=`/`PGUSER=`/URL/role-switch form
    # or sources a DSN from the generic fallback, so none may be flagged.
    assert scanner.scan_text(line) == []


def test_discovery_includes_python_dash_m_targets_and_their_imports():
    surfaces = set(scanner.discover_launch_surfaces(REPO))
    # `python -m applypilot.fleet.console_app` (run-fleet-console.ps1) and
    # `python -m applypilot.fleet.apply_home_main` (run-fleet-crash-liveness.ps1) are reached
    # via -m, not a console entry, yet are audited -- with no hand-maintained module allowlist.
    assert "src/applypilot/fleet/console_app.py" in surfaces
    assert "src/applypilot/fleet/apply_home_main.py" in surfaces


def test_python_dash_m_module_extraction_is_deterministic_and_first_party_only():
    text = (
        "& $py -m applypilot.fleet.autotriage_main @margs\n"
        "python -m applypilot.cli supervise-apply\n"
        "python -m pytest tests/\n"  # third-party -m target must be ignored
    )
    modules = scanner._python_dash_m_modules(text)
    assert modules == [
        "src/applypilot/fleet/autotriage_main.py",
        "src/applypilot/cli.py",
    ]


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


# --------------------------------------------------------------------------------------------
# Adversarial: generated wrappers / task actions / shortcuts NEVER embed the operator DSN.
# --------------------------------------------------------------------------------------------
def test_otp_startup_registration_never_embeds_the_dsn_and_fails_closed():
    text = (REPO / "register-otp-responder-startup.ps1").read_text(encoding="utf-8")
    # The generated task action + startup shortcut arguments must not carry the secret DSN.
    assert "-Dsn `\"$Dsn`\"" not in text
    assert "-Dsn $Dsn" not in text
    assert "$Dsn" not in text  # the whole -Dsn parameter is gone; nothing to embed
    # Both generated actions launch the responder with -Supervise only (runtime env channel).
    assert text.count('-File `"$Launcher`" -Supervise') == 2
    # Fail closed before registration if the secure channel (FLEET_PG_DSN) is absent.
    assert "if (-not $env:FLEET_PG_DSN)" in text
    assert "is never embedded in the task/shortcut" in text


@pytest.mark.parametrize(
    "rel,embed_tokens",
    [
        ("register-fleet-tasks.ps1", ["'$effectiveDsn'"]),
        ("register-apply-cycle.ps1", ["'$effectiveDsn'"]),
        ("fleet-daily-unguard.ps1", ["'$qDsn'", "-Dsn '$qDsn'"]),
    ],
)
def test_generated_task_wrappers_never_embed_the_dsn_literal(rel, embed_tokens):
    text = (REPO / rel).read_text(encoding="utf-8")
    # A generated wrapper here-string that sets FLEET_PG_DSN to an embedded operator DSN literal
    # (fragile single-quote quoting that an apostrophe/newline/`;` payload could break or inject,
    # and a persistent secret copy on disk) must not exist. The wrapper references the inherited
    # FLEET_PG_DSN runtime channel instead.
    assert "$env:FLEET_PG_DSN = '$effectiveDsn'" not in text
    assert "$env:APPLYPILOT_FLEET_DSN = '$effectiveDsn'" not in text
    for token in embed_tokens:
        assert token not in text, token
    # The de-embedded wrapper documents the inherited channel.
    assert "inherited from the persisted User environment" in text
