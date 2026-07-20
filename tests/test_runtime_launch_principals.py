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
        "load-canary-home.ps1",  # the primary spend-cap loader must be classified/audited
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
        "raw_psql_fleet_config_write",  # raw psql mutation of fleet_config (unapproved DB channel)
        "powershell_generic_dsn_env_source",  # bare $env:DATABASE_URL/$env:APPLYPILOT_FLEET_DSN source
        "database_client_admin_parameter_flow",  # admin/default variable reaching psql/pg_dump
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
        "postgresql://host/db?%75ser=%70ostgres",
        "postgresql://host/db?user=%66leet_worker",
        "dsn = os.environ['DATABASE_URL']",
        'dsn = os.getenv("APPLYPILOT_FLEET_DSN")',
        "$dsn = $env:DATABASE_URL",
        '$dsn = [Environment]::GetEnvironmentVariable("DATABASE_URL", "User")',
        'Set-Item Env:PGUSER "postgres"',
        "export DSN=${DATABASE_URL:-}",
        "export DSN=$APPLYPILOT_FLEET_DSN",
        "options=-c%20role%3Dpostgres",
        "options='-c session_authorization=random_superuser'",
    ],
)
def test_scanner_flags_encoded_query_env_and_role_option_bypasses(line):
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


@pytest.mark.parametrize(
    "line",
    [
        # The exact retired load-canary-home write, and its INSERT/DELETE/psql-path variants.
        '& $psql --dbname=$Dsn -v ON_ERROR_STOP=1 -c "UPDATE fleet_config SET spend_cap_usd=0 WHERE id=1;"',
        'psql "$env:FLEET_PG_DSN" -c "INSERT INTO fleet_config (id) VALUES (1)"',
        'psql $Dsn -c "DELETE FROM fleet_config WHERE id=1"',
        r'& $psql --dbname=$Dsn -c "TRUNCATE fleet_config"',
    ],
)
def test_scanner_flags_reintroduced_raw_psql_fleet_config_write(line):
    findings = {f.pattern for f in scanner.scan_text(line)}
    assert "raw_psql_fleet_config_write" in findings, line


@pytest.mark.parametrize(
    "line",
    [
        # Reads of fleet_config and writes to OTHER tables are not the unapproved-mutation channel.
        'psql "$env:FLEET_PG_DSN" -c "SELECT doctor_last_pass_at FROM fleet_config WHERE id=1;"',
        'psql $effectiveDsn -c "UPDATE fleet_desired_state SET desired_workers=1 WHERE machine_owner=' + "'m2'\"",
    ],
)
def test_scanner_ignores_psql_reads_and_other_table_writes(line):
    assert "raw_psql_fleet_config_write" not in {f.pattern for f in scanner.scan_text(line)}, line


@pytest.mark.parametrize(
    "line",
    [
        # Bare generic-DSN env used as a value/fallback source (coalesce, ternary, scriptblock).
        "$Dsn = if ($env:FLEET_PG_DSN) { $env:FLEET_PG_DSN } else { $env:APPLYPILOT_FLEET_DSN }",
        "$Dsn = $env:DATABASE_URL",
        "Start-Worker -Dsn ($env:APPLYPILOT_FLEET_DSN)",
    ],
)
def test_scanner_flags_powershell_generic_dsn_env_source(line):
    assert "powershell_generic_dsn_env_source" in {f.pattern for f in scanner.scan_text(line)}, line


@pytest.mark.parametrize(
    "line",
    [
        "$Dsn = ${env:DATABASE_URL}",
        "Start-Worker -Dsn (${env:APPLYPILOT_FLEET_DSN})",
    ],
)
def test_scanner_flags_braced_powershell_generic_dsn_env_source(line):
    assert "powershell_generic_dsn_env_source" in {f.pattern for f in scanner.scan_text(line)}, line


@pytest.mark.parametrize(
    "line",
    [
        # Aliasing the validated FLEET_PG_DSN under the legacy name is a WRITE TARGET, not a source.
        "$env:APPLYPILOT_FLEET_DSN = $env:FLEET_PG_DSN",
        "$env:APPLYPILOT_FLEET_DSN = $Dsn",
        "$env:DATABASE_URL=$foo",
        "${env:DATABASE_URL} = $foo",
    ],
)
def test_scanner_ignores_generic_dsn_env_alias_write_target(line):
    assert "powershell_generic_dsn_env_source" not in {f.pattern for f in scanner.scan_text(line)}, line


@pytest.mark.parametrize(
    "text",
    [
        '& $psql --dbname=$Dsn -c @"\nUPDATE fleet_config\nSET spend_cap_usd=0 WHERE id=1;\n"@',
        '& $psql `\n  --dbname=$Dsn `\n  -c @"\nUPDATE fleet_config SET spend_cap_usd=0 WHERE id=1\n"@',
        '$query = "UPDATE fleet_config SET spend_cap_usd=0 WHERE id=1"\n& $psql -c $query',
        '$query = "UPDATE fleet_config SET spend_cap_usd=0 WHERE id=1"\n& $psql `\n  --dbname=$Dsn `\n  -c $query',
        "$query = @'\nDELETE FROM fleet_config WHERE id=1\n'@\n& $psql --command $query",
    ],
)
def test_scanner_flags_multiline_or_indirected_raw_psql_fleet_config_write(text):
    assert "raw_psql_fleet_config_write" in {f.pattern for f in scanner.scan_text(text)}, text


@pytest.mark.parametrize(
    "text",
    [
        'param([string]$PgUser = "postgres")\n& $pgDump -U $PgUser -d applypilot_fleet',
        "param(\n [string]$DatabaseLogin = 'fleet_worker'\n)\n& $psql `\n --username $DatabaseLogin `\n -c 'SELECT 1'",
        '[string]$PgUser = "postgres"\n$selected = $PgUser\n& $pgDump --username=$selected -Fc',
        '$admin = "POSTGRES"\n$alias = "${admin}"\n& pg_dump.exe -U ${alias} -Fc',
        '& pg_dump -U postgres -d applypilot_fleet',
    ],
)
def test_scanner_flags_admin_parameter_and_variable_dataflow_into_database_clients(text):
    assert "database_client_admin_parameter_flow" in {
        finding.pattern for finding in scanner.scan_text(text)
    }, text


@pytest.mark.parametrize(
    "text",
    [
        "PGDUMP_USER=postgres\npg_dump -U $PGDUMP_USER -d applypilot_fleet",
        "export BACKUP_USER=fleet_worker\npg_dump --username=${BACKUP_USER} -Fc",
        "ADMIN_USER=postgres\nDUMP_USER=$ADMIN_USER\npg_dump --username=$DUMP_USER -Fc",
        "export ADMIN_USER=fleet_worker\nexport DUMP_USER=${ADMIN_USER}\npg_dump -U ${DUMP_USER} -Fc",
    ],
)
def test_scanner_flags_shell_admin_assignment_aliases_and_database_client_sink(text):
    findings = [
        finding
        for finding in scanner.scan_text(text)
        if finding.pattern == "database_client_admin_parameter_flow"
    ]
    assert {finding.line for finding in findings} == set(range(1, len(text.splitlines()) + 1))


@pytest.mark.parametrize("operator", ["-", ":-", "=", ":=", "+", ":+"])
@pytest.mark.parametrize("principal", ["postgres", "fleet_worker"])
@pytest.mark.parametrize("word_quote", ["", "'", '"'])
def test_scanner_flags_shell_admin_parameter_expansion_aliases_at_origin_and_sink(
    operator, principal, word_quote
):
    text = (
        f"ADMIN_USER=${{PGUSER{operator}{word_quote}{principal}{word_quote}}}\n"
        "DUMP_USER=$ADMIN_USER\n"
        "pg_dump -U $DUMP_USER -d applypilot_fleet"
    )
    findings = [
        finding
        for finding in scanner.scan_text(text)
        if finding.pattern == "database_client_admin_parameter_flow"
    ]
    assert {finding.line for finding in findings} == {1, 2, 3}


@pytest.mark.parametrize("operator", ["-", ":-", "=", ":=", "+", ":+"])
@pytest.mark.parametrize("principal", ["postgres", "fleet_worker"])
@pytest.mark.parametrize("word_quote", ["", "'", '"'])
@pytest.mark.parametrize("client_argument", ["-U {}", "--username={}"])
def test_scanner_flags_direct_shell_admin_parameter_expansion_database_client_sink(
    operator, principal, word_quote, client_argument
):
    expansion = f"${{PGUSER{operator}{word_quote}{principal}{word_quote}}}"
    text = f"pg_dump {client_argument.format(expansion)} -Fc"
    assert "database_client_admin_parameter_flow" in {
        finding.pattern for finding in scanner.scan_text(text)
    }


@pytest.mark.parametrize("operator", ["-", ":-", "=", ":=", "+", ":+"])
@pytest.mark.parametrize("principal", ["postgres", "fleet_worker"])
@pytest.mark.parametrize("word_quote", ["", "'", '"'])
def test_scanner_flags_ambient_pguser_admin_parameter_expansion_without_explicit_user_flag(
    operator, principal, word_quote
):
    line = f"PGUSER=${{SOURCE_USER{operator}{word_quote}{principal}{word_quote}}}"
    assert "pguser_admin_or_shared" in {
        finding.pattern for finding in scanner.scan_text(line)
    }


@pytest.mark.parametrize(
    "text",
    [
        "PGUSER=${PGUSER:?PGUSER is required}\npg_dump -U $PGUSER -Fc",
        "DUMP_USER=${PGUSER:-fleet_backup}\npg_dump -U $DUMP_USER -Fc",
        "export BACKUP_USER=${PGUSER:+fleet_backup}\npsql -U $BACKUP_USER -c 'SELECT 1'",
    ],
)
def test_scanner_allows_fail_closed_or_least_privilege_shell_parameter_expansions(text):
    assert "database_client_admin_parameter_flow" not in {
        finding.pattern for finding in scanner.scan_text(text)
    }


@pytest.mark.parametrize("operator", ["-", ":-", "=", ":=", "+", ":+"])
@pytest.mark.parametrize("word_quote", ["", "'", '"'])
@pytest.mark.parametrize("principal", ["fleet_backup", "postgresql", "fleet_worker_ro"])
def test_scanner_does_not_broaden_parameter_expansion_principal_matches(
    operator, word_quote, principal
):
    expansion = f"${{PGUSER{operator}{word_quote}{principal}{word_quote}}}"
    text = f"PGUSER={expansion}\npg_dump --username={expansion} -Fc"
    patterns = {finding.pattern for finding in scanner.scan_text(text)}
    assert "pguser_admin_or_shared" not in patterns
    assert "database_client_admin_parameter_flow" not in patterns


def test_scanner_allows_explicit_non_admin_backup_service_channel():
    text = (
        'param([string]$ExpectedPgPrincipal)\n'
        '$serviceReference = "service=$PgService"\n'
        '& $psql --dbname=$serviceReference -c "SELECT current_user"\n'
        '& $pgDump --dbname=$serviceReference -Fc -f $dest\n'
    )
    assert "database_client_admin_parameter_flow" not in {
        finding.pattern for finding in scanner.scan_text(text)
    }


def test_scanner_does_not_join_unrelated_psql_read_and_later_safe_python_text():
    text = (
        '& $psql --dbname=$Dsn -c "SELECT spend_cap_usd FROM fleet_config"\n'
        'print("UPDATE fleet_config appears only in a later non-psql diagnostic")\n'
    )
    assert "raw_psql_fleet_config_write" not in {f.pattern for f in scanner.scan_text(text)}


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
    # Validate the persisted future channel, not a transient process-only environment value.
    assert "Assert-ApplyPilotRuntimeChannel" in text
    assert "-RequirePersistedUserChannel" in text
    assert "Get-ScheduledTask" in text and "Remove-Item -LiteralPath $legacyShortcut" in text


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


def test_otp_child_processes_inherit_dsn_without_secret_command_line():
    text = (REPO / "run-otp-responder.ps1").read_text(encoding="utf-8")
    invocation_lines = [
        line for line in text.splitlines()
        if "& $Exe" in line or "ArgumentList" in line
    ]
    assert invocation_lines
    assert all("--dsn" not in line.casefold() for line in invocation_lines)
    assert "--machine-owner" in "\n".join(invocation_lines)


def test_discovery_follows_transitive_from_package_imports(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "src/applypilot/pkg").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project.scripts]\nentry = "applypilot.entry:main"\n', encoding="utf-8"
    )
    (tmp_path / "src/applypilot/__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src/applypilot/entry.py").write_text(
        "from applypilot.pkg import second\n", encoding="utf-8"
    )
    (tmp_path / "src/applypilot/pkg/__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src/applypilot/pkg/second.py").write_text(
        "from applypilot.pkg import third\n", encoding="utf-8"
    )
    (tmp_path / "src/applypilot/pkg/third.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)

    surfaces = set(scanner.discover_launch_surfaces(tmp_path))

    assert "src/applypilot/entry.py" in surfaces
    assert "src/applypilot/pkg/second.py" in surfaces
    assert "src/applypilot/pkg/third.py" in surfaces


def test_discovery_follows_transitive_relative_imports(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "src/applypilot/pkg").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project.scripts]\nentry = "applypilot.entry:main"\n', encoding="utf-8"
    )
    (tmp_path / "src/applypilot/__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src/applypilot/entry.py").write_text(
        "from .pkg import second\n", encoding="utf-8"
    )
    (tmp_path / "src/applypilot/pkg/__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src/applypilot/pkg/second.py").write_text(
        "from . import third\n", encoding="utf-8"
    )
    (tmp_path / "src/applypilot/pkg/third.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)

    surfaces = set(scanner.discover_launch_surfaces(tmp_path))

    assert "src/applypilot/pkg/second.py" in surfaces
    assert "src/applypilot/pkg/third.py" in surfaces


@pytest.mark.parametrize(
    "rel",
    [
        "register-fleet-tasks.ps1",
        "register-apply-cycle.ps1",
        "fleet-daily-unguard.ps1",
        "register-otp-responder-startup.ps1",
    ],
)
def test_registrars_validate_persisted_future_channel_before_side_effects(rel):
    text = (REPO / rel).read_text(encoding="utf-8")
    validation = text.index("Assert-ApplyPilotRuntimeChannel")
    effect_positions = [
        text.find(marker)
        for marker in ("New-Item -ItemType Directory", "Set-Content", "Register-ScheduledTask")
        if text.find(marker) >= 0
    ]
    assert effect_positions
    assert validation < min(effect_positions)
    assert "RequirePersistedUserChannel" in text


def test_registration_rollout_scrubs_legacy_artifacts_and_writes_post_evidence():
    fleet = (REPO / "register-fleet-tasks.ps1").read_text(encoding="utf-8")
    cadence = (REPO / "register-apply-cycle.ps1").read_text(encoding="utf-8")
    otp = (REPO / "register-otp-responder-startup.ps1").read_text(encoding="utf-8")
    daily = (REPO / "fleet-daily-unguard.ps1").read_text(encoding="utf-8")
    for text in (fleet, cadence, otp, daily):
        assert "Write-ApplyPilotLaunchRolloutEvidence" in text
    assert "Remove-Item -LiteralPath" in fleet
    assert "Set-Content -Path $path" in cadence
    assert "Remove-Item -LiteralPath" in otp
    assert "Remove-Item -LiteralPath" in daily
    assert "Unregister-ScheduledTask" in fleet
    assert "Unregister-ScheduledTask" in cadence
    assert "Unregister-ScheduledTask" in otp
    assert "Unregister-ScheduledTask" in daily
