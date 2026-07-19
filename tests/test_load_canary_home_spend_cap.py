"""Repair contract for load-canary-home.ps1 (authenticated P2).

The primary spend-cap loader used to (1) accept FLEET_PG_DSN/-Dsn behind only an
insufficient regex denylist and (2) perform a raw ``psql --dbname=$Dsn`` UPDATE of
fleet_config that could execute as the ambient (postgres) login and whose exit code was
discarded. This suite proves the closure:

  * the loader now gates every side effect behind the SAME approved runtime-launch channel
    + full offline conninfo/principal validator (Assert-ApplyPilotRuntimeChannel ->
    launch_guard -> pgqueue.assert_mapped_principal) that supported launchers use, with no
    standalone denylist or weaker parser;
  * the raw ``psql`` fleet_config mutation is gone; the spend cap is written through the
    audited apply-home CLI subcommand ``set-spend-cap`` (central pgqueue mapped-principal
    connection, parameterized statement in one transaction, audit row, exit code propagated);
  * hostile principal forms (URL postgres, percent-encoded postgres, query user, options
    role / session_authorization, keyword admin, missing principal) are rejected by that
    shared validator while a least-privilege mapped principal is accepted;
  * ordering is enforced -- a failed validation/update lets NO later step run (offline mocks).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from applypilot.apply import pgqueue


REPO = Path(__file__).resolve().parents[1]
LOADER = REPO / "load-canary-home.ps1"
_SECRET = "s3cr3t-canary-home-pw"
_MAPPED = "node_apply_home"


def _load_scanner():
    spec = importlib.util.spec_from_file_location(
        "runtime_launch_principals_scanner_cc", REPO / "scripts" / "verify-runtime-launch-principals.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------------
# Static contract: validated channel first, no weak denylist / raw psql, audited CLI + guards.
# --------------------------------------------------------------------------------------------
def test_loader_validates_approved_channel_before_any_side_effect():
    text = LOADER.read_text(encoding="utf-8")
    # Uses the shared runtime-launch channel + offline validator, not a local parser.
    assert ". (Join-Path $Root \"scripts\\runtime-launch-channel.ps1\")" in text
    assert "Assert-ApplyPilotRuntimeChannel" in text
    # The old insufficient regex denylist is gone.
    assert "-match '(?i)\\buser" not in text
    assert "selects an admin/shared principal" not in text
    # Validation happens before the first CLI side effect.
    validation = text.index("Assert-ApplyPilotRuntimeChannel")
    first_side_effect = text.index("Invoke-HomeCli")
    assert validation < first_side_effect


def test_loader_has_no_raw_psql_and_routes_spend_cap_through_audited_cli():
    text = LOADER.read_text(encoding="utf-8")
    # No raw psql channel remains (no discovery, no invocation, no inline fleet_config write).
    assert "psql.exe" not in text
    assert "$psql" not in text
    assert "UPDATE fleet_config" not in text
    # The spend cap is set through the mapped control-plane CLI subcommand.
    assert '"set-spend-cap"' in text
    # The scanner classifies + clears this launcher and would flag any reintroduced raw channel.
    scanner = _load_scanner()
    assert "load-canary-home.ps1" in set(scanner.discover_launch_surfaces(REPO))
    findings = scanner.scan_file(REPO, "load-canary-home.ps1")
    assert findings == [], findings


def test_loader_propagates_native_exit_for_every_cli_step():
    text = LOADER.read_text(encoding="utf-8")
    # Every apply-home CLI step goes through the guarded wrapper (checks $LASTEXITCODE, throws).
    assert "if ($LASTEXITCODE -ne 0)" in text
    for step in ("pull", "push", "canary", "approve", "set-spend-cap"):
        assert f'"{step}"' in text, step
    # The guarded wrapper runs the spend-cap update BEFORE launching workers, so a failed cap
    # write aborts before run-fleet-workers.ps1.
    assert text.index('"set-spend-cap"') < text.index("run-fleet-workers.ps1")


# --------------------------------------------------------------------------------------------
# The audited control-plane subcommand: parameterized, transactional, audited, fail-closed.
# --------------------------------------------------------------------------------------------
class _FakeCursor:
    def __init__(self, rowcount=1):
        self.rowcount = rowcount
        self.calls: list[tuple[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))


class _FakeConn:
    def __init__(self, rowcount=1):
        self._cur = _FakeCursor(rowcount=rowcount)
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self._cur

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def test_set_spend_cap_is_parameterized_transactional_and_audited():
    from applypilot.fleet import apply_home_main as hm

    conn = _FakeConn(rowcount=1)
    result = hm.set_spend_cap(conn, 25.0)
    assert result == {"spend_cap_usd": 25.0}
    assert conn.committed and not conn.rolled_back
    update = conn._cur.calls[0]
    audit = conn._cur.calls[1]
    # The value is bound as a parameter -- never interpolated into the SQL string.
    assert "spend_cap_usd=%s" in update[0]
    assert update[1] == (25.0,)
    assert "25.0" not in update[0]
    # A durable audit row records the mapped control-plane change.
    assert "INSERT INTO fleet_console_audit" in audit[0]
    assert audit[1][0] == "set_spend_cap"


def test_set_spend_cap_fails_closed_when_config_row_missing():
    """A least-privilege role whose write lands nowhere must raise (CLI exits non-zero)."""
    from applypilot.fleet import apply_home_main as hm

    conn = _FakeConn(rowcount=0)
    with pytest.raises(RuntimeError, match="fleet_config row id=1 is missing"):
        hm.set_spend_cap(conn, 25.0)
    assert conn.rolled_back and not conn.committed


def test_set_spend_cap_rejects_negative_cap():
    from applypilot.fleet import apply_home_main as hm

    with pytest.raises(ValueError, match="non-negative"):
        hm.set_spend_cap(_FakeConn(), -1.0)


def test_set_spend_cap_writes_and_audits_against_disposable_db(fleet_db):
    from applypilot.fleet import apply_home_main as hm

    with pgqueue.connect(fleet_db) as conn:
        out = hm.set_spend_cap(conn, 17.5)
    assert out == {"spend_cap_usd": 17.5}
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("SELECT spend_cap_usd FROM fleet_config WHERE id=1")
        assert float(cur.fetchone()["spend_cap_usd"]) == 17.5
        cur.execute("SELECT COUNT(*) AS n FROM fleet_console_audit WHERE action='set_spend_cap'")
        assert cur.fetchone()["n"] == 1
        conn.rollback()


# --------------------------------------------------------------------------------------------
# Hostile principal forms rejected by the shared validator the loader now uses.
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dsn",
    [
        # keyword admin / shared
        f"host=node.tail.ts.net dbname=fleet user=postgres password={_SECRET} sslmode=require",
        f"host=node.tail.ts.net dbname=fleet user=fleet_worker password={_SECRET} sslmode=require",
        # URL postgres
        f"postgresql://postgres:{_SECRET}@node.tail.ts.net/db?sslmode=require",
        # percent-encoded postgres
        f"postgresql://%70ostgres:{_SECRET}@node.tail.ts.net/db?sslmode=require",
        # query-string user identity
        f"postgresql://node.tail.ts.net/db?user=postgres&password={_SECRET}&sslmode=require",
        # options role / session_authorization role switch
        (
            f"host=node.tail.ts.net dbname=fleet user={_MAPPED} password={_SECRET} "
            "sslmode=require options='-c role=postgres'"
        ),
        (
            f"host=node.tail.ts.net dbname=fleet user={_MAPPED} password={_SECRET} "
            "sslmode=require options='-c session_authorization=postgres'"
        ),
        # missing explicit principal
        f"host=node.tail.ts.net dbname=fleet password={_SECRET} sslmode=require",
    ],
)
def test_shared_validator_rejects_hostile_principal_forms(dsn):
    with pytest.raises(RuntimeError) as exc:
        pgqueue.assert_mapped_principal(dsn, expected_principal=_MAPPED)
    assert _SECRET not in str(exc.value)  # never leaks the secret


def test_shared_validator_accepts_least_privilege_mapped_principal():
    dsn = (
        f"host=node.tail.ts.net dbname=fleet user={_MAPPED} password={_SECRET} sslmode=require"
    )
    assert pgqueue.assert_mapped_principal(dsn, expected_principal=_MAPPED) == _MAPPED


# --------------------------------------------------------------------------------------------
# Offline ordering proofs (no DB, no worker): a failed channel/update lets no later step run.
# --------------------------------------------------------------------------------------------
_SIDE_EFFECT_BANNERS = (
    "Pulling applied_set",
    "Arming canary",
    "Setting hard spend cap",
    "Starting",
    "HOME canary loaded",
)


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh required for the loader ordering proof")
def test_loader_fails_closed_before_any_side_effect_when_channel_invalid():
    # An incomplete runtime channel (no approved principal) must abort inside the validator,
    # before queue prep / canary / spend-cap / worker launch. Forbidden DSN, no venv, no DB.
    env = {k: v for k, v in os.environ.items() if k not in ("APPLYPILOT_PG_PRINCIPAL",)}
    env["FLEET_PG_DSN"] = f"host=h dbname=fleet user=postgres password={_SECRET} sslmode=require"
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-File", str(LOADER)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    for banner in _SIDE_EFFECT_BANNERS:
        assert banner not in combined, banner
    assert _SECRET not in combined  # the forbidden DSN value never leaks


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh required for the guard mechanism proof")
def test_native_exit_guard_stops_later_steps_after_a_failed_command(tmp_path):
    # Mirror the loader's Invoke-HomeCli guard exactly: a native nonzero exit is NOT swallowed by
    # $ErrorActionPreference='Stop', so the explicit $LASTEXITCODE check must throw and prevent
    # the later step from writing its marker.
    marker = tmp_path / "later-step-ran.txt"
    script = (
        "$ErrorActionPreference='Stop'\n"
        "function Invoke-Step { param([int]$Code,[string]$What)\n"
        "  & cmd /c \"exit $Code\"\n"
        "  if ($LASTEXITCODE -ne 0) { throw \"$What failed (exit $LASTEXITCODE)\" }\n"
        "}\n"
        "try {\n"
        "  Invoke-Step -Code 3 -What 'spend-cap update'\n"
        f"  'ran' | Out-File -LiteralPath '{marker.as_posix()}'\n"
        "  exit 0\n"
        "} catch { Write-Host \"ABORTED: $($_.Exception.Message)\"; exit 9 }\n"
    )
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 9, result.stdout + result.stderr
    assert "ABORTED: spend-cap update failed (exit 3)" in result.stdout
    assert not marker.exists()  # the later step never ran
