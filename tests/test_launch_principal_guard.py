"""Adversarial tests for the reusable mapped launch-principal validator (Task 9B).

The single hardened conninfo parser (``pgqueue.assert_mapped_principal`` + the
``applypilot.fleet.launch_guard`` entry point) must reject forbidden admin/shared
principals wherever they hide -- libpq keyword DSNs, URL DSNs (including percent-encoded
and query-string identities), mixed case / quoted spellings, the ambient ``PGUSER`` form,
and the indirect ``service`` / ``passfile`` forms -- while accepting a legitimate explicit
mapped DSN. No message may leak the DSN value.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from applypilot.apply import pgqueue
from applypilot.fleet import launch_guard


REPO = Path(__file__).resolve().parents[1]

_SECRET = "s3cr3t-launch-principal-pw"

# Every one of these selects the admin (postgres) or shared (fleet_worker) principal.
FORBIDDEN_DSNS = [
    f"host=h port=5432 dbname=applypilot_fleet user=postgres password={_SECRET}",
    f"host=h dbname=fleet user=fleet_worker password={_SECRET}",
    f"host=h dbname=fleet user='postgres' password={_SECRET}",  # quoted
    f"host=h dbname=fleet user=POSTGRES password={_SECRET}",  # mixed case
    f"host=h   dbname=fleet   user   =   fleet_worker password={_SECRET}",  # spacing
    f"postgresql://postgres:{_SECRET}@h:5432/db",  # URL principal
    f"postgres://fleet_worker:{_SECRET}@h/db",
    f"postgresql://%70ostgres:{_SECRET}@h/db",  # percent-encoded 'p'ostgres
    f"postgresql://h/db?user=postgres&password={_SECRET}",  # query-string identity
]


@pytest.mark.parametrize("dsn", FORBIDDEN_DSNS)
def test_forbidden_principals_are_rejected(dsn):
    with pytest.raises(RuntimeError) as exc:
        pgqueue.assert_mapped_principal(dsn)
    message = str(exc.value)
    assert _SECRET not in message  # never leaks the DSN/password


def test_ambient_pguser_admin_is_rejected(monkeypatch):
    monkeypatch.setenv("PGUSER", "postgres")
    with pytest.raises(RuntimeError, match="Ambient PGUSER"):
        pgqueue.assert_mapped_principal(f"host=h dbname=fleet user=node_apply_a password={_SECRET}")


@pytest.mark.parametrize("dsn", ["service=fleetsvc", f"host=h passfile=/tmp/pg user=node_a password={_SECRET}"])
def test_indirect_service_and_passfile_are_rejected(dsn):
    with pytest.raises(RuntimeError, match="indirection"):
        pgqueue.assert_mapped_principal(dsn)


def test_missing_explicit_user_fails_closed():
    with pytest.raises(RuntimeError, match="explicit mapped principal"):
        pgqueue.assert_mapped_principal(f"host=h dbname=fleet password={_SECRET}")


@pytest.mark.parametrize(
    "dsn",
    [
        f"host=h port=5432 dbname=applypilot_fleet user=node_apply_a password={_SECRET}",
        f"postgresql://node_linkedin_m2:{_SECRET}@h:5432/applypilot_fleet",
    ],
)
def test_legitimate_mapped_dsn_is_accepted(dsn):
    principal = pgqueue.assert_mapped_principal(dsn)
    assert principal not in pgqueue.FORBIDDEN_LAUNCH_PRINCIPALS


def test_launch_guard_module_fails_closed_on_missing_dsn(monkeypatch):
    monkeypatch.delenv("FLEET_PG_DSN", raising=False)
    assert launch_guard.main([]) == 78


def test_launch_guard_cli_rejects_forbidden_without_leaking(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PGUSER"}
    env["PYTHONPATH"] = str(REPO / "src")
    env["FLEET_PG_DSN"] = f"postgresql://postgres:{_SECRET}@h:5432/db"
    result = subprocess.run(
        [sys.executable, "-m", "applypilot.fleet.launch_guard"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 78
    assert _SECRET not in (result.stdout + result.stderr)
    assert "forbidden admin/shared principal" in result.stderr
