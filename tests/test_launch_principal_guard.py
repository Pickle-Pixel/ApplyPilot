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
from applypilot.fleet import launch_guard, pg_roles


REPO = Path(__file__).resolve().parents[1]

_SECRET = "s3cr3t-launch-principal-pw"
_MAPPED = "node_apply_a"

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
        pgqueue.assert_mapped_principal(dsn, expected_principal=_MAPPED)
    message = str(exc.value)
    assert _SECRET not in message  # never leaks the DSN/password


def test_ambient_pguser_admin_is_rejected(monkeypatch):
    monkeypatch.setenv("PGUSER", "postgres")
    with pytest.raises(RuntimeError, match="Ambient libpq"):
        pgqueue.assert_mapped_principal(
            f"host=h dbname=fleet user={_MAPPED} password={_SECRET} sslmode=require",
            expected_principal=_MAPPED,
        )


@pytest.mark.parametrize("dsn", ["service=fleetsvc", f"host=h passfile=/tmp/pg user=node_a password={_SECRET}"])
def test_indirect_service_and_passfile_are_rejected(dsn):
    with pytest.raises(RuntimeError, match="indirection"):
        pgqueue.assert_mapped_principal(dsn, expected_principal=_MAPPED)


def test_missing_explicit_user_fails_closed():
    with pytest.raises(RuntimeError, match="explicit mapped principal"):
        pgqueue.assert_mapped_principal(
            f"host=h dbname=fleet password={_SECRET} sslmode=require",
            expected_principal=_MAPPED,
        )


@pytest.mark.parametrize(
    "dsn",
    [
        f"host=node.tail.ts.net port=5432 dbname=applypilot_fleet user={_MAPPED} password={_SECRET} sslmode=require",
        f"postgresql://{_MAPPED}:{_SECRET}@node.tail.ts.net:5432/applypilot_fleet?sslmode=require",
    ],
)
def test_legitimate_mapped_dsn_is_accepted(dsn):
    principal = pgqueue.assert_mapped_principal(dsn, expected_principal=_MAPPED)
    assert principal not in pgqueue.FORBIDDEN_LAUNCH_PRINCIPALS


def test_random_login_is_not_accepted_as_the_expected_mapped_principal():
    dsn = (
        f"host=node.tail.ts.net dbname=fleet user=random_superuser password={_SECRET} "
        "sslmode=require"
    )
    with pytest.raises(RuntimeError, match="does not match the approved mapped principal"):
        pgqueue.assert_mapped_principal(dsn, expected_principal=_MAPPED)


@pytest.mark.parametrize(
    "dsn",
    [
        f"dbname=fleet user={_MAPPED} password={_SECRET} sslmode=require",
        f"host=node.tail.ts.net dbname=fleet user={_MAPPED} sslmode=require",
        f"host=node.tail.ts.net dbname=fleet user={_MAPPED} password={_SECRET}",
        (
            f"host=node.tail.ts.net dbname=fleet user={_MAPPED} password={_SECRET} "
            "sslmode=require options='-c role=postgres'"
        ),
    ],
)
def test_full_runtime_conninfo_contract_rejects_missing_or_role_changing_fields(dsn):
    with pytest.raises(RuntimeError):
        pgqueue.assert_mapped_principal(dsn, expected_principal=_MAPPED)


@pytest.mark.parametrize("ambient", ["PGSERVICE", "PGPASSFILE", "PGOPTIONS"])
def test_full_runtime_conninfo_contract_rejects_ambient_libpq(monkeypatch, ambient):
    monkeypatch.setenv(ambient, "forbidden")
    dsn = (
        f"host=node.tail.ts.net dbname=fleet user={_MAPPED} password={_SECRET} "
        "sslmode=require"
    )
    with pytest.raises(RuntimeError, match="Ambient libpq"):
        pgqueue.assert_mapped_principal(dsn, expected_principal=_MAPPED)


def test_launch_guard_module_fails_closed_on_missing_dsn(monkeypatch):
    monkeypatch.delenv("FLEET_PG_DSN", raising=False)
    assert launch_guard.main([]) == 78


def test_launch_guard_fails_without_approved_expected_principal(monkeypatch):
    monkeypatch.setenv(
        "FLEET_PG_DSN",
        f"host=node.tail.ts.net dbname=fleet user={_MAPPED} password={_SECRET} sslmode=require",
    )
    monkeypatch.delenv("APPLYPILOT_PG_PRINCIPAL", raising=False)
    assert launch_guard.main([]) == 78


def test_launch_guard_cli_rejects_forbidden_without_leaking(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PGUSER"}
    env["PYTHONPATH"] = str(REPO / "src")
    env["FLEET_PG_DSN"] = f"postgresql://postgres:{_SECRET}@h:5432/db"
    env["APPLYPILOT_PG_PRINCIPAL"] = _MAPPED
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


def test_connect_runtime_authenticates_exact_worker_contract(monkeypatch):
    conn = object()
    observed = []
    monkeypatch.setattr(pgqueue, "connect", lambda dsn, autocommit=False: conn)
    monkeypatch.setattr(
        pg_roles,
        "validate_runtime_principal",
        lambda value, *, worker_id, contract: observed.append(
            (value, worker_id, contract)
        ),
    )

    assert pgqueue.connect_runtime(
        "redacted-test-dsn", worker_id="node-a", contract="apply"
    ) is conn
    assert observed == [(conn, "node-a", "apply")]


def test_connect_runtime_closes_connection_when_database_mapping_rejects(monkeypatch):
    class FakeConnection:
        closed = False

        def close(self):
            self.closed = True

    conn = FakeConnection()
    monkeypatch.setattr(pgqueue, "connect", lambda dsn, autocommit=False: conn)
    monkeypatch.setattr(
        pg_roles,
        "validate_runtime_principal",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("mapping rejected")),
    )

    with pytest.raises(RuntimeError, match="mapping rejected"):
        pgqueue.connect_runtime(
            "redacted-test-dsn", worker_id="node-a", contract="apply"
        )
    assert conn.closed is True


@pytest.mark.parametrize(
    ("relpath", "contract"),
    [
        ("src/applypilot/fleet/apply_worker_main.py", "apply"),
        ("src/applypilot/fleet/compute_worker_main.py", "compute"),
        ("src/applypilot/fleet/discovery_main.py", "discovery"),
        ("src/applypilot/fleet/linkedin_worker_main.py", "linkedin"),
    ],
)
def test_runtime_workers_authenticate_database_mapping_before_schema_use(relpath, contract):
    text = (REPO / relpath).read_text(encoding="utf-8")
    authenticated = text.index("pgqueue.connect_runtime(")
    schema_use = min(
        position
        for marker in (
            "fleet_schema.require_apply_result_event_schema",
            "require_allowed(compute_worker_admission",
            "require_allowed(discovery_worker_admission",
            "emergency_admission.linkedin_worker_admission",
        )
        if (position := text.find(marker)) >= 0
    )
    assert authenticated < schema_use
    assert f'contract="{contract}"' in text[authenticated : authenticated + 180]
