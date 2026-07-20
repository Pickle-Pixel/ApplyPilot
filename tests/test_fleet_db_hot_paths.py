from __future__ import annotations

import builtins
import runpy
import sys
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")
from psycopg.conninfo import make_conninfo  # noqa: E402

from applypilot.apply import pgqueue  # noqa: E402
from applypilot.fleet import machine_blackout, machine_blackout_main  # noqa: E402
from applypilot.fleet import schema as fleet_schema  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


class _ReadOnlyConnection:
    def __init__(self) -> None:
        self.read_only = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_blackout_poll_does_not_migrate_and_marks_connection_read_only(monkeypatch, capsys):
    conn = _ReadOnlyConnection()
    monkeypatch.setenv("FLEET_PG_DSN", "fleet-test-dsn")
    monkeypatch.delenv("APPLYPILOT_FLEET_DSN", raising=False)
    monkeypatch.setattr(pgqueue, "connect", lambda dsn: conn if dsn == "fleet-test-dsn" else None)
    monkeypatch.setattr(
        fleet_schema,
        "ensure_schema_v3",
        lambda *_args, **_kwargs: pytest.fail("blackout poll attempted schema migration"),
    )

    def status_line(status_conn, label, *, role):
        assert status_conn is conn
        assert status_conn.read_only is True
        return f"OK|{label}|{role}|||"

    monkeypatch.setattr(machine_blackout, "status_line", status_line)
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    assert capsys.readouterr().out.strip() == "OK|m4|compute|||"


def test_blackout_poll_returns_keep_when_read_only_setup_fails(monkeypatch, capsys):
    class Connection:
        @property
        def read_only(self):
            return False

        @read_only.setter
        def read_only(self, _value):
            raise RuntimeError("read-only unavailable")

    monkeypatch.setenv("FLEET_PG_DSN", "fleet-test-dsn")
    monkeypatch.delenv("APPLYPILOT_FLEET_DSN", raising=False)
    monkeypatch.setattr(pgqueue, "connect", lambda _dsn: Connection())
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip().startswith("BLOCKED|m4|compute|blackout-query-error||")
    assert "read-only unavailable" in captured.err


def test_blackout_poll_preserves_ok_role_with_one_character_password(monkeypatch, capsys):
    conn = _ReadOnlyConnection()
    dsn = "host=fleet.invalid dbname=applypilot user=worker password=p"
    monkeypatch.setenv("FLEET_PG_DSN", dsn)
    monkeypatch.delenv("APPLYPILOT_FLEET_DSN", raising=False)
    monkeypatch.setattr(pgqueue, "connect", lambda _dsn: conn)
    monkeypatch.setattr(
        machine_blackout,
        "status_line",
        lambda *_args, **_kwargs: "OK|m4|compute|||",
    )
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip() == "OK|m4|compute|||"
    assert captured.err == ""


def test_blackout_poll_accepts_explicit_fleet_pg_dsn(monkeypatch, capsys):
    conn = _ReadOnlyConnection()
    monkeypatch.delenv("FLEET_PG_DSN", raising=False)
    monkeypatch.delenv("APPLYPILOT_FLEET_DSN", raising=False)
    monkeypatch.setenv("FLEET_PG_DSN", "fleet-test-dsn")
    monkeypatch.setattr(pgqueue, "connect", lambda dsn: conn if dsn == "fleet-test-dsn" else None)
    monkeypatch.setattr(
        machine_blackout,
        "status_line",
        lambda status_conn, label, *, role: f"OK|{label}|{role}|||",
    )
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    assert capsys.readouterr().out.strip() == "OK|m4|compute|||"


def test_blackout_poll_rejects_legacy_fleet_dsn_as_only_source(monkeypatch, capsys):
    monkeypatch.delenv("FLEET_PG_DSN", raising=False)
    monkeypatch.setenv("APPLYPILOT_FLEET_DSN", "legacy-fleet-test-dsn")
    monkeypatch.setattr(
        pgqueue,
        "connect",
        lambda *_args, **_kwargs: pytest.fail("legacy DSN reached the fleet connector"),
    )
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip().startswith("BLOCKED|m4|compute|blackout-query-error||")
    assert "No fleet Postgres DSN: set FLEET_PG_DSN" in captured.err
    assert "legacy-fleet-test-dsn" not in captured.out
    assert "legacy-fleet-test-dsn" not in captured.err


def test_blackout_poll_rejects_database_url_without_fleet_dsn(monkeypatch, capsys):
    monkeypatch.delenv("FLEET_PG_DSN", raising=False)
    monkeypatch.delenv("APPLYPILOT_FLEET_DSN", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://wrong-database.invalid/app")
    monkeypatch.setattr(
        pgqueue,
        "connect",
        lambda *_args, **_kwargs: pytest.fail("DATABASE_URL reached the fleet connector"),
    )
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip().startswith("BLOCKED|m4|compute|blackout-query-error||")
    assert "No fleet Postgres DSN" in captured.err


@pytest.mark.parametrize(
    "applypilot_fleet_dsn",
    [
        "postgresql://worker:other-secret@fleet-two.invalid:5432/applypilot",
        "postgresql://worker:other-secret@fleet-one.invalid:5433/applypilot",
        "postgresql://worker:other-secret@fleet-one.invalid:5432/other_database",
        "postgresql://other-user:other-secret@fleet-one.invalid:5432/applypilot",
        "postgresql://worker:different-secret@fleet-one.invalid:5432/applypilot",
        "postgresql://worker:other-secret@fleet-one.invalid:5432/applypilot?sslmode=require",
        (
            "postgresql://worker:other-secret@fleet-one.invalid:5432/applypilot"
            "?target_session_attrs=read-write"
        ),
    ],
)
def test_blackout_poll_rejects_conflicting_fleet_dsns(
    applypilot_fleet_dsn,
    monkeypatch,
    capsys,
):
    fleet_secret = "fleet-secret"
    monkeypatch.setenv(
        "FLEET_PG_DSN",
        f"postgresql://worker:{fleet_secret}@fleet-one.invalid:5432/applypilot",
    )
    monkeypatch.setenv("APPLYPILOT_FLEET_DSN", applypilot_fleet_dsn)
    monkeypatch.setattr(
        pgqueue,
        "connect",
        lambda *_args, **_kwargs: pytest.fail("conflicting fleet DSNs reached connector"),
    )
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip().startswith("BLOCKED|m4|compute|blackout-query-error||")
    assert "Inconsistent fleet Postgres DSN references" in captured.err
    assert fleet_secret not in captured.out
    assert fleet_secret not in captured.err
    assert "other-secret" not in captured.out
    assert "other-secret" not in captured.err
    assert "different-secret" not in captured.out
    assert "different-secret" not in captured.err


def test_blackout_poll_does_not_equate_omitted_host_with_localhost(monkeypatch, capsys):
    secret = "socket-secret"
    monkeypatch.delenv("PGHOST", raising=False)
    monkeypatch.setenv(
        "FLEET_PG_DSN",
        f"port=5432 dbname=applypilot user=worker password={secret}",
    )
    monkeypatch.setenv(
        "APPLYPILOT_FLEET_DSN",
        f"postgresql://worker:{secret}@localhost:5432/applypilot",
    )
    monkeypatch.setattr(
        pgqueue,
        "connect",
        lambda *_args, **_kwargs: pytest.fail("different destinations reached connector"),
    )
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip().startswith("BLOCKED|m4|compute|blackout-query-error||")
    assert "Inconsistent fleet Postgres DSN references" in captured.err
    assert secret not in captured.out
    assert secret not in captured.err


def test_blackout_poll_query_runs_in_read_only_transaction(fleet_db, monkeypatch, capsys):
    transaction_modes: list[str] = []
    original = machine_blackout.status_line
    monkeypatch.setenv("FLEET_PG_DSN", fleet_db)
    monkeypatch.delenv("APPLYPILOT_FLEET_DSN", raising=False)

    def checked_status_line(conn, label, *, role):
        with conn.cursor() as cur:
            cur.execute("SHOW transaction_read_only")
            transaction_modes.append(cur.fetchone()["transaction_read_only"])
        return original(conn, label, role=role)

    monkeypatch.setattr(machine_blackout, "status_line", checked_status_line)
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    assert capsys.readouterr().out.strip() == "OK|m4|compute|||"
    assert transaction_modes == ["on"]


def test_control_status_query_runs_in_read_only_transaction_without_migration(
    fleet_db,
    monkeypatch,
    capsys,
):
    transaction_modes: list[str] = []
    original = machine_blackout.status_line
    monkeypatch.setattr(
        fleet_schema,
        "ensure_schema_v3",
        lambda *_args, **_kwargs: pytest.fail("control status attempted schema migration"),
    )

    def checked_status_line(conn, label, *, role):
        with conn.cursor() as cur:
            cur.execute("SHOW transaction_read_only")
            transaction_modes.append(cur.fetchone()["transaction_read_only"])
        return original(conn, label, role=role)

    monkeypatch.setattr(machine_blackout, "status_line", checked_status_line)

    rc = machine_blackout_main.main(["status", "--label", "m4", "--role", "compute"])

    assert rc == 0
    assert capsys.readouterr().out.strip() == "OK|m4|compute|||"
    assert transaction_modes == ["on"]


def test_control_status_does_not_migrate_and_marks_connection_read_only(monkeypatch, capsys):
    conn = _ReadOnlyConnection()
    monkeypatch.setattr(pgqueue, "connect", lambda: conn)
    monkeypatch.setattr(
        fleet_schema,
        "ensure_schema_v3",
        lambda *_args, **_kwargs: pytest.fail("status command attempted schema migration"),
    )
    monkeypatch.setattr(
        machine_blackout,
        "status_line",
        lambda status_conn, label, *, role: (
            f"OK|{label}|{role}|||" if status_conn.read_only else pytest.fail("status connection is writable")
        ),
    )

    rc = machine_blackout_main.main(["status", "--label", "m4", "--role", "compute"])

    assert rc == 0
    assert capsys.readouterr().out.strip() == "OK|m4|compute|||"


@pytest.mark.parametrize(
    ("environment", "fleet_dsn", "applypilot_fleet_dsn"),
    [
        (
            {},
            "host=fleet.example password=secret",
            "host=fleet.example user='' password=secret",
        ),
        (
            {"PGHOST": "env.example", "PGPORT": "6543", "PGUSER": "env-worker"},
            "dbname=applypilot password=secret",
            "host='' port='' dbname=applypilot user='' password=secret",
        ),
        (
            {"PGPASSWORD": "env-secret", "PGPASSFILE": "env-passfile"},
            "host=fleet.example dbname=applypilot user=worker",
            (
                "host=fleet.example dbname=applypilot user=worker "
                "password='' passfile=''"
            ),
        ),
        (
            {},
            "host=fleet.example dbname=applypilot user=worker",
            (
                "host=fleet.example dbname=applypilot user=worker "
                "password='' passfile=''"
            ),
        ),
    ],
)
def test_blackout_poll_treats_explicit_empty_conninfo_as_effective_default(
    environment,
    fleet_dsn,
    applypilot_fleet_dsn,
    monkeypatch,
    capsys,
):
    conn = _ReadOnlyConnection()
    for variable in (
        "PGDATABASE",
        "PGHOST",
        "PGHOSTADDR",
        "PGPORT",
        "PGUSER",
        "PGPASSWORD",
        "PGPASSFILE",
    ):
        monkeypatch.delenv(variable, raising=False)
    for variable, value in environment.items():
        monkeypatch.setenv(variable, value)
    monkeypatch.setenv("FLEET_PG_DSN", fleet_dsn)
    monkeypatch.setenv("APPLYPILOT_FLEET_DSN", applypilot_fleet_dsn)
    monkeypatch.setattr(pgqueue, "connect", lambda dsn: conn if dsn == fleet_dsn else None)
    monkeypatch.setattr(
        machine_blackout,
        "status_line",
        lambda status_conn, label, *, role: f"OK|{label}|{role}|||",
    )
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip() == "OK|m4|compute|||"
    assert captured.err == ""


@pytest.mark.parametrize(
    "field",
    [
        "sslmode",
        "target_session_attrs",
        "channel_binding",
        "gssencmode",
        "sslcertmode",
    ],
)
def test_blackout_poll_rejects_explicit_empty_invalid_authority_field(
    field,
    monkeypatch,
    capsys,
):
    base_dsn = "host=fleet.example dbname=applypilot user=worker password=secret"
    monkeypatch.setenv("FLEET_PG_DSN", base_dsn)
    monkeypatch.setenv("APPLYPILOT_FLEET_DSN", f"{base_dsn} {field}=''")
    monkeypatch.setattr(
        pgqueue,
        "connect",
        lambda *_args, **_kwargs: pytest.fail("conflicting fleet DSNs reached connector"),
    )
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip().startswith("BLOCKED|m4|compute|blackout-query-error||")
    assert "Inconsistent fleet Postgres DSN references" in captured.err


@pytest.mark.parametrize(
    ("fleet_dsn", "applypilot_fleet_dsn"),
    [
        (
            "host=fleet.example port=5432 dbname=applypilot user=worker password=secret connect_timeout=5",
            "connect_timeout=5 password=secret user=worker dbname=applypilot port=5432 host=fleet.example",
        ),
        (
            "postgresql://worker:secret@fleet.example:5432/applypilot?connect_timeout=5",
            "host=fleet.example port=5432 dbname=applypilot user=worker password=secret connect_timeout=5",
        ),
        (
            "host=fleet.example dbname=applypilot user=worker password=secret",
            "postgresql://worker:secret@fleet.example:5432/applypilot",
        ),
        (
            "host=fleet.example dbname=applypilot user=worker password=secret",
            (
                "host=fleet.example port=5432 dbname=applypilot user=worker password=secret "
                "sslmode=prefer target_session_attrs=any"
            ),
        ),
        (
            "host=fleet.example user=worker password=secret",
            "host=fleet.example user=worker password=secret dbname=worker",
        ),
        (
            "postgresql://worker:secret@fleet.example",
            "postgresql://worker:secret@fleet.example/worker",
        ),
        (
            "host=fleet.example dbname=applypilot user=worker password=secret",
            (
                "host=fleet.example dbname=applypilot user=worker password=secret "
                "connect_timeout=0"
            ),
        ),
        (
            (
                "host=fleet.example dbname=applypilot user=worker password=secret "
                "connect_timeout=2"
            ),
            (
                "host=fleet.example dbname=applypilot user=worker password=secret "
                "connect_timeout=30"
            ),
        ),
        (
            "port=5432 dbname=applypilot user=worker password=secret",
            "host='' port=5432 dbname=applypilot user=worker password=secret",
        ),
        (
            "host=fleet.example port=5432 dbname=applypilot user=worker password=secret",
            (
                "host=fleet.example hostaddr='' port=5432 dbname=applypilot "
                "user=worker password=secret"
            ),
        ),
        (
            "host=fleet.example dbname=applypilot user=worker password=secret",
            "host=fleet.example port='' dbname=applypilot user=worker password=secret",
        ),
        (
            "host=fleet.example user=worker password=secret",
            "host=fleet.example user=worker password=secret dbname=''",
        ),
        (
            "host=fleet.example dbname=applypilot user=worker password=secret",
            (
                "host=fleet.example dbname=applypilot user=worker password=secret "
                "sslcertmode=allow"
            ),
        ),
    ],
)
def test_blackout_poll_accepts_semantically_equivalent_fleet_dsns(
    fleet_dsn,
    applypilot_fleet_dsn,
    monkeypatch,
    capsys,
):
    conn = _ReadOnlyConnection()
    for variable in (
        "PGDATABASE",
        "PGHOST",
        "PGHOSTADDR",
        "PGPORT",
        "PGSSLCERTMODE",
        "PGSSLMODE",
        "PGTARGETSESSIONATTRS",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("FLEET_PG_DSN", fleet_dsn)
    monkeypatch.setenv("APPLYPILOT_FLEET_DSN", applypilot_fleet_dsn)
    monkeypatch.setattr(pgqueue, "connect", lambda dsn: conn if dsn == fleet_dsn else None)
    monkeypatch.setattr(
        machine_blackout,
        "status_line",
        lambda status_conn, label, *, role: f"OK|{label}|{role}|||",
    )
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip() == "OK|m4|compute|||"
    assert captured.err == ""


@pytest.mark.parametrize("failure_point", ["connect", "query", "schema"])
def test_blackout_poll_returns_sanitized_blocked_when_database_access_fails(
    failure_point,
    monkeypatch,
    capsys,
):
    conn = _ReadOnlyConnection()
    secret = "super-secret-password"
    dsn = f"postgresql://worker:{secret}@fleet.invalid/applypilot"
    monkeypatch.setenv("FLEET_PG_DSN", dsn)
    monkeypatch.delenv("APPLYPILOT_FLEET_DSN", raising=False)

    def connect(_dsn):
        if failure_point == "connect":
            raise RuntimeError(f"could not connect using {dsn} password={secret}")
        return conn

    monkeypatch.setattr(pgqueue, "connect", connect)

    def fail_status_query(status_conn, _label, *, role):
        assert status_conn.read_only is True
        assert role == "compute"
        if failure_point in {"query", "schema"}:
            raise RuntimeError(f"{failure_point} access failed for password={secret}")
        return "OK|m4|compute|||"

    monkeypatch.setattr(machine_blackout, "status_line", fail_status_query)
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip().startswith("BLOCKED|m4|compute|blackout-query-error||")
    assert "RuntimeError" in captured.err
    assert "***" in captured.err
    assert secret not in captured.out
    assert secret not in captured.err


def test_desired_state_poll_marks_connection_read_only(monkeypatch, capsys):
    class Cursor:
        def __init__(self, conn):
            self.conn = conn

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, *_args):
            assert self.conn.read_only is True

        def fetchone(self):
            return None

    class Connection:
        def __init__(self):
            self.read_only = False

        def cursor(self):
            return Cursor(self)

    monkeypatch.setenv("FLEET_PG_DSN", "fleet-test-dsn")
    monkeypatch.delenv("APPLYPILOT_FLEET_DSN", raising=False)
    monkeypatch.setattr(pgqueue, "connect", lambda _dsn: Connection())
    monkeypatch.setattr(sys, "argv", ["fleet-agent-query.py", "m4"])

    runpy.run_path(str(ROOT / "fleet-agent-query.py"), run_name="__main__")

    assert capsys.readouterr().out.strip() == "STOP|||"


def test_desired_state_poll_returns_stop_when_query_fails(monkeypatch, capsys):
    class Cursor:
        def __init__(self, conn):
            self.conn = conn

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, *_args):
            assert self.conn.read_only is True
            raise RuntimeError("desired-state query failed")

    class Connection:
        def __init__(self):
            self.read_only = False

        def cursor(self):
            return Cursor(self)

    monkeypatch.setenv("FLEET_PG_DSN", "fleet-test-dsn")
    monkeypatch.delenv("APPLYPILOT_FLEET_DSN", raising=False)
    monkeypatch.setattr(pgqueue, "connect", lambda _dsn: Connection())
    monkeypatch.setattr(sys, "argv", ["fleet-agent-query.py", "m4"])

    runpy.run_path(str(ROOT / "fleet-agent-query.py"), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip() == "STOP|||"
    assert "control state unavailable" in captured.err
    assert "desired-state query failed" not in captured.err


@pytest.mark.parametrize(
    "sensitive_key",
    [
        "password",
        "sslpassword",
        "passfile",
        "sslkey",
        "oauth_client_secret",
        "scram_client_key",
        "scram_server_key",
        "sslkeylogfile",
    ],
)
def test_blackout_poll_redacts_every_sensitive_authority_value(
    sensitive_key,
    monkeypatch,
    capsys,
):
    parsed_secret = f"parsed-{sensitive_key}-secret"
    diagnostic_secret = f"diagnostic-{sensitive_key}-secret"
    dsn = make_conninfo(
        host="fleet.invalid",
        dbname="applypilot",
        user="worker",
        **{sensitive_key: parsed_secret},
    )
    monkeypatch.setenv("FLEET_PG_DSN", dsn)
    monkeypatch.delenv("APPLYPILOT_FLEET_DSN", raising=False)

    def fail_connect(_dsn):
        raise RuntimeError(
            f"credential failure {parsed_secret} {sensitive_key}={diagnostic_secret}"
        )

    monkeypatch.setattr(pgqueue, "connect", fail_connect)
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip().startswith("BLOCKED|m4|compute|blackout-query-error||")
    assert "***" in captured.out
    assert "***" in captured.err
    for secret in (parsed_secret, diagnostic_secret):
        assert secret not in captured.out
        assert secret not in captured.err


def test_blackout_poll_rejects_and_redacts_conflicting_sslkeylogfile(
    monkeypatch,
    capsys,
):
    first_path = r"C:\sensitive\fleet-one.keys"
    second_path = r"C:\sensitive\fleet-two.keys"
    base = "host=fleet.example dbname=applypilot user=worker password=secret"
    monkeypatch.setenv("FLEET_PG_DSN", make_conninfo(base, sslkeylogfile=first_path))
    monkeypatch.setenv(
        "APPLYPILOT_FLEET_DSN",
        make_conninfo(base, sslkeylogfile=second_path),
    )
    monkeypatch.setattr(
        pgqueue,
        "connect",
        lambda *_args, **_kwargs: pytest.fail("conflicting fleet DSNs reached connector"),
    )
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip().startswith("BLOCKED|m4|compute|blackout-query-error||")
    assert "Inconsistent fleet Postgres DSN references" in captured.err
    for path in (first_path, second_path):
        assert path not in captured.out
        assert path not in captured.err


def test_blackout_poll_import_failure_still_emits_sanitized_blocked_line(
    monkeypatch,
    capsys,
):
    secret = "import-hook-secret"
    dsn = f"postgresql://worker:{secret}@fleet.invalid/applypilot"
    real_import = builtins.__import__

    def fail_psycopg_import(name, *args, **kwargs):
        if name == "psycopg" or name.startswith("psycopg."):
            raise ImportError(f"psycopg unavailable for password={secret} using {dsn}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_psycopg_import)
    monkeypatch.setenv("FLEET_PG_DSN", dsn)
    monkeypatch.delenv("APPLYPILOT_FLEET_DSN", raising=False)
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip().startswith("BLOCKED|m4|compute|blackout-query-error||")
    assert len(captured.out.splitlines()) == 1
    assert "Traceback" not in captured.err
    assert secret not in captured.out
    assert secret not in captured.err


def test_blackout_poll_import_failure_fallback_decodes_escaped_keyword_secret(
    monkeypatch,
    capsys,
):
    decoded_secret = "alpha beta"
    escaped_secret = r"alpha\ beta"
    dsn = f"host=fleet.invalid dbname=applypilot user=worker password={escaped_secret}"
    real_import = builtins.__import__

    def fail_psycopg_import(name, *args, **kwargs):
        if name == "psycopg" or name.startswith("psycopg."):
            raise ImportError(
                f"psycopg unavailable for {decoded_secret} password={escaped_secret}"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_psycopg_import)
    monkeypatch.setenv("FLEET_PG_DSN", dsn)
    monkeypatch.delenv("APPLYPILOT_FLEET_DSN", raising=False)
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip().startswith("BLOCKED|m4|compute|blackout-query-error||")
    assert len(captured.out.splitlines()) == 1
    assert decoded_secret not in captured.out
    assert decoded_secret not in captured.err
    assert escaped_secret not in captured.out
    assert escaped_secret not in captured.err


def test_blackout_poll_sanitizes_blocked_policy_name_protocol(monkeypatch, capsys):
    conn = _ReadOnlyConnection()
    dsn_secret = "policy-dsn-secret"
    policy_secret = "policy-name-secret"
    dsn = f"postgresql://worker:{dsn_secret}@fleet.invalid/applypilot"
    monkeypatch.setenv("FLEET_PG_DSN", dsn)
    monkeypatch.delenv("APPLYPILOT_FLEET_DSN", raising=False)
    monkeypatch.setattr(pgqueue, "connect", lambda _dsn: conn)
    monkeypatch.setattr(
        machine_blackout,
        "status_line",
        lambda *_args, **_kwargs: (
            f"BLOCKED|m4|compute|policy|injected\npassword={policy_secret}||safe reason"
        ),
    )
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    captured = capsys.readouterr()
    line = captured.out.rstrip("\r\n")
    assert line.startswith("BLOCKED|")
    assert len(line.split("|")) == 6
    assert "\n" not in line
    assert "\r" not in line
    assert policy_secret not in line
    assert dsn_secret not in line


def test_blackout_poll_sanitizes_blocked_policy_reason_credentials(monkeypatch, capsys):
    conn = _ReadOnlyConnection()
    dsn_secret = "reason-dsn-secret"
    passfile_secret = "reason-passfile-secret"
    uri_secret = "reason-uri-secret"
    password_secret = "reason-password-secret"
    oauth_secret = "reason-oauth-secret"
    scram_secret = "reason-scram-secret"
    server_key_secret = "reason-server-key-secret"
    dsn = make_conninfo(
        host="fleet.invalid",
        dbname="applypilot",
        user="worker",
        password=dsn_secret,
        passfile=passfile_secret,
    )
    reason = (
        f"raw {dsn_secret} {passfile_secret}\n"
        f"password={password_secret} passfile={passfile_secret} "
        f"oauth_client_secret={oauth_secret} scram_client_key={scram_secret} "
        f"scram_server_key={server_key_secret} "
        f"postgresql://worker:{uri_secret}@fleet.invalid/applypilot|tail"
    )
    monkeypatch.setenv("FLEET_PG_DSN", dsn)
    monkeypatch.delenv("APPLYPILOT_FLEET_DSN", raising=False)
    monkeypatch.setattr(pgqueue, "connect", lambda _dsn: conn)
    monkeypatch.setattr(
        machine_blackout,
        "status_line",
        lambda *_args, **_kwargs: f"BLOCKED|m4|compute|policy||{reason}",
    )
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    captured = capsys.readouterr()
    line = captured.out.rstrip("\r\n")
    assert line.startswith("BLOCKED|")
    assert len(line.split("|")) == 6
    assert "\n" not in line
    assert "\r" not in line
    assert "***" in line
    for secret in (
        dsn_secret,
        passfile_secret,
        uri_secret,
        password_secret,
        oauth_secret,
        scram_secret,
        server_key_secret,
    ):
        assert secret not in line


@pytest.mark.parametrize("surface", ["policy", "diagnostic"])
@pytest.mark.parametrize(
    "sensitive_key",
    [
        "password",
        "sslpassword",
        "passfile",
        "sslkey",
        "oauth_client_secret",
        "scram_client_key",
        "scram_server_key",
    ],
)
def test_blackout_poll_decodes_escaped_keyword_secrets_with_psycopg(
    sensitive_key,
    surface,
    monkeypatch,
    capsys,
):
    decoded_secret = "alpha beta"
    escaped_secret = r"alpha\ beta"
    dsn = (
        "host=fleet.invalid dbname=applypilot user=worker "
        f"{sensitive_key}={escaped_secret}"
    )
    conn = _ReadOnlyConnection()
    monkeypatch.setenv("FLEET_PG_DSN", dsn)
    monkeypatch.delenv("APPLYPILOT_FLEET_DSN", raising=False)
    if surface == "policy":
        monkeypatch.setattr(pgqueue, "connect", lambda _dsn: conn)
        monkeypatch.setattr(
            machine_blackout,
            "status_line",
            lambda *_args, **_kwargs: (
                f"BLOCKED|m4|compute|policy {decoded_secret}||reason {decoded_secret}"
            ),
        )
    else:
        monkeypatch.setattr(
            pgqueue,
            "connect",
            lambda _dsn: (_ for _ in ()).throw(
                RuntimeError(
                    f"credential failure {decoded_secret} "
                    f"{sensitive_key}={escaped_secret}"
                )
            ),
        )
    monkeypatch.setattr(sys, "argv", ["fleet-blackout-query.py", "m4", "compute"])

    runpy.run_path(str(ROOT / "fleet-blackout-query.py"), run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip().startswith("BLOCKED|m4|compute|")
    assert decoded_secret not in captured.out
    assert decoded_secret not in captured.err
    assert escaped_secret not in captured.out
    assert escaped_secret not in captured.err
