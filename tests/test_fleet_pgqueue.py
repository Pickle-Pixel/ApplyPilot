"""P0 unit tests for the cloud apply-fleet Postgres layer (pgqueue).

Spins up a DISPOSABLE local Postgres (from the `applypilot-pgtest` conda env) on a temp
cluster + free port, runs the real lease/reclaim/cap/sync SQL against it, and tears it down.
No Docker, no system Postgres, no Railway. If the pg binaries aren't found the module skips.

The lease/reclaim tests are the ones that matter most: a double-grab or a re-leased
possibly-submitted job is a real-world double-application under Jonathan's name.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import threading
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")
conninfo_to_dict = psycopg.conninfo.conninfo_to_dict

from applypilot.apply import fleet_sync  # noqa: E402
from applypilot.apply import pgqueue  # noqa: E402


@pytest.fixture(autouse=True)
def _retire_score_only_queue_tests(request):
    """The v1 lease/push suite is superseded by test_fleet_v3_governor_queue."""
    allowed = {
        "test_ensure_schema_is_idempotent",
        "test_legacy_score_only_queue_apis_are_disabled",
    }
    if request.node.name not in allowed and not request.node.name.startswith("test_connection_"):
        pytest.skip("legacy score-only queue retired; canonical v3 queue owns these invariants")


def test_legacy_score_only_queue_apis_are_disabled(db):
    with pgqueue.connect_admin(db) as conn:
        with pytest.raises(RuntimeError, match="canonical"):
            pgqueue.push_jobs(conn, [])
        with pytest.raises(RuntimeError, match="canonical"):
            pgqueue.lease_one(conn, "worker")
        with pytest.raises(RuntimeError, match="canonical"):
            fleet_sync.push_offsite_jobs(pg_conn=conn)


# ---------------------------------------------------------------------------
# Disposable local Postgres
# ---------------------------------------------------------------------------

def _find_pg_bin() -> Path | None:
    """Locate the applypilot-pgtest env's Postgres bin dir (initdb/pg_ctl/postgres)."""
    cands: list[Path] = []
    if os.environ.get("APPLYPILOT_PGTEST_BIN"):
        cands.append(Path(os.environ["APPLYPILOT_PGTEST_BIN"]))
    conda = shutil.which("conda")
    bases: list[Path] = []
    if conda:
        # .../Scripts/conda.exe or .../condabin/conda -> base is two up
        bases.append(Path(conda).resolve().parent.parent)
    bases.append(Path.home() / "anaconda3")
    bases.append(Path.home() / "miniconda3")
    for base in bases:
        cands.append(base / "envs" / "applypilot-pgtest" / "Library" / "bin")  # win
        cands.append(base / "envs" / "applypilot-pgtest" / "bin")              # nix
    for c in cands:
        exe = "initdb.exe" if os.name == "nt" else "initdb"
        if (c / exe).exists():
            return c
    return None


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="session")
def pg_dsn():
    binp = _find_pg_bin()
    if binp is None:
        pytest.skip("applypilot-pgtest Postgres env not found "
                    "(conda create -n applypilot-pgtest -c conda-forge postgresql)")
    ext = ".exe" if os.name == "nt" else ""
    initdb, pg_ctl = binp / f"initdb{ext}", binp / f"pg_ctl{ext}"

    datadir = Path(tempfile.mkdtemp(prefix="ap_pgtest_"))
    logfile = datadir / "server.log"
    port = _free_port()
    try:
        subprocess.run(
            [str(initdb), "-D", str(datadir), "-U", "postgres", "-A", "trust", "-E", "UTF8"],
            check=True, capture_output=True, text=True,
        )
        # NOTE: `pg_ctl start` launches a PERSISTENT postgres that inherits this process's
        # stdout/stderr. With capture_output (PIPEs) the child holds the pipe open, so
        # subprocess.run blocks on EOF until the SERVER exits -> a hang that never returns.
        # Send the server's own log to -l <logfile> and give pg_ctl DEVNULL (no pipe to
        # inherit) so run() returns the moment pg_ctl confirms readiness and exits.
        subprocess.run(
            [str(pg_ctl), "-D", str(datadir), "-l", str(logfile),
             "-o", f"-p {port} -c listen_addresses=127.0.0.1 -c fsync=off",
             "-w", "-t", "30", "start"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        log = logfile.read_text(encoding="utf-8", errors="replace") if logfile.exists() else ""
        shutil.rmtree(datadir, ignore_errors=True)
        pytest.skip(f"could not start test Postgres (exit {e.returncode}):\n{log}")

    dsn = f"postgresql://postgres:test-password@127.0.0.1:{port}/postgres"
    try:
        yield dsn
    finally:
        subprocess.run([str(pg_ctl), "-D", str(datadir), "-m", "immediate", "-w", "stop"],
                       capture_output=True, text=True)
        shutil.rmtree(datadir, ignore_errors=True)


@pytest.fixture
def db(pg_dsn):
    """Clean schema for each test: ensure schema, truncate, reset config. Yields the DSN."""
    with pgqueue.connect_admin(pg_dsn) as conn:
        pgqueue.ensure_schema(conn)
        with conn.cursor() as cur:
            # lease_one now requires approved_batch IS NOT NULL (Task 5 gate); add the column
            # to the base schema so pgqueue tests work against the old fleet_schema.sql cluster.
            cur.execute("ALTER TABLE apply_queue ADD COLUMN IF NOT EXISTS approved_batch TEXT;")
            cur.execute("TRUNCATE apply_queue;")
            cur.execute("UPDATE fleet_config SET spend_cap_usd=0, paused=FALSE WHERE id=1;")
        conn.commit()
    return pg_dsn


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def _seed_queued(conn, n, *, score_desc=True):
    rows = [{
        "url": f"https://jobs.example.com/{i}",
        "company": f"Co{i}", "title": f"Role {i}",
        "application_url": f"https://boards.greenhouse.io/example/jobs/{i}",
        "score": float(i if score_desc else n - i),
        "apply_domain": f"greenhouse-{i}.io",   # distinct domains -> politeness never blocks
    } for i in range(n)]
    pgqueue.push_jobs(conn, rows)
    # Stamp approval so lease_one (now gated on approved_batch IS NOT NULL) can pick these up.
    with conn.cursor() as cur:
        cur.execute("UPDATE apply_queue SET approved_batch='b0'")
    conn.commit()
    return rows


def _insert_leased(conn, url, *, owner, attempts, apply_error, expires_delta_sec):
    """Insert a row already in 'leased' state with an explicit lease_expires_at offset."""
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO apply_queue
               (url, application_url, score, apply_domain, status, lease_owner,
                lease_expires_at, attempts, apply_error)
               VALUES (%s,%s,%s,%s,'leased',%s, now() + make_interval(secs => %s), %s, %s)""",
            (url, url, 5.0, "d.io", owner, expires_delta_sec, attempts, apply_error),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_CONNECTION_DEFAULTS = {
    "application_name": "ApplyPilot",
    "connect_timeout": "10",
    "keepalives": "1",
    "keepalives_idle": "30",
    "keepalives_interval": "10",
    "keepalives_count": "5",
}
_DOCUMENTED_LIBPQ_CONNECTION_ENV_VARS = (
    "PGAPPNAME",
    "PGCHANNELBINDING",
    "PGCLIENTENCODING",
    "PGCONNECT_TIMEOUT",
    "PGDATABASE",
    "PGDATESTYLE",
    "PGGEQO",
    "PGGSSDELEGATION",
    "PGGSSENCMODE",
    "PGGSSLIB",
    "PGHOST",
    "PGHOSTADDR",
    "PGKRBSRVNAME",
    "PGLOADBALANCEHOSTS",
    "PGLOCALEDIR",
    "PGMAXPROTOCOLVERSION",
    "PGMINPROTOCOLVERSION",
    "PGOPTIONS",
    "PGPASSFILE",
    "PGPASSWORD",
    "PGPORT",
    "PGREQUIREAUTH",
    "PGREQUIREPEER",
    "PGREQUIRESSL",
    "PGSERVICE",
    "PGSERVICEFILE",
    "PGSYSCONFDIR",
    "PGSSLCERT",
    "PGSSLCERTMODE",
    "PGSSLCOMPRESSION",
    "PGSSLCRL",
    "PGSSLCRLDIR",
    "PGSSLKEY",
    "PGSSLMAXPROTOCOLVERSION",
    "PGSSLMINPROTOCOLVERSION",
    "PGSSLMODE",
    "PGSSLNEGOTIATION",
    "PGSSLROOTCERT",
    "PGSSLSNI",
    "PGTARGETSESSIONATTRS",
    "PGTZ",
    "PGUSER",
)


def _clear_ambient_libpq(monkeypatch):
    for name in _DOCUMENTED_LIBPQ_CONNECTION_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APPLYPILOT_PG_PRINCIPAL", "worker")


def _capture_connection(monkeypatch):
    _clear_ambient_libpq(monkeypatch)
    captured = {}
    sentinel = object()

    def fake_connect(conninfo, **kwargs):
        captured["params"] = conninfo_to_dict(conninfo)
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(pgqueue.psycopg, "connect", fake_connect)
    return captured, sentinel


@pytest.mark.parametrize(
    "dsn, expected_host",
    [
        ("postgresql://worker:p%40ss@localhost/fleet?sslmode=require", "localhost"),
        ("postgresql://worker:p%40ss@[::1]:5432/fleet?sslmode=require", "::1"),
        ("host=127.0.0.1 dbname=fleet user=worker password='p@ss' sslmode=require", "127.0.0.1"),
        (
            "host=postgres.railway.internal dbname=fleet user=worker password='p@ss' sslmode=require",
            "postgres.railway.internal",
        ),
        (
            "host=pg-gateway.tailnet.ts.net dbname=fleet user=worker password='p@ss' sslmode=require",
            "pg-gateway.tailnet.ts.net",
        ),
    ],
)
def test_connection_private_endpoints_receive_bounded_defaults(monkeypatch, dsn, expected_host):
    captured, sentinel = _capture_connection(monkeypatch)

    assert pgqueue.connect(dsn, autocommit=True) is sentinel

    assert captured["params"]["host"] == expected_host
    if "password" in captured["params"]:
        assert captured["params"]["password"] == "p@ss"
    for key, value in _CONNECTION_DEFAULTS.items():
        assert captured["params"][key] == value
    assert captured["kwargs"]["autocommit"] is True


@pytest.mark.parametrize(
    "dsn, expected_mode",
    [
        (
            "postgresql://worker:p%40ss@db.example.com:5432/fleet?sslmode=verify-full&sslrootcert=system",
            "verify-full",
        ),
        (
            "host=db.example.com port=5432 dbname=fleet user=worker password=p "
            "sslmode=verify-full sslrootcert=/run/secrets/provider-ca.pem",
            "verify-full",
        ),
    ],
)
def test_connection_public_endpoints_require_explicit_tls(monkeypatch, dsn, expected_mode):
    captured, sentinel = _capture_connection(monkeypatch)

    assert pgqueue.connect(dsn) is sentinel

    assert captured["params"]["sslmode"] == expected_mode
    assert captured["params"]["sslrootcert"]
    for key, value in _CONNECTION_DEFAULTS.items():
        assert captured["params"][key] == value


@pytest.mark.parametrize("sslmode", [None, "disable", "allow", "prefer", "require", "verify-ca"])
def test_connection_public_endpoint_rejects_non_verifying_tls(monkeypatch, sslmode):
    _clear_ambient_libpq(monkeypatch)
    secret = "never-print-this-password"
    suffix = "" if sslmode is None else f"?sslmode={sslmode}"
    dsn = f"postgresql://worker:{secret}@roundhouse.proxy.rlwy.net:12345/fleet{suffix}"
    monkeypatch.setattr(
        pgqueue.psycopg,
        "connect",
        lambda *args, **kwargs: pytest.fail("invalid connection contract reached psycopg.connect"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        pgqueue.connect(dsn)

    assert "sslmode=verify-full" in str(exc_info.value)
    assert secret not in str(exc_info.value)


def test_connection_public_verify_full_requires_explicit_root_certificate(monkeypatch):
    _clear_ambient_libpq(monkeypatch)
    secret = "never-print-this-password"
    dsn = f"postgresql://worker:{secret}@db.example.com/fleet?sslmode=verify-full"

    with pytest.raises(RuntimeError) as exc_info:
        pgqueue.connect(dsn)

    assert "sslrootcert" in str(exc_info.value)
    assert secret not in str(exc_info.value)


def test_runtime_connect_rejects_admin_while_explicit_admin_api_allows_it(monkeypatch):
    captured, sentinel = _capture_connection(monkeypatch)
    dsn = "host=localhost dbname=fleet user=postgres password=p sslmode=require"

    with pytest.raises(RuntimeError, match="forbidden admin/shared principal"):
        pgqueue.connect(dsn)

    assert pgqueue.connect_admin(dsn) is sentinel
    assert captured["params"]["user"] == "postgres"


def test_runtime_connect_requires_an_exact_approved_principal(monkeypatch):
    _clear_ambient_libpq(monkeypatch)
    monkeypatch.delenv("APPLYPILOT_PG_PRINCIPAL")
    dsn = "host=localhost dbname=fleet user=worker password=p sslmode=require"

    with pytest.raises(RuntimeError, match="Approved mapped Postgres principal"):
        pgqueue.connect(dsn)


def test_connection_mixed_private_and_public_hosts_require_public_release_tls(monkeypatch):
    _clear_ambient_libpq(monkeypatch)
    monkeypatch.setattr(
        pgqueue.psycopg,
        "connect",
        lambda *args, **kwargs: pytest.fail("ambiguous connection reached psycopg.connect"),
    )

    with pytest.raises(RuntimeError, match="sslmode=verify-full"):
        pgqueue.connect(
            "host=localhost,roundhouse.proxy.rlwy.net dbname=fleet "
            "user=worker password=p sslmode=require"
        )


@pytest.mark.parametrize(
    "dsn",
    [
        "dbname=fleet",
        "postgresql:///fleet",
        "service=fleet",
        "host=localhost service=fleet dbname=fleet",
        "host=localhost passfile=/run/secrets/pgpass dbname=fleet",
    ],
)
def test_connection_rejects_hostless_and_indirect_targets(monkeypatch, dsn):
    _clear_ambient_libpq(monkeypatch)

    with pytest.raises(RuntimeError, match="explicit|indirection"):
        pgqueue.connect(dsn)


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://worker@localhost/fleet",
        "host=localhost dbname=fleet user=worker",
        "postgresql://worker@db.example.com/fleet?sslmode=verify-full&sslrootcert=system",
        "host=db.example.com dbname=fleet sslmode=verify-full sslrootcert=system",
    ],
)
def test_connection_rejects_passwordless_dsn_without_exposing_target(monkeypatch, dsn):
    _clear_ambient_libpq(monkeypatch)

    with pytest.raises(RuntimeError) as exc_info:
        pgqueue.connect(dsn)

    assert "explicit password" in str(exc_info.value)
    assert "localhost" not in str(exc_info.value)
    assert "db.example.com" not in str(exc_info.value)


def test_connection_environment_rejection_constant_is_exhaustive():
    assert pgqueue.LIBPQ_CONNECTION_ENV_VARS == frozenset(_DOCUMENTED_LIBPQ_CONNECTION_ENV_VARS)
    assert "PG_CONFIG" not in pgqueue.LIBPQ_CONNECTION_ENV_VARS


@pytest.mark.parametrize("variable", _DOCUMENTED_LIBPQ_CONNECTION_ENV_VARS)
def test_connection_rejects_every_documented_libpq_connection_variable(monkeypatch, variable):
    _clear_ambient_libpq(monkeypatch)
    monkeypatch.setenv(variable, "never-print-this-secret")

    with pytest.raises(RuntimeError) as exc_info:
        pgqueue.connect("host=localhost dbname=fleet")

    assert variable in str(exc_info.value)
    assert "never-print-this-secret" not in str(exc_info.value)


def test_connection_rejects_ambient_libpq_variable_even_when_empty(monkeypatch):
    _clear_ambient_libpq(monkeypatch)
    monkeypatch.setenv("PGHOST", "")

    with pytest.raises(RuntimeError, match="PGHOST"):
        pgqueue.connect("host=localhost dbname=fleet")


@pytest.mark.parametrize("variable", ["PG_CONFIG", "PG_FLEET_LABEL", "PGAPP_WORKER_ID"])
def test_connection_does_not_reject_unrelated_pg_prefixed_variables(monkeypatch, variable):
    monkeypatch.setenv(variable, "unrelated-application-value")
    monkeypatch.setenv("PG_CONFIG", "C:/PostgreSQL/bin/pg_config.exe")
    captured, sentinel = _capture_connection(monkeypatch)

    assert pgqueue.connect(
        "host=localhost dbname=fleet user=worker password=p sslmode=require"
    ) is sentinel
    assert captured["params"]["host"] == "localhost"


@pytest.mark.parametrize(
    "dsn",
    [
        "hostaddr=127.0.0.1 dbname=fleet",
        "hostaddr=127.0.0.1,127.0.0.2 dbname=fleet",
        "host=localhost hostaddr=127.0.0.1 dbname=fleet",
        "host=localhost,db.example.com hostaddr=127.0.0.1 dbname=fleet",
        "host=localhost hostaddr=8.8.8.8 dbname=fleet",
        "host=db.example.com hostaddr=127.0.0.1 dbname=fleet",
        "host=localhost, hostaddr=127.0.0.1,127.0.0.2 dbname=fleet",
    ],
)
def test_connection_rejects_all_hostaddr_targets(monkeypatch, dsn):
    _clear_ambient_libpq(monkeypatch)

    with pytest.raises(RuntimeError, match="hostaddr"):
        pgqueue.connect(dsn)


@pytest.mark.parametrize(
    "host",
    [
        "10.0.0.8",
        "100.100.100.100",
        "127.0.0.2",
        "169.254.1.1",
        "192.0.2.1",
    ],
)
def test_connection_does_not_exempt_unapproved_ip_topologies(monkeypatch, host):
    _clear_ambient_libpq(monkeypatch)

    with pytest.raises(RuntimeError, match="sslmode=verify-full"):
        pgqueue.connect(
            f"host={host} dbname=fleet user=worker password=p sslmode=require"
        )


@pytest.mark.parametrize(
    "host",
    [
        "/var/run/postgresql",
        "foo..railway.internal",
        "-bad.railway.internal",
        "*.tailnet.ts.net",
        "gateway.ts.net",
    ],
)
def test_connection_rejects_invalid_private_topology_syntax(monkeypatch, host):
    _clear_ambient_libpq(monkeypatch)

    with pytest.raises(RuntimeError, match="topology|syntax"):
        pgqueue.connect(
            f"host={host} dbname=fleet user=worker password=p sslmode=require"
        )


def test_connection_preserves_explicit_caller_settings(monkeypatch):
    captured, sentinel = _capture_connection(monkeypatch)
    dsn = (
        "host=localhost dbname=fleet user=worker password=p sslmode=require connect_timeout=3 keepalives=1 "
        "keepalives_idle=15 keepalives_interval=4 keepalives_count=2 "
        "application_name=ApplyPilotController"
    )

    assert pgqueue.connect(dsn) is sentinel

    assert captured["params"]["connect_timeout"] == "3"
    assert captured["params"]["keepalives_idle"] == "15"
    assert captured["params"]["keepalives_interval"] == "4"
    assert captured["params"]["keepalives_count"] == "2"
    assert captured["params"]["application_name"] == "ApplyPilotController"


def test_connection_invalid_conninfo_error_is_secret_safe(monkeypatch):
    _clear_ambient_libpq(monkeypatch)
    secret = "never-print-this-password"
    malformed = f"host=localhost password={secret} broken='unterminated"

    with pytest.raises(RuntimeError) as exc_info:
        pgqueue.connect(malformed)

    assert "Invalid fleet Postgres DSN" in str(exc_info.value)
    assert secret not in str(exc_info.value)


def test_connection_get_dsn_rejects_retired_applypilot_alias(monkeypatch):
    monkeypatch.delenv("FLEET_PG_DSN", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://live.example/postgres")
    monkeypatch.setenv("APPLYPILOT_FLEET_DSN", "postgresql://test.example/postgres")

    with pytest.raises(RuntimeError, match="FLEET_PG_DSN") as exc_info:
        pgqueue.get_dsn()

    assert "APPLYPILOT_FLEET_DSN" not in str(exc_info.value)


def test_ensure_schema_is_idempotent(db):
    with pgqueue.connect_admin(db) as conn:
        pgqueue.ensure_schema(conn)   # second run must not raise
        with conn.cursor() as cur:
            cur.execute("SELECT spend_cap_usd, paused FROM fleet_config WHERE id=1")
            row = cur.fetchone()
    assert row["paused"] is False


def test_lease_atomicity_under_concurrency(db):
    M, K = 30, 6
    with pgqueue.connect(db) as conn:
        _seed_queued(conn, M)

    grabbed: list[str] = []
    lock = threading.Lock()

    def worker(wid):
        local = []
        with pgqueue.connect(db) as c:
            while True:
                job = pgqueue.lease_one(c, f"w{wid}", politeness_seconds=0)
                if job is None:
                    break
                local.append(job["url"])
        with lock:
            grabbed.extend(local)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(K)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every queued row leased exactly once, no double-grab.
    assert len(grabbed) == M
    assert len(set(grabbed)) == M


def test_lease_orders_by_score_desc(db):
    with pgqueue.connect(db) as conn:
        _seed_queued(conn, 5)             # score = i, so url .../4 has the top score
        first = pgqueue.lease_one(conn, "w", politeness_seconds=0)
    assert first["url"].endswith("/4")


def test_lease_one_skips_company_blocklist_match(db):
    with pgqueue.connect(db) as conn:
        pgqueue.push_jobs(conn, [
            {
                "url": "https://jobs.example.com/openai",
                "company": "OpenAI",
                "title": "Strategy",
                "application_url": "https://jobs.ashbyhq.com/openai/1",
                "score": 10.0,
                "apply_domain": "ashbyhq.com",
            },
            {
                "url": "https://jobs.example.com/acme",
                "company": "Acme",
                "title": "Chief of Staff",
                "application_url": "https://boards.greenhouse.io/acme/1",
                "score": 9.0,
                "apply_domain": "greenhouse.io",
            },
        ])
        with conn.cursor() as cur:
            cur.execute("UPDATE apply_queue SET approved_batch='b0'")
        conn.commit()

        job = pgqueue.lease_one(conn, "w", politeness_seconds=0)

    assert job is not None
    assert job["url"] == "https://jobs.example.com/acme"


def test_reclaim_parks_all_stale_leases_never_requeues(db):
    """An expired lease == a hard crash (a clean finish always writes terminal status). Because
    `attempts` is bumped only at lease time it cannot tell a never-launched lease (attempts=1)
    from one that crashed mid-submit (also attempts=1), so EVERY stale lease is parked
    crash_unconfirmed (attempts=99) and NONE is requeued -- the conservative anti-double-submit
    default. (Was test_reclaim_requeues_prelaunch_only, which encoded the now-fixed vector.)"""
    with pgqueue.connect(db) as conn:
        _insert_leased(conn, "u/fresh", owner="dead", attempts=1, apply_error=None,
                       expires_delta_sec=-3600)
        _insert_leased(conn, "u/maybe", owner="dead", attempts=2, apply_error=None,
                       expires_delta_sec=-3600)

        reclaimed = {r["url"]: r["status"] for r in pgqueue.reclaim_stale_leases(conn, grace_seconds=0)}
        assert reclaimed["u/fresh"] == "crash_unconfirmed"
        assert reclaimed["u/maybe"] == "crash_unconfirmed"

        # Neither can ever be re-leased, even with approval stamped.
        with conn.cursor() as cur:
            cur.execute("UPDATE apply_queue SET approved_batch='b0'")
        conn.commit()
        leased_urls = set()
        while (job := pgqueue.lease_one(conn, "w", politeness_seconds=0)):
            leased_urls.add(job["url"])
        assert "u/fresh" not in leased_urls
        assert "u/maybe" not in leased_urls

        with conn.cursor() as cur:
            cur.execute("SELECT url, attempts FROM apply_queue WHERE url IN ('u/fresh','u/maybe')")
            assert all(r["attempts"] == 99 for r in cur.fetchall())
            cur.execute(
                "SELECT url, source, application_tool_calls, result_metadata "
                "FROM apply_result_events WHERE url IN ('u/fresh','u/maybe') "
                "ORDER BY url"
            )
            events = cur.fetchall()
            assert [r["url"] for r in events] == ["u/fresh", "u/maybe"]
            assert all(r["source"] == "watchdog" for r in events)
            assert all(r["application_tool_calls"] is None for r in events)
            assert all(r["result_metadata"]["reclaim_reason"] == "lease_expired" for r in events)


def test_reclaim_never_requeues_crash_at_submit(db):
    """Live incident 2026-06-29: worker home-0 hard-crashed AFTER launching the browser
    and reaching 'Now submitting the application' on
    https://hiring.cafe/viewjob/k8odglehcnugjwyu, but BEFORE writing any terminal status
    -- leaving the row at exactly attempts=1, apply_error=NULL, which is INDISTINGUISHABLE
    from a never-launched lease (attempts is bumped only once, at lease time). Reclaim must
    NOT treat that as 'safe to requeue': re-leasing it would apply a SECOND time under
    Jonathan's name. Per the owner's hard rule (NEVER double-apply), an expired lease defaults
    to crash_unconfirmed and is NEVER re-leasable."""
    url = "https://hiring.cafe/viewjob/k8odglehcnugjwyu"
    with pgqueue.connect(db) as conn:
        _insert_leased(conn, url, owner="home-0", attempts=1, apply_error=None,
                       expires_delta_sec=-3600)

        reclaimed = {r["url"]: r["status"] for r in pgqueue.reclaim_stale_leases(conn, grace_seconds=0)}
        assert reclaimed[url] == "crash_unconfirmed"

        # Even with approval stamped, the possibly-submitted row must NEVER be re-leased.
        with conn.cursor() as cur:
            cur.execute("UPDATE apply_queue SET approved_batch='b0'")
        conn.commit()
        leased = set()
        while (job := pgqueue.lease_one(conn, "w", politeness_seconds=0)):
            leased.add(job["url"])
        assert url not in leased

        with conn.cursor() as cur:
            cur.execute("SELECT attempts FROM apply_queue WHERE url=%s", (url,))
            assert cur.fetchone()["attempts"] == 99


def test_reclaim_skips_unexpired_leases(db):
    with pgqueue.connect(db) as conn:
        _insert_leased(conn, "u/live", owner="w", attempts=1, apply_error=None,
                       expires_delta_sec=+3600)   # lease still valid
        assert pgqueue.reclaim_stale_leases(conn, grace_seconds=0) == []


def test_cost_cap_halts(db):
    with pgqueue.connect(db) as conn:
        assert pgqueue.should_halt(conn) is False          # cap=0 -> no cap
        _seed_queued(conn, 3)
        # attach costs summing to $4.50
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE apply_queue SET est_cost_usd = 1.50, cumulative_cost_usd = 1.50"
            )
        conn.commit()
        pgqueue.set_spend_cap(conn, 5.00)
        assert pgqueue.should_halt(conn) is False           # 4.50 < 5.00
        pgqueue.set_spend_cap(conn, 4.00)
        assert pgqueue.should_halt(conn) is True            # 4.50 >= 4.00
        pgqueue.set_spend_cap(conn, 0)
        pgqueue.set_paused(conn, True)
        assert pgqueue.should_halt(conn) is True            # kill switch


def test_push_is_idempotent_and_preserves_leases(db):
    with pgqueue.connect(db) as conn:
        rows = _seed_queued(conn, 3)
        pgqueue.push_jobs(conn, rows)                        # re-push same rows
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM apply_queue")
            assert cur.fetchone()["n"] == 3                  # no duplicates

        # lease one, then re-push it: the leased row must NOT be reset to queued
        job = pgqueue.lease_one(conn, "w", politeness_seconds=0)
        pgqueue.push_jobs(conn, [r for r in rows if r["url"] == job["url"]])
        with conn.cursor() as cur:
            cur.execute("SELECT status, lease_owner FROM apply_queue WHERE url=%s", (job["url"],))
            r = cur.fetchone()
        assert r["status"] == "leased" and r["lease_owner"] == "w"


def test_pull_is_idempotent(db):
    with pgqueue.connect(db) as conn:
        _seed_queued(conn, 1)
        job = pgqueue.lease_one(conn, "w", politeness_seconds=0)
        pgqueue.write_result(conn, "w", job["url"], status="applied",
                             apply_status="applied", est_cost_usd=0.61, agent_model="sonnet")
        pending = pgqueue.fetch_pending_results(conn)
        assert len(pending) == 1 and pending[0]["url"] == job["url"]

        pgqueue.mark_synced(conn, job["url"])
        assert pgqueue.fetch_pending_results(conn) == []     # second pull is a no-op


def test_write_result_lease_owner_guard(db):
    with pgqueue.connect(db) as conn:
        _seed_queued(conn, 1)
        job = pgqueue.lease_one(conn, "A", politeness_seconds=0)

        # A non-holder (e.g. a worker that grabbed a reclaimed row) must not be able to close it
        assert pgqueue.write_result(conn, "B", job["url"], status="applied") is False
        with conn.cursor() as cur:
            cur.execute("SELECT status, lease_owner FROM apply_queue WHERE url=%s", (job["url"],))
            r = cur.fetchone()
        assert r["status"] == "leased" and r["lease_owner"] == "A"

        # The real holder closes it
        assert pgqueue.write_result(conn, "A", job["url"], status="applied",
                                    est_cost_usd=0.5, agent_model="deepseek-chat") is True
        with conn.cursor() as cur:
            cur.execute("SELECT status, applied_at, est_cost_usd, cumulative_cost_usd, agent_model, lease_owner "
                        "FROM apply_queue WHERE url=%s", (job["url"],))
            r = cur.fetchone()
        assert r["status"] == "applied"
        assert r["applied_at"] is not None
        assert r["lease_owner"] is None
        assert float(r["est_cost_usd"]) == 0.5
        assert float(r["cumulative_cost_usd"]) == 0.5
        assert r["agent_model"] == "deepseek-chat"


def test_write_result_unconditional_cost_zero(db):
    with pgqueue.connect(db) as conn:
        _seed_queued(conn, 1)
        job = pgqueue.lease_one(conn, "w", politeness_seconds=0)
        # CLI reported no cost -> we still write a row with est_cost_usd=0 (cap math stays sane)
        assert pgqueue.write_result(conn, "w", job["url"], status="failed",
                                    apply_error="no_result_line", est_cost_usd=None) is True
        with conn.cursor() as cur:
            cur.execute("SELECT status, est_cost_usd FROM apply_queue WHERE url=%s", (job["url"],))
            r = cur.fetchone()
        assert r["status"] == "failed"
        assert float(r["est_cost_usd"]) == 0.0


# ---------------------------------------------------------------------------
# fleet_sync: home SQLite <-> Postgres bridge
# ---------------------------------------------------------------------------

_JOBS_DDL = """
CREATE TABLE jobs (
    url TEXT PRIMARY KEY, company TEXT, title TEXT, application_url TEXT,
    audit_score REAL, fit_score INTEGER, full_description TEXT, liveness_status TEXT,
    apply_status TEXT, apply_error TEXT, duplicate_of_url TEXT,
    applied_at TEXT, agent_id TEXT, verification_confidence TEXT,
    apply_duration_ms INTEGER, apply_attempts INTEGER DEFAULT 0
);
"""


def _home_sqlite(tmp_path):
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "home.db"))
    conn.row_factory = sqlite3.Row
    conn.executescript(_JOBS_DDL)
    return conn


def _add_job(conn, url, **kw):
    cols = {"url": url, "application_url": url, "audit_score": 8.0,
            "liveness_status": "live", "full_description": "x" * 600}
    cols.update(kw)
    conn.execute(f"INSERT INTO jobs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                 list(cols.values()))
    conn.commit()


def _no_host_filters(monkeypatch):
    for fn in ("is_auth_gated_application", "is_unresolved_aggregator", "is_manual_ats"):
        monkeypatch.setattr(f"applypilot.config.{fn}", lambda u: False)


def test_push_offsite_filters(db, tmp_path, monkeypatch):
    _no_host_filters(monkeypatch)
    sq = _home_sqlite(tmp_path)
    _add_job(sq, "https://boards.greenhouse.io/acme/jobs/1", company="Acme", title="COS")  # eligible
    _add_job(sq, "https://www.linkedin.com/jobs/view/123", audit_score=9.0)         # linkedin -> skip
    _add_job(sq, "https://boards.greenhouse.io/x/jobs/2", apply_status="applied")   # applied -> skip
    _add_job(sq, "https://boards.greenhouse.io/y/jobs/3", audit_score=5.0)          # below floor -> skip
    _add_job(sq, "https://boards.greenhouse.io/thin/jobs/4", full_description="x" * 499)  # too thin -> skip

    with pgqueue.connect(db) as pg:
        assert fleet_sync.push_offsite_jobs(sqlite_conn=sq, pg_conn=pg, score_floor=7) == 1
        with pg.cursor() as cur:
            cur.execute("SELECT url, apply_domain FROM apply_queue")
            rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0]["url"].endswith("/acme/jobs/1")
    assert rows[0]["apply_domain"] == "boards.greenhouse.io"


def test_push_offsite_skips_company_blocklist_matches(db, tmp_path, monkeypatch):
    _no_host_filters(monkeypatch)
    sq = _home_sqlite(tmp_path)
    _add_job(sq, "https://boards.greenhouse.io/acme/jobs/1", company="Acme", title="COS")
    _add_job(sq, "https://boards.greenhouse.io/openai/jobs/2", company="OpenAI", title="Strategy")
    _add_job(sq, "https://hiring.cafe/viewjob/openai-3", company="HiringCafe", title="Ops",
             application_url="https://jobs.ashbyhq.com/openai/3")

    with pgqueue.connect(db) as pg:
        assert fleet_sync.push_offsite_jobs(sqlite_conn=sq, pg_conn=pg, score_floor=7) == 1
        with pg.cursor() as cur:
            cur.execute("SELECT url FROM apply_queue")
            urls = {r["url"] for r in cur.fetchall()}
    assert urls == {"https://boards.greenhouse.io/acme/jobs/1"}


def test_push_drops_auth_gated(db, tmp_path, monkeypatch):
    monkeypatch.setattr("applypilot.config.is_unresolved_aggregator", lambda u: False)
    monkeypatch.setattr("applypilot.config.is_manual_ats", lambda u: False)
    monkeypatch.setattr("applypilot.config.is_auth_gated_application",
                        lambda u: "workday" in (u or ""))
    sq = _home_sqlite(tmp_path)
    _add_job(sq, "https://acme.wd1.myworkdayjobs.com/job/1", audit_score=9.0)  # auth-gated -> skip
    _add_job(sq, "https://jobs.lever.co/acme/2", audit_score=8.0)              # eligible
    with pgqueue.connect(db) as pg:
        assert fleet_sync.push_offsite_jobs(sqlite_conn=sq, pg_conn=pg) == 1
        with pg.cursor() as cur:
            cur.execute("SELECT url FROM apply_queue")
            assert cur.fetchone()["url"].endswith("/acme/2")


def test_pull_maps_applied_and_idempotent(db, tmp_path, monkeypatch):
    _no_host_filters(monkeypatch)
    sq = _home_sqlite(tmp_path)
    url = "https://boards.greenhouse.io/acme/jobs/1"
    _add_job(sq, url)
    with pgqueue.connect(db) as pg:
        fleet_sync.push_offsite_jobs(sqlite_conn=sq, pg_conn=pg)
        # Stamp approval so lease_one (gated on approved_batch IS NOT NULL) can pick this up.
        with pg.cursor() as cur:
            cur.execute("UPDATE apply_queue SET approved_batch='b0'")
        pg.commit()
        job = pgqueue.lease_one(pg, "w", politeness_seconds=0)
        pgqueue.write_result(pg, "w", job["url"], status="applied", apply_status="applied",
                             est_cost_usd=0.6, agent_model="deepseek-chat", apply_duration_ms=42000)
        assert fleet_sync.pull_results(sqlite_conn=sq, pg_conn=pg).get("applied") == 1
        row = sq.execute("SELECT apply_status, applied_at, apply_duration_ms FROM jobs WHERE url=?",
                         (url,)).fetchone()
        assert row["apply_status"] == "applied"
        assert row["applied_at"] is not None
        assert row["apply_duration_ms"] == 42000
        # second pull is a no-op (PG row already stamped synced_to_home_at)
        assert fleet_sync.pull_results(sqlite_conn=sq, pg_conn=pg) == {}


def test_pull_failed_pins_attempts(db, tmp_path, monkeypatch):
    _no_host_filters(monkeypatch)
    sq = _home_sqlite(tmp_path)
    url = "https://jobs.lever.co/acme/9"
    _add_job(sq, url)
    with pgqueue.connect(db) as pg:
        fleet_sync.push_offsite_jobs(sqlite_conn=sq, pg_conn=pg)
        # Stamp approval so lease_one (gated on approved_batch IS NOT NULL) can pick this up.
        with pg.cursor() as cur:
            cur.execute("UPDATE apply_queue SET approved_batch='b0'")
        pg.commit()
        job = pgqueue.lease_one(pg, "w", politeness_seconds=0)
        pgqueue.write_result(pg, "w", job["url"], status="blocked", apply_error="cloudflare")
        fleet_sync.pull_results(sqlite_conn=sq, pg_conn=pg)
        row = sq.execute("SELECT apply_status, apply_error, apply_attempts FROM jobs WHERE url=?",
                         (url,)).fetchone()
        assert row["apply_status"] == "failed"      # blocked -> failed home-side
        assert row["apply_error"] == "cloudflare"
        assert row["apply_attempts"] == 99
