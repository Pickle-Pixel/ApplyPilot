"""Shared pytest fixtures for the distributed fleet v3 tests.

Provides a DISPOSABLE local Postgres (from the ``applypilot-pgtest`` conda env, or
an explicitly supplied test DSN) and a clean-schema ``fleet_db`` fixture. Mirrors the disposable cluster in
tests/test_fleet_pgqueue.py but is kept separate (distinct fixture names) so the
existing pgqueue tests are untouched.
"""

from __future__ import annotations

import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_SRC = (_REPO_ROOT / "src").resolve()
_FLEET_PG_LOGFILE: Path | None = None
_TEST_PG_PASSWORD = "applypilot-disposable-test-only"
DISPOSABLE_DATABASE_COMMENT = "applypilot:disposable-postgres-test-database:v1"
DISPOSABLE_CLUSTER_MARKER_ROLE = "applypilot_disposable_cluster_marker"
DISPOSABLE_CLUSTER_COMMENT_PREFIX = "applypilot:disposable-postgres-test-cluster:v1:"
DISPOSABLE_CLUSTER_MARKER_ENV = "APPLYPILOT_PGTEST_CLUSTER_MARKER"
_FIXED_BRAIN_TEST_ROLES = (
    "fleet_backup",
    "brain_candidate_reader",
    "brain_candidate_writer",
    "brain_status_reader",
    "brain_policy_controller",
    "brain_schema_verifier",
    "brain_schema_migrator",
)


def require_disposable_postgres(conn) -> None:
    """Require operator, database, marker-role, and isolated-cluster proof."""
    if os.environ.get("APPLYPILOT_PGTEST_DISPOSABLE") != "1":
        raise RuntimeError("destructive PostgreSQL tests require APPLYPILOT_PGTEST_DISPOSABLE=1")
    marker = conn.execute(
        "SELECT shobj_description(oid,'pg_database') AS marker FROM pg_database WHERE datname=current_database()"
    ).fetchone()["marker"]
    if marker != DISPOSABLE_DATABASE_COMMENT:
        raise RuntimeError("destructive PostgreSQL tests require the exact disposable database comment")
    cluster_nonce = os.environ.get(DISPOSABLE_CLUSTER_MARKER_ENV, "")
    if not cluster_nonce:
        raise RuntimeError("destructive PostgreSQL tests require the fixture cluster marker nonce")
    cluster_marker = conn.execute(
        "SELECT rolcanlogin,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls,"
        "shobj_description(oid,'pg_authid') AS marker FROM pg_roles WHERE rolname=%s",
        (DISPOSABLE_CLUSTER_MARKER_ROLE,),
    ).fetchone()
    if cluster_marker is None or any(
        cluster_marker[name]
        for name in (
            "rolcanlogin",
            "rolsuper",
            "rolcreatedb",
            "rolcreaterole",
            "rolreplication",
            "rolbypassrls",
        )
    ):
        raise RuntimeError("destructive PostgreSQL tests require the inert cluster marker role")
    expected_cluster_comment = DISPOSABLE_CLUSTER_COMMENT_PREFIX + cluster_nonce
    if cluster_marker["marker"] != expected_cluster_comment:
        raise RuntimeError("destructive PostgreSQL tests require the exact fixture cluster marker comment")
    other_databases = conn.execute(
        "SELECT datname FROM pg_database WHERE datallowconn AND NOT datistemplate "
        "AND datname<>current_database() ORDER BY datname"
    ).fetchall()
    if other_databases:
        raise RuntimeError(
            "cluster-wide cleanup requires an isolated cluster with no other connectable databases: "
            + ", ".join(row["datname"] for row in other_databases)
        )


def _mark_disposable_postgres(dsn: str, *, cluster_nonce: str) -> None:
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        database_name = conn.execute("SELECT current_database() AS name").fetchone()["name"]
        conn.execute(
            sql.SQL("COMMENT ON DATABASE {} IS {}").format(
                sql.Identifier(database_name), sql.Literal(DISPOSABLE_DATABASE_COMMENT)
            )
        )
        conn.execute(
            sql.SQL(
                "CREATE ROLE {} NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
            ).format(sql.Identifier(DISPOSABLE_CLUSTER_MARKER_ROLE))
        )
        conn.execute(
            sql.SQL("COMMENT ON ROLE {} IS {}").format(
                sql.Identifier(DISPOSABLE_CLUSTER_MARKER_ROLE),
                sql.Literal(DISPOSABLE_CLUSTER_COMMENT_PREFIX + cluster_nonce),
            )
        )
        conn.commit()


def _pin_local_src() -> None:
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _LOCAL_SRC]
    sys.path.insert(0, str(_LOCAL_SRC))


_pin_local_src()

psycopg = pytest.importorskip("psycopg")
from psycopg import sql  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402

from applypilot import config  # noqa: E402
from applypilot.apply import pgqueue  # noqa: E402
from applypilot.fleet import schema as fleet_schema  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def pin_local_applypilot_checkout():
    """Keep test imports bound to this worktree rather than an editable sibling."""
    _pin_local_src()
    import applypilot

    assert Path(applypilot.__file__).resolve().is_relative_to(_LOCAL_SRC)


@pytest.fixture
def acquisition_admitted(monkeypatch):
    """Explicitly cross the A1 admission seam for downstream unit behavior tests."""
    from applypilot.fleet import emergency_admission

    def admitted(*_args, **_kwargs):
        return emergency_admission.allow("explicit downstream unit-test admission")

    for name in (
        "launcher_admission",
        "worker_tick_admission",
        "linkedin_worker_admission",
        "linkedin_tick_admission",
        "workday_onboard_admission",
        "workday_rollout_admission",
        "linkedin_home_admission",
        "worker_admission",
    ):
        monkeypatch.setattr(emergency_admission, name, admitted)


# Tables truncated between tests (in addition to apply_queue / fleet_config).
_V3_TABLES = [
    "fleet_worker_principals",
    "fleet_worker_lease_ledger",
    "fleet_worker_blocklist",
    "compute_queue",
    "search_tasks",
    "linkedin_queue",
    "rate_governor",
    "llm_usage",
    "applied_set",
    "answer_bank",
    "auth_challenge",
    "otp_request",
    "inbox_events",
    "inbox_outcomes",
    "workers",
    "worker_heartbeat",
    "poison_jobs",
    "remote_commands",
    "command_acks",
    "fleet_assets",
    "discovered_postings",
    "fleet_knobs",
    "fleet_diagnoses",
    "fleet_console_audit",
    "agent_availability",
    "autotriage_actions",
    "apply_result_events",
    "apply_attempts",
    "fleet_machine_blackout",
]


@pytest.fixture(autouse=True)
def isolate_local_runtime_state(tmp_path, monkeypatch):
    """Keep durable lifecycle interlocks created by tests out of the live runtime."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "applypilot.db")


def _find_pg_bin() -> Path | None:
    cands: list[Path] = []
    if os.environ.get("APPLYPILOT_PGTEST_BIN"):
        cands.append(Path(os.environ["APPLYPILOT_PGTEST_BIN"]))
    conda = shutil.which("conda")
    bases: list[Path] = []
    if conda:
        bases.append(Path(conda).resolve().parent.parent)
    bases += [Path.home() / "anaconda3", Path.home() / "miniconda3"]
    for base in bases:
        cands.append(base / "envs" / "applypilot-pgtest" / "Library" / "bin")  # win
        cands.append(base / "envs" / "applypilot-pgtest" / "bin")  # nix
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
def fleet_pg():
    global _FLEET_PG_LOGFILE

    # CI supplies a disposable service-container DSN. This path deliberately
    # never starts or stops the server: ownership remains with the CI job.
    external_dsn = os.environ.get("APPLYPILOT_PGTEST_DSN")
    if external_dsn:
        with pgqueue.connect_admin(external_dsn) as conn:
            require_disposable_postgres(conn)
        yield external_dsn
        return

    binp = _find_pg_bin()
    if binp is None:
        pytest.skip(
            "applypilot-pgtest Postgres env not found (conda create -n applypilot-pgtest -c conda-forge postgresql)"
        )
    ext = ".exe" if os.name == "nt" else ""
    initdb, pg_ctl = binp / f"initdb{ext}", binp / f"pg_ctl{ext}"
    datadir = Path(tempfile.mkdtemp(prefix="ap_fleetpg_"))
    # Let initdb create the data directory so its service account receives the
    # platform-native ACL instead of inheriting Python's private temp ACL.
    datadir.rmdir()
    logfile = datadir / "server.log"
    pwfile = datadir.parent / f"{datadir.name}.pwfile"
    port = _free_port()
    try:
        pwfile.write_text(_TEST_PG_PASSWORD + "\n", encoding="utf-8")
        subprocess.run(
            [
                str(initdb),
                "-D",
                str(datadir),
                "-U",
                "postgres",
                "-E",
                "UTF8",
                "--auth-local=trust",
                "--auth-host=scram-sha-256",
                f"--pwfile={pwfile}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        pwfile.unlink(missing_ok=True)
        subprocess.run(
            [
                str(pg_ctl),
                "-D",
                str(datadir),
                "-l",
                str(logfile),
                "-o",
                f"-p {port} -c listen_addresses=127.0.0.1 -c fsync=off",
                "-w",
                "-t",
                "30",
                "start",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        pwfile.unlink(missing_ok=True)
        log = logfile.read_text(encoding="utf-8", errors="replace") if logfile.exists() else ""
        shutil.rmtree(datadir, ignore_errors=True)
        details = "\n".join(part for part in (e.stdout, e.stderr, log) if part)
        pytest.skip(f"could not start test Postgres (exit {e.returncode}):\n{details}")

    dsn = f"postgresql://postgres:{_TEST_PG_PASSWORD}@127.0.0.1:{port}/postgres"
    _FLEET_PG_LOGFILE = logfile
    previous_disposable_marker = os.environ.get("APPLYPILOT_PGTEST_DISPOSABLE")
    previous_cluster_marker = os.environ.get(DISPOSABLE_CLUSTER_MARKER_ENV)
    cluster_nonce = secrets.token_hex(32)
    os.environ["APPLYPILOT_PGTEST_DISPOSABLE"] = "1"
    os.environ[DISPOSABLE_CLUSTER_MARKER_ENV] = cluster_nonce
    try:
        _mark_disposable_postgres(dsn, cluster_nonce=cluster_nonce)
        with pgqueue.connect_admin(dsn) as conn:
            require_disposable_postgres(conn)
        yield dsn
    finally:
        if previous_disposable_marker is None:
            os.environ.pop("APPLYPILOT_PGTEST_DISPOSABLE", None)
        else:
            os.environ["APPLYPILOT_PGTEST_DISPOSABLE"] = previous_disposable_marker
        if previous_cluster_marker is None:
            os.environ.pop(DISPOSABLE_CLUSTER_MARKER_ENV, None)
        else:
            os.environ[DISPOSABLE_CLUSTER_MARKER_ENV] = previous_cluster_marker
        _FLEET_PG_LOGFILE = None
        try:
            subprocess.run(
                [str(pg_ctl), "-D", str(datadir), "-m", "immediate", "-w", "stop"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            pass
        shutil.rmtree(datadir, ignore_errors=True)


@pytest.fixture(scope="session")
def fleet_pg_log(fleet_pg):
    """Server log for security probes against the disposable cluster."""
    assert _FLEET_PG_LOGFILE is not None
    return _FLEET_PG_LOGFILE


def _restore_test_public_acls(conn) -> None:
    database_name = conn.execute("SELECT current_database() AS name").fetchone()["name"]
    conn.execute(sql.SQL("ALTER DATABASE {} OWNER TO postgres").format(sql.Identifier(database_name)))
    conn.execute(sql.SQL("REVOKE ALL PRIVILEGES ON DATABASE {} FROM PUBLIC").format(sql.Identifier(database_name)))
    conn.execute(
        sql.SQL("GRANT CONNECT, TEMPORARY ON DATABASE {} TO PUBLIC").format(sql.Identifier(database_name))
    )
    if conn.execute("SELECT 1 FROM pg_namespace WHERE nspname='public'").fetchone() is not None:
        conn.execute("ALTER SCHEMA public OWNER TO pg_database_owner")
        conn.execute("REVOKE ALL PRIVILEGES ON SCHEMA public FROM PUBLIC")
        conn.execute("GRANT USAGE ON SCHEMA public TO PUBLIC")
    conn.commit()


def _drop_fixed_brain_test_roles(conn) -> None:
    require_disposable_postgres(conn)
    for role_name in _FIXED_BRAIN_TEST_ROLES:
        if conn.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,)).fetchone() is None:
            continue
        column_acls = conn.execute(
            "SELECT n.nspname,c.relname,a.attname,acl.privilege_type,grantor.rolname AS grantor "
            "FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "CROSS JOIN LATERAL aclexplode(a.attacl) acl "
            "JOIN pg_roles grantor ON grantor.oid=acl.grantor "
            "WHERE a.attnum>0 AND NOT a.attisdropped "
            "AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s)",
            (role_name,),
        ).fetchall()
        for acl in column_acls:
            conn.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(acl["grantor"])))
            conn.execute(
                sql.SQL("REVOKE {} ({}) ON TABLE {}.{} FROM {} CASCADE").format(
                    sql.SQL(acl["privilege_type"]),
                    sql.Identifier(acl["attname"]),
                    sql.Identifier(acl["nspname"]),
                    sql.Identifier(acl["relname"]),
                    sql.Identifier(role_name),
                )
            )
            conn.execute("RESET ROLE")
        conn.execute(sql.SQL("DROP OWNED BY {} CASCADE").format(sql.Identifier(role_name)))
        conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role_name)))
    conn.commit()


@pytest.fixture
def fleet_db(fleet_pg, monkeypatch):
    """Clean v3 schema for each test; yields the DSN."""
    monkeypatch.setenv("FLEET_PG_DSN", fleet_pg)
    with pgqueue.connect_admin(fleet_pg) as conn:
        require_disposable_postgres(conn)
        _drop_fixed_brain_test_roles(conn)
        _restore_test_public_acls(conn)
        # This compatibility table is created by a few focused console/worker
        # tests with deliberately different column sets. It is not part of the
        # canonical v3 schema, so remove it before each clean-schema test rather
        # than allowing one test's shape to leak into the next.
        conn.execute("DROP TABLE IF EXISTS fleet_desired_state CASCADE;")
        conn.commit()
        fleet_schema.ensure_schema_v3(conn)
        with conn.cursor() as cur:
            cur.execute("TRUNCATE apply_queue CASCADE;")
            for t in _V3_TABLES:
                cur.execute(f"TRUNCATE {t} CASCADE;")
            cur.execute(
                "UPDATE fleet_config SET spend_cap_usd=0, paused=FALSE, "
                "cost_cap_daily_usd=0, cost_cap_total_usd=0, "
                "last_window_roll_at=NULL, agent_timeout_override=NULL, "
                "daily_apply_target=NULL, "
                # Most legacy queue tests focus on governor/dedup/lease behavior,
                # not the canary rail. Put test lanes in explicit steady mode with
                # canaries disarmed; canary-specific tests switch to canary mode.
                "pinned_worker_version=NULL, canary_version=NULL, canary_worker_id=NULL, "
                "ats_apply_mode='steady', canary_enabled=FALSE, canary_remaining=NULL, "
                "linkedin_apply_mode='steady', linkedin_canary_enabled=FALSE, linkedin_canary_remaining=NULL, "
                # Fleet Doctor hardening columns (H1/H2/H5/H8/H18) -- reset per test.
                "ats_paused=FALSE, ats_pause_source=NULL, doctor_budget_day=NULL, "
                "doctor_host_skips_today=0, doctor_pace_actions_today=0, "
                "doctor_last_pass_at=NULL, doctor_pause_armed_at=NULL, "
                "doctor_systemic_streak=0, "
                # DeadMan (autonomous-apply Tasks 1-4) -- reset per test.
                "deadman_alert=NULL, deadman_alert_at=NULL, deadman_hot_streak=0 "
                ", ats_policy_version=NULL, linkedin_policy_version=NULL "
                "WHERE id=1;"
            )
            cur.execute("DELETE FROM fleet_decision_policies;")
        conn.commit()
    runtime_connect = pgqueue.connect

    def connect_disposable_admin(dsn=None, *, autocommit=False):
        resolved = pgqueue.get_dsn(dsn)
        if resolved == fleet_pg:
            return pgqueue.connect_admin(resolved, autocommit=autocommit)
        return runtime_connect(dsn, autocommit=autocommit)

    # Tests using fleet_db are explicitly privileged against a disposable cluster. Keep that
    # administrative fixture path separate while production pgqueue.connect remains fail-closed.
    monkeypatch.setattr(pgqueue, "connect", connect_disposable_admin)
    try:
        yield fleet_pg
    finally:
        with pgqueue.connect_admin(fleet_pg) as conn:
            _drop_fixed_brain_test_roles(conn)
            _restore_test_public_acls(conn)
