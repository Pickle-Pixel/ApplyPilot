# tests/test_fleet_apply_lane.py
import concurrent.futures as cf
from datetime import datetime, timedelta, timezone

import pytest

from applypilot.apply import pgqueue

pytestmark = pytest.mark.usefixtures("acquisition_admitted")


def _activate_policy(conn, lane="ats"):
    policy = f"test-{lane}-policy"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO fleet_decision_policies (policy_version,lane,status) "
            "VALUES (%s,%s,'active') ON CONFLICT (policy_version) DO UPDATE SET status='active'",
            (policy, lane),
        )
        cur.execute(
            f"UPDATE fleet_config SET {lane}_policy_version=%s, approval_threshold=0 WHERE id=1",
            (policy,),
        )
    conn.commit()
    return policy


def _seed_approved_apply_rows(conn, n, *, batch="b1", start=0, distinct_hosts=False):
    policy = _activate_policy(conn)
    now = datetime.now(timezone.utc)
    with conn.cursor() as cur:
        for i in range(start, start + n):
            host = f"acme-{i}.com" if distinct_hosts else "acme.com"
            cur.execute(
                "INSERT INTO apply_queue (url, application_url, score, status, lane, approved_batch, dedup_key, apply_domain, "
                "decision_id,policy_version,decision_action,qualification_verdict,qualification_score,qualification_floor,"
                "preference_score,outcome_score,final_score,decision_confidence,decision_created_at,decision_expires_at,input_hash) "
                "VALUES (%s,%s,%s,'queued','ats',%s,%s,%s,%s,%s,'apply','qualified',9,7,8,8,%s,.9,%s,%s,%s)",
                (f"u{i}", f"http://{host}/{i}", 9.0 - i*0.01, batch, f"dk{i}", host,
                 f"d{i}", policy, 9.0 - i*0.01, now, now + timedelta(days=1), f"h{i}"))
        conn.commit()


def test_canary_columns_exist_and_default(fleet_db):
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("SELECT canary_enabled, canary_remaining FROM fleet_config WHERE id=1")
        row = cur.fetchone()
    assert row["canary_enabled"] is False
    assert row["canary_remaining"] is None


def test_lease_blocked_when_paused(fleet_db):
    from applypilot.fleet import queue
    with pgqueue.connect(fleet_db) as conn:
        _seed_approved_apply_rows(conn, 1)
        with conn.cursor() as cur:
            cur.execute("UPDATE fleet_config SET paused=TRUE WHERE id=1")
        conn.commit()
        assert queue.lease_apply(conn, "w1", home_ip="1.1.1.1") is None


def test_lease_rejects_prior_browser_or_requeue_evidence(fleet_db):
    """Review-only canary blockers must also hold at the real lease boundary."""
    from applypilot.fleet import queue
    with pgqueue.connect(fleet_db) as conn:
        _seed_approved_apply_rows(conn, 2)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE apply_queue SET apply_error='requeued_by_remediator:usage_limit' "
                "WHERE url='u1'"
            )
            cur.execute(
                "INSERT INTO apply_result_events "
                "(url, queue_name, status, apply_error, application_tool_calls, source) "
                "VALUES ('u0','apply_queue','failed','failed:budget_exhausted',3,'worker')"
            )
            cur.execute("UPDATE fleet_config SET paused=FALSE, ats_paused=FALSE WHERE id=1")
        conn.commit()
        assert queue.lease_apply(conn, "w1", home_ip="1.1.1.1") is None
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM apply_queue ORDER BY url")
            assert [row["status"] for row in cur.fetchall()] == ["queued", "queued"]


def test_canary_caps_total_leases_fleetwide(fleet_db):
    from applypilot.fleet import queue
    with pgqueue.connect(fleet_db) as conn:
        _seed_approved_apply_rows(conn, 5, distinct_hosts=True)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE fleet_config SET ats_apply_mode='canary', "
                "canary_enabled=TRUE, canary_remaining=2, paused=FALSE WHERE id=1"
            )
        conn.commit()
        a = queue.lease_apply(conn, "w1", home_ip="1.1.1.1")
        b = queue.lease_apply(conn, "w2", home_ip="1.1.1.1")
        c = queue.lease_apply(conn, "w3", home_ip="1.1.1.1")
    assert a is not None and b is not None and c is None  # exactly 2 leases
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("SELECT canary_remaining, paused, ats_apply_mode FROM fleet_config WHERE id=1")
        row = cur.fetchone()
        assert row["canary_remaining"] == 0
        assert row["paused"] is False
        assert row["ats_apply_mode"] == "stopped"


def test_canary_atomic_under_concurrency(fleet_db):
    # N concurrent workers, K=1 -> EXACTLY one lease succeeds (no overshoot).
    from applypilot.fleet import queue
    with pgqueue.connect(fleet_db) as conn:
        _seed_approved_apply_rows(conn, 8)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE fleet_config SET ats_apply_mode='canary', "
                "canary_enabled=TRUE, canary_remaining=1, paused=FALSE WHERE id=1"
            )
        conn.commit()

    def _lease(i):
        with pgqueue.connect(fleet_db) as c:
            return queue.lease_apply(c, f"w{i}", home_ip="1.1.1.1") is not None

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_lease, range(8)))
    assert sum(results) == 1  # exactly one of eight workers leased


def test_steady_mode_with_canary_disabled_does_not_decrement(fleet_db):
    from applypilot.fleet import queue
    with pgqueue.connect(fleet_db) as conn:
        _seed_approved_apply_rows(conn, 1)  # steady mode with canary disabled by fixture default
        assert queue.lease_apply(conn, "w1", home_ip="1.1.1.1") is not None
        with conn.cursor() as cur:
            cur.execute("SELECT canary_remaining FROM fleet_config WHERE id=1")
            assert cur.fetchone()["canary_remaining"] is None  # untouched


def test_lease_blocked_when_spend_cap_breached(fleet_db):
    # G5 as a HARD lease guard: cumulative apply spend >= spend_cap_usd -> no lease.
    from applypilot.fleet import queue
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        _seed_approved_apply_rows(conn, 1)
        cur.execute(
            "UPDATE apply_queue SET est_cost_usd = 5.0, cumulative_cost_usd = 5.0 "
            "WHERE url='u0'"
        )  # already-spent row
        # add a second leasable row so the SUM (5.0) is what blocks, not an empty queue
        _seed_approved_apply_rows(conn, 1, start=1)
        cur.execute("UPDATE fleet_config SET spend_cap_usd = 1.0 WHERE id=1")
        conn.commit()
        assert queue.lease_apply(conn, "w1", home_ip="1.1.1.1") is None  # 5.0 >= 1.0 cap


def test_build_apply_loop_wires_apply_fn(fleet_db, monkeypatch):
    monkeypatch.setenv("APPLYPILOT_DB_PATH", "x")  # _setup_apply_env may setdefault
    from applypilot.fleet import apply_worker_main as am
    loop = am.build_apply_loop(dsn=fleet_db, worker_id="w1", home_ip="1.1.1.1", model="sonnet", agent="claude")
    assert loop.role == "apply" and loop.apply_fn is not None


def test_apply_env_sets_base_resume(monkeypatch):
    monkeypatch.delenv("APPLYPILOT_LANE_FILTER", raising=False)
    from applypilot.fleet import apply_worker_main as am
    am._setup_apply_env()
    import os
    assert os.environ.get("APPLYPILOT_BASE_RESUME") == "1"
    assert os.environ.get("APPLYPILOT_LANE_FILTER") == "0"


def test_apply_env_preserves_existing_lane_filter(monkeypatch):
    monkeypatch.setenv("APPLYPILOT_LANE_FILTER", "1")
    from applypilot.fleet import apply_worker_main as am
    am._setup_apply_env()
    import os
    assert os.environ.get("APPLYPILOT_LANE_FILTER") == "1"


def test_apply_worker_main_validates_schema_before_building_loop(monkeypatch):
    from applypilot.apply import pgqueue
    from applypilot.fleet import apply_worker_main as am
    from applypilot.fleet import schema as fleet_schema

    calls = []
    conn = object()

    class _Ctx:
        def __enter__(self):
            calls.append("connect")
            return conn

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        am,
        "enforce_host_identity",
        lambda machine_owner, **_kwargs: calls.append("identity"),
    )
    monkeypatch.setattr(
        pgqueue,
        "connect_runtime",
        lambda dsn, *, worker_id, contract: (
            calls.append(("runtime_identity", worker_id, contract)) or _Ctx()
        ),
    )
    monkeypatch.setattr(
        fleet_schema,
        "ensure_schema_v3",
        lambda c: (_ for _ in ()).throw(AssertionError("remote worker must not run DDL")),
    )
    monkeypatch.setattr(
        fleet_schema,
        "require_apply_result_event_schema",
        lambda c: calls.append(("schema_check", c)),
        raising=False,
    )
    monkeypatch.setattr(
        fleet_schema,
        "require_apply_attempt_schema",
        lambda c: calls.append(("attempt_schema_check", c)),
        raising=False,
    )
    monkeypatch.setattr(am, "build_apply_loop", lambda **kwargs: calls.append("build") or object())
    monkeypatch.setattr(am, "install_stop_handler", lambda: calls.append("install"))
    monkeypatch.setattr(am, "run_apply", lambda conn_factory, loop, **kwargs: calls.append("run") or {})

    assert am.main(["--dsn", "postgresql://fleet", "--worker-id", "w-schema"]) == 0
    assert ("runtime_identity", "w-schema", "apply") in calls

    assert ("schema_check", conn) in calls
    assert calls.index(("schema_check", conn)) < calls.index("build")
    assert calls.index(("schema_check", conn)) < calls.index(("attempt_schema_check", conn))
    assert calls.index(("attempt_schema_check", conn)) < calls.index("build")
    assert calls.index("build") < calls.index("run")


def test_run_apply_idles_when_halted(fleet_db):
    # should_halt True (paused) -> the loop does not lease; it returns after one idle pass.
    from applypilot.apply import pgqueue
    from applypilot.fleet import apply_worker_main as am
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("UPDATE fleet_config SET paused=TRUE WHERE id=1")
        conn.commit()
    ticks = am.run_apply(lambda: pgqueue.connect(fleet_db),
                         am.build_apply_loop(dsn=fleet_db, worker_id="w1", home_ip="1.1.1.1",
                                             model="sonnet", agent="claude"),
                         max_iterations=2, idle_sleep=0)
    assert ticks["halted"] >= 1 and ticks["applied"] == 0


def test_run_apply_acks_restart_even_when_halted(fleet_db):
    from applypilot.apply import pgqueue
    from applypilot.fleet import apply_worker_main as am
    from applypilot.fleet import heartbeat

    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("UPDATE fleet_config SET paused=TRUE WHERE id=1")
        cmd_id = heartbeat.issue_command(conn, "w1", "restart", commit=False)
        conn.commit()

    loop = am.build_apply_loop(
        dsn=fleet_db,
        worker_id="w1",
        home_ip="1.1.1.1",
        model="sonnet",
        agent="claude",
    )
    am.run_apply(lambda: pgqueue.connect(fleet_db), loop, max_iterations=1, idle_sleep=0)

    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("SELECT acked_at FROM remote_commands WHERE id=%s", (cmd_id,))
        assert cur.fetchone()["acked_at"] is not None


def test_run_apply_reresolves_timeout_override_mid_flight(fleet_db, monkeypatch):
    """The Doctor sets agent_timeout_override WHILE a worker is already running. A startup-only
    read would never see it; the per-tick re-resolve must pick up a bump on the next tick and
    raise launcher.AGENT_TIMEOUT_SECONDS (which run_job reads as a module global per job)."""
    monkeypatch.setenv("APPLYPILOT_DB_PATH", "x")
    monkeypatch.setenv("APPLYPILOT_AGENT_TIMEOUT", "300")
    from applypilot.apply import pgqueue, launcher
    from applypilot.fleet import apply_worker_main as am
    # Build the loop at the baseline (no override yet) and pin the launcher global to the default.
    loop = am.build_apply_loop(dsn=fleet_db, worker_id="w1", home_ip="1.1.1.1",
                               model="sonnet", agent="claude")
    monkeypatch.setattr(launcher, "AGENT_TIMEOUT_SECONDS", 300, raising=False)
    # Doctor bumps the override AFTER the loop is built; keep the lane paused so the tick just
    # idles (the re-resolve runs BEFORE the should_halt check, so this is enough to observe it).
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("UPDATE fleet_config SET agent_timeout_override=720, paused=TRUE WHERE id=1")
        conn.commit()
    am.run_apply(lambda: pgqueue.connect(fleet_db), loop, max_iterations=1, idle_sleep=0)
    assert int(launcher.AGENT_TIMEOUT_SECONDS) == 720  # next job sees the raised value


def test_legacy_lease_one_is_disabled(fleet_db):
    from applypilot.apply import pgqueue
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO apply_queue (url, application_url, score, status, lane, apply_domain) "
                    "VALUES ('uone','http://x','9','queued','ats','x.com')")  # approved_batch NULL
        conn.commit()
    with pgqueue.connect(fleet_db) as conn:
        with pytest.raises(RuntimeError, match="canonical"):
            pgqueue.lease_one(conn, "w1", politeness_seconds=0)
