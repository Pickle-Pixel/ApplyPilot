# tests/test_codex_bridge.py
import importlib
from applypilot.apply import pgqueue


def test_module_imports_without_dsn(monkeypatch):
    # No DB access at import time: importing with FLEET_PG_DSN unset must not raise.
    monkeypatch.delenv("FLEET_PG_DSN", raising=False)
    mod = importlib.import_module("applypilot.fleet.codex_bridge")
    importlib.reload(mod)
    assert hasattr(mod, "mcp") and hasattr(mod, "main") and hasattr(mod, "_with_conn")


def test_with_conn_errors_when_dsn_unset(monkeypatch):
    monkeypatch.delenv("FLEET_PG_DSN", raising=False)
    from applypilot.fleet import codex_bridge
    out = codex_bridge._with_conn(lambda conn: {"ok": True})
    assert "error" in out and "FLEET_PG_DSN" in out["error"]


def test_with_conn_errors_on_unreachable_db(monkeypatch):
    # A syntactically-valid but dead DSN returns a structured error, not a raise.
    monkeypatch.setenv("FLEET_PG_DSN", "postgresql://postgres@127.0.0.1:1/postgres?connect_timeout=1")
    from applypilot.fleet import codex_bridge
    out = codex_bridge._with_conn(lambda conn: {"ok": True})
    assert "error" in out


def test_with_conn_runs_fn_and_closes(fleet_db, fleet_control_db, monkeypatch):
    monkeypatch.setenv("FLEET_PG_DSN", fleet_control_db)
    from applypilot.fleet import codex_bridge
    out = codex_bridge._with_conn(lambda conn: {"ok": True})
    assert out == {"ok": True}


def test_runtime_fixture_is_tls_nonadmin_and_cannot_mutate_global_controls(fleet_control_db):
    with pgqueue.connect(fleet_control_db) as conn:
        row = conn.execute(
            "SELECT session_user AS session_user,current_user AS current_user,"
            "r.rolsuper,r.rolcreatedb,r.rolcreaterole,r.rolreplication,r.rolbypassrls,"
            "COALESCE(s.ssl,false) AS ssl,"
            "has_table_privilege(current_user,'public.fleet_config','UPDATE') AS can_update_config "
            "FROM pg_catalog.pg_roles r "
            "LEFT JOIN pg_catalog.pg_stat_ssl s ON s.pid=pg_catalog.pg_backend_pid() "
            "WHERE r.rolname=current_user"
        ).fetchone()

    assert row["session_user"] == row["current_user"]
    assert row["session_user"].startswith("test_fleet_control_")
    assert row["ssl"] is True
    assert not any(
        row[name]
        for name in (
            "rolsuper",
            "rolcreatedb",
            "rolcreaterole",
            "rolreplication",
            "rolbypassrls",
            "can_update_config",
        )
    )


def _seed_caps(conn):
    with conn.cursor() as cur:
        cur.execute("UPDATE fleet_config SET paused=FALSE, cost_cap_daily_usd=10, cost_cap_total_usd=100 WHERE id=1")
        cur.execute("INSERT INTO llm_usage (cost_usd, ts) VALUES (3.0, now())")
    conn.commit()


def test_fleet_status_returns_snapshot(fleet_db, fleet_control_db, monkeypatch):
    monkeypatch.setenv("FLEET_PG_DSN", fleet_control_db)
    from applypilot.fleet import codex_bridge
    out = codex_bridge.fleet_status()
    # dashboard_snapshot keys
    for k in ("machines", "governor", "queue_depth", "captcha_backlog", "quarantine", "spend_today"):
        assert k in out


def test_caps_returns_caps_and_spend(fleet_db, fleet_control_db, monkeypatch):
    monkeypatch.setenv("FLEET_PG_DSN", fleet_control_db)
    from applypilot.fleet import codex_bridge
    with pgqueue.connect(fleet_db) as conn:
        _seed_caps(conn)
    out = codex_bridge.caps()
    assert out["paused"] is False
    assert float(out["cost_cap_daily_usd"]) == 10.0
    assert float(out["cost_cap_total_usd"]) == 100.0
    assert float(out["spend_today"]) == 3.0
    assert float(out["spend_total"]) == 3.0


def test_health_report_returns_text(fleet_db, fleet_control_db, monkeypatch):
    monkeypatch.setenv("FLEET_PG_DSN", fleet_control_db)
    from applypilot.fleet import codex_bridge
    with pgqueue.connect(fleet_db) as conn:
        _seed_caps(conn)
    out = codex_bridge.health_report()
    assert "report" in out and "NEEDS YOUR DECISION" in out["report"]


def test_recent_results_merges_lanes_newest_first(fleet_db, fleet_control_db, monkeypatch):
    monkeypatch.setenv("FLEET_PG_DSN", fleet_control_db)
    from applypilot.fleet import codex_bridge
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        # an apply terminal row (older) and a compute terminal row (newer)
        cur.execute("INSERT INTO apply_queue (url, application_url, score, status, company, title, apply_error, updated_at) "
                    "VALUES ('a1','http://x','5','failed','Acme','COS','form_error', now() - interval '2 min')")
        cur.execute("INSERT INTO compute_queue (url, task, status, est_cost_usd, updated_at) "
                    "VALUES ('c1','score','done', 0.01, now())")
        conn.commit()
    out = codex_bridge.recent_results(limit=10)
    rows = out["results"]
    assert [r["lane"] for r in rows] == ["compute", "apply"]   # newest-first
    apply_row = next(r for r in rows if r["lane"] == "apply")
    assert apply_row["url"] == "a1" and apply_row["status"] == "failed"
    assert apply_row["detail"]["apply_error"] == "form_error"
    compute_row = next(r for r in rows if r["lane"] == "compute")
    assert compute_row["detail"]["task"] == "score"


def test_recent_results_caps_limit(fleet_db, fleet_control_db, monkeypatch):
    monkeypatch.setenv("FLEET_PG_DSN", fleet_control_db)
    from applypilot.fleet import codex_bridge
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        for i in range(5):
            cur.execute("INSERT INTO compute_queue (url, task, status, updated_at) "
                        "VALUES (%s,'score','done', now())", (f"c{i}",))
        conn.commit()
    out = codex_bridge.recent_results(limit=3)
    assert len(out["results"]) == 3


def test_challenges_only_unresolved(fleet_db, fleet_control_db, monkeypatch):
    monkeypatch.setenv("FLEET_PG_DSN", fleet_control_db)
    from applypilot.fleet import codex_bridge
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO auth_challenge (url, worker_id, kind, route) VALUES ('u1','w','captcha','offsite')")
        cur.execute("INSERT INTO auth_challenge (url, worker_id, kind, route, resolved_at) "
                    "VALUES ('u2','w','captcha','offsite', now())")
        conn.commit()
    out = codex_bridge.challenges()
    urls = [c["url"] for c in out["challenges"]]
    assert "u1" in urls and "u2" not in urls


def test_restart_worker_enqueues_command(fleet_db, fleet_control_db, monkeypatch):
    monkeypatch.setenv("FLEET_PG_DSN", fleet_control_db)
    from applypilot.fleet import codex_bridge
    out = codex_bridge.restart_worker("wA")
    assert out["action"] == "restart" and out["worker_id"] == "wA"
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("SELECT command FROM remote_commands WHERE worker_id='wA'")
        assert cur.fetchone()["command"] == "restart"


def test_pause_scope_pauses(fleet_db, fleet_control_db, monkeypatch):
    monkeypatch.setenv("FLEET_PG_DSN", fleet_control_db)
    from applypilot.fleet import codex_bridge
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO rate_governor (scope_key, min_gap_seconds) VALUES ('host:z.com',5)")
        conn.commit()
    out = codex_bridge.pause_scope("host:z.com")
    assert out["action"] == "pause" and out["scope_key"] == "host:z.com"
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("SELECT breaker_state FROM rate_governor WHERE scope_key='host:z.com'")
        assert cur.fetchone()["breaker_state"] == "paused"


def test_quarantine_job_one_shot(fleet_db, fleet_control_db, monkeypatch):
    monkeypatch.setenv("FLEET_PG_DSN", fleet_control_db)
    from applypilot.fleet import codex_bridge
    out = codex_bridge.quarantine_job("jX", "wA", "owner-pulled")
    assert out["action"] == "quarantine" and out["url"] == "jX" and out["newly_quarantined"] is True
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("SELECT quarantined_at, crash_count FROM poison_jobs WHERE url='jX'")
        row = cur.fetchone()
        assert row["quarantined_at"] is not None and row["crash_count"] == 0


def test_registry_is_exactly_the_eight_tools():
    from applypilot.fleet import codex_bridge
    names = {t.name for t in codex_bridge.mcp._tool_manager.list_tools()}  # sync, no await
    assert names == {
        "fleet_status", "health_report", "recent_results", "challenges", "caps",
        "restart_worker", "pause_scope", "quarantine_job",
    }


def test_no_denied_op_is_registered():
    from applypilot.fleet import codex_bridge
    names = {t.name for t in codex_bridge.mcp._tool_manager.list_tools()}
    for denied in ("apply", "approve", "resolve_challenge", "set_cost_cap", "unpause",
                   "resume_scope", "set_paused", "query", "execute", "sql"):
        assert denied not in names


def test_bridge_end_to_end(fleet_db, fleet_control_db, monkeypatch):
    monkeypatch.setenv("FLEET_PG_DSN", fleet_control_db)
    from applypilot.fleet import codex_bridge
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO compute_queue (url, task, status, updated_at) VALUES ('c1','score','done', now())")
        cur.execute("INSERT INTO worker_heartbeat (worker_id, role, state, last_beat) VALUES ('w1','compute','idle', now())")
        conn.commit()
    # read: status + recent_results render
    assert "machines" in codex_bridge.fleet_status()
    assert codex_bridge.recent_results()["results"][0]["url"] == "c1"
    # act: pause a scope, see it reflected
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO rate_governor (scope_key, min_gap_seconds) VALUES ('host:e2e',5)")
        conn.commit()
    assert codex_bridge.pause_scope("host:e2e")["action"] == "pause"
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("SELECT breaker_state FROM rate_governor WHERE scope_key='host:e2e'")
        assert cur.fetchone()["breaker_state"] == "paused"
