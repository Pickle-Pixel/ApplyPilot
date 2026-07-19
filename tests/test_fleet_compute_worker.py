# tests/test_fleet_compute_worker.py
from applypilot.apply import pgqueue
from applypilot.fleet import queue
from applypilot.fleet.worker import WorkerLoop


def _factory(dsn):
    return lambda: pgqueue.connect(dsn)


def test_compute_entrypoint_requires_server_admission_before_context(monkeypatch):
    from applypilot.fleet import compute_worker_main as cwm
    from applypilot.fleet import emergency_admission

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        cwm.pgqueue,
        "connect_runtime",
        lambda _dsn, *, worker_id, contract: Conn(),
    )
    monkeypatch.setattr(cwm.fleet_schema, "require_apply_result_event_schema", lambda _conn: None)
    monkeypatch.setattr(cwm.fleet_schema, "require_apply_attempt_schema", lambda _conn: None)
    monkeypatch.setattr(
        cwm,
        "build_compute_loop",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("compute context loaded before admission")
        ),
    )
    monkeypatch.setattr(
        cwm,
        "compute_worker_admission",
        lambda _conn: emergency_admission.deny("compute denied by server"),
        raising=False,
    )

    import pytest

    with pytest.raises(SystemExit, match="compute denied by server"):
        cwm.main(["--dsn", "test", "--worker-id", "worker-a"])


def test_compute_worker_routes_audit_task_and_records_provider(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        queue.push_compute_jobs(conn, [{"url": "c-audit", "task": "audit",
                                        "payload": {"title": "Chief of Staff", "company": "Acme",
                                                    "full_description": "ops", "fit_score": 8}}])
        queue.push_compute_jobs(conn, [{"url": "c-score", "task": "score",
                                        "payload": {"title": "COS", "company": "Acme", "full_description": "ops"}}])

    fns = {
        "audit": lambda payload: ({"task": "audit", "research_decision": "qualified", "status": "done"}, 0.0),
        "score": lambda payload: ({"task": "score", "research_fit_score": 9, "model": "deepseek-v4-flash",
                                   "provider": "deepseek", "status": "done"}, 0.0003),
    }
    loop = WorkerLoop(_factory(fleet_db), "w-c", home_ip="1.1.1.1", role="compute", compute_fns=fns)
    a1 = loop.run_once()
    a2 = loop.run_once()
    assert {a1["action"], a2["action"]} == {"compute_done"}

    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("SELECT status, result FROM compute_queue WHERE url='c-audit'")
        r = cur.fetchone()
        assert r["status"] == "done" and r["result"]["research_decision"] == "qualified"
        cur.execute("SELECT provider, model FROM llm_usage WHERE task='score'")
        u = cur.fetchone()
        assert u["provider"] == "deepseek" and u["model"] == "deepseek-v4-flash"


def test_build_compute_loop_wires_both_handlers(fleet_db):
    from applypilot.fleet import compute_context as cc
    from applypilot.fleet import compute_worker_main as cwm
    from applypilot.apply import pgqueue
    with pgqueue.connect(fleet_db) as conn:
        cc.publish_context(conn, resume_text="R", preference_profile={}, kg_prompt="KG",
                           search_cfg={}, version="v1")
    with pgqueue.connect(fleet_db) as conn:
        loop, version = cwm.build_compute_loop(conn, dsn=fleet_db, worker_id="w1", home_ip="1.1.1.1",
                                               providers=["deepseek"], fallback=[], ensemble=False)
    assert set(loop.compute_fns) == {"score", "audit"} and loop.role == "compute"
    assert version == "v1"


def test_build_compute_loop_rejects_missing_resume_context(fleet_db):
    from applypilot.fleet import compute_worker_main as cwm
    from applypilot.apply import pgqueue

    with pgqueue.connect(fleet_db) as conn:
        try:
            cwm.build_compute_loop(conn, dsn=fleet_db, worker_id="w-empty", home_ip="1.1.1.1",
                                   providers=["deepseek"], fallback=[], ensemble=False)
        except RuntimeError as exc:
            assert "ctx:resume" in str(exc)
        else:
            raise AssertionError("missing ctx:resume must fail before any compute job is scored")


def test_maybe_refresh_context_rebuilds_on_version_change(fleet_db):
    from applypilot.fleet import compute_context as cc
    from applypilot.fleet import compute_worker_main as cwm
    from applypilot.apply import pgqueue

    # Publish v1
    with pgqueue.connect(fleet_db) as conn:
        cc.publish_context(conn, resume_text="R1", preference_profile={}, kg_prompt="KG1",
                           search_cfg={}, version="v1")

    # Build loop — captures v1
    with pgqueue.connect(fleet_db) as conn:
        loop, v1 = cwm.build_compute_loop(conn, dsn=fleet_db, worker_id="w-refresh",
                                          home_ip="1.1.1.1", providers=["deepseek"],
                                          fallback=[], ensemble=False)
    assert v1 == "v1"
    original_fns = loop.compute_fns

    # Publish v2 (updated resume)
    with pgqueue.connect(fleet_db) as conn:
        cc.publish_context(conn, resume_text="R2", preference_profile={}, kg_prompt="KG2",
                           search_cfg={}, version="v2")

    # maybe_refresh_context should detect the version bump and rebuild compute_fns
    with pgqueue.connect(fleet_db) as conn:
        v_after = cwm.maybe_refresh_context(conn, loop, current_version="v1",
                                            providers=["deepseek"], fallback=[], ensemble=False)
    assert v_after == "v2", f"expected 'v2', got {v_after!r}"
    assert loop.compute_fns is not original_fns, "compute_fns should have been replaced (rebuilt)"
    assert set(loop.compute_fns) == {"score", "audit"}

    # Calling again with current_version already at v2 must be a no-op (same dict identity)
    fns_after_rebuild = loop.compute_fns
    with pgqueue.connect(fleet_db) as conn:
        v_noop = cwm.maybe_refresh_context(conn, loop, current_version="v2",
                                           providers=["deepseek"], fallback=[], ensemble=False)
    assert v_noop == "v2"
    assert loop.compute_fns is fns_after_rebuild, "no-op refresh must not replace compute_fns"
