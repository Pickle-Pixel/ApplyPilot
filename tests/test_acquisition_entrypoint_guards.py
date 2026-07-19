from __future__ import annotations

import pytest


def test_linkedin_worker_startup_denies_after_server_admission_before_schema(monkeypatch):
    from applypilot.fleet import linkedin_worker_main as worker
    from applypilot.apply import pgqueue
    from applypilot.fleet import schema

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def execute(self, statement, params=None):
            assert "fleet_worker_admission_snapshot" in statement
        def fetchone(self):
            return {"snapshot": {"contract": "linkedin", "admission_allowed": False,
                                  "admission_reason": "linkedin_stopped"}}

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def cursor(self): return Cursor()
        def rollback(self): return None

    monkeypatch.setattr(worker, "_setup_apply_env", lambda: None)
    monkeypatch.setenv("APPLYPILOT_FLEET_LABEL", "home")
    # main() connects through pgqueue.connect_runtime(), which passes autocommit into
    # pgqueue.connect() and then authenticates the mapped worker principal. Accept the
    # production keyword and neutralize the principal check so the admission gate under
    # test is the seam that actually denies.
    monkeypatch.setattr(pgqueue, "connect", lambda _dsn, *, autocommit=False: Connection())
    from applypilot.fleet import pg_roles

    monkeypatch.setattr(
        pg_roles, "validate_runtime_principal",
        lambda _conn, *, worker_id, contract: None,
    )
    monkeypatch.setattr(
        schema, "require_apply_result_event_schema",
        lambda *_args: pytest.fail("schema validation reached after denied admission"),
    )

    with pytest.raises(SystemExit, match="linkedin_stopped"):
        worker.main(
            [
                "--dsn",
                "postgresql://unused.invalid/fleet",
                "--worker-id",
                "owner-1",
                "--machine-owner",
                "home",
                "--public-ip",
                "1.1.1.1",
                "--owner-ip",
                "1.1.1.1",
            ]
        )


def test_linkedin_worker_tick_denies_from_control_contract_before_lease():
    from applypilot.fleet import linkedin_worker_main as worker

    class Loop:
        def run_once(self):
            pytest.fail("LinkedIn lease reached before tick admission")

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def execute(self, statement, params=None):
            assert "fleet_worker_admission_snapshot" in statement
        def fetchone(self):
            return {"snapshot": {"contract": "linkedin", "admission_allowed": False,
                                  "admission_reason": "linkedin_stopped"}}

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def cursor(self): return Cursor()
        def rollback(self): return None

    result = worker.run_linkedin(
        lambda: Connection(),
        Loop(),
        max_iterations=1,
        idle_sleep=0,
    )

    assert result == {"applied": 0, "halted": 1, "idle": 0, "error": 0}


def test_workday_onboard_denies_before_profile_database_or_browser(monkeypatch):
    from applypilot.fleet import workday_onboard_main as onboard

    monkeypatch.setattr(
        onboard.config,
        "load_profile",
        lambda: pytest.fail("profile loaded before startup admission"),
    )
    monkeypatch.setattr(
        onboard,
        "launch_chrome",
        lambda *_args, **_kwargs: pytest.fail("Chrome launched before startup admission"),
    )
    monkeypatch.setattr("sys.argv", ["applypilot-workday-onboard"])

    with pytest.raises(SystemExit, match="emergency acquisition hold"):
        onboard.main()


@pytest.mark.parametrize("stage", ["shadow", "prepare", "authorize", "canary"])
def test_workday_rollout_mutation_stages_deny_before_database_or_browser(monkeypatch, stage):
    from applypilot.fleet import workday_rollout_main as rollout

    monkeypatch.setattr(
        rollout.config,
        "get_connection",
        lambda: pytest.fail("database reached before rollout admission"),
        raising=False,
    )
    monkeypatch.setattr(
        rollout,
        "launch_chrome",
        lambda *_args, **_kwargs: pytest.fail("Chrome launched before rollout admission"),
    )
    monkeypatch.setattr("sys.argv", ["applypilot-workday-rollout", stage])

    with pytest.raises(SystemExit, match="emergency acquisition hold"):
        rollout.main()


def test_workday_tenant_launch_rechecks_admission_before_browser(monkeypatch):
    from applypilot.fleet import workday_rollout_main as rollout

    monkeypatch.setattr(
        rollout.tenant_sessions,
        "select_session",
        lambda _host: {"state": "ready", "profile_dir": "unused"},
    )
    monkeypatch.setattr(
        rollout,
        "launch_chrome",
        lambda *_args, **_kwargs: pytest.fail("Chrome launched before per-launch admission"),
    )

    execute = rollout._tenant_executor({}, headless=True)
    with pytest.raises(SystemExit, match="emergency acquisition hold"):
        execute({"target_host": "acme.myworkdayjobs.com", "url": "https://example.invalid"}, submit=False)


def test_linkedin_home_mutation_denies_before_connection_or_schema(monkeypatch):
    from applypilot.fleet import linkedin_home_main as home

    monkeypatch.setattr(
        home.pgqueue,
        "connect",
        lambda _dsn: pytest.fail("connection reached before command admission"),
    )

    with pytest.raises(SystemExit, match="emergency acquisition hold"):
        home.main(["--dsn", "postgresql://unused.invalid/fleet", "push"])


def test_linkedin_home_status_uses_read_only_schema_verification(fleet_db, monkeypatch):
    from applypilot.fleet import linkedin_home_main as home
    from applypilot.fleet import schema

    monkeypatch.setattr(
        schema,
        "ensure_schema_v3",
        lambda *_args, **_kwargs: pytest.fail("LinkedIn status attempted schema DDL"),
    )

    assert home.main(["--dsn", fleet_db, "status"]) == 0
