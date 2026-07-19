from __future__ import annotations

import pytest


_VALID_ENV = {"APPLYPILOT_FLEET_LABEL": "owner-node"}


@pytest.mark.parametrize(
    "marker",
    [
        "RAILWAY_PROJECT_ID",
        "RAILWAY_ENVIRONMENT_ID",
        "RAILWAY_SERVICE_ID",
        "RAILWAY_DEPLOYMENT_ID",
    ],
)
def test_browser_host_guard_rejects_railway_runtime_markers(marker):
    from applypilot.fleet.browser_host_guard import require_enrolled_browser_host

    env = {**_VALID_ENV, marker: "present"}
    with pytest.raises(SystemExit, match="Railway"):
        require_enrolled_browser_host(
            machine_owner="owner-node",
            public_ip="1.1.1.1",
            env=env,
        )


@pytest.mark.parametrize(
    "public_ip",
    [
        "",
        "railway",
        "0.0.0.0",
        "::",
        "unknown",
        "10.0.0.1",
        "192.168.1.10",
        "100.64.0.1",
        "203.0.113.1",
    ],
)
def test_browser_host_guard_rejects_placeholder_public_ip(public_ip):
    from applypilot.fleet.browser_host_guard import require_enrolled_browser_host

    with pytest.raises(SystemExit, match="public IP"):
        require_enrolled_browser_host(
            machine_owner="owner-node",
            public_ip=public_ip,
            env=_VALID_ENV,
        )


def test_browser_host_guard_allows_matching_enrolled_node():
    from applypilot.fleet.browser_host_guard import require_enrolled_browser_host

    require_enrolled_browser_host(
        machine_owner=" Owner-Node ",
        public_ip="1.1.1.1",
        env=_VALID_ENV,
    )


def test_linkedin_guard_requires_owner_public_ip_match():
    from applypilot.fleet.browser_host_guard import require_linkedin_owner_host

    with pytest.raises(SystemExit, match="owner IP"):
        require_linkedin_owner_host(
            machine_owner="owner-node",
            public_ip="1.1.1.1",
            owner_ip="8.8.8.8",
            env=_VALID_ENV,
        )


def test_linkedin_guard_allows_matching_owner_node_and_ip():
    from applypilot.fleet.browser_host_guard import require_linkedin_owner_host

    require_linkedin_owner_host(
        machine_owner="owner-node",
        public_ip="1.1.1.1",
        owner_ip="1.1.1.1",
        env=_VALID_ENV,
    )


def test_ats_entrypoint_rejects_railway_before_database_or_browser(monkeypatch):
    from applypilot.apply import pgqueue
    from applypilot.fleet import apply_worker_main as worker

    monkeypatch.setenv("APPLYPILOT_FLEET_LABEL", "m2")
    monkeypatch.setenv("RAILWAY_PROJECT_ID", "railway-project")
    monkeypatch.setattr(
        pgqueue,
        "connect",
        lambda _dsn: pytest.fail("database reached before browser-host guard"),
    )
    monkeypatch.setattr(
        worker,
        "build_apply_loop",
        lambda **_kwargs: pytest.fail("browser/LLM initialization reached before guard"),
    )

    with pytest.raises(SystemExit, match="Railway"):
        worker.main(
            [
                "--dsn",
                "postgresql://unused.invalid/fleet",
                "--worker-id",
                "m2-0",
                "--machine-owner",
                "m2",
                "--home-ip",
                "1.1.1.1",
            ]
        )


def test_ats_entrypoint_rejects_placeholder_ip_before_database_or_browser(monkeypatch):
    from applypilot.apply import pgqueue
    from applypilot.fleet import apply_worker_main as worker

    monkeypatch.setenv("APPLYPILOT_FLEET_LABEL", "m2")
    monkeypatch.setattr(
        pgqueue,
        "connect",
        lambda _dsn: pytest.fail("database reached before browser-host guard"),
    )
    monkeypatch.setattr(
        worker,
        "build_apply_loop",
        lambda **_kwargs: pytest.fail("browser/LLM initialization reached before guard"),
    )

    with pytest.raises(SystemExit, match="public IP"):
        worker.main(
            [
                "--dsn",
                "postgresql://unused.invalid/fleet",
                "--worker-id",
                "m2-0",
                "--machine-owner",
                "m2",
                "--home-ip",
                "railway",
            ]
        )


def test_linkedin_entrypoint_rejects_railway_before_setup_database_or_browser(monkeypatch):
    from applypilot.apply import pgqueue
    from applypilot.fleet import linkedin_worker_main as worker

    monkeypatch.setenv("APPLYPILOT_FLEET_LABEL", "owner-node")
    monkeypatch.setenv("RAILWAY_SERVICE_ID", "railway-service")
    monkeypatch.setattr(
        worker,
        "_setup_apply_env",
        lambda: pytest.fail("profile environment initialized before browser-host guard"),
    )
    monkeypatch.setattr(
        pgqueue,
        "connect",
        lambda _dsn: pytest.fail("database reached before browser-host guard"),
    )
    monkeypatch.setattr(
        worker,
        "build_linkedin_loop",
        lambda **_kwargs: pytest.fail("browser/LLM initialization reached before guard"),
    )

    with pytest.raises(SystemExit, match="Railway"):
        worker.main(
            [
                "--dsn",
                "postgresql://unused.invalid/fleet",
                "--worker-id",
                "owner-linkedin-0",
                "--machine-owner",
                "owner-node",
                "--public-ip",
                "1.1.1.1",
                "--owner-ip",
                "1.1.1.1",
            ]
        )


def test_linkedin_entrypoint_rejects_placeholder_ip_before_setup_or_database(monkeypatch):
    from applypilot.apply import pgqueue
    from applypilot.fleet import linkedin_worker_main as worker

    monkeypatch.setenv("APPLYPILOT_FLEET_LABEL", "owner-node")
    monkeypatch.setattr(
        worker,
        "_setup_apply_env",
        lambda: pytest.fail("profile environment initialized before browser-host guard"),
    )
    monkeypatch.setattr(
        pgqueue,
        "connect",
        lambda _dsn: pytest.fail("database reached before browser-host guard"),
    )

    with pytest.raises(SystemExit, match="public IP"):
        worker.main(
            [
                "--dsn",
                "postgresql://unused.invalid/fleet",
                "--worker-id",
                "owner-linkedin-0",
                "--machine-owner",
                "owner-node",
                "--public-ip",
                "railway",
                "--owner-ip",
                "railway",
            ]
        )


def test_linkedin_entrypoint_rejects_control_plane_owner_mismatch_before_lock_or_browser(
    monkeypatch,
):
    from applypilot.apply import pgqueue
    from applypilot.fleet import emergency_admission, linkedin_worker_main as worker, schema

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, _statement, _params=None):
            return None

        def fetchone(self):
            return {"snapshot": {"linkedin_owner_ip": "8.8.8.8"}}

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return Cursor()

        def rollback(self):
            return None

    from applypilot.fleet import pg_roles

    monkeypatch.setenv("APPLYPILOT_FLEET_LABEL", "owner-node")
    monkeypatch.setattr(worker, "_setup_apply_env", lambda: None)
    # main() reaches the DB through pgqueue.connect_runtime(), which passes autocommit into
    # pgqueue.connect() and authenticates the mapped worker principal before returning. Accept
    # the production keyword and neutralize the principal check; the host boundary is the seam
    # under test.
    monkeypatch.setattr(pgqueue, "connect", lambda _dsn, *, autocommit=False: Connection())
    monkeypatch.setattr(
        pg_roles, "validate_runtime_principal",
        lambda _conn, *, worker_id, contract: None,
    )
    monkeypatch.setattr(emergency_admission, "linkedin_worker_admission", lambda _conn: object())
    monkeypatch.setattr(emergency_admission, "require_allowed", lambda _decision: None)
    monkeypatch.setattr(schema, "require_apply_result_event_schema", lambda _conn: None)
    monkeypatch.setattr(schema, "require_apply_attempt_schema", lambda _conn: None)
    monkeypatch.setattr(
        worker,
        "acquire_linkedin_interlock",
        lambda _conn: pytest.fail("advisory lock reached with mismatched owner IP"),
    )
    monkeypatch.setattr(
        worker,
        "build_linkedin_loop",
        lambda **_kwargs: pytest.fail("browser/LLM initialization reached with mismatched owner IP"),
    )

    with pytest.raises(SystemExit, match="owner IP"):
        worker.main(
            [
                "--dsn",
                "postgresql://unused.invalid/fleet",
                "--worker-id",
                "owner-linkedin-0",
                "--machine-owner",
                "owner-node",
                "--public-ip",
                "1.1.1.1",
                "--owner-ip",
                "1.1.1.1",
            ]
        )


def test_linkedin_entrypoint_allows_valid_owner_through_single_driver_lock(monkeypatch):
    from applypilot.apply import pgqueue
    from applypilot.fleet import emergency_admission, linkedin_worker_main as worker, schema

    calls = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, _statement, _params=None):
            return None

        def fetchone(self):
            return {"snapshot": {"linkedin_owner_ip": "1.1.1.1"}}

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return Cursor()

        def rollback(self):
            return None

        def close(self):
            calls.append("lock-closed")

    from applypilot.fleet import pg_roles

    monkeypatch.setenv("APPLYPILOT_FLEET_LABEL", "owner-node")
    monkeypatch.setattr(worker, "_setup_apply_env", lambda: calls.append("setup"))
    # connect_runtime() passes autocommit into pgqueue.connect() and authenticates the mapped
    # worker principal before returning; accept the keyword and neutralize the principal check.
    monkeypatch.setattr(pgqueue, "connect", lambda _dsn, *, autocommit=False: Connection())
    monkeypatch.setattr(
        pg_roles, "validate_runtime_principal",
        lambda _conn, *, worker_id, contract: None,
    )
    monkeypatch.setattr(emergency_admission, "linkedin_worker_admission", lambda _conn: object())
    monkeypatch.setattr(emergency_admission, "require_allowed", lambda _decision: None)
    monkeypatch.setattr(schema, "require_apply_result_event_schema", lambda _conn: None)
    monkeypatch.setattr(schema, "require_apply_attempt_schema", lambda _conn: None)
    monkeypatch.setattr(
        worker,
        "acquire_linkedin_interlock",
        lambda _conn: calls.append("lock-acquired") or True,
    )

    def build_loop(**kwargs):
        calls.append(("build", kwargs["public_ip"], kwargs["owner_ip"]))
        return object()

    monkeypatch.setattr(worker, "build_linkedin_loop", build_loop)
    monkeypatch.setattr(worker, "run_linkedin", lambda *_args, **_kwargs: calls.append("run"))

    assert worker.main(
        [
            "--dsn",
            "postgresql://unused.invalid/fleet",
            "--worker-id",
            "owner-linkedin-0",
            "--machine-owner",
            "owner-node",
            "--public-ip",
            "1.1.1.1",
            "--owner-ip",
            "1.1.1.1",
        ]
    ) == 0
    assert ("build", "1.1.1.1", "1.1.1.1") in calls
    assert calls.index("lock-acquired") < calls.index(("build", "1.1.1.1", "1.1.1.1"))
    assert calls[-1] == "lock-closed"
