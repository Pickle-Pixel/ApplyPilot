"""Authenticated tests for the function-only fleet worker database boundary."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from itertools import count
from pathlib import Path
import tempfile

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row

from applypilot.apply import pgqueue
from applypilot.fleet import pg_roles, queue
from applypilot.fleet.worker import WorkerLoop
from conftest import require_disposable_postgres


ROLE = "fleet_worker_test"
WORKER = "worker-role-test"
PASSWORD = "applypilot-disposable-worker-only"
CONTROLLER_ROLE = "fleet_controller_test"
VERIFIER_ROLE = "fleet_verifier_test"
DATABASE_OWNER_ROLE = "fleet_database_owner_test"
MIGRATOR_ROLE = "fleet_migrator_test"
RETIRED_ADMIN_ROLE = "fleet_provider_admin_test"
CONTROLLER_PASSWORD = "applypilot-disposable-controller-only"
BACKUP_PASSWORD = "applypilot-disposable-backup-only"


_EVIDENCE_TEMP = tempfile.TemporaryDirectory(prefix="applypilot-role-evidence-")
_EVIDENCE_KEY = b"applypilot-test-rollback-hmac-key-v1"
_EVIDENCE_KEY_ID = "pytest-v1"
_EVIDENCE_COUNTER = count()


def _test_evidence_paths() -> pg_roles.DurableEvidencePaths:
    sequence = next(_EVIDENCE_COUNTER)
    root = Path(_EVIDENCE_TEMP.name).resolve()
    return pg_roles.DurableEvidencePaths(
        preparation_receipt_path=root / f"prepared-{sequence}.json",
        rollback_sql_path=root / f"rollback-{sequence}.sql",
        authentication_key=_EVIDENCE_KEY,
        authentication_key_id=_EVIDENCE_KEY_ID,
    )


def _manifest(
    *,
    expected_services: tuple[str, ...] = ("postgres",),
    regrants: tuple[pg_roles.AclRegrant, ...] = (),
) -> pg_roles.RegrantManifest:
    return pg_roles.RegrantManifest(
        database_owner_role=DATABASE_OWNER_ROLE,
        controller_roles=(CONTROLLER_ROLE,),
        verifier_roles=(VERIFIER_ROLE,),
        retired_admin_roles=(RETIRED_ADMIN_ROLE,),
        migrator_role=MIGRATOR_ROLE,
        infrastructure_superuser_roles=("postgres",),
        expected_service_roles=expected_services,
        regrants=regrants,
    )


def _dsn(base: str, *, password: str = PASSWORD) -> str:
    params = conninfo_to_dict(base)
    params.update(user=ROLE, password=password)
    return make_conninfo(**params)


def _controller_dsn(base: str) -> str:
    params = conninfo_to_dict(base)
    params.update(user=CONTROLLER_ROLE, password=CONTROLLER_PASSWORD)
    return make_conninfo(**params)


def _drop_role(conn) -> None:
    require_disposable_postgres(conn)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("ALTER DATABASE {} OWNER TO postgres").format(
                sql.Identifier(conninfo_to_dict(conn.info.dsn)["dbname"])
            )
        )
        cur.execute("SELECT 1 FROM pg_catalog.pg_roles WHERE rolname=%s", (MIGRATOR_ROLE,))
        if cur.fetchone():
            cur.execute(sql.SQL("REASSIGN OWNED BY {} TO postgres").format(sql.Identifier(MIGRATOR_ROLE)))
        cur.execute("DELETE FROM public.fleet_worker_principals WHERE role_name=%s", (ROLE,))
        for transient_role in (
            "fleet_worker_grantee_child",
            "fleet_worker_grantee_approved",
            "fleet_worker_child",
        ):
            cur.execute("SELECT 1 FROM pg_catalog.pg_roles WHERE rolname=%s", (transient_role,))
            if cur.fetchone():
                cur.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(transient_role)))
                cur.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(transient_role)))
        cur.execute("SELECT 1 FROM pg_catalog.pg_roles WHERE rolname=%s", (ROLE,))
        if cur.fetchone():
            cur.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(ROLE)))
            cur.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(ROLE)))
        cur.execute("SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='fleet_worker_inherited'")
        if cur.fetchone():
            cur.execute("DROP OWNED BY fleet_worker_inherited")
            cur.execute("DROP ROLE fleet_worker_inherited")
        for support_role in (
            pg_roles.FLEET_BACKUP_ROLE,
            CONTROLLER_ROLE,
            VERIFIER_ROLE,
            DATABASE_OWNER_ROLE,
            MIGRATOR_ROLE,
            RETIRED_ADMIN_ROLE,
        ):
            cur.execute("SELECT 1 FROM pg_catalog.pg_roles WHERE rolname=%s", (support_role,))
            if cur.fetchone():
                cur.execute(sql.SQL("DROP OWNED BY {} ").format(sql.Identifier(support_role)))
                cur.execute(sql.SQL("DROP ROLE {} ").format(sql.Identifier(support_role)))
    conn.commit()


@pytest.fixture(autouse=True)
def worker_role(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        _drop_role(conn)
        conn.execute(sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER").format(sql.Identifier(DATABASE_OWNER_ROLE)))
        conn.execute(sql.SQL("CREATE ROLE {} NOLOGIN NOSUPERUSER").format(sql.Identifier(MIGRATOR_ROLE)))
        conn.execute(
            sql.SQL("CREATE ROLE {} NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS").format(
                sql.Identifier(RETIRED_ADMIN_ROLE)
            )
        )
        conn.execute(
            sql.SQL(
                "CREATE ROLE {} LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE "
                "NOREPLICATION NOBYPASSRLS PASSWORD {}"
            ).format(sql.Identifier(CONTROLLER_ROLE), sql.Literal(CONTROLLER_PASSWORD))
        )
        conn.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(VERIFIER_ROLE)))
        conn.execute(
            sql.SQL(
                "CREATE ROLE {} LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE "
                "NOREPLICATION NOBYPASSRLS CONNECTION LIMIT 1 PASSWORD {}"
            ).format(
                sql.Identifier(pg_roles.FLEET_BACKUP_ROLE),
                sql.Literal(BACKUP_PASSWORD),
            )
        )
        conn.execute(
            "INSERT INTO public.workers(worker_id,machine_owner,public_ip,validated,capabilities) "
            "VALUES(%s,'machine-a','1.1.1.1',TRUE,'{\"can_ats\":true}'::jsonb) "
            "ON CONFLICT(worker_id) DO UPDATE SET machine_owner=EXCLUDED.machine_owner,"
            "public_ip=EXCLUDED.public_ip,validated=TRUE,revoked_at=NULL",
            (WORKER,),
        )
        conn.execute(
            "INSERT INTO public.worker_heartbeat(worker_id,machine_owner,home_ip,role,state,sw_version,last_beat) "
            "VALUES(%s,'machine-a','1.1.1.1','apply','idle','v1',now()) "
            "ON CONFLICT(worker_id) DO UPDATE SET last_beat=now(),sw_version='v1'",
            (WORKER,),
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS public.fleet_desired_state("
            "machine_owner text primary key,desired_workers integer,agent text,model text,"
            "generation integer,updated_at timestamptz)"
        )
        conn.execute(
            "INSERT INTO public.fleet_desired_state(machine_owner,desired_workers,agent,model,generation,updated_at) "
            "VALUES('machine-a',1,'codex','test',1,now()) ON CONFLICT(machine_owner) DO UPDATE SET "
            "desired_workers=1,generation=1,updated_at=now()"
        )
        pg_roles._transfer_application_ownership(conn.cursor(), new_owner_role=MIGRATOR_ROLE)
        conn.execute(
            sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                sql.Identifier(conninfo_to_dict(conn.info.dsn)["dbname"]),
                sql.Identifier(DATABASE_OWNER_ROLE),
            )
        )
        conn.execute(
            sql.SQL("REVOKE CONNECT, CREATE, TEMPORARY ON DATABASE {} FROM PUBLIC").format(
                sql.Identifier(conninfo_to_dict(conn.info.dsn)["dbname"])
            )
        )
        conn.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                sql.Identifier(conninfo_to_dict(conn.info.dsn)["dbname"]),
                sql.Identifier(CONTROLLER_ROLE),
            )
        )
        conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(CONTROLLER_ROLE)))
        pg_roles._reconcile_backup_role_boundary(
            conn.cursor(),
            topology=pg_roles.BootstrapTopology(
                database_owner_role=DATABASE_OWNER_ROLE,
                controller_role=CONTROLLER_ROLE,
                verifier_role=VERIFIER_ROLE,
                migrator_role=MIGRATOR_ROLE,
                retired_admin_roles=(RETIRED_ADMIN_ROLE,),
            ),
        )
        conn.commit()

        yield
        _drop_role(conn)


def _harden(fleet_db: str, *, password: str = PASSWORD, contract: str = "apply") -> None:
    with pgqueue.connect(fleet_db) as conn:
        conn.execute(
            "UPDATE public.worker_heartbeat SET role=%s,sw_version='v1',last_beat=now() WHERE worker_id=%s",
            (contract, WORKER),
        )
        conn.execute("UPDATE public.fleet_config SET pinned_worker_version='v1',paused=FALSE WHERE id=1")
        conn.commit()
        pg_roles.ensure_fleet_worker_role(
            conn,
            password,
            role=ROLE,
            worker_id=WORKER,
            contract=contract,
            regrant_manifest=_manifest(),
            evidence_paths=_test_evidence_paths(),
        )


def test_transfer_does_not_leak_schema_create_to_controller(fleet_db):
    with psycopg.connect(fleet_db, row_factory=dict_row) as root:
        root.execute(sql.SQL("REVOKE CREATE ON SCHEMA public FROM {}").format(sql.Identifier(MIGRATOR_ROLE)))
        assert (
            root.execute(
                "SELECT has_schema_privilege(%s, 'public', 'CREATE') AS allowed",
                (CONTROLLER_ROLE,),
            ).fetchone()["allowed"]
            is False
        )

        pg_roles._transfer_application_ownership(root.cursor(), new_owner_role=MIGRATOR_ROLE)

        assert (
            root.execute(
                "SELECT has_schema_privilege(%s, 'public', 'CREATE') AS allowed",
                (CONTROLLER_ROLE,),
            ).fetchone()["allowed"]
            is False
        )
        assert (
            root.execute(
                "SELECT has_schema_privilege(%s, 'public', 'CREATE') AS allowed",
                (MIGRATOR_ROLE,),
            ).fetchone()["allowed"]
            is True
        )
        root.commit()


def test_identity_function_does_not_depend_on_controller_migrator_membership(fleet_db):
    _harden(fleet_db)
    with psycopg.connect(_controller_dsn(fleet_db), row_factory=dict_row) as controller:
        metadata = controller.execute(
            "SELECT function_owner.rolname AS function_owner,"
            "table_owner.rolname AS table_owner "
            "FROM pg_proc function "
            "JOIN pg_roles function_owner ON function_owner.oid=function.proowner "
            "JOIN pg_class relation ON relation.oid='public.fleet_worker_principals'::regclass "
            "JOIN pg_roles table_owner ON table_owner.oid=relation.relowner "
            "WHERE function.oid='public.fleet_worker_identity()'::regprocedure"
        ).fetchone()
        assert metadata == {"function_owner": MIGRATOR_ROLE, "table_owner": MIGRATOR_ROLE}
        assert (
            controller.execute(
                "SELECT 1 FROM pg_auth_members membership "
                "JOIN pg_roles parent ON parent.oid=membership.roleid "
                "WHERE membership.member=(SELECT oid FROM pg_roles WHERE rolname=%s) "
                "AND parent.rolname=%s",
                (CONTROLLER_ROLE, MIGRATOR_ROLE),
            ).fetchone()
            is None
        )

    with psycopg.connect(_dsn(fleet_db), row_factory=dict_row) as worker:
        assert worker.execute("SELECT public.fleet_worker_identity() AS identity").fetchone()["identity"] == {
            "role_name": ROLE,
            "worker_id": WORKER,
            "contract": "apply",
        }


def _seed_apply(conn, *, remaining: int = 2, policy: str = "policy-a", suffix: str = "role-boundary") -> str:
    now = datetime.now(timezone.utc)
    url = f"https://example.test/job/{suffix}"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO public.fleet_decision_policies(policy_version,lane,status) "
            "VALUES(%s,'ats','active') ON CONFLICT(policy_version) DO UPDATE SET status='active'",
            (policy,),
        )
        cur.execute(
            "UPDATE public.fleet_config SET paused=FALSE,ats_paused=FALSE,ats_apply_mode='canary',"
            "canary_enabled=TRUE,canary_remaining=%s,ats_policy_version=%s,pinned_worker_version='v1' "
            "WHERE id=1",
            (remaining, policy),
        )
        for scope in ("global", "home_ip:1.1.1.1", "host:example.test"):
            cur.execute(
                "INSERT INTO public.rate_governor(scope_key,min_gap_seconds,daily_cap) "
                "VALUES(%s,0,100) ON CONFLICT(scope_key) DO UPDATE SET min_gap_seconds=0,daily_cap=100,"
                "breaker_state='ok',breaker_until=NULL,count_24h=0,last_attempt_at=NULL,last_applied_at=NULL",
                (scope,),
            )
        cur.execute(
            "INSERT INTO public.apply_queue(url,company,title,application_url,score,apply_domain,target_host,"
            "lane,dedup_key,approved_batch,decision_id,policy_version,decision_action,qualification_verdict,"
            "qualification_score,qualification_floor,preference_score,outcome_score,final_score,"
            "decision_confidence,decision_created_at,decision_expires_at,input_hash) "
            "VALUES(%s,'Co','Role',%s,9,'example.test','example.test','ats',%s,'approved',"
            "%s,%s,'apply','qualified',9,7,9,9,9,.9,%s,%s,%s)",
            (
                url,
                url,
                f"dedup-{suffix}",
                f"decision-{suffix}",
                policy,
                now,
                now + timedelta(days=1),
                f"hash-{suffix}",
            ),
        )
    conn.commit()
    return url


def _worker_factory(fleet_db: str):
    return lambda: psycopg.connect(_dsn(fleet_db), row_factory=dict_row)


def test_authenticated_worker_loop_submission_and_challenge_boundaries(fleet_db):
    with pgqueue.connect(fleet_db) as owner:
        submitted_url = _seed_apply(owner, remaining=3, suffix="runtime-submission")
    _harden(fleet_db)

    forged = WorkerLoop(
        _worker_factory(fleet_db),
        "forged-worker-label",
        home_ip="9.9.9.9",
        role="apply",
        sw_version="forged-version",
        apply_fn=lambda _job: {"run_status": "applied", "application_tool_calls": 1},
        machine_owner="forged-owner",
        on_owner_machine=False,
    ).run_once()
    assert forged == {
        "action": "version_mismatch",
        "expected_version": "v1",
        "sw_version": "forged-version",
    }

    submitted = WorkerLoop(
        _worker_factory(fleet_db),
        "forged-worker-label",
        home_ip="9.9.9.9",
        role="apply",
        sw_version="v1",
        apply_fn=lambda _job: {"run_status": "applied", "application_tool_calls": 1},
        machine_owner="forged-owner",
        on_owner_machine=False,
    ).run_once()
    assert submitted == {"action": "applied", "url": submitted_url}

    with pgqueue.connect(fleet_db) as owner:
        row = owner.execute(
            "SELECT status,apply_status,worker_id,applied_at FROM public.apply_queue WHERE url=%s",
            (submitted_url,),
        ).fetchone()
        assert row == {
            "status": "crash_unconfirmed",
            "apply_status": "submission_claimed_unverified",
            "worker_id": WORKER,
            "applied_at": None,
        }
        challenge_url = _seed_apply(owner, remaining=2, suffix="runtime-challenge")

    challenged = WorkerLoop(
        _worker_factory(fleet_db),
        "another-forged-label",
        home_ip="8.8.8.8",
        role="apply",
        sw_version="v1",
        apply_fn=lambda _job: {"run_status": "captcha", "application_tool_calls": 1},
        machine_owner="forged-owner",
        on_owner_machine=False,
    ).run_once()
    assert challenged == {"action": "parked_challenge", "url": challenge_url}
    with pgqueue.connect(fleet_db) as owner:
        assert owner.execute(
            "SELECT worker_id,kind,resolved_at FROM public.auth_challenge WHERE url=%s",
            (challenge_url,),
        ).fetchone() == {"worker_id": WORKER, "kind": "visible_captcha", "resolved_at": None}


def test_reconciliation_is_function_only_and_removes_public_and_membership(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        conn.execute("CREATE ROLE fleet_worker_inherited")
        conn.execute("CREATE ROLE fleet_worker_child")
        conn.execute("CREATE ROLE fleet_worker_grantee_approved")
        conn.execute("CREATE ROLE fleet_worker_grantee_child")
        conn.execute("GRANT ALL ON ALL TABLES IN SCHEMA public TO fleet_worker_inherited")
        conn.execute(sql.SQL("CREATE ROLE {} LOGIN").format(sql.Identifier(ROLE)))
        conn.execute(sql.SQL("GRANT fleet_worker_inherited TO {} WITH ADMIN OPTION").format(sql.Identifier(ROLE)))
        conn.execute(sql.SQL("GRANT {} TO fleet_worker_child").format(sql.Identifier(ROLE)))
        conn.execute(sql.SQL("GRANT {} TO fleet_worker_grantee_approved").format(sql.Identifier(ROLE)))
        conn.execute("GRANT fleet_worker_grantee_approved TO fleet_worker_grantee_child")
        conn.execute("GRANT SELECT,UPDATE ON public.apply_queue TO PUBLIC")
        conn.commit()
        pg_roles.ensure_fleet_worker_role(
            conn,
            PASSWORD,
            role=ROLE,
            worker_id=WORKER,
            contract="apply",
            regrant_manifest=_manifest(),
            evidence_paths=_test_evidence_paths(),
            approved_grantee_roles=("fleet_worker_grantee_approved",),
        )
        row = conn.execute(
            "SELECT rolcanlogin,rolinherit,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
            "FROM pg_catalog.pg_roles WHERE rolname=%s",
            (ROLE,),
        ).fetchone()
        assert tuple(row.values()) == (True, False, False, False, False, False, False)
        assert (
            conn.execute(
                "SELECT pg_catalog.pg_has_role(%s,'fleet_worker_inherited','MEMBER') AS ok", (ROLE,)
            ).fetchone()["ok"]
            is False
        )
        assert (
            conn.execute(
                "SELECT pg_catalog.pg_has_role('fleet_worker_child',%s,'MEMBER') AS ok", (ROLE,)
            ).fetchone()["ok"]
            is False
        )
        assert (
            conn.execute(
                "SELECT pg_catalog.pg_has_role('fleet_worker_grantee_approved',%s,'MEMBER') AS ok", (ROLE,)
            ).fetchone()["ok"]
            is False
        )
        conn.execute("REVOKE fleet_worker_grantee_approved FROM fleet_worker_grantee_child")
        conn.execute(sql.SQL("GRANT {} TO fleet_worker_grantee_approved").format(sql.Identifier(ROLE)))
        conn.commit()
        pg_roles.ensure_fleet_worker_role(
            conn,
            PASSWORD,
            role=ROLE,
            worker_id=WORKER,
            contract="apply",
            regrant_manifest=_manifest(),
            evidence_paths=_test_evidence_paths(),
            approved_grantee_roles=("fleet_worker_grantee_approved",),
        )
        assert conn.execute(
            "SELECT pg_catalog.pg_has_role('fleet_worker_grantee_approved',%s,'MEMBER') AS ok", (ROLE,)
        ).fetchone()["ok"] is True
        for table in (
            "fleet_config",
            "apply_queue",
            "apply_attempts",
            "applied_set",
            "apply_result_events",
            "rate_governor",
            "workers",
            "worker_heartbeat",
            "auth_challenge",
            "compute_queue",
            "search_tasks",
            "discovered_postings",
            "llm_usage",
        ):
            assert (
                conn.execute(
                    "SELECT pg_catalog.has_table_privilege(%s,%s,'SELECT,INSERT,UPDATE,DELETE') AS ok",
                    (ROLE, f"public.{table}"),
                ).fetchone()["ok"]
                is False
            )
        assert (
            conn.execute(
                "SELECT pg_catalog.has_function_privilege(%s,'public.fleet_worker_lease_ats(text,text,integer,text,integer)','EXECUTE') AS ok",
                (ROLE,),
            ).fetchone()["ok"]
            is True
        )
        assert (
            conn.execute(
                "SELECT pg_catalog.has_function_privilege(%s,'public.fleet_controller_verify_submission(text,text,text,text)','EXECUTE') AS ok",
                (ROLE,),
            ).fetchone()["ok"]
            is False
        )


def test_reconciled_apply_role_executes_schema_budget_and_otp_functions(fleet_db):
    with pgqueue.connect(fleet_db) as owner:
        job_url = _seed_apply(owner, suffix="reconciled-functions")
    _harden(fleet_db, contract="apply")

    with psycopg.connect(_dsn(fleet_db), row_factory=dict_row) as worker:
        schema_contract = worker.execute("SELECT public.fleet_worker_schema_contract() AS value").fetchone()["value"]
        assert schema_contract["contract"] == "apply"
        assert schema_contract["ready"] is True
        assert (
            worker.execute("SELECT public.fleet_worker_evaluate_agent_budget('{}'::jsonb,60,60) AS value").fetchone()[
                "value"
            ]
            == {}
        )

        leased = queue.lease_apply(worker, WORKER, home_ip="1.1.1.1", sw_version="v1")
        assert leased and leased["url"] == job_url
        request_id = worker.execute(
            "SELECT public.fleet_worker_otp_request(%s,%s,%s,300) AS value",
            ("forged-worker", job_url, job_url),
        ).fetchone()["value"]
        assert (
            worker.execute("SELECT public.fleet_worker_otp_wait(%s) AS value", (request_id,)).fetchone()["value"]
            is True
        )
        assert (
            worker.execute("SELECT public.fleet_worker_otp_consume(%s) AS value", (request_id,)).fetchone()["value"]
            is None
        )


def test_reconciled_linkedin_role_gets_schema_and_otp_but_not_apply_budget(fleet_db):
    _harden(fleet_db, contract="linkedin")

    with psycopg.connect(_dsn(fleet_db), row_factory=dict_row, autocommit=True) as worker:
        schema_contract = worker.execute("SELECT public.fleet_worker_schema_contract() AS value").fetchone()["value"]
        assert schema_contract["contract"] == "linkedin"
        assert worker.execute("SELECT public.fleet_worker_otp_wait(0) AS value").fetchone()["value"] is False
        assert worker.execute("SELECT public.fleet_worker_otp_consume(0) AS value").fetchone()["value"] is None
        assert (
            worker.execute(
                "SELECT has_function_privilege(session_user,"
                "'public.fleet_worker_otp_request(text,text,text,integer)','EXECUTE') AS value"
            ).fetchone()["value"]
            is True
        )
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            worker.execute("SELECT public.fleet_worker_evaluate_agent_budget('{}'::jsonb,60,60)")


def test_reconciled_compute_role_gets_schema_but_denies_otp_and_apply_budget(fleet_db):
    _harden(fleet_db, contract="compute")

    denied_calls = (
        "SELECT public.fleet_worker_record_agent_wall('codex',now())",
        "SELECT public.fleet_worker_evaluate_agent_budget('{}'::jsonb,60,60)",
        "SELECT public.fleet_worker_otp_request('worker','job','application',300)",
        "SELECT public.fleet_worker_otp_wait(0)",
        "SELECT public.fleet_worker_otp_consume(0)",
    )
    with psycopg.connect(_dsn(fleet_db), row_factory=dict_row, autocommit=True) as worker:
        schema_contract = worker.execute("SELECT public.fleet_worker_schema_contract() AS value").fetchone()["value"]
        assert schema_contract["contract"] == "compute"
        assert (
            worker.execute(
                "SELECT has_function_privilege(session_user,"
                "'public.fleet_worker_record_agent_wall(text,timestamp with time zone)','EXECUTE') AS value"
            ).fetchone()["value"]
            is False
        )
        for statement in denied_calls:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                worker.execute(statement)


def test_record_agent_wall_sql_requires_an_apply_or_linkedin_mapping() -> None:
    source = (Path(__file__).parents[1] / "src/applypilot/fleet/schema_v3.sql").read_text(encoding="utf-8")
    body = source.split("AS $fleet_worker_record_agent_wall$", 1)[1].split("$fleet_worker_record_agent_wall$;", 1)[0]
    assert "WHERE role_name=session_user AND contract IN ('apply','linkedin')" in body
    assert "session_user<>current_user" not in body


def test_worker_claim_never_creates_canonical_applied_state(fleet_db):
    with pgqueue.connect(fleet_db) as owner:
        url = _seed_apply(owner)
    _harden(fleet_db)
    with psycopg.connect(_dsn(fleet_db), row_factory=dict_row) as worker:
        job = queue.lease_apply(worker, "forged-worker", home_ip="9.9.9.9", sw_version="forged")
        assert job and job["url"] == url
        assert queue.write_apply_result(
            worker,
            "forged-worker",
            url,
            status="applied",
            target_host="evil.test",
            home_ip="9.9.9.9",
            apply_status="applied",
            result_metadata={"claimed": True},
        )
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            worker.execute("INSERT INTO public.applied_set(dedup_key) VALUES('forged')")
    with pgqueue.connect(fleet_db) as owner:
        row = owner.execute(
            "SELECT status,apply_status,applied_at FROM public.apply_queue WHERE url=%s", (url,)
        ).fetchone()
        assert row == {
            "status": "crash_unconfirmed",
            "apply_status": "submission_claimed_unverified",
            "applied_at": None,
        }
        assert owner.execute("SELECT count(*) AS n FROM public.applied_set").fetchone()["n"] == 0
        assertion = owner.execute(
            "SELECT status,evidence_is_assertion FROM public.apply_result_events WHERE url=%s ORDER BY id DESC LIMIT 1",
            (url,),
        ).fetchone()
        assert assertion == {"status": "submission_claimed", "evidence_is_assertion": True}
        assert (
            owner.execute(
                "SELECT public.fleet_controller_verify_submission('ats',%s,'receipt-1','email_receipt') AS ok",
                (url,),
            ).fetchone()["ok"]
            is True
        )
        owner.commit()
        assert owner.execute("SELECT status,apply_status FROM public.apply_queue WHERE url=%s", (url,)).fetchone() == {
            "status": "applied",
            "apply_status": "applied",
        }
        assert owner.execute("SELECT count(*) AS n FROM public.applied_set").fetchone()["n"] == 1


def test_requeue_after_browser_checkpoint_parks_instead_of_recycling(fleet_db):
    with pgqueue.connect(fleet_db) as owner:
        url = _seed_apply(owner)
    _harden(fleet_db)
    with psycopg.connect(_dsn(fleet_db), row_factory=dict_row) as worker:
        assert queue.lease_apply(worker, WORKER, home_ip="1.1.1.1", sw_version="v1")
        worker.execute("SELECT public.fleet_worker_mark_browser_interaction()")
        assert queue.requeue_apply(worker, WORKER, url, apply_error="retry") is False
    with pgqueue.connect(fleet_db) as owner:
        assert owner.execute(
            "SELECT status,apply_status,attempts FROM public.apply_queue WHERE url=%s", (url,)
        ).fetchone() == {"status": "crash_unconfirmed", "apply_status": "verification_pending", "attempts": 1}


def test_policy_drift_refunds_once_and_stops_lane_with_capacity_above_one(fleet_db):
    with pgqueue.connect(fleet_db) as owner:
        url = _seed_apply(owner, remaining=3)
    _harden(fleet_db)
    with psycopg.connect(_dsn(fleet_db), row_factory=dict_row) as worker:
        assert queue.lease_apply(worker, WORKER, home_ip="1.1.1.1", sw_version="v1")
        with pgqueue.connect(fleet_db) as owner:
            owner.execute("UPDATE public.fleet_config SET ats_policy_version=NULL WHERE id=1")
            owner.commit()
        assert queue.requeue_apply(worker, WORKER, url, apply_error="policy drift") is True
        assert queue.requeue_apply(worker, WORKER, url, apply_error="again") is False
    with pgqueue.connect(fleet_db) as owner:
        assert owner.execute(
            "SELECT ats_apply_mode,canary_remaining FROM public.fleet_config WHERE id=1"
        ).fetchone() == {"ats_apply_mode": "stopped", "canary_remaining": 3}


def test_owned_object_fails_closed_and_role_stays_nologin(fleet_db):
    with psycopg.connect(fleet_db, row_factory=dict_row) as break_glass:
        break_glass.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(ROLE)))
        break_glass.execute("CREATE TABLE public.worker_owned_attack(id integer)")
        break_glass.execute(sql.SQL("ALTER TABLE public.worker_owned_attack OWNER TO {}").format(sql.Identifier(ROLE)))
        break_glass.commit()
    with pgqueue.connect(fleet_db) as conn:
        with pytest.raises(RuntimeError, match="table:public.worker_owned_attack"):
            pg_roles.ensure_fleet_worker_role(
                conn,
                PASSWORD,
                role=ROLE,
                worker_id=WORKER,
                contract="apply",
                regrant_manifest=_manifest(),
                evidence_paths=_test_evidence_paths(),
            )
        role = conn.execute("SELECT rolcanlogin FROM pg_catalog.pg_roles WHERE rolname=%s", (ROLE,)).fetchone()
        assert role is None or role["rolcanlogin"] is False
    with psycopg.connect(fleet_db) as break_glass:
        break_glass.execute("ALTER TABLE public.worker_owned_attack OWNER TO postgres")
        break_glass.execute("DROP TABLE public.worker_owned_attack")
        break_glass.commit()


def test_pg_temp_shadow_fails_closed_and_role_stays_nologin(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        conn.execute("CREATE TEMP TABLE fleet_worker_principals(role_name text)")
        conn.commit()
        with pytest.raises(RuntimeError, match="pg_temp object shadowing"):
            pg_roles.ensure_fleet_worker_role(
                conn,
                PASSWORD,
                role=ROLE,
                worker_id=WORKER,
                contract="apply",
                regrant_manifest=_manifest(),
                evidence_paths=_test_evidence_paths(),
            )
        row = conn.execute("SELECT rolcanlogin FROM pg_catalog.pg_roles WHERE rolname=%s", (ROLE,)).fetchone()
        assert row is None or row["rolcanlogin"] is False


def test_concurrent_first_reconciliation_and_password_rotation(fleet_db):
    def reconcile(password: str) -> None:
        with pgqueue.connect(fleet_db) as conn:
            pg_roles.ensure_fleet_worker_role(
                conn,
                password,
                role=ROLE,
                worker_id=WORKER,
                contract="apply",
                regrant_manifest=_manifest(),
                evidence_paths=_test_evidence_paths(),
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(reconcile, ("first-password", "second-password")))
    # Lock order determines the winner. Rotate once more to assert deterministic credentials.
    reconcile(PASSWORD)
    with psycopg.connect(_dsn(fleet_db)):
        pass
    with pytest.raises(psycopg.OperationalError):
        psycopg.connect(_dsn(fleet_db, password="wrong-password"), connect_timeout=3)


def test_missing_principal_mapping_fails_closed(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        with pytest.raises(RuntimeError, match="principal mapping"):
            pg_roles.ensure_fleet_worker_role(
                conn,
                PASSWORD,
                role=ROLE,
                regrant_manifest=_manifest(),
                evidence_paths=_test_evidence_paths(),
            )
        role = conn.execute("SELECT rolcanlogin FROM pg_catalog.pg_roles WHERE rolname=%s", (ROLE,)).fetchone()
        assert role is None or role["rolcanlogin"] is False


def test_precommit_evidence_survives_failure_and_database_rolls_back(fleet_db, tmp_path: Path, monkeypatch):
    receipt_path = tmp_path / "prepared.json"
    rollback_path = tmp_path / "rollback.sql"
    with pgqueue.connect(fleet_db) as conn:
        before = conn.execute(
            "SELECT datacl::text AS acl,owner.rolname AS owner FROM pg_database d "
            "JOIN pg_roles owner ON owner.oid=d.datdba WHERE d.datname=current_database()"
        ).fetchone()
        conn.commit()

        monkeypatch.setattr(
            pg_roles,
            "_install_identity_function",
            lambda _cur: (_ for _ in ()).throw(RuntimeError("injected post-evidence failure")),
        )
        with pytest.raises(RuntimeError, match="injected post-evidence failure"):
            pg_roles.ensure_fleet_worker_role(
                conn,
                PASSWORD,
                role=ROLE,
                worker_id=WORKER,
                contract="apply",
                regrant_manifest=_manifest(),
                evidence_paths=pg_roles.DurableEvidencePaths(
                    preparation_receipt_path=receipt_path.resolve(),
                    rollback_sql_path=rollback_path.resolve(),
                    authentication_key=_EVIDENCE_KEY,
                    authentication_key_id=_EVIDENCE_KEY_ID,
                ),
            )
        assert receipt_path.read_text(encoding="utf-8")
        rollback_bytes = rollback_path.read_bytes()
        assert b"REVOKE CONNECT ON DATABASE" in rollback_bytes
        prepared = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert prepared["status"] == "prepared_before_database_mutation"
        assert prepared["rollback_sql_sha256"] == hashlib.sha256(rollback_bytes).hexdigest()
        after = conn.execute(
            "SELECT datacl::text AS acl,owner.rolname AS owner FROM pg_database d "
            "JOIN pg_roles owner ON owner.oid=d.datdba WHERE d.datname=current_database()"
        ).fetchone()
        assert after == before
        assert conn.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (ROLE,)).fetchone() is None


def test_controller_is_non_superuser_and_retired_admin_has_no_elevated_attributes(fleet_db):
    _harden(fleet_db)
    with pgqueue.connect(fleet_db) as conn:
        controller = conn.execute(
            "SELECT rolinherit,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
            "FROM pg_roles WHERE rolname=%s",
            (CONTROLLER_ROLE,),
        ).fetchone()
        assert controller == {
            "rolinherit": False,
            "rolsuper": False,
            "rolcreatedb": False,
            "rolcreaterole": False,
            "rolreplication": False,
            "rolbypassrls": False,
        }
        retired = conn.execute(
            "SELECT rolcanlogin,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
            "FROM pg_roles WHERE rolname=%s",
            (RETIRED_ADMIN_ROLE,),
        ).fetchone()
        assert not any(retired.values())


def test_controller_cannot_provision_or_assume_owner_and_migrator_roles(fleet_db):
    with psycopg.connect(_controller_dsn(fleet_db), row_factory=dict_row) as controller:
        assert controller.execute(
            "SELECT pg_has_role(current_user,%s,'MEMBER') AS owner_member,"
            "pg_has_role(current_user,%s,'MEMBER') AS migrator_member",
            (DATABASE_OWNER_ROLE, MIGRATOR_ROLE),
        ).fetchone() == {"owner_member": False, "migrator_member": False}
        controller.rollback()
        with pytest.raises(RuntimeError, match="offline provider-admin identity"):
            pg_roles.ensure_fleet_worker_role(
                controller,
                PASSWORD,
                role=ROLE,
                worker_id=WORKER,
                contract="apply",
                regrant_manifest=_manifest(expected_services=("postgres",)),
                evidence_paths=_test_evidence_paths(),
            )


def test_existing_role_mapping_cannot_be_reassigned_to_another_node_or_contract(fleet_db):
    _harden(fleet_db)
    with pgqueue.connect(fleet_db) as conn:
        with pytest.raises(RuntimeError, match="different node identity"):
            pg_roles.ensure_fleet_worker_role(
                conn,
                PASSWORD,
                role=ROLE,
                worker_id="other-node",
                contract="apply",
                regrant_manifest=_manifest(),
                evidence_paths=_test_evidence_paths(),
            )
        with pytest.raises(RuntimeError, match="different contract"):
            pg_roles.ensure_fleet_worker_role(
                conn,
                PASSWORD,
                role=ROLE,
                worker_id=WORKER,
                contract="compute",
                regrant_manifest=_manifest(),
                evidence_paths=_test_evidence_paths(),
            )


def test_manifest_is_required_before_database_wide_acl_changes(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        conn.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO PUBLIC").format(
                sql.Identifier(conninfo_to_dict(fleet_db)["dbname"])
            )
        )
        conn.commit()
        with pytest.raises(RuntimeError, match="explicit regrant manifest"):
            pg_roles.ensure_fleet_worker_role(conn, PASSWORD, role=ROLE, worker_id=WORKER, contract="apply")
        assert (
            conn.execute("SELECT has_database_privilege('public',current_database(),'CONNECT') AS allowed").fetchone()[
                "allowed"
            ]
            is True
        )


def test_connect_acl_is_exact_and_receipt_contains_rollback_sql(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        conn.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO PUBLIC").format(
                sql.Identifier(conninfo_to_dict(fleet_db)["dbname"])
            )
        )
        conn.commit()
        receipt = pg_roles.ensure_fleet_worker_role(
            conn,
            PASSWORD,
            role=ROLE,
            worker_id=WORKER,
            contract="apply",
            regrant_manifest=_manifest(),
            evidence_paths=_test_evidence_paths(),
        )
        acl = conn.execute("SELECT datacl::text AS acl FROM pg_database WHERE datname=current_database()").fetchone()[
            "acl"
        ]
        assert "=Tc/" not in acl
        direct_connect = conn.execute(
            "SELECT array_agg(CASE WHEN acl.grantee=0 THEN 'PUBLIC' ELSE grantee.rolname END "
            "ORDER BY CASE WHEN acl.grantee=0 THEN 'PUBLIC' ELSE grantee.rolname END) AS roles "
            "FROM pg_database d CROSS JOIN LATERAL aclexplode(d.datacl) acl "
            "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
            "WHERE d.datname=current_database() AND acl.privilege_type='CONNECT'"
        ).fetchone()["roles"]
        assert set(direct_connect) == set(receipt.connect_allowlist)
        for principal in (DATABASE_OWNER_ROLE, CONTROLLER_ROLE, VERIFIER_ROLE, ROLE):
            assert (
                conn.execute(
                    "SELECT has_database_privilege(%s,current_database(),'CONNECT') AS allowed",
                    (principal,),
                ).fetchone()["allowed"]
                is True
            )
        assert "GRANT CONNECT ON DATABASE" in receipt.rollback_sql
        assert "PUBLIC" in receipt.rollback_sql
        assert receipt.role_name == ROLE
        assert receipt.worker_id == WORKER
        assert receipt.contract == "apply"
        owner = conn.execute(
            "SELECT owner.rolname,owner.rolcanlogin FROM pg_database d "
            "JOIN pg_roles owner ON owner.oid=d.datdba WHERE d.datname=current_database()"
        ).fetchone()
        assert owner == {"rolname": DATABASE_OWNER_ROLE, "rolcanlogin": True}
        postgres_effective = next(row for row in receipt.effective_connect_grantees if row["role_name"] == "postgres")
        assert postgres_effective == {
            "role_name": "postgres",
            "can_login": True,
            "superuser": True,
            "database_owner": False,
            "effective_connect": True,
            "reconnect_capable": True,
        }
        with psycopg.connect(fleet_db, connect_timeout=3):
            pass


def test_generated_rollback_sql_restores_public_connect_and_removes_new_allowlist_grants(
    fleet_db,
):
    with pgqueue.connect(fleet_db) as conn:
        database = sql.Identifier(conninfo_to_dict(fleet_db)["dbname"])
        conn.execute(sql.SQL("GRANT CONNECT, CREATE ON DATABASE {} TO PUBLIC").format(database))
        conn.commit()
        receipt = pg_roles.ensure_fleet_worker_role(
            conn,
            PASSWORD,
            role=ROLE,
            worker_id=WORKER,
            contract="apply",
            regrant_manifest=_manifest(),
            evidence_paths=_test_evidence_paths(),
        )
        with psycopg.connect(fleet_db, row_factory=dict_row) as break_glass:
            break_glass.execute(receipt.rollback_sql)
            break_glass.commit()
        assert (
            conn.execute("SELECT has_database_privilege('public',current_database(),'CONNECT') AS allowed").fetchone()[
                "allowed"
            ]
            is True
        )
        assert (
            conn.execute("SELECT has_database_privilege('public',current_database(),'CREATE') AS allowed").fetchone()[
                "allowed"
            ]
            is True
        )
        direct = conn.execute(
            "SELECT COALESCE(array_agg(grantee.rolname ORDER BY grantee.rolname),ARRAY[]::name[]) AS roles "
            "FROM pg_database d CROSS JOIN LATERAL aclexplode(d.datacl) acl "
            "JOIN pg_roles grantee ON grantee.oid=acl.grantee "
            "WHERE d.datname=current_database() AND acl.privilege_type='CONNECT' "
            "AND grantee.rolname=ANY(%s)",
            ([DATABASE_OWNER_ROLE, CONTROLLER_ROLE, VERIFIER_ROLE, ROLE],),
        ).fetchone()["roles"]
        assert direct == [CONTROLLER_ROLE, DATABASE_OWNER_ROLE]


def test_existing_mapped_role_rollback_restores_catalog_state_and_quarantines_login(fleet_db):
    regrant = pg_roles.AclRegrant(
        object_kind="table",
        qualified_name="public.apply_queue",
        privileges=("UPDATE",),
        grantee=VERIFIER_ROLE,
    )
    with pgqueue.connect(fleet_db) as conn:
        conn.execute("DROP SEQUENCE IF EXISTS public.rollback_acl_sequence")
        conn.execute("DROP TYPE IF EXISTS public.rollback_acl_type")
        conn.execute("CREATE SEQUENCE public.rollback_acl_sequence")
        conn.execute(
            "GRANT SELECT ON SEQUENCE public.rollback_acl_sequence TO fleet_backup"
        )
        conn.execute("CREATE TYPE public.rollback_acl_type AS ENUM ('before')")
        conn.execute(
            sql.SQL(
                "CREATE ROLE {} LOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE "
                "NOREPLICATION NOBYPASSRLS CONNECTION LIMIT 7 PASSWORD {}"
            ).format(sql.Identifier(ROLE), sql.Literal("rollback-original-secret"))
        )
        conn.execute("CREATE ROLE fleet_worker_inherited")
        conn.execute("CREATE ROLE fleet_worker_child")
        conn.execute(
            sql.SQL("GRANT fleet_worker_inherited TO {} WITH ADMIN TRUE, INHERIT FALSE, SET TRUE").format(
                sql.Identifier(ROLE)
            )
        )
        conn.execute(
            sql.SQL("GRANT {} TO fleet_worker_child WITH ADMIN FALSE, INHERIT TRUE, SET FALSE").format(
                sql.Identifier(ROLE)
            )
        )
        conn.execute(
            sql.SQL("GRANT TEMPORARY ON DATABASE {} TO {}").format(
                sql.Identifier(conninfo_to_dict(fleet_db)["dbname"]), sql.Identifier(ROLE)
            )
        )
        conn.execute(sql.SQL("GRANT CREATE ON SCHEMA public TO {}").format(sql.Identifier(ROLE)))
        conn.execute(sql.SQL("GRANT SELECT ON public.apply_queue TO {} WITH GRANT OPTION").format(sql.Identifier(ROLE)))
        conn.execute(sql.SQL("GRANT USAGE ON SEQUENCE public.rollback_acl_sequence TO {}").format(sql.Identifier(ROLE)))
        conn.execute(
            sql.SQL("GRANT EXECUTE ON FUNCTION public.fleet_worker_schema_contract() TO {}").format(
                sql.Identifier(ROLE)
            )
        )
        conn.execute(sql.SQL("GRANT USAGE ON TYPE public.rollback_acl_type TO {}").format(sql.Identifier(ROLE)))
        conn.execute(
            sql.SQL(
                "ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA public GRANT SELECT ON TABLES TO {} WITH GRANT OPTION"
            ).format(sql.Identifier(MIGRATOR_ROLE), sql.Identifier(ROLE))
        )
        conn.execute(sql.SQL("GRANT SELECT ON public.apply_queue TO {}").format(sql.Identifier(VERIFIER_ROLE)))
        conn.execute(
            "INSERT INTO public.fleet_worker_principals(role_name,worker_id,contract,created_at) "
            "VALUES(%s,%s,%s,'2001-02-03 04:05:06+00'::timestamptz)",
            (ROLE, WORKER, "apply"),
        )
        conn.commit()
        membership_query = (
            "SELECT parent.rolname AS parent_role,member.rolname AS member_role,"
            "grantor.rolname AS grantor_role,m.admin_option,m.inherit_option,m.set_option "
            "FROM pg_auth_members m JOIN pg_roles parent ON parent.oid=m.roleid "
            "JOIN pg_roles member ON member.oid=m.member JOIN pg_roles grantor ON grantor.oid=m.grantor "
            "WHERE parent.rolname=%s OR member.rolname=%s "
            "ORDER BY parent.rolname,member.rolname,grantor.rolname"
        )
        memberships_before = conn.execute(membership_query, (ROLE, ROLE)).fetchall()
        principal_before = conn.execute(
            "SELECT role_name::text,worker_id,contract,created_at "
            "FROM public.fleet_worker_principals WHERE role_name=%s",
            (ROLE,),
        ).fetchone()
        with conn.cursor() as cur:
            before = pg_roles._target_role_snapshot(cur, role_name=ROLE, role_exists=True)
        conn.commit()

        receipt = pg_roles.ensure_fleet_worker_role(
            conn,
            PASSWORD,
            role=ROLE,
            worker_id=WORKER,
            contract="apply",
            regrant_manifest=_manifest(regrants=(regrant,)),
            evidence_paths=_test_evidence_paths(),
        )
        assert receipt.credential_forward_reconcile_required is True
        assert receipt.inventory["credential_forward_reconcile_required"] is True
        evidence = json.dumps(receipt.inventory, sort_keys=True)
        assert "rollback-original-secret" not in evidence
        assert PASSWORD not in evidence
        assert "SCRAM-SHA" not in evidence
        assert "rollback-original-secret" not in receipt.rollback_sql
        assert PASSWORD not in receipt.rollback_sql
        assert "SCRAM-SHA" not in receipt.rollback_sql

    with psycopg.connect(fleet_db, row_factory=dict_row) as break_glass:
        assert (
            break_glass.execute(
                "SELECT has_table_privilege(%s,'public.apply_queue','UPDATE') AS allowed",
                (VERIFIER_ROLE,),
            ).fetchone()["allowed"]
            is True
        )
        break_glass.execute(receipt.rollback_sql)
        break_glass.commit()
        with break_glass.cursor() as cur:
            after = pg_roles._target_role_snapshot(cur, role_name=ROLE, role_exists=True)
        assert after["attributes"] == {**before["attributes"], "rolcanlogin": False}
        assert break_glass.execute(membership_query, (ROLE, ROLE)).fetchall() == memberships_before
        assert after["principal_mapping"] == before["principal_mapping"]
        assert (
            break_glass.execute(
                "SELECT role_name::text,worker_id,contract,created_at "
                "FROM public.fleet_worker_principals WHERE role_name=%s",
                (ROLE,),
            ).fetchone()
            == principal_before
        )
        assert principal_before["created_at"] == datetime(2001, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
        assert after["direct_acls"] == before["direct_acls"]
        assert after["default_acls"] == before["default_acls"]
        assert break_glass.execute(
            "SELECT has_table_privilege(%s,'public.apply_queue','SELECT') AS selected,"
            "has_table_privilege(%s,'public.apply_queue','UPDATE') AS updated",
            (VERIFIER_ROLE, VERIFIER_ROLE),
        ).fetchone() == {"selected": True, "updated": False}
        break_glass.execute("DROP SEQUENCE public.rollback_acl_sequence")
        break_glass.execute("DROP TYPE public.rollback_acl_type")
        break_glass.commit()


def test_new_mapped_role_rollback_drops_role_and_mapping(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        receipt = pg_roles.ensure_fleet_worker_role(
            conn,
            PASSWORD,
            role=ROLE,
            worker_id=WORKER,
            contract="apply",
            regrant_manifest=_manifest(),
            evidence_paths=_test_evidence_paths(),
        )
        assert receipt.credential_forward_reconcile_required is False
        assert "DROP OWNED BY" in receipt.rollback_sql
        assert "DROP ROLE" in receipt.rollback_sql
    with psycopg.connect(fleet_db, row_factory=dict_row) as break_glass:
        break_glass.execute(receipt.rollback_sql)
        break_glass.commit()
        assert break_glass.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (ROLE,)).fetchone() is None
        assert (
            break_glass.execute("SELECT 1 FROM public.fleet_worker_principals WHERE role_name=%s", (ROLE,)).fetchone()
            is None
        )


def test_structured_regrant_is_generated_and_public_or_freeform_values_are_rejected(fleet_db):
    allowed = pg_roles.AclRegrant(
        object_kind="table",
        qualified_name="public.fleet_config",
        privileges=("SELECT",),
        grantee=VERIFIER_ROLE,
    )
    with pgqueue.connect(fleet_db) as conn:
        receipt = pg_roles.ensure_fleet_worker_role(
            conn,
            PASSWORD,
            role=ROLE,
            worker_id=WORKER,
            contract="apply",
            regrant_manifest=_manifest(regrants=(allowed,)),
            evidence_paths=_test_evidence_paths(),
        )
        assert (
            conn.execute(
                "SELECT has_table_privilege(%s,'public.fleet_config','SELECT') AS allowed",
                (VERIFIER_ROLE,),
            ).fetchone()["allowed"]
            is True
        )
        assert receipt.inventory["regrants"] == (
            {
                "object_kind": "table",
                "qualified_name": "public.fleet_config",
                "privileges": ("SELECT",),
                "grantee": VERIFIER_ROLE,
            },
        )

    for rejected in (
        pg_roles.AclRegrant("table", "public.fleet_config", ("SELECT",), "PUBLIC"),
        pg_roles.AclRegrant("table", "public.fleet_config; DROP TABLE workers", ("SELECT",), VERIFIER_ROLE),
    ):
        with pgqueue.connect(fleet_db) as conn:
            with pytest.raises(ValueError):
                pg_roles.ensure_fleet_worker_role(
                    conn,
                    PASSWORD,
                    role=ROLE,
                    worker_id=WORKER,
                    contract="apply",
                    regrant_manifest=_manifest(regrants=(rejected,)),
                    evidence_paths=_test_evidence_paths(),
                )


def test_unknown_active_service_aborts_before_public_acl_change(fleet_db):
    with pgqueue.connect(fleet_db) as owner:
        owner.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO PUBLIC").format(
                sql.Identifier(conninfo_to_dict(fleet_db)["dbname"])
            )
        )
        owner.execute("CREATE ROLE unknown_service LOGIN PASSWORD 'disposable-unknown-service'")
        owner.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO unknown_service").format(
                sql.Identifier(conninfo_to_dict(fleet_db)["dbname"])
            )
        )
        owner.commit()
        params = conninfo_to_dict(fleet_db)
        params.update(user="unknown_service", password="disposable-unknown-service")
        service = psycopg.connect(make_conninfo(**params))
        try:
            with pytest.raises(
                RuntimeError,
                match="unknown (database service principals|explicit CONNECT dependencies)",
            ):
                pg_roles.ensure_fleet_worker_role(
                    owner,
                    PASSWORD,
                    role=ROLE,
                    worker_id=WORKER,
                    contract="apply",
                    regrant_manifest=_manifest(),
                    evidence_paths=_test_evidence_paths(),
                )
            assert (
                owner.execute(
                    "SELECT has_database_privilege('public',current_database(),'CONNECT') AS allowed"
                ).fetchone()["allowed"]
                is True
            )
        finally:
            service.close()
            require_disposable_postgres(owner)
            owner.execute(
                sql.SQL("REVOKE CONNECT ON DATABASE {} FROM unknown_service").format(
                    sql.Identifier(conninfo_to_dict(fleet_db)["dbname"])
                )
            )
            owner.execute("DROP ROLE unknown_service")
            owner.commit()


def test_runtime_principal_validation_binds_session_role_worker_and_contract(fleet_db):
    _harden(fleet_db)
    with psycopg.connect(_dsn(fleet_db), row_factory=dict_row) as worker:
        identity = pg_roles.validate_runtime_principal(worker, worker_id=WORKER, contract="apply")
        assert identity == pg_roles.RuntimePrincipal(
            session_user=ROLE,
            current_user=ROLE,
            worker_id=WORKER,
            contract="apply",
        )
        with pytest.raises(RuntimeError, match="worker identity mismatch"):
            pg_roles.validate_runtime_principal(worker, worker_id="forged-worker", contract="apply")


def test_authenticated_compute_contract_cannot_forge_usage_or_queue(fleet_db):
    with pgqueue.connect(fleet_db) as owner:
        owner.execute(
            "INSERT INTO public.compute_queue(url,task,payload) VALUES('compute-role','score','{\"input\":1}'::jsonb)"
        )
        owner.commit()
    _harden(fleet_db, contract="compute")
    with pgqueue.connect(fleet_db) as owner:
        owner.execute("UPDATE public.fleet_config SET paused=TRUE WHERE id=1")
        owner.commit()
    with psycopg.connect(_dsn(fleet_db), row_factory=dict_row) as worker:
        assert queue.lease_compute(worker, "forged", sw_version="forged") is None
    with pgqueue.connect(fleet_db) as owner:
        owner.execute("UPDATE public.fleet_config SET paused=FALSE WHERE id=1")
        owner.execute(
            "UPDATE public.fleet_desired_state SET desired_workers=0,updated_at=now() WHERE machine_owner='machine-a'"
        )
        owner.commit()
    with psycopg.connect(_dsn(fleet_db), row_factory=dict_row) as worker:
        assert queue.lease_compute(worker, "forged", sw_version="forged") is None
    with pgqueue.connect(fleet_db) as owner:
        owner.execute(
            "UPDATE public.fleet_desired_state SET desired_workers=1,updated_at=now() WHERE machine_owner='machine-a'"
        )
        owner.commit()
    with psycopg.connect(_dsn(fleet_db), row_factory=dict_row) as worker:
        job = queue.lease_compute(worker, "forged", sw_version="forged")
        assert job == {"url": "compute-role", "task": "score", "payload": {"input": 1}, "attempts": 1}
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            worker.execute("INSERT INTO public.llm_usage(worker_id,cost_usd) VALUES('forged',999)")
        worker.rollback()
        assert queue.write_compute_result(
            worker,
            "forged",
            "compute-role",
            task="score",
            result={"score": 8},
            cost_usd=0.1,
            model="m",
            provider="p",
        )
    with pgqueue.connect(fleet_db) as owner:
        assert owner.execute(
            "SELECT status::text,lease_owner FROM public.compute_queue WHERE url='compute-role'"
        ).fetchone() == {"status": "done", "lease_owner": None}
        assert (
            owner.execute("SELECT worker_id,cost_usd FROM public.llm_usage WHERE task='score'").fetchone()["worker_id"]
            == WORKER
        )


def test_authenticated_discovery_requires_existing_governor_and_derives_worker(fleet_db):
    with pgqueue.connect(fleet_db) as owner:
        owner.execute(
            "INSERT INTO public.search_tasks(task_id,query,board,location) "
            "VALUES('search-role','chief of staff','indeed','Remote')"
        )
        owner.commit()
    _harden(fleet_db, contract="discovery")
    with psycopg.connect(_dsn(fleet_db), row_factory=dict_row) as worker:
        assert queue.lease_search(worker, "forged", sw_version="forged") is None
    with pgqueue.connect(fleet_db) as owner:
        owner.execute(
            "INSERT INTO public.rate_governor(scope_key,daily_cap,min_gap_seconds) VALUES('board:indeed',10,0)"
        )
        owner.execute("UPDATE public.fleet_config SET paused=TRUE WHERE id=1")
        owner.commit()
    with psycopg.connect(_dsn(fleet_db), row_factory=dict_row) as worker:
        assert queue.lease_search(worker, "forged", sw_version="forged") is None
    with pgqueue.connect(fleet_db) as owner:
        owner.execute("UPDATE public.fleet_config SET paused=FALSE WHERE id=1")
        owner.execute(
            "UPDATE public.fleet_desired_state SET desired_workers=0,updated_at=now() WHERE machine_owner='machine-a'"
        )
        owner.commit()
    with psycopg.connect(_dsn(fleet_db), row_factory=dict_row) as worker:
        assert queue.lease_search(worker, "forged", sw_version="forged") is None
    with pgqueue.connect(fleet_db) as owner:
        owner.execute(
            "UPDATE public.fleet_desired_state SET desired_workers=1,updated_at=now() WHERE machine_owner='machine-a'"
        )
        owner.commit()
    with psycopg.connect(_dsn(fleet_db), row_factory=dict_row) as worker:
        task = queue.lease_search(worker, "forged", sw_version="forged")
        assert task and task["task_id"] == "search-role"
        assert queue.complete_search(
            worker,
            "forged",
            "search-role",
            board="indeed",
            postings=[{"url": "https://example.test/discovered"}],
        )
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            worker.execute("UPDATE public.rate_governor SET daily_cap=999")
    with pgqueue.connect(fleet_db) as owner:
        assert (
            owner.execute("SELECT worker_id FROM public.discovered_postings WHERE task_id='search-role'").fetchone()[
                "worker_id"
            ]
            == WORKER
        )


def test_candidate_capability_roles_are_nologin_and_cannot_create_or_escalate(fleet_db):
    with psycopg.connect(fleet_db, row_factory=dict_row) as provider:
        try:
            provider.execute(
                "DO $$ BEGIN CREATE ROLE brain_schema_migrator NOLOGIN NOINHERIT NOSUPERUSER "
                "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS; "
                "EXCEPTION WHEN duplicate_object THEN NULL; END $$"
            )
            provider.commit()
            receipt = pg_roles.ensure_brain_candidate_roles(provider)
            assert receipt.reader_role == "brain_candidate_reader"
            assert receipt.writer_role == "brain_candidate_writer"
            provider.execute("CREATE ROLE candidate_parent_unapproved")
            provider.execute("CREATE ROLE candidate_grantee_approved")
            provider.execute("CREATE ROLE candidate_grantee_unapproved")
            provider.execute("GRANT candidate_parent_unapproved TO brain_candidate_writer")
            provider.execute("GRANT brain_candidate_writer TO candidate_grantee_approved")
            provider.execute("GRANT candidate_grantee_approved TO candidate_grantee_unapproved")
            provider.commit()
            pg_roles.ensure_brain_candidate_roles(
                provider,
                writer_approved_grantees=("candidate_grantee_approved",),
            )
            membership_edges = provider.execute(
                "SELECT parent.rolname AS parent_role,member.rolname AS member_role "
                "FROM pg_auth_members membership "
                "JOIN pg_roles parent ON parent.oid=membership.roleid "
                "JOIN pg_roles member ON member.oid=membership.member "
                "WHERE parent.rolname='brain_candidate_writer' OR member.rolname='brain_candidate_writer'"
            ).fetchall()
            assert membership_edges == []

            provider.execute("REVOKE candidate_grantee_approved FROM candidate_grantee_unapproved")
            provider.execute("GRANT brain_candidate_writer TO candidate_grantee_approved")
            provider.commit()
            pg_roles.ensure_brain_candidate_roles(
                provider,
                writer_approved_grantees=("candidate_grantee_approved",),
            )
            assert provider.execute(
                "SELECT pg_has_role('candidate_grantee_approved','brain_candidate_writer','MEMBER') AS allowed"
            ).fetchone()["allowed"] is True
            rows = provider.execute(
                "SELECT rolname,rolcanlogin,rolinherit,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
                "FROM pg_roles WHERE rolname IN ('brain_candidate_reader','brain_candidate_writer','brain_schema_migrator') "
                "ORDER BY rolname"
            ).fetchall()
            assert all(
                not any(
                    row[key]
                    for key in (
                        "rolcanlogin",
                        "rolinherit",
                        "rolsuper",
                        "rolcreatedb",
                        "rolcreaterole",
                        "rolreplication",
                        "rolbypassrls",
                    )
                )
                for row in rows
            )
            assert (
                provider.execute(
                    "SELECT has_schema_privilege('brain_candidate_writer','public','CREATE') AS allowed"
                ).fetchone()["allowed"]
                is False
            )
        finally:
            provider.rollback()
            require_disposable_postgres(provider)
            for role_name in (
                "brain_candidate_reader",
                "brain_candidate_writer",
                "brain_schema_migrator",
                "candidate_parent_unapproved",
                "candidate_grantee_unapproved",
                "candidate_grantee_approved",
            ):
                if provider.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,)).fetchone():
                    provider.execute(sql.SQL("DROP OWNED BY {} CASCADE").format(sql.Identifier(role_name)))
                    provider.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role_name)))
            provider.commit()


def test_artifact_authority_roles_are_fixed_nologin_and_membership_closed(fleet_db):
    with psycopg.connect(fleet_db, row_factory=dict_row) as provider:
        try:
            provider.execute(
                "DO $$ BEGIN CREATE ROLE brain_schema_migrator NOLOGIN NOINHERIT NOSUPERUSER "
                "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS; "
                "EXCEPTION WHEN duplicate_object THEN NULL; END $$"
            )
            provider.commit()
            with provider.transaction():
                with provider.cursor() as cursor:
                    assert pg_roles.ensure_brain_artifact_authority_roles_in_transaction(cursor) == (
                        "brain_artifact_authority_owner",
                        "brain_artifact_authority_writer",
                    )
            roles = provider.execute(
                "SELECT rolname,rolcanlogin,rolinherit,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
                "FROM pg_roles WHERE rolname LIKE 'brain_artifact_authority_%' ORDER BY rolname"
            ).fetchall()
            assert len(roles) == 5
            for role in roles:
                assert role["rolcanlogin"] is role["rolname"].endswith("_login")
                assert not any(
                    role[key]
                    for key in (
                        "rolinherit", "rolsuper", "rolcreatedb", "rolcreaterole",
                        "rolreplication", "rolbypassrls",
                    )
                )
            assert provider.execute(
                "SELECT array_agg(parent.rolname || '->' || member.rolname ORDER BY parent.rolname,member.rolname) "
                "AS edges FROM pg_auth_members membership "
                "JOIN pg_roles parent ON parent.oid=membership.roleid "
                "JOIN pg_roles member ON member.oid=membership.member "
                "WHERE parent.rolname LIKE 'brain_artifact_authority_%' "
                "OR member.rolname LIKE 'brain_artifact_authority_%'"
            ).fetchone()["edges"] == [
                "brain_artifact_authority_owner->brain_schema_migrator",
                "brain_artifact_authority_verifier->brain_artifact_authority_verifier_login",
                "brain_artifact_authority_writer->brain_artifact_authority_writer_login",
            ]
            assert provider.execute(
                "SELECT has_schema_privilege('brain_artifact_authority_owner','public','CREATE') AS owner_create,"
                "has_schema_privilege('brain_artifact_authority_writer','public','CREATE') AS writer_create"
            ).fetchone() == {"owner_create": False, "writer_create": False}
        finally:
            provider.rollback()
            require_disposable_postgres(provider)
            for role_name in (
                "brain_artifact_authority_verifier_login",
                "brain_artifact_authority_writer_login",
                "brain_artifact_authority_verifier",
                "brain_artifact_authority_writer",
                "brain_artifact_authority_owner",
                "brain_schema_migrator",
            ):
                if provider.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,)).fetchone():
                    provider.execute(sql.SQL("DROP OWNED BY {} CASCADE").format(sql.Identifier(role_name)))
                    provider.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role_name)))
            provider.commit()
