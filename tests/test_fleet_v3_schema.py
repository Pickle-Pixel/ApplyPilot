"""v3 schema: loads cleanly against real Postgres, idempotent, all tables + columns present."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import uuid
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")
from psycopg import sql  # noqa: E402
from psycopg.conninfo import conninfo_to_dict, make_conninfo  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402

from applypilot.apply import pgqueue  # noqa: E402
from applypilot.fleet import apply_worker_main, compute_context, emergency_admission, queue  # noqa: E402
from applypilot.fleet import schema as fleet_schema  # noqa: E402

EXPECTED_TABLES = {
    "apply_queue", "fleet_config", "fleet_assets",  # base
    "compute_queue", "search_tasks", "linkedin_queue", "rate_governor", "llm_usage",
    "applied_set", "answer_bank", "auth_challenge", "otp_request", "inbox_events",
    "workers", "worker_heartbeat", "poison_jobs", "remote_commands", "autotriage_actions",
    "fleet_decision_policies",
    "apply_attempts",
}


def test_read_only_compatibility_contract_covers_runtime_tables():
    assert {"apply_attempts", "fleet_decision_policies"} <= fleet_schema._REQUIRED_TABLES
    assert fleet_schema._APPLY_ATTEMPT_REQUIRED_COLUMNS <= fleet_schema._REQUIRED_COLUMNS["apply_attempts"]
    assert {"policy_version", "lane", "status"} <= fleet_schema._REQUIRED_COLUMNS[
        "fleet_decision_policies"
    ]


CANONICAL_QUEUE_COLUMNS = {
    "decision_id", "policy_version", "decision_action", "qualification_verdict",
    "qualification_score", "qualification_floor", "preference_score", "outcome_score",
    "final_score", "decision_confidence", "decision_created_at",
    "decision_expires_at", "input_hash",
}

CANONICAL_CONSTRAINT_SUFFIXES = {
    "canonical_provenance_ck", "lane_ck", "policy_lane_fk",
}


@pytest.mark.parametrize("contract", ["apply", "linkedin", "compute", "discovery"])
def test_mapped_scram_worker_starts_through_narrow_contract(fleet_db, contract):
    role = f"runtime_{contract}_worker"
    worker_id = f"runtime-box-{contract}"
    password = f"runtime-{contract}-scram-secret"
    now = datetime.now(timezone.utc)
    with pgqueue.connect(fleet_db) as owner:
        owner.execute(
            "CREATE TABLE IF NOT EXISTS public.fleet_desired_state("
            "machine_owner text primary key,desired_workers integer,agent text,model text,"
            "generation integer,updated_at timestamptz)"
        )
        owner.execute(
            "INSERT INTO public.fleet_desired_state(machine_owner,desired_workers,agent,model,generation,updated_at) "
            "VALUES('runtime-box',4,'codex','test-model',1,now()) "
            "ON CONFLICT(machine_owner) DO UPDATE SET desired_workers=4,agent='codex',model='test-model',"
            "generation=1,updated_at=now()"
        )
        owner.execute(
            "INSERT INTO public.workers(worker_id,machine_owner,public_ip,validated,capabilities) "
            "VALUES(%s,'runtime-box','1.1.1.1',true,'{}'::jsonb)", (worker_id,)
        )
        owner.execute(
            "INSERT INTO public.worker_heartbeat(worker_id,machine_owner,home_ip,role,state,sw_version,last_beat) "
            "VALUES(%s,'runtime-box','1.1.1.1',%s,'idle','runtime-v1',now())", (worker_id, contract)
        )
        owner.execute(
            "INSERT INTO public.fleet_decision_policies(policy_version,lane,status) VALUES"
            "('runtime-ats-policy','ats','active'),('runtime-linkedin-policy','linkedin','active') "
            "ON CONFLICT(policy_version) DO UPDATE SET status='active'"
        )
        owner.execute(
            "UPDATE public.fleet_config SET paused=false,ats_paused=false,ats_apply_mode='steady',"
            "linkedin_apply_mode='steady',ats_policy_version='runtime-ats-policy',"
            "linkedin_policy_version='runtime-linkedin-policy',linkedin_owner_ip='1.1.1.1',"
            "pinned_worker_version='runtime-v1',spend_cap_usd=0 WHERE id=1"
        )
        owner.execute(
            "INSERT INTO public.rate_governor(scope_key,daily_cap,min_gap_seconds,count_24h) "
            "VALUES('account:linkedin',20,0,0),('global',100,0,0) ON CONFLICT(scope_key) DO UPDATE SET "
            "daily_cap=EXCLUDED.daily_cap,min_gap_seconds=0,count_24h=0,halted_until=NULL,"
            "breaker_state='ok',last_applied_at=NULL"
        )
        if contract == "linkedin":
            owner.execute(
                "INSERT INTO public.linkedin_queue(url,application_url,score,status,lane,approved_batch,dedup_key,"
                "linkedin_resolve_status,linkedin_resolved_at,decision_id,policy_version,decision_action,"
                "qualification_verdict,qualification_score,qualification_floor,preference_score,outcome_score,"
                "final_score,decision_confidence,decision_created_at,decision_expires_at,input_hash) "
                "VALUES('runtime-linkedin-job','https://linkedin.com/jobs/runtime',9,'queued','linkedin','runtime',"
                "'runtime-linkedin-dedup','easy_apply',now(),'runtime-linkedin-decision','runtime-linkedin-policy',"
                "'apply','qualified',9,7,9,9,9,.9,%s,%s,'runtime-linkedin-hash')",
                (now, now + timedelta(days=1)),
            )
        elif contract == "apply":
            owner.execute(
                "INSERT INTO public.llm_usage(worker_id,task,provider,cost_usd) "
                "VALUES(%s,'apply_agent','runtime-agent',2)", (worker_id,)
            )
        owner.execute(sql.SQL(
            "CREATE ROLE {} LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE "
            "NOREPLICATION NOBYPASSRLS PASSWORD {}"
        ).format(sql.Identifier(role), sql.Literal(password)))
        owner.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
            sql.Identifier(conninfo_to_dict(fleet_db)["dbname"]), sql.Identifier(role)
        ))
        owner.execute(
            "INSERT INTO public.fleet_worker_principals(role_name,worker_id,contract) VALUES(%s,%s,%s)",
            (role, worker_id, contract),
        )
        owner.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(role)))
        owner.execute(sql.SQL(
            "GRANT EXECUTE ON FUNCTION public.fleet_worker_schema_contract(), "
            "public.fleet_worker_admission_snapshot(), public.fleet_worker_runtime_state(text) TO {}"
        ).format(sql.Identifier(role)))
        if contract in {"apply", "linkedin"}:
            owner.execute(sql.SQL(
                "GRANT EXECUTE ON FUNCTION public.fleet_worker_evaluate_agent_budget(jsonb,integer,integer), "
                "public.fleet_worker_agent_blocks() TO {}"
            ).format(sql.Identifier(role)))
        if contract == "apply":
            owner.execute(sql.SQL(
                "GRANT USAGE ON TYPE public.apply_queue TO {}; "
                "GRANT EXECUTE ON FUNCTION public.fleet_worker_lease_ats(text,text,integer,text,integer) TO {}"
            ).format(sql.Identifier(role), sql.Identifier(role)))
        elif contract == "linkedin":
            owner.execute(sql.SQL(
                "GRANT USAGE ON TYPE public.linkedin_queue TO {}; "
                "GRANT EXECUTE ON FUNCTION public.fleet_worker_lease_linkedin(text,text,text,integer,text) TO {}"
            ).format(sql.Identifier(role), sql.Identifier(role)))
        owner.commit()

    params = conninfo_to_dict(fleet_db)
    params.update(user=role, password=password)
    try:
        with psycopg.connect(make_conninfo(**params), row_factory=dict_row) as worker:
            fleet_schema.ensure_schema_v3(worker)
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                worker.execute("SELECT * FROM public.fleet_config")
            worker.rollback()
            snapshot = worker.execute(
                "SELECT public.fleet_worker_admission_snapshot() AS snapshot"
            ).fetchone()["snapshot"]
            assert snapshot["worker_id"] == worker_id
            assert snapshot["admission_allowed"] is True
            if contract == "apply":
                assert emergency_admission.worker_admission(
                    worker, machine_label="runtime-box", machine_owner="runtime-box",
                    worker_id=worker_id,
                ).allowed
                worker.rollback()
                apply_worker_main.PgAgentBudget(
                    soft_caps={"runtime-agent": 1}, window_seconds=3600,
                    cooldown_seconds=60, eval_interval_seconds=0,
                ).maybe_evaluate(worker)
                blocks = worker.execute(
                    "SELECT public.fleet_worker_agent_blocks() AS blocks"
                ).fetchone()["blocks"]
                assert "runtime-agent" in blocks
            elif contract == "linkedin":
                assert emergency_admission.linkedin_worker_admission(worker).allowed
                leased = queue.lease_linkedin(
                    worker, "forged-worker", public_ip="1.1.1.1", owner_ip="1.1.1.1",
                    sw_version="forged-version", min_gap_seconds=0,
                )
                assert leased and leased["url"] == "runtime-linkedin-job"
                with pgqueue.connect(fleet_db) as owner:
                    counts = owner.execute(
                        "SELECT scope_key,count_24h FROM public.rate_governor "
                        "WHERE scope_key IN ('global','account:linkedin') ORDER BY scope_key"
                    ).fetchall()
                assert {row["scope_key"]: row["count_24h"] for row in counts} == {
                    "account:linkedin": 1, "global": 1,
                }
            elif contract == "compute":
                context, version = compute_context.load_context(worker, providers=("test",))
                assert context.providers == ["test"]
                assert version == ""
            else:
                state = worker.execute(
                    "SELECT public.fleet_worker_runtime_state('forged-worker') AS state"
                ).fetchone()["state"]
                assert state["state"] == "idle"
            if contract in {"apply", "linkedin"}:
                worker.rollback()
                lane_column = "ats_apply_mode" if contract == "apply" else "linkedin_apply_mode"
                with pgqueue.connect(fleet_db) as owner:
                    owner.execute(sql.SQL("UPDATE public.fleet_config SET {}='stopped' WHERE id=1")
                                  .format(sql.Identifier(lane_column)))
                    owner.commit()
                if contract == "apply":
                    denied = emergency_admission.worker_tick_admission(worker)
                else:
                    denied = emergency_admission.linkedin_tick_admission(worker)
                assert not denied.allowed
    finally:
        with pgqueue.connect(fleet_db) as owner:
            owner.execute("DELETE FROM public.fleet_worker_principals WHERE role_name=%s", (role,))
            owner.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
            owner.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
            owner.commit()


def _canonical_row(*, table: str, url: str = "canonical", **overrides):
    row = {
        "url": url,
        "application_url": "https://example.test/apply",
        "score": 0.75,
        "lane": "ats" if table == "apply_queue" else "linkedin",
        "decision_id": "decision-1",
        "policy_version": "policy-1",
        "decision_action": "apply",
        "qualification_verdict": "qualified",
        "qualification_score": 0.8,
        "qualification_floor": 0.7,
        "preference_score": 0.6,
        "outcome_score": 0.5,
        "final_score": 0.75,
        "decision_confidence": 0.9,
        "decision_created_at": "2026-07-11T12:00:00Z",
        "decision_expires_at": "2026-07-12T12:00:00Z",
        "input_hash": "sha256:abc",
    }
    row.update(overrides)
    return row


def test_worker_principal_mapping_is_unique_per_worker_contract(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        conn.execute(
            "INSERT INTO public.workers(worker_id,machine_owner,validated,capabilities) "
            "VALUES('unique-runtime-worker','runtime-box',true,'{}'::jsonb)"
        )
        conn.execute(
            "INSERT INTO public.fleet_worker_principals(role_name,worker_id,contract) "
            "VALUES('runtime-role-a','unique-runtime-worker','apply')"
        )
        conn.commit()
        with pytest.raises(psycopg.errors.UniqueViolation):
            conn.execute(
                "INSERT INTO public.fleet_worker_principals(role_name,worker_id,contract) "
                "VALUES('runtime-role-b','unique-runtime-worker','apply')"
            )
        conn.rollback()


def _insert_canonical(cur, table: str, row: dict) -> None:
    columns = list(row)
    cur.execute(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))})",
        [row[column] for column in columns],
    )


def _canonical_schema_objects(cur, table: str) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    cur.execute(
        """
        SELECT c.conname, pg_get_constraintdef(c.oid, true) AS definition
        FROM pg_constraint c
        JOIN pg_class rel ON rel.oid = c.conrelid
        JOIN pg_namespace n ON n.oid = rel.relnamespace
        WHERE n.nspname = current_schema() AND rel.relname = %s
          AND (c.conname LIKE %s OR c.conname LIKE %s OR c.conname LIKE %s)
        """,
        (table, "%canonical_provenance_ck", "%lane_ck", "%policy_lane_fk"),
    )
    constraints = {(r["conname"], r["definition"]) for r in cur.fetchall()}
    cur.execute(
        """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = current_schema() AND tablename = %s
          AND indexname LIKE %s
        """,
        (table, "%canonical_lease"),
    )
    indexes = {(r["indexname"], r["indexdef"]) for r in cur.fetchall()}
    return constraints, indexes


def _tables(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        return {r["table_name"] for r in cur.fetchall()}


def test_v3_schema_creates_all_tables(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        got = _tables(conn)
    missing = EXPECTED_TABLES - got
    assert not missing, f"missing tables: {missing}"


def test_v3_schema_idempotent(fleet_db):
    # the fixture already applied it once; applying twice more must not error.
    with pgqueue.connect(fleet_db) as conn:
        fleet_schema.ensure_schema_v3(conn)
        fleet_schema.ensure_schema_v3(conn)
        got = _tables(conn)
    assert {"rate_governor", "search_tasks", "compute_queue"} <= got


def test_constraint_lookup_ignores_same_name_on_other_schema(fleet_db):
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS canonical_shadow CASCADE")
        cur.execute("CREATE SCHEMA canonical_shadow")
        cur.execute(
            "CREATE TABLE canonical_shadow.apply_queue ("
            "value INTEGER CONSTRAINT apply_queue_canonical_provenance_ck CHECK (value > 0))"
        )
        cur.execute("ALTER TABLE public.apply_queue DROP CONSTRAINT apply_queue_canonical_provenance_ck")
        conn.commit()

        fleet_schema.ensure_schema_v3(conn)

        cur.execute(
            """
            SELECT count(*) AS count
            FROM pg_constraint c
            JOIN pg_class rel ON rel.oid = c.conrelid
            JOIN pg_namespace n ON n.oid = rel.relnamespace
            WHERE n.nspname = 'public' AND rel.relname = 'apply_queue'
              AND c.conname = 'apply_queue_canonical_provenance_ck'
            """
        )
        assert cur.fetchone()["count"] == 1
        cur.execute("DROP SCHEMA canonical_shadow CASCADE")
        conn.commit()


def test_v3_schema_declares_dedup_repair_actions_once():
    schema = (Path(__file__).resolve().parents[1] / "src" / "applypilot" / "fleet" / "schema_v3.sql").read_text(
        encoding="utf-8"
    )

    assert schema.count("CREATE TABLE IF NOT EXISTS dedup_repair_actions") == 1


def test_v3_schema_declares_autotriage_actions_once():
    schema = (Path(__file__).resolve().parents[1] / "src" / "applypilot" / "fleet" / "schema_v3.sql").read_text(
        encoding="utf-8"
    )

    assert schema.count("CREATE TABLE IF NOT EXISTS autotriage_actions") == 1


def test_v3_schema_migrates_compute_queue_url_primary_key(fleet_pg):
    with pgqueue.connect_admin(fleet_pg) as conn:
        pgqueue.ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS compute_queue")
            cur.execute(
                """
                CREATE TABLE compute_queue (
                    url TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    payload JSONB,
                    status fleet_task_status NOT NULL DEFAULT 'queued',
                    lease_owner TEXT,
                    lease_expires_at TIMESTAMPTZ,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    result JSONB,
                    est_cost_usd NUMERIC(10,4) NOT NULL DEFAULT 0,
                    synced_to_home_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
        conn.commit()

        fleet_schema.ensure_schema_v3(conn)

        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO compute_queue (url, task) VALUES "
                "('same-url', 'score'), ('same-url', 'audit')"
            )
            cur.execute(
                """
                SELECT a.attname
                FROM pg_constraint c
                JOIN unnest(c.conkey) WITH ORDINALITY u(attnum, ordinality) ON true
                JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = u.attnum
                WHERE c.conrelid = 'compute_queue'::regclass AND c.contype = 'p'
                ORDER BY u.ordinality
                """
            )
            pk_cols = [r["attname"] for r in cur.fetchall()]
        conn.commit()

    assert pk_cols == ["url", "task"]


def test_apply_queue_v3_columns(fleet_db):
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='apply_queue'")
        cols = {r["column_name"] for r in cur.fetchall()}
    for c in (
        "worker_home_ip", "target_host", "lane", "dedup_key", "approved_batch",
        "liveness_check_count", "liveness_consecutive_uncertain",
    ):
        assert c in cols, f"apply_queue missing {c}"
    assert CANONICAL_QUEUE_COLUMNS <= cols


def test_linkedin_queue_has_canonical_provenance(fleet_db):
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='linkedin_queue'")
        cols = {r["column_name"] for r in cur.fetchall()}
    assert CANONICAL_QUEUE_COLUMNS <= cols


def test_v3_schema_backfills_cumulative_cost_from_latest_or_events(fleet_db):
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO apply_queue "
            "(url, application_url, score, est_cost_usd, cumulative_cost_usd) "
            "VALUES ('legacy-latest', 'legacy-latest', 1, 1.25, 0), "
            "('legacy-events', 'legacy-events', 1, 0, 0)"
        )
        cur.execute(
            "INSERT INTO apply_result_events (url, est_cost_usd) "
            "VALUES ('legacy-latest', 0.25), ('legacy-events', 0.40), "
            "('legacy-events', 0.35)"
        )
        conn.commit()

        fleet_schema.ensure_schema_v3(conn)

        cur.execute(
            "SELECT url, cumulative_cost_usd FROM apply_queue "
            "WHERE url LIKE 'legacy-%%' ORDER BY url"
        )
        rows = {row["url"]: float(row["cumulative_cost_usd"]) for row in cur.fetchall()}

    assert rows == {"legacy-events": 0.75, "legacy-latest": 1.25}


def test_v3_schema_adds_and_backfills_cumulative_cost_for_pre_column_queues(fleet_pg):
    database_name = f"applypilot_legacy_{uuid.uuid4().hex}"
    legacy_dsn = psycopg.conninfo.make_conninfo(fleet_pg, dbname=database_name)
    base_schema = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "applypilot"
        / "apply"
        / "fleet_schema.sql"
    ).read_text(encoding="utf-8")
    legacy_schema = "\n".join(
        line for line in base_schema.splitlines()
        if "cumulative_cost_usd" not in line
    )
    assert legacy_schema != base_schema
    assert "cumulative_cost_usd" not in legacy_schema

    with pgqueue.connect_admin(fleet_pg, autocommit=True) as admin:
        admin.execute(
            psycopg.sql.SQL("CREATE DATABASE {}").format(
                psycopg.sql.Identifier(database_name)
            )
        )
    try:
        with pgqueue.connect_admin(legacy_dsn) as conn, conn.cursor() as cur:
            cur.execute(legacy_schema)
            cur.execute(
                "CREATE TABLE linkedin_queue "
                "(LIKE apply_queue INCLUDING DEFAULTS INCLUDING INDEXES)"
            )
            cur.execute(
                "INSERT INTO apply_queue (url, application_url, score, est_cost_usd) "
                "VALUES ('pre-column-apply', 'pre-column-apply', 1, 1.25)"
            )
            cur.execute(
                "INSERT INTO linkedin_queue (url, application_url, score, est_cost_usd) "
                "VALUES ('pre-column-linkedin', 'pre-column-linkedin', 1, 2.25)"
            )
            conn.commit()
            cur.execute(
                "SELECT table_name FROM information_schema.columns "
                "WHERE table_schema='public' "
                "AND table_name IN ('apply_queue', 'linkedin_queue') "
                "AND column_name='cumulative_cost_usd'"
            )
            assert cur.fetchall() == []
            conn.rollback()

            fleet_schema.ensure_schema_v3(conn)

            cur.execute(
                "SELECT table_name FROM information_schema.columns "
                "WHERE table_schema='public' "
                "AND table_name IN ('apply_queue', 'linkedin_queue') "
                "AND column_name='cumulative_cost_usd'"
            )
            assert {row["table_name"] for row in cur.fetchall()} == {
                "apply_queue",
                "linkedin_queue",
            }
            cur.execute(
                "SELECT table_name, is_nullable, column_default "
                "FROM information_schema.columns "
                "WHERE table_schema='public' "
                "AND table_name IN ('apply_queue', 'linkedin_queue') "
                "AND column_name='cumulative_cost_usd'"
            )
            metadata = {row["table_name"]: row for row in cur.fetchall()}
            assert {table: row["is_nullable"] for table, row in metadata.items()} == {
                "apply_queue": "NO",
                "linkedin_queue": "NO",
            }
            assert all(row["column_default"] == "0" for row in metadata.values())
            cur.execute(
                "SELECT cumulative_cost_usd FROM apply_queue WHERE url='pre-column-apply'"
            )
            apply_cost = float(cur.fetchone()["cumulative_cost_usd"])
            cur.execute(
                "SELECT cumulative_cost_usd FROM linkedin_queue "
                "WHERE url='pre-column-linkedin'"
            )
            linkedin_cost = float(cur.fetchone()["cumulative_cost_usd"])
    finally:
        with pgqueue.connect_admin(fleet_pg, autocommit=True) as admin:
            admin.execute(
                psycopg.sql.SQL("DROP DATABASE {} WITH (FORCE)").format(
                    psycopg.sql.Identifier(database_name)
                )
            )

    assert apply_cost == 1.25
    assert linkedin_cost == 2.25


def test_v3_schema_does_not_rewrite_already_backfilled_queue_rows(fleet_db):
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO apply_queue "
            "(url, application_url, score, est_cost_usd, cumulative_cost_usd) "
            "VALUES ('stable-apply-cost', 'stable-apply-cost', 1, 1.25, 2.50)"
        )
        cur.execute(
            "INSERT INTO linkedin_queue "
            "(url, application_url, score, est_cost_usd, cumulative_cost_usd) "
            "VALUES ('stable-linkedin-cost', 'stable-linkedin-cost', 1, 1.25, 2.50)"
        )
        conn.commit()
        cur.execute(
            "SELECT 'apply' AS queue, xmin::text FROM apply_queue WHERE url='stable-apply-cost' "
            "UNION ALL "
            "SELECT 'linkedin' AS queue, xmin::text FROM linkedin_queue WHERE url='stable-linkedin-cost'"
        )
        before = {row["queue"]: row["xmin"] for row in cur.fetchall()}
        conn.rollback()

        fleet_schema.ensure_schema_v3(conn)

        cur.execute(
            "SELECT 'apply' AS queue, xmin::text FROM apply_queue WHERE url='stable-apply-cost' "
            "UNION ALL "
            "SELECT 'linkedin' AS queue, xmin::text FROM linkedin_queue WHERE url='stable-linkedin-cost'"
        )
        after = {row["queue"]: row["xmin"] for row in cur.fetchall()}

    assert after == before


def test_v3_schema_initializes_null_legacy_cumulative_costs(fleet_db):
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("ALTER TABLE apply_queue ALTER COLUMN cumulative_cost_usd DROP NOT NULL")
        cur.execute("ALTER TABLE apply_queue ALTER COLUMN cumulative_cost_usd DROP DEFAULT")
        cur.execute("ALTER TABLE linkedin_queue ALTER COLUMN cumulative_cost_usd DROP NOT NULL")
        cur.execute("ALTER TABLE linkedin_queue ALTER COLUMN cumulative_cost_usd DROP DEFAULT")
        cur.execute(
            "INSERT INTO apply_queue "
            "(url, application_url, score, est_cost_usd, cumulative_cost_usd) "
            "VALUES ('null-legacy-apply', 'null-legacy-apply', 1, 1.25, NULL)"
        )
        cur.execute(
            "INSERT INTO apply_result_events (url, est_cost_usd) "
            "VALUES ('null-legacy-apply', 0.75), ('null-legacy-apply', 0.80)"
        )
        cur.execute(
            "INSERT INTO linkedin_queue "
            "(url, application_url, score, est_cost_usd, cumulative_cost_usd) "
            "VALUES ('null-legacy-linkedin', 'null-legacy-linkedin', 1, 2.25, NULL)"
        )
        conn.commit()

        fleet_schema.ensure_schema_v3(conn)

        cur.execute(
            "SELECT cumulative_cost_usd FROM apply_queue WHERE url='null-legacy-apply'"
        )
        apply_cost = float(cur.fetchone()["cumulative_cost_usd"])
        cur.execute(
            "SELECT cumulative_cost_usd FROM linkedin_queue WHERE url='null-legacy-linkedin'"
        )
        linkedin_cost = float(cur.fetchone()["cumulative_cost_usd"])
        cur.execute(
            "SELECT table_name, is_nullable, column_default "
            "FROM information_schema.columns "
            "WHERE table_schema='public' "
            "AND table_name IN ('apply_queue', 'linkedin_queue') "
            "AND column_name='cumulative_cost_usd'"
        )
        metadata = {row["table_name"]: row for row in cur.fetchall()}

    assert apply_cost == 1.55
    assert linkedin_cost == 2.25
    assert {table: row["is_nullable"] for table, row in metadata.items()} == {
        "apply_queue": "NO",
        "linkedin_queue": "NO",
    }
    assert all(row["column_default"] == "0" for row in metadata.values())


def test_apply_result_events_include_cost_router_metadata(fleet_db):
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='apply_result_events'"
        )
        cols = {r["column_name"] for r in cur.fetchall()}

    assert "route" in cols
    assert "failure_class" in cols
    assert "tool_calls_total" in cols
    assert "application_tool_calls" in cols
    assert "last_tool" in cols
    assert "host_policy" in cols
    assert "result_metadata" in cols
    assert "job_log_path" in cols
    assert "transcript_digest" in cols
    assert "final_result_source" in cols


def test_apply_attempts_schema_and_unresolved_index(fleet_db):
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='apply_attempts'"
        )
        cols = {r["column_name"] for r in cur.fetchall()}
        cur.execute(
            "SELECT indexdef FROM pg_indexes "
            "WHERE tablename='apply_attempts' AND indexname='uq_apply_attempts_unresolved_dedup'"
        )
        indexdef = cur.fetchone()["indexdef"]

    assert {
        "attempt_id", "queue_name", "url", "dedup_key", "worker_id", "route",
        "route_version", "state", "submit_started_at", "finalized_at",
        "verification_method", "verification_ref", "evidence", "created_at",
    } <= cols
    assert "submit_started" in indexdef
    assert "submitted_unverified" in indexdef


def test_apply_attempts_schema_never_restores_broad_worker_privileges():
    schema = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "applypilot"
        / "fleet"
        / "schema_v3.sql"
    ).read_text(encoding="utf-8")

    assert "GRANT SELECT, INSERT, UPDATE ON apply_attempts TO fleet_worker" not in schema
    assert (
        "REVOKE DELETE, TRUNCATE, REFERENCES, TRIGGER ON apply_attempts FROM fleet_worker"
        in schema
    )


def test_apply_worker_schema_check_passes_with_current_schema(fleet_db):
    with pgqueue.connect(fleet_db) as conn:
        fleet_schema.require_apply_result_event_schema(conn)
        fleet_schema.require_apply_attempt_schema(conn)


def test_apply_worker_schema_check_reports_missing_columns():
    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, *args, **kwargs):
            pass

        def fetchone(self):
            return {"contract": {"contract": "apply", "apply_result_event_ready": False}}

    class _Conn:
        def cursor(self):
            return _Cursor()

        def rollback(self):
            pass

    with pytest.raises(RuntimeError, match="apply-result worker contract"):
        fleet_schema.require_apply_result_event_schema(_Conn())


def test_apply_attempt_schema_check_reports_missing_columns():
    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, *args, **kwargs):
            pass

        def fetchone(self):
            return {"contract": {"contract": "apply", "apply_attempt_ready": False}}

    class _Conn:
        def cursor(self):
            return _Cursor()

        def rollback(self):
            pass

    with pytest.raises(RuntimeError, match="apply-attempt worker contract"):
        fleet_schema.require_apply_attempt_schema(_Conn())


def test_linkedin_queue_freshness_columns(fleet_db):
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='linkedin_queue'")
        cols = {r["column_name"] for r in cur.fetchall()}
    for c in (
        "linkedin_resolve_status",
        "linkedin_resolved_at",
        "linkedin_resolve_error",
        "linkedin_unresolved_kind",
        "linkedin_next_action",
    ):
        assert c in cols, f"linkedin_queue missing {c}"


def test_worker_canary_security_definer_schema_and_upgrade(fleet_db):
    charge_columns = {
        "linkedin_canary_charge_attempt",
        "linkedin_canary_charge_worker",
        "linkedin_canary_charge_policy_version",
        "linkedin_canary_charge_capacity",
        "linkedin_canary_charge_exhausted",
        "linkedin_canary_refunded_attempt",
    }
    with pgqueue.connect(fleet_db) as conn:
        with conn.cursor() as cur:
            worker_functions = {
                "fleet_worker_authorize_lease",
                "fleet_worker_refund_linkedin_canary",
                "fleet_worker_lease_ats",
                "fleet_worker_lease_linkedin",
                "fleet_worker_requeue",
                "fleet_worker_park_infrastructure",
                "fleet_worker_terminalize",
                "fleet_worker_park",
            }
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name='linkedin_queue'"
            )
            assert charge_columns <= {row["column_name"] for row in cur.fetchall()}
            cur.execute(
                "SELECT p.proname, p.prosecdef, p.proconfig, "
                "COALESCE(bool_or(acl.grantee=0 AND acl.privilege_type='EXECUTE'),FALSE) AS public_execute "
                "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
                "LEFT JOIN LATERAL aclexplode(COALESCE(p.proacl, acldefault('f',p.proowner))) acl ON TRUE "
                "WHERE n.nspname='public' AND p.proname = ANY(%s) "
                "GROUP BY p.proname,p.prosecdef,p.proconfig ORDER BY p.proname"
                , (list(worker_functions),)
            )
            rows = cur.fetchall()
            assert {row["proname"] for row in rows} == worker_functions
            for row in rows:
                assert row["prosecdef"] is True
                assert row["proconfig"] == ["search_path=pg_catalog, public"]
                assert row["public_execute"] is False

            for column in charge_columns:
                cur.execute(sql.SQL("ALTER TABLE linkedin_queue DROP COLUMN {}").format(sql.Identifier(column)))
            cur.execute("ALTER TABLE apply_queue DROP COLUMN worker_lease_id")
            cur.execute("ALTER TABLE linkedin_queue DROP COLUMN worker_lease_id")
            cur.execute("DROP TABLE fleet_worker_lease_ledger")
        conn.commit()
        fleet_schema.ensure_schema_v3(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name='linkedin_queue'"
            )
            assert charge_columns <= {row["column_name"] for row in cur.fetchall()}
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name='fleet_worker_lease_ledger'"
            )
            assert cur.fetchone() is not None
            for table in ("apply_queue", "linkedin_queue"):
                cur.execute(
                    "SELECT 1 FROM information_schema.columns WHERE table_schema='public' "
                    "AND table_name=%s AND column_name='worker_lease_id'",
                    (table,),
                )
                assert cur.fetchone() is not None


def test_fleet_config_v3_columns(fleet_db):
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='fleet_config'")
        cols = {r["column_name"] for r in cur.fetchall()}
    for c in ("approval_threshold", "approval_policy", "approval_sampling_rate",
              "cost_cap_daily_usd", "cost_cap_total_usd", "pinned_worker_version",
              "canary_enabled", "canary_remaining", "ats_apply_mode",
              "linkedin_canary_enabled", "linkedin_canary_remaining",
              "linkedin_apply_mode", "daily_apply_target", "ats_policy_version",
              "linkedin_policy_version"):
        assert c in cols, f"fleet_config missing {c}"


def test_policy_registry_enforces_lane_status_and_one_active_policy(fleet_db):
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO fleet_decision_policies (policy_version, lane, status) "
            "VALUES ('ats-v1', 'ats', 'active')"
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            cur.execute(
                "INSERT INTO fleet_decision_policies (policy_version, lane, status) "
                "VALUES ('ats-v2', 'ats', 'active')"
            )
        conn.rollback()
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                "INSERT INTO fleet_decision_policies (policy_version, lane, status) "
                "VALUES ('bad', 'email', 'active')"
            )


def test_fleet_config_policy_foreign_keys_are_lane_specific(fleet_db):
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO fleet_decision_policies (policy_version, lane, status) "
            "VALUES ('linkedin-v1', 'linkedin', 'active')"
        )
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            cur.execute("UPDATE fleet_config SET ats_policy_version='linkedin-v1' WHERE id=1")


def test_queue_provenance_constraints_and_policy_foreign_key(fleet_db):
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO fleet_decision_policies (policy_version, lane, status) "
            "VALUES ('ats-v1', 'ats', 'active')"
        )
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            _insert_canonical(
                cur,
                "apply_queue",
                _canonical_row(table="apply_queue", policy_version="missing"),
            )
        conn.rollback()
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                "INSERT INTO apply_queue "
                "(url, application_url, score, decision_action) "
                "VALUES ('u2', 'https://example.test/apply', 1, 'maybe')"
            )


@pytest.mark.parametrize("table", ["apply_queue", "linkedin_queue"])
@pytest.mark.parametrize(
    ("overrides", "case"),
    [
        ({"decision_id": "only", "policy_version": None, "decision_action": None,
          "qualification_verdict": None, "qualification_score": None,
          "qualification_floor": None, "preference_score": None, "outcome_score": None,
          "final_score": None, "decision_confidence": None, "decision_created_at": None,
          "decision_expires_at": None, "input_hash": None}, "decision-id-only"),
        ({"decision_created_at": None}, "missing-created-at"),
        ({"decision_expires_at": None}, "missing-expires-at"),
        ({"outcome_score": None}, "missing-component-score"),
        ({"decision_id": "  "}, "blank-decision-id"),
        ({"policy_version": "  "}, "blank-policy-version"),
        ({"input_hash": "  "}, "blank-hash"),
        ({"decision_expires_at": "2026-07-10T12:00:00Z"}, "reversed-expiry"),
        ({"qualification_score": float("nan")}, "nan-score"),
        ({"decision_confidence": float("inf")}, "infinite-confidence"),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_queue_rejects_incomplete_or_invalid_canonical_provenance(fleet_db, table, overrides, case):
    del case
    lane = "ats" if table == "apply_queue" else "linkedin"
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO fleet_decision_policies (policy_version, lane, status) "
            "VALUES ('policy-1', %s, 'active')",
            (lane,),
        )
        with pytest.raises(psycopg.errors.CheckViolation):
            _insert_canonical(cur, table, _canonical_row(table=table, **overrides))


@pytest.mark.parametrize("table", ["apply_queue", "linkedin_queue"])
def test_queue_accepts_legacy_all_null_and_complete_canonical_rows(fleet_db, table):
    lane = "ats" if table == "apply_queue" else "linkedin"
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {table} (url, application_url, score) VALUES ('legacy', 'https://example.test', 1)"
        )
        cur.execute(
            "INSERT INTO fleet_decision_policies (policy_version, lane, status) "
            "VALUES ('policy-1', %s, 'active')",
            (lane,),
        )
        _insert_canonical(cur, table, _canonical_row(table=table))


@pytest.mark.parametrize("table", ["apply_queue", "linkedin_queue"])
def test_existing_queue_is_explicitly_upgraded_to_fresh_canonical_schema(fleet_pg, table):
    lane = "ats" if table == "apply_queue" else "linkedin"
    canonical_index = f"idx_{table}_canonical_lease"
    with pgqueue.connect_admin(fleet_pg) as conn:
        pgqueue.ensure_schema(conn)
        fleet_schema.ensure_schema_v3(conn)
        with conn.cursor() as cur:
            fresh_objects = _canonical_schema_objects(cur, table)
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE {table}")
            cur.execute(f"DROP INDEX IF EXISTS idx_{table}_canonical_lease")
            for column in CANONICAL_QUEUE_COLUMNS:
                cur.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {column} CASCADE")
            cur.execute(
                f"INSERT INTO {table} (url, application_url, score) "
                "VALUES ('legacy', 'https://linkedin.com/jobs/view/1', 1)"
            )
        conn.commit()

        fleet_schema.ensure_schema_v3(conn)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema=current_schema() AND table_name=%s",
                (table,),
            )
            cols = {r["column_name"] for r in cur.fetchall()}
            cur.execute(f"SELECT lane, decision_id FROM {table} WHERE url='legacy'")
            legacy = cur.fetchone()
            upgraded_objects = _canonical_schema_objects(cur, table)
        conn.commit()

    assert CANONICAL_QUEUE_COLUMNS <= cols
    assert legacy == {"lane": lane, "decision_id": None}
    assert upgraded_objects == fresh_objects
    constraint_names = {name for name, _ in upgraded_objects[0]}
    assert constraint_names == {f"{table}_{suffix}" for suffix in CANONICAL_CONSTRAINT_SUFFIXES}
    assert {name for name, _ in upgraded_objects[1]} == {canonical_index}
    index_definition = next(iter(upgraded_objects[1]))[1]
    for column in CANONICAL_QUEUE_COLUMNS:
        assert column in index_definition
    assert "decision_action = 'apply'" in index_definition
    assert "qualification_verdict = 'qualified'" in index_definition


def test_autotriage_actions_schema(fleet_db):
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='autotriage_actions'")
        cols = {r["column_name"] for r in cur.fetchall()}
    for c in (
        "url", "worker_id", "chosen_action", "decision_source", "confidence",
        "action_status", "prior_status", "prior_apply_error", "evidence",
        "how_to_reverse",
    ):
        assert c in cols, f"autotriage_actions missing {c}"


def test_fleet_console_audit_schema(fleet_db):
    with pgqueue.connect(fleet_db) as conn, conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='fleet_console_audit'")
        cols = {r["column_name"] for r in cur.fetchall()}
    for c in ("id", "action", "actor", "lane", "target", "message", "ok", "created_at"):
        assert c in cols, f"fleet_console_audit missing {c}"


def test_challenge_rate_is_generated(fleet_db):
    # the generated challenge_rate = (captcha+block)/(success+captcha+block)
    with pgqueue.connect(fleet_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO rate_governor (scope_key, daily_cap, success_24h, captcha_24h, block_24h) "
                "VALUES ('host:example.com', 100, 8, 1, 1)"
            )
        conn.commit()
        with conn.cursor() as cur:
            cur.execute("SELECT challenge_rate FROM rate_governor WHERE scope_key='host:example.com'")
            rate = cur.fetchone()["challenge_rate"]
    assert abs(rate - 0.2) < 1e-6  # (1+1)/(8+1+1)


def test_worker_heartbeat_agent_model_columns(fleet_db):
    from applypilot.apply import pgqueue

    with pgqueue.connect(fleet_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='worker_heartbeat'"
            )
            cols = {r["column_name"] for r in cur.fetchall()}

    assert "current_agent" in cols
    assert "current_model" in cols
    assert "agent_chain" in cols
    assert "last_agent_switch_at" in cols
    assert "last_agent_switch_reason" in cols
