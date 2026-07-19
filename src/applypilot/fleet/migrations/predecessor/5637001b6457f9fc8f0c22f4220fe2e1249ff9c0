"""Apply the v3 fleet schema (base + extensions) idempotently."""
from __future__ import annotations

import os
import math
import time
from pathlib import Path

from applypilot.apply import pgqueue

_SCHEMA_V3_SQL = Path(__file__).with_name("schema_v3.sql")
_SCHEMA_LOCK_KEY = "applypilot:schema:v3"

_APPLY_RESULT_EVENT_REQUIRED_COLUMNS = frozenset({
    "queue_name",
    "url",
    "worker_id",
    "status",
    "apply_status",
    "apply_error",
    "target_host",
    "home_ip",
    "agent",
    "agent_model",
    "est_cost_usd",
    "apply_duration_ms",
    "result_line",
    "source",
    "route",
    "failure_class",
    "tool_calls_total",
    "application_tool_calls",
    "last_tool",
    "host_policy",
    "result_metadata",
    "job_log_path",
    "transcript_digest",
    "final_result_source",
})
_APPLY_ATTEMPT_REQUIRED_COLUMNS = frozenset({
    "attempt_id",
    "queue_name",
    "url",
    "dedup_key",
    "worker_id",
    "route",
    "route_version",
    "state",
    "submit_started_at",
    "finalized_at",
    "verification_method",
    "verification_ref",
    "evidence",
    "created_at",
})
_APPLY_QUEUE_COST_REQUIRED_COLUMNS = frozenset({"cumulative_cost_usd"})


def _nonnegative_timeout(value, default: float = 30.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) and parsed >= 0 else default


_SCHEMA_LOCK_TIMEOUT_SECONDS = _nonnegative_timeout(
    os.environ.get("APPLYPILOT_SCHEMA_LOCK_TIMEOUT_SECONDS"),
)

_REQUIRED_TABLES = {
    "agent_availability",
    "answer_bank",
    "applied_set",
    "apply_queue",
    "apply_result_events",
    "auth_challenge",
    "autotriage_actions",
    "command_acks",
    "compute_queue",
    "dedup_repair_actions",
    "discovered_postings",
    "email_reconcile_actions",
    "fleet_assets",
    "fleet_config",
    "fleet_console_audit",
    "fleet_diagnoses",
    "fleet_knobs",
    "fleet_machine_blackout",
    "inbox_events",
    "inbox_outcomes",
    "linkedin_queue",
    "llm_usage",
    "otp_request",
    "poison_jobs",
    "rate_governor",
    "remote_commands",
    "search_tasks",
    "worker_heartbeat",
    "workers",
}

_REQUIRED_COLUMNS = {
    "apply_queue": {
        "approved_batch",
        "dedup_key",
        "eligibility_required",
        "eligibility_status",
        "execution_route",
        "lane",
        "liveness_check_count",
        "liveness_check_expires_at",
        "liveness_check_owner",
        "liveness_checked_at",
        "liveness_consecutive_uncertain",
        "liveness_reason",
        "liveness_required",
        "liveness_status",
        "routing_required",
        "session_required",
        "target_host",
        "tenant_profile_id",
    },
    "fleet_config": {
        "ats_apply_mode",
        "ats_paused",
        "canary_enabled",
        "canary_remaining",
        "linkedin_apply_mode",
        "linkedin_canary_enabled",
        "linkedin_canary_remaining",
    },
    "linkedin_queue": {
        "approved_batch",
        "dedup_key",
        "linkedin_next_action",
        "linkedin_resolve_status",
        "linkedin_resolved_at",
    },
}


def _verify_schema_v3(conn) -> None:
    """Read-only compatibility check for least-privilege fleet connections."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name = ANY(%s)",
            (sorted(_REQUIRED_TABLES),),
        )
        present_tables = {row["table_name"] for row in cur.fetchall()}
        cur.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name = ANY(%s)",
            (sorted(_REQUIRED_COLUMNS),),
        )
        present_columns: dict[str, set[str]] = {}
        for row in cur.fetchall():
            present_columns.setdefault(row["table_name"], set()).add(row["column_name"])

    missing_tables = sorted(_REQUIRED_TABLES - present_tables)
    missing_columns = sorted(
        f"{table}.{column}"
        for table, required in _REQUIRED_COLUMNS.items()
        for column in required - present_columns.get(table, set())
    )
    if missing_tables or missing_columns:
        details = []
        if missing_tables:
            details.append("tables=" + ",".join(missing_tables))
        if missing_columns:
            details.append("columns=" + ",".join(missing_columns))
        raise RuntimeError(
            "fleet schema v3 is incomplete or inaccessible to this database role, "
            "and the role cannot migrate it; run a home/controller command with the owner DSN, "
            "then repair fleet_worker grants if needed (" + "; ".join(details) + ")"
        )


def _acquire_migration_lock(conn, *, timeout_seconds: float) -> None:
    timeout_seconds = _nonnegative_timeout(timeout_seconds)
    deadline = time.monotonic() + timeout_seconds
    while True:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_try_advisory_lock(hashtext(%s)) AS acquired",
                (_SCHEMA_LOCK_KEY,),
            )
            if bool(cur.fetchone()["acquired"]):
                return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for the fleet schema migration lock; "
                "another controller may be stalled while migrating"
            )
        time.sleep(0.1)


def ensure_schema_v3(conn, *, lock_timeout_seconds: float = _SCHEMA_LOCK_TIMEOUT_SECONDS) -> None:
    """Idempotently apply the base fleet schema then the v3 extensions.

    Safe to run on every broker/home/worker startup. Schema owners run
    ``pgqueue.ensure_schema`` and the v3 migration under an advisory lock.
    Least-privilege workers perform a read-only compatibility check instead.
    """
    with conn.cursor() as cur:
        cur.execute(
            "WITH existing AS (SELECT to_regclass('public.apply_queue') AS oid) "
            "SELECT CASE "
            "WHEN existing.oid IS NULL "
            "THEN has_schema_privilege(current_user, 'public', 'CREATE') "
            "ELSE EXISTS ("
            "  SELECT 1 FROM pg_class c "
            "  WHERE c.oid = existing.oid "
            "    AND (c.relowner = (SELECT oid FROM pg_roles WHERE rolname=current_user) "
            "         OR pg_has_role(current_user, c.relowner, 'MEMBER'))"
            ") END AS can_migrate FROM existing"
        )
        can_migrate = bool(cur.fetchone()["can_migrate"])
    if not can_migrate:
        _verify_schema_v3(conn)
        return

    _acquire_migration_lock(conn, timeout_seconds=lock_timeout_seconds)
    try:
        pgqueue.ensure_schema(conn)
        sql = _SCHEMA_V3_SQL.read_text(encoding="utf-8")
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(hashtext(%s))", (_SCHEMA_LOCK_KEY,))
        conn.commit()


def _table_columns(conn, table_name: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema=current_schema() AND table_name=%s",
            (table_name,),
        )
        columns = {
            row["column_name"] if hasattr(row, "get") else row[0]
            for row in cur.fetchall()
        }
    try:
        conn.rollback()
    except Exception:
        pass
    return columns


def require_apply_result_event_schema(conn) -> None:
    """Fail closed when a worker cannot persist complete result evidence."""
    missing = sorted(
        _APPLY_RESULT_EVENT_REQUIRED_COLUMNS - _table_columns(conn, "apply_result_events")
    )
    if missing:
        raise RuntimeError(
            "fleet schema is missing apply_result_events columns: "
            + ", ".join(missing)
            + "; run applypilot-fleet-apply-home with the owner/home DSN once to migrate "
            "before starting remote apply workers"
        )
    missing_queue = sorted(
        _APPLY_QUEUE_COST_REQUIRED_COLUMNS - _table_columns(conn, "apply_queue")
    )
    if missing_queue:
        raise RuntimeError(
            "fleet schema is missing apply_queue columns: "
            + ", ".join(missing_queue)
            + "; run applypilot-fleet-apply-home with the owner/home DSN once to migrate "
            "before starting remote apply workers"
        )


def require_apply_attempt_schema(conn) -> None:
    """Fail closed before adapter submit ownership when its ledger is incomplete."""
    missing = sorted(
        _APPLY_ATTEMPT_REQUIRED_COLUMNS - _table_columns(conn, "apply_attempts")
    )
    if missing:
        raise RuntimeError(
            "fleet schema is missing apply_attempts columns: "
            + ", ".join(missing)
            + "; run applypilot-fleet-apply-home with the owner/home DSN once before "
            "enabling APPLYPILOT_GREENHOUSE_ADAPTER_SUBMIT"
        )
