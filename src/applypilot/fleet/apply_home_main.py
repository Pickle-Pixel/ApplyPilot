"""applypilot-fleet-apply-home: the owner driver for the offsite apply lane.
push (stage UNAPPROVED + backfill applied_set + push email-outcome summaries), approve
(arm a batch; refuse unless the canary is armed), pull, canary/lift-canary,
challenges + resolve-challenge, status."""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
from pathlib import Path
import re
import sys
import uuid

from applypilot import config
from applypilot.apply import pgqueue
from applypilot.fleet import emergency_admission, queue, queue_diagnosis, schema as fleet_schema, sync

logger = logging.getLogger("applypilot.fleet.apply_home_main")


def crash_liveness_task_diagnosis(log_path: str | os.PathLike[str] | None = None) -> dict:
    """Report freshness of the scheduled evidence-only liveness runner log."""
    path = Path(log_path) if log_path else Path(__file__).resolve().parents[3] / ".fleet-logs" / "crash-liveness.log"
    if not path.exists():
        return {"available": False, "log_path": str(path), "healthy": False}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    exit_match = None
    for line in reversed(lines):
        match = re.search(r"^\[(?P<at>[^]]+)\] exit=(?P<code>\d+)$", line.strip())
        if match:
            exit_match = match
            break
    last_run_at = None
    last_exit = None
    if exit_match:
        try:
            last_run_at = _dt.datetime.fromisoformat(exit_match.group("at"))
        except ValueError:
            last_run_at = None
        last_exit = int(exit_match.group("code"))
    now = _dt.datetime.now(_dt.timezone.utc)
    age_seconds = None
    if last_run_at:
        if last_run_at.tzinfo is None:
            last_run_at = last_run_at.replace(tzinfo=_dt.timezone.utc)
        age_seconds = max(0.0, (now - last_run_at.astimezone(_dt.timezone.utc)).total_seconds())
    healthy = last_exit == 0 and age_seconds is not None and age_seconds <= 90 * 60
    return {
        "available": True,
        "log_path": str(path),
        "last_run_at": last_run_at,
        "last_exit": last_exit,
        "age_seconds": age_seconds,
        "healthy": healthy,
    }


def _configure_stdio() -> None:
    """Keep Windows owner-review commands usable with Unicode job metadata."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            pass


def push_home(conn, *, sqlite_conn=None, score_floor: float = 7.0, limit: int | None = None,
              lane_filter: bool = True, diagnostics: bool = False,
              approved_batch: str | None = None) -> int | dict[str, int]:
    """The home 'push' cadence: stage apply-eligible jobs (+ backfill applied_set,
    inside push_apply_eligible), and best-effort push the brain's email_events outcome
    summaries into PG inbox_outcomes (R8 feedback loop). The outcomes push is advisory
    reporting only -- a transient/UndefinedTable failure must never block staging the
    apply queue, so it is logged and swallowed rather than raised."""
    pushed = sync.push_apply_eligible(sqlite_conn=sqlite_conn, pg_conn=conn,
                                       score_floor=score_floor, limit=limit,
                                       lane_filter=lane_filter,
                                       diagnostics=diagnostics,
                                       approved_batch=approved_batch)
    try:
        sync.push_inbox_outcomes(sqlite_conn=sqlite_conn, pg_conn=conn)
    except Exception:
        logger.warning("push_inbox_outcomes failed (best-effort, non-fatal)", exc_info=True)
    return pushed


def set_canary(conn, k: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE fleet_config "
            "SET ats_apply_mode='canary', canary_enabled=TRUE, canary_remaining=%s, "
            "updated_at=now() "
            "WHERE id=1 AND ats_paused=FALSE",
            (k,),
        )
        if cur.rowcount == 0:
            conn.rollback()
            raise RuntimeError("cannot arm canary while ATS pause is active")
    conn.commit()


def lift_canary(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE fleet_config "
            "SET ats_apply_mode='stopped', canary_enabled=FALSE, canary_remaining=NULL "
            "WHERE id=1"
        )
    conn.commit()


def set_spend_cap(conn, cap_usd: float) -> dict[str, float]:
    """Set the fleet-wide hard $ spend ceiling on ``fleet_config`` (spec S6).

    This is the audited control-plane replacement for the retired raw ``psql`` write to
    ``fleet_config`` in ``load-canary-home.ps1``. The caller's
    connection is the central ``pgqueue.connect`` mapped-principal channel (validated
    by ``validate_runtime_conninfo``/``assert_mapped_principal`` -- never the postgres
    admin or the shared ``fleet_worker`` role, never an ``options``/ambient/indirection
    bypass). The write is a single parameterized statement inside one transaction; it
    fails closed (raising, so the CLI exits non-zero) when the singleton config row is
    absent, and records a durable ``fleet_console_audit`` row for the change.
    """
    if cap_usd < 0:
        raise ValueError("spend cap must be non-negative (0 means UNCAPPED)")
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE fleet_config SET spend_cap_usd=%s, updated_at=now() WHERE id=1",
            (cap_usd,),
        )
        if cur.rowcount != 1:
            conn.rollback()
            raise RuntimeError("fleet_config row id=1 is missing; refusing to set the spend cap")
        cur.execute(
            "INSERT INTO fleet_console_audit "
            "(action, actor, lane, target, message, ok) VALUES (%s,%s,%s,%s,%s,%s)",
            (
                "set_spend_cap",
                "apply-home",
                "ats",
                f"${cap_usd:.2f}",
                "set fleet-wide hard spend cap via the mapped control-plane principal",
                True,
            ),
        )
    conn.commit()
    return {"spend_cap_usd": cap_usd}


def _canary_capacity(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT ats_apply_mode, canary_enabled, canary_remaining FROM fleet_config WHERE id=1")
        row = cur.fetchone()
    if not row or row["ats_apply_mode"] != "canary" or not row["canary_enabled"]:
        return 0
    remaining = row["canary_remaining"]
    if remaining is None:
        return 0
    return max(int(remaining), 0)


def _canary_armed(conn) -> bool:
    return _canary_capacity(conn) > 0


def approve(conn, *, urls=None, all_pushed=False) -> str:
    """Stamp a fresh batch token on the given (or all queued-unapproved) rows. REFUSES
    unless the canary is armed (so the runbook's arm-then-approve order can't invert)."""
    capacity = _canary_capacity(conn)
    if capacity <= 0:
        raise SystemExit("refusing to approve: arm the canary with positive remaining capacity first (apply-home canary <K>)")
    if all_pushed:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT q.url FROM apply_queue q CROSS JOIN fleet_config cfg "
                "WHERE q.status='queued' AND q.approved_batch IS NULL "
                "AND q.score >= COALESCE(cfg.approval_threshold, 7) "
                "ORDER BY q.score DESC, q.url LIMIT %s",
                (capacity,),
            )
            urls = [r["url"] for r in cur.fetchall()]
    token = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    queue.approve_jobs(conn, urls or [], token)
    return token


def retire_stale_unapproved(
    conn, *, older_days: int = 7, limit: int = 1000, execute: bool = False,
    include_safe_requeues: bool = False,
) -> dict[str, int | bool]:
    """Retire never-leased ATS rows that have remained unapproved past the TTL.

    These rows are inventory drift, not retryable applications: the lease gate
    requires ``approved_batch`` and they have no attempts or lease timestamp.
    Dry-run is the default; execution marks only those rows ``failed/expired``
    with an explicit reason and records one operator audit row.
    """
    if older_days < 1:
        raise ValueError("older_days must be at least 1")
    if limit < 1:
        raise ValueError("limit must be positive")
    requeue_clause = ""
    if include_safe_requeues:
        requeue_clause = (
            " OR (COALESCE(apply_error,'') = 'requeued_by_autotriage:pre_touch_crash' "
            "AND NOT EXISTS (SELECT 1 FROM apply_result_events e "
            "WHERE e.queue_name='apply_queue' AND e.url=apply_queue.url "
            "AND COALESCE(e.application_tool_calls,0)>0))"
        )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT url FROM apply_queue "
            "WHERE status='queued' AND lane='ats' AND approved_batch IS NULL "
            "AND COALESCE(attempts, 0)=0 AND (last_attempted_at IS NULL" + requeue_clause + ") "
            "AND updated_at < now() - make_interval(days => %s) "
            "ORDER BY updated_at ASC, url LIMIT %s",
            (older_days, limit),
        )
        urls = [row["url"] for row in cur.fetchall()]
        if not execute:
            conn.rollback()
            return {"dry_run": True, "candidates": len(urls), "retired": 0}
        if urls:
            cur.execute(
                "UPDATE apply_queue SET status='failed', apply_status='expired', "
                "apply_error='stale_unapproved', updated_at=now() "
                "WHERE url = ANY(%s) AND status='queued' AND lane='ats' "
                "AND approved_batch IS NULL AND COALESCE(attempts,0)=0 "
                "AND (last_attempted_at IS NULL" + requeue_clause + ") RETURNING url",
                (urls,),
            )
            retired = cur.rowcount
            cur.execute(
                "INSERT INTO fleet_console_audit "
                "(action, actor, lane, target, message, ok) VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    "retire_stale_unapproved",
                    "apply-home",
                    "ats",
                    f"{retired} rows",
                    f"retired stale unapproved ATS rows older than {older_days} days",
                    True,
                ),
            )
        else:
            retired = 0
    conn.commit()
    return {"dry_run": False, "candidates": len(urls), "retired": retired}


def quarantine_unsafe_approved(
    conn, *, limit: int = 1000, execute: bool = False
) -> dict[str, int | bool]:
    """Remove previously attempted approved rows from the blind lease backlog.

    These rows remain reviewable as blocked/manual-review records. They are not
    marked applied or failed because the original submission outcome is unknown.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    predicate = (
        "q.status='queued' AND q.lane='ats' AND q.approved_batch IS NOT NULL "
        "AND (COALESCE(q.apply_error,'') ILIKE 'requeued_by_%%' OR EXISTS ("
        "SELECT 1 FROM apply_result_events e WHERE e.queue_name='apply_queue' "
        "AND e.url=q.url AND (COALESCE(e.application_tool_calls,0)>0 "
        "OR COALESCE(e.apply_error,'') ILIKE 'requeued_by_%%')))"
    )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT q.url FROM apply_queue q WHERE " + predicate +
            " ORDER BY q.score DESC, q.url LIMIT %s",
            (limit,),
        )
        urls = [row["url"] for row in cur.fetchall()]
        if not execute:
            conn.rollback()
            return {"dry_run": True, "candidates": len(urls), "blocked": 0}
        blocked = 0
        if urls:
            cur.execute(
                "UPDATE apply_queue q SET status='blocked', "
                "apply_status='manual_review_required', apply_error='unsafe_prior_attempt', "
                "lease_owner=NULL, lease_expires_at=NULL, synced_to_home_at=NULL, updated_at=now() "
                "WHERE q.url = ANY(%s) AND " + predicate + " RETURNING q.url",
                (urls,),
            )
            blocked = cur.rowcount
            cur.execute(
                "INSERT INTO fleet_console_audit "
                "(action, actor, lane, target, message, ok) VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    "quarantine_unsafe_approved", "apply-home", "ats", f"{blocked} rows",
                    "quarantined approved rows with prior retry/browser evidence", True,
                ),
            )
    conn.commit()
    return {"dry_run": False, "candidates": len(urls), "blocked": blocked}


def clear_blocked_leases(conn, *, limit: int = 1000, execute: bool = False) -> dict[str, int | bool]:
    """Clear lease metadata from terminal blocked rows without changing outcomes."""
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT url FROM apply_queue WHERE status='blocked' "
            "AND (lease_owner IS NOT NULL OR lease_expires_at IS NOT NULL) "
            "ORDER BY updated_at ASC, url LIMIT %s",
            (limit,),
        )
        urls = [row["url"] for row in cur.fetchall()]
        cleared = 0
        if execute and urls:
            cur.execute(
                "UPDATE apply_queue SET lease_owner=NULL, lease_expires_at=NULL, "
                "updated_at=now() WHERE status='blocked' AND url=ANY(%s) "
                "RETURNING url",
                (urls,),
            )
            cleared = cur.rowcount
            cur.execute(
                "INSERT INTO fleet_console_audit "
                "(action, actor, lane, target, message, ok) VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    "clear_blocked_leases", "apply-home", "ats", f"{cleared} rows",
                    "cleared lease metadata from blocked rows without changing outcomes", True,
                ),
            )
    (conn.commit() if execute else conn.rollback())
    return {"dry_run": not execute, "candidates": len(urls), "cleared": cleared}


def clear_terminal_leases(conn, *, limit: int = 5000, execute: bool = False) -> dict[str, int | bool]:
    """Clear stale lease metadata from terminal rows without changing outcomes."""
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT url
              FROM apply_queue
             WHERE status IN ('applied','failed','blocked','crash_unconfirmed')
               AND (lease_owner IS NOT NULL OR lease_expires_at IS NOT NULL)
             ORDER BY updated_at ASC, url
             LIMIT %s
            """,
            (limit,),
        )
        urls = [row["url"] for row in cur.fetchall()]
        cleared = 0
        if execute and urls:
            cur.execute(
                """
                UPDATE apply_queue
                   SET lease_owner=NULL, lease_expires_at=NULL, updated_at=now()
                 WHERE url=ANY(%s)
                   AND status IN ('applied','failed','blocked','crash_unconfirmed')
                """,
                (urls,),
            )
            cleared = cur.rowcount
            cur.execute(
                """
                INSERT INTO fleet_console_audit
                    (action, actor, lane, target, message, ok)
                SELECT 'clear_terminal_leases','apply-home','ats',url,
                       'cleared stale lease metadata from terminal row; outcome unchanged',TRUE
                  FROM apply_queue WHERE url=ANY(%s)
                """,
                (urls,),
            )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(urls), "cleared": cleared}


def quarantine_invalid_queued(conn, *, limit: int = 1000, execute: bool = False) -> dict[str, int | bool]:
    """Park queued ATS rows that violate source-lane or identity invariants."""
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT url
              FROM apply_queue
             WHERE lane='ats' AND status='queued'
               AND (url ILIKE '%%linkedin.com/%%'
                    OR NULLIF(BTRIM(company),'') IS NULL
                    OR NULLIF(BTRIM(title),'') IS NULL)
             ORDER BY updated_at ASC, url
             LIMIT %s
            """,
            (limit,),
        )
        urls = [row["url"] for row in cur.fetchall()]
        quarantined = 0
        if execute and urls:
            cur.execute(
                """
                UPDATE apply_queue
                   SET status='blocked', apply_status='manual_review_required',
                       apply_error='invalid_queued_inventory', updated_at=now(),
                       lease_owner=NULL, lease_expires_at=NULL
                 WHERE url=ANY(%s) AND lane='ats' AND status='queued'
                """,
                (urls,),
            )
            quarantined = cur.rowcount
            cur.execute(
                """
                INSERT INTO fleet_console_audit
                    (action, actor, lane, target, message, ok)
                SELECT 'quarantine_invalid_queued','apply-home','ats',url,
                       'parked invalid queued inventory for manual review; no submission occurred',TRUE
                  FROM apply_queue WHERE url=ANY(%s)
                """,
                (urls,),
            )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(urls), "quarantined": quarantined}


def quarantine_missing_evidence_failures(
    conn, *, limit: int = 1000, execute: bool = False,
) -> dict[str, int | bool]:
    """Move malformed failed rows with no result event to explicit manual review."""
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.url
              FROM apply_queue q
             WHERE q.lane='ats' AND q.status='failed'
               AND q.apply_error IN ('failed:reason','failed:unspecified','unspecified')
               AND NOT EXISTS (
                   SELECT 1 FROM apply_result_events e
                    WHERE e.queue_name='apply_queue' AND e.url=q.url
               )
             ORDER BY q.updated_at ASC, q.url
             LIMIT %s
            """,
            (limit,),
        )
        urls = [row["url"] for row in cur.fetchall()]
        quarantined = 0
        if execute and urls:
            cur.execute(
                """
                UPDATE apply_queue
                   SET status='blocked', apply_status='manual_review_required',
                       apply_error='submission_uncertain:evidence_gap',
                       lease_owner=NULL, lease_expires_at=NULL, updated_at=now()
                 WHERE url=ANY(%s) AND status='failed'
                """,
                (urls,),
            )
            quarantined = cur.rowcount
            cur.execute(
                """
                INSERT INTO fleet_console_audit
                    (action, actor, lane, target, message, ok)
                SELECT 'quarantine_missing_evidence_failure','apply-home','ats',url,
                       'parked failed row with no durable result event for manual review',TRUE
                  FROM apply_queue WHERE url=ANY(%s)
                """,
                (urls,),
            )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(urls), "quarantined": quarantined}


def quarantine_missing_infrastructure_evidence(
    conn, *, limit: int = 1000, execute: bool = False,
) -> dict[str, int | bool]:
    """Park browser-infrastructure failures whose execution event is missing.

    Without a result event the tool cannot prove that the browser was never
    reached, so these rows are deliberately made explicit manual-review items
    rather than being retried as if they were preflight-only failures.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.url
              FROM apply_queue q
             WHERE q.lane='ats' AND q.status='failed'
               AND (q.apply_error ILIKE '%%browser%%'
                    OR q.apply_error ILIKE '%%playwright%%'
                    OR q.apply_error ILIKE '%%backend%%'
                    OR q.apply_error ILIKE '%%connection_lost%%'
                    OR q.apply_error ILIKE '%%page_error%%')
               AND NOT EXISTS (
                   SELECT 1 FROM apply_result_events e
                    WHERE e.queue_name='apply_queue' AND e.url=q.url
               )
             ORDER BY q.updated_at ASC, q.url
             LIMIT %s
            """,
            (limit,),
        )
        urls = [row["url"] for row in cur.fetchall()]
        quarantined = 0
        if execute and urls:
            cur.execute(
                """
                UPDATE apply_queue
                   SET status='blocked', apply_status='manual_review_required',
                       apply_error='submission_uncertain:infrastructure_evidence_gap',
                       lease_owner=NULL, lease_expires_at=NULL, updated_at=now()
                 WHERE url=ANY(%s) AND status='failed'
                """,
                (urls,),
            )
            quarantined = cur.rowcount
            cur.execute(
                """
                INSERT INTO fleet_console_audit
                    (action, actor, lane, target, message, ok)
                SELECT 'quarantine_missing_infrastructure_evidence','apply-home','ats',url,
                       'parked browser failure with no durable result event; retry prohibited',TRUE
                  FROM apply_queue WHERE url=ANY(%s) AND status='blocked'
                """,
                (urls,),
            )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(urls), "quarantined": quarantined}


def quarantine_incomplete_infrastructure_evidence(
    conn, *, limit: int = 1000, execute: bool = False,
) -> dict[str, int | bool]:
    """Park infrastructure events that exist but omit tool-call evidence."""
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.url
              FROM apply_queue q
              JOIN LATERAL (
                  SELECT e.id, e.application_tool_calls
                    FROM apply_result_events e
                   WHERE e.queue_name='apply_queue' AND e.url=q.url
                   ORDER BY e.created_at DESC, e.id DESC
                   LIMIT 1
              ) e ON TRUE
             WHERE q.lane='ats' AND q.status='failed'
               AND (q.apply_error ILIKE '%%browser%%'
                    OR q.apply_error ILIKE '%%playwright%%'
                    OR q.apply_error ILIKE '%%backend%%'
                    OR q.apply_error ILIKE '%%connection_lost%%'
                    OR q.apply_error ILIKE '%%page_error%%')
               AND e.application_tool_calls IS NULL
             ORDER BY q.updated_at ASC, q.url
             LIMIT %s
            """,
            (limit,),
        )
        urls = [row["url"] for row in cur.fetchall()]
        quarantined = 0
        if execute and urls:
            cur.execute(
                """
                UPDATE apply_queue
                   SET status='blocked', apply_status='manual_review_required',
                       apply_error='submission_uncertain:incomplete_infrastructure_evidence',
                       lease_owner=NULL, lease_expires_at=NULL, updated_at=now()
                 WHERE url=ANY(%s) AND status='failed'
                """,
                (urls,),
            )
            quarantined = cur.rowcount
            cur.execute(
                """
                INSERT INTO fleet_console_audit
                    (action, actor, lane, target, message, ok)
                SELECT 'quarantine_incomplete_infrastructure_evidence','apply-home','ats',url,
                       'parked browser failure with incomplete result evidence; retry prohibited',TRUE
                  FROM apply_queue WHERE url=ANY(%s) AND status='blocked'
                """,
                (urls,),
            )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(urls), "quarantined": quarantined}


def quarantine_tool_touched_infrastructure(
    conn, *, limit: int = 1000, execute: bool = False,
) -> dict[str, int | bool]:
    """Protect infrastructure failures whose latest event shows browser activity."""
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.url
              FROM apply_queue q
              JOIN LATERAL (
                  SELECT e.application_tool_calls
                    FROM apply_result_events e
                   WHERE e.queue_name='apply_queue' AND e.url=q.url
                   ORDER BY e.created_at DESC, e.id DESC
                   LIMIT 1
              ) e ON TRUE
             WHERE q.lane='ats' AND q.status='failed'
               AND (q.apply_error ILIKE '%%browser%%'
                    OR q.apply_error ILIKE '%%playwright%%'
                    OR q.apply_error ILIKE '%%backend%%'
                    OR q.apply_error ILIKE '%%connection_lost%%'
                    OR q.apply_error ILIKE '%%page_error%%')
               AND e.application_tool_calls > 0
             ORDER BY q.updated_at ASC, q.url
             LIMIT %s
            """,
            (limit,),
        )
        urls = [row["url"] for row in cur.fetchall()]
        quarantined = 0
        if execute and urls:
            cur.execute(
                """
                UPDATE apply_queue
                   SET status='blocked', apply_status='manual_review_required',
                       apply_error='submission_uncertain:browser_tool_touched',
                       lease_owner=NULL, lease_expires_at=NULL, updated_at=now()
                 WHERE url=ANY(%s) AND status='failed'
                """,
                (urls,),
            )
            quarantined = cur.rowcount
            cur.execute(
                """
                INSERT INTO fleet_console_audit
                    (action, actor, lane, target, message, ok)
                SELECT 'quarantine_tool_touched_infrastructure','apply-home','ats',url,
                       'protected browser-touched failure from retry; outcome requires review',TRUE
                  FROM apply_queue WHERE url=ANY(%s) AND status='blocked'
                """,
                (urls,),
            )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(urls), "quarantined": quarantined}


def quarantine_no_confirmation_failures(
    conn, *, limit: int = 1000, execute: bool = False,
) -> dict[str, int | bool]:
    """Make failed no-confirmation outcomes explicit protected review rows."""
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT url FROM apply_queue
             WHERE lane='ats' AND status='failed' AND apply_error='failed:no_confirmation'
             ORDER BY updated_at ASC, url LIMIT %s
            """,
            (limit,),
        )
        urls = [row["url"] for row in cur.fetchall()]
        quarantined = 0
        if execute and urls:
            cur.execute(
                """
                UPDATE apply_queue
                   SET status='blocked', apply_status='manual_review_required',
                       apply_error='submission_uncertain:no_confirmation',
                       lease_owner=NULL, lease_expires_at=NULL, updated_at=now()
                 WHERE url=ANY(%s) AND status='failed' AND apply_error='failed:no_confirmation'
                """,
                (urls,),
            )
            quarantined = cur.rowcount
            cur.execute(
                """
                INSERT INTO fleet_console_audit
                    (action, actor, lane, target, message, ok)
                SELECT 'quarantine_no_confirmation_failure','apply-home','ats',url,
                       'protected no-confirmation failure from retry; outcome requires review',TRUE
                  FROM apply_queue WHERE url=ANY(%s) AND status='blocked'
                """,
                (urls,),
            )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(urls), "quarantined": quarantined}


def normalize_terminal_inventory(
    conn, *, limit: int = 5000, execute: bool = False,
) -> dict[str, int | bool]:
    """Move stale/expired inventory out of the failed retry pool.

    The original ``apply_error`` is retained for reporting.  No row with an
    application timestamp is eligible, and the transition is audit logged.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    reasons = ("stale_unapproved", "expired", "canonical_provenance_missing")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT url FROM apply_queue
             WHERE lane='ats' AND status='failed' AND apply_error=ANY(%s)
               AND applied_at IS NULL
             ORDER BY updated_at ASC, url LIMIT %s
            """,
            (list(reasons), limit),
        )
        urls = [row["url"] for row in cur.fetchall()]
        normalized = 0
        if execute and urls:
            cur.execute(
                """
                UPDATE apply_queue
                   SET status='blocked', apply_status='terminal_non_retryable',
                       lease_owner=NULL, lease_expires_at=NULL, updated_at=now()
                 WHERE url=ANY(%s) AND status='failed' AND applied_at IS NULL
                   AND apply_error=ANY(%s)
                """,
                (urls, list(reasons)),
            )
            normalized = cur.rowcount
            cur.execute(
                """
                INSERT INTO fleet_console_audit
                    (action, actor, lane, target, message, ok)
                SELECT 'normalize_terminal_inventory','apply-home','ats',url,
                       'moved stale or expired non-retryable inventory out of failed pool; reason retained',TRUE
                  FROM apply_queue WHERE url=ANY(%s) AND status='blocked'
                """,
                (urls,),
            )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(urls), "normalized": normalized}


def normalize_nonretryable_policy_failures(
    conn, *, limit: int = 1000, execute: bool = False,
) -> dict[str, int | bool]:
    """Move deterministic eligibility/routing failures out of the retry pool."""
    if limit < 1:
        raise ValueError("limit must be positive")
    reasons = (
        "failed:not_eligible_location", "adapter_unsupported",
        "failed:not_a_job_application", "host_policy:workday_tenant_requires_trust",
        "host_policy:login_gate_prone", "failed:contract_position_not_fulltime",
        "failed:part_time_role_not_eligible", "failed:contract_role_not_full_time_salaried",
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT url FROM apply_queue
             WHERE lane='ats' AND status='failed' AND apply_error=ANY(%s)
             ORDER BY updated_at ASC, url LIMIT %s
            """,
            (list(reasons), limit),
        )
        urls = [row["url"] for row in cur.fetchall()]
        normalized = 0
        if execute and urls:
            cur.execute(
                """
                UPDATE apply_queue
                   SET status='blocked', apply_status='terminal_non_retryable',
                       lease_owner=NULL, lease_expires_at=NULL, updated_at=now()
                 WHERE url=ANY(%s) AND status='failed' AND apply_error=ANY(%s)
                """,
                (urls, list(reasons)),
            )
            normalized = cur.rowcount
            cur.execute(
                """
                INSERT INTO fleet_console_audit
                    (action, actor, lane, target, message, ok)
                SELECT 'normalize_nonretryable_policy_failure','apply-home','ats',url,
                       'moved deterministic policy failure out of failed retry pool; reason retained',TRUE
                  FROM apply_queue WHERE url=ANY(%s) AND status='blocked'
                """,
                (urls,),
            )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(urls), "normalized": normalized}


def quarantine_suspicious_stuck_failures(
    conn, *, limit: int = 1000, execute: bool = False,
) -> dict[str, int | bool]:
    """Protect suspicious/stuck flows from automatic retry."""
    if limit < 1:
        raise ValueError("limit must be positive")
    reasons = ("failed:suspicious_page", "failed:stuck")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT url FROM apply_queue
             WHERE lane='ats' AND status='failed' AND apply_error=ANY(%s)
             ORDER BY updated_at ASC, url LIMIT %s
            """,
            (list(reasons), limit),
        )
        urls = [row["url"] for row in cur.fetchall()]
        quarantined = 0
        if execute and urls:
            cur.execute(
                """
                UPDATE apply_queue
                   SET status='blocked', apply_status='manual_review_required',
                       apply_error='submission_uncertain:suspicious_or_stuck',
                       lease_owner=NULL, lease_expires_at=NULL, updated_at=now()
                 WHERE url=ANY(%s) AND status='failed' AND apply_error=ANY(%s)
                """,
                (urls, list(reasons)),
            )
            quarantined = cur.rowcount
            cur.execute(
                """
                INSERT INTO fleet_console_audit
                    (action, actor, lane, target, message, ok)
                SELECT 'quarantine_suspicious_stuck_failure','apply-home','ats',url,
                       'protected suspicious or stuck flow from retry; outcome requires review',TRUE
                  FROM apply_queue WHERE url=ANY(%s) AND status='blocked'
                """,
                (urls,),
            )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(urls), "quarantined": quarantined}


def normalize_resolved_auth_failures(
    conn, *, limit: int = 1000, execute: bool = False,
) -> dict[str, int | bool]:
    """Move auth failures with a conclusively resolved challenge to review."""
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.url
              FROM apply_queue q
              JOIN LATERAL (
                  SELECT a.resolved_at, a.outcome
                    FROM auth_challenge a
                   WHERE a.url=q.url
                   ORDER BY a.raised_at DESC, a.id DESC
                   LIMIT 1
              ) a ON TRUE
             WHERE q.lane='ats' AND q.status='failed'
               AND (q.apply_error ILIKE '%%auth%%'
                    OR q.apply_error ILIKE '%%login%%'
                    OR q.apply_error ILIKE '%%email_verification%%')
               AND a.resolved_at IS NOT NULL
             ORDER BY q.updated_at ASC, q.url
             LIMIT %s
            """,
            (limit,),
        )
        urls = [row["url"] for row in cur.fetchall()]
        normalized = 0
        if execute and urls:
            cur.execute(
                """
                UPDATE apply_queue
                   SET status='blocked', apply_status='manual_review_required',
                       apply_error='auth_review:challenge_resolved',
                       lease_owner=NULL, lease_expires_at=NULL, updated_at=now()
                 WHERE url=ANY(%s) AND status='failed'
                """,
                (urls,),
            )
            normalized = cur.rowcount
            cur.execute(
                """
                INSERT INTO fleet_console_audit
                    (action, actor, lane, target, message, ok)
                SELECT 'normalize_resolved_auth_failure','apply-home','ats',url,
                       'challenge is resolved; retained auth outcome for manual review',TRUE
                  FROM apply_queue WHERE url=ANY(%s) AND status='blocked'
                """,
                (urls,),
            )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(urls), "normalized": normalized}


def quarantine_budget_limit_failures(
    conn, *, limit: int = 1000, execute: bool = False,
) -> dict[str, int | bool]:
    """Protect budget/rate-limit failures whose submission outcome is uncertain."""
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT url FROM apply_queue
             WHERE lane='ats' AND status='failed'
               AND (apply_error ILIKE '%%budget%%' OR apply_error ILIKE '%%limit%%')
             ORDER BY updated_at ASC, url LIMIT %s
            """,
            (limit,),
        )
        urls = [row["url"] for row in cur.fetchall()]
        quarantined = 0
        if execute and urls:
            cur.execute(
                """
                UPDATE apply_queue
                   SET status='blocked', apply_status='manual_review_required',
                       apply_error='submission_uncertain:budget_or_limit',
                       lease_owner=NULL, lease_expires_at=NULL, updated_at=now()
                 WHERE url=ANY(%s) AND status='failed'
                   AND (apply_error ILIKE '%%budget%%' OR apply_error ILIKE '%%limit%%')
                """,
                (urls,),
            )
            quarantined = cur.rowcount
            cur.execute(
                """
                INSERT INTO fleet_console_audit
                    (action, actor, lane, target, message, ok)
                SELECT 'quarantine_budget_limit_failure','apply-home','ats',url,
                       'protected budget or limit failure from retry; outcome requires review',TRUE
                  FROM apply_queue WHERE url=ANY(%s) AND status='blocked'
                """,
                (urls,),
            )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(urls), "quarantined": quarantined}


def normalize_clear_terminal_failures(
    conn, *, limit: int = 1000, execute: bool = False,
) -> dict[str, int | bool]:
    """Normalize exact terminal outcomes that cannot be retried safely."""
    if limit < 1:
        raise ValueError("limit must be positive")
    reasons = (
        "failed:already_applied", "failed:already_applied_recently",
        "failed:spam_blocked", "failed:spam_flagged", "failed:spam_filter",
        "failed:spam_filter_blocked_submission", "failed:missing_required_references",
        "failed:references_required_no_data",
    )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT url FROM apply_queue WHERE lane='ats' AND status='failed' "
            "AND apply_error=ANY(%s) ORDER BY updated_at ASC,url LIMIT %s",
            (list(reasons), limit),
        )
        urls = [row["url"] for row in cur.fetchall()]
        normalized = 0
        if execute and urls:
            cur.execute(
                """UPDATE apply_queue SET status='blocked', apply_status='terminal_non_retryable',
                       lease_owner=NULL, lease_expires_at=NULL, updated_at=now()
                 WHERE url=ANY(%s) AND status='failed' AND apply_error=ANY(%s)""",
                (urls, list(reasons)),
            )
            normalized = cur.rowcount
            cur.execute(
                """INSERT INTO fleet_console_audit (action,actor,lane,target,message,ok)
                SELECT 'normalize_clear_terminal_failure','apply-home','ats',url,
                       'normalized deterministic terminal outcome; reason retained',TRUE
                  FROM apply_queue WHERE url=ANY(%s) AND status='blocked'""",
                (urls,),
            )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(urls), "normalized": normalized}


def normalize_dedup_terminal_failures(
    conn, *, limit: int = 2000, execute: bool = False,
) -> dict[str, int | bool]:
    """Move dedup-protected already-applied rows out of the failed pool."""
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.url
              FROM apply_queue q
              JOIN applied_set s ON s.dedup_key=q.dedup_key
             WHERE q.lane='ats' AND q.status='failed'
               AND q.apply_error='dedup:already_applied'
               AND q.apply_status='skipped'
             ORDER BY q.updated_at ASC, q.url LIMIT %s
            """,
            (limit,),
        )
        urls = [row["url"] for row in cur.fetchall()]
        normalized = 0
        if execute and urls:
            cur.execute(
                """
                UPDATE apply_queue
                   SET status='blocked', apply_status='terminal_non_retryable',
                       lease_owner=NULL, lease_expires_at=NULL, updated_at=now()
                 WHERE url=ANY(%s) AND status='failed'
                   AND apply_error='dedup:already_applied' AND apply_status='skipped'
                """,
                (urls,),
            )
            normalized = cur.rowcount
            cur.execute(
                """
                INSERT INTO fleet_console_audit
                    (action, actor, lane, target, message, ok)
                SELECT 'normalize_dedup_terminal_failure','apply-home','ats',url,
                       'dedup-protected already-applied row moved out of failed pool; guard retained',TRUE
                  FROM apply_queue WHERE url=ANY(%s) AND status='blocked'
                """,
                (urls,),
            )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(urls), "normalized": normalized}


def normalize_final_protected_failures(
    conn, *, limit: int = 1000, execute: bool = False,
) -> dict[str, int | bool]:
    """Make final crash/auth/tooling outcomes explicit protected review rows."""
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.url
              FROM apply_queue q
             WHERE q.lane='ats' AND q.status='failed'
               AND (q.apply_error='crash_unconfirmed'
                    OR q.apply_error ILIKE '%%email_verification%%'
                    OR q.apply_error ILIKE '%%gmail_mcp%%'
                    OR q.apply_error IN ('failed:tooling_unavailable','failed:browser_tools_unavailable',
                                         'failed:blocked_by_cloudflare_challenge',
                                         'failed:unsafe_verification','failed:too_many_attempts_try_again_later'))
               AND NOT EXISTS (SELECT 1 FROM auth_challenge a
                                WHERE a.url=q.url AND a.resolved_at IS NULL)
             ORDER BY q.updated_at ASC, q.url LIMIT %s
            """,
            (limit,),
        )
        urls = [row["url"] for row in cur.fetchall()]
        normalized = 0
        if execute and urls:
            cur.execute(
                """
                UPDATE apply_queue
                   SET status='blocked', apply_status='manual_review_required',
                       apply_error='submission_uncertain:final_protected_outcome',
                       lease_owner=NULL, lease_expires_at=NULL, updated_at=now()
                 WHERE url=ANY(%s) AND status='failed'
                """,
                (urls,),
            )
            normalized = cur.rowcount
            cur.execute(
                """
                INSERT INTO fleet_console_audit
                    (action, actor, lane, target, message, ok)
                SELECT 'normalize_final_protected_failure','apply-home','ats',url,
                       'protected final crash/auth/tooling outcome; no open challenge remains',TRUE
                  FROM apply_queue WHERE url=ANY(%s) AND status='blocked'
                """,
                (urls,),
            )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(urls), "normalized": normalized}


def quarantine_residual_failures(
    conn, *, limit: int = 1000, execute: bool = False,
) -> dict[str, int | bool]:
    """Park the residual failed tail for explicit operator review."""
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """SELECT url, apply_error FROM apply_queue
                WHERE lane='ats' AND status='failed'
                ORDER BY updated_at ASC,url LIMIT %s""",
            (limit,),
        )
        rows = cur.fetchall()
        urls = [row["url"] for row in rows]
        quarantined = 0
        if execute and urls:
            cur.execute(
                """UPDATE apply_queue
                    SET status='blocked', apply_status='manual_review_required',
                        apply_error='manual_review:residual_failure',
                        lease_owner=NULL, lease_expires_at=NULL, updated_at=now()
                  WHERE url=ANY(%s) AND status='failed'""",
                (urls,),
            )
            quarantined = cur.rowcount
            cur.executemany(
                """INSERT INTO fleet_console_audit
                    (action,actor,lane,target,message,ok)
                   VALUES ('quarantine_residual_failure','apply-home','ats',%s,%s,TRUE)""",
                [
                    (
                        row["url"],
                        "parked residual failure for explicit operator review; original_reason="
                        + str(row["apply_error"] or "(none)"),
                    )
                    for row in rows
                ],
            )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(urls), "quarantined": quarantined}


def retry_infrastructure_pending(
    conn, *, limit: int = 100, cooldown_hours: int = 1,
    max_failures: int = 1, execute: bool = False,
) -> dict[str, int | bool]:
    """Requeue only untouched, low-count browser-preflight parks.

    A preflight park is retryable because the worker recorded zero browser calls;
    this command never selects rows that reached the form.  The cooldown and
    failure-count bounds prevent a broken host from creating an unbounded retry
    loop, while the applied_set guard preserves the double-submit invariant.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    if cooldown_hours < 0:
        raise ValueError("cooldown_hours must be non-negative")
    if max_failures < 1:
        raise ValueError("max_failures must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.url
              FROM apply_queue q
              JOIN LATERAL (
                  SELECT e.application_tool_calls
                    FROM apply_result_events e
                   WHERE e.queue_name='apply_queue' AND e.url=q.url
                   ORDER BY e.created_at DESC, e.id DESC
                   LIMIT 1
              ) ev ON TRUE
             WHERE q.lane='ats' AND q.status='failed'
               AND q.apply_status='infrastructure_pending'
               AND COALESCE(q.infrastructure_failure_count,0) <= %s
               AND q.infrastructure_last_failure_at <= now() - make_interval(hours => %s)
               AND ev.application_tool_calls=0
               AND NOT EXISTS (SELECT 1 FROM applied_set s WHERE s.dedup_key=q.dedup_key)
               AND NOT EXISTS (
                   SELECT 1 FROM fleet_console_audit a
                    WHERE a.action='retry_infrastructure_pending' AND a.target=q.url
                      AND a.created_at > now() - make_interval(hours => %s)
               )
             ORDER BY q.infrastructure_last_failure_at ASC, q.url
             LIMIT %s
            """,
            (max_failures, cooldown_hours, cooldown_hours, limit),
        )
        urls = [row["url"] for row in cur.fetchall()]
        requeued = 0
        if execute and urls:
            cur.execute(
                """
                UPDATE apply_queue
                   SET status='queued', apply_status='queued',
                       apply_error='requeued_by_repair:infrastructure_pending',
                       attempts=0, lease_owner=NULL, lease_expires_at=NULL, updated_at=now()
                 WHERE url=ANY(%s) AND status='failed' AND apply_status='infrastructure_pending'
                """,
                (urls,),
            )
            requeued = cur.rowcount
            cur.execute(
                """
                INSERT INTO fleet_console_audit
                    (action, actor, lane, target, message, ok)
                SELECT 'retry_infrastructure_pending','apply-home','ats',url,
                       'requeued untouched browser-preflight park after cooldown; dedup guard passed',TRUE
                  FROM apply_queue WHERE url=ANY(%s) AND status='queued'
                """,
                (urls,),
            )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(urls), "requeued": requeued}


def dedup_review(conn, *, limit: int = 100) -> list[dict]:
    """List applied-set entries with sparse metadata and their queue evidence."""
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.dedup_key, a.company, a.normalized_role, a.first_applied_at,
                   a.applied_url, a.got_response,
                   COUNT(q.url) AS queue_rows,
                   MAX(q.updated_at) AS last_queue_at,
                   (array_agg(q.apply_error ORDER BY q.updated_at DESC NULLS LAST))[1]
                     AS last_queue_error
              FROM applied_set a
              LEFT JOIN apply_queue q ON q.dedup_key=a.dedup_key
             WHERE NULLIF(a.company,'') IS NULL
               AND NULLIF(a.normalized_role,'') IS NULL
               AND NULLIF(a.applied_url,'') IS NULL
             GROUP BY a.dedup_key, a.company, a.normalized_role,
                      a.first_applied_at, a.applied_url, a.got_response
             ORDER BY a.first_applied_at ASC, a.dedup_key
             LIMIT %s
            """,
            (limit,),
        )
        rows = [dict(row) for row in cur.fetchall()]
    conn.rollback()
    return rows


def canary_readiness(conn, *, limit: int = 25) -> dict:
    """Return a read-only per-row readiness report for approved ATS canary work."""
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.url, q.company, q.title, q.score, q.approved_batch,
                   q.apply_error, q.updated_at, q.attempts,
                   EXISTS (SELECT 1 FROM applied_set a WHERE a.dedup_key=q.dedup_key) AS dedup_blocked,
                   EXISTS (SELECT 1 FROM auth_challenge ch
                           WHERE ch.url=q.url AND ch.resolved_at IS NULL) AS open_challenge,
                   ev.status AS last_event_status, ev.apply_error AS last_event_error,
                   ev.application_tool_calls, ev.source AS last_event_source
              FROM apply_queue q
              LEFT JOIN LATERAL (
                  SELECT e.status, e.apply_error, e.application_tool_calls, e.source
                    FROM apply_result_events e
                   WHERE e.queue_name='apply_queue' AND e.url=q.url
                   ORDER BY e.created_at DESC, e.id DESC
                   LIMIT 1
              ) ev ON TRUE
             WHERE q.status='queued' AND q.lane='ats' AND q.approved_batch IS NOT NULL
             ORDER BY q.score DESC, q.url
             LIMIT %s
            """,
            (limit,),
        )
        rows = []
        for row in cur.fetchall():
            blockers = []
            warnings = []
            if row["dedup_blocked"]:
                blockers.append("dedup_blocked")
            if row["open_challenge"]:
                blockers.append("open_challenge")
            if row["last_event_status"] == "crash_unconfirmed":
                blockers.append("prior_crash_unconfirmed")
            if any(
                value and "requeued_by_" in value
                for value in (row["apply_error"], row["last_event_error"])
            ):
                blockers.append("prior_requeue")
            if row["application_tool_calls"]:
                blockers.append("prior_browser_tool_calls")
            if int(row["attempts"] or 0) > 0:
                blockers.append("prior_attempts")
            rows.append({
                "url": row["url"],
                "company": row["company"],
                "title": row["title"],
                "score": row["score"],
                "blockers": blockers,
                "warnings": warnings,
                "last_event_source": row["last_event_source"],
                "last_event_status": row["last_event_status"],
                "last_event_error": row["last_event_error"],
                "application_tool_calls": row["application_tool_calls"],
                "attempts": row["attempts"],
                "ready": not blockers,
            })
        cur.execute(
            "SELECT paused, ats_paused, ats_pause_source, ats_apply_mode, canary_enabled, "
            "canary_remaining FROM fleet_config WHERE id=1"
        )
        cfg = dict(cur.fetchone())
    conn.rollback()
    if not rows:
        next_action = "stage approved ATS jobs, then rerun canary-readiness"
    elif not any(row["ready"] for row in rows):
        next_action = "repair or quarantine the listed canary blockers, then rerun canary-readiness"
    else:
        next_action = "review ready rows, then arm a bounded canary"
    return {
        "gate_ready": (
            not cfg["paused"] and not cfg["ats_paused"]
            and cfg["ats_apply_mode"] in ("canary", "steady")
            and bool(cfg["canary_enabled"] or cfg["ats_apply_mode"] == "steady")
            and int(cfg["canary_remaining"] or 0) > 0
        ),
        "pause_source": cfg["ats_pause_source"],
        "candidate_count": len(rows),
        "ready_count": sum(1 for row in rows if row["ready"]),
        "next_action": next_action,
        "rows": rows,
    }


def readiness(conn) -> dict:
    """Return a concise, read-only operator resume report."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT paused, ats_paused, ats_pause_source, ats_apply_mode, "
            "canary_enabled, canary_remaining FROM fleet_config WHERE id=1"
        )
        cfg = dict(cur.fetchone())
        cur.execute(
            "SELECT status, COUNT(*) AS n FROM apply_queue GROUP BY status"
        )
        queue_counts = {str(row["status"]): int(row["n"] or 0) for row in cur.fetchall()}
        cur.execute(
            "SELECT COUNT(*) AS n FROM apply_queue "
            "WHERE lane='ats' AND status='crash_unconfirmed'"
        )
        crash_unconfirmed = int(cur.fetchone()["n"] or 0)
        cur.execute(
            "SELECT COUNT(*) AS n FROM auth_challenge a "
            "WHERE a.resolved_at IS NULL AND ("
            "EXISTS (SELECT 1 FROM apply_queue q WHERE q.url=a.url AND q.lane='ats' "
            "AND q.status='leased' AND q.apply_status='challenge_pending') OR "
            "(NOT EXISTS (SELECT 1 FROM apply_queue q WHERE q.url=a.url) AND "
            "NOT EXISTS (SELECT 1 FROM linkedin_queue l WHERE l.url=a.url)))"
        )
        open_challenges = int(cur.fetchone()["n"] or 0)
        cur.execute(
            "SELECT COUNT(*) FILTER (WHERE last_beat >= now() - interval '2 minutes') AS recent, "
            "COUNT(*) FILTER (WHERE last_beat < now() - interval '2 minutes') AS stale "
            "FROM worker_heartbeat WHERE role='apply'"
        )
        heartbeat = dict(cur.fetchone())
        cur.execute("SELECT to_regclass('fleet_desired_state') AS rel")
        desired_available = cur.fetchone()["rel"] is not None
        desired_row = None
        if desired_available:
            cur.execute(
                "SELECT desired_workers FROM fleet_desired_state "
                "WHERE machine_owner='home'"
            )
            desired_row = cur.fetchone()
        desired_workers = int(desired_row["desired_workers"] or 0) if desired_row else 0
        cur.execute(
            "SELECT state, last_error, last_beat FROM worker_heartbeat "
            "WHERE worker_id='otp_responder'"
        )
        otp_row = cur.fetchone()
        cur.execute(
            "SELECT COUNT(*) FILTER (WHERE consumed_at IS NULL AND code IS NULL "
            "AND expires_at > now()) AS pending, COUNT(*) FILTER "
            "(WHERE code IS NOT NULL AND consumed_at IS NULL) AS ready FROM otp_request"
        )
        otp_counts = cur.fetchone()
    conn.rollback()

    now = _dt.datetime.now(_dt.timezone.utc)
    otp_healthy = bool(
        otp_row and otp_row["last_beat"]
        and otp_row["last_beat"] >= now - _dt.timedelta(seconds=45)
        and not otp_row["last_error"]
    )
    liveness = crash_liveness_task_diagnosis()
    blockers = ["emergency_acquisition_hold"]
    if cfg["paused"] or cfg["ats_paused"]:
        blockers.append("operator_pause")
    if desired_workers <= 0:
        blockers.append("desired_workers_zero")
    if crash_unconfirmed:
        blockers.append("crash_unconfirmed_present")
    if not liveness["healthy"]:
        blockers.append("crash_liveness_unhealthy")
    if open_challenges:
        blockers.append("open_auth_challenges")
    if not otp_healthy:
        blockers.append("otp_unhealthy")
    if int(queue_counts.get("queued", 0)):
        blockers.append("queued_work_present")
    heartbeat_blocks = bool(
        not cfg["paused"]
        and not cfg["ats_paused"]
        and desired_workers > 0
        and int(heartbeat.get("stale", 0) or 0)
    )
    if heartbeat_blocks:
        blockers.append("stale_apply_heartbeats")
    if cfg["ats_apply_mode"] not in ("canary", "steady"):
        blockers.append("invalid_apply_mode")
    if cfg["ats_apply_mode"] == "canary" and not cfg["canary_enabled"]:
        blockers.append("canary_disabled")
    if int(cfg["canary_remaining"] or 0) <= 0:
        blockers.append("canary_budget_empty")
    if blockers:
        next_action = "keep fleet paused and resolve the listed blockers"
    else:
        next_action = "run canary-readiness before arming workers"
    return {
        "ready": not blockers,
        "blockers": blockers,
        "next_action": next_action,
        "pause_source": cfg["ats_pause_source"],
        "queue": queue_counts,
        "crash_unconfirmed": crash_unconfirmed,
        "open_auth_challenges": open_challenges,
        "desired_control_available": desired_available,
        "desired_workers": desired_workers,
        "apply_heartbeats": heartbeat,
        "heartbeat_blocks": heartbeat_blocks,
        "crash_liveness_task": liveness,
        "otp": {
            "healthy": otp_healthy,
            "state": otp_row["state"] if otp_row else None,
            "pending": int(otp_counts["pending"] or 0),
            "ready": int(otp_counts["ready"] or 0),
        },
        "apply_mode": cfg["ats_apply_mode"],
        "canary_enabled": bool(cfg["canary_enabled"]),
        "canary_remaining": int(cfg["canary_remaining"] or 0),
    }


def review_crash_liveness(
    conn, *, older_days: int = 7, limit: int = 25,
    priority: bool = True, execute: bool = False, refresh_days: int = 7,
) -> dict:
    """Probe old crash postings and optionally stamp liveness metadata only."""
    if older_days < 0:
        raise ValueError("older_days must be non-negative")
    if limit < 1:
        raise ValueError("limit must be positive")
    if refresh_days < 1:
        raise ValueError("refresh_days must be at least 1")
    from applypilot.apply.liveness import probe_url

    order_sql = "q.score DESC, q.updated_at ASC, q.url" if priority else "q.updated_at ASC, q.url"
    with conn.cursor() as cur:
        cur.execute(
            "SELECT url, application_url, company, title, score FROM apply_queue q "
            "WHERE status='crash_unconfirmed' AND lane='ats' "
            "AND updated_at < now() - make_interval(days => %s) "
            "AND (liveness_checked_at IS NULL OR liveness_checked_at < now() - make_interval(days => %s)) "
            "ORDER BY " + order_sql + " LIMIT %s",
            (older_days, refresh_days, limit),
        )
        results = []
        for row in cur.fetchall():
            status, reason = probe_url(row["application_url"] or row["url"])
            results.append({
                "url": row["url"], "company": row["company"], "title": row["title"],
                "score": row["score"], "liveness_status": status,
                "liveness_reason": reason,
            })
            if execute:
                cur.execute(
                    "UPDATE apply_queue SET liveness_status=%s, liveness_reason=%s, "
                    "liveness_checked_at=now(), updated_at=updated_at "
                    "WHERE url=%s AND status='crash_unconfirmed'",
                    (status, reason[:200], row["url"]),
                )
        if execute and results:
            cur.execute(
                "INSERT INTO fleet_console_audit "
                "(action, actor, lane, target, message, ok) VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    "review_crash_liveness", "apply-home", "ats", f"{len(results)} rows",
                    f"stamped liveness results for crash rows older than {older_days} days", True,
                ),
            )
    (conn.commit() if execute else conn.rollback())
    return {
        "dry_run": not execute,
        "checked": len(results),
        **{status: sum(1 for row in results if row["liveness_status"] == status)
           for status in ("live", "dead", "uncertain")},
        "rows": results,
    }


def resolve_crash(
    conn, url: str, *, outcome: str, reason: str, actor: str = "apply-home"
) -> dict:
    """Close one crash row only after an explicit operator outcome.

    Liveness and autotriage evidence are advisory; neither is sufficient to infer
    whether a submission happened. This command is the deliberate human decision
    point and refuses to touch a row that has already changed state.
    """
    allowed = {"applied", "failed", "blocked"}
    if outcome not in allowed:
        raise ValueError(f"outcome must be one of: {', '.join(sorted(allowed))}")
    reason = str(reason or "").strip()
    if not reason:
        raise ValueError("reason is required")
    if len(reason) > 500:
        raise ValueError("reason must be 500 characters or fewer")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT url, status, apply_status, apply_error, attempts, worker_id "
            "FROM apply_queue WHERE url=%s AND lane='ats' FOR UPDATE",
            (url,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"no ATS queue row found for {url}")
        if row["status"] != "crash_unconfirmed":
            raise ValueError(
                f"refusing to resolve {url}: current status is {row['status']!r}, "
                "not crash_unconfirmed"
            )

        apply_status = "applied" if outcome == "applied" else f"{outcome}:{reason}"
        apply_error = None if outcome == "applied" else reason
        cur.execute(
            "UPDATE apply_queue SET status=%s, apply_status=%s, apply_error=%s, "
            "lease_owner=NULL, lease_expires_at=NULL, applied_at=%s, "
            "synced_to_home_at=NULL, updated_at=now() WHERE url=%s "
            "AND status='crash_unconfirmed'",
            (
                outcome,
                apply_status,
                apply_error,
                _dt.datetime.now(_dt.timezone.utc) if outcome == "applied" else None,
                url,
            ),
        )
        if cur.rowcount != 1:
            raise ValueError(f"refusing to resolve {url}: row changed during update")
        cur.execute(
            "INSERT INTO apply_result_events "
            "(queue_name, url, worker_id, status, apply_status, apply_error, "
            "application_tool_calls, final_result_source, result_metadata, result_line, source) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                "apply_queue", url, actor, outcome, apply_status, apply_error,
                None, "operator_crash_resolution",
                pgqueue.Jsonb({"resolution": "manual_crash_resolution", "reason": reason}),
                f"manual_resolution:{outcome}:{reason}", "operator",
            ),
        )
        cur.execute(
            "INSERT INTO fleet_console_audit "
            "(action, actor, lane, target, message, ok) VALUES (%s,%s,%s,%s,%s,%s)",
            (
                "resolve_crash", actor, "ats", url,
                f"manual crash resolution: {outcome}; reason={reason}", True,
            ),
        )
    conn.commit()
    return {
        "url": url,
        "prior_status": "crash_unconfirmed",
        "status": outcome,
        "apply_status": apply_status,
        "reason": reason,
        "audit": True,
    }


def park_dead_crashes(
    conn, *, older_days: int = 7, refresh_days: int = 7,
    limit: int = 25, execute: bool = False, actor: str = "apply-home",
    include_applied_set: bool = False,
) -> dict:
    """Move only evidence-free, recently dead crash rows to explicit uncertainty.

    A dead posting is not proof that an application failed. This action removes such
    rows from the crash backlog only by recording ``blocked:submission_uncertain``;
    it never claims success and never makes the row leaseable.
    """
    if older_days < 0:
        raise ValueError("older_days must be non-negative")
    if refresh_days < 1:
        raise ValueError("refresh_days must be at least 1")
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS dead_rows, "
            "COUNT(*) FILTER (WHERE q.updated_at < now() - make_interval(days => %s) "
            "  AND q.liveness_checked_at >= now() - make_interval(days => %s)) AS scoped_dead, "
            "COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM apply_result_events e "
            "  WHERE e.queue_name='apply_queue' AND e.url=q.url "
            "    AND COALESCE(e.application_tool_calls,0)>0)) AS tool_touched, "
            "COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM apply_result_events e "
            "  WHERE e.queue_name='apply_queue' AND e.url=q.url "
            "    AND (e.status='applied' OR e.apply_status='applied'))) AS applied_event, "
            "COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM applied_set a "
            "  WHERE a.dedup_key=q.dedup_key)) AS applied_set, "
            "COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM inbox_outcomes io "
            "  WHERE io.dedup_key=q.dedup_key OR io.job_url IN (q.url,q.application_url))) "
            "AS inbox_outcome "
            "FROM apply_queue q WHERE q.lane='ats' AND q.status='crash_unconfirmed' "
            "AND q.liveness_status='dead'",
            (older_days, refresh_days),
        )
        guard_counts = dict(cur.fetchone())
        cur.execute(
            (
            "SELECT q.url, q.application_url, q.dedup_key, q.company, q.title, "
            "q.liveness_reason, q.liveness_checked_at "
            "FROM apply_queue q "
            "WHERE q.lane='ats' AND q.status='crash_unconfirmed' "
            "AND q.updated_at < now() - make_interval(days => %s) "
            "AND q.liveness_status='dead' "
            "AND q.liveness_checked_at >= now() - make_interval(days => %s) "
            "AND NOT EXISTS (SELECT 1 FROM apply_result_events e "
            "  WHERE e.queue_name='apply_queue' AND e.url=q.url "
            "    AND (COALESCE(e.application_tool_calls,0)>0 "
            "      OR e.status='applied' OR e.apply_status='applied')) "
            + ("" if include_applied_set else
               "AND NOT EXISTS (SELECT 1 FROM applied_set a WHERE a.dedup_key=q.dedup_key) ")
            + "AND NOT EXISTS (SELECT 1 FROM inbox_outcomes io "
            "  WHERE io.dedup_key=q.dedup_key "
            "     OR io.job_url IN (q.url, q.application_url)) "
            "ORDER BY q.updated_at ASC, q.url LIMIT %s"
            ),
            (older_days, refresh_days, limit),
        )
        rows = [dict(row) for row in cur.fetchall()]
        parked = 0
        if execute:
            for row in rows:
                reason = "submission_uncertain:dead_posting"
                cur.execute(
                    "UPDATE apply_queue SET status='blocked', "
                    "apply_status='blocked:submission_uncertain', apply_error=%s, "
                    "lease_owner=NULL, lease_expires_at=NULL, synced_to_home_at=NULL, "
                    "updated_at=now() WHERE url=%s AND status='crash_unconfirmed'",
                    (reason, row["url"]),
                )
                if cur.rowcount != 1:
                    continue
                cur.execute(
                    "INSERT INTO apply_result_events "
                    "(queue_name, url, worker_id, status, apply_status, apply_error, "
                    "application_tool_calls, final_result_source, result_metadata, "
                    "result_line, source) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        "apply_queue", row["url"], actor, "blocked",
                        "blocked:submission_uncertain", reason, None,
                        "operator_dead_posting_park",
                        pgqueue.Jsonb({
                            "resolution": "park_dead_crash",
                            "liveness_status": "dead",
                            "liveness_reason": row["liveness_reason"],
                            "liveness_checked_at": row["liveness_checked_at"].isoformat()
                            if row["liveness_checked_at"] else None,
                            "outcome_confirmed": False,
                        }),
                        f"operator_park_dead:{reason}", "operator",
                    ),
                )
                cur.execute(
                    "INSERT INTO fleet_console_audit "
                    "(action, actor, lane, target, message, ok) VALUES (%s,%s,%s,%s,%s,%s)",
                    (
                        "park_dead_crash", actor, "ats", row["url"],
                        "parked dead crash as submission_uncertain; outcome remains unresolved",
                        True,
                    ),
                )
                parked += 1
    (conn.commit() if execute else conn.rollback())
    return {
        "dry_run": not execute,
        "candidates": len(rows),
        "parked": parked,
        "include_applied_set": include_applied_set,
        "guard_counts": guard_counts,
        "rows": rows,
    }


def park_protected_crashes(
    conn, *, older_days: int = 7, limit: int = 100,
    execute: bool = False, actor: str = "apply-home",
    include_tool_touched: bool = False, require_applied_set: bool = True,
    include_inbox: bool = False, require_inbox: bool = False,
) -> dict:
    """Reclassify old evidence-gap crashes as unresolved uncertainty."""
    if older_days < 0:
        raise ValueError("older_days must be non-negative")
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS protected_rows, "
            "COUNT(*) FILTER (WHERE q.attempts >= 99) AS attempts_99, "
            "COUNT(*) FILTER (WHERE q.liveness_status='live') AS live_rows, "
            "COUNT(*) FILTER (WHERE q.liveness_status='uncertain') AS uncertain_rows, "
            "COUNT(*) FILTER (WHERE q.liveness_status IS NULL) AS unchecked_rows, "
            "COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM inbox_outcomes io "
            "  WHERE io.dedup_key=q.dedup_key OR io.job_url IN (q.url,q.application_url))) "
            "AS inbox_outcome, "
            "COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM apply_result_events e "
            "  WHERE e.queue_name='apply_queue' AND e.url=q.url "
            "    AND COALESCE(e.application_tool_calls,0)>0)) AS tool_touched "
            "FROM apply_queue q WHERE q.lane='ats' AND q.status='crash_unconfirmed' "
            "AND q.updated_at < now() - make_interval(days => %s) "
            + ("AND EXISTS (SELECT 1 FROM applied_set a WHERE a.dedup_key=q.dedup_key)"
               if require_applied_set else ""),
            (older_days,),
        )
        guard_counts = dict(cur.fetchone())
        cur.execute(
            (
            "SELECT q.url, q.application_url, q.dedup_key, q.company, q.title, "
            "q.liveness_status, q.liveness_reason, q.liveness_checked_at, q.attempts "
            "FROM apply_queue q WHERE q.lane='ats' AND q.status='crash_unconfirmed' "
            "AND q.updated_at < now() - make_interval(days => %s) "
            + ("AND EXISTS (SELECT 1 FROM applied_set a WHERE a.dedup_key=q.dedup_key) "
               if require_applied_set else "")
            + ("" if include_tool_touched else
               "AND NOT EXISTS (SELECT 1 FROM apply_result_events e "
               "  WHERE e.queue_name='apply_queue' AND e.url=q.url "
               "    AND COALESCE(e.application_tool_calls,0)>0) ")
            + "AND NOT EXISTS (SELECT 1 FROM apply_result_events e "
            "  WHERE e.queue_name='apply_queue' AND e.url=q.url "
            "    AND (e.status='applied' OR e.apply_status='applied')) "
            + ("" if include_inbox else
               "AND NOT EXISTS (SELECT 1 FROM inbox_outcomes io "
               "  WHERE io.dedup_key=q.dedup_key OR io.job_url IN (q.url,q.application_url)) ")
            + ("AND EXISTS (SELECT 1 FROM inbox_outcomes io "
               "  WHERE io.dedup_key=q.dedup_key OR io.job_url IN (q.url,q.application_url)) "
               if require_inbox else "")
            + "ORDER BY q.updated_at ASC, q.url LIMIT %s"
            ),
            (older_days, limit),
        )
        rows = [dict(row) for row in cur.fetchall()]
        parked = 0
        if execute:
            for row in rows:
                reason = (
                    "submission_uncertain:email_match_review"
                    if require_inbox else
                    "submission_uncertain:tool_touched"
                    if include_tool_touched else
                    "submission_uncertain:applied_set_protected"
                    if require_applied_set else
                    "submission_uncertain:evidence_gap"
                )
                cur.execute(
                    "UPDATE apply_queue SET status='blocked', "
                    "apply_status='blocked:submission_uncertain', apply_error=%s, "
                    "lease_owner=NULL, lease_expires_at=NULL, synced_to_home_at=NULL, "
                    "updated_at=now() WHERE url=%s AND status='crash_unconfirmed'",
                    (reason, row["url"]),
                )
                if cur.rowcount != 1:
                    continue
                cur.execute(
                    "INSERT INTO apply_result_events "
                    "(queue_name, url, worker_id, status, apply_status, apply_error, "
                    "application_tool_calls, final_result_source, result_metadata, "
                    "result_line, source) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        "apply_queue", row["url"], actor, "blocked",
                        "blocked:submission_uncertain", reason, None,
                        "operator_protected_crash_park",
                        pgqueue.Jsonb({
                            "resolution": "park_protected_crash",
                            "dedup_guard_present": True,
                            "liveness_status": row["liveness_status"],
                            "liveness_reason": row["liveness_reason"],
                            "outcome_confirmed": False,
                        }),
                        f"operator_park_protected:{reason}", "operator",
                    ),
                )
                cur.execute(
                    "INSERT INTO fleet_console_audit "
                    "(action, actor, lane, target, message, ok) VALUES (%s,%s,%s,%s,%s,%s)",
                    (
                        "park_protected_crash", actor, "ats", row["url"],
                        "parked evidence-gap crash as submission_uncertain; outcome unresolved",
                        True,
                    ),
                )
                parked += 1
    (conn.commit() if execute else conn.rollback())
    return {
        "dry_run": not execute,
        "candidates": len(rows),
        "parked": parked,
        "include_tool_touched": include_tool_touched,
        "require_applied_set": require_applied_set,
        "include_inbox": include_inbox,
        "require_inbox": require_inbox,
        "guard_counts": guard_counts,
        "rows": rows,
    }


def arm_canary_if_safe(conn, k: int) -> bool:
    """Guarded ApplyCycle arm: re-open canary leasing only when safety flags allow it."""
    if queue._cost_cap_exceeded(conn):
        return False
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE fleet_config "
            "SET ats_apply_mode=CASE WHEN paused OR ats_paused THEN ats_apply_mode ELSE 'canary' END, "
            "canary_enabled=CASE WHEN paused OR ats_paused THEN FALSE ELSE TRUE END, "
            "canary_remaining=CASE WHEN paused OR ats_paused THEN 0 ELSE %s END, "
            "updated_at=now() "
            "WHERE id=1 RETURNING paused, ats_paused",
            (k,),
        )
        state = cur.fetchone()
    conn.commit()
    return bool(state and not state["paused"] and not state["ats_paused"])


def _stale_codex_canary_pause_clearable(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT ats_paused, COALESCE(ats_pause_source, '') AS ats_pause_source "
                    "FROM fleet_config WHERE id=1")
        row = cur.fetchone()
        if not row or not row["ats_paused"] or not row["ats_pause_source"].startswith("codex_canary_"):
            return False
        cur.execute("SELECT EXISTS (SELECT 1 FROM fleet_knobs WHERE active) AS has_active_knob")
        if cur.fetchone()["has_active_knob"]:
            return False
        cur.execute("SELECT EXISTS (SELECT 1 FROM rate_governor WHERE breaker_state='paused') "
                    "AS has_paused_breaker")
        if cur.fetchone()["has_paused_breaker"]:
            return False
    return True


def resume_if_safe(conn) -> bool:
    """Keep every pause intact while the emergency acquisition hold is active."""
    del conn
    return False


def resolve_challenge_cmd(conn, url: str, *, skip: bool) -> bool:
    return queue.resolve_challenge(conn, url, requeue=not skip)


def cleanup_terminal_challenges(conn, *, older_days: int = 1, execute: bool = False) -> dict[str, int | bool]:
    """Close ATS challenge records whose queue row is already terminal.

    This is inbox housekeeping only: it never changes an apply_queue row and never
    touches LinkedIn challenges or a live challenge_pending lease.
    """
    if older_days < 0:
        raise ValueError("older_days must be non-negative")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS n
            FROM auth_challenge a
            JOIN apply_queue q ON q.url = a.url AND q.lane = 'ats'
            WHERE a.resolved_at IS NULL
              AND a.raised_at < now() - make_interval(days => %s)
              AND q.status IN ('applied', 'failed', 'blocked', 'crash_unconfirmed')
              AND NOT EXISTS (
                SELECT 1 FROM linkedin_queue l
                WHERE l.url = a.url AND l.status IN ('queued', 'leased')
              )
            """,
            (older_days,),
        )
        candidates = int(cur.fetchone()["n"] or 0)
        closed = 0
        if execute and candidates:
            cur.execute(
                """
                UPDATE auth_challenge a
                SET resolved_at = now(), outcome = 'stale_terminal_queue'
                FROM apply_queue q
                WHERE q.url = a.url AND q.lane = 'ats'
                  AND a.resolved_at IS NULL
                  AND a.raised_at < now() - make_interval(days => %s)
                  AND q.status IN ('applied', 'failed', 'blocked', 'crash_unconfirmed')
                  AND NOT EXISTS (
                    SELECT 1 FROM linkedin_queue l
                    WHERE l.url = a.url AND l.status IN ('queued', 'leased')
                  )
                """,
                (older_days,),
            )
            closed = cur.rowcount
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": candidates, "closed": closed}


def retire_stale_parked_challenges(
    conn, *, older_days: int = 7, limit: int = 100, execute: bool = False,
) -> dict[str, int | bool | list[dict]]:
    """Retire old frozen ATS challenge holds without retrying their jobs."""
    if older_days < 0:
        raise ValueError("older_days must be non-negative")
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (q.url)
                   q.url, q.company, q.title, q.score, q.worker_id,
                   q.updated_at, q.lease_expires_at, a.kind, a.raised_at
              FROM apply_queue q
              JOIN auth_challenge a ON a.url=q.url AND a.resolved_at IS NULL
             WHERE q.lane='ats'
               AND q.status='leased'
               AND q.apply_status='challenge_pending'
               AND q.lease_expires_at > now() + interval '1 day'
               AND a.raised_at < now() - make_interval(days => %s)
             ORDER BY q.url, a.raised_at ASC
             LIMIT %s
            """,
            (older_days, limit),
        )
        rows = [dict(row) for row in cur.fetchall()]
        retired = 0
        if execute:
            for row in rows:
                if not queue.resolve_challenge(conn, row["url"], requeue=False, commit=False):
                    continue
                cur.execute(
                    "INSERT INTO fleet_console_audit "
                    "(action, actor, lane, target, message, ok) VALUES (%s,%s,%s,%s,%s,%s)",
                    (
                        "retire_stale_parked_challenge", "apply-home", "ats", row["url"],
                        "retired stale frozen challenge hold without retrying job", True,
                    ),
                )
                retired += 1
    (conn.commit() if execute else conn.rollback())
    return {"dry_run": not execute, "candidates": len(rows), "retired": retired, "rows": rows}


def repair_orphan_challenges(conn, *, older_days: int = 1, execute: bool = False) -> dict[str, int | bool]:
    """Create owner-inbox records for frozen ATS leases missing challenge metadata."""
    if older_days < 0:
        raise ValueError("older_days must be non-negative")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS n
            FROM apply_queue q
            WHERE q.lane = 'ats' AND q.status = 'leased'
              AND q.apply_status = 'challenge_pending'
              AND q.lease_expires_at > now() + interval '1 day'
              AND q.updated_at < now() - make_interval(days => %s)
              AND NOT EXISTS (
                SELECT 1 FROM auth_challenge a
                WHERE a.url = q.url AND a.resolved_at IS NULL
              )
            """,
            (older_days,),
        )
        candidates = int(cur.fetchone()["n"] or 0)
        created = 0
        if execute and candidates:
            cur.execute(
                """
                INSERT INTO auth_challenge (url, worker_id, kind, route, raised_at)
                SELECT q.url, q.lease_owner, 'manual_auth', 'owner_inbox', q.updated_at
                FROM apply_queue q
                WHERE q.lane = 'ats' AND q.status = 'leased'
                  AND q.apply_status = 'challenge_pending'
                  AND q.lease_expires_at > now() + interval '1 day'
                  AND q.updated_at < now() - make_interval(days => %s)
                  AND NOT EXISTS (
                    SELECT 1 FROM auth_challenge a
                    WHERE a.url = q.url AND a.resolved_at IS NULL
                  )
                """,
                (older_days,),
            )
            created = cur.rowcount
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": candidates, "created": created}


def normalize_resolved_challenge_blocks(
    conn, *, limit: int = 1000, execute: bool = False,
) -> dict[str, int | bool]:
    """Normalize terminal blocked rows whose resolved challenge was skipped.

    This only repairs stale labeling. It deliberately keeps the job blocked and never
    requeues a row, because a skipped challenge is not evidence that submission is safe.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.url
              FROM apply_queue q
              JOIN LATERAL (
                SELECT a.outcome
                  FROM auth_challenge a
                 WHERE a.url=q.url AND a.resolved_at IS NOT NULL
                 ORDER BY a.resolved_at DESC
                 LIMIT 1
              ) a ON TRUE
             WHERE q.lane='ats' AND q.status='blocked'
               AND q.apply_error='challenge_pending'
               AND a.outcome='skipped'
             ORDER BY q.updated_at ASC, q.url
             LIMIT %s
            """,
            (limit,),
        )
        urls = [row["url"] for row in cur.fetchall()]
        normalized = 0
        if execute and urls:
            cur.execute(
                """
                UPDATE apply_queue
                   SET apply_status='challenge_skipped', apply_error='challenge_skipped', updated_at=now()
                 WHERE lane='ats' AND status='blocked' AND apply_error='challenge_pending'
                   AND url = ANY(%s)
                """,
                (urls,),
            )
            normalized = cur.rowcount
            cur.execute(
                """
                INSERT INTO fleet_console_audit
                    (action, actor, lane, target, message, ok)
                SELECT 'normalize_resolved_challenge_block', 'apply-home', 'ats', url,
                       'normalized resolved skipped challenge label; row remains blocked', TRUE
                  FROM apply_queue
                 WHERE url=ANY(%s)
                """,
                (urls,),
            )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(urls), "normalized": normalized}


def backfill_dedup_metadata(
    conn, *, limit: int = 500, execute: bool = False,
) -> dict[str, int | bool]:
    """Fill only unambiguous missing applied-set metadata from protected queue rows."""
    if limit < 1:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH candidates AS (
                SELECT a.dedup_key,
                       CASE WHEN COUNT(DISTINCT NULLIF(q.company,'')) = 1
                            THEN MIN(NULLIF(q.company,'')) END AS company,
                       CASE WHEN COUNT(DISTINCT NULLIF(COALESCE(q.application_url,q.url),'')) = 1
                            THEN MIN(NULLIF(COALESCE(q.application_url,q.url),'')) END AS applied_url
                  FROM applied_set a
                  JOIN apply_queue protected
                    ON protected.dedup_key=a.dedup_key
                   AND protected.status='blocked'
                   AND protected.apply_error='submission_uncertain:applied_set_protected'
                  JOIN apply_queue q ON q.dedup_key=a.dedup_key
                 WHERE a.company IS NULL OR a.applied_url IS NULL
                 GROUP BY a.dedup_key
                HAVING COUNT(DISTINCT NULLIF(q.company,'')) = 1
                    OR COUNT(DISTINCT NULLIF(COALESCE(q.application_url,q.url),'')) = 1
                 ORDER BY a.dedup_key
                 LIMIT %s
            )
            SELECT dedup_key, company, applied_url FROM candidates
            """,
            (limit,),
        )
        rows = cur.fetchall()
        updated = 0
        if execute and rows:
            for row in rows:
                cur.execute(
                    """
                    UPDATE applied_set
                       SET company=COALESCE(company,%s), applied_url=COALESCE(applied_url,%s)
                     WHERE dedup_key=%s AND (company IS NULL OR applied_url IS NULL)
                    """,
                    (row["company"], row["applied_url"], row["dedup_key"]),
                )
                updated += cur.rowcount
                cur.execute(
                    """
                    INSERT INTO fleet_console_audit
                        (action, actor, lane, target, message, ok)
                    VALUES ('backfill_dedup_metadata','apply-home','ats',%s,
                            'filled only unambiguous applied-set metadata; dedup protection unchanged',TRUE)
                    """,
                    (row["dedup_key"],),
                )
    if execute:
        conn.commit()
    else:
        conn.rollback()
    return {"dry_run": not execute, "candidates": len(rows), "updated": updated}


def list_challenges(
    conn,
    *,
    stale_days: int | None = None,
    limit: int | None = None,
    kind: str | None = None,
    priority: bool = False,
) -> list:
    where: list[str] = []
    params: list[object] = []
    if stale_days is not None:
        if stale_days < 0:
            raise ValueError("stale_days must be non-negative")
        where.append("c.raised_at < now() - make_interval(days => %s)")
        params.append(stale_days)
    if kind:
        where.append("c.kind = %s")
        params.append(kind)
    limit_sql = ""
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        limit_sql = " LIMIT %s"
        params.append(limit)
    with conn.cursor() as cur:
        cur.execute(
            "WITH challenge_rows AS ("
            "SELECT a.id, a.url, a.worker_id, a.kind, a.route, a.raised_at, FALSE AS orphan, "
            "q.company, q.title, q.score, q.application_url "
            "FROM auth_challenge a LEFT JOIN apply_queue q ON q.url=a.url "
            "WHERE a.resolved_at IS NULL "
            "AND (q.lane='ats' OR (q.url IS NULL AND NOT EXISTS ("
            "SELECT 1 FROM linkedin_queue l WHERE l.url=a.url))) "
            "UNION ALL "
            "SELECT NULL::bigint, q.url, q.lease_owner, "
            "'parked_without_open_challenge', 'owner_inbox', q.updated_at, TRUE, "
            "q.company, q.title, q.score, q.application_url "
            "FROM apply_queue q "
            "WHERE q.lane='ats' AND q.status='leased' AND q.apply_status='challenge_pending' "
            "AND q.lease_expires_at > now() + interval '1 day' "
            "AND NOT EXISTS (SELECT 1 FROM auth_challenge a "
            "WHERE a.url=q.url AND a.resolved_at IS NULL)"
            ") "
            "SELECT c.id, c.url, c.worker_id, c.kind, c.route, c.raised_at, "
            "c.company, c.title, c.score, c.application_url, "
            "EXTRACT(EPOCH FROM (now() - c.raised_at)) / 3600.0 AS age_hours, "
            "EXISTS (SELECT 1 FROM apply_queue q WHERE q.url=c.url "
            "AND q.lane='ats' AND q.status='leased' AND q.apply_status='challenge_pending' "
            "AND q.lease_expires_at > now() + interval '1 day') AS parked, c.orphan "
            "FROM challenge_rows c "
            + ("WHERE " + " AND ".join(where) if where else "") +
            " ORDER BY " + (
                "c.score DESC NULLS LAST, c.raised_at ASC"
                if priority
                else "c.raised_at " + ("ASC" if stale_days is not None else "DESC")
            ) + limit_sql,
            params,
        )
        return [queue.challenge_metadata(r) for r in cur.fetchall()]


def list_crash_reviews(
    conn, *, limit: int = 25, evidence: str = "all", unreviewed: bool = False,
    older_days: int | None = None, priority: bool = False, liveness: str = "all",
) -> list:
    """Read-only oldest-first crash review queue with durable evidence and audit state."""
    if limit < 1:
        raise ValueError("limit must be positive")
    evidence_filters = {
        "all": None,
        "missing": "ev.application_tool_calls IS NULL",
        "zero_tool_calls": "ev.application_tool_calls = 0",
        "tool_touched": "ev.application_tool_calls > 0",
        "no_result_event": "ev.event_id IS NULL",
    }
    if evidence not in evidence_filters:
        raise ValueError(f"evidence must be one of: {', '.join(evidence_filters)}")
    if liveness not in {"all", "live", "dead", "uncertain", "unchecked"}:
        raise ValueError("liveness must be one of: all, live, dead, uncertain, unchecked")
    if older_days is not None and older_days < 0:
        raise ValueError("older_days must be non-negative")
    evidence_where = evidence_filters[evidence]
    review_where = "aa.action_status IS NULL" if unreviewed else None
    age_where = (
        "q.updated_at < now() - make_interval(days => %s)"
        if older_days is not None else None
    )
    liveness_where = {
        "live": "q.liveness_status='live'",
        "dead": "q.liveness_status='dead'",
        "uncertain": "q.liveness_status='uncertain'",
        "unchecked": "q.liveness_status IS NULL",
    }.get(liveness)
    filters = [x for x in (evidence_where, review_where, age_where, liveness_where) if x]
    params: list = []
    if older_days is not None:
        params.append(older_days)
    params.append(limit)
    order_sql = "q.score DESC, q.updated_at ASC, q.url" if priority else "q.updated_at ASC, q.url"
    with conn.cursor() as cur:
        cur.execute(
            "SELECT q.url, q.company, q.title, q.application_url, q.score, q.worker_id, "
            "q.attempts, q.apply_error, q.updated_at, q.liveness_status, "
            "q.liveness_reason, q.liveness_checked_at, "
            "ev.application_tool_calls, ev.job_log_path, ev.transcript_digest, ev.source AS event_source, "
            "ev.event_id, "
            "aa.chosen_action, aa.action_status, aa.reason AS audit_reason, aa.created_at AS audited_at, "
            "aa.pre_touch_requeue_count "
            "FROM apply_queue q "
            "LEFT JOIN LATERAL ("
            "  SELECT e.id AS event_id, e.application_tool_calls, e.job_log_path, e.transcript_digest, e.source "
            "  FROM apply_result_events e WHERE e.queue_name='apply_queue' AND e.url=q.url "
            "  ORDER BY e.created_at DESC, e.id DESC LIMIT 1"
            ") ev ON TRUE "
            "LEFT JOIN LATERAL ("
            "  SELECT a.chosen_action, a.action_status, a.reason, a.created_at, "
            "    (SELECT COUNT(*) FROM autotriage_actions r "
            "     WHERE r.url=a.url AND r.chosen_action='requeue_pre_touch_crash' "
            "       AND r.action_status='applied') AS pre_touch_requeue_count "
            "  FROM autotriage_actions a WHERE a.url=q.url "
            "  ORDER BY a.created_at DESC, a.id DESC LIMIT 1"
            ") aa ON TRUE "
            "WHERE q.lane='ats' AND q.status='crash_unconfirmed' "
            + (("AND " + " AND ".join(filters) + " ") if filters else "") +
            "ORDER BY " + order_sql + " LIMIT %s",
            tuple(params),
        )
        return [dict(r) for r in cur.fetchall()]


def print_crash_reviews(
    conn, *, limit: int = 25, evidence: str = "all", unreviewed: bool = False,
    older_days: int | None = None, priority: bool = False, liveness: str = "all",
) -> None:
    rows = list_crash_reviews(
        conn, limit=limit, evidence=evidence, unreviewed=unreviewed,
        older_days=older_days, priority=priority, liveness=liveness,
    )
    if not rows:
        print("no crash_unconfirmed rows")
        return
    for row in rows:
        print(row)


def print_challenges_grouped(conn) -> None:
    """Plain kind x host -> count table for the apply lane only, sourced from the
    SHARED queue.challenge_summary (same helper the console's build_challenges
    detail view and the linkedin-home CLI use, lane='apply' here)."""
    rows = queue.challenge_summary(conn, "apply")
    if not rows:
        print("no open/parked challenges")
        return
    for r in sorted(rows, key=lambda r: (r["kind"], r["host"])):
        print(f"{r['kind']}\t{r['host']}\t{r['count']}")


def _blocklist_match_sql(*, pg: bool) -> tuple[str, dict | list]:
    names, patterns = config.load_blocked_companies()
    if pg:
        return (
            "(LOWER(TRIM(COALESCE(company,''))) = ANY(%(blocked_names)s) "
            "OR url ILIKE ANY(%(blocked_pats)s) "
            "OR COALESCE(application_url,'') ILIKE ANY(%(blocked_pats)s))",
            {"blocked_names": list(names), "blocked_pats": patterns},
        )
    clauses: list[str] = []
    params: list[str] = []
    if names:
        placeholders = ",".join("?" * len(names))
        clauses.append(f"LOWER(TRIM(COALESCE(company,''))) IN ({placeholders})")
        params.extend(sorted(names))
    for pattern in patterns:
        clauses.append("url LIKE ?")
        clauses.append("COALESCE(application_url,'') LIKE ?")
        params.extend([pattern, pattern])
    if not clauses:
        return "0", []
    return "(" + " OR ".join(clauses) + ")", params


def blocklist_backfill(*, sqlite_conn=None, pg_conn=None, execute: bool = False) -> dict[str, int]:
    own_sq, own_pg = sqlite_conn is None, pg_conn is None
    sq = sqlite_conn or sync._home_conn()
    pg = pg_conn or pgqueue.connect()
    counts: dict[str, int] = {}
    brain_match, brain_params = _blocklist_match_sql(pg=False)
    pg_match, pg_params = _blocklist_match_sql(pg=True)
    try:
        brain_where = (
            f"{brain_match} "
            "AND COALESCE(apply_status,'') NOT IN ('applied','in_progress','crash_unconfirmed')"
        )
        counts["brain_matches"] = sq.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE " + brain_where,
            brain_params,
        ).fetchone()["n"]
        with pg.cursor() as cur:
            for table in ("apply_queue", "linkedin_queue"):
                key = f"{table}_matches"
                cur.execute(
                    f"SELECT COUNT(*) AS n FROM {table} WHERE status='queued' AND {pg_match}",
                    pg_params,
                )
                counts[key] = cur.fetchone()["n"]
        if not execute:
            pg.rollback()
            return counts

        cur = sq.execute(
            "UPDATE jobs SET apply_status='blocked', apply_error='company_blocklist' "
            "WHERE " + brain_where,
            brain_params,
        )
        counts["brain_blocked"] = cur.rowcount
        sq.commit()
        with pg.cursor() as cur:
            for table in ("apply_queue", "linkedin_queue"):
                cur.execute(
                    f"UPDATE {table} SET status='blocked', apply_error='company_blocklist', updated_at=now() "
                    f"WHERE status='queued' AND {pg_match}",
                    pg_params,
                )
                counts[f"{table}_blocked"] = cur.rowcount
        pg.commit()
        return counts
    finally:
        if own_sq:
            sq.close()
        if own_pg:
            pg.close()


def main(argv=None) -> int:  # pragma: no cover - CLI wiring
    _configure_stdio()
    p = argparse.ArgumentParser(prog="applypilot-fleet-apply-home")
    p.add_argument("--dsn", default=os.environ.get("FLEET_PG_DSN"))
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("push")
    sp.add_argument("--score-floor", type=float, default=7.0)
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--no-lane-filter", action="store_true",
                    help="Disable the default off-lane drift filter for this push.")
    sp.add_argument("--diagnostic", action="store_true",
                    help="Report candidate, dedup-skipped, and touched counts.")
    sp.add_argument("--approved-batch", default=None,
                    help="Explicit human-approved batch token to stamp on staged rows.")
    sub.add_parser("pull")
    ca = sub.add_parser("canary")
    ca.add_argument("k", type=int)
    sub.add_parser("lift-canary")
    ac = sub.add_parser("arm-canary-if-safe")
    ac.add_argument("k", type=int)
    ap = sub.add_parser("approve")
    ap.add_argument("--all-pushed", action="store_true")
    ssc = sub.add_parser("set-spend-cap")
    ssc.add_argument(
        "cap_usd",
        type=float,
        help="Hard $ ceiling written to fleet_config (0 = UNCAPPED). Routed through the "
        "mapped control-plane principal (never raw psql / postgres / fleet_worker).",
    )
    chp = sub.add_parser("challenges")
    chp.add_argument("--grouped", action="store_true")
    chp.add_argument("--stale-days", type=int, default=None,
                     help="Show only unresolved challenges older than N days, oldest first.")
    chp.add_argument("--limit", type=int, default=None,
                     help="Limit per-row challenge output.")
    chp.add_argument("--kind", default=None,
                     help="Show only one challenge kind, such as login_gate or visible_captcha.")
    chp.add_argument("--priority", action="store_true",
                     help="Sort by score descending, then oldest first.")
    cr = sub.add_parser("crash-review")
    cr.add_argument("--limit", type=int, default=25,
                    help="Oldest crash rows to show (default: 25).")
    cr.add_argument("--evidence", choices=("all", "missing", "zero_tool_calls", "tool_touched", "no_result_event"),
                    default="all", help="Filter by durable execution evidence.")
    cr.add_argument("--unreviewed", action="store_true",
                    help="Show only crash rows without any autotriage audit.")
    cr.add_argument("--older-days", type=int, default=None,
                    help="Show only rows older than N days.")
    cr.add_argument("--liveness", choices=("all", "live", "dead", "uncertain", "unchecked"),
                    default="all", help="Filter by public-posting liveness evidence.")
    cr.add_argument("--priority", action="store_true",
                    help="Sort by score descending, then oldest first.")
    rc = sub.add_parser("resolve-challenge")
    rc.add_argument("url")
    rc.add_argument("--skip", action="store_true")
    cc = sub.add_parser("cleanup-terminal-challenges")
    cc.add_argument("--older-days", type=int, default=1)
    cc.add_argument("--execute", action="store_true", help="Close matching challenge records.")
    rsc = sub.add_parser("retire-stale-parked-challenges")
    rsc.add_argument("--older-days", type=int, default=7)
    rsc.add_argument("--limit", type=int, default=100)
    rsc.add_argument("--execute", action="store_true",
                     help="Skip matching stale frozen ATS challenges and clear their leases.")
    oc = sub.add_parser("repair-orphan-challenges")
    oc.add_argument("--older-days", type=int, default=1)
    oc.add_argument("--execute", action="store_true", help="Create missing owner-inbox records.")
    nc = sub.add_parser("normalize-resolved-challenge-blocks")
    nc.add_argument("--limit", type=int, default=1000)
    nc.add_argument("--execute", action="store_true",
                    help="Normalize skipped resolved challenges; rows remain blocked and are never requeued.")
    bm = sub.add_parser("backfill-dedup-metadata")
    bm.add_argument("--limit", type=int, default=500)
    bm.add_argument("--execute", action="store_true",
                    help="Fill only unambiguous missing applied-set metadata; never remove dedup guards.")
    su = sub.add_parser("retire-stale-unapproved")
    su.add_argument("--older-days", type=int, default=7)
    su.add_argument("--limit", type=int, default=1000)
    su.add_argument("--include-safe-requeues", action="store_true",
                    help="Also retire old unapproved pre-touch requeues with zero browser calls.")
    su.add_argument("--execute", action="store_true", help="Retire matching never-leased rows.")
    qu = sub.add_parser("quarantine-unsafe-approved")
    qu.add_argument("--limit", type=int, default=1000)
    qu.add_argument("--execute", action="store_true", help="Block matching rows for manual review.")
    cbl = sub.add_parser("clear-blocked-leases")
    cbl.add_argument("--limit", type=int, default=1000)
    cbl.add_argument("--execute", action="store_true",
                     help="Clear lease metadata from blocked rows without changing outcomes.")
    ctl = sub.add_parser("clear-terminal-leases")
    ctl.add_argument("--limit", type=int, default=5000)
    ctl.add_argument("--execute", action="store_true",
                     help="Clear lease metadata from terminal rows without changing outcomes.")
    iq = sub.add_parser("quarantine-invalid-queued")
    iq.add_argument("--limit", type=int, default=1000)
    iq.add_argument("--execute", action="store_true",
                     help="Block queued ATS rows with LinkedIn source URLs or missing identity fields.")
    me = sub.add_parser("quarantine-missing-evidence-failures")
    me.add_argument("--limit", type=int, default=1000)
    me.add_argument("--execute", action="store_true",
                    help="Park failed rows with no result event for manual review; never requeue them.")
    mie = sub.add_parser("quarantine-missing-infrastructure-evidence")
    mie.add_argument("--limit", type=int, default=1000)
    mie.add_argument("--execute", action="store_true",
                     help="Park browser failures with no result event; never requeue them.")
    cie = sub.add_parser("quarantine-incomplete-infrastructure-evidence")
    cie.add_argument("--limit", type=int, default=1000)
    cie.add_argument("--execute", action="store_true",
                     help="Park browser failures with missing tool-call evidence; never requeue them.")
    tti = sub.add_parser("quarantine-tool-touched-infrastructure")
    tti.add_argument("--limit", type=int, default=1000)
    tti.add_argument("--execute", action="store_true",
                     help="Protect browser-touched infrastructure failures; never requeue them.")
    ncf = sub.add_parser("quarantine-no-confirmation-failures")
    ncf.add_argument("--limit", type=int, default=1000)
    ncf.add_argument("--execute", action="store_true",
                     help="Protect failed no-confirmation outcomes; never requeue them.")
    nti = sub.add_parser("normalize-terminal-inventory")
    nti.add_argument("--limit", type=int, default=5000)
    nti.add_argument("--execute", action="store_true",
                     help="Move stale/expired inventory out of failed; preserve reasons and never requeue.")
    npr = sub.add_parser("normalize-nonretryable-policy-failures")
    npr.add_argument("--limit", type=int, default=1000)
    npr.add_argument("--execute", action="store_true",
                     help="Move deterministic policy failures out of failed; never requeue.")
    ssf = sub.add_parser("quarantine-suspicious-stuck-failures")
    ssf.add_argument("--limit", type=int, default=1000)
    ssf.add_argument("--execute", action="store_true",
                     help="Protect suspicious/stuck flows; never requeue them.")
    raf = sub.add_parser("normalize-resolved-auth-failures")
    raf.add_argument("--limit", type=int, default=1000)
    raf.add_argument("--execute", action="store_true",
                     help="Move auth failures with resolved challenges to manual review.")
    blf = sub.add_parser("quarantine-budget-limit-failures")
    blf.add_argument("--limit", type=int, default=1000)
    blf.add_argument("--execute", action="store_true",
                     help="Protect budget/rate-limit failures; never requeue them.")
    ctf = sub.add_parser("normalize-clear-terminal-failures")
    ctf.add_argument("--limit", type=int, default=1000)
    ctf.add_argument("--execute", action="store_true",
                     help="Normalize exact already-applied/spam/reference terminal outcomes.")
    dtf = sub.add_parser("normalize-dedup-terminal-failures")
    dtf.add_argument("--limit", type=int, default=2000)
    dtf.add_argument("--execute", action="store_true",
                     help="Move dedup-protected already-applied rows out of failed.")
    fpf = sub.add_parser("normalize-final-protected-failures")
    fpf.add_argument("--limit", type=int, default=1000)
    fpf.add_argument("--execute", action="store_true",
                     help="Protect final crash/auth/tooling outcomes with no open challenge.")
    rrf = sub.add_parser("quarantine-residual-failures")
    rrf.add_argument("--limit", type=int, default=1000)
    rrf.add_argument("--execute", action="store_true",
                     help="Park residual failed rows for manual review; never requeue them.")
    rip = sub.add_parser("retry-infrastructure-pending")
    rip.add_argument("--limit", type=int, default=100)
    rip.add_argument("--cooldown-hours", type=int, default=1)
    rip.add_argument("--max-failures", type=int, default=1,
                     help="Maximum prior infrastructure failures eligible for retry.")
    rip.add_argument("--execute", action="store_true",
                     help="Requeue only zero-browser-call infrastructure parks.")
    dr = sub.add_parser("dedup-review")
    dr.add_argument("--limit", type=int, default=100)
    crd = sub.add_parser("canary-readiness")
    crd.add_argument("--limit", type=int, default=25)
    rd = sub.add_parser("readiness")
    rd.add_argument("--strict", action="store_true",
                    help="Exit 2 when readiness is false; still read-only.")
    cl = sub.add_parser("crash-liveness")
    cl.add_argument("--older-days", type=int, default=7)
    cl.add_argument("--limit", type=int, default=25)
    cl.add_argument("--oldest-first", action="store_true")
    cl.add_argument("--refresh-days", type=int, default=7)
    cl.add_argument("--execute", action="store_true", help="Stamp liveness metadata only.")
    cres = sub.add_parser("resolve-crash")
    cres.add_argument("url")
    cres.add_argument("--outcome", choices=("applied", "failed", "blocked"), required=True)
    cres.add_argument("--reason", required=True, help="Operator evidence supporting the outcome.")
    cres.add_argument("--actor", default="apply-home", help="Audit actor name.")
    pdc = sub.add_parser("park-dead-crashes")
    pdc.add_argument("--older-days", type=int, default=7)
    pdc.add_argument("--refresh-days", type=int, default=7)
    pdc.add_argument("--limit", type=int, default=25)
    pdc.add_argument("--execute", action="store_true",
                     help="Park recent dead-posting rows as submission_uncertain.")
    pdc.add_argument("--include-applied-set", action="store_true",
                     help="Also park rows protected only by applied_set; never delete that guard.")
    pdc.add_argument("--actor", default="apply-home", help="Audit actor name.")
    ppc = sub.add_parser("park-protected-crashes")
    ppc.add_argument("--older-days", type=int, default=7)
    ppc.add_argument("--limit", type=int, default=100)
    ppc.add_argument("--execute", action="store_true",
                     help="Park old applied_set-protected crashes as submission_uncertain.")
    ppc.add_argument("--include-tool-touched", action="store_true",
                     help="Also park rows with browser calls when no applied/inbox outcome exists.")
    ppc.add_argument("--actor", default="apply-home", help="Audit actor name.")
    pge = sub.add_parser("park-evidence-gap-crashes")
    pge.add_argument("--older-days", type=int, default=0)
    pge.add_argument("--limit", type=int, default=100)
    pge.add_argument("--execute", action="store_true",
                     help="Park rows without positive execution evidence as submission_uncertain.")
    pge.add_argument("--actor", default="apply-home", help="Audit actor name.")
    per = sub.add_parser("park-email-review-crashes")
    per.add_argument("--older-days", type=int, default=0)
    per.add_argument("--limit", type=int, default=100)
    per.add_argument("--execute", action="store_true",
                     help="Park inbox-linked rows as email_reconcile_review_required.")
    per.add_argument("--actor", default="apply-home", help="Audit actor name.")
    sub.add_parser("status")
    sub.add_parser("resume-if-safe")
    bf = sub.add_parser("blocklist-backfill")
    mode = bf.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    if not args.dsn:
        raise SystemExit("set --dsn or FLEET_PG_DSN")
    admission = emergency_admission.apply_home_admission(args.cmd)
    if not admission.allowed:
        print(admission.reason, file=sys.stderr)
        return 2
    with pgqueue.connect(args.dsn) as conn:
        fleet_schema._verify_schema_v3(conn)
        if args.cmd == "push":
            print("pushed", push_home(conn, score_floor=args.score_floor, limit=args.limit,
                                      lane_filter=not args.no_lane_filter,
                                      diagnostics=args.diagnostic,
                                      approved_batch=args.approved_batch))
        elif args.cmd == "pull":
            print("pulled", sync.pull_apply_results(pg_conn=conn))
        elif args.cmd == "canary":
            set_canary(conn, args.k)
            print("canary armed", args.k)
        elif args.cmd == "lift-canary":
            lift_canary(conn)
            print("canary lifted")
        elif args.cmd == "arm-canary-if-safe":
            if arm_canary_if_safe(conn, args.k):
                print("canary armed", args.k)
            else:
                print("left-disarmed (ats_paused or cost cap exceeded)")
                return 2
        elif args.cmd == "approve":
            print("approved batch", approve(conn, all_pushed=args.all_pushed))
        elif args.cmd == "set-spend-cap":
            print("spend cap set", set_spend_cap(conn, args.cap_usd))
        elif args.cmd == "challenges":
            if args.grouped:
                print_challenges_grouped(conn)
            else:
                for c in list_challenges(
                    conn, stale_days=args.stale_days, limit=args.limit,
                    kind=args.kind, priority=args.priority
                ):
                    print(c)
        elif args.cmd == "crash-review":
            print_crash_reviews(
                conn, limit=args.limit, evidence=args.evidence,
                unreviewed=args.unreviewed, older_days=args.older_days,
                priority=args.priority, liveness=args.liveness,
            )
        elif args.cmd == "resolve-challenge":
            print("resolved", resolve_challenge_cmd(conn, args.url, skip=args.skip))
        elif args.cmd == "cleanup-terminal-challenges":
            print(cleanup_terminal_challenges(conn, older_days=args.older_days, execute=args.execute))
        elif args.cmd == "retire-stale-parked-challenges":
            print(retire_stale_parked_challenges(
                conn, older_days=args.older_days, limit=args.limit, execute=args.execute,
            ))
        elif args.cmd == "repair-orphan-challenges":
            print(repair_orphan_challenges(conn, older_days=args.older_days, execute=args.execute))
        elif args.cmd == "normalize-resolved-challenge-blocks":
            print(normalize_resolved_challenge_blocks(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "backfill-dedup-metadata":
            print(backfill_dedup_metadata(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "retire-stale-unapproved":
            print(retire_stale_unapproved(
                conn, older_days=args.older_days, limit=args.limit, execute=args.execute,
                include_safe_requeues=args.include_safe_requeues,
            ))
        elif args.cmd == "quarantine-unsafe-approved":
            print(quarantine_unsafe_approved(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "clear-blocked-leases":
            print(clear_blocked_leases(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "clear-terminal-leases":
            print(clear_terminal_leases(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "quarantine-invalid-queued":
            print(quarantine_invalid_queued(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "quarantine-missing-evidence-failures":
            print(quarantine_missing_evidence_failures(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "quarantine-missing-infrastructure-evidence":
            print(quarantine_missing_infrastructure_evidence(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "quarantine-incomplete-infrastructure-evidence":
            print(quarantine_incomplete_infrastructure_evidence(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "quarantine-tool-touched-infrastructure":
            print(quarantine_tool_touched_infrastructure(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "quarantine-no-confirmation-failures":
            print(quarantine_no_confirmation_failures(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "normalize-terminal-inventory":
            print(normalize_terminal_inventory(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "normalize-nonretryable-policy-failures":
            print(normalize_nonretryable_policy_failures(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "quarantine-suspicious-stuck-failures":
            print(quarantine_suspicious_stuck_failures(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "normalize-resolved-auth-failures":
            print(normalize_resolved_auth_failures(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "quarantine-budget-limit-failures":
            print(quarantine_budget_limit_failures(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "normalize-clear-terminal-failures":
            print(normalize_clear_terminal_failures(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "normalize-dedup-terminal-failures":
            print(normalize_dedup_terminal_failures(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "normalize-final-protected-failures":
            print(normalize_final_protected_failures(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "quarantine-residual-failures":
            print(quarantine_residual_failures(conn, limit=args.limit, execute=args.execute))
        elif args.cmd == "retry-infrastructure-pending":
            print(retry_infrastructure_pending(
                conn, limit=args.limit, cooldown_hours=args.cooldown_hours,
                max_failures=args.max_failures, execute=args.execute,
            ))
        elif args.cmd == "dedup-review":
            for row in dedup_review(conn, limit=args.limit):
                print(row)
        elif args.cmd == "canary-readiness":
            print(canary_readiness(conn, limit=args.limit))
        elif args.cmd == "readiness":
            report = readiness(conn)
            print(report)
            if args.strict and not report["ready"]:
                return 2
        elif args.cmd == "crash-liveness":
            print(review_crash_liveness(
                conn, older_days=args.older_days, limit=args.limit,
                priority=not args.oldest_first, execute=args.execute,
                refresh_days=args.refresh_days,
            ))
        elif args.cmd == "resolve-crash":
            print(resolve_crash(
                conn, args.url, outcome=args.outcome, reason=args.reason, actor=args.actor,
            ))
        elif args.cmd == "park-dead-crashes":
            print(park_dead_crashes(
                conn, older_days=args.older_days, refresh_days=args.refresh_days,
                limit=args.limit, execute=args.execute, actor=args.actor,
                include_applied_set=args.include_applied_set,
            ))
        elif args.cmd == "park-protected-crashes":
            print(park_protected_crashes(
                conn, older_days=args.older_days, limit=args.limit,
                execute=args.execute, actor=args.actor,
                include_tool_touched=args.include_tool_touched,
            ))
        elif args.cmd == "park-evidence-gap-crashes":
            print(park_protected_crashes(
                conn, older_days=args.older_days, limit=args.limit,
                execute=args.execute, actor=args.actor,
                require_applied_set=False,
            ))
        elif args.cmd == "park-email-review-crashes":
            print(park_protected_crashes(
                conn, older_days=args.older_days, limit=args.limit,
                execute=args.execute, actor=args.actor,
                include_tool_touched=True, require_applied_set=False,
                include_inbox=True, require_inbox=True,
            ))
        elif args.cmd == "status":
            _print_status(conn)
        elif args.cmd == "resume-if-safe":
            if queue._cost_cap_exceeded(conn):
                print("left-paused (cap exceeded)")
            elif resume_if_safe(conn):
                print("resumed")
            else:
                print("left-paused (ats_paused or already running)")
        elif args.cmd == "blocklist-backfill":
            print(blocklist_backfill(pg_conn=conn, execute=bool(args.execute)))
    return 0


def _print_status(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT status, count(*) AS n FROM apply_queue GROUP BY status")
        depth = {r["status"]: r["n"] for r in cur.fetchall()}
        cur.execute(
            "SELECT paused, ats_paused, ats_pause_source, ats_apply_mode, canary_enabled, "
            "canary_remaining, spend_cap_usd, updated_at FROM fleet_config WHERE id=1"
        )
        cfg = cur.fetchone()
        cur.execute("SELECT to_regclass('fleet_desired_state') AS rel")
        desired_control = {"available": False, "machine_owner": "home", "desired_workers": None}
        if cur.fetchone()["rel"] is not None:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema=current_schema() AND table_name='fleet_desired_state'"
            )
            available_columns = {str(row["column_name"]) for row in cur.fetchall()}
            desired_columns = ["machine_owner", "desired_workers"] + [
                name for name in ("agent", "model", "generation", "updated_by", "updated_at")
                if name in available_columns
            ]
            cur.execute(
                f"SELECT {', '.join(desired_columns)} FROM fleet_desired_state "
                "WHERE machine_owner='home'"
            )
            desired_row = cur.fetchone()
            if desired_row:
                desired_control = {"available": True, **dict(desired_row)}
            else:
                desired_control = {
                    "available": True, "machine_owner": "home", "desired_workers": 0,
                    "reason": "no desired-state row",
                }
        cur.execute(
            "SELECT COUNT(*) AS registered, "
            "COUNT(*) FILTER (WHERE last_beat >= now() - interval '2 minutes') AS recent, "
            "COUNT(*) FILTER (WHERE last_beat < now() - interval '2 minutes') AS stale "
            "FROM worker_heartbeat WHERE role='apply'"
        )
        apply_heartbeat_summary = dict(cur.fetchone())
        cur.execute("SELECT COALESCE(SUM(est_cost_usd),0) AS s FROM apply_queue")
        spend = float(cur.fetchone()["s"])
        cur.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM auth_challenge a
               WHERE a.resolved_at IS NULL
                 AND (EXISTS (SELECT 1 FROM apply_queue q WHERE q.url=a.url AND q.lane='ats'
                              AND q.status='leased' AND q.apply_status='challenge_pending')
                      OR (NOT EXISTS (SELECT 1 FROM apply_queue q WHERE q.url=a.url)
                          AND NOT EXISTS (SELECT 1 FROM linkedin_queue l WHERE l.url=a.url)))) AS ats_open,
              (SELECT COUNT(*) FROM auth_challenge a
               WHERE a.resolved_at IS NULL
                 AND EXISTS (SELECT 1 FROM linkedin_queue l WHERE l.url=a.url
                             AND l.status='leased' AND l.apply_status='challenge_pending')) AS linkedin_open
            """
        )
        challenge_counts = cur.fetchone()
        open_ch = challenge_counts["ats_open"]
        linkedin_open_ch = challenge_counts["linkedin_open"]
        cur.execute(
            """
            WITH ats_auth AS (
              SELECT a.*
              FROM auth_challenge a
              WHERE a.resolved_at IS NULL
                AND (EXISTS (SELECT 1 FROM apply_queue q WHERE q.url=a.url AND q.lane='ats'
                             AND q.status='leased' AND q.apply_status='challenge_pending')
                     OR (NOT EXISTS (SELECT 1 FROM apply_queue q WHERE q.url=a.url)
                         AND NOT EXISTS (SELECT 1 FROM linkedin_queue l WHERE l.url=a.url)))
            )
            SELECT
              (SELECT COUNT(*) FROM ats_auth) AS open_count,
              (SELECT COUNT(*) FROM ats_auth
               WHERE raised_at < now() - interval '1 day') AS old_open,
              (SELECT COUNT(*) FROM apply_queue
               WHERE lane='ats' AND status = 'leased' AND apply_status = 'challenge_pending'
                 AND lease_expires_at > now() + interval '1 day') AS parked_frozen,
              (SELECT COUNT(*) FROM apply_queue
               WHERE lane='ats' AND status = 'blocked' AND apply_status = 'challenge_skipped') AS blocked_skipped,
              (SELECT COUNT(*) FROM apply_queue
               WHERE lane='ats' AND status = 'leased' AND apply_status = 'challenge_pending') AS active_pending,
              (SELECT COUNT(*) FROM apply_queue q
               WHERE q.status = 'leased' AND q.apply_status = 'challenge_pending'
                 AND q.lease_expires_at > now() + interval '1 day'
                 AND NOT EXISTS (
                   SELECT 1 FROM auth_challenge open_ch
                   WHERE open_ch.url = q.url AND open_ch.resolved_at IS NULL
                 )) AS parked_without_open
            """
        )
        challenge_diag = dict(cur.fetchone())
        cur.execute(
            """
            SELECT a.kind, COUNT(*) AS n
            FROM auth_challenge a
            JOIN apply_queue q ON q.url = a.url AND q.lane = 'ats'
            WHERE a.resolved_at IS NULL
              AND q.status = 'leased' AND q.apply_status = 'challenge_pending'
            GROUP BY a.kind ORDER BY n DESC, a.kind
            """
        )
        challenge_kinds = {str(row["kind"] or "unknown"): int(row["n"] or 0) for row in cur.fetchall()}
        cur.execute(
            """
            SELECT
              COUNT(*) FILTER (WHERE ar.application_tool_calls IS NULL) AS missing_evidence,
              COUNT(*) FILTER (WHERE ar.application_tool_calls = 0) AS zero_tool_calls,
              COUNT(*) FILTER (WHERE ar.application_tool_calls > 0) AS tool_touched,
              COUNT(*) FILTER (WHERE ar.url IS NULL) AS no_result_event,
              COUNT(*) FILTER (WHERE ar.event_source = 'worker_lease') AS lease_started_no_terminal,
              COUNT(*) FILTER (WHERE q.liveness_status = 'dead') AS liveness_dead,
              COUNT(*) FILTER (WHERE q.liveness_status = 'live') AS liveness_live,
              COUNT(*) FILTER (WHERE q.liveness_status = 'uncertain') AS liveness_uncertain,
              COUNT(*) FILTER (
                WHERE q.liveness_status = 'dead'
                  AND q.updated_at < now() - interval '7 days'
                  AND q.liveness_checked_at >= now() - interval '7 days'
                  AND NOT EXISTS (
                    SELECT 1 FROM apply_result_events pe
                    WHERE pe.queue_name='apply_queue' AND pe.url=q.url
                      AND (COALESCE(pe.application_tool_calls,0)>0
                           OR pe.status='applied' OR pe.apply_status='applied')
                  )
                  AND NOT EXISTS (SELECT 1 FROM applied_set pas WHERE pas.dedup_key=q.dedup_key)
                  AND NOT EXISTS (
                    SELECT 1 FROM inbox_outcomes pio
                    WHERE pio.dedup_key=q.dedup_key
                       OR pio.job_url IN (q.url, q.application_url)
                  )
              ) AS parkable_dead,
              COUNT(*) FILTER (WHERE q.updated_at < now() - interval '7 days') AS older_7d,
              COUNT(*) FILTER (WHERE q.updated_at >= now() - interval '7 days'
                                      AND q.updated_at < now() - interval '1 day') AS age_1d_to_7d,
              MIN(q.updated_at) AS oldest_crash_at,
              COUNT(*) FILTER (
                WHERE ar.application_tool_calls = 0
                  AND COALESCE(q.apply_error, '') IN ('failed:no_result_line', 'failed:timeout')
              ) AS safe_pre_touch_candidates,
              COUNT(*) FILTER (WHERE aa.action_status = 'manual_review_required') AS manual_review_audited,
              COUNT(*) FILTER (WHERE aa.action_status IS NULL) AS no_autotriage_audit
            FROM apply_queue q
            LEFT JOIN LATERAL (
              SELECT e.url, e.application_tool_calls, e.source AS event_source
              FROM apply_result_events e
              WHERE e.queue_name = 'apply_queue' AND e.url = q.url
              ORDER BY e.created_at DESC, e.id DESC
              LIMIT 1
            ) ar ON TRUE
            LEFT JOIN LATERAL (
              SELECT a.action_status
              FROM autotriage_actions a
              WHERE a.url = q.url
              ORDER BY a.created_at DESC, a.id DESC
              LIMIT 1
            ) aa ON TRUE
            WHERE q.lane = 'ats' AND q.status = 'crash_unconfirmed'
            """
        )
        crash_diag = dict(cur.fetchone())
        cur.execute(
            """
            SELECT
              COUNT(*) AS events_24h,
              COUNT(*) FILTER (WHERE source = 'worker') AS worker_events_24h,
              COUNT(*) FILTER (WHERE source = 'worker' AND status = 'crash_unconfirmed') AS worker_crash_events_24h,
              COUNT(*) FILTER (WHERE source = 'worker' AND application_tool_calls IS NULL) AS worker_missing_evidence_24h,
              COUNT(*) FILTER (WHERE source = 'worker' AND application_tool_calls IS NOT NULL) AS worker_evidenced_events_24h,
              COUNT(*) FILTER (WHERE source = 'watchdog') AS watchdog_events_24h,
              COUNT(*) FILTER (WHERE source LIKE 'historical_reaudit:%') AS historical_reaudit_events_24h
            FROM apply_result_events
            WHERE queue_name = 'apply_queue' AND created_at >= now() - interval '24 hours'
            """
        )
        evidence_freshness = dict(cur.fetchone())
        crash_liveness_task = crash_liveness_task_diagnosis()
        cur.execute(
            "SELECT last_beat, state, last_error FROM worker_heartbeat "
            "WHERE worker_id = 'otp_responder'"
        )
        otp_heartbeat = cur.fetchone()
        cur.execute(
            """
            SELECT
              COUNT(*) FILTER (WHERE consumed_at IS NULL AND code IS NULL AND expires_at > now()) AS pending_requests,
              COUNT(*) FILTER (WHERE code IS NOT NULL AND consumed_at IS NULL) AS ready_requests,
              COUNT(*) FILTER (WHERE matched_email_ts >= now() - interval '24 hours') AS matched_requests_24h
            FROM otp_request
            """
        )
        otp_counts = cur.fetchone()
        otp_diagnosis = {
            "heartbeat_healthy": bool(
                otp_heartbeat and otp_heartbeat["last_beat"] >= _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=45)
            ),
            "state": otp_heartbeat["state"] if otp_heartbeat else None,
            "last_error": otp_heartbeat["last_error"] if otp_heartbeat else None,
            "pending_requests": int(otp_counts["pending_requests"] or 0),
            "ready_requests": int(otp_counts["ready_requests"] or 0),
            "matched_requests_24h": int(otp_counts["matched_requests_24h"] or 0),
        }
    print({"queue": depth, "paused": cfg["paused"], "ats_paused": cfg["ats_paused"],
           "ats_pause_source": cfg["ats_pause_source"], "ats_apply_mode": cfg["ats_apply_mode"],
           "fleet_config_updated_at": cfg["updated_at"],
           "desired_control": desired_control,
           "apply_heartbeat_summary": apply_heartbeat_summary,
           "canary_enabled": cfg["canary_enabled"],
           "canary_remaining": cfg["canary_remaining"],
           "spend_cap_usd": float(cfg["spend_cap_usd"] or 0), "apply_spend": spend,
           "open_challenges": open_ch, "linkedin_open_challenges": linkedin_open_ch,
           "challenge_diagnosis": challenge_diag, "challenge_kinds": challenge_kinds,
           "crash_diagnosis": crash_diag,
           "crash_liveness_task": crash_liveness_task,
           "evidence_freshness": evidence_freshness,
           "otp_diagnosis": otp_diagnosis,
           "queue_diagnosis": queue_diagnosis.queue_diagnosis(conn)})
