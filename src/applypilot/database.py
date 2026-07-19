"""ApplyPilot database layer: schema, migrations, stats, and connection helpers.

Single source of truth for the jobs table schema. All columns from every
pipeline stage are created up front so any stage can run independently
without migration ordering issues.
"""

import os
import re
import sqlite3
import threading
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from applypilot.auth_event_storage import (
    OUTDATED_STORAGE_PREDICATE,
    scrub_inbox_events,
)
from applypilot.config import DB_PATH

MIN_GOOD_DESCRIPTION_CHARS = 200
THIN_DESCRIPTION_CHARS = 500
DEFAULT_LLM_MAX_JOB_AGE_DAYS = 45


def _llm_max_job_age_days() -> int:
    raw = os.environ.get("APPLYPILOT_MAX_LLM_JOB_AGE_DAYS")
    if not raw:
        return DEFAULT_LLM_MAX_JOB_AGE_DAYS
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_LLM_MAX_JOB_AGE_DAYS


def llm_stage_liveness_sql() -> str:
    """SQL guard for token-spending stages."""
    max_age_days = _llm_max_job_age_days()
    freshness = ""
    if max_age_days > 0:
        freshness = (
            "AND (COALESCE(posted_at, discovered_at) IS NULL "
            f"OR julianday(COALESCE(posted_at, discovered_at)) >= julianday('now', '-{max_age_days} days') "
            "OR COALESCE(liveness_status, '') = 'live')"
        )
    return f"AND COALESCE(liveness_status, '') != 'dead' {freshness}"

def _ensure_desc_quality(row: dict[str, Any]) -> tuple[str, int]:
    """Compute deterministic description-quality telemetry for one job row."""
    from applypilot.discovery import desc_quality

    title = row.get("title")
    full_description = row.get("full_description")
    flags, score = desc_quality.assess_description(title, full_description)
    return ",".join(flags), score


# Thread-local connection storage — each thread gets its own connection
# (required for SQLite thread safety with parallel workers)
_local = threading.local()


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Get a thread-local cached SQLite connection with WAL mode enabled.

    Each thread gets its own connection (required for SQLite thread safety).
    Connections are cached and reused within the same thread.

    Args:
        db_path: Override the default DB_PATH. Useful for testing.

    Returns:
        sqlite3.Connection configured with WAL mode and row factory.
    """
    path = str(db_path or DB_PATH)

    if not hasattr(_local, 'connections'):
        _local.connections = {}

    conn = _local.connections.get(path)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except sqlite3.ProgrammingError:
            pass

    conn = sqlite3.connect(path, timeout=60)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    # Tolerate write contention from parallel stages/workers and OneDrive sync
    # locking the DB file: wait up to 60s for the writer lock instead of failing
    # with "database is locked". synchronous=NORMAL is WAL-safe and cuts fsync churn.
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    _local.connections[path] = conn
    return conn


def close_connection(db_path: Path | str | None = None) -> None:
    """Close the cached connection for the current thread."""
    path = str(db_path or DB_PATH)
    if hasattr(_local, 'connections'):
        conn = _local.connections.pop(path, None)
        if conn is not None:
            conn.close()


def init_db(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Create the full jobs table with all columns from every pipeline stage.

    This is idempotent -- safe to call on every startup. Uses CREATE TABLE IF NOT EXISTS
    so it won't destroy existing data.

    Schema columns by stage:
      - Discovery:  url, title, salary, description, location, site, strategy, discovered_at
      - Enrichment: full_description, application_url, detail_scraped_at, detail_error
      - Scoring:    fit_score, score_reasoning, score_model, score_provider, scored_at
      - Diagnosis:  fit_gap_category, fit_gap_severity, resume_fixability,
                    recommended_action, fit_diagnosis, fit_diagnosis_json,
                    diagnosis_model, diagnosis_provider, diagnosed_at
      - Tailoring:  tailored_resume_path, tailored_at, tailor_attempts
      - Cover:      cover_letter_path, cover_letter_at, cover_attempts
      - Apply:      applied_at, apply_status, apply_error, apply_attempts,
                   agent_id, last_attempted_at, apply_duration_ms, apply_task_id,
                   verification_confidence

    Args:
        db_path: Override the default DB_PATH.

    Returns:
        sqlite3.Connection with the schema initialized.
    """
    path = db_path or DB_PATH

    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            -- Discovery stage (smart_extract / job_search)
            url                   TEXT PRIMARY KEY,
            title                 TEXT,
            salary                TEXT,
            description           TEXT,
            location              TEXT,
            site                  TEXT,
            company               TEXT,
            source_board          TEXT,
            strategy              TEXT,
            discovered_at         TEXT,
            dedupe_key            TEXT,
            duplicate_of_url      TEXT,
            duplicate_reason      TEXT,
            duplicate_detected_at TEXT,

            -- Enrichment stage (detail_scraper)
            full_description      TEXT,
            application_url       TEXT,
            detail_scraped_at     TEXT,
            detail_error          TEXT,

            -- Scoring stage (job_scorer)
            fit_score             INTEGER,
            score_reasoning       TEXT,
            fit_verdict           TEXT,
            score_error           TEXT,
            score_error_at        TEXT,
            score_attempts        INTEGER DEFAULT 0,
            score_model           TEXT,
            score_provider        TEXT,
            scored_at             TEXT,
            role_fit_score        INTEGER,
            audit_score           REAL,
            audit_label           TEXT,
            audit_flags           TEXT,
            audit_reason          TEXT,
            audited_at            TEXT,

            -- Diagnosis stage (fit diagnosis / resume gap analysis)
            fit_gap_category      TEXT,
            fit_gap_severity      TEXT,
            resume_fixability     TEXT,
            recommended_action    TEXT,
            fit_diagnosis         TEXT,
            fit_diagnosis_json    TEXT,
            diagnosis_model       TEXT,
            diagnosis_provider    TEXT,
            diagnosed_at          TEXT,

            -- Tailoring stage (resume tailor)
            tailored_resume_path  TEXT,
            tailored_at           TEXT,
            tailor_attempts       INTEGER DEFAULT 0,

            -- Cover letter stage
            cover_letter_path     TEXT,
            cover_letter_at       TEXT,
            cover_attempts        INTEGER DEFAULT 0,

            -- Application stage
            applied_at            TEXT,
            apply_status          TEXT,
            apply_error           TEXT,
            apply_attempts        INTEGER DEFAULT 0,
            agent_id              TEXT,
            last_attempted_at     TEXT,
            apply_duration_ms     INTEGER,
            apply_task_id         TEXT,
            verification_confidence TEXT
        )
    """)
    conn.commit()

    # Run migrations for any columns added after initial schema
    ensure_columns(conn)
    backfill_company_source_fields(conn)
    repair_retryable_score_errors(conn)
    ensure_job_indexes(conn)
    ensure_application_tables(conn)
    ensure_inbox_auth_tables(conn)
    ensure_pipeline_tables(conn)
    ensure_research_tables(conn)
    ensure_desc_quality_tables(conn)
    ensure_outcome_tables(conn)
    ensure_canonical_decision_tables(conn)
    ensure_tenant_tables(conn)

    return conn


# Complete column registry: column_name -> SQL type with optional default.
# This is the single source of truth. Adding a column here is all that's needed
# for it to appear in both new databases and migrated ones.
_ALL_COLUMNS: dict[str, str] = {
    # Discovery
    "url": "TEXT PRIMARY KEY",
    "title": "TEXT",
    "salary": "TEXT",
    "description": "TEXT",
    "location": "TEXT",
    "site": "TEXT",
    "company": "TEXT",
    "source_board": "TEXT",
    "strategy": "TEXT",
    "discovered_at": "TEXT",
    "dedupe_key": "TEXT",
    "duplicate_of_url": "TEXT",
    "duplicate_reason": "TEXT",
    "duplicate_detected_at": "TEXT",
    # Enrichment
    "full_description": "TEXT",
    "application_url": "TEXT",
    "detail_scraped_at": "TEXT",
    "detail_error": "TEXT",
    # LinkedIn external apply URL resolver
    "linkedin_resolved_at": "TEXT",
    "linkedin_resolve_status": "TEXT",
    "linkedin_resolve_error": "TEXT",
    "linkedin_resolve_attempts": "INTEGER DEFAULT 0",
    "linkedin_resolve_final_url": "TEXT",
    "linkedin_unresolved_kind": "TEXT",
    "linkedin_next_action": "TEXT",
    # Cross-source apply URL resolver. This lets LinkedIn rows inherit a real
    # ATS/company apply URL from an already-discovered matching job.
    "apply_url_resolved_at": "TEXT",
    "apply_url_resolution_strategy": "TEXT",
    "apply_url_resolution_confidence": "REAL",
    "apply_url_resolution_source": "TEXT",
    "apply_url_resolution_error": "TEXT",
    "apply_url_resolution_attempts": "INTEGER DEFAULT 0",
    "apply_url_resolution_matched_url": "TEXT",
    "detail_attempts": "INTEGER DEFAULT 0",
    # Scoring
    "fit_score": "INTEGER",
    "score_reasoning": "TEXT",
    "fit_verdict": "TEXT",
    "score_error": "TEXT",
    "score_error_at": "TEXT",
    "score_attempts": "INTEGER DEFAULT 0",
    "score_model": "TEXT",
    "score_provider": "TEXT",
    "scored_at": "TEXT",
    # Score audit / reranking
    "role_fit_score": "INTEGER",
    "audit_score": "REAL",
    "audit_label": "TEXT",
    "audit_flags": "TEXT",
    "audit_reason": "TEXT",
    "audited_at": "TEXT",
    # External recommendation-engine decision ("brainstorm"). When present, the
    # decision is authoritative for the apply gate (written into audit_score),
    # ApplyPilot's own audit stage skips the row, and its LLM fit_score is kept
    # as a benchmark datapoint to compare against external_decision_score.
    "decision_source": "TEXT",
    "decision_verdict": "TEXT",
    "external_decision_score": "REAL",
    "decision_at": "TEXT",
    # Canonical decision cache. These fields have no apply-eligible defaults;
    # job_decisions remains the authoritative immutable record.
    "canonical_decision_id": "TEXT",
    "canonical_policy_version": "TEXT",
    "canonical_action": "TEXT",
    "canonical_score": "REAL",
    "canonical_decided_at": "TEXT",
    # Fit diagnosis / gap analysis
    "fit_gap_category": "TEXT",
    "fit_gap_severity": "TEXT",
    "resume_fixability": "TEXT",
    "recommended_action": "TEXT",
    "fit_diagnosis": "TEXT",
    "fit_diagnosis_json": "TEXT",
    "diagnosis_model": "TEXT",
    "diagnosis_provider": "TEXT",
    "diagnosed_at": "TEXT",
    # Tailoring
    "tailored_resume_path": "TEXT",
    "tailored_at": "TEXT",
    "tailor_attempts": "INTEGER DEFAULT 0",
    # Cover letter
    "cover_letter_path": "TEXT",
    "cover_letter_at": "TEXT",
    "cover_attempts": "INTEGER DEFAULT 0",
    # Application
    "applied_at": "TEXT",
    "apply_status": "TEXT",
    "apply_error": "TEXT",
    "apply_attempts": "INTEGER DEFAULT 0",
    "agent_id": "TEXT",
    "last_attempted_at": "TEXT",
    "apply_duration_ms": "INTEGER",
    "apply_task_id": "TEXT",
    "verification_confidence": "TEXT",
    # Pre-apply posting-liveness gate (additive; NEVER deletes rows — dead jobs
    # stay in the DB with full JD/scores + this close-marker, for future training)
    "liveness_status": "TEXT",        # live | dead | uncertain (NULL = unchecked)
    "last_verified_live": "TEXT",     # ISO8601 timestamp of last liveness probe
    "liveness_reason": "TEXT",        # signal that produced the verdict
    "posted_at": "TEXT",
    "valid_through": "TEXT",
    # Research advisory scores (denormalized "current best research opinion" written by the
    # TS KG scorer; detail rows live in research_scores). ADVISORY ONLY -- NEVER folded into
    # COALESCE(audit_score, fit_score); reaches the apply gate only via owner promotion.
    # Spec: docs/superpowers/specs/2026-06-25-unified-brain-pipeline-design.md §1b.
    "research_fit_score": "REAL",
    "research_decision": "TEXT",
    "research_model": "TEXT",
    "research_scored_at": "TEXT",
    "research_opt_in": "INTEGER DEFAULT 0",
    "desc_quality_flags": "TEXT",
    "desc_quality_score": "INTEGER",
    "desc_quality_at": "TEXT",
}


def ensure_columns(conn: sqlite3.Connection | None = None) -> list[str]:
    """Add any missing columns to the jobs table (forward migration).

    Reads the current table schema via PRAGMA table_info and compares against
    the full column registry. Any missing columns are added with ALTER TABLE.

    This makes it safe to upgrade the database from any previous version --
    columns are only added, never removed or renamed.

    Args:
        conn: Database connection. Uses get_connection() if None.

    Returns:
        List of column names that were added (empty if schema was already current).
    """
    if conn is None:
        conn = get_connection()

    existing = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    added = []

    for col, dtype in _ALL_COLUMNS.items():
        if col not in existing:
            # PRIMARY KEY columns can't be added via ALTER TABLE, but url
            # is always created with the table itself so this is safe
            if "PRIMARY KEY" in dtype:
                continue
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {dtype}")
            added.append(col)

    if added:
        conn.commit()

    return added


def backfill_company_source_fields(conn: sqlite3.Connection | None = None) -> None:
    """Populate company/source_board for databases created before those fields existed."""
    if conn is None:
        conn = get_connection()

    board_names = (
        "indeed",
        "linkedin",
        "zip_recruiter",
        "glassdoor",
        "the_muse",
        "remotejobs_org",
        "remotive",
        "arbeitnow",
    )
    placeholders = ",".join("?" for _ in board_names)
    conn.execute(
        f"""
        UPDATE jobs
           SET source_board = COALESCE(source_board, site)
         WHERE source_board IS NULL
           AND lower(COALESCE(site, '')) IN ({placeholders})
        """,
        board_names,
    )
    conn.execute(
        f"""
        UPDATE jobs
           SET company = COALESCE(company, site)
         WHERE company IS NULL
           AND lower(COALESCE(site, '')) NOT IN ({placeholders})
        """,
        board_names,
    )
    conn.execute(
        """
        UPDATE jobs
           SET source_board = COALESCE(source_board, strategy, site)
         WHERE source_board IS NULL
        """
    )
    conn.commit()


def repair_retryable_score_errors(conn: sqlite3.Connection | None = None) -> int:
    """Move legacy LLM error scores back to pending scoring.

    Older runs persisted provider failures as fit_score=0, which made them look
    completed. Those rows should remain retryable and keep the failure note.
    """
    if conn is None:
        conn = get_connection()

    cursor = conn.execute(
        """
        UPDATE jobs
           SET score_error = COALESCE(score_error, score_reasoning),
               score_error_at = COALESCE(score_error_at, scored_at),
               fit_score = NULL,
               scored_at = NULL
         WHERE fit_score = 0
           AND score_reasoning LIKE '%LLM error:%'
        """
    )
    repaired = int(cursor.rowcount or 0)
    conn.execute(
        """
        UPDATE jobs
           SET role_fit_score = NULL,
               audit_score = NULL,
               audit_label = NULL,
               audit_flags = NULL,
               audit_reason = NULL,
               audited_at = NULL,
               fit_gap_category = NULL,
               fit_gap_severity = NULL,
               resume_fixability = NULL,
               recommended_action = NULL,
               fit_diagnosis = NULL,
               fit_diagnosis_json = NULL,
               diagnosis_model = NULL,
               diagnosis_provider = NULL,
               diagnosed_at = NULL
         WHERE fit_score IS NULL
           AND score_error IS NOT NULL
        """
    )
    conn.commit()
    return repaired


def ensure_job_indexes(conn: sqlite3.Connection | None = None) -> None:
    """Create indexes used by status and resumable pipeline stages."""
    if conn is None:
        conn = get_connection()

    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_source_board ON jobs(source_board)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_discovered_at ON jobs(discovered_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_dedupe_key ON jobs(dedupe_key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_duplicate_of_url ON jobs(duplicate_of_url)")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_jobs_pending_enrich
            ON jobs(site, discovered_at)
         WHERE detail_scraped_at IS NULL AND duplicate_of_url IS NULL
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_jobs_pending_score
            ON jobs(discovered_at)
         WHERE full_description IS NOT NULL AND fit_score IS NULL AND duplicate_of_url IS NULL
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_jobs_pending_tailor
            ON jobs(COALESCE(audit_score, fit_score), discovered_at)
         WHERE full_description IS NOT NULL
           AND tailored_resume_path IS NULL
           AND COALESCE(tailor_attempts, 0) < 5
           AND duplicate_of_url IS NULL
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_jobs_ready_to_apply
            ON jobs(COALESCE(audit_score, fit_score), discovered_at)
         WHERE tailored_resume_path IS NOT NULL AND applied_at IS NULL AND duplicate_of_url IS NULL
        """
    )
    conn.commit()


def ensure_application_tables(conn: sqlite3.Connection | None = None) -> None:
    """Create first-class application tracking tables.

    jobs.applied_at is the quick pipeline flag. These tables are the durable
    tracker: current application status plus an append-only status history.
    """
    if conn is None:
        conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            job_url               TEXT NOT NULL UNIQUE,
            title                 TEXT,
            company               TEXT,
            source                TEXT,
            source_url            TEXT,
            application_url       TEXT,
            status                TEXT NOT NULL DEFAULT 'shortlisted',
            channel               TEXT,
            applied_at            TEXT,
            last_status_at        TEXT,
            next_follow_up_at     TEXT,
            resume_path           TEXT,
            cover_letter_path     TEXT,
            contact_name          TEXT,
            contact_email         TEXT,
            contact_url           TEXT,
            notes                 TEXT,
            created_at            TEXT NOT NULL,
            updated_at            TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS application_events (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            job_url               TEXT NOT NULL,
            status                TEXT NOT NULL,
            event_type            TEXT NOT NULL,
            channel               TEXT,
            happened_at           TEXT NOT NULL,
            notes                 TEXT
        )
    """)
    _remove_application_job_foreign_keys(conn)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_applications_status ON applications(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_applications_applied_at ON applications(applied_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_applications_followup ON applications(next_follow_up_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_application_events_job_url ON application_events(job_url)")

    # Backfill current applied jobs into the tracker once, preserving dates.
    rows = conn.execute("""
        SELECT j.*
        FROM jobs j
        LEFT JOIN applications a ON a.job_url = j.url
        WHERE j.applied_at IS NOT NULL AND a.job_url IS NULL
    """).fetchall()
    for row in rows:
        applied_at = row["applied_at"]
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            INSERT INTO applications (
                job_url, title, company, source, source_url, application_url,
                status, channel, applied_at, last_status_at, resume_path,
                cover_letter_path, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'applied', 'legacy', ?, ?, ?, ?, ?, ?)
        """, (
            row["url"], row["title"], row["company"] or row["site"], row["source_board"] or row["site"], row["url"],
            row["application_url"], applied_at, applied_at,
            row["tailored_resume_path"], row["cover_letter_path"], now, now,
        ))
        conn.execute("""
            INSERT INTO application_events (
                job_url, status, event_type, channel, happened_at, notes
            )
            VALUES (?, 'applied', 'backfill', 'legacy', ?, 'Backfilled from jobs.applied_at')
        """, (row["url"], applied_at))
    conn.commit()


def _remove_application_job_foreign_keys(conn: sqlite3.Connection) -> None:
    """Keep the durable application ledger independent from the mutable jobs catalog."""
    table_specs = {
        "applications": (
            "id,job_url,title,company,source,source_url,application_url,status,channel,"
            "applied_at,last_status_at,next_follow_up_at,resume_path,cover_letter_path,"
            "contact_name,contact_email,contact_url,notes,created_at,updated_at",
            """
            CREATE TABLE applications_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_url TEXT NOT NULL UNIQUE,
                title TEXT, company TEXT, source TEXT, source_url TEXT,
                application_url TEXT,
                status TEXT NOT NULL DEFAULT 'shortlisted',
                channel TEXT, applied_at TEXT, last_status_at TEXT,
                next_follow_up_at TEXT, resume_path TEXT, cover_letter_path TEXT,
                contact_name TEXT, contact_email TEXT, contact_url TEXT, notes TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )
            """,
        ),
        "application_events": (
            "id,job_url,status,event_type,channel,happened_at,notes",
            """
            CREATE TABLE application_events_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_url TEXT NOT NULL,
                status TEXT NOT NULL,
                event_type TEXT NOT NULL,
                channel TEXT,
                happened_at TEXT NOT NULL,
                notes TEXT
            )
            """,
        ),
    }
    for table, (columns, create_sql) in table_specs.items():
        if not conn.execute(f"PRAGMA foreign_key_list({table})").fetchall():
            continue
        replacement = f"{table}_new"
        conn.execute(f"DROP TABLE IF EXISTS {replacement}")
        conn.execute(create_sql)
        conn.execute(
            f"INSERT INTO {replacement} ({columns}) SELECT {columns} FROM {table}"
        )
        conn.execute(f"DROP TABLE {table}")
        conn.execute(f"ALTER TABLE {replacement} RENAME TO {table}")


def ensure_inbox_auth_tables(conn: sqlite3.Connection | None = None) -> None:
    """Create durable inbox event and auth challenge tracking tables."""
    if conn is None:
        conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS inbox_events (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id            TEXT NOT NULL UNIQUE,
            thread_id             TEXT,
            sender                TEXT,
            sender_domain         TEXT,
            subject               TEXT,
            received_at           TEXT,
            event_type            TEXT NOT NULL,
            confidence            TEXT NOT NULL,
            matched_job_url       TEXT,
            matched_company       TEXT,
            matched_method        TEXT,
            snippet               TEXT,
            created_at            TEXT NOT NULL,
            storage_version       INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(matched_job_url) REFERENCES jobs(url)
        )
    """)
    inbox_event_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(inbox_events)").fetchall()
    }
    if "storage_version" not in inbox_event_columns:
        conn.execute(
            "ALTER TABLE inbox_events "
            "ADD COLUMN storage_version INTEGER NOT NULL DEFAULT 0"
        )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS auth_challenges (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            job_url               TEXT NOT NULL,
            application_url       TEXT,
            provider              TEXT,
            challenge_type        TEXT NOT NULL,
            status                TEXT NOT NULL,
            requested_at          TEXT NOT NULL,
            expires_at            TEXT NOT NULL,
            resolved_at           TEXT,
            inbox_event_id        INTEGER,
            attempt_count         INTEGER NOT NULL DEFAULT 0,
            last_error            TEXT,
            created_at            TEXT NOT NULL,
            updated_at            TEXT NOT NULL,
            FOREIGN KEY(job_url) REFERENCES jobs(url),
            FOREIGN KEY(inbox_event_id) REFERENCES inbox_events(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_inbox_events_job_url ON inbox_events(matched_job_url)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_inbox_events_received_at ON inbox_events(received_at)")
    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_inbox_events_storage_outdated
        ON inbox_events(id)
        WHERE {OUTDATED_STORAGE_PREDICATE}
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_challenges_status ON auth_challenges(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_challenges_job_url ON auth_challenges(job_url)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_challenges_expires_at ON auth_challenges(expires_at)")
    migration_now = datetime.now(timezone.utc).isoformat()
    claim_index_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
        ("idx_auth_challenges_inbox_event_unique",),
    ).fetchone()
    if claim_index_exists is None:
        conn.execute("""
            UPDATE auth_challenges
               SET status = 'failed',
                   resolved_at = NULL,
                   inbox_event_id = NULL,
                   last_error = 'message_claim_conflict',
                   updated_at = ?
             WHERE inbox_event_id IS NOT NULL
               AND id NOT IN (
                   SELECT MIN(id)
                     FROM auth_challenges
                    WHERE inbox_event_id IS NOT NULL
                    GROUP BY inbox_event_id
               )
        """, (migration_now,))
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_auth_challenges_inbox_event_unique
        ON auth_challenges(inbox_event_id)
        WHERE inbox_event_id IS NOT NULL
    """)
    scrub_inbox_events(conn, migration_now=migration_now)
    conn.commit()


def ensure_outcome_tables(conn: sqlite3.Connection | None = None) -> None:
    """Create the email-driven outcome-tracking table (outcomes tracker spec
    2026-06-29). Distinct from inbox_events (which tracks apply-time auth/OTP
    challenges): email_events is the evidence layer for application OUTCOMES --
    one row per recruiter email, idempotent by Gmail message_id. ADDITIVE."""
    if conn is None:
        conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS email_events (
            message_id     TEXT PRIMARY KEY,
            thread_id      TEXT,
            job_url        TEXT,
            occurred_at    TEXT NOT NULL,
            sender         TEXT,
            sender_domain  TEXT,
            subject        TEXT,
            stage          TEXT NOT NULL,
            outcome        TEXT,
            reason         TEXT,
            title          TEXT,
            company        TEXT,
            match_method   TEXT,
            match_score    REAL,
            confidence     TEXT,
            body_text      TEXT,
            snippet        TEXT,
            extracted_by   TEXT,
            scanned_at     TEXT NOT NULL,
            FOREIGN KEY(job_url) REFERENCES jobs(url)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_email_events_job ON email_events(job_url)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_email_events_occurred ON email_events(occurred_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_email_events_stage ON email_events(stage)")

    # Match-quarantine tier (outcome-loop-integrity Task 3). ADDITIVE + idempotent.
    ee_existing = {row[1] for row in conn.execute("PRAGMA table_info(email_events)").fetchall()}
    for col in ("match_status", "match_reason", "prev_job_url"):
        if col not in ee_existing:
            conn.execute(f"ALTER TABLE email_events ADD COLUMN {col} TEXT")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS email_event_reviews (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id           TEXT NOT NULL,
            review_action        TEXT NOT NULL,
            reviewed_by          TEXT,
            reviewed_at          TEXT NOT NULL,
            corrected_job_url    TEXT,
            corrected_stage      TEXT,
            corrected_outcome    TEXT,
            corrected_confidence TEXT,
            resolution           TEXT NOT NULL CHECK(
                resolution IN ('trusted','needs_review','ignored','corrected')
            ),
            note                 TEXT,
            FOREIGN KEY(message_id) REFERENCES email_events(message_id)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_email_event_reviews_message_id "
        "ON email_event_reviews(message_id, id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_email_event_reviews_resolution "
        "ON email_event_reviews(resolution)"
    )

    conn.commit()


def ensure_canonical_decision_tables(conn: sqlite3.Connection | None = None) -> None:
    """Create additive policy, immutable decision, and reviewed-outcome tables."""
    if conn is None:
        conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS decision_policy_versions (
            policy_version       TEXT PRIMARY KEY,
            lane                 TEXT NOT NULL CHECK(lane IN ('ats', 'linkedin')),
            status               TEXT NOT NULL CHECK(status IN ('draft', 'validated', 'canary', 'active', 'retired')),
            qualification_model  TEXT,
            preference_model     TEXT,
            outcome_model        TEXT,
            kg_version           TEXT,
            label_snapshot       TEXT,
            pairwise_snapshot    TEXT,
            outcome_snapshot     TEXT,
            config_json          TEXT,
            metrics_json         TEXT,
            created_at           TEXT NOT NULL,
            validated_at         TEXT,
            activated_at         TEXT,
            retired_at           TEXT,
            UNIQUE(policy_version, lane)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS job_decisions (
            decision_id            TEXT PRIMARY KEY,
            job_url                TEXT NOT NULL,
            policy_version         TEXT NOT NULL,
            lane                   TEXT NOT NULL CHECK(lane IN ('ats', 'linkedin')),
            qualification_score    REAL,
            preference_score       REAL,
            outcome_score          REAL,
            final_score            REAL,
            qualification_verdict  TEXT NOT NULL CHECK(
                qualification_verdict IN ('qualified', 'unqualified', 'uncertain')
            ),
            action                 TEXT NOT NULL CHECK(action IN ('apply', 'review', 'reject')),
            confidence             REAL,
            uncertainty_json       TEXT,
            blockers_json          TEXT,
            requirements_json      TEXT,
            evidence_node_ids_json TEXT,
            title_signals_json      TEXT,
            explanation            TEXT,
            decision_snapshot_json  TEXT,
            input_hash             TEXT NOT NULL,
            created_at             TEXT NOT NULL,
            expires_at             TEXT,
            UNIQUE(job_url, policy_version, input_hash),
            FOREIGN KEY(job_url) REFERENCES jobs(url),
            FOREIGN KEY(policy_version, lane)
                REFERENCES decision_policy_versions(policy_version, lane)
        )
    """)
    job_decision_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(job_decisions)").fetchall()
    }
    if "decision_snapshot_json" not in job_decision_columns:
        try:
            conn.execute("ALTER TABLE job_decisions ADD COLUMN decision_snapshot_json TEXT")
        except sqlite3.OperationalError:
            # Another startup process can complete the same additive migration
            # after our PRAGMA snapshot but before this ALTER obtains its lock.
            # Suppress only that successfully completed race; every other DDL
            # failure remains fatal.
            concurrent_columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(job_decisions)").fetchall()
            }
            if "decision_snapshot_json" not in concurrent_columns:
                raise
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviewed_outcomes (
            event_id          TEXT NOT NULL,
            job_url           TEXT NOT NULL,
            attribution_json  TEXT,
            review_status     TEXT NOT NULL CHECK(
                review_status IN ('accepted', 'rejected', 'needs_review')
            ),
            normalized_stage  TEXT,
            weight            REAL,
            reviewer          TEXT,
            reason            TEXT,
            created_at        TEXT NOT NULL,
            reviewed_at       TEXT,
            updated_at        TEXT,
            PRIMARY KEY(event_id, job_url),
            FOREIGN KEY(event_id) REFERENCES email_events(message_id),
            FOREIGN KEY(job_url) REFERENCES jobs(url)
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_decision_policy_versions_status_lane
        ON decision_policy_versions(status, lane)
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_decision_policy_versions_version_lane
        ON decision_policy_versions(policy_version, lane)
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_decision_policy_versions_active_lane
        ON decision_policy_versions(lane) WHERE status = 'active'
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_job_decisions_job
        ON job_decisions(job_url, created_at)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_job_decisions_policy
        ON job_decisions(policy_version)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_job_decisions_action
        ON job_decisions(action)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_job_decisions_expires
        ON job_decisions(expires_at)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_reviewed_outcomes_job
        ON reviewed_outcomes(job_url)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_reviewed_outcomes_status
        ON reviewed_outcomes(review_status)
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_reviewed_outcomes_event
        ON reviewed_outcomes(event_id)
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_job_decisions_policy_lane_insert
        BEFORE INSERT ON job_decisions
        FOR EACH ROW
        WHEN NOT EXISTS (
            SELECT 1
            FROM decision_policy_versions
            WHERE policy_version = NEW.policy_version
              AND lane = NEW.lane
        )
        BEGIN
            SELECT RAISE(ABORT, 'decision policy lane mismatch');
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_job_decisions_policy_lane_update
        BEFORE UPDATE OF policy_version, lane ON job_decisions
        FOR EACH ROW
        WHEN NOT EXISTS (
            SELECT 1
            FROM decision_policy_versions
            WHERE policy_version = NEW.policy_version
              AND lane = NEW.lane
        )
        BEGIN
            SELECT RAISE(ABORT, 'decision policy lane mismatch');
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_decision_policy_versions_lane_update
        BEFORE UPDATE OF lane ON decision_policy_versions
        FOR EACH ROW
        WHEN EXISTS (
            SELECT 1
            FROM job_decisions
            WHERE policy_version = OLD.policy_version
              AND lane <> NEW.lane
        )
        BEGIN
            SELECT RAISE(ABORT, 'policy lane has decisions');
        END
    """)
    conn.commit()

    violations: list[str] = []
    for table in ("job_decisions", "reviewed_outcomes"):
        for row in conn.execute(f"PRAGMA foreign_key_check({table})").fetchall():
            violations.append(
                f"foreign key violation: {row[0]} rowid={row[1]} "
                f"references {row[2]} (fk={row[3]})"
            )

    lane_mismatches = conn.execute("""
        SELECT d.decision_id, d.policy_version, d.lane, p.lane
        FROM job_decisions d
        JOIN decision_policy_versions p ON p.policy_version = d.policy_version
        WHERE d.lane <> p.lane
        ORDER BY d.decision_id
    """).fetchall()
    for row in lane_mismatches:
        violations.append(
            f"lane mismatch: decision_id={row[0]} policy_version={row[1]} "
            f"decision_lane={row[2]} policy_lane={row[3]}"
        )

    if violations:
        raise RuntimeError(
            "Canonical decision schema integrity check failed: " + "; ".join(violations)
        )


def ensure_tenant_tables(conn: sqlite3.Connection | None = None) -> None:
    """Create the ats_tenants registry (auth-gated-tenant-lane Task 1). ADDITIVE +
    idempotent. Tracks per-host (e.g. a Workday tenant subdomain) rollout status for
    login-gated ATS lanes: excluded (never attempted) -> supervised (human-in-the-loop
    submits) -> trusted (autonomous submits allowed once evidence clears the bar).
    Stores NO credentials/secrets -- status + counters only."""
    if conn is None:
        conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ats_tenants (
            host             TEXT PRIMARY KEY,
            status           TEXT NOT NULL DEFAULT 'excluded',
            clean_submits    INTEGER NOT NULL DEFAULT 0,
            failed_submits   INTEGER NOT NULL DEFAULT 0,
            daily_cap        INTEGER NOT NULL DEFAULT 5,
            halted_until     TEXT,
            last_result      TEXT,
            updated_at       TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ats_tenants_status ON ats_tenants(status)")
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(ats_tenants)").fetchall()}
    for column, definition in (
        ("profile_id", "TEXT"),
        ("session_state", "TEXT NOT NULL DEFAULT 'supervised'"),
        ("session_checked_at", "TEXT"),
        ("session_expires_at", "TEXT"),
        ("session_reason", "TEXT"),
    ):
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE ats_tenants ADD COLUMN {column} {definition}")

    conn.commit()


def ensure_research_tables(conn: sqlite3.Connection | None = None) -> None:
    """Create the research layer the TS KG scorer writes into (unified-brain pipeline,
    spec 2026-06-25-unified-brain-pipeline-design §1). ADDITIVE + idempotent. Python OWNS this
    schema; the TS `brainDb.ts` layer only INSERTs into these tables, never issues DDL. ALL of
    this is ADVISORY -- it never changes the apply gate unless promoted via import-decisions.

    research_labels is a ONE-WAY sink fed by both the JSONL logs and Supabase label_events
    (deduped by event id; §9.2). KG artifacts/runs version the graph each score was computed
    against (§5). Keyed to jobs.url."""
    if conn is None:
        conn = get_connection()

    # KG artifact + provenance first (research_scores.kg_version references them).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research_kg_artifacts (
            kg_version            TEXT PRIMARY KEY,
            compact_kg_json       BLOB,
            built_at              TEXT,
            input_label_count     INTEGER,
            inputs_sha            TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research_kg_runs (
            kg_version            TEXT PRIMARY KEY,
            built_at              TEXT,
            resume_path           TEXT,
            resume_sha            TEXT,
            n_label_events        INTEGER,
            n_capabilities        INTEGER,
            compact_kg_path       TEXT,
            source                TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research_scores (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            job_url                TEXT NOT NULL,
            item_id                TEXT,
            provider               TEXT,
            model                  TEXT,
            research_fit_score     REAL,
            research_decision      TEXT,
            confidence             TEXT,
            reason                 TEXT,
            positive_signals_json  TEXT,
            gaps_json              TEXT,
            evidence_node_ids_json TEXT,
            score_source           TEXT,
            raw_fit_score          REAL,
            kg_version             TEXT,
            scored_at              TEXT,
            ingested_at            TEXT,
            UNIQUE(job_url, provider, model, scored_at),
            FOREIGN KEY(job_url) REFERENCES jobs(url)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research_labels (
            id                    TEXT PRIMARY KEY,
            job_url               TEXT,
            item_id               TEXT,
            source_project_id     TEXT,
            decision              TEXT,
            rating                INTEGER,
            reason                TEXT,
            cleaned_reason        TEXT,
            tags_json             TEXT,
            method                TEXT,
            fit_map_feedback_json TEXT,
            review_queue_json     TEXT,
            item_status_at_review TEXT,
            created_at            TEXT,
            raw_event_json        TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research_pairwise_labels (
            id                    TEXT PRIMARY KEY,
            left_job_url          TEXT,
            right_job_url         TEXT,
            left_item_id          TEXT,
            right_item_id         TEXT,
            winner                TEXT,
            method                TEXT,
            source_project_id     TEXT,
            created_at            TEXT,
            raw_event_json        TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research_label_confidence (
            label_id      TEXT PRIMARY KEY,
            item_id       TEXT,
            weight        REAL NOT NULL CHECK(weight > 0 AND weight <= 1),
            item_flip_rate REAL NOT NULL CHECK(item_flip_rate >= 0 AND item_flip_rate <= 1),
            method        TEXT NOT NULL,
            imported_at   TEXT NOT NULL,
            raw_json      TEXT NOT NULL,
            FOREIGN KEY(label_id) REFERENCES research_labels(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_research_scores_job_url ON research_scores(job_url)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_research_scores_model ON research_scores(model)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_research_scores_scored_at ON research_scores(scored_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_research_labels_job_url ON research_labels(job_url)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_research_pairwise_left ON research_pairwise_labels(left_job_url)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_research_pairwise_right ON research_pairwise_labels(right_job_url)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_research_label_confidence_method ON research_label_confidence(method)")
    conn.commit()


def ensure_desc_quality_tables(conn: sqlite3.Connection | None = None) -> None:
    """Create durable parse-quality drift/audit tables."""
    if conn is None:
        conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS desc_quality_drift (
            snapshot_at        TEXT NOT NULL,
            board              TEXT NOT NULL,
            window_days        INTEGER NOT NULL,
            total              INTEGER NOT NULL,
            null_rate          REAL NOT NULL,
            stub_rate          REAL NOT NULL,
            short_rate         REAL NOT NULL,
            html_rate          REAL NOT NULL,
            junk_rate          REAL NOT NULL,
            board_summary_rate  REAL NOT NULL,
            title_echo_rate    REAL NOT NULL,
            no_req_marker_rate REAL NOT NULL,
            PRIMARY KEY(snapshot_at, board)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS parse_spot_audit (
            audited_at  TEXT NOT NULL,
            url         TEXT NOT NULL,
            board       TEXT NOT NULL,
            band        TEXT NOT NULL,
            complete    INTEGER NOT NULL,
            defects     TEXT,
            model       TEXT,
            PRIMARY KEY (audited_at, url)
        )
    """)
    conn.commit()


def refresh_desc_quality(conn: sqlite3.Connection | None = None, limit: int | None = None) -> dict[str, int]:
    """Recompute desc_quality_* where stale or missing.

    Returns counters in a stable shape for tests and diagnostics:
    {"scanned": N, "updated": M}.
    """
    if conn is None:
        conn = get_connection()

    where = (
        "WHERE duplicate_of_url IS NULL "
        "AND (desc_quality_at IS NULL OR desc_quality_at < COALESCE(detail_scraped_at, discovered_at))"
    )
    query = (
        "SELECT url, title, full_description, COALESCE(detail_scraped_at, discovered_at) AS source_ts "
        f"FROM jobs {where}"
    )
    if limit:
        query += " LIMIT ?"
    rows = conn.execute(query, () if not limit else (limit,)).fetchall()

    updates = 0
    for row in rows:
        row_dict = {
            "url": row[0],
            "title": row[1],
            "full_description": row[2],
        }
        desc_flags, desc_score = _ensure_desc_quality(row_dict)
        source_ts = row["source_ts"] or datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            UPDATE jobs
               SET desc_quality_flags = ?,
                   desc_quality_score = ?,
                   desc_quality_at = ?
             WHERE url = ?
            """,
            (desc_flags, int(desc_score), source_ts, row[0]),
        )
        updates += 1
    conn.commit()
    return {"scanned": len(rows), "updated": updates}


def snapshot_desc_quality(conn: sqlite3.Connection | None = None, window_days: int = 7) -> dict[str, int | list[dict]]:
    """Compute rolling-window parse quality rates per board and persist a snapshot."""
    if conn is None:
        conn = get_connection()

    cutoff = datetime.now(timezone.utc).astimezone().isoformat()
    if window_days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()

    rows = conn.execute(
        """
        SELECT
            COALESCE(source_board, strategy, site, 'unknown') AS board,
            title,
            full_description,
            desc_quality_flags
        FROM jobs
        WHERE duplicate_of_url IS NULL
          AND discovered_at IS NOT NULL
          AND datetime(discovered_at) >= datetime(?)
        """,
        (cutoff,),
    ).fetchall()

    by_board: dict[str, dict[str, float | int]] = {}
    for board, title, full_description, flags in rows:
        stats = by_board.setdefault(
            board,
            {"total": 0, "empty": 0, "stub": 0, "short": 0, "html": 0, "junk": 0,
             "board_summary": 0, "title_echo": 0, "no_req_marker": 0},
        )
        flags_iter = _flag_set(title, full_description, flags)
        stats["total"] += 1
        if "empty" in flags_iter:
            stats["empty"] += 1
        if "stub_lt200" in flags_iter:
            stats["stub"] += 1
        if "short_lt500" in flags_iter:
            stats["short"] += 1
        if "html_residue" in flags_iter:
            stats["html"] += 1
        if "junk_boilerplate" in flags_iter:
            stats["junk"] += 1
        if "board_summary_stub" in flags_iter:
            stats["board_summary"] += 1
        if "title_echo" in flags_iter:
            stats["title_echo"] += 1
        if "no_requirements_marker" in flags_iter:
            stats["no_req_marker"] += 1

    snapshot_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    report_rows: list[dict] = []
    for board, s in by_board.items():
        total = int(s["total"])
        if total <= 0:
            continue
        total_f = float(total)
        rec = {
            "board": board,
            "total": total,
            "null_rate": round((s["empty"] or 0) / total_f, 4),
            "stub_rate": round((s["stub"] or 0) / total_f, 4),
            "short_rate": round((s["short"] or 0) / total_f, 4),
            "html_rate": round((s["html"] or 0) / total_f, 4),
            "junk_rate": round((s["junk"] or 0) / total_f, 4),
            "board_summary_rate": round((s["board_summary"] or 0) / total_f, 4),
            "title_echo_rate": round((s["title_echo"] or 0) / total_f, 4),
            "no_req_marker_rate": round((s["no_req_marker"] or 0) / total_f, 4),
        }
        conn.execute(
            """
            INSERT INTO desc_quality_drift (
                snapshot_at, board, window_days, total,
                null_rate, stub_rate, short_rate, html_rate, junk_rate,
                board_summary_rate, title_echo_rate, no_req_marker_rate
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_at, board) DO UPDATE SET
                total = excluded.total,
                null_rate = excluded.null_rate,
                stub_rate = excluded.stub_rate,
                short_rate = excluded.short_rate,
                html_rate = excluded.html_rate,
                junk_rate = excluded.junk_rate,
                board_summary_rate = excluded.board_summary_rate,
                title_echo_rate = excluded.title_echo_rate,
                no_req_marker_rate = excluded.no_req_marker_rate,
                window_days = excluded.window_days
            """,
            (
                snapshot_at,
                board,
                window_days,
                total,
                rec["null_rate"],
                rec["stub_rate"],
                rec["short_rate"],
                rec["html_rate"],
                rec["junk_rate"],
                rec["board_summary_rate"],
                rec["title_echo_rate"],
                rec["no_req_marker_rate"],
            ),
        )
        report_rows.append(rec)

    total_all = 0
    aggregate = {"empty": 0.0, "stub": 0.0, "short": 0.0, "html": 0.0, "junk": 0.0,
                 "board_summary": 0.0, "title_echo": 0.0, "no_req_marker": 0.0}
    for s in by_board.values():
        board_total = int(s["total"])
        if board_total <= 0:
            continue
        total_all += board_total
        aggregate["empty"] += float(s["empty"] or 0)
        aggregate["stub"] += float(s["stub"] or 0)
        aggregate["short"] += float(s["short"] or 0)
        aggregate["html"] += float(s["html"] or 0)
        aggregate["junk"] += float(s["junk"] or 0)
        aggregate["board_summary"] += float(s["board_summary"] or 0)
        aggregate["title_echo"] += float(s["title_echo"] or 0)
        aggregate["no_req_marker"] += float(s["no_req_marker"] or 0)
    all_board = {
        "board": "__all__",
        "total": total_all,
        "null_rate": 0.0,
        "stub_rate": 0.0,
        "short_rate": 0.0,
        "html_rate": 0.0,
        "junk_rate": 0.0,
        "board_summary_rate": 0.0,
        "title_echo_rate": 0.0,
        "no_req_marker_rate": 0.0,
    }
    if total_all:
        all_board["null_rate"] = round(aggregate["empty"] / total_all, 4)
        all_board["stub_rate"] = round(aggregate["stub"] / total_all, 4)
        all_board["short_rate"] = round(aggregate["short"] / total_all, 4)
        all_board["html_rate"] = round(aggregate["html"] / total_all, 4)
        all_board["junk_rate"] = round(aggregate["junk"] / total_all, 4)
        all_board["board_summary_rate"] = round(aggregate["board_summary"] / total_all, 4)
        all_board["title_echo_rate"] = round(aggregate["title_echo"] / total_all, 4)
        all_board["no_req_marker_rate"] = round(aggregate["no_req_marker"] / total_all, 4)

    conn.execute(
        """
        INSERT INTO desc_quality_drift (
            snapshot_at, board, window_days, total,
            null_rate, stub_rate, short_rate, html_rate, junk_rate,
            board_summary_rate, title_echo_rate, no_req_marker_rate
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(snapshot_at, board) DO UPDATE SET
            total = excluded.total,
            null_rate = excluded.null_rate,
            stub_rate = excluded.stub_rate,
            short_rate = excluded.short_rate,
            html_rate = excluded.html_rate,
            junk_rate = excluded.junk_rate,
            board_summary_rate = excluded.board_summary_rate,
            title_echo_rate = excluded.title_echo_rate,
            no_req_marker_rate = excluded.no_req_marker_rate,
            window_days = excluded.window_days
        """,
        (
            snapshot_at,
            "__all__",
            window_days,
            all_board["total"],
            all_board["null_rate"],
            all_board["stub_rate"],
            all_board["short_rate"],
            all_board["html_rate"],
            all_board["junk_rate"],
            all_board["board_summary_rate"],
            all_board["title_echo_rate"],
            all_board["no_req_marker_rate"],
        ),
    )

    report_rows.append(all_board)
    conn.commit()
    return {"snapshot_at": snapshot_at, "rows": report_rows}


def _flag_set(title: str | None, full_description: str | None, raw_flags: str | None) -> set[str]:
    """Return a deterministic set of quality flags, recomputing if needed."""
    if raw_flags is not None:
        return {f for f in str(raw_flags).split(",") if f}
    flags, _ = _ensure_desc_quality({"title": title, "full_description": full_description})
    return {f for f in flags.split(",") if f}


def ensure_pipeline_tables(conn: sqlite3.Connection | None = None) -> None:
    """Create durable pipeline run/checkpoint tables."""
    if conn is None:
        conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            stages                TEXT NOT NULL,
            mode                  TEXT NOT NULL,
            status                TEXT NOT NULL,
            min_score             INTEGER,
            batch_size            INTEGER,
            workers               INTEGER,
            validation_mode       TEXT,
            started_at            TEXT NOT NULL,
            ended_at              TEXT,
            error                 TEXT,
            created_at            TEXT NOT NULL,
            updated_at            TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_stage_runs (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id                INTEGER NOT NULL,
            stage                 TEXT NOT NULL,
            status                TEXT NOT NULL,
            pending_before        INTEGER,
            pending_after         INTEGER,
            elapsed_seconds       REAL,
            started_at            TEXT NOT NULL,
            ended_at              TEXT,
            error                 TEXT,
            created_at            TEXT NOT NULL,
            updated_at            TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES pipeline_runs(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_usage (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            stage                 TEXT,
            model                 TEXT,
            provider              TEXT,
            prompt_tokens         INTEGER DEFAULT 0,
            completion_tokens     INTEGER DEFAULT 0,
            thinking_tokens       INTEGER DEFAULT 0,
            total_tokens          INTEGER DEFAULT 0,
            est_cost_usd          REAL,
            prompt_variant        TEXT,
            created_at            TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scoring_context_events (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            event                 TEXT NOT NULL,
            detail                TEXT,
            created_at            TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_runs_started_at ON pipeline_runs(started_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON pipeline_runs(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_stage_runs_run_id ON pipeline_stage_runs(run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_stage_runs_stage ON pipeline_stage_runs(stage)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_stage ON llm_usage(stage)")

    # Forward-migrate prompt_variant into existing llm_usage tables BEFORE
    # indexing it: an older DB's llm_usage predates this column, so creating the
    # index first throws "no such column: prompt_variant" and aborts init_db.
    existing_llm_cols = {row[1] for row in conn.execute("PRAGMA table_info(llm_usage)").fetchall()}
    if "prompt_variant" not in existing_llm_cols:
        conn.execute("ALTER TABLE llm_usage ADD COLUMN prompt_variant TEXT")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_variant ON llm_usage(prompt_variant)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scoring_context_events_event ON scoring_context_events(event)")

    conn.commit()


def record_llm_usage(
    stage: str | None,
    model: str | None,
    provider: str | None,
    usage: dict | None,
    est_cost_usd: float | None = None,
    conn: sqlite3.Connection | None = None,
    prompt_variant: str | None = None,
) -> None:
    """Persist one LLM call's token usage (best-effort; never raises).

    `usage` is the LLMClient.last_usage dict (prompt/completion/thinking/total).
    `prompt_variant` identifies which A/B prompt template was used (MAB tracking).
    """
    if not usage:
        return
    if conn is None:
        conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO llm_usage (stage, model, provider, prompt_tokens,
                completion_tokens, thinking_tokens, total_tokens, est_cost_usd,
                prompt_variant, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stage, model, provider,
                int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0),
                int(usage.get("thinking_tokens") or 0),
                int(usage.get("total_tokens") or 0),
                est_cost_usd,
                prompt_variant,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    except Exception:
        pass  # usage logging must never break a real LLM call


def record_scoring_context_event(
    event: str,
    detail: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Persist scorer context anomalies (best-effort; never raises)."""
    try:
        if conn is None:
            conn = get_connection()
        conn.execute(
            """
            INSERT INTO scoring_context_events (event, detail, created_at)
            VALUES (?, ?, ?)
            """,
            (event, detail, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    except Exception:
        pass


def get_llm_usage_summary(conn: sqlite3.Connection | None = None) -> dict:
    """Aggregate LLM token usage and estimated cost by stage, model, and prompt variant."""
    if conn is None:
        conn = get_connection()
    by_stage = [
        dict(zip(("stage", "calls", "prompt", "completion", "thinking", "total", "cost"), row))
        for row in conn.execute(
            """
            SELECT COALESCE(stage, '?'), COUNT(*), SUM(prompt_tokens), SUM(completion_tokens),
                   SUM(thinking_tokens), SUM(total_tokens), SUM(est_cost_usd)
              FROM llm_usage GROUP BY stage ORDER BY SUM(total_tokens) DESC
            """
        ).fetchall()
    ]
    by_model = [
        dict(zip(("model", "calls", "total", "cost"), row))
        for row in conn.execute(
            """
            SELECT COALESCE(model, '?'), COUNT(*), SUM(total_tokens), SUM(est_cost_usd)
              FROM llm_usage GROUP BY model ORDER BY SUM(total_tokens) DESC
            """
        ).fetchall()
    ]
    by_variant = [
        dict(zip(("variant", "stage", "calls", "total", "cost"), row))
        for row in conn.execute(
            """
            SELECT COALESCE(prompt_variant, '(default)'), COALESCE(stage, '?'),
                   COUNT(*), SUM(total_tokens), SUM(est_cost_usd)
              FROM llm_usage
             WHERE prompt_variant IS NOT NULL
             GROUP BY prompt_variant, stage
             ORDER BY prompt_variant, SUM(total_tokens) DESC
            """
        ).fetchall()
    ]
    totals = conn.execute(
        "SELECT COUNT(*), SUM(total_tokens), SUM(est_cost_usd) FROM llm_usage"
    ).fetchone()
    return {
        "by_stage": by_stage,
        "by_model": by_model,
        "by_variant": by_variant,
        "total_calls": totals[0] or 0,
        "total_tokens": totals[1] or 0,
        "total_cost_usd": totals[2] or 0.0,
    }


def get_scrape_quality_report(conn: sqlite3.Connection | None = None) -> dict:
    """Per-board null-rate report computed from the live jobs table.

    Null rate = fraction of non-duplicate jobs on that board that are missing
    at least one of: title, full_description, location, company.  High null
    rates indicate selector drift or anti-bot placeholder substitution.

    Implemented as a single GROUP BY query to avoid N+1 database round-trips.
    """
    if conn is None:
        conn = get_connection()

    rows = conn.execute(
        """
        SELECT
            COALESCE(source_board, strategy, site, 'unknown') AS board,
            COUNT(*)                                           AS total,
            SUM(CASE WHEN title IS NULL OR trim(title) = ''                         THEN 1 ELSE 0 END) AS null_title,
            SUM(CASE WHEN full_description IS NULL OR trim(full_description) = ''   THEN 1 ELSE 0 END) AS null_full_description,
            SUM(CASE WHEN location IS NULL OR trim(location) = ''                   THEN 1 ELSE 0 END) AS null_location,
            SUM(CASE WHEN company IS NULL OR trim(company) = ''                     THEN 1 ELSE 0 END) AS null_company,
            SUM(CASE WHEN
                    title IS NULL OR trim(title) = '' OR
                    full_description IS NULL OR trim(full_description) = '' OR
                    location IS NULL OR trim(location) = '' OR
                    company IS NULL OR trim(company) = ''
                THEN 1 ELSE 0 END)                           AS with_any_null
          FROM jobs
         WHERE duplicate_of_url IS NULL
         GROUP BY COALESCE(source_board, strategy, site, 'unknown')
        """
    ).fetchall()

    reports = []
    for row in rows:
        board, total, n_title, n_desc, n_loc, n_co, with_any_null = row
        if not total:
            continue
        reports.append({
            "board": board,
            "total": total,
            "null_rate": round(with_any_null / total, 4),
            "null_counts": {
                "title": n_title or 0,
                "full_description": n_desc or 0,
                "location": n_loc or 0,
                "company": n_co or 0,
            },
        })

    return {
        "boards": sorted(reports, key=lambda r: r["null_rate"], reverse=True),
    }


def get_apply_analytics(conn: sqlite3.Connection | None = None) -> dict:
    """Apply funnel + success-rate analytics from the jobs/applications tables."""
    if conn is None:
        conn = get_connection()

    funnel = {
        (row[0] or "(unattempted)"): row[1]
        for row in conn.execute(
            "SELECT apply_status, COUNT(*) FROM jobs WHERE duplicate_of_url IS NULL GROUP BY apply_status"
        ).fetchall()
    }
    applied = funnel.get("applied", 0)
    terminal_fail = sum(
        n for s, n in funnel.items()
        if s not in ("applied", "in_progress", "(unattempted)", "manual")
    )
    attempted = applied + terminal_fail
    success_rate = (applied / attempted) if attempted else None

    by_site = [
        dict(zip(("site", "applied", "failed"), row))
        for row in conn.execute(
            """
            SELECT site,
                   SUM(CASE WHEN apply_status = 'applied' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN apply_status IS NOT NULL AND apply_status NOT IN
                            ('applied', 'in_progress', 'manual') THEN 1 ELSE 0 END)
              FROM jobs
             WHERE apply_status IS NOT NULL AND apply_status != 'in_progress'
             GROUP BY site
             ORDER BY 2 DESC, 3 DESC
             LIMIT 15
            """
        ).fetchall()
    ]
    fail_reasons = [
        dict(zip(("reason", "count"), row))
        for row in conn.execute(
            """
            SELECT COALESCE(apply_error, 'unknown'), COUNT(*)
              FROM jobs
             WHERE apply_status IS NOT NULL
               AND apply_status NOT IN ('applied', 'in_progress', 'manual')
             GROUP BY apply_error ORDER BY 2 DESC LIMIT 12
            """
        ).fetchall()
    ]
    avg_ms = conn.execute(
        "SELECT AVG(apply_duration_ms) FROM jobs WHERE apply_duration_ms IS NOT NULL"
    ).fetchone()[0]

    # Outcome funnel from the application tracker (interview/offer/rejected/...)
    outcomes: dict[str, int] = {}
    try:
        outcomes = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT status, COUNT(*) FROM applications GROUP BY status"
            ).fetchall()
        }
    except sqlite3.OperationalError:
        pass  # applications table may not exist yet

    return {
        "funnel": funnel,
        "applied": applied,
        "attempted": attempted,
        "success_rate": success_rate,
        "by_site": by_site,
        "fail_reasons": fail_reasons,
        "avg_apply_seconds": (avg_ms / 1000.0) if avg_ms else None,
        "outcomes": outcomes,
    }


_COMPANY_SUFFIXES = {
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "llc",
    "llp",
    "lp",
    "ltd",
    "limited",
    "plc",
}
_STATE_NAMES = {
    "alabama": "al",
    "alaska": "ak",
    "arizona": "az",
    "arkansas": "ar",
    "california": "ca",
    "colorado": "co",
    "connecticut": "ct",
    "delaware": "de",
    "florida": "fl",
    "georgia": "ga",
    "hawaii": "hi",
    "idaho": "id",
    "illinois": "il",
    "indiana": "in",
    "iowa": "ia",
    "kansas": "ks",
    "kentucky": "ky",
    "louisiana": "la",
    "maine": "me",
    "maryland": "md",
    "massachusetts": "ma",
    "michigan": "mi",
    "minnesota": "mn",
    "mississippi": "ms",
    "missouri": "mo",
    "montana": "mt",
    "nebraska": "ne",
    "nevada": "nv",
    "new hampshire": "nh",
    "new jersey": "nj",
    "new mexico": "nm",
    "new york": "ny",
    "north carolina": "nc",
    "north dakota": "nd",
    "ohio": "oh",
    "oklahoma": "ok",
    "oregon": "or",
    "pennsylvania": "pa",
    "rhode island": "ri",
    "south carolina": "sc",
    "south dakota": "sd",
    "tennessee": "tn",
    "texas": "tx",
    "utah": "ut",
    "vermont": "vt",
    "virginia": "va",
    "washington": "wa",
    "west virginia": "wv",
    "wisconsin": "wi",
    "wyoming": "wy",
}
_LOCATION_ALIASES = {
    "nyc": "new york ny",
    "new york city": "new york ny",
    "new york ny": "new york ny",
    "sf": "san francisco ca",
    "bay area": "san francisco ca",
    "san francisco ca": "san francisco ca",
}
_DUPLICATE_REASON = "same_company_title_location"


def _fold_ascii(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def _normalize_tokens(value: Any) -> list[str]:
    text = _fold_ascii(value)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return [token for token in text.split() if token]


def _normalize_company(value: Any) -> str:
    tokens = _normalize_tokens(value)
    while tokens and tokens[-1] in _COMPANY_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _normalize_title(value: Any) -> str:
    return " ".join(_normalize_tokens(value))


def _normalize_location(value: Any) -> str:
    tokens = _normalize_tokens(value)
    if not tokens:
        return "unknown"
    if "remote" in tokens:
        return "remote"

    text = " ".join(token for token in tokens if token not in {"usa", "us", "united", "states"})
    if text in _LOCATION_ALIASES:
        return _LOCATION_ALIASES[text]

    for state_name, abbr in sorted(_STATE_NAMES.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(rf"\b{re.escape(state_name)}\b", abbr, text)
    text = _LOCATION_ALIASES.get(text, text)
    return re.sub(r"\s+", " ", text).strip() or "unknown"


def build_job_dedupe_key(job: Mapping[str, Any]) -> str | None:
    """Build a conservative same-job fingerprint from company, title, and location."""
    title = _normalize_title(job.get("title"))
    company = _normalize_company(job.get("company") or job.get("site"))
    if not title or not company:
        return None
    location = _normalize_location(job.get("location"))
    return f"v1|{company}|{title}|{location}"


def _row_quality(row: Mapping[str, Any]) -> tuple[int, int, str, str]:
    has_application = 0 if row.get("application_url") else 1
    has_description = 0 if row.get("full_description") else 1
    return has_application, has_description, str(row.get("discovered_at") or ""), str(row.get("url") or "")


def _canonical_duplicate(
    conn: sqlite3.Connection,
    dedupe_key: str,
    *,
    exclude_url: str | None = None,
) -> sqlite3.Row | None:
    params: list[Any] = [dedupe_key]
    where = "dedupe_key = ? AND duplicate_of_url IS NULL"
    if exclude_url:
        where += " AND url != ?"
        params.append(exclude_url)
    return conn.execute(
        f"""
        SELECT *
          FROM jobs
         WHERE {where}
         ORDER BY
              CASE WHEN application_url IS NULL OR application_url = '' THEN 1 ELSE 0 END,
              CASE WHEN full_description IS NULL OR full_description = '' THEN 1 ELSE 0 END,
              discovered_at,
              url
         LIMIT 1
        """,
        params,
    ).fetchone()


def insert_discovered_job(
    conn: sqlite3.Connection,
    job: Mapping[str, Any],
    *,
    site: str | None = None,
    strategy: str | None = None,
    source_board: str | None = None,
    discovered_at: str | None = None,
) -> str:
    """Insert one discovered job and mark same-job duplicates non-destructively.

    Returns one of: "new", "duplicate", "existing", or "skipped".
    """
    url = str(job.get("url") or "").strip()
    if not url:
        return "skipped"

    now = discovered_at or datetime.now(timezone.utc).isoformat()
    site_value = site or job.get("site") or job.get("company")
    company = job.get("company") or site_value
    board = source_board or job.get("source_board") or site_value
    dedupe_key = build_job_dedupe_key({**dict(job), "site": site_value, "company": company})
    canonical = _canonical_duplicate(conn, dedupe_key, exclude_url=url) if dedupe_key else None
    duplicate_of_url = canonical["url"] if canonical else None
    duplicate_reason = _DUPLICATE_REASON if canonical else None
    duplicate_detected_at = now if canonical else None

    desc_flags = None
    desc_score = None
    desc_quality_at = now
    if job.get("full_description") is not None:
        desc_flags, desc_score = _ensure_desc_quality(job)
        desc_quality_at = job.get("detail_scraped_at") or now

    try:
        conn.execute(
            """
            INSERT INTO jobs (
                url, title, salary, description, location, site,
                company, source_board, strategy, discovered_at,
                dedupe_key, duplicate_of_url, duplicate_reason, duplicate_detected_at,
                full_description, application_url, detail_scraped_at, detail_error,
                posted_at, valid_through,
                desc_quality_flags, desc_quality_score, desc_quality_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                url,
                job.get("title"),
                job.get("salary"),
                job.get("description"),
                job.get("location"),
                site_value,
                company,
                board,
                strategy or job.get("strategy"),
                now,
                dedupe_key,
                duplicate_of_url,
                duplicate_reason,
                duplicate_detected_at,
                job.get("full_description"),
                job.get("application_url"),
                job.get("detail_scraped_at"),
                job.get("detail_error"),
                job.get("posted_at"),
                job.get("valid_through"),
                desc_flags,
                desc_score,
                desc_quality_at,
            ),
        )
    except sqlite3.IntegrityError:
        if dedupe_key:
            conn.execute(
                "UPDATE jobs SET dedupe_key = COALESCE(dedupe_key, ?) WHERE url = ?",
                (dedupe_key, url),
            )
        return "existing"

    return "duplicate" if duplicate_of_url else "new"


def dedupe_existing_jobs(conn: sqlite3.Connection | None = None, *, dry_run: bool = False) -> dict[str, int | bool]:
    """Recompute dedupe keys and mark same-job duplicates already in the DB."""
    if conn is None:
        conn = get_connection()

    rows = [dict(row) for row in conn.execute("SELECT * FROM jobs ORDER BY discovered_at, url").fetchall()]
    desired: dict[str, dict[str, str | None]] = {}
    groups: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        key = build_job_dedupe_key(row)
        desired[row["url"]] = {
            "dedupe_key": key,
            "duplicate_of_url": None,
            "duplicate_reason": None,
            "duplicate_detected_at": None,
        }
        if key:
            groups.setdefault(key, []).append(row)

    duplicate_groups = 0
    duplicate_count = 0
    now = datetime.now(timezone.utc).isoformat()

    for key, group in groups.items():
        if len(group) <= 1:
            continue
        duplicate_groups += 1
        canonical = sorted(group, key=_row_quality)[0]
        for row in group:
            if row["url"] == canonical["url"]:
                continue
            desired[row["url"]] = {
                "dedupe_key": key,
                "duplicate_of_url": canonical["url"],
                "duplicate_reason": _DUPLICATE_REASON,
                "duplicate_detected_at": row.get("duplicate_detected_at") or now,
            }
            duplicate_count += 1

    if not dry_run:
        for url, values in desired.items():
            conn.execute(
                """
                UPDATE jobs
                   SET dedupe_key = ?,
                       duplicate_of_url = ?,
                       duplicate_reason = ?,
                       duplicate_detected_at = ?
                 WHERE url = ?
                """,
                (
                    values["dedupe_key"],
                    values["duplicate_of_url"],
                    values["duplicate_reason"],
                    values["duplicate_detected_at"],
                    url,
                ),
            )
        conn.commit()

    return {
        "processed": len(rows),
        "keys": sum(1 for values in desired.values() if values["dedupe_key"]),
        "duplicates": duplicate_count,
        "groups": duplicate_groups,
        "dry_run": dry_run,
    }


def create_pipeline_run(
    conn: sqlite3.Connection | None = None,
    *,
    stages: list[str],
    mode: str,
    min_score: int,
    batch_size: int | None,
    workers: int,
    validation_mode: str,
) -> int:
    """Record a new pipeline run and return its id."""
    if conn is None:
        conn = get_connection()
    ensure_pipeline_tables(conn)

    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """
        INSERT INTO pipeline_runs (
            stages, mode, status, min_score, batch_size, workers,
            validation_mode, started_at, created_at, updated_at
        )
        VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ",".join(stages),
            mode,
            min_score,
            batch_size,
            workers,
            validation_mode,
            now,
            now,
            now,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def finish_pipeline_run(
    conn: sqlite3.Connection | None,
    run_id: int,
    *,
    status: str,
    error: str | None = None,
) -> None:
    """Mark a pipeline run complete, partial, or failed."""
    if conn is None:
        conn = get_connection()

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        UPDATE pipeline_runs
           SET status = ?,
               ended_at = ?,
               error = ?,
               updated_at = ?
         WHERE id = ?
        """,
        (status, now, error, now, run_id),
    )
    conn.commit()


def start_pipeline_stage(
    conn: sqlite3.Connection | None,
    run_id: int,
    stage: str,
    pending_before: int | None = None,
) -> int:
    """Record the start of one pipeline stage and return its stage-run id."""
    if conn is None:
        conn = get_connection()

    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """
        INSERT INTO pipeline_stage_runs (
            run_id, stage, status, pending_before,
            started_at, created_at, updated_at
        )
        VALUES (?, ?, 'running', ?, ?, ?, ?)
        """,
        (run_id, stage, pending_before, now, now, now),
    )
    conn.commit()
    return int(cursor.lastrowid)


def finish_pipeline_stage(
    conn: sqlite3.Connection | None,
    stage_run_id: int,
    *,
    status: str,
    pending_after: int | None = None,
    elapsed_seconds: float | None = None,
    error: str | None = None,
) -> None:
    """Record the end state for one pipeline stage."""
    if conn is None:
        conn = get_connection()

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        UPDATE pipeline_stage_runs
           SET status = ?,
               pending_after = ?,
               elapsed_seconds = ?,
               ended_at = ?,
               error = ?,
               updated_at = ?
         WHERE id = ?
        """,
        (status, pending_after, elapsed_seconds, now, error, now, stage_run_id),
    )
    conn.commit()


def get_stats(conn: sqlite3.Connection | None = None) -> dict:
    """Return job counts by pipeline stage.

    Provides a snapshot of how many jobs are at each stage, useful for
    dashboard display and pipeline progress tracking.

    Args:
        conn: Database connection. Uses get_connection() if None.

    Returns:
        Dictionary with keys:
            total, by_site, pending_detail, with_description,
            scored, unscored, tailored, untailored_eligible,
            with_cover_letter, applied, score_distribution
    """
    if conn is None:
        conn = get_connection()

    stats: dict = {}

    # Total jobs
    stats["total"] = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    stats["duplicates"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE duplicate_of_url IS NOT NULL"
    ).fetchone()[0]
    stats["active_total"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE duplicate_of_url IS NULL"
    ).fetchone()[0]

    # By site breakdown
    rows = conn.execute(
        """
        SELECT COALESCE(source_board, strategy, site, 'Unknown') AS source, COUNT(*) as cnt
          FROM jobs
         WHERE duplicate_of_url IS NULL
         GROUP BY COALESCE(source_board, strategy, site, 'Unknown')
         ORDER BY cnt DESC
        """
    ).fetchall()
    stats["by_site"] = [(row[0], row[1]) for row in rows]

    # Enrichment stage
    stats["pending_detail"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE detail_scraped_at IS NULL AND duplicate_of_url IS NULL"
    ).fetchone()[0]

    stats["with_description"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE full_description IS NOT NULL AND duplicate_of_url IS NULL"
    ).fetchone()[0]

    stats["detail_errors"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE detail_error IS NOT NULL AND duplicate_of_url IS NULL"
    ).fetchone()[0]

    # Scoring stage
    stats["scored"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE fit_score IS NOT NULL AND duplicate_of_url IS NULL"
    ).fetchone()[0]

    stats["unscored"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE full_description IS NOT NULL AND fit_score IS NULL AND duplicate_of_url IS NULL"
    ).fetchone()[0]

    # Score distribution
    dist_rows = conn.execute(
        "SELECT fit_score, COUNT(*) as cnt FROM jobs "
        "WHERE fit_score IS NOT NULL AND duplicate_of_url IS NULL "
        "GROUP BY fit_score ORDER BY fit_score DESC"
    ).fetchall()
    stats["score_distribution"] = [(row[0], row[1]) for row in dist_rows]

    stats["audited"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE audited_at IS NOT NULL AND duplicate_of_url IS NULL"
    ).fetchone()[0]

    stats["recommended"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE audit_label IN ('priority', 'recommended') AND duplicate_of_url IS NULL"
    ).fetchone()[0]

    stats["audit_excluded"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE audit_label = 'exclude' AND duplicate_of_url IS NULL"
    ).fetchone()[0]

    stats["diagnosed"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE diagnosed_at IS NOT NULL AND duplicate_of_url IS NULL"
    ).fetchone()[0]

    stats["undiagnosed"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE fit_score IS NOT NULL AND full_description IS NOT NULL "
        "AND duplicate_of_url IS NULL "
        "AND (diagnosed_at IS NULL "
        "OR (scored_at IS NOT NULL AND diagnosed_at < scored_at) "
        "OR (audited_at IS NOT NULL AND diagnosed_at < audited_at))"
    ).fetchone()[0]

    # Tailoring stage
    stats["tailored"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL AND duplicate_of_url IS NULL"
    ).fetchone()[0]

    stats["untailored_eligible"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE COALESCE(audit_score, fit_score) >= 7 AND full_description IS NOT NULL "
        "AND duplicate_of_url IS NULL AND tailored_resume_path IS NULL"
    ).fetchone()[0]

    stats["tailor_exhausted"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE COALESCE(tailor_attempts, 0) >= 5 "
        "AND tailored_resume_path IS NULL AND duplicate_of_url IS NULL"
    ).fetchone()[0]

    # Cover letter stage
    stats["with_cover_letter"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE cover_letter_path IS NOT NULL AND duplicate_of_url IS NULL"
    ).fetchone()[0]

    stats["cover_exhausted"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE COALESCE(cover_attempts, 0) >= 5 "
        "AND duplicate_of_url IS NULL "
        "AND (cover_letter_path IS NULL OR cover_letter_path = '')"
    ).fetchone()[0]

    # Application stage
    stats["applied"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE applied_at IS NOT NULL AND duplicate_of_url IS NULL"
    ).fetchone()[0]

    stats["apply_errors"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE apply_error IS NOT NULL AND duplicate_of_url IS NULL"
    ).fetchone()[0]

    stats["ready_to_apply"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE tailored_resume_path IS NOT NULL "
        "AND duplicate_of_url IS NULL "
        "AND applied_at IS NULL "
        "AND (apply_status IS NULL OR apply_status = 'failed') "
        "AND COALESCE(audit_score, fit_score, 0) >= 7"
    ).fetchone()[0]

    stats["applications_tracked"] = conn.execute(
        "SELECT COUNT(*) FROM applications"
    ).fetchone()[0]

    stats["interviews"] = conn.execute(
        "SELECT COUNT(*) FROM applications "
        "WHERE status IN ('recruiter_screen', 'hiring_manager', 'assessment', 'final')"
    ).fetchone()[0]

    stats["rejections"] = conn.execute(
        "SELECT COUNT(*) FROM applications WHERE status = 'rejected'"
    ).fetchone()[0]

    today = datetime.now(timezone.utc).date().isoformat()
    stats["followups_due"] = conn.execute(
        "SELECT COUNT(*) FROM applications "
        "WHERE next_follow_up_at IS NOT NULL AND substr(next_follow_up_at, 1, 10) <= ? "
        "AND status NOT IN ('rejected', 'closed', 'offer', 'withdrawn')",
        (today,),
    ).fetchone()[0]

    return stats


def store_jobs(conn: sqlite3.Connection, jobs: list[dict],
               site: str, strategy: str) -> tuple[int, int]:
    """Store discovered jobs, skipping exact URLs and marking same-job duplicates.

    Args:
        conn: Database connection.
        jobs: List of job dicts with keys: url, title, salary, description, location.
        site: Source site name (e.g. "RemoteOK", "Dice").
        strategy: Extraction strategy used (e.g. "json_ld", "api_response", "css_selectors").

    Returns:
        Tuple of (new_count, duplicate_count).
    """
    new = 0
    existing = 0

    for job in jobs:
        status = insert_discovered_job(conn, job, site=job.get("company") or site, strategy=strategy)
        if status == "new":
            new += 1
        elif status in {"existing", "duplicate"}:
            existing += 1

    conn.commit()
    return new, existing


def get_jobs_by_stage(conn: sqlite3.Connection | None = None,
                      stage: str = "discovered",
                      min_score: int | None = None,
                      limit: int = 100) -> list[dict]:
    """Fetch jobs filtered by pipeline stage.

    Args:
        conn: Database connection. Uses get_connection() if None.
        stage: One of "discovered", "enriched", "scored", "tailored", "applied".
        min_score: Minimum fit_score filter (only relevant for scored+ stages).
        limit: Maximum number of rows to return.

    Returns:
        List of job dicts.
    """
    if conn is None:
        conn = get_connection()

    liveness_guard = llm_stage_liveness_sql()
    conditions = {
        "discovered": "1=1",
        "pending_detail": "detail_scraped_at IS NULL AND duplicate_of_url IS NULL",
        "enriched": "full_description IS NOT NULL AND duplicate_of_url IS NULL",
        "pending_score": (
            f"full_description IS NOT NULL AND LENGTH(full_description) >= {MIN_GOOD_DESCRIPTION_CHARS} "
            f"AND fit_score IS NULL AND duplicate_of_url IS NULL {liveness_guard}"
        ),
        "scored": "fit_score IS NOT NULL AND duplicate_of_url IS NULL",
        "pending_diagnosis": (
            "fit_score IS NOT NULL AND full_description IS NOT NULL "
            "AND duplicate_of_url IS NULL "
            "AND (diagnosed_at IS NULL "
            "OR (scored_at IS NOT NULL AND diagnosed_at < scored_at) "
            "OR (audited_at IS NOT NULL AND diagnosed_at < audited_at))"
        ),
        "pending_tailor": (
            "COALESCE(audit_score, fit_score) >= ? AND full_description IS NOT NULL "
            f"AND LENGTH(COALESCE(full_description,'')) >= {THIN_DESCRIPTION_CHARS} "
            "AND duplicate_of_url IS NULL "
            f"AND tailored_resume_path IS NULL AND COALESCE(tailor_attempts, 0) < 5 {liveness_guard}"
        ),
        "tailored": "tailored_resume_path IS NOT NULL AND duplicate_of_url IS NULL",
        "pending_apply": (
            "tailored_resume_path IS NOT NULL AND applied_at IS NULL "
            "AND duplicate_of_url IS NULL "
            "AND (apply_status IS NULL OR apply_status = 'failed') "
            f"AND LENGTH(COALESCE(full_description,'')) >= {THIN_DESCRIPTION_CHARS} "
            f"AND COALESCE(audit_score, fit_score, 0) >= 7 {liveness_guard}"
        ),
        "applied": "applied_at IS NOT NULL AND duplicate_of_url IS NULL",
    }

    where = conditions.get(stage, "1=1")
    params: list = []

    if "?" in where and min_score is not None:
        params.append(min_score)
    elif "?" in where:
        params.append(7)  # default min_score

    if min_score is not None and "fit_score" not in where and stage in ("scored", "tailored", "applied"):
        where += " AND COALESCE(audit_score, fit_score) >= ?"
        params.append(min_score)

    query = (
        f"SELECT * FROM jobs WHERE {where} "
        "ORDER BY COALESCE(audit_score, fit_score) DESC NULLS LAST, "
        "fit_score DESC NULLS LAST, discovered_at DESC"
    )
    if limit > 0:
        query += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(query, params).fetchall()

    # Convert sqlite3.Row objects to dicts
    if rows:
        columns = rows[0].keys()
        return [dict(zip(columns, row)) for row in rows]
    return []
