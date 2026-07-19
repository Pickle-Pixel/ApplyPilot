"""Import the canonical ApplyPilot SQLite brain into the Postgres brain.

The importer is intentionally explicit about source identity and lane state.
It is staging-only until its parity report is accepted. Rows without any
supported label endpoint are quarantined with their original evidence.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb
from psycopg.pq import TransactionStatus

from applypilot.brain.schema import ensure_policy_partition
from applypilot.brain.importer import SOURCE_TABLES
from applypilot.brain.policy_artifacts import ArtifactDescriptor, compile_policy_artifacts
from applypilot.brain.parity import ParityResult
from applypilot.brain.source_receipt import SQLiteSourceReceipt, capture_sqlite_source_receipt
from applypilot.brain.sqlite_source_guard import SQLiteSourceGuard


SOURCE_NAMESPACE = "applypilot-sqlite"
CONTROLLER_ROLE = "brain_schema_migrator"


class BrainImportError(RuntimeError):
    """The source or target failed the import contract."""


@dataclass(frozen=True, slots=True)
class ImportSummary:
    migration_run_id: int
    source_sha256: str
    imported: dict[str, int]
    quarantined: dict[str, int]
    finalized: bool = False
    parity: dict[str, dict[str, Any]] = field(default_factory=dict)
    terminal_event_id: int | None = None
    recovered: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "migration_run_id": self.migration_run_id,
            "source_sha256": self.source_sha256,
            "imported": dict(self.imported),
            "quarantined": dict(self.quarantined),
            "finalized": self.finalized,
            "parity": {name: dict(result) for name, result in self.parity.items()},
            "terminal_event_id": self.terminal_event_id,
            "recovered": self.recovered,
        }


def _job_id(url: str) -> str:
    return hashlib.sha256(f"{SOURCE_NAMESPACE}:job:{url}".encode()).hexdigest()


def _application_id(source_id: int) -> str:
    return hashlib.sha256(f"{SOURCE_NAMESPACE}:application:{source_id}".encode()).hexdigest()


def _json(value: Any, default: Any = None) -> Jsonb:
    if value in (None, ""):
        value = {} if default is None else default
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {"raw": value}
    return Jsonb(value)


def _rows(connection: sqlite3.Connection, table: str, key: str) -> Iterator[sqlite3.Row]:
    quoted = '"' + table.replace('"', '""') + '"'
    yield from connection.execute(f"SELECT * FROM {quoted} ORDER BY \"{key}\"")


def _decision_rows(
    connection: sqlite3.Connection,
    policy_version: str,
    *,
    after_decision_id: str | None = None,
) -> Iterator[sqlite3.Row]:
    # Do not pull descriptions or other large historical columns for the
    # decision backfill; these are the only source fields needed by the target.
    query = """SELECT decision_id,job_url,policy_version,lane,qualification_score,
                  preference_score,outcome_score,final_score,qualification_verdict,
                  action,confidence,uncertainty_json,blockers_json,requirements_json,
                  evidence_node_ids_json,title_signals_json,explanation,input_hash,
                  created_at,expires_at
           FROM job_decisions WHERE policy_version = ?"""
    params: tuple[Any, ...] = (policy_version,)
    if after_decision_id is not None:
        query += " AND decision_id > ?"
        params += (after_decision_id,)
    query += " ORDER BY decision_id"
    yield from connection.execute(query, params)


def _migration_run_resume_event(head: dict[str, Any] | None) -> tuple[str, int | None] | None:
    if head is None:
        return ("started", None)
    event_type = str(head["event_type"])
    event_id = int(head["migration_run_event_id"])
    if event_type == "started":
        return None
    if event_type in {"planned", "failed"}:
        return ("started", event_id)
    raise BrainImportError(
        f"migration run is terminal ({event_type}); corrected imports require a new run_key"
    )


def _ensure_source_and_run(pg, receipt: SQLiteSourceReceipt, run_key: str) -> int:
    with pg.cursor() as cur:
        cur.execute(
            """INSERT INTO brain_migration_sources
               (source_namespace, source_fingerprint, byte_length, schema_metadata)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (source_namespace, source_fingerprint) DO NOTHING
               RETURNING migration_source_id""",
            (SOURCE_NAMESPACE, receipt.sha256, receipt.byte_length, Jsonb(receipt.as_dict())),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "SELECT migration_source_id FROM brain_migration_sources "
                "WHERE source_namespace=%s AND source_fingerprint=%s",
                (SOURCE_NAMESPACE, receipt.sha256),
            )
            row = cur.fetchone()
        source_id = row["migration_source_id"]
        cur.execute(
            """INSERT INTO brain_migration_runs
               (migration_source_id, source_namespace, run_key, metadata)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (source_namespace, migration_source_id, run_key) DO NOTHING
               RETURNING migration_run_id""",
            (source_id, SOURCE_NAMESPACE, run_key, Jsonb({"source_sha256": receipt.sha256})),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "SELECT migration_run_id FROM brain_migration_runs "
                "WHERE source_namespace=%s AND migration_source_id=%s AND run_key=%s",
                (SOURCE_NAMESPACE, source_id, run_key),
            )
            row = cur.fetchone()
        run_id = row["migration_run_id"]
        cur.execute(
            "SELECT migration_run_event_id,event_type FROM brain_migration_run_events WHERE migration_run_id=%s "
            "AND source_namespace=%s ORDER BY migration_run_event_id DESC LIMIT 1",
            (run_id, SOURCE_NAMESPACE),
        )
        resume_event = _migration_run_resume_event(cur.fetchone())
        if resume_event is not None:
            _, supersedes_event_id = resume_event
            cur.execute(
                """INSERT INTO brain_migration_run_events
                   (migration_run_id, source_namespace, event_type, actor_id, metadata, supersedes_run_event_id)
                   VALUES (%s, %s, 'started', 'sqlite-to-postgres-importer', %s, %s)""",
                (run_id, SOURCE_NAMESPACE, Jsonb({"source_sha256": receipt.sha256}), supersedes_event_id),
            )
    return int(run_id)


def _mark_run_failed(pg, run_id: int, error: BaseException) -> None:
    pg.rollback()
    with pg.cursor() as cur:
        cur.execute(
            "SELECT migration_run_event_id,event_type FROM brain_migration_run_events "
            "WHERE migration_run_id=%s AND source_namespace=%s "
            "ORDER BY migration_run_event_id DESC LIMIT 1",
            (run_id, SOURCE_NAMESPACE),
        )
        head = cur.fetchone()
        if head is None or head["event_type"] != "started":
            pg.rollback()
            return
        cur.execute(
            "INSERT INTO brain_migration_run_events "
            "(migration_run_id,source_namespace,event_type,actor_id,metadata,supersedes_run_event_id) "
            "VALUES (%s,%s,'failed','sqlite-to-postgres-importer',%s,%s)",
            (
                run_id,
                SOURCE_NAMESPACE,
                Jsonb({"error_type": type(error).__name__}),
                head["migration_run_event_id"],
            ),
        )
    pg.commit()


def _commit_phase(pg, recheck_source: Callable[[], None] | None = None) -> None:
    """Commit one durable phase bracketed by sealed-source identity checks."""
    if recheck_source is not None:
        recheck_source()
    pg.commit()
    if recheck_source is not None:
        recheck_source()


def _commit_terminal_phase(
    pg,
    *,
    recheck_source: Callable[[], None],
    verify_final_source: Callable[[], None],
) -> None:
    """Verify the sealed source after staging completion, then commit once."""
    verify_final_source()
    recheck_source()
    pg.commit()


def _activate_controller(pg) -> None:
    """Activate and verify the exact append-only brain controller identity."""
    info = getattr(pg, "info", None)
    transaction_status = getattr(info, "transaction_status", None)
    if transaction_status is not None and transaction_status != TransactionStatus.IDLE:
        raise BrainImportError("brain import requires an idle dedicated Postgres connection")
    try:
        with pg.cursor() as cur:
            cur.execute(f"SET ROLE {CONTROLLER_ROLE}")
            cur.execute("SET search_path TO public")
            cur.execute("SELECT current_user AS current_user")
            row = cur.fetchone()
            if row is None or row["current_user"] != CONTROLLER_ROLE:
                raise BrainImportError(f"required controller role is not active: {CONTROLLER_ROLE}")
        pg.commit()
    except Exception:
        pg.rollback()
        try:
            with pg.cursor() as cur:
                cur.execute("RESET ROLE")
            pg.commit()
        except Exception:
            pg.rollback()
        raise


def _acquire_import_lock(pg, source_sha256: str) -> None:
    """Hold a session lock so one process owns a sealed source at a time."""
    with pg.cursor() as cur:
        cur.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS acquired",
            (f"applypilot-brain-import:{source_sha256}",),
        )
        row = cur.fetchone()
    if row is None or not row["acquired"]:
        pg.rollback()
        raise BrainImportError("another importer already owns source fingerprint")
    pg.commit()


def _release_import_context(pg, source_sha256: str, *, lock_acquired: bool) -> None:
    """Release session-level import state even after an aborted transaction."""
    pg.rollback()
    try:
        with pg.cursor() as cur:
            if lock_acquired:
                cur.execute(
                    "SELECT pg_advisory_unlock(hashtextextended(%s, 0)) AS released",
                    (f"applypilot-brain-import:{source_sha256}",),
                )
            cur.execute("RESET ROLE")
        pg.commit()
    except Exception:
        pg.rollback()
        raise


def _source_job_map(pg) -> dict[str, str]:
    with pg.cursor() as cur:
        cur.execute(
            "SELECT canonical_url, job_id FROM brain_jobs "
            "WHERE source_namespace=%s AND canonical_url IS NOT NULL",
            (SOURCE_NAMESPACE,),
        )
        return {row["canonical_url"]: row["job_id"] for row in cur.fetchall()}


def _insert_jobs(pg, source: sqlite3.Connection, job_map: dict[str, str], batch_size: int) -> int:
    count = 0
    pending = []
    for row in _rows(source, "jobs", "url"):
        url = str(row["url"])
        jid = _job_id(url)
        job_map[url] = jid
        metadata = {key: row[key] for key in row.keys() if key not in {"url", "title", "company"}}
        pending.append((jid, SOURCE_NAMESPACE, url, url, row["title"], row["company"], Jsonb(metadata)))
        if len(pending) >= batch_size:
            _flush_jobs(pg, pending)
            count += len(pending)
            pending.clear()
    if pending:
        _flush_jobs(pg, pending)
        count += len(pending)
    return count


def _flush_jobs(pg, rows: list[tuple[Any, ...]]) -> None:
    with pg.cursor() as cur:
        cur.executemany(
            """INSERT INTO brain_jobs
               (job_id, source_namespace, source_job_id, canonical_url, title, company, current_metadata)
               VALUES (%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (job_id) DO UPDATE SET title=EXCLUDED.title,
                 company=EXCLUDED.company, current_metadata=EXCLUDED.current_metadata,
                 updated_at=now()""",
            rows,
        )


def _insert_aliases(pg, source: sqlite3.Connection, job_map: dict[str, str], fingerprint: str, batch_size: int) -> int:
    pending = []
    count = 0
    for row in _rows(source, "jobs", "url"):
        url = str(row["url"])
        pending.append((job_map[url], SOURCE_NAMESPACE, fingerprint, url, url, "sqlite-job", Jsonb({})))
        if len(pending) >= batch_size:
            count += _flush_aliases(pg, pending)
            pending.clear()
    if pending:
        count += _flush_aliases(pg, pending)
    return count


def _flush_aliases(pg, rows: list[tuple[Any, ...]]) -> int:
    with pg.cursor() as cur:
        cur.executemany(
            """INSERT INTO brain_job_aliases
               (job_id, source_namespace, source_database_fingerprint, source_item_id, source_url, alias_type, alias_metadata)
               VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
            rows,
        )
    return len(rows)


def _insert_observations(
    pg,
    source: sqlite3.Connection,
    job_map: dict[str, str],
    *,
    recheck_source: Callable[[], None] | None = None,
) -> int:
    pending: list[tuple[Any, ...]] = []
    copied = 0
    for row in _rows(source, "jobs", "url"):
        observed_at = row["discovered_at"] or row["detail_scraped_at"] or row["scored_at"] or row["applied_at"]
        pending.append((
            SOURCE_NAMESPACE,
            hashlib.sha256(f"observation:{row['url']}".encode()).hexdigest(),
            job_map[row["url"]],
            observed_at,
            _json({"source_url": row["url"], "site": row["site"], "source_board": row["source_board"]}),
        ))
        copied += 1
        if len(pending) >= 500:
            _flush_observations(pg, pending)
            pending.clear()
    if pending:
        _flush_observations(pg, pending)
    _commit_phase(pg, recheck_source)
    return copied


def _flush_observations(pg, rows: list[tuple[Any, ...]]) -> None:
    with pg.cursor() as cur:
        cur.executemany(
            """INSERT INTO brain_job_observations
               (source_namespace,source_observation_id,job_id,observed_at,observation_metadata)
               VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT (source_namespace,source_observation_id) DO NOTHING""",
            rows,
        )


def _insert_applications(
    pg,
    source: sqlite3.Connection,
    job_map: dict[str, str],
    *,
    recheck_source: Callable[[], None] | None = None,
) -> tuple[int, dict[str, str]]:
    application_map: dict[str, str] = {}
    rows: list[tuple[Any, ...]] = []
    for row in _rows(source, "applications", "id"):
        app_id = _application_id(int(row["id"]))
        application_map[row["job_url"]] = app_id
        channel = row["channel"] or row["source"] or "unknown"
        lane = "linkedin" if "linkedin" in str(channel).lower() or "linkedin.com" in row["job_url"] else "ats"
        metadata = {key: row[key] for key in row.keys() if key not in {
            "id", "job_url", "channel", "status", "created_at", "updated_at"
        }}
        rows.append((app_id, job_map[row["job_url"]], SOURCE_NAMESPACE, str(row["id"]), channel, lane,
                     row["status"], Jsonb(metadata), row["created_at"], row["updated_at"]))
    with pg.cursor() as cur:
        cur.executemany(
            """INSERT INTO brain_applications
               (application_id,job_id,source_namespace,source_application_id,source_channel,lane,
                current_state,application_metadata,created_at,updated_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
            rows,
        )
    _commit_phase(pg, recheck_source)
    return len(rows), application_map


def _insert_application_events(
    pg,
    source: sqlite3.Connection,
    application_map: dict[str, str],
    *,
    recheck_source: Callable[[], None] | None = None,
) -> int:
    rows = []
    for row in _rows(source, "application_events", "id"):
        rows.append((application_map[row["job_url"]], SOURCE_NAMESPACE, str(row["id"]),
                     row["event_type"] or row["status"] or "event", row["channel"], row["happened_at"],
                     _json({"status": row["status"], "notes": row["notes"], "job_url": row["job_url"]})))
    with pg.cursor() as cur:
        cur.executemany(
            """INSERT INTO brain_application_events
               (application_id,source_namespace,source_event_id,event_type,source_channel,occurred_at,event_metadata)
               VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
            rows,
        )
    _commit_phase(pg, recheck_source)
    return len(rows)


def _insert_email_events(
    pg,
    source: sqlite3.Connection,
    job_map: dict[str, str],
    *,
    recheck_source: Callable[[], None] | None = None,
) -> dict[str, int]:
    rows = []
    for row in _rows(source, "email_events", "message_id"):
        metadata = {key: row[key] for key in row.keys() if key not in {"message_id", "job_url", "occurred_at"}}
        rows.append((SOURCE_NAMESPACE, row["message_id"], job_map.get(row["job_url"]),
                     row["stage"] or row["outcome"] or "email", row["occurred_at"], Jsonb(metadata)))
    with pg.cursor() as cur:
        cur.executemany(
            """INSERT INTO brain_email_events
               (source_namespace,source_event_id,job_id,event_type,occurred_at,event_metadata)
               VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
            rows,
        )
    _commit_phase(pg, recheck_source)
    with pg.cursor() as cur:
        cur.execute("SELECT source_event_id,email_event_id FROM brain_email_events WHERE source_namespace=%s", (SOURCE_NAMESPACE,))
        return {row["source_event_id"]: int(row["email_event_id"]) for row in cur.fetchall()}


def _insert_email_reviews(
    pg,
    source: sqlite3.Connection,
    job_map: dict[str, str],
    *,
    recheck_source: Callable[[], None] | None = None,
) -> int:
    rows = []
    for row in _rows(source, "email_event_reviews", "id"):
        source_email = source.execute("SELECT job_url FROM email_events WHERE message_id=?", (row["message_id"],)).fetchone()
        url = row["corrected_job_url"] or (source_email["job_url"] if source_email else None)
        rows.append((SOURCE_NAMESPACE, f"review:{row['id']}", job_map.get(url), "review", row["reviewed_at"],
                     _json(dict(row))))
    with pg.cursor() as cur:
        cur.executemany(
            """INSERT INTO brain_email_events
               (source_namespace,source_event_id,job_id,event_type,occurred_at,event_metadata)
               VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
            rows,
        )
    _commit_phase(pg, recheck_source)
    return len(rows)


def _insert_outcomes(
    pg,
    source: sqlite3.Connection,
    job_map: dict[str, str],
    email_map: dict[str, int],
    *,
    recheck_source: Callable[[], None] | None = None,
) -> int:
    del email_map
    rows = []
    with pg.cursor() as cur:
        for row in source.execute("SELECT * FROM reviewed_outcomes ORDER BY event_id,job_url"):
            job_id = job_map[row["job_url"]]
            cur.execute(
                "SELECT email_event_id FROM brain_email_events "
                "WHERE source_namespace=%s AND source_event_id=%s AND job_id=%s",
                (SOURCE_NAMESPACE, row["event_id"], job_id),
            )
            target_email = cur.fetchone()
            if target_email is None:
                source_email = source.execute(
                    "SELECT occurred_at,stage,outcome FROM email_events WHERE message_id=?", (row["event_id"],)
                ).fetchone()
                attribution_id = "outcome-attribution:" + hashlib.sha256(
                    f"{row['event_id']}:{row['job_url']}".encode()
                ).hexdigest()
                cur.execute(
                    """INSERT INTO brain_email_events
                       (source_namespace,source_event_id,job_id,event_type,occurred_at,event_metadata)
                       VALUES (%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (source_namespace,source_event_id) DO NOTHING
                       RETURNING email_event_id""",
                    (SOURCE_NAMESPACE, attribution_id, job_id, "outcome-attribution",
                     source_email["occurred_at"], _json({"source_message_id": row["event_id"],
                                                        "source_stage": source_email["stage"],
                                                        "source_outcome": source_email["outcome"],
                                                        "attribution": row["attribution_json"]})),
                )
                target_email = cur.fetchone()
                if target_email is None:
                    cur.execute(
                        "SELECT email_event_id FROM brain_email_events WHERE source_namespace=%s AND source_event_id=%s",
                        (SOURCE_NAMESPACE, attribution_id),
                    )
                    target_email = cur.fetchone()
            rows.append((SOURCE_NAMESPACE, f"{row['event_id']}:{row['job_url']}", job_id,
                         target_email["email_event_id"], row["review_status"], row["normalized_stage"],
                         row["weight"], row["reviewer"], row["reason"], _json(row["attribution_json"]),
                         row["created_at"], row["reviewed_at"], row["updated_at"]))
        cur.executemany(
            """INSERT INTO brain_reviewed_outcomes
               (source_namespace,source_event_id,job_id,email_event_id,review_status,normalized_stage,
                weight,reviewer,reason,review_metadata,created_at,reviewed_at,updated_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
            rows,
        )
    _commit_phase(pg, recheck_source)
    return len(rows)


def _insert_label_confidence(
    pg,
    source: sqlite3.Connection,
    job_map: dict[str, str],
    run_id: int,
    *,
    recheck_source: Callable[[], None] | None = None,
) -> tuple[int, int]:
    source_count = int(source.execute("SELECT count(*) FROM research_label_confidence").fetchone()[0])
    with pg.cursor() as cur:
        cur.execute(
            "SELECT count(*) AS n FROM brain_label_events WHERE source_namespace=%s AND source_event_id LIKE 'confidence:%%'",
            (SOURCE_NAMESPACE,),
        )
        existing = int(cur.fetchone()["n"])
        cur.execute(
            "SELECT count(*) AS n FROM brain_migration_quarantine "
            "WHERE migration_run_id=%s AND source_namespace=%s AND source_table='research_label_confidence'",
            (run_id, SOURCE_NAMESPACE),
        )
        existing_quarantine = int(cur.fetchone()["n"])
    if existing + existing_quarantine >= source_count:
        return existing, existing_quarantine
    inserted = quarantined = 0
    with pg.cursor() as cur:
        for confidence in _rows(source, "research_label_confidence", "label_id"):
            label = source.execute("SELECT * FROM research_labels WHERE id=?", (confidence["label_id"],)).fetchone()
            if label is None:
                cur.execute(
                    """INSERT INTO brain_migration_quarantine
                       (migration_run_id,source_namespace,source_table,source_key,reason_code,unresolved_evidence)
                       VALUES (%s,%s,'research_label_confidence',%s,'missing_label_predecessor',%s)
                       ON CONFLICT DO NOTHING""",
                    (run_id, SOURCE_NAMESPACE, confidence["label_id"], _json(dict(confidence))),
                )
                quarantined += 1
                continue
            source_item_id = label["item_id"] or confidence["item_id"]
            source_item_url = label["job_url"]
            job_id = job_map.get(source_item_url)
            if job_id is None and not source_item_id and not source_item_url:
                cur.execute(
                    """INSERT INTO brain_migration_quarantine
                       (migration_run_id,source_namespace,source_table,source_key,reason_code,unresolved_evidence)
                       VALUES (%s,%s,'research_label_confidence',%s,'missing_label_endpoint',%s)
                       ON CONFLICT DO NOTHING""",
                    (run_id, SOURCE_NAMESPACE, confidence["label_id"], _json(dict(confidence))),
                )
                quarantined += 1
                continue
            cur.execute(
                "SELECT label_event_id FROM brain_label_events WHERE source_namespace=%s AND source_event_id=%s",
                (SOURCE_NAMESPACE, confidence["label_id"]),
            )
            original = cur.fetchone()
            if original is None:
                cur.execute(
                    """INSERT INTO brain_migration_quarantine
                       (migration_run_id,source_namespace,source_table,source_key,reason_code,unresolved_evidence)
                       VALUES (%s,%s,'research_label_confidence',%s,'missing_label_predecessor',%s)
                       ON CONFLICT DO NOTHING""",
                    (run_id, SOURCE_NAMESPACE, confidence["label_id"], _json(dict(confidence))),
                )
                quarantined += 1
                continue
            cur.execute(
                """INSERT INTO brain_label_events
                   (source_namespace,source_event_id,job_id,source_item_id,source_item_url,project,method,
                    confidence,weight,label_name,label_value,occurred_at,event_metadata,supersedes_label_event_id)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'fitmap_decision',%s,%s,%s,%s)
                   ON CONFLICT (source_namespace,source_event_id) DO NOTHING""",
                (SOURCE_NAMESPACE, f"confidence:{confidence['label_id']}", job_id,
                 source_item_id, source_item_url, label["source_project_id"] or "applypilot",
                 confidence["method"] or "confidence-import",
                 max(Decimal(0), Decimal(1) - Decimal(str(confidence["item_flip_rate"] or 0))),
                 Decimal(str(confidence["weight"])),
                 _json({"decision": label["decision"], "rating": label["rating"]}),
                 confidence["imported_at"], _json(dict(confidence)), original["label_event_id"]),
            )
            inserted += 1
    _commit_phase(pg, recheck_source)
    return inserted, quarantined


def _insert_labels(pg, source: sqlite3.Connection, job_map: dict[str, str], run_id: int) -> tuple[int, int]:
    imported = quarantined = 0
    with pg.cursor() as cur:
        for row in _rows(source, "research_labels", "id"):
            url = row["job_url"]
            job_id = job_map.get(url)
            if job_id is None and not row["item_id"] and not url:
                cur.execute(
                    """INSERT INTO brain_migration_quarantine
                       (migration_run_id, source_namespace, source_table, source_key, reason_code, unresolved_evidence)
                       VALUES (%s,%s,'research_labels',%s,'missing_label_endpoint',%s)
                       ON CONFLICT DO NOTHING""",
                    (run_id, SOURCE_NAMESPACE, str(row["id"]), _json(dict(row))),
                )
                quarantined += 1
                continue
            cur.execute(
                """INSERT INTO brain_label_events
                   (source_namespace, source_event_id, job_id, source_item_id, source_item_url,
                    project, method, confidence, weight, label_name, label_value, occurred_at, event_metadata)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,1,'fitmap_decision',%s,%s,%s)
                   ON CONFLICT (source_namespace, source_event_id) DO NOTHING""",
                (SOURCE_NAMESPACE, str(row["id"]), job_id, row["item_id"], url,
                 row["source_project_id"] or "applypilot", row["method"] or "import",
                 1 if row["rating"] is not None else None,
                 _json({"decision": row["decision"], "rating": row["rating"], "reason": row["reason"], "tags": row["tags_json"]}),
                 row["created_at"], _json(row["raw_event_json"])),
            )
            imported += 1
    return imported, quarantined


def _insert_pairwise(pg, source: sqlite3.Connection, job_map: dict[str, str]) -> int:
    count = 0
    with pg.cursor() as cur:
        for row in _rows(source, "research_pairwise_labels", "id"):
            preference = {"left": "left", "right": "right", "tie": "tie"}.get(str(row["winner"]), "unclear")
            cur.execute(
                """INSERT INTO brain_pairwise_events
                   (source_namespace, source_event_id, left_job_id, right_job_id, left_source_item_id,
                    right_source_item_id, left_source_url, right_source_url, project, method, weight,
                    preference, occurred_at, event_metadata)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1,%s,%s,%s)
                   ON CONFLICT (source_namespace, source_event_id) DO NOTHING""",
                (SOURCE_NAMESPACE, str(row["id"]), job_map.get(row["left_job_url"]), job_map.get(row["right_job_url"]),
                 row["left_item_id"], row["right_item_id"], row["left_job_url"], row["right_job_url"],
                 row["source_project_id"] or "applypilot", row["method"] or "import", preference,
                 row["created_at"], _json(row["raw_event_json"])),
            )
            count += 1
    return count


def _verify_kg_artifacts(pg, source: sqlite3.Connection) -> int:
    """Derive every legacy KG hash from source bytes and verify its target ledger row."""
    count = 0
    for row in _rows(source, "research_kg_artifacts", "kg_version"):
        raw = row["compact_kg_json"]
        if raw is None:
            raise BrainImportError(f"knowledge graph {row['kg_version']} has no content")
        content = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        try:
            json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BrainImportError(f"knowledge graph {row['kg_version']} is invalid JSON") from exc
        artifact_hash = hashlib.sha256(content).hexdigest()
        claimed_hash = row["inputs_sha"]
        if claimed_hash and claimed_hash != artifact_hash:
            raise BrainImportError(
                f"knowledge graph {row['kg_version']} source hash mismatch: "
                f"claimed={claimed_hash}, actual={artifact_hash}"
            )
        with pg.cursor() as cur:
            cur.execute(
                "SELECT artifact_hash,byte_length,media_type FROM brain_artifacts WHERE artifact_hash=%s",
                (artifact_hash,),
            )
            target = cur.fetchone()
        if (
            target is None
            or int(target["byte_length"]) != len(content)
            or target["media_type"] != "application/json"
        ):
            raise BrainImportError(
                f"knowledge graph {row['kg_version']} is missing or inconsistent in target artifact ledger"
            )
        count += 1
    return count


_BINDABLE_COMPILED_POLICY_ROLES = frozenset({
    "qualification_model",
    "preference_model",
    "outcome_model",
    "config",
    "metrics",
    "label_snapshot",
    "pairwise_snapshot",
    "outcome_snapshot",
})


def _require_artifact(pg, artifact: ArtifactDescriptor) -> None:
    with pg.cursor() as cur:
        cur.execute(
            "SELECT artifact_hash,byte_length,media_type,provenance FROM brain_artifacts WHERE artifact_hash=%s",
            (artifact.sha256,),
        )
        target = cur.fetchone()
    if (
        target is None
        or int(target["byte_length"]) != artifact.byte_length
        or target["media_type"] != artifact.media_type
    ):
        raise BrainImportError(f"policy artifact is missing or inconsistent: {artifact.sha256}")
    if artifact.policy_source_id is not None:
        provenance = target.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("policy_source_id") != artifact.policy_source_id
        ):
            raise BrainImportError(f"policy artifact provenance is missing or inconsistent: {artifact.sha256}")


def _require_referenced_artifact(pg, artifact_hash: str, role: str) -> None:
    with pg.cursor() as cur:
        cur.execute(
            "SELECT artifact_hash,byte_length,media_type FROM brain_artifacts WHERE artifact_hash=%s",
            (artifact_hash,),
        )
        target = cur.fetchone()
    if target is None:
        raise BrainImportError(f"policy {role} artifact is missing from target ledger: {artifact_hash}")


def _bind_policy_artifact(cur, policy_version: str, artifact_role: str, artifact_hash: str) -> None:
    cur.execute(
        """INSERT INTO brain_policy_artifacts
           (policy_version,artifact_role,artifact_hash)
           VALUES (%s,%s,%s) ON CONFLICT DO NOTHING""",
        (policy_version, artifact_role, artifact_hash),
    )
    cur.execute(
        "SELECT artifact_hash FROM brain_policy_artifacts WHERE policy_version=%s AND artifact_role=%s",
        (policy_version, artifact_role),
    )
    existing = cur.fetchone()
    if existing is None or existing["artifact_hash"] != artifact_hash:
        raise BrainImportError(
            f"existing policy artifact binding conflicts with sealed source: {policy_version}/{artifact_role}"
        )


def _insert_policies(
    pg,
    source: sqlite3.Connection,
    *,
    recheck_source: Callable[[], None] | None = None,
) -> int:
    count = 0
    with pg.cursor() as cur:
        for row in _rows(source, "decision_policy_versions", "policy_version"):
            compiled = compile_policy_artifacts(dict(row))
            metadata = compiled.metadata_object()
            for artifact in compiled.artifacts:
                _require_artifact(pg, artifact)
            cur.execute(
                "SELECT policy_version,lane,lifecycle,policy_metadata FROM brain_decision_policies "
                "WHERE policy_version=%s",
                (compiled.policy_version,),
            )
            existing = cur.fetchone()
            if existing is not None:
                if (
                    existing["lane"] != compiled.lane
                    or existing["lifecycle"] != compiled.target_lifecycle
                    or existing["policy_metadata"] != metadata
                ):
                    raise BrainImportError(
                        f"existing policy conflicts with sealed source: {compiled.policy_version}"
                    )
            else:
                cur.execute(
                    """INSERT INTO brain_decision_policies
                       (policy_version,lane,lifecycle,policy_metadata,created_at)
                       VALUES (%s,%s,'draft',%s,%s)""",
                    (
                        compiled.policy_version,
                        compiled.lane,
                        Jsonb(metadata),
                        row["created_at"],
                    ),
                )
            for artifact in compiled.artifacts:
                if artifact.role not in _BINDABLE_COMPILED_POLICY_ROLES:
                    continue
                _bind_policy_artifact(cur, compiled.policy_version, artifact.role, artifact.sha256)
            graph_hash = metadata["sourceReferences"]["kgVersion"]
            if graph_hash is not None:
                _require_referenced_artifact(pg, graph_hash, "knowledge_graph")
                _bind_policy_artifact(cur, compiled.policy_version, "knowledge_graph", graph_hash)
            # ensure_policy_partition intentionally requires an idle connection
            # because it changes owner-controlled DDL under its own transaction.
            _commit_phase(pg, recheck_source)
            ensure_policy_partition(pg, compiled.policy_version)
            if recheck_source is not None:
                recheck_source()
            count += 1
    return count


def _insert_decisions(
    pg,
    source: sqlite3.Connection,
    job_map: dict[str, str],
    batch_size: int,
    *,
    recheck_source: Callable[[], None] | None = None,
) -> int:
    count = 0
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    policies = source.execute(
        "SELECT policy_version, count(*) AS n FROM job_decisions GROUP BY policy_version ORDER BY policy_version"
    ).fetchall()
    for policy in policies:
        if recheck_source is not None:
            recheck_source()
        policy_version = str(policy["policy_version"])
        source_count = int(policy["n"])
        with pg.cursor() as cur:
            cur.execute(
                "SELECT source_decision_id FROM brain_job_decisions "
                "WHERE source_namespace=%s AND policy_version=%s",
                (SOURCE_NAMESPACE, policy_version),
            )
            target_ids = {str(row["source_decision_id"]) for row in cur}
        target_count = len(target_ids)
        if target_count > source_count:
            raise BrainImportError(f"policy partition contains more rows than the source: {policy_version}")
        remaining_target_ids = set(target_ids)
        copied = target_count
        pending: list[tuple[Any, ...]] = []
        for row in _decision_rows(source, policy_version):
            decision_id = str(row["decision_id"])
            if decision_id in remaining_target_ids:
                remaining_target_ids.remove(decision_id)
                continue
            if row["job_url"] not in job_map:
                raise BrainImportError(f"job decision has no job URL target: {row['decision_id']}")
            pending.append((
                decision_id, SOURCE_NAMESPACE, decision_id,
                job_map[row["job_url"]], row["policy_version"], row["lane"],
                row["qualification_score"], None, row["preference_score"], row["outcome_score"],
                row["final_score"], row["qualification_verdict"], row["action"], row["confidence"],
                _json(row["uncertainty_json"], []), _json(row["blockers_json"], []),
                _json(row["requirements_json"], []), _json(row["evidence_node_ids_json"], []),
                _json(row["title_signals_json"], []), row["explanation"], row["input_hash"],
                row["created_at"], row["expires_at"],
            ))
            if len(pending) >= batch_size:
                copied += _flush_decisions(pg, pending)
                pending.clear()
                _commit_phase(pg, recheck_source)
        if pending:
            copied += _flush_decisions(pg, pending)
            _commit_phase(pg, recheck_source)
        if remaining_target_ids:
            raise BrainImportError(
                f"policy partition contains decisions absent from the sealed source: {policy_version}"
            )
        if copied != source_count:
            raise BrainImportError(
                f"policy copy count mismatch for {policy_version}: expected={source_count}, copied={copied}"
            )
        count += copied
    return count


def _source_bounds(source: sqlite3.Connection, table_name: str, key_columns: tuple[str, ...]) -> tuple[str, str]:
    order = ",".join(f'"{column}"' for column in key_columns)
    columns = ",".join(f'"{column}"' for column in key_columns)
    first = source.execute(f'SELECT {columns} FROM "{table_name}" ORDER BY {order} LIMIT 1').fetchone()
    if first is None:
        return "", ""
    last = source.execute(f'SELECT {columns} FROM "{table_name}" ORDER BY {order} DESC LIMIT 1').fetchone()

    def encode(row: sqlite3.Row, boundary: str) -> str:
        values = json.dumps([row[column] for column in key_columns], ensure_ascii=True, separators=(",", ":"))
        return f"{boundary}:{values}"

    return encode(first, "0"), encode(last, "1")


def _record_completed_batches(
    pg,
    source: sqlite3.Connection,
    receipt: SQLiteSourceReceipt,
    run_id: int,
    parity_results: Mapping[str, ParityResult],
    *,
    recheck_source: Callable[[], None] | None = None,
) -> None:
    worker_id = "sqlite-to-postgres-importer"
    for table in SOURCE_TABLES:
        source_count = int(receipt.table_counts[table.name])
        parity = parity_results.get(table.name)
        if parity is None:
            raise BrainImportError(f"missing independent parity result for {table.name}")
        target_count = int(parity.target_count)
        if parity.source_count != source_count or not parity.passed:
            raise BrainImportError(
                f"independent parity failed for {table.name}: "
                f"receipt={source_count}, source={parity.source_count}, target={target_count}, "
                f"mismatches={parity.mismatch_count}, unresolved={parity.unresolved_count}"
            )
        key_start, key_end = _source_bounds(source, table.name, table.key_columns)
        canonical_batch_hash = hashlib.sha256(json.dumps({
            "source_fingerprint": receipt.sha256,
            "source_table": table.name,
            "source_count": source_count,
            "target_count": target_count,
            "source_hash": parity.source_hash,
            "target_hash": parity.target_hash,
            "key_start": key_start,
            "key_end": key_end,
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with pg.cursor() as cur:
            cur.execute(
                "SELECT migration_batch_id,key_start,key_end FROM brain_migration_batches "
                "WHERE migration_run_id=%s AND source_namespace=%s AND source_table=%s AND batch_ordinal=1",
                (run_id, SOURCE_NAMESPACE, table.name),
            )
            batch = cur.fetchone()
            if batch is None:
                cur.execute(
                    """INSERT INTO brain_migration_batches
                       (migration_run_id,source_namespace,source_table,batch_ordinal,key_start,key_end)
                       VALUES (%s,%s,%s,1,%s,%s) RETURNING migration_batch_id""",
                    (run_id, SOURCE_NAMESPACE, table.name, key_start, key_end),
                )
                batch = cur.fetchone()
                batch = {
                    "migration_batch_id": batch["migration_batch_id"],
                    "key_start": key_start,
                    "key_end": key_end,
                }
            elif batch["key_start"] != key_start or batch["key_end"] != key_end:
                raise BrainImportError(f"existing migration batch bounds conflict for {table.name}")
            batch_id = int(batch["migration_batch_id"])
            cur.execute(
                "SELECT migration_batch_event_id,event_type,attempt,worker_id,source_count,target_count,"
                "canonical_batch_hash FROM brain_migration_batch_events "
                "WHERE migration_batch_id=%s ORDER BY migration_batch_event_id DESC LIMIT 1",
                (batch_id,),
            )
            head = cur.fetchone()
            if head is None:
                cur.execute(
                    """INSERT INTO brain_migration_batch_events
                       (migration_run_id,source_namespace,source_table,migration_batch_id,batch_ordinal,event_type,attempt)
                       VALUES (%s,%s,%s,%s,1,'pending',0) RETURNING migration_batch_event_id""",
                    (run_id, SOURCE_NAMESPACE, table.name, batch_id),
                )
                head = {"migration_batch_event_id": cur.fetchone()["migration_batch_event_id"],
                        "event_type": "pending", "attempt": 0, "worker_id": None}
            if head["event_type"] != "completed":
                if head["event_type"] != "claimed":
                    attempt = 1 if head["event_type"] == "pending" else int(head["attempt"]) + 1
                    cur.execute(
                        """INSERT INTO brain_migration_batch_events
                           (migration_run_id,source_namespace,source_table,migration_batch_id,batch_ordinal,
                            event_type,attempt,worker_id,lease_expires_at,supersedes_batch_event_id)
                           VALUES (%s,%s,%s,%s,1,'claimed',%s,%s,now()+interval '10 minutes',%s)
                           RETURNING migration_batch_event_id""",
                        (run_id, SOURCE_NAMESPACE, table.name, batch_id, attempt, worker_id,
                         head["migration_batch_event_id"]),
                    )
                    head = {"migration_batch_event_id": cur.fetchone()["migration_batch_event_id"],
                            "event_type": "claimed", "attempt": attempt, "worker_id": worker_id}
                cur.execute(
                    """INSERT INTO brain_migration_batch_events
                       (migration_run_id,source_namespace,source_table,migration_batch_id,batch_ordinal,
                        event_type,attempt,worker_id,source_count,target_count,canonical_batch_hash,
                        supersedes_batch_event_id)
                       VALUES (%s,%s,%s,%s,1,'completed',%s,%s,%s,%s,%s,%s)
                       RETURNING migration_batch_event_id""",
                    (run_id, SOURCE_NAMESPACE, table.name, batch_id, head["attempt"], worker_id,
                     source_count, target_count, canonical_batch_hash, head["migration_batch_event_id"]),
                )
                completed_event_id = int(cur.fetchone()["migration_batch_event_id"])
            else:
                completed_event_id = int(head["migration_batch_event_id"])
                if (
                    int(head["source_count"]) != source_count
                    or int(head["target_count"]) != target_count
                    or head["canonical_batch_hash"] != canonical_batch_hash
                ):
                    raise BrainImportError(f"existing completed migration batch conflicts for {table.name}")
            checkpoint_hash = hashlib.sha256(
                f"{canonical_batch_hash}:{completed_event_id}:{key_end}".encode()
            ).hexdigest()
            cur.execute(
                "SELECT last_key,migration_batch_id,migration_batch_event_id,canonical_checkpoint_hash "
                "FROM brain_migration_checkpoints "
                "WHERE migration_run_id=%s AND source_namespace=%s AND source_table=%s AND batch_ordinal=1",
                (run_id, SOURCE_NAMESPACE, table.name),
            )
            checkpoint = cur.fetchone()
            if checkpoint is None:
                cur.execute(
                    """INSERT INTO brain_migration_checkpoints
                       (migration_run_id,source_namespace,source_table,batch_ordinal,last_key,migration_batch_id,
                        migration_batch_event_id,canonical_checkpoint_hash)
                       VALUES (%s,%s,%s,1,%s,%s,%s,%s)""",
                    (run_id, SOURCE_NAMESPACE, table.name, key_end, batch_id, completed_event_id, checkpoint_hash),
                )
            elif (
                checkpoint["last_key"] != key_end
                or int(checkpoint["migration_batch_id"]) != batch_id
                or int(checkpoint["migration_batch_event_id"]) != completed_event_id
                or checkpoint["canonical_checkpoint_hash"] != checkpoint_hash
            ):
                raise BrainImportError(f"existing migration checkpoint conflicts for {table.name}")
        _commit_phase(pg, recheck_source)


def _flush_decisions(pg, rows: list[tuple[Any, ...]]) -> int:
    with pg.cursor() as cur:
        with cur.copy(
            """COPY brain_job_decisions
               (decision_id,source_namespace,source_decision_id,job_id,policy_version,lane,
                qualification_score,qualification_floor,preference_score,outcome_score,final_score,
                qualification_verdict,action,confidence,uncertainty,blockers,requirements,evidence_nodes,
                title_signals,explanation,input_hash,created_at,expires_at)
               FROM STDIN"""
        ) as copy:
            for row in rows:
                copy.write_row(row)
    return len(rows)


def import_sqlite_to_postgres(
    pg,
    sqlite_path: str | Path,
    *,
    expected_sha256: str,
    run_key: str,
    batch_size: int = 500,
    dry_run: bool = False,
) -> ImportSummary:
    """Import the canonical source, refusing an unexpected database identity."""
    receipt = capture_sqlite_source_receipt(sqlite_path)
    if receipt.sha256 != expected_sha256:
        raise BrainImportError(f"source SHA mismatch: expected {expected_sha256}, got {receipt.sha256}")
    guard = SQLiteSourceGuard(sqlite_path, expected_sha256)
    source = guard.connection
    source.row_factory = sqlite3.Row
    run_id: int | None = None
    controller_active = False
    lock_acquired = False
    try:
        if dry_run:
            guard.verify_final()
            return ImportSummary(
                0,
                receipt.sha256,
                {},
                {
                    "research_labels_missing_job": 0,
                    "research_label_confidence_missing_job": 0,
                },
            )
        _activate_controller(pg)
        controller_active = True
        _acquire_import_lock(pg, receipt.sha256)
        lock_acquired = True
        run_id = _ensure_source_and_run(pg, receipt, run_key)
        _commit_phase(pg, guard.recheck)
        job_map = _source_job_map(pg)
        with pg.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM brain_label_events "
                "WHERE source_namespace=%s AND source_event_id NOT LIKE 'confidence:%%'",
                (SOURCE_NAMESPACE,),
            )
            label_count = cur.fetchone()["n"]
            cur.execute(
                "SELECT count(*) AS n FROM brain_migration_quarantine "
                "WHERE migration_run_id=%s AND source_table='research_labels'",
                (run_id,),
            )
            quarantine_count = cur.fetchone()["n"]
            cur.execute("SELECT count(*) AS n FROM brain_pairwise_events WHERE source_namespace=%s", (SOURCE_NAMESPACE,))
            pairwise_count = cur.fetchone()["n"]
        labels_expected = receipt.table_counts["research_labels"]
        imported = {
            "jobs": _insert_jobs(pg, source, job_map, batch_size),
            "aliases": _insert_aliases(pg, source, job_map, receipt.sha256, batch_size),
        }
        guard.recheck()
        if label_count + quarantine_count >= labels_expected:
            labels, missing = label_count, quarantine_count
        else:
            labels, missing = _insert_labels(pg, source, job_map, run_id)
        imported["labels"] = labels
        imported["observations"] = _insert_observations(
            pg, source, job_map, recheck_source=guard.recheck
        )
        imported["applications"], application_map = _insert_applications(
            pg, source, job_map, recheck_source=guard.recheck
        )
        imported["application_events"] = _insert_application_events(
            pg, source, application_map, recheck_source=guard.recheck
        )
        email_map = _insert_email_events(pg, source, job_map, recheck_source=guard.recheck)
        imported["email"] = receipt.table_counts["email_events"]
        imported["email_reviews"] = _insert_email_reviews(
            pg, source, job_map, recheck_source=guard.recheck
        )
        imported["outcomes"] = _insert_outcomes(
            pg, source, job_map, email_map, recheck_source=guard.recheck
        )
        confidence, confidence_quarantine = _insert_label_confidence(
            pg, source, job_map, run_id, recheck_source=guard.recheck
        )
        imported["label_confidence"] = confidence
        imported["pairwise"] = pairwise_count if pairwise_count >= receipt.table_counts["research_pairwise_labels"] else _insert_pairwise(pg, source, job_map)
        imported["policies"] = _insert_policies(pg, source, recheck_source=guard.recheck)
        imported["decisions"] = _insert_decisions(
            pg,
            source,
            job_map,
            batch_size,
            recheck_source=guard.recheck,
        )
        kg_artifact_count = _verify_kg_artifacts(pg, source)
        if receipt.table_counts["research_kg_runs"] or receipt.table_counts["research_scores"]:
            raise BrainImportError(
                "nonempty research_kg_runs/research_scores require an explicit canonical mapping"
            )
        imported["kg_artifacts"] = kg_artifact_count
        _commit_phase(pg, guard.recheck)
        guard.verify_final()
        return ImportSummary(
            run_id,
            receipt.sha256,
            imported,
            {
                "research_labels_missing_job": missing,
                "research_label_confidence_missing_job": confidence_quarantine,
            },
        )
    except Exception as exc:
        if run_id is not None:
            _mark_run_failed(pg, run_id, exc)
        else:
            pg.rollback()
        raise
    finally:
        try:
            if controller_active:
                _release_import_context(pg, receipt.sha256, lock_acquired=lock_acquired)
        finally:
            guard.close()


def _started_run_id(pg, source_sha256: str, run_key: str) -> int:
    with pg.cursor() as cur:
        cur.execute(
            """SELECT run.migration_run_id
               FROM brain_migration_runs run
               JOIN brain_migration_sources source
                 ON source.migration_source_id=run.migration_source_id
                AND source.source_namespace=run.source_namespace
               WHERE run.source_namespace=%s AND source.source_fingerprint=%s AND run.run_key=%s""",
            (SOURCE_NAMESPACE, source_sha256, run_key),
        )
        row = cur.fetchone()
        if row is None:
            raise BrainImportError("migration run does not exist; load the sealed source before finalizing")
        run_id = int(row["migration_run_id"])
        cur.execute(
            "SELECT event_type FROM brain_migration_run_events WHERE migration_run_id=%s "
            "AND source_namespace=%s ORDER BY migration_run_event_id DESC LIMIT 1",
            (run_id, SOURCE_NAMESPACE),
        )
        head = cur.fetchone()
    if head is None or head["event_type"] != "started":
        raise BrainImportError("migration run must have a started head before parity finalization")
    return run_id


def _parity_metadata(parity_results: Mapping[str, ParityResult]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "source_count": result.source_count,
            "target_count": result.target_count,
            "source_hash": result.source_hash,
            "target_hash": result.target_hash,
            "mismatch_count": result.mismatch_count,
            "unresolved_count": result.unresolved_count,
            "passed": result.passed,
        }
        for name, result in sorted(parity_results.items())
    }


def _validated_terminal_parity(value: Any) -> dict[str, dict[str, Any]]:
    expected_tables = {table.name for table in SOURCE_TABLES}
    if not isinstance(value, dict) or set(value) != expected_tables:
        raise BrainImportError("terminal recovery parity metadata is incomplete")
    required_fields = {
        "source_count",
        "target_count",
        "source_hash",
        "target_hash",
        "mismatch_count",
        "unresolved_count",
        "passed",
    }
    normalized: dict[str, dict[str, Any]] = {}
    for name in sorted(expected_tables):
        record = value[name]
        if not isinstance(record, dict) or set(record) != required_fields:
            raise BrainImportError(f"terminal recovery parity metadata is invalid for {name}")
        counts = (
            record["source_count"],
            record["target_count"],
            record["mismatch_count"],
            record["unresolved_count"],
        )
        if any(type(item) is not int or item < 0 for item in counts):
            raise BrainImportError(f"terminal recovery parity counts are invalid for {name}")
        source_hash = record["source_hash"]
        target_hash = record["target_hash"]
        if any(
            not isinstance(item, str)
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in (source_hash, target_hash)
        ):
            raise BrainImportError(f"terminal recovery parity hashes are invalid for {name}")
        if (
            record["passed"] is not True
            or record["source_count"] != record["target_count"]
            or source_hash != target_hash
            or record["mismatch_count"] != 0
            or record["unresolved_count"] != 0
        ):
            raise BrainImportError(f"terminal recovery parity is not clean for {name}")
        normalized[name] = dict(record)
    return normalized


def _is_canonical_uuid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value
    except (ValueError, AttributeError):
        return False


def _validated_destination_binding(value: Any) -> dict[str, Any]:
    required = {"database", "systemIdentifier", "databaseOid", "databaseIncarnationId"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise BrainImportError("destination binding is incomplete")
    normalized = dict(value)
    for name in ("database", "systemIdentifier", "databaseOid"):
        if not isinstance(normalized[name], str) or not normalized[name]:
            raise BrainImportError(f"destination binding {name} is invalid")
    # A durable database incarnation is mandatory: the terminal write and every resumed recovery
    # must bind to the exact destination singleton, never an absent or malformed incarnation.
    if not _is_canonical_uuid(normalized["databaseIncarnationId"]):
        raise BrainImportError("destination binding databaseIncarnationId must be a canonical UUID")
    return normalized


def recover_finalized_sqlite_to_postgres_import(
    pg,
    *,
    expected_sha256: str,
    run_key: str,
    expected_destination_binding: Mapping[str, Any],
) -> ImportSummary:
    """Reconstruct a finalized result exclusively from its durable terminal event."""
    with pg.cursor() as cur:
        cur.execute(
            """SELECT run.migration_run_id,event.migration_run_event_id,event.event_type,event.metadata,
                      source.source_fingerprint
               FROM brain_migration_runs run
               JOIN brain_migration_sources source
                 ON source.migration_source_id=run.migration_source_id
                AND source.source_namespace=run.source_namespace
               JOIN LATERAL (
                   SELECT migration_run_event_id,event_type,metadata
                   FROM brain_migration_run_events
                   WHERE migration_run_id=run.migration_run_id
                     AND source_namespace=run.source_namespace
                   ORDER BY migration_run_event_id DESC LIMIT 1
               ) event ON TRUE
               WHERE run.source_namespace=%s AND source.source_fingerprint=%s AND run.run_key=%s""",
            (SOURCE_NAMESPACE, expected_sha256, run_key),
        )
        row = cur.fetchone()
    if row is None or row["event_type"] != "completed":
        raise BrainImportError("terminal recovery requires an existing completed migration run")
    if row["source_fingerprint"] != expected_sha256:
        raise BrainImportError("terminal recovery source fingerprint does not match")
    metadata = row["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != {
        "source_sha256",
        "independent_parity",
        "destination_binding",
    }:
        raise BrainImportError("terminal recovery metadata is invalid")
    if metadata["source_sha256"] != expected_sha256:
        raise BrainImportError("terminal recovery metadata source fingerprint does not match")
    parity = _validated_terminal_parity(metadata["independent_parity"])
    durable_destination = _validated_destination_binding(metadata["destination_binding"])
    expected_destination = _validated_destination_binding(expected_destination_binding)
    if durable_destination != expected_destination:
        raise BrainImportError("terminal recovery destination binding does not match")
    return ImportSummary(
        migration_run_id=int(row["migration_run_id"]),
        source_sha256=expected_sha256,
        imported={name: int(result["target_count"]) for name, result in parity.items()},
        quarantined={},
        finalized=True,
        parity=parity,
        terminal_event_id=int(row["migration_run_event_id"]),
        recovered=True,
    )


def finalize_sqlite_to_postgres_import(
    pg,
    sqlite_path: str | Path,
    *,
    expected_sha256: str,
    run_key: str,
    destination_binding: Mapping[str, Any],
) -> ImportSummary:
    """Compute independent parity and finalize a loaded run only when it is clean."""
    receipt = capture_sqlite_source_receipt(sqlite_path)
    durable_destination = _validated_destination_binding(destination_binding)
    if receipt.sha256 != expected_sha256:
        raise BrainImportError(f"source SHA mismatch: expected {expected_sha256}, got {receipt.sha256}")
    guard = SQLiteSourceGuard(sqlite_path, expected_sha256)
    source = guard.connection
    source.row_factory = sqlite3.Row
    run_id: int | None = None
    controller_active = False
    lock_acquired = False
    try:
        _activate_controller(pg)
        controller_active = True
        _acquire_import_lock(pg, receipt.sha256)
        lock_acquired = True
        run_id = _started_run_id(pg, receipt.sha256, run_key)
        # Import locally to keep the projection adapter independent from the loader.
        from applypilot.brain.parity_adapters import compute_import_parity

        parity_results = compute_import_parity(pg, source, source_fingerprint=receipt.sha256)
        failed_parity = [name for name, result in parity_results.items() if not result.passed]
        if failed_parity:
            raise BrainImportError("independent parity failed: " + ", ".join(failed_parity))
        _record_completed_batches(
            pg,
            source,
            receipt,
            run_id,
            parity_results,
            recheck_source=guard.recheck,
        )
        with pg.cursor() as cur:
            cur.execute(
                "SELECT migration_run_event_id,event_type FROM brain_migration_run_events "
                "WHERE migration_run_id=%s AND source_namespace=%s "
                "ORDER BY migration_run_event_id DESC LIMIT 1",
                (run_id, SOURCE_NAMESPACE),
            )
            head = cur.fetchone()
            if head is None or head["event_type"] != "started":
                raise BrainImportError("migration run lost its started head during finalization")
            parity_metadata = _parity_metadata(parity_results)
            cur.execute(
                """INSERT INTO brain_migration_run_events
                   (migration_run_id,source_namespace,event_type,actor_id,metadata,supersedes_run_event_id)
                   VALUES (%s,%s,'completed','sqlite-to-postgres-importer',%s,%s)
                   RETURNING migration_run_event_id""",
                (
                    run_id,
                    SOURCE_NAMESPACE,
                    Jsonb(
                        {
                            "source_sha256": receipt.sha256,
                            "independent_parity": parity_metadata,
                            "destination_binding": durable_destination,
                        }
                    ),
                    head["migration_run_event_id"],
                ),
            )
            terminal_event_id = int(cur.fetchone()["migration_run_event_id"])
        _commit_terminal_phase(
            pg,
            recheck_source=guard.recheck,
            verify_final_source=guard.verify_final,
        )
        return ImportSummary(
            migration_run_id=run_id,
            source_sha256=receipt.sha256,
            imported={name: result.target_count for name, result in parity_results.items()},
            quarantined={},
            finalized=True,
            parity=parity_metadata,
            terminal_event_id=terminal_event_id,
        )
    except Exception as exc:
        if run_id is not None:
            _mark_run_failed(pg, run_id, exc)
        else:
            pg.rollback()
        raise
    finally:
        try:
            if controller_active:
                _release_import_context(pg, receipt.sha256, lock_acquired=lock_acquired)
        finally:
            guard.close()
