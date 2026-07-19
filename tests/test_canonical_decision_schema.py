from __future__ import annotations

import sqlite3

import pytest

from applypilot import database


_NOW = "2026-07-10T00:00:00Z"


def _insert_policy(conn, policy_version: str = "policy-v1", lane: str = "ats", status: str = "draft") -> None:
    conn.execute(
        """
        INSERT INTO decision_policy_versions(policy_version, lane, status, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (policy_version, lane, status, _NOW),
    )


def _insert_decision(
    conn,
    decision_id: str,
    *,
    job_url: str = "https://example.test/job",
    policy_version: str = "policy-v1",
    lane: str = "ats",
    input_hash: str = "input-hash",
) -> None:
    conn.execute(
        """
        INSERT INTO job_decisions(
            decision_id, job_url, policy_version, lane, qualification_verdict,
            action, input_hash, created_at
        ) VALUES (?, ?, ?, ?, 'qualified', 'apply', ?, ?)
        """,
        (decision_id, job_url, policy_version, lane, input_hash, _NOW),
    )


def _create_previous_canonical_schema(db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE jobs (
            url TEXT PRIMARY KEY,
            title TEXT
        );
        CREATE TABLE email_events (
            message_id TEXT PRIMARY KEY,
            job_url TEXT,
            occurred_at TEXT NOT NULL,
            stage TEXT NOT NULL,
            scanned_at TEXT NOT NULL,
            FOREIGN KEY(job_url) REFERENCES jobs(url)
        );
        CREATE TABLE decision_policy_versions (
            policy_version TEXT PRIMARY KEY,
            lane TEXT NOT NULL CHECK(lane IN ('ats', 'linkedin')),
            status TEXT NOT NULL CHECK(status IN ('draft', 'validated', 'canary', 'active', 'retired')),
            created_at TEXT NOT NULL
        );
        CREATE TABLE job_decisions (
            decision_id TEXT PRIMARY KEY,
            job_url TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            lane TEXT NOT NULL CHECK(lane IN ('ats', 'linkedin')),
            qualification_verdict TEXT NOT NULL,
            action TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            UNIQUE(job_url, policy_version, input_hash),
            FOREIGN KEY(job_url) REFERENCES jobs(url),
            FOREIGN KEY(policy_version) REFERENCES decision_policy_versions(policy_version)
        );
        CREATE TABLE reviewed_outcomes (
            event_id TEXT NOT NULL,
            job_url TEXT NOT NULL,
            review_status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(event_id, job_url),
            FOREIGN KEY(event_id) REFERENCES email_events(message_id),
            FOREIGN KEY(job_url) REFERENCES jobs(url)
        );
        """
    )
    conn.execute(
        "INSERT INTO jobs(url, title) VALUES ('https://example.test/job', 'Existing role')"
    )
    conn.execute(
        """
        INSERT INTO email_events(message_id, occurred_at, stage, scanned_at)
        VALUES ('event-1', ?, 'screen', ?)
        """,
        (_NOW, _NOW),
    )
    conn.execute(
        """
        INSERT INTO decision_policy_versions(policy_version, lane, status, created_at)
        VALUES ('policy-v1', 'ats', 'draft', ?)
        """,
        (_NOW,),
    )
    conn.commit()
    return conn


def test_init_db_creates_canonical_decision_schema(tmp_path) -> None:
    conn = database.init_db(tmp_path / "brain.db")

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {"decision_policy_versions", "job_decisions", "reviewed_outcomes"} <= tables

    decision_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(job_decisions)").fetchall()
    }
    assert "decision_snapshot_json" in decision_columns

    job_columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    assert {
        "canonical_decision_id",
        "canonical_policy_version",
        "canonical_action",
        "canonical_score",
        "canonical_decided_at",
    } <= job_columns

    conn.execute("INSERT INTO jobs(url) VALUES ('https://example.test/unscored')")
    projection = conn.execute(
        """
        SELECT canonical_decision_id, canonical_policy_version, canonical_action,
               canonical_score, canonical_decided_at
        FROM jobs
        WHERE url = 'https://example.test/unscored'
        """
    ).fetchone()
    assert tuple(projection) == (None, None, None, None, None)


def test_init_db_enables_foreign_keys_and_is_idempotent(tmp_path) -> None:
    db_path = tmp_path / "brain.db"

    first = database.init_db(db_path)
    second = database.init_db(db_path)

    assert first is second
    assert second.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert {
        "decision_policy_versions",
        "job_decisions",
        "reviewed_outcomes",
    } <= {
        row[0]
        for row in second.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }

    database.close_connection(db_path)
    reopened = database.init_db(db_path)
    assert reopened is not first
    assert reopened.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_init_db_migrates_existing_jobs_database_additively(tmp_path) -> None:
    db_path = tmp_path / "existing.db"
    legacy = sqlite3.connect(db_path)
    legacy.execute("CREATE TABLE jobs(url TEXT PRIMARY KEY, title TEXT)")
    legacy.execute(
        "INSERT INTO jobs(url, title) VALUES ('https://example.test/legacy', 'Legacy role')"
    )
    legacy.commit()
    legacy.close()

    conn = database.init_db(db_path)

    columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    assert {
        "canonical_decision_id",
        "canonical_policy_version",
        "canonical_action",
        "canonical_score",
        "canonical_decided_at",
    } <= columns
    assert conn.execute(
        "SELECT title FROM jobs WHERE url = 'https://example.test/legacy'"
    ).fetchone()[0] == "Legacy role"
    assert {"decision_policy_versions", "job_decisions", "reviewed_outcomes"} <= {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }


def test_init_db_adds_decision_snapshot_without_rewriting_existing_decisions(tmp_path) -> None:
    db_path = tmp_path / "previous-canonical.db"
    previous = _create_previous_canonical_schema(db_path)
    _insert_decision(previous, "preserved-decision")
    previous.commit()
    previous.close()

    conn = database.init_db(db_path)

    columns = {
        row[1]: row for row in conn.execute("PRAGMA table_info(job_decisions)").fetchall()
    }
    assert columns["decision_snapshot_json"][2].upper() == "TEXT"
    assert columns["decision_snapshot_json"][3] == 0
    assert columns["decision_snapshot_json"][4] is None
    assert conn.execute(
        "SELECT decision_snapshot_json FROM job_decisions WHERE decision_id = ?",
        ("preserved-decision",),
    ).fetchone()[0] is None

    snapshot = (
        '{"recommendationLane":"core_fit","requiredCoverage":10,'
        '"schemaVersion":"applypilot_canonical_decision_snapshot_v1"}'
    )
    conn.execute(
        "UPDATE job_decisions SET decision_snapshot_json = ? WHERE decision_id = ?",
        (snapshot, "preserved-decision"),
    )
    conn.commit()
    database.close_connection(db_path)

    reopened = database.init_db(db_path)
    assert reopened.execute(
        "SELECT decision_snapshot_json FROM job_decisions WHERE decision_id = ?",
        ("preserved-decision",),
    ).fetchone()[0] == snapshot


def test_upgrade_installs_lane_consistency_triggers(tmp_path) -> None:
    db_path = tmp_path / "previous.db"
    previous = _create_previous_canonical_schema(db_path)
    previous.close()

    conn = database.init_db(db_path)
    triggers = {
        row[0]
        for row in conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'trigger'
              AND tbl_name IN ('job_decisions', 'decision_policy_versions')
            """
        ).fetchall()
    }
    assert {
        "trg_job_decisions_policy_lane_insert",
        "trg_job_decisions_policy_lane_update",
        "trg_decision_policy_versions_lane_update",
    } <= triggers

    with pytest.raises(sqlite3.IntegrityError, match="decision policy lane mismatch"):
        _insert_decision(conn, "insert-mismatch", lane="linkedin")

    _insert_decision(conn, "valid-decision")
    with pytest.raises(sqlite3.IntegrityError, match="decision policy lane mismatch"):
        conn.execute(
            "UPDATE job_decisions SET lane = 'linkedin' WHERE decision_id = 'valid-decision'"
        )

    with pytest.raises(sqlite3.IntegrityError, match="policy lane has decisions"):
        conn.execute(
            "UPDATE decision_policy_versions SET lane = 'linkedin' WHERE policy_version = 'policy-v1'"
        )

    _insert_policy(conn, "childless-policy", lane="ats")
    conn.execute(
        "UPDATE decision_policy_versions SET lane = 'linkedin' WHERE policy_version = 'childless-policy'"
    )
    assert conn.execute(
        "SELECT lane FROM decision_policy_versions WHERE policy_version = 'childless-policy'"
    ).fetchone()[0] == "linkedin"


@pytest.mark.parametrize(
    ("corruption", "expected"),
    (
        ("orphan_decision", "foreign key violation: job_decisions"),
        ("orphan_outcome", "foreign key violation: reviewed_outcomes"),
        ("lane_mismatch", "lane mismatch: decision_id=lane-mismatch"),
    ),
)
def test_upgrade_rejects_preexisting_canonical_integrity_violations(
    tmp_path, corruption: str, expected: str
) -> None:
    db_path = tmp_path / f"{corruption}.db"
    previous = _create_previous_canonical_schema(db_path)
    if corruption == "orphan_decision":
        previous.execute(
            """
            INSERT INTO job_decisions(
                decision_id, job_url, policy_version, lane, qualification_verdict,
                action, input_hash, created_at
            ) VALUES (
                'orphan-decision', 'https://example.test/missing', 'policy-v1',
                'ats', 'qualified', 'review', 'orphan-hash', ?
            )
            """,
            (_NOW,),
        )
    elif corruption == "orphan_outcome":
        previous.execute(
            """
            INSERT INTO reviewed_outcomes(event_id, job_url, review_status, created_at)
            VALUES ('missing-event', 'https://example.test/job', 'needs_review', ?)
            """,
            (_NOW,),
        )
    else:
        previous.execute(
            """
            INSERT INTO job_decisions(
                decision_id, job_url, policy_version, lane, qualification_verdict,
                action, input_hash, created_at
            ) VALUES (
                'lane-mismatch', 'https://example.test/job', 'policy-v1',
                'linkedin', 'qualified', 'review', 'lane-hash', ?
            )
            """,
            (_NOW,),
        )
    previous.commit()
    previous.close()

    with pytest.raises(RuntimeError, match=expected):
        database.init_db(db_path)
    database.close_connection(db_path)


def test_orphan_decision_and_reviewed_outcome_foreign_keys_fail(tmp_path) -> None:
    conn = database.init_db(tmp_path / "brain.db")
    conn.execute("INSERT INTO jobs(url) VALUES ('https://example.test/job')")
    conn.execute(
        """
        INSERT INTO email_events(message_id, occurred_at, stage, scanned_at)
        VALUES ('event-1', ?, 'screen', ?)
        """,
        (_NOW, _NOW),
    )
    _insert_policy(conn)

    with pytest.raises(sqlite3.IntegrityError):
        _insert_decision(
            conn,
            "orphan-job",
            job_url="https://example.test/missing",
            input_hash="orphan-job",
        )
    with pytest.raises(sqlite3.IntegrityError):
        _insert_decision(
            conn,
            "orphan-policy",
            policy_version="missing-policy",
            input_hash="orphan-policy",
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO reviewed_outcomes(event_id, job_url, review_status, created_at)
            VALUES ('missing-event', 'https://example.test/job', 'needs_review', ?)
            """,
            (_NOW,),
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO reviewed_outcomes(event_id, job_url, review_status, created_at)
            VALUES ('event-1', 'https://example.test/missing', 'needs_review', ?)
            """,
            (_NOW,),
        )


def test_decision_lane_must_match_policy_lane(tmp_path) -> None:
    conn = database.init_db(tmp_path / "brain.db")
    conn.execute("INSERT INTO jobs(url) VALUES ('https://example.test/job')")
    _insert_policy(conn, lane="ats")

    with pytest.raises(sqlite3.IntegrityError):
        _insert_decision(conn, "lane-mismatch", lane="linkedin")


def test_fresh_schema_protects_policy_lane_updates_with_decisions(tmp_path) -> None:
    conn = database.init_db(tmp_path / "brain.db")
    conn.execute("INSERT INTO jobs(url) VALUES ('https://example.test/job')")
    _insert_policy(conn)
    _insert_decision(conn, "decision-1")

    with pytest.raises(sqlite3.IntegrityError, match="policy lane has decisions"):
        conn.execute(
            "UPDATE decision_policy_versions SET lane = 'linkedin' WHERE policy_version = 'policy-v1'"
        )

    _insert_policy(conn, "childless-policy", lane="ats")
    conn.execute(
        "UPDATE decision_policy_versions SET lane = 'linkedin' WHERE policy_version = 'childless-policy'"
    )
    assert conn.execute(
        "SELECT lane FROM decision_policy_versions WHERE policy_version = 'childless-policy'"
    ).fetchone()[0] == "linkedin"


def test_decision_input_identity_is_unique(tmp_path) -> None:
    conn = database.init_db(tmp_path / "brain.db")
    conn.execute("INSERT INTO jobs(url) VALUES ('https://example.test/job')")
    _insert_policy(conn)
    _insert_decision(conn, "decision-1")

    with pytest.raises(sqlite3.IntegrityError):
        _insert_decision(conn, "decision-2")


def test_only_one_active_policy_is_allowed_per_lane(tmp_path) -> None:
    conn = database.init_db(tmp_path / "brain.db")
    _insert_policy(conn, "ats-active", lane="ats", status="active")

    with pytest.raises(sqlite3.IntegrityError):
        _insert_policy(conn, "second-ats-active", lane="ats", status="active")

    _insert_policy(conn, "linkedin-active", lane="linkedin", status="active")


def test_canonical_schema_has_required_indexes(tmp_path) -> None:
    conn = database.init_db(tmp_path / "brain.db")
    indexes = {
        row[0]
        for row in conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'index'
              AND tbl_name IN (
                  'decision_policy_versions', 'job_decisions', 'reviewed_outcomes'
              )
            """
        ).fetchall()
    }

    assert {
        "idx_decision_policy_versions_status_lane",
        "idx_decision_policy_versions_active_lane",
        "idx_decision_policy_versions_version_lane",
        "idx_job_decisions_job",
        "idx_job_decisions_policy",
        "idx_job_decisions_action",
        "idx_job_decisions_expires",
        "idx_reviewed_outcomes_job",
        "idx_reviewed_outcomes_status",
    } <= indexes


def test_canonical_schema_constraints_reject_invalid_values(tmp_path) -> None:
    conn = database.init_db(tmp_path / "brain.db")
    conn.execute("INSERT INTO jobs(url) VALUES ('https://example.test/job')")
    conn.execute(
        """
        INSERT INTO email_events(message_id, occurred_at, stage, scanned_at)
        VALUES ('event-1', '2026-07-10T00:00:00Z', 'screen', '2026-07-10T00:00:00Z')
        """
    )
    conn.execute(
        """
        INSERT INTO decision_policy_versions(policy_version, lane, status, created_at)
        VALUES ('policy-v1', 'ats', 'draft', '2026-07-10T00:00:00Z')
        """
    )

    invalid_statements = (
        """
        INSERT INTO decision_policy_versions(policy_version, lane, status, created_at)
        VALUES ('bad-status', 'ats', 'wrong', '2026-07-10T00:00:00Z')
        """,
        """
        INSERT INTO decision_policy_versions(policy_version, lane, status, created_at)
        VALUES ('bad-lane', 'email', 'draft', '2026-07-10T00:00:00Z')
        """,
        """
        INSERT INTO job_decisions(
            decision_id, job_url, policy_version, lane, qualification_verdict,
            action, input_hash, created_at
        ) VALUES (
            'bad-decision-lane', 'https://example.test/job', 'policy-v1', 'email',
            'qualified', 'apply', 'hash-lane', '2026-07-10T00:00:00Z'
        )
        """,
        """
        INSERT INTO job_decisions(
            decision_id, job_url, policy_version, lane, qualification_verdict,
            action, input_hash, created_at
        ) VALUES (
            'bad-verdict', 'https://example.test/job', 'policy-v1', 'ats',
            'maybe', 'apply', 'hash-verdict', '2026-07-10T00:00:00Z'
        )
        """,
        """
        INSERT INTO job_decisions(
            decision_id, job_url, policy_version, lane, qualification_verdict,
            action, input_hash, created_at
        ) VALUES (
            'bad-action', 'https://example.test/job', 'policy-v1', 'ats',
            'qualified', 'queue', 'hash-action', '2026-07-10T00:00:00Z'
        )
        """,
        """
        INSERT INTO reviewed_outcomes(
            event_id, job_url, review_status, created_at
        ) VALUES (
            'event-1', 'https://example.test/job', 'pending',
            '2026-07-10T00:00:00Z'
        )
        """,
    )

    for statement in invalid_statements:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(statement)
