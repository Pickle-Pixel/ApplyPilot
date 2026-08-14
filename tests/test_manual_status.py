"""Regression tests for manual application status updates."""

import sqlite3

import pytest
from typer.testing import CliRunner

from applypilot import cli
from applypilot.apply import launcher


@pytest.fixture
def jobs_db(monkeypatch: pytest.MonkeyPatch) -> sqlite3.Connection:
    """Provide the minimal jobs schema used by ``mark_job``."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE jobs (
            url TEXT PRIMARY KEY,
            application_url TEXT,
            applied_at TEXT,
            apply_status TEXT,
            apply_error TEXT,
            apply_attempts INTEGER DEFAULT 0,
            agent_id TEXT
        )
        """
    )
    monkeypatch.setattr(launcher, "get_connection", lambda: conn)
    yield conn
    conn.close()


def _insert_job(conn: sqlite3.Connection, url: str, application_url: str) -> None:
    conn.execute(
        "INSERT INTO jobs (url, application_url) VALUES (?, ?)",
        (url, application_url),
    )
    conn.commit()


def test_mark_job_accepts_canonical_url(jobs_db: sqlite3.Connection) -> None:
    canonical_url = "https://board.example/jobs/123"
    _insert_job(jobs_db, canonical_url, "https://ats.example/apply/123")

    updated_url = launcher.mark_job(canonical_url, "applied")

    row = jobs_db.execute(
        "SELECT apply_status, applied_at FROM jobs WHERE url = ?", (canonical_url,)
    ).fetchone()
    assert updated_url == canonical_url
    assert row["apply_status"] == "applied"
    assert row["applied_at"] is not None


def test_mark_job_resolves_application_url(jobs_db: sqlite3.Connection) -> None:
    canonical_url = "https://board.example/jobs/123"
    application_url = "https://ats.example/apply/123"
    _insert_job(jobs_db, canonical_url, application_url)

    updated_url = launcher.mark_job(application_url, "failed", reason="manual email")

    row = jobs_db.execute(
        "SELECT apply_status, apply_error, apply_attempts FROM jobs WHERE url = ?",
        (canonical_url,),
    ).fetchone()
    assert updated_url == canonical_url
    assert row["apply_status"] == "failed"
    assert row["apply_error"] == "manual email"
    assert row["apply_attempts"] == 99


def test_mark_job_rejects_unknown_url(jobs_db: sqlite3.Connection) -> None:
    with pytest.raises(LookupError, match="No job found"):
        launcher.mark_job("https://ats.example/apply/missing", "applied")


def test_mark_job_rejects_ambiguous_application_url(jobs_db: sqlite3.Connection) -> None:
    application_url = "https://ats.example/apply/shared"
    _insert_job(jobs_db, "https://board.example/jobs/123", application_url)
    _insert_job(jobs_db, "https://board.example/jobs/456", application_url)

    with pytest.raises(ValueError, match="matches multiple jobs"):
        launcher.mark_job(application_url, "applied")

    statuses = jobs_db.execute("SELECT apply_status FROM jobs ORDER BY url").fetchall()
    assert [row["apply_status"] for row in statuses] == [None, None]


def test_mark_job_rejects_invalid_status(jobs_db: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="status must be"):
        launcher.mark_job("https://board.example/jobs/123", "previewed")


def test_mark_applied_cli_exits_when_no_job_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI must not print a success message after a zero-row update."""
    monkeypatch.setattr(cli, "_bootstrap", lambda: None)

    def missing_job(*_args: object, **_kwargs: object) -> str:
        raise LookupError("No job found for URL: https://example.invalid/missing")

    monkeypatch.setattr(launcher, "mark_job", missing_job)

    result = CliRunner().invoke(
        cli.app,
        ["apply", "--mark-applied", "https://example.invalid/missing"],
    )

    assert result.exit_code == 1
    assert "Could not mark job as applied" in result.output
    assert "Marked as applied" not in result.output
