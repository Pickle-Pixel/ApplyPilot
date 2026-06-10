"""F4: apply --url must select fresh (NULL) jobs and never re-apply."""
import applypilot.database as db
from applypilot.apply.launcher import acquire_job


def _seed(conn, url, status, applied_at=None):
    conn.execute(
        "INSERT INTO jobs (url, title, site, tailored_resume_path, fit_score, "
        "apply_status, applied_at) VALUES (?,?,?,?,?,?,?)",
        (url, "Engineer", "linkedin", "/tmp/r.txt", 8, status, applied_at),
    )
    conn.commit()


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    conn = db.init_db()
    return conn


def test_fresh_null_job_is_selected(tmp_path, monkeypatch):
    conn = _setup(tmp_path, monkeypatch)
    url = "https://example.com/job-null"
    _seed(conn, url, None)
    assert acquire_job(target_url=url) is not None


def test_failed_job_is_selected(tmp_path, monkeypatch):
    conn = _setup(tmp_path, monkeypatch)
    url = "https://example.com/job-failed"
    _seed(conn, url, "failed")
    assert acquire_job(target_url=url) is not None


def test_applied_job_is_not_reselected(tmp_path, monkeypatch):
    conn = _setup(tmp_path, monkeypatch)
    url = "https://example.com/job-applied"
    _seed(conn, url, "applied", applied_at="2026-01-01T00:00:00Z")
    assert acquire_job(target_url=url) is None
