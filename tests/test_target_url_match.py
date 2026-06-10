"""apply --url must select exactly the targeted job, never a different one.

Regression: the old target-url fallback stripped the query string to build
its LIKE pattern, but on indeed/linkedin the query string IS the job
identity (?jk=, currentJobId=). '%indeed.com/viewjob%' matched every indeed
job in the DB and LIMIT 1 applied to an arbitrary one.
"""
import applypilot.database as db
from applypilot.apply.launcher import acquire_job


def _seed(conn, url):
    conn.execute(
        "INSERT INTO jobs (url, title, site, tailored_resume_path, fit_score, "
        "apply_status) VALUES (?,?,?,?,?,?)",
        (url, "Engineer", "indeed", "/tmp/r.txt", 8, "pending"),
    )
    conn.commit()


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    return db.init_db(db_path=tmp_path / "test.db")


def test_target_url_query_string_is_job_identity(tmp_path, monkeypatch):
    conn = _setup(tmp_path, monkeypatch)
    _seed(conn, "https://www.indeed.com/viewjob?jk=aaaa1111")
    _seed(conn, "https://www.indeed.com/viewjob?jk=bbbb2222")
    job = acquire_job(target_url="https://www.indeed.com/viewjob?jk=bbbb2222")
    assert job is not None
    assert job["url"].endswith("jk=bbbb2222")


def test_target_url_unknown_job_returns_none(tmp_path, monkeypatch):
    conn = _setup(tmp_path, monkeypatch)
    _seed(conn, "https://www.indeed.com/viewjob?jk=aaaa1111")
    assert acquire_job(target_url="https://www.indeed.com/viewjob?jk=zzzz9999") is None


def test_target_url_tolerates_scheme_and_slash_variants(tmp_path, monkeypatch):
    conn = _setup(tmp_path, monkeypatch)
    _seed(conn, "https://boards.greenhouse.io/acme/jobs/123")
    job = acquire_job(target_url="http://boards.greenhouse.io/acme/jobs/123/")
    assert job is not None
    assert job["url"] == "https://boards.greenhouse.io/acme/jobs/123"
