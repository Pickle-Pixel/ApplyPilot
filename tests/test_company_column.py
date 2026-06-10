"""F7: the real company name is stored and migrated, not discarded."""
import sqlite3

import pandas as pd

import applypilot.database as db
from applypilot.discovery.jobspy import store_jobspy_results


def test_store_jobspy_persists_company(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    conn = db.init_db()
    df = pd.DataFrame([{
        "job_url": "https://example.com/job1",
        "title": "Engineer",
        "company": "Acme Corp",
        "location": "Remote",
        "site": "linkedin",
    }])
    store_jobspy_results(conn, df, "linkedin")
    row = conn.execute("SELECT company FROM jobs WHERE url = ?",
                       ("https://example.com/job1",)).fetchone()
    assert row["company"] == "Acme Corp"


def test_ensure_columns_adds_company_to_old_db(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    # Simulate a pre-F7 schema with no company column.
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE jobs (url TEXT PRIMARY KEY, title TEXT, site TEXT)")
    old.commit()
    old.close()

    monkeypatch.setattr(db, "DB_PATH", path)
    conn = db.get_connection(path)
    db.ensure_columns(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    assert "company" in cols
