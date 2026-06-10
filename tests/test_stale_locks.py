"""F13: stale in_progress locks are recovered at apply startup."""
import applypilot.database as db
from applypilot.apply.launcher import reset_stale_locks


def test_reset_stale_locks(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    conn = db.init_db()
    conn.execute(
        "INSERT INTO jobs (url, title, apply_status, agent_id) VALUES (?,?,?,?)",
        ("https://example.com/stuck", "Engineer", "in_progress", "worker-0"),
    )
    conn.commit()

    assert reset_stale_locks() == 1
    row = conn.execute("SELECT apply_status, agent_id FROM jobs WHERE url=?",
                       ("https://example.com/stuck",)).fetchone()
    assert row["apply_status"] is None
    assert row["agent_id"] is None
