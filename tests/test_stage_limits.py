"""F12: sequential pipeline must not silently cap tailor/cover at 20."""
import applypilot.database as db


def test_get_jobs_by_stage_limit_zero_is_unlimited(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    conn = db.init_db()
    for i in range(25):
        conn.execute(
            "INSERT INTO jobs (url, title, fit_score, full_description, tailored_resume_path) "
            "VALUES (?,?,?,?,NULL)",
            (f"https://example.com/{i}", "Engineer", 8, "x"),
        )
    conn.commit()

    all_jobs = db.get_jobs_by_stage(conn=conn, stage="pending_tailor", min_score=7, limit=0)
    assert len(all_jobs) == 25
    capped = db.get_jobs_by_stage(conn=conn, stage="pending_tailor", min_score=7, limit=10)
    assert len(capped) == 10
