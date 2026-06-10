"""F8: robust score parsing + never persist permanent zero on failure."""
import applypilot.database as db
import applypilot.scoring.scorer as scorer
from applypilot.scoring.scorer import _parse_score_response, run_scoring


def test_markdown_score_line():
    assert _parse_score_response("**SCORE:** 8\nKEYWORDS: a\nREASONING: b")["score"] == 8


def test_score_with_slash():
    assert _parse_score_response("Score: 7/10")["score"] == 7


def test_no_score_line_is_none():
    assert _parse_score_response("the model rambled with no verdict")["score"] is None


def test_unparseable_score_is_none():
    assert _parse_score_response("SCORE: N/A")["score"] is None


def test_plain_score():
    assert _parse_score_response("SCORE: 10")["score"] == 10


def _seed(conn, url):
    conn.execute(
        "INSERT INTO jobs (url, title, full_description) VALUES (?,?,?)",
        (url, "Engineer", "a long enough description " * 20),
    )
    conn.commit()


def test_failed_score_left_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(scorer, "RESUME_PATH", tmp_path / "resume.txt")
    (tmp_path / "resume.txt").write_text("my resume")
    conn = db.init_db()
    _seed(conn, "https://example.com/ok")
    _seed(conn, "https://example.com/fail")

    def fake_score(resume, job):
        if job["url"].endswith("fail"):
            return {"score": None, "keywords": "", "reasoning": "boom"}
        return {"score": 9, "keywords": "k", "reasoning": "r"}

    monkeypatch.setattr(scorer, "score_job", fake_score)
    out = run_scoring()

    assert out["scored"] == 1 and out["errors"] == 1
    ok = conn.execute("SELECT fit_score FROM jobs WHERE url=?",
                      ("https://example.com/ok",)).fetchone()
    bad = conn.execute("SELECT fit_score FROM jobs WHERE url=?",
                       ("https://example.com/fail",)).fetchone()
    assert ok["fit_score"] == 9          # committed
    assert bad["fit_score"] is None      # left pending, not a permanent 0


def test_pending_score_stage_still_includes_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(scorer, "RESUME_PATH", tmp_path / "resume.txt")
    (tmp_path / "resume.txt").write_text("my resume")
    conn = db.init_db()
    _seed(conn, "https://example.com/fail")
    monkeypatch.setattr(scorer, "score_job",
                        lambda r, j: {"score": None, "keywords": "", "reasoning": "x"})
    run_scoring()
    pending = db.get_jobs_by_stage(conn=conn, stage="pending_score", limit=0)
    assert any(j["url"] == "https://example.com/fail" for j in pending)
