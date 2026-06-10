"""F14: one flaky site must not abort the whole smart-extract stage."""
import applypilot.database as db
import applypilot.discovery.smartextract as se


def test_one_failing_site_does_not_abort(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")

    def fake_run_one_site(name, url):
        if name == "A":
            raise RuntimeError("network timeout")
        return {
            "name": "B", "status": "PASS", "strategy": "test",
            "total": 1, "titles": 1,
            "jobs": [{"url": "https://example.com/1", "title": "T",
                      "salary": None, "description": None, "location": "Remote"}],
        }

    monkeypatch.setattr(se, "_run_one_site", fake_run_one_site)
    stats = se._run_all(
        [{"name": "A", "url": "u"}, {"name": "B", "url": "v"}], [], [], workers=1)

    assert stats["errors"] == 1
    assert stats["passed"] == 1
    assert stats["total_new"] == 1
