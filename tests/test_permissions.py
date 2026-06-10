"""F5: secret-bearing files are written owner-only (0600)."""
import os
import stat

import applypilot.config as config
import applypilot.database as db


def test_write_private_text_is_owner_only(tmp_path):
    p = tmp_path / "secret.env"
    config.write_private_text(p, "GEMINI_API_KEY=abc")
    assert p.read_text() == "GEMINI_API_KEY=abc"
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600


def test_init_db_is_owner_only(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    assert stat.S_IMODE(os.stat(tmp_path / "test.db").st_mode) == 0o600
