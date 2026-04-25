import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from applypilot.database import init_db
from applypilot.scoring import cover_letter
from applypilot.scoring.validator import validate_cover_letter


class CoverLetterValidationTests(unittest.TestCase):
    def test_empty_cover_letter_is_invalid(self):
        result = validate_cover_letter("")

        self.assertFalse(result["passed"])
        self.assertIn("empty", " ".join(result["errors"]).lower())

    def test_too_short_cover_letter_is_invalid(self):
        result = validate_cover_letter("Dear Hiring Manager,\nDarrin")

        self.assertFalse(result["passed"])
        self.assertIn("too short", " ".join(result["errors"]).lower())


class CoverLetterBatchTests(unittest.TestCase):
    def test_invalid_generated_letter_does_not_save_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "jobs.db"
            conn = init_db(db_path)
            conn.execute(
                """
                INSERT INTO jobs (
                    url, title, site, application_url, fit_score,
                    full_description, tailored_resume_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "https://example.com/job",
                    "Reporting Analyst",
                    "Direct Employer",
                    "https://example.com/job/apply",
                    9,
                    "Remote reporting work with dashboards and operational metrics.",
                    str(root / "resume.txt"),
                ),
            )
            conn.commit()
            (root / "resume.txt").write_text("resume text", encoding="utf-8")
            output_dir = root / "cover_letters"

            with patch.object(cover_letter, "get_connection", return_value=conn), patch.object(
                cover_letter, "load_profile", return_value={"personal": {"full_name": "Darrin Biby"}}
            ), patch.object(cover_letter, "RESUME_PATH", root / "resume.txt"), patch.object(
                cover_letter, "COVER_LETTER_DIR", output_dir
            ), patch.object(
                cover_letter, "generate_cover_letter", return_value=""
            ):
                result = cover_letter.run_cover_letters(min_score=7, limit=1)

            row = conn.execute(
                "SELECT cover_letter_path, cover_attempts FROM jobs WHERE url=?",
                ("https://example.com/job",),
            ).fetchone()
            conn.close()

            self.assertEqual(result["generated"], 0)
            self.assertEqual(result["errors"], 1)
            self.assertIsNone(row["cover_letter_path"])
            self.assertEqual(row["cover_attempts"], 1)
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
