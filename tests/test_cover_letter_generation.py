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
    VALID_LETTER = """Dear Hiring Manager,

I support documentation review, SQL checks, and Excel reporting that keep remote operations accurate.

At NCO Group, I handled workforce scheduling, real-time adherence, and variance tracking across three call center locations. At Jack Henry, I supported ATM and POS troubleshooting for more than 40 bank clients with careful documentation and escalation. That mix matches your reporting, documentation, and quality review needs.

Happy to walk through any of this in more detail.

Darrin"""

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

    def test_blocked_jobs_are_skipped_when_generating_covers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "jobs.db"
            conn = init_db(db_path)
            rows = [
                (
                    "https://example.com/blocked",
                    "Blocked Reporting Analyst",
                    "Direct Employer",
                    "https://example.com/blocked/apply",
                    10,
                    "Remote reporting work with dashboards and operational metrics.",
                    str(root / "blocked_resume.txt"),
                    "failed",
                ),
                (
                    "https://example.com/clean",
                    "Clean Reporting Analyst",
                    "Direct Employer",
                    "https://example.com/clean/apply",
                    9,
                    "Remote reporting work with dashboards and operational metrics.",
                    str(root / "clean_resume.txt"),
                    None,
                ),
            ]
            conn.executemany(
                """
                INSERT INTO jobs (
                    url, title, site, application_url, fit_score,
                    full_description, tailored_resume_path, apply_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            (root / "runtime_resume.txt").write_text("resume text", encoding="utf-8")
            output_dir = root / "cover_letters"

            with patch.object(cover_letter, "get_connection", return_value=conn), patch.object(
                cover_letter, "load_profile", return_value={"personal": {"full_name": "Darrin Biby"}}
            ), patch.object(cover_letter, "RESUME_PATH", root / "runtime_resume.txt"), patch.object(
                cover_letter, "COVER_LETTER_DIR", output_dir
            ), patch.object(
                cover_letter, "generate_cover_letter", return_value=self.VALID_LETTER
            ):
                result = cover_letter.run_cover_letters(min_score=7, limit=5)

            blocked = conn.execute(
                "SELECT cover_letter_path, cover_attempts FROM jobs WHERE url=?",
                ("https://example.com/blocked",),
            ).fetchone()
            clean = conn.execute(
                "SELECT cover_letter_path, cover_attempts FROM jobs WHERE url=?",
                ("https://example.com/clean",),
            ).fetchone()
            conn.close()

            self.assertEqual(result["generated"], 1)
            self.assertEqual(result["errors"], 0)
            self.assertIsNone(blocked["cover_letter_path"])
            self.assertEqual(blocked["cover_attempts"], 0)
            self.assertIsNotNone(clean["cover_letter_path"])
            self.assertEqual(clean["cover_attempts"], 1)

    def test_cover_generation_can_filter_to_primary_lane(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "jobs.db"
            conn = init_db(db_path)
            rows = [
                (
                    "https://example.com/flex",
                    "Flex Reporting Analyst",
                    "Direct Employer",
                    "https://example.com/flex/apply",
                    10,
                    "Remote reporting work.",
                    str(root / "flex_resume.txt"),
                    "flex",
                ),
                (
                    "https://example.com/primary",
                    "Primary Reporting Analyst",
                    "Direct Employer",
                    "https://example.com/primary/apply",
                    9,
                    "Remote reporting work.",
                    str(root / "primary_resume.txt"),
                    "primary",
                ),
            ]
            conn.executemany(
                """
                INSERT INTO jobs (
                    url, title, site, application_url, fit_score,
                    full_description, tailored_resume_path, job_lane
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            (root / "runtime_resume.txt").write_text("resume text", encoding="utf-8")
            output_dir = root / "cover_letters"

            with patch.object(cover_letter, "get_connection", return_value=conn), patch.object(
                cover_letter, "load_profile", return_value={"personal": {"full_name": "Darrin Biby"}}
            ), patch.object(cover_letter, "RESUME_PATH", root / "runtime_resume.txt"), patch.object(
                cover_letter, "COVER_LETTER_DIR", output_dir
            ), patch.object(
                cover_letter, "generate_cover_letter", return_value=self.VALID_LETTER
            ):
                result = cover_letter.run_cover_letters(min_score=7, limit=5, lane="primary")

            flex = conn.execute(
                "SELECT cover_letter_path, cover_attempts FROM jobs WHERE url=?",
                ("https://example.com/flex",),
            ).fetchone()
            primary = conn.execute(
                "SELECT cover_letter_path, cover_attempts FROM jobs WHERE url=?",
                ("https://example.com/primary",),
            ).fetchone()
            conn.close()

            self.assertEqual(result["generated"], 1)
            self.assertIsNone(flex["cover_letter_path"])
            self.assertEqual(flex["cover_attempts"], 0)
            self.assertIsNotNone(primary["cover_letter_path"])
            self.assertEqual(primary["cover_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
