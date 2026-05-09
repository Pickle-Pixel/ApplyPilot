import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from applypilot.apply import launcher
from applypilot.database import init_db


class ApplyLaneFilterTests(unittest.TestCase):
    def test_acquire_job_skips_flex_lane_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "jobs.db"
            conn = init_db(db_path)
            conn.execute(
                """
                INSERT INTO jobs (
                    url, title, site, application_url, tailored_resume_path, cover_letter_path,
                    fit_score, full_description, job_lane
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "https://example.com/flex",
                    "Remote Business Analyst",
                    "linkedin",
                    "https://example.com/flex/apply",
                    "C:/resume.pdf",
                    "C:/cover.pdf",
                    10,
                    "Turing platform role",
                    "flex",
                ),
            )
            conn.execute(
                """
                INSERT INTO jobs (
                    url, title, site, application_url, tailored_resume_path, cover_letter_path,
                    fit_score, full_description, job_lane
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "https://example.com/primary",
                    "Data Analyst I",
                    "indeed",
                    "https://example.com/primary/apply",
                    "C:/resume.pdf",
                    "C:/cover.pdf",
                    9,
                    "Direct employer role",
                    "primary",
                ),
            )
            conn.commit()

            with patch("applypilot.apply.launcher.get_connection", return_value=conn), patch(
                "applypilot.apply.launcher._load_blocked", return_value=(set(), [])
            ), patch("applypilot.apply.launcher.config.DEFAULTS", {"max_apply_attempts": 3}), patch(
                "applypilot.apply.launcher.config.is_manual_ats", return_value=False
            ):
                job = launcher.acquire_job(min_score=7, worker_id=1)

            self.assertIsNotNone(job)
            self.assertEqual(job["url"], "https://example.com/primary")
            conn.close()

    def test_dry_run_does_not_mark_job_applied(self):
        job = {
            "url": "https://example.com/primary",
            "title": "Document Review Specialist",
            "site": "Talent.com",
            "fit_score": 9,
        }
        launcher._stop_event.clear()

        with patch.object(launcher, "acquire_job", return_value=job), patch.object(
            launcher, "launch_chrome", return_value=object()
        ), patch.object(
            launcher, "run_job", return_value=("applied", 123)
        ), patch.object(
            launcher, "mark_result"
        ) as mark_result, patch.object(
            launcher, "cleanup_worker"
        ), patch.object(
            launcher, "update_state"
        ), patch.object(
            launcher, "add_event"
        ):
            applied, failed = launcher.worker_loop(limit=1, dry_run=True)

        self.assertEqual((applied, failed), (1, 0))
        mark_result.assert_called_once_with(
            "https://example.com/primary", "dry_run_ready", None, permanent=False, duration_ms=123
        )

    def test_acquire_target_url_allows_unlocked_null_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "jobs.db"
            conn = init_db(db_path)
            conn.execute(
                """
                INSERT INTO jobs (
                    url, title, site, application_url, tailored_resume_path, cover_letter_path,
                    fit_score, full_description, job_lane, apply_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "https://example.com/job",
                    "Document Review Specialist",
                    "Talent.com",
                    "https://example.com/job/apply",
                    "C:/resume.pdf",
                    "C:/cover.pdf",
                    9,
                    "Remote document review role",
                    "primary",
                    None,
                ),
            )
            conn.commit()

            with patch("applypilot.apply.launcher.get_connection", return_value=conn), patch(
                "applypilot.apply.launcher.config.is_manual_ats", return_value=False
            ):
                job = launcher.acquire_job(
                    target_url="https://example.com/job/apply",
                    min_score=7,
                    worker_id=1,
                )

            self.assertIsNotNone(job)
            self.assertEqual(job["url"], "https://example.com/job")
            conn.close()

    def test_acquire_target_url_respects_min_score_and_cover_letter_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "jobs.db"
            conn = init_db(db_path)
            conn.execute(
                """
                INSERT INTO jobs (
                    url, title, site, application_url, tailored_resume_path, cover_letter_path,
                    fit_score, full_description, job_lane
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "https://example.com/job",
                    "Document Review Specialist",
                    "Talent.com",
                    "https://example.com/job/apply",
                    "C:/resume.txt",
                    "C:/cover.txt",
                    6,
                    "Remote document review role",
                    "primary",
                ),
            )
            conn.commit()

            with patch("applypilot.apply.launcher.get_connection", return_value=conn), patch(
                "applypilot.apply.launcher.config.is_manual_ats", return_value=False
            ):
                job = launcher.acquire_job(
                    target_url="https://example.com/job/apply",
                    min_score=7,
                    worker_id=1,
                )

            self.assertIsNone(job)
            conn.close()

    def test_acquire_target_url_skips_blocked_statuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "jobs.db"
            conn = init_db(db_path)
            conn.execute(
                """
                INSERT INTO jobs (
                    url, title, site, application_url, tailored_resume_path, cover_letter_path,
                    fit_score, full_description, job_lane, apply_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "https://example.com/blocked",
                    "Blocked Job",
                    "Talent.com",
                    "https://example.com/blocked/apply",
                    "C:/resume.txt",
                    "C:/cover.txt",
                    9,
                    "Remote role.",
                    "primary",
                    "manual_review",
                ),
            )
            conn.commit()

            with patch("applypilot.apply.launcher.get_connection", return_value=conn), patch(
                "applypilot.apply.launcher.config.is_manual_ats", return_value=False
            ):
                job = launcher.acquire_job(
                    target_url="https://example.com/blocked/apply",
                    min_score=7,
                    worker_id=1,
                )

            self.assertIsNone(job)
            conn.close()


if __name__ == "__main__":
    unittest.main()
