import tempfile
import unittest
from pathlib import Path

from applypilot.database import get_stats, init_db


class DatabaseStatsTests(unittest.TestCase):
    def test_ready_to_apply_counts_primary_url_without_application_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = init_db(Path(tmp) / "jobs.db")
            conn.execute(
                """
                INSERT INTO jobs (
                    url, title, site, fit_score, full_description,
                    tailored_resume_path, cover_letter_path, job_lane
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "https://www.linkedin.com/jobs/view/123",
                    "Release of Information Quality Assurance Review Specialist I",
                    "linkedin",
                    8,
                    "Remote records QA role.",
                    "C:/resume.txt",
                    "C:/cover.txt",
                    "primary",
                ),
            )
            conn.commit()

            stats = get_stats(conn)
            conn.close()

            self.assertEqual(stats["ready_to_apply"], 1)
