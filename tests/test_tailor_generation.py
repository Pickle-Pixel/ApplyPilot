import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from applypilot.database import init_db
from applypilot.scoring import tailor
from applypilot.scoring.validator import validate_json_fields


class TailorValidationTests(unittest.TestCase):
    def test_json_validator_accepts_preserved_company_in_subtitle(self):
        profile = {
            "resume_facts": {
                "preserved_companies": ["Aegis Communications"],
                "preserved_school": "Crowder College",
            }
        }
        data = {
            "title": "Operations Analyst",
            "summary": "Operations analyst focused on reporting accuracy and workflow control.",
            "skills": {"Tools": "Advanced Excel, SQL"},
            "experience": [
                {
                    "header": "Dialer Team Lead / Analyst",
                    "subtitle": "Aegis Communications | 2007 - 2008",
                    "bullets": ["Analyzed campaign performance data and adjusted dialer settings."],
                }
            ],
            "projects": [
                {
                    "header": "BuddyOS - Workflow Prototype",
                    "subtitle": "TypeScript | 2025",
                    "bullets": ["Built a workflow prototype for task routing."],
                }
            ],
            "education": "Crowder College | Coursework",
        }

        result = validate_json_fields(data, profile)

        self.assertTrue(result["passed"], result)


class TailorBatchTests(unittest.TestCase):
    def test_failed_tailoring_does_not_leave_resume_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "jobs.db"
            conn = init_db(db_path)
            conn.execute(
                """
                INSERT INTO jobs (
                    url, title, site, application_url, fit_score, full_description
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "https://example.com/job",
                    "Forecasting Analyst",
                    "Direct Employer",
                    "https://example.com/job/apply",
                    9,
                    "Forecasting and WFM reporting role.",
                ),
            )
            conn.commit()
            resume_path = root / "resume.txt"
            resume_path.write_text("base resume", encoding="utf-8")
            output_dir = root / "tailored"

            with patch.object(tailor, "get_connection", return_value=conn), patch.object(
                tailor, "RESUME_PATH", resume_path
            ), patch.object(tailor, "TAILORED_DIR", output_dir), patch.object(
                tailor, "load_profile", return_value={}
            ), patch.object(
                tailor,
                "tailor_resume",
                return_value=("", {"status": "exhausted_retries", "attempts": 4}),
            ):
                result = tailor.run_tailoring(min_score=9, limit=1)

            row = conn.execute(
                "SELECT tailored_resume_path, tailor_attempts FROM jobs WHERE url=?",
                ("https://example.com/job",),
            ).fetchone()
            conn.close()

            self.assertEqual(result["approved"], 0)
            self.assertIsNone(row["tailored_resume_path"])
            self.assertEqual(row["tailor_attempts"], 1)
            self.assertEqual(list(output_dir.glob("*.txt")), [])


if __name__ == "__main__":
    unittest.main()
