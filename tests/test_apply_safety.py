import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from applypilot import cli
from applypilot.apply import launcher, prompt as prompt_mod
from applypilot.database import init_db


def _profile() -> dict:
    return {
        "personal": {
            "full_name": "Darrin Example",
            "preferred_name": "Darrin",
            "email": "darrin@example.com",
            "phone": "555-0100",
            "address": "123 Main St",
            "city": "St. Louis",
            "province_state": "MO",
            "country": "US",
            "postal_code": "63101",
            "password": "super-secret-password",
        },
        "work_authorization": {
            "legally_authorized_to_work": "Yes",
            "require_sponsorship": "No",
            "work_permit_type": "",
        },
        "compensation": {
            "salary_expectation": "50000",
            "salary_range_min": "50000",
            "salary_range_max": "65000",
            "salary_currency": "USD",
        },
        "experience": {
            "years_of_experience_total": "10",
            "target_role": "technical support specialist",
            "education_level": "Some college",
        },
        "availability": {"earliest_start_date": "Immediately"},
        "eeo_voluntary": {},
    }


def _search_config() -> dict:
    return {"location": {"primary": "St. Louis", "accept_patterns": ["St. Louis", "Remote"]}}


class ApplySafetyTests(unittest.TestCase):
    def test_mcp_config_excludes_gmail_by_default(self):
        mcp_config = launcher._make_mcp_config(9222)

        self.assertIn("playwright", mcp_config["mcpServers"])
        self.assertNotIn("gmail", mcp_config["mcpServers"])

    def test_claude_command_does_not_bypass_permissions(self):
        cmd = launcher._build_claude_command(
            model="haiku",
            mcp_config_path=Path("C:/tmp/.mcp-apply-0.json"),
        )

        self.assertNotIn("bypassPermissions", cmd)
        self.assertNotIn("--permission-mode", cmd)

    def test_apply_mode_requires_explicit_submit_confirmation(self):
        self.assertTrue(cli._resolve_apply_dry_run(dry_run=False, submit=False, confirm_submit=None))

        with self.assertRaises(ValueError):
            cli._resolve_apply_dry_run(dry_run=False, submit=True, confirm_submit=None)

        with self.assertRaises(ValueError):
            cli._resolve_apply_dry_run(dry_run=True, submit=True, confirm_submit=cli.SUBMIT_CONFIRMATION)

        self.assertFalse(
            cli._resolve_apply_dry_run(
                dry_run=False,
                submit=True,
                confirm_submit=cli.SUBMIT_CONFIRMATION,
            )
        )

    def test_reset_failed_only_resets_failed_statuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = init_db(Path(tmp) / "jobs.db")
            rows = [
                ("https://example.com/failed", "failed", "timeout"),
                ("https://example.com/failed-prefixed", "failed:timeout", "timeout"),
                ("https://example.com/manual", "manual", "manual ATS"),
                ("https://example.com/dry-run", "dry_run_ready", None),
                ("https://example.com/avoid", "avoid", "AVOID list"),
            ]
            for url, status, error in rows:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        url, title, site, fit_score, full_description,
                        tailored_resume_path, cover_letter_path, job_lane,
                        apply_status, apply_error, apply_attempts
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        url,
                        "Safety Test",
                        "direct",
                        9,
                        "Remote role.",
                        "C:/resume.txt",
                        "C:/cover.txt",
                        "primary",
                        status,
                        error,
                        99,
                    ),
                )
            conn.commit()

            with patch("applypilot.apply.launcher.get_connection", return_value=conn):
                reset_count = launcher.reset_failed()

            remaining = {
                row["url"]: row["apply_status"]
                for row in conn.execute("SELECT url, apply_status FROM jobs").fetchall()
            }
            conn.close()

        self.assertEqual(reset_count, 2)
        self.assertIsNone(remaining["https://example.com/failed"])
        self.assertIsNone(remaining["https://example.com/failed-prefixed"])
        self.assertEqual(remaining["https://example.com/manual"], "manual")
        self.assertEqual(remaining["https://example.com/dry-run"], "dry_run_ready")
        self.assertEqual(remaining["https://example.com/avoid"], "avoid")

    def test_prompt_does_not_embed_mail_tools_password_or_captcha_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            resume_txt = tmp_path / "resume.txt"
            resume_pdf = tmp_path / "resume.pdf"
            resume_txt.write_text("Technical support resume text.", encoding="utf-8")
            resume_pdf.write_bytes(b"%PDF-1.4\n" + (b"0" * 2048))

            worker_dir = tmp_path / "workers"
            job = {
                "url": "https://example.com/job",
                "application_url": "https://example.com/apply",
                "title": "Technical Support Specialist",
                "site": "ExampleCo",
                "fit_score": 9,
                "tailored_resume_path": str(resume_txt),
            }

            with patch.object(prompt_mod.config, "load_profile", return_value=_profile()), patch.object(
                prompt_mod.config, "load_search_config", return_value=_search_config()
            ), patch.object(prompt_mod.config, "APPLY_WORKER_DIR", worker_dir), patch.dict(
                "os.environ", {"CAPSOLVER_API_KEY": "capsolver-secret-value"}, clear=False
            ):
                text = prompt_mod.build_prompt(job=job, tailored_resume="Technical support resume text.", dry_run=True)

        self.assertNotIn("send_email", text)
        self.assertNotIn("search_emails", text)
        self.assertNotIn("read_email", text)
        self.assertNotIn("super-secret-password", text)
        self.assertNotIn("capsolver-secret-value", text)
        self.assertNotIn("clientKey", text)


if __name__ == "__main__":
    unittest.main()
