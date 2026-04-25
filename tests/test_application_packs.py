import sqlite3
import tempfile
import unittest
from pathlib import Path

from applypilot.database import init_db
from applypilot.application_packs import export_ready_packs


class ApplicationPackTests(unittest.TestCase):
    def test_exports_ready_primary_pack_with_pdfs_and_next_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "jobs.db"
            conn = init_db(db_path)

            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            resume_txt = artifact_dir / "resume.txt"
            cover_txt = artifact_dir / "cover.txt"
            resume_pdf = artifact_dir / "resume.pdf"
            cover_pdf = artifact_dir / "cover.pdf"
            resume_txt.write_text("resume", encoding="utf-8")
            cover_txt.write_text("cover", encoding="utf-8")
            resume_pdf.write_bytes(b"%PDF-1.4\n" + b"x" * 2048)
            cover_pdf.write_bytes(b"%PDF-1.4\n" + b"x" * 2048)

            conn.execute(
                """
                INSERT INTO jobs (
                    url, title, site, application_url, fit_score, full_description,
                    tailored_resume_path, cover_letter_path, job_lane
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "https://example.com/job",
                    "Document Review Specialist",
                    "Direct Employer",
                    "https://example.com/job/apply",
                    9,
                    "Remote document review role.",
                    str(resume_txt),
                    str(cover_txt),
                    "primary",
                ),
            )
            conn.commit()

            result = export_ready_packs(
                conn=conn,
                output_root=root / "application_packs",
                batch_name="ready_primary_test",
                min_score=7,
                limit=5,
            )

            conn.close()

            self.assertEqual(result["exported"], 1)
            self.assertEqual(result["skipped"], 0)
            pack_dir = Path(result["packs"][0]["path"])
            self.assertTrue(pack_dir.is_dir())
            self.assertTrue((pack_dir / "01_resume.pdf").is_file())
            self.assertTrue((pack_dir / "02_cover_letter.pdf").is_file())
            next_steps = (pack_dir / "00_NEXT_STEPS.md").read_text(encoding="utf-8")
            self.assertIn("Document Review Specialist", next_steps)
            self.assertIn("applypilot.exe apply --dry-run --headless", next_steps)
            self.assertIn("https://example.com/job/apply", next_steps)

            index = (root / "application_packs" / "ready_primary_test" / "00_INDEX.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Document Review Specialist", index)

    def test_skips_pack_when_pdf_artifact_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "jobs.db"
            conn = init_db(db_path)
            resume_txt = root / "resume.txt"
            cover_txt = root / "cover.txt"
            resume_txt.write_text("resume", encoding="utf-8")
            cover_txt.write_text("cover", encoding="utf-8")

            conn.execute(
                """
                INSERT INTO jobs (
                    url, title, site, application_url, fit_score, full_description,
                    tailored_resume_path, cover_letter_path, job_lane
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "https://example.com/job",
                    "WFM Analyst",
                    "Direct Employer",
                    "https://example.com/job/apply",
                    9,
                    "Remote WFM role.",
                    str(resume_txt),
                    str(cover_txt),
                    "primary",
                ),
            )
            conn.commit()

            result = export_ready_packs(
                conn=conn,
                output_root=root / "application_packs",
                batch_name="ready_primary_test",
            )

            conn.close()

            self.assertEqual(result["exported"], 0)
            self.assertEqual(result["skipped"], 1)
            self.assertIn("missing resume PDF", result["skips"][0]["reason"])

    def test_exports_ready_pack_with_original_url_when_application_url_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "jobs.db"
            conn = init_db(db_path)

            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            resume_txt = artifact_dir / "resume.txt"
            cover_txt = artifact_dir / "cover.txt"
            resume_pdf = artifact_dir / "resume.pdf"
            cover_pdf = artifact_dir / "cover.pdf"
            resume_txt.write_text("resume", encoding="utf-8")
            cover_txt.write_text("cover", encoding="utf-8")
            resume_pdf.write_bytes(b"%PDF-1.4\n" + b"x" * 2048)
            cover_pdf.write_bytes(b"%PDF-1.4\n" + b"x" * 2048)

            conn.execute(
                """
                INSERT INTO jobs (
                    url, title, site, application_url, fit_score, full_description,
                    tailored_resume_path, cover_letter_path, job_lane
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "https://www.linkedin.com/jobs/view/123",
                    "Chat Support Assistant",
                    "linkedin",
                    None,
                    8,
                    "Remote chat support role.",
                    str(resume_txt),
                    str(cover_txt),
                    "primary",
                ),
            )
            conn.commit()

            result = export_ready_packs(
                conn=conn,
                output_root=root / "application_packs",
                batch_name="ready_primary_test",
            )

            conn.close()

            self.assertEqual(result["exported"], 1)
            next_steps = (Path(result["packs"][0]["path"]) / "00_NEXT_STEPS.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("https://www.linkedin.com/jobs/view/123", next_steps)
