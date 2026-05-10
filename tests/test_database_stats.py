import tempfile
import unittest
from pathlib import Path

from applypilot.database import (
    get_apply_queue_stats,
    get_jobs_by_stage,
    get_ready_to_apply_by_blocker_reason,
    get_stale_lock_count,
    get_stats,
    init_db,
)


class DatabaseStatsTests(unittest.TestCase):
    def _write_ready_artifacts(self, tmp: str, stem: str) -> tuple[str, str]:
        root = Path(tmp)
        resume_txt = root / f"{stem}_resume.txt"
        cover_txt = root / f"{stem}_cover.txt"
        resume_txt.write_text("resume", encoding="utf-8")
        cover_txt.write_text("cover", encoding="utf-8")
        resume_txt.with_suffix(".pdf").write_bytes(b"%PDF-1.4\n" + b"0" * 2048)
        cover_txt.with_suffix(".pdf").write_bytes(b"%PDF-1.4\n" + b"0" * 2048)
        return str(resume_txt), str(cover_txt)

    def test_ready_to_apply_counts_primary_url_without_application_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = init_db(Path(tmp) / "jobs.db")
            resume_path, cover_path = self._write_ready_artifacts(tmp, "ready")
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
                    resume_path,
                    cover_path,
                    "primary",
                ),
            )
            conn.commit()

            stats = get_stats(conn)
            conn.close()

            self.assertEqual(stats["ready_to_apply"], 1)

    def test_pending_tailor_skips_blocked_apply_statuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = init_db(Path(tmp) / "jobs.db")
            rows = [
                ("https://example.com/blocked", "Blocked Job", "direct", 9, "Remote role.", "failed"),
                ("https://example.com/clean", "Clean Job", "direct", 8, "Remote role.", None),
            ]
            conn.executemany(
                """
                INSERT INTO jobs (
                    url, title, site, fit_score, full_description, apply_status
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

            jobs = get_jobs_by_stage(conn=conn, stage="pending_tailor", min_score=7, limit=10)
            conn.close()

            self.assertEqual([job["url"] for job in jobs], ["https://example.com/clean"])

    def test_pending_tailor_can_filter_to_primary_lane(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = init_db(Path(tmp) / "jobs.db")
            rows = [
                ("https://example.com/flex", "Flex Job", "direct", 10, "Remote role.", "flex"),
                ("https://example.com/primary", "Primary Job", "direct", 8, "Remote role.", "primary"),
            ]
            conn.executemany(
                """
                INSERT INTO jobs (
                    url, title, site, fit_score, full_description, job_lane
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

            jobs = get_jobs_by_stage(conn=conn, stage="pending_tailor", min_score=7, limit=10, lane="primary")
            conn.close()

            self.assertEqual([job["url"] for job in jobs], ["https://example.com/primary"])

    def test_apply_queue_stats_exposes_ready_and_blocker_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = init_db(Path(tmp) / "jobs.db")
            resume_path, cover_path = self._write_ready_artifacts(tmp, "ready")
            conn.execute(
                """
                INSERT INTO jobs (
                    url, title, site, fit_score, full_description,
                    tailored_resume_path, cover_letter_path, job_lane, apply_status, apply_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "https://example.com/ready",
                    "Ready Job",
                    "direct",
                    9,
                    "Remote role.",
                    resume_path,
                    cover_path,
                    "primary",
                    None,
                    None,
                ),
            )
            conn.execute(
                """
                INSERT INTO jobs (
                    url, title, site, fit_score, full_description,
                    tailored_resume_path, cover_letter_path, job_lane, apply_status, apply_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "https://example.com/blocked",
                    "Blocked Job",
                    "direct",
                    9,
                    "Remote role.",
                    "C:/resume.txt",
                    "C:/cover.txt",
                    "primary",
                    "manual",
                    "manual ATS",
                ),
            )
            conn.execute(
                """
                INSERT INTO jobs (
                    url, title, site, fit_score, full_description,
                    tailored_resume_path, cover_letter_path, job_lane, apply_status, apply_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "https://example.com/failed",
                    "Failed Job",
                    "direct",
                    9,
                    "Remote role.",
                    "C:/resume.txt",
                    "C:/cover.txt",
                    "primary",
                    "failed",
                    "already_applied",
                ),
            )
            conn.commit()

            queue_stats = get_apply_queue_stats(conn)
            conn.close()

            self.assertEqual(queue_stats["ready_count"], 1)
            self.assertEqual(queue_stats["manual_count"], 1)
            self.assertEqual(queue_stats["failed_count"], 1)
            self.assertEqual(queue_stats["failed_reasons"], {"already_applied": 1})

    def test_ready_to_apply_by_blocker_reason_exposes_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = init_db(Path(tmp) / "jobs.db")
            conn.execute(
                """
                INSERT INTO jobs (
                    url, title, site, fit_score, full_description,
                    tailored_resume_path, cover_letter_path, job_lane, apply_status, apply_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "https://example.com/manual",
                    "Blocked Manual Job",
                    "direct",
                    9,
                    "Remote role.",
                    "C:/resume.txt",
                    "C:/cover.txt",
                    "primary",
                    "manual",
                    "manual ATS",
                ),
            )
            conn.execute(
                """
                INSERT INTO jobs (
                    url, title, site, fit_score, full_description,
                    tailored_resume_path, cover_letter_path, job_lane, apply_status, apply_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "https://example.com/in_progress",
                    "In progress job",
                    "direct",
                    9,
                    "Remote role.",
                    "C:/resume.txt",
                    "C:/cover.txt",
                    "primary",
                    "in_progress",
                    "running",
                ),
            )
            conn.commit()

            summary = get_ready_to_apply_by_blocker_reason(conn)
            conn.close()

            self.assertEqual(summary["blocked_by_status"]["manual"], 1)
            self.assertEqual(summary["blocked_by_status"]["in_progress"], 1)
            self.assertEqual(summary["ready_count"], 0)

    def test_ready_queue_stats_exclude_missing_pdf_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = init_db(Path(tmp) / "jobs.db")
            resume_txt = Path(tmp) / "missing_resume.txt"
            cover_txt = Path(tmp) / "missing_cover.txt"
            resume_txt.write_text("resume", encoding="utf-8")
            cover_txt.write_text("cover", encoding="utf-8")
            conn.execute(
                """
                INSERT INTO jobs (
                    url, title, site, fit_score, full_description,
                    tailored_resume_path, cover_letter_path, job_lane, apply_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "https://example.com/missing-pdfs",
                    "Missing PDFs Job",
                    "direct",
                    9,
                    "Remote role.",
                    str(resume_txt),
                    str(cover_txt),
                    "primary",
                    None,
                ),
            )
            conn.commit()

            queue_stats = get_apply_queue_stats(conn)
            conn.close()

            self.assertEqual(queue_stats["ready_count"], 0)
            self.assertEqual(queue_stats["invalid_artifact_count"], 1)

    def test_stale_lock_count_detects_old_in_progress_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = init_db(Path(tmp) / "jobs.db")
            conn.execute(
                """
                INSERT INTO jobs (
                    url, title, site, fit_score, full_description,
                    tailored_resume_path, cover_letter_path, apply_status, last_attempted_at, job_lane
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "https://example.com/old_lock",
                    "Old lock job",
                    "direct",
                    9,
                    "Remote role.",
                    "C:/resume.txt",
                    "C:/cover.txt",
                    "in_progress",
                    "2026-01-01T00:00:00",
                    "primary",
                ),
            )
            conn.commit()

            stale_count = get_stale_lock_count(conn)
            conn.close()

            self.assertEqual(stale_count, 1)
