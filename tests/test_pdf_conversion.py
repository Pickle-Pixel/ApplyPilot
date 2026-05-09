import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import applypilot.scoring.pdf as pdf_module
from applypilot.scoring.pdf import batch_convert, convert_cover_letter_to_pdf, convert_to_pdf


class PdfConversionTests(unittest.TestCase):
    def test_resume_pdf_conversion_rejects_empty_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "empty.txt"
            source.write_text("", encoding="utf-8")

            with self.assertRaises(ValueError):
                convert_to_pdf(source, html_only=True)

    def test_cover_letter_html_conversion_uses_letter_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "cover.txt"
            source.write_text(
                "Dear Hiring Manager,\n\n"
                "I built dashboard checks that kept operations teams aligned with real metrics.\n\n"
                "Darrin",
                encoding="utf-8",
            )

            html_path = convert_cover_letter_to_pdf(source, html_only=True)
            html = html_path.read_text(encoding="utf-8")

            self.assertIn("Dear Hiring Manager", html)
            self.assertIn("cover-letter", html)
            self.assertNotIn("Technical Skills", html)

    def test_batch_convert_includes_cover_letters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tailored_dir = root / "tailored_resumes"
            cover_dir = root / "cover_letters"
            tailored_dir.mkdir()
            cover_dir.mkdir()

            resume = tailored_dir / "role.txt"
            job_sidecar = tailored_dir / "role_JOB.txt"
            cover = cover_dir / "role_CL.txt"
            resume.write_text("resume", encoding="utf-8")
            job_sidecar.write_text("job", encoding="utf-8")
            cover.write_text("cover", encoding="utf-8")

            def fake_resume_converter(path):
                Path(path).with_suffix(".pdf").write_bytes(b"%PDF-1.4\nresume")

            def fake_cover_converter(path):
                Path(path).with_suffix(".pdf").write_bytes(b"%PDF-1.4\ncover")

            with (
                patch.object(pdf_module, "TAILORED_DIR", tailored_dir),
                patch.object(pdf_module, "COVER_LETTER_DIR", cover_dir),
                patch.object(pdf_module, "convert_to_pdf", side_effect=fake_resume_converter),
                patch.object(pdf_module, "convert_cover_letter_to_pdf", side_effect=fake_cover_converter),
            ):
                converted = batch_convert(limit=10)

            self.assertEqual(converted, 2)
            self.assertTrue((tailored_dir / "role.pdf").exists())
            self.assertFalse((tailored_dir / "role_JOB.pdf").exists())
            self.assertTrue((cover_dir / "role_CL.pdf").exists())


if __name__ == "__main__":
    unittest.main()
