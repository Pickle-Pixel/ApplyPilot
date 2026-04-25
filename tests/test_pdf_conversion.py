import tempfile
import unittest
from pathlib import Path

from applypilot.scoring.pdf import convert_cover_letter_to_pdf, convert_to_pdf


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


if __name__ == "__main__":
    unittest.main()
