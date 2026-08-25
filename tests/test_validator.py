"""Unit tests for ApplyPilot scoring/validator.py using unittest"""
import unittest
from applypilot.scoring.validator import sanitize_text, validate_json_fields

class TestValidator(unittest.TestCase):
    def test_sanitize_text(self):
        raw = "Senior Engineer \u2014 Python \u2013 \u201cGreat\u201d \u2018Work\u2019"
        cleaned = sanitize_text(raw)
        self.assertEqual(cleaned, 'Senior Engineer, Python - "Great" \'Work\'')

    def test_validate_json_fields_missing_key(self):
        data = {"title": "Software Engineer"}
        profile = {}
        result = validate_json_fields(data, profile)
        self.assertFalse(result["passed"])
        self.assertTrue(any("Missing required field" in err for err in result["errors"]))

    def test_validate_json_fields_valid(self):
        data = {
            "title": "Backend Developer",
            "summary": "Built distributed services using Python and PostgreSQL.",
            "skills": {"Languages": "Python, SQL", "Tools": "Docker, Git"},
            "experience": [
                {
                    "header": "Acme Corp | Software Engineer",
                    "bullets": ["Optimized API latency by 40%."]
                }
            ],
            "projects": [
                {
                    "header": "Open Source CLI",
                    "bullets": ["Developed file processing engine."]
                }
            ],
            "education": "BS Computer Science"
        }
        profile = {
            "resume_facts": {"preserved_companies": ["Acme Corp"]}
        }
        result = validate_json_fields(data, profile, mode="lenient")
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["errors"]), 0)

    def test_validate_json_fields_fabricated_skill(self):
        data = {
            "title": "Backend Developer",
            "summary": "Experienced Rust developer",
            "skills": {"Languages": "Python, Rust, Scala"},
            "experience": [{"header": "Acme Corp", "bullets": ["Built app"]}],
            "projects": [{"header": "Proj", "bullets": ["Built app"]}],
            "education": "BS CS"
        }
        profile = {"resume_facts": {"preserved_companies": ["Acme Corp"]}}
        result = validate_json_fields(data, profile, mode="normal")
        self.assertFalse(result["passed"])
        self.assertTrue(any("Fabricated skill" in err for err in result["errors"]))

if __name__ == "__main__":
    unittest.main()
