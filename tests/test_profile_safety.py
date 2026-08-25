"""Production profile safety gates."""

import unittest

from applypilot.config import profile_safety_reasons, validate_profile_for_application


class TestProfileSafety(unittest.TestCase):
    def test_legacy_sample_profile_is_rejected(self):
        profile = {"name": "Firstname Lastname (Sample Resume Candidate)", "email": "youremail@example.com"}
        self.assertTrue(profile_safety_reasons(profile))
        with self.assertRaises(ValueError):
            validate_profile_for_application(profile)

    def test_complete_structured_profile_is_allowed(self):
        profile = {
            "personal": {"full_name": "Real Person", "email": "person@domain.test"},
            "work_authorization": {}, "compensation": {}, "experience": {}, "resume_facts": {}
        }
        self.assertEqual(profile_safety_reasons(profile), [])


if __name__ == "__main__":
    unittest.main()
