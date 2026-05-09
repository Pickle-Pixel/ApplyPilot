import unittest

try:
    from applypilot.discovery import jobspy
    _HAS_JOBSPY = True
except ModuleNotFoundError:
    jobspy = None
    _HAS_JOBSPY = False


@unittest.skipIf(not _HAS_JOBSPY, "jobspy dependency is not installed")
class JobSpyFilterTests(unittest.TestCase):
    def test_load_location_config_supports_nested_location_block(self):
        cfg = {
            "location": {
                "accept_patterns": ["Remote", "United States"],
                "reject_patterns": ["hybrid", "India"],
            }
        }

        accept, reject = jobspy._load_location_config(cfg)

        self.assertEqual(accept, ["Remote", "United States"])
        self.assertEqual(reject, ["hybrid", "India"])

    def test_search_filters_reject_title_matches(self):
        cfg = {
            "exclude_titles": ["customer service representative", "call center agent"],
            "exclude_description_keywords": [],
        }

        allowed, reason = jobspy._passes_search_filters(
            title="Customer Service Representative",
            description="Email support role",
            search_cfg=cfg,
        )

        self.assertFalse(allowed)
        self.assertEqual(reason, "title")

    def test_search_filters_reject_description_matches(self):
        cfg = {
            "exclude_titles": [],
            "exclude_description_keywords": ["we are not the employer", "talent platform"],
        }

        allowed, reason = jobspy._passes_search_filters(
            title="Business Analyst",
            description="Swooped is a talent platform. We are not the employer for this role.",
            search_cfg=cfg,
        )

        self.assertFalse(allowed)
        self.assertEqual(reason, "description")

    def test_search_filters_reject_excluded_company_matches(self):
        cfg = {
            "exclude_titles": [],
            "exclude_description_keywords": [],
            "exclude_companies": ["RBC", "Magna International"],
        }

        allowed, reason = jobspy._passes_search_filters(
            title="Document Review Specialist",
            company="RBC",
            description="Remote role reviewing documents for accuracy.",
            search_cfg=cfg,
        )

        self.assertFalse(allowed)
        self.assertEqual(reason, "company")

    def test_search_filters_allow_clean_non_phone_role(self):
        cfg = {
            "exclude_titles": ["customer service representative"],
            "exclude_description_keywords": ["cold calling", "commission only"],
        }

        allowed, reason = jobspy._passes_search_filters(
            title="Reporting Specialist",
            company="Direct Employer",
            description="Remote reporting role focused on dashboards, QA, and operational metrics.",
            search_cfg=cfg,
        )

        self.assertTrue(allowed)
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
