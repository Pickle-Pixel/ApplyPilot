import unittest

from applypilot.scoring.scorer import _title_quick_filter


class ScorerQuickFilterTests(unittest.TestCase):
    def test_rejects_phone_heavy_language(self):
        score = _title_quick_filter(
            "Customer Support Specialist",
            "Handle 80+ inbound calls per day in a high-volume call center environment.",
        )

        self.assertEqual(score, (2, "AUTO-REJECT (description): phone-heavy"))

    def test_rejects_voice_support_language(self):
        score = _title_quick_filter(
            "Chat Support Specialist",
            "Provide voice support, manage the phone queue, and answer customer calls.",
        )

        self.assertEqual(score, (2, "AUTO-REJECT (description): phone-heavy"))

    def test_rejects_commission_only_language(self):
        score = _title_quick_filter(
            "Account Executive",
            "Remote sales role with commission-only compensation.",
        )

        self.assertEqual(score, (1, "AUTO-REJECT (title): sales role"))

    def test_rejects_sales_social_media_manager_hybrid(self):
        score = _title_quick_filter(
            "Sales, Events & Social Media Manager",
            "Manage events, sales campaigns, and social media.",
        )

        self.assertEqual(score, (1, "AUTO-REJECT (title): sales role"))

    def test_rejects_content_creation_roles(self):
        score = _title_quick_filter(
            "Social Media Manager/YouTube Video Editor",
            "Create video content and edit YouTube clips.",
        )

        self.assertEqual(score, (1, "AUTO-REJECT (title): content creation role"))

    def test_rejects_brand_strategy_social_media_roles(self):
        score = _title_quick_filter(
            "Social Media Manager - Brand Strategy",
            "Own brand strategy and campaigns.",
        )

        self.assertEqual(score, (1, "AUTO-REJECT (title): marketing specialty"))

    def test_rejects_in_person_role_hidden_as_remote(self):
        score = _title_quick_filter(
            "Social Media Manager (In-Person | Wig & Beauty Industry)",
            "Listed as remote but requires in-person work.",
        )

        self.assertEqual(score, (1, "AUTO-REJECT (title): on-site role"))

    def test_rejects_account_owner_roles(self):
        score = _title_quick_filter(
            "Social Media Manager / Account Owner",
            "Own client accounts and strategy.",
        )

        self.assertEqual(score, (1, "AUTO-REJECT (title): account ownership role"))

    def test_rejects_recruiting_roles(self):
        score = _title_quick_filter(
            "Independent Driver Recruiter",
            "Source and recruit drivers.",
        )

        self.assertEqual(score, (1, "AUTO-REJECT (title): recruiting role"))

    def test_keeps_clean_remote_data_role(self):
        score = _title_quick_filter(
            "Data Analyst I",
            "Remote role focused on SQL, reporting, dashboards, and healthcare operations.",
        )

        self.assertIsNone(score)

    def test_allows_social_media_manager_exception(self):
        score = _title_quick_filter(
            "Social Media Manager",
            "Remote role responding to social media DMs, comments, chat, and email. No phone support.",
        )

        self.assertIsNone(score)


if __name__ == "__main__":
    unittest.main()
