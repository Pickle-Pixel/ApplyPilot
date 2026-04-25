import unittest

from applypilot.scoring.scorer import _apply_contextual_caps, _title_has_strong_fit


class ScorerContextualCapsTests(unittest.TestCase):
    def test_generic_data_analyst_title_is_not_auto_strong_fit(self):
        self.assertFalse(_title_has_strong_fit("Data Analyst"))

    def test_data_operations_analyst_stays_strong_fit(self):
        self.assertTrue(_title_has_strong_fit("Data Operations Analyst"))

    def test_time_and_scheduling_specialist_stays_strong_fit(self):
        self.assertTrue(_title_has_strong_fit("UKG Ready Time & Scheduling Support Specialist"))

    def test_back_office_processing_titles_are_strong_fit(self):
        self.assertTrue(_title_has_strong_fit("Back Office Processing Specialist"))

    def test_document_verification_titles_are_strong_fit(self):
        self.assertTrue(_title_has_strong_fit("Document Verification Specialist"))

    def test_chat_email_support_titles_are_strong_fit(self):
        self.assertTrue(_title_has_strong_fit("Email and Chat Support Specialist"))

    def test_social_media_manager_title_is_strong_fit(self):
        self.assertTrue(_title_has_strong_fit("Social Media Manager"))

    def test_caps_non_us_remote_role_hidden_in_description(self):
        score, note = _apply_contextual_caps(
            score=9,
            title="QA Analyst I",
            location="Remote",
            description=(
                "Join Upland India as a QA Analyst I for our product team. "
                "Upland India has the flexibility to manage your working hours."
            ),
        )

        self.assertEqual(score, 3)
        self.assertIn("non-US", note)

    def test_caps_non_us_country_qualified_remote_title(self):
        score, note = _apply_contextual_caps(
            score=8,
            title="Data Analytics Support Analyst (Philippines, Remote)",
            location="Remote",
            description="",
        )

        self.assertEqual(score, 3)
        self.assertIn("non-US", note)

    def test_caps_internship_roles(self):
        score, note = _apply_contextual_caps(
            score=8,
            title="Data Analyst Intern",
            location="Remote",
            description=(
                "3 month internship for candidates currently pursuing or recently "
                "completed a degree. Certificate of Internship provided."
            ),
        )

        self.assertEqual(score, 4)
        self.assertIn("internship", note.lower())

    def test_caps_entry_level_roles(self):
        score, note = _apply_contextual_caps(
            score=8,
            title="Data Analyst",
            location="Remote",
            description=(
                "Junior Data Analyst role ideal for someone early in their career. "
                "Required qualifications include 0-2 years of experience."
            ),
        )

        self.assertEqual(score, 7)
        self.assertIn("entry-level", note.lower())

    def test_caps_programmer_heavy_early_career_roles(self):
        score, note = _apply_contextual_caps(
            score=8,
            title="Early Career Python Programmer/Data Analyst",
            location="Remote, US (Remote)",
            description=(
                "Programmer/Analyst role with approximately 90% of time spent "
                "developing coding expertise in Python and SQL analytic programming."
            ),
        )

        self.assertEqual(score, 5)
        self.assertIn("programming-heavy", note.lower())

    def test_caps_annotation_only_roles_below_top_tier(self):
        score, note = _apply_contextual_caps(
            score=9,
            title="Data Quality Analyst",
            location="Remote",
            description=(
                "Accurately label machine learning textual data through data "
                "annotation. Pre-screening includes typing speed and reading "
                "comprehension. Bachelor's degree in English, Literature, Creative "
                "Writing, Journalism, or related field."
            ),
        )

        self.assertEqual(score, 7)
        self.assertIn("annotation", note.lower())

    def test_caps_marketing_heavy_social_media_roles(self):
        score, note = _apply_contextual_caps(
            score=8,
            title="Social Media Manager",
            location="Remote, US (Remote)",
            description=(
                "Own paid social campaigns, brand strategy, content calendar, "
                "influencer partnerships, and marketing performance reporting."
            ),
        )

        self.assertEqual(score, 6)
        self.assertIn("marketing-heavy", note.lower())


if __name__ == "__main__":
    unittest.main()
