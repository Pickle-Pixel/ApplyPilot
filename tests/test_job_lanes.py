import unittest

from applypilot.scoring.job_lanes import classify_job_lane


class JobLaneTests(unittest.TestCase):
    def test_platform_marketplace_goes_to_flex_lane(self):
        lane = classify_job_lane(
            title="Remote Business Analyst",
            site="linkedin",
            description="About Turing: work on projects to help fine-tune large language models.",
        )

        self.assertEqual(lane, "flex")

    def test_freelance_contract_role_goes_to_flex_lane(self):
        lane = classify_job_lane(
            title="English Language Specialist (US Only) - Freelance AI Trainer Project",
            site="linkedin",
            description="Freelance AI trainer project with hourly contract compensation.",
        )

        self.assertEqual(lane, "flex")

    def test_direct_employer_role_stays_primary_lane(self):
        lane = classify_job_lane(
            title="Data Operations Analyst",
            site="linkedin",
            description="Maintain data integrity across operational systems and collaborate with multiple teams.",
        )

        self.assertEqual(lane, "primary")


if __name__ == "__main__":
    unittest.main()
