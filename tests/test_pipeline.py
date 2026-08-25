"""Unit tests for ApplyPilot pipeline.py using unittest"""
import unittest
from applypilot.pipeline import STAGE_ORDER, STAGE_META, _UPSTREAM

class TestPipeline(unittest.TestCase):
    def test_stage_order(self):
        self.assertEqual(STAGE_ORDER, ("discover", "enrich", "score", "tailor", "cover", "pdf"))

    def test_stage_dependencies(self):
        self.assertIsNone(_UPSTREAM["discover"])
        self.assertEqual(_UPSTREAM["enrich"], "discover")
        self.assertEqual(_UPSTREAM["score"], "enrich")
        self.assertEqual(_UPSTREAM["tailor"], "score")
        self.assertEqual(_UPSTREAM["cover"], "tailor")
        self.assertEqual(_UPSTREAM["pdf"], "cover")

    def test_stage_meta_completeness(self):
        for stage in STAGE_ORDER:
            self.assertIn(stage, STAGE_META)
            self.assertIn("desc", STAGE_META[stage])

if __name__ == "__main__":
    unittest.main()
