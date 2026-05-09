"""Unit tests for the 2026-04-29 tailoring quality patches.

Bug 1: validator banlist now catches Unicode dash variants of "detail-oriented".
Bug 2: empty PROJECTS section heading is suppressed in the renderer.
Bug 3: EXPERIENCE entries render in reverse-chronological order.
"""
from __future__ import annotations

import unittest

from applypilot.scoring import tailor
from applypilot.scoring import validator


class DetailOrientedBanlistTests(unittest.TestCase):
    """Bug 1: every dash variant of 'detail-oriented' must be flagged."""

    PROFILE = {
        "skills_boundary": {"Tools": ["Excel"]},
        "resume_facts": {
            "preserved_companies": [],
            "preserved_school": "Crowder College",
            "real_metrics": [],
        },
    }

    BASE_RESUME = (
        "Darrin Biby\n"
        "Operations Analyst\n\n"
        "SUMMARY\nOperations analyst with quality background.\n\n"
        "TECHNICAL SKILLS\nTools: Excel\n\n"
        "EXPERIENCE\nAnalyst at Acme | 2024 - Present\n- Did stuff.\n\n"
        "EDUCATION\nCrowder College | Coursework\n"
    )

    def _check(self, summary: str) -> dict:
        text = (
            "Darrin Biby\nOperations Analyst\nx@y.com\n\n"
            f"SUMMARY\n{summary}\n\n"
            "TECHNICAL SKILLS\nTools: Excel\n\n"
            "EXPERIENCE\nAnalyst at Acme\nTools | 2024 - Present\n- Did stuff.\n\n"
            "EDUCATION\nCrowder College | Coursework\n"
        )
        return validator.validate_tailored_resume(text, self.PROFILE, self.BASE_RESUME)

    def test_ascii_hyphen_variant_caught(self):
        result = self._check("Detail-oriented operations analyst with QA background.")
        # Strict mode is default in the policed path; even in normal mode the
        # validator surfaces banned words as warnings.
        all_msgs = " ".join(result.get("errors", []) + result.get("warnings", []))
        self.assertIn("detail-oriented", all_msgs.lower())

    def test_unhyphenated_variant_caught(self):
        result = self._check("Detail oriented operations analyst with QA background.")
        all_msgs = " ".join(result.get("errors", []) + result.get("warnings", []))
        self.assertIn("detail oriented", all_msgs.lower())

    def test_en_dash_variant_caught(self):
        result = self._check("Detail–oriented analyst with QA background.")
        all_msgs = " ".join(result.get("errors", []) + result.get("warnings", []))
        self.assertIn("detail", all_msgs.lower())

    def test_em_dash_variant_caught(self):
        result = self._check("Detail—oriented analyst with QA background.")
        all_msgs = " ".join(result.get("errors", []) + result.get("warnings", []))
        self.assertIn("detail", all_msgs.lower())


class EmptyProjectsSectionTests(unittest.TestCase):
    """Bug 2: PROJECTS heading must not appear when projects array is empty."""

    PROFILE = {
        "personal": {"full_name": "Darrin Biby", "email": "x@y.com"},
        "skills_boundary": {},
        "resume_facts": {},
    }

    def _data(self, projects):
        return {
            "title": "Operations Analyst",
            "summary": "Analyst with strong reporting background.",
            "skills": {"Tools": "Excel"},
            "experience": [
                {
                    "header": "Analyst at Acme",
                    "subtitle": "Excel | 2024 - Present",
                    "bullets": ["Built reports."],
                }
            ],
            "projects": projects,
            "education": "Crowder College | Coursework",
        }

    def test_empty_projects_no_heading(self):
        text = tailor.assemble_resume_text(self._data([]), self.PROFILE)
        self.assertNotIn("PROJECTS", text)

    def test_projects_with_only_falsy_entries_no_heading(self):
        text = tailor.assemble_resume_text(self._data([None, {}]), self.PROFILE)
        self.assertNotIn("PROJECTS", text)

    def test_projects_with_real_entry_renders_heading(self):
        data = self._data([
            {
                "header": "BuddyOS - Workflow",
                "subtitle": "TypeScript | 2025",
                "bullets": ["Built workflow prototype."],
            }
        ])
        text = tailor.assemble_resume_text(data, self.PROFILE)
        self.assertIn("PROJECTS", text)
        self.assertIn("BuddyOS", text)


class ChronologicalOrderTests(unittest.TestCase):
    """Bug 3: EXPERIENCE rendered reverse-chronological by end date."""

    PROFILE = {
        "personal": {"full_name": "Darrin Biby", "email": "x@y.com"},
        "skills_boundary": {},
        "resume_facts": {},
    }

    def _data(self, experience):
        return {
            "title": "Analyst",
            "summary": "Analyst.",
            "skills": {"Tools": "Excel"},
            "experience": experience,
            "projects": [],
            "education": "Crowder | Coursework",
        }

    def test_present_role_is_at_top(self):
        experience = [
            {"header": "NCO Tier 2", "subtitle": "Avaya | 2009 - 2014",
             "bullets": ["Tier 2 work."]},
            {"header": "Aegis Lead", "subtitle": "Avaya | 2007 - 2008",
             "bullets": ["Dialer."]},
            {"header": "Sharecare Analyst", "subtitle": "SQL | 2024 - Present",
             "bullets": ["Reporting."]},
        ]
        text = tailor.assemble_resume_text(self._data(experience), self.PROFILE)
        # Sharecare should come before NCO and Aegis
        i_share = text.find("Sharecare")
        i_nco = text.find("NCO")
        i_aegis = text.find("Aegis")
        self.assertLess(i_share, i_nco)
        self.assertLess(i_nco, i_aegis)
        self.assertGreater(i_share, 0)

    def test_year_only_dates_descending(self):
        experience = [
            {"header": "Old Job", "subtitle": "Tools | 2007 - 2008",
             "bullets": ["..."]},
            {"header": "Recent Job", "subtitle": "Tools | 2024 - 2025",
             "bullets": ["..."]},
            {"header": "Mid Job", "subtitle": "Tools | 2009 - 2014",
             "bullets": ["..."]},
        ]
        text = tailor.assemble_resume_text(self._data(experience), self.PROFILE)
        i_recent = text.find("Recent")
        i_mid = text.find("Mid")
        i_old = text.find("Old")
        self.assertLess(i_recent, i_mid)
        self.assertLess(i_mid, i_old)

    def test_sort_key_present_token(self):
        key = tailor._experience_sort_key({"subtitle": "Excel | 2024 - Present"})
        self.assertEqual(key, (9999, 12))

    def test_sort_key_extracts_max_year(self):
        key = tailor._experience_sort_key({"subtitle": "Tools | 2007 - 2014"})
        self.assertEqual(key[0], 2014)

    def test_sort_key_falls_back_to_zero(self):
        key = tailor._experience_sort_key({"subtitle": "Tools"})
        self.assertEqual(key, (0, 0))


if __name__ == "__main__":
    unittest.main()
