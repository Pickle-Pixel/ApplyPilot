"""Classify jobs into primary and flex lanes."""

from __future__ import annotations

import re

FLEX_PLATFORM_KEYWORDS = (
    "turing",
    "mercor",
    "alignerr",
    "outlier",
    "upwork",
    "fiverr",
    "toptal",
    "swooped",
    "pinnacle method",
    "prism data",
    "rex.zone",
    "talent platform",
    "not the employer",
    "staffing firm",
    "staffing agency",
)

FLEX_CONTRACT_KEYWORDS = (
    "freelance",
    "1099",
    "independent contractor",
    "contract opportunity",
    "hourly contract",
    "project-based",
    "set your rate",
)


def classify_job_lane(title: str | None, site: str | None, description: str | None) -> str:
    """Return the lane this job belongs to.

    primary: standard direct-employer queue
    flex: platform, freelance, and contract work
    """
    text = "\n".join(part for part in (title or "", site or "", description or "") if part).lower()

    if any(keyword in text for keyword in FLEX_PLATFORM_KEYWORDS):
        return "flex"

    if any(keyword in text for keyword in FLEX_CONTRACT_KEYWORDS):
        return "flex"

    if re.search(r"\b(contract|contractor)\b", text) and "full-time" not in text:
        return "flex"

    return "primary"
