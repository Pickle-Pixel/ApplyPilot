"""
LLM-powered resume parsing for ApplyPilot.

Extracts structured profile data from resume text using the configured LLM.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class PersonalInfo(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    city: Optional[str] = None
    province_state: Optional[str] = None
    country: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None


class ProfessionalInfo(BaseModel):
    current_title: Optional[str] = None
    years_experience: Optional[int] = Field(None, ge=0, le=60)
    education_level: Optional[str] = None
    education_institution: Optional[str] = None


class SkillsInfo(BaseModel):
    programming_languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)


class ExtractedResume(BaseModel):
    personal: PersonalInfo = Field(default_factory=PersonalInfo)
    professional: ProfessionalInfo = Field(default_factory=ProfessionalInfo)
    skills: SkillsInfo = Field(default_factory=SkillsInfo)
    preserved_companies: list[str] = Field(default_factory=list)
    preserved_projects: list[str] = Field(default_factory=list)
    real_metrics: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """Extract structured data from this resume. Return ONLY valid JSON, no explanation.

{
  "personal": {
    "full_name": "string or null",
    "email": "string or null",
    "phone": "string or null",
    "city": "string or null",
    "province_state": "string or null",
    "country": "string or null",
    "linkedin_url": "full URL or null",
    "github_url": "full URL or null",
    "portfolio_url": "full URL or null"
  },
  "professional": {
    "current_title": "most recent job title or null",
    "years_experience": "integer calculated from work history dates, or null",
    "education_level": "Bachelor's, Master's, PhD, Associate's, Bootcamp, or null",
    "education_institution": "school name or null"
  },
  "skills": {
    "programming_languages": ["array of languages mentioned"],
    "frameworks": ["array of frameworks/libraries mentioned"],
    "tools": ["array of tools/platforms like Docker, AWS, Git, etc."]
  },
  "preserved_companies": ["array of company names from work history"],
  "preserved_projects": ["array of notable project names"],
  "real_metrics": ["array of quantified achievements like '50% faster', 'served 100k users'"]
}

Rules:
1. Extract ONLY what is explicitly in the resume
2. Do NOT fabricate or guess missing information - use null
3. For years_experience, calculate from date ranges (e.g., 2019-2024 = 5 years)
4. For skills, only include technologies explicitly mentioned
5. Return valid JSON only - no markdown, no explanation

Resume:
"""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_json_from_response(response: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try direct parse first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding raw JSON object
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not extract valid JSON from LLM response")


def extract_resume_data(resume_text: str) -> tuple[Optional[ExtractedResume], dict]:
    """Extract structured profile data from resume text using LLM.

    Args:
        resume_text: Plain text content of the resume.

    Returns:
        (extracted_data, metadata) where:
        - extracted_data: ExtractedResume or None if extraction failed
        - metadata: {"success": bool, "errors": list[str], "warnings": list[str]}
    """
    from applypilot.llm import get_client

    metadata = {
        "success": False,
        "errors": [],
        "warnings": [],
    }

    try:
        client = get_client()
        prompt = EXTRACTION_PROMPT + resume_text

        response = client.ask(prompt, temperature=0.0, max_tokens=2048)

        # Parse JSON from response
        try:
            data = extract_json_from_response(response)
        except ValueError as e:
            metadata["errors"].append(f"Invalid JSON from LLM: {e}")
            log.error("Resume extraction failed: %s", e)
            return None, metadata

        # Validate with Pydantic
        try:
            extracted = ExtractedResume(**data)
        except Exception as e:
            metadata["errors"].append(f"Validation error: {e}")
            log.error("Resume validation failed: %s", e)
            return None, metadata

        # Post-extraction validation
        warnings = _validate_extraction(extracted)
        metadata["warnings"] = warnings

        metadata["success"] = True
        return extracted, metadata

    except Exception as e:
        metadata["errors"].append(f"Extraction failed: {e}")
        log.error("Resume extraction error: %s", e)
        return None, metadata


def _validate_extraction(extracted: ExtractedResume) -> list[str]:
    """Validate extracted data and return warnings."""
    warnings = []

    if not extracted.personal.full_name:
        warnings.append("Could not extract name")

    if not extracted.personal.email:
        warnings.append("Could not extract email")

    skill_count = (
        len(extracted.skills.programming_languages) + len(extracted.skills.frameworks) + len(extracted.skills.tools)
    )
    if skill_count == 0:
        warnings.append("No skills extracted")

    if not extracted.preserved_companies:
        warnings.append("No companies extracted from work history")

    return warnings


def extracted_to_profile(extracted: ExtractedResume) -> dict:
    """Convert ExtractedResume to ApplyPilot profile.json format."""
    return {
        "personal": {
            "full_name": extracted.personal.full_name or "",
            "preferred_name": "",
            "email": extracted.personal.email or "",
            "phone": extracted.personal.phone or "",
            "city": extracted.personal.city or "",
            "province_state": extracted.personal.province_state or "",
            "country": extracted.personal.country or "",
            "postal_code": "",
            "address": "",
            "linkedin_url": extracted.personal.linkedin_url or "",
            "github_url": extracted.personal.github_url or "",
            "portfolio_url": extracted.personal.portfolio_url or "",
            "website_url": "",
            "password": "",
        },
        "experience": {
            "years_of_experience_total": str(extracted.professional.years_experience or ""),
            "education_level": extracted.professional.education_level or "",
            "current_title": extracted.professional.current_title or "",
            "target_role": extracted.professional.current_title or "",  # Default to current
        },
        "skills_boundary": {
            "programming_languages": extracted.skills.programming_languages,
            "frameworks": extracted.skills.frameworks,
            "tools": extracted.skills.tools,
        },
        "resume_facts": {
            "preserved_companies": extracted.preserved_companies,
            "preserved_projects": extracted.preserved_projects,
            "preserved_school": extracted.professional.education_institution or "",
            "real_metrics": extracted.real_metrics,
        },
    }
