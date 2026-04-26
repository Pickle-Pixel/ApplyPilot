"""Template-based cover letter generation with auto-selection and customization.

Reads the user's pre-written cover letter templates from Pending-Review,
auto-selects the best one based on job title + description keyword matching,
and customizes it (fills placeholders, adjusts emphasis, adds company-specific
sentence). Falls back to LLM generation if templates are missing or selection
fails.

All personal data comes from the profile at runtime. No hardcoded personal info.
"""

import logging
import re
from pathlib import Path

from applypilot.config import APP_DIR, load_profile

log = logging.getLogger(__name__)

# ── Template Registry ────────────────────────────────────────────────────

PENDING_REVIEW_DIR = APP_DIR / "Pending-Review"

# Each template has a filename, keyword weights, and a paused flag.
# Keywords are mapped to integer weights; higher = stronger signal.
TEMPLATE_REGISTRY: dict[str, dict] = {
    "chat_email_backoffice": {
        "filename": "COVER_LETTER_TEMPLATE_Chat_Email_BackOffice.txt",
        "keywords": {
            "chat": 3, "live chat": 4, "messaging": 4, "email": 3,
            "correspondence": 3, "back office": 3, "back-office": 3,
            "non-phone": 3, "non-voice": 3, "text-based": 2,
            "digital support": 2, "web-based": 2,
            "customer service representative": 2, "customer support": 2,
            "live messaging": 4, "chat support": 4, "email support": 4,
            "order processing": 2, "data entry": 2, "messaging agent": 5,
            "chat support specialist": 5, "email support specialist": 5,
            "live chat agent": 5, "back office specialist": 5,
            "back office associate": 5, "correspondence specialist": 5,
            "inbound messages": 2, "outbound messages": 2, "live agent": 2,
            "help desk": 2, "helpdesk": 2, "ticketing": 2, "zendesk": 2,
            "intercom": 2, "livechat": 2, "slack": 1, "discord": 1,
        },
        "paused": False,
    },
    "wfm_operations": {
        "filename": "COVER_LETTER_TEMPLATE_WFM_Operations.txt",
        "keywords": {
            "workforce": 4, "wfm": 4, "forecasting": 3, "scheduling": 3,
            "adherence": 3, "erlang": 3, "real-time": 3, "intraday": 3,
            "staffing": 3, "capacity": 3, "shrinkage": 2, "occupancy": 2,
            "aspect": 2, "ewfm": 2, "avaya": 2, "cms supervisor": 2,
            "nice": 2, "verint": 2, "iex": 2,
            "workforce management": 5, "workforce analyst": 5,
            "scheduler": 4, "operations analyst": 4,
            "real-time analyst": 5, "capacity planner": 5,
            "service level manager": 4,
        },
        "paused": False,
    },
    "general_qa": {
        "filename": "COVER_LETTER_TEMPLATE_General_Operations_QA.txt",
        "keywords": {
            "qa": 3, "quality assurance": 4, "quality control": 4,
            "audit": 2, "compliance": 3, "documentation": 3,
            "document review": 3, "records": 2, "verification": 2,
            "hipaa": 3, "cms": 2, "medicare": 3, "medicaid": 2,
            "enrollment": 2, "prior authorization": 3, "claims": 2,
            "roi": 2, "accuracy": 2, "data entry": 2,
            "document processor": 4, "qa analyst": 5,
            "release of information": 4, "medical records": 4,
            "claims processor": 4, "compliance analyst": 4,
            "health information": 3, "h.i.s.": 3, "his": 2,
            "healthcare": 3, "health care": 3, "clinical": 3,
            "patient care": 3, "revenue cycle": 3, "utilization review": 4,
            "referrals": 2, "scheduling": 2, "hospital": 2,
            "health system": 2, "prior auth": 3,
        },
        "paused": False,
    },
    "tech_support": {
        "filename": "COVER_LETTER_TEMPLATE_Tech_Support.txt",
        "keywords": {
            "technical support": 4, "sql": 2, "atm": 2, "pos": 2,
            "incident": 2, "troubleshooting": 2, "tier 2": 3,
            "banking systems": 2, "technical support engineer": 5,
            "application support": 4, "sql support": 4,
        },
        "paused": True,  # Lane 4 paused per user instruction
    },
}

# Minimum score required to consider a template a match.
# Below this, we default to the primary chat/email template.
MIN_TEMPLATE_SCORE = 3

# Default template when no clear match is found.
DEFAULT_TEMPLATE_KEY = "chat_email_backoffice"


# ── Helpers ────────────────────────────────────────────────────────────────

def _template_path(key: str) -> Path | None:
    """Resolve template file path from registry key."""
    entry = TEMPLATE_REGISTRY.get(key)
    if not entry:
        return None
    path = PENDING_REVIEW_DIR / entry["filename"]
    if path.exists():
        return path
    log.warning("Template file missing: %s", path)
    return None


def _score_text_against_keywords(text: str, keywords: dict[str, int]) -> int:
    """Score a block of text against a keyword-weight map.

    Longer keywords are checked first so "live chat" beats "chat".
    """
    text_lower = text.lower()
    score = 0
    # Sort by length descending so multi-word phrases match first
    for keyword, weight in sorted(keywords.items(), key=lambda kv: -len(kv[0])):
        if keyword in text_lower:
            score += weight
    return score


def select_template(title: str, description: str) -> tuple[str, Path | None]:
    """Auto-select the best cover letter template for a job.

    Args:
        title: Job title.
        description: Full job description.

    Returns:
        (template_key, template_path) — path may be None if missing.
    """
    combined = f"{title or ''}\n{description or ''}"
    best_key = DEFAULT_TEMPLATE_KEY
    best_score = 0

    for key, entry in TEMPLATE_REGISTRY.items():
        if entry.get("paused"):
            continue
        score = _score_text_against_keywords(combined, entry["keywords"])
        log.debug("Template '%s' scored %d for job '%s'", key, score, title[:50])
        if score > best_score:
            best_score = score
            best_key = key

    if best_score < MIN_TEMPLATE_SCORE:
        best_key = DEFAULT_TEMPLATE_KEY
        log.debug(
            "No template scored above %d (best=%d); defaulting to '%s'",
            MIN_TEMPLATE_SCORE, best_score, best_key,
        )
    else:
        log.debug(
            "Selected template '%s' with score %d for job '%s'",
            best_key, best_score, title[:50],
        )

    path = _template_path(best_key)
    return best_key, path


# ── Customization ──────────────────────────────────────────────────────────

def _extract_company_detail(description: str, company: str) -> str:
    """Generate a company-specific sentence from the job description.

    Tries to find a sentence in the JD that hints at mission, values, or
    scope, and frames it as a reason the candidate is drawn to the company.
    Falls back to a generic sentence if nothing usable is found.
    """
    if not description:
        return (
            f"I am drawn to {company}'s mission and would welcome the "
            "opportunity to contribute to your team."
        )

    # Split on sentence boundaries
    raw_sentences = re.split(r"[.!?\n]", description)
    mission_words = {
        "mission", "values", "committed", "dedicated", "focus", "leading",
        "largest", "fast-growing", "innovative", "patient care",
        "customer experience", "transform", "improve", "community",
        "excellence", "quality", "integrity", "collaboration",
        "team-oriented", "people-first", "client-centered", "remote-first",
        "nationwide", "award-winning", "recognized", "trusted",
    }

    candidates: list[str] = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if len(sentence) < 15 or len(sentence) > 180:
            continue
        lower = sentence.lower()
        # Skip boilerplate EEO sentences
        if "equal opportunity" in lower or "eeo" in lower or "disability" in lower:
            continue
        if any(w in lower for w in mission_words):
            # Clean bullet markers and excess whitespace
            cleaned = re.sub(r"^[-*\s•]+", "", sentence).strip()
            if cleaned:
                candidates.append(cleaned)

    if candidates:
        # Pick the first reasonable candidate
        detail = candidates[0]
        detail_lower = detail[0].lower() + detail[1:]
        return (
            f"I am drawn to {company}'s focus on {detail_lower} and would welcome "
            "the opportunity to contribute to your team."
        )

    # Fallback
    return (
        f"I am drawn to {company}'s commitment to operational excellence and "
        "would welcome the opportunity to contribute to your team."
    )


def _remove_customization_checklist(text: str) -> str:
    """Strip the 'CUSTOMIZATION CHECKLIST' block and everything after it."""
    marker = "---"
    idx = text.rfind(marker)
    if idx != -1:
        return text[:idx].strip()
    return text.strip()


def _fill_placeholders(text: str, job: dict, profile: dict) -> str:
    """Replace [ROLE_NAME] and [COMPANY_NAME] placeholders."""
    role = job.get("title") or "this position"
    company = job.get("company") or job.get("site") or "your company"

    text = text.replace("[ROLE_NAME]", role)
    text = text.replace("[COMPANY_NAME]", company)
    return text


def _fill_company_sentence(text: str, job: dict) -> str:
    """Replace the [COMPANY-SPECIFIC SENTENCE: ...] placeholder."""
    company = job.get("company") or job.get("site") or "your company"
    description = job.get("full_description") or ""

    sentence = _extract_company_detail(description, company)

    # Replace the placeholder, handling nested brackets (e.g., [Company])
    start = text.find("[COMPANY-SPECIFIC SENTENCE:")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    text = text[:start] + sentence + text[i + 1 :]
                    break
    return text


def _adjust_emphasis(text: str, job: dict, template_key: str) -> str:
    """Optionally reorder or tweak paragraphs based on JD keyword emphasis.

    Currently supports:
      - Healthcare-heavy JDs → lead with GoHealth/Teleperformance paragraph
      - Scheduling-heavy JDs → ensure NCO Group scheduling leads (already default)
    """
    description = (job.get("full_description") or "").lower()
    title = (job.get("title") or "").lower()
    combined = f"{title}\n{description}"

    healthcare_keywords = {
        "medicare", "medicaid", "hipaa", "cms", "prior authorization",
        "utilization review", "referrals", "health information", "h.i.s.",
        "medical records", "enrollment", "claims", "patient",
        "healthcare", "health care", "clinical", "radiology",
    }
    healthcare_score = sum(1 for k in healthcare_keywords if k in combined)

    # Only reorder for the chat template (others already have appropriate order)
    if template_key == "chat_email_backoffice" and healthcare_score >= 2:
        # Try to find the GoHealth + Teleperformance paragraph and move it up
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) >= 4:
            # Look for the paragraph containing both GoHealth and Teleperformance
            health_idx = None
            for i, para in enumerate(paragraphs):
                lower = para.lower()
                if "gohealth" in lower and "teleperformance" in lower:
                    health_idx = i
                    break
            if health_idx is not None and health_idx > 1:
                # Move it to position 1 (right after opening paragraph)
                para = paragraphs.pop(health_idx)
                paragraphs.insert(1, para)
                text = "\n\n".join(paragraphs)
                log.debug("Reordered paragraphs for healthcare emphasis.")

    return text


def customize_cover_letter(template_text: str, job: dict, profile: dict,
                         template_key: str) -> str:
    """Fully customize a selected template for a specific job.

    Steps:
      1. Fill [ROLE_NAME] and [COMPANY_NAME].
      2. Generate and insert company-specific sentence.
      3. Remove the customization checklist footer.
      4. Adjust paragraph emphasis based on JD keywords.
    """
    text = template_text
    text = _fill_placeholders(text, job, profile)
    text = _fill_company_sentence(text, job)
    text = _remove_customization_checklist(text)
    text = _adjust_emphasis(text, job, template_key)
    return text


# ── Public API ─────────────────────────────────────────────────────────────

def templates_available() -> bool:
    """Return True if at least one template file exists in Pending-Review."""
    for entry in TEMPLATE_REGISTRY.values():
        path = PENDING_REVIEW_DIR / entry["filename"]
        if path.exists():
            return True
    return False


def generate_cover_letter_from_template(job: dict, profile: dict | None = None) -> str | None:
    """Generate a cover letter using template matching and customization.

    Args:
        job: Job dict with title, site, company, full_description.
        profile: User profile dict (loaded automatically if None).

    Returns:
        Customized cover letter text, or None if templates are unavailable.
    """
    if profile is None:
        profile = load_profile()

    title = job.get("title") or ""
    description = job.get("full_description") or ""

    template_key, path = select_template(title, description)
    if path is None:
        log.warning("No template available for key '%s'; skipping template match.", template_key)
        return None

    template_text = path.read_text(encoding="utf-8")
    letter = customize_cover_letter(template_text, job, profile, template_key)

    log.info(
        "Template cover letter generated (%s, %d chars) for %s @ %s",
        template_key, len(letter), title[:40], job.get("site", "?"),
    )
    return letter
