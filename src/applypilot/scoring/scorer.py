"""Job fit scoring: LLM-powered evaluation of candidate-job match quality.

Scores jobs on a 1-10 scale by comparing the user's resume against each
job description. All personal data is loaded at runtime from the user's
profile and resume file.

SCORING PHILOSOPHY (Darrin-specific):
    Darrin is a WFM/QA/Ops/Tech-Support professional. He is NOT a developer,
    NOT a manager, and cannot do phone-heavy roles. Many job titles read
    "technical" but require a senior software-engineer background he doesn't
    have. The scorer must:
      1. Hard-cap disqualifiers (phone, management, senior-dev, sales,
         commission, Jack Henry) to a max score of 2 BEFORE the LLM runs.
      2. Weight domain fit (WFM / QA / Ops / Tech Support / Healthcare /
         Insurance) above generic "technical skills."
      3. Treat BuddyOS / AI-assisted dev as a "shows learning agility"
         signal, NOT a qualification for Software Engineer roles.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone

from applypilot.config import RESUME_PATH, load_profile
from applypilot.database import get_connection, get_jobs_by_stage
from applypilot.llm import get_client
from applypilot.scoring.job_lanes import classify_job_lane

log = logging.getLogger(__name__)


# -- Title-level hard caps and boosts --------------------------------------
# Applied BEFORE the LLM call. Saves compute on obvious disqualifiers.

# Patterns that force a maximum score of 2 (auto-reject) regardless of body.
# Checked against the job TITLE (case-insensitive).
TITLE_HARD_CAPS: list[tuple[str, str]] = [
    # Sales / marketing / content noise that commonly leaks into social media searches
    (r"\bsales\b[\w\s,&/+-]{0,80}\bmanager\b",       "sales role"),
    (r"\b(bdr|sdr)\b",                               "sales role"),
    (r"\b(recruiter|talent\s+acquisition)\b",        "recruiting role"),
    (r"\b(content\s+writer|copywriter|video\s+editor|content\s+creator)\b","content creation role"),
    (r"\b(ppc|google\s+ads|paid\s+social|brand\s+strategy|seo|sem)\b","marketing specialty"),
    (r"\baccount\s+owner\b",                         "account ownership role"),
    (r"\b(in[\s-]?person|on[\s-]?site|onsite)\b",    "on-site role"),
    # Management / leadership — Darrin is IC track
    (r"\b(senior\s+)?manager\b",                    "management role"),
    (r"\b(director|vp|vice\s+president|head\s+of)\b","exec/director level"),
    (r"\blead(er)?\s+(engineer|developer|architect)\b","engineering leadership"),
    (r"\bprincipal\b",                               "principal-level (always senior IC)"),
    # Senior / staff software engineering — not his background
    (r"\b(sr\.?|senior|staff)\s+[\w\s-]*?(engineer|architect|scientist|developer)\b","senior engineering/architect role"),
    (r"\b(software|full[\s-]?stack|backend|back[\s-]?end|frontend|front[\s-]?end|platform|site\s+reliability|sre)\s+engineer\b","software engineering role"),
    (r"\b(devops|cloud|ml|ai|llm|data|compute|infrastructure|security)\s+engineer\b","specialized engineering role"),
    (r"\b(solutions?|enterprise|cloud|data|security)\s+architect\b","architect role"),
    # Sales / BD / commission-coded titles
    (r"\b(sales|account)\s+(executive|manager|representative|rep|consultant|director)\b","sales role"),
    (r"\b(business\s+development|biz\s*dev|bd)\b",   "business development"),
    (r"\bchannel\s+(sales|partner)\b",               "channel sales"),
    (r"\b(territory|prime)\s+(account|sales)\b",     "territory sales"),
    (r"\bdeveloper\s+relations\b",                   "DevRel (presentation-heavy)"),
    # Phone-coded titles
    (r"\b(inbound|outbound)\s+(agent|representative|rep|specialist)\b","phone agent"),
    (r"\bcollections?\s+(agent|specialist)\b",       "collections (phone-heavy)"),
    (r"\btelesales\b",                               "phone sales"),
    # Licenses / field work
    (r"\bfield\s+(engineer|technician|service|application)\b","field role (on-site)"),
    (r"\b(rn|registered\s+nurse|lpn|cna|physician|md|pharmacist)\b","licensed medical role"),
    # Deep domain experts outside his wheelhouse
    (r"\b(tax|accounting|cpa|controller|audit)\b",   "accounting/finance specialty"),
    (r"\b(attorney|paralegal|legal\s+counsel)\b",    "legal specialty"),
]

# Patterns that strongly suggest a good fit. Used as a soft prior signal
# passed to the LLM rubric — not a hard floor.
TITLE_STRONG_FIT: list[str] = [
    r"\bworkforce\s+(management|analyst|scheduler|planner)\b",
    r"\b(real[\s-]?time|rta)\s+(analyst|scheduler|manager)\b",
    r"\b(wfm|wfp|capacity)\s+(analyst|specialist|planner|coordinator)\b",
    r"\b(qa|quality)\s+(analyst|tester|specialist|auditor)\b",
    r"\b(test|software)\s+(analyst|tester)\b",
    r"\boperations\s+(analyst|specialist|coordinator)\b",
    r"\bbusiness\s+operations\b",
    r"\b(tier\s*[12]|technical)\s+support\s+(specialist|analyst|engineer)\b",
    r"\b(application|product)\s+support\s+(specialist|analyst|engineer)\b",
    r"\b(service|help)\s+desk\s+(specialist|analyst|engineer)\b",
    r"\b(time\s*(and|&)\s*scheduling|timekeeping)\s+(support\s+)?(specialist|analyst)\b",
    r"\b(customer|client)\s+success\s+(analyst|specialist)\b",
    r"\b(data\s+operations|reporting|forecasting|fraud|payments|business\s+intelligence)\s+(analyst|specialist)\b",
    r"\bback[\s-]?office\s+(processing|processor|specialist|associate|clerk|coordinator)\b",
    r"\b(document|records?)\s+(processing|processor|verification|verifier|review(er)?|specialist|control|indexing|imaging)\b",
    r"\b(document|records?)\s+(processing|verification|review)\s+(specialist|associate|analyst|processor|clerk|coordinator)\b",
    r"\b(verification|eligibility)\s+(specialist|analyst|processor|associate|coordinator)\b",
    r"\b(email\s+(and|&)\s+chat|chat\s+(and|&)\s+email)\s+(support\s+)?(specialist|representative|agent|advisor|associate)\b",
    r"\b(chat|email)\s+(support|specialist|representative|agent|advisor|associate)\b",
    r"\b(live\s+chat|online\s+chat)\s+(support\s+)?(specialist|representative|agent|associate)\b",
    r"\bsocial\s+media\s+(support\s+)?(specialist|moderator|manager|coordinator|associate)\b",
    r"\bsocial\s+care\s+(specialist|moderator|manager|coordinator|associate)\b",
    r"\bcommunity\s+(support\s+)?(specialist|moderator|manager|coordinator|associate)\b",
    r"\b(ai|llm)\s+(rater|evaluator|quality|qa|reviewer|annotator|trainer)\b",
    r"\bcontent\s+moderator\b",
    r"\bclaims?\s+(analyst|specialist|examiner)\b",
]

# Body-level phrase filters — substrings in description that are red flags.
BODY_HARD_CAPS: list[tuple[str, str]] = [
    (r"\b(inbound|outbound|high[\s-]?volume)\s+call(s|ing)?\b","phone-heavy"),
    (r"\b\d+\+?\s*calls?\s+per\s+(day|hour)\b",      "call quota"),
    (r"\bvoice\s+support\b",                         "phone-heavy"),
    (r"\bphone\s+queue\b",                           "phone-heavy"),
    (r"\b(answer|handle|take|make|place|field)\s+(inbound\s+|outbound\s+)?(phone\s+)?calls?\b","phone-heavy"),
    (r"\bcall\s+(queue|center|centre)\b",            "phone-heavy"),
    (r"\bcold[\s-]?call(ing|s)?\b",                  "cold calling"),
    (r"\bcommission[\s-]?only\b",                    "commission-only"),
    (r"\b(24\s*/\s*7|on[\s-]?call\s+rotation|pager\s+duty)\b","on-call rotation"),
    (r"\b(leetcode|hackerrank|live\s+coding)\b",     "coding interview required"),
    (r"\bjack\s+henry\b",                             "Jack Henry — banned employer"),
]

MANAGER_TITLE_EXCEPTIONS: list[str] = [
    r"\bsocial\s+media\s+manager\b",
    r"\bsocial\s+care\s+manager\b",
    r"\bcommunity\s+manager\b",
]

NON_US_MARKETS = r"india|philippines|uk|united kingdom|europe|emea|apac|australia|new zealand|singapore|malaysia"

NON_US_ROLE_PATTERNS: list[str] = [
    rf"\bjoin\s+[\w& .'-]*?(?:{NON_US_MARKETS})\b",
    rf"\b(?:{NON_US_MARKETS})\s+(?:team|office|location|remote)\b",
    rf"\b(?:{NON_US_MARKETS})\b[^.\n]{{0,20}}\bremote\b",
    rf"\b(?:remote|hybrid|onsite|on-site)[^.\n]{{0,40}}\b(?:{NON_US_MARKETS})\b",
    rf"\b(?:candidates|applicants|employees|must|required|eligible)[^.\n]{{0,60}}\b(?:in|based in|located in|reside in|work from)\s+(?:the\s+)?(?:{NON_US_MARKETS})\b",
]

INTERNSHIP_PATTERNS: list[str] = [
    r"\bintern(ship)?\b",
    r"\bcurrently\s+pursuing\b",
    r"\brecently\s+completed\b",
    r"\bcertificate\s+of\s+internship\b",
    r"\bletter\s+of\s+recommendation\b",
]

ENTRY_LEVEL_PATTERNS: list[str] = [
    r"\b(junior|entry[\s/-]?level)\b",
    r"\bearly\s+career\b",
    r"\bearly\s+in\s+(their|your)\s+career\b",
    r"\b0\s*[-–]\s*2\s+years?\b",
    r"\b1\s*[-–]\s*2\s+years?\b",
]

PROGRAMMER_ROLE_PATTERNS: list[str] = [
    r"\bprogrammer\b",
    r"\bstatistical\s+programming\b",
    r"\banalytic\s+programming\b",
]

ANNOTATION_KEYWORDS: tuple[str, ...] = (
    "data annotation",
    "data labeling",
    "data labelling",
    "label machine learning",
    "label textual data",
    "annotating large volumes",
)

ANNOTATION_SUPPORT_CUES: tuple[str, ...] = (
    "typing speed",
    "reading comprehension",
    "creative writing",
    "journalism",
    "english, literature",
    "english, literature, creative writing",
)

ANNOTATION_COUNTERSIGNALS: tuple[str, ...] = (
    "sql",
    "dashboard",
    "power bi",
    "looker",
    "tableau",
    "test case",
    "manual testing",
    "quality assurance",
    "technical support",
    "workforce management",
    "forecasting",
    "claims",
    "compliance",
)

SOCIAL_MEDIA_TITLE_PATTERNS: list[str] = [
    r"\bsocial\s+media\s+manager\b",
    r"\bsocial\s+care\s+manager\b",
    r"\bcommunity\s+manager\b",
]

SOCIAL_MEDIA_MARKETING_CUES: tuple[str, ...] = (
    "paid social",
    "brand strategy",
    "content calendar",
    "influencer",
    "campaign",
    "marketing performance",
    "growth marketing",
    "seo",
    "sem",
)

SOCIAL_MEDIA_SUPPORT_CUES: tuple[str, ...] = (
    "dm",
    "direct message",
    "inbox",
    "comments",
    "comment moderation",
    "moderation",
    "customer support",
    "customer care",
    "social care",
    "respond to",
    "reply to",
)


def _title_quick_filter(title: str, description: str) -> tuple[int, str] | None:
    """Return (score, reason) if the title/body triggers a hard cap.

    Hard-cap hits short-circuit the LLM call and score 1-2 automatically.
    Returns None if no cap triggered — normal LLM scoring proceeds.
    """
    t = (title or "").lower()
    d = (description or "").lower()

    for pattern, reason in TITLE_HARD_CAPS:
        if re.search(pattern, t, flags=re.IGNORECASE):
            if reason == "management role" and any(
                re.search(exc, t, flags=re.IGNORECASE) for exc in MANAGER_TITLE_EXCEPTIONS
            ):
                continue
            return 1, f"AUTO-REJECT (title): {reason}"

    for pattern, reason in BODY_HARD_CAPS:
        if re.search(pattern, d, flags=re.IGNORECASE):
            return 2, f"AUTO-REJECT (description): {reason}"

    return None


def _title_has_strong_fit(title: str) -> bool:
    """Whether the title contains one of Darrin's target-role patterns."""
    t = (title or "").lower()
    return any(re.search(p, t, flags=re.IGNORECASE) for p in TITLE_STRONG_FIT)


def _apply_contextual_caps(
    score: int,
    title: str,
    location: str,
    description: str,
) -> tuple[int, str]:
    """Cap scores for patterns that the LLM tends to overrate.

    These are not auto-rejects; they are realism checks that prevent inflated
    top-tier scores for roles that are technically adjacent but poor targets
    in practice for Darrin's search.
    """
    caps: list[tuple[int, str]] = []
    text = "\n".join(part for part in (title or "", description or "") if part).lower()
    loc = (location or "").lower()

    if "us" not in loc and "united states" not in loc:
        for pattern in NON_US_ROLE_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                caps.append((3, "non-US location / hiring market"))
                break

    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in INTERNSHIP_PATTERNS):
        caps.append((4, "internship / student-targeted role"))
    elif any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in ENTRY_LEVEL_PATTERNS):
        caps.append((7, "entry-level role likely to screen out as overqualified"))

    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in PROGRAMMER_ROLE_PATTERNS):
        caps.append((5, "programming-heavy analyst role"))

    has_annotation_keywords = any(keyword in text for keyword in ANNOTATION_KEYWORDS)
    has_annotation_cues = any(keyword in text for keyword in ANNOTATION_SUPPORT_CUES)
    has_countersignals = any(keyword in text for keyword in ANNOTATION_COUNTERSIGNALS)
    if has_annotation_keywords and (has_annotation_cues or not has_countersignals):
        caps.append((7, "annotation-first AI data role"))

    is_social_media_manager = any(
        re.search(pattern, text, flags=re.IGNORECASE) for pattern in SOCIAL_MEDIA_TITLE_PATTERNS
    )
    has_marketing_cues = any(keyword in text for keyword in SOCIAL_MEDIA_MARKETING_CUES)
    has_support_cues = any(keyword in text for keyword in SOCIAL_MEDIA_SUPPORT_CUES)
    if is_social_media_manager and has_marketing_cues and not has_support_cues:
        caps.append((6, "marketing-heavy social media role"))

    if not caps:
        return score, ""

    cap, note = min(caps, key=lambda item: item[0])
    return min(score, cap), note


# -- Scoring Prompt (Darrin-specific rubric) -------------------------------

SCORE_PROMPT = """You are a job fit evaluator scoring roles for Darrin Biby.

CANDIDATE PROFILE (authoritative — override anything the resume implies):
- Background: Workforce Management (WFM), QA, Operations, Technical Support,
  Insurance/Healthcare ops. ~15 years in contact-center / financial-services /
  insurance domains.
- Track: Individual Contributor (IC). Not a manager. Not seeking to manage.
- Technical depth: SQL, advanced Excel, EWFM/Avaya CMS, ATM/POS troubleshooting,
  basic TypeScript, API troubleshooting. Uses AI tools to build personal
  automations (BuddyOS, ApplyPilot). Is NOT a software engineer and cannot
  pass a live-coding interview.
- Must be: remote and non-phone (chat/ticket/email OK; phone-heavy is NO).
- Standard W-2 roles are the main target, but contract/freelance/platform
  roles can still be useful and should be kept in the flex lane.
- Insurance-domain experience in regulated environments. Do NOT reward
  insurance licensing unless the job explicitly needs insurance licensing.

TARGET ROLES (score HIGHER when title matches):
- Back-office processing, document processing, document verification, records
  processing, eligibility verification, and other quiet production workflows
- Chat-only, email-only, ticket-only, and social inbox / social care support
- Social Media Manager / Community Manager ONLY when focused on inboxes,
  DMs, comments, moderation, or customer support; score marketing campaign,
  paid social, influencer, and brand-strategy roles lower
- WFM / Real-Time / Capacity Analyst (9-10 range if fit)
- QA Analyst / Test Analyst / Software QA (non-dev)
- Operations Analyst / Business Operations
- Tier 1-2 Technical Support (non-phone)
- Reporting / Data Operations / Fraud / Payments Analyst
- Healthcare/Insurance QA, Claims Analyst, HIPAA-adjacent
- AI Rater / LLM Evaluator / Prompt QA / Content Moderator

DISQUALIFIERS (must score 1-2 — these already auto-reject upstream but
re-verify from the description):
- Phone-heavy: "inbound calls", "outbound calls", call quotas, cold calling
- Management / leadership / people management / direct reports
- Senior / staff / principal / lead software engineering
- Sales: quota-carrying, commission-only, territory sales, channel sales
- On-call rotation, Jack Henry & Associates

SCORING SCALE:
- 9-10: Target-list title + core skills directly match + remote + non-phone.
- 7-8:  Strong adjacent fit (e.g. operations analyst at a call center,
        QA-adjacent at insurance company); minor gaps.
- 5-6:  Could be bridged — partial skill overlap, unclear phone load, or
        mid-level seniority ambiguity.
- 3-4:  Weak fit — unrelated domain or significant skill gap that ramp-up
        won't bridge.
- 1-2:  Disqualified — management, senior engineering, sales, phone-heavy,
        or otherwise off-profile.

IMPORTANT:
- BuddyOS / AI-assisted dev work DOES NOT qualify Darrin for Software
  Engineer roles. Treat it as a "learning agility" signal only, not as
  engineering credentials.
- A job that LOOKS technical (mentions SQL, Python, APIs) is not automatically
  a fit — check the role level. "Data Analyst" with SQL = good. "Senior
  Backend Engineer with SQL" = NOT a fit.
- When in doubt about seniority level, score LOWER, not higher.
- Plain "Data Analyst" titles are not automatically top-tier fits. Reserve
  8-10 for roles with clear reporting, operations, fraud/payments, QA,
  healthcare, or workforce overlap.
- Internships, student-only roles, non-US hiring markets, and annotation-only
  typing/writing roles should be scored conservatively, usually below 8.

RESPOND IN EXACTLY THIS FORMAT (no other text):
SCORE: [1-10]
KEYWORDS: [comma-separated ATS keywords from the job description that
         actually match Darrin's background — no padding]
REASONING: [2-3 sentences: what fits, what doesn't, why this score]"""


def _parse_score_response(response: str) -> dict:
    """Parse the LLM's score response into structured data.

    Args:
        response: Raw LLM response text.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    score = 0
    keywords = ""
    reasoning = response

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                score = int(re.search(r"\d+", line).group())
                score = max(1, min(10, score))
            except (AttributeError, ValueError):
                score = 0
        elif line.startswith("KEYWORDS:"):
            keywords = line.replace("KEYWORDS:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    return {"score": score, "keywords": keywords, "reasoning": reasoning}


def score_job(resume_text: str, job: dict) -> dict:
    """Score a single job against the resume.

    Applies a title/body hard-cap filter first. On a hit, returns an
    auto-reject score WITHOUT calling the LLM (saves compute and prevents
    the model from "finding reasons" to pass a disqualified role).

    Args:
        resume_text: The candidate's full resume text.
        job: Job dict with keys: title, site, location, full_description.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    title = job.get("title", "") or ""
    description = job.get("full_description", "") or ""

    # Hard-cap short-circuit: auto-reject disqualifiers without burning
    # an LLM call. This is both a quality gate and a huge speed win.
    quick = _title_quick_filter(title, description)
    if quick is not None:
        score, reason = quick
        return {
            "score": score,
            "keywords": "",
            "reasoning": reason,
            "lane": classify_job_lane(title, job.get("site", ""), description),
        }

    strong_hint = (
        "\n\nHINT: Title matches one of Darrin's target role patterns — "
        "if the description also fits (remote, non-phone, IC), score 7+."
        if _title_has_strong_fit(title)
        else ""
    )

    job_text = (
        f"TITLE: {title}\n"
        f"COMPANY: {job['site']}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{description[:6000]}{strong_hint}"
    )

    messages = [
        {"role": "system", "content": SCORE_PROMPT},
        {"role": "user", "content": f"RESUME:\n{resume_text}\n\n---\n\nJOB POSTING:\n{job_text}"},
    ]

    try:
        client = get_client()
        response = client.chat(messages, max_tokens=512, temperature=0.2)
        parsed = _parse_score_response(response)
        raw_score = parsed["score"]
        adjusted_score, cap_note = _apply_contextual_caps(
            score=raw_score,
            title=title,
            location=job.get("location", "") or "",
            description=description,
        )
        parsed["score"] = adjusted_score
        if cap_note and adjusted_score < raw_score:
            parsed["reasoning"] = f"{parsed['reasoning']} Score capped due to {cap_note}."
        parsed["lane"] = classify_job_lane(title, job.get("site", ""), description)
        return parsed
    except Exception as e:
        log.error("LLM error scoring job '%s': %s", job.get("title", "?"), e)
        return {
            "score": 0,
            "keywords": "",
            "reasoning": f"LLM error: {e}",
            "lane": classify_job_lane(title, job.get("site", ""), description),
        }


def run_scoring(limit: int = 0, rescore: bool = False) -> dict:
    """Score unscored jobs that have full descriptions.

    Args:
        limit: Maximum number of jobs to score in this run.
        rescore: If True, re-score all jobs (not just unscored ones).

    Returns:
        {"scored": int, "errors": int, "elapsed": float, "distribution": list}
    """
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    if rescore:
        query = "SELECT * FROM jobs WHERE full_description IS NOT NULL"
        if limit > 0:
            query += f" LIMIT {limit}"
        jobs = conn.execute(query).fetchall()
    else:
        jobs = get_jobs_by_stage(conn=conn, stage="pending_score", limit=limit)

    if not jobs:
        log.info("No unscored jobs with descriptions found.")
        return {"scored": 0, "errors": 0, "elapsed": 0.0, "distribution": []}

    # Convert sqlite3.Row to dicts if needed
    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    log.info("Scoring %d jobs sequentially...", len(jobs))
    t0 = time.time()
    completed = 0
    errors = 0
    auto_rejected = 0
    results: list[dict] = []

    for job in jobs:
        result = score_job(resume_text, job)
        result["url"] = job["url"]
        completed += 1

        if result["score"] == 0:
            errors += 1
        elif result["reasoning"].startswith("AUTO-REJECT"):
            auto_rejected += 1

        results.append(result)

        log.info(
            "[%d/%d] score=%d  %s",
            completed, len(jobs), result["score"], job.get("title", "?")[:60],
        )

    # Write scores to DB
    now = datetime.now(timezone.utc).isoformat()
    for r in results:
        conn.execute(
            "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ?, job_lane = ? WHERE url = ?",
            (r["score"], f"{r['keywords']}\n{r['reasoning']}", now, r.get("lane", "primary"), r["url"]),
        )
    conn.commit()

    elapsed = time.time() - t0
    log.info(
        "Done: %d scored in %.1fs (%.1f jobs/sec). Auto-rejected: %d.",
        len(results), elapsed,
        len(results) / elapsed if elapsed > 0 else 0,
        auto_rejected,
    )

    # Score distribution
    dist = conn.execute("""
        SELECT fit_score, COUNT(*) FROM jobs
        WHERE fit_score IS NOT NULL
        GROUP BY fit_score ORDER BY fit_score DESC
    """).fetchall()
    distribution = [(row[0], row[1]) for row in dist]

    return {
        "scored": len(results),
        "errors": errors,
        "elapsed": elapsed,
        "distribution": distribution,
        "auto_rejected": auto_rejected,
    }
