"""Cover letter generation: LLM-powered, profile-driven, with validation.

Generates concise, engineering-voice cover letters tailored to specific job
postings. All personal data (name, skills, achievements) comes from the user's
profile at runtime. No hardcoded personal information.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone

from applypilot.config import (
    COVER_LETTER_BODY_PATH,
    COVER_LETTER_DIR,
    COVER_LETTER_PROMPT_PATH,
    RESUME_PATH,
    WRITING_SAMPLE_PATH,
    load_profile,
)
from applypilot.database import get_connection, get_jobs_by_stage
from applypilot.llm import get_client
from applypilot.scoring.validator import (
    BANNED_WORDS,
    LLM_LEAK_PHRASES,
    sanitize_text,
    validate_cover_letter,
)

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5  # max cross-run retries before giving up


# ── Prompt Builder (profile-driven) ──────────────────────────────────────

def _build_cover_letter_prompt(profile: dict) -> str:
    """Build the cover letter system prompt from the user's profile.

    All personal data, skills, and sign-off name come from the profile.
    """
    personal = profile.get("personal", {})
    boundary = profile.get("skills_boundary", {})
    resume_facts = profile.get("resume_facts", {})

    # Preferred name for the sign-off (falls back to full name)
    sign_off_name = personal.get("preferred_name") or personal.get("full_name", "")

    # Flatten all allowed skills
    all_skills: list[str] = []
    for items in boundary.values():
        if isinstance(items, list):
            all_skills.extend(items)
    skills_str = ", ".join(all_skills) if all_skills else "the tools listed in the resume"

    # Real metrics from resume_facts
    real_metrics = resume_facts.get("real_metrics", [])
    preserved_projects = resume_facts.get("preserved_projects", [])

    # Build achievement examples for the prompt
    projects_hint = ""
    if preserved_projects:
        projects_hint = f"\nKnown projects to reference: {', '.join(preserved_projects)}"

    metrics_hint = ""
    if real_metrics:
        metrics_hint = f"\nReal metrics to use: {', '.join(real_metrics)}"

    # Build the full banned list from the validator so the prompt stays in sync
    # with what will actually be rejected — the validator checks all of these.
    all_banned = ", ".join(f'"{w}"' for w in BANNED_WORDS)
    leak_banned = ", ".join(f'"{p}"' for p in LLM_LEAK_PHRASES)

    return f"""Write a cover letter for {sign_off_name}. The goal is to get an interview.

STRUCTURE: 3 short paragraphs. Under 250 words. Every sentence must earn its place.

PARAGRAPH 1 (2-3 sentences): Open with a specific thing YOU built that solves THEIR problem. Not "I'm excited about this role." Not "This role aligns with my experience." Start with the work.

PARAGRAPH 2 (3-4 sentences): Pick 2 achievements from the resume that are MOST relevant to THIS job. Use numbers. Frame as solving their problem, not listing your accomplishments.{projects_hint}{metrics_hint}

PARAGRAPH 3 (1-2 sentences): One specific thing about the company from the job description (a product, a technical challenge, a team structure). Then close. "Happy to walk through any of this in more detail." or "Let's discuss." Nothing else.

BANNED WORDS AND PHRASES (automated validator rejects ANY of these — do not use even once):
{all_banned}

ALSO BANNED (meta-commentary the validator catches):
{leak_banned}

BANNED PUNCTUATION: No em dashes (—) or en dashes (–). Use commas or periods.

VOICE:
- Write like a real engineer emailing someone they respect. Not formal, not casual. Just direct.
- NEVER narrate or explain what you're doing. BAD: "This demonstrates my commitment to X." GOOD: Just state the fact and move on.
- NEVER hedge. BAD: "might address some of your challenges." GOOD: "solves the same problem your team is facing."
- Every sentence should contain either a number, a tool name, or a specific outcome. If it doesn't, cut it.
- Read it out loud. If it sounds like a robot wrote it, rewrite it.

FABRICATION = INSTANT REJECTION:
The candidate's real tools are ONLY: {skills_str}.
Do NOT mention ANY tool not in this list. If the job asks for tools not listed, talk about the work you did, not the tools.

Sign off: just "{sign_off_name}"

Output ONLY the letter text. No subject lines. No "Here is the cover letter:" preamble. No notes after the sign-off.
Start DIRECTLY with "Dear Hiring Manager," and end with the name."""


# ── Helpers ──────────────────────────────────────────────────────────────

DEFAULT_COVER_LETTER_INSTRUCTIONS = """\
Write a cover letter for this job. 3 or 4 short paragraphs, under 300 words.

- Open with who I am (school, degree, and graduation date from my resume) and the role I'm applying for.
- Middle: connect 2 or 3 specific things from my resume to what the job description asks for. Use the real numbers.
- Close by saying why this company specifically, based on the job description, and that I'd welcome a conversation.
- Sound like a person, not a template. No buzzwords, no flattery, no "I am excited to apply".
- Don't guess at the company's problems or claim to know their internal challenges.
"""


def _build_custom_cover_letter_prompt(instructions: str, profile: dict) -> str:
    """User-written instructions plus the rules the renderer and validator rely on."""
    personal = profile.get("personal", {})
    sign_off = personal.get("preferred_name") or personal.get("full_name", "")
    return f"""{instructions.strip()}

== NON-NEGOTIABLE RULES ==
- Only use facts from the RESUME. Never invent experience, tools, numbers, or details about the company beyond the job description.
- Do not write a header, address, or date (they are added automatically).
- Start with "Dear Hiring Manager," and end with a sign-off followed by "{sign_off}".
- Plain text: paragraphs separated by a blank line, no em dashes. The only markdown allowed is **bold** for a quality label at the start of a paragraph.
- Output ONLY the letter. No "Here is your letter" or notes."""


# Words that mark text as AI-written (from Wikipedia's "Signs of AI writing")
_AI_TELLS = (
    "delve", "crucial", "pivotal", "landscape", "tapestry", "testament", "underscore", "underscores",
    "showcase", "showcases", "showcasing", "foster", "fostering", "enhance", "enhancing", "intricate",
    "vibrant", "align", "aligns", "aligned", "seamless", "seamlessly", "leverage", "leveraging",
    "embark", "journey", "thrilled", "resonate", "resonates", "invaluable",
)
# Scale/seniority claims a student's intro tends to inflate. OK only if the resume says it.
_CL_INFLATION = (
    "production-ready", "production-grade", "production models", "production systems", "high-volume",
    "large-scale", "scalable", "enterprise", "state-of-the-art", "extensive", "expert", "seasoned",
)

_CL_JUDGE_PROMPT = """You fact-check the INTRO and CLOSING of a cover letter.
Claims about the candidate must be supported by the RESUME. Claims about the company or role must be
supported by the JOB DESCRIPTION. FAIL if either paragraph states anything that isn't, including
inflated descriptions of the candidate's work (e.g. calling student or intern work "production").
Reply exactly:
VERDICT: PASS or FAIL
ISSUES: none, or the unsupported phrases"""


def _cl_tells(text: str, resume_text: str) -> list[str]:
    lower, resume_lower = text.lower(), resume_text.lower()
    has = lambda w, s: re.search(rf"\b{re.escape(w)}\b", s)
    issues = []
    tells = [w for w in _AI_TELLS if has(w, lower)]
    if tells:
        issues.append(f"Remove AI-sounding words: {', '.join(tells)}")
    inflated = [w for w in _CL_INFLATION if has(w, lower) and not has(w, resume_lower)]
    if inflated:
        issues.append(f"Remove claims my resume doesn't support: {', '.join(inflated)}")
    return issues


def _cl_judge(intro: str, closing: str, resume_text: str, job_text: str) -> str | None:
    """Return a problem description, or None if the fact-check passes."""
    raw = get_client().chat([
        {"role": "system", "content": _CL_JUDGE_PROMPT},
        {"role": "user", "content": (
            f"RESUME:\n{resume_text}\n\n---\n\nJOB DESCRIPTION:\n{job_text}\n\n---\n\n"
            f"INTRO:\n{intro}\n\nCLOSING:\n{closing}"
        )},
    ], max_tokens=4096, temperature=0.0)
    if "VERDICT: PASS" in raw.upper():
        return None
    return "Fact-check failed: " + (raw.split("ISSUES:", 1)[-1].strip() if "ISSUES:" in raw else raw.strip())[:400]


def _body_labels(body: str) -> list[str]:
    """Bold quality labels that start each body paragraph (e.g. **Machine Learning:**)."""
    return [m.strip().rstrip(":").strip() for m in re.findall(r"^\*\*([^*]+)\*\*", body, re.M)]


def _mentions(text: str, label: str) -> bool:
    """True if text refers to the quality, by stem: "Communicator" matches "communicate"."""
    lower = text.lower()
    words = [w for w in re.findall(r"[a-z]+", label.lower()) if len(w) >= 4] or [label.lower()]
    return any(w[:max(4, len(w) - 3)] in lower for w in words)


def _voice_sample() -> str:
    if WRITING_SAMPLE_PATH.exists():
        sample = WRITING_SAMPLE_PATH.read_text(encoding="utf-8").strip()
        if sample:
            return (
                "\n\n== MY WRITING SAMPLE (match this voice) ==\n"
                "Match its sentence length, word choice, and punctuation habits. Don't copy its content.\n"
                f"{sample[:3000]}"
            )
    return ""


def _generate_intro_closing(instructions: str, body: str, resume_text: str, job_text: str,
                            profile: dict, validation_mode: str, max_retries: int) -> str:
    """Write only the intro and closing around the user's fixed body paragraphs."""
    personal = profile.get("personal", {})
    name = personal.get("full_name", "")
    labels = _body_labels(body)
    qualities = ", ".join(labels) if labels else "the qualities in the body paragraphs"

    system = f"""{instructions.strip()}{_voice_sample()}

== YOUR TASK FOR THIS LETTER ==
The three body paragraphs are already written (below) and must not change. Write ONLY:
- "intro": the introduction paragraph. It must name these three qualities, in this order: {qualities}.
- "closing": the closing paragraph, specific to this company and role.
Only use facts from the RESUME and the job description. Don't invent anything about me or the company.
No greeting and no sign-off (they are added automatically). No em dashes.
Return ONLY a JSON object: {{"intro": "...", "closing": "..."}}"""

    client = get_client()
    avoid: list[str] = []
    letter = ""
    for attempt in range(max_retries + 1):
        prompt = system + ("\n\n== FIX THESE ISSUES ==\n" + "\n".join(f"- {a}" for a in avoid[-5:]) if avoid else "")
        raw = client.chat([
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"CANDIDATE NAME: {name}\n\nRESUME:\n{resume_text}\n\n---\n\n"
                f"BODY PARAGRAPHS (fixed):\n{body}\n\n---\n\nTARGET JOB:\n{job_text}"
            )},
        ], max_tokens=4096, temperature=0.7)
        try:
            parts = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            intro, closing = sanitize_text(parts["intro"]).strip(), sanitize_text(parts["closing"]).strip()
        except (AttributeError, ValueError, KeyError, TypeError):
            avoid.append('Return valid JSON with exactly the keys "intro" and "closing".')
            continue

        letter = f"Dear Hiring Manager,\n\n{intro}\n\n{body.strip()}\n\n{closing}\n\nSincerely,\n{name}"
        missing = [q for q in labels if not _mentions(intro, q)]
        validation = validate_cover_letter(letter, mode=validation_mode)
        errors = validation["errors"] + ([f"Intro must name: {', '.join(missing)}"] if missing else [])
        if "thank" not in closing.lower():
            errors.append("The closing must start by thanking the reader for their time.")
        if validation_mode != "lenient":
            errors += _cl_tells(f"{intro}\n{closing}", resume_text)
            if not errors:
                problem = _cl_judge(intro, closing, resume_text, job_text)
                if problem:
                    errors.append(problem)
        if not errors:
            return letter
        avoid.extend(errors)
    return letter


def _strip_preamble(text: str) -> str:
    """Remove LLM preamble before 'Dear Hiring Manager,' if present.

    Gemini and other models sometimes output "Here is the cover letter:" or
    similar meta-commentary before the actual letter text. Strip everything
    before the first occurrence of "Dear" so the validator's start-check passes.
    """
    dear_idx = text.lower().find("dear")
    if dear_idx > 0:
        return text[dear_idx:]
    return text


# ── Core Generation ──────────────────────────────────────────────────────

def generate_cover_letter(
    resume_text: str, job: dict, profile: dict,
    max_retries: int = 3, validation_mode: str = "normal",
) -> str:
    """Generate a cover letter with fresh context on each retry + auto-sanitize.

    Same design as tailor_resume: fresh conversation per attempt, issues noted
    in the prompt, no conversation history stacking.

    Args:
        resume_text:      The candidate's resume text (base or tailored).
        job:              Job dict with title, site, location, full_description.
        profile:          User profile dict.
        max_retries:      Maximum retry attempts.
        validation_mode:  "strict", "normal", or "lenient".

    Returns:
        The cover letter text (best attempt even if validation failed).
    """
    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job['site']}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )

    avoid_notes: list[str] = []
    letter = ""
    client = get_client()
    custom = (COVER_LETTER_PROMPT_PATH.read_text(encoding="utf-8").strip()
              if COVER_LETTER_PROMPT_PATH.exists() else "")
    body = (COVER_LETTER_BODY_PATH.read_text(encoding="utf-8").strip()
            if COVER_LETTER_BODY_PATH.exists() else "")
    if body:
        return _generate_intro_closing(custom or DEFAULT_COVER_LETTER_INSTRUCTIONS, body, resume_text,
                                       job_text, profile, validation_mode, max_retries)
    if custom:
        cl_prompt_base = _build_custom_cover_letter_prompt(custom, profile) + _voice_sample()
    else:
        cl_prompt_base = _build_cover_letter_prompt(profile)

    for attempt in range(max_retries + 1):
        # Fresh conversation every attempt
        prompt = cl_prompt_base
        if avoid_notes:
            prompt += "\n\n## AVOID THESE ISSUES:\n" + "\n".join(
                f"- {n}" for n in avoid_notes[-5:]
            )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"RESUME:\n{resume_text}\n\n---\n\n"
                f"TARGET JOB:\n{job_text}\n\n"
                "Write the cover letter:"
            )},
        ]

        letter = client.chat(messages, max_tokens=4096, temperature=0.7)
        letter = sanitize_text(letter)  # auto-fix em dashes, smart quotes
        letter = _strip_preamble(letter)  # remove any "Here is the letter:" prefix

        validation = validate_cover_letter(letter, mode=validation_mode)
        if validation["passed"]:
            return letter

        avoid_notes.extend(validation["errors"])
        # Warnings never block — only hard errors trigger a retry
        log.debug(
            "Cover letter attempt %d/%d failed: %s",
            attempt + 1, max_retries + 1, validation["errors"],
        )

    return letter  # last attempt even if failed


# ── Batch Entry Point ────────────────────────────────────────────────────

def run_cover_letters(min_score: int = 7, limit: int = 20,
                      validation_mode: str = "normal") -> dict:
    """Generate cover letters for high-scoring jobs that have tailored resumes.

    Args:
        min_score:       Minimum fit_score threshold.
        limit:           Maximum jobs to process.
        validation_mode: "strict", "normal", or "lenient".

    Returns:
        {"generated": int, "errors": int, "elapsed": float}
    """
    profile = load_profile()
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    # Fetch jobs that have tailored resumes but no cover letter yet
    jobs = conn.execute(
        "SELECT * FROM jobs "
        "WHERE fit_score >= ? AND tailored_resume_path IS NOT NULL "
        "AND full_description IS NOT NULL "
        "AND (cover_letter_path IS NULL OR cover_letter_path = '') "
        "AND COALESCE(cover_attempts, 0) < ? "
        "ORDER BY fit_score DESC LIMIT ?",
        (min_score, MAX_ATTEMPTS, limit),
    ).fetchall()

    if not jobs:
        log.info("No jobs needing cover letters (score >= %d).", min_score)
        return {"generated": 0, "errors": 0, "elapsed": 0.0}

    # Convert rows to dicts
    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    COVER_LETTER_DIR.mkdir(parents=True, exist_ok=True)
    log.info(
        "Generating cover letters for %d jobs (score >= %d)...",
        len(jobs), min_score,
    )
    t0 = time.time()
    completed = 0
    results: list[dict] = []
    error_count = 0

    for job in jobs:
        completed += 1
        try:
            letter = generate_cover_letter(resume_text, job, profile,
                                          validation_mode=validation_mode)

            # Build safe filename prefix
            safe_title = re.sub(r"[^\w\s-]", "", job["title"])[:50].strip().replace(" ", "_")
            safe_site = re.sub(r"[^\w\s-]", "", job["site"])[:20].strip().replace(" ", "_")
            prefix = f"{safe_site}_{safe_title}"

            cl_path = COVER_LETTER_DIR / f"{prefix}_CL.txt"
            cl_path.write_text(letter, encoding="utf-8")

            # Plain black-and-white .docx + PDF (best-effort)
            pdf_path = None
            try:
                from applypilot.scoring.letter_doc import render_letter
                pdf_path = str(render_letter(cl_path, profile, company=job.get("site", "")))
            except Exception:
                log.debug("PDF generation failed for %s", cl_path, exc_info=True)

            result = {
                "url": job["url"],
                "path": str(cl_path),
                "pdf_path": pdf_path,
                "title": job["title"],
                "site": job["site"],
            }
            results.append(result)

            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            log.info(
                "%d/%d [OK] | %.1f jobs/min | %s",
                completed, len(jobs), rate * 60, result["title"][:40],
            )
        except Exception as e:
            result = {
                "url": job["url"], "title": job["title"], "site": job["site"],
                "path": None, "pdf_path": None, "error": str(e),
            }
            error_count += 1
            results.append(result)
            log.error("%d/%d [ERROR] %s -- %s", completed, len(jobs), job["title"][:40], e)

    # Persist to DB: increment attempt counter for ALL, save path only for successes
    now = datetime.now(timezone.utc).isoformat()
    saved = 0
    for r in results:
        if r.get("path"):
            conn.execute(
                "UPDATE jobs SET cover_letter_path=?, cover_letter_at=?, "
                "cover_attempts=COALESCE(cover_attempts,0)+1 WHERE url=?",
                (r["path"], now, r["url"]),
            )
            saved += 1
        else:
            conn.execute(
                "UPDATE jobs SET cover_attempts=COALESCE(cover_attempts,0)+1 WHERE url=?",
                (r["url"],),
            )
    conn.commit()

    elapsed = time.time() - t0
    log.info("Cover letters done in %.1fs: %d generated, %d errors", elapsed, saved, error_count)

    return {
        "generated": saved,
        "errors": error_count,
        "elapsed": elapsed,
    }
