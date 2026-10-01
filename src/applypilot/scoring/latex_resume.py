"""LaTeX resume tailoring: edit the user's own (e.g. Overleaf) resume per job.

Used instead of the built-in JSON/HTML template when ~/.applypilot/resume.tex
exists. The LLM rewrites only the document body; the preamble (packages,
macros, layout) is always restored from the original so formatting can't
drift. Each attempt is compiled and checked:

  1. compiles (LaTeX errors are fed back to the LLM on retry)
  2. same page count as the original
  3. no numbers that aren't in the original (catches invented metrics)
  4. an LLM fact-check against the original resume text

Instructions come from resume_prompt.md when present, else a default.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from applypilot import config
from applypilot.llm import get_client

log = logging.getLogger(__name__)

DEFAULT_RESUME_INSTRUCTIONS = """\
Tailor my resume to this job so a recruiter scanning it for 6 seconds sees a match.

- Reword bullet points to use the job description's language and keywords where my real experience supports it.
- Put the most relevant bullets first within each role, and the most relevant projects first.
- Reorder skills so the ones the job asks for come first. Only list skills already on my resume.
- Keep every job, every project, education, and leadership entry. Only trim a bullet if needed to stay on one page.
- Keep all names, titles, dates, locations, contact info, and section headings exactly as they are.
- Keep every number exactly as written. Do not add new metrics.
- Write like an engineer: short, concrete, no buzzwords ("leveraged", "spearheaded", "passionate", "synergy").
"""

_HARD_RULES = r"""
== NON-NEGOTIABLE RULES ==
- You are editing a LaTeX document. Return the COMPLETE document from \documentclass to \end{document}.
- Do not change the preamble (everything before \begin{document}); it will be restored from the original anyway.
- Only change text inside the existing commands/environments. Keep every macro name and argument structure (e.g. \resumeItem{...}, \resumeSubheading{..}{..}{..}{..}) exactly as used in the original.
- Never invent experience, employers, degrees, tools, or numbers. Everything must be supported by the original resume.
- Escape LaTeX special characters in any text you write: \% \& \$ \# \_
- It must fit on the same number of pages as the original.
- Output ONLY the LaTeX source. No markdown code fences, no commentary.
"""

_JUDGE_PROMPT = """You are a strict fact-checker comparing a tailored resume to the candidate's ORIGINAL resume.
Rewording and reordering are fine. FAIL the resume if ANY bullet:
- adds something the original does not support: employers, roles, dates, degrees, tools, responsibilities, numbers
- changes what was done (e.g. "coordinated" -> "analyzed", "detects" -> "extracts", "built" -> "trained")
- replaces a specific technical term with a different one (e.g. "LLMs" -> "deep learning models")
- adds scale or scope the original doesn't state (e.g. "large-scale", "scalable", "enterprise", "for users")
- turns an estimate or partial result into a stronger claim (e.g. "reduce" -> "eliminate")
Compare bullet by bullet. Be strict: when in doubt, FAIL and name the bullet.

Reply in exactly this format:
VERDICT: PASS or FAIL
ISSUES: none, or a short list of unsupported claims"""

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Scale/prestige/buzzwords the LLM likes to sneak in. Rejected unless the original already uses them.
_INFLATION_WORDS = (
    "scalable", "large-scale", "high-scale", "enterprise", "advanced", "state-of-the-art",
    "production-grade", "cutting-edge", "world-class", "robust", "seamless", "seamlessly",
    "leveraged", "leveraging", "spearheaded", "utilized", "synergy", "innovative",
)


# ---------------------------------------------------------------------------
# LaTeX compilation
# ---------------------------------------------------------------------------

def find_latex_engine() -> list[str] | None:
    """Return the base command for a LaTeX compiler, preferring Tectonic."""
    candidates = [os.environ.get("TECTONIC_PATH", ""), shutil.which("tectonic") or "",
                  str(Path.home() / ".local/bin/tectonic")]
    for c in candidates:
        if c and Path(c).exists():
            return [c, "-X", "compile"]
    if shutil.which("pdflatex"):
        return ["pdflatex", "-interaction=nonstopmode", "-halt-on-error"]
    return None


# pdfTeX-only lines common in Overleaf templates (Jake's Resume). XeTeX, which
# Tectonic uses, already emits Unicode-mapped (ATS-readable) text without them.
_PDFTEX_ONLY = re.compile(r"^[ \t]*(\\input\{glyphtounicode\}|\\pdfgentounicode\s*=\s*1)[ \t]*$", re.M)


def compile_tex(tex: str, out_pdf: Path) -> tuple[bool, str]:
    """Compile LaTeX source to out_pdf. Returns (ok, log_tail)."""
    engine = find_latex_engine()
    if engine is None:
        return False, "No LaTeX compiler found (install tectonic or pdflatex)."
    if "pdflatex" not in engine[0]:
        tex = _PDFTEX_ONLY.sub(r"% \1  (pdfTeX-only, skipped)", tex)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "resume.tex"
        src.write_text(tex, encoding="utf-8")
        try:
            proc = subprocess.run(engine + [src.name], cwd=tmp, capture_output=True,
                                  text=True, timeout=240)
        except subprocess.TimeoutExpired:
            return False, "LaTeX compile timed out"
        built = Path(tmp) / "resume.pdf"
        if proc.returncode != 0 or not built.exists():
            out = (proc.stdout + proc.stderr).strip().splitlines()
            errs = [ln for ln in out if "error" in ln.lower() or ln.startswith("!")]
            return False, "\n".join((errs or out)[-15:])
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(built, out_pdf)
    return True, ""


def pdf_info(pdf: Path) -> tuple[int, str]:
    """Return (page_count, extracted_text) for a PDF."""
    from pypdf import PdfReader
    reader = PdfReader(str(pdf))
    return len(reader.pages), "\n".join(p.extract_text() or "" for p in reader.pages)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split(tex: str) -> tuple[str, str]:
    idx = tex.find(r"\begin{document}")
    if idx < 0:
        raise ValueError(r"resume.tex has no \begin{document}")
    return tex[:idx], tex[idx:]


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:latex|tex)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER_RE.findall(text)}


# Skill lines in the Jake's Resume style: \textbf{Languages}{: Python, Java, ...}
_SKILL_LINE_RE = re.compile(r"(\\textbf\{([^{}]+)\}\{:\s*)([^{}]*)(\})")
_SKILL_SPLIT_RE = re.compile(r",\s*(?![^()]*\))")  # commas not inside "AWS (DynamoDB, Cognito)"
_MIN_SKILLS = 3


def _skills_instructions(instructions: str) -> str | None:
    """The '## Choosing skills' section of the user's prompt, if present."""
    m = re.search(r"^##\s*Choosing skills\s*$(.*?)(?=^##\s|\Z)", instructions, re.M | re.S | re.I)
    return m.group(1).strip() if m else None


def _original_skills(orig_body: str) -> dict[str, list[str]]:
    return {label: [s.strip() for s in _SKILL_SPLIT_RE.split(items) if s.strip()]
            for _, label, items, _ in _SKILL_LINE_RE.findall(orig_body)}


def choose_skills(orig_body: str, job_text: str, rules: str) -> dict[str, list[str]] | None:
    """Ask the LLM which of the ORIGINAL skills to show for this job (one call per job)."""
    import json

    original = _original_skills(orig_body)
    if not original:
        return None
    listing = "\n".join(f"{label}: {', '.join(items)}" for label, items in original.items())
    raw = get_client().chat([
        {"role": "system", "content": (
            f"{rules}\n\nReturn ONLY a JSON object mapping each category label to the list of skills to show, "
            "most relevant first, copied exactly from the list. No commentary.")},
        {"role": "user", "content": f"SKILLS ON RESUME:\n{listing}\n\nTARGET JOB:\n{job_text}"},
    ], max_tokens=4096, temperature=0.2)
    try:
        chosen = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    except (AttributeError, ValueError):
        log.warning("Skill selection returned no JSON; keeping skills as tailored")
        return None
    return {k: v for k, v in chosen.items() if isinstance(v, list)}


def apply_skills(body: str, orig_body: str, chosen: dict[str, list[str]]) -> str:
    """Rebuild skill lines from the original names in the chosen order.

    A skill can be dropped or reordered but never added, renamed, or misspelled.
    """
    original = _original_skills(orig_body)

    def rebuild(m: re.Match) -> str:
        label = m.group(2)
        pool = original.get(label)
        if not pool or label not in chosen:
            return m.group(0)
        by_key = {s.lower(): s for s in pool}
        picked = []
        for s in chosen[label]:
            name = by_key.get(str(s).strip().lower())
            if name and name not in picked:
                picked.append(name)
        for s in pool:  # top up to the minimum in original order
            if len(picked) >= _MIN_SKILLS:
                break
            if s not in picked:
                picked.append(s)
        return f"{m.group(1)}{', '.join(picked)}{m.group(4)}"

    return _SKILL_LINE_RE.sub(rebuild, body)


def _sections(body: str) -> list[str]:
    """Section titles in order, ignoring commented-out lines."""
    live = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("%"))
    return re.findall(r"\\section\*?\{([^}]*)\}", live)


def load_instructions(path: Path, default: str) -> str:
    if path.exists():
        custom = path.read_text(encoding="utf-8").strip()
        if custom:
            return custom
    return default


def _judge(original_text: str, tailored_text: str, job_title: str) -> dict:
    messages = [
        {"role": "system", "content": _JUDGE_PROMPT},
        {"role": "user", "content": (
            f"JOB: {job_title}\n\nORIGINAL RESUME:\n{original_text}\n\n---\n\nTAILORED RESUME:\n{tailored_text}"
        )},
    ]
    raw = get_client().chat(messages, max_tokens=4096, temperature=0.0)
    passed = "VERDICT: PASS" in raw.upper()
    issues = raw.split("ISSUES:", 1)[-1].strip() if "ISSUES:" in raw else raw.strip()
    return {"passed": passed, "issues": issues[:500]}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_original_cache: dict[str, tuple[int, str]] = {}


def original_info(tex: str) -> tuple[int, str]:
    """Compile the untouched resume once to learn its page count and text."""
    key = str(hash(tex))
    if key not in _original_cache:
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "orig.pdf"
            ok, err = compile_tex(tex, pdf)
            if not ok:
                raise RuntimeError(f"Your resume.tex does not compile:\n{err}")
            _original_cache[key] = pdf_info(pdf)
    return _original_cache[key]


def tailor_latex(tex: str, job: dict, out_pdf: Path, max_retries: int = 3,
                 validation_mode: str = "normal") -> tuple[str, str, dict]:
    """Tailor a LaTeX resume for one job and compile it to out_pdf.

    Returns:
        (tailored_tex, tailored_plain_text, report). report["status"] is
        "approved" when a compiled, checked PDF was written.
    """
    preamble, orig_body = _split(tex)
    orig_pages, orig_text = original_info(tex)
    orig_numbers = _numbers(orig_text)
    orig_sections = _sections(orig_body)
    instructions = load_instructions(config.RESUME_PROMPT_PATH, DEFAULT_RESUME_INSTRUCTIONS)
    skills_rules = _skills_instructions(instructions)
    job_text = (
        f"TITLE: {job['title']}\nCOMPANY: {job['site']}\nLOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )

    report: dict = {"status": "failed", "attempts": 0, "issues": []}
    client = get_client()

    chosen_skills = None
    if skills_rules:
        try:
            chosen_skills = choose_skills(orig_body, job_text, skills_rules)
            report["skills"] = chosen_skills
        except Exception as e:
            log.warning("Skill selection failed (%s); keeping skills as tailored", e)
    best: tuple[str, str] = ("", "")

    for attempt in range(max_retries + 1):
        report["attempts"] = attempt + 1
        system = f"{instructions}\n{_HARD_RULES}"
        if report["issues"]:
            system += "\n== FIX THESE PROBLEMS FROM YOUR LAST ATTEMPT ==\n" + "\n".join(
                f"- {i}" for i in report["issues"][-4:])

        raw = client.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": f"ORIGINAL LATEX RESUME:\n{tex}\n\n---\n\nTARGET JOB:\n{job_text}"},
        ], max_tokens=16000, temperature=0.4)

        try:
            _, body = _split(_strip_fences(raw))
        except ValueError:
            report["issues"].append(r"Output was not a complete LaTeX document (missing \begin{document}).")
            continue
        if chosen_skills:
            body = apply_skills(body, orig_body, chosen_skills)
        candidate = preamble + body  # formatting always comes from the original

        ok, err = compile_tex(candidate, out_pdf)
        if not ok:
            report["issues"].append(f"LaTeX failed to compile:\n{err}")
            continue

        pages, text = pdf_info(out_pdf)
        best = (candidate, text)
        if pages > orig_pages:
            report["issues"].append(f"Resume is {pages} pages; it must fit on {orig_pages}. Shorten bullets.")
            continue

        if _sections(body) != orig_sections:
            report["issues"].append(
                f"Sections changed. Keep exactly these sections in this order: {', '.join(orig_sections)}")
            continue

        tailored_numbers = _numbers(text)
        invented = sorted(tailored_numbers - orig_numbers)
        dropped = sorted(orig_numbers - tailored_numbers)
        if invented and validation_mode != "lenient":
            report["issues"].append(f"These numbers are not in the original resume, remove them: {', '.join(invented[:8])}")
            continue
        if dropped and validation_mode != "lenient":
            report["issues"].append(f"These numbers from the original are missing, keep them: {', '.join(dropped[:8])}")
            continue
        report["dropped_numbers"] = dropped

        if validation_mode != "lenient":
            lower, orig_lower = text.lower(), orig_text.lower()
            inflated = [w for w in _INFLATION_WORDS
                        if re.search(rf"\b{re.escape(w)}\b", lower) and not re.search(rf"\b{re.escape(w)}\b", orig_lower)]
            if inflated:
                report["issues"].append(f"Remove inflated wording not in the original: {', '.join(inflated)}")
                continue

        if validation_mode != "lenient":
            verdict = _judge(orig_text, text, job.get("title", ""))
            report["judge"] = verdict
            if not verdict["passed"]:
                report["issues"].append(f"Fact-check failed: {verdict['issues']}")
                continue

        report["status"] = "approved"
        return candidate, text, report

    # Out of retries: keep the last compiled version on disk for inspection
    if best[0]:
        compile_tex(best[0], out_pdf)
    return best[0], best[1], report
