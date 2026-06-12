"""LaTeX resume support for the tailor stage.

When ~/.applypilot/resume.tex exists, tailoring operates on the LaTeX source
instead of plain text: the preamble is preserved verbatim (the LLM never sees
it), only the document body is rewritten, and the result is compiled with
latexmk/pdflatex so the layout -- and crucially the page count -- is
deterministic. The compiled page count is checked against the base resume's
page count and overflowing attempts are retried with shortening feedback.
"""

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from applypilot.config import RESUME_PATH, RESUME_TEX_PATH
from applypilot.scoring.validator import BANNED_WORDS

log = logging.getLogger(__name__)

COMPILE_TIMEOUT = 120  # seconds per latexmk run


class TexCompileError(Exception):
    """Raised when the LaTeX source fails to compile. Message holds log excerpt."""


# ── Source splitting ─────────────────────────────────────────────────────

def split_tex(source: str) -> tuple[str, str, str]:
    """Split a .tex file into (preamble, body, ending).

    preamble: everything up to and including \\begin{document}
    body:     content between \\begin{document} and \\end{document}
    ending:   \\end{document} and anything after it

    Raises:
        ValueError: if the source has no \\begin{document}.
    """
    m_begin = re.search(r"\\begin\{document\}", source)
    if not m_begin:
        raise ValueError("resume.tex has no \\begin{document} -- not a complete LaTeX document")

    m_end = re.search(r"\\end\{document\}", source)
    end_pos = m_end.start() if m_end else len(source)

    preamble = source[: m_begin.end()]
    body = source[m_begin.end(): end_pos]
    ending = source[end_pos:] if m_end else "\\end{document}\n"
    return preamble, body, ending


def extract_tex_body(raw: str) -> str:
    """Extract the LaTeX body from an LLM response (handles fences, full docs)."""
    raw = raw.strip()

    # Prefer fenced code blocks if present
    if "```" in raw:
        blocks = raw.split("```")[1::2]
        if blocks:
            best = max(blocks, key=len).strip()
            for tag in ("latex", "tex"):
                if best.lower().startswith(tag):
                    best = best[len(tag):].strip()
            raw = best

    # If the model returned a full document, slice out the body
    m_begin = re.search(r"\\begin\{document\}", raw)
    if m_begin:
        m_end = re.search(r"\\end\{document\}", raw)
        raw = raw[m_begin.end(): m_end.start() if m_end else len(raw)].strip()

    return raw


# ── Plain-text extraction (for validation / judge / apply prompt) ────────

def strip_tex(tex: str) -> str:
    """Convert LaTeX source to approximate plain text.

    Good enough for substring validation (company names, banned words) and
    for the LLM judge -- not meant to be a faithful renderer.
    """
    text = re.sub(r"(?<!\\)%.*", "", tex)                    # comments
    text = text.replace("\\\\", "\n")                        # line breaks
    # Spacing/format commands whose arguments are dimensions, not content
    text = re.sub(r"\\(?:vspace|hspace|fontsize|titlespacing|needspace|setlength)\*?(?:\{[^}]*\})+", "", text)
    text = re.sub(r"\\kern\s*[-\d.]+\s*(?:pt|cm|mm|em|ex|in)", "", text)
    # \href and aliases like \hrefWithoutArrow
    text = re.sub(r"\\[a-zA-Z]*[Hh]ref[a-zA-Z]*\{([^}]*)\}\{([^}]*)\}", r"\2 (\1)", text)
    text = re.sub(r"\((?:mailto|tel):[^)]*\)", "", text)     # redundant link targets
    text = re.sub(r"\\(?:sub)*section\*?\{([^}]*)\}", r"\n\n\1\n", text)
    text = re.sub(r"\\item\b", "\n- ", text)
    text = re.sub(r"\\begin\{[^}]*\}(\[[^\]]*\])?", "\n", text)
    text = re.sub(r"\\end\{[^}]*\}", "\n", text)
    text = re.sub(r"(?<!\\)&", " | ", text)                  # tabular alignment
    # Collapse single-argument commands to their content, innermost-out
    for _ in range(6):
        new = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", text)
        if new == text:
            break
        text = new
    text = re.sub(r"\\[a-zA-Z@]+\*?", "", text)              # bare commands
    text = text.replace("~", " ")
    for esc, ch in (("\\%", "%"), ("\\&", "&"), ("\\$", "$"), ("\\#", "#"), ("\\_", "_")):
        text = text.replace(esc, ch)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Drop lines with no alphanumeric content (leftover symbols like "$|$")
    lines = [ln for ln in text.splitlines() if re.search(r"[A-Za-z0-9]", ln) or not ln.strip()]
    text = "\n".join(ln.rstrip() for ln in lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_base_resume_text() -> str:
    """Load the base resume as plain text for scoring/cover-letter stages.

    Prefers resume.txt; falls back to stripping resume.tex (and caches the
    result to resume.txt so subsequent stages can read it directly).
    """
    tex_exists = RESUME_TEX_PATH.exists()
    if RESUME_PATH.exists():
        # Regenerate if resume.tex was edited after the cached txt was made
        if not (tex_exists and RESUME_TEX_PATH.stat().st_mtime > RESUME_PATH.stat().st_mtime):
            return RESUME_PATH.read_text(encoding="utf-8")
        log.info("resume.tex is newer than resume.txt -- regenerating")
    if tex_exists:
        _, body, _ = split_tex(RESUME_TEX_PATH.read_text(encoding="utf-8"))
        text = strip_tex(body)
        RESUME_PATH.write_text(text, encoding="utf-8")
        log.info("Generated %s from resume.tex", RESUME_PATH)
        return text
    raise FileNotFoundError(
        f"No resume found. Add {RESUME_PATH} or {RESUME_TEX_PATH} (run `applypilot init`)."
    )


# ── Compilation ──────────────────────────────────────────────────────────

def _pick_engine(source: str) -> str:
    """Pick xelatex for fontspec/CJK documents, pdflatex otherwise."""
    if re.search(r"fontspec|xeCJK|unicode-math|ctex", source):
        return "xelatex"
    return "pdflatex"


def _extract_log_errors(log_text: str) -> str:
    """Pull the '!' error lines (plus context) out of a LaTeX log."""
    lines = log_text.splitlines()
    errors: list[str] = []
    for i, line in enumerate(lines):
        if line.startswith("!"):
            errors.extend(lines[i: i + 3])
    return "\n".join(errors)[:600] or log_text[-600:]


def compile_tex(source: str) -> tuple[bytes, int]:
    """Compile LaTeX source in a temp dir.

    Returns:
        (pdf_bytes, page_count)

    Raises:
        TexCompileError: with a log excerpt if compilation fails.
        FileNotFoundError: if no LaTeX engine is installed.
    """
    engine = _pick_engine(source)

    with tempfile.TemporaryDirectory() as td:
        tex_file = Path(td) / "resume.tex"
        tex_file.write_text(source, encoding="utf-8")

        latexmk = shutil.which("latexmk")
        if latexmk:
            flag = "-xelatex" if engine == "xelatex" else "-pdf"
            cmds = [[latexmk, flag, "-interaction=nonstopmode", "-halt-on-error", "resume.tex"]]
        elif shutil.which(engine):
            # No latexmk: run the engine twice for cross-references
            cmd = [engine, "-interaction=nonstopmode", "-halt-on-error", "resume.tex"]
            cmds = [cmd, cmd]
        else:
            raise FileNotFoundError(
                "No LaTeX engine found. Install TeX Live / MacTeX (latexmk, pdflatex)."
            )

        for cmd in cmds:
            try:
                proc = subprocess.run(
                    cmd, cwd=td, capture_output=True, text=True, timeout=COMPILE_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                raise TexCompileError(f"LaTeX compilation timed out after {COMPILE_TIMEOUT}s")
            if proc.returncode != 0:
                break  # inspect log below

        log_file = Path(td) / "resume.log"
        log_text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""

        pdf_file = Path(td) / "resume.pdf"
        if not pdf_file.exists():
            raise TexCompileError(_extract_log_errors(log_text or proc.stdout))

        m = re.search(r"Output written on .*?\((\d+) pages?", log_text)
        pages = int(m.group(1)) if m else 0
        if not pages:
            # Fallback: count page objects in the PDF itself
            pages = len(re.findall(rb"/Type\s*/Page\b(?!s)", pdf_file.read_bytes())) or 1

        return pdf_file.read_bytes(), pages


# ── Validation (tex-mode counterpart of validate_json_fields) ────────────

def validate_tex_body(new_body: str, original_body: str, profile: dict,
                      mode: str = "normal") -> dict:
    """Programmatic checks on a tailored LaTeX body.

    Structure must match the original (same \\section titles), preserved
    companies must survive, banned words follow the validation mode.

    Returns:
        {"passed": bool, "errors": list[str], "warnings": list[str]}
    """
    errors: list[str] = []
    warnings: list[str] = []

    if re.search(r"\\documentclass|\\usepackage|\\begin\{document\}", new_body):
        errors.append(
            "Output must be the document BODY only -- no \\documentclass, "
            "\\usepackage, or \\begin{document}."
        )

    # Same section structure as the original
    orig_sections = re.findall(r"\\section\*?\{([^}]*)\}", original_body)
    for sec in orig_sections:
        if sec not in new_body:
            errors.append(f"Missing \\section{{{sec}}} -- keep the original section structure.")

    plain_lower = strip_tex(new_body).lower()
    resume_facts = profile.get("resume_facts", {})

    for company in resume_facts.get("preserved_companies", []):
        if company.lower() not in plain_lower:
            errors.append(f"Company '{company}' missing -- cannot remove real experience.")

    for metric in resume_facts.get("real_metrics", []):
        if str(metric).lower() not in plain_lower:
            warnings.append(f"Real metric '{metric}' not found -- verify it wasn't altered.")

    if mode != "lenient":
        found = [w for w in BANNED_WORDS if re.search(rf"\b{re.escape(w)}\b", plain_lower)]
        if found:
            msg = "Banned words found: " + ", ".join(found)
            (errors if mode == "strict" else warnings).append(msg)

    return {"passed": not errors, "errors": errors, "warnings": warnings}
