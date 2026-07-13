"""Resume file helpers.

Supports plain text resumes, PDF inputs, and LaTeX source resumes. The LaTeX
path is used to derive a text version for LLM stages and to auto-compile a PDF
when possible.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from applypilot.config import RESUME_PATH, RESUME_PDF_PATH, RESUME_TEX_PATH

log = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_latex(tex: str) -> str:
    """Best-effort LaTeX to plain-text conversion.

    Uses pandoc when available. Falls back to a lightweight regex-based
    stripper so AI stages still have a usable resume text file.
    """
    pandoc = shutil.which("pandoc")
    if pandoc:
        try:
            proc = subprocess.run(
                [pandoc, "-f", "latex", "-t", "plain", "--wrap=none"],
                input=tex,
                text=True,
                capture_output=True,
                check=True,
            )
            return _normalize_text(proc.stdout)
        except subprocess.CalledProcessError:
            log.debug("pandoc LaTeX extraction failed", exc_info=True)

    text = tex
    text = re.sub(r"(?m)^%.*$", "", text)
    text = re.sub(r"\\documentclass(?:\[[^\]]*\])?\{[^}]*\}", "", text)
    text = re.sub(r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\}", "", text)
    text = re.sub(r"\\begin\{(?:document|center|flushleft|flushright)\}", "\n", text)
    text = re.sub(r"\\end\{(?:document|center|flushleft|flushright)\}", "\n", text)
    text = re.sub(r"\\section\*?\{([^}]*)\}", r"\n\1\n", text)
    text = re.sub(r"\\subsection\*?\{([^}]*)\}", r"\n\1\n", text)
    text = re.sub(r"\\subsubsection\*?\{([^}]*)\}", r"\n\1\n", text)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\url\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\item\s*", "\n- ", text)
    text = re.sub(r"\\begin\{(?:itemize|enumerate)\}", "\n", text)
    text = re.sub(r"\\end\{(?:itemize|enumerate)\}", "\n", text)
    text = re.sub(r"\\\\", "\n", text)
    text = re.sub(r"\\[a-zA-Z@]+(?:\[[^\]]*\])?(?:\{[^}]*\})*", "", text)
    text = re.sub(r"\$[^$]*\$", "", text)
    text = re.sub(r"[{}]", "", text)
    return _normalize_text(text)


def load_resume_text() -> str:
    """Load the base resume text for AI stages.

    Prefers `resume.txt`, then falls back to a LaTeX source by extracting
    plain text from it.
    """
    if RESUME_PATH.exists():
        return RESUME_PATH.read_text(encoding="utf-8")

    if RESUME_TEX_PATH.exists():
        return extract_text_from_latex(RESUME_TEX_PATH.read_text(encoding="utf-8"))

    raise FileNotFoundError(
        f"Resume not found. Expected {RESUME_PATH} or {RESUME_TEX_PATH}."
    )


def compile_latex_to_pdf(tex_path: Path, output_path: Path | None = None) -> Path:
    """Compile a LaTeX document into PDF.

    Uses latexmk when available, falling back to xelatex or pdflatex.
    """
    tex_path = Path(tex_path).expanduser().resolve()
    if not tex_path.exists():
        raise FileNotFoundError(tex_path)

    output_path = Path(output_path) if output_path else tex_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    latexmk = shutil.which("latexmk")
    xelatex = shutil.which("xelatex")
    pdflatex = shutil.which("pdflatex")
    if not any((latexmk, xelatex, pdflatex)):
        raise FileNotFoundError("No LaTeX engine found (latexmk, xelatex, or pdflatex)")

    with tempfile.TemporaryDirectory(prefix="applypilot-latex-") as tmp:
        tmpdir = Path(tmp)
        cwd = tex_path.parent

        if latexmk:
            cmd = [
                latexmk,
                "-cd",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                f"-output-directory={tmpdir}",
            ]
            if xelatex:
                cmd.append("-xelatex")
            else:
                cmd.append("-pdf")
            cmd.append(tex_path.name)
            proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError((proc.stdout + "\n" + proc.stderr).strip())
        else:
            engine = xelatex or pdflatex
            assert engine is not None
            cmd = [
                engine,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                f"-output-directory={tmpdir}",
                tex_path.name,
            ]
            proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError((proc.stdout + "\n" + proc.stderr).strip())
            # A second pass helps resolve references when latexmk is unavailable.
            proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError((proc.stdout + "\n" + proc.stderr).strip())

        produced = tmpdir / tex_path.with_suffix(".pdf").name
        if not produced.exists():
            raise RuntimeError(f"LaTeX compile finished without producing {produced.name}")

        shutil.copy2(produced, output_path)
        return output_path


def ingest_resume_source(src: Path) -> dict[str, Path]:
    """Copy a source resume into the ApplyPilot profile directory.

    Returns a mapping of artifact name to written path.
    """
    src = Path(src).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    artifacts: dict[str, Path] = {}
    suffix = src.suffix.lower()

    if suffix == ".txt":
        shutil.copy2(src, RESUME_PATH)
        artifacts["resume_txt"] = RESUME_PATH
    elif suffix == ".pdf":
        shutil.copy2(src, RESUME_PDF_PATH)
        artifacts["resume_pdf"] = RESUME_PDF_PATH
    elif suffix == ".tex":
        shutil.copy2(src, RESUME_TEX_PATH)
        artifacts["resume_tex"] = RESUME_TEX_PATH

        extracted = extract_text_from_latex(src.read_text(encoding="utf-8"))
        RESUME_PATH.write_text(extracted + "\n", encoding="utf-8")
        artifacts["resume_txt"] = RESUME_PATH

        try:
            compile_latex_to_pdf(RESUME_TEX_PATH, RESUME_PDF_PATH)
            artifacts["resume_pdf"] = RESUME_PDF_PATH
        except Exception as exc:
            log.warning("LaTeX resume compile failed: %s", exc)
    else:
        raise ValueError("Unsupported resume format. Use .txt, .pdf, or .tex.")

    return artifacts
