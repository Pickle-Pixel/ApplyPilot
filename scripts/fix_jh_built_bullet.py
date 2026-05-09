"""Replace the JH 'Built standardized trace-file diagnostics' bullet across
all 5 resume files (master + 4 variants) and the runtime resume.txt.

Replacement (per Darrin / Claw I1 audit):
  "Built standardized trace-file diagnostics and knowledge-base guidance"
  -> "Documented trace-file diagnostic procedures and knowledge-base guidance"

Backs up every touched file with the suffix `.bak-2026-04-28-jh-fix`.

Does NOT rebuild .pdf files. The runtime apply pipeline does its own per-job
PDF conversion from the tailored .txt; the resume-overhaul .pdf files are
reference artifacts whose rebuild is gated on having Word/LibreOffice/pandoc
with a PDF engine, none of which is available on this machine right now.
That rebuild is logged as a follow-up.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from docx import Document

OLD = "Built standardized trace-file diagnostics and knowledge-base guidance"
NEW = "Documented trace-file diagnostic procedures and knowledge-base guidance"

BACKUP_SUFFIX = ".bak-2026-04-28-jh-fix"

OVERHAUL_DIR = Path.home() / ".applypilot" / "resume-overhaul"
RUNTIME_TXT = Path.home() / ".applypilot" / "resume.txt"

DOCX_FILES = [
    OVERHAUL_DIR / "MASTER_RESUME.docx",
    OVERHAUL_DIR / "Resume_QA_Analyst.docx",
    OVERHAUL_DIR / "Resume_WFM_Analyst.docx",
    OVERHAUL_DIR / "Resume_Operations_TechSupport.docx",
    OVERHAUL_DIR / "Resume_Document_QA_Ops.docx",
]

TXT_FILES = [
    RUNTIME_TXT,
    OVERHAUL_DIR / "MASTER_RESUME.txt",
    OVERHAUL_DIR / "Resume_QA_Analyst.txt",
    OVERHAUL_DIR / "Resume_WFM_Analyst.txt",
    OVERHAUL_DIR / "Resume_Operations_TechSupport.txt",
    OVERHAUL_DIR / "Resume_Document_QA_Ops.txt",
]


def backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if bak.exists():
        # Don't overwrite an existing backup of this name (idempotent re-run)
        return bak
    shutil.copy2(path, bak)
    return bak


def fix_docx(path: Path) -> tuple[bool, int]:
    """Replace OLD with NEW in every paragraph and every run.

    python-docx splits paragraphs into "runs" (runs of consistent formatting).
    Doing a paragraph-level text replacement loses formatting; we instead
    replace at the run level when the OLD substring appears in a single run.
    If OLD spans multiple runs we fall back to a paragraph-level replace.
    """
    if not path.exists():
        return False, 0

    doc = Document(path)
    replacements = 0

    for para in doc.paragraphs:
        if OLD not in para.text:
            continue

        # Try run-level first
        for run in para.runs:
            if OLD in run.text:
                run.text = run.text.replace(OLD, NEW)
                replacements += 1

        # If still present, OLD spans multiple runs -> rebuild paragraph
        if OLD in para.text:
            # Concatenate run texts to verify, then rewrite paragraph text
            full = "".join(r.text for r in para.runs)
            if OLD in full:
                new_text = full.replace(OLD, NEW)
                # Clear all runs except the first; rewrite first run's text
                if para.runs:
                    para.runs[0].text = new_text
                    for run in para.runs[1:]:
                        run.text = ""
                    replacements += 1

    if replacements > 0:
        doc.save(path)
    return True, replacements


def fix_txt(path: Path) -> tuple[bool, int]:
    if not path.exists():
        return False, 0
    text = path.read_text(encoding="utf-8")
    if OLD not in text:
        return True, 0
    new_text = text.replace(OLD, NEW)
    path.write_text(new_text, encoding="utf-8")
    return True, text.count(OLD)


def main() -> int:
    print(f"OLD: {OLD!r}")
    print(f"NEW: {NEW!r}")
    print()

    print("=== .docx files ===")
    for p in DOCX_FILES:
        bak = backup(p)
        ok, n = fix_docx(p)
        status = "OK" if ok else "MISSING"
        print(f"  [{status}] {p.name:<35} replacements={n}  backup={bak.name if bak else '-'}")

    print()
    print("=== .txt files ===")
    for p in TXT_FILES:
        bak = backup(p)
        ok, n = fix_txt(p)
        status = "OK" if ok else "MISSING"
        print(f"  [{status}] {p.name:<35} replacements={n}  backup={bak.name if bak else '-'}")

    print()
    print("DONE. PDFs in resume-overhaul/ NOT rebuilt (no Word/LibreOffice/pdflatex on system).")
    print("Tonight's apply runs read from runtime resume.txt and re-tailor per job — fix is live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
