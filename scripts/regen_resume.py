"""Regenerate resume.txt from MASTER_RESUME.docx.

ApplyPilot reads `C:\\Users\\bibyd\\.applypilot\\resume.txt` on every tailoring
run. The canonical master/fact bank lives at
`C:\\Users\\bibyd\\.applypilot\\resume-overhaul\\MASTER_RESUME.docx` (canonical
as of 2026-04-27 per .applypilot\\00_START_HERE\\CANONICAL.md). The previous default was
`.applypilot\\job_application_ready_2026-04-24\\04_fact_bank_do_not_submit\\MASTER_RESUME.docx`;
that path is retained as LEGACY_SOURCE for one release.

This script converts the .docx to the flat text form ApplyPilot expects,
preserving bullet structure and heading separation.

Usage (run from anywhere; defaults point to the canonical locations):

    python scripts/regen_resume.py
    python scripts/regen_resume.py --source path\\to\\MASTER.docx --output path\\to\\resume.txt
    python scripts/regen_resume.py --dry-run    # show what would change without writing

Exit codes:
    0   success
    2   source docx not found
    3   parse error
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docx import Document

DEFAULT_SOURCE = Path(
    r"C:\Users\bibyd\.applypilot\resume-overhaul\MASTER_RESUME.docx"
)
# Legacy path retained for one release in case a stale workflow still calls it;
# resume-overhaul/ is canonical as of 2026-04-27
# (see .applypilot\00_START_HERE\CANONICAL.md).
LEGACY_SOURCE = Path(
    r"C:\Users\bibyd\.applypilot\job_application_ready_2026-04-24\04_fact_bank_do_not_submit\MASTER_RESUME.docx"
)
DEFAULT_OUTPUT = Path(r"C:\Users\bibyd\.applypilot\resume.txt")

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _is_bullet(para) -> bool:
    """Detect list items by probing the paragraph's numPr XML node.

    python-docx does not expose list-style directly, so we inspect the
    underlying XML like the original one-shot converter did.
    """
    pPr = para._p.find(f"{_W_NS}pPr")
    if pPr is None:
        return False
    return pPr.find(f"{_W_NS}numPr") is not None


def convert(source: Path, output: Path, *, dry_run: bool = False) -> int:
    if not source.exists():
        # Canonical master missing. If resume.txt already exists, treat this as
        # a no-op instead of hard-failing the nightly pipeline — the resume-overhaul
        # skill may have split the master into role-specific variants without
        # restoring MASTER_RESUME.docx yet. Surface a clear warning either way.
        if output.exists():
            print(
                f"WARNING: master docx not found: {source}. "
                f"Leaving existing resume.txt at {output} unchanged. "
                f"(Restore the fact-bank MASTER_RESUME.docx to re-enable regen.)",
                file=sys.stderr,
            )
            return 0
        print(f"ERROR: source docx not found and no existing resume.txt: {source}", file=sys.stderr)
        return 2

    try:
        doc = Document(source)
    except Exception as exc:  # python-docx raises a variety; treat all as parse errors
        print(f"ERROR: could not parse {source}: {exc}", file=sys.stderr)
        return 3

    lines: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            lines.append("")
            continue
        style = ((para.style.name if para.style else "") or "").lower()
        if _is_bullet(para):
            lines.append(f"- {text}")
        elif "heading" in style:
            lines.append("")
            lines.append(text)
            lines.append("=" * min(len(text), 60))
        else:
            lines.append(text)

    # Collapse runs of 2+ blank lines down to 1.
    collapsed: list[str] = []
    blank_run = 0
    for ln in lines:
        if ln == "":
            blank_run += 1
            if blank_run <= 1:
                collapsed.append(ln)
        else:
            blank_run = 0
            collapsed.append(ln)

    payload = "\n".join(collapsed).rstrip() + "\n"

    if dry_run:
        existing = output.read_text(encoding="utf-8") if output.exists() else ""
        if existing == payload:
            print(f"NO-CHANGE {output} ({len(payload.encode('utf-8'))} bytes)")
        else:
            print(f"WOULD-WRITE {output} ({len(payload.encode('utf-8'))} bytes, {len(collapsed)} lines)")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(f"WROTE {output} ({output.stat().st_size} bytes, {len(collapsed)} lines)")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="regen_resume",
        description="Regenerate .applypilot/resume.txt from the do-not-submit MASTER_RESUME.docx fact bank.",
    )
    p.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Path to master .docx (default: {DEFAULT_SOURCE})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to resume.txt (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing to disk.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return convert(args.source, args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
