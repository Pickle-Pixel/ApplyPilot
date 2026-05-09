"""Rebuild the 5 resume-overhaul PDFs from their .txt sources via ApplyPilot's
Playwright-based converter. Backups already created by fix_jh_built_bullet.py
for the .pdf files; this script writes fresh PDFs in place."""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\bibyd\Projects\dev\ApplyPilot\src")
from applypilot.scoring.pdf import convert_to_pdf  # noqa: E402

OVERHAUL = Path.home() / ".applypilot" / "resume-overhaul"
FILES = [
    "MASTER_RESUME",
    "Resume_QA_Analyst",
    "Resume_WFM_Analyst",
    "Resume_Operations_TechSupport",
    "Resume_Document_QA_Ops",
]


def main() -> int:
    for f in FILES:
        txt = OVERHAUL / f"{f}.txt"
        pdf = OVERHAUL / f"{f}.pdf"
        if pdf.exists():
            bak = pdf.with_suffix(".pdf.bak-2026-04-28-jh-fix")
            if not bak.exists():
                shutil.copy2(pdf, bak)
        if not txt.exists():
            print(f"  MISSING: {txt}")
            continue
        t0 = time.time()
        try:
            convert_to_pdf(txt, pdf)
            print(f"  OK: {pdf.name} ({pdf.stat().st_size} bytes, {time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  FAIL {pdf.name}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
