"""Measure actual PDF page counts (current vs. backup), and word counts."""
from pathlib import Path
import re

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader  # type: ignore

BASE = Path(r"C:\Users\bibyd\.applypilot\resume-overhaul")
FILES = [
    "MASTER_RESUME",
    "Resume_WFM_Analyst",
    "Resume_QA_Analyst",
    "Resume_Operations_TechSupport",
    "Resume_Document_QA_Ops",
]

print(f"{'File':<35} {'CurPages':>9} {'CurWords':>9} {'OldPages':>9} {'OldWords':>9} {'CurKB':>6} {'OldKB':>6}")
print("-" * 100)
for f in FILES:
    cur = BASE / f"{f}.pdf"
    old = BASE / f"{f}.pdf.bak-2026-04-28-jh-fix"
    rows = []
    for p in (cur, old):
        if not p.exists():
            rows.append(("?", 0, 0))
            continue
        r = PdfReader(str(p))
        text = ""
        for page in r.pages:
            text += page.extract_text() or ""
        wc = len(re.findall(r"\b\w[\w'\-/]*\b", text))
        rows.append((len(r.pages), wc, p.stat().st_size // 1024))
    (cp, cw, ck), (op, ow, ok) = rows
    print(f"{f:<35} {cp:>9} {cw:>9} {op:>9} {ow:>9} {ck:>6} {ok:>6}")
