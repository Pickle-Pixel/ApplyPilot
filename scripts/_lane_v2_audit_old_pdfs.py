"""Verify older bak PDFs had 2 pages + jobs section."""
from pathlib import Path
from pypdf import PdfReader

BASE = Path(r"C:\Users\bibyd\.applypilot\resume-overhaul")
samples = [
    "MASTER_RESUME.pdf.bak-2026-04-27-loop-r1",
    "MASTER_RESUME.pdf.bak-2026-04-27-tradeoffs",
    "MASTER_RESUME.pdf.bak-2026-04-28-jh-fix",
    "MASTER_RESUME.pdf",  # current
]
for s in samples:
    p = BASE / s
    if not p.exists():
        continue
    r = PdfReader(str(p))
    pages = len(r.pages)
    text = "".join((pg.extract_text() or "") for pg in r.pages)
    has_jh = "Jack Henry" in text
    has_nco = "NCO Group" in text
    has_aegis = "Aegis" in text
    has_highlights = "PROFESSIONAL HIGHLIGHTS" in text or "Professional Highlights" in text or "Document and QA operations" in text
    print(f"{s}: pages={pages} bytes={p.stat().st_size//1024}KB jh={has_jh} nco={has_nco} aegis={has_aegis} hilites={has_highlights}")
