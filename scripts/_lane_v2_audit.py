"""Audit word count, bullet count per role, and page estimate for each resume."""
import re
from pathlib import Path
from docx import Document

BASE = Path(r"C:\Users\bibyd\.applypilot\resume-overhaul")
FILES = [
    "MASTER_RESUME",
    "Resume_WFM_Analyst",
    "Resume_QA_Analyst",
    "Resume_Operations_TechSupport",
    "Resume_Document_QA_Ops",
]


def words(s: str) -> int:
    return len(re.findall(r"\b\w[\w'\-/]*\b", s))


print(f"{'File':<35} {'Words':>6} {'Pages':>6} {'Bullets':>8} {'docxKB':>7} {'pdfKB':>7}")
print("-" * 80)

for f in FILES:
    docx_p = BASE / f"{f}.docx"
    pdf_p = BASE / f"{f}.pdf"
    txt_p = BASE / f"{f}.txt"

    doc = Document(docx_p)
    total_text = "\n".join(p.text for p in doc.paragraphs)
    word_count = words(total_text)

    # Bullets: count list-marked paragraphs
    W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    bullets = 0
    for para in doc.paragraphs:
        pPr = para._p.find(f"{W_NS}pPr")
        if pPr is not None and pPr.find(f"{W_NS}numPr") is not None:
            bullets += 1

    pages = round(word_count / 450, 2)  # ~450 words per page is typical
    docx_kb = round(docx_p.stat().st_size / 1024, 1)
    pdf_kb = round(pdf_p.stat().st_size / 1024, 1)
    print(f"{f:<35} {word_count:>6} {pages:>6.2f} {bullets:>8} {docx_kb:>7} {pdf_kb:>7}")

# Per-role bullet breakdown for MASTER (representative)
print()
print("Per-role bullet count (using MASTER_RESUME.txt for clarity):")
master_txt = (BASE / "MASTER_RESUME.txt").read_text(encoding="utf-8")
# Parse by role headers (lines without leading hyphen and containing ' | ' typically)
sections = re.split(r"\n(?=[A-Z][^\n]*\| )", master_txt)
for s in sections:
    first = s.strip().splitlines()[0] if s.strip() else ""
    if "|" not in first or first.startswith("- "):
        continue
    bullets_in_role = sum(1 for ln in s.splitlines() if ln.strip().startswith("- "))
    print(f"  [{bullets_in_role:>2}] {first[:90]}")
