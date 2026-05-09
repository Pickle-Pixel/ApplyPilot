"""Surgically replace headline (para 1) + summary (para after SUMMARY heading) in all 5 docx files.

Preserves first-run formatting. Removes all other runs in target paragraphs.
"""
from docx import Document
from pathlib import Path

NEW_HEADLINE = "Licensed Insurance Compliance & QA Specialist | Medicare Member Documentation | Remote Non-Phone"

NEW_SUMMARY = (
    "Licensed Insurance Compliance & QA Specialist with 10+ years across regulated insurance documentation, "
    "contact-center QA, and back-office operations. Active Health Insurance Agent license and recent Medicare "
    "AEP/OEP enrollment-documentation experience at GoHealth, ResultsCX, and TTEC across multi-state CMS "
    "Marketplace workflows. Prior QA Analyst work at Teleperformance auditing agent calls and documentation "
    "for HIPAA privacy expectations and carrier compliance across UHC, Humana, Aetna, BCBS, and Anthem programs. "
    "Targeting remote, non-phone roles in insurance compliance, member documentation, claims processing, and "
    "Medicare/healthcare QA review."
)

BASE = Path(r"C:\Users\bibyd\.applypilot\resume-overhaul")
FILES = [
    "MASTER_RESUME.docx",
    "Resume_WFM_Analyst.docx",
    "Resume_QA_Analyst.docx",
    "Resume_Operations_TechSupport.docx",
    "Resume_Document_QA_Ops.docx",
]


def replace_paragraph_text(para, new_text: str) -> None:
    """Replace paragraph text, preserving first run's formatting."""
    runs = list(para.runs)
    if not runs:
        para.add_run(new_text)
        return
    first = runs[0]
    first.text = new_text
    # Remove other runs
    for r in runs[1:]:
        r._element.getparent().remove(r._element)


for fn in FILES:
    p = BASE / fn
    doc = Document(p)
    paragraphs = doc.paragraphs

    # Headline = paragraph[1] (after DARRIN BIBY heading)
    old_headline = paragraphs[1].text
    replace_paragraph_text(paragraphs[1], NEW_HEADLINE)

    # Summary = paragraph immediately following the SUMMARY heading
    summary_idx = None
    for i, para in enumerate(paragraphs):
        if para.text.strip().upper() == "SUMMARY" and "heading" in (para.style.name or "").lower():
            summary_idx = i + 1
            break
    if summary_idx is None:
        print(f"FAIL no SUMMARY heading: {fn}")
        continue
    old_summary = paragraphs[summary_idx].text[:80]
    replace_paragraph_text(paragraphs[summary_idx], NEW_SUMMARY)

    doc.save(p)
    print(f"OK {fn}")
    print(f"   old headline: {old_headline!r}")
    print(f"   old summary:  {old_summary!r}...")
print("---DONE---")
