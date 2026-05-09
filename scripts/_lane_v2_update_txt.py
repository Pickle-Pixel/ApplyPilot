"""Update headline (line 2) + SUMMARY paragraph in all 5 resume .txt files."""
import re
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
    "MASTER_RESUME.txt",
    "Resume_WFM_Analyst.txt",
    "Resume_QA_Analyst.txt",
    "Resume_Operations_TechSupport.txt",
    "Resume_Document_QA_Ops.txt",
]

for fn in FILES:
    p = BASE / fn
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    # Line 0 = "DARRIN BIBY". Line 1 = headline. Replace.
    old_headline = lines[1]
    lines[1] = NEW_HEADLINE
    text2 = "\n".join(lines)
    # Replace SUMMARY block: from "SUMMARY\n" to next blank line.
    pattern = re.compile(r"(SUMMARY\n)(.*?)(\n\n)", re.DOTALL)
    m = pattern.search(text2)
    if not m:
        print(f"FAIL no SUMMARY block: {fn}")
        continue
    text3 = pattern.sub(r"\1" + NEW_SUMMARY + r"\3", text2, count=1)
    # ensure trailing newline
    if not text3.endswith("\n"):
        text3 += "\n"
    p.write_text(text3, encoding="utf-8")
    print(f"OK {fn} | old headline: {old_headline!r}")
print("---DONE---")
