"""Dump intermediate HTML for MASTER to verify which sections are being rendered."""
import sys
sys.path.insert(0, r"C:\Users\bibyd\Projects\dev\ApplyPilot\src")
from pathlib import Path
from applypilot.scoring.pdf import parse_resume, build_html

txt = Path(r"C:\Users\bibyd\.applypilot\resume-overhaul\MASTER_RESUME.txt").read_text(encoding="utf-8")
resume = parse_resume(txt)
print("Section keys:", list(resume["sections"].keys()))
html = build_html(resume)
out = Path(r"C:\Users\bibyd\.applypilot\resume-overhaul\_diagnostic_master.html")
out.write_text(html, encoding="utf-8")
print(f"HTML chars: {len(html)}")
print("Contains 'Jack Henry':", "Jack Henry" in html)
print("Contains 'Professional Experience':", "Professional Experience" in html or "Experience</div>" in html)
print("Contains 'PROFESSIONAL EXPERIENCE' section:", "PROFESSIONAL EXPERIENCE" in resume["sections"])
print("Contains 'EXPERIENCE' section (legacy expected):", "EXPERIENCE" in resume["sections"])
print(f"Output: {out}")
