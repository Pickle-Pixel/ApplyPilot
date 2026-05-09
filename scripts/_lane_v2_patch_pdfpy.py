"""Patch pdf.py to fix the section-matching regression.

Bug: build_html checks `"EXPERIENCE" in sections` but real key is "PROFESSIONAL EXPERIENCE",
and there is no handler for "PROFESSIONAL HIGHLIGHTS". Result: PDFs drop ~75% of content.

Fix: introduce _find_section helper that does substring-key matching, add HIGHLIGHTS rendering,
and route EXPERIENCE / EDUCATION / SUMMARY / TECHNICAL SKILLS through the helper.
"""
from pathlib import Path

p = Path(r"C:\Users\bibyd\Projects\dev\ApplyPilot\src\applypilot\scoring\pdf.py")
src = p.read_text(encoding="utf-8")

# Sanity: ensure we're patching the same source we expect (the regression form).
assert 'if "EXPERIENCE" in sections:' in src, "old EXPERIENCE check not found"
assert 'if "TECHNICAL SKILLS" in sections:' in src, "old TECHNICAL SKILLS check not found"
assert "_find_section" not in src, "patch already applied"

# 1. Inject the helper just before `def build_html`.
helper = '''def _find_section(sections: dict, *needles: str) -> str | None:
    """Return the first section value whose key contains any of the needles.

    Resume txt files use heading variants ("PROFESSIONAL EXPERIENCE",
    "WORK EXPERIENCE", "EXPERIENCE"); this helper makes the renderer
    forgiving instead of exact-key matching.
    """
    for needle in needles:
        for key, val in sections.items():
            if needle in key:
                return val
    return None


'''
marker = "# ── HTML Template ────────────────────────────────────────────────────────\n\ndef build_html"
assert marker in src
src = src.replace(marker, "# ── HTML Template ────────────────────────────────────────────────────────\n\n" + helper + "def build_html", 1)

# 2. Rewrite SUMMARY block to use _find_section
old_summary_block = '''    # Summary
    summary_html = ""
    if "SUMMARY" in sections:
        summary_html = f'<div class="section"><div class="section-title">Summary</div><div class="summary">{sections["SUMMARY"].strip()}</div></div>'
'''
new_summary_block = '''    # Summary
    summary_html = ""
    summary_text = _find_section(sections, "SUMMARY")
    if summary_text:
        summary_html = f'<div class="section"><div class="section-title">Summary</div><div class="summary">{summary_text.strip()}</div></div>'

    # Professional Highlights (rendered as a bullet list)
    highlights_html = ""
    highlights_text = _find_section(sections, "HIGHLIGHTS")
    if highlights_text:
        bullets = []
        for line in highlights_text.strip().splitlines():
            stripped = line.strip()
            if stripped.startswith("- ") or stripped.startswith("\\u2022 "):
                bullets.append(stripped[2:].strip())
            elif stripped:
                bullets.append(stripped)
        if bullets:
            items = "".join(f"<li>{b}</li>" for b in bullets)
            highlights_html = f\'<div class="section"><div class="section-title">Professional Highlights</div><ul>{items}</ul></div>\'
'''
assert old_summary_block in src, "summary block not found verbatim"
src = src.replace(old_summary_block, new_summary_block, 1)

# 3. Rewrite TECHNICAL SKILLS block to use _find_section
old_skills_block = '''    # Skills
    skills_html = ""
    if "TECHNICAL SKILLS" in sections:
        skills = parse_skills(sections["TECHNICAL SKILLS"])
'''
new_skills_block = '''    # Skills
    skills_html = ""
    skills_text = _find_section(sections, "TECHNICAL SKILLS", "SKILLS")
    if skills_text:
        skills = parse_skills(skills_text)
'''
assert old_skills_block in src, "skills block not found verbatim"
src = src.replace(old_skills_block, new_skills_block, 1)

# 4. Rewrite EXPERIENCE block to use _find_section
old_exp_block = '''    # Experience
    exp_html = ""
    if "EXPERIENCE" in sections:
        entries = parse_entries(sections["EXPERIENCE"])
'''
new_exp_block = '''    # Experience
    exp_html = ""
    exp_text = _find_section(sections, "EXPERIENCE")
    if exp_text:
        entries = parse_entries(exp_text)
'''
assert old_exp_block in src, "exp block not found verbatim"
src = src.replace(old_exp_block, new_exp_block, 1)

# 5. Rewrite PROJECTS block
old_proj_block = '''    # Projects
    proj_html = ""
    if "PROJECTS" in sections:
        entries = parse_entries(sections["PROJECTS"])
'''
new_proj_block = '''    # Projects
    proj_html = ""
    proj_text = _find_section(sections, "PROJECTS")
    if proj_text:
        entries = parse_entries(proj_text)
'''
assert old_proj_block in src
src = src.replace(old_proj_block, new_proj_block, 1)

# 6. Rewrite EDUCATION block
old_edu_block = '''    # Education
    edu_html = ""
    if "EDUCATION" in sections:
        edu_text = sections["EDUCATION"].strip()
'''
new_edu_block = '''    # Education
    edu_html = ""
    edu_text_raw = _find_section(sections, "EDUCATION")
    if edu_text_raw:
        edu_text = edu_text_raw.strip()
'''
assert old_edu_block in src
src = src.replace(old_edu_block, new_edu_block, 1)

# 7. Insert highlights_html into the f-string body (after summary, before skills).
old_body = "{summary_html}\n{skills_html}\n{exp_html}\n{proj_html}\n{edu_html}\n"
new_body = "{summary_html}\n{highlights_html}\n{skills_html}\n{exp_html}\n{proj_html}\n{edu_html}\n"
assert old_body in src, "f-string body not found"
src = src.replace(old_body, new_body, 1)

p.write_text(src, encoding="utf-8")
print("OK pdf.py patched")
print("  - _find_section helper added")
print("  - SUMMARY/SKILLS/EXPERIENCE/PROJECTS/EDUCATION now use forgiving lookup")
print("  - PROFESSIONAL HIGHLIGHTS section now rendered as bullet list")
