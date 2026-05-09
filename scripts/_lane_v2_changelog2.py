"""Insert depth-fix dispatch entry at top of CHANGELOG.md."""
from pathlib import Path

p = Path(r"C:\Users\bibyd\Projects\dev\ApplyPilot\CHANGELOG.md")
text = p.read_text(encoding="utf-8")

new_entry = (
    "2026-04-28 17:26 | dispatch:lane-consolidation-v2-depth-fix | "
    "Projects\\dev\\ApplyPilot\\LANE_CONSOLIDATION_2026-04-28.md | "
    "Resume PDFs were rendering tiny (1 page / ~190 extracted words) due to a regression in "
    "src/applypilot/scoring/pdf.py: build_html keyed sections on exact 'EXPERIENCE' but the txt "
    "uses 'PROFESSIONAL EXPERIENCE'; 'PROFESSIONAL HIGHLIGHTS' had no handler at all. Added "
    "_find_section() forgiving lookup + HIGHLIGHTS bullet renderer. Regenerated 5 PDFs (now "
    "2 pages / 758-783 words / 79 KB each, matching pre-regression behavior). Backups: "
    ".bak-2026-04-28-depth on pdf.py + 5 PDFs.\n\n"
)

marker = "2026-04-28 17:10 | dispatch:lane-consolidation-v2"
assert marker in text, "marker not found"
text2 = text.replace(marker, new_entry + marker, 1)
assert text2 != text, "no replacement made"
p.write_text(text2, encoding="utf-8")
print("OK CHANGELOG.md prepended with depth-fix entry")
