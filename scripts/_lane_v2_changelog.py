"""Insert lane-consolidation-v2 dispatch entry at top of CHANGELOG.md."""
from pathlib import Path

p = Path(r"C:\Users\bibyd\Projects\dev\ApplyPilot\CHANGELOG.md")
text = p.read_text(encoding="utf-8")

new_entry = (
    "2026-04-28 17:10 | dispatch:lane-consolidation-v2 | "
    "Projects\\dev\\ApplyPilot\\LANE_CONSOLIDATION_2026-04-28.md | "
    "Council top correction #2: collapsed 4-lane positioning to "
    "\"Licensed Insurance Compliance & QA Specialist\". Master + 4 variants + "
    "profile.json target_role updated.\n\n"
)

marker = "2026-04-28 15:43 | dispatch:council-review"
assert marker in text, "marker not found"
text2 = text.replace(marker, new_entry + marker, 1)
assert text2 != text, "no replacement made"
p.write_text(text2, encoding="utf-8")
print("OK CHANGELOG.md prepended with dispatch:lane-consolidation-v2")
