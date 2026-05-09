"""One-shot: update profile.json target_role to lane-consolidated list (2026-04-28)."""
import json
from pathlib import Path

p = Path(r"C:\Users\bibyd\.applypilot\profile.json")
data = json.loads(p.read_text(encoding="utf-8"))
old = data["experience"]["target_role"]
new = [
    "Licensed Insurance Compliance Specialist",
    "Insurance Compliance & QA Specialist",
    "Insurance Verification Specialist",
    "Claims Processor",
    "Medicare Compliance Auditor",
    "Member Documentation Specialist",
    "QA Review Analyst",
]
data["experience"]["target_role"] = new
p.write_text(json.dumps(data, indent=4), encoding="utf-8")
print("OLD type:", type(old).__name__)
print("OLD:", repr(old))
print("NEW count:", len(new))
print("NEW:", new)
print("OK profile.json updated")
