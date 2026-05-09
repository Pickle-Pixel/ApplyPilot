"""Insurance-license priority boost (Darrin dispatch 2026-04-28).

Boosts fit_score by +3 (capped at 10) for jobs with both:
  (a) an insurance-license signal in the title or full_description, and
  (b) a non-phone / chat / email / back-office / digital signal.

Also boosts roles at any of Darrin's target insurance companies
(Oscar, Lemonade, Root, Hippo, Devoted Health, UnitedHealth,
Humana, Aetna, Centene, Bright Health, GoHealth, Datavant, Epiq, Thatch)
even without an explicit license signal, because their digital/chat
divisions are the strategic targets per the dispatch.

Idempotent: re-runs reapply the SAME boost based on a marker column
('boost_license_v1'). To rebuild the boost from scratch, drop and
re-add the marker:

    ALTER TABLE jobs DROP COLUMN boost_license_v1;

Run from the ApplyPilot project root:
    .\.venv\Scripts\python.exe scripts\boost_license_priority.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

DB = Path.home() / ".applypilot" / "applypilot.db"

LICENSE_SIGNAL_TITLE = (
    "license", "licensed",
)
LICENSE_SIGNAL_DESC = (
    "insurance license", "active license", "state license",
    "resident license", "health license", "life and health license",
)

NON_PHONE_SIGNAL_TITLE = (
    "chat", "email", "back office", "back-office", "non-phone",
    "non phone", "non-voice", "digital", "verification",
    "underwriting", "processor", "coordinator", "advocate",
    "trainer", "benefit", "correspondence", "back office specialist",
    "messaging",
)

TARGET_COMPANIES = (
    "oscar", "lemonade", "root", "hippo",
    "devoted health", "devoted-health", "devotedhealth",
    "unitedhealth", "uhc", "humana", "aetna", "cvs aetna",
    "centene", "ambetter", "bright health", "gohealth",
    "datavant", "epiq", "thatch",
)


def _build_signal_clause(col: str, needles: tuple[str, ...]) -> tuple[str, list]:
    parts = [f"LOWER({col}) LIKE ?" for _ in needles]
    params = [f"%{n}%" for n in needles]
    return "(" + " OR ".join(parts) + ")", params


def main() -> int:
    if not DB.exists():
        print(f"DB not found at {DB}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(DB))
    cur = conn.cursor()

    # Add marker column if absent (so we can audit which rows got boosted)
    cols = {r[1] for r in cur.execute("PRAGMA table_info(jobs)")}
    if "boost_license_v1" not in cols:
        cur.execute("ALTER TABLE jobs ADD COLUMN boost_license_v1 INTEGER DEFAULT 0")
        conn.commit()

    # Build the WHERE clause
    license_title_clause, license_title_params = _build_signal_clause("title", LICENSE_SIGNAL_TITLE)
    license_desc_clause, license_desc_params = _build_signal_clause("full_description", LICENSE_SIGNAL_DESC)
    non_phone_title_clause, non_phone_title_params = _build_signal_clause("title", NON_PHONE_SIGNAL_TITLE)
    non_phone_desc_clause, non_phone_desc_params = _build_signal_clause("full_description", NON_PHONE_SIGNAL_TITLE)
    target_co_clause, target_co_params = _build_signal_clause("COALESCE(company, '') || ' ' || COALESCE(site, '')", TARGET_COMPANIES)

    # Combined: (license signal title OR license signal desc) AND (non-phone title OR non-phone desc)
    license_clause = f"({license_title_clause} OR {license_desc_clause})"
    license_params = license_title_params + license_desc_params

    non_phone_clause = f"({non_phone_title_clause} OR {non_phone_desc_clause})"
    non_phone_params = non_phone_title_params + non_phone_desc_params

    # Set 1: license + non-phone signal
    set1_where = f"applied_at IS NULL AND COALESCE(boost_license_v1, 0) = 0 AND {license_clause} AND {non_phone_clause}"
    set1_params = license_params + non_phone_params

    matches_1 = cur.execute(
        f"SELECT url, title, COALESCE(company,site), fit_score FROM jobs WHERE {set1_where}",
        set1_params,
    ).fetchall()

    # Set 2: target company (regardless of license)
    set2_where = f"applied_at IS NULL AND COALESCE(boost_license_v1, 0) = 0 AND {target_co_clause}"
    set2_params = target_co_params

    matches_2 = cur.execute(
        f"SELECT url, title, COALESCE(company,site), fit_score FROM jobs WHERE {set2_where}",
        set2_params,
    ).fetchall()

    # Combine (de-dup by url)
    seen = set()
    combined = []
    for src, rows in (("license+non-phone", matches_1), ("target-company", matches_2)):
        for r in rows:
            if r[0] in seen:
                continue
            seen.add(r[0])
            combined.append((src, *r))

    print(f"Matched {len(matches_1)} license+non-phone + {len(matches_2)} target-company "
          f"= {len(combined)} unique rows to boost (+3, capped at 10).")

    # Apply boost
    for src, url, title, co, fit in combined:
        new_fit = min(10, (fit or 0) + 3)
        cur.execute(
            "UPDATE jobs SET fit_score = ?, boost_license_v1 = 1 WHERE url = ?",
            (new_fit, url),
        )
        if len(combined) <= 60:
            print(f"  [{src:>16}] fit {fit or 0}->{new_fit:>2} | {(title or '')[:60]:<60} | {(co or '')[:25]}")

    conn.commit()

    # Surface top 20 after boost
    print()
    print("=== TOP 20 NON-APPLIED, fit_score DESC, boosted-or-not (post-boost) ===")
    for r in cur.execute("""
        SELECT title, COALESCE(company,site), fit_score, location, boost_license_v1, url
        FROM jobs
        WHERE applied_at IS NULL AND full_description IS NOT NULL
        ORDER BY boost_license_v1 DESC, fit_score DESC, scored_at DESC
        LIMIT 20
    """):
        marker = "B" if r[4] else " "
        print(f"  {marker} | fit={r[2]:>2} | {(r[0] or '')[:55]:<55} | {(r[1] or '?')[:22]:<22} | {(r[3] or '?')[:25]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
