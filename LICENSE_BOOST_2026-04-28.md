# Insurance-License Priority Boost — 2026-04-28

> Per Darrin dispatch 2026-04-28: insurance-licensed + non-phone (chat / email / written / back-office / digital) roles boosted +3 in `fit_score`. Target insurance companies (Oscar, Lemonade, Root, Hippo, Devoted Health, UnitedHealth, Humana, Aetna, Centene, Bright Health, GoHealth, Datavant, Epiq, Thatch) get +3 on any role at the company even without an explicit license signal.

## Implementation

- **Script:** `scripts/boost_license_priority.py` in the ApplyPilot project root.
- **DB column added:** `boost_license_v1` (INTEGER, default 0). Marker prevents re-boosting the same row on subsequent runs.
- **Boost rule:** `fit_score = MIN(10, fit_score + 3)` on rows where (license signal in title OR JD) AND (non-phone signal in title OR JD), OR target-company match.
- **Idempotent:** safe to re-run after every `applypilot run score`. Only un-boosted rows get touched.
- **Re-boost from scratch:** `ALTER TABLE jobs DROP COLUMN boost_license_v1` then re-run the script.

## Run output

- 37 rows matched on license + non-phone signal
- 11 rows matched on target-company signal
- 48 unique rows boosted (de-duped by URL)

Most matches landed below the apply threshold (fit < 7) and won't enter the apply queue. The high-impact hits are:

| Boosted to fit | Title | Company | Notes |
|---|---|---|---|
| **10** | Health Information Specialist I | Datavant | Remote US, Document_QA_Ops lane, target company |
| **10** | Vendor Support Specialist | Datavant | Remote US, NON-VOICE, target company |
| **10** | Live Messaging Agent (JR109855) | GoHealth Urgent Care | Remote US, NON-VOICE, target company, chat-primary |
| **8** | Credentialing Coordinator | UT MD Anderson | Remote, license-adjacent |

These are now the top 4 boosted rows at fit>=7. The broader queue still has 193 jobs at fit>=7; the boosted set surfaces ahead of them via the `boost_license_v1 DESC, fit_score DESC` ordering.

## Caveats

1. **No native ingest-time filter yet.** The boost is a one-shot SQL update. To make it persistent, run the script after every `applypilot run score`. A proper integration would call this from the scoring stage finalizer. Logged as a follow-up.
2. **Target-company list is hand-curated.** Add to `scripts/boost_license_priority.py` `TARGET_COMPANIES` tuple as new targets come up.
3. **Some noise hits.** A handful of nursing / banking / real-estate roles matched the license + non-phone signal heuristically (RN with "license required" + "coordinator" in title). They got boosted to fit 4-5, which keeps them OUT of the apply queue (threshold 7). Acceptable noise level.
4. **Fit=10 is the cap.** A row at fit 7+ gets capped at 10 after +3. The ordering still uses `boost_license_v1 DESC` first so boosted rows stay above non-boosted fit=10 rows.

## Recommended next step

Operator review of the 4 boosted-and-eligible rows, then the same 5-app cover-letter dry-run pattern for the boosted set. Once approved, the overnight `applypilot apply` run will pull those first by virtue of the ordering.

## Files

- `scripts/boost_license_priority.py` — runner (versioned in ApplyPilot project)
- `~/.applypilot/dry_run_2026-04-28/BOOSTED_QUEUE.txt` — point-in-time snapshot of post-boost top 30
- `~/.applypilot/dry_run_2026-04-28/boost_license.log` — full run log
