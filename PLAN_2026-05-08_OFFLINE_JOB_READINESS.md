# ApplyPilot Offline Job Readiness Audit (2026-05-09 Endgame Reconciliation)

## 2026-05-08 Privacy/Safety Execution Update

- `applypilot apply` is now manual-safe by default and resolves to dry-run unless the operator passes `--submit --confirm-submit APPLY`.
- Default apply worker MCP config is Playwright-only; Gmail/mail tooling is not available to the apply agent.
- Claude apply workers no longer request blanket `bypassPermissions`.
- Apply prompts no longer include stored passwords, raw CAPTCHA keys, mailbox-read instructions, or email-send actions.
- `reset_failed()` only reopens `failed` and `failed:*`; manual, avoid, manual_review, captcha, expired, and dry_run_ready buckets remain protected.
- `status --ready` subtracts rows whose resume/cover sibling PDFs are missing or too small and reports them as `Invalid artifacts`.
- `pack --lane primary --limit 1` is a local-mutating export smoke, not a read-only check.

## Final Live Baseline (authoritative for handoff)

- `python -m applypilot.cli status`: 6,429 total jobs, 6,132 with full description, 0 strict-ready primary jobs, 11 applied, 31 apply errors.
- `python -m applypilot.cli status --ready --lane primary`: manual 5 / avoid 1 / manual_review 1 / failed 10 / in_progress 0 / stale 0.
- `python -m applypilot.cli pack --lane primary --limit 1`: 0 exported, 1 skipped (`already failed: expired`).
- Upstream and repo posture check: `origin` fetch from `https://github.com/Pickle-Pixel/ApplyPilot.git`, push remains disabled locally.
- Full-file sweep attached via `applier_insight_audit_scan.json` (235-file fast scan) and `APPLYPILOT_FILE_AUDIT_MANIFEST_2026-05-09.json` (244-file authoritative manifest after excluding Git internals, virtualenvs, and bytecode caches).

## Historical sections are archive only

The sections below are preserved only as historical context. Any conflict should defer to the live baseline above.
## 2026-05-09 Live Baseline (current, replaces older counts for queue handoff)

- Live checks under manual/offline posture:
  - `python -m applypilot.cli status`: 6,429 total jobs, 6,132 with full description, 0 strict-ready primary jobs, 11 applied, 31 apply errors.
  - `python -m applypilot.cli status --ready --lane primary`: Manual 5 / Avoid 1 / Manual Review 1 / Failed 10 / In progress 0 / Stale 0.
  - `python -m applypilot.cli pack --lane primary --limit 1`: 0 exported, 1 skipped (`already failed: expired`).
  - Targeted tests: `test_apply_result_parser`, `test_application_packs`, `test_database_stats`, `test_apply_lane_filter` pass.
- Any older date-specific "Current Verified State" values in this file are historical unless they exactly match this lane baseline.

## Current Verified State

| Area | Checked result |
| --- | --- |
| Repo | `C:\Users\bibyd\Projects\dev\ApplyPilot` exists on branch `codex/applypilot-job-readiness`. |
| Working tree | Dirty before this audit. Existing modified/untracked code and docs were left untouched except this audit doc and the stale status-note fix. |
| Git remote | `origin` fetches from `https://github.com/Pickle-Pixel/ApplyPilot.git`. |
| Public push posture | `origin` push URL is `DISABLED-PUSH-TO-KEEP-LOCAL`; accidental normal pushes should fail. |
| Upstream fetch check | `git ls-remote origin HEAD` resolved `4a8d521f67f5139811c0a910ef37410f8e6d836a`. |
| Repo venv | `.venv\Scripts\python.exe` exists and reports Python 3.12.8. |
| ApplyPilot doctor | Profile, resume, searches, LLM, Claude CLI, Chromium, Node all OK. CapSolver remains optional/unset. |
| Automation capability | CLI reports `Tier 3 - Full Auto-Apply`, so auto-apply is technically available if someone runs it. Keep using manual/dry-run commands only. |
| Scheduled tasks | `ApplyPilot Daily Pipeline` and `ApplyPilot Indeed Scan MWF` are present but Disabled. |
| Pipeline status | 6,429 total jobs, 4,770 scored, 71 tailored resumes, 58 cover letters, 0 strict-ready primary-lane jobs, 11 applied, 31 apply errors. |
| Runtime DB | Active runtime DB is `C:\Users\bibyd\.applypilot\applypilot.db` at about 64 MB. Repo-root `applypilot.db` is a 12 KB orphan/stub. |
| Profile data | Active profile is `C:\Users\bibyd\.applypilot\profile.json`. It contains personal, work authorization, availability, compensation, experience, skills boundary, resume facts, and EEO sections. |
| Resume source | Canonical resume set is `C:\Users\bibyd\.applypilot\resume-overhaul`; runtime resume text is `C:\Users\bibyd\.applypilot\resume.txt`. |
| Current resumes | Quick-access current resumes are also mirrored under `C:\Users\bibyd\.applypilot\00_CURRENT_RESUMES`. |
| Application packs | Recent active pack: `C:\Users\bibyd\.applypilot\application_packs\micro1_Insurance_Specialist_2026-04-28`. Older manual/ready/submitted packs remain under `C:\Users\bibyd\.applypilot\application_packs`. |
| Coordination docs | `C:\Users\bibyd\Projects\20_INCOME\PLAN_2026-05-08_INCOME_SURVIVAL.md` and `C:\Users\bibyd\Projects\30_LIFE_ADMIN\PLAN_00_LIFE_OPS_COORDINATION.md` both correctly say manual/draft-only and no public pushes. |

## Ready

- Local repo can fetch upstream patches while staying push-disabled.
- Runtime profile, resume, searches, browser, local LLM endpoint, Claude CLI, and Node checks pass.
- Scheduled ApplyPilot tasks are disabled, which matches the manual/offline posture.
- Strict-ready primary-lane jobs are currently 0 under the false-ready hardening gate (status null/empty + valid tailor + cover PDFs).
- Resume artifacts exist as DOCX/PDF/TXT variants in the canonical resume folder and quick-access current-resumes folder.

## Risky

- The CLI can still run full auto-apply because the environment is Tier 3 capable. Do not run `applypilot apply` without `--dry-run` unless Darrin explicitly approves real submissions.
- The repo worktree was already dirty, with many existing modified and untracked files. Treat all unknown changes as active work from another agent/person.
- Older docs still contain historical references to April queues and prior automation states. This audit supersedes those when there is a conflict.
- `applypilot status` reports 19 apply errors and 5 applied records. Review before trusting the next target list.
- Some application pack folders are older or historical; use timestamps and handoff files before assuming a pack is today's priority.

## Next Manual-Safe Steps

1. Open `C:\Users\bibyd\.applypilot\application_packs\micro1_Insurance_Specialist_2026-04-28\HANDOFF_FOR_DARRIN.md` and decide whether that pack is still worth submitting manually.
2. Review `C:\Users\bibyd\.applypilot\application_packs\ready_primary_2026-04-24\00_TODAY_SUBMIT_QUEUE.md` as the older queue, then mark stale, submitted, skipped, or still-active items.
3. Run only read-only checks before work: `.\.venv\Scripts\applypilot.exe doctor` and `.\.venv\Scripts\applypilot.exe status`.
4. If refreshing packets, use pack/export generation only; do not run real submission. Keep any browser run to `applypilot apply --dry-run` unless explicitly approved.
5. Before touching code, inspect the dirty working tree and identify which changes belong to which agent/person.
6. Keep income/life-admin coordination docs as approval gates: local drafts are allowed; external actions require Darrin approval.

## Commands Verified

```powershell
git status --short
git branch --show-current
git remote -v
git config --get-regexp '^remote\..*\.(url|pushurl)$'
git ls-remote origin HEAD
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m applypilot.cli doctor
.\.venv\Scripts\python.exe -m applypilot.cli status
Get-ScheduledTask | Where-Object { $_.TaskName -like '*ApplyPilot*' }
```

