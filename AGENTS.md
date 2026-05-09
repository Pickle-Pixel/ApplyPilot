# ApplyPilot Repo Operating Instructions

Scope: `C:\Users\bibyd\Projects\dev\ApplyPilot`

## 2026-05-09 Full Repo Insight Sweep (Authoritative Addendum)

- Repository-wide surface scan completed across all reachable files (tracked + untracked): quick scan covered `235` files, and the authoritative per-file manifest covered `244` source/operator files after excluding Git internals, virtualenvs, bytecode caches, and the manifest itself.
- Authoritative per-file manifest: `APPLYPILOT_FILE_AUDIT_MANIFEST_2026-05-09.json` records path, size, extension, SHA-256, line count where text-readable, and marker counts without dumping private contents.
- No unresolved `TODO|FIXME|XXX|HACK` markers remain in executable/source surfaces (`src`, `tests`, `scripts`), while review-relevant markers in docs/logs were inventoried as audit noise and retained.
- Marker scan output (`applier_insight_audit_scan.json`) recorded `47` files with high-risk keyword hits and `3` files with >4,000-char lines; these remain non-blocking after manual inspection.
- Full scan artifact and next controls:
  - `applier_insight_audit_scan.json` (raw counts)
  - `APPLYPILOT_FILE_AUDIT_MANIFEST_2026-05-09.json` (per-file manifest)
  - `APPLYPILOT_FULL_INSIGHT_AUDIT_2026-05-09.md` (decision trace)
  - `ENDGAME_EXECUTION_LOG_2026-05-09.md` (append new blocker deltas before every handoff)
- New action settings to study immediately:
  - enforce consistent blocker-count baseline in queue dashboards by lane: `manual`, `avoid`, `manual_review`, `failed`, `stale`
  - keep stale `in_progress` recovery reason as explicit `manual_review` with reason `stale_in_progress_orphaned`
  - keep a stable `CHROME_PATH` and `PYTHONIOENCODING=utf-8` fallback for reproducible runtime checks
- If any new branch includes `--ready`/`pack` deltas that increase blocked buckets, pause handoff and run parser/launcher targeted tests before proceeding.

## Golden rules (non-negotiable)

- Keep runtime and execution posture manual/offline unless an explicit approval request is approved in chat.
- Never run or script real application submissions from this repo without explicit user approval. Real submission now requires `applypilot apply --submit --confirm-submit APPLY`; otherwise the apply command defaults to dry-run behavior.
- Never push to `origin` from this branch unless the user explicitly requests it.
- Before editing, read and preserve local context; do not reset or revert unrelated workspace edits.
- Preserve current architecture. This pass is constrained to reliability and safety of queue/readiness behavior.

## Daily operating rhythm

1. Read live state (`git status`, `git branch --show-current`, `git remote -v`).
2. Audit source-of-truth files (`launcher.py`, `application_packs.py`, `database.py`, `cli.py`, targeted tests, status docs).
3. Use read-only validation first (`python -m applypilot.cli status`, `python -m applypilot.cli status --ready`). Treat `python -m applypilot.cli pack --lane primary --limit N` as a local-mutating export smoke because it writes pack folders and `00_INDEX.md`.
4. Avoid broad architecture changes; make minimal safe slices.

## 2026-05-08 Privacy/Safety Controls

- Default apply MCP config includes Playwright only; Gmail/email MCP is not part of the apply worker path.
- Claude apply workers must not run with blanket `bypassPermissions`.
- Apply prompts must not include raw CAPTCHA keys or stored account passwords.
- Email-only applications, email verification, SSO, and account creation are manual approval gates, not agent actions.
- `reset_failed()` may only reopen `failed` or `failed:*` statuses; it must not reset `manual`, `manual_review`, `avoid`, `captcha`, `expired`, or `dry_run_ready` buckets.
- Worker Chrome profiles are clean by default. Reusing a real Chrome profile requires `APPLYPILOT_REUSE_CHROME_PROFILE=1` and still excludes credential/session stores.

## Reliability posture targets

- Parser outcomes are constrained to strict statuses:
  - `applied`
  - `dry_run_ready`
  - `expired`
  - `captcha`
  - `login_issue`
  - `failed:<reason>`
- Never write `skipped` as a first-class apply status; only internal failures map through `failed:<reason>`.
- Export readiness must require `tailored_resume_path` + `cover_letter_path` valid PDFs and a ready apply status (`NULL`/empty).
- Stale `in_progress` locks should be moved to explicit manual-review buckets.
- `release_lock` must clear `agent_id` and remain idempotent.

## Update surfaces (this repo)

- `src/applypilot/apply/launcher.py`
- `src/applypilot/apply/prompt.py`
- `src/applypilot/apply/chrome.py`
- `src/applypilot/application_packs.py`
- `src/applypilot/database.py`
- `src/applypilot/cli.py`
- `tests/test_apply_result_parser.py`
- `tests/test_apply_safety.py`
- `tests/test_apply_lane_filter.py`
- `tests/test_database_stats.py`
- `tests/test_application_packs.py`

See `CLAUDE.md` for operational check commands and current acceptance gates.

## Full insight + audit posture (this pass)

- Read the following as hard controls before any non-read-only reliability edits:
  - `AGENTS.md` (this file) and `CLAUDE.md`
  - `INDEX.md` and `PLAN_2026-05-08_OFFLINE_JOB_READINESS.md`
  - `STATUS.md` and `APPLYPILOT_FULL_INSIGHT_AUDIT_2026-05-09.md`
  - 20_INCOME handoff docs before touching queue-related behavior
- Keep a stable repository evidence snapshot:
  - one `git status --short`
  - one tracked-file manifest via `git ls-files`
  - one untracked manifest via `git status --short | Select-Object` in this environment
- Treat these readiness checks as required in this lane:
  - `python -m applypilot.cli status`
  - `python -m applypilot.cli status --ready --lane primary`
  - `python -m applypilot.cli pack --lane primary --limit 1`
- Persist blocker-level drift evidence:
  - save failed reason buckets, stale-in-progress bucket, and manual/avoid/manual_review counts before/after each handoff attempt.

## 2026-05-09 Endgame Completion Baseline (post-upstream pull)

- Branch: `codex/applypilot-job-readiness`
- Upstream comparison: `HEAD...origin/main` shows branch is ahead by 3 commits while keeping fetch safety.
- Live verification snapshot:
  - `status`: 6,429 total / 6,132 with full description / 71 tailored / 58 cover letters / 0 strict-ready primary.
  - `status --ready --lane primary`: manual 5 / avoid 1 / manual_review 1 / failed 10 / in_progress 0 / stale 0.
  - `pack --lane primary --limit 1`: 0 exported, 1 skipped (`already failed: expired`).
  - `status --ready --lane primary` failed-bucket reasons currently include: browser_initialization_error(1), captcha(1), contractor_not_full_time_salaried(2), expired(4), expired_removed(1), below_salary_floor_46k(1).
- Validation status:
  - Targeted tests in this slice pass (`test_apply_result_parser`, `test_application_packs`, `test_database_stats`, `test_apply_lane_filter`).
  - Full test sweep now passes: `92` tests run, `OK`.
- Documentation rule for this pass:
  - Any lower dated `ready`/`pool` notes in `PLAN_2026-05-08_OFFLINE_JOB_READINESS.md` and `STATUS.md` are historical unless they explicitly match this live baseline section.
