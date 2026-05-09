## 2026-05-09 Full Repo File Scan Layer (v1.2)

- Files audited: `235` discovered (`rg --files` + workspace walk).
- Authoritative per-file manifest: `244` source/operator files walked, hashed, and counted in `APPLYPILOT_FILE_AUDIT_MANIFEST_2026-05-09.json`; excluded `.git`, `.venv`, `__pycache__`, `.pytest_cache`, and the manifest file itself.
- Extension counts: `97` Python, `76` Markdown, `22` text, `16` JSON, `6` YAML, `3` PowerShell.
- Marker scan (`TODO|FIXME|XXX|HACK` + `print(` + `pdb.set_trace`): `47` files matched at least once.
- `python -m compileall -q src tests`: pass.
- `rg -n "TODO|FIXME|XXX|HACK" src tests scripts`: no matches.
- Raw evidence artifact: `applier_insight_audit_scan.json`.
- Key-like string scan: `15` files contain config-key-looking labels or examples. This is not proof of leaked secrets, but `NIM_LOCAL_SETUP.md`, `APPLYPILOT_DRY_RUN_2026-04-28.md`, and historical lead/workbench docs should remain private until a manual secret scrub is completed.
- New control rule: queue-handoff actions must treat blocker-bucket deltas (`manual`, `avoid`, `manual_review`, `failed`, `stale`) as mandatory evidence before export.
# ApplyPilot Full Insight Audit - 2026-05-09

## What I audited

- Repo code and test surfaces tied to queue integrity: `src/applypilot/apply/launcher.py`, `src/applypilot/application_packs.py`, `src/applypilot/database.py`, `src/applypilot/cli.py`, `tests/test_apply_result_parser.py`, `tests/test_apply_lane_filter.py`, `tests/test_database_stats.py`, `tests/test_application_packs.py`.
- Operational posture docs: `STATUS.md`, `PLAN_2026-05-08_OFFLINE_JOB_READINESS.md`, `intent.md`, `20_INCOME` status/review docs.
- Tooling posture and environment from memory/task history in `MEMORY.md` and upstream notes.
- Repo-level indexed inventory from `git ls-files` and key doc scans.
- Full-repo file-surface sweep used a manifest of all tracked files (`git ls-files`) plus dirty-tree scan (`git status --short --untracked-files=all`) before code-path edits; current tracked file count is 70 and untracked working artifacts are 134 in this pass.
- Repo-wide high-signal scan also checked for actionable `TODO/FIXME/HACK/X` markers across tracked `.py`/`.md` surfaces to catch stale instructions and quick follow-up hooks.

### 2026-05-09 Snapshot (v1.1 final-hardening pass)
- Parser contract validated for strict outputs: `applied`, `dry_run_ready`, `expired`, `captcha`, `login_issue`, `failed:<reason>`.
- Runtime queue checks now include:
  - `python -m applypilot.cli status --ready --lane primary`
  - `python -m applypilot.cli pack --lane primary --limit 1`
  - targeted unit suite: `tests.test_apply_result_parser`, `tests.test_application_packs`, `tests.test_database_stats`, `tests.test_apply_lane_filter`.
- Queue signal now reads: 0 ready primary jobs, 31 apply errors, 11 applied, 6,132 jobs with full description.
- `release_stale_locks()` still routes stale work into manual-review buckets with explicit reasons.

## Evidence-based decisions made

1. Upstream posture
- `git fetch origin` completed.
- Branch remained at `codex/applypilot-job-readiness` with head at `c6457c8` and `origin/main` at `4a8d521`.
- Upstream fetch safety is in place and branch stays local.

2. Parser and lock safety
- Parser contract is now constrained to explicit status outputs and explicit failure categories.
- Expired/no-result and backend-loss cases now map to explicit failure reasons for retry policy.
- `PERMANENT_FAILURES` includes terminal outcomes from this lane.

3. Queue and export integrity
- Ready export now excludes blocked and failed rows and verifies both tailor+cover PDFs are valid.
- Skip reporting now carries user-readable reasons in 00_INDEX output.

4. Queue observability
- DB stats helpers now provide explicit ready/manual/avoid/manual_review/failed/stale buckets.
- CLI `status --ready` now reports blocker reason buckets and failed-reason breakdown.

5. Manual/offline alignment
- No architecture drift was introduced in this pass.
- The lane still defaults to dry-run mode in operation guidance.

### Chat-history-derived reliability insights

- Manual safety is the highest-priority invariant across the user’s recent interactions: treat any ready signal as advisory until `status --ready` blocker buckets are reviewed and artifacts are valid PDFs.
- False-ready risk was repeatedly observed when queue docs drifted from live metrics; the acceptance criteria now demands stale notes are tagged historical unless overwritten by an explicit live refresh.
- The user repeatedly requests end-to-end slices without waiting for handoff approvals; this sweep keeps execution continuous but bounded to read-only/live-read checks for handoff-critical commands.
- Settings/configuration to study in the next lane:
  - `CHROME_PATH` path drift in `~/.applypilot/.env` (Playwright path changes still occur on updates).
  - `apply_status` contract in DB cleanup filters (`NULL`/empty only) before pack export.
  - stale lock TTL and heartbeat cadence to prevent silent worker deadlocks (`in_progress` lingering > 60 mins).
  - `origin` push URL remains `DISABLED-PUSH-TO-KEEP-LOCAL`; leave untouched until explicit approval to re-enable.
  - `status --ready` blocked-bucket deltas (`manual`, `avoid`, `manual_review`, `failed`, `stale`) should be captured before every handoff and compared against prior run.

## Hard constraints that should stay active

- `applypilot` real submits stay off by default.
- `origin` push URL must stay disabled unless explicitly approved.
- Do not clear unrelated rows when target-specific acquire/target_url fails.
- Keep `release_lock` behavior idempotent and always clear `agent_id`.

## Actionable settings / configurations to study next

1. Runtime environment
- Keep `CHROME_PATH` explicit and valid for the machine path used by Playwright.
- Keep `PYTHONIOENCODING=utf-8` if doctor/console rendering shows encoding truncation.

2. Reliability operations
- Re-run `python -m applypilot.cli status --ready` daily.
- Capture `get_apply_queue_stats()` deltas before pack export if queue behavior changes.
- Consider a temporary alert for rising `in_progress` without matching `last_attempted_at` heartbeat.

3. LLM/Model profile
- Track actual token/runtime behavior when using multi-tier chains; provider order changes should stay in `~/.applypilot/.env` only.
- Keep old model chains working as-is unless user asks for migration.

## Files updated/verified

- Verified status surfaces and smoke tests are now aligned with the upgrade plan.
- All above files are now stable enough for manual review and queue handoff.

## Open risks

- Full suite now passes after restoring the annotation-first score cap at 7 (`92` tests, `OK`).
- Queue counts in historical notes can drift; treat `2026-04` snapshots as historical unless refreshed.
- Any broad queue-sourced leads from legacy review files must be re-checked before re-activation.
- `repo runtime` still has many non-source artifacts and manual sourcing notes in untracked state; they are intentionally outside this plan slice but may affect operational interpretation until manually reconciled.

## 2026-05-08 Privacy/Safety Execution Baseline

- Local shell timestamp: `2026-05-08T22:07:34.1819119-05:00`.
- Active branch: `codex/applypilot-job-readiness`.
- Push posture remains disabled locally: do not change `origin` push URL without explicit approval.
- Baseline `git status --short --branch --untracked-files=all` showed a dirty private worktree, including staged `SECURITY.md`, staged/modified `src/applypilot/view.py`, 21 modified tracked source/test/doc files, and many untracked private operator artifacts.
- Baseline `git diff --stat` showed 21 tracked files changed with `1770` insertions and `239` deletions before this safety pass.
- Baseline `git diff --check` exited clean.
- Execution controls for this pass:
  - no real applications;
  - no email send/read/delete/archive/move actions;
  - no public push, tag, publish, or deployment;
  - no destructive cleanup of private drafts, leads, council outputs, application logs, or Claw artifacts.
- Specialist findings to resolve first:
  - default apply path exposed too much authority through Gmail MCP and permission bypass;
  - dry-run was prompt-only rather than mechanically guarded;
  - raw secrets/PII could enter prompts or page scripts;
  - `reset_failed()` could reopen safety blockers;
  - `pack` is local-mutating export, not read-only validation.

### Actions Applied In Privacy/Safety Pass

- Installed declared dev extras into the local venv so `pytest` and `ruff` are available for real verification.
- Added regression coverage in `tests/test_apply_safety.py` for:
  - no Gmail MCP in default apply config;
  - no blanket Claude permission bypass;
  - explicit submit confirmation gate;
  - failed-status-only reset behavior;
  - no stored passwords, CAPTCHA secrets, or mail actions in prompts.
- Added PDF-artifact readiness coverage in `tests/test_database_stats.py`.
- Changed `applypilot apply` so real submission requires `--submit --confirm-submit APPLY`; otherwise it resolves to dry-run behavior.
- Removed Gmail/mail MCP and blanket `bypassPermissions` from default apply workers.
- Reworked prompt instructions so email-only applications, email verification, account creation, SSO, and CAPTCHA solving stop for manual handoff instead of using mailbox/API actions.
- Changed worker Chrome setup to create clean profiles by default; real-profile reuse is opt-in via `APPLYPILOT_REUSE_CHROME_PROFILE=1` and excludes credential/session stores.
- Constrained `reset_failed()` to `failed` and `failed:*` statuses only.
- Updated `status --ready` to subtract rows whose sibling resume/cover PDFs are missing or too small and report them as `Invalid artifacts`.
- Expanded `.gitignore` for private job-search/operator artifacts without moving or deleting any existing files.

### Final Verification For Privacy/Safety Pass

- Dev tooling repair completed in the existing local venv with `python -m pip install -e ".[dev]"`.
  - `pytest --version`: `pytest 9.0.3`.
  - `ruff --version`: `ruff 0.15.12`.
- Verification commands after the final safety and lint cleanup:
  - `.\.venv\Scripts\python.exe -m compileall -q src tests`: PASS.
  - `.\.venv\Scripts\python.exe -m unittest discover tests -v`: PASS, `98` tests.
  - `.\.venv\Scripts\python.exe -m pytest tests -v -p no:cacheprovider`: PASS, `102` tests.
  - `.\.venv\Scripts\python.exe -m ruff check src tests scripts`: PASS.
  - `$env:PYTHONIOENCODING='utf-8'; .\.venv\Scripts\python.exe -m applypilot.cli doctor`: PASS.
  - `$env:PYTHONIOENCODING='utf-8'; .\.venv\Scripts\python.exe -m applypilot.cli status --ready --lane primary`: PASS.
  - `git diff --check`: PASS.
- Live doctor state:
  - profile, resume, searches, python-jobspy, LLM key, Claude Code CLI, Chrome/Chromium, and Node/npx all report OK.
  - CapSolver remains optional; CAPTCHA provider keys are not placed in prompts or browser scripts.
  - CLI reports Tier 3 capability, but real submission is still guarded by `--submit --confirm-submit APPLY`.
- Live primary-lane readiness:
  - Ready: `0`.
  - Invalid artifacts: `0`.
  - Manual: `5`.
  - Avoid: `1`.
  - Manual Review: `1`.
  - Failed: `10`.
  - In progress: `0`.
  - Stale in-progress: `0`.
  - Failed reasons: `below_salary_floor_46k(1)`, `browser_initialization_error(1)`, `captcha(1)`, `contractor_not_full_time_salaried(2)`, `expired(4)`, `expired_removed(1)`.
- Contradiction scan:
  - No stale `pack is read-only` guidance found in active docs; current docs classify pack as local-mutating export.
  - No default Gmail MCP config found in the apply worker path.
  - No prompt references found for `send_email`, `search_emails`, `read_email`, raw CAPTCHA `clientKey`, token injection, or provider API task creation.
  - `bypassPermissions` only appears in regression tests and historical audit context, not in the worker command path.
- Lint cleanup note:
  - Safe Ruff fixes cleaned pre-existing unused imports and trivial f-string issues across source/tests/scripts so the declared verification command now passes.
  - Manual cleanup removed unused prompt/tailor variables and kept the `scripts/score_batch.py` import-path bootstrap explicit with `# noqa: E402`.
- Current git posture:
  - Dirty private worktree intentionally preserved.
  - Existing staged `SECURITY.md` and staged/modified `src/applypilot/view.py` were not reverted.
  - No untracked private artifacts were moved or deleted.
  - No push, tag, publish, email action, browser submission, real job application, or external automation was performed.

