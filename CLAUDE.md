# CLAUDE.md

This repository uses manual/offline-safe queue handling.

## 2026-05-09 Endgame Insight Overlay

- Complete repository audit pass executed before this handoff lane:
  - `git fetch origin`
  - `git log --oneline --decorate -n 20`
  - `git log --oneline --left-right --count HEAD...origin/main`
  - `rg --files`
  - `rg -n` marker scan across `.py` and `.md`
  - `python -m compileall -q src tests`
- Fast full-repo scan generated `applier_insight_audit_scan.json` (235 files, 97 `.py`, 76 `.md`, 47 high-risk marker hits, 3 files with >4000-char lines).
- Authoritative per-file walk generated `APPLYPILOT_FILE_AUDIT_MANIFEST_2026-05-09.json` (244 source/operator files, excluding `.git`, `.venv`, `__pycache__`, `.pytest_cache`, and the manifest file itself).
- Executable surfaces (`src`, `tests`, `scripts`) were clean for unresolved `TODO|FIXME|HACK|XXX` markers on this pass.
- Actionable controls added to this lane:
  - add blocker-delta recording for `status --ready --lane primary` before any handoff move;
  - treat `manual_review`, `manual`, `avoid`, `stale_in_progress_orphaned`, and `failed*` as blocking for pack export;
  - keep parser outcomes and queue-lock state changes in one central status write path and test that path first.
- Keep these as required pre-handoff checks:
  - `python -m applypilot.cli status`
  - `python -m applypilot.cli status --ready --lane primary`
  - `python -m applypilot.cli pack --lane primary --limit 1`
  - `python -m unittest discover tests -p \"test_apply_result_parser.py\"`,
    `test_application_packs.py`, `test_database_stats.py`, `test_apply_lane_filter.py`

## Immediate constraints

- Manual posture by default: `applypilot apply` now resolves to dry-run unless real submission is explicitly requested with `--submit --confirm-submit APPLY`.
- Email-only applications, mailbox verification, SSO, account creation, CAPTCHA solving, and public/employer communication are manual handoff gates.
- Apply workers use Playwright MCP only by default; do not re-add Gmail/mail MCP or blanket permission bypass without explicit approval.
- Do not push to public upstream by default.
- Keep `.applypilot` runtime as the operational source of truth.
- Preserve `applypilot` queue integrity around `in_progress`, `apply_status`, and `attempts`.

## Required read-before-change list

- `src/applypilot/apply/launcher.py`
- `src/applypilot/apply/prompt.py`
- `src/applypilot/apply/chrome.py`
- `src/applypilot/application_packs.py`
- `src/applypilot/database.py`
- `src/applypilot/cli.py`
- `tests/test_apply_safety.py`
- `tests/test_apply_result_parser.py`
- `tests/test_apply_lane_filter.py`
- `tests/test_database_stats.py`
- `tests/test_application_packs.py`
- `APPLYPILOT_FULL_INSIGHT_AUDIT_2026-05-09.md`
- `PLAN_2026-05-08_OFFLINE_JOB_READINESS.md`
- `STATUS.md`

## Evidence checks

- Upstream sync: `git fetch origin`
- Status command: `python -m applypilot.cli status`
- Ready view: `python -m applypilot.cli status --ready`
- Pack export smoke, local-mutating: `python -m applypilot.cli pack --lane primary --limit 1`
- Parser unit smoke: `python -m unittest discover tests -p "test_apply_result_parser.py"`
- Queue health smoke: `python -m applypilot.cli status --ready --lane primary`

## Full-insight sweep protocol (mandatory for endgame execution)

- Keep a one-file-per-surface audit chain in this order:
  - `AGENTS.md`
  - `INDEX.md`
  - `STATUS.md`
  - `PLAN_2026-05-08_OFFLINE_JOB_READINESS.md`
  - `APPLYPILOT_FULL_INSIGHT_AUDIT_2026-05-09.md`
- Run these live-readiness probes together and archive outputs:
  - `python -m applypilot.cli status`
  - `python -m applypilot.cli status --ready --lane primary`
  - `python -m applypilot.cli pack --lane primary --limit 1`
- If local shell does not resolve `applypilot` modules, set `PYTHONPATH=src` and use `.venv\\Scripts\\python.exe`.
- If runtime CLI output appears garbled on this Windows shell, re-run with `PYTHONIOENCODING=utf-8` to avoid dropped status text.
- Run the queue correctness regression suite together before handoff:
  - `python -m unittest discover tests -p "test_apply_result_parser.py"`
  - `python -m unittest discover tests -p "test_application_packs.py"`
  - `python -m unittest discover tests -p "test_database_stats.py"`
  - `python -m unittest discover tests -p "test_apply_lane_filter.py"`
- Keep a rolling acceptance note in:
  - `APPLYPILOT_FULL_INSIGHT_AUDIT_2026-05-09.md`
- Do not mark anything queue-safe until:
  - `apply_status` is null/empty, and
  - both tailored resume/cover PDFs are valid file-backed artifacts,
  - no terminal blockers (`manual`, `avoid`, `manual_review`, `stale_in_progress_orphaned`, `failed*`) are present.
- `status --ready` now subtracts DB-ready rows whose sibling resume/cover PDFs are missing or too small, and reports those rows as `Invalid artifacts`.

- Before any pack handoff, keep one explicit queue-drift point:
  - `python -m applypilot.cli status --ready --lane primary` and preserve `Manual / Avoid / Manual Review / Failed / Stale` deltas in `APPLYPILOT_FULL_INSIGHT_AUDIT_2026-05-09.md`.
