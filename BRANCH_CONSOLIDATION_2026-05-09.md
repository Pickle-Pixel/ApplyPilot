# ApplyPilot Branch Consolidation - 2026-05-09

## Scope

Local/manual-only consolidation of ApplyPilot agent and PR work. No public push, no job applications, no email, and no scheduler or cron re-enable actions were performed.

## Preservation

- Full repo ref bundle: `C:\Users\bibyd\.applypilot\90_BACKUPS_AND_OLD_ROOT_FILES\applypilot-branch-preservation-20260509-153526\applypilot-all-refs.bundle`
- Dirty worktree backup patches and untracked zip: `C:\Users\bibyd\.applypilot\90_BACKUPS_AND_OLD_ROOT_FILES\applypilot-branch-preservation-20260509-153526`
- Resume backup set: `C:\Users\bibyd\.applypilot\90_BACKUPS_AND_OLD_ROOT_FILES\resume-upgrade-backup-20260509-154214`
- Local dirty work preserved as branch: `preserve/local-working-tree-20260509-153526`
- All GitHub PR heads fetched into local branches: `preserve/pr-7`, `preserve/pr-8`, `preserve/pr-20`, `preserve/pr-23`, `preserve/pr-24`, `preserve/pr-25`, `preserve/pr-28`, `preserve/pr-29`, `preserve/pr-30`, `preserve/pr-31`, `preserve/pr-34`, `preserve/pr-38`, `preserve/pr-41`, `preserve/pr-47`, `preserve/pr-48`

## Consolidation Branch

Current local integration branch:

`codex/full-branch-consolidation-20260509`

Merged into the consolidation branch:

- `preserve/local-working-tree-20260509-153526` - local ApplyPilot safety/docs/tests/worktree work
- `preserve/pr-7` - submitted/failed application tables
- `preserve/pr-8` - applied/failed dashboard indicators and auto-apply command copy button
- `preserve/pr-25` - Greenhouse discovery/config/CLI support
- `preserve/pr-28` - `SECURITY.md`
- `preserve/pr-29` - security and quality hardening, merged conservatively with current safety lane winning overlapping conflicts
- `preserve/pr-31` - Windows Unicode doctor fix
- `preserve/pr-38` - OpenAI `max_completion_tokens` handling
- `preserve/pr-41` - duplicate PDF filename collision fix
- `preserve/pr-47` - local-only ApplyPilot publish sync documentation

Preserved but not merged:

- `preserve/pr-20` - auto-apply backend expansion. Skipped because it changes safety-sensitive apply launcher/prompt/CLI behavior.
- `preserve/pr-23` - older LiteLLM/onboarding branch, closed.
- `preserve/pr-24` - LiteLLM migration. Skipped because it adds a production dependency and collides with current provider fallback work.
- `preserve/pr-30` - LLM-first onboarding branch built on LiteLLM-era changes; production dependency gate remains.
- `preserve/pr-34` - Gustaf preset/profile branch. Skipped because it adds unrelated external profile/resume data.
- `preserve/pr-48` - large dispatch/web/analyze branch. Skipped because it adds `litellm` and optional web dependencies plus a large dashboard/server surface.
- `origin/dev` - online branch with LLM-first wizard and `pypdf` dependency; skipped under production-dependency approval gate.

## Resume Updates

Updated local ApplyPilot resume/profile surfaces only:

- `C:\Users\bibyd\.applypilot\resume.txt`
- `C:\Users\bibyd\.applypilot\profile.json`
- `C:\Users\bibyd\.applypilot\resume-overhaul\*.txt`
- `C:\Users\bibyd\.applypilot\resume-overhaul\*.docx`
- `C:\Users\bibyd\.applypilot\resume-overhaul\*.pdf`
- `C:\Users\bibyd\.applypilot\00_CURRENT_RESUMES\*.docx`
- `C:\Users\bibyd\.applypilot\00_CURRENT_RESUMES\*.pdf`
- `C:\Users\bibyd\.applypilot\00_CURRENT_RESUMES\MASTER_RESUME.txt`
- `C:\Users\bibyd\.applypilot\00_CURRENT_RESUMES\resume.txt`

Truth-only changes:

- Sharpened master runtime summary toward benefits verification, eligibility documentation, member documentation, claims quality, release-of-information, and healthcare QA review.
- Added Alorica seasonal regulated-program support where already supported by existing profile/audit history.
- Tightened WFM variant summary toward full-time WFM, real-time analysis, capacity planning, and analytical operations.
- Added profile target roles and skills already supported by the resume system: benefits verification, eligibility verification, medical records, release-of-information terminology, claims quality, and healthcare documentation.
- Added `Alorica` to `profile.json` preserved companies.
- Disabled ligatures in the PDF renderer so ATS text extraction keeps words like `benefits` and `office` parseable.

No new numeric achievements were invented.

## Verification

Run from `C:\Users\bibyd\Projects\dev\ApplyPilot\.worktrees\pr-integration-20260509` with `PYTHONPATH=src`:

- `python -m compileall -q src tests` - passed
- `python -m unittest discover tests` - passed, `98` tests, `OK`
- `python -m applypilot.cli status` - `6429` jobs, `6132` full descriptions, `71` tailored resumes, `58` cover letters, `0` ready primary, `11` applied, `31` apply errors
- `python -m applypilot.cli status --ready --lane primary` - Ready `0`, Invalid artifacts `0`, Manual `5`, Avoid `1`, Manual Review `1`, Failed `10`, In progress `0`, Stale in-progress `0`
- `python -m applypilot.cli pack --lane primary --limit 1` - `0` exported, `1` skipped (`already failed: expired`), output folder `C:\Users\bibyd\.applypilot\application_packs\ready_primary_2026-05-09`
- PDF text extraction check - rebuilt PDFs are 2 pages each and no longer contain `fi` or `ffi` ligatures in extracted text

## Current Branch Status

Clean worktrees after verification:

- `C:\Users\bibyd\Projects\dev\ApplyPilot` on `preserve/local-working-tree-20260509-153526`
- `C:\Users\bibyd\Projects\dev\ApplyPilot\.worktrees\pr-integration-20260509` on `codex/full-branch-consolidation-20260509`

Remote push remains disabled:

`origin DISABLED-PUSH-TO-KEEP-LOCAL (push)`
