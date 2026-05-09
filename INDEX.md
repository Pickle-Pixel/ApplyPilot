# ApplyPilot Repository Index and Inspection Map

Generated for the 2026-05-09 Full Insight sweep.

2026-05-08 privacy/safety execution update:
- Apply workers are dry-run/manual-safe by default; real submission requires `--submit --confirm-submit APPLY`.
- Default apply MCP config no longer includes Gmail/mail tooling and no longer uses blanket permission bypass.
- Prompt generation must not include raw CAPTCHA keys, stored passwords, mailbox-reading instructions, or email-send actions.
- `status --ready` now reports only rows with valid sibling resume/cover PDFs as ready and surfaces missing/invalid PDFs as `Invalid artifacts`.
- `pack` remains useful for export smoke checks, but it is local-mutating and writes application-pack folders.

Execution evidence refresh for this sweep:
- Fast full-file inventory executed on 2026-05-09: 235 total files discovered; 97 Python, 76 Markdown.
- Authoritative per-file manifest executed on 2026-05-09: 244 source/operator files walked, hashed, and counted in `APPLYPILOT_FILE_AUDIT_MANIFEST_2026-05-09.json`.
- High-risk marker hits: 47 files (`applier_insight_audit_scan.json`), 3 files with very long lines.
- No unresolved `TODO|FIXME|XXX|HACK` markers remain in source/test/scripting runtime files.
- `python -m compileall -q src tests` passed on this pass.
- Upstream sync and branch safety completed before edits.
- `python -m applypilot.cli status` (repo runtime): 6,429 total; 6,132 full descriptions; 0 strict-ready primary jobs.
- `python -m applypilot.cli status --ready --lane primary`: manual 5, avoid 1, manual_review 1, failed 10, stale 0.
- `python -m applypilot.cli pack --lane primary --limit 1`: 0 exported, 1 skipped (`already failed: expired`).
- `get_apply_queue_stats()` blocker-bucket visibility is now first-class in this pass and treated as the acceptance control for any queue handoff.

- Tracked inventory: 70 files (via `git ls-files`).
- Working-tree drift: 134 items in untracked/unstaged state (`git status --short --untracked-files=all`).
- Live docs and source-of-truth:
  - Authoritative baseline: `APPLYPILOT_FULL_INSIGHT_AUDIT_2026-05-09.md` + `ENDGAME_EXECUTION_LOG_2026-05-09.md` + `AGENTS.md` + `CLAUDE.md` + `PLAN_2026-05-08_OFFLINE_JOB_READINESS.md`.
  - Older queue notes are preserved as historical context only unless explicitly aligned to this live baseline.
- Last audited: 2026-05-09.

## Tracked source-of-truth files

### Controls
- .env.example
- .gitignore
- .github/workflows/ci.yml
- .github/workflows/publish.yml
- CHANGELOG.md
- CONTRIBUTING.md
- LICENSE
- README.md
- docs/reports/2026-04-18-cleanup-report.md
- docs/reports/2026-04-25-application-cycle.md
- profile.example.json
- pyproject.toml
- STATUS.md
- PLAN_2026-05-08_OFFLINE_JOB_READINESS.md
- INDEX.md
- STORAGE_README.md
- LANE_CONSOLIDATION_2026-04-28.md

### Runtime config
- src/applypilot/config/employers.yaml
- src/applypilot/config/searches.example.yaml
- src/applypilot/config/sites.yaml

### Scripts
- scripts/ingest_from_xlsx.py
- scripts/read_docx.py
- scripts/regen_resume.py
- scripts/score_batch.py
- scripts/test_llm_smoke.py

### Runtime code
- src/applypilot/__init__.py
- src/applypilot/__main__.py
- src/applypilot/application_packs.py
- src/applypilot/cli.py
- src/applypilot/config.py
- src/applypilot/database.py
- src/applypilot/pipeline.py
- src/applypilot/view.py
- src/applypilot/wizard/__init__.py
- src/applypilot/wizard/init.py

### Apply stage
- src/applypilot/apply/__init__.py
- src/applypilot/apply/chrome.py
- src/applypilot/apply/dashboard.py
- src/applypilot/apply/launcher.py
- src/applypilot/apply/prompt.py

### Discovery stage
- src/applypilot/discovery/__init__.py
- src/applypilot/discovery/jobspy.py
- src/applypilot/discovery/smartextract.py
- src/applypilot/discovery/workday.py

### Enrichment stage
- src/applypilot/enrichment/__init__.py
- src/applypilot/enrichment/detail.py

### LLM and scoring
- src/applypilot/llm.py
- src/applypilot/scoring/__init__.py
- src/applypilot/scoring/cover_letter.py
- src/applypilot/scoring/job_lanes.py
- src/applypilot/scoring/pdf.py
- src/applypilot/scoring/role_signals.py
- src/applypilot/scoring/scorer.py
- src/applypilot/scoring/tailor.py
- src/applypilot/scoring/template_matcher.py
- src/applypilot/scoring/validator.py

### Test matrix
- tests/test_apply_lane_filter.py
- tests/test_apply_result_parser.py
- tests/test_application_packs.py
- tests/test_chrome_profile_setup.py
- tests/test_config_setup.py
- tests/test_cover_letter_generation.py
- tests/test_database_stats.py
- tests/test_job_lanes.py
- tests/test_jobspy_filters.py
- tests/test_llm_fallback.py
- tests/test_llm_max_tokens.py
- tests/test_pdf_conversion.py
- tests/test_role_signals.py
- tests/test_scorer_contextual_caps.py
- tests/test_scorer_quick_filters.py
- tests/test_tailor_generation.py
- tests/test_tailor_quality_2026_04_29.py

### Templates
- templates/README.md
- templates/cover_letter_template.md
- templates/daily_run_checklist.md
- templates/pack_quality_gate.md
- templates/resume_tailoring_framework.md

## Untracked runtime/workbench files (audited as non-source)

These are currently present in this worktree and were intentionally left out of the core patch surface:

- .claw_dispatch.ps1
- .claw_response_raw.json
- APPLYPILOT_DRY_RUN_2026-04-28.md
- AGENTS.md
- CLAUDE.md
- intent.md
- APPLYPILOT_FULL_INSIGHT_AUDIT_2026-05-09.md
- CLAW_DEFENSIBILITY_AUDIT_2026-04-27.md
- CRISIS_PUSH_2026-04-28.md
- HIGH_LEVERAGE_2026-04-28.md
- RESUME_AUDIT_2026-04-27.md
- RESUME_FIXES_APPLIED_2026-04-27.md
- PIPELINE_FOLLOWUPS_2026-04-27.md
- TITLEFIT_LOOP_FINAL_2026-04-27.md
- TIMELINE_FIXES_APPLIED_2026-04-27.md
- NIM_LOCAL_SETUP.md
- PANEL_FIXES_APPLIED_2026-04-27.md
- PANEL_LOOP_FINAL_2026-04-27.md
- PANEL_REVIEW_2026-04-27.md
- LICENSE_BOOST_2026-04-28.md
- LANE_CONSOLIDATION_2026-04-28.md
- interview_prep/
- applications_log/
- council/
- drafts/
- leads/
- loop_journal/
- scripts/_lane_v2_audit.py
- scripts/_lane_v2_audit_old_pdfs.py
- scripts/_lane_v2_audit_pdf.py
- scripts/_lane_v2_changelog.py
- scripts/_lane_v2_changelog2.py
- scripts/_lane_v2_html_dump.py
- scripts/_lane_v2_inspect_docx.py
- scripts/_lane_v2_patch_pdfpy.py
- scripts/_lane_v2_update_docx.py
- scripts/_lane_v2_update_profile.py
- scripts/_lane_v2_update_txt.py
- scripts/boost_license_priority.py
- scripts/fix_jh_built_bullet.py
- scripts/rebuild_overhaul_pdfs.py

## Control rule for this repository

- Do not touch untracked workbench artifacts unless they are part of an approved task slice.
- Keep `origin` push-path disabled.
- Treat this index as the baseline for future diff and audit comparisons.
