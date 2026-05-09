# Changelog


2026-04-28 17:26 | dispatch:lane-consolidation-v2-depth-fix | Projects\dev\ApplyPilot\LANE_CONSOLIDATION_2026-04-28.md | Resume PDFs were rendering tiny (1 page / ~190 extracted words) due to a regression in src/applypilot/scoring/pdf.py: build_html keyed sections on exact 'EXPERIENCE' but the txt uses 'PROFESSIONAL EXPERIENCE'; 'PROFESSIONAL HIGHLIGHTS' had no handler at all. Added _find_section() forgiving lookup + HIGHLIGHTS bullet renderer. Regenerated 5 PDFs (now 2 pages / 758-783 words / 79 KB each, matching pre-regression behavior). Backups: .bak-2026-04-28-depth on pdf.py + 5 PDFs.

2026-04-28 17:10 | dispatch:lane-consolidation-v2 | Projects\dev\ApplyPilot\LANE_CONSOLIDATION_2026-04-28.md | Council top correction #2: collapsed 4-lane positioning to "Licensed Insurance Compliance & QA Specialist". Master + 4 variants + profile.json target_role updated.

2026-04-28 15:43 | dispatch:council-review | Projects\dev\ApplyPilot\council\COUNCIL_VERDICT_2026-04-28.md | 3-LLM council (Kimi K2.5 fallback / DeepSeek V3.2 / GLM 5.1) reviewed tonight's job-search effort. Top 5 corrections surfaced. Kimi K2.6 was attempted but token-burned on internal reasoning; switched to K2.5 same family non-thinking variant.
All notable changes to ApplyPilot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Duplicate PDF filename collisions** (#11) - Appended MD5 hash of the job URL to the filename prefix for tailored resumes and cover letters to prevent overwriting when multiple jobs have identical titles.
- **OpenAI newer models (gpt-4.1+, gpt-5.x, o-series) rejected with HTTP 400** - these models
  require `max_completion_tokens` instead of the legacy `max_tokens` parameter. `_chat_compat()`
  now detects the active model prefix and sends the correct parameter automatically.

### Dispatch log

2026-05-08 22:20 | dispatch:privacy-safety-hardening | Projects\dev\ApplyPilot\APPLYPILOT_FULL_INSIGHT_AUDIT_2026-05-09.md | ApplyPilot auto-apply made manual-first: `applypilot apply` defaults to dry-run unless `--submit --confirm-submit APPLY` is present, default MCP config is Playwright-only, blanket permission bypass was removed, prompts no longer expose stored passwords/CAPTCHA secrets/mailbox actions, `reset_failed()` only resets failed statuses, clean Chrome worker profiles are default, and `status --ready` now excludes rows with missing/invalid PDFs.

2026-04-28 20:30 | dispatch:alorica-non-phone-check | Projects\dev\ApplyPilot\leads\INSURANCE_LICENSED_NON_PHONE_2026-04-28.md (Alorica section appended) | Verified Alorica US openings via direct Oracle HCM REST API (`fa-euxw-saasfaprod1.fa.ocs.oraclecloud.com/hcmRestApi/...recruitingCEJobRequisitions`). 34 US reqs total â€” 32 Customer Experience (all phone CSR / sales / advocate variants), 2 Operations Support (TN warehouse + computer repair). ZERO non-phone, ZERO healthcare/claims/QA/compliance/documentation/verification roles in US public ATS. 4 remote-flagged US roles all phone-primary (234903 Healthcare CSR, 235412 Customer Solutions Rep telecom, 235224 WAH Bilingual CSR, 236120 Hawaii-only Healthcare Advocate). Verdict: Alorica portal worth 10 min ONLY for the rehire-eligibility paper trail Alorica's email asked for; submit closest-to-Saint-Joseph req (237915 Lead CX Agent, 234923 CSR St. Joseph, or 236213 Learning Delivery Associate) and tag rehire-eligibility-only in PIPELINE_TRACKER so future ApplyPilot runs don't re-prioritize Alorica off this signal. Alorica's back-office / claims / QA work is staffed in India (Bangalore 64 reqs) and Philippines (Quezon City 31 reqs), not US.

2026-04-28 15:30 | dispatch:outreach-execute | Projects\dev\ApplyPilot\applications_log\OUTREACH_LOG_2026-04-28.md | Sent 8 outreach messages: 1 Concentrix Gmail reply (from bibyd21@gmail.com, draft's darrinbiby243 was wrong, real thread is on bibyd21, user confirmed override) + 7 LinkedIn connect requests. Five with notes: Kelsie Hester (ASI), Pravin Selvaraj (Cloudious), Rohini M (BayOne), Bethany Pence (Dexian), Corinne Brown (Daniels Health). Two without notes due to LinkedIn free-tier 5-personalized-invite monthly cap being hit: Vignesh R (Trident), Shaik Wahab (TEK NINJAS). Phone-screen call-to-action included in 6 of 8 initial messages. The 2 no-note connects will carry the phone-screen ask in their follow-up DM after acceptance per existing draft. LinkedIn note limit is now 200 chars (drafts assumed 300), so all 5 with-note messages were compressed in-flight. Concentrix reply message ID 19dd5b3300d45d90 on existing thread 19dc1436991d752a confirmed via Gmail MCP get_thread. All 7 LinkedIn invites confirmed by "Pending" UI state + "Invitation sent" toast. 90s pacing held, no captcha.

2026-04-28 20:06 | dispatch:insurance-licensed-source | Projects\dev\ApplyPilot\leads\INSURANCE_LICENSED_NON_PHONE_2026-04-28.md | 15 insurance-licensed non-phone openings sourced and verified where possible. Tier 1 (3 verified live via direct ATS fetch): Sagility BroadPath Claims Processor, Capital Rx WFM Analyst Customer Care, Sana Benefits Member Advocate. Tier 2 (7 LinkedIn-only â€” eye-check needed): CVS BV x2, Lash Group BV, Option Care BV, Cornerstone BV, Insight Global IV, Versant Health Claims Processor III. Tier 3 (5 lower-confidence). 11 companies dry/disqualified: Pie (wrong license), Slide (FL license), Centene (clinical), Lemonade, Sharecare (phone), CVS Service Advocate (phone), Connexion Point + Exact Medicare + 1840 & Co (sales/1099/phone), GoHealth sales tier, Devoted Health (TODO direct Workday), Humana/Aetna/Cigna/UHG digital-advocate (TODO direct Workday). Sana Benefits Claims Processor 117a9f33 confirmed 404. Caveat: LinkedIn URLs login-walled â€” direct verification only for Greenhouse/Lever/Workday/careers ATS.

2026-04-28 15:05 | dispatch:applypilot-license-boost | Projects\dev\ApplyPilot\LICENSE_BOOST_2026-04-28.md | Insurance-license priority boost script `scripts/boost_license_priority.py` shipped. 48 rows boosted +3; 4 land at fit>=7 (Datavant Health Info Specialist I, Datavant Vendor Support, GoHealth Urgent Care Live Messaging Agent, UT MD Anderson Credentialing Coordinator). DB column `boost_license_v1` tracks who got boosted. Idempotent across re-runs.

2026-04-28 15:00 | dispatch:applypilot-dry-run | Projects\dev\ApplyPilot\APPLYPILOT_DRY_RUN_2026-04-28.md | OLLAMA_API_KEY set at User scope. Live testing surfaced two real issues â€” Ollama Cloud 60s gateway timeout and reasoning-model token-budget exhaustion. Pivoted to `ollama:minimax-m2.5:cloud` via local proxy as Tier 0. 5 cover letters generated (4 PASS, 1 empty Sharecare anomaly). LLMClient `_TIMEOUT` 120â†’300, new `ocloud:` prefix for cloud-direct chain entries. cover_letter.py `max_tokens` reset to 2048 for non-reasoning chain. AUTO_APPLY remains procedurally locked.

2026-04-28 23:55 | dispatch:high-leverage-track | Projects\dev\ApplyPilot\HIGH_LEVERAGE_2026-04-28.md | LinkedIn DM drafts for 7 recruiters (3 substitutions documented: Concentrixâ†’ASI, Sagilityâ†’Cloudious, Jobgetherâ†’TEK NINJAS) + Track B chat-primary career-page scout for 10 listed companies (0 viable packets, 2 manual handoffs surfaced for Wayfair + ModSquad).

2026-04-28 14:23 | dispatch:applypilot-dry-run-blocked | Projects\dev\ApplyPilot\APPLYPILOT_DRY_RUN_2026-04-28.md | Initial state assessment: ApplyPilot prompts audited, API key tier verified (BLOCKED â€” no cloud key set), dry-run NOT generated. Cerebras 8B fallback rejected per Darrin standing rule. Replaced by 15:00 entry once OLLAMA_API_KEY arrived.

### Changed (2026-04-28, cover letter prompt audit)
- `src/applypilot/scoring/validator.py` â€” extended `BANNED_WORDS` with: `briefly`, `minimize`, `minimizing`, `minimized`, `superpower`, `hack`, `journey`, plus hedge phrases `might address`, `may help`, `could potentially`.
- `src/applypilot/scoring/cover_letter.py` â€” added Jack Henry verb guardrail block (forbidden: built/designed/architected/engineered/automated/owned/implemented/developed; required substitutes: supported/troubleshot/queried/diagnosed/researched/documented/helped implement/investigated/resolved/escalated). Added JD-keyword-echo requirement to PARAGRAPH 2.

## [0.2.0] - 2026-02-17

### Added
- **Parallel workers for discovery/enrichment** - `applypilot run --workers N` enables
  ThreadPoolExecutor-based parallelism for Workday scraping, smart extract, and detail
  enrichment. Default is sequential (1); power users can scale up.
- **Apply utility modes** - `--gen` (generate prompt for manual debugging), `--mark-applied`,
  `--mark-failed`, `--reset-failed` flags on `applypilot apply`
- **Dry-run mode** - `applypilot apply --dry-run` fills forms without clicking Submit
- **5 new tracking columns** - `agent_id`, `last_attempted_at`, `apply_duration_ms`,
  `apply_task_id`, `verification_confidence` for better apply-stage observability
- **Manual ATS detection** - `manual_ats` list in `config/sites.yaml` skips sites with
  unsolvable CAPTCHAs (e.g. TCS iBegin)
- **Qwen3 `/no_think` optimization** - automatically saves tokens when using Qwen models
- **`config.DEFAULTS`** - centralized dict for magic numbers (`min_score`, `max_apply_attempts`,
  `poll_interval`, `apply_timeout`, `viewport`)

### Fixed
- **Config YAML not found after install** - moved `config/` into the package at
  `src/applypilot/config/` so YAML files (employers, sites, searches) ship with `pip install`
- **Search config format mismatch** - wizard wrote `searches:` key but discovery code
  expected `queries:` with tier support. Aligned wizard output and example config
- **JobSpy install isolation** - removed python-jobspy from package dependencies due to
  broken numpy==1.26.3 exact pin in jobspy metadata. Installed separately with `--no-deps`
- **Scoring batch limit** - default limit of 50 silently left jobs unscored across runs.
  Changed to no limit (scores all pending jobs in one pass)
- **Missing logging output** - added `logging.basicConfig(INFO)` so per-job progress for
  scoring, tailoring, and cover letters is visible during pipeline runs

### Changed
- **Blocked sites externalized** - moved from hardcoded sets in launcher.py to
  `config/sites.yaml` under `blocked:` key
- **Site base URLs externalized** - moved from hardcoded dict in detail.py to
  `config/sites.yaml` under `base_urls:` key
- **SSO domains externalized** - moved from hardcoded list in prompt.py to
  `config/sites.yaml` under `blocked_sso:` key
- **Prompt improvements** - screening context uses `target_role` from profile,
  salary section includes `currency_conversion_note` and dynamic hourly rate examples
- **`acquire_job()` fixed** - writes `agent_id` and `last_attempted_at` to proper columns
  instead of misusing `apply_error`
- **`profile.example.json`** - added `currency_conversion_note` and `target_role` fields

## [0.1.0] - 2026-02-17

### Added
- 6-stage pipeline: discover, enrich, score, tailor, cover letter, apply
- Multi-source job discovery: Indeed, LinkedIn, Glassdoor, ZipRecruiter, Google Jobs
- Workday employer portal support (46 preconfigured employers)
- Direct career site scraping (28 preconfigured sites)
- 3-tier job description extraction cascade (JSON-LD, CSS selectors, AI fallback)
- AI-powered job scoring (1-10 fit scale with rationale)
- Resume tailoring with factual preservation (no fabrication)
- Cover letter generation per job
- Autonomous browser-based application submission via Playwright
- Interactive setup wizard (`applypilot init`)
- Cross-platform Chrome/Chromium detection (Windows, macOS, Linux)
- Multi-provider LLM support (Gemini, OpenAI, local models via OpenAI-compatible endpoints)
- Pipeline stats and HTML results dashboard
- YAML-based configuration for employers, career sites, and search queries
- Job deduplication across sources
- Configurable score threshold filtering
- Safety limits for maximum applications per run
- Detailed application results logging
