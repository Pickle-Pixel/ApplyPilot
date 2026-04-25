# ApplyPilot Application Cycle Report

**Date:** 2026-04-25
**Operator:** Codex
**Primary lane:** document, records, QA, healthcare operations, WFM, and back-office processing

## What Changed In The Local Application System

- Cleaned `C:\Users\bibyd\.applypilot\resume-overhaul` so it contains only the send-ready resume files:
  - `Resume_Document_QA_Ops.docx/.pdf/.txt`
  - `Resume_QA_Analyst.docx/.pdf/.txt`
  - `Resume_Operations_TechSupport.docx/.pdf/.txt`
  - `Resume_WFM_Analyst.docx/.pdf/.txt`
- Moved advisory audit markdown files out of `resume-overhaul` into `C:\Users\bibyd\.applypilot\resume-audit-reports`.
- Created/updated the local application-cycle summaries:
  - `C:\Users\bibyd\.applypilot\application_packs\APPLICATION_CYCLE_2026-04-25.md`
  - `C:\Users\bibyd\.applypilot\job_application_ready_2026-04-24\APPLICATION_CYCLE_2026-04-25.md`
- Moved blocked/manual packs out of the active ready folder into:
  - `C:\Users\bibyd\.applypilot\application_packs\manual_or_blocked_2026-04-25`
  - `C:\Users\bibyd\.applypilot\job_application_ready_2026-04-24\02_ready_application_packs\manual_or_blocked_2026-04-25`
- Left `ready_primary_2026-04-24` with only the Sharecare ROI Medical Records Specialist draft to resume after Workday maintenance.

## Live Submission Results

Submitted successfully:

- Lingraphica, Funding Verification Specialist - Remote, through SmartRecruiters.
- Tenavi Staffing, Document Review Specialist, through Crelate.

Saved evidence:

- `C:\Users\bibyd\.applypilot\_archive\20260425-queue-refresh\lingraphica_live_after_submit.png`
- `C:\Users\bibyd\.applypilot\_archive\20260425-queue-refresh\tenavi_live_after_submit.png`

## Draft Or Blocked Applications

- Sharecare, ROI Medical Records Specialist - Remote: Workday account/draft created and visible in Candidate Home as `Not Submitted`. Completion is blocked by Workday planned maintenance from 1:00 AM to 4:00 AM CDT on 2026-04-25.
- Navitus, Analyst, Claims Quality: active iCIMS posting, blocked by invisible hCaptcha at email/login.
- ComPsych, Forecasting Analyst: active iCIMS posting, blocked by invisible hCaptcha at email/login.
- Wink/Snaphunt, Data Processing Specialist: account/sign-in works and resume upload is accepted, but final profile requires a listed city and rejects truthful `0` current salary. This needs user-approved handling, not guessing.
- Mercer Advisors, Compliance Operations Analyst: LinkedIn-gated and weaker fit because it asks for investment-adviser compliance background and a bachelor requirement.

## Code Changes Made To Support The Cycle

- `src/applypilot/config.py`
  - Reads `profile.json` using `utf-8-sig` so a PowerShell-written BOM does not break live apply runs.
  - Treats `C:\Users\bibyd\.applypilot\.env` as the source of truth for provider selection and avoids falling back to repo `.env` when the ApplyPilot-local env exists.
- `src/applypilot/apply/prompt.py`
  - Replaced broad software-engineer tool-claim guidance with role-accurate document/QA/ops guidance to avoid overclaiming specialized tools or credentials.
- `src/applypilot/llm.py`
  - Allows per-stage `:cloud` model names to fall back to the configured default client when `OLLAMA_API_KEY` is unavailable, instead of crashing tailoring immediately.
- Added local portal scripts under `C:\Users\bibyd\.applypilot\_archive\20260425-queue-refresh` for Sharecare Workday, SmartRecruiters, Crelate, and Snaphunt attempts. These scripts are operational evidence and local workbench artifacts, not committed repo source.

## Verification

- `applypilot doctor`: passed; CapSolver remains optional and unset.
- Resume TXT scan: passed for all four variants; no placeholders, both emails present, gap entry present, Six Sigma is in-progress only.
- Resume PDF text scan: passed for all four variants; no placeholders, both emails present, no Yellow Belt certified wording.
- Focused config/chrome tests: passed.
- Focused score-8+ tailoring retry: provider routing repair worked and produced several tailored PDFs, but the provider rate-limited before the batch could become a clean ready-pack update.

## Next Operational Step

Resume the Sharecare Workday draft after 4:00 AM CDT on 2026-04-25. The active ready folder intentionally contains only that post-maintenance task.
