---
id: "intent_applypilot_2026-04-26"
version: 1
status: "active"
priority: "high"
last_updated: "2026-04-26"
---
# ApplyPilot: Autonomous Job Application Pipeline

## Objective

Discover, score, tailor, and (eventually) submit remote non-phone job applications end to end without manual intervention, while keeping a dry-run policy in force until Darrin reviews the artifacts and explicitly approves real submissions.

## Outcomes (Success Criteria)

- [ ] 6-stage pipeline (discover, enrich, score, tailor, cover letter, auto-apply) operates without manual hand-offs
- [ ] AVOID filter blocks phone-heavy and 1099 roles at ingest, pack export, and apply acquire
- [ ] Tier 3 doctor passes (profile, resume, searches, LLM, Claude CLI, Chromium, Node detected)
- [ ] Application packs render valid `01_resume.pdf` plus `02_cover_letter.pdf` per job
- [ ] Real submissions occur only after Darrin reviews dry-run artifacts and authorizes
- [ ] LLM fallback chain (Kimi, NVIDIA, Cerebras, Ollama) walks tiers correctly with 3-attempt cap

## Dependencies

- Profile at `C:\Users\bibyd\.applypilot\profile.json` (preserved companies, projects, EEO defaults)
- Runtime resume at `C:\Users\bibyd\.applypilot\resume.txt`
- LLM provider keys (`OLLAMA_API_KEY`, `NVIDIA_API_KEY`, `CEREBRAS_API_KEY` or legacy `LLM_API_KEY`)
- Claude Code CLI for auto-apply browser-driving step
- Chrome / Chromium for browser automation
- Windows Task Scheduler entries: `ApplyPilot Daily Pipeline`, `ApplyPilot Indeed Scan MWF`

## Current State

As of 2026-04-24 18:32 America/Chicago: 22/23 tests passing (1 pre-existing env-leak failure unrelated to pipeline). 4,940 jobs discovered, 4,517 scored, 19 tailored, 19 cover letters generated, 15 ready in primary lane, 0 real submissions. Doctor green. Scheduled tasks Ready. AVOID filter verified at ingest, pack, and apply. Ready packs at `C:\Users\bibyd\.applypilot\application_packs\ready_primary_2026-04-21\`. GitHub origin is the public Pickle-Pixel/ApplyPilot repo (do not push or PR there).

## Risks / Edge Cases

- **Cover-letter fabrication**: Cerebras llama3.1-8b invents specific percentages even with explicit "do not invent metrics" instructions. Each pack flags hand-edit lines.
- **3-attempt fallback cap**: A 5-tier chain only gets 3 tries before failing fast.
- **NVIDIA tier dark**: `NVIDIA_API_KEY` unset on 2026-04-25, tier 2 will skip until set.
- **Local Ollama tier dark**: requires `ollama pull llama3.1:8b` and `ollama pull qwen2.5:14b` first.
- **Frontdoor WFM pack**: deprioritized due to likely bachelor-degree hard screen and iCIMS login issues.
