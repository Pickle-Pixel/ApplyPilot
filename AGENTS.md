# ApplyPilot – Agent Guide

## Mission
- Maintain a human-in-the-loop job application agent tailored to **Gustaf Garnow**.
- Keep the Gustaf preset (`profiles/gustaf/`) current: profile.json, resume.txt, searches.yaml, `.env`.
- Ensure every outbound application requires explicit approval (no autonomous submissions by default).
- Document operational changes (this file + MEMORY.md) and keep AGPL compliance intact.

## Guardrails
- **Never** push to `main` – create a feature branch per change.
- **Do not** send outbound emails/messages nor apply to jobs without the user's thumbs-up.
- If unsure whether a credential/endpoint exists, stop and ask instead of guessing.
- Respect `APPLYPILOT_REQUIRE_APPROVAL`: do not remove/disable it unless the user explicitly requests.

## Environment Setup
1. `python3 -m venv venv && source venv/bin/activate` (if not already active).
2. `pip install -e .` plus `pip install --no-deps python-jobspy && pip install pydantic tls-client requests markdownify regex`.
3. Copy Gustaf's preset: `python scripts/bootstrap_profile.py gustaf --force` (updates `~/.applypilot`).
4. Run `applypilot doctor` to confirm Tier-2+ readiness (Gemini key, resume, searches, `.env`).
5. LLM defaults: `GEMINI_API_KEY`, `LLM_MODEL=gemini-2.0-flash`, `APPLYPILOT_REQUIRE_APPROVAL=1`.

## Workflow Cheatsheet
- Discovery/tailoring: `applypilot run --stream --workers 4 --validation normal`.
- Manual review: inspect `applypilot dashboard` or query SQLite (`~/.applypilot/applypilot.db`).
- Human approval apply mode (default):
  ```bash
  applypilot apply --limit 3 --min-score 8
  ```
  CLI presents each job; only approved ones trigger Chrome/Claude.
- Dry-run apply (fills forms, no submit): `applypilot apply --dry-run --limit 1`.
- Mark manually handled jobs: `applypilot apply --mark-applied URL` or `--mark-failed URL --fail-reason "notes"`.

## Files to Touch Frequently
- `profiles/gustaf/profile.json` – skills, compensation, dealbreakers.
- `profiles/gustaf/resume.txt` – base resume (keep SUMMARY/TECHNICAL SKILLS/EXPERIENCE sections for validator).
- `profiles/gustaf/searches.yaml` – queries and negative filters for Stockholm/remote-EU roles.
- `scripts/bootstrap_profile.py` – helper to sync preset into `~/.applypilot`.
- `AGENTS.md` / `MEMORY.md` – document ops decisions.

## Reporting
- Summaries go to the user in Discord (#general) plus a short operational note for #ops when instructed.
- Include: what changed, commands used, outstanding blockers, next steps.
