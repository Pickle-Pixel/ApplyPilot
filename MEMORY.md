/Users/matriarch/.openclaw/completions/openclaw.zsh:3570: command not found: compdef
# ApplyPilot – Memory Log

## Current Status
- Gustaf preset (`profiles/gustaf/`) contains profile.json, resume.txt, searches.yaml, `.env`.
- Human-in-loop safety: `APPLYPILOT_REQUIRE_APPROVAL=1` enforced in CLI (approval prompts + sequential apply loop).
- `scripts/bootstrap_profile.py` copies preset files into `~/.applypilot/` (respects `--force`).
- `.env.example` documents safety toggles (`APPLYPILOT_REQUIRE_APPROVAL`, `APPLYPILOT_APPROVAL_WINDOW`).
- 2026-03-11 smoke-run (`discover -> enrich -> score`) populated the SQLite DB with 662 jobs; discover succeeded partially (ZipRecruiter geo-blocked) and scoring currently returns 0 because the default `GEMINI_API_KEY=PUT_YOUR_KEY_HERE` value causes 400 errors.

## Pending / Nice-to-haves
- Add automated tests for the approval workflow (mock DB + CLI confirm).
- Expand searches.yaml with curated employers (Workday list) for Swedish fintech/pension orgs.
- Capture resume facts in a structured YAML for easier updates.
- Hook AGENT logs into Obsidian vault once available.
- Provide a real `GEMINI_API_KEY` (or OpenAI/local LLM config) so tailor/cover stages can run.
- Install/configure Claude Code CLI to unlock Tier-3 apply loop.

## Recently Completed
- Repository cloned locally and fully read (CLI, pipeline, database, apply launcher, wizard).
- Added HITL enforcement path to `applypilot apply` + README documentation.
- Authored Gustaf-specific resume/profile/search config + bootstrap script.
- Created AGENTS.md & MEMORY.md for future collaborators.
