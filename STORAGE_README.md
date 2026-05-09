# ApplyPilot repo - storage notes

This is the canonical repo for ApplyPilot code. Runtime data lives elsewhere.

- Outputs (tailored_resumes, cover_letters, application_packs, applications_log) go to ~/.applypilot/, not the repo. Empty dirs at the repo root are vestigial and ignored via .gitignore.
- Runtime DB is ~/.applypilot/applypilot.db, not the repo's applypilot.db (orphan, 12 KB).
- Master resumes live at ~/.applypilot/resume-overhaul/MASTER_RESUME.{docx,pdf,txt}.
- Audit / decision docs about this repo go in this repo (not in ~/.applypilot/).

See C:\Users\bibyd\Projects\personal\dev\STORAGE_RULES.md for the full ruleset.
