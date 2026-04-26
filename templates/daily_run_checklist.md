# Daily ApplyPilot Run Checklist

The standard daily sequence, formalized from `START_HERE.html`.

## Commands

```
cd C:\Users\bibyd\Projects\dev\ApplyPilot
.venv\Scripts\applypilot.exe run discover enrich score
.venv\Scripts\applypilot.exe pack --min-score 8 --limit 30 --lane primary
.venv\Scripts\applypilot.exe apply --dry-run
```

## Rules
- Always read each pack's `NEXT_STEPS.md` before any real (non-dry-run) submission.
- Run `pack_quality_gate.md` against each pack before flipping off `--dry-run`.
- If discover or enrich crash, do NOT proceed to score. Fix the upstream first.

## On unexpected output
- Score distribution wildly different from baseline? Pause. Compare against last week's run.
- Any pack with a fabricated metric or invented employer? Stop the whole batch and triage `cover_letter.py` / `tailor.py`.
