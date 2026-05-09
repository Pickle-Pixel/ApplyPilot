# LANE_CONSOLIDATION — 2026-04-28

**Dispatch:** `lane-consolidation-v2`
**Driver:** Council top correction #2 (`council/COUNCIL_VERDICT_2026-04-28.md`)
**Action:** Collapse 4-lane positioning to ONE lane: **Licensed Insurance Compliance & QA Specialist**
**Outreach:** None. Disk-only. No DMs, no emails, no LinkedIn pushes.

---

## Step 1 — 3 Candidate Headlines

All 3 lead with the lane phrase. None contain em dashes, "passionate", "results-driven", "superpower", "hack", "journey", or "solutions".

- **A (PICK):** `Licensed Insurance Compliance & QA Specialist | Medicare Member Documentation | Remote Non-Phone` (97 chars)
- **B:** `Licensed Insurance Compliance & QA Specialist | HIPAA, CMS Marketplace, Carrier Audit | Remote` (94 chars)
- **C:** `Licensed Insurance Compliance & QA Specialist | Claims Quality Review & Member Records | 10+ yrs` (97 chars)

**Pick = A.** Reason: "Medicare Member Documentation" is the council-named differentiating, defensible niche; "Remote Non-Phone" hits Darrin's hard rule on the same line.

Alternates: B has densest ATS keyword surface (HIPAA + CMS + audit); C anchors years-of-experience for ATS systems that weight tenure.

## Step 2 — Consolidated Summary (verbatim, applied to all 5 resumes)

> Licensed Insurance Compliance & QA Specialist with 10+ years across regulated insurance documentation, contact-center QA, and back-office operations. Active Health Insurance Agent license and recent Medicare AEP/OEP enrollment-documentation experience at GoHealth, ResultsCX, and TTEC across multi-state CMS Marketplace workflows. Prior QA Analyst work at Teleperformance auditing agent calls and documentation for HIPAA privacy expectations and carrier compliance across UHC, Humana, Aetna, BCBS, and Anthem programs. Targeting remote, non-phone roles in insurance compliance, member documentation, claims processing, and Medicare/healthcare QA review.

4 sentences. Cross-checked against `profile.json.real_metrics` — every concrete claim maps to a preserved entry (10+ yrs, GoHealth/ResultsCX/TTEC Medicare, multi-state CMS Marketplace, Teleperformance QA Analyst auditing for HIPAA + carrier compliance). No fab metrics. No Jack Henry "built/designed/architected/engineered/automated/owned/implemented/developed" verbs (JH paragraph wasn't touched anyway). No em dashes.

## Step 3 — profile.json target_role Diff

**Old (string, 9 entries):**
```
"Chat Support Specialist, Email Support Specialist, Live Messaging Agent, Back Office Specialist,
 Correspondence Specialist, Non-Voice Customer Service, Utilization Review Specialist,
 Referrals Coordinator, Health Information Specialist"
```

**New (array, 7 entries):**
```json
"target_role": [
  "Licensed Insurance Compliance Specialist",
  "Insurance Compliance & QA Specialist",
  "Insurance Verification Specialist",
  "Claims Processor",
  "Medicare Compliance Auditor",
  "Member Documentation Specialist",
  "QA Review Analyst"
]
```

Schema change: `string` → `array`. Counts: 9 → 7. Every other key in `profile.json` preserved (personal, work_authorization, availability, compensation, skills_boundary, resume_facts, eeo_voluntary, salary band 40-60K, Joplin/Missouri, immediate-start, no-sponsorship). Backed up to `profile.json.bak-2026-04-28-lane-v2`.

## Step 4 — Files Updated

**16 files modified** (5 resumes × 3 formats = 15 + profile.json). **16 backups** created with `.bak-2026-04-28-lane-v2` suffix.

| File (path under `C:\Users\bibyd\.applypilot\`) | Headline | Summary | .docx | .pdf |
|---|---|---|---|---|
| `resume-overhaul\MASTER_RESUME.txt` | NEW | NEW | NEW (55 KB) | NEW (55 KB) |
| `resume-overhaul\Resume_WFM_Analyst.txt` | NEW | NEW | NEW | NEW (55 KB) |
| `resume-overhaul\Resume_QA_Analyst.txt` | NEW | NEW | NEW | NEW (54 KB) |
| `resume-overhaul\Resume_Operations_TechSupport.txt` | NEW | NEW | NEW | NEW (54 KB) |
| `resume-overhaul\Resume_Document_QA_Ops.txt` | NEW | NEW | NEW | NEW (55 KB) |
| `profile.json` | n/a | n/a — `target_role` swapped | n/a | n/a |
| `resume.txt` (runtime) | regenerated | regenerated | n/a | n/a |

PDFs regenerated via existing `scripts/rebuild_overhaul_pdfs.py` (Playwright converter). DOCX edits surgical (replaced text in headline paragraph + summary paragraph; preserved first-run formatting; removed orphan runs). Variant-specific TECHNICAL SKILLS and bullet content untouched.

### Variant Conflicts Surfaced (NOT auto-restructured per dispatch rule)

- **`Resume_Operations_TechSupport.txt/.docx/.pdf`** — TECHNICAL SKILLS still leads with `Technical Support: Tier 2/3 support, ticket triage, incident management, problem management, knowledge-base authoring, ATM/POS systems`. This conflicts with the new "Licensed Insurance Compliance" lane per council correction #5 (drop JH + tech-support pitch from the $60K+ tier). Recommendation: either retire this variant entirely (council majority) or restructure its TECHNICAL SKILLS section to lead with compliance/QA. **Did not touch — surface for Darrin's call.**
- **`Resume_WFM_Analyst.txt/.docx/.pdf`** — TECHNICAL SKILLS leads with `Workforce Management: Aspect eWFM, Avaya CMS Supervisor`. Softer mismatch (WFM doesn't fight the new lane the way Tech Support does). Variant still useful for hybrid WFM/Compliance roles like the Capital Rx WFM Analyst Customer Care lead from tonight's sourcing. **Kept.**
- **`Resume_QA_Analyst.txt/.docx/.pdf`** — TECHNICAL SKILLS leads with `Quality and Compliance: quality assurance, QA scorecards, call auditing, HIPAA, CMS Marketplace, carrier compliance`. Aligns cleanly with new lane.
- **`Resume_Document_QA_Ops.txt/.docx/.pdf`** — TECHNICAL SKILLS leads with `Document and Quality Operations: documentation review, medical records, records management, healthcare documentation review, claims quality, audit documentation`. Aligns cleanly with new lane.

## Step 5 — LinkedIn Proposal

Saved: `Projects\dev\ApplyPilot\drafts\linkedin_lane_consolidation_2026-04-28.md`

Contents: proposed Headline (matches resume), proposed About paragraph (matches resume SUMMARY), 2 alternate headlines, real-metrics cross-check table, manual operator steps. **Not pushed.**

## Step 6 — regen_resume.py Output Verification

```
WROTE C:\Users\bibyd\.applypilot\resume.txt (6484 bytes, 59 lines)
```

Verified by `Read` on lines 1-15 of the regenerated `resume.txt`:
- Line 4: `Licensed Insurance Compliance & QA Specialist | Medicare Member Documentation | Remote Non-Phone` ✓
- Lines 9-10: full new SUMMARY paragraph present ✓
- Lines 11+: PROFESSIONAL HIGHLIGHTS unchanged (still uses pre-existing master highlights) ✓

## Things That Need Darrin's Hands

1. **LinkedIn upload of new `MASTER_RESUME.pdf`** — replace the parked `Resume_Operations_TechSupport.pdf` in your LinkedIn Featured / Resume slot. (Council corrections #2 + #4 both depend on this.)
2. **LinkedIn Headline + About paragraph paste** — see `drafts\linkedin_lane_consolidation_2026-04-28.md`. Both auto-produced; you click paste.
3. **Headline pick decision** — A is my pick (Medicare Member Documentation + Remote Non-Phone). If you want B (densest ATS keywords) or C (years-of-experience anchor), say so and I'll regenerate the 5 resumes + LinkedIn draft.
4. **`Resume_Operations_TechSupport.pdf` retirement decision** — surfaced as variant conflict. Council majority says retire this lane entirely. I did **not** auto-decide; awaiting your call.
5. **LinkedIn Job Preferences titles** — optional: mirror the new 7-entry `target_role` list under LinkedIn > Open To Work > Job Titles so recruiter inbound matches the lane.

## Hard Rules Enforced

- No outreach. No DMs. No emails. No LinkedIn pushes. Per Darrin's standing rule against spam/blacklist exposure.
- No fabrication — every metric / claim cross-checked against `profile.json.real_metrics`.
- No forbidden Jack Henry verbs (built/designed/architected/engineered/automated/owned/implemented/developed). No new content was written for the JH paragraph; it was preserved verbatim.
- No em dashes, "briefly", "passionate", "results-driven", "superpower", "hack", "journey".
- Backups before edit with `.bak-2026-04-28-lane-v2` suffix on all 16 touched files (5 resumes × 3 formats + profile.json).
- No subagents.
- Opus 4.7 reasoning per standing rule.

---

## TL;DR

- One lane everywhere now: Licensed Insurance Compliance & QA Specialist | Medicare Member Documentation | Remote Non-Phone.
- 5 resumes (txt+docx+pdf) + profile.json updated; 16 backups with `.bak-2026-04-28-lane-v2`; runtime `resume.txt` regenerated.
- LinkedIn draft saved to disk; **not pushed** — Darrin pastes manually.
- Operations_TechSupport variant has a positioning conflict — surfaced, not restructured.
