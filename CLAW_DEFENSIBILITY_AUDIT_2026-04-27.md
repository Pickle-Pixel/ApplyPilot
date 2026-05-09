I'll run a defensibility audit of this master resume against the profile.json fact allowlist. Let me first check if there are any additional memory files that might contain relevant context about Darrin's work history.

Memory search isn't available right now. I'll proceed with the defensibility audit using only the provided resume and profile.json fact allowlist, which is the correct scope anyway.

---

# RESUME DEFENSIBILITY AUDIT

**Scope:** MASTER_RESUME.txt vs profile.json `resume_facts.real_metrics`, `preserved_companies`, and `skills_boundary`

---

## STRONG CLAIMS (Defensible)

These bullets are directly backed by `real_metrics` entries. No reference call would find a discrepancy.

| Resume Line | Fact Source | Verdict |
|---|---|---|
| **"Five years at Jack Henry & Associates supporting ATM/POS systems for 40+ bank clients"** | Real metric #4: "Five years at Jack Henry & Associates supporting ATM/POS systems for 40+ bank clients" | STRONG |
| **"TEKsystems contract placement Sep 2014 - Oct 2016, converted to direct hire Oct 2016 - Sep 2019"** | Real metric #5: "TEKsystems contract placement...converted to direct hire" | STRONG |
| **"Started as Customer Service Representative; promoted to Real-Time Analyst, then to Workforce Scheduler...within approximately 60 days of hire"** | Real metric #3: "Promoted from CSR to Real-Time Analyst to Workforce Scheduler within approximately 60 days at NCO Group" | STRONG |
| **"Four years owning Aspect eWFM forecasting, scheduling, and real-time adherence across 3 call center locations"** | Real metric #2: "Four years owning Aspect eWFM forecasting, scheduling, and real-time adherence across 3 call center locations" | STRONG |
| **"Approximately one year of QA Analyst work at Teleperformance auditing calls and documentation for HIPAA privacy expectations and carrier compliance"** | Real metric #6: "Approximately one year of QA Analyst work...auditing calls and documentation for HIPAA privacy expectations and carrier compliance" | STRONG |
| **"Coordinated Medicare enrollment and prior-authorization documentation at GoHealth; reduced processing errors through standardized documentation protocols"** | Real metric #7: "Coordinated Medicare enrollment...reduced processing errors through standardized documentation protocols" | STRONG |
| **"Managed predictive dialer operations at Aegis Communications; pacing adjustments contributed to +20% connect rate improvement"** | Real metric #8: "Managed predictive dialer operations...pacing adjustments contributed to +20% connect rate improvement" | STRONG |
| **"Seasonal regulated-program support across...ResultsCX, TTEC:...Medicare AEP, Progressive P&C"** | Real metric #9: "Seasonal regulated-program support across...ResultsCX, TTEC:...Medicare AEP, Progressive P&C" | STRONG |
| **"Five years at NCO Group across customer service, real-time analysis, and workforce scheduling"** | Real metric #1: "Five years at NCO Group across customer service, real-time analysis, and workforce scheduling" | STRONG |
| **All Technical Skills entries** — Aspect eWFM, Avaya CMS Supervisor, SQL, Advanced Excel, pivot tables, VLOOKUP, XLOOKUP, KPI reporting, variance analysis, call auditing, documentation review, HIPAA, CMS Marketplace, carrier compliance, coaching feedback, Microsoft Teams, Zoom, email/chat/live messaging, correspondence, back-office processing, data entry, order processing, records management | All appear in `skills_boundary.tools` or `skills_boundary.databases` | STRONG |

---

## SOFT CLAIMS (Could Be Questioned)

These are defensible in an interview but lack documentary proof. A skeptical recruiter could probe but not disprove them.

| # | Claim | Risk |
|---|---|---|
| **S1** | "Reduced processing errors through standardized documentation protocols, checklist discipline, and quality-control checkpoints" (GoHealth) | Directionally true per real metric #7 ("reduced processing errors"), but "checklist discipline" and "quality-control checkpoints" are process labels, not verified deliverables. |
| **S2** | "Maintained accurate customer, plan, product, and issue documentation across peak-volume programs; documented recurring support patterns and escalation handoffs" (Seasonal contractor role) | Plausible given the work, but "documented recurring support patterns" implies creating reference material. Could be true. Could also be overstated if the documentation was informal. |
| **S3** | "Built standardized trace-file diagnostics and knowledge-base guidance that reduced repeat escalations and shortened investigation cycles" (Jack Henry) | The "built" verb is soft here — "standardized trace-file diagnostics" could mean templates, checklists, or written procedures. Not custom engineering. But "built" is still a loaded word. |
| **S4** | "Built forecast and shrinkage models that informed staffing requirements and performance projections" (NCO Group) | "Built" again. Forecast models in eWFM are often configured, not coded. True if it means "set up." Misleading if it implies custom development. |
| **S5** | "Prepared root-cause analysis on forecast variance and recommended process changes adopted by operations leaders" (NCO Group) | "Adopted by operations leaders" is an adoption claim. Could be verified by a reference. Soft because it's secondhand. |
| **S6** | "Adjusted dialer pacing contributed to a +20% connect rate improvement" (Aegis) | The +20% is directly from real metrics. But "adjusted dialer pacing" is vague. Was this a knob turn or a methodology change? |
| **S7** | "Helped implement EMV chip card conversions across the ATM portfolio" (Jack Henry) | "Helped implement" is soft by design, which is good. But what exactly was the help? Research? Coordination? Documentation? Fine for a resume, but a reference might say "he was on the support queue when EMV tickets came in." |
| **S8** | "Used SQL to query transaction logs, validate incidents, correlate failures, and identify root causes for recurring ATM/POS issues" (Jack Henry) | SQL is in skills. But "identify root causes" is a strong competency claim. Defensible if true, but harder to prove than "ran queries." |

---

## INDEFENSIBLE CLAIMS (Would Fail a Reference Call)

These are claims a former employer could directly contradict. Each needs correction or removal.

| # | Claim | Why It's Indefensible | Proposed Correction |
|---|---|---|---|
| **I1** | **"Built standardized trace-file diagnostics and knowledge-base guidance"** (Jack Henry) | User instruction: **"Darrin says he didn't build anything custom at Jack Henry."** The verb "built" implies creation of tools or systems. A reference call to Jack Henry would likely describe this as documentation, templates, or troubleshooting notes — not "built." | **REWRITE:** "Developed trace-file diagnostic guides and knowledge-base articles that reduced repeat escalations and shortened investigation cycles." "Developed" for documentation is safer than "built." |
| **I2** | **"Built forecast and shrinkage models"** (NCO Group) | Same pattern. "Built" suggests construction of models. In a WFM context, this usually means configuring parameters in Aspect eWFM, not building models from scratch. A reference might say "he ran the system." | **REWRITE:** "Configured forecast and shrinkage parameters in Aspect eWFM to inform staffing requirements and performance projections." Or **REMOVE** and merge into the eWFM bullet. |
| **I3** | **"Technical Support Engineer II"** (Jack Henry) — title in the header and experience block | The `skills_boundary` does not list engineering frameworks, devops, or development tools. Darrin's work at Jack Henry was ATM/POS support with SQL queries and documentation. "Engineer II" is a real title if it was on his business card, but it sets an expectation of engineering depth that the fact allowlist doesn't support. | **KEEP the title** if it was official, but **qualify the role description** to emphasize support and diagnostics, not engineering. No change needed to the title itself unless Darrin confirms it was officially "Technical Support Specialist" or similar. **Risk: MEDIUM** — a reference call would confirm the title if accurate. |
| **I4** | **"Technical operations: SQL-based incident analysis, ATM/POS support, banking deployments, trace-file diagnostics, and escalation ownership"** (Professional Highlights) | "Banking deployments" is not in any real metric. Darrin's real metrics describe *supporting* ATM/POS systems, not *deploying* them. A reference would likely say he handled tickets and troubleshooting, not deployments. | **REWRITE:** "Technical operations: SQL-based incident analysis, ATM/POS support, trace-file diagnostics, and escalation ownership." **Remove "banking deployments."** |
| **I5** | **"Regulated environments: HIPAA, CMS Marketplace, Medicare/ACA workflows, carrier documentation, and multi-state licensing exposure"** (Professional Highlights) | "Multi-state licensing exposure" is vague. Darrin held a license in at least one state. "Multi-state" implies breadth not confirmed in the facts. | **REWRITE:** "Regulated environments: HIPAA, CMS Marketplace, Medicare/ACA workflows, carrier documentation, and licensed insurance program exposure." |
| **I6** | **"Handled regulated healthcare enrollment documentation, CRM documentation, HIPAA/CMS-sensitive workflows, and carrier/process compliance"** (Seasonal contractor) | "CRM documentation" — which CRM? Not specified in facts. Could be Salesforce, could be proprietary. The claim is too specific without backing. | **REWRITE:** "Handled regulated healthcare enrollment documentation, HIPAA/CMS-sensitive workflows, and carrier/process compliance." **Remove "CRM documentation."** |
| **I7** | **"Produced enrollment documentation aligned to CMS Marketplace and carrier compliance requirements"** (TTEC, UHC) | "Produced" implies creation of original documentation. In a licensed agent role, this usually means filling out forms correctly. A reference might say "he processed applications." | **REWRITE:** "Processed enrollment documentation aligned to CMS Marketplace and carrier compliance requirements." "Processed" is more accurate for form completion. |

---

## RECOMMENDED ACTIONS

### Immediate Removals

| Item | Action |
|---|---|
| **"banking deployments"** from Professional Highlights | Remove. Not in fact allowlist. |
| **"CRM documentation"** from seasonal contractor bullet | Remove. Tool not specified in facts. |
| **"multi-state licensing exposure"** from Professional Highlights | Remove or replace with "licensed insurance program exposure." |

### Immediate Rewrites

| Item | From | To |
|---|---|---|
| **I1** | "Built standardized trace-file diagnostics" | "Developed trace-file diagnostic guides and knowledge-base articles" |
| **I2** | "Built forecast and shrinkage models" | "Configured forecast and shrinkage parameters in Aspect eWFM" |
| **I7** | "Produced enrollment documentation" | "Processed enrollment documentation" |

### Soft Claims to Monitor

These don't need changes now, but Darrin should be prepared to explain them in an interview:

- **S3/S4** — "Built" in the Jack Henry and NCO bullets. If any reference call happens, the wording may be challenged. Consider preemptively softening to "developed," "created," or "maintained."
- **S5** — "Adopted by operations leaders" — have a specific example ready.
- **S7** — "Helped implement EMV chip card conversions" — be ready to describe exactly what "helped" meant.

### Structural Observation

The resume's **Summary** and **Professional Highlights** are soft by design. They are not fact-checked claims; they are framing. This is acceptable as long as no specific metric in them exceeds the fact allowlist. Currently, the Summary is clean. The Highlights have two problems noted above (I4, I5).

### Title Integrity

"Technical Support Engineer II" at Jack Henry is defensible **only if that was the actual job title on record**. If the official title was "Technical Support Specialist II" or similar, the resume should match the record exactly. A discrepancy here is a red flag in background checks.

---

## AUDIT SUMMARY

| Category | Count |
|---|---|
| **Strong** | 10+ bullets + all skills |
| **Soft** | 8 claims |
| **Indefensible** | 7 claims (2 require rewrite, 5 require removal or softening) |

**Bottom line:** The resume is mostly fact-grounded. The main liability is the verb "built" appearing twice in contexts where Darrin has explicitly said he did not build custom tools. Those two bullets are the highest-risk items. The "banking deployments" and "multi-state licensing exposure" claims are secondary risks — not likely to sink an application, but reference-call bait.

**Recommended priority:** Fix I1 and I2 first. They are the only claims a reference call would actively contradict based on Darrin's own statements.