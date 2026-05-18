<!-- logo here -->

> **⚠️ ApplyPilot** is the original open-source project, created by [Pickle-Pixel](https://github.com/Pickle-Pixel) and first published on GitHub on **February 17, 2026**. We are **not affiliated** with applypilot.app, useapplypilot.com, or any other product using the "ApplyPilot" name. These sites are **not associated with this project** and may misrepresent what they offer. If you're looking for the autonomous, open-source job application agent — you're in the right place.

# ApplyPilot — Warehousing & Logistics Edition (Greater Vancouver)

**Autonomous warehouse job applications across Metro Vancouver. Open source.**

This fork is tuned for warehouse, order-picker, packager, and material-handler
roles in the Greater Vancouver Regional District. The discovery, scoring, and
auto-apply pipeline targets:

- **Roles:** Warehouse Associate, Order Picker, Warehouse Worker, Order
  Selector, Warehouse Operator, Warehouse Assistant, Warehouse Person,
  Warehouse Stocker, Packager (plus wider-net terms like Shipper-Receiver,
  Material Handler, Forklift Operator, Fulfillment Associate).
- **Locations:** Vancouver, Burnaby, Richmond, Surrey, Delta, Langley,
  Coquitlam, Port Coquitlam, New Westminster, North Vancouver, Port Moody,
  Pitt Meadows, Maple Ridge, White Rock, Tsawwassen.
- **Boards & employers:** Job Bank Canada, WorkBC, Indeed CA, LinkedIn,
  ZipRecruiter, Eluta, Talent.com, CareerJet CA, Jooble CA, Randstad,
  Adecco, Aerotek, Drake, plus direct career sites for Amazon (YVR4/YVR7),
  Loblaw, Save-On-Foods/Pattison, Walmart Canada, Sysco, Costco, IKEA,
  Home Depot Canada, UPS, Purolator, FedEx, Gordon Food Service, DHL,
  18 Wheels, Stryder, Radius, Day & Ross, and more.

[![PyPI version](https://img.shields.io/pypi/v/applypilot?color=blue)](https://pypi.org/project/applypilot/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Pickle-Pixel/ApplyPilot?style=social)](https://github.com/Pickle-Pixel/ApplyPilot)
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/S6S01UL5IO)




https://github.com/user-attachments/assets/7ee3417f-43d4-4245-9952-35df1e77f2df


---

## What It Does

ApplyPilot is a 6-stage autonomous job application pipeline. It discovers warehouse jobs across Canadian boards and direct employer career sites, scores them against your resume with AI, tailors your resume per job, writes cover letters, and **submits applications for you**. It navigates forms, uploads documents, answers screening questions, all hands-free.

Three commands. That's it.

```bash
pip install applypilot
pip install --no-deps python-jobspy && pip install pydantic tls-client requests markdownify regex
applypilot init          # one-time setup: resume, profile, preferences, API keys
applypilot doctor        # verify your setup — shows what's installed and what's missing
applypilot run           # discover > enrich > score > tailor > cover letters
applypilot run -w 4      # same but parallel (4 threads for discovery/enrichment)
applypilot apply         # autonomous browser-driven submission
applypilot apply -w 3    # parallel apply (3 Chrome instances)
applypilot apply --dry-run  # fill forms without submitting
```

> **Why two install commands?** `python-jobspy` pins an exact numpy version in its metadata that conflicts with pip's resolver, but works fine at runtime with any modern numpy. The `--no-deps` flag bypasses the resolver; the second command installs jobspy's actual runtime dependencies. Everything except `python-jobspy` installs normally.

---

## Two Paths

### Full Pipeline (recommended)
**Requires:** Python 3.11+, Node.js (for npx), Gemini API key (free), Claude Code CLI, Chrome

Runs all 6 stages, from job discovery to autonomous application submission. This is the full power of ApplyPilot.

### Discovery + Tailoring Only
**Requires:** Python 3.11+, Gemini API key (free)

Runs stages 1-5: discovers jobs, scores them, tailors your resume, generates cover letters. You submit applications manually with the AI-prepared materials.

---

## The Pipeline

| Stage | What Happens |
|-------|-------------|
| **1. Discover** | Scrapes 4 job boards (Indeed, LinkedIn, ZipRecruiter, Google Jobs) + the Canadian warehouse employer registry on Workday + ~20 direct career sites (Amazon, Costco, IKEA, UPS, Purolator, Gordon Food Service, etc.) + staffing agencies (Randstad, Adecco, Aerotek, Drake) |
| **2. Enrich** | Fetches full job descriptions via JSON-LD, CSS selectors, or AI-powered extraction |
| **3. Score** | AI rates every job 1-10 based on your resume, equipment certifications (forklift, WHMIS), and shift availability. Only high-fit jobs proceed |
| **4. Tailor** | AI rewrites your resume per job: reorganizes, emphasizes relevant warehouse/material-handling experience, adds keywords. Never fabricates certifications |
| **5. Cover Letter** | AI generates a targeted cover letter per job |
| **6. Auto-Apply** | Claude Code navigates application forms, fills fields, uploads documents, answers questions, and submits |

Each stage is independent. Run them all or pick what you need.

---

## ApplyPilot vs The Alternatives

| Feature | ApplyPilot (Warehouse / Vancouver) | AIHawk | Manual |
|---------|-----------|--------|--------|
| Job discovery | 4 boards + Canadian warehouse Workday tenants + ~20 direct sites + 4 staffing agencies | LinkedIn only | One board at a time |
| AI scoring | 1-10 fit score per job, weighted for warehouse certs & shift availability | Basic filtering | Your gut feeling |
| Resume tailoring | Per-job AI rewrite, never fabricates certifications | Template-based | Hours per application |
| Auto-apply | Full form navigation + submission | LinkedIn Easy Apply only | Click, type, repeat |
| Supported sites | Indeed CA, LinkedIn, ZipRecruiter, Google Jobs, Job Bank, WorkBC, Eluta, Talent.com, CareerJet, Jooble + Amazon, Loblaw, Save-On-Foods, Walmart, Sysco, Costco, IKEA, Home Depot, UPS, Purolator, FedEx, GFS, DHL, 18 Wheels, Day & Ross, Stryder, Radius | LinkedIn | Whatever you open |
| License | AGPL-3.0 | MIT | N/A |

---

## Requirements

| Component | Required For | Details |
|-----------|-------------|---------|
| Python 3.11+ | Everything | Core runtime |
| Node.js 18+ | Auto-apply | Needed for `npx` to run Playwright MCP server |
| Gemini API key | Scoring, tailoring, cover letters | Free tier (15 RPM / 1M tokens/day) is enough |
| Chrome/Chromium | Auto-apply | Auto-detected on most systems |
| Claude Code CLI | Auto-apply | Install from [claude.ai/code](https://claude.ai/code) |

**Gemini API key is free.** Get one at [aistudio.google.com](https://aistudio.google.com). OpenAI and local models (Ollama/llama.cpp) are also supported.

### Optional

| Component | What It Does |
|-----------|-------------|
| CapSolver API key | Solves CAPTCHAs during auto-apply (hCaptcha, reCAPTCHA, Turnstile, FunCaptcha). Without it, CAPTCHA-blocked applications just fail gracefully |

> **Note:** python-jobspy is installed separately with `--no-deps` because it pins an exact numpy version in its metadata that conflicts with pip's resolver. It works fine with modern numpy at runtime.

---

## Configuration

All generated by `applypilot init`:

### `profile.json`
Your personal data in one structured file: contact info, work authorization, compensation, experience, skills, resume facts (preserved during tailoring), and EEO defaults. Powers scoring, tailoring, and form auto-fill.

### `searches.yaml`
Job search queries, target titles, locations, boards. Run multiple searches with different parameters.

### `.env`
API keys and runtime config: `GEMINI_API_KEY`, `LLM_MODEL`, `CAPSOLVER_API_KEY` (optional).

### Package configs (shipped with ApplyPilot)
- `config/employers.yaml` - Workday employer registry — Canadian warehouse / logistics / grocery distribution (Loblaw DCs, Save-On-Foods/Pattison, Walmart Canada, Sysco, Home Depot Canada, FedEx Canada, McKesson, DHL, Day & Ross, Magna, Canadian Tire)
- `config/sites.yaml` - Direct career sites, staffing agencies, Canadian job boards, blocked sites, base URLs, manual ATS domains — all tuned for Metro Vancouver warehouse roles
- `config/searches.example.yaml` - Example search configuration with the 9 target warehouse titles, every Greater Vancouver municipality, and warehouse-specific exclude filters

---

## How Stages Work

### Discover
Queries Indeed CA, LinkedIn, ZipRecruiter, Google Jobs via JobSpy with `country: "Canada"`. Scrapes the Canadian warehouse Workday tenants in `employers.yaml` (Loblaw, Save-On-Foods, Walmart Canada, Sysco, etc.). Hits direct career sites (Amazon Jobs, Costco, IKEA, UPS, Purolator, Gordon Food Service, etc.) plus Canadian boards (Job Bank, WorkBC, Eluta, Talent.com, Jooble, CareerJet) and staffing agencies (Randstad, Adecco, Aerotek, Drake). Deduplicates by URL. Glassdoor is intentionally blocked — its bot detection makes scraping unreliable.

### Enrich
Visits each job URL and extracts the full description. 3-tier cascade: JSON-LD structured data, then CSS selector patterns, then AI-powered extraction for unknown layouts.

### Score
AI scores every job 1-10 against your profile, weighting forklift / WHMIS / First Aid certifications, RF-scanner & WMS experience, shift availability, and physical capability. 9-10 = strong match, 7-8 = good, 5-6 = moderate, 1-4 = skip. Jobs outside Greater Vancouver are heavily downweighted. Only jobs above your threshold proceed to tailoring.

### Tailor
Generates a custom resume per job: reorders experience, emphasizes relevant warehouse / material-handling / equipment skills, incorporates keywords from the job description. Your `resume_facts` (past employers, achievements, real metrics) are preserved exactly. The AI reorganizes but never fabricates — it will never claim a forklift cert, WHMIS, or other credential that isn't in your profile.

### Cover Letter
Writes a targeted cover letter per job referencing the specific company, role, and how your experience maps to their requirements.

### Auto-Apply
Claude Code launches a Chrome instance, navigates to each application page, detects the form type, fills personal information and work history, uploads the tailored resume and cover letter, answers screening questions with AI, and submits. A live dashboard shows progress in real-time.

The Playwright MCP server is configured automatically at runtime per worker. No manual MCP setup needed.

```bash
# Utility modes (no Chrome/Claude needed)
applypilot apply --mark-applied URL    # manually mark a job as applied
applypilot apply --mark-failed URL     # manually mark a job as failed
applypilot apply --reset-failed        # reset all failed jobs for retry
applypilot apply --gen --url URL       # generate prompt file for manual debugging
```

---

## CLI Reference

```
applypilot init                         # First-time setup wizard
applypilot doctor                       # Verify setup, diagnose missing requirements
applypilot run [stages...]              # Run pipeline stages (or 'all')
applypilot run --workers 4              # Parallel discovery/enrichment
applypilot run --stream                 # Concurrent stages (streaming mode)
applypilot run --min-score 8            # Override score threshold
applypilot run --dry-run                # Preview without executing
applypilot run --validation lenient     # Relax validation (recommended for Gemini free tier)
applypilot run --validation strict      # Strictest validation (retries on any banned word)
applypilot apply                        # Launch auto-apply
applypilot apply --workers 3            # Parallel browser workers
applypilot apply --dry-run              # Fill forms without submitting
applypilot apply --continuous           # Run forever, polling for new jobs
applypilot apply --headless             # Headless browser mode
applypilot apply --url URL              # Apply to a specific job
applypilot status                       # Pipeline statistics
applypilot dashboard                    # Open HTML results dashboard
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and PR guidelines.

---

## License

ApplyPilot is licensed under the [GNU Affero General Public License v3.0](LICENSE).

You are free to use, modify, and distribute this software. If you deploy a modified version as a service, you must release your source code under the same license.
