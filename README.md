# ApplyAssist

**An AI job-search copilot. It does the tedious work — discovering, scoring, tailoring, and filling out applications — then *you* review and click Submit.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-green.svg)](LICENSE)

> ApplyAssist is a fork of [**ApplyPilot**](https://github.com/Pickle-Pixel/ApplyPilot) by [Pickle-Pixel](https://github.com/Pickle-Pixel) (AGPL-3.0). It keeps the genuinely useful machinery — multi-board discovery, AI fit-scoring, per-job resume tailoring and cover letters — and **deliberately removes the "fully autonomous, apply to 1,000 jobs" behavior**. See [Why this fork exists](#why-this-fork-exists).

---

## Why this fork exists

The original tool optimizes for **volume**: apply to as many jobs as possible, hands-free, solving CAPTCHAs along the way. In practice that is the wrong goal:

- **Volume isn't the bottleneck — relevance and signal are.** 30 targeted applications beat 1,000 sprayed ones. Recruiters and ATS systems detect and down-rank bulk submissions.
- **It gets accounts banned.** LinkedIn and others actively flag high-volume automated applying, especially combined with scraping and CAPTCHA-solving. Losing your account mid-search is a real setback.
- **Auto-submitting without review replicates mistakes at scale** — one wrong field, one mismatched role, multiplied by hundreds.
- **Solving CAPTCHAs is bot-detection evasion.** A CAPTCHA is a site explicitly asking for a human. ApplyAssist's answer is to give it one — you.

So ApplyAssist keeps a human in the loop. It removes the friction (finding jobs, tailoring, filling forms) but leaves the **judgment and the Submit click** with you. You review every application in seconds instead of filling it out in minutes — that's how you scale responsibly.

---

## What it does

A pipeline that prepares applications, plus an **assisted-apply** step that fills the form and stops:

```bash
pip install applyassist
pip install --no-deps python-jobspy && pip install pydantic tls-client requests markdownify regex
applyassist init          # one-time setup: resume, profile, preferences, API keys
applyassist doctor        # verify setup — shows what's installed and what's missing
applyassist run           # discover → enrich → score → tailor → cover letters → pdf
applyassist apply         # REVIEW MODE (default): opens a browser, fills the form, STOPS for you to submit
```

> **Why two install commands?** `python-jobspy` pins an exact numpy version in its metadata that conflicts with pip's resolver but works fine at runtime. `--no-deps` bypasses the resolver; the second command installs jobspy's actual runtime deps.

---

## How applying works

`applyassist apply` defaults to **review mode**:

1. It picks your highest-fit prepared job and opens it in a **visible** browser.
2. The agent navigates the form, uploads your tailored resume + cover letter, answers screening questions, and fills every field.
3. It **stops before Submit**, leaves the browser open, and hands control to you.
4. You eyeball the form, fix anything, and **click Submit yourself**. Then tell ApplyAssist whether you submitted (`a`), want to skip (`s`), or quit (`q`).

It never solves CAPTCHAs. If a challenge appears, it pauses and you finish that step in the open browser.

A **daily cap** (default 30, set `APPLYASSIST_DAILY_CAP`) keeps your pace human — recruiters down-rank bursts and platforms flag them.

### Autopilot (opt-in, not recommended)

```bash
applyassist apply --autopilot     # also clicks Submit. Prompts for confirmation. Higher ban/ToS risk.
```

Autopilot still never solves CAPTCHAs and still respects the daily cap. Use it sparingly, if at all.

---

## The pipeline

| Stage | What happens |
|-------|-------------|
| **1. Discover** | Scrapes job boards (Indeed, LinkedIn, Glassdoor, ZipRecruiter, Google Jobs) + Workday employer portals + direct career sites |
| **2. Enrich** | Fetches full job descriptions via JSON-LD, CSS selectors, or AI extraction |
| **3. Score** | AI rates every job 1–10 against your resume and preferences; only high-fit jobs proceed |
| **4. Tailor** | AI reorganizes your resume per job to emphasize relevant experience. **Never fabricates** |
| **5. Cover letter** | AI drafts a targeted cover letter per job |
| **6. PDF** | Converts tailored resumes and cover letters to PDF for upload |

Then **assisted apply** (review mode) fills the form for you to submit. Each stage is independent — run them all or pick what you need.

---

## Requirements

| Component | Required for | Details |
|-----------|-------------|---------|
| Python 3.11+ | Everything | Core runtime |
| Gemini API key | Scoring, tailoring, cover letters | Free tier is enough — get one at [aistudio.google.com](https://aistudio.google.com) |
| Node.js 18+ | Assisted apply | Needed for `npx` to run the Playwright MCP server |
| Chrome/Chromium | Assisted apply | Auto-detected on most systems |
| Claude Code CLI | Assisted apply | Install from [claude.ai/code](https://claude.ai/code) |

OpenAI and local models (Ollama/llama.cpp) are also supported.

There is **no CAPTCHA-solver dependency** — that was removed by design.

---

## Open source vs hosted

ApplyAssist is **free and fully featured when self-hosted.** Every capability above runs locally with your own API keys; nothing is paywalled.

A future **paid hosted** tier would sell *convenience*, not features: managed API keys, multi-device sync, and zero local setup. Because ApplyAssist is AGPL-3.0, **any hosted version's source stays public** — that's a deliberate constraint, not an afterthought. Open-core via managed hosting is compatible with AGPL; closed-source SaaS is not.

The capability tiers (`applyassist doctor` shows yours) describe what your *local install* can do based on installed dependencies — they are not a billing boundary.

---

## CLI reference

```
applyassist init                        # First-time setup wizard
applyassist doctor                      # Verify setup, diagnose missing requirements
applyassist run [stages...]             # Run pipeline stages (discover enrich score tailor cover pdf, or 'all')
applyassist run --workers 4             # Parallel discovery/enrichment
applyassist run --min-score 8           # Override score threshold
applyassist apply                       # REVIEW MODE: fill the form, stop for you to submit (default)
applyassist apply --url URL             # Prepare a specific job
applyassist apply --autopilot           # Opt-in: also clicks Submit (confirmation required)
applyassist apply --mark-applied URL    # Mark a job applied after you submitted it
applyassist apply --mark-failed URL     # Mark a job failed
applyassist apply --reset-failed        # Reset failed jobs for retry
applyassist apply --gen --url URL       # Generate the agent prompt for manual debugging
applyassist status                      # Pipeline statistics
applyassist dashboard                   # Open HTML results dashboard
```

---

## Responsible use

- More applications is not the goal — more **interviews per application** is.
- Review and submit every application yourself.
- No CAPTCHA solving, no bot-detection evasion.
- Respect the daily cap; vary your pace.
- The hours you save on form-filling are best spent on **referrals and follow-ups** — they convert far better than cold applications.

---

## Attribution

ApplyAssist is a fork of [ApplyPilot](https://github.com/Pickle-Pixel/ApplyPilot) by [Pickle-Pixel](https://github.com/Pickle-Pixel), licensed under AGPL-3.0. Significant credit for the discovery, scoring, and tailoring machinery belongs to the original project. ApplyAssist re-purposes it as a human-in-the-loop assistant rather than an autonomous submitter.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and PR guidelines.

---

## License

ApplyAssist is licensed under the [GNU Affero General Public License v3.0](LICENSE), inherited from ApplyPilot.

You are free to use, modify, and distribute this software. **If you deploy a modified version as a service, you must release your source code under the same license.**
