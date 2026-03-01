<!-- logo here -->

# ApplyPilot

**Applied to 1,000 jobs in 2 days. Fully autonomous. Open source.**

[![PyPI version](https://img.shields.io/pypi/v/applypilot?color=blue)](https://pypi.org/project/applypilot/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Pickle-Pixel/ApplyPilot?style=social)](https://github.com/Pickle-Pixel/ApplyPilot)
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/S6S01UL5IO)




https://github.com/user-attachments/assets/7ee3417f-43d4-4245-9952-35df1e77f2df


---

## What It Does

ApplyPilot is a 6-stage autonomous job application pipeline. It discovers jobs across 5+ boards, scores them against your resume with AI, tailors your resume per job, writes cover letters, and **submits applications for you**. It navigates forms, uploads documents, answers screening questions, all hands-free.

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
**Requires:** Python 3.11+, Node.js (for npx), Gemini API key (free), Claude Code CLI or OpenCode CLI, Chrome

Runs all 6 stages, from job discovery to autonomous application submission. This is the full power of ApplyPilot.

### Discovery + Tailoring Only
**Requires:** Python 3.11+, Gemini API key (free)

Runs stages 1-5: discovers jobs, scores them, tailors your resume, generates cover letters. You submit applications manually with the AI-prepared materials.

---

## The Pipeline

| Stage | What Happens |
|-------|-------------|
| **1. Discover** | Scrapes 5 job boards (Indeed, LinkedIn, Glassdoor, ZipRecruiter, Google Jobs) + 48 Workday employer portals + 30 direct career sites |
| **2. Enrich** | Fetches full job descriptions via JSON-LD, CSS selectors, or AI-powered extraction |
| **3. Score** | AI rates every job 1-10 based on your resume and preferences. Only high-fit jobs proceed |
| **4. Tailor** | AI rewrites your resume per job: reorganizes, emphasizes relevant experience, adds keywords. Never fabricates |
| **5. Cover Letter** | AI generates a targeted cover letter per job |
| **6. Auto-Apply** | Orchestrates browser-driven submission using an external backend (Claude or OpenCode). The backend launches a browser, detects the form type, fills personal information and work history, uploads the tailored resume and cover letter, answers screening questions with AI, and submits. |

Each stage is independent. Run them all or pick what you need.

---

## ApplyPilot vs The Alternatives

| Feature | ApplyPilot | AIHawk | Manual |
|---------|-----------|--------|--------|
| Job discovery | 5 boards + Workday + direct sites | LinkedIn only | One board at a time |
| AI scoring | 1-10 fit score per job | Basic filtering | Your gut feeling |
| Resume tailoring | Per-job AI rewrite | Template-based | Hours per application |
| Auto-apply | Full form navigation + submission | LinkedIn Easy Apply only | Click, type, repeat |
| Supported sites | Indeed, LinkedIn, Glassdoor, ZipRecruiter, Google Jobs, 46 Workday portals, 28 direct sites | LinkedIn | Whatever you open |
| License | AGPL-3.0 | MIT | N/A |

---

## Requirements

| Component | Required For | Details |
|-----------|-------------|---------|
| Python 3.11+ | Everything | Core runtime |
| Node.js 18+ | Auto-apply | Needed for `npx` to run Playwright MCP server |
| Gemini API key | Scoring, tailoring, cover letters | Free tier (15 RPM / 1M tokens/day) is enough |
| Chrome/Chromium | Auto-apply | Auto-detected on most systems |
| Claude Code CLI or OpenCode CLI | Auto-apply | Claude: install from https://claude.ai/code; OpenCode: install from https://opencode.ai and register MCPs |

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
API keys and runtime config: `GEMINI_API_KEY`, `LLM_MODEL`, `CAPSOLVER_API_KEY` (optional). See Backend and Gateway configuration for details on multi-backend selection and gateway compatibility.

---

## Backend and Gateway configuration (Gemini first, OpenCode backend)

ApplyPilot supports multiple LLM backends. The baseline-first approach for LLMs is Gemini. For the auto-apply orchestration, the code's runtime default backend is Claude (APPLY_BACKEND unset => "claude"). OpenCode (opencode) is the recommended production path and is supported as an alternative; set APPLY_BACKEND=opencode to use it. Configure your environment carefully and never commit real keys.

1) Baseline LLM (Gemini)
- Set GEMINI_API_KEY to use Google Gemini for scoring, tailoring, and cover letters. This is the recommended default and is used automatically when present.

2) Gateway compatibility (9router / OpenAI-compatible gateways)
- If you need a proxy or gateway that speaks the OpenAI-compatible API (for example, 9router, self-hosted gateways, or Ollama with a REST wrapper), set these env vars in your `.env` or runtime environment:

  - LLM_URL: Base URL of your gateway, for example `https://my-9router.example.com/v1`
  - LLM_API_KEY: API key for that gateway (keep secret)
  - LLM_MODEL: Model name exposed by the gateway, for example `gpt-4o-mini`

- Example (do not paste real keys):

  export LLM_URL="https://my-9router.example.com/v1"
  export LLM_API_KEY="sk-xxxxxxxx"
  export LLM_MODEL="gpt-4o-mini"

3) Backend selection for auto-apply and orchestration
- Use APPLY_BACKEND to select which orchestration backend the system will auto-apply with. Supported values:
  - opencode: Use the OpenCode backend and its MCP integrations (recommended)
  - claude: Use Claude Code CLI for auto-apply (current code default when APPLY_BACKEND is not set)

- Backend defaults are configurable:
  - `APPLY_CLAUDE_MODEL` (default: `haiku`)
  - `APPLY_OPENCODE_MODEL` (fallback: `LLM_MODEL`, then `gpt-4o-mini`)
  - `APPLY_OPENCODE_AGENT` (passed as `--agent` to `opencode run`)

  Example (use OpenCode):

  export APPLY_BACKEND=opencode
  export APPLY_OPENCODE_MODEL="gh/claude-sonnet-4.5"
  export APPLY_OPENCODE_AGENT="coder"

4) OpenCode MCP prerequisite
- When using the opencode backend you must register the OpenCode MCP provider before first run. Run:

  opencode mcp add my-mcp --provider=openai --url "$LLM_URL" --api-key "$LLM_API_KEY" --model "$LLM_MODEL"

- Replace the provider and flags according to your MCP. This registers the gateway so OpenCode can reach it at runtime. Note: OpenCode manages MCP servers globally in its own config; you cannot pass an MCP config file per invocation.
- For parity with Claude apply flow, ensure `opencode mcp list` contains both MCP server names:
  - `playwright`
  - `gmail`
  ApplyPilot validates this baseline before running the OpenCode backend.

5) Claude fallback / code default
- The code default backend when APPLY_BACKEND is not set is `claude`. If you plan to rely on the default behavior or explicitly set APPLY_BACKEND=claude, ensure Claude Code CLI is installed and configured. Claude remains supported as a fallback orchestration backend.

6) Security and secret handling
- Never add API keys to git. Use a local file outside the repo (for example `~/.applypilot/.env`) or a secret manager.
- Add `.env` or `~/.applypilot/.env` to your `.gitignore`.
- Rotate keys regularly and treat gateway keys like production secrets.
- When sharing examples, replace any keys with `sk-xxxxxxxx` or `GEMINI_API_KEY=xxxxx` placeholders.

7) 9router example variables
- 9router and similar gateways expect the following env variables for compatibility with ApplyPilot's AI stages: `LLM_URL`, `LLM_API_KEY`, `LLM_MODEL`. Make sure the gateway exposes an OpenAI-compatible v1 completions/chat endpoint.

8) Verification
- After setting env vars and optionally registering MCPs for opencode, run `applypilot doctor`. It will report configured providers and flag missing MCP registration or missing CLI binaries. If doctor reports issues, follow its guidance.

### OpenCode Configuration Details

ApplyPilot uses OpenCode in **project mode** with isolated configuration to prevent conflicts with your personal OpenCode setup:

**Configuration File:** `~/.applypilot/.opencode/opencode.jsonc`

This file is where you define:
- The custom `applypilot-apply` agent
- MCP server configurations
- Permission scopes
- Model defaults

**XDG Directory Isolation:**
ApplyPilot sets `XDG_CONFIG_HOME=~/.applypilot` when running OpenCode, which means:
- OpenCode loads **only** `~/.applypilot/.opencode/opencode.jsonc`
- Your global `~/.config/opencode/` is ignored
- Auth credentials (`~/.opencode/auth.json`) still work (different XDG var)
- State and sessions use default locations

This isolation ensures ApplyPilot's agent configuration doesn't interfere with your personal OpenCode workflows.

**Example `~/.applypilot/.opencode/opencode.jsonc`:**
```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "permission": {
    "*": "allow",
    "question": "deny"
  },
  "tools": {
    "question": false
  },
  "agent": {
    "applypilot-apply": {
      "description": "Autonomous job application agent",
      "mode": "primary",
      "model": "github-copilot/gpt-5-mini",
      "prompt": "{file:../prompts/apply-agent.md}",
      "permission": {
        // Safety: deny file editing and bash
        "task": "deny",
        "edit": "deny",
        "write": "deny",
        "bash": "deny",
        // Read-only tools allowed
        "read": "allow",
        "grep": "allow",
        "glob": "allow",
        "lsp": "allow",
        // All Playwright browser tools enabled
        "playwright_browser_navigate": "allow",
        "playwright_browser_click": "allow",
        "playwright_browser_fill_form": "allow",
        "playwright_browser_snapshot": "allow",
        "playwright_browser_evaluate": "allow",
        "playwright_browser_file_upload": "allow",
        "playwright_browser_tabs": "allow",
        "playwright_browser_wait_for": "allow",
        "playwright_browser_screenshot": "allow",
        // All Gmail tools enabled
        "gmail_search_emails": "allow",
        "gmail_read_email": "allow",
        "gmail_send_email": "allow",
        "gmail_create_draft": "allow"
      }
    }
  },
  "mcp": {
    "playwright": {
      "type": "local",
      "enabled": true,
      "command": [
        "npx", "@playwright/mcp@latest",
        "--cdp-endpoint=http://localhost:9222"
      ]
    },
    "gmail": {
      "type": "local",
      "enabled": true,
      "command": [
        "npx", "-y", "@gongrzhe/server-gmail-autoauth-mcp"
      ]
    }
  }
}
```

**MCP Server Registration:**
```bash
# Register Playwright MCP for browser automation
opencode mcp add playwright --provider=openai --url="$LLM_URL" --api-key="$LLM_API_KEY"

# Verify registration
opencode mcp list
```

**Key Points:**
- The agent name `applypilot-apply` is referenced in the backend code
- The prompt file path is relative to the `~/.applypilot/` directory
- Browser tools are explicitly allowed; file editing is denied for safety
- Model can be overridden via `APPLY_OPENCODE_MODEL` environment variable

---

### Package configs (shipped with ApplyPilot)
- `config/employers.yaml` - Workday employer registry (48 preconfigured)
- `config/sites.yaml` - Direct career sites (30+), blocked sites, base URLs, manual ATS domains
- `config/searches.example.yaml` - Example search configuration

---

## How Stages Work

### Discover
Queries Indeed, LinkedIn, Glassdoor, ZipRecruiter, Google Jobs via JobSpy. Scrapes 48 Workday employer portals (configurable in `employers.yaml`). Hits 30 direct career sites with custom extractors. Deduplicates by URL.

### Enrich
Visits each job URL and extracts the full description. 3-tier cascade: JSON-LD structured data, then CSS selector patterns, then AI-powered extraction for unknown layouts.

### Score
AI scores every job 1-10 against your profile. 9-10 = strong match, 7-8 = good, 5-6 = moderate, 1-4 = skip. Only jobs above your threshold proceed to tailoring.

### Tailor
Generates a custom resume per job: reorders experience, emphasizes relevant skills, incorporates keywords from the job description. Your `resume_facts` (companies, projects, metrics) are preserved exactly. The AI reorganizes but never fabricates.

### Cover Letter
Writes a targeted cover letter per job referencing the specific company, role, and how your experience maps to their requirements.

### Auto-Apply
Auto-apply is implemented via a pluggable backend with two supported options:

- **Claude**: uses the Claude Code CLI (default when APPLY_BACKEND is unset)
- **OpenCode**: uses the OpenCode CLI with pre-configured MCP servers

Both backends perform the same high-level tasks: launch a browser, detect form types, fill personal details, upload tailored documents, answer screening questions, and submit applications. A live dashboard shows progress in real-time.

Choose your backend by setting APPLY_BACKEND=claude or APPLY_BACKEND=opencode. Each requires the respective CLI to be installed and configured.

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
