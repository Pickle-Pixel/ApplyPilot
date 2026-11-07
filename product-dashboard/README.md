# ApplyPilot CTO Dashboard

Product-intelligence dashboard reverse-engineered from source code. All conclusions have direct evidence.

## Quick Start

```bash
# Go to ApplyPilot project root
cd d:\obsidian\4_项目\2025逃跑计划\ApplyPilot

# Start a simple HTTP server (required because fetch() needs http:// not file://)
python -m http.server 8765

# Open in browser
# http://localhost:8765/product-dashboard/
```

### Why HTTP server?

The dashboard fetches `.json` files from `.project-intelligence/` using `fetch()`. Browsers block `fetch()` on `file://` URLs for security — that's why you need `python -m http.server` (or any other static server).

### Alternative: Node.js

```bash
npx http-server -p 8765
```

## Navigation Hierarchy

The primary navigation follows the product perspective, not source-code structure:

```
Overview (landing)
  └─ Roles  ← START HERE
       └─ Journeys
            └─ Steps (user actions)
                 ├─ CLI Entry Point
                 ├─ Backend Entry Point
                 ├─ Related Atomic Capabilities
                 ├─ Related Business Feature
                 ├─ Exact file path
                 ├─ Symbol + line number
                 └─ Evidence + Confidence
```

Secondary views (top tabs): Features, Capabilities, Code/Architecture, Risks.

## Tier Filtering

The blue/purple/gold tier filter at the top lets you focus on a specific capability level:

| Tier | Capabilities | Unlock Condition |
|------|-------------|-----------------|
| 1 · Discovery | `init`, `run discover`, `run enrich`, `status`, `doctor`, `dashboard` | Python + pip only |
| 2 · AI Scoring | `run score`, `run tailor`, `run cover`, `run pdf`, `run all`, `run --stream` | Tier 1 + LLM API key env var |
| 3 · Auto-Apply | `apply`, `apply --workers`, `apply --continuous` | Tier 2 + claude CLI + Chrome + Node.js |

Note: Tier is a capability-detection mechanism, not role-based access. The same user can run all tiers.

## Data Source

All data comes from `.project-intelligence/` — produced by the reverse product discovery pipeline (Stages 0–6):

| Artifact | Content |
|----------|---------|
| `.project-intelligence/roles/user_roles.json` | 1 human role + 7 system components |
| `.project-intelligence/journeys/user_journeys.json` | 9 user journeys · 34 ordered steps |
| `.project-intelligence/features/business_features.json` | 10 business features |
| `.project-intelligence/capabilities/atomic_capabilities.json` | 28 atomic capabilities |
| `.project-intelligence/inventory/symbols.json` | 120+ AST symbols with qualified names and line numbers |
| `.project-intelligence/inventory/files.json` | 23 source files with file_id → path mapping |
| `.project-intelligence/dashboard/cto_dashboard.json` | Architecture, tier boundaries, risk register |

**Stage 7 does not modify any Stage 0–6 artifacts.** All join operations happen at runtime in `data-loader.js`.

## Files in This Directory

```
product-dashboard/
  ├─ index.html         # SPA skeleton
  ├─ styles.css         # Dark theme + tier color coding
  ├─ data-loader.js     # Loads 6 JSON files, builds join tables
  ├─ app.js             # All navigation + rendering logic
  └─ README.md
```

## "I see missing mappings..."

Gaps are intentionally shown as:
- **Missing** (red dashed border) — no data in source
- **System behavior** (gray badge) — internal, not user-facing
- **Inferred** (yellow badge) — conclusion with partial evidence

This is a deliberate design choice: we show uncertainty, we don't invent relationships.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Page loads but says "Failed to load data" | Make sure you're running `python -m http.server` from the **ApplyPilot project root**, NOT from `product-dashboard/` |
| Tier filter does nothing | The filter applies to Roles and Journeys tabs — go there and try |
| Symbols show as `SYM-0012` instead of function names | `files.json` may be missing — check `.project-intelligence/inventory/files.json` exists |