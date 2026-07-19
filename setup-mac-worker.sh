#!/usr/bin/env bash
# setup-mac-worker.sh -- ONE-TIME interactive bootstrap of a Mac as an ApplyPilot
# offsite-ATS apply worker (worker-id mac-0). Copy JUST THIS FILE to the Mac and run:
#   bash setup-mac-worker.sh
# It clones the private repo via a READ-ONLY deploy key (your GitHub credentials never
# touch this machine), prompts for the PG password + API keys (stored ONLY in chmod-600
# files here), hydrates profile/resume from the fleet_assets PG table, and registers the
# launchd agent so the worker runs whenever this Mac is on. LinkedIn is NEVER installed
# here. Prereqs done by the owner first: Tailscale installed+joined on this Mac, and
# setup-fleet-pg-tailscale.ps1 run on the home box (it prints the values prompted below).
# macOS ships bash 3.2 -- keep this file 3.2-compatible.
set -eu

REPO_SSH="git@github.com:thefulmination/applypilot-private.git"
INSTALL_DIR="${INSTALL_DIR:-$HOME/applypilot-fleet}"
KEY="$HOME/.ssh/applypilot_deploy"
say() { printf '\n[setup] %s\n' "$*"; }

# --- 0. sanity: macOS + Tailscale up -----------------------------------------
[ "$(uname)" = "Darwin" ] || { echo "This script is for macOS."; exit 1; }
if ! /Applications/Tailscale.app/Contents/MacOS/Tailscale status >/dev/null 2>&1 \
   && ! command -v tailscale >/dev/null 2>&1; then
  say "Tailscale not found. Install from https://tailscale.com/download, sign in, then re-run."
  exit 1
fi

# --- 1. toolchain (Homebrew, python, node/npx, git, agent CLIs) --------------
if ! command -v brew >/dev/null 2>&1; then
  say "Installing Homebrew (you may be prompted for the Mac's password)..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv)"
fi
say "Installing python, node, git via Homebrew..."
brew install python@3.12 node git >/dev/null
say "Installing Claude Code CLI..."
npm install -g @anthropic-ai/claude-code >/dev/null
say "Installing Codex CLI..."
npm install -g @openai/codex >/dev/null

# --- 2. read-only deploy key + clone ------------------------------------------
if [ ! -f "$KEY" ]; then
  ssh-keygen -t ed25519 -f "$KEY" -N "" -C "applypilot-mac-worker" >/dev/null
fi
say "Add this READ-ONLY deploy key to the private repo, then press Enter:"
echo "  GitHub -> thefulmination/applypilot-private -> Settings -> Deploy keys -> Add (leave 'write access' UNCHECKED)"
echo ""
cat "$KEY.pub"
read -r _
GIT_SSH="ssh -i $KEY -o IdentitiesOnly=yes"
printf 'Branch to run [applypilot-hardening-and-brainstorm-integration]: '; read -r BRANCH
BRANCH="${BRANCH:-applypilot-hardening-and-brainstorm-integration}"
if [ ! -d "$INSTALL_DIR/.git" ]; then
  say "Cloning $REPO_SSH ($BRANCH) -> $INSTALL_DIR"
  GIT_SSH_COMMAND="$GIT_SSH" git clone --branch "$BRANCH" "$REPO_SSH" "$INSTALL_DIR"
fi

# --- 3. venv + package + playwright chromium ----------------------------------
cd "$INSTALL_DIR"
say "Creating venv + installing applypilot (editable)..."
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -e . && ./.venv/bin/pip install -q "psycopg[binary]" mcp pyyaml
say "Installing Playwright chromium (project-local)..."
PLAYWRIGHT_BROWSERS_PATH="$INSTALL_DIR/.playwright-browsers" ./.venv/bin/python -m playwright install chromium

# --- 4. mapped fleet DSN + prompts ---------------------------------------------
# The unique mapped per-node fleet DSN (with its own credential) is supplied whole via
# FLEET_PG_DSN. This installer never prompts for the shared fleet_worker/postgres password
# and never constructs a shared-role DSN. Fail closed before any database action.
DSN="${FLEET_PG_DSN:-}"
if [ -z "$DSN" ]; then
  echo "setup-mac-worker: FLEET_PG_DSN is not set. Export an explicit mapped per-node fleet DSN (with its own password; never the postgres admin or the shared fleet_worker role) before running this installer." >&2
  exit 78
fi
if printf '%s' "$DSN" | grep -qiE 'user=(postgres|fleet_worker)'; then
  echo "setup-mac-worker: FLEET_PG_DSN selects an admin/shared principal. Supply an explicit mapped fleet DSN." >&2
  exit 78
fi
printf 'ANTHROPIC_API_KEY: '; read -rs ANTHROPIC_KEY; echo
printf 'DEEPSEEK_API_KEY: '; read -rs DEEPSEEK_KEY; echo

# --- 6. env file (everything run-worker-mac.sh needs) --------------------------
mkdir -p "$INSTALL_DIR/.applypilot" "$INSTALL_DIR/logs"
ENV_FILE="$INSTALL_DIR/.applypilot/fleet-worker.env"
CLAUDE_BIN="$(command -v claude || true)"
CODEX_BIN="$(command -v codex || true)"
if [ -z "$CLAUDE_BIN" ]; then
  say "WARNING: 'claude' CLI not found on PATH after npm install -g."
  say "  The worker falls back to a PATH lookup at runtime, but if that also fails, applies will not run."
  say "  Check 'npm bin -g' is on PATH, then re-run this script (safe to re-run)."
fi
if [ -z "$CODEX_BIN" ]; then
  say "WARNING: 'codex' CLI not found on PATH after npm install -g."
  say "  Claude-primary workers need Codex available for quota fallback."
  say "  Check 'npm bin -g' is on PATH, then re-run this script (safe to re-run)."
fi
# Values are LITERAL-QUOTED (single quotes in the WRITTEN file): this file is sourced
# by bash (set -a; . file), and a value containing $ or backticks (e.g. a future
# API-key format) must not re-expand at source time. The heredoc below still expands
# each $VAR ONCE at write time (unquoted heredoc delimiter); only the output is single-quoted.
cat > "$ENV_FILE" <<EOF
FLEET_PG_DSN='$DSN'
APPLYPILOT_FLEET_DSN='$DSN'
APPLYPILOT_DIR='$INSTALL_DIR/.applypilot'
PLAYWRIGHT_BROWSERS_PATH='$INSTALL_DIR/.playwright-browsers'
APPLYPILOT_DB_PATH='/tmp/fleet_apply_throwaway_0.db'
APPLYPILOT_ENABLE_GMAIL_MCP='0'
APPLYPILOT_AGENT_TIMEOUT='600'
ANTHROPIC_API_KEY='$ANTHROPIC_KEY'
DEEPSEEK_API_KEY='$DEEPSEEK_KEY'
CLAUDE_PATH='$CLAUDE_BIN'
CODEX_PATH='$CODEX_BIN'
WORKER_LABEL='mac'
WORKER_SLOT='0'
WORKER_AGENT='claude'
WORKER_MODEL='sonnet'
WORKER_FALLBACK_AGENT='codex'
FLEET_MACHINE_OWNER='mac-$(hostname -s)'
APPLYPILOT_BRANCH='$BRANCH'
UPDATE_CHECK_SECONDS='21600'
RESTART_BACKOFF_SECONDS='30'
APPLYPILOT_MAC_CAFFEINATE='1'
GIT_SSH_COMMAND='$GIT_SSH'
APPLYPILOT_INBOX_AUTH='1'
APPLYPILOT_INBOX_AUTH_MODE='relay'
FLEET_WORKER_ID='mac-0'
EOF
chmod 600 "$ENV_FILE"
unset ANTHROPIC_KEY DEEPSEEK_KEY
say "Env file written (chmod 600): $ENV_FILE"

# --- 7. connectivity + asset hydration from PG ---------------------------------
say "Testing Postgres over Tailscale..."
APPLYPILOT_TEST_DSN="$DSN" APPLYPILOT_DIR="$INSTALL_DIR/.applypilot" ./.venv/bin/python - <<'PY'
import os, pathlib
from applypilot.apply import pgqueue
dsn = os.environ["APPLYPILOT_TEST_DSN"]
conn = pgqueue.connect(dsn)
print("[setup] PG connection OK")
appdir = pathlib.Path(os.environ.get("APPLYPILOT_DIR", "")) or pathlib.Path.cwd() / ".applypilot"
appdir.mkdir(parents=True, exist_ok=True)
for fname in ("profile.json", "resume.pdf"):
    data = pgqueue.get_asset(conn, fname)
    if data:
        (appdir / fname).write_bytes(data)
        print(f"[setup] hydrated {fname} ({len(data)} bytes) from fleet_assets")
    elif not (appdir / fname).exists():
        print(f"[setup] WARNING: {fname} missing -- push it from the home box "
              f"(see docs/fleet-mac-worker-runbook.md) or copy it here manually")
conn.close()
PY

# --- 8. launchd -----------------------------------------------------------------
PLIST="$HOME/Library/LaunchAgents/com.applypilot.fleetworker.plist"
mkdir -p "$HOME/Library/LaunchAgents"
sed "s|__INSTALL_DIR__|$INSTALL_DIR|g" "$INSTALL_DIR/com.applypilot.fleetworker.plist.template" > "$PLIST"
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load -w "$PLIST"
say "launchd agent loaded. The worker now runs whenever this Mac is on."
say "Status : launchctl list | grep applypilot"
say "Logs   : tail -f $INSTALL_DIR/logs/wrapper.log $INSTALL_DIR/.applypilot/logs/worker-0.log"
