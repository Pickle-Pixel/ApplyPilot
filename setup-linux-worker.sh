#!/usr/bin/env bash
# setup-linux-worker.sh -- one-time interactive bootstrap of a Linux Mint/Ubuntu
# machine as an ApplyPilot offsite ATS apply worker.
#
# Copy JUST THIS FILE to the Linux machine and run:
#   bash setup-linux-worker.sh
#
# What this machine receives:
#   - a clean read-only clone of applypilot-private
#   - a least-privilege fleet_worker Postgres connection over Tailscale
#   - profile.json and resume.pdf hydrated from the fleet_assets table
#   - local API keys in chmod-600 files
#   - a systemd service that supervises one applypilot-fleet-apply worker
#
# What it does NOT receive:
#   - the home SQLite brain
#   - Gmail OAuth tokens
#   - LinkedIn cookies
#   - broad Postgres/admin credentials
set -euo pipefail

REPO_SSH="${REPO_SSH:-git@github.com:thefulmination/applypilot-private.git}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/applypilot-fleet}"
KEY="${APPLYPILOT_DEPLOY_KEY:-$HOME/.ssh/applypilot_deploy}"
SERVICE_NAME="${SERVICE_NAME:-applypilot-fleet-worker}"
WORKER_LABEL="${WORKER_LABEL:-mint}"
WORKER_SLOT="${WORKER_SLOT:-0}"
DEFAULT_BRANCH="${APPLYPILOT_BRANCH:-applypilot-hardening-and-brainstorm-integration}"

say() { printf '\n[setup] %s\n' "$*"; }
die() { printf '\n[setup] FATAL: %s\n' "$*" >&2; exit 1; }

shell_quote() {
  local s=${1//\'/\'\\\'\'}
  printf "'%s'" "$s"
}

write_env_line() {
  printf '%s=%s\n' "$1" "$(shell_quote "${2:-}")"
}

require_normal_user() {
  if [ "$(id -u)" -eq 0 ]; then
    die "Run as your normal Linux user, not with sudo. The script will ask sudo only where needed."
  fi
}

check_platform() {
  [ "$(uname -s)" = "Linux" ] || die "This script is for Linux."
  command -v apt-get >/dev/null 2>&1 || die "This script expects Linux Mint/Ubuntu/Debian with apt-get."
  if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    case "${ID:-} ${ID_LIKE:-}" in
      *linuxmint*|*ubuntu*|*debian*) ;;
      *) say "WARNING: untested distro '${PRETTY_NAME:-unknown}'. Continuing because apt-get exists." ;;
    esac
  fi
}

need_sudo() {
  say "Checking sudo access..."
  sudo -v
}

install_packages() {
  say "Installing system packages..."
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git curl ca-certificates openssh-client openssh-server \
    python3 python3-venv python3-pip build-essential
  sudo systemctl enable --now ssh >/dev/null

  if command -v ufw >/dev/null 2>&1 && sudo ufw status 2>/dev/null | grep -qi '^Status: active'; then
    sudo ufw allow from 100.64.0.0/10 to any port 22 proto tcp >/dev/null || true
  fi
}

install_node22() {
  local major node_dir version arch tarball url
  major="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || true)"
  if [ -z "$major" ] || [ "$major" -lt 22 ]; then
    if sudo -n true 2>/dev/null; then
      say "Installing Node.js 22 for Claude Code/Codex CLIs..."
      curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs
    else
      say "Installing Node.js 22 under ~/.local because sudo is not currently cached..."
      node_dir="$HOME/.local/node-v22"
      arch="$(uname -m)"
      case "$arch" in
        x86_64) arch="x64" ;;
        aarch64|arm64) arch="arm64" ;;
        *) die "Unsupported CPU architecture for local Node install: $arch" ;;
      esac
      version="$(python3 - <<'PY'
import json, urllib.request
data = json.load(urllib.request.urlopen("https://nodejs.org/dist/index.json", timeout=20))
for row in data:
    version = row.get("version", "")
    if version.startswith("v22."):
        print(version)
        break
PY
)"
      [ -n "$version" ] || die "Could not resolve latest Node 22 version."
      tarball="/tmp/node-${version}-linux-${arch}.tar.xz"
      url="https://nodejs.org/dist/${version}/node-${version}-linux-${arch}.tar.xz"
      mkdir -p "$node_dir"
      curl -fsSL "$url" -o "$tarball"
      tar -xJf "$tarball" -C "$node_dir" --strip-components=1
      mkdir -p "$HOME/.local/bin"
      ln -sf "$node_dir/bin/node" "$HOME/.local/bin/node"
      ln -sf "$node_dir/bin/npm" "$HOME/.local/bin/npm"
      ln -sf "$node_dir/bin/npx" "$HOME/.local/bin/npx"
      export PATH="$node_dir/bin:$HOME/.local/bin:$PATH"
    fi
  fi
}

install_tailscale() {
  if ! command -v tailscale >/dev/null 2>&1; then
    say "Installing Tailscale..."
    curl -fsSL https://tailscale.com/install.sh | sh
  fi
  sudo systemctl enable --now tailscaled >/dev/null
  if ! tailscale status >/dev/null 2>&1; then
    say "Joining Tailscale. A browser/device-login prompt may appear."
    sudo tailscale up
  fi
}

install_agent_clis() {
  say "Installing/updating Claude Code and Codex CLIs..."
  npm install -g @anthropic-ai/claude-code @openai/codex >/dev/null || \
    sudo npm install -g @anthropic-ai/claude-code @openai/codex >/dev/null
}

install_operator_ssh_key() {
  local key_text="${APPLYPILOT_FLEET_SSH_PUBLIC_KEY:-}"
  if [ -z "$key_text" ]; then
    printf 'Fleet SSH public key for home repair access (optional; paste one line or press Enter): '
    read -r key_text
  fi
  if [ -n "$key_text" ]; then
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"
    touch "$HOME/.ssh/authorized_keys"
    chmod 600 "$HOME/.ssh/authorized_keys"
    if ! grep -qxF "$key_text" "$HOME/.ssh/authorized_keys"; then
      printf '%s\n' "$key_text" >> "$HOME/.ssh/authorized_keys"
      say "Installed fleet SSH public key in ~/.ssh/authorized_keys."
    else
      say "Fleet SSH public key already present."
    fi
  fi
}

ensure_deploy_key() {
  mkdir -p "$HOME/.ssh"
  chmod 700 "$HOME/.ssh"
  if [ ! -f "$KEY" ]; then
    ssh-keygen -t ed25519 -f "$KEY" -N "" -C "applypilot-linux-worker-${WORKER_LABEL}-${WORKER_SLOT}" >/dev/null
  fi
  chmod 600 "$KEY"
  say "Add this READ-ONLY deploy key to the private repo, then press Enter:"
  echo "  GitHub -> thefulmination/applypilot-private -> Settings -> Deploy keys -> Add"
  echo "  Leave 'Allow write access' UNCHECKED."
  echo ""
  cat "$KEY.pub"
  read -r _
}

clone_or_update_repo() {
  printf 'Branch to run [%s]: ' "$DEFAULT_BRANCH"
  read -r BRANCH
  BRANCH="${BRANCH:-$DEFAULT_BRANCH}"
  GIT_SSH="ssh -i $KEY -o IdentitiesOnly=yes"

  if [ ! -d "$INSTALL_DIR/.git" ]; then
    say "Cloning $REPO_SSH ($BRANCH) -> $INSTALL_DIR"
    GIT_SSH_COMMAND="$GIT_SSH" git clone --branch "$BRANCH" "$REPO_SSH" "$INSTALL_DIR"
  else
    say "Updating existing clone at $INSTALL_DIR"
    GIT_SSH_COMMAND="$GIT_SSH" git -C "$INSTALL_DIR" fetch origin "$BRANCH"
    GIT_SSH_COMMAND="$GIT_SSH" git -C "$INSTALL_DIR" checkout "$BRANCH" >/dev/null
    GIT_SSH_COMMAND="$GIT_SSH" git -C "$INSTALL_DIR" merge --ff-only "origin/$BRANCH"
  fi
}

install_python_runtime() {
  cd "$INSTALL_DIR"
  say "Creating venv + installing ApplyPilot..."
  python3 -m venv .venv
  ./.venv/bin/pip install -q --upgrade pip setuptools wheel
  ./.venv/bin/pip install -q -e . "psycopg[binary]" mcp pyyaml

  say "Installing Playwright Chromium and OS browser dependencies..."
  PLAYWRIGHT_BROWSERS_PATH="$INSTALL_DIR/.playwright-browsers" \
    ./.venv/bin/python -m playwright install --with-deps chromium
}

prompt_runtime_values() {
  # The unique mapped per-node fleet DSN (with its own credential) is supplied whole via
  # FLEET_PG_DSN. This installer never prompts for the shared fleet_worker/postgres password
  # and never constructs a shared-role DSN. Fail closed before any database action.
  DSN="${FLEET_PG_DSN:-}"
  [ -n "$DSN" ] || die "FLEET_PG_DSN is not set. Export an explicit mapped per-node fleet DSN (with its own password; never the postgres admin or the shared fleet_worker role) before running this installer."
  if printf '%s' "$DSN" | grep -qiE 'user=(postgres|fleet_worker)'; then
    die "FLEET_PG_DSN selects an admin/shared principal. Supply an explicit mapped fleet DSN."
  fi

  printf 'ANTHROPIC_API_KEY (optional; leave blank for Claude subscription login or Codex-only): '
  read -rs ANTHROPIC_KEY
  echo

  USE_CLAUDE_SUBSCRIPTION="0"
  if [ -z "$ANTHROPIC_KEY" ]; then
    printf 'Use Claude Code subscription login for this worker? [Y/n]: '
    read -r use_sub
    case "${use_sub:-Y}" in
      y|Y|yes|YES) USE_CLAUDE_SUBSCRIPTION="1" ;;
      *) USE_CLAUDE_SUBSCRIPTION="0" ;;
    esac
  fi

  printf 'DEEPSEEK_API_KEY: '
  read -rs DEEPSEEK_KEY
  echo

  printf 'CAPSOLVER_API_KEY (optional; press Enter to skip): '
  read -rs CAPSOLVER_KEY
  echo
}

write_pgpass() {
  # The mapped per-node credential lives in FLEET_PG_DSN itself; no shared fleet_worker
  # pgpass entry is written. Retained as a no-op so callers stay unchanged.
  :
}

write_env_file() {
  local env_file claude_bin codex_bin
  mkdir -p "$INSTALL_DIR/.applypilot" "$INSTALL_DIR/logs"
  env_file="$INSTALL_DIR/.applypilot/fleet-worker.env"
  claude_bin="$(command -v claude || true)"
  codex_bin="$(command -v codex || true)"

  {
    write_env_line FLEET_PG_DSN "$DSN"
    write_env_line APPLYPILOT_FLEET_DSN "$DSN"
    write_env_line APPLYPILOT_DIR "$INSTALL_DIR/.applypilot"
    write_env_line PLAYWRIGHT_BROWSERS_PATH "$INSTALL_DIR/.playwright-browsers"
    write_env_line APPLYPILOT_DB_PATH "/tmp/fleet_apply_throwaway_${WORKER_SLOT}.db"
    write_env_line APPLYPILOT_ENABLE_GMAIL_MCP "0"
    write_env_line APPLYPILOT_AGENT_TIMEOUT "600"
    write_env_line PATH "$PATH"
    if [ -n "$ANTHROPIC_KEY" ]; then
      write_env_line ANTHROPIC_API_KEY "$ANTHROPIC_KEY"
    fi
    write_env_line DEEPSEEK_API_KEY "$DEEPSEEK_KEY"
    write_env_line CAPSOLVER_API_KEY "$CAPSOLVER_KEY"
    write_env_line CLAUDE_PATH "$claude_bin"
    write_env_line CODEX_PATH "$codex_bin"
    write_env_line WORKER_LABEL "$WORKER_LABEL"
    write_env_line WORKER_SLOT "$WORKER_SLOT"
    if [ -n "$ANTHROPIC_KEY" ] || [ "${USE_CLAUDE_SUBSCRIPTION:-0}" = "1" ]; then
      write_env_line WORKER_AGENT "claude"
      write_env_line WORKER_FALLBACK_AGENT "codex"
    else
      write_env_line WORKER_AGENT "codex"
      write_env_line WORKER_FALLBACK_AGENT ""
    fi
    write_env_line WORKER_MODEL "sonnet"
    write_env_line FLEET_MACHINE_OWNER "$WORKER_LABEL"
    write_env_line APPLYPILOT_FLEET_LABEL "$WORKER_LABEL"
    write_env_line APPLYPILOT_BRANCH "$BRANCH"
    write_env_line UPDATE_CHECK_SECONDS "21600"
    write_env_line RESTART_BACKOFF_SECONDS "30"
    write_env_line APPLYPILOT_LINUX_INHIBIT "1"
    write_env_line GIT_SSH_COMMAND "$GIT_SSH"
    write_env_line APPLYPILOT_INBOX_AUTH "1"
    write_env_line APPLYPILOT_INBOX_AUTH_MODE "relay"
    write_env_line FLEET_WORKER_ID "${WORKER_LABEL}-${WORKER_SLOT}"
  } > "$env_file"
  chmod 600 "$env_file"
  unset ANTHROPIC_KEY DEEPSEEK_KEY CAPSOLVER_KEY
  say "Env file written: $env_file"
}

hydrate_assets() {
  say "Testing Postgres over Tailscale and hydrating profile/resume..."
  APPLYPILOT_TEST_DSN="$DSN" APPLYPILOT_DIR="$INSTALL_DIR/.applypilot" \
    "$INSTALL_DIR/.venv/bin/python" - <<'PY'
import os
import pathlib
from applypilot.apply import pgqueue

dsn = os.environ["APPLYPILOT_TEST_DSN"]
conn = pgqueue.connect(dsn)
print("[setup] PG connection OK")
appdir = pathlib.Path(os.environ["APPLYPILOT_DIR"])
appdir.mkdir(parents=True, exist_ok=True)
for fname in ("profile.json", "resume.pdf"):
    data = pgqueue.get_asset(conn, fname)
    if data:
        (appdir / fname).write_bytes(data)
        print(f"[setup] hydrated {fname} ({len(data)} bytes) from fleet_assets")
    elif not (appdir / fname).exists():
        print(f"[setup] WARNING: {fname} missing. Push it from the home box fleet_assets table.")
conn.close()
PY
}

write_worker_wrapper() {
  say "Writing Linux worker wrapper..."
  cat > "$INSTALL_DIR/run-worker-linux.sh" <<'EOF'
#!/usr/bin/env bash
# run-worker-linux.sh -- systemd-supervised offsite ATS apply worker for Linux.
set -u

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$INSTALL_DIR/.applypilot/fleet-worker.env"
inhibit_pid=0
child=0

log() { printf '%s [wrapper] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

load_env() {
  # shellcheck disable=SC1090
  set -a
  . "$ENV_FILE"
  set +a
}

resolve_chrome_path() {
  local bin
  bin=$(find "$PLAYWRIGHT_BROWSERS_PATH" -path '*/chrome-linux/chrome' -type f -executable 2>/dev/null | sort -V | tail -1)
  if [ -n "$bin" ]; then
    printf '%s' "$bin"
    return 0
  fi
  for bin in google-chrome-stable google-chrome chromium chromium-browser; do
    if command -v "$bin" >/dev/null 2>&1; then
      command -v "$bin"
      return 0
    fi
  done
  return 1
}

detect_egress_ip() {
  curl -fsS --max-time 10 https://api.ipify.org 2>/dev/null || printf '0.0.0.0'
}

start_sleep_inhibit() {
  if [ "${APPLYPILOT_LINUX_INHIBIT:-1}" = "1" ] && command -v systemd-inhibit >/dev/null 2>&1; then
    systemd-inhibit --what=sleep --why="ApplyPilot fleet worker active" --mode=block sleep infinity &
    inhibit_pid=$!
    log "sleep inhibit active pid=$inhibit_pid"
  fi
}

updates_available() {
  git -C "$INSTALL_DIR" fetch --quiet origin "$APPLYPILOT_BRANCH" || return 1
  [ "$(git -C "$INSTALL_DIR" rev-parse HEAD)" != "$(git -C "$INSTALL_DIR" rev-parse "origin/$APPLYPILOT_BRANCH")" ]
}

apply_update() {
  local old_sha
  old_sha=$(git -C "$INSTALL_DIR" rev-parse HEAD)
  if ! git -C "$INSTALL_DIR" merge --ff-only "origin/$APPLYPILOT_BRANCH"; then
    log "WARN: update is not a clean fast-forward; leaving current checkout untouched"
    return 1
  fi
  if ! git -C "$INSTALL_DIR" diff --quiet "$old_sha" HEAD -- pyproject.toml; then
    log "pyproject.toml changed; reinstalling package"
    "$INSTALL_DIR/.venv/bin/pip" install -q -e "$INSTALL_DIR" || return 1
  fi
  log "updated $old_sha -> $(git -C "$INSTALL_DIR" rev-parse --short HEAD)"
}

shutdown() {
  log "SIGTERM: draining worker"
  [ "$child" -gt 0 ] && kill -TERM "$child" 2>/dev/null
  [ "$inhibit_pid" -gt 0 ] && kill "$inhibit_pid" 2>/dev/null
  [ "$child" -gt 0 ] && wait "$child" 2>/dev/null
  exit 0
}

main() {
  load_env
  mkdir -p "$INSTALL_DIR/logs"
  start_sleep_inhibit
  CHROME_PATH="$(resolve_chrome_path)" || { log "FATAL: no Chromium/Chrome found"; exit 1; }
  export CHROME_PATH
  FLEET_HOME_IP="$(detect_egress_ip)"
  export FLEET_HOME_IP
  export APPLYPILOT_INBOX_AUTH=1
  export APPLYPILOT_INBOX_AUTH_MODE=relay
  export APPLYPILOT_ENABLE_GMAIL_MCP=0
  log "egress=$FLEET_HOME_IP chrome=$CHROME_PATH branch=$APPLYPILOT_BRANCH"

  trap shutdown TERM INT

  while true; do
    if updates_available; then
      apply_update || log "WARN: update failed; running current code"
    fi

    fallback_args=()
    if [ -n "${WORKER_FALLBACK_AGENT:-}" ]; then
      fallback_args=(--fallback-agent "${WORKER_FALLBACK_AGENT}")
    elif [ "${WORKER_AGENT:-claude}" = "claude" ]; then
      fallback_args=(--fallback-agent "codex")
    fi

    "$INSTALL_DIR/.venv/bin/applypilot-fleet-apply" \
      --worker-id "${WORKER_LABEL:-mint}-${WORKER_SLOT:-0}" \
      --agent "${WORKER_AGENT:-claude}" \
      --model "${WORKER_MODEL:-sonnet}" \
      --machine-owner "${FLEET_MACHINE_OWNER:-mint}" \
      "${fallback_args[@]}" &
    child=$!
    log "worker started pid=$child id=${WORKER_LABEL:-mint}-${WORKER_SLOT:-0}"

    waited=0
    while kill -0 "$child" 2>/dev/null; do
      sleep 60
      waited=$((waited + 60))
      if [ "$waited" -ge "${UPDATE_CHECK_SECONDS:-21600}" ]; then
        waited=0
        if updates_available; then
          log "update available: draining worker after current job"
          kill -TERM "$child" 2>/dev/null
          wait "$child" 2>/dev/null
          break
        fi
      fi
    done

    wait "$child" 2>/dev/null
    child=0
    log "worker exited; restart in ${RESTART_BACKOFF_SECONDS:-30}s"
    sleep "${RESTART_BACKOFF_SECONDS:-30}"
  done
}

main "$@"
EOF
  chmod +x "$INSTALL_DIR/run-worker-linux.sh"
}

write_systemd_service() {
  local service_file
  service_file="/etc/systemd/system/${SERVICE_NAME}.service"
  say "Installing systemd service: $SERVICE_NAME"
  sudo tee "$service_file" >/dev/null <<EOF
[Unit]
Description=ApplyPilot Linux Fleet Worker (${WORKER_LABEL}-${WORKER_SLOT})
After=network-online.target tailscaled.service
Wants=network-online.target tailscaled.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment=HOME=$HOME
ExecStart=$INSTALL_DIR/run-worker-linux.sh
Restart=always
RestartSec=15
KillSignal=SIGTERM
TimeoutStopSec=900

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable --now "$SERVICE_NAME" >/dev/null
}

main() {
  require_normal_user
  check_platform
  need_sudo
  install_packages
  install_node22
  install_tailscale
  install_operator_ssh_key
  install_agent_clis
  ensure_deploy_key
  clone_or_update_repo
  install_python_runtime
  prompt_runtime_values
  write_pgpass
  write_env_file
  hydrate_assets
  write_worker_wrapper
  write_systemd_service

  say "Linux fleet worker installed."
  say "Status: sudo systemctl status ${SERVICE_NAME} --no-pager"
  say "Logs  : journalctl -u ${SERVICE_NAME} -f"
  say "Worker: tail -f ${INSTALL_DIR}/.applypilot/logs/worker-${WORKER_SLOT}.log"
}

main "$@"
