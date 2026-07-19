<#
================================================================================
  setup-fleet-machine.ps1
  Set up a SECOND machine to run Codex and talk to the ApplyPilot fleet Postgres,
  preferably over Tailscale so Wi-Fi/Ethernet LAN address changes do not break it.

  The home box's current Tailscale IP is 100.90.104.99. Both machines must be on
  the same Tailscale tailnet, or you must override the prompt below with a valid
  private LAN address for the home box.

  --------------------------------------------------------------------------------
  STEP 0 -- DO THIS ONCE ON THE HOME BOX (the machine that has Postgres), AS ADMIN:
  --------------------------------------------------------------------------------
    # Allow Postgres from the private LAN only (a public source can't match this rule,
    # and TCP can't spoof a 192.168.x source, so this is safe even with the public
    # Ethernet still plugged in):
    Add-Content "C:\Program Files\PostgreSQL\18\data\pg_hba.conf" "host all all 192.168.1.0/24 scram-sha-256"
    New-NetFirewallRule -DisplayName "PostgreSQL LAN" -Direction Inbound -Protocol TCP -LocalPort 5432 -Action Allow -RemoteAddress 192.168.1.0/24
    Restart-Service postgresql-x64-18
    # (listen_addresses is already '*'. pg_hba + firewall limit reach to the LAN.)

  --------------------------------------------------------------------------------
  THEN, ON THIS (SECOND) MACHINE -- also on the 192.168.1.x router -- run:
      powershell -ExecutionPolicy Bypass -File .\setup-fleet-machine.ps1
  --------------------------------------------------------------------------------
#>
param([string]$FleetPgDsn = $env:FLEET_PG_DSN)

$ErrorActionPreference = "Stop"
function Say($m,$c="White"){ Write-Host $m -ForegroundColor $c }
function Set-CapSolverKey([string]$InstallDir) {
  $capKey = (Read-Host "CapSolver API key (Enter to skip)").Trim()
  if (-not $capKey) { Say "  SKIPPED -- set CAPSOLVER_API_KEY before launching apply workers." Yellow; return }

  [Environment]::SetEnvironmentVariable("CAPSOLVER_API_KEY", $capKey, "User")
  $env:CAPSOLVER_API_KEY = $capKey
  $appDir = Join-Path $InstallDir ".applypilot"
  New-Item -ItemType Directory -Force -Path $appDir | Out-Null
  $envFile = Join-Path $appDir ".env"
  $lines = @()
  if (Test-Path $envFile) { $lines = @(Get-Content $envFile | Where-Object { $_ -notmatch '^CAPSOLVER_API_KEY=' }) }
  $lines += "CAPSOLVER_API_KEY=$capKey"
  Set-Content -Path $envFile -Value $lines -Encoding ascii
  Say "  [ok] CapSolver key saved for this Windows user and $envFile" Green
}

Say "`n=== ApplyPilot fleet -- second-machine setup ===" Cyan

# --- Prereqs this script can't automate (interactive sign-ins) ---
Say "`nBefore continuing, make sure:" Yellow
Say "  [ ] This machine can reach the home box over Tailscale or the same private LAN"
Say "  [ ] The home box STEP 0 above is done (PG opened to the LAN, service restarted)"
Say "  [ ] Codex CLI is installed + signed in here (npm i -g @openai/codex ; codex login)"
if ((Read-Host "All set? (y/n)") -ne 'y') { Say "Finish those, then re-run." Yellow; exit 1 }

# Sanity: are WE on the private LAN?
$gw = (Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway } | Select-Object -First 1).IPv4DefaultGateway.NextHop
if ($gw -notlike '192.168.*' -and $gw -notlike '10.*' -and $gw -notlike '172.*') {
  Say "  WARNING: your primary gateway is $gw (not private). Connect this machine to the same router as the home box first." Yellow
  if ((Read-Host "Continue anyway? (y/n)") -ne 'y') { exit 1 }
}

# --- 1. Base tools via winget (skips anything already present) ---
function Ensure($name,$id,$cmd){
  if (Get-Command $cmd -ErrorAction SilentlyContinue){ Say "  [ok] $name" Green }
  else { Say "  installing $name ..." ; winget install -e --id $id --accept-source-agreements --accept-package-agreements }
}
Ensure "Git"         "Git.Git"            "git"
Ensure "Python 3.12" "Python.Python.3.12" "python"
Ensure "Node.js LTS" "OpenJS.NodeJS.LTS"  "node"

# --- 2. Inputs ---
$homeIp = (Read-Host "Home box IP [default 100.90.104.99]").Trim()
if (-not $homeIp){ $homeIp = "100.90.104.99" }
$installDir = (Read-Host "Folder to install the code [default C:\ApplyPilot]").Trim()
if (-not $installDir){ $installDir = "C:\ApplyPilot" }

# --- 3. Reachability check BEFORE the heavy install ---
Say "`n  checking the home box is reachable ..." Cyan
if (-not (Test-Connection -ComputerName $homeIp -Count 2 -Quiet)) {
  Say "  Could not ping $homeIp. Confirm Tailscale/private LAN connectivity and the IP is right." Yellow
  if ((Read-Host "Continue anyway? (y/n)") -ne 'y') { exit 1 }
}

# --- 4. Get the code (private repo -- you may be prompted to sign in to GitHub) ---
$repo = "https://github.com/thefulmination/applypilot-private.git"
if (-not (Test-Path (Join-Path $installDir ".git"))){
  Say "  cloning the fleet repo -> $installDir (sign in to GitHub if asked) ..."
  git clone $repo $installDir
} else { Say "  [ok] repo already at $installDir" Green }
Set-Location $installDir

Say "`n  configuring CapSolver CAPTCHA service ..." Cyan
Set-CapSolverKey $installDir

# --- 5. Python env + install (bridge needs psycopg + mcp; pyyaml for config) ---
if (-not (Test-Path ".\.venv")){ python -m venv .venv }
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
& $py -m pip install --upgrade pip --quiet
& $py -m pip install -e . --quiet
& $py -m pip install "psycopg[binary]" mcp pyyaml --quiet

# --- 6. Postgres connectivity: require an explicit mapped fleet DSN (never postgres) ---
# The mapped per-node DSN carries its OWN credential; this script never prompts for the
# home box's postgres password nor writes a postgres pgpass. Fail closed if it is absent
# or selects an admin/shared principal, before any database action.
if (-not $FleetPgDsn) {
  throw "FLEET_PG_DSN is not set. Provide an explicit mapped per-node fleet DSN (with its own password; never the postgres admin or the shared fleet_worker role) via -FleetPgDsn or the FLEET_PG_DSN environment variable."
}
if ($FleetPgDsn -match '(?i)\buser\s*=\s*(?:postgres|fleet_worker)\b') {
  throw "setup-fleet-machine: fleet DSN selects an admin/shared principal. Supply an explicit mapped fleet DSN."
}
$dsn = $FleetPgDsn
# FLEET_PG_DSN -> the fleet CLIs + Codex bridge; APPLYPILOT_FLEET_DSN -> the low-level connect() fallback.
[Environment]::SetEnvironmentVariable("FLEET_PG_DSN", $dsn, "User")
[Environment]::SetEnvironmentVariable("APPLYPILOT_FLEET_DSN", $dsn, "User")
$env:FLEET_PG_DSN = $dsn
$env:APPLYPILOT_FLEET_DSN = $dsn

# --- 7. Test the actual Postgres connection ---
Say "`n  testing Postgres connectivity ..." Cyan
& $py -c "from applypilot.apply import pgqueue; pgqueue.connect(); print('CONNECTED to the fleet Postgres')"

# --- 8. Wire the Codex bridge into ~/.codex/config.toml (single-quoted = literal Windows paths) ---
$codexDir = Join-Path $env:USERPROFILE ".codex"
New-Item -ItemType Directory -Force -Path $codexDir | Out-Null
$block = @"
[mcp_servers.applypilot-fleet]
command = '$py'
args = ["-m", "applypilot.fleet.codex_bridge"]
cwd = '$installDir'
enabled = true

[mcp_servers.applypilot-fleet.env]
FLEET_PG_DSN = '$dsn'
"@
$cfg = Join-Path $codexDir "config.toml"
if (Test-Path $cfg){ Add-Content $cfg "`n$block"; Say "  appended the fleet bridge to existing ~/.codex/config.toml" Green }
else { Set-Content $cfg $block -Encoding utf8; Say "  wrote ~/.codex/config.toml" Green }

Say "`n=== DONE ===" Green
Say "Open Codex on this machine and ask:  'what is the applypilot fleet status?'"
Say "It should call fleet_status and report the live numbers from the home box's Postgres."
Say "(If a tool says 'FLEET_PG_DSN is not set', fully restart Codex so it re-reads config.toml.)"
Say "`nTo also have THIS machine APPLY jobs (not just monitor), that's the heavier worker setup"
Say "(Playwright browser + an authed apply agent) -- ask and I'll extend this."
