# load-canary-remote.ps1
#   Start REMOTE (second-box) apply workers at HIGH concurrency -- the workhorse of the fleet.
#
#   The canary is armed ONCE on the HOME box (load-canary-home.ps1) and lives in the shared
#   Postgres fleet_config. This box does NOT arm anything and NEVER touches the SQLite brain --
#   it only LEASES approved jobs from the home Postgres, applies them, and heartbeats back.
#   Because the canary + spend-cap in fleet_config bound the WHOLE fleet, you can run as many
#   workers here as the box's RAM allows without changing the blast radius.
#
#   Prereqs (one-time, via setup-fleet-worker.ps1 from machine2-bundle.zip):
#     * same LAN as the home box; home Postgres reachable (pg_hba 192.168.1.0/24, firewall :5432)
#     * .venv with `pip install -e .`, Playwright Chromium installed, codex/claude logged in
#     * pgpass.conf has the home Postgres password
#
#   RUN THIS IN YOUR OWN POWERSHELL on the remote box.
#
#   Examples:
#     .\load-canary-remote.ps1                       # auto-size workers from this box's RAM
#     .\load-canary-remote.ps1 -Count 8              # force 8 workers
#     .\load-canary-remote.ps1 -HomeIp 100.90.104.99 -Label m2 -Agent codex
#
param(
  [int]$Count          = 0,                    # 0 = auto-size from TOTAL RAM (this box is the workhorse)
  [string]$Agent       = "codex",              # codex rides the ChatGPT plan (no per-apply $)
  [string]$Label       = "m2",                 # MUST differ from home so worker-ids don't collide
  [string]$HomeIp      = "100.90.104.99",      # home box Tailscale IP (where Postgres lives)
  [string]$MachineOwner= $env:COMPUTERNAME,    # heartbeat attribution (captcha inbox / monitoring)
  [string]$EgressIp    = ""                    # this box's residential egress IP (governor key); auto-detected if blank
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# --- 1. point at the HOME Postgres over the LAN (NOT localhost) -----------------------------
# The mapped per-node fleet DSN (with its own credential) must be supplied in FLEET_PG_DSN.
# Never construct a local-admin (the postgres admin) or shared fleet_worker DSN here.
if (-not $env:FLEET_PG_DSN) {
  throw "FLEET_PG_DSN is not set. Export an explicit mapped per-node fleet DSN (never the postgres admin or the shared fleet_worker role) before starting remote workers."
}
$Dsn = $env:FLEET_PG_DSN
$env:APPLYPILOT_FLEET_DSN = $Dsn

Write-Host "Checking home Postgres at ${HomeIp}:5432 ..." -ForegroundColor Cyan
if (-not (Test-NetConnection -ComputerName $HomeIp -Port 5432 -InformationLevel Quiet -WarningAction SilentlyContinue)) {
  throw "Cannot reach home Postgres at ${HomeIp}:5432. Verify: same LAN, pg_hba allows 192.168.1.0/24, firewall opens TCP 5432, pgpass has the password."
}

# --- 2. per-machine identity (fixes the heartbeat attribution + per-IP governor gap) --------
$env:FLEET_MACHINE_OWNER = $MachineOwner
if (-not $EgressIp) {
  try { $EgressIp = (Invoke-RestMethod -Uri "https://api.ipify.org" -TimeoutSec 5).Trim() } catch { $EgressIp = "" }
}
if ($EgressIp) {
  $env:FLEET_HOME_IP = $EgressIp
  Write-Host "egress IP (per-IP governor key): $EgressIp" -ForegroundColor DarkCyan
} else {
  Write-Host "WARNING: could not detect egress IP; per-IP throttle stays inert (home_ip=0.0.0.0). Pass -EgressIp <ip>." -ForegroundColor Yellow
}

# --- 3. memory-aware sizing: scale to THIS box's RAM ---------------------------------------
$totalGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)
$perWorkerGB = 1.5; $reserveGB = 4.0
if ($Count -le 0) {
  $Count = [int][math]::Floor(($totalGB - $reserveGB) / $perWorkerGB)
  if ($Count -lt 1)  { $Count = 1 }
  if ($Count -gt 10) { $Count = 10 }    # run-fleet-workers.ps1 hard cap
}
Write-Host ("REMOTE box '{0}': {1} GB RAM -> {2} worker(s) @ ~{3} GB each (reserve {4} GB)" -f $Label,$totalGB,$Count,$perWorkerGB,$reserveGB) -ForegroundColor Cyan

# --- 4. start REMOTE workers (distinct -Label keeps worker-ids unique fleet-wide) ----------
& (Join-Path $Root "run-fleet-workers.ps1") -Count $Count -Agent $Agent -Label $Label

Write-Host ""
Write-Host ("REMOTE workers up: {0} x '{1}' (agent={2}), leasing from {3}." -f $Count,$Label,$Agent,$HomeIp) -ForegroundColor Green
Write-Host "The canary + spend cap on the HOME box bound the whole fleet -- this box just adds throughput." -ForegroundColor Green
