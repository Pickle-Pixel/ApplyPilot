<#
================================================================================
  setup-fleet-discovery.ps1
  Bootstrap a FRESH Windows machine into an ApplyPilot DISCOVERY worker:
  it leases search tasks from the home box's Postgres over Tailscale or your private LAN,
  scrapes job boards with JobSpy, and stages the raw postings back to Postgres.
  The home box alone ingests them into the brain (see run-discovery-home-loop.ps1).

  Discovery is PURE SCRAPE -- no apply agent (no Codex/Claude), no Playwright
  login, no resume, no brain write. So this is far lighter than the apply-worker
  bootstrap: Git + Python + the package + JobSpy, nothing else.

  WHY a dedicated script (don't reuse setup-fleet-worker / setup-fleet-machine):
  JobSpy (python-jobspy) is NOT a normal dependency -- its metadata pins an exact
  numpy that breaks pip's resolver, so it must be installed with
  `pip install --no-deps python-jobspy`. The apply/monitor bootstraps SKIP that
  step, so a worker set up with them imports fine but dies the moment it scrapes.

  PREREQS:
    - This machine can reach the home box's Tailscale IP, or both boxes are on the same private LAN.
    - The home box already serves Postgres to the LAN -- it does: pg_hba +
      firewall rule + listen_addresses='*' are all set. NOTHING to do there.
    - This should NOT be the machine that holds your LinkedIn / apply session.
      Keep scraping IPs separate from the apply IP (account-safety rule).

  RUN (normal PowerShell, NOT admin):
      powershell -ExecutionPolicy Bypass -File .\setup-fleet-discovery.ps1
================================================================================
#>
param([string]$HomeIp = "100.90.104.99", [string]$InstallDir = "C:\ApplyPilot", [string]$FleetPgDsn = $env:FLEET_PG_DSN)
$ErrorActionPreference = "Stop"
function Say($m,$c="White"){ Write-Host $m -ForegroundColor $c }
function Refresh-Path { $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User") }

Say "`n=== ApplyPilot fleet -- DISCOVERY worker bootstrap (pure scrape) ===" Cyan

# --- 0. On the private LAN? ---
$gw = (Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway } | Select-Object -First 1).IPv4DefaultGateway.NextHop
Say "primary gateway: $gw" Gray
if ($gw -notlike '192.168.*' -and $gw -notlike '10.*' -and $gw -notlike '172.*') {
  Say "WARNING: not on a private LAN -- connect to the same router as the home box first." Yellow
  if ((Read-Host "  continue anyway? (y/n)") -ne 'y') { exit 1 }
}

# --- 1. Reach the home box's Postgres BEFORE the heavy install ---
Say "`n[1/6] checking ${HomeIp}:5432 is reachable ..." Cyan
if (-not (Test-NetConnection $HomeIp -Port 5432 -WarningAction SilentlyContinue).TcpTestSucceeded) {
  Say "  cannot reach ${HomeIp}:5432 -- confirm same router + the home box is up." Yellow
  if ((Read-Host "  continue anyway? (y/n)") -ne 'y') { exit 1 }
} else { Say "  reachable." Green }

# --- 2. Tooling: prefer what's already installed; winget only as a fallback ---
#   winget is absent on plenty of Windows builds (Server, LTSC, fresh images). Don't hard-depend
#   on it: use an existing git / python / project venv if present, fall back to winget if we have
#   it, and otherwise print manual download links instead of crashing.
Say "`n[2/6] locating Git + Python (reusing what's installed) ..." Cyan
$haveWinget = [bool](Get-Command winget -ErrorAction SilentlyContinue)

function Ensure-Tool($name, $wingetId, [string[]]$cmds) {
  foreach ($c in $cmds) { if (Get-Command $c -ErrorAction SilentlyContinue) { Say "  [ok] $name" Green; return $true } }
  if ($haveWinget) {
    Say "  installing $name via winget ..."; winget install -e --id $wingetId --accept-source-agreements --accept-package-agreements; Refresh-Path
    foreach ($c in $cmds) { if (Get-Command $c -ErrorAction SilentlyContinue) { return $true } }
  }
  return $false
}
function Find-Python {
  foreach ($v in @((Join-Path $InstallDir ".venv\Scripts\python.exe"), (Join-Path $InstallDir ".conda-env\Scripts\python.exe"))) {
    if (Test-Path $v) { return (Resolve-Path $v).Path }
  }
  foreach ($c in @("py","python","python3")) { $g = Get-Command $c -ErrorAction SilentlyContinue; if ($g) { return $g.Source } }
  return $null
}

$gitOk = Ensure-Tool "Git" "Git.Git" @("git")
$pyExe = Find-Python
if (-not $pyExe -and $haveWinget) {
  Say "  installing Python 3.12 via winget ..."; winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements; Refresh-Path
  $pyExe = Find-Python
}
if (-not $gitOk -or -not $pyExe) {
  Say "`n  Missing prerequisite(s) and winget is unavailable on this machine:" Yellow
  if (-not $gitOk) { Say "    - Git:    https://git-scm.com/download/win" Yellow }
  if (-not $pyExe) { Say "    - Python: https://www.python.org/downloads/  (tick 'Add python.exe to PATH')" Yellow }
  Say "  Install the above, open a NEW PowerShell, and re-run this script." Yellow
  exit 1
}
Say "  python: $pyExe" Green

# --- 3. Clone the fleet repo (GitHub sign-in if prompted) ---
Say "`n[3/6] cloning the fleet repo -> $InstallDir ..." Cyan
$repo = "https://github.com/thefulmination/applypilot-private.git"
if (-not (Test-Path (Join-Path $InstallDir ".git"))) { git clone $repo $InstallDir } else { Say "  [ok] already cloned" Green }
Set-Location $InstallDir

# --- 4. Python venv + package + JobSpy (the separate --no-deps step) ---
Say "`n[4/6] Python env + package + JobSpy (a few minutes) ..." Cyan
# Reuse an existing project venv/conda if present (machine 2 may already be an apply worker);
# otherwise create one with whatever Python step 2 found.
if (-not (Test-Path ".\.venv") -and -not (Test-Path ".\.conda-env")) { & $pyExe -m venv .venv }
$py = @(".\.venv\Scripts\python.exe", ".\.conda-env\Scripts\python.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
$py = (Resolve-Path $py).Path
& $py -m pip install --upgrade pip --quiet
& $py -m pip install -e . --quiet
& $py -m pip install "psycopg[binary]" pyyaml --quiet
# JobSpy: pinned-numpy metadata breaks the resolver, so install it WITHOUT deps
# (the package already declares jobspy's import-time runtime deps).
& $py -m pip install --no-deps python-jobspy --quiet
& $py -c "import jobspy; print('  jobspy import OK')"
& $py -m pip check

# --- 5. Postgres connectivity: require an explicit mapped fleet DSN (never postgres) ---
# The mapped DSN carries its own credential; this script never prompts for the home box's
# postgres password nor writes a postgres pgpass. Fail closed before any database action.
Say "`n[5/6] Postgres connection ..." Cyan
if (-not $FleetPgDsn) {
  throw "FLEET_PG_DSN is not set. Provide an explicit mapped per-node fleet DSN (with its own password; never the postgres admin or the shared fleet_worker role) via -FleetPgDsn or the FLEET_PG_DSN environment variable."
}
if ($FleetPgDsn -match '(?i)\buser\s*=\s*(?:postgres|fleet_worker)\b') {
  throw "setup-fleet-discovery: fleet DSN selects an admin/shared principal. Supply an explicit mapped fleet DSN."
}
$dsn = $FleetPgDsn
[Environment]::SetEnvironmentVariable("FLEET_PG_DSN", $dsn, "User")
[Environment]::SetEnvironmentVariable("APPLYPILOT_FLEET_DSN", $dsn, "User")
$env:FLEET_PG_DSN = $dsn; $env:APPLYPILOT_FLEET_DSN = $dsn
& $py -c "from applypilot.apply import pgqueue; pgqueue.connect(); print('  CONNECTED to the fleet Postgres')"

# --- 6. Done ---
Say "`n[6/6] done." Green
Say "`n=== DISCOVERY BOOTSTRAP COMPLETE ===" Green
Say "Open a NEW PowerShell (so it picks up FLEET_PG_DSN), then start scraping:" Cyan
Say "  cd $InstallDir"
Say "  .\run-fleet-discovery.ps1 -Label m2"
Say "`nIt idles until the HOME box seeds search tasks. On the home box run once:" Gray
Say "  .\run-discovery-home-loop.ps1 -Seed" Gray
Say "which seeds the searches and then continuously pulls scraped postings into the brain." Gray
