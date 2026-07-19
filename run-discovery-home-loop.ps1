# run-discovery-home-loop.ps1  [-Seed] [-Config <path>] [-IntervalSec 120] [-Once]
#
#   Run on the HOME box. This is the "pipe into the pipeline" half of fleet
#   discovery. It (optionally) seeds search_tasks from your searches config, then
#   loops forever pulling postings that the residential discovery workers have
#   staged in Postgres (discovered_postings) into the LIVE brain (applypilot.db)
#   via store_jobspy_results (dedups by jobs.url). Once a posting is in the brain,
#   the normal pipeline (enrich -> score -> audit -> ...) processes it like any
#   other discovered job.
#
#   The residential SCRAPE half runs on machine 2:  .\run-fleet-discovery.ps1
#
#   Parameters:
#     -Seed         run `expand` once at start to seed search_tasks from -Config
#     -Config       searches YAML/JSON (default .applypilot\searches.yaml; point
#                   at .applypilot\searches_tuned.yaml to use your curated 41-query
#                   set -- the fleet expand path reads it directly, so the live
#                   tool's "searches_tuned is ignored" config bug does NOT apply here)
#     -IntervalSec  seconds between pulls (default 120)
#     -Once         seed (if -Seed) + a single pull, then exit (good for cron)
#
#   FULL LOOP, three commands:
#     1. (home)     .\run-discovery-home-loop.ps1 -Seed     <- seed + keep ingesting
#     2. (machine2) .\run-fleet-discovery.ps1               <- scrape + stage
#     3. profit: new jobs flow into applypilot.db automatically
param(
  [switch]$Seed,
  [string]$Config = ".applypilot\searches.yaml",
  [int]$IntervalSec = 120,
  [switch]$Once
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# home exe lives in .conda-env on the home box (.venv on a bootstrapped machine)
$exe = $null
foreach ($d in @(".\.conda-env\Scripts", ".\.venv\Scripts")) {
  $cand = Join-Path $d "applypilot-fleet-discovery-home.exe"
  if (Test-Path $cand) { $exe = (Resolve-Path $cand).Path; break }
}
if (-not $exe) { throw "applypilot-fleet-discovery-home not found in .conda-env or .venv." }

# DSN: an explicit mapped fleet DSN must be supplied in FLEET_PG_DSN (the setup script
# persists it). Never fall back to a local admin (the postgres admin) or the shared fleet_worker.
if (-not $env:FLEET_PG_DSN) { throw "FLEET_PG_DSN is not set. Supply an explicit mapped fleet DSN (never the postgres admin or the shared fleet_worker role); refusing to start before any database action." }
$env:APPLYPILOT_FLEET_DSN = $env:FLEET_PG_DSN
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

# CRITICAL: `pull` writes the SQLite brain at config.DB_PATH. That MUST be the LIVE
# local brain (AppData), NOT the stale OneDrive backup, or postings ingest into the
# wrong DB. Prefer an inherited value; else the persisted User var; else the known
# live path -- and refuse to run if the file isn't there.
if (-not $env:APPLYPILOT_DB_PATH) {
  $env:APPLYPILOT_DB_PATH = [Environment]::GetEnvironmentVariable("APPLYPILOT_DB_PATH","User")
}
if (-not $env:APPLYPILOT_DB_PATH) {
  $env:APPLYPILOT_DB_PATH = Join-Path $env:LOCALAPPDATA "ApplyPilot\applypilot.db"
  Write-Host "[discovery-home] APPLYPILOT_DB_PATH was unset; defaulted to $env:APPLYPILOT_DB_PATH" -ForegroundColor Yellow
}
if (-not (Test-Path $env:APPLYPILOT_DB_PATH)) {
  throw "brain not found at APPLYPILOT_DB_PATH=$($env:APPLYPILOT_DB_PATH) -- refusing to pull into a missing/wrong DB."
}
Write-Host "[discovery-home] brain: $($env:APPLYPILOT_DB_PATH)" -ForegroundColor Gray
Write-Host "[discovery-home] DSN:   $($env:FLEET_PG_DSN)" -ForegroundColor Gray

if ($Seed) {
  if (-not (Test-Path $Config)) { throw "searches config not found: $Config" }
  Write-Host "[discovery-home] seeding search_tasks from $Config ..." -ForegroundColor Cyan
  & $exe expand --config $Config
}

Write-Host "[discovery-home] ingesting staged postings into the brain every ${IntervalSec}s (Ctrl-C to stop) ..." -ForegroundColor Cyan
while ($true) {
  $ts = Get-Date -Format "HH:mm:ss"
  try { $out = (& $exe pull 2>&1) -join " "; Write-Host "[$ts] $out" }
  catch { Write-Host "[$ts] pull error: $_" -ForegroundColor Yellow }
  if ($Once) { break }
  Start-Sleep -Seconds $IntervalSec
}
