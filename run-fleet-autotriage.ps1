# run-fleet-autotriage.ps1 [-Once] [-Interval 600] [-Limit 25] [-WindowMinutes 1440]
#   Launch autonomous bounded fleet triage on the HOME box. It inspects recent ATS
#   terminal failures and lets the LLM choose only from a fixed, validated action menu.
#   The executor still performs the duplicate/apply-email guards before mutating state.
param(
  [switch]$Once,
  [int]$Interval = 600,
  [int]$Limit = 25,
  [int]$WindowMinutes = 1440,
  [switch]$DryRun,
  [switch]$DisableLlm,
  [switch]$CrashOnly
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not $env:FLEET_PG_DSN) {
  throw "FLEET_PG_DSN is not set. Supply an explicit mapped per-node fleet DSN (never the postgres admin or the shared fleet_worker role); refusing to start before any database action."
}
$env:APPLYPILOT_FLEET_DSN = $env:FLEET_PG_DSN
$env:APPLYPILOT_DB_PATH = Join-Path $env:LOCALAPPDATA "ApplyPilot\applypilot.db"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$exe = $null
foreach ($d in @(".\.conda-env\Scripts", ".\.venv\Scripts")) {
  $cand = Join-Path $d "applypilot-fleet-autotriage.exe"
  if (Test-Path $cand) { $exe = (Resolve-Path $cand).Path; break }
}

$python = $null
if (-not $exe) {
  foreach ($p in @(".\.conda-env\python.exe", ".\.venv\Scripts\python.exe")) {
    if (Test-Path $p) { $python = (Resolve-Path $p).Path; break }
  }
}
if (-not $exe -and -not $python) {
  throw "applypilot-fleet-autotriage not found and no repo Python found -- run 'pip install -e .' first."
}

$margs = @(
  "--dsn", $env:FLEET_PG_DSN,
  "--brain-path", $env:APPLYPILOT_DB_PATH,
  "--limit", $Limit,
  "--window-minutes", $WindowMinutes
)
if ($Once) {
  $margs += "--once"
} else {
  $margs += @("--interval", $Interval)
}
if (-not $DisableLlm) { $margs += "--enable-llm" }
if ($DryRun) { $margs += "--dry-run" }
if ($CrashOnly) { $margs += "--crash-only" }

Write-Host "[fleet-autotriage] limit=$Limit window=$WindowMinutes min interval=$Interval s once=$($Once.IsPresent) llm=$(-not $DisableLlm) dry_run=$($DryRun.IsPresent) crash_only=$($CrashOnly.IsPresent)"
if ($exe) {
  & $exe @margs
} else {
  & $python -m applypilot.fleet.autotriage_main @margs
}
