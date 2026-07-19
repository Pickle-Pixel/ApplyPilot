# run-fleet-doctor.ps1 [-Once] [-Interval 300] [-WindowMinutes 60]
#   Launch the FLEET DOCTOR on the HOME box. The Doctor reads the fleet's centralized failure
#   data and applies BOUNDED, REVERSIBLE, MONOTONICALLY-CONSERVATIVE auto-fixes only
#   (host_skip / timeout_bump / quarantine / pace_or_pause). It can NEVER un-pause, re-approve,
#   apply, raise the spend cap, lower the inter-apply gap, or touch the LinkedIn lane. Everything
#   else it finds becomes a human RECOMMENDATION in the LAN console (Diagnostics section).
#
#   -Once runs a single pass (good for a manual check); otherwise it loops every -Interval seconds.
#   Requires FLEET_PG_DSN (the setup script persists it; mirrors run-fleet-worker.ps1).
param([switch]$Once, [int]$Interval = 300, [int]$WindowMinutes = 60)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# Python env: the home box uses .conda-env; a bootstrapped machine uses .venv. Find whichever exists.
$exe = $null
foreach ($d in @(".\.conda-env\Scripts", ".\.venv\Scripts")) {
  $cand = Join-Path $d "applypilot-fleet-doctor.exe"
  if (Test-Path $cand) { $exe = (Resolve-Path $cand).Path; break }
}
if (-not $exe) { throw "applypilot-fleet-doctor not found in .conda-env or .venv -- run the setup script (pip install -e .) first." }

# DSN comes from the environment (pgpass makes it passwordless). Never echoed.
if (-not $env:FLEET_PG_DSN) {
  throw "FLEET_PG_DSN is not set. Supply an explicit mapped per-node fleet DSN (never the postgres admin or the shared fleet_worker role); refusing to start before any database action."
}
$env:APPLYPILOT_FLEET_DSN = $env:FLEET_PG_DSN
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

$margs = @("--interval", $Interval, "--window-minutes", $WindowMinutes)
if ($Once) { $margs += "--once" }
Write-Host "[fleet-doctor] window=$WindowMinutes min  interval=$Interval s  once=$($Once.IsPresent)  (conservative auto-fixes + recommendation queue)"
& $exe --dsn $env:FLEET_PG_DSN @margs
