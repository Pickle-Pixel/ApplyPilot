# run-compute-home-loop.ps1 [-ScoreFloor 7] [-IntervalSec 900] [-Once] [-Task score]
#
#   Run on the HOME box. The "feed + harvest" half of the compute/scoring lane. Each cycle it
#   PUSHES brain jobs worth an LLM pass (COALESCE(audit_score, fit_score) >= -ScoreFloor, best
#   first) into compute_queue, then PULLS the advisory results the m4 scorers wrote back into the
#   live brain (research_fit_score / research_decision -- ADVISORY, never demotes fit_score).
#
#   Score floor 7 is deliberate: only jobs the cheap first-pass scorer already ranked >=7 are
#   worth the paid LLM pass. Lower it to widen the net (e.g. -ScoreFloor 5), raise it to conserve.
#
#   The compute SCORE half runs on machine 4:  .\run-fleet-compute.ps1 -Workers 16
param(
  [int]$ScoreFloor = 7,
  [string]$Task = "score",
  [int]$IntervalSec = 900,
  [switch]$IncludeUnscored,
  [int]$UnscoredLimit = 500,
  [switch]$Once
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$exe = $null
foreach ($d in @(".\.conda-env\Scripts", ".\.venv\Scripts")) {
  $cand = Join-Path $d "applypilot-fleet-compute-home.exe"
  if (Test-Path $cand) { $exe = (Resolve-Path $cand).Path; break }
}
if (-not $exe) { throw "applypilot-fleet-compute-home not found in .conda-env or .venv." }

if (-not $env:FLEET_PG_DSN) { throw "FLEET_PG_DSN is not set. Supply an explicit mapped fleet DSN (never the postgres admin or the shared fleet_worker role); refusing to start before any database action." }
$env:APPLYPILOT_FLEET_DSN = $env:FLEET_PG_DSN
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

# CRITICAL (mirrors run-discovery-home-loop): `push` reads the SQLite brain at config.DB_PATH.
# That MUST be the LIVE local brain (AppData), not the stale OneDrive backup, or it queues the
# wrong jobs. Prefer inherited -> persisted User var -> known live path; refuse if absent.
if (-not $env:APPLYPILOT_DB_PATH) {
  $env:APPLYPILOT_DB_PATH = [Environment]::GetEnvironmentVariable("APPLYPILOT_DB_PATH", "User")
}
if (-not $env:APPLYPILOT_DB_PATH) {
  $env:APPLYPILOT_DB_PATH = Join-Path $env:LOCALAPPDATA "ApplyPilot\applypilot.db"
  Write-Host "[compute-home] APPLYPILOT_DB_PATH was unset; defaulted to $env:APPLYPILOT_DB_PATH" -ForegroundColor Yellow
}
if (-not (Test-Path $env:APPLYPILOT_DB_PATH)) {
  throw "brain not found at APPLYPILOT_DB_PATH=$($env:APPLYPILOT_DB_PATH) -- refusing to push from a missing/wrong DB."
}
Write-Host "[compute-home] brain: $($env:APPLYPILOT_DB_PATH)" -ForegroundColor Gray
Write-Host "[compute-home] DSN:   $($env:FLEET_PG_DSN)  task=$Task floor=$ScoreFloor" -ForegroundColor Gray

function Invoke-ComputeHomeStep {
  param(
    [string]$Name,
    [string[]]$ArgsList
  )

  $ts = Get-Date -Format "HH:mm:ss"
  $oldEap = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    $out = & $exe @ArgsList 2>&1
    $code = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $oldEap
  }

  if ($code -ne 0) {
    Write-Host "[$ts] $Name FAILED (exit $code):" -ForegroundColor Yellow
    foreach ($line in $out) { Write-Host $line }
    return
  }

  Write-Host "[$ts] $(($out | ForEach-Object { "$_" }) -join ' ')"
}

Write-Host "[compute-home] push (fill compute_queue) + pull (harvest results) every ${IntervalSec}s (Ctrl-C to stop) ..." -ForegroundColor Cyan
while ($true) {
  Invoke-ComputeHomeStep -Name "push" -ArgsList @("push", "--task", $Task, "--score-floor", "$ScoreFloor")
  if ($IncludeUnscored) {
    Invoke-ComputeHomeStep -Name "push-unscored" -ArgsList @("push", "--task", "score", "--unscored-only", "--limit", "$UnscoredLimit")
  }
  Invoke-ComputeHomeStep -Name "pull" -ArgsList @("pull")
  if ($Once) { break }
  Start-Sleep -Seconds $IntervalSec
}
