# run-fleet-crash-liveness.ps1 [-Once] [-Interval 1800] [-Limit 25] [-OlderDays 7] [-RefreshDays 7] [-LogPath <path>]
# Refresh public-posting liveness for old crash rows. This stamps evidence only; it never
# retries jobs, changes apply outcomes, or changes the fleet gate.
param(
  [switch]$Once,
  [int]$Interval = 1800,
  [int]$Limit = 25,
  [int]$OlderDays = 7,
  [int]$RefreshDays = 7,
  [string]$LogPath = ".fleet-logs\crash-liveness.log"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$LogPath = if ([System.IO.Path]::IsPathRooted($LogPath)) { $LogPath } else { Join-Path $ProjectRoot $LogPath }
$logDir = Split-Path -Parent $LogPath
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

if ($Interval -lt 60) { throw "Interval must be at least 60 seconds." }
if ($Limit -lt 1 -or $Limit -gt 1000) { throw "Limit must be between 1 and 1000." }
if ($OlderDays -lt 0 -or $RefreshDays -lt 0) { throw "OlderDays and RefreshDays cannot be negative." }

if (-not $env:FLEET_PG_DSN) {
  throw "FLEET_PG_DSN is not set. Supply an explicit mapped per-node fleet DSN (never the postgres admin or the shared fleet_worker role); refusing to start before any database action."
}
$env:APPLYPILOT_FLEET_DSN = $env:FLEET_PG_DSN
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$exe = $null
foreach ($d in @(".\.conda-env\Scripts", ".\.venv\Scripts")) {
  $cand = Join-Path $d "applypilot-fleet-apply-home.exe"
  if (Test-Path $cand) { $exe = (Resolve-Path $cand).Path; break }
}

$python = $null
if (-not $exe) {
  foreach ($p in @(".\.conda-env\python.exe", ".\.venv\Scripts\python.exe")) {
    if (Test-Path $p) { $python = (Resolve-Path $p).Path; break }
  }
}
if (-not $exe -and -not $python) {
  throw "applypilot-fleet-apply-home not found and no repo Python found -- run 'pip install -e .' first."
}

function Invoke-LivenessPass {
  $margs = @(
    "--dsn", $env:FLEET_PG_DSN,
    "crash-liveness",
    "--older-days", $OlderDays,
    "--refresh-days", $RefreshDays,
    "--limit", $Limit,
    "--execute"
  )
  $started = Get-Date
  $header = "[$($started.ToString('o'))] start older_days=$OlderDays refresh_days=$RefreshDays limit=$Limit once=$($Once.IsPresent)"
  Add-Content -LiteralPath $LogPath -Value $header
  Write-Host "[fleet-crash-liveness] older_days=$OlderDays refresh_days=$RefreshDays limit=$Limit once=$($Once.IsPresent)"
  $output = ""
  $code = 0
  try {
    if ($exe) {
      $output = (& $exe @margs 2>&1 | Out-String)
    } else {
      $output = (& $python -m applypilot.fleet.apply_home_main @margs 2>&1 | Out-String)
    }
    $code = $LASTEXITCODE
  } catch {
    $output = ($_ | Out-String)
    $code = 1
  }
  if ($output) {
    Add-Content -LiteralPath $LogPath -Value $output.TrimEnd()
    Write-Host $output.TrimEnd()
  }
  Add-Content -LiteralPath $LogPath -Value "[$((Get-Date).ToString('o'))] exit=$code"
  if ($code -ne 0) { exit $code }
}

if ($Once) {
  Invoke-LivenessPass
} else {
  while ($true) {
    Invoke-LivenessPass
    Start-Sleep -Seconds $Interval
  }
}
