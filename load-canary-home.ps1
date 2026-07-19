# load-canary-home.ps1
#   Arm the OFFSITE (ATS) apply canary AND start HOME apply workers at LOW concurrency.
#
#   The home box is memory-tight (it also runs Postgres, the brain, ibgateway, etc.) and an
#   OOM hang is what killed worker home-0. So this loader deliberately keeps the worker count
#   small and auto-sizes it from FREE RAM. For the heavy lifting, run load-canary-remote.ps1
#   on the second box -- the canary you arm here is fleet-wide (shared Postgres fleet_config),
#   so the remote workers consume the SAME canary; you only arm it once, here.
#
#   RUN THIS IN YOUR OWN POWERSHELL on the home box (real environment) -- not through Claude --
#   so the worker windows open on your desktop and the Claude/Codex agent creds resolve.
#
#   Examples:
#     .\load-canary-home.ps1                       # arm canary=3, 1 home Claude worker, $25 cap
#     .\load-canary-home.ps1 -Canary 5 -Count 2    # bigger first run, 2 workers (only if RAM allows)
#     .\load-canary-home.ps1 -SkipQueuePrep        # queue already populated: just arm+approve+run
#     .\load-canary-home.ps1 -WithWatchdog         # also launch the self-healing watchdog
#
param(
  [int]$Canary        = 3,        # blast radius for THIS run; re-arms (overwrites) canary_remaining
  [int]$Count         = 1,        # home workers; keep 1-2. Pass 0 to auto-size from free RAM.
  [string]$Agent      = "claude", # home default; each Claude apply is ~$0.65-0.90
  [double]$SpendCapUsd= 25.0,     # hard $ ceiling (0 = UNCAPPED, so always set a real one)
  [int]$ScoreFloor    = 7,        # only push jobs scoring >= this into the queue
  [switch]$SkipQueuePrep,         # skip pull/push (just arm + approve + start workers)
  [switch]$WithWatchdog,          # also start the self-healing watchdog (reclaim + restart)
  [string]$Dsn = $env:FLEET_PG_DSN
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# --- 0. gate EVERY side effect behind the SAME approved runtime-launch channel + full offline
# conninfo/principal validator that the supported launchers use. There is deliberately NO
# standalone regex denylist or weaker parser here: Assert-ApplyPilotRuntimeChannel runs
# `python -m applypilot.fleet.launch_guard`, which reuses the single hardened conninfo parser
# (pgqueue.assert_mapped_principal) to reject the postgres admin, the shared fleet_worker role,
# and every URL / percent-encoded / query-string / options role-switch / ambient / indirection /
# weak-TLS bypass, and to require the exact APPLYPILOT_PG_PRINCIPAL. It fails closed BEFORE the
# queue prep, canary arm/approve, spend-cap update, or any worker launch below.
. (Join-Path $Root "scripts\runtime-launch-channel.ps1")
$runtimeChannel = Assert-ApplyPilotRuntimeChannel -RepoRoot $Root -OverrideDsn $Dsn
$Dsn = $runtimeChannel.Dsn
$env:FLEET_PG_DSN = $Dsn
$env:APPLYPILOT_FLEET_DSN = $Dsn

function Find-Exe([string]$name) {
  foreach ($d in @(".\.conda-env\Scripts", ".\.venv\Scripts")) {
    $c = Join-Path $d "$name.exe"
    if (Test-Path $c) { return (Resolve-Path $c).Path }
  }
  throw "$name not found in .conda-env or .venv -- is the home box set up (pip install -e .)?"
}

# --- memory-aware sizing: the home box OOM-crashed, so keep concurrency LOW -----------------
$totalGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)
$freeGB  = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB/1024,1)
$perWorkerGB = 1.5
if ($Count -le 0) {
  $Count = [int][math]::Floor(($freeGB - 2.0) / $perWorkerGB)
  if ($Count -lt 1) { $Count = 1 }
  if ($Count -gt 2) { $Count = 2 }     # hard HOME ceiling -- never run the home box hot
}
Write-Host ("HOME box: {0} GB total, {1} GB free now -> {2} worker(s) @ ~{3} GB each" -f $totalGB,$freeGB,$Count,$perWorkerGB) -ForegroundColor Cyan
if ($freeGB -lt ($Count * $perWorkerGB + 1.5)) {
  Write-Host ("  WARNING: only {0} GB free; {1} worker(s) want ~{2} GB. Close ibgateway/other apps, run fewer workers, or lean on the REMOTE box -- this is the OOM that hung home-0." -f $freeGB,$Count,($Count*$perWorkerGB)) -ForegroundColor Yellow
}

$homeExe = Find-Exe "applypilot-fleet-apply-home"

# Native (external .exe) commands do NOT honor $ErrorActionPreference, so a nonzero exit from the
# CLI would otherwise be ignored and the script would barrel on to the next mutation. Check
# $LASTEXITCODE after every native call and throw so NO later step (approve, spend-cap, workers)
# runs after a failed validation/update.
function Invoke-HomeCli {
  param([Parameter(Mandatory = $true)][string[]]$CliArgs, [Parameter(Mandatory = $true)][string]$What)
  & $homeExe @CliArgs
  if ($LASTEXITCODE -ne 0) {
    throw "load-canary-home: $What failed (exit $LASTEXITCODE); aborting before any later step."
  }
}

# --- 1. queue prep (touches the SQLite brain -- HOME box only) ------------------------------
if (-not $SkipQueuePrep) {
  Write-Host "Pulling applied_set + pushing eligible jobs (score >= $ScoreFloor) ..." -ForegroundColor Cyan
  Invoke-HomeCli -CliArgs @("--dsn", $Dsn, "pull") -What "queue pull"
  Invoke-HomeCli -CliArgs @("--dsn", $Dsn, "push", "--score-floor", "$ScoreFloor") -What "queue push"
}

# --- 2. ARM the canary to K (canary_enabled=TRUE, remaining=K, paused=FALSE; overwrites prior)
Write-Host "Arming canary -> remaining=$Canary (auto-pauses the fleet after $Canary applies) ..." -ForegroundColor Cyan
Invoke-HomeCli -CliArgs @("--dsn", $Dsn, "canary", "$Canary") -What "canary arm"

# --- 3. approve the pushed batch (CLI refuses unless the canary is armed) -------------------
Invoke-HomeCli -CliArgs @("--dsn", $Dsn, "approve", "--all-pushed") -What "approve"

# --- 4. hard $ ceiling (belt-and-suspenders with the canary count) --------------------------
# Routed through the SAME audited, mapped control-plane CLI as the steps above (one
# parameterized write + audit row in a single transaction) -- never a raw psql write to the
# fleet_config table. The exit code is propagated, so a least-privilege role that cannot write
# the cap aborts the run here rather than silently starting workers under an unset cap.
Write-Host "Setting hard spend cap = `$$SpendCapUsd via the mapped control-plane principal ..." -ForegroundColor Cyan
Invoke-HomeCli -CliArgs @("--dsn", $Dsn, "set-spend-cap", "$SpendCapUsd") -What "spend-cap update"
Write-Host "Spend cap set to `$$SpendCapUsd (remember: 0 would mean UNCAPPED)." -ForegroundColor Cyan

# --- 5. optional: self-healing watchdog (reclaims stale leases + restarts dead workers) -----
if ($WithWatchdog) {
  try {
    $wd = Find-Exe "applypilot-fleet-watchdog"
    Start-Process powershell.exe -ArgumentList @("-NoExit","-ExecutionPolicy","Bypass","-Command","& '$wd' --dsn '$Dsn'") -WorkingDirectory $Root
    Write-Host "Watchdog launched (would have caught the home-0 hang)." -ForegroundColor Green
  } catch { Write-Host ("Watchdog NOT started: {0}" -f $_.Exception.Message) -ForegroundColor Yellow }
}

# --- 6. start HOME workers at LOW concurrency ----------------------------------------------
Write-Host "Starting $Count HOME apply worker(s) (label=home, agent=$Agent) ..." -ForegroundColor Cyan
& (Join-Path $Root "run-fleet-workers.ps1") -Count $Count -Agent $Agent -Label home

Write-Host ""
Write-Host ("HOME canary loaded: K={0}, {1} worker(s), cap=`${2}. Watch: http://100.90.104.99:8787" -f $Canary,$Count,$SpendCapUsd) -ForegroundColor Green
Write-Host ("Fleet auto-pauses after {0} applies (canary) or `${1} spend, whichever comes first." -f $Canary,$SpendCapUsd) -ForegroundColor Green
Write-Host "Scale the throughput on the REMOTE box: .\load-canary-remote.ps1" -ForegroundColor Green
