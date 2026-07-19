# register-apply-cycle.ps1 [-Dsn <override>] [-Unregister]
#
#   ONE-COMMAND path to wire the autonomous apply CADENCE as Windows Scheduled Tasks. Run ONCE,
#   ELEVATED (right-click PowerShell -> Run as Administrator), on the HOME box only. This is a
#   SEPARATE script from register-fleet-tasks.ps1 (which registers the fleet actuator/watchdog/
#   doctor/outcome-scan/discovery/compute tasks) -- do not edit that file, and do not merge these
#   two registration paths. Per docs/superpowers/specs/2026-06-30-autonomous-apply-loop-roadmap.md
#   Task 5 (auto-task-5-brief.md).
#
#   Registers THREE tasks:
#     ApplyPilot VerifyLive  -- every 6h.  Wrapper: run-applypilot.ps1 verify-live (read-only probe,
#                                writes liveness columns to the live brain via run-applypilot.ps1 so
#                                APPLYPILOT_DB_PATH resolves to the live brain + gets backed up).
#     ApplyPilot ApplyCycle  -- every 4h.  Wrapper chain, each step logged, wrapper exits non-zero
#                                if ANY step fails (Task Scheduler Last-Result visibility):
#                                  1. run-applypilot.ps1 verify-live --limit 300   (fresh pre-push check)
#                                  2. applypilot-fleet-apply-home push --score-floor 5.8
#                                  3. applypilot-fleet-apply-home arm-canary-if-safe <K>
#                                  4. applypilot-fleet-apply-home approve --all-pushed
#                                  5. applypilot-fleet-apply-home resume-if-safe
#                                  6. applypilot-fleet-apply-home pull
#     ApplyPilot DeadMan     -- every 20 min. Wrapper: applypilot-fleet-deadman.exe (read-only
#                                watcher), inheriting the validated persisted Runtime channel.
#
#   Each scheduled task action runs a small generated WRAPPER SCRIPT under
#   .fleet-logs\_task-wrappers\ via "powershell.exe -File" (never -Command with embedded quoting --
#   Task Scheduler passes -Argument as one literal command line, and content containing double
#   quotes inside a -Command "..." wrapper corrupts argv parsing). This mirrors
#   register-fleet-tasks.ps1's -File pattern exactly.
#
#   Examples:
#     .\register-apply-cycle.ps1                     # register all three home tasks
#     .\register-apply-cycle.ps1 -Dsn "host=localhost port=5432 dbname=applypilot_fleet user=<mapped-fleet-role> connect_timeout=5"
#     .\register-apply-cycle.ps1 -Unregister          # remove the three tasks
#
#   Idempotent: re-running replaces existing "ApplyPilot <Name>" tasks (unregister-then-register)
#   and regenerates their wrapper scripts.
[CmdletBinding()]
param(
  [string]$Dsn,
  # Per-cycle apply-lane canary budget. Cost caps and rate governors still bind.
  [int]$CanaryK = 40,
  [switch]$Unregister,
  [string]$RenderApplyCycleWrapper,
  [string]$ReadinessCommand,
  [string[]]$ReadinessArgument,
  [string]$PostReadinessCommand
)
$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
if (-not $repo) { $repo = Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location $repo

function New-ReadinessGateContent([string]$Command, [string[]]$Arguments, [string]$OutputRedirection = '') {
  $escapedCommand = $Command.Replace("'", "''")
  $escapedArguments = @($Arguments | ForEach-Object { "'" + ([string]$_).Replace("'", "''") + "'" }) -join ', '
  return @"
`$ReadinessCommand = '$escapedCommand'
`$ReadinessArgs = @($escapedArguments)
& `$ReadinessCommand @ReadinessArgs $OutputRedirection
`$readinessExit = `$LASTEXITCODE
if (`$readinessExit -ne 0) {
  Write-Error "ApplyCycle readiness denied with exit `$readinessExit"
  exit `$readinessExit
}
"@
}

# Source-only test seam: render the exact readiness gate used by the registered wrapper
# without requiring elevation or touching Task Scheduler.
if ($RenderApplyCycleWrapper) {
  if (-not $ReadinessCommand -or -not $PostReadinessCommand) {
    throw '-RenderApplyCycleWrapper requires -ReadinessCommand and -PostReadinessCommand'
  }
  $gate = New-ReadinessGateContent $ReadinessCommand $ReadinessArgument
  $escapedPost = $PostReadinessCommand.Replace("'", "''")
  $content = @"
`$ErrorActionPreference = 'Continue'
$gate
& '$escapedPost'
exit `$LASTEXITCODE
"@
  [IO.File]::WriteAllText($RenderApplyCycleWrapper, $content, [Text.UTF8Encoding]::new($false))
  exit 0
}

$TaskPrefix = "ApplyPilot "
$wrapperDir = Join-Path $repo ".fleet-logs\_task-wrappers"
$logDir = Join-Path $repo ".fleet-logs"

# ---- elevation check (the whole point of this script is registering scheduled tasks) ----
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
  Write-Host "[register-apply-cycle] NOT elevated. Re-run this script from an Administrator PowerShell (right-click -> Run as Administrator)." -ForegroundColor Red
  exit 1
}

# ============================================================================================
# -Unregister MODE: remove the three ApplyPilot cadence tasks.
# ============================================================================================
function Get-CadenceTaskNames() {
  return @("${TaskPrefix}VerifyLive", "${TaskPrefix}ApplyCycle", "${TaskPrefix}DeadMan")
}

function Remove-CadenceTask([string]$name) {
  $existing = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
  if ($existing) {
    Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "  removed $name" -ForegroundColor Yellow
  }
}

if ($Unregister) {
  Write-Host "[register-apply-cycle] -Unregister: removing 'ApplyPilot VerifyLive/ApplyCycle/DeadMan' tasks..." -ForegroundColor Cyan
  Get-CadenceTaskNames | ForEach-Object { Remove-CadenceTask $_ }
  Write-Host "[register-apply-cycle] done." -ForegroundColor Green
  exit 0
}

# ============================================================================================
# REGISTRATION MODE
# ============================================================================================
. (Join-Path $repo "scripts\runtime-launch-channel.ps1")
$runtimeChannel = Assert-ApplyPilotRuntimeChannel -RepoRoot $repo -OverrideDsn $Dsn `
  -RequirePersistedUserChannel
$effectiveDsn = $runtimeChannel.Dsn
if (-not (Test-Path $wrapperDir)) { New-Item -ItemType Directory -Path $wrapperDir -Force | Out-Null }
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

# ---- common scheduled-task settings (mirrors register-fleet-tasks.ps1's New-FleetSettings) ----
function New-CadenceSettings([int]$restartCount = 3, [TimeSpan]$restartInterval = (New-TimeSpan -Minutes 1), [TimeSpan]$executionTimeLimit = ([TimeSpan]::Zero)) {
  return New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit $executionTimeLimit -RestartCount $restartCount -RestartInterval $restartInterval
}

# Writes a wrapper .ps1 (UTF8) and returns its full path. Mirrors register-fleet-tasks.ps1's
# Write-Wrapper -- keeping each task's env-setup in its own tiny file avoids ALL the Win32-argv /
# PowerShell double-quote-nesting hazards of building a giant -Command string by hand.
$script:generatedRuntimeWrappers = @()
$script:registeredRuntimeTasks = @()
function Write-Wrapper([string]$name, [string]$content) {
  $path = Join-Path $wrapperDir "$name.ps1"
  $runtimePython = Get-ApplyPilotRuntimePython -RepoRoot $repo
  $guard = @"
& '$runtimePython' -m applypilot.fleet.launch_guard
if (`$LASTEXITCODE -ne 0) { exit 78 }
"@
  Set-Content -Path $path -Value ($guard + $content) -Encoding UTF8
  $script:generatedRuntimeWrappers += $path
  return $path
}

function Register-CadenceTask {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$WrapperPath,
    [Parameter(Mandatory = $true)]$Trigger,
    [Parameter(Mandatory = $true)]$Settings,
    [string]$Description = ""
  )
  # idempotent: unregister first, then create fresh (mirrors register-fleet-tasks.ps1's
  # Register-FleetTask -- Register-ScheduledTask -Force still leaves stale trigger/settings
  # combinations in some Windows builds, so an explicit remove is safer).
  Remove-CadenceTask $Name
  $argument = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$WrapperPath`""
  $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argument -WorkingDirectory $repo
  $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
  Register-ScheduledTask -TaskName $Name -Action $action -Trigger $Trigger -Settings $Settings `
    -Principal $principal -Description $Description -Force | Out-Null
  Write-Host "  registered $Name -> $WrapperPath" -ForegroundColor Green
  $script:registeredRuntimeTasks += $Name
}

$runApplyPilotPs1 = Join-Path $repo "run-applypilot.ps1"
if (-not (Test-Path $runApplyPilotPs1)) { throw "run-applypilot.ps1 not found at $runApplyPilotPs1" }

# resolve the fleet apply-home + deadman console-script exes (.conda-env first -- home's real
# runtime; .venv is stale there -- then .venv, matching every other fleet script's resolution
# order in register-fleet-tasks.ps1).
$applyHomeExe = $null
$deadmanExe = $null
foreach ($d in @(".\.conda-env\Scripts", ".\.venv\Scripts")) {
  $c1 = Join-Path $repo (Join-Path $d "applypilot-fleet-apply-home.exe")
  $c2 = Join-Path $repo (Join-Path $d "applypilot-fleet-deadman.exe")
  if (-not $applyHomeExe -and (Test-Path $c1)) { $applyHomeExe = $c1 }
  if (-not $deadmanExe -and (Test-Path $c2)) { $deadmanExe = $c2 }
}
if (-not $applyHomeExe -or -not $deadmanExe) {
  Write-Host "[register-apply-cycle] applypilot-fleet-apply-home.exe and/or applypilot-fleet-deadman.exe not found in .conda-env or .venv (run 'pip install -e .' first)." -ForegroundColor Red
  throw "required console-script exe(s) missing -- see message above."
}

# ---------------------------------------------------------------------------------------------
# VerifyLive -- every 6h. Read-only liveness probe; brain-touching (writes liveness columns), so
# it runs via run-applypilot.ps1 (resolves APPLYPILOT_DB_PATH to the live brain + backs it up).
# ---------------------------------------------------------------------------------------------
Write-Host "`n[register-apply-cycle] registering VerifyLive (every 6h)..." -ForegroundColor Cyan
$verifyLiveLog = Join-Path $logDir "verify-live.log"
$verifyLiveWrapperContent = @"
# Auto-generated by register-apply-cycle.ps1 -- do not hand-edit; re-run the register script instead.
`$ErrorActionPreference = 'Continue'
# FLEET_PG_DSN is inherited from the persisted User environment; never embed the secret here.
Set-Location '$repo'
`$log = '$verifyLiveLog'
function Invoke-VerifyLive {
  param([ScriptBlock]`$Action)
  `$mutex = New-Object System.Threading.Mutex(`$false, 'Global\ApplyPilotVerifyLive')
  `$acquired = `$false
  try {
    try {
      `$acquired = `$mutex.WaitOne([TimeSpan]::FromMinutes(15))
    } catch [System.Threading.AbandonedMutexException] {
      `$acquired = `$true
    }
    if (-not `$acquired) {
      Add-Content -Path `$log -Value ('[' + (Get-Date -Format 'o') + '] verify-live skipped: mutex wait timeout (15m)')
      return 0
    }
    try {
      & `$Action
      return `$LASTEXITCODE
    } finally {
      if (`$acquired) { `$null = `$mutex.ReleaseMutex() }
    }
  } finally {
    `$mutex = `$null
  }
}

Add-Content -Path `$log -Value ('[' + (Get-Date -Format 'o') + '] === VerifyLive start ===')
`$code = Invoke-VerifyLive { & '$runApplyPilotPs1' verify-live --max-age-days 3 --limit 1200 *>> `$log }
Add-Content -Path `$log -Value ('[' + (Get-Date -Format 'o') + '] verify-live exit=' + `$code)
exit `$code
"@
$verifyLiveWrapper = Write-Wrapper "verify-live-task" $verifyLiveWrapperContent
$verifyLiveTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 6) -RepetitionDuration (New-TimeSpan -Days 3650)
$verifyLiveSettings = New-CadenceSettings -restartCount 3 -restartInterval (New-TimeSpan -Minutes 1) -executionTimeLimit (New-TimeSpan -Minutes 30)
Register-CadenceTask -Name "${TaskPrefix}VerifyLive" -WrapperPath $verifyLiveWrapper -Trigger $verifyLiveTrigger -Settings $verifyLiveSettings `
  -Description "ApplyPilot verify-live: read-only liveness probe of scored postings, every 6h. Marks dead postings so apply selection skips them."

# ---------------------------------------------------------------------------------------------
# ApplyCycle -- every 4h. verify-live (fresh pre-push check) -> push -> arm-canary ->
# approve --all-pushed -> resume-if-safe -> pull. Each step logged; wrapper exits non-zero if any fails.
# ---------------------------------------------------------------------------------------------
Write-Host "`n[register-apply-cycle] registering ApplyCycle (every 4h: verify-live -> push -> arm-canary -> approve -> resume-if-safe -> pull)..." -ForegroundColor Cyan
$applyCycleLog = Join-Path $logDir "apply-cycle.log"
$applyCycleWrapperContent = @"
# Auto-generated by register-apply-cycle.ps1 -- do not hand-edit; re-run the register script instead.
`$ErrorActionPreference = 'Continue'
# FLEET_PG_DSN is inherited from the persisted User environment; never embed the secret here.
Set-Location '$repo'
`$log = '$applyCycleLog'
function Invoke-VerifyLive {
  param([ScriptBlock]`$Action)
  `$mutex = New-Object System.Threading.Mutex(`$false, 'Global\ApplyPilotVerifyLive')
  `$acquired = `$false
  try {
    try {
      `$acquired = `$mutex.WaitOne([TimeSpan]::FromMinutes(15))
    } catch [System.Threading.AbandonedMutexException] {
      `$acquired = `$true
    }
    if (-not `$acquired) {
      Add-Content -Path `$log -Value ('[' + (Get-Date -Format 'o') + '] verify-live skipped: mutex wait timeout (15m)')
      return 0
    }
    try {
      & `$Action
      return `$LASTEXITCODE
    } finally {
      if (`$acquired) { `$null = `$mutex.ReleaseMutex() }
    }
  } finally {
    `$mutex = `$null
  }
}
function Step(`$stepName, [ScriptBlock]`$action) {
  Add-Content -Path `$log -Value ('[' + (Get-Date -Format 'o') + '] --- ' + `$stepName + ' start ---')
  & `$action *>> `$log
  `$code = `$LASTEXITCODE
  Add-Content -Path `$log -Value ('[' + (Get-Date -Format 'o') + '] ' + `$stepName + ' exit=' + `$code)
  return `$code
}
Add-Content -Path `$log -Value ('[' + (Get-Date -Format 'o') + '] === ApplyCycle start ===')
# Best-effort: run ALL steps even if one fails (a transient verify-live/pull hiccup must not skip
# the apply push+resume for a whole 4h cycle -- the steps are largely independent). Track the last
# failure and exit non-zero at the end so Task Scheduler's Last-Result still surfaces it.
`$fail = 0
$(New-ReadinessGateContent $applyHomeExe @('readiness', '--strict') '*>> $log')
`$rc = Step 'verify-live' { Invoke-VerifyLive { & '$runApplyPilotPs1' verify-live --max-age-days 3 --limit 300 } }
if (`$rc -ne 0) { `$fail = `$rc; Add-Content -Path `$log -Value 'ApplyCycle: verify-live FAILED (continuing best-effort)' }
`$rc = Step 'push' { & '$applyHomeExe' push --score-floor 5.8 }
if (`$rc -ne 0) { `$fail = `$rc; Add-Content -Path `$log -Value 'ApplyCycle: push FAILED (continuing best-effort)' }
`$rc = Step 'arm-canary' { & '$applyHomeExe' arm-canary-if-safe $CanaryK }
`$canaryArmed = (`$rc -eq 0)
if (-not `$canaryArmed) {
  if (`$rc -eq 2) {
    Add-Content -Path `$log -Value 'ApplyCycle: canary left disarmed by safety pause (expected no-op)'
  } else {
    `$fail = `$rc
    Add-Content -Path `$log -Value 'ApplyCycle: arm-canary FAILED; skipping approve/resume'
  }
}
if (`$canaryArmed) {
  `$rc = Step 'approve' { & '$applyHomeExe' approve --all-pushed }
  if (`$rc -ne 0) { `$fail = `$rc; Add-Content -Path `$log -Value 'ApplyCycle: approve FAILED; skipping resume' }
  if (`$rc -eq 0) {
    `$rc = Step 'resume-if-safe' { & '$applyHomeExe' resume-if-safe }
    if (`$rc -ne 0) { `$fail = `$rc; Add-Content -Path `$log -Value 'ApplyCycle: resume-if-safe FAILED' }
  }
}
`$rc = Step 'pull' { & '$applyHomeExe' pull }
if (`$rc -ne 0) { `$fail = `$rc; Add-Content -Path `$log -Value 'ApplyCycle: pull FAILED (continuing best-effort)' }
Add-Content -Path `$log -Value ('[' + (Get-Date -Format 'o') + '] === ApplyCycle done (fail=' + `$fail + ') ===')
exit `$fail
"@
$applyCycleWrapper = Write-Wrapper "apply-cycle-task" $applyCycleWrapperContent
$applyCycleTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 4) -RepetitionDuration (New-TimeSpan -Days 3650)
$applyCycleSettings = New-CadenceSettings -restartCount 3 -restartInterval (New-TimeSpan -Minutes 1) -executionTimeLimit (New-TimeSpan -Hours 1)
Register-CadenceTask -Name "${TaskPrefix}ApplyCycle" -WrapperPath $applyCycleWrapper -Trigger $applyCycleTrigger -Settings $applyCycleSettings `
  -Description "ApplyPilot autonomous apply cycle: verify-live -> push -> arm-canary -> approve --all-pushed -> resume-if-safe -> pull, every 4h. Best-effort: runs all steps, exits non-zero if any failed."

# ---------------------------------------------------------------------------------------------
# DeadMan -- every 20 min. Read-only watcher; inherits the validated persisted Runtime channel.
# ---------------------------------------------------------------------------------------------
Write-Host "`n[register-apply-cycle] registering DeadMan (every 20 min, read-only watcher)..." -ForegroundColor Cyan
$deadManLog = Join-Path $logDir "deadman.log"
$deadManWrapperContent = @"
# Auto-generated by register-apply-cycle.ps1 -- do not hand-edit; re-run the register script instead.
`$ErrorActionPreference = 'Continue'
# FLEET_PG_DSN is inherited from the persisted User environment; never embed the secret here.
Set-Location '$repo'
& '$deadmanExe' *>> '$deadManLog'
"@
$deadManWrapper = Write-Wrapper "deadman-task" $deadManWrapperContent
$deadManTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 20) -RepetitionDuration (New-TimeSpan -Days 3650)
$deadManSettings = New-CadenceSettings -restartCount 3 -restartInterval (New-TimeSpan -Minutes 1) -executionTimeLimit (New-TimeSpan -Minutes 10)
Register-CadenceTask -Name "${TaskPrefix}DeadMan" -WrapperPath $deadManWrapper -Trigger $deadManTrigger -Settings $deadManSettings `
  -Description "ApplyPilot dead-man watcher: read-only check for silent fleet death / stalled queue / self-healer death / running hot against the daily cost cap, every 20 min."

# ---------------------------------------------------------------------------------------------
# DONE = verification checklist
# ---------------------------------------------------------------------------------------------
Write-Host "`n=================================================================================" -ForegroundColor Cyan
Write-Host " Done = " -ForegroundColor Cyan -NoNewline
Write-Host "verify these once the tasks have had a chance to run:" -ForegroundColor Cyan
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "  [ ] .fleet-logs\verify-live.log, apply-cycle.log, deadman.log are growing"
Write-Host "  [ ] apply-cycle.log shows '=== ApplyCycle done (fail=0) ===' after a full run"
Write-Host "  [ ] Get-ScheduledTask -TaskName 'ApplyPilot *' shows LastTaskResult=0 for all three"
Write-Host "`n[register-apply-cycle] Get-ScheduledTask -TaskName 'ApplyPilot *' to list what's registered." -ForegroundColor Cyan
Write-Host "[register-apply-cycle] wrapper scripts live under $wrapperDir (regenerated on every run)." -ForegroundColor Cyan
Write-ApplyPilotLaunchRolloutEvidence `
  -Path (Join-Path $logDir "runtime-launch-rollout-apply-cycle.json") `
  -RuntimeChannel $runtimeChannel `
  -WrapperPath $generatedRuntimeWrappers `
  -TaskName $registeredRuntimeTasks
Write-Host "[register-apply-cycle] registration complete." -ForegroundColor Green
