# register-fleet-tasks.ps1 -Machine home|m2|m4 [-Dsn <override>] [-SetDesired "home=1,m2=6,m4=2"] [-ComputeWorkers 16] [-AllowZero] [-Unregister]
#
#   ONE-COMMAND path to wire ApplyPilot's control loop as Windows Scheduled Tasks. Run ONCE PER
#   MACHINE, ELEVATED (right-click PowerShell -> Run as Administrator). Implements roadmap Phase 1
#   ("Restore the loop") per docs/superpowers/specs/2026-06-30-autonomous-apply-loop-roadmap.md,
#   AMENDMENT SET v2 (that section GOVERNS over the v1 body text). Key amendments honored here:
#     S1  -SetDesired writes fleet_desired_state directly (generation=generation+1, never a literal)
#     S2  registering FleetAgent subsumes hand-killing PIDs -- do NOT also run canary loaders here
#     C3  fleet_desired_state has NO linkedin/role field; no actuator path in this repo can start
#         a LinkedIn worker (run-fleet-worker.ps1 hardcodes applypilot-fleet-apply.exe)
#     S6a Watchdog is registered for its rate-governor/roll_window/cap duties ONLY. Its restart leg
#         (_handle_stuck -> remote_commands) talks to a DEAD channel nothing consumes -- respawn to
#         match fleet_desired_state is fleet-agent's job, not watchdog's.
#     S4  Doctor has NO dependency on llm_usage/2.1 -- safe to schedule immediately. -Once / a single
#         pass legitimately exits 3 on lock contention; that is NOT a failure.
#     C1  outcomes-scan never reads APPLYPILOT_ENABLE_GMAIL_MCP -- do not set it here.
#     C13 run-discovery-home-loop.ps1 is ingest-only (expand/pull), no egress, NO -Proxy parameter.
#         run-fleet-discovery.ps1 (the scraper) also has no -Proxy parameter. Scrape (m2) and
#         ingest (home) are two separate scheduled tasks -- never pass -Proxy to either here.
#     S2 / S1 warning: NEVER co-locate a canary loader (load-canary-*.ps1) with fleet-agent on one
#         machine -- they are two competing actuators on the same <Label>-<Slot> worker-id
#         namespace and WILL kill-fight.
#
#   Task names are prefixed "ApplyPilotFleet-" so they're easy to find/remove as a group and never
#   collide with the pre-existing "ApplyPilotKeepAlive" task (register-keepalive.ps1).
#
#   Per-machine task sets:
#     ALL machines : FleetAgent            (the actuator -- fleet-agent.ps1 -Label <machine>)
#     home only    : Watchdog, OtpResponder, Doctor, Autotriage, CrashLiveness, OutcomeScan, DiscoveryIngest
#     m2 only      : DiscoveryScrape       (in addition to FleetAgent)
#     m4           : FleetAgent only (no local Postgres, no Gmail creds, not the discovery IP)
#
#   Each scheduled task action runs a small generated WRAPPER SCRIPT under
#   .fleet-logs\_task-wrappers\ via "powershell.exe -File" (never -Command with embedded quoting --
#   Task Scheduler passes -Argument as one literal command line, and content containing double
#   quotes inside a -Command "..." wrapper corrupts argv parsing). Each wrapper sets env vars for
#   ITS OWN task only, then calls the real repo script/exe. This mirrors register-keepalive.ps1's
#   -File pattern.
#
#   Examples:
#     .\register-fleet-tasks.ps1 -Machine home                     # register home's full task set
#     .\register-fleet-tasks.ps1 -Machine m2                       # register m2's task set
#     .\register-fleet-tasks.ps1 -Machine m4 -Dsn "host=100.90.104.99 port=5432 dbname=applypilot_fleet user=<mapped-fleet-role> connect_timeout=5"
#     .\register-fleet-tasks.ps1 -Machine home -Unregister         # remove this machine's tasks
#     .\register-fleet-tasks.ps1 -Machine home -SetDesired "home=1,m2=6,m4=2"   # one-shot PG write, no task changes
#
#   Idempotent: re-running replaces existing ApplyPilotFleet-* tasks (unregister-then-register) and
#   regenerates their wrapper scripts.
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][ValidateSet("home", "m2", "m4")][string]$Machine,
  [string]$Dsn,
  [string]$SetDesired,
  [ValidateRange(1,16)][int]$ComputeWorkers = 16,
  [switch]$AllowZero,
  [switch]$Unregister
)
$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
if (-not $repo) { $repo = Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location $repo

$TaskPrefix = "ApplyPilotFleet-"
$wrapperDir = Join-Path $repo ".fleet-logs\_task-wrappers"
$logDir = Join-Path $repo ".fleet-logs"

# ---- elevation check (the whole point of this script is registering scheduled tasks) ----
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
  Write-Host "[register-fleet-tasks] NOT elevated. Re-run this script from an Administrator PowerShell (right-click -> Run as Administrator)." -ForegroundColor Red
  exit 1
}

# ---- fleet DSN: explicit -Dsn or FLEET_PG_DSN, an explicit mapped fleet DSN. There is NO
# ---- per-machine local-admin default anymore: a postgres/fleet_worker DSN would bypass the
# ---- mapped-role topology. Resolve it here but only REQUIRE it on the paths that touch the
# ---- database or register tasks (Assert-FleetDsn); -Unregister needs no DSN. ----
$effectiveDsn = if ($Dsn) { $Dsn } elseif ($env:FLEET_PG_DSN) { $env:FLEET_PG_DSN } else { $null }
function Assert-FleetDsn {
  if (-not $effectiveDsn) {
    throw "register-fleet-tasks: no fleet DSN for machine '$Machine'. Pass -Dsn or set FLEET_PG_DSN to an explicit mapped fleet DSN (never the postgres admin or the shared fleet_worker role). Refusing to touch the database or register tasks."
  }
  if ($effectiveDsn -match '(?i)\buser\s*=\s*(?:postgres|fleet_worker)\b') {
    throw "register-fleet-tasks: fleet DSN selects an admin/shared principal. Supply an explicit mapped fleet DSN."
  }
}

function Get-Psql {
  $p = Get-ChildItem "C:\Program Files\PostgreSQL\*\bin\psql.exe" -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
  if (-not $p) { throw "psql.exe not found under C:\Program Files\PostgreSQL\*\bin -- install the PostgreSQL client, or run with a machine that has it." }
  return $p.FullName
}

# ============================================================================================
# -SetDesired MODE (S1): one-shot PG write, runs INSTEAD of task registration. No task changes.
# ============================================================================================
if ($SetDesired) {
  Assert-FleetDsn
  $psql = Get-Psql
  Write-Host "[register-fleet-tasks] -SetDesired mode: writing fleet_desired_state via DSN '$effectiveDsn'" -ForegroundColor Cyan
  $pairs = $SetDesired -split "," | Where-Object { $_.Trim() -ne "" }
  if (-not $pairs) { throw "-SetDesired must look like `"home=1,m2=6,m4=2`" (got '$SetDesired')" }
  foreach ($pair in $pairs) {
    $kv = $pair.Trim() -split "="
    if ($kv.Count -ne 2) { throw "malformed -SetDesired entry '$pair' (want machine=count)" }
    $m = $kv[0].Trim()
    $nRaw = $kv[1].Trim()
    $n = 0
    if (-not [int]::TryParse($nRaw, [ref]$n)) { throw "desired_workers for '$m' is not an integer: '$nRaw'" }
    if ($n -lt 0) { throw "desired_workers for '$m' is negative ($n) -- rejected." }
    # generation=generation+1 (NEVER a literal) per S7's concurrent-writer rule.
    $sql = "UPDATE fleet_desired_state SET desired_workers=$n, generation=generation+1, updated_by='roadmap-bringup' WHERE machine_owner='$m';"
    Write-Host "  [$m] -> desired_workers=$n" -ForegroundColor Gray
    & $psql $effectiveDsn -v ON_ERROR_STOP=1 -c $sql
    if ($LASTEXITCODE -ne 0) { throw "psql UPDATE failed for machine_owner='$m' (exit $LASTEXITCODE)" }
  }
  Write-Host "`n[register-fleet-tasks] fleet_desired_state after write:" -ForegroundColor Cyan
  & $psql $effectiveDsn -c "SELECT machine_owner, desired_workers, agent, model, generation, updated_by, updated_at FROM fleet_desired_state ORDER BY machine_owner;"
  exit 0
}

# ============================================================================================
# -Unregister MODE: remove all ApplyPilotFleet-* tasks for this machine's role set.
# ============================================================================================
function Get-TaskNamesForMachine([string]$m) {
  $names = @("${TaskPrefix}FleetAgent")
  if ($m -eq "home") { $names += @("${TaskPrefix}Watchdog", "${TaskPrefix}OtpResponder", "${TaskPrefix}Doctor", "${TaskPrefix}Autotriage", "${TaskPrefix}CrashLiveness", "${TaskPrefix}OutcomeScan", "${TaskPrefix}DiscoveryIngest", "${TaskPrefix}ComputeIngest") }
  if ($m -eq "m2")   { $names += @("${TaskPrefix}DiscoveryScrape") }
  if ($m -eq "m4")   { $names += @("${TaskPrefix}ComputeScore") }
  return $names
}

function Remove-FleetTask([string]$name) {
  $existing = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
  if ($existing) {
    Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "  removed $name" -ForegroundColor Yellow
  }
}

if ($Unregister) {
  Write-Host "[register-fleet-tasks] -Unregister: removing ApplyPilotFleet-* tasks for machine '$Machine'..." -ForegroundColor Cyan
  Get-TaskNamesForMachine $Machine | ForEach-Object { Remove-FleetTask $_ }
  Write-Host "[register-fleet-tasks] done." -ForegroundColor Green
  exit 0
}

# ============================================================================================
# REGISTRATION MODE
# ============================================================================================
Assert-FleetDsn
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host " NEVER co-locate canary loaders (load-canary-*.ps1) with fleet-agent on one machine" -ForegroundColor Red
Write-Host " -- they will kill-fight over the same <Label>-<Slot> worker-id namespace (S1)." -ForegroundColor Red
Write-Host "=================================================================================" -ForegroundColor Cyan

if (-not (Test-Path $wrapperDir)) { New-Item -ItemType Directory -Path $wrapperDir -Force | Out-Null }
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

# ---- Phase-1.1 / S2 safety guard: refuse to register FleetAgent if desired_workers=0 without ----
# ---- an explicit -AllowZero. Registering with 0 is a real trap: fleet-agent's first poll kills ----
# ---- ANY locally-running workers (including ones a canary loader just started) to match 0.     ----
Write-Host "`n[register-fleet-tasks] pre-flight: reading fleet_desired_state row for machine_owner='$Machine' via DSN '$effectiveDsn'..." -ForegroundColor Cyan
# Worker boxes (m2/m4) have no psql.exe -- read via the python helper (psycopg ships with every
# bootstrap; fleet-agent.ps1 already relies on it). psql is only a fallback for odd setups.
$rowRaw = $null; $readOk = $false
# .conda-env FIRST (home's real runtime; .venv is stale there) then .venv (worker boxes have
# only .venv). Matches the resolution order every other fleet script uses.
$pyPre = @((Join-Path $repo ".conda-env\python.exe"), (Join-Path $repo ".venv\Scripts\python.exe")) |
  Where-Object { Test-Path $_ } | Select-Object -First 1
if ($pyPre) {
  $prevDsn = $env:FLEET_PG_DSN; $env:FLEET_PG_DSN = $effectiveDsn
  $line = (& $pyPre (Join-Path $repo "fleet-agent-query.py") $Machine 2>$null | Select-Object -Last 1)
  if ($null -ne $prevDsn) { $env:FLEET_PG_DSN = $prevDsn } else { Remove-Item Env:FLEET_PG_DSN -ErrorAction SilentlyContinue }
  # helper prints "workers|agent|model|gen" (missing row -> "0|claude||0"; DB error -> "KEEP|||")
  if ("$line" -match '^\d+\|') { $rowRaw = "$line"; $readOk = $true }
}
if (-not $readOk) {
  $psql = try { Get-Psql } catch { $null }
  if ($psql) {
    $rowRaw = & $psql $effectiveDsn -t -A -F '|' -c "SELECT desired_workers, agent, model, generation FROM fleet_desired_state WHERE machine_owner='$Machine';" 2>&1
    $readOk = ($LASTEXITCODE -eq 0)
  }
}
if (-not $readOk) {
  Write-Host "[register-fleet-tasks] WARNING: could not read fleet_desired_state (python helper + psql both failed): $rowRaw" -ForegroundColor Yellow
  Write-Host "[register-fleet-tasks] Cannot verify desired_workers before registering FleetAgent. Continuing is riskier without this check." -ForegroundColor Yellow
  if (-not $AllowZero) {
    throw "Refusing to register without a readable fleet_desired_state row (pass -AllowZero to override, or fix the DSN/connectivity first)."
  }
} else {
  $row = "$rowRaw".Trim()
  if (-not $row) {
    Write-Host "[register-fleet-tasks] WARNING: no fleet_desired_state row for machine_owner='$Machine' yet." -ForegroundColor Yellow
    if (-not $AllowZero) {
      throw "No row for '$Machine' in fleet_desired_state -- seed it first (see -SetDesired), or pass -AllowZero to register anyway (fleet-agent will idle at 0 workers until a row exists)."
    }
  } else {
    $fields = $row -split '\|'
    $desired = [int]$fields[0]
    Write-Host "[register-fleet-tasks] current row: desired_workers=$desired agent=$($fields[1]) model=$($fields[2]) generation=$($fields[3])" -ForegroundColor Gray
    if ($desired -eq 0 -and -not $AllowZero) {
      Write-Host "`n*** LOUD WARNING ***" -ForegroundColor Red
      Write-Host "fleet_desired_state.desired_workers=0 for machine_owner='$Machine'." -ForegroundColor Red
      Write-Host "Registering FleetAgent NOW will make it obey that 0 and KILL any workers currently" -ForegroundColor Red
      Write-Host "running on this box (including ones started by a canary loader) on its first poll." -ForegroundColor Red
      Write-Host "Fix desired_workers first (.\register-fleet-tasks.ps1 -Machine $Machine -SetDesired `"$Machine=<n>`")" -ForegroundColor Red
      Write-Host "or pass -AllowZero if you deliberately want this box to register the agent at 0 (idle-armed)." -ForegroundColor Red
      throw "Refusing to register FleetAgent on '$Machine' with desired_workers=0 (pass -AllowZero to proceed anyway)."
    }
  }
}

# ---- common scheduled-task settings ----
function New-FleetSettings([int]$restartCount = 3, [TimeSpan]$restartInterval = (New-TimeSpan -Minutes 1)) {
  return New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount $restartCount -RestartInterval $restartInterval
}

# Writes a wrapper .ps1 (UTF8) and returns its full path. Keeping each task's env-setup in its own
# tiny file avoids ALL the Win32-argv / PowerShell double-quote-nesting hazards of building a
# giant -Command string by hand, and gives us a stable file to `Get-Content` for troubleshooting.
function Write-Wrapper([string]$name, [string]$content) {
  $path = Join-Path $wrapperDir "$name.ps1"
  Set-Content -Path $path -Value $content -Encoding UTF8
  return $path
}

function Register-FleetTask {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$WrapperPath,
    [Parameter(Mandatory = $true)]$Trigger,
    [Parameter(Mandatory = $true)]$Settings,
    [string]$Description = ""
  )
  # idempotent: unregister first, then create fresh (Register-ScheduledTask -Force still leaves
  # stale trigger/settings combinations in some Windows builds, so an explicit remove is safer).
  Remove-FleetTask $Name
  $argument = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$WrapperPath`""
  $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argument -WorkingDirectory $repo
  $taskUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
  $principal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
  Register-ScheduledTask -TaskName $Name -Action $action -Trigger $Trigger -Settings $Settings `
    -Principal $principal -Description $Description -Force | Out-Null
  Write-Host "  registered $Name -> $WrapperPath" -ForegroundColor Green
}

# ---------------------------------------------------------------------------------------------
# ALL MACHINES: FleetAgent -- the one actuator (Phase 1.1). At logon, restart-on-failure, no
# execution-time limit. Sets FLEET_PG_DSN for this machine inline in the wrapper (does NOT rely
# on user-profile env).
# ---------------------------------------------------------------------------------------------
Write-Host "`n[register-fleet-tasks] registering FleetAgent (actuator) on '$Machine'..." -ForegroundColor Cyan
$fleetAgentPs1 = Join-Path $repo "fleet-agent.ps1"
if (-not (Test-Path $fleetAgentPs1)) { throw "fleet-agent.ps1 not found at $fleetAgentPs1" }
# -AutoUpdate on worker boxes only: home is the dev origin (pushes, never pulls); its dirty/ahead
# tree would no-op the updater anyway, but the explicit gate keeps intent obvious (spec 2026-07-03).
$agentAutoUpdate = if ($Machine -ne "home") { " -AutoUpdate" } else { "" }
$agentWrapperContent = @"
# Auto-generated by register-fleet-tasks.ps1 -- do not hand-edit; re-run the register script instead.
`$ErrorActionPreference = 'Continue'
`$env:FLEET_PG_DSN = '$effectiveDsn'
Set-Location '$repo'
& '$fleetAgentPs1' -Label $Machine$($agentAutoUpdate)
exit `$LASTEXITCODE
"@
$agentWrapper = Write-Wrapper "fleet-agent-task" $agentWrapperContent
$agentTrigger = New-ScheduledTaskTrigger -AtLogOn
# restartCount 10 (not the house default 3): fleet-agent is THE actuator -- if it crash-loops past
# the restart budget it stays dark until next logon and nothing else can revive it (watchdog's
# restart leg is dead, S6a). 10 restarts rides out a multi-minute PG outage.
$agentSettings = New-FleetSettings -restartCount 10 -restartInterval (New-TimeSpan -Minutes 1)
Register-FleetTask -Name "${TaskPrefix}FleetAgent" -WrapperPath $agentWrapper -Trigger $agentTrigger -Settings $agentSettings `
  -Description "ApplyPilot fleet actuator: polls fleet_desired_state, starts/stops local apply workers to match (machine=$Machine)."

# ---------------------------------------------------------------------------------------------
# HOME ONLY: Watchdog, OtpResponder, Doctor, OutcomeScan, DiscoveryIngest
# ---------------------------------------------------------------------------------------------
if ($Machine -eq "home") {

  # -- Watchdog (Phase 1.2 / amendment S6a) -----------------------------------------------------
  # Registered for its rate-governor / roll_window / rolling-24h-cap duties ONLY. Its restart leg
  # (_handle_stuck -> heartbeat.issue_command(...,"restart") -> INSERT INTO remote_commands) writes
  # to a channel NOTHING consumes (1,968 issued / 0 acked, live-verified) -- do NOT rely on the
  # watchdog to respawn dead workers. Respawn-to-match-desired-state is FleetAgent's job.
  Write-Host "`n[register-fleet-tasks] registering Watchdog (governor/roll_window/cap duties ONLY -- respawn is fleet-agent's job, S6a)..." -ForegroundColor Cyan
  $watchdogExe = $null
  foreach ($d in @(".\.conda-env\Scripts", ".\.venv\Scripts")) {
    $cand = Join-Path $repo (Join-Path $d "applypilot-fleet-watchdog.exe")
    if (Test-Path $cand) { $watchdogExe = $cand; break }
  }
  if (-not $watchdogExe) {
    Write-Host "  SKIPPED: applypilot-fleet-watchdog.exe not found in .conda-env or .venv (run 'pip install -e .' first)." -ForegroundColor Yellow
  } else {
    $watchdogWrapperContent = @"
# Auto-generated by register-fleet-tasks.ps1 -- do not hand-edit; re-run the register script instead.
# NOTE (S6a): this watchdog's restart leg writes to the dead remote_commands channel (nothing
# consumes it). It is registered for rate-governor/roll_window/24h-cap duties ONLY -- respawn to
# match fleet_desired_state is FleetAgent's job, not this one's.
`$ErrorActionPreference = 'Continue'
`$env:FLEET_PG_DSN = '$effectiveDsn'
Set-Location '$repo'
function Stop-StaleWatchdogProcesses {
  `$self = `$PID
  `$repoRoot = '$repo'
  Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
      `$_.ProcessId -ne `$self -and `$_.CommandLine -and
      `$_.CommandLine -like "*`$repoRoot*" -and
      (`$_.CommandLine -like '*applypilot-fleet-watchdog.exe*' -or `$_.CommandLine -like '*watchdog-task.ps1*')
    } |
    ForEach-Object {
      Stop-Process -Id `$_.ProcessId -Force -ErrorAction SilentlyContinue
    }
}
Stop-StaleWatchdogProcesses
& '$watchdogExe' --dsn `$env:FLEET_PG_DSN
"@
    $watchdogWrapper = Write-Wrapper "watchdog-task" $watchdogWrapperContent
    $watchdogTrigger = New-ScheduledTaskTrigger -AtLogOn
    # Same elevated restart budget as fleet-agent: an always-on daemon that exhausts its restarts
    # goes dark until next logon (see comment at $agentSettings).
    $watchdogSettings = New-FleetSettings -restartCount 10 -restartInterval (New-TimeSpan -Minutes 1)
    Register-FleetTask -Name "${TaskPrefix}Watchdog" -WrapperPath $watchdogWrapper -Trigger $watchdogTrigger -Settings $watchdogSettings `
      -Description "ApplyPilot fleet watchdog: rate-governor/roll_window/24h-cap ONLY. Restart leg writes to a dead channel (S6a) -- fleet-agent handles respawn."
  }

  # -- OtpResponder (Gmail email-verification relay) -------------------------------------------
  # Runs only on the home box, where Gmail credentials live. Remote workers write otp_request
  # rows and poll Postgres; this responder reads Gmail read-only only when a request is pending.
  Write-Host "`n[register-fleet-tasks] registering OtpResponder (home Gmail relay for email-verification codes)..." -ForegroundColor Cyan
  $otpLauncher = Join-Path $repo "run-otp-responder.ps1"
  if (-not (Test-Path $otpLauncher)) {
    Write-Host "  SKIPPED: run-otp-responder.ps1 not found at $otpLauncher." -ForegroundColor Yellow
  } else {
    $otpWrapperContent = @"
# Auto-generated by register-fleet-tasks.ps1 -- do not hand-edit; re-run the register script instead.
# Home-only Gmail relay: answers otp_request rows for remote apply workers. Codes are never logged.
`$ErrorActionPreference = 'Stop'
`$env:FLEET_PG_DSN = '$effectiveDsn'
Set-Location '$repo'
& '$otpLauncher' -Dsn `$env:FLEET_PG_DSN -Supervise
"@
    $otpWrapper = Write-Wrapper "otp-responder-task" $otpWrapperContent
    $otpTrigger = New-ScheduledTaskTrigger -AtLogOn
    $otpSettings = New-FleetSettings -restartCount 10 -restartInterval (New-TimeSpan -Minutes 1)
    Register-FleetTask -Name "${TaskPrefix}OtpResponder" -WrapperPath $otpWrapper -Trigger $otpTrigger -Settings $otpSettings `
      -Description "ApplyPilot home OTP responder: reads Gmail read-only for pending email-verification requests and relays single-use codes via Postgres."
  }

  # -- Doctor (Phase 1.3 / amendment S4) --------------------------------------------------------
  # No dependency on llm_usage/2.1 -- Doctor is cost-agnostic (reads only apply_queue failure rows).
  # Runs every 5 minutes via a wrapper that treats exit code 3 (lock contention) as success.
  Write-Host "`n[register-fleet-tasks] registering Doctor (every 5 min; exit 3 = lock contention = OK, not a failure)..." -ForegroundColor Cyan
  $doctorPs1 = Join-Path $repo "run-fleet-doctor.ps1"
  if (-not (Test-Path $doctorPs1)) { throw "run-fleet-doctor.ps1 not found at $doctorPs1" }
  $doctorLog = Join-Path $logDir "doctor.log"
  $doctorWrapperContent = @"
# Auto-generated by register-fleet-tasks.ps1 -- do not hand-edit; re-run the register script instead.
# Wraps run-fleet-doctor.ps1 -Once and normalizes exit code 3 (lock contention, S4) to 0 so Task
# Scheduler's history reflects real failures only.
`$ErrorActionPreference = 'Continue'
`$env:FLEET_PG_DSN = '$effectiveDsn'
Set-Location '$repo'
& '$doctorPs1' -Once *>> '$doctorLog'
`$code = `$LASTEXITCODE
if (`$code -eq 3) { exit 0 } else { exit `$code }
"@
  $doctorWrapper = Write-Wrapper "doctor-task" $doctorWrapperContent
  $doctorTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 3650)
  $doctorSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 4)
  Register-FleetTask -Name "${TaskPrefix}Doctor" -WrapperPath $doctorWrapper -Trigger $doctorTrigger -Settings $doctorSettings `
    -Description "ApplyPilot fleet doctor: bounded, reversible, monotonically-conservative auto-fixes, every 5 min. Never touches LinkedIn."

  # -- Autotriage (bounded LLM/rule action menu) ----------------------------------------------
  # Lets the LLM classify opaque apply failures, but the executor still validates every action
  # and keeps duplicate/apply-email guards authoritative. Runs less frequently than Doctor
  # because it can spend LLM calls; load_contexts also skips known no-op buckets and recent
  # terminal audit rows to avoid repeated spend on the same URL.
  Write-Host "`n[register-fleet-tasks] registering Autotriage (every 10 min; bounded LLM choices + validator guards)..." -ForegroundColor Cyan
  $autotriagePs1 = Join-Path $repo "run-fleet-autotriage.ps1"
  if (-not (Test-Path $autotriagePs1)) { throw "run-fleet-autotriage.ps1 not found at $autotriagePs1" }
  $autotriageLog = Join-Path $logDir "autotriage.log"
  $autotriageWrapperContent = @"
# Auto-generated by register-fleet-tasks.ps1 -- do not hand-edit; re-run the register script instead.
`$ErrorActionPreference = 'Continue'
`$env:FLEET_PG_DSN = '$effectiveDsn'
Set-Location '$repo'
& '$autotriagePs1' -Once -CrashOnly -Limit 25 -WindowMinutes 1440 *>> '$autotriageLog'
"@
  $autotriageWrapper = Write-Wrapper "autotriage-task" $autotriageWrapperContent
  $autotriageTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration (New-TimeSpan -Days 3650)
  $autotriageSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 9)
  Register-FleetTask -Name "${TaskPrefix}Autotriage" -WrapperPath $autotriageWrapper -Trigger $autotriageTrigger -Settings $autotriageSettings `
    -Description "ApplyPilot fleet autotriage: bounded LLM/rule action menu with validator gates, every 10 min. Writes autotriage_actions audit rows."

  # -- CrashLiveness (read-only evidence refresh) ----------------------------------------------
  # This task classifies old crash rows by current public-posting liveness. It never retries a
  # job or resolves an outcome; the result is evidence for operator review and autotriage guards.
  Write-Host "`n[register-fleet-tasks] registering CrashLiveness (every 30 min; evidence-only) ..." -ForegroundColor Cyan
  $crashLivenessPs1 = Join-Path $repo "run-fleet-crash-liveness.ps1"
  if (-not (Test-Path $crashLivenessPs1)) { throw "run-fleet-crash-liveness.ps1 not found at $crashLivenessPs1" }
  $crashLivenessLog = Join-Path $logDir "crash-liveness.log"
  $crashLivenessWrapperContent = @"
# Auto-generated by register-fleet-tasks.ps1 -- do not hand-edit; re-run the register script instead.
# Evidence-only: crash-liveness stamps posting status and never retries or resolves applications.
`$ErrorActionPreference = 'Continue'
`$env:FLEET_PG_DSN = '$effectiveDsn'
Set-Location '$repo'
& '$crashLivenessPs1' -Once -Limit 25 -OlderDays 7 -RefreshDays 7 *>> '$crashLivenessLog'
"@
  $crashLivenessWrapper = Write-Wrapper "crash-liveness-task" $crashLivenessWrapperContent
  $crashLivenessTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Days 3650)
  $crashLivenessSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 9)
  Register-FleetTask -Name "${TaskPrefix}CrashLiveness" -WrapperPath $crashLivenessWrapper -Trigger $crashLivenessTrigger -Settings $crashLivenessSettings `
    -Description "ApplyPilot crash liveness refresh: evidence-only, bounded 25 rows every 30 min. Never retries or resolves applications."

  # -- OutcomeScan (Phase 1.4, amendment C1) ----------------------------------------------------
  # C1: outcomes-scan never reads APPLYPILOT_ENABLE_GMAIL_MCP -- do not set it. Real deps are
  # pre-authorized gmail_credentials.json/gmail_token.json in APPLYPILOT_DIR, APPLYPILOT_DB_PATH
  # pointed at the real LOCALAPPDATA brain, and .conda-env python. Chain: scan, then (only if scan
  # exits 0) reconcile-email --apply. Fail loud: non-zero exit is recorded by Task Scheduler.
  Write-Host "`n[register-fleet-tasks] registering OutcomeScan (every 6h: outcomes-scan -> reconcile-email --apply)..." -ForegroundColor Cyan
  $applypilotExe = $null
  $reconcileExe = $null
  foreach ($d in @(".\.conda-env\Scripts", ".\.venv\Scripts")) {
    $c1 = Join-Path $repo (Join-Path $d "applypilot.exe")
    $c2 = Join-Path $repo (Join-Path $d "applypilot-fleet-reconcile-email.exe")
    if (-not $applypilotExe -and (Test-Path $c1)) { $applypilotExe = $c1 }
    if (-not $reconcileExe -and (Test-Path $c2)) { $reconcileExe = $c2 }
  }
  if (-not $applypilotExe -or -not $reconcileExe) {
    Write-Host "  SKIPPED: applypilot.exe and/or applypilot-fleet-reconcile-email.exe not found in .conda-env or .venv (run 'pip install -e .' first)." -ForegroundColor Yellow
  } else {
    $outcomeLog = Join-Path $logDir "outcome-scan.log"
    $appDir = Join-Path $repo ".applypilot"
    # Env set INSIDE the wrapper (not relying on user-profile env), per spec. Deliberately does
    # NOT set APPLYPILOT_ENABLE_GMAIL_MCP (C1: outcomes-scan never reads it). Fails loud: the
    # wrapper's own exit code mirrors whichever step failed, so Task Scheduler logs it as a failure.
    $outcomeWrapperContent = @"
# Auto-generated by register-fleet-tasks.ps1 -- do not hand-edit; re-run the register script instead.
`$ErrorActionPreference = 'Continue'
`$env:APPLYPILOT_DB_PATH = Join-Path `$env:LOCALAPPDATA 'ApplyPilot\applypilot.db'
`$env:APPLYPILOT_DIR = '$appDir'
`$env:APPLYPILOT_FLEET_DSN = '$effectiveDsn'
`$env:PYTHONUTF8 = '1'
`$env:PYTHONIOENCODING = 'utf-8'
Set-Location '$repo'
`$log = '$outcomeLog'
Add-Content -Path `$log -Value ('[' + (Get-Date -Format 'o') + '] === OutcomeScan start ===')
& '$applypilotExe' outcomes-scan --days 7 *>> `$log
`$scanExit = `$LASTEXITCODE
Add-Content -Path `$log -Value ('[' + (Get-Date -Format 'o') + '] outcomes-scan exit=' + `$scanExit)
if (`$scanExit -ne 0) {
  Add-Content -Path `$log -Value ('[' + (Get-Date -Format 'o') + '] scan failed -- skipping reconcile-email')
  exit `$scanExit
}
# --no-scan: the standalone outcomes-scan above already refreshed email_events; without it the
# reconcile runs its own built-in Phase-0 Gmail scan and every cycle would scan Gmail twice.
& '$reconcileExe' --dsn `$env:APPLYPILOT_FLEET_DSN --no-scan --apply *>> `$log
`$reconcileExit = `$LASTEXITCODE
Add-Content -Path `$log -Value ('[' + (Get-Date -Format 'o') + '] reconcile-email exit=' + `$reconcileExit)
exit `$reconcileExit
"@
    $outcomeWrapper = Write-Wrapper "outcome-scan-task" $outcomeWrapperContent
    $outcomeTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 6) -RepetitionDuration (New-TimeSpan -Days 3650)
    $outcomeSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
      -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
    Register-FleetTask -Name "${TaskPrefix}OutcomeScan" -WrapperPath $outcomeWrapper -Trigger $outcomeTrigger -Settings $outcomeSettings `
      -Description "ApplyPilot outcome scan + reconcile: Gmail scan every 6h, then reconcile-email --apply if scan succeeds. apply_queue scope only."
  }

  # -- DiscoveryIngest (Phase 1.5 / S8, amendment C13) -----------------------------------------
  # C13: run-discovery-home-loop.ps1 is ingest-only (expand/pull); it has NO -Proxy parameter and
  # passing one would error the task. Never pass -Proxy here.
  Write-Host "`n[register-fleet-tasks] registering DiscoveryIngest (every 6h, ingest-only, no -Proxy)..." -ForegroundColor Cyan
  $discIngestPs1 = Join-Path $repo "run-discovery-home-loop.ps1"
  if (-not (Test-Path $discIngestPs1)) { throw "run-discovery-home-loop.ps1 not found at $discIngestPs1" }
  $discIngestLog = Join-Path $logDir "discovery-ingest.log"
  $discIngestWrapperContent = @"
# Auto-generated by register-fleet-tasks.ps1 -- do not hand-edit; re-run the register script instead.
# Ingest-only (expand/pull), no egress. NEVER pass -Proxy here (C13 -- the script has no such param).
`$ErrorActionPreference = 'Continue'
`$env:FLEET_PG_DSN = '$effectiveDsn'
Set-Location '$repo'
& '$discIngestPs1' -Once *>> '$discIngestLog'
"@
  $discIngestWrapper = Write-Wrapper "discovery-ingest-task" $discIngestWrapperContent
  $discIngestTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 6) -RepetitionDuration (New-TimeSpan -Days 3650)
  $discIngestSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
  Register-FleetTask -Name "${TaskPrefix}DiscoveryIngest" -WrapperPath $discIngestWrapper -Trigger $discIngestTrigger -Settings $discIngestSettings `
    -Description "ApplyPilot discovery ingest: pulls staged postings from Postgres into the live brain. Ingest-only, no egress, NEVER pass -Proxy (C13)."

  # -- ComputeIngest (scoring lane feed+harvest) ----------------------------------------------
  # push worth-scoring brain jobs into compute_queue + pull the m4 scorers' ADVISORY results
  # back into the brain. Reads the LIVE brain (APPLYPILOT_DB_PATH resolved inside the loop
  # script). Every 15 min so the queue stays fed and results land promptly during a burst.
  Write-Host "`n[register-fleet-tasks] registering ComputeIngest (every 15m, push+pull, home-only)..." -ForegroundColor Cyan
  $computeIngestPs1 = Join-Path $repo "run-compute-home-loop.ps1"
  if (-not (Test-Path $computeIngestPs1)) { throw "run-compute-home-loop.ps1 not found at $computeIngestPs1" }
  $computeIngestLog = Join-Path $logDir "compute-ingest.log"
  $computeIngestWrapperContent = @"
# Auto-generated by register-fleet-tasks.ps1 -- do not hand-edit; re-run the register script instead.
`$ErrorActionPreference = 'Continue'
`$env:FLEET_PG_DSN = '$effectiveDsn'
Set-Location '$repo'
& '$computeIngestPs1' -Once -IncludeUnscored *>> '$computeIngestLog'
"@
  $computeIngestWrapper = Write-Wrapper "compute-ingest-task" $computeIngestWrapperContent
  $computeIngestTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Days 3650)
  $computeIngestSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 14)
  Register-FleetTask -Name "${TaskPrefix}ComputeIngest" -WrapperPath $computeIngestWrapper -Trigger $computeIngestTrigger -Settings $computeIngestSettings `
    -Description "ApplyPilot compute ingest: push worth-scoring jobs to compute_queue + pull advisory scores back into the brain. Home-only, reads live brain."
}

# ---------------------------------------------------------------------------------------------
# M2 ONLY: DiscoveryScrape -- residential-IP scrape (Phase 1.5 / S8, amendment S8)
# ---------------------------------------------------------------------------------------------
if ($Machine -eq "m2") {
  Write-Host "`n[register-fleet-tasks] registering DiscoveryScrape (every 6h, residential-IP scrape, no -Proxy param exists yet)..." -ForegroundColor Cyan
  $discScrapePs1 = Join-Path $repo "run-fleet-discovery.ps1"
  if (-not (Test-Path $discScrapePs1)) { throw "run-fleet-discovery.ps1 not found at $discScrapePs1" }
  $discScrapeLog = Join-Path $logDir "discovery-scrape.log"
  $discScrapeWrapperContent = @"
# Auto-generated by register-fleet-tasks.ps1 -- do not hand-edit; re-run the register script instead.
# Pure scrape (JobSpy) staged to discovered_postings. No login, no apply agent, no brain write.
`$ErrorActionPreference = 'Continue'
`$env:FLEET_PG_DSN = '$effectiveDsn'
Set-Location '$repo'
& '$discScrapePs1' -Label $Machine *>> '$discScrapeLog'
"@
  $discScrapeWrapper = Write-Wrapper "discovery-scrape-task" $discScrapeWrapperContent
  $discScrapeTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 6) -RepetitionDuration (New-TimeSpan -Days 3650)
  $discScrapeSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
  Register-FleetTask -Name "${TaskPrefix}DiscoveryScrape" -WrapperPath $discScrapeWrapper -Trigger $discScrapeTrigger -Settings $discScrapeSettings `
    -Description "ApplyPilot discovery scrape: leases search_tasks, scrapes JobSpy, stages to discovered_postings. Pure scrape, no login/agent/brain-write."
}

# ---------------------------------------------------------------------------------------------
# M4 ONLY: ComputeScore -- IP-free LLM score/audit workers (owner-machine compute lane, R14)
# ---------------------------------------------------------------------------------------------
if ($Machine -eq "m4") {
  # Persistent scorers (run-fleet-compute spawns N forever-workers + exits). Re-run hourly via
  # the repetition trigger self-heals any that died (it kills+respawns a clean slate of N).
  # IP-free + cost-cap gated: no apply-IP hygiene needed, and it is NOT halted by the apply pause.
  Write-Host "`n[register-fleet-tasks] registering ComputeScore ($ComputeWorkers scorers, self-heal hourly, IP-free)..." -ForegroundColor Cyan
  $computeScorePs1 = Join-Path $repo "run-fleet-compute.ps1"
  if (-not (Test-Path $computeScorePs1)) { throw "run-fleet-compute.ps1 not found at $computeScorePs1" }
  $computeScoreLog = Join-Path $logDir "compute-score.log"
  $computeScoreWrapperContent = @"
# Auto-generated by register-fleet-tasks.ps1 -- do not hand-edit; re-run the register script instead.
# IP-free LLM score/audit. Reads DEEPSEEK key from ~/.applypilot/.env (loaded by the script).
`$ErrorActionPreference = 'Continue'
`$env:FLEET_PG_DSN = '$effectiveDsn'
Set-Location '$repo'
& '$computeScorePs1' -Label $Machine -Workers $ComputeWorkers *>> '$computeScoreLog'
"@
  $computeScoreWrapper = Write-Wrapper "compute-score-task" $computeScoreWrapperContent
  $computeScoreTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Days 3650)
  $computeScoreSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
  Register-FleetTask -Name "${TaskPrefix}ComputeScore" -WrapperPath $computeScoreWrapper -Trigger $computeScoreTrigger -Settings $computeScoreSettings `
    -Description "ApplyPilot compute score: $ComputeWorkers IP-free LLM scorers leasing compute_queue. Cost-cap gated (not apply spend_cap, not apply pause). Advisory results only."
}

# ---------------------------------------------------------------------------------------------
# DONE = verification checklist
# ---------------------------------------------------------------------------------------------
Write-Host "`n=================================================================================" -ForegroundColor Cyan
Write-Host " Done = " -ForegroundColor Cyan -NoNewline
Write-Host "verify these once the tasks have had a chance to run:" -ForegroundColor Cyan
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "  [ ] fresh worker_heartbeat beats < 2 min old for '$Machine'"
Write-Host "        psql `"$effectiveDsn`" -c `"SELECT worker_id, last_beat, now()-last_beat AS age FROM worker_heartbeat WHERE worker_id LIKE '$Machine-%' ORDER BY last_beat DESC;`""
if ($Machine -eq "home") {
  Write-Host "  [ ] doctor_last_pass_at advancing"
  Write-Host "        psql `"$effectiveDsn`" -c `"SELECT doctor_last_pass_at FROM fleet_config WHERE id=1;`""
  Write-Host "  [ ] autotriage_actions advancing; no repeated churn on same URL"
  Write-Host "        psql `"$effectiveDsn`" -c `"SELECT chosen_action, action_status, count(*) FROM autotriage_actions WHERE created_at > now() - interval '30 minutes' GROUP BY 1,2 ORDER BY 3 DESC;`""
  Write-Host "  [ ] email_events.scanned_at advancing every ~6h"
  Write-Host "        psql `"$effectiveDsn`" -c `"SELECT max(scanned_at) FROM email_events;`""
  Write-Host "  [ ] discovered_postings > 0 (after m2's DiscoveryScrape has run at least once)"
  Write-Host "        psql `"$effectiveDsn`" -c `"SELECT count(*) FROM discovered_postings;`""
  Write-Host "  [ ] .fleet-logs\doctor.log, autotriage.log, outcome-scan.log, discovery-ingest.log are growing"
}
if ($Machine -eq "home") {
  Write-Host "  [ ] compute_queue draining + advisory scores landing (ComputeIngest + m4 scorers)"
  Write-Host "        psql `"$effectiveDsn`" -c `"SELECT status, count(*) FROM compute_queue GROUP BY status;`""
  Write-Host "  [ ] .fleet-logs\compute-ingest.log shows 'pushed N' / 'pulled N' lines"
}
if ($Machine -eq "m2") {
  Write-Host "  [ ] .fleet-logs\discovery-scrape.log is growing; search_tasks.last_run_at populated"
  Write-Host "        psql `"$effectiveDsn`" -c `"SELECT count(*) FROM search_tasks WHERE last_run_at IS NOT NULL;`""
}
if ($Machine -eq "m4") {
  Write-Host "  [ ] fresh m4-score-* heartbeats + advisory scores flowing (needs .applypilot\.env copied here)"
  Write-Host "        psql `"$effectiveDsn`" -c `"SELECT worker_id, state, now()-last_beat AS age FROM worker_heartbeat WHERE worker_id LIKE 'm4-score-%' ORDER BY last_beat DESC;`""
  Write-Host "  [ ] .fleet-logs\compute-score.log shows 'pre-flight OK: LLM tier=' then scoring (not the tier-fail)"
  Write-Host "  [ ] apply lane resumes only after you UNPAUSE (fleet is paused=true today); compute runs regardless"
}
Write-Host "`n[register-fleet-tasks] Get-ScheduledTask -TaskName '${TaskPrefix}*' to list what's registered on this box." -ForegroundColor Cyan
Write-Host "[register-fleet-tasks] wrapper scripts live under $wrapperDir (regenerated on every run)." -ForegroundColor Cyan
Write-Host "[register-fleet-tasks] registration for '$Machine' complete." -ForegroundColor Green
