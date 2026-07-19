# fleet-agent.ps1 -Label m2 [-PollSec 20] [-AutoUpdate] [-UpdateEverySec 900]
#
#   Run this ONCE on a worker box (m2, m4, or even home). It is the actuator that makes the home
#   box's `fleet.ps1` able to control THIS machine: every -PollSec it reads this box's row in the
#   Postgres table `fleet_desired_state` and starts/stops LOCAL apply workers to match. So setting
#   m2=4 from the home box makes this agent bring m2 to 4 workers; setting m2=0 stops them.
#
#   Enroll once per box (ideally as a Task Scheduler "At log on" task so it auto-starts):
#     m2:  $env:FLEET_PG_DSN="host=100.90.104.99 port=5432 dbname=applypilot_fleet user=<mapped-fleet-role> connect_timeout=5"
#          cd C:\ApplyPilot ;  .\fleet-agent.ps1 -Label m2 -AutoUpdate
#     m4:  $env:FLEET_PG_DSN="host=100.90.104.99 port=5432 dbname=applypilot_fleet user=<mapped-fleet-role> connect_timeout=5"
#          cd C:\ApplyPilot ;  .\fleet-agent.ps1 -Label m4 -AutoUpdate
#
#   -AutoUpdate (worker boxes ONLY, never home -- home is the dev origin and pushes, never pulls):
#   every -UpdateEverySec git-fetch the current branch's upstream; if behind and fast-forwardable
#   with a clean tree, wait for a BETWEEN-JOBS window (fleet-agent-update-gate.py: no fresh
#   non-idle heartbeat + no live lease for this Label; fail-closed), merge the captured pinned
#   target while workers remain running, and persist a target-SHA restart marker. A valid
#   post-merge policy plus a still-IDLE gate completes the controlled worker restart. The old
#   in-memory agent exits 1 for scheduled-task relaunch; a new agent reconciles directly.
#   Dependency-changing updates are deferred. Spec: docs/superpowers/specs/2026-07-03-fleet-pull-updater-design.md
#
#   Fail-safe: on any DB blip the agent leaves local workers untouched (never kills on a transient error).
param([Parameter(Mandatory = $true)][string]$Label, [int]$PollSec = 20,
      [switch]$AutoUpdate, [int]$UpdateEverySec = 900)
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo
$agentStartHead = "$(& git rev-parse HEAD 2>$null)".Trim()
if ($LASTEXITCODE -ne 0 -or $agentStartHead -cnotmatch '^[0-9a-fA-F]{40}$') {
  $agentStartHead = $null
}

# ---- host-identity guard: never run another machine's workers on this box ----
# A box declares its OWN fleet label in APPLYPILOT_FLEET_LABEL (home/m2/m4). Running this
# agent with a foreign -Label (e.g. `-Label m2` on the HOME box) would reconcile+spawn that
# machine's workers HERE -- the live 2026-07-04 incident where home's desired=0 yet 4 m2
# (TARPON) workers ran on the home box. If this box is labeled and disagrees, refuse to start.
# Unset label = unknown identity = permissive (back-compat for not-yet-labeled boxes).
$boxLabel = "$env:APPLYPILOT_FLEET_LABEL".Trim()
if ($boxLabel -and ($boxLabel -ne $Label)) {
  throw "[fleet-agent:$Label] host-identity guard: this box is '$boxLabel' but the agent was started with -Label '$Label'. Refusing to spawn another machine's workers here (e.g. m2/TARPON workers must never run on the home box). Re-run with -Label $boxLabel, or start this agent on the '$Label' box."
}

$py = $null
foreach ($d in @(".\.conda-env\python.exe", ".\.venv\Scripts\python.exe")) { if (Test-Path $d) { $py = (Resolve-Path $d).Path; break } }
if (-not $py) { throw "python not found (.conda-env or .venv) -- run the box setup first." }

function Assert-NoLifecycleFaults {
  $dbPath = if ($env:APPLYPILOT_DB_PATH) {
    $env:APPLYPILOT_DB_PATH
  } else {
    Join-Path $env:LOCALAPPDATA "ApplyPilot\applypilot.db"
  }
  $faults = @()
  $stateDirs = @(
    (Split-Path -Parent ([IO.Path]::GetFullPath($dbPath))),
    (Join-Path $env:LOCALAPPDATA "ApplyPilot"),
    $env:TEMP
  ) | Where-Object { $_ } | Select-Object -Unique
  foreach ($stateDir in $stateDirs) {
    $legacyFault = Join-Path $stateDir "keepalive.hard-fault.json"
    $faultDir = Join-Path $stateDir "lifecycle-faults"
    if (Test-Path -LiteralPath $legacyFault -PathType Leaf) {
      $faults += Get-Item -LiteralPath $legacyFault
    }
    if (Test-Path -LiteralPath $faultDir -PathType Container) {
      $faults += Get-ChildItem -LiteralPath $faultDir -Filter "fault-*.json" -File
    }
  }
  if ($faults.Count -gt 0) {
    throw "[fleet-agent:$Label] unresolved lifecycle hard-fault record(s); operator reconciliation is required before worker launch."
  }
}

Assert-NoLifecycleFaults
$applypilotCli = $null
foreach ($d in @(".\.conda-env\Scripts", ".\.venv\Scripts")) {
  $cand = Join-Path $d "applypilot.exe"
  if (Test-Path $cand) { $applypilotCli = (Resolve-Path $cand).Path; break }
}
if (-not $env:FLEET_PG_DSN) {
  throw "FLEET_PG_DSN is not set for fleet-agent label '$Label'. Set it to an explicit mapped per-node fleet DSN (host=<home Tailscale IP or localhost> port=5432 dbname=applypilot_fleet user=<mapped-fleet-role> ...; never the postgres admin or the shared fleet_worker role) before starting this agent."
}
$worker = Join-Path $repo "run-fleet-worker.ps1"
if (-not (Test-Path $worker)) { throw "run-fleet-worker.ps1 not found next to fleet-agent.ps1 ($worker)" }

# ---- startup PRE-FLIGHT: is THIS box ready to run workers + reach the home box? ----
Write-Host "[fleet-agent:$Label] pre-flight (DSN=$env:FLEET_PG_DSN)..." -ForegroundColor Cyan
$pf = @()
& $py -c "import applypilot" 2>$null; if (-not $?) { $pf += "applypilot not importable -> $py -m pip install -e ." }
if (-not (@(".\.conda-env\Scripts\applypilot-fleet-apply.exe", ".\.venv\Scripts\applypilot-fleet-apply.exe") | Where-Object { Test-Path $_ })) { $pf += "applypilot-fleet-apply.exe MISSING -> pip install -e ." }
if (-not $applypilotCli) {
  $pf += "CapSolver readiness UNKNOWN: applypilot.exe MISSING -> pip install -e ."
} else {
  $capProbe = & $applypilotCli fleet-capsolver-check --json 2>$null
  if ($LASTEXITCODE -ne 0) {
    $pf += "CapSolver readiness FAILED on this box -> set CAPSOLVER_API_KEY and verify with applypilot fleet-capsolver-check --json (got '$($capProbe -join ' ')')"
  }
}
if (-not (Get-ChildItem ".\.playwright-browsers\chromium-*" -Directory -ErrorAction SilentlyContinue)) { $pf += "Chromium MISSING in .playwright-browsers (set PLAYWRIGHT_BROWSERS_PATH + reinstall)" }
$probe = (& $py "fleet-agent-query.py" $Label 2>$null | Select-Object -Last 1)
if ("$probe" -notmatch '^\d+\|') {
  $pf += "CANNOT reach home Postgres over FLEET_PG_DSN (got '$probe') -- check the DSN host/LAN/firewall/pg_hba"
} else {
  $wantAgent = ($probe -split '\|')[1]
  $cli = if ($wantAgent -eq 'codex') { Join-Path $env:APPDATA "npm\codex.cmd" } else { Join-Path $env:APPDATA "npm\claude.cmd" }
  if (-not (Test-Path $cli)) { $pf += "$wantAgent CLI missing ($cli) -- workers will flash-and-die until it's installed + logged in" }
  Write-Host "[fleet-agent:$Label] home box wants this machine = '$probe' (workers|agent|model|gen)" -ForegroundColor Gray
}
if ($pf.Count) {
  Write-Host "[fleet-agent:$Label] PRE-FLIGHT PROBLEMS on this box:" -ForegroundColor Red
  $pf | ForEach-Object { Write-Host "   - $_" -ForegroundColor Red }
  Write-Host "[fleet-agent:$Label] fix those and re-run. (Reconciling anyway in 8s; Ctrl-C to abort.)" -ForegroundColor Yellow
  Start-Sleep -Seconds 8
} else {
  Write-Host "[fleet-agent:$Label] pre-flight PASS -- ready to obey the home box." -ForegroundColor Green
}

function Get-LocalWorkers {
  $rx = '--worker-id\s+"?' + [regex]::Escape($Label) + '-(\d+)'
  @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='applypilot-fleet-apply.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match $rx })
}
function Slot-Of($proc) { if ($proc.CommandLine -match ('--worker-id\s+"?' + [regex]::Escape($Label) + '-(\d+)')) { [int]$matches[1] } else { -1 } }

function Get-MachineBlackoutStatus([string]$Role) {
  $lines = @(& $py "fleet-blackout-query.py" $Label $Role 2>$null)
  $queryExit = $LASTEXITCODE
  if ($queryExit -ne 0) {
    return [pscustomobject]@{ State = "KEEP"; Line = "ERROR|blackout-query-exit=$queryExit" }
  }
  if ($lines.Count -eq 0) {
    return [pscustomobject]@{ State = "KEEP"; Line = "ERROR|empty-blackout-status" }
  }
  if ($lines.Count -ne 1) {
    return [pscustomobject]@{ State = "KEEP"; Line = "ERROR|multiline-blackout-status" }
  }

  $line = "$($lines[0])"
  if ($line -match "[`r`n]") {
    return [pscustomobject]@{ State = "KEEP"; Line = "ERROR|multiline-blackout-status" }
  }
  $expectedLabel = $Label.Trim().ToLowerInvariant()
  $expectedRole = $Role.Trim().ToLowerInvariant()
  if ($line -ceq "OK|$expectedLabel|$expectedRole|||") {
    return [pscustomobject]@{ State = "OK"; Line = $line }
  }

  $parts = $line -split '\|', 7
  $expiration = [datetimeoffset]::MinValue
  [string[]]$expirationFormats = @(
    "yyyy-MM-dd'T'HH:mm:sszzz",
    "yyyy-MM-dd'T'HH:mm:ss.FFFFFFFzzz"
  )
  $expirationValid = $false
  if ($parts.Count -eq 6) {
    $expirationValid = [datetimeoffset]::TryParseExact(
      $parts[4],
      $expirationFormats,
      [Globalization.CultureInfo]::InvariantCulture,
      [Globalization.DateTimeStyles]::None,
      [ref]$expiration
    )
  }
  if ($parts.Count -eq 6 -and $parts[0] -ceq "BLOCKED" -and
      $parts[1] -ceq $expectedLabel -and $parts[2] -ceq $expectedRole -and
      $parts[3].Trim().Length -gt 0 -and $expirationValid -and
      $expiration -gt [datetimeoffset]::UtcNow) {
    return [pscustomobject]@{ State = "BLOCKED"; Line = $line }
  }
  return [pscustomobject]@{ State = "KEEP"; Line = $line }
}

# ---- auto-update (spec: 2026-07-03-fleet-pull-updater-design.md) ----
$updateLog = Join-Path $repo ".fleet-logs\fleet-agent-update.log"
$updateRestartMarker = Join-Path $repo ".fleet-logs\fleet-agent-update-restart.sha"
function Log-Update([string]$msg, [string]$color = "Gray") {
  $line = "[{0}] [fleet-agent:{1}] {2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Label, $msg
  Write-Host $line -ForegroundColor $color
  try {
    $dir = Split-Path -Parent $updateLog
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $f = Get-Item $updateLog -ErrorAction SilentlyContinue
    if ($f -and $f.Length -gt 50MB) { Move-Item $updateLog "$updateLog.1" -Force }  # single-slot rotation
    Add-Content -Path $updateLog -Value $line
  } catch { Write-Host "[fleet-agent:$Label] WARNING: update-log write failed: $_" -ForegroundColor Yellow }
}
function Get-ShortSha([string]$sha) { if ($sha -and $sha.Length -ge 9) { $sha.Substring(0, 9) } else { "$sha" } }

function Get-PinnedWorkerVersion {
  $versionLines = @(& $py "fleet-agent-version.py" 2>$null)
  $versionExit = $LASTEXITCODE
  if ($versionExit -ne 0) {
    return [pscustomobject]@{ Valid = $false; Reason = "query exit=$versionExit"; CurrentVersion = $null; PinnedVersion = $null }
  }
  if ($versionLines.Count -ne 1) {
    return [pscustomobject]@{ Valid = $false; Reason = "expected one line, got $($versionLines.Count)"; CurrentVersion = $null; PinnedVersion = $null }
  }
  $versionLine = "$($versionLines[0])"
  $vf = $versionLine -split '\|', 4
  if ($vf.Count -lt 4 -or $vf[0] -cne "OK") {
    return [pscustomobject]@{ Valid = $false; Reason = "malformed status '$versionLine'"; CurrentVersion = $null; PinnedVersion = $null }
  }
  if ([string]::IsNullOrWhiteSpace($vf[2])) {
    return [pscustomobject]@{ Valid = $false; Reason = "pinned_worker_version is missing or blank"; CurrentVersion = $vf[1]; PinnedVersion = $null }
  }
  return [pscustomobject]@{ Valid = $true; Reason = $null; CurrentVersion = $vf[1]; PinnedVersion = $vf[2] }
}

function Confirm-PinnedWorkerVersion([string]$CapturedPinnedVersion) {
  $freshVersion = Get-PinnedWorkerVersion
  if (-not $freshVersion.Valid) {
    Log-Update "UPDATE BLOCKED: pinned_worker_version revalidation failed immediately before merge ($($freshVersion.Reason))" "Red"
    return $false
  }
  if ($freshVersion.PinnedVersion -cne $CapturedPinnedVersion) {
    Log-Update "UPDATE BLOCKED: pinned_worker_version changed immediately before merge (captured '$CapturedPinnedVersion', now '$($freshVersion.PinnedVersion)')" "Red"
    return $false
  }
  return $true
}

function Test-InstalledAgentWrapper {
  $wrapperPath = Join-Path $repo ".fleet-logs\_task-wrappers\fleet-agent-task.ps1"
  $reRegister = "re-run register-fleet-tasks.ps1 elevated on this node"
  if (-not (Test-Path -LiteralPath $wrapperPath -PathType Leaf)) {
    Log-Update "UPDATE DEFERRED: installed FleetAgent wrapper is missing; operator re-registration required ($reRegister)" "Red"
    return $false
  }

  try {
    $tokens = $null
    $parseErrors = $null
    $wrapperAst = [System.Management.Automation.Language.Parser]::ParseFile(
      $wrapperPath, [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count -ne 0 -or $null -eq $wrapperAst.EndBlock) { throw "wrapper does not parse cleanly" }
    $expectedAgent = [IO.Path]::GetFullPath((Join-Path $repo "fleet-agent.ps1"))
    $agentCommands = @($wrapperAst.FindAll({
      param($node)
      if ($node -isnot [System.Management.Automation.Language.CommandAst]) { return $false }
      $commandName = $node.GetCommandName()
      if (-not $commandName) { return $false }
      try { return [IO.Path]::GetFullPath($commandName) -ieq $expectedAgent } catch { return $false }
    }, $true))
    if ($agentCommands.Count -ne 1) { throw "expected one direct fleet-agent.ps1 invocation, got $($agentCommands.Count)" }

    $statements = @($wrapperAst.EndBlock.Statements)
    $commandOffset = $agentCommands[0].Extent.StartOffset
    $commandStatementIndex = -1
    for ($i = 0; $i -lt $statements.Count; $i++) {
      if ($statements[$i].Extent.StartOffset -le $commandOffset -and
          $statements[$i].Extent.EndOffset -ge $agentCommands[0].Extent.EndOffset) {
        $commandStatementIndex = $i
        break
      }
    }
    if ($commandStatementIndex -ne ($statements.Count - 2)) { throw "fleet-agent invocation is not immediately before the final statement" }
    if ($statements[-1].Extent.Text.Trim() -cnotmatch '(?i)^exit\s+\$LASTEXITCODE$') {
      throw "final statement is not explicit exit `$LASTEXITCODE"
    }
    return $true
  } catch {
    Log-Update "UPDATE DEFERRED: installed FleetAgent wrapper is old or ambiguous; operator re-registration required ($reRegister): $_" "Red"
    return $false
  }
}

function Get-UpdateRestartTarget {
  if (-not (Test-Path -LiteralPath $updateRestartMarker -PathType Leaf)) { return $null }
  try {
    $lines = @(Get-Content -LiteralPath $updateRestartMarker -ErrorAction Stop)
    if ($lines.Count -ne 1) { throw "expected one line, got $($lines.Count)" }
    $markerFields = "$($lines[0])".Trim() -split '\|', 2
    $target = $markerFields[0]
    if ($target -cnotmatch '^[0-9a-fA-F]{40}$') { throw "target SHA is invalid" }
    $preMergeHead = $null
    if ($markerFields.Count -eq 2) {
      if ($markerFields[1] -cnotmatch '^[0-9a-fA-F]{40}$') { throw "pre-merge SHA is invalid" }
      $preMergeHead = $markerFields[1]
    }
    return [pscustomobject]@{ Target = $target; PreMergeHead = $preMergeHead }
  } catch {
    Log-Update "restart marker invalid; retaining it and preserving workers: $_" "Red"
    return $null
  }
}

function Write-UpdateRestartMarker([string]$Target, [string]$PreMergeHead) {
  $dir = Split-Path -Parent $updateRestartMarker
  $tempMarker = "$updateRestartMarker.tmp.$PID"
  try {
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    Set-Content -LiteralPath $tempMarker -Value "$Target|$PreMergeHead" -Encoding Ascii -NoNewline -ErrorAction Stop
    Move-Item -LiteralPath $tempMarker -Destination $updateRestartMarker -Force -ErrorAction Stop
    return $true
  } catch {
    Remove-Item -LiteralPath $tempMarker -Force -ErrorAction SilentlyContinue
    Log-Update "UPDATE BLOCKED: could not persist restart marker for $(Get-ShortSha $Target): $_" "Red"
    return $false
  }
}

function Clear-UpdateRestartMarker {
  if (-not (Test-Path -LiteralPath $updateRestartMarker -PathType Leaf)) { return $true }
  try {
    Remove-Item -LiteralPath $updateRestartMarker -Force -ErrorAction Stop
    return $true
  } catch {
    Log-Update "controlled restart marker could not be cleared; supervisor relaunch required: $_" "Red"
    return $false
  }
}

function Clear-UnappliedUpdateRestartMarker(
  [string]$PreMergeHead, [string]$ObservedHead, [int]$ObservedExit, [string]$Context) {
  if ($ObservedExit -eq 0 -and
      $ObservedHead -cmatch '^[0-9a-fA-F]{40}$' -and
      $ObservedHead -ceq $PreMergeHead) {
    Log-Update "${Context}: merge confirmed not applied; clearing restart marker at $(Get-ShortSha $PreMergeHead)" "Yellow"
    return (Clear-UpdateRestartMarker)
  }
  Log-Update "${Context}: HEAD is unavailable, malformed, or not the captured pre-merge commit; retaining restart marker" "Red"
  return $false
}

function Complete-PendingUpdateRestart([pscustomobject]$MachinePolicy) {
  if (-not (Test-Path -LiteralPath $updateRestartMarker -PathType Leaf)) { return "NONE" }
  $restartState = Get-UpdateRestartTarget
  if (-not $restartState) { return "WAIT" }
  $target = $restartState.Target

  $currentHead = "$(& git rev-parse HEAD 2>$null)".Trim()
  $headExit = $LASTEXITCODE
  if ($headExit -ne 0 -or $currentHead -cnotmatch '^[0-9a-fA-F]{40}$') {
    Log-Update "restart marker deferred: current HEAD is unavailable or invalid" "Red"
    return "WAIT"
  }
  if ($currentHead -cne $target) {
    if ($restartState.PreMergeHead -and $currentHead -ceq $restartState.PreMergeHead) {
      Log-Update "discarding unapplied restart marker for $(Get-ShortSha $target); HEAD remains captured pre-merge $(Get-ShortSha $currentHead)" "Yellow"
      if (-not (Clear-UpdateRestartMarker)) { return "WAIT" }
      return "NONE"
    }
    Log-Update "restart marker deferred: HEAD $(Get-ShortSha $currentHead) is neither target $(Get-ShortSha $target) nor a known captured pre-merge commit" "Red"
    return "WAIT"
  }
  if ($null -eq $MachinePolicy -or @("OK", "BLOCKED") -cnotcontains $MachinePolicy.State) {
    Log-Update "post-merge restart deferred: machine blackout policy is not a valid OK/BLOCKED verdict" "Yellow"
    return "WAIT"
  }

  $gate = (& $py "fleet-agent-update-gate.py" $Label 2>$null | Select-Object -Last 1)
  if ("$gate" -cne "IDLE") {
    Log-Update "post-merge restart deferred: update gate is not IDLE (got '$gate')" "Yellow"
    return "WAIT"
  }

  Log-Update "post-merge policy and IDLE gate valid -- restarting workers for $(Get-ShortSha $target)" "Cyan"
  Get-LocalWorkers | ForEach-Object {
    Log-Update "-stop $Label-$(Slot-Of $_) (controlled post-update restart)" "Yellow"
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
  }
  Start-Sleep -Seconds 2
  if (-not (Clear-UpdateRestartMarker)) { return "EXIT" }

  if (-not $agentStartHead -or $agentStartHead -cne $target) { return "EXIT" }
  return "RECONCILE"
}

# KEEP recovery is deliberately limited to the agent, blackout query/policy, and their focused tests.
# Any other target path must wait for the normal between-jobs updater after a valid policy verdict.
$recoveryFiles = @("fleet-agent.ps1", "fleet-blackout-query.py",
                   "register-fleet-tasks.ps1",
                   "src/applypilot/fleet/machine_blackout.py",
                   "tests/test_fleet_agent_autoupdate_script.py",
                   "tests/test_fleet_machine_blackout.py",
                   "tests/test_fleet_machine_blackout_scripts.py")
$script:lastUpdateCheck = [datetime]::MinValue
$script:updatePending = $false
$script:updBranch = $null
$script:updRemote = $null

function Invoke-AutoUpdate([pscustomobject]$ExpectedMachinePolicy, [switch]$RecoveryOnly) {
  # Normal mode merges with workers running, then completes a marker-controlled restart.
  # Recovery mode never mutates workers and exits after an allowlisted update is applied.
  if (Test-Path -LiteralPath $updateRestartMarker -PathType Leaf) { return $false }
  $due = ((Get-Date) - $script:lastUpdateCheck).TotalSeconds -ge $UpdateEverySec
  if (-not ($due -or $script:updatePending)) { return $false }

  if ($due) {
    $script:lastUpdateCheck = Get-Date
    $branch = (& git rev-parse --abbrev-ref HEAD 2>$null)
    $branchExit = $LASTEXITCODE
    if ($branchExit -ne 0 -or -not $branch -or $branch -eq "HEAD") { Log-Update "skip: could not resolve a branch ('$branch', exit=$branchExit)" "Yellow"; $script:updatePending = $false; return $false }
    $remote = (& git config "branch.$branch.remote" 2>$null); if (-not $remote) { $remote = "origin" }
    & git fetch $remote --quiet 2>$null
    if ($LASTEXITCODE -ne 0) { Log-Update "skip: git fetch $remote failed (offline?)" "Yellow"; $script:updatePending = $false; return $false }
    $script:updBranch = $branch; $script:updRemote = $remote
  }
  $branch = $script:updBranch; $remote = $script:updRemote
  if (-not $branch -or -not $remote) { $script:updatePending = $false; return $false }
  $target = (& git rev-parse "$remote/$branch" 2>$null)
  $targetExit = $LASTEXITCODE
  $target = "$target".Trim()
  if ($targetExit -ne 0 -or $target -cnotmatch '^[0-9a-fA-F]{40}$') {
    Log-Update "UPDATE BLOCKED: could not capture a valid target commit for $remote/$branch (exit=$targetExit)" "Red"
    $script:updatePending = $false; return $false
  }
  # Pin-aware guard: branch heads are transport; fleet_config.pinned_worker_version is
  # the release contract. This prevents a clean worker from pulling an unpinned branch
  # tip just because someone pushed ahead of the fleet pin.
  $versionStatus = Get-PinnedWorkerVersion
  if (-not $versionStatus.Valid) {
    Log-Update "UPDATE BLOCKED: could not capture pinned_worker_version ($($versionStatus.Reason))" "Red"
    $script:updatePending = $false; return $false
  }
  $currentVersion = $versionStatus.CurrentVersion
  $pinnedVersion = $versionStatus.PinnedVersion
  $packageVersion = $currentVersion
  if ($packageVersion -match '^(.+)\+git\.') { $packageVersion = $matches[1] } else { $packageVersion = "0.3.0" }
  $targetTree = (& git rev-parse "$target^{tree}" 2>$null)
  $targetTreeExit = $LASTEXITCODE
  $targetTree = "$targetTree".Trim()
  if ($targetTreeExit -ne 0 -or $targetTree -cnotmatch '^[0-9a-fA-F]{40}$') {
    Log-Update "UPDATE BLOCKED: could not resolve a valid tree for captured target $(Get-ShortSha $target) (exit=$targetTreeExit)" "Red"
    $script:updatePending = $false; return $false
  }
  $targetVersion = "$packageVersion+git.tree.$($targetTree.Substring(0, 7))"
  if ($pinnedVersion) {
    if (-not $targetVersion) {
      Log-Update "UPDATE BLOCKED: pinned version $pinnedVersion but could not resolve remote tree for $remote/$branch" "Red"
      $script:updatePending = $false; return $false
    }
    if ($targetVersion -ne $pinnedVersion) {
      if ($currentVersion -eq $pinnedVersion) {
        Log-Update "skip: remote tree $targetVersion is not pinned $pinnedVersion" "Yellow"
      } else {
        Log-Update "UPDATE BLOCKED: pinned version $pinnedVersion but remote tree $targetVersion is not pinned" "Red"
      }
      $script:updatePending = $false; return $false
    }
  }
  $local = (& git rev-parse HEAD 2>$null)
  $localExit = $LASTEXITCODE
  $local = "$local".Trim()
  if ($localExit -ne 0 -or $local -cnotmatch '^[0-9a-fA-F]{40}$') {
    Log-Update "UPDATE BLOCKED: could not resolve a valid local HEAD (exit=$localExit)" "Red"
    $script:updatePending = $false; return $false
  }
  if ($local -ceq $target) { $script:updatePending = $false; return $false }

  # guards: worker boxes must never own local commits or edits
  $statusLines = @(& git status --porcelain 2>$null)
  $statusExit = $LASTEXITCODE
  if ($statusExit -ne 0) {
    Log-Update "UPDATE BLOCKED: git status failed (exit=$statusExit)" "Red"
    $script:updatePending = $false; return $false
  }
  if ($statusLines.Count -gt 0) {
    Log-Update "UPDATE BLOCKED: working tree dirty -- this box must be a clean clone" "Red"
    $script:updatePending = $false; return $false
  }
  & git merge-base --is-ancestor $local $target 2>$null
  if ($LASTEXITCODE -ne 0) {
    Log-Update "UPDATE BLOCKED: history diverged from captured target $(Get-ShortSha $target) -- fix by hand" "Red"
    $script:updatePending = $false; return $false
  }

  $targetChanges = @(& git diff --name-only $local $target 2>$null |
    ForEach-Object { $_ -replace '\\', '/' })
  if ($LASTEXITCODE -ne 0) {
    Log-Update "UPDATE BLOCKED: could not inspect target changes for $remote/$branch" "Red"
    $script:updatePending = $false; return $false
  }

  if ($RecoveryOnly) {
    $broaderChanges = @($targetChanges | Where-Object { $recoveryFiles -notcontains $_ })
    if ($broaderChanges.Count -gt 0) {
      Log-Update "KEEP recovery deferred: target includes non-recovery files ($($broaderChanges -join ', '))" "Yellow"
      $script:updatePending = $false; return $false
    }

    if (-not (Test-InstalledAgentWrapper)) {
      $script:updatePending = $false; return $false
    }
    Log-Update "KEEP recovery update: $(Get-ShortSha $local) -> $(Get-ShortSha $target) ($($targetChanges.Count) allowlisted files); workers remain running" "Cyan"
    if (-not (Confirm-PinnedWorkerVersion $pinnedVersion)) {
      $script:updatePending = $false; return $false
    }
    & git merge --ff-only --quiet $target 2>$null
    if ($LASTEXITCODE -ne 0) {
      Log-Update "KEEP RECOVERY FAILED: ff-only merge refused -- leaving tree at $(Get-ShortSha $local)" "Red"
      $script:updatePending = $false; return $false
    }
    $mergedHead = (& git rev-parse HEAD 2>$null)
    $mergedHeadExit = $LASTEXITCODE
    $mergedHead = "$mergedHead".Trim()
    if ($mergedHeadExit -ne 0 -or $mergedHead -cne $target) {
      Log-Update "KEEP RECOVERY FAILED: HEAD verification did not match captured target $(Get-ShortSha $target)" "Red"
      $script:updatePending = $false
      exit 1
    }
    $script:updatePending = $false
    Log-Update "KEEP recovery applied -- exiting for supervisor relaunch before the next policy verdict" "Yellow"
    exit 1
  }

  if ($targetChanges -contains "pyproject.toml") {
    Log-Update "pyproject.toml changed; automatic update deferred because dependency mutation is not restart-safe" "Yellow"
    $script:updatePending = $false; return $false
  }

  if (-not $script:updatePending) { Log-Update "update available: $(Get-ShortSha $local) -> $(Get-ShortSha $target) (waiting for between-jobs window)" "Cyan" }
  $script:updatePending = $true

  # between-jobs gate (fail-closed: anything but IDLE means wait)
  $gate = (& $py "fleet-agent-update-gate.py" $Label 2>$null | Select-Object -Last 1)
  if ("$gate" -ne "IDLE") { return $false }

  $freshMachinePolicy = Get-MachineBlackoutStatus "all"
  if ($null -eq $ExpectedMachinePolicy -or
      $freshMachinePolicy.State -cne $ExpectedMachinePolicy.State -or
      $freshMachinePolicy.Line -cne $ExpectedMachinePolicy.Line) {
    Log-Update "update deferred: machine blackout policy changed before worker mutation" "Yellow"
    return $false
  }

  if (-not (Test-InstalledAgentWrapper)) {
    $script:updatePending = $false; return $false
  }

  if (-not (Write-UpdateRestartMarker $target $local)) {
    $script:updatePending = $false; return $false
  }

  Log-Update "between-jobs window open -- merging captured target while current workers remain running" "Cyan"
  if (-not (Confirm-PinnedWorkerVersion $pinnedVersion)) {
    $headBeforeMerge = "$(& git rev-parse HEAD 2>$null)".Trim()
    $headBeforeMergeExit = $LASTEXITCODE
    Clear-UnappliedUpdateRestartMarker $local $headBeforeMerge $headBeforeMergeExit "UPDATE ABORTED" | Out-Null
    $script:updatePending = $false; return $false
  }
  & git merge --ff-only --quiet $target 2>$null
  if ($LASTEXITCODE -ne 0) {
    $failedMergeHead = "$(& git rev-parse HEAD 2>$null)".Trim()
    $failedMergeHeadExit = $LASTEXITCODE
    Clear-UnappliedUpdateRestartMarker $local $failedMergeHead $failedMergeHeadExit "UPDATE FAILED" | Out-Null
    Log-Update "UPDATE FAILED: ff-only merge refused -- leaving tree at $(Get-ShortSha $local); workers remain running" "Red"
    $script:updatePending = $false; return $false
  }
  $mergedHead = (& git rev-parse HEAD 2>$null)
  $mergedHeadExit = $LASTEXITCODE
  $mergedHead = "$mergedHead".Trim()
  if ($mergedHeadExit -ne 0 -or $mergedHead -cne $target) {
    Clear-UnappliedUpdateRestartMarker $local $mergedHead $mergedHeadExit "UPDATE FAILED" | Out-Null
    Log-Update "UPDATE FAILED: HEAD verification did not match captured target $(Get-ShortSha $target); workers remain running" "Red"
    $script:updatePending = $false; return $false
  }
  $script:updatePending = $false
  $changed = $targetChanges
  Log-Update "updated $(Get-ShortSha $local) -> $(Get-ShortSha $target) ($($changed.Count) files)" "Green"

  $postMergePolicy = Get-MachineBlackoutStatus "all"
  $restartAction = Complete-PendingUpdateRestart -MachinePolicy $postMergePolicy
  if ($restartAction -eq "EXIT") {
    Log-Update "controlled restart complete under old agent code -- exiting for supervisor relaunch" "Yellow"
    exit 1
  }
  return ($restartAction -eq "RECONCILE")
}

$lastGen = $null
Write-Host "[fleet-agent:$Label] online -- reconciling LOCAL workers to fleet_desired_state every ${PollSec}s (Ctrl-C to stop)" -ForegroundColor Cyan
while ($true) {
  $machinePolicy = Get-MachineBlackoutStatus "all"
  $restartAction = Complete-PendingUpdateRestart -MachinePolicy $machinePolicy
  if ($restartAction -eq "EXIT") {
    Log-Update "pending controlled restart complete under old agent code -- exiting for supervisor relaunch" "Yellow"
    exit 1
  }
  if ($restartAction -eq "WAIT") {
    Start-Sleep -Seconds $PollSec
    continue
  }
  if ($machinePolicy.State -eq "KEEP") {
    Write-Host "[fleet-agent:$Label] blackout status unavailable or invalid; preserving existing workers for this tick. $($machinePolicy.Line)" -ForegroundColor Yellow
    if ($AutoUpdate) { Invoke-AutoUpdate -RecoveryOnly | Out-Null }
    Start-Sleep -Seconds $PollSec
    continue
  }

  # auto-update runs BEFORE reconcile so a post-update respawn happens in this same tick
  if ($AutoUpdate) {
    Invoke-AutoUpdate -ExpectedMachinePolicy $machinePolicy | Out-Null
    $machinePolicy = Get-MachineBlackoutStatus "all"
    $restartAction = Complete-PendingUpdateRestart -MachinePolicy $machinePolicy
    if ($restartAction -eq "EXIT") {
      Log-Update "post-update controlled restart complete under old agent code -- exiting for supervisor relaunch" "Yellow"
      exit 1
    }
    if ($restartAction -eq "WAIT") {
      Start-Sleep -Seconds $PollSec
      continue
    }
    if ($machinePolicy.State -eq "KEEP") {
      Write-Host "[fleet-agent:$Label] blackout status changed or became uncertain after update check; preserving existing workers for this tick. $($machinePolicy.Line)" -ForegroundColor Yellow
      Start-Sleep -Seconds $PollSec
      continue
    }
  }

  $line = (& $py "fleet-agent-query.py" $Label 2>$null | Select-Object -Last 1)
  $f = "$line" -split '\|'
  if ($f.Count -lt 4 -or $f[0] -eq 'KEEP') { Start-Sleep -Seconds $PollSec; continue }  # DB blip -> leave as-is
  $want = [int]$f[0]; $agent = $f[1]; $model = $f[2]; $gen = [int]$f[3]
  $desiredWant = $want
  if ($machinePolicy.State -eq "BLOCKED") {
    $parts = $machinePolicy.Line -split '\|', 6
    $policyName = if ($parts.Count -ge 4) { $parts[3] } else { "machine blackout" }
    $expiresAt = if ($parts.Count -ge 5) { $parts[4] } else { "" }
    if ($want -gt 0) {
      Write-Host "[fleet-agent:$Label] machine blackout active; effective desired_workers 0 (configured $want). policy=$policyName until=$expiresAt" -ForegroundColor Yellow
    }
    $want = 0
  }
  $blackout = (& $py -m applypilot.fleet.work_hours $Label 2>$null | Select-Object -Last 1)
  if ("$blackout" -match '^BLACKOUT\|') {
    if ($want -gt 0) {
      Write-Host "[fleet-agent:$Label] work-hours blackout active; effective desired_workers 0 (configured $want). Set APPLYPILOT_ALLOW_WORK_HOURS_APPLY=1 to override." -ForegroundColor Yellow
    }
    $want = 0
  }

  $procs = Get-LocalWorkers
  # A worker is TWO OS processes (the pip .exe wrapper + its python.exe child), both carrying the
  # same --worker-id. Count DISTINCT SLOTS, never raw processes: counting processes reads 1 worker
  # as 2 -> "over target" -> kill -> next poll sees 0 -> respawn, forever (live 7/03: this loop
  # leaked 51 launcher windows in 40 min on home and killed home-0 mid-apply every ~42s).
  $slotGroups = @($procs | Group-Object { Slot-Of $_ })
  $have = $slotGroups.Count

  # generation bump -> the home box asked for a clean restart: kill all local, then re-spawn to $want
  if ($null -ne $lastGen -and $gen -ne $lastGen -and $want -gt 0) {
    Write-Host "[fleet-agent:$Label] generation $lastGen->$gen : restarting all local workers" -ForegroundColor Yellow
    $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2; $procs = @(); $have = 0
  } elseif ($null -ne $lastGen -and $gen -ne $lastGen -and $desiredWant -gt 0 -and $want -eq 0) {
    Write-Host "[fleet-agent:$Label] generation $lastGen->$gen observed during work-hours blackout; deferring restart until blackout clears" -ForegroundColor Yellow
  }
  $lastGen = $gen

  if ($have -lt $want) {
    $running = @($slotGroups | ForEach-Object { [int]$_.Name })
    $started = 0; $slot = 0
    while ($started -lt ($want - $have) -and $slot -le 20) {
      if ($running -notcontains $slot) {
        Assert-NoLifecycleFaults
        # -File must be pre-quoted: Start-Process joins array args with spaces WITHOUT quoting,
        # so a repo path containing spaces (home's OneDrive checkout) otherwise truncates to
        # "-File C:\...\New" and the child exits before the launcher ever runs (m2/m4's
        # C:\ApplyPilot masked this for days).
        # No -NoExit: a launcher window must close when its worker exits, or every worker death
        # leaves an eternal window (transcripts live in .applypilot\logs\worker-<slot>.log).
        $argList = @("-ExecutionPolicy", "Bypass", "-File", "`"$worker`"", "-Slot", $slot, "-Agent", $agent, "-Label", $Label)
        if ($model) { $argList += @("-Model", $model) }
        Start-Process powershell.exe -ArgumentList $argList -WorkingDirectory $repo
        Write-Host "[fleet-agent:$Label] +start $Label-$slot ($agent$(if($model){"/$model"}))" -ForegroundColor Green
        $running += $slot; $started++; Start-Sleep -Milliseconds 800
      }
      $slot++
    }
  }
  elseif ($have -gt $want) {
    # Scale down by SLOT (highest first), killing every process of that slot (exe wrapper + python child).
    $slotGroups | Sort-Object { [int]$_.Name } -Descending | Select-Object -First ($have - $want) | ForEach-Object {
      Write-Host "[fleet-agent:$Label] -stop $Label-$($_.Name) (scale-down / offload)" -ForegroundColor Yellow
      $_.Group | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }
  }
  Start-Sleep -Seconds $PollSec
}
