# run-fleet-worker.ps1 [-Slot N] [-Agent claude|codex] [-Model name] [-Label name]
#                      [-FallbackAgent "codex"]
#   Launch ONE offsite apply worker. Use a DISTINCT -Slot per worker on the same machine
#   (slot keys the browser profile + CDP port + logs so they don't collide).
#   Use a DISTINCT -Label per MACHINE (home box = "home", second box = e.g. "m2") so the
#   worker-id (= "<Label>-<Slot>") is unique fleet-wide -- otherwise two machines both stamp
#   "home-0" and lease attribution / monitoring can't tell them apart.
#   Run each in its own window:  .\run-fleet-worker.ps1 -Slot 0 -Agent codex -Label m2
#   Requires FLEET_PG_DSN already set (the setup script persists it). LinkedIn never runs here.
#
#   -FallbackAgent: an ordered chain the worker switches to when -Agent hits its usage/session
#   limit (each an INDEPENDENT quota pool), e.g. "codex". Defaults to the
#   APPLYPILOT_FALLBACK_AGENT env var, else "codex" when -Agent is claude (so a Claude
#   session-limit wall fails over to Codex instead of stalling).
param([int]$Slot = 0, [string]$Agent = "claude", [string]$Model = "", [string]$Label = "home",
      [string]$HomeIp = $env:FLEET_HOME_IP,
      [string]$FallbackAgent = $env:APPLYPILOT_FALLBACK_AGENT)
$WorkerId = "$Label-$Slot"
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# Host-identity guard (see fleet-agent.ps1): a box declares its own fleet label in
# APPLYPILOT_FLEET_LABEL (home/m2/m4) and must NOT physically host another machine's workers.
# Blocks the direct-launch path too (someone running this script with a foreign -Label). The
# Python worker (enforce_host_identity) backstops this for manual/SSH launches. Unset = permissive.
$boxLabel = "$env:APPLYPILOT_FLEET_LABEL".Trim()
if ($boxLabel -and ($boxLabel -ne $Label)) {
  throw "run-fleet-worker: this box is '$boxLabel' but -Label '$Label' was requested. Refusing to launch another machine's worker here (m2/TARPON workers must never run on the home box). Use -Label $boxLabel, or run this on the '$Label' box."
}

# Guard against the multi-tab collision: refuse a SECOND worker on the SAME worker-id/slot.
# Multiple processes on one slot share ONE Chrome profile + CDP debug port and fight over it
# (the "3-4 tabs, stuck/crash" symptom). For parallelism use DISTINCT -Slot values (each gets
# its own isolated browser), or run-fleet-workers.ps1 -Count N.
$dupe = Get-CimInstance Win32_Process -Filter "Name='applypilot-fleet-apply.exe' OR Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match ('--worker-id\s+"?' + [regex]::Escape($WorkerId) + '"?(\s|$)') }
if ($dupe) {
  throw "Worker '$WorkerId' (slot $Slot) is ALREADY running (PID $($dupe.ProcessId -join ', ')). Another worker on the SAME slot collides on one Chrome. Use a different -Slot for parallelism, or stop the existing one first."
}

# Python env: the home box uses .conda-env; a bootstrapped machine uses .venv. Find whichever exists.
$exe = $null
foreach ($d in @(".\.conda-env\Scripts", ".\.venv\Scripts")) {
  $cand = Join-Path $d "applypilot-fleet-apply.exe"
  if (Test-Path $cand) { $exe = (Resolve-Path $cand).Path; break }
}
if (-not $exe) { throw "applypilot-fleet-apply not found in .conda-env or .venv -- run the setup script first." }
$scriptsDir = Split-Path -Parent $exe
$applypilotCli = Join-Path $scriptsDir "applypilot.exe"
if (-not (Test-Path $applypilotCli)) { throw "applypilot.exe not found next to applypilot-fleet-apply.exe -- run pip install -e . first." }
$py = $null
foreach ($d in @(".\.conda-env\python.exe", ".\.venv\Scripts\python.exe")) {
  if (Test-Path $d) { $py = (Resolve-Path $d).Path; break }
}
if (-not $py) { throw "python not found (.conda-env or .venv) -- run the setup script first." }

function Test-MachineBlackout([string]$Role) {
  $stderrPath = [IO.Path]::GetTempFileName()
  try {
    $lines = @(& $py (Join-Path $ProjectRoot "fleet-blackout-query.py") $Label $Role 2> $stderrPath)
    $queryExit = $LASTEXITCODE
    Get-Content -LiteralPath $stderrPath -ErrorAction SilentlyContinue | ForEach-Object {
      if (-not [string]::IsNullOrWhiteSpace("$_")) { [Console]::Error.WriteLine("$_") }
    }
  } finally {
    Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue
  }
  if ($queryExit -ne 0) { return "ERROR|blackout-query-exit=$queryExit" }
  if ($lines.Count -eq 0) { return "ERROR|empty-blackout-status" }
  if ($lines.Count -ne 1) { return "ERROR|multiline-blackout-status" }
  $line = "$($lines[0])"
  if ($line -match "[`r`n]") { return "ERROR|multiline-blackout-status" }
  $expected = "OK|$($Label.Trim().ToLowerInvariant())|$($Role.Trim().ToLowerInvariant())|||"
  if ($line -ceq $expected) { return $null }
  return $line
}

$machinePolicyFailure = Test-MachineBlackout "apply"
if ($machinePolicyFailure) {
  throw "Refusing to start apply worker '$WorkerId': machine blackout status did not return exact OK. $machinePolicyFailure"
}

function Resolve-FleetHomeIp([string]$Candidate) {
  $ip = "$Candidate".Trim()
  if ($ip -and $ip -ne "0.0.0.0" -and $ip -ne "::") { return $ip }

  $tailscale = Get-Command "tailscale.exe" -ErrorAction SilentlyContinue
  if ($tailscale) {
    $tsIp = (& $tailscale.Source ip -4 2>$null | Where-Object { $_ -like "100.*" } | Select-Object -First 1)
    if ($tsIp) { return "$tsIp".Trim() }
  }
  return $null
}

$HomeIp = Resolve-FleetHomeIp $HomeIp
if (-not $HomeIp -or $HomeIp -eq "0.0.0.0" -or $HomeIp -eq "::") {
  throw "Refusing to start worker '$WorkerId': FLEET_HOME_IP is missing/invalid and tailscale.exe ip -4 did not return a 100.x address. Set FLEET_HOME_IP to this machine's Tailscale IP, then restart FleetAgent."
}
$env:FLEET_HOME_IP = $HomeIp
$env:FLEET_MACHINE_OWNER = $Label

$env:APPLYPILOT_DIR = Join-Path $ProjectRoot ".applypilot"
if (-not $env:FLEET_PG_DSN) { throw "FLEET_PG_DSN is not set (the setup script persists it; open a fresh window)." }
$env:APPLYPILOT_FLEET_DSN = $env:FLEET_PG_DSN
Write-Host "[fleet-worker] checking fleet Postgres connectivity ..."
$probeLines = @(& $py (Join-Path $ProjectRoot "fleet-agent-query.py") $Label 2>&1)
$probe = "$($probeLines | Select-Object -Last 1)"
if ($probe -notmatch '^\d+\|') {
  throw "Cannot reach fleet Postgres over FLEET_PG_DSN before starting worker '$WorkerId' (probe='$($probeLines -join ' ')'). On m2/m4 set FLEET_PG_DSN to your mapped per-node fleet DSN (host=<home Tailscale IP> port=5432 dbname=applypilot_fleet ...; never the postgres admin or the shared fleet_worker role)."
}
Write-Host "[fleet-worker] checking CapSolver fleet readiness ..."
$capProbe = & $applypilotCli fleet-capsolver-check --json 2>&1
if ($LASTEXITCODE -ne 0) {
  throw "Refusing to start worker '$WorkerId': CapSolver fleet readiness failed. $($capProbe -join ' ')"
}
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $ProjectRoot ".playwright-browsers"
$chromium = Get-ChildItem -Path $env:PLAYWRIGHT_BROWSERS_PATH -Directory -Filter "chromium-*" -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 1
if ($chromium) { $env:CHROME_PATH = Join-Path $chromium.FullName "chrome-win64\chrome.exe" }
$env:Path = "C:\Program Files\nodejs;$env:Path"

# Apply-agent CLI path (must be the AUTHENTICATED one): claude or codex.
if ($Agent -eq "claude") {
  $a = Join-Path $env:APPDATA "npm\claude.cmd"; if (Test-Path $a) { $env:CLAUDE_PATH = $a }
  if (-not $Model) { $Model = "sonnet" }
} elseif ($Agent -eq "codex") {
  $a = Join-Path $env:APPDATA "npm\codex.cmd"; if (Test-Path $a) { $env:CODEX_PATH = $a }
  # leave $Model empty -> codex uses its default model
} else { throw "unknown -Agent '$Agent' (use claude or codex)" }

# Failover chain: default a claude worker to codex so a Claude session-limit wall fails over
# instead of stalling. Resolve the CLI path for every fallback agent so the switcher can
# actually launch it.
if (-not $FallbackAgent -and $Agent -eq "claude") { $FallbackAgent = "codex" }
foreach ($fa in ($FallbackAgent -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
  if ($fa -eq "claude") { $c = Join-Path $env:APPDATA "npm\claude.cmd"; if (Test-Path $c) { $env:CLAUDE_PATH = $c } }
  elseif ($fa -eq "codex") { $c = Join-Path $env:APPDATA "npm\codex.cmd"; if (Test-Path $c) { $env:CODEX_PATH = $c } }
}

# Per-LAUNCH throwaway cost-DB (never the real brain), relay-based inbox auth, 10-min timeout.
# Unique per launch (slot+PID): a corrupt leftover husk from a crashed run (seen live 7/03:
# a 4KB unopenable fleet_apply_throwaway_0.db from 6/28 flash-killed every home-0 spawn at
# startup, before the worker log existed) must never block the next launch. Best-effort
# cleanup of old husks below keeps TEMP bounded.
Get-ChildItem (Join-Path $env:TEMP "fleet_apply_throwaway_*.db*") -ErrorAction SilentlyContinue |
  Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-1) } |
  ForEach-Object { try { Remove-Item $_.FullName -Force -ErrorAction Stop } catch {} }
$env:APPLYPILOT_DB_PATH = Join-Path $env:TEMP "fleet_apply_throwaway_${Slot}_$PID.db"
$env:APPLYPILOT_INBOX_AUTH = "1"
$env:APPLYPILOT_INBOX_AUTH_MODE = "relay"
$env:APPLYPILOT_ENABLE_GMAIL_MCP = "0"
$env:APPLYPILOT_AGENT_TIMEOUT = "600"
# Cheap read-only liveness probe before each agent launch: ~15% of queued postings are dead
# (expired/closed) and would otherwise burn a full agent launch. linkedin.com is guarded in
# liveness.py (never probed). Container/supervisor lanes already run with this ON.
$env:APPLYPILOT_PREFLIGHT_LIVENESS = "1"
# Phase 2B rollout: inventory deterministic Greenhouse plans in shadow, but
# hard-disable ownership of the irreversible submit until shadow acceptance.
$env:APPLYPILOT_GREENHOUSE_ADAPTER = "1"
$programDataRoot = if ($env:ProgramData) { $env:ProgramData } else { "C:\ProgramData" }
$greenhouseSubmitFlag = Join-Path $programDataRoot "ApplyPilot\greenhouse-submit.enabled"
if (Test-Path -LiteralPath $greenhouseSubmitFlag) {
  $env:APPLYPILOT_GREENHOUSE_ADAPTER_SUBMIT = "1"
} else {
  $env:APPLYPILOT_GREENHOUSE_ADAPTER_SUBMIT = "0"
}
# Ashby shadow discovery is deterministic and uses no model. Irreversible
# ownership remains independently gated by a machine-local sentinel.
$env:APPLYPILOT_ASHBY_ADAPTER = "1"
$ashbySubmitFlag = Join-Path $programDataRoot "ApplyPilot\ashby-submit.enabled"
if (Test-Path -LiteralPath $ashbySubmitFlag) {
  $env:APPLYPILOT_ASHBY_ADAPTER_SUBMIT = "1"
} else {
  $env:APPLYPILOT_ASHBY_ADAPTER_SUBMIT = "0"
}
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

$margs = @(); if ($Model) { $margs = @("--model", $Model) }
$fargs = @(); if ($FallbackAgent) { $fargs = @("--fallback-agent", $FallbackAgent) }
$faultProbe = & $py -c "from applypilot.apply.lifecycle_fault import enforce_no_lifecycle_faults; enforce_no_lifecycle_faults(); print('CLEAR')" 2>$null
if ($LASTEXITCODE -ne 0 -or "$faultProbe" -ne "CLEAR") {
  throw "Refusing to start worker '$WorkerId': unresolved lifecycle hard-fault record(s); operator reconciliation is required."
}
Write-Host "[fleet-worker] worker $WorkerId  owner=$Label  home_ip=$HomeIp  agent=$Agent  model=$(if($Model){$Model}else{'default'})  fallback=$(if($FallbackAgent){$FallbackAgent}else{'none'})  -> logs .applypilot\logs\worker-$Slot.log"
& $exe --dsn $env:FLEET_PG_DSN --worker-id "$WorkerId" --home-ip "$HomeIp" --machine-owner "$Label" --chrome-slot $Slot --agent $Agent @margs @fargs
