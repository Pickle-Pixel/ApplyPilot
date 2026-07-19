# run-fleet-compute.ps1 [-Label m4] [-Workers 16] [-Providers deepseek]
#   Run compute/scoring worker(s): lease jobs from compute_queue, run the LLM score/audit
#   pass, write ADVISORY results back. IP-FREE (no browser, no site traffic) -- so it is safe
#   on any owner machine and does NOT need the apply IP hygiene. Cost is gated by the fleet's
#   OWN compute caps (fleet_config.cost_cap_daily_usd / cost_cap_total_usd vs llm_usage) --
#   SEPARATE from the apply spend_cap_usd, and NOT affected by the apply `paused` switch.
#
#   -Workers N runs N scorers in TRUE parallel, each in its OWN window with a DISTINCT
#   worker-id ($Label-score-0 .. $Label-score-(N-1)). Scoring is API-bound (not CPU-bound), so
#   N is about DeepSeek concurrency/rate-limits, not cores. Re-running with -Workers >1 first
#   STOPS any compute workers already up (clean slate -> no two processes on one id).
#
#   The DeepSeek key comes from ~/.applypilot/.env (same file the home tool uses). This script
#   loads it and refuses to start if no usable LLM tier is present -- copy the home box's
#   .applypilot\.env to this machine if the pre-flight fails.
#
#   FULL LOOP:
#     1. HOME seeds + pulls:   .\run-compute-home-loop.ps1        (fills compute_queue, pulls results)
#     2. THIS script (m4):     .\run-fleet-compute.ps1 -Workers 16 (scores what's queued)
param(
  [string]$Label = "m4",
  [int]$Workers = 16,
  [string]$Providers = "deepseek",
  [int]$Index = -1   # internal: set when the launcher self-spawns a child worker
)
$ErrorActionPreference = "Stop"
$WorkerCap = 16
if ($Workers -lt 1 -or $Workers -gt $WorkerCap) { throw "-Workers must be 1..$WorkerCap (each is a scorer leasing its own jobs)." }
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# .conda-env on the home box, .venv on a bootstrapped machine -- use whichever has the exe.
$exe = $null
foreach ($d in @(".\.conda-env\Scripts", ".\.venv\Scripts")) {
  $cand = Join-Path $d "applypilot-fleet-compute.exe"
  if (Test-Path $cand) { $exe = (Resolve-Path $cand).Path; break }
}
if (-not $exe) { throw "applypilot-fleet-compute not found in .conda-env or .venv -- run the setup script first." }

# DSN: inherit a persisted FLEET_PG_DSN (setup sets it at User scope -> home box over LAN/Tailscale),
# else default to the home box's local Postgres. Set in THIS process so child windows inherit it.
if (-not $env:FLEET_PG_DSN) { throw "FLEET_PG_DSN is not set. Supply an explicit mapped per-node fleet DSN (never the postgres admin or the shared fleet_worker role); refusing to start before any database action." }
$env:APPLYPILOT_FLEET_DSN = $env:FLEET_PG_DSN
$env:LLM_SCORE_PROVIDER = $Providers
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

# Load ~/.applypilot/.env (KEY=VALUE lines) so DEEPSEEK_API_KEY et al. reach the worker, exactly
# like the home tool. Existing process env wins (never clobber an explicitly-set key).
$envFile = Join-Path $HOME ".applypilot\.env"
if (Test-Path $envFile) {
  Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and $line -notlike '#*' -and $line -match '^\s*([^=\s]+)\s*=\s*(.*)$') {
      $k = $matches[1]; $v = $matches[2].Trim('"').Trim("'")
      if (-not [Environment]::GetEnvironmentVariable($k)) { Set-Item -Path "Env:$k" -Value $v }
    }
  }
}

# NO local key? Fetch it from the fleet Postgres (fleet_assets, stored once from home) -- the
# same box already holds the DSN, so a worker machine needs NO manual .env copy. This is what
# lets m4 come up key-less. (Children spawned below inherit this env, so only the parent fetches.)
if (-not $env:DEEPSEEK_API_KEY) {
  $pyKey = @(".\.conda-env\python.exe", ".\.venv\Scripts\python.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
  if ($pyKey) {
    $k = (& $pyKey (Join-Path $ProjectRoot "fleet-secret.py") get deepseek_api_key 2>$null | Out-String).Trim()
    if ($k) { $env:DEEPSEEK_API_KEY = $k; Write-Host "[fleet-compute] DeepSeek key loaded from fleet Postgres (no .env needed)." -ForegroundColor Gray }
  }
}

$pyGuard = @(".\.conda-env\python.exe", ".\.venv\Scripts\python.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $pyGuard) { $pyGuard = "python" }

function Test-MachineBlackout([string]$Role) {
  $stderrPath = [IO.Path]::GetTempFileName()
  try {
    $lines = @(& $pyGuard (Join-Path $ProjectRoot "fleet-blackout-query.py") $Label $Role 2> $stderrPath)
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

function Start-OneWorker([string]$wid) {
  $blocked = Test-MachineBlackout "compute"
  if ($blocked) { throw "Refusing to start compute workers for '$Label': machine blackout status did not return exact OK. $blocked" }
  Write-Host "[fleet-compute] worker $wid  providers=$Providers  (IP-free score/audit, cost-cap gated)"
  & $exe --worker-id "$wid" --home-ip "0.0.0.0" --machine-owner $Label
}

# --- child invocation: run ONE worker with a distinct id in the foreground ---
if ($Index -ge 0) { Start-OneWorker "$Label-score-$Index"; return }

Write-Host ("[fleet-compute] target_workers={0} cap={1} providers={2} label={3}" -f $Workers, $WorkerCap, $Providers, $Label) -ForegroundColor Cyan

# --- pre-flight (parent only): is a usable LLM tier present? ---
$pyExe = @(".\.conda-env\python.exe", ".\.venv\Scripts\python.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($pyExe) {
  $tier = (& $pyExe -c "from applypilot import config; print(config.get_tier())" 2>$null | Select-Object -Last 1)
  if ("$tier" -notmatch '^\d+$' -or [int]$tier -lt 2) {
    Write-Host "[fleet-compute] PRE-FLIGHT FAIL: LLM tier='$tier' (<2). No usable scoring key." -ForegroundColor Red
    Write-Host "  Copy the home box's .applypilot\.env (with DEEPSEEK_API_KEY=...) to $envFile and retry." -ForegroundColor Yellow
    throw "no usable LLM tier -- refusing to start scorers that would immediately die."
  }
  Write-Host "[fleet-compute] pre-flight OK: LLM tier=$tier" -ForegroundColor Green
}

# --- single worker (default path when -Workers 1): foreground, unsuffixed-index id ---
if ($Workers -le 1) { Start-OneWorker "$Label-score-0"; return }

# --- multi-worker: clean slate, then one window per worker on a DISTINCT id ---
$blocked = Test-MachineBlackout "compute"
if ($blocked) { throw "Refusing to start compute workers for '$Label': machine blackout status did not return exact OK. $blocked" }
$self = $MyInvocation.MyCommand.Path
$existing = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
  $_.Name -eq 'applypilot-fleet-compute.exe' -or
  ($_.Name -eq 'python.exe' -and $_.CommandLine -match 'fleet-compute') -or
  ($_.Name -eq 'powershell.exe' -and $_.CommandLine -match [regex]::Escape($self) -and $_.CommandLine -match '\s-Index\s+\d+')
}
if ($existing) {
  Write-Host ("Stopping {0} existing compute worker process(es) for a clean slate..." -f @($existing).Count) -ForegroundColor Yellow
  $existing | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 2
}

$logDir = Join-Path $ProjectRoot ".fleet-logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
for ($i = 0; $i -lt $Workers; $i++) {
  $outLog = Join-Path $logDir ("compute-{0}-score-{1}.out.log" -f $Label, $i)
  $errLog = Join-Path $logDir ("compute-{0}-score-{1}.err.log" -f $Label, $i)
  $argList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$self`"",
               "-Index", $i, "-Label", $Label, "-Providers", $Providers)
  Start-Process -FilePath "powershell.exe" -ArgumentList $argList -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog
  Write-Host ("  launched {0}-score-{1}" -f $Label, $i) -ForegroundColor Green
  Start-Sleep -Milliseconds 800
}
Write-Host ("`n{0} compute scorers up (ids {1}-score-0..{2}), each its own window. They idle until the home box seeds compute_queue." -f $Workers, $Label, ($Workers - 1)) -ForegroundColor Cyan
