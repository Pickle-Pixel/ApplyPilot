# run-fleet-discovery.ps1 [-Label m2] [-Workers N] [-ResultsPerSite 50] [-HoursOld 72]
#   Run discovery worker(s): lease search_tasks from Postgres, scrape JobSpy, and stage postings
#   to discovered_postings. PURE SCRAPE -- no login, no apply agent (no Codex/Claude), no brain
#   write. Best run on a residential machine (machine 2), NOT the home box, so scraping stays off
#   the IP that holds the LinkedIn apply session.
#
#   -Workers N runs N scrapers in TRUE parallel, each in its OWN window with a DISTINCT worker-id
#   ($Label-disc-0 .. $Label-disc-(N-1)), so they lease different tasks instead of fighting over one
#   lease. -Workers 1 (default) runs a single worker in THIS window (Ctrl-C to stop), id $Label-disc.
#   Re-running with -Workers >1 first STOPS any discovery workers already up (clean slate -> no two
#   processes on the same id). Scraping is light (no browser), so N can go higher than the apply
#   launcher; the practical cap is your egress bandwidth + how fast the home box can `pull`.
#
#   FULL LOOP:
#     1. SEED + INGEST (home box):       .\run-discovery-home-loop.ps1 -Seed
#     2. SCRAPE (this script, machine 2): .\run-fleet-discovery.ps1 -Workers 4
#   Without step 1 there are no tasks to lease and the workers just idle.
#
#   Examples:
#     .\run-fleet-discovery.ps1                # 1 worker, foreground (unchanged behavior)
#     .\run-fleet-discovery.ps1 -Workers 4     # 4 parallel scrapers, one window each
param(
  [string]$Label = "m2",
  [int]$Workers = 1,
  [int]$ResultsPerSite = 50,
  [int]$HoursOld = 72,
  [int]$Index = -1   # internal: set when the launcher self-spawns a child worker
)
$ErrorActionPreference = "Stop"
if ($Workers -lt 1 -or $Workers -gt 16) { throw "-Workers must be 1..16 (each is a scraper leasing its own tasks)." }
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# .conda-env on the home box, .venv on a bootstrapped machine -- use whichever has the exe.
$exe = $null
foreach ($d in @(".\.conda-env\Scripts", ".\.venv\Scripts")) {
  $cand = Join-Path $d "applypilot-fleet-discovery.exe"
  if (Test-Path $cand) { $exe = (Resolve-Path $cand).Path; break }
}
if (-not $exe) { throw "applypilot-fleet-discovery not found in .conda-env or .venv -- run the setup script first." }

# DSN: inherit a persisted FLEET_PG_DSN (machine 2 sets it at User scope -> the home box), else
# default to the home box's local Postgres. Set in THIS process so child windows inherit it.
if (-not $env:FLEET_PG_DSN) { throw "FLEET_PG_DSN is not set. Supply an explicit mapped per-node fleet DSN (never the postgres admin or the shared fleet_worker role); refusing to start before any database action." }
$env:APPLYPILOT_FLEET_DSN = $env:FLEET_PG_DSN
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

$py = $null
foreach ($d in @(".\.conda-env\python.exe", ".\.venv\Scripts\python.exe")) {
  if (Test-Path $d) { $py = (Resolve-Path $d).Path; break }
}
if (-not $py) { $py = "python" }

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

# Run exactly one worker (helper used by both the single-worker path and each spawned child).
function Start-OneWorker([string]$wid) {
  $blocked = Test-MachineBlackout "discovery"
  if ($blocked) { throw "Refusing to start discovery workers for '$Label': machine blackout status did not return exact OK. $blocked" }
  Write-Host "[fleet-discovery] worker $wid  results/site=$ResultsPerSite  hours-old=$HoursOld  (pure scrape, no agent)"
  & $exe --worker-id "$wid" --results-per-site $ResultsPerSite --hours-old $HoursOld
}

# --- child invocation: run ONE worker with a distinct id in the foreground ---
if ($Index -ge 0) { Start-OneWorker "$Label-disc-$Index"; return }

# --- single worker (default): foreground, unsuffixed id -- unchanged behavior ---
if ($Workers -le 1) { Start-OneWorker "$Label-disc"; return }

# --- multi-worker: clean slate, then one window per worker on a DISTINCT id ---
$blocked = Test-MachineBlackout "discovery"
if ($blocked) { throw "Refusing to start discovery workers for '$Label': machine blackout status did not return exact OK. $blocked" }
$existing = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
  $_.Name -eq 'applypilot-fleet-discovery.exe' -or
  ($_.Name -eq 'python.exe' -and $_.CommandLine -match 'fleet-discovery' -and $_.CommandLine -notmatch 'discovery-home')
}
if ($existing) {
  Write-Host ("Stopping {0} existing discovery worker process(es) for a clean slate..." -f @($existing).Count) -ForegroundColor Yellow
  $existing | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 2
}

$self = $MyInvocation.MyCommand.Path
$logDir = Join-Path $ProjectRoot ".fleet-logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
for ($i = 0; $i -lt $Workers; $i++) {
  $outLog = Join-Path $logDir ("discovery-{0}-disc-{1}.out.log" -f $Label, $i)
  $errLog = Join-Path $logDir ("discovery-{0}-disc-{1}.err.log" -f $Label, $i)
  $argList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$self`"",
               "-Index", $i, "-Label", $Label,
               "-ResultsPerSite", $ResultsPerSite, "-HoursOld", $HoursOld)
  Start-Process -FilePath "powershell.exe" -ArgumentList $argList -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog
  Write-Host ("  launched {0}-disc-{1}" -f $Label, $i) -ForegroundColor Green
  Start-Sleep -Milliseconds 800
}
Write-Host ("`n{0} discovery scrapers up (ids {1}-disc-0..{2}), each its own window. They idle until the home box seeds tasks." -f $Workers, $Label, ($Workers - 1)) -ForegroundColor Cyan
