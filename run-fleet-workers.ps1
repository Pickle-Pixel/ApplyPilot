# run-fleet-workers.ps1 -Count N [-Agent claude|codex] [-Label home] [-Model name] [-StartSlot 0]
#   Spin up N apply workers in TRUE parallel: each gets a DISTINCT slot (0..N-1), so each runs
#   its OWN isolated Chrome profile + CDP port + log -- N jobs applied at once, no shared-browser
#   collision. (Multiple workers on the SAME slot is the "3-4 tabs fighting over one Chrome" bug;
#   this launcher avoids it by construction.)
#
#   It first STOPS any running apply workers (a clean slate so stale same-slot processes can't
#   keep colliding), then opens one window per worker.
#
#   Examples:
#     .\run-fleet-workers.ps1 -Count 3                      # 3 Claude workers on the home box
#     .\run-fleet-workers.ps1 -Count 4 -Agent codex -Label m2                 # slots 0..3
#     .\run-fleet-workers.ps1 -Count 4 -Agent claude -Label m4 -StartSlot 4    # slots 4..7
#
#   Cost note: each home-box (claude) worker is ~$0.65-0.90/apply, so N workers = N x spend RATE
#   (the canary + cost cap still bound the TOTAL). Codex workers ride your ChatGPT Pro plan.
#   LinkedIn NEVER runs here -- that lane is one-IP, one-worker, by hand.
param([int]$Count = 2, [string]$Agent = "claude", [string]$Label = "home", [string]$Model = "",
      [int]$StartSlot = 0)
$ErrorActionPreference = "Stop"
if ($Count -lt 1 -or $Count -gt 10) { throw "-Count must be 1..10 (each worker is a full Chrome + apply agent; keep it sane)." }
if ($StartSlot -lt 0 -or ($StartSlot + $Count - 1) -gt 9) { throw "-StartSlot plus -Count must stay within Chrome slots 0..9." }
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# DSN: inherit a persisted FLEET_PG_DSN (an explicit mapped per-node fleet DSN, set at User
# scope by the setup script). There is NO local-admin fallback: a missing DSN used to spawn
# workers that looped forever on Postgres connection timeouts, and a postgres/fleet_worker
# fallback would bypass the mapped-role topology. Fail closed instead.
if (-not $env:FLEET_PG_DSN) {
  throw "FLEET_PG_DSN is not set for worker label '$Label'. Set it to an explicit mapped per-node fleet DSN (never the postgres admin or the shared fleet_worker role) before launching workers."
}
$env:APPLYPILOT_FLEET_DSN = $env:FLEET_PG_DSN

# 1. Clean slate: stop any apply workers already running (they may be stacked on one slot).
$slotPattern = "--chrome-slot\s+(" + (($StartSlot..($StartSlot + $Count - 1)) -join "|") + ")(\s|$)"
$existing = Get-CimInstance Win32_Process -Filter "Name='applypilot-fleet-apply.exe' OR Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match 'fleet-apply|apply_worker_main' -and $_.CommandLine -match $slotPattern }
if ($existing) {
  Write-Host ("Stopping {0} existing apply worker process(es) in slots {1}..{2} for a clean slate..." -f @($existing).Count, $StartSlot, ($StartSlot + $Count - 1)) -ForegroundColor Yellow
  $existing | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 2
}

# 2. Launch N workers, each on a DISTINCT slot, each in its own window.
$worker = Join-Path $ProjectRoot "run-fleet-worker.ps1"
for ($i = $StartSlot; $i -lt ($StartSlot + $Count); $i++) {
  $argList = @("-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$worker`"", "-Slot", $i, "-Agent", $Agent, "-Label", $Label)
  if ($Model) { $argList += @("-Model", $Model) }
  Start-Process -FilePath "powershell.exe" -ArgumentList $argList -WorkingDirectory $ProjectRoot
  Write-Host ("  launched {0}-{1} (slot {1})" -f $Label, $i) -ForegroundColor Green
  Start-Sleep -Milliseconds 800
}
Write-Host ("`n{0} isolated workers up (slots {1}..{2}), each its own Chrome. Watch them at the console: http://<your-lan-ip>:8787" -f $Count, $StartSlot, ($StartSlot + $Count - 1)) -ForegroundColor Cyan
