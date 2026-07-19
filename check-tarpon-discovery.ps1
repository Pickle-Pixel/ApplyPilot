[CmdletBinding()]
param(
  [string]$Dsn = "",
  [int]$AliveSeconds = 150,
  [int]$RecentWarningMinutes = 60,
  [switch]$RestartTask
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repo = $PSScriptRoot
if (-not $repo) { $repo = "C:\ApplyPilot" }
Set-Location $repo

function Write-Section([string]$Title) {
  Write-Host ""
  Write-Host "=== $Title ===" -ForegroundColor Cyan
}

function Get-RepoPython {
  foreach ($candidate in @(
    (Join-Path $repo ".venv\Scripts\python.exe"),
    (Join-Path $repo ".conda-env\Scripts\python.exe"),
    (Join-Path $repo "venv\Scripts\python.exe"),
    (Join-Path $repo ".env\Scripts\python.exe")
  )) {
    if (Test-Path -LiteralPath $candidate) { return $candidate }
  }
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  return $null
}

if (-not $Dsn) {
  $Dsn = [Environment]::GetEnvironmentVariable("FLEET_PG_DSN", "Process")
  if (-not $Dsn) { $Dsn = [Environment]::GetEnvironmentVariable("FLEET_PG_DSN", "User") }
  if (-not $Dsn) { $Dsn = [Environment]::GetEnvironmentVariable("FLEET_PG_DSN", "Machine") }
  if (-not $Dsn) { throw "No fleet DSN: pass -Dsn or set FLEET_PG_DSN to an explicit mapped fleet DSN (never the postgres admin or the shared fleet_worker role). Refusing to read discovery status without it." }
}

$python = Get-RepoPython
if (-not $python) { throw "Python not found under $repo or PATH." }

$taskName = "ApplyPilotFleet-DiscoveryScrape"
$logPath = Join-Path $repo ".fleet-logs\discovery-scrape.log"
$issues = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]

Write-Section "Inputs"
Write-Host "Repo    : $repo"
Write-Host "Python  : $python"
Write-Host "DSN     : $Dsn"
Write-Host "Machine : $env:COMPUTERNAME"

Write-Section "Tarpon Reachability"
if (Get-Command tailscale -ErrorAction SilentlyContinue) {
  $ts = (& tailscale status 2>&1 | Out-String)
  $tarpon = ($ts -split "`r?`n") | Where-Object { $_ -match "\btarpon\b|100\.77\.65\.8" }
  if ($tarpon) { $tarpon | ForEach-Object { Write-Host $_ } }
  else { Write-Host "Tarpon not found in tailscale status." -ForegroundColor Yellow }
} else {
  Write-Host "tailscale CLI not found on this machine." -ForegroundColor Yellow
}

Write-Section "Local Scheduled Task"
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
$taskInfo = $null
if ($task) {
  if ($RestartTask) {
    Write-Host "Restart requested: starting $taskName"
    Start-ScheduledTask -TaskName $taskName
    Start-Sleep -Seconds 10
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
  }
  $taskInfo = Get-ScheduledTaskInfo -TaskName $taskName -ErrorAction SilentlyContinue
  $task | Select-Object TaskName, State | Format-Table -AutoSize
  if ($taskInfo) {
    $taskInfo | Select-Object LastRunTime, NextRunTime, LastTaskResult, NumberOfMissedRuns | Format-Table -AutoSize
  }
} else {
  Write-Host "Task is not registered on this machine: $taskName"
  Write-Host "That is OK when running this script on home instead of Tarpon."
}

Write-Section "Local Discovery Process"
$procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
  $_.CommandLine -and $_.CommandLine -match "applypilot-fleet-discovery|run-fleet-discovery|m2-disc|DiscoveryScrape"
}
if ($procs) {
  $procs | Select-Object ProcessId, Name, CommandLine | Format-Table -Wrap
} else {
  Write-Host "No local discovery process found. This is OK when running from home."
}

Write-Section "Local Discovery Log"
if (Test-Path -LiteralPath $logPath) {
  $log = Get-Item -LiteralPath $logPath
  $ageMin = [math]::Round(((Get-Date) - $log.LastWriteTime).TotalMinutes, 1)
  Write-Host "Path      : $logPath"
  Write-Host "LastWrite : $($log.LastWriteTime)"
  Write-Host "Age       : ${ageMin}m"
  Get-Content -LiteralPath $logPath -Tail 30
} else {
  Write-Host "No local discovery log found: $logPath"
}

Write-Section "Database"
$env:FLEET_PG_DSN = $Dsn
$code = @'
import datetime as dt
import decimal
import json
import os
import sys

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception as exc:
    print(json.dumps({"error": "import", "detail": f"{type(exc).__name__}: {exc}"}))
    sys.exit(2)

dsn = os.environ.get("FLEET_PG_DSN")

def clean(value):
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    return value

def rows(cur):
    return [{k: clean(v) for k, v in dict(r).items()} for r in cur.fetchall()]

try:
    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as conn:
        cur = conn.cursor()
        cur.execute("""
          SELECT worker_id, machine_owner, home_ip, role, state, current_job,
                 sw_version, last_beat,
                 round(EXTRACT(EPOCH FROM (now() - last_beat)))::int AS age_s,
                 left(coalesce(last_error,''), 500) AS last_error
          FROM worker_heartbeat
          WHERE worker_id='m2-disc' OR worker_id LIKE 'm2-disc-%' OR role='discovery'
          ORDER BY worker_id
        """)
        discovery_workers = rows(cur)

        cur.execute("""
          SELECT status, count(*)::int AS n,
                 count(*) FILTER (WHERE enabled AND next_due_at <= now())::int AS due_now,
                 max(last_run_at) AS max_last_run_at
          FROM search_tasks
          GROUP BY status
          ORDER BY status
        """)
        search_status = rows(cur)

        cur.execute("""
          SELECT count(*)::int AS total,
                 count(*) FILTER (WHERE last_run_at IS NOT NULL)::int AS with_last_run,
                 max(last_run_at) AS max_last_run_at,
                 min(next_due_at) FILTER (WHERE enabled AND status='queued') AS next_due_at
          FROM search_tasks
        """)
        search_meta = {k: clean(v) for k, v in dict(cur.fetchone()).items()}

        cur.execute("""
          SELECT count(*)::int AS leaseable_now
          FROM search_tasks s
          LEFT JOIN rate_governor g ON g.scope_key = 'board:' || s.board
          WHERE s.status='queued' AND s.enabled AND s.next_due_at <= now()
            AND COALESCE(g.breaker_state, 'ok') != 'demoted'
            AND COALESCE(NOT (g.breaker_state = 'paused'
                AND COALESCE(g.breaker_until, 'infinity'::timestamptz) >= now()), TRUE)
            AND COALESCE(g.count_24h, 0) < COALESCE(g.daily_cap, 2000000000)
            AND (COALESCE(g.last_applied_at, g.last_attempt_at) IS NULL
                 OR COALESCE(g.last_applied_at, g.last_attempt_at) < now()
                      - make_interval(secs => COALESCE(g.min_gap_seconds, 90)))
        """)
        leaseable = {k: clean(v) for k, v in dict(cur.fetchone()).items()}

        cur.execute("""
          SELECT task_id, query, board, location, lease_owner, lease_expires_at,
                 last_run_at, next_due_at, left(coalesce(last_error,''), 300) AS last_error
          FROM search_tasks
          WHERE status='leased'
          ORDER BY lease_expires_at
          LIMIT 10
        """)
        leased_tasks = rows(cur)

        cur.execute("""
          SELECT count(*)::int AS total,
                 count(*) FILTER (WHERE discovered_at >= now() - interval '30 minutes')::int AS recent_30m,
                 count(*) FILTER (WHERE discovered_at >= now() - interval '24 hours')::int AS recent_24h,
                 max(discovered_at) AS newest
          FROM discovered_postings
        """)
        postings = {k: clean(v) for k, v in dict(cur.fetchone()).items()}

        cur.execute("""
          SELECT task_id, query, board, location, last_run_at, next_due_at, result_count,
                 left(coalesce(last_error,''), 300) AS last_error
          FROM search_tasks
          WHERE last_error IS NOT NULL AND last_error <> ''
          ORDER BY updated_at DESC
          LIMIT 8
        """)
        task_errors = rows(cur)

    print(json.dumps({
        "discovery_workers": discovery_workers,
        "search_status": search_status,
        "search_meta": search_meta,
        "leaseable": leaseable,
        "leased_tasks": leased_tasks,
        "postings": postings,
        "task_errors": task_errors,
    }))
except Exception as exc:
    print(json.dumps({"error": "db", "detail": f"{type(exc).__name__}: {exc}"}))
    sys.exit(3)
'@

$raw = $code | & $python -
if ($LASTEXITCODE -ne 0) { throw "DB probe failed: $raw" }
$data = $raw | ConvertFrom-Json
if ($data.error) { throw "DB probe failed: $($data.error) $($data.detail)" }

Write-Host "Discovery workers:"
if ($data.discovery_workers) {
  $data.discovery_workers |
    Select-Object worker_id, role, state, age_s, current_job, last_beat |
    Format-Table -AutoSize
}
else { Write-Host "(none)" }

Write-Host ""
Write-Host "Search task status:"
$data.search_status | Format-Table -AutoSize

Write-Host ""
Write-Host "Search summary:"
$data.search_meta | Format-List
Write-Host "leaseable_now: $($data.leaseable.leaseable_now)"

Write-Host ""
Write-Host "Discovered postings:"
$data.postings | Format-List

if ($data.leased_tasks) {
  Write-Host ""
  Write-Host "Leased search tasks:"
  $data.leased_tasks | Format-Table -AutoSize
}

if ($data.task_errors) {
  Write-Host ""
  Write-Host "Recent search task errors:"
  $data.task_errors | Format-Table -AutoSize
}

$mainDisc = $null
if ($data.discovery_workers) {
  $mainDisc = $data.discovery_workers | Where-Object { $_.worker_id -eq "m2-disc" } | Select-Object -First 1
  if (-not $mainDisc) { $mainDisc = $data.discovery_workers | Select-Object -First 1 }
}

if (-not $mainDisc) {
  $issues.Add("No discovery heartbeat row found for m2-disc or role=discovery.")
} elseif ([int]$mainDisc.age_s -gt $AliveSeconds) {
  $issues.Add("Discovery heartbeat is stale: $($mainDisc.age_s)s > ${AliveSeconds}s.")
}

if ([int]$data.search_meta.total -le 0) {
  $issues.Add("No search_tasks exist. Seed search tasks from the home loop.")
} elseif ([int]$data.search_meta.with_last_run -le 0) {
  $issues.Add("No search_tasks have last_run_at. Discovery has not completed a search.")
}

if ($data.leased_tasks) {
  foreach ($taskRow in $data.leased_tasks) {
    if ($taskRow.lease_expires_at) {
      $leaseExpiry = [datetime]::Parse($taskRow.lease_expires_at)
      if ($leaseExpiry -lt (Get-Date).ToUniversalTime().AddMinutes(-2)) {
        $warnings.Add("Expired search lease: $($taskRow.task_id) owner=$($taskRow.lease_owner)")
      }
    }
  }
}

if ([int]$data.postings.recent_30m -eq 0) {
  $newest = $data.postings.newest
  if ($newest) {
    $warnings.Add("No discovered_postings in the last 30 minutes. Newest posting: $newest.")
  } else {
    $warnings.Add("No discovered_postings exist yet.")
  }
}

if ($task -and $taskInfo -and ($taskInfo.LastTaskResult -notin @(0, 267009, 267014))) {
  $warnings.Add("Scheduled task LastTaskResult is unusual: $($taskInfo.LastTaskResult).")
}

Write-Section "Verdict"
if ($issues.Count -eq 0) {
  Write-Host "HEALTHY: discovery worker heartbeat and search task completion are present." -ForegroundColor Green
} else {
  Write-Host "DEGRADED" -ForegroundColor Yellow
  $issues | ForEach-Object { Write-Host " - $_" -ForegroundColor Yellow }
}

if ($warnings.Count -gt 0) {
  Write-Host ""
  Write-Host "Warnings:" -ForegroundColor Yellow
  $warnings | ForEach-Object { Write-Host " - $_" -ForegroundColor Yellow }
}
