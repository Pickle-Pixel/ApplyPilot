# fleet-daily-unguard.ps1 -RunOnce -Dsn <override> [-RegisterTask]
#
# Removes guardrails for apply/compute/discovery/linkedin and clears the operator lock
# so the whole fleet can run through the day. This is an explicit operator action;
# scheduled invocations are intentionally no-ops unless -Force is supplied.
#
# Use cases:
#   - run immediately: .\fleet-daily-unguard.ps1
#   - schedule once daily at 03:00AM: .\fleet-daily-unguard.ps1 -RegisterTask
param(
  [string]$Dsn = "",
  [string]$Reason = "3am daily guardrail release",
  [switch]$RegisterTask,
  [switch]$Force,
  [string]$TaskName = "ApplyPilotFleet-DailyUnpause",
  [int]$HeartbeatMinutes = 24
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
if (-not $repo) { $repo = Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location $repo

$logs = Join-Path $repo ".fleet-logs"
$wrapperDir = Join-Path $logs "_task-wrappers"
if (-not (Test-Path $logs)) { New-Item -ItemType Directory -Path $logs | Out-Null }
if (-not (Test-Path $wrapperDir)) { New-Item -ItemType Directory -Path $wrapperDir | Out-Null }

$python = @((Join-Path $repo ".conda-env\python.exe"), (Join-Path $repo ".venv\Scripts\python.exe")) |
  Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $python) { throw "python not found (.conda-env or .venv)." }

$effectiveDsn = if ($Dsn) { $Dsn } else { $env:FLEET_PG_DSN }
if (-not $effectiveDsn) {
  throw "Need FLEET_PG_DSN in environment or -Dsn argument."
}
$env:FLEET_PG_DSN = $effectiveDsn

function Invoke-PgUnguard([int]$HeartbeatMinutes, [string]$ReasonText) {
  $code = @"
import os
from psycopg import connect
from psycopg.rows import dict_row

heart_minutes = int(os.environ["UNGUARD_HEARTBEAT_MINUTES"])
reason = os.environ["UNGUARD_REASON"]
dsn = os.environ["FLEET_PG_DSN"]

with connect(dsn, row_factory=dict_row) as conn:
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = '15s'")
        cur.execute("SET lock_timeout = '3s'")

        # Capture pre-change state for operator visibility
        cur.execute("SELECT id, active, source, reason, updated_at FROM applypilot_operator_lock WHERE id = 1")
        pre_lock = cur.fetchone()

        cur.execute(
            "UPDATE applypilot_operator_lock SET active = FALSE, source = '', reason = %s, updated_at = now() WHERE id = 1",
            (reason,),
        )
        lock_updated = cur.rowcount

        # Remove all lane-level guardrails + canaries + apply spend cap for the day.
        cur.execute(
            "UPDATE fleet_config\n"
            "SET paused = FALSE, ats_paused = FALSE, ats_pause_source = NULL,\n"
            "    canary_enabled = FALSE, canary_remaining = NULL,\n"
            "    linkedin_canary_enabled = FALSE, linkedin_canary_remaining = NULL,\n"
            "    spend_cap_usd = 0, updated_at = now()\n"
            "WHERE id = 1"
        )
        cfg_updated = cur.rowcount

        # Clear stale open commands so old restart/pause signals cannot suppress run.
        cur.execute("UPDATE remote_commands SET acked_at = now() WHERE acked_at IS NULL")
        commands_cleared = cur.rowcount

        # Resume all active front-facing workers (apply/compute/discovery/linkedin).
        cur.execute(
            "INSERT INTO remote_commands (worker_id, command)\n"
            "SELECT DISTINCT worker_id, 'resume'\n"
            "FROM worker_heartbeat\n"
            "WHERE role IN ('apply', 'compute', 'discovery', 'linkedin')\n"
            "  AND worker_id IS NOT NULL\n"
            "  AND last_beat > now() - (%s * interval '1 minute')",
            (heart_minutes,),
        )
        resume_inserts = cur.rowcount

        # Verification snapshots
        cur.execute("SELECT id, paused, ats_paused, ats_pause_source, canary_enabled, canary_remaining, linkedin_canary_enabled, linkedin_canary_remaining, spend_cap_usd FROM fleet_config WHERE id = 1")
        post_cfg = cur.fetchone()

        cur.execute("SELECT id, active, source, reason, updated_at FROM applypilot_operator_lock WHERE id = 1")
        post_lock = cur.fetchone()

        cur.execute(
            "SELECT command, count(*) AS open FROM remote_commands WHERE acked_at IS NULL GROUP BY command ORDER BY command"
        )
        open_commands = cur.fetchall()

        cur.execute(
            "SELECT count(*) AS active_front_workers\n"
            "FROM (\n"
            "    SELECT DISTINCT worker_id\n"
            "    FROM worker_heartbeat\n"
            "    WHERE role IN ('apply','compute','discovery','linkedin')\n"
            "      AND worker_id IS NOT NULL\n"
            "      AND last_beat > now() - (%s * interval '1 minute')\n"
            ") AS alive",
            (heart_minutes,),
        )
        active_front_workers = cur.fetchone()["active_front_workers"]

        print("pre_lock=" + str(pre_lock))
        print("post_lock=" + str(post_lock))
        print("post_cfg=" + str(post_cfg))
        print("lock_updated=" + str(lock_updated))
        print("cfg_updated=" + str(cfg_updated))
        print("commands_cleared=" + str(commands_cleared))
        print("resume_inserts=" + str(resume_inserts))
        print("open_remote_commands=" + str(open_commands))
        print("active_front_workers=" + str(active_front_workers))

        conn.commit()
"@

  $env:UNGUARD_REASON = $ReasonText
  $env:UNGUARD_HEARTBEAT_MINUTES = [string]$HeartbeatMinutes
  & $python -c $code
}

if ($RegisterTask) {
  # Register a daily local task that runs this script at 03:00 and writes logs.
  $scriptPath = $MyInvocation.MyCommand.Path
  $wrapperPath = Join-Path $wrapperDir "applypilot-daily-unguard-wrapper.ps1"
  $wrapperLog = Join-Path $logs "fleet-daily-unguard.log"
  $qReason = $Reason.Replace("'", "''")
@"
# FLEET_PG_DSN is inherited from the persisted User environment; the secret DSN is never
# embedded in this generated wrapper. The daily task reads it from the runtime channel.
Set-Location '$repo'
& '$scriptPath' -Dsn `$env:FLEET_PG_DSN -Reason '$qReason' -HeartbeatMinutes $HeartbeatMinutes *>> '$wrapperLog'
"@ | Set-Content -Encoding UTF8 -Path $wrapperPath

  $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$wrapperPath`""
  $trigger = New-ScheduledTaskTrigger -Daily -At 3:00am
  $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1) -MultipleInstances IgnoreNew
  if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue | Out-Null
  }
  Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Description "Daily 03:00 fleet guardrail release for apply/compute/discovery/LinkedIn fronts." -Force | Out-Null
  Write-Host "[fleet-daily-unguard] registered task '$TaskName' for daily 03:00 (local log: $wrapperLog)." -ForegroundColor Green
  return
}

if (-not $Force) {
  throw "Refusing to remove fleet guardrails: explicit -Force is required."
}

Invoke-PgUnguard -HeartbeatMinutes $HeartbeatMinutes -ReasonText $Reason
