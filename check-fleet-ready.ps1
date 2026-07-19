# check-fleet-ready.ps1 - strict go/no-go gate before letting fleet apply run.
[CmdletBinding()]
param(
  [string]$Dsn = "",
  [int]$MaxHeartbeatAgeSeconds = 300,
  [int]$VerifyLiveMaxAgeMinutes = 390,
  [switch]$RunVerifyLive,
  [switch]$AllowPaused,
  [int]$VerifyLimit = 1200,
  [switch]$Watch,
  [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
if (-not $repo) { $repo = Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location $repo

if (-not $Dsn) {
  if ($env:FLEET_PG_DSN) {
    $Dsn = $env:FLEET_PG_DSN
  } else {
    throw "No fleet DSN: pass -Dsn or set FLEET_PG_DSN to an explicit mapped fleet DSN (never the postgres admin or the shared fleet_worker role). Refusing the go/no-go gate without it."
  }
}

$py = @(
  (Join-Path $repo ".conda-env\python.exe"),
  (Join-Path $repo ".venv\Scripts\python.exe")
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $py) { throw "python not found (.conda-env or .venv)." }

$verifyLog = Join-Path $repo ".fleet-logs\verify-live.log"
$runApplyPilot = Join-Path $repo "run-applypilot.ps1"

function Invoke-ReadyCheck {
  if ($RunVerifyLive) {
    if (-not (Test-Path $runApplyPilot)) { throw "run-applypilot.ps1 not found at $runApplyPilot" }
    Write-Host "[check-fleet-ready] running VerifyLive first..." -ForegroundColor Cyan
    & $runApplyPilot verify-live --max-age-days 3 --limit $VerifyLimit
    if ($LASTEXITCODE -ne 0) {
      Write-Host "NOT READY: VerifyLive exited $LASTEXITCODE" -ForegroundColor Red
      return 2
    }
  }

  $env:FLEET_PG_DSN = $Dsn
  $env:CHECK_FLEET_VERIFY_LOG = $verifyLog
  $env:CHECK_FLEET_MAX_HEARTBEAT_AGE_SECONDS = [string]$MaxHeartbeatAgeSeconds
  $env:CHECK_FLEET_VERIFY_MAX_AGE_MINUTES = [string]$VerifyLiveMaxAgeMinutes
  $env:CHECK_FLEET_ALLOW_PAUSED = if ($AllowPaused) { "1" } else { "0" }

  $code = @'
from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from applypilot.fleet.agent_readiness import blocked_desired_agent_chains

dsn = os.environ["FLEET_PG_DSN"]
verify_log = Path(os.environ["CHECK_FLEET_VERIFY_LOG"])
max_age = int(os.environ["CHECK_FLEET_MAX_HEARTBEAT_AGE_SECONDS"])
verify_max_age_minutes = int(os.environ["CHECK_FLEET_VERIFY_MAX_AGE_MINUTES"])
allow_paused = os.environ.get("CHECK_FLEET_ALLOW_PAUSED") == "1"

blockers: list[str] = []
warnings: list[str] = []

def _age_s(row) -> int:
    value = row["age_s"]
    return 999999 if value is None else int(value)

def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None

def check_verify_live(now: datetime) -> None:
    if not verify_log.exists():
        blockers.append(f"VerifyLive log missing: {verify_log}")
        return
    raw = verify_log.read_bytes()
    text = "\n".join(
        decoded.replace("\x00", "")
        for decoded in (
            raw.decode("utf-8", errors="ignore"),
            raw.decode("utf-16-le", errors="ignore"),
        )
    )
    starts = list(re.finditer(r"\[([^\]]+)\] === VerifyLive start ===", text))
    exits = list(re.finditer(r"\[([^\]]+)\] verify-live exit=(\d+)", text))
    if not starts:
        blockers.append("VerifyLive has never started.")
        return
    if not exits:
        blockers.append("VerifyLive has no completed exit line.")
        return
    last_start = _parse_ts(starts[-1].group(1))
    last_exit = _parse_ts(exits[-1].group(1))
    last_code = int(exits[-1].group(2))
    if last_start and last_exit and last_start > last_exit:
        blockers.append(f"VerifyLive is still running or did not record an exit after {last_start.isoformat()}.")
        return
    if last_code != 0:
        blockers.append(f"VerifyLive last exit was {last_code}.")
        return
    if last_exit:
        age_min = (now - last_exit).total_seconds() / 60.0
        if age_min > verify_max_age_minutes:
            blockers.append(f"VerifyLive is stale: last successful exit {age_min:.0f} minutes ago.")

with psycopg.connect(dsn, row_factory=dict_row) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT now() AS now")
        db_now = cur.fetchone()["now"]
        now = db_now

        check_verify_live(now)

        cur.execute("SELECT paused, COALESCE(ats_paused, FALSE) AS ats_paused, spend_cap_usd FROM fleet_config WHERE id=1")
        cfg = cur.fetchone()
        if cfg:
            if cfg["paused"] and not allow_paused:
                blockers.append("fleet_config.paused is true.")
            elif cfg["paused"]:
                warnings.append("fleet_config.paused is true; ignored because -AllowPaused was passed.")
            if cfg["ats_paused"]:
                blockers.append("fleet_config.ats_paused is true.")

        cur.execute(
            "SELECT agent, blocked_until, reason FROM agent_availability "
            "WHERE blocked_until IS NOT NULL AND blocked_until > now() ORDER BY agent"
        )
        active_agent_blocks = {row["agent"]: row for row in cur.fetchall()}

        cur.execute(
            "SELECT worker_id, command, issued_at FROM remote_commands "
            "WHERE acked_at IS NULL ORDER BY issued_at"
        )
        commands = cur.fetchall()
        if commands:
            sample = ", ".join(f"{r['worker_id']}:{r['command']}" for r in commands[:8])
            more = "" if len(commands) <= 8 else f", +{len(commands) - 8} more"
            blockers.append(f"{len(commands)} open remote command(s): {sample}{more}.")

        cur.execute(
            "SELECT worker_id, role, state, sw_version, "
            "round(EXTRACT(EPOCH FROM (now() - last_beat)))::int AS age_s "
            "FROM worker_heartbeat"
        )
        beats = {r["worker_id"]: r for r in cur.fetchall()}

        cur.execute("SELECT machine_owner, desired_workers, agent, COALESCE(model,'') AS model FROM fleet_desired_state ORDER BY machine_owner")
        desired = cur.fetchall()
        blockers.extend(blocked_desired_agent_chains(desired, active_agent_blocks))
        for row in desired:
            owner = row["machine_owner"]
            wanted = int(row["desired_workers"] or 0)
            for i in range(wanted):
                wid = f"{owner}-{i}"
                beat = beats.get(wid)
                if beat is None:
                    blockers.append(f"missing desired worker heartbeat: {wid}.")
                elif _age_s(beat) > max_age:
                    blockers.append(f"stale desired worker {wid}: age {beat['age_s']}s.")

        cur.execute("SELECT count(*) AS n FROM linkedin_queue WHERE status='queued'")
        linkedin_queued = int(cur.fetchone()["n"] or 0)
        if linkedin_queued:
            linkedin_beats = [r for r in beats.values() if r["role"] == "linkedin"]
            fresh_linkedin = [r for r in linkedin_beats if _age_s(r) <= max_age]
            if not fresh_linkedin:
                if linkedin_beats:
                    ages = ", ".join(f"{r['worker_id']} age {r['age_s']}s" for r in linkedin_beats)
                    blockers.append(f"no fresh LinkedIn worker heartbeat ({ages}).")
                else:
                    blockers.append("missing LinkedIn worker heartbeat.")
            elif len(fresh_linkedin) > 1:
                names = ", ".join(r["worker_id"] for r in fresh_linkedin)
                warnings.append(f"multiple fresh LinkedIn worker heartbeats: {names}.")

        cur.execute(
            "SELECT count(*) FILTER (WHERE status='queued') AS queued, "
            "count(*) FILTER (WHERE status='queued' AND approved_batch IS NOT NULL) AS approved "
            "FROM apply_queue"
        )
        aq = cur.fetchone()
        if int(aq["approved"] or 0) <= 0:
            blockers.append("no approved ATS queued jobs are available.")

        cur.execute("SELECT pinned_worker_version FROM fleet_config WHERE id=1")
        pinned = (cur.fetchone() or {}).get("pinned_worker_version")
        if pinned:
            drift = [
                (wid, r["sw_version"]) for wid, r in beats.items()
                if r["role"] in ("apply", "compute", "discovery", "linkedin")
                and _age_s(r) <= max_age
                and r["sw_version"] not in (None, pinned)
            ]
            if drift:
                sample = ", ".join(f"{wid}={ver}" for wid, ver in drift[:6])
                more = "" if len(drift) <= 6 else f", +{len(drift) - 6} more"
                warnings.append(f"version drift from pinned {pinned}: {sample}{more}.")

print(f"DB now: {now.isoformat()}")
if blockers:
    print("NOT READY")
    for item in blockers:
        print(f"  BLOCKER: {item}")
else:
    print("READY")
if warnings:
    for item in warnings:
        print(f"  WARNING: {item}")
sys.exit(2 if blockers else 0)
'@

  $output = $code | & $py -
  $rc = $LASTEXITCODE
  if ($output) { $output | ForEach-Object { Write-Host $_ } }
  return $rc
}

do {
  $rc = Invoke-ReadyCheck
  if ($rc -eq 0) { exit 0 }
  if (-not $Watch) { exit $rc }
  Write-Host "[check-fleet-ready] not ready; polling again in $PollSeconds seconds..." -ForegroundColor Yellow
  Start-Sleep -Seconds $PollSeconds
} while ($true)
