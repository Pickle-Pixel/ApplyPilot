"""applypilot-fleet-console: a LOCAL, LAN-ONLY web control panel for the apply fleet.

This replaces hand-run SQL with a tiny, audited HTTP surface that is safe enough to
operate a system that submits REAL job applications under the owner's name.

DESIGN / SAFETY (the review rubric S1-S6 lives here):
  * stdlib ONLY -- http.server.ThreadingHTTPServer + json + argparse. NO new pip deps.
    The HTML/CSS/JS frontend is embedded as a string below.
  * Every consequential mutation goes through a STRICT ALLOW-LIST (a dict), never
    eval/getattr on the request action string, and never SQL built from request input.
  * Reuses the proven, tested fleet functions (pgqueue / apply_home_main / codex_bridge)
    rather than reinventing their queries. Connections are short-lived (one per request)
    and always closed in a finally.
  * NO LinkedIn WRITE controls anywhere -- LinkedIn is the catastrophe lane and is shown
    strictly read-only. There is no arm/resume/apply action for it.
  * LAN-ONLY bind: the server refuses to bind 0.0.0.0 or any public IP. It binds a private
    IPv4 (10/8, 172.16/12, 192.168/16) or 127.0.0.1 only.
  * No secrets are written to disk or logged; the DSN is never printed.

WORKER LIVENESS SOURCE (per the task's explicit instruction):
  The apply workers DO write worker_heartbeat. WorkerLoop._beat -> worker.py:_heartbeat
  UPSERTs worker_heartbeat with last_beat=now() on EVERY tick (idle, applying, etc.), and
  apply_worker_main builds a WorkerLoop(role='apply'). So liveness here is REAL, derived
  from worker_heartbeat.last_beat: alive = last_beat within 150s. We filter to role='apply'
  (the LinkedIn lane is excluded from the controllable view) and join current_job (a url)
  back to apply_queue for the company/title of the job a worker is on.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hmac
import http.cookies
import ipaddress
import json
import os
import re
import secrets
import socket
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from psycopg import errors

from applypilot.apply import pgqueue
from applypilot.fleet import apply_home_main as H
from applypilot.fleet import codex_bridge as cb
from applypilot.fleet import compute_home_main as compute_home
from applypilot.fleet import config as fleet_config
from applypilot.fleet import console_machines
from applypilot.fleet import queue as _queue
from applypilot.fleet.failure_taxonomy import canonical_failure_group
from applypilot.fleet.worker import _scrub  # reuse the worker's secret redactor (S1: scrub on READ too)

DEFAULT_PORT = 8787
_HB_ALIVE_SECONDS = 150          # apply workers beat ~every tick; >150s stale = dead
_INFERRED_WINDOW_SECONDS = 300   # fallback only (unused: apply workers DO beat)
_LOG_LAST_ERROR_CAP = 4000       # mirror the worker write caps on read (S3)
_LOG_RECENT_CAP = 8000
_MACHINE_NAME_FALLBACK = {"home": "home", "m2": "tarpon", "m4": "gggtower"}
_BROWSER_ISSUE_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("browser backend crashed", ("browser backend crashed", "browser process crashed", "backend crashed")),
    ("browser service unavailable", ("browser service unavailable", "service unavailable", "browser down")),
    ("playwright/mcp disconnected", ("playwright", "mcp", "disconnected", "browser disconnected")),
    ("file chooser crash", ("file chooser", "filechooser",)),
    ("connection refused", ("connection refused", "[errno 111]", "could not connect")),
    ("timeout after browser action", ("timeout after", "browser timeout", "timed out waiting for")),
    ("captcha solver unsupported", ("captcha solver unsupported", "unsupported captcha", "captcha-solver")),
    ("CAPTCHA present", ("captcha", "visible_captcha", "hcaptcha")),
    ("login gate", ("login required", "login gate", "sign-in required", "please sign in")),
    ("email/OTP verification", ("otp", "verification code", "email verification", "2fa", "two-factor")),
    ("employer application cap", ("application cap", "employer application cap", "application limit reached")),
    ("usage limit", ("usage limit", "usage-limit", "account limit")),
    ("no result line", ("no result line", "could not extract", "no result")),
]
_BROWSER_BACKEND_ISSUES = {
    "browser backend crashed",
    "browser service unavailable",
    "playwright/mcp disconnected",
    "file chooser crash",
    "connection refused",
    "timeout after browser action",
}
_BROWSER_MANUAL_ISSUES = {
    "CAPTCHA present",
    "login gate",
    "email/OTP verification",
}
_BROWSER_OPERATOR_LIMIT_ISSUES = {
    "captcha solver unsupported",
    "employer application cap",
    "usage limit",
    "no result line",
}
_BROWSER_ISSUE_GUIDANCE = {
    "browser backend crashed": "Restart that worker's browser subsystem (worker restart usually recovers).",
    "browser service unavailable": "Check Playwright/browser backend service health on that host; restart if unavailable.",
    "playwright/mcp disconnected": "Verify backend process and transport channel between worker and browser; recycle worker profile.",
    "file chooser crash": "Pause and restart the worker; resume when the environment can open file dialogs.",
    "connection refused": "Check whether the browser backend host/port is reachable and restart affected automation stack.",
    "timeout after browser action": "Browser action is stalling; inspect the worker's recent log tail for the failing page.",
    "CAPTCHA present": "Resolve captcha/login in worker session and reopen lane once solved.",
    "login gate": "Manually resolve login gate or rotate auth profile for that worker.",
    "email/OTP verification": "Open challenge triage and complete OTP/email verification for affected rows.",
    "captcha solver unsupported": "Skip or manually solve this host; the automated browser cannot handle this captcha path.",
    "employer application cap": "Skip this job or host group; the employer or board is refusing more applications.",
    "usage limit": "Switch the worker model/account or wait for the reset time before re-queueing affected rows.",
    "no result line": "Inspect the worker log and adapter path; the apply agent ended without a terminal RESULT line.",
}
_FROZEN_LEASE_THRESHOLD = "300 days"
_HISTORICAL_MACHINE_LABEL = "historical/no owner recorded"
_APPLY_LLM_TASKS = ("apply", "apply_agent")
_DEADMAN_SEVERITY = {
    "critical": 0,
    "warning": 1,
    "warn": 1,
    "info": 2,
}
_DEADMAN_KIND_DETAILS: dict[str, tuple[str, str, str]] = {
    "silent_death": (
        "No apply worker heartbeat",
        "No apply worker heartbeat has been recorded recently; the apply lane may be down or disconnected.",
        "severe",
    ),
    "stalled_queue": (
        "Queued backlog is stalled",
        "Approved backlog is not converting into applied rows; workers may be blocked, parked, or no longer making forward progress.",
        "severe",
    ),
    "selfheal_dead": (
        "Fleet self-healer is unavailable",
        "Watchdog/Doctor recovery loop is not healthy; stalled queues will not auto-heal.",
        "severe",
    ),
    "otp_relay_down": (
        "OTP relay is unavailable",
        "Email-2FA relay is not healthy; jobs waiting on email verification will stall.",
        "severe",
    ),
    "owner_inbox_backlog": (
        "Manual auth backlog requires attention",
        "A manual-auth challenge backlog is building on the owner inbox path; review parked challenges in the console.",
        "severe",
    ),
    "running_hot": (
        "Daily spend is running hot",
        "24h spend has been near or above cap threshold; continue cautiously and watch the cap.",
        "warn",
    ),
}


def _parse_deadman_alert(raw: Any) -> list[tuple[str, str]]:
    """Parse ``deadman_alert`` string into (code, detail) tuples."""
    if not raw:
        return []
    if not isinstance(raw, str):
        return []
    items: list[tuple[str, str]] = []
    for part in raw.split("|"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            kind, detail = part.split(":", 1)
            kind = kind.strip()
            detail = detail.strip()
        else:
            kind = part.strip()
            detail = part.strip()
        if kind:
            items.append((kind, detail))
    # Keep one row per kind, preferring first-seen details where duplicates appear.
    unique: dict[str, str] = {}
    for kind, detail in items:
        unique.setdefault(kind, detail)
    return [(kind, detail) for kind, detail in unique.items()]


def _classify_deadman(kind: str, detail: str) -> dict[str, str]:
    known = kind in _DEADMAN_KIND_DETAILS
    title, reason, severity = _DEADMAN_KIND_DETAILS.get(kind, (
        f"Unknown DeadMan alert: {kind}",
        (
            f"Unhandled watchdog alert '{kind}' was raised"
            + (f" with detail: {detail}" if detail else ".")
        ),
        "warn",
    ))
    if detail:
        detail_text = detail.strip()
        if known and detail_text and detail_text.lower() not in reason.lower():
            reason = f"{reason} ({detail_text})"
    return {"kind": kind, "title": title, "reason": reason, "detail": detail, "severity": severity}


def _model_family(model: str | None) -> str:
    if not model:
        return "unknown"
    m = str(model).lower()
    if "claude" in m or "sonnet" in m or "haiku" in m or "opus" in m:
        return "claude"
    if "gpt" in m or "codex" in m or "deepseek" in m or "gemini" in m or "qwen" in m or "llama" in m or "mixtral" in m:
        return "codex-like"
    return "other"


def _model_vendor(model: str | None) -> str:
    if not model:
        return "unknown"
    m = str(model).lower()
    if "claude" in m or "sonnet" in m or "haiku" in m or "opus" in m:
        return "claude"
    if "gpt" in m or "codex" in m or "o1" in m or "o3" in m or "gpt-4" in m or "gpt-5" in m:
        return "codex"
    if "gemini" in m:
        return "gemini"
    if "deepseek" in m or "qwen" in m or "llama" in m or "mixtral" in m:
        return "other"
    return "other"


_CLAUDE_MODEL_TIERS = frozenset({"sonnet", "opus", "haiku"})


def _normalize_agent_model(agent: Any, model: Any) -> str | None:
    """Normalize legacy telemetry to the model the launcher actually selected."""
    normalized_agent = str(agent or "").strip().lower()
    normalized_model = str(model or "").strip()
    if not normalized_model:
        return None
    if normalized_agent == "codex" and normalized_model.lower() in _CLAUDE_MODEL_TIERS:
        return "codex-default"
    return normalized_model


def _merge_agent_model(heartbeat: Any, usage: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """Merge telemetry without pairing an agent with another agent's model."""
    heartbeat_agent = str(heartbeat.get("current_agent") or "").strip() or None
    usage_agent = str((usage or {}).get("current_agent") or "").strip() or None
    current_agent = heartbeat_agent or usage_agent
    heartbeat_model = _normalize_agent_model(heartbeat_agent, heartbeat.get("current_model"))
    if heartbeat_model:
        return current_agent, heartbeat_model
    if usage_agent and heartbeat_agent and usage_agent.lower() != heartbeat_agent.lower():
        usage_model = None
    else:
        usage_model = _normalize_agent_model(current_agent, (usage or {}).get("current_model"))
    if usage_model:
        return current_agent, usage_model
    if str(current_agent or "").lower() == "codex":
        return current_agent, "codex-default"
    return current_agent, None


def _machine_name_map() -> dict[str, str]:
    raw = os.environ.get("APPLYPILOT_MACHINE_NAMES")
    if not raw:
        return dict(_MACHINE_NAME_FALLBACK)
    try:
        parsed = json.loads(raw)
    except Exception:
        return dict(_MACHINE_NAME_FALLBACK)
    if not isinstance(parsed, dict):
        return dict(_MACHINE_NAME_FALLBACK)
    out: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if key.strip():
            out[key.strip().lower()] = value.strip()
    if out:
        return out
    return dict(_MACHINE_NAME_FALLBACK)


def _machine_display_name(machine_owner: str | None) -> str:
    if not machine_owner:
        return "unknown"
    m = str(machine_owner).strip()
    if not m:
        return "unknown"
    name = _machine_name_map().get(m.lower())
    if name:
        return name
    owner_prefix = m.split("-", 1)[0]
    return _machine_name_map().get(owner_prefix.lower(), m)


def _historical_machine_label(machine_owner: str | None) -> str:
    if machine_owner:
        return _machine_display_name(machine_owner)
    return _HISTORICAL_MACHINE_LABEL


def _latest_worker_llm_usage(conn, *, tasks: tuple[str, ...] = _APPLY_LLM_TASKS) -> dict[str, dict[str, Any]]:
    """Latest durable worker/model evidence, including zero-cost apply runs."""
    placeholders = ", ".join(["%s"] * len(tasks))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS ("
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema=current_schema() AND table_name='apply_result_events' "
            "AND column_name='machine_owner') AS present"
        )
        event_machine_owner = (
            "machine_owner" if cur.fetchone()["present"] else "NULL::TEXT AS machine_owner"
        )
        cur.execute(
            f"""
            WITH evidence AS (
                SELECT worker_id, machine_owner, provider, model, ts, id, 2 AS source_priority
                FROM llm_usage
                WHERE worker_id IS NOT NULL
                  AND task IN ({placeholders})
                UNION ALL
                SELECT worker_id, {event_machine_owner}, agent AS provider, agent_model AS model,
                       created_at AS ts, id, 1 AS source_priority
                FROM apply_result_events
                WHERE worker_id IS NOT NULL
                  AND agent IS NOT NULL
                  AND agent_model IS NOT NULL
            )
            SELECT DISTINCT ON (worker_id)
                worker_id, machine_owner, provider, model, ts
            FROM evidence
            ORDER BY worker_id, ts DESC, source_priority DESC, id DESC
            """,
            tuple(tasks),
        )
        rows = cur.fetchall()
    conn.rollback()  # read-only

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        wid = row.get("worker_id")
        if not wid:
            continue
        out[str(wid)] = {
            "machine_owner": row.get("machine_owner"),
            "current_agent": row.get("provider"),
            "current_model": row.get("model"),
            "ts": row.get("ts"),
        }
    return out


def _agent_parts(value: Any) -> list[str]:
    if not value:
        return []
    if not isinstance(value, str):
        return []
    out = []
    for token in value.split(","):
        token = token.strip()
        if token:
            out.append(token)
    return out


def _agents_state() -> dict[str, Any]:
    conn = pgqueue.connect()  # reads FLEET_PG_DSN only (no generic fallback)
    try:
        apply_placeholders = ", ".join(["%s"] * len(_APPLY_LLM_TASKS))
        with conn.cursor() as cur:
            cur.execute(
                "SELECT wh.worker_id, wh.machine_owner, wh.current_agent, wh.current_model, "
                "wh.agent_chain, wh.last_agent_switch_at, wh.last_agent_switch_reason, "
                "wh.state, wh.last_beat "
                "FROM worker_heartbeat wh WHERE wh.role='apply'"
            )
            rows = [dict(r) if hasattr(r, "keys") else r for r in cur.fetchall()]

            cur.execute(
                "SELECT agent, blocked_until, reason, updated_at FROM agent_availability"
            )
            blocks_rows = cur.fetchall()

            cur.execute(
                "SELECT provider, model, COALESCE(COUNT(*),0) AS apply_calls, "
                "COALESCE(SUM(cost_usd),0) AS spend_usd "
                "FROM llm_usage "
                f"WHERE ts >= now() - interval '24 hours' AND task IN ({apply_placeholders}) "
                "GROUP BY provider, model"
            , tuple(_APPLY_LLM_TASKS))
            usage_rows = cur.fetchall()
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.close()

    active_blocks: dict[str, dict[str, Any]] = {}
    for row in blocks_rows:
        if not row:
            continue
        agent = row["agent"] if hasattr(row, "keys") or isinstance(row, dict) else row[0]
        blocked_until = row["blocked_until"] if hasattr(row, "keys") or isinstance(row, dict) else row[1]
        reason = row["reason"] if hasattr(row, "keys") or isinstance(row, dict) else row[2]
        updated_at = row["updated_at"] if hasattr(row, "keys") or isinstance(row, dict) else row[3]
        if blocked_until is not None and blocked_until > _utcnow():
            active_blocks[agent] = {
                "blocked_until": _iso(blocked_until),
                "reason": reason,
                "updated_at": _iso(updated_at),
            }

    usage_by_agent: dict[str, dict[str, Any]] = {}
    for row in usage_rows:
        if not row:
            continue
        agent = row["provider"] if hasattr(row, "keys") or isinstance(row, dict) else row[0]
        model = row["model"] if hasattr(row, "keys") or isinstance(row, dict) else row[1]
        if not agent:
            continue
        stat = usage_by_agent.setdefault(agent, {"apply_calls": 0, "spend_usd": 0.0, "models": {}})
        stat["apply_calls"] += int((row["apply_calls"] if hasattr(row, "keys") or isinstance(row, dict) else row[2]) or 0)
        stat["spend_usd"] += float((row["spend_usd"] if hasattr(row, "keys") or isinstance(row, dict) else row[3]) or 0)
        if model:
            stat["models"][model] = stat["models"].get(model, 0) + int((row["apply_calls"] if hasattr(row, "keys") or isinstance(row, dict) else row[2]) or 0)

    workers: list[dict[str, Any]] = []
    model_usage: dict[str, int] = {}
    family_usage: dict[str, int] = {}
    chain_workers = 0
    switched_workers = 0
    partial_workers = 0
    blocked_workers = 0
    model_missing_workers = 0
    for r in rows:
        row = dict(r) if hasattr(r, "keys") or isinstance(r, dict) else {
            "worker_id": r[0], "machine_owner": r[1], "current_agent": r[2], "current_model": r[3],
            "agent_chain": r[4], "last_agent_switch_at": r[5], "last_agent_switch_reason": r[6],
            "state": r[7], "last_beat": r[8],
        }
        chain = _agent_parts(row.get("agent_chain"))
        current_agent = row.get("current_agent")
        verdict = "unknown"
        if chain:
            if len(chain) == 1:
                if current_agent == chain[0]:
                    verdict = "not triggered"
                else:
                    verdict = "misconfigured"
            elif current_agent in chain:
                blocked = [a for a in chain if a in active_blocks]
                if blocked:
                    if len(blocked) == len(chain):
                        verdict = "blocked"
                    elif current_agent == chain[0]:
                        verdict = "partial"
                    else:
                        verdict = "working"
                else:
                    verdict = "not triggered" if current_agent == chain[0] else "working"
            else:
                verdict = "misconfigured"
        blocks = []
        if current_agent and current_agent in active_blocks:
            block = active_blocks[current_agent].copy()
            block["agent"] = current_agent
            blocks.append(block)
        if chain and len(chain) > 1:
            chain_workers += 1
        if current_agent and chain and current_agent != chain[0]:
            switched_workers += 1
        if verdict == "partial":
            partial_workers += 1
        if verdict == "blocked":
            blocked_workers += 1
        current_model = _normalize_agent_model(current_agent, row.get("current_model")) or ""
        if current_model:
            model_usage[current_model] = model_usage.get(current_model, 0) + 1
            fam = _model_family(current_model)
            family_usage[fam] = family_usage.get(fam, 0) + 1
        elif current_agent:
            model_missing_workers += 1
        workers.append({
            "worker_id": row.get("worker_id"),
            "machine": _machine_display_name(row.get("machine_owner")),
            "machine_owner": row.get("machine_owner"),
            "current_agent": current_agent,
            "current_model": current_model or None,
            "current_model_family": _model_family(current_model),
            "current_model_vendor": _model_vendor(current_model),
            "agent_chain": chain,
            "last_agent_switch_at": _iso(row.get("last_agent_switch_at")),
            "last_switch_reason": row.get("last_agent_switch_reason"),
            "switch_verdict": verdict,
            "last_beat": _iso(row.get("last_beat")),
            "state": row.get("state"),
            "active_blocked_agents": [k for k in chain if k in active_blocks],
            "agent_blocks": blocks,
            "current_agent_stats": usage_by_agent.get(current_agent) if current_agent else None,
        })

    return {
        "workers": workers,
        "agent_blocks": {
            k: {"blocked_until": v["blocked_until"], "reason": v["reason"], "updated_at": v["updated_at"]}
            for k, v in active_blocks.items()
        },
        "agent_spend": {
            k: {"apply_calls": int(v["apply_calls"]), "spend_usd": round(float(v["spend_usd"]), 6),
                "models": dict(v["models"])}
            for k, v in usage_by_agent.items()
        },
        "summary": {
            "dynamic_workers": chain_workers,
            "switched_workers": switched_workers,
            "partial_workers": partial_workers,
            "blocked_workers": blocked_workers,
            "model_usage": dict(sorted(
                model_usage.items(), key=lambda kv: kv[1], reverse=True
            )),
            "model_family_usage": dict(sorted(
                family_usage.items(), key=lambda kv: kv[1], reverse=True
            )),
            "model_count": sum(model_usage.values()),
            "model_missing_workers": model_missing_workers,
        },
    }


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _iso(v: Any) -> Any:
    return v.isoformat() if hasattr(v, "isoformat") else v


def _seconds_since(ts: Any) -> int | None:
    if ts is None or not hasattr(ts, "tzinfo"):
        return None
    now = _utcnow()
    t = ts if ts.tzinfo is not None else ts.replace(tzinfo=_dt.timezone.utc)
    return int((now - t).total_seconds())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_text(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=_repo_root(), text=True, capture_output=True,
            timeout=3, check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _deployment_info(conn) -> dict[str, Any]:
    """Read-only console/schema/version facts for the dashboard."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='worker_heartbeat' "
            "AND column_name IN ('current_agent','current_model','agent_chain',"
            "'last_agent_switch_at','last_agent_switch_reason')"
        )
        telemetry_cols = {r["column_name"] for r in cur.fetchall()}
        cur.execute("SELECT to_regclass('public.fleet_console_audit') AS audit_table")
        audit_table = cur.fetchone()["audit_table"]
        cur.execute(
            "SELECT COALESCE(sw_version, '(unknown)') AS version, worker_id, machine_owner "
            "FROM worker_heartbeat ORDER BY 1, worker_id"
        )
        groups: dict[str, dict[str, Any]] = {}
        for row in cur.fetchall():
            version = row["version"]
            owner = console_machines.infer_machine_owner(row["worker_id"], row.get("machine_owner"))
            machine = console_machines.display_name(owner)
            group = groups.setdefault(
                version, {"version": version, "workers": 0, "machines": [], "worker_ids": []}
            )
            group["workers"] += 1
            group["worker_ids"].append(row["worker_id"])
            if machine not in group["machines"]:
                group["machines"].append(machine)
    conn.rollback()
    return {
        "console": {
            "branch": _git_text(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown",
            "commit": _git_text(["rev-parse", "--short", "HEAD"]) or "unknown",
            "path": str(_repo_root()),
        },
        "schema": {
            "agent_telemetry": {
                "current_agent", "current_model", "agent_chain",
                "last_agent_switch_at", "last_agent_switch_reason",
            }.issubset(telemetry_cols),
            "audit_table": bool(audit_table),
        },
        "worker_versions": list(groups.values()),
    }


def _classify_browser_issue(text: Any) -> str | None:
    if not text:
        return None
    if not isinstance(text, str):
        return None
    normalized = text.lower()
    for issue, triggers in _BROWSER_ISSUE_PATTERNS:
        if issue == "playwright/mcp disconnected":
            if (
                "browser disconnected" in normalized
                or "playwright disconnected" in normalized
                or "mcp disconnected" in normalized
                or ("playwright" in normalized and "disconnected" in normalized)
                or ("mcp" in normalized and "disconnected" in normalized)
            ):
                return issue
            continue
        for trigger in triggers:
            if trigger in normalized:
                return issue
        # continue trying other issues if no trigger matches this one
    return None


def _summarize_browser_issue_rows(
    issues: list[dict[str, Any]], tracked: set[str]
) -> list[str]:
    """Build compact ``issue (machines/models)`` summaries for status recommendations."""
    by_issue: dict[str, list[str]] = {}
    for issue in issues:
        issue_name = issue.get("issue")
        if issue_name not in tracked:
            continue
        model = issue.get("current_model") or "unknown model"
        machine = issue.get("machine") or "unknown"
        worker = issue.get("worker_id") or "worker"
        age = issue.get("age")
        age_text = f"{age}s ago" if isinstance(age, int) else "age unknown"
        issue_key = f"{issue_name} on {machine}"
        bucket = by_issue.setdefault(issue_key, [])
        summary = f"{worker} ({model}, {age_text})"
        if summary not in bucket:
            bucket.append(summary)
    return [f"{issue} [{'; '.join(details)}]" for issue, details in sorted(by_issue.items())]


def _browser_issue_guidance(issues: list[dict[str, Any]], tracked: set[str]) -> list[str]:
    guidance: list[str] = []
    for issue in issues:
        issue_name = issue.get("issue")
        if issue_name not in tracked:
            continue
        hint = _BROWSER_ISSUE_GUIDANCE.get(str(issue_name))
        if hint and hint not in guidance:
            guidance.append(hint)
    return guidance


def _browser_health(conn) -> dict[str, Any]:
    """Read-only browser/backend health from heartbeats.

    We keep this lightweight (metadata, not full log dump) and only return the newest
    issue signal per worker. The endpoint stays on /api/diagnostics, so no 4-second
    status polling penalty.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT worker_id, machine_owner, role, state, last_beat, last_error, recent_log, "
            "       current_agent, current_model "
            "FROM worker_heartbeat ORDER BY worker_id"
        )
        rows = cur.fetchall()
    conn.rollback()  # read-only

    total = len(rows)
    active_total = 0
    stale_problem_workers = 0
    inactive_problem_workers = 0
    issues: list[dict[str, Any]] = []
    by_issue: dict[str, int] = {}
    for r in rows:
        age = _seconds_since(r.get("last_beat"))
        alive = age is not None and age <= _HB_ALIVE_SECONDS
        if alive:
            active_total += 1
        worker_issue = _classify_browser_issue(_scrub(r.get("last_error"))) or _classify_browser_issue(_scrub(r.get("recent_log")))
        if worker_issue is None:
            continue
        if not alive:
            stale_problem_workers += 1
            continue
        if str(r.get("state") or "").lower() in {"paused", "stopped", "drained", "version_mismatch"}:
            inactive_problem_workers += 1
            continue
        by_issue[worker_issue] = by_issue.get(worker_issue, 0) + 1
        issues.append({
            "worker_id": r.get("worker_id"),
            "machine": _machine_display_name(r.get("machine_owner")),
            "role": r.get("role"),
            "state": r.get("state"),
            "issue": worker_issue,
            "last_beat": _iso(r.get("last_beat")),
            "age": age,
            "current_agent": r.get("current_agent"),
            "current_model": r.get("current_model"),
            "current_model_family": _model_family(r.get("current_model")),
            "current_model_vendor": _model_vendor(r.get("current_model")),
            "evidence": _scrub((r.get("last_error") or r.get("recent_log")) or "")[:250],
        })
    return {
        "issues": issues,
        "summary": {
            "workers": active_total,
            "tracked_workers": total,
            "stale_workers": total - active_total,
            "problem_workers": len(issues),
            "stale_problem_workers": stale_problem_workers,
            "inactive_problem_workers": inactive_problem_workers,
            "by_issue": by_issue,
        },
    }


def _desired_machine_workers(conn) -> dict[str, int]:
    """Optional desired worker counts by human-readable machine name.

    Some installs have fleet_desired_state; older/test installs do not. Treat the table
    as advisory telemetry only and keep /api/diagnosis available when it is absent.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('fleet_desired_state') AS rel")
        if cur.fetchone()["rel"] is None:
            conn.rollback()
            return {}
        cur.execute(
            "SELECT machine_owner, desired_workers "
            "FROM fleet_desired_state WHERE desired_workers > 0"
        )
        rows = cur.fetchall()
    conn.rollback()

    desired: dict[str, int] = {}
    for row in rows:
        name = _machine_display_name(row.get("machine_owner"))
        desired[name] = desired.get(name, 0) + int(row.get("desired_workers") or 0)
    return desired


def _network_rollup(conn) -> dict[str, Any]:
    """All-role worker_heartbeat rollup for fleet-network visibility."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT worker_id, machine_owner, role, state, last_beat "
            "FROM worker_heartbeat ORDER BY worker_id"
        )
        rows = cur.fetchall()
    conn.rollback()

    machines: dict[str, dict[str, Any]] = {}
    by_role: dict[str, int] = {}
    alive_total = 0
    stale_total = 0
    for row in rows:
        machine = _machine_display_name(row.get("machine_owner"))
        role = row.get("role") or "unknown"
        secs = _seconds_since(row.get("last_beat"))
        alive = secs is not None and secs <= _HB_ALIVE_SECONDS
        item = machines.setdefault(
            machine,
            {"workers": 0, "alive": 0, "stale": 0, "by_role": {}},
        )
        item["workers"] += 1
        item["by_role"][role] = item["by_role"].get(role, 0) + 1
        by_role[role] = by_role.get(role, 0) + 1
        if alive:
            item["alive"] += 1
            alive_total += 1
        else:
            item["stale"] += 1
            stale_total += 1

    return {
        "workers": len(rows),
        "alive": alive_total,
        "stale": stale_total,
        "by_role": dict(sorted(by_role.items())),
        "machines": {
            name: {
                "workers": int(data["workers"]),
                "alive": int(data["alive"]),
                "stale": int(data["stale"]),
                "by_role": dict(sorted(data["by_role"].items())),
            }
            for name, data in sorted(machines.items())
        },
    }


# ---------------------------------------------------------------------------
# READ: status snapshot (all queries via short-lived connection)
# ---------------------------------------------------------------------------
def _gate_and_queue(conn) -> tuple[dict, dict, dict]:
    """fleet_config gate + apply_queue depth + leasable + spend, in one read txn.

    base_leasable is the approved queued pool before global stop gates.
    leasable is the operator-facing effective count after pause, ATS pause,
    spend cap, and canary exhaustion are applied.
    spent_usd = SUM(cumulative_cost_usd) over apply_queue (same SUM the cap halt uses)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT paused, ats_paused, ats_pause_source, ats_apply_mode, "
            "canary_enabled, canary_remaining, spend_cap_usd "
            "FROM fleet_config WHERE id=1"
        )
        cfg = cur.fetchone() or {}

        cur.execute("SELECT COALESCE(SUM(cumulative_cost_usd),0) AS s FROM apply_queue")
        spent = float(cur.fetchone()["s"])

        cur.execute("SELECT status, COUNT(*) AS n FROM apply_queue GROUP BY status")
        depth = {r["status"]: int(r["n"]) for r in cur.fetchall()}
        cur.execute(
            "SELECT COALESCE(lane, 'unknown') AS lane, status, COALESCE(COUNT(*),0) AS n "
            "FROM apply_queue GROUP BY lane, status"
        )
        depth_by_lane: dict[str, dict[str, int]] = {}
        for row in cur.fetchall():
            lane = row["lane"]
            status = row["status"]
            if not lane:
                lane = "unknown"
            if lane not in depth_by_lane:
                depth_by_lane[lane] = {}
            depth_by_lane[lane][status] = int(row["n"])
        cur.execute(
            "SELECT COALESCE(lane, 'unknown') AS lane, "
            "       COUNT(*) FILTER (WHERE status='leased') AS leased, "
            "       COUNT(*) FILTER (WHERE status='leased' AND lease_expires_at IS NOT NULL "
            f"                        AND lease_expires_at > now() + interval '{_FROZEN_LEASE_THRESHOLD}') AS parked_frozen, "
            "       COUNT(*) FILTER (WHERE status='leased' AND lease_expires_at IS NOT NULL "
            f"                        AND lease_expires_at > now() + interval '{_FROZEN_LEASE_THRESHOLD}' "
            "                        AND apply_status='challenge_pending') AS challenge_parked, "
            "       COUNT(*) FILTER (WHERE status='leased' AND lease_expires_at IS NOT NULL "
            f"                        AND lease_expires_at > now() + interval '{_FROZEN_LEASE_THRESHOLD}' "
            "                        AND apply_status IS DISTINCT FROM 'challenge_pending') AS unexpected_frozen, "
            "       COUNT(*) FILTER (WHERE status='leased' AND (lease_expires_at IS NULL "
            f"                        OR lease_expires_at <= now() + interval '{_FROZEN_LEASE_THRESHOLD}')) AS active_leased "
            "FROM apply_queue GROUP BY lane"
        )
        lease_by_lane: dict[str, dict[str, int]] = {}
        for row in cur.fetchall():
            lane = row["lane"] or "unknown"
            lease_by_lane[lane] = {
                "leased": int(row.get("leased") or 0),
                "parked_frozen": int(row.get("parked_frozen") or 0),
                "challenge_parked": int(row.get("challenge_parked") or 0),
                "unexpected_frozen": int(row.get("unexpected_frozen") or 0),
                "active_leased": int(row.get("active_leased") or 0),
            }

        cur.execute(
            "SELECT COUNT(*) AS n FROM apply_queue q "
            "WHERE q.status='queued' AND q.lane='ats' AND q.approved_batch IS NOT NULL "
            "  AND NOT EXISTS (SELECT 1 FROM applied_set a WHERE a.dedup_key = q.dedup_key)"
        )
        base_leasable = int(cur.fetchone()["n"])
        cur.execute(
            "SELECT COUNT(*) AS n FROM apply_queue q "
            "WHERE q.status='queued' AND q.approved_batch IS NOT NULL "
            "  AND EXISTS (SELECT 1 FROM applied_set a WHERE a.dedup_key = q.dedup_key)"
        )
        dedup_blocked = int(cur.fetchone()["n"])
        cur.execute(
            "SELECT COUNT(*) AS n FROM apply_queue "
            "WHERE status='queued' AND approved_batch IS NOT NULL"
        )
        approved = int(cur.fetchone()["n"])
    conn.rollback()  # read-only

    paused = bool(cfg.get("paused"))
    ats_paused = bool(cfg.get("ats_paused"))
    ats_apply_mode = cfg.get("ats_apply_mode") or "stopped"
    canary_enabled = bool(cfg.get("canary_enabled"))
    canary_remaining = (int(cfg["canary_remaining"])
                        if cfg.get("canary_remaining") is not None else None)
    spend_cap = float(cfg.get("spend_cap_usd") or 0)
    lane_stopped = ats_apply_mode == "stopped"
    canary_exhausted = (
        ats_apply_mode == "canary"
        and (not canary_enabled or not (canary_remaining is not None and canary_remaining > 0))
    )
    spend_halted = spend_cap > 0 and spent >= spend_cap
    lease_gate_open = not (paused or ats_paused or lane_stopped or canary_exhausted or spend_halted)
    halt_reasons = []
    operator_paused = str(cfg.get("ats_pause_source") or "").lower() == "operator" and (paused or ats_paused)
    if paused:
        halt_reasons.append("paused")
    if ats_paused:
        halt_reasons.append("ats_paused")
    if lane_stopped:
        halt_reasons.append("ats_stopped")
    if canary_exhausted:
        halt_reasons.append("canary_exhausted")
    if spend_halted:
        halt_reasons.append("spend_cap")

    gate = {
        "paused": paused,
        "ats_paused": ats_paused,
        "ats_pause_source": cfg.get("ats_pause_source"),
        "ats_apply_mode": ats_apply_mode,
        "canary_enabled": canary_enabled,
        "canary_remaining": canary_remaining,
        "spend_cap_usd": spend_cap,
        "spent_usd": spent,
        "base_leasable": base_leasable,
        "leasable": base_leasable if lease_gate_open else 0,
        "lease_gate_open": lease_gate_open,
        "halt_reasons": halt_reasons,
    }
    queue = {
        "apply": {
            "queued": depth.get("queued", 0),
            "leased": depth.get("leased", 0),
            "active_leased": lease_by_lane.get("ats", {}).get("active_leased", 0)
            + lease_by_lane.get("unknown", {}).get("active_leased", 0)
            + lease_by_lane.get("compute", {}).get("active_leased", 0),
            "parked_frozen": lease_by_lane.get("ats", {}).get("parked_frozen", 0)
            + lease_by_lane.get("unknown", {}).get("parked_frozen", 0)
            + lease_by_lane.get("compute", {}).get("parked_frozen", 0),
            "challenge_parked": lease_by_lane.get("ats", {}).get("challenge_parked", 0)
            + lease_by_lane.get("unknown", {}).get("challenge_parked", 0)
            + lease_by_lane.get("compute", {}).get("challenge_parked", 0),
            "unexpected_frozen": lease_by_lane.get("ats", {}).get("unexpected_frozen", 0)
            + lease_by_lane.get("unknown", {}).get("unexpected_frozen", 0)
            + lease_by_lane.get("compute", {}).get("unexpected_frozen", 0),
            "applied": depth.get("applied", 0),
            "failed": depth.get("failed", 0) + depth.get("blocked", 0),
            "crash_unconfirmed": depth.get("crash_unconfirmed", 0),
            "diagnosis": diagnosis,
        "approved": approved,
        "dedup_blocked": dedup_blocked,
        "base_leasable": base_leasable,
        "by_lane": {
            "ats": {
                "queued": depth_by_lane.get("ats", {}).get("queued", 0),
                "leased": depth_by_lane.get("ats", {}).get("leased", 0),
                "active_leased": lease_by_lane.get("ats", {}).get("active_leased", 0),
                "parked_frozen": lease_by_lane.get("ats", {}).get("parked_frozen", 0),
                "challenge_parked": lease_by_lane.get("ats", {}).get("challenge_parked", 0),
                "unexpected_frozen": lease_by_lane.get("ats", {}).get("unexpected_frozen", 0),
                "applied": depth_by_lane.get("ats", {}).get("applied", 0),
            },
            "compute": {
                "queued": depth_by_lane.get("compute", {}).get("queued", 0),
                "leased": depth_by_lane.get("compute", {}).get("leased", 0),
                "active_leased": lease_by_lane.get("compute", {}).get("active_leased", 0),
                "parked_frozen": lease_by_lane.get("compute", {}).get("parked_frozen", 0),
                "challenge_parked": lease_by_lane.get("compute", {}).get("challenge_parked", 0),
                "unexpected_frozen": lease_by_lane.get("compute", {}).get("unexpected_frozen", 0),
                "applied": depth_by_lane.get("compute", {}).get("applied", 0),
            },
            "other": {
                "queued": depth_by_lane.get("unknown", {}).get("queued", 0),
                "leased": depth_by_lane.get("unknown", {}).get("leased", 0),
                "active_leased": lease_by_lane.get("unknown", {}).get("active_leased", 0),
                "parked_frozen": lease_by_lane.get("unknown", {}).get("parked_frozen", 0),
                "challenge_parked": lease_by_lane.get("unknown", {}).get("challenge_parked", 0),
                "unexpected_frozen": lease_by_lane.get("unknown", {}).get("unexpected_frozen", 0),
                "applied": depth_by_lane.get("unknown", {}).get("applied", 0),
            },
        }
    }
    }
    if operator_paused:
        apply_state = {
            "code": "operator_paused",
            "severity": "info",
            "reason": "Fleet is intentionally paused by the operator.",
        }
    elif paused:
        apply_state = {
            "code": "paused",
            "severity": "halted",
            "reason": "Fleet is paused by the shared kill switch.",
        }
    elif ats_paused:
        apply_state = {
            "code": "ats_paused",
            "severity": "halted",
            "reason": "ATS lane is paused.",
        }
    elif canary_exhausted:
        apply_state = {
            "code": "ats_canary_exhausted",
            "severity": "halted",
            "reason": "ATS canary is exhausted.",
        }
    elif spend_halted:
        apply_state = {
            "code": "spend_cap_reached",
            "severity": "halted",
            "reason": "Spend cap reached; leasing is temporarily halted.",
        }
    elif gate["leasable"] > 0:
        apply_state = {
            "code": "ready_to_apply",
            "severity": "ok",
            "reason": "Leaseable ATS jobs are available.",
        }
    elif approved > 0 and dedup_blocked >= approved:
        apply_state = {
            "code": "ready_jobs_not_leaseable",
            "severity": "warn",
            "reason": "Queued approved ATS rows are blocked by dedup; no new leaseable jobs.",
        }
    else:
        apply_state = {
            "code": "no_leaseable_jobs",
            "severity": "warn",
            "reason": "No leaseable ATS jobs are available.",
        }
    return gate, queue, apply_state


def _workers(conn) -> list[dict]:
    """Per-apply-worker liveness from worker_heartbeat (REAL: apply workers beat every
    tick). alive = last_beat within 150s. current_job (a url) is joined to apply_queue
    for the company/title.

    The applied count is derived from AUTHORITATIVE apply_queue rows
    (status='applied' AND worker_id = this worker), NOT worker_heartbeat.success_today:
    the apply path (WorkerLoop._beat -> worker._heartbeat) never writes success_today,
    so it would be a misleading permanent 0. apply_queue.worker_id is set on every
    terminal write by pgqueue.write_result, so this count is real."""
    out: list[dict] = []
    latest_usage = _latest_worker_llm_usage(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT wh.worker_id, wh.machine_owner, wh.home_ip, wh.state, wh.role, wh.current_job, "
            "       wh.sw_version, wh.last_beat, wh.current_agent, wh.current_model, "
            "       wh.agent_chain, wh.last_agent_switch_at, wh.last_agent_switch_reason, "
            "       aq.company AS cur_company, aq.title AS cur_title, "
            "       (SELECT COUNT(*) FROM apply_queue done "
            "          WHERE done.worker_id = wh.worker_id "
            "            AND done.status = 'applied') AS applied_n "
            "FROM worker_heartbeat wh "
            "LEFT JOIN apply_queue aq ON aq.url = wh.current_job "
            "WHERE wh.role = 'apply' "
            "ORDER BY wh.worker_id"
        )
        rows = cur.fetchall()
    conn.rollback()  # read-only
    for r in rows:
        secs = _seconds_since(r["last_beat"])
        alive = secs is not None and secs <= _HB_ALIVE_SECONDS
        usage = latest_usage.get(str(r.get("worker_id") or ""))
        machine_owner = r.get("machine_owner") or (usage or {}).get("machine_owner")
        current_agent, current_model = _merge_agent_model(r, usage)
        current = None
        if r["current_job"] and r["state"] == "applying":
            current = {"company": r["cur_company"], "title": r["cur_title"]}
        machine_owner = console_machines.infer_machine_owner(
            r["worker_id"], r.get("machine_owner")
        )
        out.append({
            "worker_id": r["worker_id"],
            "machine": _machine_display_name(machine_owner),
            "machine_owner": machine_owner,
            "machine_display_name": console_machines.display_name(machine_owner),
            "home_ip": r.get("home_ip"),
            "alive": bool(alive),
            "health": "alive" if alive else "stale",
            "last_beat": _iso(r["last_beat"]),
            "seconds_since": secs,
            "role": r["role"],
            "state": r["state"],
            "lane": r["role"],
            "sw_version": r.get("sw_version"),
            "applied": int(r["applied_n"] or 0),
            "current_agent": current_agent,
            "current_model": current_model,
            "current_model_family": _model_family(current_model),
            "current_model_vendor": _model_vendor(current_model),
            "agent_chain": r.get("agent_chain"),
            "last_agent_switch_at": _iso(r["last_agent_switch_at"]),
            "last_agent_switch_reason": r.get("last_agent_switch_reason"),
            "current": current,
        })
    return out


def _versions(conn) -> dict:
    """Read-only software-version rollup for drift visibility.

    Application workers use their lane-specific canary pin. Non-application workers
    use the generic staged rollout canary, preserving the legacy behavior.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pinned_worker_version, canary_version, canary_worker_id, "
            "ats_canary_version, ats_canary_worker_id, "
            "linkedin_canary_version, linkedin_canary_worker_id "
            "FROM fleet_config WHERE id=1"
        )
        cfg = cur.fetchone() or {}
        cur.execute(
            "SELECT worker_id, machine_owner, role, sw_version "
            "FROM worker_heartbeat "
            "WHERE role IN ('apply', 'linkedin', 'compute', 'discovery') "
            "ORDER BY worker_id"
        )
        rows = cur.fetchall()
    conn.rollback()  # read-only

    counts: dict[str, int] = {}
    drifted: list[dict] = []
    for row in rows:
        version = row.get("sw_version") or "(unreported)"
        counts[version] = counts.get(version, 0) + 1
        expected = fleet_config.version_for_worker_config(
            cfg, row.get("worker_id"), lane=row.get("role")
        )
        if expected and version != expected:
            drifted.append({
                "worker_id": row.get("worker_id"),
                "machine_owner": row.get("machine_owner"),
                "sw_version": row.get("sw_version"),
            })
    return {
        "pinned_worker_version": cfg.get("pinned_worker_version"),
        "canary_version": cfg.get("canary_version"),
        "canary_worker_id": cfg.get("canary_worker_id"),
        "ats_canary_version": cfg.get("ats_canary_version"),
        "ats_canary_worker_id": cfg.get("ats_canary_worker_id"),
        "linkedin_canary_version": cfg.get("linkedin_canary_version"),
        "linkedin_canary_worker_id": cfg.get("linkedin_canary_worker_id"),
        "worker_versions": counts,
        "drifted_workers": drifted,
    }


def _recent(conn, limit: int = 15) -> list[dict]:
    """Newest-first recent terminal events from cb.recent_results, flattened to the
    panel's row shape (apply lane only carries company/title detail)."""
    res = cb._recent_results(conn, limit)  # rolls back internally; reuses the bridge query
    rows: list[dict] = []
    for r in res.get("results", []):
        detail = r.get("detail") or {}
        rows.append({
            "time": r.get("finished_at"),
            "worker": detail.get("worker_id"),  # apply rows don't carry a worker; stays None
            "status": r.get("status"),
            "company": detail.get("company"),
            "title": detail.get("title"),
            "lane": r.get("lane"),
        })
    return rows[:limit]


def _linkedin_summary(conn) -> dict:
    """READ-ONLY LinkedIn rollup. halted = account:linkedin governor halted_until in the
    future. There is intentionally NO write path for any of this."""
    with conn.cursor() as cur:
        cur.execute("SELECT status, COUNT(*) AS n FROM linkedin_queue GROUP BY status")
        depth = {r["status"]: int(r["n"]) for r in cur.fetchall()}
        cur.execute(
            "SELECT linkedin_apply_mode, linkedin_canary_enabled, linkedin_canary_remaining "
            "FROM fleet_config WHERE id=1"
        )
        cfg = cur.fetchone() or {}
        cur.execute(
            "SELECT halted_until FROM rate_governor WHERE scope_key='account:linkedin'"
        )
        gov = cur.fetchone()
    conn.rollback()  # read-only
    halted = False
    if gov and gov.get("halted_until") is not None:
        secs = _seconds_since(gov["halted_until"])
        halted = secs is not None and secs < 0  # halted_until still in the future
    return {
        "queued": depth.get("queued", 0),
        "applied": depth.get("applied", 0),
        "apply_mode": cfg.get("linkedin_apply_mode") or "stopped",
        "canary_enabled": bool(cfg.get("linkedin_canary_enabled")),
        "canary_remaining": cfg.get("linkedin_canary_remaining"),
        "halted": bool(halted),
    }


def _deadman_status(conn) -> dict:
    """READ-ONLY watchdog summary for the operator-facing diagnosis surface."""
    with conn.cursor() as cur:
        cur.execute("SELECT deadman_alert, deadman_alert_at FROM fleet_config WHERE id=1")
        cfg = cur.fetchone() or {}
    conn.rollback()  # read-only
    alert = cfg.get("deadman_alert")
    code = None
    title = None
    reason = None
    alerts = []
    top: dict[str, str] | None = None
    deadman_items = _parse_deadman_alert(alert)
    if deadman_items:
        for kind, detail in deadman_items:
            classified = _classify_deadman(kind, detail)
            alerts.append(classified)
            if top is None:
                top = classified
            elif _DEADMAN_SEVERITY.get(classified["severity"], 9) < _DEADMAN_SEVERITY.get(top.get("severity", "warn"), 9):
                top = classified
    if alert:
        code = top.get("kind") if top is not None else deadman_items[0][0]
        title = top.get("title") if top is not None else f"Deadman alert: {code}"
        reason = top.get("reason") if top is not None else (deadman_items[0][1] if deadman_items else str(alert))
    return {
        "active": bool(alert),
        "code": code,
        "title": title,
        "reason": reason,
        "alerts": alerts,
        "alert": _scrub(alert)[:300] if alert else None,
        "alert_at": _iso(cfg.get("deadman_alert_at")),
    }


def _agent_routing_summary(conn, workers: list[dict]) -> dict:
    """READ-ONLY model/agent summary for the top diagnosis surface."""
    with conn.cursor() as cur:
        cur.execute("SELECT agent, blocked_until FROM agent_availability")
        block_rows = cur.fetchall()
    conn.rollback()  # read-only

    active_blocks = {
        r["agent"]
        for r in block_rows
        if r.get("agent") and r.get("blocked_until") is not None and r["blocked_until"] > _utcnow()
    }
    model_usage: dict[str, int] = {}
    family_usage: dict[str, int] = {}
    dynamic_workers = 0
    switched_workers = 0
    partial_workers = 0
    blocked_workers = 0
    model_missing_workers = 0
    stale_model_missing_workers = sum(
        1 for worker in workers
        if not worker.get("alive") and worker.get("current_agent")
        and not str(worker.get("current_model") or "").strip()
    )
    active_workers = [worker for worker in workers if worker.get("alive")]
    for worker in active_workers:
        chain = _agent_parts(worker.get("agent_chain"))
        current_agent = worker.get("current_agent")
        if len(chain) > 1:
            dynamic_workers += 1
        if current_agent and chain and current_agent != chain[0]:
            switched_workers += 1
        if chain:
            blocked = [agent for agent in chain if agent in active_blocks]
            if blocked:
                if len(blocked) == len(chain):
                    blocked_workers += 1
                elif current_agent == chain[0]:
                    partial_workers += 1
        current_model = str(worker.get("current_model") or "").strip()
        if current_model:
            model_usage[current_model] = model_usage.get(current_model, 0) + 1
            family = _model_family(current_model)
            family_usage[family] = family_usage.get(family, 0) + 1
        elif current_agent:
            model_missing_workers += 1

    return {
        "workers": len(active_workers),
        "tracked_workers": len(workers),
        "stale_workers": len(workers) - len(active_workers),
        "dynamic_workers": dynamic_workers,
        "switched_workers": switched_workers,
        "partial_workers": partial_workers,
        "blocked_workers": blocked_workers,
        "active_agent_blocks": sorted(active_blocks),
        "model_usage": dict(sorted(model_usage.items(), key=lambda kv: (-kv[1], kv[0]))),
        "model_family_usage": dict(sorted(family_usage.items(), key=lambda kv: (-kv[1], kv[0]))),
        "model_count": sum(model_usage.values()),
        "model_missing_workers": model_missing_workers,
        "stale_model_missing_workers": stale_model_missing_workers,
    }


def _compute_summary(conn) -> dict:
    """READ-ONLY compute/scoring health for the operator diagnosis surface."""
    try:
        return compute_home.status_report(pg_conn=conn, task="score")
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        return {
            "task": "score",
            "unavailable": True,
            "error": _scrub(str(exc))[:200],
            "context": {"version": "", "resume_chars": 0, "kg_chars": 0},
            "queue": {},
            "active_workers": {"total": 0, "by_state": {}},
            "score_workers": {
                "expected": None,
                "cap": None,
                "active": 0,
                "within_expected": None,
                "by_owner": {},
                "recent_done_15m": 0,
                "recent_failed_15m": 0,
                "recent_rate_limited_15m": 0,
                "failure_rate_15m": None,
            },
            "bad_reasoning_unsynced_done": 0,
            "eta_seconds": None,
            "recent_done_15m": 0,
        }


def _throughput_forecast(conn, gate: dict, workers: list[dict]) -> dict:
    """READ-ONLY lightweight apply-throughput forecast for the diagnosis surface."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT "
            "COUNT(*) FILTER (WHERE status='applied' AND applied_at >= now() - interval '1 hour') AS applied_1h, "
            "COUNT(*) FILTER (WHERE status='applied' AND applied_at >= now() - interval '24 hours') AS applied_24h, "
            "MAX(applied_at) FILTER (WHERE status='applied') AS last_applied "
            "FROM apply_queue"
        )
        apply_stats = cur.fetchone() or {}
        cur.execute("SELECT COUNT(*) AS n FROM auth_challenge WHERE resolved_at IS NULL")
        challenge_stats = cur.fetchone() or {}
    conn.rollback()  # read-only

    applied_1h = int(apply_stats.get("applied_1h") or 0)
    applied_24h = int(apply_stats.get("applied_24h") or 0)
    live_apply_workers = sum(1 for row in workers if row.get("alive"))
    leaseable = int(gate.get("leasable") or 0)
    estimated_per_hour = float(applied_1h)
    eta_hours = None
    if estimated_per_hour > 0 and leaseable > 0:
        eta_hours = round(leaseable / estimated_per_hour, 2)
    return {
        "applies_last_1h": applied_1h,
        "applies_last_24h": applied_24h,
        "live_apply_workers": live_apply_workers,
        "leaseable_jobs": leaseable,
        "open_challenges": int(challenge_stats.get("n") or 0),
        "estimated_applies_per_hour": estimated_per_hour,
        "eta_hours_to_exhaust_leaseable": eta_hours,
        "last_successful_apply_at": _iso(apply_stats.get("last_applied")),
        "assumption": "Estimated rate uses confirmed applies in the last hour; ETA is blank until at least one recent apply exists.",
    }


def _daily_goals(conn, gate: dict) -> dict:
    """READ-ONLY optional daily target strip for the diagnosis surface."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS ("
            "  SELECT 1 FROM information_schema.columns "
            "  WHERE table_name='fleet_config' AND column_name='daily_apply_target'"
            ") AS has_target"
        )
        has_target = bool((cur.fetchone() or {}).get("has_target"))
        if has_target:
            cur.execute("SELECT daily_apply_target FROM fleet_config WHERE id=1")
            cfg = cur.fetchone() or {}
            target = cfg.get("daily_apply_target")
        else:
            target = None
        cur.execute(
            "SELECT COUNT(*) AS n FROM apply_queue "
            "WHERE status='applied' "
            "AND applied_at >= (date_trunc('day', now() AT TIME ZONE 'America/New_York') "
            "                   AT TIME ZONE 'America/New_York')"
        )
        applied_today = int((cur.fetchone() or {}).get("n") or 0)
    conn.rollback()  # read-only

    target_today = int(target) if target is not None else None
    remaining = max(target_today - applied_today, 0) if target_today is not None else None
    return {
        "configured": target_today is not None,
        "message": "Daily target active" if target_today is not None else "No daily target configured",
        "applied_today": applied_today,
        "target_today": target_today,
        "remaining_target": remaining,
        "canary_remaining": gate.get("canary_remaining"),
        "projected_shortfall": remaining,
    }


def _data_freshness(conn, *, endpoint: str = "diagnosis") -> dict:
    """READ-ONLY freshness rollup for the diagnosis endpoint and major dashboard panels."""
    generated_at = _utcnow()
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(last_beat) AS ts FROM worker_heartbeat")
        worker_ts = (cur.fetchone() or {}).get("ts")
        cur.execute("SELECT MAX(applied_at) AS ts FROM apply_queue WHERE status='applied'")
        apply_ts = (cur.fetchone() or {}).get("ts")
        cur.execute("SELECT MAX(discovered_at) AS ts FROM discovered_postings")
        discovery_ts = (cur.fetchone() or {}).get("ts")
        cur.execute("SELECT MAX(updated_at) AS ts FROM compute_queue")
        compute_ts = (cur.fetchone() or {}).get("ts")
        cur.execute("SELECT updated_at AS ts FROM fleet_config WHERE id=1")
        config_ts = (cur.fetchone() or {}).get("ts")
    conn.rollback()  # read-only

    return {
        "endpoint": endpoint,
        "generated_at": _iso(generated_at),
        "last_worker_beat_at": _iso(worker_ts),
        "last_apply_at": _iso(apply_ts),
        "last_discovery_at": _iso(discovery_ts),
        "last_compute_at": _iso(compute_ts),
        "config_updated_at": _iso(config_ts),
        "ages": {
            "last_worker_beat_seconds": _seconds_since(worker_ts),
            "last_apply_seconds": _seconds_since(apply_ts),
            "last_discovery_seconds": _seconds_since(discovery_ts),
            "last_compute_seconds": _seconds_since(compute_ts),
            "config_updated_seconds": _seconds_since(config_ts),
        },
    }


def _status_recommendation(
    *,
    gate: dict,
    queue_apply: dict,
    apply_state: dict,
    workers: list[dict],
    linkedin: dict,
    discovery: dict | None,
    doctor_sig: dict | None,
    agents: dict,
    browser: dict,
) -> dict[str, str]:
    """Small next-action summary for the fast /api/status poll."""
    deadman_alert = (doctor_sig or {}).get("deadman_alert")
    if deadman_alert:
        deadman_entries = _parse_deadman_alert(deadman_alert)
        summary_parts = []
        deadman_reason = "Fleet deadman is signaling an alert."
        for kind, detail in deadman_entries:
            classified = _classify_deadman(kind, detail or "")
            summary_parts.append(f"{classified['title']}: {classified['reason']}")
            if not deadman_reason and classified.get("reason"):
                deadman_reason = classified["reason"]
        if summary_parts:
            deadman_reason = summary_parts[0]
        if summary_parts:
            if len(summary_parts) > 1:
                deadman_reason += " (+ multiple alerts)"
                deadman_reason += f" (+ {len(summary_parts)-1} additional)"
        return {
            "title": "Investigate DeadMan alert",
            "severity": "severe",
            "reason": deadman_reason,
        }

    operator_paused = (
        str(gate.get("ats_pause_source") or "").lower() == "operator"
        and bool(gate.get("paused") or gate.get("ats_paused"))
    )
    if operator_paused:
        return {
            "title": "Operator pause active",
            "severity": "info",
            "reason": "Fleet activity is intentionally suspended by the operator; no resume action is recommended.",
        }

    backend_issue_count = sum(
        1 for issue in browser.get("issues", []) if issue.get("issue") in _BROWSER_BACKEND_ISSUES
    )
    manual_issue_count = sum(
        1 for issue in browser.get("issues", []) if issue.get("issue") in _BROWSER_MANUAL_ISSUES
    )
    operator_limit_count = sum(
        1 for issue in browser.get("issues", []) if issue.get("issue") in _BROWSER_OPERATOR_LIMIT_ISSUES
    )
    if backend_issue_count > 0:
        details = _summarize_browser_issue_rows(browser.get("issues", []), _BROWSER_BACKEND_ISSUES)
        guidance = []
        for issue in browser.get("issues", []):
            issue_name = issue.get("issue")
            if issue_name in _BROWSER_BACKEND_ISSUES:
                hint = _BROWSER_ISSUE_GUIDANCE.get(issue_name)
                if hint and hint not in guidance:
                    guidance.append(hint)
        detail_text = "; ".join(details) if details else "multiple browser/backend faults"
        guidance_text = "; ".join(guidance) if guidance else "inspect browser health details"
        return {
            "title": "Fix browser/backend",
            "severity": "warn",
            "reason": (
                f"{backend_issue_count} worker(s) report browser/backend failures: {detail_text}. "
                f"Suggested actions: {guidance_text}."
            ),
        }
    if manual_issue_count > 0:
        details = _summarize_browser_issue_rows(browser.get("issues", []), _BROWSER_MANUAL_ISSUES)
        guidance = []
        for issue in browser.get("issues", []):
            issue_name = issue.get("issue")
            if issue_name in _BROWSER_MANUAL_ISSUES:
                hint = _BROWSER_ISSUE_GUIDANCE.get(issue_name)
                if hint and hint not in guidance:
                    guidance.append(hint)
        guidance_text = "; ".join(guidance) if guidance else "manual resolution workflow"
        detail_text = "; ".join(details) if details else "captcha/login/verification required"
        return {
            "title": "Resolve browser challenges",
            "severity": "warn",
            "reason": (
                f"{manual_issue_count} worker(s) report manual browser challenges: {detail_text}. "
                f"Suggested actions: {guidance_text}."
            ),
        }
    if operator_limit_count > 0:
        details = _summarize_browser_issue_rows(browser.get("issues", []), _BROWSER_OPERATOR_LIMIT_ISSUES)
        guidance = _browser_issue_guidance(browser.get("issues", []), _BROWSER_OPERATOR_LIMIT_ISSUES)
        detail_text = "; ".join(details) if details else "browser/model limit or missing terminal result"
        guidance_text = "; ".join(guidance) if guidance else "inspect worker log and route manually"
        return {
            "title": "Resolve browser/model limit",
            "severity": "warn",
            "reason": (
                f"{operator_limit_count} worker(s) report browser/model limits: {detail_text}. "
                f"Suggested actions: {guidance_text}."
            ),
        }

    if gate.get("paused"):
        return {
            "title": "Resume fleet",
            "severity": "severe",
            "reason": "Fleet is globally paused.",
        }
    if gate.get("ats_paused"):
        return {
            "title": "Resume ATS lane",
            "severity": "warn",
            "reason": "ATS lane paused; review ATS controls and resume when ready.",
        }
    if gate.get("canary_enabled") and (gate.get("canary_remaining") or 0) <= 0:
        return {
            "title": "Arm lift",
            "severity": "warn",
            "reason": "ATS canary exhausted; lift canary to continue bounded applying.",
        }
    if linkedin.get("halted") and linkedin.get("queued", 0) > 0:
        return {
            "title": "Clear LinkedIn halt",
            "severity": "warn",
            "reason": "LinkedIn lane has queued work but the account governor is halted.",
        }
    discovery_due = 0
    discovery_workers_alive = 0
    if discovery:
        discovery_due = int(discovery.get("tasks", {}).get("due_now") or 0)
        discovery_workers_alive = sum(1 for row in discovery.get("workers", []) if row.get("alive"))
    if discovery_due > 0 and discovery_workers_alive == 0 and queue_apply.get("queued", 0) == 0:
        return {
            "title": "Start discovery workers",
            "severity": "warn",
            "reason": f"{discovery_due} discovery task(s) are due, but no discovery worker is alive.",
        }
    if queue_apply.get("queued", 0) == 0 and apply_state.get("code") == "ready_to_apply":
        return {
            "title": "Inspect upstream queues",
            "severity": "warn",
            "reason": "Ready state is reporting no queued ATS jobs currently.",
        }
    if queue_apply.get("approved", 0) == 0:
        return {
            "title": "Seed approvals",
            "severity": "info",
            "reason": "No approved ATS jobs are queued yet.",
        }
    alive_workers = sum(1 for row in workers if row.get("alive"))
    if apply_state.get("code") == "ready_to_apply" and alive_workers == 0:
        return {
            "title": "Start apply workers",
            "severity": "severe",
            "reason": "Leaseable jobs exist, but no apply workers are heartbeating alive.",
        }
    if apply_state.get("code") == "ready_to_apply" and agents.get("blocked_workers", 0) > 0:
        return {
            "title": "Unblock agent routing",
            "severity": "warn",
            "reason": f"{agents.get('blocked_workers', 0)} dynamic worker(s) have all configured agents blocked.",
        }
    return {
        "title": "Monitor throughput",
        "severity": "info",
        "reason": "Fleet is eligible and should be applying if workers are healthy.",
    }


def _console_audit_summary(conn, limit: int = 10) -> dict:
    """READ-ONLY operator console action audit rows."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('fleet_console_audit') AS rel")
        if cur.fetchone()["rel"] is None:
            conn.rollback()
            return {"available": False, "rows": [], "summary": {"rows": 0}}
        cur.execute(
            "SELECT action, actor, lane, target, message, ok, created_at "
            "FROM fleet_console_audit ORDER BY created_at DESC, id DESC LIMIT %s",
            (int(limit),),
        )
        rows = cur.fetchall()
    conn.rollback()  # read-only
    out = []
    for row in rows:
        out.append({
            "action": _scrub(row.get("action") or "")[:80],
            "actor": _scrub(row.get("actor") or "")[:80],
            "lane": _scrub(row.get("lane") or "")[:40],
            "target": _scrub(row.get("target") or "")[:160],
            "message": _scrub(row.get("message") or "")[:300],
            "ok": bool(row.get("ok")),
            "created_at": _iso(row.get("created_at")),
        })
    return {
        "available": True,
        "rows": out,
        "summary": {
            "rows": len(out),
            "failed": sum(1 for row in out if not row["ok"]),
        },
    }


def _host_source_quality(conn, limit: int = 8) -> dict:
    """READ-ONLY host/source quality rollup for tuning noisy boards and sources."""
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH open_challenges AS (
                SELECT DISTINCT url FROM auth_challenge WHERE resolved_at IS NULL
            )
            SELECT
                COALESCE(NULLIF(q.target_host, ''), NULLIF(q.apply_domain, ''), 'unknown') AS host,
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE q.status='applied') AS applied,
                COUNT(*) FILTER (WHERE q.status IN ('failed','blocked','crash_unconfirmed')) AS failed,
                COUNT(*) FILTER (
                    WHERE q.apply_status='challenge_pending' OR oc.url IS NOT NULL
                ) AS challenges,
                COUNT(*) FILTER (WHERE q.status='queued') AS queued,
                COUNT(*) FILTER (
                    WHERE q.status='queued'
                      AND q.approved_batch IS NOT NULL
                      AND (q.dedup_key IS NULL OR a.dedup_key IS NULL)
                ) AS leaseable
            FROM apply_queue q
            LEFT JOIN applied_set a ON a.dedup_key = q.dedup_key
            LEFT JOIN open_challenges oc ON oc.url = q.url
            GROUP BY host
            ORDER BY challenges DESC, failed DESC, queued DESC, applied DESC, host ASC
            LIMIT %s
            """,
            (int(limit),),
        )
        rows = cur.fetchall()
    conn.rollback()  # read-only

    hosts: list[dict[str, Any]] = []
    for row in rows:
        total = int(row.get("total") or 0)
        failed = int(row.get("failed") or 0)
        challenges = int(row.get("challenges") or 0)
        hosts.append({
            "host": row.get("host") or "unknown",
            "total": total,
            "applied": int(row.get("applied") or 0),
            "failed": failed,
            "challenges": challenges,
            "queued": int(row.get("queued") or 0),
            "leaseable": int(row.get("leaseable") or 0),
            "failure_rate": round(failed / total, 4) if total else 0.0,
            "challenge_rate": round(challenges / total, 4) if total else 0.0,
        })
    return {
        "hosts": hosts,
        "summary": {
            "hosts": len(hosts),
            "challenge_hosts": sum(1 for row in hosts if row["challenges"] > 0),
            "failed": sum(row["failed"] for row in hosts),
            "challenges": sum(row["challenges"] for row in hosts),
            "queued": sum(row["queued"] for row in hosts),
            "leaseable": sum(row["leaseable"] for row in hosts),
        },
    }


def _worker_comparison(conn, workers: list[dict], limit: int = 12) -> dict:
    """READ-ONLY per-worker comparison for spotting bad workers or machines."""
    latest_usage = _latest_worker_llm_usage(conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                worker_id,
                COUNT(*) FILTER (WHERE status='applied') AS applied,
                COUNT(*) FILTER (WHERE status IN ('failed','blocked','crash_unconfirmed')) AS failed,
                COALESCE(SUM(est_cost_usd), 0) AS cost_usd,
                AVG(apply_duration_ms) FILTER (WHERE apply_duration_ms IS NOT NULL) AS avg_duration_ms,
                MAX(updated_at) AS last_terminal_at
            FROM apply_queue
            WHERE worker_id IS NOT NULL
            GROUP BY worker_id
            """
        )
        result_rows = cur.fetchall()
        cur.execute(
            """
            SELECT worker_id, COUNT(*) AS n
            FROM auth_challenge
            WHERE resolved_at IS NULL AND worker_id IS NOT NULL
            GROUP BY worker_id
            """
        )
        challenge_rows = cur.fetchall()
    conn.rollback()  # read-only

    metrics: dict[str, dict[str, Any]] = {}
    for row in result_rows:
        wid = row.get("worker_id")
        if not wid:
            continue
        avg_duration = row.get("avg_duration_ms")
        metrics[wid] = {
            "applied": int(row.get("applied") or 0),
            "failed": int(row.get("failed") or 0),
            "challenges": 0,
            "cost_usd": round(float(row.get("cost_usd") or 0), 4),
            "avg_duration_ms": int(round(float(avg_duration))) if avg_duration is not None else None,
            "last_terminal_at": _iso(row.get("last_terminal_at")),
        }
    for row in challenge_rows:
        wid = row.get("worker_id")
        if not wid:
            continue
        item = metrics.setdefault(
            wid,
            {
                "applied": 0,
                "failed": 0,
                "challenges": 0,
                "cost_usd": 0.0,
                "avg_duration_ms": None,
                "last_terminal_at": None,
            },
        )
        item["challenges"] = int(row.get("n") or 0)

    by_worker = {row.get("worker_id"): row for row in workers if row.get("worker_id")}
    worker_ids = set(by_worker) | set(metrics)
    rows: list[dict[str, Any]] = []
    for wid in worker_ids:
        heartbeat = by_worker.get(wid, {})
        usage = latest_usage.get(str(wid or ""))
        machine_owner = heartbeat.get("machine_owner") or (usage or {}).get("machine_owner")
        current_agent, current_model = _merge_agent_model(heartbeat, usage)
        stat = metrics.get(
            wid,
            {
                "applied": 0,
                "failed": 0,
                "challenges": 0,
                "cost_usd": 0.0,
                "avg_duration_ms": None,
                "last_terminal_at": None,
            },
        )
        rows.append({
            "worker_id": wid,
            "machine": (
                heartbeat.get("machine")
                or _historical_machine_label(machine_owner)
            ),
            "machine_owner": machine_owner,
            "alive": bool(heartbeat.get("alive")),
            "state": heartbeat.get("state"),
            "last_beat": heartbeat.get("last_beat"),
            "current_agent": current_agent,
            "current_model": current_model,
            **stat,
        })
    rows.sort(
        key=lambda row: (
            -int(row.get("failed") or 0),
            -int(row.get("challenges") or 0),
            -int(row.get("applied") or 0),
            str(row.get("worker_id") or ""),
        )
    )
    rows = rows[: int(limit)]
    return {
        "workers": rows,
        "summary": {
            "workers": len(rows),
            "applied": sum(int(row.get("applied") or 0) for row in rows),
            "failed": sum(int(row.get("failed") or 0) for row in rows),
            "challenges": sum(int(row.get("challenges") or 0) for row in rows),
            "cost_usd": round(sum(float(row.get("cost_usd") or 0) for row in rows), 4),
        },
    }


def _discovery(conn) -> dict:
    """Discovery-lane status (READ-ONLY, PG-only — never touches the SQLite brain):
    the search_tasks work-list, the discovered_postings staging area, and discovery-worker
    liveness (worker_heartbeat role='discovery', alive = beat within 150s)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) AS total, count(*) FILTER (WHERE enabled) AS enabled, "
            "       count(*) FILTER (WHERE enabled AND next_due_at IS NOT NULL "
            "                        AND next_due_at <= now()) AS due_now "
            "FROM search_tasks"
        )
        t = dict(cur.fetchone())
        cur.execute(
            "SELECT count(*) AS total, "
            "       count(*) FILTER (WHERE synced_to_home_at IS NULL) AS pending, "
            "       count(*) FILTER (WHERE discovered_at > now() - interval '24 hours') AS last24h "
            "FROM discovered_postings"
        )
        d = dict(cur.fetchone())
        cur.execute(
            "SELECT wh.worker_id, wh.machine_owner, wh.home_ip, wh.state, wh.last_beat, "
            "       (SELECT COUNT(*) FROM discovered_postings dp "
            "          WHERE dp.worker_id = wh.worker_id "
            "            AND dp.discovered_at > now() - interval '24 hours') AS found_24h "
            "FROM worker_heartbeat wh WHERE wh.role = 'discovery' ORDER BY wh.worker_id"
        )
        wrows = cur.fetchall()
        cur.execute(
            "SELECT dp.discovered_at, dp.source_label, st.query, st.board "
            "FROM discovered_postings dp LEFT JOIN search_tasks st ON st.task_id = dp.task_id "
            "ORDER BY dp.discovered_at DESC LIMIT 10"
        )
        rrows = cur.fetchall()
    conn.rollback()  # read-only
    workers = []
    for r in wrows:
        secs = _seconds_since(r["last_beat"])
        machine_owner = console_machines.infer_machine_owner(
            r["worker_id"], r.get("machine_owner")
        )
        workers.append({
            "worker_id": r["worker_id"],
            "machine_owner": machine_owner,
            "machine_display_name": console_machines.display_name(machine_owner),
            "home_ip": r.get("home_ip"),
            "alive": bool(secs is not None and secs <= _HB_ALIVE_SECONDS),
            "last_beat": _iso(r["last_beat"]),
            "seconds_since": secs,
            "state": r["state"],
            "found_24h": int(r["found_24h"] or 0),
        })
    recent: list[dict[str, Any]] = []
    recent_by_key: dict[tuple[str | None, str | None, str | None, str | None], dict[str, Any]] = {}
    for r in rrows:
        item = {
            "time": _iso(r["discovered_at"]),
            "source": r["source_label"],
            "query": r["query"],
            "board": r["board"],
            "count": 1,
        }
        key = (item["time"], item["source"], item["query"], item["board"])
        existing = recent_by_key.get(key)
        if existing is not None:
            existing["count"] += 1
            continue
        recent_by_key[key] = item
        recent.append(item)
    return {
        "tasks": {"total": int(t["total"] or 0), "enabled": int(t["enabled"] or 0),
                  "due_now": int(t["due_now"] or 0)},
        "postings": {"total": int(d["total"] or 0), "pending_ingest": int(d["pending"] or 0),
                     "last24h": int(d["last24h"] or 0)},
        "workers": workers,
        "recent": recent,
    }


def _challenge_backlog_summary(conn) -> dict:
    """READ-ONLY rollup separating actionable parks from stale challenge records."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT url, kind, raised_at FROM auth_challenge WHERE resolved_at IS NULL ORDER BY raised_at DESC"
        )
        rows = cur.fetchall()
    conn.rollback()  # read-only
    _, parked_by_url = _queue._challenge_rows(conn, None)
    by_kind: dict[str, int] = {}
    fresh_open = 0
    stale_open = 0
    stale_parked = 0
    stale_nonparked = 0
    missing_metadata_rows = 0
    park_only_rows = 0
    now = _utcnow()
    for row in rows:
        kind = str(row.get("kind") or "(unknown)")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        raised_at = row.get("raised_at")
        age_hours = _age_hours(raised_at, now)
        if age_hours is not None and age_hours > 24:
            stale_open += 1
            if str(row.get("url") or "") in parked_by_url:
                stale_parked += 1
            else:
                stale_nonparked += 1
        else:
            fresh_open += 1
    for _, parked in parked_by_url.values():
        if not ((parked or {}).get("company") and (parked or {}).get("title")):
            missing_metadata_rows += 1
    park_only_rows = sum(1 for url in parked_by_url if url not in {str(r.get("url") or "") for r in rows})
    parked_open = sum(1 for row in rows if str(row.get("url") or "") in parked_by_url)
    nonparked_open = len(rows) - parked_open
    return {
        "open_auth": len(rows),
        "parked_urls": len(parked_by_url),
        "parked_open": parked_open,
        "nonparked_open": nonparked_open,
        "fresh_open": fresh_open,
        "stale_open": stale_open,
        "stale_parked": stale_parked,
        "stale_nonparked": stale_nonparked,
        "missing_metadata_rows": missing_metadata_rows,
        "park_only_rows": park_only_rows,
        "by_kind": dict(sorted(by_kind.items())),
    }


def _diagnostics_backlog_summary(conn) -> dict:
    """READ-ONLY summary of active recommended diagnosis backlog for operator prioritization."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT reason, host, sample_count, diagnosis, recommendation "
            "FROM fleet_diagnoses "
            "WHERE status='recommended' AND (expires_at IS NULL OR expires_at > now())"
        )
        rows = cur.fetchall()
        cur.execute(
            "SELECT COALESCE(target_host, apply_domain, 'unknown') AS host, "
            "       COALESCE(NULLIF(apply_error,''), NULLIF(apply_status,''), status::text) AS reason, "
            "       COUNT(*) AS n "
            "FROM apply_queue WHERE lane='ats' AND status IN ('failed','crash_unconfirmed','blocked') "
            "GROUP BY 1,2"
        )
        current_failure_rows = cur.fetchall()
    conn.rollback()  # read-only
    current_other_by_host: dict[str, int] = {}
    for row in current_failure_rows:
        if canonical_failure_group(row.get("reason")) != "other":
            continue
        host = str(row.get("host") or "unknown")
        current_other_by_host[host] = current_other_by_host.get(host, 0) + int(row.get("n") or 0)

    remaining_other = dict(current_other_by_host)
    effective_rows = []
    for row in rows:
        item = dict(row)
        if item.get("reason") == "other":
            host = str(item.get("host") or "unknown")
            current = int(remaining_other.get(host) or 0)
            samples = min(int(item.get("sample_count") or 0), current)
            if samples <= 0:
                continue
            item["sample_count"] = samples
            remaining_other[host] = current - samples
        effective_rows.append(item)
    rows = effective_rows
    recommendations = _diagnostic_recommendations([{
        "diagnosis_id": None,
        "reason": r.get("reason"),
        "host": _scrub(r.get("host"))[:200],
        "machine": None,
        "lane": None,
        "sample_count": int(r.get("sample_count") or 0),
        "severity": "info",
        "diagnosis": _scrub(r.get("diagnosis"))[:300],
        "recommendation": _scrub(r.get("recommendation"))[:300],
        "created_at": None,
    } for r in rows])
    other_hosts: dict[str, int] = {}
    other_buckets: dict[str, int] = {}
    other_bucket_hosts: dict[str, dict[str, int]] = {}
    for r in rows:
        if r.get("reason") != "other":
            continue
        host = str(_scrub(r.get("host"))[:200] or "unknown")
        samples = int(r.get("sample_count") or 0)
        other_hosts[host] = other_hosts.get(host, 0) + samples
        bucket = _classify_unclassified_failure({
            "diagnosis": _scrub(r.get("diagnosis"))[:300],
            "recommendation": _scrub(r.get("recommendation"))[:300],
            "host": host,
        })
        other_buckets[bucket] = other_buckets.get(bucket, 0) + samples
        bucket_hosts = other_bucket_hosts.setdefault(bucket, {})
        bucket_hosts[host] = bucket_hosts.get(host, 0) + samples
    top_other_hosts = [
        {"host": host, "samples": samples}
        for host, samples in sorted(other_hosts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    ]
    top_other_buckets = [
        {"bucket": bucket, "samples": samples}
        for bucket, samples in sorted(other_buckets.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    ]
    top_other_bucket_hosts = {
        bucket: [
            {"host": host, "samples": samples}
            for host, samples in sorted(hosts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        ]
        for bucket, hosts in sorted(other_bucket_hosts.items())
    }
    return {
        "total": len(recommendations),
        "other_hosts": len({r["host"] for r in recommendations if r.get("reason") == "other"}),
        "other_samples": sum(int(r.get("sample_count") or 0) for r in recommendations if r.get("reason") == "other"),
        "dead_rows": sum(1 for r in recommendations if r.get("reason") == "dead"),
        "top_other_hosts": top_other_hosts,
        "other_buckets": dict(sorted(other_buckets.items())),
        "top_other_buckets": top_other_buckets,
        "top_other_bucket_hosts": top_other_bucket_hosts,
    }


def _host_risk_summary(source_quality: dict[str, Any]) -> dict[str, Any]:
    """Summarize leaseable ATS hosts that are burning attempts through high failure/challenge rates."""
    risky_hosts: list[dict[str, Any]] = []
    for row in source_quality.get("hosts", []):
        failed = int(row.get("failed") or 0)
        challenges = int(row.get("challenges") or 0)
        leaseable = int(row.get("leaseable") or 0)
        queued = int(row.get("queued") or 0)
        backlog = max(leaseable, queued)
        failure_rate = float(row.get("failure_rate") or 0.0)
        challenge_rate = float(row.get("challenge_rate") or 0.0)
        if backlog <= 0:
            continue
        if failed >= 3 and failure_rate >= 0.5:
            risky_hosts.append(dict(row))
            continue
        if challenges >= 3 and challenge_rate >= 0.1:
            risky_hosts.append(dict(row))
    risky_hosts.sort(
        key=lambda row: (
            -max(float(row.get("failure_rate") or 0.0), float(row.get("challenge_rate") or 0.0)),
            -int(row.get("leaseable") or 0),
            str(row.get("host") or ""),
        )
    )
    actions: list[dict[str, Any]] = []
    for row in risky_hosts[:5]:
        failure_rate = float(row.get("failure_rate") or 0.0)
        challenge_rate = float(row.get("challenge_rate") or 0.0)
        failed = int(row.get("failed") or 0)
        challenges = int(row.get("challenges") or 0)
        queued = int(row.get("queued") or 0)
        if failure_rate >= 0.5 and challenge_rate >= 0.1:
            action = "adapter_fix"
        elif failure_rate >= 0.5 and failed >= 3:
            action = "adapter_fix"
        elif challenge_rate >= 0.1 and challenges >= 3:
            action = "deprioritize"
        else:
            action = "watch"
        actions.append({
            "host": row.get("host") or "unknown",
            "action": action,
            "failure_rate": round(failure_rate, 4),
            "challenge_rate": round(challenge_rate, 4),
            "queued": queued,
            "leaseable": int(row.get("leaseable") or 0),
            "failed": failed,
            "challenges": challenges,
        })
    top = risky_hosts[0] if risky_hosts else None
    return {
        "hosts": len(risky_hosts),
        "top_host": top.get("host") if top else None,
        "top_failure_rate": float(top.get("failure_rate") or 0.0) if top else 0.0,
        "top_challenge_rate": float(top.get("challenge_rate") or 0.0) if top else 0.0,
        "top_leaseable": int(top.get("leaseable") or 0) if top else 0,
        "top_actions": actions,
    }

_UNCLASSIFIED_BUCKET_HINTS: dict[str, str] = {
    "expired_posting": "Run availability re-check for these rows and skip dead postings.",
    "access_denied": "Investigate anti-bot/auth barrier on the host; likely needs manual verification.",
    "broken_redirect": "Verify navigation/host-specific selectors; page target may have changed.",
    "parse_failure": "Inspect adapter parsing assumptions for this host and update selectors/workflow mapping.",
    "adapter_gap": "Host workflow drift detected; this likely needs ATS-specific adapter or target mapping update.",
    "unknown_other": "Still unclassified; open fresh worker logs and keep manual.",
}


def _classify_unclassified_failure(row: dict[str, Any]) -> str:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("diagnosis", "recommendation", "host")
    ).lower()
    if any(token in text for token in ("expired", "no longer available", "job closed", "posting removed", "404", "not found")):
        return "expired_posting"
    if any(token in text for token in ("access denied", "forbidden", "blocked", "captcha", "login required", "sign in", "unauthorized", "429", "rate limit")):
        return "access_denied"
    if any(token in text for token in ("redirect", "broken target", "target missing", "landed on search", "apply target missing", "wrong page")):
        return "broken_redirect"
    if any(token in text for token in ("parse", "extract", "selector", "dom", "field missing", "no result line", "could not read")):
        return "parse_failure"
    if any(token in text for token in ("adapter", "unsupported", "not implemented", "site-specific", "workflow mismatch", "integration gap")):
        return "adapter_gap"
    return "unknown_other"


def _unclassified_bucket_text(bucket: str) -> str:
    hint = _UNCLASSIFIED_BUCKET_HINTS.get(bucket)
    if hint:
        return f"Likely {bucket.replace('_', ' ')} pattern. {hint}"
    return (
        "Unrecognized failure token. Human review only -- the Doctor never auto-acts on an unclassified cluster."
    )


def _apply_unclassified_bucket(row: dict[str, Any], evidence: dict[str, Any] | None = None) -> None:
    host = str(row.get("host") or "unknown")
    n = int(row.get("sample_count") or 0)
    if n <= 0:
        n = 1
    source = evidence or row
    bucket = _classify_unclassified_failure({
        "diagnosis": str(source.get("diagnosis") or ""),
        "recommendation": str(source.get("recommendation") or ""),
        "host": host,
    })
    row["diagnosis"] = f"{n} unclassified failure(s) on {host}."
    row["recommendation"] = _unclassified_bucket_text(bucket)
    row["bucket"] = bucket


def _apply_state_blockers(
    apply_state: dict[str, Any],
    queue_apply: dict[str, Any],
    challenge_backlog: dict[str, Any],
    host_risk: dict[str, Any],
) -> dict[str, Any]:
    state = dict(apply_state)
    blockers: list[dict[str, Any]] = []

    if state.get("code") == "ready_jobs_not_leaseable":
        dedup_blocked = int(queue_apply.get("dedup_blocked") or 0)
        blockers.append({
            "code": "dedup_blocked",
            "title": "Approved rows are dedup-blocked",
            "count": dedup_blocked,
            "severity": "warn",
            "reason": (
                f"{dedup_blocked} approved ATS row(s) are blocked by dedup, so the queue looks ready "
                "but no new rows can be leased."
            ),
            "next_action": "Review dedup groups before expecting new leases.",
        })
    elif state.get("code") == "no_leaseable_jobs":
        stale_open = int(challenge_backlog.get("stale_open") or 0)
        stale_nonparked = int(challenge_backlog.get("stale_nonparked") or 0)
        stale_parked = int(challenge_backlog.get("stale_parked") or 0)
        fresh_open = int(challenge_backlog.get("fresh_open") or 0)
        open_auth = int(challenge_backlog.get("open_auth") or 0)
        missing_metadata_rows = int(challenge_backlog.get("missing_metadata_rows") or 0)
        unexpected_frozen = int(queue_apply.get("unexpected_frozen") or 0)
        approved = int(queue_apply.get("approved") or 0)
        queued = int(queue_apply.get("queued") or 0)
        risky_hosts = int(host_risk.get("hosts") or 0)
        top_host = host_risk.get("top_host") or "unknown"

        if stale_nonparked > 0:
            blockers.append({
                "code": "stale_challenge_records",
                "title": "Stale challenge records need cleanup",
                "count": stale_nonparked,
                "severity": "warn",
                "reason": (
                    f"{stale_nonparked} old auth record(s) have no corresponding parked queue row. "
                    "They are bookkeeping residue, not jobs waiting for authentication."
                ),
                "next_action": "Reconcile or close stale challenge records.",
            })
        if stale_parked > 0:
            blockers.append({
                "code": "aging_parked_challenges",
                "title": "Aging parked challenges need a decision",
                "count": stale_parked,
                "severity": "warn",
                "reason": (
                    f"{stale_parked} job(s) are still intentionally parked behind authentication "
                    "after more than 24 hours."
                ),
                "next_action": "Resolve or skip those parked jobs.",
            })
        if unexpected_frozen > 0:
            blockers.append({
                "code": "frozen_leases",
                "title": "Parked leases are still frozen",
                "count": unexpected_frozen,
                "severity": "warn",
                "reason": (
                    f"{unexpected_frozen} ATS row(s) are unexpectedly leased/frozen without an auth challenge, which keeps queue inventory "
                    "out of circulation."
                ),
                "next_action": "Release or reclaim parked leases.",
            })
        if approved <= 0 and queued > 0:
            blockers.append({
                "code": "approval_gap",
                "title": "Queued backlog has no approved ATS rows",
                "count": queued,
                "severity": "info",
                "reason": (
                    f"{queued} ATS row(s) are queued, but none are approved, so zero rows are eligible "
                    "for leasing."
                ),
                "next_action": "Approve ATS backlog before expecting new leases.",
            })
        if risky_hosts > 0:
            blockers.append({
                "code": "host_risk",
                "title": "Queued backlog is concentrated on hostile ATS hosts",
                "count": risky_hosts,
                "severity": "warn",
                "reason": (
                    f"{risky_hosts} queued host group(s) show elevated failure/challenge rates; top host "
                    f"{top_host} is a likely adapter/deprioritization candidate."
                ),
                "next_action": "Review hostile ATS hosts before spending more attempts there.",
            })
        if fresh_open > 0 and stale_open == 0:
            blockers.append({
                "code": "fresh_challenges",
                "title": "Fresh auth challenges are holding rows",
                "count": fresh_open,
                "severity": "warn",
                "reason": (
                    f"{fresh_open} fresh auth challenge(s) remain open; those rows need manual resolution "
                    "before they can flow again."
                ),
                "next_action": "Resolve fresh challenges worth saving.",
            })
        elif open_auth > 0 and stale_open > 0:
            blockers.append({
                "code": "open_challenges",
                "title": "Open auth challenges still dominate the queue",
                "count": open_auth,
                "severity": "warn",
                "reason": (
                    f"{open_auth} total auth challenge record(s) are open across parked jobs and "
                    "nonparked bookkeeping records."
                ),
                "next_action": "Resolve parked jobs and reconcile nonparked records separately.",
            })
        if missing_metadata_rows > 0:
            blockers.append({
                "code": "challenge_metadata_gap",
                "title": "Challenge metadata is missing on parked rows",
                "count": missing_metadata_rows,
                "severity": "info",
                "reason": (
                    f"{missing_metadata_rows} parked challenge row(s) are missing company/title metadata, "
                    "which slows operator triage."
                ),
                "next_action": "Backfill challenge metadata to speed manual triage.",
            })

    if blockers:
        severity_rank = {"severe": 0, "warn": 1, "info": 2}
        blocker_rank = {
            "stale_challenge_records": 0,
            "aging_parked_challenges": 1,
            "frozen_leases": 2,
            "approval_gap": 2,
            "host_risk": 3,
            "fresh_challenges": 4,
            "open_challenges": 5,
            "challenge_metadata_gap": 6,
            "dedup_blocked": 7,
        }
        blockers.sort(
            key=lambda item: (
                severity_rank.get(str(item.get("severity") or "info"), 9),
                blocker_rank.get(str(item.get("code") or ""), 99),
                -int(item.get("count") or 0),
                str(item.get("code") or ""),
            )
        )
        top = blockers[0]
        state["reason"] = str(top.get("reason") or state.get("reason") or "")
        state["next_action"] = str(top.get("next_action") or "")
    state["blockers"] = blockers
    return state


def _diagnostic_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = [dict(r) for r in rows if (r.get("reason") or "") != "dead"]
    aggregated: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    ordered: list[tuple[str, str, str | None]] = []
    for row in filtered:
        reason = str(row.get("reason") or "")
        if reason != "other":
            key = (str(row.get("diagnosis_id") or row.get("diagnosis") or ""), reason, row.get("lane"))
            aggregated[key] = row
            ordered.append(key)
            continue
        key = (reason, str(row.get("host") or ""), row.get("lane"))
        existing = aggregated.get(key)
        if existing is None:
            aggregated[key] = row
            _apply_unclassified_bucket(aggregated[key])
            ordered.append(key)
            continue
        existing["sample_count"] = int(existing.get("sample_count") or 0) + int(row.get("sample_count") or 0)
        if str(row.get("created_at") or "") > str(existing.get("created_at") or ""):
            existing["diagnosis_id"] = row.get("diagnosis_id")
            existing["created_at"] = row.get("created_at")
        existing["machine"] = None
        _apply_unclassified_bucket(existing, row)
    return [aggregated[key] for key in ordered]


def _diagnostics(conn) -> dict:
    """READ-ONLY Fleet Doctor surface (its own endpoint, NOT in the 4s /api/status poll):
      ``clusters``        -- live failure clusters over the rolling window (analyze())
      ``auto_fixes``      -- ACTIVE fleet_knobs joined to their recent auto_applied diagnoses
                             (type, scope, reason, expires, knob_id, diagnosis_id, how_to_reverse)
      ``recommendations`` -- status='recommended' fleet_diagnoses rows (human queue)

    All free text is re-scrubbed on read (defense in depth -- S1) and sizes are capped. The
    Doctor already scrubbed before storing; we scrub AGAIN here. LinkedIn never appears: the
    Doctor only ever clusters / acts on the apply (ats) lane."""
    from applypilot.fleet import doctor as _doctor
    try:
        clustered = _doctor.analyze(conn, window_minutes=60)
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        clustered = {"cluster_rows": []}
    clusters = []
    for c in (clustered.get("cluster_rows") or [])[:50]:
        clusters.append({
            "reason": c.get("reason"),
            "host": _scrub(c.get("host"))[:200],
            "machine": _machine_display_name(_scrub(c.get("machine"))[:120]),
            "lane": c.get("lane"),
            "sample_count": int(c.get("sample_count") or 0),
        })
    with conn.cursor() as cur:
        # Active knobs + the most recent auto-applied diagnosis for the same scope (for the
        # Reverse button + how_to_reverse text). LEFT JOIN LATERAL keeps it one row per knob.
        # H18: the join is CONSTRAINED TO THE KNOB'S SCOPE (not just the auto_action type prefix),
        # so with two active host_skips each knob shows ITS OWN diagnosis -- the old type-prefix-only
        # match showed the most-recent diagnosis text for ALL knobs of a type (operator could Reverse
        # on misleading evidence). Per type: host_skip -> d2.host=scope; pace -> d2.host=substr(scope,6);
        # timeout_bump/pause -> d2.lane=scope; quarantine -> d2.host=knob host (url has no diag host,
        # fall back to type-only via the OR).
        cur.execute(
            "SELECT k.id AS knob_id, k.knob_type, k.scope_key, k.value_text, k.reason, "
            "       k.created_at, k.expires_at, d.id AS diagnosis_id, d.how_to_reverse, d.diagnosis "
            "FROM fleet_knobs k "
            "LEFT JOIN LATERAL ("
            "   SELECT id, how_to_reverse, diagnosis FROM fleet_diagnoses d2 "
            "   WHERE d2.auto_action LIKE k.knob_type || '%' AND d2.status='auto_applied' "
            "     AND ( "
            "       (k.knob_type='host_skip'    AND d2.host = "
            "           CASE WHEN k.scope_key LIKE 'host:%' THEN substr(k.scope_key, 6) ELSE k.scope_key END) "
            "       OR (k.knob_type='pace_or_pause' AND k.scope_key LIKE 'host:%' "
            "           AND d2.host = substr(k.scope_key, 6)) "
            "       OR (k.knob_type='pace_or_pause' AND k.scope_key='ats' AND d2.lane = k.scope_key) "
            "       OR (k.knob_type='timeout_bump' AND d2.lane = k.scope_key) "
            "       OR (k.knob_type='quarantine') "
            "     ) "
            "   ORDER BY d2.created_at DESC LIMIT 1"
            ") d ON TRUE "
            "WHERE k.active AND (k.expires_at IS NULL OR k.expires_at > now()) "
            "ORDER BY k.created_at DESC LIMIT 100"
        )
        krows = cur.fetchall()
        cur.execute(
            "SELECT id, cluster_key, reason, host, machine, lane, sample_count, severity, "
            "       diagnosis, recommendation, created_at "
            "FROM fleet_diagnoses WHERE status='recommended' "
            "AND (expires_at IS NULL OR expires_at > now()) "
            "ORDER BY created_at DESC LIMIT 200"
        )
        rrows = cur.fetchall()
        # H11/H15: quarantined URLs surfaced INDEPENDENT of fleet_knobs, so a still-quarantined url
        # whose audit knob already expired (the 7-day knob TTL only expires the audit row, not the
        # permanent quarantine) is still visible + has a reversible path. manual: reasons only.
        cur.execute(
            "SELECT url, reason, quarantined_at FROM poison_jobs "
            "WHERE quarantined_at IS NOT NULL AND reason LIKE 'manual:%' "
            "ORDER BY quarantined_at DESC LIMIT 100"
        )
        qrows = cur.fetchall()
    conn.rollback()  # read-only
    browser_health = _browser_health(conn)
    auto_fixes = [{
        "knob_id": r["knob_id"],
        "diagnosis_id": r["diagnosis_id"],
        "knob_type": r["knob_type"],
        "scope_key": _scrub(r.get("scope_key"))[:300],
        "value_text": _scrub(r.get("value_text"))[:200],
        "reason": _scrub(r.get("reason"))[:200],
        "created_at": _iso(r.get("created_at")),
        "expires_at": _iso(r.get("expires_at")),
        "how_to_reverse": _scrub(r.get("how_to_reverse"))[:400],
    } for r in krows]
    recommendations = [{
        "diagnosis_id": r["id"],
        "reason": r.get("reason"),
        "host": _scrub(r.get("host"))[:200],
        "machine": _machine_display_name(_scrub(r.get("machine"))[:120]),
        "lane": r.get("lane"),
        "sample_count": int(r.get("sample_count") or 0),
        "severity": r.get("severity"),
        "diagnosis": _scrub(r.get("diagnosis"))[:600],
        "recommendation": _scrub(r.get("recommendation"))[:600],
        "created_at": _iso(r.get("created_at")),
    } for r in rrows]
    recommendations = _diagnostic_recommendations(recommendations)
    recommendations_summary = {
        "total": len(recommendations),
        "other_hosts": len({r["host"] for r in recommendations if r.get("reason") == "other"}),
        "other_samples": sum(int(r.get("sample_count") or 0) for r in recommendations if r.get("reason") == "other"),
    }
    quarantined = [{
        "url": _scrub(r.get("url"))[:300],
        "reason": _scrub(r.get("reason"))[:200],
        "quarantined_at": _iso(r.get("quarantined_at")),
    } for r in qrows]
    return {"clusters": clusters, "auto_fixes": auto_fixes,
            "recommendations": recommendations, "recommendations_summary": recommendations_summary,
            "quarantined": quarantined,
            "browser_health": browser_health}


def _doctor_signal(conn) -> dict:
    """H18: a SMALL, above-the-fold Doctor signal folded into the fast /api/status poll (NOT the
    heavy diagnostics blob): last_pass_at, the active auto-fix count, and the newest active
    auto-fix timestamp/id so the frontend can show a Doctor card + fire a toast on a NEW auto-fix.
    Also surfaces ats_paused provenance (H8) so the banner can label a Doctor pause vs operator."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT doctor_last_pass_at, ats_paused, ats_pause_source, "
            "deadman_alert, deadman_alert_at FROM fleet_config WHERE id=1")
        cfg = cur.fetchone() or {}
        cur.execute(
            "SELECT count(*) AS n, max(created_at) AS newest, max(id) AS newest_id "
            "FROM fleet_knobs WHERE active AND (expires_at IS NULL OR expires_at > now())")
        k = cur.fetchone() or {}
    conn.rollback()  # read-only
    return {
        "last_pass_at": _iso(cfg.get("doctor_last_pass_at")),
        "active_auto_fix_count": int(k.get("n") or 0),
        "newest_auto_action_at": _iso(k.get("newest")),
        "newest_auto_fix_id": k.get("newest_id"),
        "ats_paused": bool(cfg.get("ats_paused")),
        "ats_pause_source": cfg.get("ats_pause_source"),
        "deadman_alert": cfg.get("deadman_alert"),
        "deadman_alert_at": _iso(cfg.get("deadman_alert_at")),
    }


def diagnostics() -> dict:
    """Open a short-lived connection, build the Doctor surface, always close."""
    conn = pgqueue.connect()
    try:
        return _diagnostics(conn)
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.close()


def diagnosis() -> dict:
    """Build an operator-facing /api/diagnosis payload (why-not-applying + safety rails +
    machine rollups + browser summary + lightweight recommendations)."""
    conn = pgqueue.connect()
    try:
        gate, queue, apply_state = _gate_and_queue(conn)
        workers = _workers(conn)
        queue_apply = queue.get("apply", {})
        browser = _browser_health(conn)
        desired_workers_by_machine = _desired_machine_workers(conn)
        network = _network_rollup(conn)
        linkedin = _linkedin_summary(conn)
        deadman = _deadman_status(conn)
        compute = _compute_summary(conn)
        forecast = _throughput_forecast(conn, gate, workers)
        daily_goals = _daily_goals(conn, gate)
        freshness = _data_freshness(conn)
        audit = _console_audit_summary(conn)
        challenge_backlog = _challenge_backlog_summary(conn)
        diagnostics_summary = _diagnostics_backlog_summary(conn)
        source_quality = _host_source_quality(conn)
        host_risk = _host_risk_summary(source_quality)
        apply_state = _apply_state_blockers(apply_state, queue_apply, challenge_backlog, host_risk)
        worker_comparison = _worker_comparison(conn, workers)
        try:
            agents = _agent_routing_summary(conn, workers)
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            agents = {
                "workers": len(workers),
                "dynamic_workers": 0,
                "switched_workers": 0,
                "partial_workers": 0,
                "blocked_workers": 0,
                "active_agent_blocks": [],
                "model_usage": {},
                "model_family_usage": {},
                "model_count": 0,
            }
        try:
            discovery = _discovery(conn)
            for worker in discovery.get("workers", []):
                worker["machine"] = _machine_display_name(worker.get("machine_owner"))
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            discovery = None

        machines: dict[str, dict[str, int]] = {}
        for row in workers:
            m = row.get("machine", "unknown")
            item = machines.setdefault(m, {"workers": 0, "alive": 0, "idle": 0, "working": 0})
            item["workers"] += 1
            if row.get("alive"):
                item["alive"] += 1
            if row.get("alive") and row.get("state") == "applying":
                item["working"] += 1
            elif row.get("alive"):
                item["idle"] += 1
            if row.get("applied", 0) > 0:
                item["applied_workers"] = item.get("applied_workers", 0) + 1
        for name, desired_count in desired_workers_by_machine.items():
            item = machines.setdefault(name, {"workers": 0, "alive": 0, "idle": 0, "working": 0})
            item["desired"] = desired_count
            item["missing"] = max(desired_count - int(item.get("alive", 0)), 0)
        for name, data in network.get("machines", {}).items():
            item = machines.setdefault(name, {"workers": 0, "alive": 0, "idle": 0, "working": 0})
            item["fleet_workers"] = int(data.get("workers", 0))
            item["fleet_alive"] = int(data.get("alive", 0))
            item["fleet_stale"] = int(data.get("stale", 0))
        if desired_workers_by_machine:
            for name, item in machines.items():
                item.setdefault("desired", desired_workers_by_machine.get(name, 0))
                item.setdefault("missing", max(int(item.get("desired", 0)) - int(item.get("alive", 0)), 0))
        desired_total = sum(desired_workers_by_machine.values())
        alive_total = sum(int(item.get("alive", 0)) for item in machines.values())
        missing_total = sum(int(item.get("missing", 0)) for item in machines.values())
        queue_funnel = {
            "ats": queue_apply.get("by_lane", {}).get("ats", {}),
            "compute": queue_apply.get("by_lane", {}).get("compute", {}),
            "other": queue_apply.get("by_lane", {}).get("other", {}),
        }

        recommendations: list[dict[str, str]] = []
        recommendation_titles: set[str] = set()
        def add_recommendation(title: str, severity: str, reason: str) -> None:
            key = f"{title}|{reason}"
            if key in recommendation_titles:
                return
            recommendation_titles.add(key)
            recommendations.append({
                "title": title,
                "severity": severity,
                "reason": reason,
            })
        alive_workers = sum(1 for row in workers if row.get("alive"))
        browser_issues = browser.get("issues") or []
        backend_problem_workers = sum(
            1 for issue in browser_issues
            if issue.get("issue") in _BROWSER_BACKEND_ISSUES
        )
        manual_problem_workers = sum(
            1 for issue in browser_issues
            if issue.get("issue") in _BROWSER_MANUAL_ISSUES
        )
        operator_limit_workers = sum(
            1 for issue in browser_issues
            if issue.get("issue") in _BROWSER_OPERATOR_LIMIT_ISSUES
        )
        discovery_workers_alive = 0
        discovery_due = 0
        if discovery:
            discovery_due = int(discovery.get("tasks", {}).get("due_now") or 0)
            discovery_workers_alive = sum(1 for row in discovery.get("workers", []) if row.get("alive"))
        compute_queue = compute.get("queue", {}) if compute else {}
        compute_queued = int(compute_queue.get("queued", {}).get("count") or 0)
        compute_workers_alive = int(compute.get("score_workers", {}).get("active") or 0) if compute else 0
        compute_recent_failed = int(compute.get("score_workers", {}).get("recent_failed_15m") or 0) if compute else 0
        if deadman.get("active"):
            for deadman_alert in deadman.get("alerts") or []:
                add_recommendation(
                    "Investigate DeadMan alert",
                    deadman_alert.get("severity", "severe"),
                    (
                        deadman_alert.get("reason")
                        or deadman.get("reason")
                        or deadman.get("alert")
                        or "Fleet watchdog has raised an alert."
                    ),
                )
        operator_paused = (
            str(gate.get("ats_pause_source") or "").lower() == "operator"
            and bool(gate.get("paused") or gate.get("ats_paused"))
        )
        if operator_paused:
            add_recommendation(
                "Operator pause active", "info",
                "Fleet activity is intentionally suspended by the operator; no resume action is recommended.",
            )
        elif gate.get("paused"):
            add_recommendation("Resume fleet", "severe", "Fleet is globally paused.")
        if gate.get("ats_paused") and not operator_paused:
            add_recommendation("Resume ATS lane", "warn", "ATS lane paused; review ATS controls and resume when ready.")
        if gate.get("canary_enabled") and (gate.get("canary_remaining") or 0) <= 0:
            add_recommendation("Arm lift", "warn", "ATS canary exhausted; lift canary to continue bounded applying.")
        if linkedin.get("halted") and linkedin.get("queued", 0) > 0:
            add_recommendation("Clear LinkedIn halt", "warn", "LinkedIn lane has queued work but the account governor is halted.")
        if discovery_due > 0 and discovery_workers_alive == 0 and queue_apply.get("queued", 0) == 0:
            add_recommendation("Start discovery workers", "warn", f"{discovery_due} discovery task(s) are due, but no discovery worker is alive.")
        if compute_queued > 0 and compute_workers_alive == 0:
            add_recommendation("Start compute workers", "warn", f"{compute_queued} score task(s) are queued, but no compute worker is alive.")
        if compute_recent_failed > 0:
            add_recommendation("Review compute failures", "warn", f"{compute_recent_failed} score task(s) failed in the last 15 minutes.")
        if queue_apply.get("queued", 0) == 0 and apply_state.get("code") == "ready_to_apply":
            add_recommendation("Inspect upstream queues", "warn", "Ready state is reporting no queued ATS jobs currently.")
        if apply_state.get("code") == "no_leaseable_jobs" and (apply_state.get("blockers") or []):
            for blocker in apply_state.get("blockers") or []:
                blocker_code = str((blocker or {}).get("code") or "")
                blocker_reason = str((blocker or {}).get("reason") or apply_state.get("reason") or "No leaseable ATS jobs are available.")
                if blocker_code == "stale_challenge_records":
                    add_recommendation("Clean stale challenge records", "warn", blocker_reason)
                elif blocker_code == "aging_parked_challenges":
                    add_recommendation("Resolve aging parked challenges", "warn", blocker_reason)
                elif blocker_code == "frozen_leases":
                    add_recommendation("Release parked leases", "warn", blocker_reason)
                elif blocker_code == "approval_gap":
                    add_recommendation("Approve ATS backlog", "info", blocker_reason)
                elif blocker_code == "host_risk":
                    add_recommendation("Review hostile ATS hosts", "warn", blocker_reason)
                elif blocker_code in {"fresh_challenges", "open_challenges"}:
                    add_recommendation("Resolve browser challenges", "warn", blocker_reason)
                elif blocker_code == "challenge_metadata_gap":
                    add_recommendation("Backfill challenge metadata", "info", blocker_reason)
        if queue_apply.get("approved", 0) == 0 and not operator_paused:
            add_recommendation("Seed approvals", "info", "No approved ATS jobs are queued yet.")
        if apply_state.get("code") == "ready_to_apply" and alive_workers == 0:
            add_recommendation("Start apply workers", "severe", "Leaseable jobs exist, but no apply workers are heartbeating alive.")
        if apply_state.get("code") == "ready_to_apply" and backend_problem_workers > 0:
            browser_issues = browser.get("issues") or []
            details = _summarize_browser_issue_rows(browser_issues, _BROWSER_BACKEND_ISSUES)
            guidance = _browser_issue_guidance(browser_issues, _BROWSER_BACKEND_ISSUES)
            detail_suffix = f" {', '.join(details)}" if details else ""
            guidance_text = "; ".join(guidance) if guidance else "inspect browser health details"
            add_recommendation(
                "Fix browser/backend",
                "warn",
                f"{backend_problem_workers} worker(s) report browser/backend failures; {detail_suffix or 'inspect browser logs'}."
                f" Suggested actions: {guidance_text}.",
            )
            if (
                apply_state.get("code") == "ready_to_apply"
                and int(queue_apply.get("unexpected_frozen") or 0) > 0
                and int(queue_apply.get("active_leased") or 0) == 0
            ):
                add_recommendation(
                    "Release parked leases",
                    "warn",
                    f"{int(queue_apply.get('unexpected_frozen') or 0)} ATS row(s) are unexpectedly frozen without an auth challenge while zero leased rows look actively in progress.",
                )
        if apply_state.get("code") == "ready_to_apply" and manual_problem_workers > 0:
            browser_issues = browser.get("issues") or []
            details = _summarize_browser_issue_rows(browser_issues, _BROWSER_MANUAL_ISSUES)
            guidance = _browser_issue_guidance(browser_issues, _BROWSER_MANUAL_ISSUES)
            detail_suffix = f" {', '.join(details)}" if details else ""
            guidance_text = "; ".join(guidance) if guidance else "manual challenge triage"
            add_recommendation(
                "Resolve browser challenges",
                "warn",
                f"{manual_problem_workers} worker(s) report CAPTCHA/login/verification; "
                f"{detail_suffix or 'manual intervention is required'}."
                f" Suggested actions: {guidance_text}.",
            )
            if apply_state.get("code") == "ready_to_apply" and agents.get("blocked_workers", 0) > 0:
                add_recommendation("Unblock agent routing", "warn", f"{agents.get('blocked_workers', 0)} dynamic worker(s) have all configured agents blocked.")

        if apply_state.get("code") == "ready_to_apply" and operator_limit_workers > 0:
            browser_issues = browser.get("issues") or []
            details = _summarize_browser_issue_rows(browser_issues, _BROWSER_OPERATOR_LIMIT_ISSUES)
            guidance = _browser_issue_guidance(browser_issues, _BROWSER_OPERATOR_LIMIT_ISSUES)
            detail_suffix = f" {', '.join(details)}" if details else ""
            guidance_text = "; ".join(guidance) if guidance else "inspect worker log and route manually"
            add_recommendation(
                "Resolve browser/model limit",
                "warn",
                f"{operator_limit_workers} worker(s) report browser/model limits; "
                f"{detail_suffix or 'quota/cap/no-result intervention is required'}."
                f" Suggested actions: {guidance_text}.",
            )

        if int(queue_apply.get("unexpected_frozen") or 0) > 0 and int(queue_apply.get("active_leased") or 0) == 0:
            add_recommendation(
                "Release parked leases",
                "warn",
                f"{int(queue_apply.get('unexpected_frozen') or 0)} ATS row(s) are unexpectedly frozen without an auth challenge while zero leased rows look actively in progress.",
            )
        if apply_state.get("code") == "ready_to_apply" and missing_total > 0:
            add_recommendation("Restore missing workers", "warn", f"{missing_total} desired apply worker(s) are not heartbeating alive.")
        if not recommendations:
            add_recommendation("Monitor throughput", "info", "Fleet is eligible and should be applying if workers are healthy.")
        if challenge_backlog.get("stale_nonparked", 0) > 0:
            add_recommendation(
                "Clean stale challenge records",
                "warn",
                f"{int(challenge_backlog.get('stale_nonparked') or 0)} old open auth record(s) no longer have a parked queue row and should be reconciled or closed.",
            )
        if challenge_backlog.get("stale_parked", 0) > 0:
            add_recommendation(
                "Resolve aging parked challenges",
                "warn",
                f"{int(challenge_backlog.get('stale_parked') or 0)} job(s) remain deliberately parked behind authentication for more than 24 hours.",
            )
        if challenge_backlog.get("missing_metadata_rows", 0) > 0:
            add_recommendation(
                "Backfill challenge metadata",
                "info",
                f"{int(challenge_backlog.get('missing_metadata_rows') or 0)} parked challenge row(s) are missing company/title metadata; triage will be slower until those rows can be identified.",
            )
        if agents.get("model_missing_workers", 0) > 0:
            add_recommendation(
                "Backfill apply model telemetry",
                "info",
                f"{int(agents.get('model_missing_workers') or 0)} apply worker(s) report an agent but no model name; operator comparisons and model-level spend views are incomplete until that telemetry is filled.",
            )
        if diagnostics_summary.get("other_hosts", 0) > 0:
            top_hosts = diagnostics_summary.get("top_other_hosts") or []
            top_buckets = diagnostics_summary.get("top_other_buckets") or []
            top_hosts_text = ", ".join(
                f"{str(item.get('host') or 'unknown')} ({int(item.get('samples') or 0)})"
                for item in top_hosts[:3]
            )
            top_buckets_text = ", ".join(
                f"{str(item.get('bucket') or 'unknown')} ({int(item.get('samples') or 0)})"
                for item in top_buckets[:3]
            )
            add_recommendation(
                "Review unclassified ATS failures",
                "info",
                f"{int(diagnostics_summary.get('other_hosts') or 0)} host group(s) and {int(diagnostics_summary.get('other_samples') or 0)} sample(s) remain unclassified."
                + (f" Likely buckets: {top_buckets_text}." if top_buckets_text else "")
                + (f" Top hosts: {top_hosts_text}." if top_hosts_text else ""),
            )
        if host_risk.get("hosts", 0) > 0:
            top_actions = host_risk.get("top_actions") or []
            top_actions_text = ", ".join(
                f"{str(item.get('host') or 'unknown')} ({str(item.get('action') or 'watch')})"
                for item in top_actions[:3]
            )
            add_recommendation(
                "Review hostile ATS hosts",
                "warn",
                f"{int(host_risk.get('hosts') or 0)} host(s) with queued backlog show elevated failure/challenge rates; "
                f"top offender {host_risk.get('top_host') or 'unknown'} "
                f"(failure {float(host_risk.get('top_failure_rate') or 0.0):.0%}, "
                f"challenge {float(host_risk.get('top_challenge_rate') or 0.0):.0%}, "
                f"leaseable {int(host_risk.get('top_leaseable') or 0)})."
                + (f" Suggested actions: {top_actions_text}." if top_actions_text else ""),
            )

        return {
            "apply_state": apply_state,
            "queue": {"apply": queue_apply},
            "safety": {
                "paused": bool(gate.get("paused")),
                "ats_paused": bool(gate.get("ats_paused")),
                "spend_cap_usd": gate.get("spend_cap_usd"),
                "spent_usd": gate.get("spent_usd"),
                "canary_enabled": bool(gate.get("canary_enabled")),
                "canary_remaining": gate.get("canary_remaining"),
            },
            "browser": {
                "summary": browser.get("summary", {}),
                "counts": browser.get("summary", {}).get("by_issue", {}),
            },
            "linkedin": linkedin,
            "discovery": discovery,
            "compute": compute,
            "forecast": forecast,
            "challenges": challenge_backlog,
            "diagnostics_summary": diagnostics_summary,
            "daily_goals": daily_goals,
            "freshness": freshness,
            "audit": audit,
            "source_quality": source_quality,
            "worker_comparison": worker_comparison,
            "deadman": deadman,
            "agents": agents,
            "rollups": {
                "machines": {k: dict(v) for k, v in machines.items()},
                "fleet": {
                    "desired_configured": bool(desired_workers_by_machine),
                    "desired_workers": desired_total,
                    "alive_workers": alive_total,
                    "missing_workers": missing_total,
                },
                "network": network,
                "funnel": queue_funnel,
            },
            "recommendations": recommendations,
        }
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.close()


def _outcomes(conn, limit: int = 25) -> list[dict]:
    """READ-ONLY recent inbox_outcomes (newest first). Parameterized LIMIT only (S5);
    free text re-scrubbed + capped (S1/S3); timestamps ISO-serialized; conn.rollback()
    marks it read-only."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT message_id, company, title, stage, outcome, sender_domain, occurred_at "
            "FROM inbox_outcomes ORDER BY occurred_at DESC NULLS LAST LIMIT %s",
            (int(limit),),
        )
        rows = cur.fetchall()
    conn.rollback()  # read-only
    return [{
        "message_id": r.get("message_id"),
        "time": _iso(r.get("occurred_at")),
        "company": _scrub(r.get("company") or "")[:200],
        "title": _scrub(r.get("title") or "")[:200],
        "stage": r.get("stage"),
        "outcome": r.get("outcome"),
        "sender_domain": r.get("sender_domain"),
    } for r in rows]


def outcomes() -> dict:
    """Open a short-lived connection, read recent outcomes, always close."""
    conn = pgqueue.connect()
    try:
        return {"outcomes": _outcomes(conn)}
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.close()


def agents() -> dict:
    """Build /api/agents payload."""
    try:
        return _agents_state()
    except Exception as e:
        raise RuntimeError(f"agent telemetry read failed: {e}") from e


def _liveness_summary(conn) -> dict:
    """Return queue liveness totals plus actionable uncertain-reason groups."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              COUNT(*) FILTER (WHERE liveness_status='live') AS live,
              COUNT(*) FILTER (WHERE liveness_status='dead') AS dead,
              COUNT(*) FILTER (WHERE liveness_status='uncertain') AS uncertain,
              COUNT(*) FILTER (WHERE liveness_status IS NULL) AS unchecked
            FROM apply_queue
            WHERE lane='ats'
            """
        )
        totals = dict(cur.fetchone())
        cur.execute(
            """
            SELECT COALESCE(NULLIF(BTRIM(liveness_reason),''),'missing_reason') AS reason,
                   COUNT(*) AS rows,
                   COUNT(DISTINCT COALESCE(NULLIF(target_host,''), NULLIF(apply_domain,''))) AS hosts,
                   MAX(liveness_checked_at) AS latest_checked_at,
                   MAX(COALESCE(liveness_consecutive_uncertain,0)) AS max_consecutive_uncertain
            FROM apply_queue
            WHERE lane='ats' AND liveness_status='uncertain'
            GROUP BY 1
            ORDER BY rows DESC, reason
            LIMIT 20
            """
        )
        uncertain_reasons = [
            {
                "reason": row["reason"],
                "rows": int(row["rows"] or 0),
                "hosts": int(row["hosts"] or 0),
                "latest_checked_at": _iso(row["latest_checked_at"]),
                "max_consecutive_uncertain": int(row["max_consecutive_uncertain"] or 0),
                "retry_class": _queue.liveness_retry_policy(
                    row["reason"], int(row["max_consecutive_uncertain"] or 0)
                )[0],
                "retry_after_seconds": _queue.liveness_retry_policy(
                    row["reason"], int(row["max_consecutive_uncertain"] or 0)
                )[1],
            }
            for row in cur.fetchall()
        ]
        cur.execute(
            """
            WITH latest AS (
                SELECT DISTINCT ON (COALESCE(NULLIF(target_host,''), NULLIF(apply_domain,'')))
                       COALESCE(NULLIF(target_host,''), NULLIF(apply_domain,'')) AS host,
                       liveness_reason AS reason,
                       liveness_checked_at AS checked_at
                FROM apply_queue
                WHERE lane='ats'
                  AND liveness_status='uncertain'
                  AND liveness_checked_at >=
                      now() - make_interval(secs => %s)
                  AND (
                      COALESCE(liveness_reason,'') LIKE 'server_%%'
                      OR COALESCE(liveness_reason,'') LIKE 'neterr:%%'
                      OR COALESCE(liveness_reason,'') LIKE 'error:%%'
                      OR COALESCE(liveness_reason,'') = 'blocked_429'
                      OR COALESCE(liveness_reason,'') LIKE 'access_gate:%%'
                  )
                  AND COALESCE(NULLIF(target_host,''), NULLIF(apply_domain,'')) IS NOT NULL
                ORDER BY COALESCE(NULLIF(target_host,''), NULLIF(apply_domain,'')),
                         liveness_checked_at DESC
            )
            SELECT latest.host, latest.reason, latest.checked_at,
                   COUNT(q.url) FILTER (
                       WHERE q.status='queued' AND COALESCE(q.liveness_required,FALSE)
                   ) AS deferred_rows,
                   GREATEST(0, %s - EXTRACT(EPOCH FROM (now() - latest.checked_at)))::INTEGER
                       AS remaining_seconds
            FROM latest
            LEFT JOIN apply_queue q
              ON COALESCE(NULLIF(q.target_host,''), NULLIF(q.apply_domain,'')) = latest.host
            GROUP BY latest.host, latest.reason, latest.checked_at
            ORDER BY remaining_seconds DESC, latest.host
            """,
            (_queue.LIVENESS_HOST_COOLDOWN_SECONDS, _queue.LIVENESS_HOST_COOLDOWN_SECONDS),
        )
        host_cooldowns = [
            {
                "host": row["host"],
                "reason": row["reason"],
                "checked_at": _iso(row["checked_at"]),
                "deferred_rows": int(row["deferred_rows"] or 0),
                "remaining_seconds": int(row["remaining_seconds"] or 0),
            }
            for row in cur.fetchall()
        ]
    conn.rollback()
    return {
        "available": True,
        "error": None,
        "totals": {key: int(totals.get(key) or 0) for key in ("live", "dead", "uncertain", "unchecked")},
        "uncertain_reasons": uncertain_reasons,
        "host_cooldowns": host_cooldowns,
    }


def build_status() -> dict:
    """Assemble the full /api/status object. Opens its OWN short-lived connections
    (each helper rolls back and we always close) so no connection is ever leaked."""
    conn = pgqueue.connect()  # reads FLEET_PG_DSN only (no generic fallback)
    try:
        gate, queue, apply_state = _gate_and_queue(conn)
        try:
            from applypilot.fleet import console_diagnosis

            fleet_diagnosis = console_diagnosis.queue_diagnosis(conn)
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            fleet_diagnosis = None
        try:
            gate["should_halt"] = bool(pgqueue.should_halt(conn))
        except Exception:
            gate["should_halt"] = gate["paused"]
        workers = _workers(conn)
        try:
            deadman = _deadman_status(conn)
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            deadman = {
                "active": False,
                "alerts": [],
                "code": None,
                "title": None,
                "reason": None,
                "alert": None,
                "alert_at": None,
            }
        try:
            browser = _browser_health(conn)
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            browser = {"summary": {}, "issues": []}
        versions = _versions(conn)
        deployment = _deployment_info(conn)
        recent = _recent(conn, 15)
        ch = cb._challenges(conn)
        challenges = len(ch.get("challenges", []))
        linkedin = _linkedin_summary(conn)
        try:
            doctor_sig = _doctor_signal(conn)
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            doctor_sig = None
        try:
            discovery = _discovery(conn)
            for worker in discovery.get("workers", []):
                worker["machine"] = _machine_display_name(worker.get("machine_owner"))
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            discovery = None
        try:
            agents_summary = _agent_routing_summary(conn, workers)
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            agents_summary = {
                "workers": len(workers),
                "dynamic_workers": 0,
                "switched_workers": 0,
                "partial_workers": 0,
                "blocked_workers": 0,
                "active_agent_blocks": [],
                "model_usage": {},
                "model_family_usage": {},
                "model_count": 0,
            }
        try:
            freshness = _data_freshness(conn, endpoint="status")
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            freshness = {"endpoint": "status", "generated_at": _utcnow().isoformat(), "ages": {}}
        try:
            liveness = _liveness_summary(conn)
        except Exception as exc:
            try:
                conn.rollback()
            except Exception:
                pass
            liveness = {
                "available": False,
                "error": type(exc).__name__,
                "totals": None,
                "uncertain_reasons": [],
                "host_cooldowns": [],
            }
        recommendation = _status_recommendation(
            gate=gate,
            queue_apply=queue.get("apply", {}),
            apply_state=apply_state,
            workers=workers,
            linkedin=linkedin,
            discovery=discovery,
            doctor_sig=doctor_sig,
            agents=agents_summary,
            browser=browser,
        )
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.close()
    return {
        "now": _utcnow().isoformat(),
        "gate": gate,
        "queue": queue,
        "fleet_diagnosis": fleet_diagnosis,
        "workers": workers,
        "versions": versions,
        "deployment": deployment,
        "recent": recent,
        "challenges": challenges,
        "linkedin": linkedin,
        "apply_state": apply_state,
        "agents": agents_summary,
        "freshness": freshness,
        "liveness": liveness,
        "recommendation": recommendation,
        "doctor": doctor_sig,
        "discovery": discovery,
        "deadman": deadman,
        # H19: the DeadMan alert (fleet_config.deadman_alert/_at, written by
        # applypilot.fleet.deadman's run_deadman) surfaced top-level -- NOT nested
        # under "doctor" -- so a silent fleet death/stall/running-hot is visible
        # even when the Doctor signal itself failed to load.
        "deadman_alert": (doctor_sig or {}).get("deadman_alert"),
        "deadman_alert_at": (doctor_sig or {}).get("deadman_alert_at"),
    }


# ---------------------------------------------------------------------------
# READ: /api/challenges -- unions open auth_challenge rows with PARKED rows
# (apply_status='challenge_pending') from BOTH lanes, grouped kind -> host, so the
# operator can see the whole captcha/OTP backlog (including rows a worker parked
# but that never raised/kept an auth_challenge row) in one place.
# ---------------------------------------------------------------------------
# _host_of is the console's name for the shared queue.host_of helper (kept as a
# module-level alias so this module's other call sites need no change). The
# single host/kind derivation lives in fleet/queue.py so this detail view and
# the CLI `challenges --grouped` count view (queue.challenge_summary) on both
# home mains can never diverge.
_host_of = _queue.host_of


def _age_hours(ts: Any, now: _dt.datetime) -> float | None:
    if ts is None or not hasattr(ts, "tzinfo"):
        return None
    t = ts if ts.tzinfo is not None else ts.replace(tzinfo=_dt.timezone.utc)
    return round((now - t).total_seconds() / 3600.0, 1)


def _challenges_data(conn) -> dict:
    """READ-ONLY: open auth_challenge rows + parked rows from apply_queue/linkedin_queue,
    unioned by url, grouped kind -> host. Parked rows with no matching open challenge
    group under kind '(no challenge row)'. Fetched via the SHARED queue._challenge_rows
    (lane=None -- both queues) so this detail view and queue.challenge_summary's count
    view can never diverge on what counts as an open/parked row."""
    now = _utcnow()
    open_challenges, parked_by_url = _queue._challenge_rows(conn, None)

    challenge_by_url: dict[str, dict] = {r["url"]: r for r in open_challenges}

    urls = set(challenge_by_url) | set(parked_by_url)
    groups: dict[tuple[str, str], list[dict]] = {}
    by_kind: dict[str, int] = {}
    by_staleness: dict[str, int] = {}
    stale_by_host: dict[str, int] = {}
    missing_metadata_by_host: dict[str, int] = {}
    stale_rows = 0
    fresh_rows = 0
    missing_metadata_rows = 0
    for url in urls:
        ch = challenge_by_url.get(url)
        lane, park = parked_by_url.get(url, (None, None))
        kind = ch["kind"] if ch else "(no challenge row)"
        host = _host_of(url)
        raised_at = ch["raised_at"] if ch else None
        updated_at = park["updated_at"] if park else None
        age_hours = _age_hours(raised_at if ch else updated_at, now)
        staleness = "stale" if age_hours is not None and age_hours > 24 else "fresh"
        has_metadata = bool((park or {}).get("metadata_complete"))
        row = {
            "url": url,
            "lane": lane,
            "company": park["company"] if park else None,
            "title": park["title"] if park else None,
            "score": park["score"] if park else None,
            "kind": kind,
            "machine": _machine_display_name(ch["machine_owner"]) if ch else None,
            "screenshot_url": ch["screenshot_url"] if ch else None,
            "age_hours": age_hours,
            "staleness": staleness,
            "has_metadata": has_metadata,
            "metadata_inferred_fields": list((park or {}).get("metadata_inferred_fields") or []),
        }
        by_kind[kind] = by_kind.get(kind, 0) + 1
        by_staleness[staleness] = by_staleness.get(staleness, 0) + 1
        if staleness == "stale":
            stale_rows += 1
            stale_by_host[host] = stale_by_host.get(host, 0) + 1
        else:
            fresh_rows += 1
        if not has_metadata:
            missing_metadata_rows += 1
            missing_metadata_by_host[host] = missing_metadata_by_host.get(host, 0) + 1
        groups.setdefault((kind, host), []).append(row)

    group_payloads = []
    clear_first = []
    resolve_first = []
    for (kind, host), rows in groups.items():
        stale_count = sum(1 for row in rows if row["staleness"] == "stale")
        fresh_count = len(rows) - stale_count
        missing_metadata_count = sum(1 for row in rows if not row["has_metadata"])
        park_only_count = sum(1 for row in rows if row["kind"] == "(no challenge row)")
        ages = [float(row["age_hours"]) for row in rows if row["age_hours"] is not None]
        oldest_age_hours = round(max(ages), 1) if ages else None
        machines = sorted({str(row["machine"]) for row in rows if row.get("machine")})
        lanes = [row.get("lane") for row in rows if row.get("lane")]
        lane = max(sorted(set(lanes)), key=lanes.count) if lanes else None
        group_summary = {
            "host": host,
            "kind": kind,
            "rows": len(rows),
            "stale_rows": stale_count,
            "fresh_rows": fresh_count,
            "missing_metadata_rows": missing_metadata_count,
            "park_only_rows": park_only_count,
            "oldest_age_hours": oldest_age_hours,
            "lane": lane,
            "machines": machines,
        }
        group_payloads.append({
            "kind": kind,
            "host": host,
            "rows": rows,
            "summary": group_summary,
        })
        if stale_count > 0 or missing_metadata_count > 0:
            clear_first.append(group_summary)
        else:
            resolve_first.append(group_summary)

    group_payloads.sort(
        key=lambda group: (
            -int(group["summary"]["stale_rows"]),
            -int(group["summary"]["missing_metadata_rows"]),
            -int(group["summary"]["rows"]),
            -float(group["summary"]["oldest_age_hours"] or 0.0),
            str(group["host"]),
            str(group["kind"]),
        )
    )
    clear_first.sort(
        key=lambda group: (
            -int(group["stale_rows"]),
            -int(group["missing_metadata_rows"]),
            -int(group["rows"]),
            -float(group["oldest_age_hours"] or 0.0),
            str(group["host"]),
            str(group["kind"]),
        )
    )
    resolve_first.sort(
        key=lambda group: (
            -int(group["fresh_rows"]),
            -int(group["rows"]),
            float(group["oldest_age_hours"] or 0.0),
            str(group["host"]),
            str(group["kind"]),
        )
    )

    return {
        "groups": group_payloads,
        "counts": {
            "open_challenges": len(open_challenges),
            "parked": len(parked_by_url),
        },
        "summary": {
            "rows": sum(len(rows) for rows in groups.values()),
            "fresh_rows": fresh_rows,
            "stale_rows": stale_rows,
            "missing_metadata_rows": missing_metadata_rows,
            "by_kind": dict(sorted(by_kind.items())),
            "by_staleness": dict(sorted(by_staleness.items())),
        },
        "triage": {
            "clear_first": clear_first,
            "resolve_first": resolve_first,
            "stale_by_host": dict(sorted(stale_by_host.items())),
            "missing_metadata_by_host": dict(sorted(missing_metadata_by_host.items())),
        },
    }


def build_challenges() -> dict:
    """Open a short-lived connection, build the /api/challenges payload, always close."""
    conn = pgqueue.connect()  # reads FLEET_PG_DSN only (no generic fallback)
    try:
        return _challenges_data(conn)
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.close()


# ---------------------------------------------------------------------------
# READ: per-worker crash + log tail (separate endpoint; NOT in the 4s /api/status
# poll, so the big text blobs never bloat it). The worker already scrubbed this text
# before storing it; we scrub AGAIN on read (defense in depth -- S1) and cap sizes (S3).
# ---------------------------------------------------------------------------
def _worker_logs(conn, worker_id: str) -> dict | None:
    """Return {worker_id, last_beat, last_error, recent_log} for one worker, or None if
    no such worker_heartbeat row exists (handler -> 404). The query is PARAMETERIZED
    (never SQL built from request input -- S5). Text is re-scrubbed + size-capped."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT worker_id, last_beat, last_error, recent_log "
            "FROM worker_heartbeat WHERE worker_id = %s",
            (worker_id,),
        )
        row = cur.fetchone()
    conn.rollback()  # read-only
    if row is None:
        return None
    last_error = _scrub(row.get("last_error"))[:_LOG_LAST_ERROR_CAP]
    recent_log = _scrub(row.get("recent_log"))
    if len(recent_log) > _LOG_RECENT_CAP:
        recent_log = recent_log[-_LOG_RECENT_CAP:]
    return {
        "worker_id": row["worker_id"],
        "last_beat": _iso(row.get("last_beat")),
        "last_error": last_error,
        "recent_log": recent_log,
    }


def worker_logs(worker_id: str) -> dict | None:
    """Open a short-lived connection, read one worker's logs, always close. None -> 404."""
    conn = pgqueue.connect()
    try:
        return _worker_logs(conn, worker_id)
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.close()


# ---------------------------------------------------------------------------
# WRITE: the strict action allow-list. NO LinkedIn action exists here by design.
# ---------------------------------------------------------------------------
def _do_arm_canary(conn, body: dict) -> str:
    k = body.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1 or k > 200:
        raise ValueError("arm_canary requires integer k in [1,200]")
    if not H.arm_canary_if_safe(conn, k):
        return (False, "Canary not armed: ATS pause or cost cap is active.")
    return f"Canary armed: up to {k} real applications will be submitted, then auto-pause."


def _do_lift_canary(conn, body: dict) -> str:
    # Disarming a canary must not also resume the fleet. Resume is an explicit
    # operator action with a separate confirmation path.
    H.lift_canary(conn)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE fleet_config SET ats_apply_mode='stopped', "
            "canary_enabled=FALSE, canary_remaining=NULL WHERE id=1"
        )
    conn.commit()
    return "Canary disarmed: no new applications will be leased until a positive canary is armed."


def _do_pause(conn, body: dict) -> str:
    pgqueue.set_paused(conn, True)
    return "Fleet PAUSED: no new applications will be leased."


def _do_resume(conn, body: dict) -> str:
    pgqueue.set_paused(conn, False)
    return "Fleet RESUMED: leasing is enabled again."


def _do_reclaim(conn, body: dict) -> str:
    rows = pgqueue.reclaim_stale_leases(conn)  # THE crash-safe reclaim; never a raw UPDATE
    return f"Reclaimed {len(rows)} stale lease(s)."


def _do_set_cap(conn, body: dict) -> str:
    usd = body.get("usd")
    normalized = pgqueue.validate_spend_cap(usd)
    pgqueue.set_spend_cap(conn, normalized)
    cap_txt = "no cap" if normalized == 0 else f"${normalized:.2f}"
    return f"Spend cap set to {cap_txt}."


def _searches_config_path() -> str:
    """Locate the project's searches config. Prefer APPLYPILOT_DIR (the launcher sets it),
    else <repo>/.applypilot/searches.yaml. Raises if absent."""
    candidates = []
    app_dir = os.environ.get("APPLYPILOT_DIR")
    if app_dir:
        candidates.append(os.path.join(app_dir, "searches.yaml"))
    repo_root = Path(__file__).resolve().parents[3]  # fleet -> applypilot -> src -> repo
    candidates.append(str(repo_root / ".applypilot" / "searches.yaml"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("searches.yaml not found (set APPLYPILOT_DIR or place it in .applypilot/)")


def _apply_searches_to_fleet(cfg: dict) -> dict:
    """Translate the apply tool's searches.yaml (queries/locations/sites) into the fleet
    discovery scheduler's {'searches':[...]} shape. A config that already has 'searches' is
    returned unchanged.

    SAFETY: LinkedIn is EXCLUDED from discovery boards. The home box holds the LinkedIn apply
    session (the catastrophe lane); anonymous LinkedIn scraping from that IP could correlate
    and endanger it. Discovery scrapes the non-LinkedIn boards (e.g. indeed) only."""
    if cfg.get("searches"):
        return cfg  # already fleet-native
    queries = []
    for q in (cfg.get("queries") or []):
        val = q.get("query") if isinstance(q, dict) else q
        if val:
            queries.append(val)
    boards = [s for s in (cfg.get("sites") or ["indeed"]) if s != "linkedin"] or ["indeed"]
    locations = []
    for loc in (cfg.get("locations") or []):
        if isinstance(loc, dict) and loc.get("location"):
            locations.append(loc["location"])
        elif isinstance(loc, str) and loc:
            locations.append(loc)
    if "Remote" not in locations:
        locations.append("Remote")
    if not locations:
        locations = ["Remote"]
    defaults = cfg.get("defaults") or {}
    params = {
        "results_wanted": int(defaults.get("results_per_site", 50)),
        "hours_old": int(defaults.get("hours_old", 72)),
    }
    searches = [{"query": q, "boards": boards, "locations": locations, "params": params}
                for q in queries]
    return {"searches": searches}


def _do_expand_searches(conn, body: dict) -> str:
    """Seed/refresh search_tasks from searches.yaml (PG-only — does NOT write the brain and
    does NOT submit any application). Calls the scheduler directly so the console never has to
    import the JobSpy-heavy discovery worker module."""
    from applypilot.fleet.scheduler import expand_search_config  # lazy: keeps the console light
    path = _searches_config_path()
    if path.endswith((".yaml", ".yml")):
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    else:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    config = _apply_searches_to_fleet(raw)
    n = expand_search_config(conn, config)
    conn.commit()  # run_action rolls back in finally, so write actions must commit themselves
    roles = len(config.get("searches") or [])
    return (f"Seeded/refreshed {n} search task(s) across {roles} roles "
            f"(LinkedIn excluded from discovery for safety). Discovery workers will pick them up.")


# ---------------------------------------------------------------------------
# FLEET DOCTOR controls (exactly two, both CONSERVATIVE / bookkeeping). These are the
# ONLY new actions added to the allow-list. Reversing a Doctor knob is conservative-or-
# neutral (clear host_skip = nothing else re-approved; clear timeout_override = back to
# NULL/default; deactivate a pace/pause knob = audit-only, the human still resumes
# explicitly). Dismissing a recommendation is pure bookkeeping. There is deliberately NO
# action that APPLIES a recommendation or does anything activity-increasing -- those stay
# manual via the existing controls. NO LinkedIn action exists here.
# ---------------------------------------------------------------------------
def _do_doctor_revert(conn, body: dict) -> str:
    """Reverse a single Doctor auto-fix: deactivate its fleet_knobs row, undo the knob's
    conservative-or-neutral side (clear the timeout override / restore the paced gap to its
    base), and mark its diagnosis 'reverted'. Identified by knob_id (preferred) or
    diagnosis_id. Parameterized SQL only. NEVER re-approves jobs, never resumes a pause, never
    raises a cap -- a revert can only relax a conservative knob, which is itself safe."""
    knob_id = body.get("knob_id")
    diagnosis_id = body.get("diagnosis_id")
    if not _is_pos_int(knob_id) and not _is_pos_int(diagnosis_id):
        raise ValueError("doctor_revert requires a positive integer knob_id or diagnosis_id")
    with conn.cursor() as cur:
        if _is_pos_int(knob_id):
            cur.execute("SELECT id, knob_type, scope_key, value_text FROM fleet_knobs WHERE id=%s AND active",
                        (int(knob_id),))
        else:
            # Resolve the active knob for this diagnosis via the auto_action prefix + scope.
            cur.execute(
                "SELECT k.id, k.knob_type, k.scope_key, k.value_text FROM fleet_knobs k "
                "JOIN fleet_diagnoses d ON d.auto_action LIKE k.knob_type || '%' "
                "WHERE d.id=%s AND k.active "
                "ORDER BY k.created_at DESC LIMIT 1",
                (int(diagnosis_id),),
            )
        knob = cur.fetchone()
        if knob is None:
            raise ValueError("no active Doctor knob found to revert")
        ktype, scope, vtext = knob["knob_type"], knob["scope_key"], knob.get("value_text")
        # Deactivate the knob (audit twin stays for history; active=FALSE removes its effect).
        cur.execute("UPDATE fleet_knobs SET active=FALSE WHERE id=%s", (knob["id"],))
        # Undo the knob's mechanical side -- each branch is conservative or neutral. H11: the
        # quarantine + host_skip branches now ACTUALLY clear the underlying state (the old
        # comment-only fall-through made those Reverse buttons lies).
        if ktype == "timeout_bump":
            # H7: restore the PRE-bump override captured in value_text ('new|prev'), not an
            # unconditional NULL -- preserving an operator-set override. Restore only if no OTHER
            # active timeout_bump remains (single shared column).
            prev = None
            if vtext and "|" in str(vtext):
                tail = str(vtext).split("|", 1)[1].strip()
                prev = int(tail) if tail.isdigit() else None
            cur.execute("SELECT count(*) AS n FROM fleet_knobs WHERE active AND knob_type='timeout_bump'")
            if int(cur.fetchone()["n"]) == 0:
                cur.execute("UPDATE fleet_config SET agent_timeout_override=%s, updated_at=now() WHERE id=1",
                            (prev,))
        elif ktype == "pace_or_pause":
            if scope and scope.startswith("host:"):
                # H4: a host pace lived in the Doctor-owned doctor_min_gap_floor; clearing it lets
                # the host run at its breaker-owned gap again. (We never touch min_gap_seconds --
                # that is the watchdog breaker's column.)
                cur.execute(
                    "UPDATE rate_governor SET doctor_min_gap_floor=NULL, updated_at=now() WHERE scope_key=%s",
                    (scope,))
            elif scope == "ats":
                # H8: a Doctor-authored ATS pause (ats_pause_source='doctor') -- Reverse DOES clear
                # it (the old Reverse was a no-op so it looked broken). NEVER clears an operator/cost
                # ats pause. This only ever touches ats_paused, never the shared fleet_config.paused,
                # so it cannot resume the LinkedIn lane.
                cur.execute(
                    "UPDATE fleet_config SET ats_paused=FALSE, ats_pause_source=NULL, "
                    "doctor_pause_armed_at=NULL, updated_at=now() "
                    "WHERE id=1 AND ats_pause_source='doctor'")
        elif ktype == "host_skip":
            # H6/H11: a host_skip lived in rate_governor.doctor_skip_until on the host scope.
            # Clearing it lets the host lease again immediately (approval was preserved -- nothing
            # to re-approve). This is the REAL undo the how_to_reverse text promises.
            cur.execute(
                "UPDATE rate_governor SET doctor_skip_until=NULL, updated_at=now() WHERE scope_key=%s",
                ("host:" + scope if scope and not scope.startswith("host:") else scope,))
        elif ktype == "quarantine":
            # H11: actually clear poison_jobs.quarantined_at for this url (manual: reasons only --
            # never un-quarantine a real crash-accumulated poison). The job is retained, never deleted.
            cur.execute(
                "UPDATE poison_jobs SET quarantined_at=NULL, reviewed=TRUE "
                "WHERE url=%s AND reason LIKE 'manual:%%'", (scope,))
        # Mark the audit diagnosis 'reverted'. If the caller named a diagnosis_id, mark exactly
        # that row. Otherwise (the UI's knob_id path) match the audit rows by the knob's
        # IDENTITY -- the same auto_action prefix the _diagnostics LATERAL join uses -- scoped to
        # the host/lane the diagnosis actually carries for this knob_type. The OLD COALESCE(host,
        # lane,'') = COALESCE(scope,...) match was a NO-OP for production-shaped diagnoses:
        # timeout_bump carries host=<triggering host>+lane='ats' but knob scope='ats' ('host'!='ats'),
        # and pace carries host=<host> but knob scope='host:'||<host> -- neither matched, leaving the
        # row stuck in 'auto_applied' (a phantom active auto-fix; corrupts the D3 audit trail).
        if _is_pos_int(diagnosis_id):
            cur.execute(
                "UPDATE fleet_diagnoses SET status='reverted', updated_at=now() "
                "WHERE id=%s AND status IN ('auto_applied','open')", (int(diagnosis_id),))
        else:
            # Per-knob-type scoping of the audit rows this knob produced:
            #   host_skip   -> diagnosis.host == knob.scope (the host)
            #   timeout_bump-> diagnosis.lane == knob.scope ('ats'); host is the *triggering* host
            #   pace_or_pause(pace)  -> knob.scope='host:'||<host>, diagnosis.host==<host>
            #   pace_or_pause(pause) -> knob.scope='ats', diagnosis.lane=='ats'
            if ktype == "host_skip":
                # A5: the host_skip knob scope is now canonical 'host:<h>'; the diagnosis carries the
                # BARE host, so strip the prefix to match.
                bare = scope[len("host:"):] if scope and scope.startswith("host:") else scope
                where, params = "host = %s", (bare,)
            elif ktype == "timeout_bump":
                where, params = "lane = %s", (scope,)
            elif ktype == "pace_or_pause" and scope and scope.startswith("host:"):
                where, params = "host = %s", (scope[len("host:"):],)
            else:  # pace_or_pause pause (scope='ats') or any lane-scoped knob
                where, params = "lane = %s", (scope,)
            cur.execute(
                "UPDATE fleet_diagnoses SET status='reverted', updated_at=now() "
                "WHERE status IN ('auto_applied','open') AND auto_action LIKE %s || '%%' AND " + where,
                (ktype, *params),
            )
    conn.commit()  # run_action rolls back in finally, so write actions must commit themselves
    return f"Reverted Doctor {ktype} (scope {scope!r}). The knob is deactivated; nothing was re-approved or resumed."


def _do_doctor_dismiss(conn, body: dict) -> str:
    """Dismiss a Doctor RECOMMENDATION (status -> 'dismissed'). Pure bookkeeping: it does NOT
    apply the recommendation or change any fleet state. Parameterized SQL only."""
    diagnosis_id = body.get("diagnosis_id")
    if not _is_pos_int(diagnosis_id):
        raise ValueError("doctor_dismiss requires a positive integer diagnosis_id")
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE fleet_diagnoses SET status='dismissed', updated_at=now() "
            "WHERE id=%s AND status='recommended'", (int(diagnosis_id),))
        n = cur.rowcount
    conn.commit()
    if n == 0:
        raise ValueError("no open recommendation with that id")
    return "Recommendation dismissed (bookkeeping only -- no fleet state changed)."


def _is_pos_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


# ---------------------------------------------------------------------------
# Challenge triage ops -- the ONLY mutations here go through queue.resolve_challenge
# (apply/ats lane) or queue.resolve_linkedin_challenge (linkedin lane). Both already
# stamp auth_challenge.resolved_at/outcome internally (queue.py:219-240 / :613-635),
# so we never re-stamp it here, and neither function touches rate_governor -- a
# LinkedIn halt (account:linkedin) is untouched by resolving a challenge. Routing is
# strictly by the request's ``lane`` field so an ats op can never reach a
# linkedin_queue row (and vice versa) even if the SAME url happens to be parked in
# both queues.
# ---------------------------------------------------------------------------
_LANE_RESOLVERS = {
    "apply": _queue.resolve_challenge,
    "linkedin": _queue.resolve_linkedin_challenge,
}


def _invalid_lane_message(lane: Any) -> str:
    valid = ", ".join(sorted(_LANE_RESOLVERS))
    got = "missing" if lane is None else str(lane)
    return f"lane must be one of: {valid}; got {got}"


def _do_challenge_resolve(conn, body: dict, *, requeue: bool):
    """Shared body for challenge_requeue (requeue=True) / challenge_skip (requeue=False).
    Idempotent: a url with no parked row is a success no-op (queue_rows: 0), never an
    error, per the triage contract."""
    url = body.get("url")
    lane = body.get("lane")
    if not isinstance(url, str) or not url:
        raise ValueError("url is required")
    resolver = _LANE_RESOLVERS.get(lane)
    if resolver is None:
        return False, _invalid_lane_message(lane)
    found = resolver(conn, url, requeue=requeue, commit=True)
    n = 1 if found else 0
    verb = "requeued" if requeue else "skipped"
    return f"Challenge {verb} for {lane} url (queue_rows: {n})."


def _do_challenge_requeue(conn, body: dict):
    return _do_challenge_resolve(conn, body, requeue=True)


def _do_challenge_skip(conn, body: dict):
    return _do_challenge_resolve(conn, body, requeue=False)


def _do_challenge_skip_host(conn, body: dict):
    """Skip (requeue=False) up to 200 parked urls on one host, in one lane. Selects the
    candidate urls read-only first, then routes each through the SAME
    resolve_challenge/resolve_linkedin_challenge primitive as challenge_skip -- no new
    UPDATE of queue rows. host+lane scoped so this can never spill into the other lane
    or another host."""
    host = body.get("host")
    lane = body.get("lane")
    if not isinstance(host, str) or not host:
        raise ValueError("host is required")
    resolver = _LANE_RESOLVERS.get(lane)
    if resolver is None:
        return False, _invalid_lane_message(lane)
    table = "apply_queue" if lane == "apply" else "linkedin_queue"
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT url FROM {table} WHERE apply_status='challenge_pending' "
            "AND (url LIKE %s OR url LIKE %s) LIMIT 200",
            (f"https://{host}/%", f"https://www.{host}/%"),
        )
        urls = [r["url"] for r in cur.fetchall()]
    n = 0
    for url in urls:
        if resolver(conn, url, requeue=False, commit=True):
            n += 1
    return f"Skipped {n} parked challenge(s) on {host} ({lane}) (queue_rows: {n})."


# Explicit allow-list. The handler indexes THIS dict by the action string; it never
# eval()s or getattr()s anything derived from the request. Any other action -> 400.
# NOTE: discovery's expand_searches is PG-only (no brain write); there is deliberately NO
# brain-ingest ("pull") action and no LinkedIn APPLY/scrape action on this surface. The
# challenge_* ops below are an exception carved out deliberately: they never apply,
# scrape, or resume LinkedIn -- they ONLY resolve a parked auth wall via the existing
# resolve_challenge/resolve_linkedin_challenge primitives, lane-routed so the two queues
# can never cross-contaminate, and they never touch rate_governor (a LinkedIn halt is
# untouched). The two doctor_* actions are CONSERVATIVE/bookkeeping (revert a
# conservative knob / dismiss a rec) -- they never apply a recommendation and never do
# anything activity-increasing.
_ACTIONS = {
    "arm_canary": _do_arm_canary,
    "lift_canary": _do_lift_canary,
    "pause": _do_pause,
    "resume": _do_resume,
    "reclaim": _do_reclaim,
    "set_cap": _do_set_cap,
    "expand_searches": _do_expand_searches,
    "doctor_revert": _do_doctor_revert,
    "doctor_dismiss": _do_doctor_dismiss,
    "challenge_requeue": _do_challenge_requeue,
    "challenge_skip": _do_challenge_skip,
    "challenge_skip_host": _do_challenge_skip_host,
}

_AUDIT_ACTION_CAP = 120
_AUDIT_MESSAGE_CAP = 500
_AUDIT_LANE_CAP = 120
_AUDIT_TARGET_CAP = 300
_AUDIT_READ_LIMIT = 25
_AUDIT_SECRET_WORD_RE = re.compile(
    r"(?i)\b(token|password|passwd|pwd|secret|api[_-]?key)\s+[^\s,'\"}{]+"
)


def _audit_text(value: Any, limit: int) -> str:
    safe = _scrub("" if value is None else str(value))
    safe = _AUDIT_SECRET_WORD_RE.sub(r"\1 [REDACTED]", safe)
    return safe[:limit]


def _audit_optional_text(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    return _audit_text(value, limit)


def _audit_action(conn, *, action: str, ok: bool, message: str,
                  lane: str | None = None, target: str | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO fleet_console_audit (action, actor, ok, message, lane, target) "
            "VALUES (%s,'console',%s,%s,%s,%s)",
            (
                _audit_text(action or "unknown", _AUDIT_ACTION_CAP),
                bool(ok),
                _audit_text(message or "", _AUDIT_MESSAGE_CAP),
                _audit_optional_text(lane, _AUDIT_LANE_CAP),
                _audit_optional_text(target, _AUDIT_TARGET_CAP),
            ),
        )
    conn.commit()


def _rollback_quietly(conn) -> None:
    try:
        conn.rollback()
    except Exception:
        pass


def _try_audit_action(conn, **kwargs) -> None:
    try:
        _audit_action(conn, **kwargs)
    except Exception:
        _rollback_quietly(conn)


def audit_lifecycle_event(action: str, message: str) -> None:
    """Best-effort audit for console process lifecycle events.

    Lifecycle audit must never prevent the dashboard from starting or stopping.
    """
    conn = None
    try:
        conn = pgqueue.connect()
        _audit_action(conn, action=action, ok=True, message=message, lane="console")
    except Exception:
        if conn is not None:
            _rollback_quietly(conn)
    finally:
        if conn is not None:
            conn.close()


def audit_rows(limit: int = _AUDIT_READ_LIMIT) -> dict[str, Any]:
    limit = max(1, min(int(limit), _AUDIT_READ_LIMIT))
    conn = pgqueue.connect()
    try:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, created_at, action, ok, message, lane, target "
                    "FROM fleet_console_audit "
                    "ORDER BY created_at DESC, id DESC "
                    "LIMIT %s",
                    (limit,),
                )
                rows = cur.fetchall()
        except errors.UndefinedTable:
            _rollback_quietly(conn)
            return {
                "rows": [],
                "schema_missing": True,
                "reason": "Console audit table is not installed; apply the fleet v3 schema migration.",
            }
        return {
            "rows": [
                {
                    "time": _iso(r["created_at"]),
                    "action": _audit_text(r["action"], _AUDIT_ACTION_CAP),
                    "ok": bool(r["ok"]),
                    "result": "ok" if r["ok"] else "failed",
                    "message": _audit_text(r["message"] or "", _AUDIT_MESSAGE_CAP),
                    "lane": _audit_optional_text(r["lane"], _AUDIT_LANE_CAP),
                    "target": _audit_optional_text(r["target"], _AUDIT_TARGET_CAP),
                }
                for r in rows
            ],
            "schema_missing": False,
        }
    finally:
        _rollback_quietly(conn)
        conn.close()


def _audit_lane(action: str, body: dict) -> str | None:
    lane = body.get("lane")
    if isinstance(lane, str) and lane:
        return lane[:40]
    if action in {"arm_canary", "lift_canary", "doctor_revert", "doctor_dismiss"}:
        return "ats"
    return None


def _audit_target(action: str, body: dict) -> str | None:
    if action in {"pause", "resume", "reclaim"}:
        return "fleet"
    if action in {"arm_canary", "lift_canary"}:
        return "ats"
    if action == "set_cap":
        return "spend_cap"
    if action == "expand_searches":
        return "search_tasks"
    if action == "doctor_revert":
        if _is_pos_int(body.get("knob_id")):
            return f"knob:{int(body['knob_id'])}"
        if _is_pos_int(body.get("diagnosis_id")):
            return f"diagnosis:{int(body['diagnosis_id'])}"
    if action == "doctor_dismiss" and _is_pos_int(body.get("diagnosis_id")):
        return f"diagnosis:{int(body['diagnosis_id'])}"
    if action == "challenge_skip_host" and isinstance(body.get("host"), str):
        return "host:" + _scrub(body["host"])[:120]
    if action in {"challenge_requeue", "challenge_skip"} and isinstance(body.get("url"), str):
        return "host:" + _scrub(_host_of(body["url"]))[:120]
    return action[:120]


def _record_console_audit(conn, action: str, body: dict, *, ok: bool, message: str) -> None:
    """Append a minimal, secret-free audit row for an existing allow-listed console action."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('fleet_console_audit') AS rel")
            if cur.fetchone()["rel"] is None:
                conn.rollback()
                return
            cur.execute(
                "INSERT INTO fleet_console_audit (action, actor, lane, target, message, ok) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    _scrub(action)[:80],
                    "console",
                    _audit_lane(action, body),
                    _audit_target(action, body),
                    _scrub(message)[:300],
                    bool(ok),
                ),
            )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass


def run_action(body: dict) -> tuple[bool, str]:
    """Validate + dispatch a single allowed action against a short-lived connection.
    Returns (ok, message). Connection always closed.

    An action fn normally returns a success message (str) -> (True, msg). It may
    instead return a (False, msg) tuple for an expected, non-exceptional failure
    (e.g. an unknown lane) -- distinct from raising ValueError/Exception, which
    the HTTP layer maps to 400/500."""
    action = body.get("action")
    fn = _ACTIONS.get(action) if isinstance(action, str) else None
    if fn is None:
        conn = None
        try:
            conn = pgqueue.connect()
            _try_audit_action(conn, action=str(action), ok=False, message="unknown action")
        except Exception:
            if conn is not None:
                _rollback_quietly(conn)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        return False, "unknown action"
    conn = pgqueue.connect()
    try:
        try:
            result = fn(conn, body)
        except Exception as exc:
            _rollback_quietly(conn)
            _try_audit_action(conn, action=action, ok=False, message=str(exc),
                              lane=_audit_lane(action, body), target=_audit_target(action, body))
            raise
        ok, message = result if isinstance(result, tuple) else (True, result)
        _try_audit_action(conn, action=action, ok=bool(ok), message=str(message),
                          lane=_audit_lane(action, body), target=_audit_target(action, body))
        return ok, message
    finally:
        _rollback_quietly(conn)
        conn.close()


# ---------------------------------------------------------------------------
# Bind safety (S2): private IPv4 / loopback only -- never 0.0.0.0, never public.
# ---------------------------------------------------------------------------
def _is_private_bind(host: str) -> bool:
    """True only for a loopback or RFC1918 private IPv4 literal. 0.0.0.0 and any public
    address are rejected. We require a literal IP (no DNS) so the bind target is explicit."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.version != 4:
        return False
    if ip == ipaddress.IPv4Address("0.0.0.0"):
        return False
    return bool(ip.is_loopback or ip.is_private)


def _detect_private_ip() -> str:
    """Best-effort discovery of THIS box's private LAN IPv4 (no traffic actually sent;
    connect() on a UDP socket just picks the egress interface). Falls back to 127.0.0.1."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        cand = s.getsockname()[0]
    except Exception:
        cand = "127.0.0.1"
    finally:
        s.close()
    if _is_private_bind(cand):
        return cand
    # Scan resolved interface addresses for any private IPv4 before giving up.
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = info[4][0]
            if _is_private_bind(addr) and not ipaddress.ip_address(addr).is_loopback:
                return addr
    except Exception:
        pass
    return "127.0.0.1"


def resolve_host(requested: str | None) -> str:
    """Resolve + validate the bind host. If none requested, auto-detect a private IP.
    A requested host that is not a private/loopback IPv4 literal raises (refuse to start)."""
    if not requested:
        return _detect_private_ip()
    if not _is_private_bind(requested):
        raise SystemExit(
            f"refusing to bind {requested!r}: LAN-only. Use a private IPv4 "
            "(10.x / 172.16-31.x / 192.168.x) or 127.0.0.1 -- never 0.0.0.0 or a public IP."
        )
    return requested


# ---------------------------------------------------------------------------
# Mutation token gate (audit finding: the console had zero auth). Every POST
# /api/action must present the token via X-Console-Token header or the
# console_token cookie. GET / can arm the cookie one time via ?token=<t>.
# NEVER print or log the token except the one arm-URL line at startup.
# ---------------------------------------------------------------------------
_CACHED_TOKEN: str | None = None
_TOKEN_COOKIE_NAME = "console_token"
_TOKEN_HEADER_NAME = "X-Console-Token"


def get_console_token() -> str:
    """Returns APPLYPILOT_CONSOLE_TOKEN if set, else a process-cached random token
    generated once via secrets.token_urlsafe(24)."""
    global _CACHED_TOKEN
    env_tok = os.environ.get("APPLYPILOT_CONSOLE_TOKEN")
    if env_tok:
        return env_tok
    if _CACHED_TOKEN is None:
        _CACHED_TOKEN = secrets.token_urlsafe(24)
    return _CACHED_TOKEN


def _token_ok(handler: Any) -> bool:
    """Checks the request's X-Console-Token header, falling back to the
    console_token cookie, against get_console_token() via a constant-time compare."""
    expected = get_console_token()
    header_tok = handler.headers.get(_TOKEN_HEADER_NAME)
    if header_tok and hmac.compare_digest(header_tok, expected):
        return True
    cookie_header = handler.headers.get("Cookie")
    if cookie_header:
        jar = http.cookies.SimpleCookie()
        try:
            jar.load(cookie_header)
        except Exception:
            return False
        morsel = jar.get(_TOKEN_COOKIE_NAME)
        if morsel and hmac.compare_digest(morsel.value, expected):
            return True
    return False


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    server_version = "ApplyPilotFleetConsole/1.0"

    # Silence the default access log so no URLs/IPs/secrets land in stdout.
    def log_message(self, *args, **kwargs):  # noqa: D401
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, code: int, obj: dict) -> None:
        self._send(
            code,
            json.dumps(obj, default=str).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return
        if path == "/":
            import urllib.parse as _up
            qs = _up.parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
            tok = (qs.get("token") or [None])[0]
            if tok and hmac.compare_digest(tok, get_console_token()):
                self.send_response(302)
                self.send_header("Location", "/")
                self.send_header("Cache-Control", "no-store")
                self.send_header(
                    "Set-Cookie", f"{_TOKEN_COOKIE_NAME}={tok}; HttpOnly; Path=/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._send(200, _INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/status":
            try:
                self._send_json(200, build_status())
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return
        if path == "/api/diagnosis":
            try:
                from applypilot.fleet import console_diagnosis  # noqa: F401

                self._send_json(200, diagnosis())
            except Exception as e:
                self._send_json(500, {"error": _audit_text(str(e), 500)})
            return
        if path == "/api/diagnostics":
            # Dedicated endpoint (NOT folded into the 4s /api/status poll): the Doctor blob
            # (clusters + auto-fixes + recommendations) can be large, so it is fetched on its
            # own cadence by the Diagnostics section.
            try:
                self._send_json(200, diagnostics())
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return
        if path == "/api/outcomes":
            try:
                self._send_json(200, outcomes())
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return
        if path == "/api/challenges":
            try:
                self._send_json(200, build_challenges())
            except Exception as e:
                self._send_json(500, {"error": str(e)})   # match sibling GET routes
            return
        if path == "/api/agents":
            try:
                self._send_json(200, agents())
            except Exception as e:
                self._send_json(500, {"error": _audit_text(str(e), 500)})
            return
        if path == "/api/audit":
            try:
                self._send_json(200, audit_rows())
            except Exception as e:
                self._send_json(500, {"error": _audit_text(str(e), 500)})
            return
        if path == "/api/logs":
            # Separate endpoint (NOT folded into /api/status) so the big text blobs
            # never bloat the 4s poll. The `worker` param is validated against an
            # existing worker_heartbeat row; an unknown/missing worker -> 400/404.
            import urllib.parse as _up
            qs = _up.parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
            wid = (qs.get("worker") or [None])[0]
            if not wid or not isinstance(wid, str):
                self._send_json(400, {"error": "missing worker param"})
                return
            try:
                logs = worker_logs(wid)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
                return
            if logs is None:
                self._send_json(404, {"error": "unknown worker"})
                return
            self._send_json(200, logs)
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not _token_ok(self):
            self._send_json(401, {"ok": False, "error": "bad token"})
            return
        path = self.path.split("?", 1)[0]
        if path != "/api/action":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            # parse_constant rejects bare NaN/Infinity/-Infinity tokens at the
            # parser (json.loads accepts them by default); _do_set_cap's
            # isfinite check is the necessary guard, this is defense in depth.
            def _reject_nonfinite(tok):  # noqa: ANN001
                raise ValueError(f"non-finite JSON literal not allowed: {tok}")
            body = json.loads(raw or b"{}", parse_constant=_reject_nonfinite)
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
        except Exception:
            self._send_json(400, {"ok": False, "error": "bad request body"})
            return
        try:
            ok, msg = run_action(body)
        except ValueError as e:
            self._send_json(400, {"ok": False, "error": str(e)})
            return
        except Exception as e:
            self._send_json(500, {"ok": False, "error": str(e)})
            return
        if not ok:
            self._send_json(400, {"ok": False, "error": msg})
            return
        try:
            status = build_status()
        except Exception as e:
            status = {"error": str(e)}
        self._send_json(200, {"ok": True, "message": msg, "status": status})


# ---------------------------------------------------------------------------
# Embedded frontend (single page; polls /api/status every 4s)
# ---------------------------------------------------------------------------
_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ApplyPilot Fleet Console</title>
<style>
  :root{
    --bg:#0d1117; --panel:#161b22; --panel2:#1c2230; --border:#2a3140;
    --fg:#e6edf3; --muted:#8b96a6; --green:#2ea043; --green2:#3fb950;
    --red:#da3633; --red2:#f85149; --amber:#d29922; --blue:#388bfd; --grey:#5b6470;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
    font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:14px}
  header{padding:14px 20px;border-bottom:1px solid var(--border);display:flex;
    align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
  h1{font-size:16px;margin:0;font-weight:600;letter-spacing:.3px}
  .sub{color:var(--muted);font-size:12px}
  .wrap{padding:18px 20px;max-width:1200px;margin:0 auto}
  .banner{padding:16px 20px;border-radius:10px;font-size:20px;font-weight:700;
    display:flex;align-items:center;gap:14px;margin-bottom:16px;border:1px solid var(--border)}
  .banner .dot{width:14px;height:14px;border-radius:50%}
  .banner.run{background:rgba(46,160,67,.12);color:var(--green2);border-color:rgba(46,160,67,.4)}
  .banner.run .dot{background:var(--green2);box-shadow:0 0 10px var(--green2)}
  .banner.pause{background:rgba(218,54,51,.12);color:var(--red2);border-color:rgba(218,54,51,.4)}
  .banner.pause .dot{background:var(--red2);box-shadow:0 0 10px var(--red2)}
  .banner.warn{background:rgba(210,153,34,.12);color:var(--amber);border-color:rgba(210,153,34,.4)}
  .banner.warn .dot{background:var(--amber);box-shadow:0 0 10px var(--amber)}
  .banner small{margin-left:auto;font-size:12px;font-weight:400;color:var(--muted)}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px;margin-bottom:18px}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px}
  .card .label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.6px}
  .card .val{font-size:24px;font-weight:700;margin-top:6px}
  .card .hint{color:var(--muted);font-size:11px;margin-top:4px}
  .band.primary{border-left:4px solid var(--blue)}
  .headline{font-size:24px;font-weight:750;margin:6px 0}
  .actionline{margin-top:10px;color:var(--fg);font-weight:600}
  .metric-grid,.diagnosis-grid,.machine-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
  .mini{background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:10px}
  .mini span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.4px}
  .mini b{display:block;font-size:20px;margin-top:4px}
  .funnel{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px}
  .fstep{background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:10px;min-height:70px}
  .fstep span{display:block;color:var(--muted);font-size:11px}
  .fstep b{font-size:22px}
  .apply-state{border-color:var(--border)}
  .apply-state.ok{border-color:rgba(86,211,100,.65)}
  .apply-state.warn{border-color:rgba(210,153,34,.65)}
  .apply-state.halted{border-color:rgba(248,81,73,.65)}
  .apply-state .val{color:var(--fg)}
  .apply-state.ok .val{color:#56d364}
  .apply-state.warn .val{color:#d29922}
  .apply-state.halted .val{color:#f85149}
  section{background:var(--panel);border:1px solid var(--border);border-radius:10px;
    padding:14px 16px;margin-bottom:16px}
  section h2{font-size:13px;margin:0 0 10px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}
  .table-wrap{width:100%;overflow:auto}
  table{width:100%;border-collapse:collapse;font-size:13px}
  .table-scroll{max-width:100%;overflow-x:auto;-webkit-overflow-scrolling:touch;overflow-wrap:anywhere}
  .table-scroll table{min-width:620px}
  th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--border);white-space:nowrap}
  th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.4px}
  tr:last-child td{border-bottom:none}
  .family-chip{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;line-height:1.1;border:1px solid var(--border)}
  .family-chip.claude{color:#b18cff;border-color:rgba(177,140,255,.5);background:rgba(177,140,255,.12)}
  .family-chip.codex-like{color:#58a6ff;border-color:rgba(88,166,255,.45);background:rgba(88,166,255,.12)}
  .family-chip.other{color:#8b949e;border-color:rgba(139,148,158,.35);background:rgba(139,148,158,.12)}
  .family-chip.unknown{color:#f8eec7;border-color:rgba(248,238,199,.45);background:rgba(248,238,199,.1)}
  .vendor-chip{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;line-height:1.1;border:1px solid var(--border)}
  .vendor-chip.claude{color:#c9a4ff;border-color:rgba(201,164,255,.45);background:rgba(201,164,255,.12)}
  .vendor-chip.codex{color:#58a6ff;border-color:rgba(88,166,255,.45);background:rgba(88,166,255,.12)}
  .vendor-chip.gemini{color:#ffbe6a;border-color:rgba(255,190,106,.45);background:rgba(255,190,106,.12)}
  .vendor-chip.other{color:#8b949e;border-color:rgba(139,148,158,.35);background:rgba(139,148,158,.12)}
  .vendor-chip.unknown{color:#f8eec7;border-color:rgba(248,238,199,.45);background:rgba(248,238,199,.1)}
  .verdict-chip{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;line-height:1.1;font-weight:600;border:1px solid var(--border)}
  .v-working{color:#56d364;border-color:rgba(86,211,100,.45);background:rgba(86,211,100,.1)}
  .v-blocked{color:#f85149;border-color:rgba(248,81,73,.45);background:rgba(248,81,73,.1)}
  .v-partial{color:#d29922;border-color:rgba(210,153,34,.45);background:rgba(210,153,34,.1)}
  .v-not-triggered{color:#8b949e;border-color:rgba(139,148,158,.45);background:rgba(139,148,158,.08)}
  .v-misconfigured{color:#d2a8ff;border-color:rgba(210,168,255,.45);background:rgba(210,168,255,.1)}
  .v-unknown{color:#b0b7c3;border-color:rgba(176,183,195,.45);background:rgba(176,183,195,.08)}
  tr.v-working td{background-color:rgba(86,211,100,.04)}
  tr.v-blocked td{background-color:rgba(248,81,73,.05)}
  tr.v-partial td{background-color:rgba(210,153,34,.05)}
  tr.v-not-triggered td{background-color:rgba(139,148,158,.04)}
  tr.v-misconfigured td{background-color:rgba(210,168,255,.05)}
  tr.v-unknown td{background-color:rgba(176,183,195,.03)}
  .family-legend{display:flex;flex-wrap:wrap;gap:8px;color:var(--muted);font-size:11px;margin:0 0 10px}
  .family-legend .tag{display:inline-flex;align-items:center;gap:6px}
  .verdict-legend{display:flex;flex-wrap:wrap;gap:8px;color:var(--muted);font-size:11px;margin:8px 0 0}
  .verdict-legend .tag{display:inline-flex;align-items:center;gap:6px}
  .ldot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;vertical-align:middle}
  .ldot.on{background:var(--green2);box-shadow:0 0 6px var(--green2)}
  .ldot.off{background:var(--grey)}
  .s-applied{color:var(--green2);font-weight:600}
  .s-failed{color:var(--red2);font-weight:600}
  .s-crash{color:var(--amber);font-weight:600}
  .mut{color:var(--muted)}
  .controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
  button{font:inherit;border:1px solid var(--border);background:var(--panel2);color:var(--fg);
    padding:8px 14px;border-radius:8px;cursor:pointer}
  button:hover{border-color:var(--blue)}
  button.danger{border-color:rgba(218,54,51,.5);color:var(--red2)}
  button.go{border-color:rgba(46,160,67,.5);color:var(--green2)}
  input[type=number]{font:inherit;background:var(--bg);border:1px solid var(--border);
    color:var(--fg);padding:8px;border-radius:8px;width:90px}
  .li-strip{display:flex;flex-wrap:wrap;gap:18px;align-items:center}
  .li-strip .chip{background:var(--panel2);border:1px solid var(--border);border-radius:8px;
    padding:6px 12px}
  .li-strip .ro{margin-left:auto;color:var(--amber);font-size:12px}
  .toast{position:fixed;right:20px;bottom:20px;max-width:420px;background:var(--panel2);
    border:1px solid var(--border);border-left:4px solid var(--blue);border-radius:8px;
    padding:12px 16px;box-shadow:0 8px 24px rgba(0,0,0,.4);opacity:0;transform:translateY(8px);
    transition:opacity .2s,transform .2s;font-size:13px}
  .toast.show{opacity:1;transform:translateY(0)}
  .toast.err{border-left-color:var(--red2)}
  .toast.ok{border-left-color:var(--green2)}
  label.inl{display:inline-flex;align-items:center;gap:6px;color:var(--muted);font-size:12px}
  select{font:inherit;background:var(--bg);border:1px solid var(--border);color:var(--fg);
    padding:8px;border-radius:8px}
  .logblk{margin-top:10px}
  .logblk .lbl{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
  pre.log{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px 12px;
    margin:0;max-height:280px;overflow:auto;white-space:pre-wrap;word-break:break-word;
    font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:var(--fg)}
  pre.log.err{border-color:rgba(218,54,51,.45)}
  .dgrp{margin-bottom:14px}
  .dgrp .lbl{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
  .rec{background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin-bottom:8px;overflow:hidden;overflow-wrap:anywhere}
  .rec .rh{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:4px}
  .rec .rh .mut{min-width:0;flex:1 1 180px;overflow-wrap:anywhere}
  .rec .rh button.sm{flex:0 0 auto}
  .rec .sev{font-size:10px;text-transform:uppercase;letter-spacing:.5px;padding:2px 7px;border-radius:6px;
    border:1px solid var(--border);color:var(--muted)}
  .rec .sev.warn{color:var(--amber);border-color:rgba(210,153,34,.4)}
  .rec .sev.severe{color:var(--red2);border-color:rgba(218,54,51,.4)}
  .rec .body{font-size:13px}
  .rec .recm{color:var(--muted);font-size:12px;margin-top:4px}
  @media(max-width:560px){.rec .rh button.sm{margin-left:0!important}}
  button.sm{padding:5px 10px;font-size:12px}
  #challenges .chgrp{margin-bottom:14px;border:1px solid var(--border);border-radius:8px;overflow:hidden}
  #challenges .chgrp-h{display:flex;flex-wrap:wrap;align-items:center;gap:10px;
    background:var(--panel2);padding:8px 12px;font-size:12px;color:var(--muted)}
  #challenges .chgrp-h b{color:var(--fg)}
  #challenges .chrow{display:flex;flex-wrap:wrap;align-items:center;gap:10px;
    padding:10px 12px;border-top:1px solid var(--border)}
  #challenges .chrow .meta{flex:1 1 220px;min-width:0}
  #challenges .chrow .meta .ct{font-weight:600;overflow-wrap:anywhere}
  #challenges .chrow .meta .sub{color:var(--muted);font-size:11px;margin-top:2px}
  #challenges .chrow .btns{display:flex;flex-wrap:wrap;gap:8px}
  #challenges button{min-height:36px}
  .tokbanner{position:fixed;top:0;left:0;right:0;z-index:1000;background:var(--red2);
    color:#1a0000;font-weight:700;text-align:center;padding:10px 14px;font-size:13px;
    display:none}
  .tokbanner.show{display:block}
  .deadmanbanner{position:fixed;top:0;left:0;right:0;z-index:1001;background:var(--red2);
    color:#1a0000;font-weight:700;text-align:center;padding:10px 14px;font-size:13px;
    display:none}
  .deadmanbanner.show{display:block}
</style>
</head>
<body>
<div class="tokbanner" id="tokBanner">token expired — open the ?token= URL printed by the console window</div>
<div class="deadmanbanner" id="deadmanBanner"></div>
<header>
  <div>
    <h1>ApplyPilot Fleet Console</h1>
    <div class="sub">LAN-only control panel &mdash; this operates a system that submits REAL applications</div>
    <div class="sub" id="deploymentMeta">console version loading</div>
  </div>
  <div class="sub" id="conn">connecting&hellip;</div>
</header>
<div class="wrap">
  <div class="banner pause" id="banner">
    <span class="dot"></span><span id="bannerText">LOADING</span>
    <small id="updated"></small>
  </div>

  <div class="cards">
    <div class="card apply-state" id="applyStateCard">
      <div class="label">Apply state</div>
      <div class="val" id="applyState">&mdash;</div><div class="hint" id="applyStateHint">&nbsp;</div></div>
    <div class="card"><div class="label">Canary</div>
      <div class="val" id="cCanary">&mdash;</div><div class="hint" id="cCanaryHint"></div></div>
    <div class="card"><div class="label">Spend</div>
      <div class="val" id="cSpend">&mdash;</div><div class="hint" id="cSpendHint">of cap</div></div>
    <div class="card"><div class="label">ATS queued</div>
      <div class="val" id="cQueuedAts">&mdash;</div><div class="hint">queued / leased / applied</div></div>
    <div class="card"><div class="label">Apply queue</div>
      <div class="val" id="cQueued">&mdash;</div><div class="hint">queued</div></div>
    <div class="card"><div class="label">Leasable now</div>
      <div class="val" id="cLeasable">&mdash;</div><div class="hint">approved &amp; not deduped</div></div>
    <div class="card"><div class="label">Compute queued</div>
      <div class="val" id="cQueuedCompute">&mdash;</div><div class="hint">queued / leased / applied</div></div>
    <div class="card"><div class="label">Other queued</div>
      <div class="val" id="cQueuedOther">&mdash;</div><div class="hint">queued / leased / applied</div></div>
    <div class="card"><div class="label">Flow (10m)</div>
      <div class="val" id="cQueueFlow">&mdash;</div><div class="hint" id="cQueueFlowHint">leases/applied per minute</div></div>
    <div class="card"><div class="label">ATS lane flow</div>
      <div class="val" id="cLaneFlowAts">&mdash;</div><div class="hint" id="cLaneFlowAtsHint">leases/applied per minute</div></div>
    <div class="card"><div class="label">Compute lane flow</div>
      <div class="val" id="cLaneFlowCompute">&mdash;</div><div class="hint" id="cLaneFlowComputeHint">leases/applied per minute</div></div>
    <div class="card"><div class="label">Other lane flow</div>
      <div class="val" id="cLaneFlowOther">&mdash;</div><div class="hint" id="cLaneFlowOtherHint">leases/applied per minute</div></div>
    <div class="card"><div class="label">Open challenges</div>
      <div class="val" id="cChallenges">&mdash;</div><div class="hint">need a human</div></div>
    <div class="card"><div class="label">Doctor</div>
      <div class="val" id="cDoctor">&mdash;</div><div class="hint" id="cDoctorHint">auto-fixes active</div></div>
  </div>

  <section id="fleetState" class="band primary">
    <h2>Fleet State</h2>
    <div id="stateHeadline" class="headline">Loading</div>
    <div id="stateReason" class="sub"></div>
    <div id="stateAction" class="actionline">—</div>
    <div id="primaryBlocker" class="actionline blockerline">Primary blocker: loading</div>
    <div id="nextAction" class="actionline">Recommended Next Action: Loading</div>
    <h2 style="margin-top:14px">Action Queue</h2>
    <div id="recommendationList" class="rec-list"><div class="mut">loading recommendations</div></div>
  </section>

  <section id="laneActivity">
    <h2>Lane Activity</h2>
    <div id="laneActivityGrid" class="diagnosis-grid"></div>
  </section>

  <section id="safetyRails">
    <h2>Safety Rails</h2>
    <div class="metric-grid" id="safetyGrid"></div>
    <h2 style="margin-top:14px">Lane State</h2>
    <div class="diagnosis-grid lane-state" id="laneStateGrid"></div>
  </section>

  <section id="whyNotApplying">
    <h2>Why Not Applying</h2>
    <div class="diagnosis-grid" id="whyBody"></div>
  </section>

  <section id="applyReadiness">
    <h2>Apply Readiness</h2>
    <div id="applyReadinessVerdict" class="headline">loading</div>
    <div id="applyReadinessSummary" class="sub">waiting for fleet status, diagnosis, and agent telemetry</div>
    <div id="applyReadinessChecks" class="readiness-checks"></div>
  </section>

  <section id="discoveryBacklog">
    <h2>Discovery Backlog</h2>
    <div class="diagnosis-grid" id="discoveryBacklogGrid"></div>
  </section>

  <section id="machineHealth">
    <h2>Machine Health</h2>
    <div class="machine-grid" id="machineMap"></div>
    <div class="dgrp" style="margin-top:12px"><div class="lbl">Stale Apply Workers</div>
      <div id="staleWorkers" class="mut">loading stale worker details</div></div>
  </section>

  <section id="agentRouting">
    <h2>Agent Routing</h2>
    <div id="agentVerdict" class="sub"></div>
    <div id="agentSummary" class="diagnosis-grid"></div>
    <div class="table-scroll"><table><thead><tr><th>Worker</th><th>Machine</th><th>Agent</th><th>Model</th><th>Chain</th><th>Version</th><th>Switch</th></tr></thead>
      <tbody id="agentWorkers"><tr><td colspan="7" class="mut">loading</td></tr></tbody></table></div>
    <div class="funnel" id="agentBody"></div>
  </section>

  <section id="machineNetwork">
    <h2>Machine Network</h2>
    <div class="table-scroll"><table><thead><tr><th>Machine</th><th>Home IP</th><th>Roles</th><th>Live</th><th>Workers</th></tr></thead>
      <tbody id="machineNetworkRows"><tr><td colspan="5" class="mut">loading</td></tr></tbody></table></div>
  </section>

  <section id="deploymentDrift">
    <h2>Deployment Drift</h2><div id="deploymentDriftSummary" class="sub"></div>
    <div class="table-scroll"><table><thead><tr><th>Version</th><th>Count</th><th>Machines</th><th>Workers</th><th>Status</th><th>Deployment target</th></tr></thead>
      <tbody id="deploymentDriftRows"><tr><td colspan="6" class="mut">loading</td></tr></tbody></table></div>
  </section>

  <section id="browserHealth">
    <h2>Browser Health</h2><div id="browserBody" class="diagnosis-grid"></div>
    <div id="browserWallQueue" class="mut">loading browser wall queue</div>
  </section>

  <section id="workerComparison">
    <h2>Worker Comparison</h2>
    <div class="table-scroll"><table><thead><tr><th>Worker</th><th>Applied</th><th>Total</th><th>Success</th><th>Crash</th><th>Cost / Apply</th></tr></thead>
      <tbody id="workerComparisonRows"><tr><td colspan="6" class="mut">loading</td></tr></tbody></table></div>
    <div class="funnel" id="workerComparisonBody"></div>
  </section>

  <section id="queueFunnel">
    <h2>Queue Funnel</h2>
    <div class="funnel" id="funnelBody"></div>
  </section>

  <section id="livenessReasons">
    <h2>Job Availability</h2>
    <div class="funnel" id="livenessBody"></div>
  </section>

  <section id="throughputForecast">
    <h2>Throughput Forecast</h2>
    <div class="funnel" id="forecastBody"></div>
  </section>

  <section id="throughputSummary">
    <h2>Throughput</h2><div class="diagnosis-grid" id="throughputGrid"></div>
  </section>

  <section id="hostQuality">
    <h2>Host Quality</h2>
    <div class="table-scroll"><table><thead><tr><th>Host</th><th>Total</th><th>Applied</th><th>Failed</th><th>Challenges</th><th>Bad Rate</th></tr></thead>
      <tbody id="hostQualityRows"><tr><td colspan="6" class="mut">loading</td></tr></tbody></table></div>
  </section>

  <section id="auditLog">
    <h2>Audit Log</h2>
    <div class="table-scroll"><table><thead><tr><th>Time</th><th>Action</th><th>Result</th><th>Message</th></tr></thead>
      <tbody id="auditRows"><tr><td colspan="4" class="mut">audit endpoint not loaded</td></tr></tbody></table></div>
  </section>

  <section id="dailyGoals">
    <h2>Daily Goals</h2>
    <div class="funnel" id="dailyGoalsBody"></div>
  </section>

  <section id="dataFreshness">
    <h2>Data Freshness</h2>
    <div class="funnel" id="freshnessBody"></div>
  </section>

  <section id="operatorAudit">
    <h2>Operator Audit</h2>
    <div class="funnel" id="operatorAuditBody"></div>
  </section>

  <section id="hostSourceQuality">
    <h2>Host / Source Quality</h2>
    <div class="funnel" id="hostQualityBody"></div>
  </section>

  <section id="linkedinLane">
    <h2>LinkedIn Lane</h2>
    <div class="funnel" id="linkedinBody"></div>
  </section>

  <section id="discoveryPipeline">
    <h2>Discovery Pipeline</h2>
    <div class="funnel" id="discoveryBody"></div>
  </section>

  <section id="computeHealth">
    <h2>Compute Health</h2>
    <div class="funnel" id="computeBody"></div>
  </section>

  <section id="deadmanWatchdog">
    <h2>Fleet Watchdog</h2>
    <div class="funnel" id="deadmanBody"></div>
  </section>

  <section id="queueRecommendations">
    <h2>Recommended Next Action</h2>
    <div class="mut" id="stateRecommendation">—</div>
  </section>

  <section>
    <h2>Controls</h2>
    <div class="controls">
      <label class="inl">k <input type="number" id="canaryK" min="1" max="200" value="10"></label>
      <button class="go" id="btnArm">Arm canary</button>
      <button class="go" id="btnLift">Lift canary (run unbounded)</button>
      <button class="danger" id="btnPause">Pause</button>
      <button id="btnResume">Resume</button>
      <button id="btnReclaim">Reclaim stale leases</button>
      <span style="width:1px;height:24px;background:var(--border)"></span>
      <label class="inl">$ <input type="number" id="capUsd" min="0" step="0.01" placeholder="cap"></label>
      <button id="btnCap">Set $ cap</button>
    </div>
  </section>

  <section id="challenges">
    <h2>Challenges (need a human)</h2>
    <div class="sub" style="margin-bottom:12px">
      Parked applications waiting on a captcha / OTP / login wall, grouped by kind then host.
      Solve it in the browser, then Re-queue (or Skip if it's not worth it).
    </div>
    <div id="chList" class="mut">&hellip;</div>
  </section>

  <section>
    <h2>Software versions</h2>
    <div class="cards" style="margin-bottom:12px">
      <div class="card"><div class="label">Pinned</div>
        <div class="val" id="versionPinned">&mdash;</div><div class="hint">expected worker version</div></div>
      <div class="card"><div class="label">Generic canary</div>
        <div class="val" id="versionCanary">&mdash;</div><div class="hint" id="versionCanaryHint">optional non-application worker</div></div>
      <div class="card"><div class="label">ATS canary</div>
        <div class="val" id="versionAtsCanary">&mdash;</div><div class="hint" id="versionAtsCanaryHint">off</div></div>
      <div class="card"><div class="label">LinkedIn canary</div>
        <div class="val" id="versionLinkedinCanary">&mdash;</div><div class="hint" id="versionLinkedinCanaryHint">off</div></div>
      <div class="card"><div class="label">Drifted</div>
        <div class="val" id="versionDrift">&mdash;</div><div class="hint">workers off expected version</div></div>
    </div>
    <div class="table-wrap">
      <table><thead><tr><th>Version</th><th>Workers</th></tr></thead>
        <tbody id="versionRows"><tr><td colspan="2" class="mut">&hellip;</td></tr></tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Workers (apply lane)</h2>
    <div class="table-wrap">
      <table><thead><tr><th>Machine</th><th>Worker</th><th>Live</th><th>Last beat</th><th>Applied</th><th>Agent</th><th>Model</th><th>Model vendor</th><th>Model family</th><th>Last switch</th><th>Current job</th></tr></thead>
        <tbody id="workers"><tr><td colspan="11" class="mut">&hellip;</td></tr></tbody>
      </table>
    </div>
  </section>

    <section>
    <h2>Agents and models</h2>
    <div class="cards" style="margin-bottom:12px">
      <div class="card"><div class="label">Active blocks</div>
        <div class="val" id="aBlocks">&mdash;</div><div class="hint">agents currently blocked</div></div>
      <div class="card"><div class="label">Spending telemetry</div>
        <div class="val" id="aSpendRows">&mdash;</div><div class="hint">agents with apply spend in last 24h</div></div>
      <div class="card"><div class="label">Dynamic switching (live)</div>
        <div class="val" id="aSwitch">&mdash;</div><div class="hint">working / blocked / partial / misconfigured</div></div>
      <div class="card"><div class="label">Dynamic switching</div>
        <div class="val" id="aDynamicWorked">&mdash;</div><div class="hint">whether alternate agent is currently active</div></div>
      <div class="card"><div class="label">Model in use</div>
        <div class="val" id="aPrimaryModel">&mdash;</div><div class="hint" id="aPrimaryModelHint">top current model by worker count</div></div>
      <div class="card"><div class="label">Dynamic switching summary</div>
        <div class="val" id="aSwitchStatus">&mdash;</div><div class="hint" id="aSwitchStatusHint">workers on alternate agent now</div></div>
      <div class="card"><div class="label">Model family</div>
        <div class="val" id="aModelFamily">&mdash;</div><div class="hint" id="aModelFamilyHint">model family mix</div></div>
      <div class="card"><div class="label">Model trend (9m)</div>
        <div class="val" id="aModelTrend">&mdash;</div><div class="hint" id="aModelTrendHint">rolling snapshot trend</div></div>
      <div class="card"><div class="label">Latest switch</div>
        <div class="val" id="aLatestSwitch">&mdash;</div><div class="hint">most recent model/agent switch</div></div>
    </div>
    <div class="family-legend" aria-label="model family legend">
      <span class="tag"><span class="family-chip claude">claude</span> Claude family</span>
      <span class="tag"><span class="family-chip codex-like">codex-like</span> Codex-like</span>
      <span class="tag"><span class="family-chip other">other</span> Other</span>
      <span class="tag"><span class="family-chip unknown">unknown</span> Unknown</span>
    </div>
    <div class="family-legend" aria-label="model vendor legend">
      <span class="tag"><span class="vendor-chip codex">codex</span> Codex</span>
      <span class="tag"><span class="vendor-chip claude">claude</span> Claude</span>
      <span class="tag"><span class="vendor-chip gemini">gemini</span> Gemini</span>
      <span class="tag"><span class="vendor-chip other">other</span> Other</span>
    </div>
    <div class="verdict-legend" aria-label="agent switching verdict legend">
      <span class="tag"><span class="verdict-chip v-working">working</span> Working</span>
      <span class="tag"><span class="verdict-chip v-blocked">blocked</span> Blocked</span>
      <span class="tag"><span class="verdict-chip v-partial">partial</span> Partial</span>
      <span class="tag"><span class="verdict-chip v-misconfigured">misconfigured</span> Misconfigured</span>
      <span class="tag"><span class="verdict-chip v-not-triggered">not triggered</span> Not triggered</span>
      <span class="tag"><span class="verdict-chip v-unknown">unknown</span> Unknown</span>
    </div>
    <div class="table-wrap">
      <table><thead><tr><th>Machine</th><th>Worker</th><th>Current agent</th><th>Model</th><th>Model vendor</th><th>Model family</th><th>Configured chain</th><th>Verdict</th><th>Blocked agents</th><th>Last switch</th></tr></thead>
        <tbody id="agents"><tr><td colspan="10" class="mut">&hellip;</td></tr></tbody>
      </table>
    </div>
    <div class="table-wrap">
      <table><thead><tr><th>Machine</th><th>Workers</th><th>Codex</th><th>Claude</th><th>Gemini</th><th>Other</th><th>Unknown</th></tr></thead>
        <tbody id="aModelByMachine"><tr><td colspan="7" class="mut">&hellip;</td></tr></tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Recent activity</h2>
   <div class="table-wrap">
     <table><thead><tr><th>Time</th><th>Worker</th><th>Status</th><th>Company / Title</th></tr></thead>
       <tbody id="recent"><tr><td colspan="4" class="mut">&hellip;</td></tr></tbody>
     </table>
   </div>
  </section>

  <section>
    <h2>Worker logs (crash + recent activity)</h2>
    <div class="controls" style="margin-bottom:12px">
      <label class="inl">worker
        <select id="logWorker"><option value="">&mdash;</option></select>
      </label>
      <button id="btnLogs">Refresh</button>
      <span class="mut" style="font-size:12px">Last crash traceback + recent log tail, shipped via the heartbeat (secrets scrubbed). Read-only.</span>
    </div>
    <div id="logArea" class="mut" style="font-size:12px">Select a worker to view its logs.</div>
  </section>

  <section>
    <h2>Browser and backend health</h2>
    <div class="cards" style="margin-bottom:12px">
      <div class="card"><div class="label">Workers</div>
        <div class="val" id="bWorkers">&mdash;</div><div class="hint">heartbeat rows scanned</div></div>
      <div class="card"><div class="label">Problem workers</div>
        <div class="val" id="bProblem">&mdash;</div><div class="hint" id="bProblemHint">workers with browser/backend symptoms</div></div>
      <div class="card"><div class="label">Healthy workers</div>
        <div class="val" id="bHealthy">&mdash;</div><div class="hint">currently no browser/backend symptom</div></div>
    </div>
    <div class="table-wrap">
      <table><thead><tr><th>Worker</th><th>Machine</th><th>Lane</th><th>State</th><th>Issue</th><th>Age</th><th>Evidence</th><th>Last beat</th></tr></thead>
        <tbody id="browserHealth"><tr><td colspan="8" class="mut">&hellip;</td></tr></tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Discovery lane</h2>
    <div class="cards" style="margin-bottom:12px">
      <div class="card"><div class="label">Search tasks</div>
        <div class="val" id="dTasks">&mdash;</div><div class="hint" id="dTasksHint">enabled</div></div>
      <div class="card"><div class="label">Due now</div>
        <div class="val" id="dDue">&mdash;</div><div class="hint">ready to scrape</div></div>
      <div class="card"><div class="label">Postings staged</div>
        <div class="val" id="dStaged">&mdash;</div><div class="hint" id="dStagedHint">pending ingest</div></div>
      <div class="card"><div class="label">Found (24h)</div>
        <div class="val" id="dFound24">&mdash;</div><div class="hint">new postings</div></div>
    </div>
    <div class="controls" style="margin-bottom:12px">
      <button class="go" id="btnExpand">Expand searches (seed tasks)</button>
      <span class="mut" style="font-size:12px">Seeds <code>search_tasks</code> from searches.yaml &mdash; queues scrape work, submits nothing. Discovery workers run per-machine (<code>applypilot-fleet-discovery</code>).</span>
    </div>
    <div class="table-wrap">
      <table><thead><tr><th>Discovery worker</th><th>Machine</th><th>Live</th><th>Last beat</th><th>Found 24h</th></tr></thead>
        <tbody id="discWorkers"><tr><td colspan="5" class="mut">&hellip;</td></tr></tbody>
      </table>
    </div>
    <div class="table-wrap" style="margin-top:10px">
      <table><thead><tr><th>Found</th><th>Source</th><th>Search</th></tr></thead>
        <tbody id="discRecent"><tr><td colspan="3" class="mut">&hellip;</td></tr></tbody>
      </table>
    </div>
  </section>

<section>
  <h2>Recent outcomes (read-only)</h2>
   <div class="table-wrap">
     <table><thead><tr><th>Time</th><th>Company / Title</th><th>Stage</th><th>Outcome</th></tr></thead>
       <tbody id="outcomes"><tr><td colspan="4" class="mut">&hellip;</td></tr></tbody>
     </table>
   </div>
</section>

  <section>
    <h2>Diagnostics (Doctor)</h2>
    <div class="sub" style="margin-bottom:12px">
      Bounded, reversible, <b>conservative-only</b> auto-fixes (skip a blocking host, raise a
      timeout within a ceiling, quarantine a poison url, pace down / pause). The Doctor can never
      resume, re-approve, raise the spend cap, lower the gap, or touch LinkedIn. Everything else is
      a recommendation for you.
    </div>
    <div class="dgrp"><div class="lbl">Live failure clusters (last 60m)</div>
      <div class="table-wrap">
        <table><thead><tr><th>Reason</th><th>Host</th><th>Machine</th><th>Samples</th></tr></thead>
          <tbody id="docClusters"><tr><td colspan="4" class="mut">&hellip;</td></tr></tbody>
        </table>
      </div>
    </div>
    <div class="dgrp"><div class="lbl">Active auto-fixes</div>
      <div class="table-wrap">
        <table><thead><tr><th>Type</th><th>Scope</th><th>Reason</th><th>Expires</th><th></th></tr></thead>
          <tbody id="docAuto"><tr><td colspan="5" class="mut">&hellip;</td></tr></tbody>
        </table>
      </div>
    </div>
    <div class="dgrp"><div class="lbl">Recommendations (need a human)</div>
      <div id="docRecs" class="mut">&hellip;</div>
    </div>
  </section>

  <section>
    <h2>LinkedIn (read-only &mdash; the catastrophe lane has no controls here)</h2>
    <div class="li-strip">
      <div class="chip">Queued: <b id="liQueued">&mdash;</b></div>
      <div class="chip">Applied: <b id="liApplied">&mdash;</b></div>
      <div class="chip">Canary: <b id="liCanary">&mdash;</b></div>
      <div class="chip">Halted: <b id="liHalted">&mdash;</b></div>
      <span class="ro">read-only</span>
    </div>
  </section>
</div>
<div class="toast" id="toast"></div>

<script>
"use strict";
let lastUpdate = null;
let queueFlowSnapshots = [];
let laneFlowSnapshots = {ats: [], compute: [], other: []};
const QUEUE_FLOW_MAX_SAMPLES = 16;
const QUEUE_FLOW_WINDOW_SECONDS = 600;

function esc(s){ return s==null ? "" : String(s).replace(/[&<>"]/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

function pct(v){
  if(v==null || isNaN(v)) return "—";
  return Math.round(Number(v) * 100) + "%";
}

function money(v){
  if(v==null || isNaN(v)) return "—";
  return "$" + Number(v).toFixed(2);
}

function displayMachineName(value){
  const raw = value == null ? "" : String(value).trim();
  if(!raw) return "(unknown)";
  const names = {
    "home": "Home",
    "m2": "TARPON",
    "m4": "GGGTower",
    "mac": "Paloma Mac",
    "mac-mac": "Paloma Mac",
  };
  return names[raw.toLowerCase()] || raw;
}

function machineName(obj, fallback){
  return displayMachineName((obj && obj.machine_display_name) || fallback);
}

function isUsefulHomeIp(ip){
  const value = String(ip || "").trim();
  return Boolean(value && value !== "0.0.0.0" && value !== "::" && value.toLowerCase() !== "unknown");
}

function formatMachineCounts(items){
  const counts = {};
  (items || []).forEach(item => {
    const name = machineName(item, item && item.machine_owner);
    counts[name] = (counts[name] || 0) + 1;
  });
  return Object.keys(counts).sort().map(name => counts[name] > 1 ? name + "×" + counts[name] : name).join(", ");
}

function openWorkerLogs(workerId){
  const sel = document.getElementById("logWorker");
  if(!sel) return false;
  if(workerId && !Array.from(sel.options).some(o => o.value === workerId)){
    const opt = document.createElement("option");
    opt.value = workerId;
    opt.textContent = workerId;
    sel.appendChild(opt);
  }
  sel.value = workerId || "";
  loadLogs();
  document.getElementById("logArea").scrollIntoView({block:"center"});
  return false;
}

function rel(iso){
  if(!iso) return "never";
  const t = new Date(iso).getTime();
  const d = Math.max(0, Math.round((Date.now()-t)/1000));
  if(d<60) return d+"s ago";
  if(d<3600) return Math.round(d/60)+"m ago";
  return Math.round(d/3600)+"h ago";
}

let modelTrendSnapshots = [];
const MODEL_TREND_MAX_SAMPLES = 12;
const MODEL_TREND_WINDOW_SECONDS = 900;

function familyChipHtml(f){
  const fam = esc(f || "unknown");
  const cls = (f || "unknown").toLowerCase().replace(/[^a-z0-9-]+/g, "");
  return '<span class="family-chip ' + esc(cls) + '">' + fam + '</span>';
}

function updateQueueFlowMetrics(queue){
  if(!queue) return {text: "—", hint: "no queue telemetry"};
  const now = Date.now();
  queueFlowSnapshots.push({
    at: now,
    queued: Number(queue.queued || 0),
    leased: Number(queue.leased || 0),
    applied: Number(queue.applied || 0),
    failed: Number(queue.failed || 0),
  });
  const cutoff = now - (QUEUE_FLOW_WINDOW_SECONDS * 1000);
  while(queueFlowSnapshots.length && queueFlowSnapshots[0].at < cutoff){
    queueFlowSnapshots.shift();
  }
  while(queueFlowSnapshots.length > QUEUE_FLOW_MAX_SAMPLES){
    queueFlowSnapshots.shift();
  }
  if(queueFlowSnapshots.length < 2){
    return {text: "bootstrapping", hint: "collecting queue samples"};
  }
  const first = queueFlowSnapshots[0];
  const last = queueFlowSnapshots[queueFlowSnapshots.length - 1];
  const elapsedMs = Math.max(1, last.at - first.at);
  const elapsedMin = elapsedMs / 60000.0;
  const leasedDelta = Math.max(0, last.leased - first.leased);
  const appliedDelta = Math.max(0, last.applied - first.applied);
  const queuedDelta = last.queued - first.queued;
  const leasedRate = leasedDelta / elapsedMin;
  const appliedRate = appliedDelta / elapsedMin;
  const queuedBacklog = last.queued;
  const queueSlope = queuedDelta === 0 ? "steady" : (queuedDelta < 0 ? "improving" : "growing");
  const qSuffix = queuedBacklog > 0 ? (" · backlog " + queuedBacklog) : "";
  const leasedFmt = Math.round(leasedRate * 10) / 10;
  const appliedFmt = Math.round(appliedRate * 10) / 10;
  return {
    text: "leases " + leasedFmt + "/m · applied " + appliedFmt + "/m",
    hint: queueSlope + " queue · " + (new Date(first.at).toISOString() + " to " + new Date(last.at).toISOString()) + qSuffix,
  };
}

function updateModelTrend(modelUsageEntries, modelText){
  const now = Date.now();
  modelTrendSnapshots.push({
    at: now,
    top: (modelText && modelText !== "—") ? modelText : null,
    items: modelUsageEntries.slice(0, 3),
  });
  const cutoff = now - (MODEL_TREND_WINDOW_SECONDS * 1000);
  while(modelTrendSnapshots.length && modelTrendSnapshots[0].at < cutoff){
    modelTrendSnapshots.shift();
  }
  while(modelTrendSnapshots.length > MODEL_TREND_MAX_SAMPLES){
    modelTrendSnapshots.shift();
  }
  if(!modelTrendSnapshots.length || modelTrendSnapshots.length === 1){
    return {
      text: modelText || "—",
      trend: "steady",
      window: "capturing",
    };
  }
  const first = modelTrendSnapshots[0].top || "";
  const last = modelTrendSnapshots[modelTrendSnapshots.length - 1].top || "";
  return {
    text: modelText || "—",
    trend: first && last && first !== last ? "trend: " + first.split(" ")[0] + " → " + last.split(" ")[0] : "steady",
    window: rel(new Date(modelTrendSnapshots[0].at).toISOString()) + " – now",
  };
}

function _queueFlowForLane(flowSamples, laneData, laneText){
  const queue = laneData || {};
  const now = Date.now();
  flowSamples.push({
    at: now,
    queued: Number(queue.queued || 0),
    leased: Number(queue.leased || 0),
    applied: Number(queue.applied || 0),
  });
  const cutoff = now - (QUEUE_FLOW_WINDOW_SECONDS * 1000);
  while(flowSamples.length && flowSamples[0].at < cutoff){
    flowSamples.shift();
  }
  while(flowSamples.length > QUEUE_FLOW_MAX_SAMPLES){
    flowSamples.shift();
  }
  if(flowSamples.length < 2){
    return {
      text: laneText ? (laneText + " — bootstrapping") : "bootstrapping",
      hint: "collecting queue samples",
    };
  }
  const first = flowSamples[0];
  const last = flowSamples[flowSamples.length - 1];
  const elapsedMin = Math.max(1, (last.at - first.at) / 60000.0);
  const leasedRate = Math.max(0, last.leased - first.leased) / elapsedMin;
  const appliedRate = Math.max(0, last.applied - first.applied) / elapsedMin;
  const queuedBacklog = last.queued;
  const qSuffix = queuedBacklog > 0 ? (" · backlog " + queuedBacklog) : "";
  const leasedFmt = Math.round(leasedRate * 10) / 10;
  const appliedFmt = Math.round(appliedRate * 10) / 10;
  return {
    text: "leases " + leasedFmt + "/m · applied " + appliedFmt + "/m",
    hint: (laneText ? laneText : "lane") + " " + qSuffix + " (" +
      (new Date(first.at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) + " to " +
      new Date(last.at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })) + ")",
  };
}

function updateLaneFlowMetrics(key, laneData){
  if(!key) return {text: "—", hint: "no lane telemetry"};
  if(!laneFlowSnapshots[key]){
    laneFlowSnapshots[key] = [];
  }
  return _queueFlowForLane(laneFlowSnapshots[key], laneData, key.toUpperCase());
}

function laneRateText(d){
  const lane = d || {};
  const queued = Number(lane.queued || 0);
  const leased = Number(lane.leased || 0);
  const applied = Number(lane.applied || 0);
  const conversion = queued > 0 ? Math.round((leased / queued) * 100) + "%" : "—";
  return `${queued}/${leased}/${applied} · lease ${conversion}`;
}

function vendorChipHtml(v){
  const vendor = esc(v || "unknown");
  const cls = (v || "unknown").toLowerCase().replace(/[^a-z0-9-]+/g, "");
  return '<span class="vendor-chip ' + esc(cls) + '">' + vendor + '</span>';
}

function toast(msg, kind){
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = "toast show " + (kind||"");
  clearTimeout(el._t);
  el._t = setTimeout(()=>{ el.className = "toast " + (kind||""); }, 5000);
}

async function loadDiagnosis(){
  try{
    const r = await fetch("/api/diagnosis", {cache:"no-store"});
    if(!r.ok) return;
    diagnosisSnapshot = await r.json();
    renderDiagnosis(diagnosisSnapshot);
    renderApplyReadiness();
  }catch(e){}
}

async function loadAgents(){
  try{
    const r = await fetch("/api/agents", {cache:"no-store"});
    if(!r.ok) return;
    agentsSnapshot = await r.json();
    renderAgents(agentsSnapshot);
    renderApplyReadiness();
  }catch(e){}
}

function renderDiagnosis(d){
  const q = d.queue || {};
  const ats = q.ats || {};
  const li = q.linkedin || {};
  const roll = d.rollups || {};
  const state = q.state || {};
  document.getElementById("stateHeadline").textContent = state.code || "unknown";
  document.getElementById("stateReason").textContent = state.reason || "";
  const recs = d.recommendations || [];
  document.getElementById("nextAction").textContent = recs.length
    ? "Recommended Next Action: " + recs[0].title + " — " + recs[0].reason
    : "Recommended Next Action: No recommendation";
  renderPrimaryBlocker();
  renderRecommendationList(recs);
  document.getElementById("whyBody").innerHTML = [
    ["ATS queued", ats.queued],
    ["ATS approved", ats.approved],
    ["ATS leaseable", ats.leaseable],
    ["ATS dedup-blocked", ats.dedup_blocked],
    ["LinkedIn queued", li.queued],
    ["LinkedIn leaseable", li.leaseable],
    ["LinkedIn canary exhausted", li.canary_exhausted ? "yes" : "no"]
  ].map(([k,v]) => '<div class="mini"><span>'+esc(k)+'</span><b>'+esc(v)+'</b></div>').join("");
  const browser = d.browser || {};
  const counts = browser.counts || {};
  const examples = browser.examples || {};
  const keys = Object.keys(counts);
  document.getElementById("browserBody").innerHTML = keys.length
    ? keys.map(k => {
        const ex = examples[k] || {};
        const detail = ex.worker_id
          ? '<small>'+esc(machineName(ex, ex.machine_owner))+' / '+esc(ex.worker_id)+
            ' · <a href="'+esc(ex.logs_url||"#")+'" onclick="return openWorkerLogs('+
            esc(JSON.stringify(String(ex.worker_id)))+')">logs</a></small>'
          : "";
        return '<div class="mini"><span>'+esc(k)+'</span><b>'+esc(counts[k])+'</b>'+detail+'</div>';
      }).join("")
    : '<div class="mut">no classified browser failures</div>';
  renderBrowserWallQueue(browser.wall_queue || []);
  document.getElementById("funnelBody").innerHTML = [
    ["Queued", ats.queued],
    ["Approved", ats.approved],
    ["Leaseable", ats.leaseable],
    ["Leased", ats.leased],
    ["Applied", ats.applied],
    ["Failed", ats.failed],
    ["Crash unconfirmed", ats.crash_unconfirmed]
  ].map(([k,v]) => '<div class="fstep"><span>'+esc(k)+'</span><b>'+esc(v)+'</b></div>').join("");
  renderThroughput(roll.throughput || {}, roll.freshness || {}, roll.daily_goal || {});
  renderHostQuality(roll.host_quality || []);
  const machines = roll.machines || {};
  document.getElementById("machineMap").innerHTML = Object.keys(machines).length
    ? Object.keys(machines).map(k => '<div class="mini"><span>'+esc(machines[k].display_name || k)+'</span><b>'+
      esc(machines[k].workers)+'</b><small class="mut">'+esc(machines[k].alive_workers||0)+
      ' alive / '+esc(machines[k].stale_workers||0)+' stale</small></div>').join("")
    : '<div class="mut">no machine heartbeats</div>';
  renderWorkerComparison(roll.worker_comparison || []);
}

function renderPrimaryBlocker(){
  const el = document.getElementById("primaryBlocker");
  if(!el) return;
  const s = statusSnapshot || {};
  const d = diagnosisSnapshot || {};
  const gate = s.gate || {};
  const q = d.queue || s.fleet_diagnosis || {};
  const ats = q.ats || {};
  const linkedin = q.linkedin || {};
  const doctor = s.doctor || {};
  const agents = d.agents || {};
  const agentVerdict = agents.verdict || {};
  let state = "ok";
  let text = "Primary blocker: No blocking gate";
  if(agentVerdict.code === "all_agents_blocked"){
    state = "severe";
    const blocked = Object.keys(agents.availability || {})
      .filter(a => (agents.availability[a] || {}).blocked)
      .map(a => a + "(until " + ((agents.availability[a] || {}).blocked_until || "unknown") + ")")
      .join(", ");
    text = "Primary blocker: Apply agents blocked — " +
      (blocked || "all known agents currently blocked") + "; unblock before applying";
  } else if(gate.should_halt){
    state = "severe";
    text = "Primary blocker: Spend cap halt — " +
      money(gate.spent_usd || 0) + " spent over cap " + money(gate.spend_cap_usd || 0);
  } else if(gate.paused || ats.paused){
    state = "severe";
    text = "Primary blocker: Fleet pause — shared pause is active";
  } else if(ats.ats_paused || doctor.ats_paused){
    state = "warn";
    text = "Primary blocker: ATS pause — source " + (doctor.ats_pause_source || "unknown");
  } else if(Number(ats.approved || 0) > 0 && Number(ats.leaseable || 0) === 0){
    state = "warn";
    text = "Primary blocker: ATS lease gate — " + ats.approved + " approved but 0 leaseable";
  } else if(linkedin.owner_ip_context_known && !linkedin.owner_ip_ready){
    state = "warn";
    text = "Primary blocker: LinkedIn owner IP — blocked";
  } else if(linkedin.canary_exhausted){
    state = "warn";
    text = "Primary blocker: LinkedIn canary gate — exhausted";
  } else if(Number(linkedin.queued || 0) > 0 && Number(linkedin.leaseable || 0) === 0){
    state = "warn";
    text = "Primary blocker: LinkedIn lease gate — " +
      "queued but not leaseable";
  }
  el.textContent = text;
  el.className = "actionline blockerline " + state;
}

function renderAgents(d){
  const verdict = d.verdict || {};
  document.getElementById("agentVerdict").textContent = (verdict.code || "unknown") + " — " + (verdict.reason || "");
  renderAgentSummary(d);
  const rows = d.workers || [];
  const body = document.getElementById("agentWorkers");
  body.innerHTML = rows.length ? rows.map(w =>
    '<tr><td>'+esc(w.worker_id)+'</td><td>'+esc(w.machine_display_name||w.machine_owner||"")+'</td><td>'+
    esc(w.current_agent||"unknown")+'</td><td>'+esc(w.current_model||"unknown")+'</td><td>'+
    esc(w.agent_chain||"")+'</td><td>'+esc(w.sw_version||"unknown")+'</td><td>'+
    esc(w.last_agent_switch_reason||"")+'</td></tr>'
  ).join("") : '<tr><td colspan="7" class="mut">no apply worker agent telemetry</td></tr>';
}

function renderAgentSummary(d){
  const el = document.getElementById("agentSummary");
  if(!el) return;
  const workers = d.workers || [];
  const verdict = d.verdict || {};
  const availability = d.availability || {};
  const spend = d.spend_24h || [];
  const modelCounts = {};
  workers.forEach(w => {
    if(w.current_agent || w.current_model){
      const key = (w.current_agent || "unknown") + " / " + (w.current_model || "model unknown");
      modelCounts[key] = (modelCounts[key] || 0) + 1;
    }
  });
  const modelKeys = Object.keys(modelCounts);
  const modelText = modelKeys.length
    ? modelKeys.map(k => k + "×" + modelCounts[k]).join(", ")
    : "unknown";
  const switched = workers.filter(w =>
    w.last_agent_switch_reason || String(w.agent_chain || "").indexOf(">") >= 0
  ).length;
  const telemetryMissing = verdict.code === "telemetry_missing" || !modelKeys.length;
  const switchingText = telemetryMissing ? "not observable" : (switched ? "observed" : "not observed");
  const cards = [
    ["Model in use", modelText,
      modelKeys.length ? "current worker heartbeat telemetry" : "workers have not reported agent/model telemetry"],
    ["Dynamic switching", switchingText,
      telemetryMissing ? "redeploy/restart workers before judging switching" : (switched ? switched+" worker(s) show switch evidence" : "no switch rows reported")],
  ];
  Object.keys(availability).sort().forEach(agent => {
    const row = availability[agent] || {};
    cards.push([
      "Agent " + agent,
      row.blocked ? "blocked" : "ready",
      row.blocked ? ((row.reason || "blocked") + (row.blocked_until ? " until " + row.blocked_until : "")) : "not currently blocked",
    ]);
  });
  if(spend.length){
    spend.forEach(row => cards.push([
      "24h spend " + (row.provider || "unknown"),
      money(row.cost_usd || 0),
      (row.count || 0) + " call(s)" + (row.model ? " / " + row.model : " / model unknown"),
    ]));
  } else {
    cards.push(["24h spend", "$0.00", "no apply-agent usage recorded"]);
  }
  el.innerHTML = cards.map(([label, value, hint]) =>
    '<div class="mini"><span>'+esc(label)+'</span><b>'+esc(value)+'</b><small>'+esc(hint)+'</small></div>'
  ).join("");
}

function renderRecommendationList(recs){
  const el = document.getElementById("recommendationList");
  if(!el) return;
  if(!recs.length){
    el.innerHTML = '<div class="mut">no current recommendations</div>';
    return;
  }
  el.innerHTML = recs.map(r => {
    const sev = r.severity || "info";
    return '<div class="action-item '+esc(sev)+'"><div class="k">'+
      esc([sev, r.lane, r.action_type].filter(Boolean).join(" / "))+'</div><div class="t">'+
      esc(r.title || r.code || "Recommendation")+'</div><div class="r">'+
      esc(r.reason || "")+'</div><div class="cmd"><b>Operator step:</b> '+
      esc(r.command || "Review the linked runbook before taking action.")+'</div></div>';
  }).join("");
}

function renderBrowserWallQueue(rows){
  const el = document.getElementById("browserWallQueue");
  if(!el) return;
  if(!rows.length){
    el.innerHTML = '<div class="mut">no browser/login/captcha walls in worker logs</div>';
    return;
  }
  el.innerHTML = '<div class="table-scroll"><table><thead><tr>'+
    '<th>Kind</th><th>Machine</th><th>Worker</th><th>Sample</th><th>Action</th><th>Logs</th>'+
    '</tr></thead><tbody>' + rows.map(r =>
      '<tr><td>'+esc(r.kind||"unknown")+'</td><td>'+esc(r.machine_display_name||r.machine_owner||"")+
      '</td><td>'+esc(r.worker_id||"")+'</td><td>'+esc(r.sample||"")+'</td><td>'+
      esc(r.action||"Open logs and inspect before scaling applies.")+'</td><td><a href="'+
      esc(r.logs_url||"#")+'" onclick="return openWorkerLogs('+esc(JSON.stringify(String(r.worker_id||"")))+')">logs</a></td></tr>'
    ).join("") + '</tbody></table></div>';
}

function renderWorkerComparison(rows){
  const body = document.getElementById("workerComparisonRows");
  if(!body) return;
  body.innerHTML = rows.length ? rows.map(w =>
    '<tr><td>'+esc(w.worker_id)+'</td><td>'+esc(w.applied||0)+'</td><td>'+esc(w.total||0)+
    '</td><td>'+esc(pct(w.success_rate))+'</td><td>'+esc(pct(w.crash_rate))+
    '</td><td>'+esc(money(w.cost_per_applied))+'</td></tr>'
  ).join("") : '<tr><td colspan="6" class="mut">no worker outcome data yet</td></tr>';
}

function renderHostQuality(rows){
  const body = document.getElementById("hostQualityRows");
  if(!body) return;
  if(!rows.length){
    body.innerHTML = '<tr><td colspan="6" class="mut">no host outcome data yet</td></tr>';
    return;
  }
  body.innerHTML = rows.map(h => {
    const total = Number(h.total || 0);
    const bad = Number(h.failed || 0) + Number(h.challenges || 0);
    const badRate = total ? (bad / total) : null;
    return '<tr><td>'+esc(h.host || "(unknown)")+'</td><td>'+esc(total)+'</td><td>'+
      esc(h.applied || 0)+'</td><td>'+esc(h.failed || 0)+'</td><td>'+
      esc(h.challenges || 0)+'</td><td>'+esc(pct(badRate))+'</td></tr>';
  }).join("");
}

function renderThroughput(throughput, freshness, dailyGoal){
  const grid = document.getElementById("throughputGrid");
  if(!grid) return;
  const configured = Boolean(dailyGoal && dailyGoal.configured);
  const goalValue = configured ? String(dailyGoal.applied_today || 0) + " / " + String(dailyGoal.target || 0) : "not set";
  const goalHint = configured
    ? String(dailyGoal.remaining == null ? "unknown" : dailyGoal.remaining) + " remaining today"
    : "daily apply target not configured";
  const cards = [
    ["Last apply", freshness.last_apply_at ? rel(freshness.last_apply_at) : "never",
      freshness.last_apply_at || "no applied timestamp recorded"],
    ["Applied 1h", throughput.applied_1h == null ? "—" : throughput.applied_1h,
      "recent confirmed applies"],
    ["Applied 24h", throughput.applied_24h == null ? "—" : throughput.applied_24h,
      "confirmed applies in the last day"],
    ["Daily goal", goalValue, goalHint],
  ];
  grid.innerHTML = cards.map(([label, value, hint]) =>
    '<div class="mini"><span>'+esc(label)+'</span><b>'+esc(value)+'</b><small>'+esc(hint)+'</small></div>'
  ).join("");
}

function renderApplyReadiness(){
  const checksEl = document.getElementById("applyReadinessChecks");
  const verdictEl = document.getElementById("applyReadinessVerdict");
  const summaryEl = document.getElementById("applyReadinessSummary");
  if(!checksEl || !verdictEl || !summaryEl) return;
  const s = statusSnapshot || {};
  const d = diagnosisSnapshot || {};
  const agents = agentsSnapshot || {};
  const q = d.queue || s.fleet_diagnosis || {};
  const ats = q.ats || {};
  const linkedin = q.linkedin || {};
  const gate = s.gate || {};
  const doctor = s.doctor || {};
  const deployment = s.deployment || {};
  const browser = d.browser || {};
  const browserCounts = browser.counts || {};
  const wallRows = browser.wall_queue || [];
  const rollups = d.rollups || {};
  const dailyGoal = rollups.daily_goal || {};
  const versions = deployment.worker_versions || [];
  const applyWorkers = (s.workers || []).filter(w => (w.role || "apply") === "apply");
  const staleApply = applyWorkers.filter(w => w.alive === false);
  const staleActiveApply = staleApply.filter(w => ["applying", "challenge_pending"].includes(w.state || ""));
  const agentWorkers = agents.workers || [];
  const availability = agents.availability || {};
  const availabilityNames = Object.keys(availability);
  const blockedAgents = availabilityNames.filter(name => availability[name] && availability[name].blocked);
  const readyAgents = availabilityNames.filter(name => availability[name] && !availability[name].blocked);
  const agentVerdict = (agents.verdict || {}).code || "";
  const telemetryMissing = (agents.verdict || {}).code === "telemetry_missing" ||
    (agentWorkers.length > 0 && !agentWorkers.some(w => w.current_agent || w.current_model));
  const dirtyOrUnknown = versions.some(row => {
    const v = String(row.version || "(unknown)");
    return v === "(unknown)" || v.indexOf(".dirty") >= 0 || v.indexOf("-dirty") >= 0;
  });
  const mixedVersions = versions.length > 1;
  const paused = Boolean(gate.paused || gate.should_halt || ats.paused || ats.ats_paused || doctor.ats_paused);
  const leaseable = Number(ats.leaseable || 0);
  const approved = Number(ats.approved || 0);
  const linkedinLeaseable = Number(linkedin.leaseable || 0);
  const linkedinQueued = Number(linkedin.queued || 0);
  const linkedinOwnerBlocked = linkedin.owner_ip_context_known && !linkedin.owner_ip_ready;
  const linkedinCanaryExhausted = Boolean(linkedin.canary_exhausted);
  const authKinds = ["login_gate", "captcha", "email_otp"];
  const backendKinds = ["browser_backend_crashed", "browser_service_unavailable", "browser_server_unavailable"];
  const authWallCount = authKinds.reduce((total, kind) => total + Number(browserCounts[kind] || 0), 0) ||
    wallRows.filter(row => authKinds.includes(row.kind || "")).length;
  const browserBackendCount = backendKinds.reduce((total, kind) => total + Number(browserCounts[kind] || 0), 0) ||
    wallRows.filter(row => backendKinds.includes(row.kind || "")).length;
  const goalConfigured = Boolean(dailyGoal.configured);
  const leaseableHint = leaseable > 0
    ? approved + " approved and leaseable"
    : (approved > 0 ? approved + " approved but not leaseable" : "no approved ATS queue ready");
  const staleMachineSummary = formatMachineCounts(staleApply);
  const staleActiveMachineSummary = formatMachineCounts(staleActiveApply);
  const checks = [
    ["Pause gates", paused ? "blocked" : "ok", paused ? "blocked" : "clear",
      paused ? "shared or ATS pause is active" : "no pause gate reported"],
    ["Leaseable queue", leaseable > 0 ? "ok" : "blocked", leaseable,
      leaseableHint],
    ["LinkedIn readiness",
      linkedinLeaseable > 0 ? "ok" : ((linkedinQueued || linkedinOwnerBlocked || linkedinCanaryExhausted) ? "warn" : "ok"),
      linkedinLeaseable > 0 ? linkedinLeaseable + " leaseable" : (linkedinQueued ? linkedinQueued + " queued" : "idle"),
      linkedinOwnerBlocked ? "owner IP blocked" : (linkedinCanaryExhausted ? "canary exhausted" : (linkedinQueued && !linkedinLeaseable ? "queued but not leaseable" : "catastrophe lane ready"))],
    ["Worker versions", (mixedVersions || dirtyOrUnknown) ? "warn" : "ok",
      versions.length ? versions.length + " group(s)" : "unknown",
      (mixedVersions || dirtyOrUnknown) ? "reconcile dirty/unknown workers before scale" : "one reported worker build"],
    ["Model telemetry", telemetryMissing ? "warn" : "ok",
      telemetryMissing ? "missing" : "reported",
      telemetryMissing ? "cannot prove Codex/Claude model switching yet" : "workers report agent/model fields"],
    ["Agent failover",
      blockedAgents.length && !readyAgents.length ? "blocked" : ((blockedAgents.length || agentVerdict === "partial") ? "warn" : "ok"),
      blockedAgents.length ? blockedAgents.length + " blocked provider(s)" : (readyAgents.length ? readyAgents.join(", ") + " ready" : "loading"),
      blockedAgents.length
        ? (readyAgents.length ? readyAgents.join(", ") + " still available" : "no provider still available")
        : (readyAgents.length ? "no blocked provider reported" : "waiting for provider availability")],
    ["Browser auth walls", authWallCount ? "warn" : "ok", authWallCount,
      authWallCount ? "login/captcha/OTP needs operator review" : "no auth wall samples"],
    ["Browser backend", browserBackendCount ? "warn" : "ok", browserBackendCount,
      browserBackendCount ? "restart browser backend or worker service" : "no backend crash/server samples"],
    ["Stale workers", staleApply.length ? "warn" : "ok", staleApply.length,
      staleApply.length ? staleMachineSummary : "all apply heartbeats fresh"],
    ["Stale active workers", staleActiveApply.length ? "warn" : "ok", staleActiveApply.length,
      staleActiveApply.length ? staleActiveMachineSummary + " still marked active" : "no stale applying/challenge workers"],
    ["Daily goal", goalConfigured ? "ok" : "warn",
      goalConfigured ? String(dailyGoal.applied_today || 0) + " / " + String(dailyGoal.target || 0) : "not set",
      goalConfigured ? String(dailyGoal.remaining == null ? "unknown" : dailyGoal.remaining) + " remaining today" : "set a daily target before unattended scale"],
  ];
  const blockers = checks.filter(c => c[1] === "blocked").length;
  const warnings = checks.filter(c => c[1] === "warn").length;
  verdictEl.textContent = blockers
    ? "Not ready to apply"
    : (warnings ? "Unsafe to scale" : "Ready to apply carefully");
  summaryEl.textContent = blockers
    ? blockers + " blocking gate(s), " + warnings + " scale risk(s)"
    : (warnings
      ? "apply lane has queue, but " + warnings + " scale risk(s) remain"
      : "all readiness checks clear; keep canary and daily caps in place");
  checksEl.innerHTML = checks.map(([label, state, value, hint]) =>
    '<div class="ready-check '+esc(state)+'"><div class="name">'+esc(label)+
    '</div><div class="value">'+esc(value)+'</div><div class="hint">'+esc(hint)+'</div></div>'
  ).join("");
}

function renderDiscoveryBacklog(s){
  const grid = document.getElementById("discoveryBacklogGrid");
  if(!grid) return;
  const d = (s && s.discovery) || null;
  if(!d){
    grid.innerHTML = '<div class="mut">no discovery telemetry reported</div>';
    return;
  }
  const tasks = d.tasks || {};
  const postings = d.postings || {};
  const workers = d.workers || [];
  const pending = Number(postings.pending_ingest || 0);
  const last24 = Number(postings.last24h || 0);
  const due = Number(tasks.due_now || 0);
  const alive = workers.filter(w => w.alive).length;
  const latest = (d.recent && d.recent.length && d.recent[0].time) ? rel(d.recent[0].time) : "none";
  const pressure = pending >= 500 ? "high" : (pending >= 100 ? "watch" : (pending > 0 ? "low" : "clear"));
  const cards = [
    ["Pending ingest", pending, pending ? "postings staged but not imported" : "no ingest backlog"],
    ["Ingest pressure", pressure, due ? due + " search task(s) due now" : "no search tasks due now"],
    ["Discovery workers", alive + " / " + workers.length, alive ? "heartbeat active" : "no live discovery worker"],
    ["Found 24h", last24, "latest find " + latest],
  ];
  grid.innerHTML = cards.map(([label, value, hint]) =>
    '<div class="mini"><span>'+esc(label)+'</span><b>'+esc(value)+'</b><small>'+esc(hint)+'</small></div>'
  ).join("");
}

function renderMachineNetwork(s){
  const body = document.getElementById("machineNetworkRows");
  if(!body) return;
  const machines = {};
  function addWorker(w, fallbackRole){
    if(!w) return;
    const machine = w.machine_display_name || w.machine_owner || "(unknown)";
    const row = machines[machine] || {
      machine,
      ips: [],
      roles: {},
      alive: 0,
      stale: 0,
      workers: [],
    };
    const ip = w.home_ip || "";
    if(isUsefulHomeIp(ip) && row.ips.indexOf(ip) < 0) row.ips.push(ip);
    const role = w.role || fallbackRole || "worker";
    row.roles[role] = (row.roles[role] || 0) + 1;
    if(w.alive) row.alive += 1; else row.stale += 1;
    if(w.worker_id) row.workers.push(w.worker_id);
    machines[machine] = row;
  }
  (s.workers || []).forEach(w => addWorker(w, "apply"));
  (((s.discovery || {}).workers) || []).forEach(w => addWorker(w, "discovery"));
  const rows = Object.keys(machines).sort().map(k => machines[k]);
  if(!rows.length){
    body.innerHTML = '<tr><td colspan="5" class="mut">no machine network heartbeats</td></tr>';
    return;
  }
  body.innerHTML = rows.map(row => {
    const roles = Object.keys(row.roles).sort().map(role => role + "×" + row.roles[role]).join(", ");
    return '<tr><td>'+esc(row.machine)+'</td><td>'+esc(row.ips.join(", ") || "unknown")+
      '</td><td>'+esc(roles || "unknown")+'</td><td>'+esc(row.alive + " alive / " + row.stale + " stale")+
      '</td><td>'+esc(row.workers.join(", "))+'</td></tr>';
  }).join("");
}

function renderStaleWorkers(workers){
  const el = document.getElementById("staleWorkers");
  if(!el) return;
  const stale = (workers || []).filter(w => !w.alive);
  if(!stale.length){
    el.innerHTML = '<div class="mut">no stale apply workers</div>';
    return;
  }
  el.innerHTML = '<div class="table-scroll"><table><thead><tr>'+
    '<th>Worker</th><th>Machine</th><th>Last beat</th><th>State</th><th>Version</th>'+
    '</tr></thead><tbody>' + stale.map(w =>
      '<tr><td>'+esc(w.worker_id)+'</td><td>'+esc(w.machine_display_name||w.machine_owner||"")+
      '</td><td>'+esc(w.last_beat ? rel(w.last_beat) : "never")+
      '</td><td>'+esc(w.state||"")+'</td><td>'+esc(w.sw_version||"unknown")+'</td></tr>'
    ).join("") + '</tbody></table></div>';
}

function classifyBuild(version){
  const v = String(version || "(unknown)");
  if(v === "(unknown)") return "unknown build";
  if(v.indexOf(".dirty") >= 0 || v.indexOf("-dirty") >= 0) return "dirty build";
  return "reported build";
}

function renderDeploymentDrift(dep){
  const rowsEl = document.getElementById("deploymentDriftRows");
  const summaryEl = document.getElementById("deploymentDriftSummary");
  if(!rowsEl || !summaryEl) return;
  const versions = (dep && dep.worker_versions) || [];
  if(!versions.length){
    summaryEl.textContent = "No worker versions reported.";
    rowsEl.innerHTML = '<tr><td colspan="6" class="mut">no worker version telemetry</td></tr>';
    return;
  }
  const total = versions.reduce((n, row) => n + Number(row.workers || 0), 0);
  const dirty = versions.filter(row => classifyBuild(row.version) === "dirty build")
    .reduce((n, row) => n + Number(row.workers || 0), 0);
  const unknown = versions.filter(row => classifyBuild(row.version) === "unknown build")
    .reduce((n, row) => n + Number(row.workers || 0), 0);
  const consoleInfo = dep && dep.console || {};
  const targetCommit = String(consoleInfo.commit || "").toLowerCase();
  const targetShort = targetCommit ? targetCommit.slice(0, 7) : "";
  const targetLabel = (consoleInfo.branch || "branch") + "@" +
    (targetCommit ? targetCommit.slice(0, 7) : "unknown");
  const targetHits = targetShort
    ? versions.filter(row => String(row.version || "").toLowerCase().indexOf("+git.tree." + targetShort) >= 0).reduce((n, row) => n + Number(row.workers || 0), 0)
    : 0;
  summaryEl.textContent = total + " worker heartbeat(s) across " + versions.length +
    " version group(s) — target " + targetLabel + "; " +
    (targetShort ? (targetHits + " on-target / " + total + " reported") : "target not available") +
    (targetShort && targetHits < total ? " - reconcile these machines before comparing telemetry." : "") +
    (dirty || unknown ? " · reconcile mixed/dirty groups before scale" : "");
  rowsEl.innerHTML = versions.map(row => {
    const status = classifyBuild(row.version);
    const onTarget = targetShort && String(row.version || "").toLowerCase().indexOf("+git.tree." + targetShort) >= 0;
    const alignment = onTarget ? "on target" : "off target";
    return '<tr><td>'+esc(row.version || "(unknown)")+'</td><td>'+esc(row.workers || 0)+
      '</td><td>'+esc((row.machines || []).join(", ") || "(unknown)")+'</td><td>'+
      esc((row.worker_ids || []).join(", ") || "(none)")+'</td><td>'+esc(status)+
      '</td><td>'+esc(alignment)+'</td></tr>';
  }).join("");
}

function renderLaneState(s){
  const grid = document.getElementById("laneStateGrid");
  if(!grid) return;
  const diag = s.fleet_diagnosis || {};
  const ats = diag.ats || {};
  const linkedin = diag.linkedin || {};
  const gate = s.gate || {};
  const statusLinkedin = s.linkedin || {};
  const doctor = s.doctor || {};
  const sharedPaused = Boolean(gate.paused || ats.paused);
  const atsPaused = Boolean(ats.ats_paused || doctor.ats_paused);
  const ownerKnown = linkedin.owner_ip_context_known;
  const ownerReady = linkedin.owner_ip_ready;
  const linkedinCanaryEnabled = linkedin.canary_enabled != null
    ? linkedin.canary_enabled : statusLinkedin.canary_enabled;
  const linkedinCanaryRemaining = linkedin.canary_remaining;
  const linkedinCanary = linkedinCanaryEnabled
    ? (linkedinCanaryRemaining == null ? "armed" : linkedinCanaryRemaining)
    : "off";
  const cells = [
    ["Shared pause", sharedPaused ? "on" : "off",
      sharedPaused ? "apply lane halted; compute/discovery may continue" : "shared kill switch clear"],
    ["ATS pause", atsPaused ? "on" : "off",
      atsPaused ? "source " + (doctor.ats_pause_source || "unknown") : "ATS pause clear"],
    ["ATS leaseable", ats.leaseable == null ? "—" : ats.leaseable,
      "approved, canary-safe, not deduped"],
    ["LinkedIn owner IP", ownerKnown ? (ownerReady ? "ready" : "blocked") : "unknown",
      ownerKnown ? "owner-IP context checked" : "owner-IP context unavailable"],
    ["LinkedIn canary", linkedinCanary,
      linkedin.canary_exhausted ? "exhausted; re-arm before lease" : "catastrophe lane cap"],
  ];
  grid.innerHTML = cells.map(([label, value, hint]) =>
    '<div class="mini"><span>'+esc(label)+'</span><b>'+esc(value)+'</b><small>'+esc(hint)+'</small></div>'
  ).join("");
}

function renderLaneActivity(s){
  const grid = document.getElementById("laneActivityGrid");
  if(!grid) return;
  const diag = s.fleet_diagnosis || {};
  const ats = diag.ats || {};
  const liDiag = diag.linkedin || {};
  const gate = s.gate || {};
  const q = (s.queue && s.queue.apply) || {};
  const recent = s.recent || [];
  const discovery = s.discovery || {};
  const discoveryWorkers = discovery.workers || [];
  const linkedin = s.linkedin || {};
  const computeRecent = recent.filter(r => r.lane === "compute").length;
  const applyStatus = (gate.paused || ats.paused || ats.ats_paused || gate.should_halt)
    ? "halted" : (Number(ats.leaseable || 0) > 0 ? "ready" : "idle");
  const computeStatus = computeRecent ? "active" : "idle";
  const discoveryAlive = discoveryWorkers.filter(w => w.alive).length;
  const discoveryStatus = discoveryAlive ? "active" : "idle";
  const linkedinStatus = linkedin.halted ? "halted" : (Number(liDiag.leaseable || 0) > 0 ? "ready" : "limited");
  const cards = [
    ["Apply lane", applyStatus,
      (ats.leaseable == null ? "leaseable unknown" : ats.leaseable + " leaseable") +
      " / " + (q.queued || 0) + " queued"],
    ["Compute lane", computeStatus,
      computeRecent + " recent compute result(s)"],
    ["Discovery lane", discoveryStatus,
      discoveryAlive + " worker(s) alive / " + (((discovery.tasks || {}).due_now) || 0) + " due search task(s)"],
    ["LinkedIn lane", linkedinStatus,
      (liDiag.leaseable == null ? "leaseable unknown" : liDiag.leaseable + " leaseable") +
      " / " + (linkedin.queued || liDiag.queued || 0) + " queued"],
  ];
  grid.innerHTML = cards.map(([label, value, hint]) =>
    '<div class="mini"><span>'+esc(label)+'</span><b>'+esc(value)+'</b><small>'+esc(hint)+'</small></div>'
  ).join("");
}

function renderAudit(d){
  const body = document.getElementById("auditRows");
  const rows = (d && d.rows) || [];
  if(!rows.length){
    body.innerHTML = '<tr><td colspan="4" class="mut">no audit entries yet</td></tr>';
    return;
  }
  body.innerHTML = rows.map(r =>
    '<tr><td class="mut">'+esc(rel(r.time))+'</td><td>'+esc(r.action||"—")+
    '</td><td>'+esc(r.result || (r.ok ? "ok" : "failed"))+'</td><td>'+
    esc(r.message||"—")+'</td></tr>'
  ).join("");
}

async function loadAudit(){
  const body = document.getElementById("auditRows");
  try{
    const r = await fetch("/api/audit", {cache:"no-store"});
    if(!r.ok){ body.innerHTML = '<tr><td colspan="4" class="mut">audit unavailable</td></tr>'; return; }
    renderAudit(await r.json());
  }catch(e){
    body.innerHTML = '<tr><td colspan="4" class="mut">audit unavailable</td></tr>';
  }
}

function render(s){
  statusSnapshot = s;
  lastUpdate = Date.now();
  document.getElementById("conn").textContent = "connected";
  const g = s.gate, q = s.queue.apply, li = s.linkedin, doc = s.doctor || {};
  const state = s.apply_state || {};

  // DeadMan: a RED fixed banner when fleet_config.deadman_alert is set (silent
  // fleet death / stall / running-hot) so the owner sees it even glancing at a phone.
  const dmBanner = document.getElementById("deadmanBanner");
  if(dmBanner){
    const deadman = s.deadman || {};
    const dmTitle = (deadman.title || deadman.alert || s.deadman_alert || "deadman alert");
    const dmReason = deadman.reason || (s.deadman_alert_at ? ("active at " + rel(s.deadman_alert_at)) : "");
    if(deadman.active || s.deadman_alert){
      dmBanner.textContent = "DEADMAN ALERT: " + dmTitle + (dmReason ? " — " + dmReason : "") + " (" + rel(s.deadman_alert_at) + ")";
      dmBanner.className = "deadmanbanner show";
    } else {
      dmBanner.textContent = "";
      dmBanner.className = "deadmanbanner";
    }
  }
  // H8: an ATS-only Doctor pause (ats_paused, source='doctor') is a distinct, milder state than a
  // full operator/cost halt -- label it so the operator knows the LinkedIn lane is untouched and
  // that it auto-reverts. The shared paused/spend-cap halt still shows as the hard PAUSE/HALTED.
  const atsPause = !!(doc.ats_paused || g.ats_paused);
  const atsPauseSource = doc.ats_pause_source || g.ats_pause_source || "operator";
  const paused = g.paused || g.should_halt;
  const banner = document.getElementById("banner");
  const stateSeverity = (state.severity || "warn").toLowerCase();
  const bannerState = (stateSeverity === "ok" ? "run" : (stateSeverity === "halted" ? "pause" : "warn"));
  banner.className = "banner " + bannerState;
  document.getElementById("bannerText").textContent =
    paused ? (g.should_halt && !g.paused ? "HALTED (spend cap)" : "PAUSED")
           : (atsPause ? ("ATS PAUSED (" + atsPauseSource + ")") : "RUNNING");
  document.getElementById("updated").textContent =
    "updated " + rel(s.now) + (g.should_halt ? " • halt active"
      : atsPause ? " • LinkedIn lane UNAFFECTED" : "");

  const statusRecommendation = s.recommendation || {};
  const statusRecommendationEl = document.getElementById("stateRecommendation");
  if(statusRecommendationEl && statusRecommendation.title){
    statusRecommendationEl.className = "mut";
    statusRecommendationEl.textContent =
      statusRecommendation.title + " — " + (statusRecommendation.reason || "review recommended action.");
  }

  // H18: above-the-fold Doctor card + a toast when a NEW auto-fix appears between polls.
  const dc = document.getElementById("cDoctor");
  if(dc){
    dc.textContent = (doc.active_auto_fix_count!=null) ? doc.active_auto_fix_count : "—";
    document.getElementById("cDoctorHint").textContent =
      "active • last pass " + rel(doc.last_pass_at);
    if(window._lastDoctorFixId!=null && doc.newest_auto_fix_id!=null
       && doc.newest_auto_fix_id > window._lastDoctorFixId){
      toast("Fleet Doctor applied a new auto-fix", "ok");
    }
    if(doc.newest_auto_fix_id!=null) window._lastDoctorFixId = doc.newest_auto_fix_id;
  }

  document.getElementById("cCanary").textContent =
    g.canary_enabled ? (g.canary_remaining==null ? "on" : g.canary_remaining) : "off";
  document.getElementById("cCanaryHint").textContent =
    g.canary_enabled ? "remaining (auto-pause at 0)" : "disabled (unbounded if running)";
  document.getElementById("cSpend").textContent = "$" + (g.spent_usd||0).toFixed(2);
  document.getElementById("cSpendHint").textContent =
    g.spend_cap_usd>0 ? ("of $" + g.spend_cap_usd.toFixed(2) + " cap") : "no cap set";
  document.getElementById("cQueued").textContent = q.queued;
  document.getElementById("cLeasable").textContent =
    liveAts.leaseable == null ? g.leasable : liveAts.leaseable;
  document.getElementById("cLeasableHint").textContent =
    liveAts.leaseable == null ? "eligible if running" : "leaseable after pause/canary gates";
  document.getElementById("cChallenges").textContent = s.challenges;
  document.getElementById("cQueuedAts").textContent = laneRateText((q.by_lane && q.by_lane.ats) || {});
  document.getElementById("cQueuedCompute").textContent = laneRateText((q.by_lane && q.by_lane.compute) || {});
  document.getElementById("cQueuedOther").textContent = laneRateText((q.by_lane && q.by_lane.other) || {});
  const queueFlow = updateQueueFlowMetrics(q);
  document.getElementById("cQueueFlow").textContent = queueFlow.text;
  document.getElementById("cQueueFlowHint").textContent = queueFlow.hint;
  const byLane = q.by_lane || {};
  const atsFlow = updateLaneFlowMetrics("ats", byLane.ats || {});
  const computeFlow = updateLaneFlowMetrics("compute", byLane.compute || {});
  const otherFlow = updateLaneFlowMetrics("other", byLane.other || {});
  document.getElementById("cLaneFlowAts").textContent = atsFlow.text;
  document.getElementById("cLaneFlowAtsHint").textContent = atsFlow.hint;
  document.getElementById("cLaneFlowCompute").textContent = computeFlow.text;
  document.getElementById("cLaneFlowComputeHint").textContent = computeFlow.hint;
  document.getElementById("cLaneFlowOther").textContent = otherFlow.text;
  document.getElementById("cLaneFlowOtherHint").textContent = otherFlow.hint;
  const asCode = state.code || "unknown";
  const asSeverity = (state.severity || "warn").toLowerCase();
  const asCard = document.getElementById("applyStateCard");
  if(asCard){ asCard.className = "card apply-state " + asSeverity; }
  document.getElementById("applyState").textContent = asCode;
  document.getElementById("applyStateHint").textContent = state.reason || "—";

  const liveness = s.liveness || {};
  const livenessAvailable = liveness.available !== false;
  const lt = liveness.totals || {};
  const uncertainReasons = liveness.uncertain_reasons || [];
  const hostCooldowns = liveness.host_cooldowns || [];
  const livenessRows = livenessAvailable ? [
    ["Available", Number(lt.live || 0), "confirmed live"],
    ["Unavailable", Number(lt.dead || 0), "confirmed dead"],
    ["Uncertain", Number(lt.uncertain || 0), "grouped by reason below"],
    ["Unchecked", Number(lt.unchecked || 0), "no availability probe yet"],
  ].concat(uncertainReasons.slice(0, 8).map((item) => [
    String(item.reason || "missing_reason").replace(/_/g, " "),
    Number(item.rows || 0),
    String(item.retry_class || "uncertain") + " retry / " +
      Math.round(Number(item.retry_after_seconds || 0) / 3600) + "h / " +
      "max streak " + Number(item.max_consecutive_uncertain || 0) + " / " +
      Number(item.hosts || 0) + " host(s)" +
      (item.latest_checked_at ? (" / checked " + rel(item.latest_checked_at)) : " / check time missing"),
  ])) : [["Telemetry unavailable", "—", "availability query failed: " + String(liveness.error || "unknown error")]];
  if(livenessAvailable) hostCooldowns.slice(0, 8).forEach((item) => {
    livenessRows.push([
      "Cooldown: " + String(item.host || "unknown"),
      Math.ceil(Number(item.remaining_seconds || 0) / 60) + "m",
      Number(item.deferred_rows || 0) + " row(s) deferred / " + String(item.reason || "unknown"),
    ]);
  });
  document.getElementById("livenessBody").innerHTML = livenessRows.map(([label, value, hint]) => (
    '<div class="fstep"><span>' + esc(label) + '</span><b>' + esc(value) +
    '</b><div class="sub">' + esc(hint) + '</div></div>'
  )).join("");

  const versions = s.versions || {};
  document.getElementById("versionPinned").textContent = versions.pinned_worker_version || "not pinned";
  document.getElementById("versionCanary").textContent = versions.canary_version || "off";
  document.getElementById("versionCanaryHint").textContent =
    versions.canary_worker_id ? ("worker " + versions.canary_worker_id) : "optional non-application worker";
  document.getElementById("versionAtsCanary").textContent = versions.ats_canary_version || "off";
  document.getElementById("versionAtsCanaryHint").textContent =
    versions.ats_canary_worker_id ? ("worker " + versions.ats_canary_worker_id) : "off";
  document.getElementById("versionLinkedinCanary").textContent = versions.linkedin_canary_version || "off";
  document.getElementById("versionLinkedinCanaryHint").textContent =
    versions.linkedin_canary_worker_id ? ("worker " + versions.linkedin_canary_worker_id) : "off";
  document.getElementById("versionDrift").textContent = (versions.drifted_workers || []).length;
  const vt = document.getElementById("versionRows");
  const versionRows = Object.entries(versions.worker_versions || {});
  if(!versionRows.length){ vt.innerHTML = '<tr><td colspan="2" class="mut">no worker versions reported</td></tr>'; }
  else vt.innerHTML = versionRows.map(([ver,n])=>{
    return '<tr><td><code>'+esc(ver)+'</code></td><td>'+esc(n)+'</td></tr>';
  }).join("");

  const wt = document.getElementById("workers");
    if(!s.workers.length){ wt.innerHTML = '<tr><td colspan="10" class="mut">no apply workers heartbeating</td></tr>'; }
  else wt.innerHTML = s.workers.map(w=>{
    const sw = w.last_agent_switch_at ? (rel(w.last_agent_switch_at)+
      (w.last_agent_switch_reason ? (" · " + esc(w.last_agent_switch_reason)) : "")) : "—";
    const cur = w.current ? esc((w.current.company||"")+" — "+(w.current.title||"")) : '<span class="mut">idle</span>';
    const beat = w.last_beat ? (rel(w.last_beat)+(w.seconds_since!=null?" ("+w.seconds_since+"s)":"")) : "never";
    return '<tr><td>'+esc(w.machine||"")+'</td><td>'+esc(w.worker_id)+'</td>'+
       '<td><span class="ldot '+(w.alive?"on":"off")+'"></span>'+(w.alive?"alive":"down")+'</td>'+
       '<td class="mut">'+esc(beat)+'</td><td>'+(w.applied||0)+'</td>'+
       '<td>'+esc(w.current_agent || "—")+'</td><td>'+esc(w.current_model || "—")+'</td>'+
       '<td class="mut">'+vendorChipHtml(w.current_model_vendor || "unknown")+'</td>'+
       '<td class="mut">'+familyChipHtml(w.current_model_family || "unknown")+'</td>'+
       '<td class="mut">'+esc(sw)+'</td><td>'+cur+'</td></tr>';
  }).join("");

  const rt = document.getElementById("recent");
  if(!s.recent.length){ rt.innerHTML = '<tr><td colspan="4" class="mut">no recent activity</td></tr>'; }
  else rt.innerHTML = s.recent.map(r=>{
    const cls = r.status==="applied"?"s-applied":(r.status==="crash_unconfirmed"?"s-crash":
      (r.status==="failed"||r.status==="blocked"?"s-failed":"mut"));
    const ct = esc([r.company,r.title].filter(Boolean).join(" — "));
    return '<tr><td class="mut">'+esc(rel(r.time))+'</td><td>'+(r.worker?esc(r.worker):'<span class="mut">—</span>')+
      '</td><td class="'+cls+'">'+esc(r.status)+'</td><td>'+(ct||'<span class="mut">—</span>')+'</td></tr>';
  }).join("");

  document.getElementById("liQueued").textContent = li.queued;
  document.getElementById("liApplied").textContent = li.applied;
  document.getElementById("liCanary").textContent = li.canary_enabled ? "armed" : "off";
  document.getElementById("liHalted").textContent = li.halted ? "YES" : "no";

  // Populate the worker-logs <select> from apply + discovery workers, preserving the
  // current selection. Only rebuild when the worker set changes (don't clobber an open
  // dropdown every 4s).
  const workerRows = [];
  (s.workers || []).forEach(w=>{
    workerRows.push({
      id: w.worker_id,
      label: (w.machine ? (w.machine + " · ") : "") + w.worker_id,
    });
  });
  if(s.discovery && s.discovery.workers){
    s.discovery.workers.forEach(w=>{
      workerRows.push({
        id: w.worker_id,
        label: (w.machine ? ("[disc] " + w.machine + " · ") : "[disc] ") + w.worker_id,
      });
    });
  }
  const ids = workerRows.map(item => item.id);
  const sel = document.getElementById("logWorker");
  const sig = workerRows.map(item => item.id + ":" + item.label).join("|");
  if(sel._sig !== sig){
    sel._sig = sig;
    const cur = sel.value;
    sel.innerHTML = '<option value="">— select worker —</option>' +
      workerRows.map(item=>'<option value="'+esc(item.id)+'">'+esc(item.label)+'</option>').join("");
    if(ids.indexOf(cur) >= 0) sel.value = cur;
  }

  const d = s.discovery;
  if(d){
    document.getElementById("dTasks").textContent = d.tasks.total;
    document.getElementById("dTasksHint").textContent = d.tasks.enabled + " enabled";
    document.getElementById("dDue").textContent = d.tasks.due_now;
    document.getElementById("dStaged").textContent = d.postings.total;
    document.getElementById("dStagedHint").textContent = d.postings.pending_ingest + " pending ingest";
    document.getElementById("dFound24").textContent = d.postings.last24h;
    const dw = document.getElementById("discWorkers");
    if(!d.workers.length){ dw.innerHTML = '<tr><td colspan="5" class="mut">no discovery workers heartbeating &mdash; start one with applypilot-fleet-discovery</td></tr>'; }
    else dw.innerHTML = d.workers.map(w=>{
      const beat = w.last_beat ? (rel(w.last_beat)+(w.seconds_since!=null?" ("+w.seconds_since+"s)":"")) : "never";
      return '<tr><td>'+esc(w.worker_id)+'</td>'+
        '<td>'+esc(w.machine || "unknown")+'</td>'+
        '<td><span class="ldot '+(w.alive?"on":"off")+'"></span>'+(w.alive?"alive":"down")+'</td>'+
        '<td class="mut">'+esc(beat)+'</td><td>'+(w.found_24h||0)+'</td></tr>';
    }).join("");
    const dr = document.getElementById("discRecent");
    if(!d.recent.length){ dr.innerHTML = '<tr><td colspan="3" class="mut">no postings discovered yet</td></tr>'; }
    else dr.innerHTML = d.recent.map(r=>{
      const search = esc([r.query,r.board].filter(Boolean).join(" · "));
      return '<tr><td class="mut">'+esc(rel(r.time))+'</td><td>'+esc(r.source||"—")+
        '</td><td>'+(search||'<span class="mut">—</span>')+'</td></tr>';
    }).join("");
  }
}

async function loadAgents(){
  try{
    const r = await fetch("/api/agents", {cache:"no-store"});
    if(!r.ok) return;
    const d = await r.json();
    const rows = (d.workers || []);
    const tb = document.getElementById("agents");
    if(!rows.length){
      tb.innerHTML = '<tr><td colspan="10" class="mut">no apply telemetry yet</td></tr>';
      document.getElementById("aModelByMachine").innerHTML = '<tr><td colspan="7" class="mut">no apply telemetry yet</td></tr>';
      document.getElementById("aDynamicWorked").textContent = "no";
      document.getElementById("aLatestSwitch").textContent = "—";
      document.getElementById("aSwitch").textContent = "unknown";
      document.getElementById("aModelFamily").textContent = "—";
      document.getElementById("aModelFamilyHint").textContent = "no model telemetry yet";
      document.getElementById("aPrimaryModel").textContent = "—";
      document.getElementById("aPrimaryModelHint").textContent = "no model telemetry yet";
      return;
    }
    const machineRows = {};
    let latestSwitchTs = null;
    for(const w of rows){
      const m = w.machine || "unknown";
      const v = String(w.current_model_vendor || "unknown").toLowerCase();
      if(!machineRows[m]) machineRows[m] = {workers:0, codex:0, claude:0, gemini:0, other:0, unknown:0};
      machineRows[m].workers += 1;
      const key = (v === "codex" || v === "claude" || v === "gemini" || v === "other" || v === "unknown")
        ? v : "other";
      machineRows[m][key] += 1;
      if(w.last_agent_switch_at){
        const t = Date.parse(w.last_agent_switch_at);
        if(!Number.isNaN(t)){
          latestSwitchTs = (!latestSwitchTs || t > latestSwitchTs) ? t : latestSwitchTs;
        }
      }
    }
    tb.innerHTML = rows.map(w=>{
      const blocked = (w.active_blocked_agents && w.active_blocked_agents.length)
        ? esc(w.active_blocked_agents.join(", ")) : "none";
      const chain = (w.agent_chain || []).join(" → ");
      const verdict = String(w.switch_verdict || "unknown");
      const verdictClass = "v-" + verdict.toLowerCase().replace(/[^a-z-]/g, "-");
      const verdictChip = esc(verdict);
      const when = w.last_agent_switch_at ? esc(rel(w.last_agent_switch_at)) : "—";
       return '<tr class="'+verdictClass+'"><td>'+esc(w.machine || "unknown")+'</td><td>'+esc(w.worker_id)+'</td>'+
        '<td>'+esc(w.current_agent || "—")+'</td><td>'+esc(w.current_model || "—")+'</td>'+
        '<td class="mut">'+vendorChipHtml(w.current_model_vendor || "unknown")+'</td>'+
        '<td class="mut">'+familyChipHtml(w.current_model_family || "unknown")+'</td>'+
        '<td class="mut">'+esc(chain || "—")+'</td><td class="mut">'+
        '<span class="verdict-chip '+verdictClass+'">'+verdictChip+'</span></td>'+
        '<td class="mut">'+blocked+'</td><td class="mut">'+when+(w.last_switch_reason?(" · "+esc(w.last_switch_reason)):"")+'</td></tr>';
    }).join("");
    document.getElementById("aBlocks").textContent = (d.agent_blocks ? Object.keys(d.agent_blocks).length : 0);
    document.getElementById("aSpendRows").textContent = (d.agent_spend ? Object.keys(d.agent_spend).length : 0);
    const machineList = Object.entries(machineRows).map(([name, s]) => ({
      name,
      workers: s.workers,
      codex: s.codex,
      claude: s.claude,
      gemini: s.gemini,
      other: s.other,
      unknown: s.unknown,
    }));
    const mt = document.getElementById("aModelByMachine");
    if(!machineList.length){
      mt.innerHTML = '<tr><td colspan="7" class="mut">no model telemetry yet</td></tr>';
    } else {
      mt.innerHTML = machineList.sort((a,b)=>a.name.localeCompare(b.name)).map(v=>(
        '<tr><td>'+esc(v.name)+'</td><td>'+esc(v.workers)+'</td><td>'+esc(v.codex)+'</td><td>'+
        esc(v.claude)+'</td><td>'+esc(v.gemini)+'</td><td>'+esc(v.other)+'</td><td>'+esc(v.unknown)+'</td></tr>'
      )).join("");
    }
  const switchingSummary = d.summary || {};
    const switched = switchingSummary.switched_workers || 0;
    document.getElementById("aDynamicWorked").textContent = (switched > 0 ? "yes" : "no");
    document.getElementById("aLatestSwitch").textContent = latestSwitchTs ? rel(new Date(latestSwitchTs).toISOString()) : "—";

  const modelUsage = Object.entries(switchingSummary.model_usage || {});
  let modelText = "—";
  if(modelUsage.length){
      const [topModel, topCount] = modelUsage[0];
      const totalModels = switchingSummary.model_count || modelUsage.reduce((n, [, c]) => n + c, 0);
      modelText = topModel + " (" + topCount + "/" + totalModels + ")";
      const mix = modelUsage.slice(0, 2).map(([m, c]) => m + ": " + c).join(" · ");
      document.getElementById("aPrimaryModelHint").textContent = "model mix: " + (mix || "—");
    } else {
      document.getElementById("aPrimaryModelHint").textContent = "no model telemetry yet";
    }
  document.getElementById("aPrimaryModel").textContent = modelText;
    const modelTrend = updateModelTrend(modelUsage, modelText);
    if(document.getElementById("aModelTrend")){
      document.getElementById("aModelTrend").textContent = modelText;
    }
    if(document.getElementById("aModelTrendHint")){
      document.getElementById("aModelTrendHint").textContent = modelTrend.trend + " · " + modelTrend.window;
    }

  const familyUsage = Object.entries(switchingSummary.model_family_usage || {});
  if(familyUsage.length){
    const fam = familyUsage.map(([name,count]) => name + ": " + count).join(" · ");
    document.getElementById("aModelFamilyHint").textContent = fam;
    document.getElementById("aModelFamily").textContent = familyUsage[0][0];
  } else {
    document.getElementById("aModelFamily").textContent = "—";
    document.getElementById("aModelFamilyHint").textContent = "no model family telemetry yet";
  }

  const dynamicEnabled = switchingSummary.dynamic_workers || 0;
    const blocked = switchingSummary.blocked_workers || 0;
    if(dynamicEnabled > 0){
      document.getElementById("aSwitchStatus").textContent = switched + "/" + dynamicEnabled;
      document.getElementById("aSwitchStatusHint").textContent =
        "dynamic chain workers; blocked: " + blocked;
    } else {
      document.getElementById("aSwitchStatus").textContent = "off";
      document.getElementById("aSwitchStatusHint").textContent = "no active dynamic chain configured";
    }

    const a = Array.isArray(rows) ? rows : [];
    const tally = a.reduce((m, w)=>{
      const v = w.switch_verdict || "unknown";
      m[v] = (m[v] || 0) + 1;
      return m;
    }, {});
    const parts = ["working", "blocked", "partial", "misconfigured", "unknown", "not triggered"];
    document.getElementById("aSwitch").textContent = parts
      .filter((p) => (tally[p] || 0) > 0)
      .map((p) => p + "=" + tally[p])
      .join(" / ") || "unknown";
  }catch(e){ /* no-op; /api/status stays authoritative for connection */ }
}

async function poll(){
  try{
    const r = await fetch("/api/status", {cache:"no-store"});
    if(!r.ok) throw new Error("status "+r.status);
    render(await r.json());
  }catch(e){
    document.getElementById("conn").textContent = "disconnected — retrying";
  }
}

function showTokenBanner(){
  const b = document.getElementById("tokBanner");
  if(b) b.className = "tokbanner show";
}

async function act(action, extra, confirmMsg){
  if(confirmMsg && !window.confirm(confirmMsg)) return;
  try{
    const r = await fetch("/api/action", {method:"POST",
      credentials: "same-origin",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify(Object.assign({action}, extra||{}))});
    if(r.status === 401){
      showTokenBanner();
      toast("token expired — open the ?token= URL printed by the console window", "err");
      return {ok:false, error:"token expired"};
    }
    const j = await r.json();
    if(!r.ok || !j.ok){ toast(j.error || ("action failed ("+r.status+")"), "err"); }
    else{ toast(j.message || "done", "ok"); if(j.status) render(j.status); }
    return j;
  }catch(e){ toast("request failed: "+e.message, "err"); return {ok:false, error:e.message}; }
}

document.getElementById("btnArm").onclick = ()=>{
  const k = parseInt(document.getElementById("canaryK").value, 10);
  if(!(k>=1 && k<=200)){ toast("k must be an integer 1..200", "err"); return; }
  act("arm_canary", {k},
    "Arm "+k+" → the fleet will submit up to "+k+" REAL job applications under your name, then auto-pause. Continue?");
};
document.getElementById("btnLift").onclick = ()=>act("lift_canary", null,
  "Lift canary → disarm the current canary without resuming the fleet. Continue?");
document.getElementById("btnResume").onclick = ()=>act("resume", null,
  "Resume → leasing is re-enabled and the fleet may submit real applications again. Continue?");
document.getElementById("btnPause").onclick = ()=>act("pause", null);
document.getElementById("btnReclaim").onclick = ()=>act("reclaim", null);
document.getElementById("btnCap").onclick = ()=>{
  const v = document.getElementById("capUsd").value;
  const usd = parseFloat(v);
  if(!(usd>=0)){ toast("cap must be a number ≥ 0", "err"); return; }
  act("set_cap", {usd},
    "Set spend cap to $"+usd.toFixed(2)+" → the fleet halts once total est cost reaches this. ($0 = NO cap.) Continue?");
};

document.getElementById("btnExpand").onclick = ()=>act("expand_searches", null,
  "Expand searches → seed/refresh the discovery queue (search_tasks) from your searches.yaml. This queues scrape work; it does NOT submit any applications and does NOT write your brain. Continue?");

async function loadLogs(){
  const wid = document.getElementById("logWorker").value;
  const area = document.getElementById("logArea");
  if(!wid){ area.className = "mut"; area.style.fontSize = "12px";
    area.textContent = "Select a worker to view its logs."; return; }
  area.className = "mut"; area.textContent = "loading…";
  try{
    const r = await fetch("/api/logs?worker=" + encodeURIComponent(wid), {cache:"no-store"});
    const j = await r.json();
    if(!r.ok){ area.textContent = (j && j.error) ? j.error : ("error " + r.status); return; }
    const err = j.last_error ? j.last_error : "(no crash recorded)";
    const log = j.recent_log ? j.recent_log : "(no recent log)";
    area.className = "";
    area.innerHTML =
      '<div class="logblk"><div class="lbl">Last crash (' + esc(wid) +
        ', beat ' + esc(rel(j.last_beat)) + ')</div>' +
        '<pre class="log err">' + esc(err) + '</pre></div>' +
      '<div class="logblk"><div class="lbl">Recent log tail</div>' +
        '<pre class="log">' + esc(log) + '</pre></div>';
  }catch(e){ area.textContent = "request failed: " + e.message; }
}
document.getElementById("btnLogs").onclick = loadLogs;
document.getElementById("logWorker").onchange = loadLogs;

// --- Fleet Doctor: its own fetch (NOT in the 4s status poll; the blob can be large) ---
function renderDiagnostics(d){
  const ct = document.getElementById("docClusters");
  if(!d.clusters || !d.clusters.length){ ct.innerHTML = '<tr><td colspan="4" class="mut">no failure clusters in the window</td></tr>'; }
  else ct.innerHTML = d.clusters.map(c=>
    '<tr><td>'+esc(c.reason)+'</td><td class="mut">'+esc(c.host)+'</td><td class="mut">'+
    esc(displayMachineName(c.machine))+'</td><td>'+(c.sample_count||0)+'</td></tr>').join("");

  const at = document.getElementById("docAuto");
  if(!d.auto_fixes || !d.auto_fixes.length){ at.innerHTML = '<tr><td colspan="5" class="mut">no active auto-fixes</td></tr>'; }
  else at.innerHTML = d.auto_fixes.map(a=>{
    const exp = a.expires_at ? rel(a.expires_at).replace(" ago"," from now").replace("never","never") : "—";
    return '<tr><td>'+esc(a.knob_type)+'</td><td class="mut">'+esc(a.scope_key)+'</td><td class="mut">'+
      esc(a.reason||"—")+'</td><td class="mut">'+esc(a.expires_at||"—")+'</td>'+
      '<td><button class="sm" data-revert="'+esc(a.knob_id)+'">Reverse</button></td></tr>';
  }).join("");
  at.querySelectorAll("button[data-revert]").forEach(b=>{
    b.onclick = ()=>act("doctor_revert", {knob_id: parseInt(b.getAttribute("data-revert"),10)},
      "Reverse this Doctor auto-fix? It deactivates the conservative knob (e.g. clears a host_skip or restores a paced gap). Nothing is re-approved and the fleet is NOT resumed.")
      .then(loadDiagnostics);
  });

  const rc = document.getElementById("docRecs");
  if(!d.recommendations || !d.recommendations.length){ rc.className="mut"; rc.textContent = "no open recommendations"; }
  else { rc.className=""; rc.innerHTML = d.recommendations.map(r=>{
    const sev = (r.severity==="severe"?"severe":(r.severity==="warn"?"warn":""));
    const scopeParts = [];
    if(r.lane){ scopeParts.push("lane=" + esc(r.lane)); }
    if(r.machine){ scopeParts.push("machine=" + esc(r.machine)); }
    if(r.host){ scopeParts.push("host=" + esc(r.host)); }
    if(r.sample_count){ scopeParts.push("samples=" + Number(r.sample_count)); }
    const scopeLabel = scopeParts.length ? " · " + scopeParts.join(" · ") : "";
    return '<div class="rec"><div class="rh"><span class="sev '+sev+'">'+esc(r.severity||"info")+
      '</span><span class="mut" style="font-size:12px">'+esc(r.reason||"")+scopeLabel+'</span>'+
      '<button class="sm" style="margin-left:auto" data-dismiss="'+esc(r.diagnosis_id)+'">Dismiss</button></div>'+
      '<div class="body">'+esc(r.diagnosis||"")+'</div>'+
      '<div class="recm">'+esc(r.recommendation||"")+'</div></div>';
  }).join(""); }
  rc.querySelectorAll("button[data-dismiss]").forEach(b=>{
    b.onclick = ()=>act("doctor_dismiss", {diagnosis_id: parseInt(b.getAttribute("data-dismiss"),10)})
      .then(loadDiagnostics);
  });

  const bh = d.browser_health || {};
  const bi = bh.summary || {};
  document.getElementById("bWorkers").textContent = bi.workers || 0;
  document.getElementById("bProblem").textContent = bi.problem_workers || 0;
  document.getElementById("bHealthy").textContent = Math.max((bi.workers || 0) - (bi.problem_workers || 0), 0);
  const byKind = bh.summary && bh.summary.by_issue ? Object.entries(bh.summary.by_issue) : [];
  if(byKind.length){
    const breakdown = byKind
      .map(([name, count]) => esc(name + ": " + (count || 0)))
      .join(" · ");
    document.getElementById("bProblemHint").textContent = breakdown;
  } else {
    document.getElementById("bProblemHint").textContent = "workers with browser/backend symptoms";
  }
  const bhRows = document.getElementById("browserHealth");
  const issues = (bh.issues || []);
  if(!issues.length){ bhRows.innerHTML = '<tr><td colspan="8" class="mut">no browser/backend issues detected</td></tr>'; }
  else bhRows.innerHTML = issues.map(x=>{
    const cls = x.age !== null ? (" (" + x.age + "s ago)") : "";
    const lastBeat = x.last_beat ? esc(rel(x.last_beat)) : "—";
    return '<tr><td>'+esc(x.worker_id || "—")+'</td><td>'+esc(x.machine || "unknown")+'</td><td>'+
      esc(x.role || "—")+'</td><td class="mut">'+esc(x.state || "—")+'</td><td>'+
      '<span class="mut">'+esc(x.issue || "—")+'</span></td><td>'+(x.age !== null ? esc(String(x.age)) : "—")+
      '</td><td class="mut">'+esc(x.evidence || "—")+'</td><td class="mut">'+lastBeat + cls+'</td></tr>';
  }).join("");
}

function markDiagnosticsStale(reason){
  const rc = document.getElementById("docRecs");
  if(rc){
    rc.className = "";
    rc.innerHTML = '<div class="rec"><div class="rh"><span class="sev warn">warn</span>' +
      '<span class="mut" style="font-size:12px">diagnostics endpoint</span></div>' +
      '<div class="body">Diagnostics data is stale.</div>' +
      '<div class="recm">' + esc(reason || "diagnostics endpoint failed") + '</div></div>';
  }
  const bhRows = document.getElementById("browserHealth");
  if(bhRows){
    bhRows.innerHTML = '<tr><td colspan="8" class="mut">diagnostics unavailable: ' +
      esc(reason || "endpoint failed") + '</td></tr>';
  }
}

async function loadDiagnostics(){
  try{
    const r = await fetch("/api/diagnostics", {cache:"no-store"});
    if(!r.ok){ markDiagnosticsStale("endpoint returned " + r.status); return; }
    renderDiagnostics(await r.json());
  }catch(e){ markDiagnosticsStale("endpoint error: " + e.message); }
}

function renderDiagnosis(d){
  const state = d.apply_state || {};
  const headline = String(state.code || "unknown").replace(/_/g, " ");
  document.getElementById("stateHeadline").textContent = headline;
  document.getElementById("stateReason").textContent = state.reason || "No blocking reason detected.";
  const action =
    state.next_action
      ? String(state.next_action)
      : (state.severity === "halted")
      ? "Resume from halt before applies can proceed."
      : "Monitor queue and worker health until the next state transition.";
  document.getElementById("stateAction").textContent = action;

  const safety = d.safety || {};
  const safetyRows = [
    ["Fleet paused", safety.paused ? "yes" : "no"],
    ["ATS paused", safety.ats_paused ? "yes" : "no"],
    ["Spend cap", safety.spend_cap_usd > 0 ? "$" + Number(safety.spend_cap_usd).toFixed(2) : "none"],
    ["Spent", "$" + Number(safety.spent_usd || 0).toFixed(2)],
    ["Canary", safety.canary_enabled ? "enabled" : "off"],
    ["Canary remaining", safety.canary_remaining == null ? "—" : String(safety.canary_remaining)],
  ];
  document.getElementById("safetyGrid").innerHTML = safetyRows.map(([label, value]) =>
    '<div class="mini"><span>' + esc(label) + '</span><b>' + esc(value) + '</b></div>'
  ).join("");

  const recs = d.recommendations || [];
  const why = document.getElementById("whyBody");
  if(!recs.length){
    why.innerHTML = '<div class="mini"><span>Why not applying</span><b>no open recommendations</b></div>';
  } else {
    why.innerHTML = recs.map(r=>(
      '<div class="mini"><span>' + esc(r.severity || "info") + '</span><b>' + esc(r.title || "recommendation") +
      '</b><div class="sub" style="margin-top:4px">' + esc(r.reason || "—") + '</div></div>'
    )).join("");
  }

  const rec0 = recs[0] || {};
  const stateRecommendation = document.getElementById("stateRecommendation");
  stateRecommendation.className = "mut";
  stateRecommendation.textContent =
    rec0.title
      ? rec0.title + " — " + (rec0.reason || "review recommended action.")
      : "No recommended action now.";

  const machines = (d.rollups && d.rollups.machines) || {};
  const networkMachines = (d.rollups && d.rollups.network && d.rollups.network.machines) || {};
  const machineRows = Object.entries(machines);
  const machineWrap = document.getElementById("machineMap");
  if(!machineRows.length){
    machineWrap.innerHTML = '<div class="mini"><span>Machine health</span><b>no machine telemetry</b></div>';
  } else {
    machineWrap.innerHTML = machineRows.map(([name, m]) => {
      const network = networkMachines[name] || {};
      const roles = network.by_role || {};
      const roleText = Object.entries(roles).map(([role, count]) => role + ":" + count).join(", ");
      const workers = Number(m.workers || 0);
      const alive = Number(m.alive || 0);
      const working = Number(m.working || 0);
      const applied = Number(m.applied_workers || 0);
      const stale = Number(network.stale || 0);
      const desired = m.desired == null ? null : Number(m.desired || 0);
      const missing = m.missing == null ? null : Number(m.missing || 0);
      const fleetWorkers = m.fleet_workers == null ? Number(network.workers || workers) : Number(m.fleet_workers || 0);
      const fleetAlive = m.fleet_alive == null ? Number(network.alive || alive) : Number(m.fleet_alive || 0);
      const fleetStale = m.fleet_stale == null ? stale : Number(m.fleet_stale || 0);
      let meta = "fleet " + fleetAlive + "/" + fleetWorkers + " / apply alive " + alive +
        " / working " + working + " / applied workers " + applied;
      if(roleText){
        meta += " / roles " + roleText + " / stale " + fleetStale;
      }
      if(desired !== null){
        meta += " / desired " + desired + " / missing " + (missing || 0);
      }
      return '<div class="mini"><span>' + esc(name) + '</span><b>' + esc(workers) +
        '</b><div class="sub" style="margin-top:4px">' +
        esc(meta) + '</div></div>';
    }).join("");
  }

  const agents = d.agents || {};
  const agentBlocks = agents.active_agent_blocks || [];
  const agentModels = Object.entries(agents.model_usage || {});
  const modelText = agentModels.length
    ? agentModels.slice(0, 2).map(([model, count]) => model + ":" + count).join(", ")
    : "none";
  const dynamicWorkers = Number(agents.dynamic_workers || 0);
  const switchedWorkers = Number(agents.switched_workers || 0);
  document.getElementById("agentBody").innerHTML = [
    ["Workers", Number(agents.workers || 0), "apply workers with routing telemetry"],
    ["Models", modelText, Number(agents.model_count || 0) + " reported model(s)"],
    ["Dynamic switched", switchedWorkers + "/" + dynamicWorkers, switchedWorkers > 0 ? "worked" : "not triggered"],
    ["Blocked agents", agentBlocks.length ? agentBlocks.join(", ") : "none", "active availability blocks"],
  ].map(([label, value, hint]) => (
    '<div class="fstep"><span>' + esc(label) + '</span><b>' + esc(value) +
    '</b><div class="sub">' + esc(hint) + '</div></div>'
  )).join("");

  const comparison = d.worker_comparison || {};
  const comparisonRows = comparison.workers || [];
  if(!comparisonRows.length){
    document.getElementById("workerComparisonBody").innerHTML =
      '<div class="fstep"><span>Workers</span><b>none</b><div class="sub">no apply worker comparison yet</div></div>';
  } else {
    document.getElementById("workerComparisonBody").innerHTML = comparisonRows.slice(0, 6).map((w) => {
      const meta = "machine " + (w.machine || "unknown") +
        " / applied " + Number(w.applied || 0) +
        " / failed " + Number(w.failed || 0) +
        " / challenges " + Number(w.challenges || 0) +
        " / cost $" + Number(w.cost_usd || 0).toFixed(2) +
        (w.avg_duration_ms == null ? "" : " / avg " + Math.round(Number(w.avg_duration_ms) / 1000) + "s");
      return '<div class="fstep"><span>' + esc(w.worker_id || "unknown") + '</span><b>' +
        esc(w.alive ? "alive" : "down") + '</b><div class="sub">' + esc(meta) + '</div></div>';
    }).join("");
  }

  const funnel = (d.rollups && d.rollups.funnel) || {};
  document.getElementById("funnelBody").innerHTML = ["ats", "compute", "other"].map((lane) => {
    const q = funnel[lane] || {};
    const queued = Number(q.queued || 0);
    const leased = Number(q.leased || 0);
    const applied = Number(q.applied || 0);
    return '<div class="fstep"><span>' + esc(lane) +
      '</span><b>' + esc(queued + "/" + leased + "/" + applied) +
      '</b><div class="sub">queued/leased/applied</div></div>';
  }).join("");

  const forecast = d.forecast || {};
  document.getElementById("forecastBody").innerHTML = [
    ["Applies", Number(forecast.applies_last_1h || 0) + "/" + Number(forecast.applies_last_24h || 0), "last 1h / last 24h"],
    ["Live workers", Number(forecast.live_apply_workers || 0), "apply workers alive"],
    ["Leaseable", Number(forecast.leaseable_jobs || 0), "jobs available now"],
    ["Challenges", Number(forecast.open_challenges || 0), "open auth challenges"],
    ["ETA", forecast.eta_hours_to_exhaust_leaseable == null ? "—" : forecast.eta_hours_to_exhaust_leaseable + "h", forecast.assumption || "recent apply rate"],
  ].map(([label, value, hint]) => (
    '<div class="fstep"><span>' + esc(label) + '</span><b>' + esc(value) +
    '</b><div class="sub">' + esc(hint) + '</div></div>'
  )).join("");

  const goals = d.daily_goals || {};
  document.getElementById("dailyGoalsBody").innerHTML = [
    ["Applied today", Number(goals.applied_today || 0), goals.message || "No daily target configured"],
    ["Target today", goals.target_today == null ? "none" : Number(goals.target_today || 0), goals.configured ? "configured" : "not configured"],
    ["Remaining", goals.remaining_target == null ? "—" : Number(goals.remaining_target || 0), "remaining target"],
    ["Shortfall", goals.projected_shortfall == null ? "—" : Number(goals.projected_shortfall || 0), "projected from current confirmed applies"],
    ["Canary", goals.canary_remaining == null ? "—" : Number(goals.canary_remaining || 0), "ATS canary remaining"],
  ].map(([label, value, hint]) => (
    '<div class="fstep"><span>' + esc(label) + '</span><b>' + esc(value) +
    '</b><div class="sub">' + esc(hint) + '</div></div>'
  )).join("");

  const freshness = d.freshness || {};
  document.getElementById("freshnessBody").innerHTML = [
    ["Diagnosis", freshness.generated_at ? rel(freshness.generated_at) : "stale", freshness.endpoint || "diagnosis"],
    ["Worker beat", freshness.last_worker_beat_at ? rel(freshness.last_worker_beat_at) : "none", "freshest fleet heartbeat"],
    ["Last apply", freshness.last_apply_at ? rel(freshness.last_apply_at) : "none", "confirmed apply timestamp"],
    ["Discovery", freshness.last_discovery_at ? rel(freshness.last_discovery_at) : "none", "latest staged posting"],
    ["Compute", freshness.last_compute_at ? rel(freshness.last_compute_at) : "none", "latest score queue update"],
  ].map(([label, value, hint]) => (
    '<div class="fstep"><span>' + esc(label) + '</span><b>' + esc(value) +
    '</b><div class="sub">' + esc(hint) + '</div></div>'
  )).join("");

  const audit = d.audit || {};
  const auditRows = audit.rows || [];
  if(!audit.available){
    document.getElementById("operatorAuditBody").innerHTML =
      '<div class="fstep"><span>Audit</span><b>unavailable</b><div class="sub">fleet_console_audit table not found</div></div>';
  } else if(!auditRows.length){
    document.getElementById("operatorAuditBody").innerHTML =
      '<div class="fstep"><span>Audit</span><b>empty</b><div class="sub">no console actions recorded yet</div></div>';
  } else {
    document.getElementById("operatorAuditBody").innerHTML = auditRows.slice(0, 6).map((row) => {
      const meta = (row.created_at ? rel(row.created_at) : "time unknown") +
        " / " + (row.lane || "fleet") +
        " / " + (row.target || "target unknown") +
        " / " + (row.message || "no message");
      return '<div class="fstep"><span>' + esc(row.action || "action") + '</span><b>' +
        esc(row.ok ? "ok" : "failed") + '</b><div class="sub">' + esc(meta) + '</div></div>';
    }).join("");
  }

  const sourceQuality = d.source_quality || {};
  const hostRows = sourceQuality.hosts || [];
  if(!hostRows.length){
    document.getElementById("hostQualityBody").innerHTML =
      '<div class="fstep"><span>Hosts</span><b>none</b><div class="sub">no apply host telemetry yet</div></div>';
  } else {
    document.getElementById("hostQualityBody").innerHTML = hostRows.slice(0, 6).map((h) => {
      const meta = "applied " + Number(h.applied || 0) +
        " / failed " + Number(h.failed || 0) +
        " / challenges " + Number(h.challenges || 0) +
        " / queued " + Number(h.queued || 0) +
        " / leaseable " + Number(h.leaseable || 0);
      return '<div class="fstep"><span>' + esc(h.host || "unknown") + '</span><b>' +
        esc(Number(h.challenge_rate || 0).toFixed(2)) +
        '</b><div class="sub">' + esc(meta) + '</div></div>';
    }).join("");
  }

  const li = d.linkedin || {};
  document.getElementById("linkedinBody").innerHTML = [
    ["Queued", Number(li.queued || 0), "ready for LinkedIn lane"],
    ["Applied", Number(li.applied || 0), "terminal LinkedIn submits"],
    ["Canary", li.canary_enabled ? "on" : "off", "LinkedIn bounded mode"],
    ["Governor", li.halted ? "halted" : "open", "account:linkedin"],
  ].map(([label, value, hint]) => (
    '<div class="fstep"><span>' + esc(label) + '</span><b>' + esc(value) +
    '</b><div class="sub">' + esc(hint) + '</div></div>'
  )).join("");

  const disc = d.discovery || {};
  const discTasks = disc.tasks || {};
  const discPostings = disc.postings || {};
  const discWorkers = disc.workers || [];
  const liveDiscoveryWorkers = discWorkers.filter((w) => !!w.alive).length;
  document.getElementById("discoveryBody").innerHTML = [
    ["Due tasks", Number(discTasks.due_now || 0), (discTasks.enabled || 0) + " enabled"],
    ["Staged", Number(discPostings.pending_ingest || 0), "pending ingest"],
    ["Found 24h", Number(discPostings.last24h || 0), "new postings"],
    ["Workers", liveDiscoveryWorkers + "/" + discWorkers.length, "alive/total discovery"],
  ].map(([label, value, hint]) => (
    '<div class="fstep"><span>' + esc(label) + '</span><b>' + esc(value) +
    '</b><div class="sub">' + esc(hint) + '</div></div>'
  )).join("");

  const compute = d.compute || {};
  const computeQueue = compute.queue || {};
  const computeWorkers = compute.score_workers || {};
  const computeContext = compute.context || {};
  document.getElementById("computeBody").innerHTML = [
    ["Context", computeContext.version || (compute.unavailable ? "unavailable" : "missing"), "score context"],
    ["Queue", Number((computeQueue.queued || {}).count || 0) + "/" + Number((computeQueue.leased || {}).count || 0), "queued/leased score tasks"],
    ["Workers", Number(computeWorkers.active || 0), "active compute workers"],
    ["Failures 15m", Number(computeWorkers.recent_failed_15m || 0), "rate limited " + Number(computeWorkers.recent_rate_limited_15m || 0)],
    ["Unsynced done", Number((computeQueue.done || {}).unsynced || 0), "waiting for home pull"],
  ].map(([label, value, hint]) => (
    '<div class="fstep"><span>' + esc(label) + '</span><b>' + esc(value) +
    '</b><div class="sub">' + esc(hint) + '</div></div>'
  )).join("");

  const deadman = d.deadman || {};
  document.getElementById("deadmanBody").innerHTML = [
    ["Status", deadman.active ? "alert" : "clear", deadman.active ? "watchdog raised" : "no active alert"],
    ["Alert", deadman.title || deadman.alert || "none", deadman.reason || (deadman.alert_at ? rel(deadman.alert_at) : "not set")],
  ].map(([label, value, hint]) => (
    '<div class="fstep"><span>' + esc(label) + '</span><b>' + esc(value) +
    '</b><div class="sub">' + esc(hint) + '</div></div>'
  )).join("");
}

function renderOutcomes(d){
  const t = document.getElementById("outcomes");
  const rows = (d && d.outcomes) || [];
  if(!rows.length){ t.innerHTML = '<tr><td colspan="4" class="mut">no outcomes yet</td></tr>'; return; }
  t.innerHTML = rows.map(r=>{
    const ct = esc([r.company,r.title].filter(Boolean).join(" — "));
    return '<tr><td class="mut">'+esc(rel(r.time))+'</td><td>'+(ct||'<span class="mut">—</span>')+
      '</td><td>'+esc(r.stage||"—")+'</td><td>'+esc(r.outcome||"—")+'</td></tr>';
  }).join("");
}
async function loadOutcomes(){
  try{
    const r = await fetch("/api/outcomes", {cache:"no-store"});
    if(!r.ok) return;
    renderOutcomes(await r.json());
  }catch(e){ /* leave last render */ }
}
loadOutcomes();
setInterval(loadOutcomes, 15000);

function markDiagnosisStale(reason){
  const fb = document.getElementById("freshnessBody");
  if(!fb) return;
  fb.innerHTML = '<div class="fstep"><span>Diagnosis</span><b>diagnosis stale</b><div class="sub">' +
    esc(reason || "diagnosis endpoint failed") + '</div></div>';
}

async function loadDiagnosis(){
  try{
    const r = await fetch("/api/diagnosis", {cache:"no-store"});
    if(!r.ok){ markDiagnosisStale("endpoint returned " + r.status); return; }
    renderDiagnosis(await r.json());
  }catch(e){ markDiagnosisStale("endpoint error"); }
}
loadDiagnosis();
setInterval(loadDiagnosis, 15000);

// --- Challenges triage: its own fetch of /api/challenges, folded into the same 4s cycle. ---
function renderChallenges(d){
  const wrap = document.getElementById("chList");
  const groups = (d && d.groups) || [];
  if(!groups.length){ wrap.className = "mut"; wrap.textContent = "no open challenges — nothing parked"; return; }
  const triage = (d && d.triage) || {};
  const clearFirst = triage.clear_first || [];
  const resolveFirst = triage.resolve_first || [];
  const fmtGroup = (g)=>{
    if(!g) return "";
    const parts = [];
    parts.push((g.host || "unknown") + " / " + (g.kind || "unknown"));
    parts.push(String(Number(g.rows || 0)) + " row(s)");
    if(Number(g.stale_rows || 0) > 0) parts.push(String(Number(g.stale_rows || 0)) + " stale");
    if(Number(g.missing_metadata_rows || 0) > 0) parts.push(String(Number(g.missing_metadata_rows || 0)) + " missing metadata");
    if(g.oldest_age_hours != null) parts.push(String(g.oldest_age_hours) + "h oldest");
    return esc(parts.join(" · "));
  };
  const triageHtml = (
    '<div class="chgrp">' +
      '<div class="chgrp-h"><b>Clear / drop first</b><span>·</span><span>' + esc(String(clearFirst.length)) + '</span></div>' +
      (clearFirst.length
        ? clearFirst.slice(0, 3).map((g)=>'<div class="chrow"><div class="meta"><div class="ct">' + fmtGroup(g) + '</div></div></div>').join("")
        : '<div class="chrow"><div class="meta"><div class="sub">no stale or metadata-poor groups</div></div></div>') +
    '</div>' +
    '<div class="chgrp">' +
      '<div class="chgrp-h"><b>Worth resolving</b><span>·</span><span>' + esc(String(resolveFirst.length)) + '</span></div>' +
      (resolveFirst.length
        ? resolveFirst.slice(0, 3).map((g)=>'<div class="chrow"><div class="meta"><div class="ct">' + fmtGroup(g) + '</div></div></div>').join("")
        : '<div class="chrow"><div class="meta"><div class="sub">no fresh groups ready for manual resolution</div></div></div>') +
    '</div>'
  );
  wrap.className = "";
  wrap.innerHTML = triageHtml + groups.map((g, gi)=>{
    const rows = g.rows || [];
    const lane = rows.length ? rows[0].lane : null;
    const summary = g.summary || {};
    const rowsHtml = rows.map((r, ri)=>{
      const ct = esc([r.company, r.title].filter(Boolean).join(" — ")) || '<span class="mut">—</span>';
      const scoreTxt = (r.score!=null) ? (" · score " + r.score) : "";
      const ageTxt = (r.age_hours!=null) ? (r.age_hours + "h old") : "age unknown";
      return '<div class="chrow">' +
        '<div class="meta"><div class="ct">' + ct + '</div>' +
        '<div class="sub">' + esc(ageTxt + scoreTxt + (r.lane ? (" · " + r.lane) : "")) + '</div></div>' +
        '<div class="btns">' +
        '<a href="' + esc(r.url) + '" target="_blank" rel="noopener"><button class="sm" type="button">Open job</button></a>' +
        '<button class="sm go" data-req="' + gi + ':' + ri + '">I solved it → Re-queue</button>' +
        '<button class="sm danger" data-skip="' + gi + ':' + ri + '">Skip</button>' +
        '</div></div>';
    }).join("");
    const summaryParts = [String(rows.length)];
    if(Number(summary.stale_rows || 0) > 0) summaryParts.push(String(Number(summary.stale_rows || 0)) + " stale");
    if(Number(summary.missing_metadata_rows || 0) > 0) summaryParts.push(String(Number(summary.missing_metadata_rows || 0)) + " missing metadata");
    return '<div class="chgrp">' +
      '<div class="chgrp-h"><b>' + esc(g.kind||"") + '</b><span>·</span><b>' + esc(g.host||"") +
      '</b><span>·</span><span>' + esc(summaryParts.join(" / ")) + '</span>' +
      '<button class="sm danger" style="margin-left:auto" data-skiphost="' + gi + '">Skip all on host</button>' +
      '</div>' + rowsHtml + '</div>';
  }).join("");

  wrap.querySelectorAll("button[data-req]").forEach(b=>{
    b.onclick = async ()=>{
      const [gi, ri] = b.getAttribute("data-req").split(":").map(Number);
      const row = groups[gi].rows[ri];
      const j = await act("challenge_requeue", {url: row.url, lane: row.lane});
      if(j && j.ok !== false) loadChallenges();
    };
  });
  wrap.querySelectorAll("button[data-skip]").forEach(b=>{
    b.onclick = async ()=>{
      const [gi, ri] = b.getAttribute("data-skip").split(":").map(Number);
      const row = groups[gi].rows[ri];
      const j = await act("challenge_skip", {url: row.url, lane: row.lane});
      if(j && j.ok !== false) loadChallenges();
    };
  });
  wrap.querySelectorAll("button[data-skiphost]").forEach(b=>{
    b.onclick = async ()=>{
      const gi = Number(b.getAttribute("data-skiphost"));
      const g = groups[gi];
      const lane = g.rows.length ? g.rows[0].lane : null;
      if(!window.confirm("Skip all " + g.rows.length + " parked challenge(s) on " + g.host + "?")) return;
      const j = await act("challenge_skip_host", {host: g.host, lane});
      if(j && j.ok !== false) loadChallenges();
    };
  });
}

async function loadChallenges(){
  try{
    const r = await fetch("/api/challenges", {cache:"no-store"});
    if(r.status === 401){ showTokenBanner(); return; }
    if(!r.ok) return;
    renderChallenges(await r.json());
  }catch(e){ /* leave last render; the status poll drives the connection banner */ }
}
loadChallenges();
setInterval(loadChallenges, 4000);
loadAgents();
setInterval(loadAgents, 4000);

loadDiagnosis();
setInterval(loadDiagnosis, 15000);
loadAgents();
setInterval(loadAgents, 15000);
loadAudit();
setInterval(loadAudit, 15000);
poll();
setInterval(poll, 4000);
loadDiagnostics();
setInterval(loadDiagnostics, 15000);
</script>
</body>
</html>
"""

# Keep every legacy data table inside the same responsive overflow container as
# the operations tables added by the console merge.
_INDEX_HTML = re.sub(
    r'class="table-wrap([^\"]*)"',
    r'class="table-scroll\1"',
    _INDEX_HTML,
)
_INDEX_HTML = re.sub(
    r'(class="table-scroll"[^>]*)>\s*<table',
    r'\1><table',
    _INDEX_HTML,
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="applypilot-fleet-console",
                                description="LAN-only web control panel for the apply fleet.")
    p.add_argument("--host", default=None,
                   help="Private IPv4 to bind (10.x/172.16-31.x/192.168.x) or 127.0.0.1. "
                        "Auto-detects a private IP if omitted. Refuses 0.0.0.0 / public IPs.")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = p.parse_args(argv)

    host = resolve_host(args.host)
    if not _is_private_bind(host):  # belt-and-suspenders: never reach bind() on a public host
        raise SystemExit(f"refusing to bind non-private host {host!r}")

    from applypilot.fleet import schema as fleet_schema
    with pgqueue.connect() as schema_conn:
        fleet_schema.ensure_schema_v3(schema_conn)
    server = ThreadingHTTPServer((host, args.port), _Handler)
    # Do NOT print the DSN or any secret -- only the LAN URL the owner should open.
    print(f"ApplyPilot fleet console (LAN-only) -> http://{host}:{args.port}")
    print("Open this from a machine on your LAN. Do NOT port-forward it.")
    print(f"One-tap arm URL (sets the mutation-auth cookie): "
          f"http://{host}:{args.port}/?token={get_console_token()}")
    audit_lifecycle_event(
        "console_start",
        f"started on http://{host}:{args.port} from {_repo_root()}",
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        audit_lifecycle_event(
            "console_stop",
            f"stopped on http://{host}:{args.port} from {_repo_root()}",
        )
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
