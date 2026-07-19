"""Codex monitoring bridge (spec 2026-06-27) -- a local FastMCP stdio server that
surfaces the fleet's read-only telemetry and three bounded-safe actions into Codex.

SAFETY: the guarantee is the tool REGISTRY -- exactly 8 functions are @mcp.tool().
Action tools delegate only to monitor.MonitorActions (whose surface is restart /
quarantine / pause, no apply/unpause/cap/challenge-resolve). We import only the
`connect` symbol from apply.pgqueue (not the module) so set_paused/set_spend_cap are
not even bound here. No DB access happens at import or in main(); every tool goes
through _with_conn, which reads FLEET_PG_DSN itself and rolls back + closes per call.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from applypilot.apply.pgqueue import connect
from applypilot.fleet import heartbeat, monitor

mcp = FastMCP("applypilot-fleet")
logger = logging.getLogger("applypilot.fleet.codex_bridge")


def _with_conn(fn: Callable[[Any], dict]) -> dict:
    """Open a short-lived dict_row connection from FLEET_PG_DSN, run fn(conn), and
    return its dict result (or a structured {"error": ...}). Rolls back (read-only
    discipline; no-op after an action's own commit) and closes in finally. Reads the
    DSN directly from FLEET_PG_DSN (connect() has no generic fallback)."""
    dsn = os.environ.get("FLEET_PG_DSN")
    if not dsn:
        return {"error": "FLEET_PG_DSN is not set; set it in the Codex MCP env block"}
    try:
        conn = connect(dsn)
    except Exception as e:  # RuntimeError (no DSN) / OperationalError (dead DB) / etc.
        return {"error": f"could not connect: {e}"}
    try:
        return fn(conn)
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.close()


@mcp.tool()
def fleet_status() -> dict[str, Any]:
    """Fleet health rollup: machines, breaker states, queue depths, captcha backlog,
    quarantine count, 24h spend."""
    return _with_conn(lambda conn: heartbeat.dashboard_snapshot(conn))


def _read_caps(conn) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT paused, cost_cap_daily_usd, cost_cap_total_usd FROM fleet_config WHERE id=1")
        cfg = cur.fetchone() or {}
        cur.execute("SELECT COALESCE(SUM(cost_usd),0) AS s FROM llm_usage WHERE ts >= now() - interval '24 hours'")
        spend_today = float(cur.fetchone()["s"])
        cur.execute("SELECT COALESCE(SUM(cost_usd),0) AS s FROM llm_usage")
        spend_total = float(cur.fetchone()["s"])
    return {
        "paused": cfg.get("paused"),
        "cost_cap_daily_usd": float(cfg.get("cost_cap_daily_usd") or 0),
        "cost_cap_total_usd": float(cfg.get("cost_cap_total_usd") or 0),
        "spend_today": spend_today,
        "spend_total": spend_total,
    }


@mcp.tool()
def caps() -> dict[str, Any]:
    """Cost caps + spend: paused flag, daily/total caps, 24h spend, all-time spend."""
    return _with_conn(_read_caps)


@mcp.tool()
def health_report() -> dict[str, Any]:
    """The text health report (incl. a NEEDS YOUR DECISION anomaly section). The 24h
    spend is compared against the DAILY cap (apples-to-apples)."""
    def _build(conn):
        snap = heartbeat.dashboard_snapshot(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT cost_cap_daily_usd FROM fleet_config WHERE id=1")
            row = cur.fetchone()
        daily = float((row or {}).get("cost_cap_daily_usd") or 0)
        text = monitor.build_health_report(snap, captcha_threshold=0.4, cost_cap_total=daily)
        return {"report": text}
    return _with_conn(_build)


def _iso(v: Any) -> Any:
    return v.isoformat() if hasattr(v, "isoformat") else v


def _recent_results(conn, limit: int) -> dict[str, Any]:
    n = max(1, min(int(limit), 100))
    rows: list[dict[str, Any]] = []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT url, status, updated_at, company, title, apply_error FROM apply_queue "
            "WHERE status IN ('applied','failed','blocked','crash_unconfirmed') "
            "ORDER BY updated_at DESC LIMIT %s", (n,))
        for r in cur.fetchall():
            rows.append({"lane": "apply", "url": r["url"], "status": r["status"],
                         "finished_at": _iso(r["updated_at"]),
                         "_sort": r["updated_at"],
                         "detail": {"company": r["company"], "title": r["title"],
                                    "apply_error": r["apply_error"]}})
        cur.execute(
            "SELECT url, status, updated_at, task, est_cost_usd FROM compute_queue "
            "WHERE status IN ('done','failed','quarantined') "
            "ORDER BY updated_at DESC LIMIT %s", (n,))
        for r in cur.fetchall():
            rows.append({"lane": "compute", "url": r["url"], "status": r["status"],
                         "finished_at": _iso(r["updated_at"]),
                         "_sort": r["updated_at"],
                         "detail": {"task": r["task"], "cost": float(r["est_cost_usd"] or 0)}})
    # Sort on native datetime (None = oldest); avoids fragile ISO-string lexicographic comparison.
    rows.sort(key=lambda x: (x["_sort"] is not None, x["_sort"] or datetime.min), reverse=True)
    for row in rows:
        del row["_sort"]
    return {"results": rows[:n]}


@mcp.tool()
def recent_results(limit: int = 20) -> dict[str, Any]:
    """The most recent terminal fleet events (apply + compute) merged newest-first,
    each row carrying a lane-specific structured detail dict. limit capped at 100."""
    return _with_conn(lambda conn: _recent_results(conn, limit))


def _challenges(conn) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, url, worker_id, machine_owner, kind, route, raised_at "
                    "FROM auth_challenge WHERE resolved_at IS NULL ORDER BY raised_at DESC")
        out = [{**r, "raised_at": _iso(r["raised_at"])} for r in cur.fetchall()]
    return {"challenges": out}


@mcp.tool()
def challenges() -> dict[str, Any]:
    """Open (unresolved) auth challenges — the captcha backlog detail."""
    return _with_conn(_challenges)


@mcp.tool()
def restart_worker(worker_id: str) -> dict[str, Any]:
    """Enqueue a 'restart' command for a worker (conservative — only slows the fleet)."""
    def _do(conn):
        command_id = monitor.MonitorActions(conn).restart_worker(worker_id)
        logger.info("bridge action: restart_worker worker_id=%s command_id=%s", worker_id, command_id)
        return {"action": "restart", "worker_id": worker_id, "command_id": command_id}
    return _with_conn(_do)


@mcp.tool()
def pause_scope(scope_key: str) -> dict[str, Any]:
    """Pause a host/board scope. Does NOT unpause (resume is owner-only, absent here).

    A4: ONLY 'host:'/'board:' scopes are pausable. account:linkedin / global / home_ip:
    scopes are rejected with a structured error -- this surface can never halt the LinkedIn
    catastrophe lane."""
    def _do(conn):
        try:
            monitor.MonitorActions(conn).pause_scope(scope_key)
        except monitor.ScopeNotPausable as e:
            logger.warning("bridge action REJECTED: pause_scope scope_key=%s (%s)", scope_key, e)
            return {"error": str(e), "rejected": True, "action": "pause", "scope_key": scope_key}
        logger.info("bridge action: pause_scope scope_key=%s", scope_key)
        return {"action": "pause", "scope_key": scope_key}
    return _with_conn(_do)


@mcp.tool()
def quarantine_job(url: str, worker: str, reason: str) -> dict[str, Any]:
    """Manually quarantine a job (one-shot: pulls it now, does not pollute crash_count)."""
    def _do(conn):
        newly = monitor.MonitorActions(conn).quarantine(url, worker=worker, reason=reason)
        logger.info("bridge action: quarantine_job url=%s worker=%s newly=%s", url, worker, newly)
        return {"action": "quarantine", "url": url, "newly_quarantined": newly}
    return _with_conn(_do)


def main() -> int:  # pragma: no cover - stdio server loop, not unit-testable
    """Entry point: run the FastMCP server over stdio. No DB access here."""
    mcp.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    # So `python -m applypilot.fleet.codex_bridge` actually starts the server. Without this
    # guard, -m imported the module and exited without calling main() -> the MCP client saw
    # the process close on `initialize` (Codex Bridge silently failed to start, 2026-07-04).
    raise SystemExit(main())
