"""applypilot-fleet-apply: an OFFSITE apply worker for owner-controlled machines.
Wraps the proven launcher.run_job into an apply_fn and drives WorkerLoop(role='apply').
Respects the shared kill switch AND the Fleet Doctor's ATS-only pause via ats_should_halt
(H1); never leases through a pause/canary-pause. The LinkedIn lane uses its own
linkedin_should_halt gate so a Doctor ATS pause can never halt it."""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import signal
import threading
import time

from applypilot.fleet import emergency_admission
from applypilot.fleet.browser_host_guard import require_enrolled_browser_host
from applypilot.fleet.version import worker_version

logger = logging.getLogger("applypilot.fleet.apply_worker_main")

# --- graceful stop (SIGTERM) -------------------------------------------------
# The macOS launchd wrapper (run-worker-mac.sh) and `launchctl unload` send SIGTERM to
# restart the worker for a code update. Mid-apply death parks the job crash_unconfirmed
# ("may-have-submitted"), so instead: SIGTERM sets a flag, the CURRENT job finishes, and
# run_apply exits before the next lease. SIGINT (Ctrl+C) keeps default abort behavior.
_STOP_REQUESTED = threading.Event()


def request_stop(signum=None, frame=None) -> None:
    _STOP_REQUESTED.set()


def stop_requested() -> bool:
    return _STOP_REQUESTED.is_set()


def install_stop_handler() -> None:
    try:
        signal.signal(signal.SIGTERM, request_stop)
    except (ValueError, OSError):  # pragma: no cover - non-main thread / exotic platform
        pass


def _setup_apply_env() -> None:
    """Point config at writable locations + base-resume BEFORE importing applypilot
    (ports container_worker._setup_env, home-box flavored). run_job records agent cost
    to a home SQLite; the fleet has none, so sink it to a throwaway DB and read the REAL
    cost from launcher._last_run_stats."""
    os.environ["APPLYPILOT_BASE_RESUME"] = "1"
    # Fleet row selection is lane-filtered at push time; keep worker-side acquire opt-in.
    os.environ.setdefault("APPLYPILOT_LANE_FILTER", "0")
    os.environ.setdefault("APPLYPILOT_DB_PATH", os.path.join(os.environ.get("TEMP", "/tmp"), "fleet_apply_throwaway.db"))
    os.environ.setdefault("CHROME_WORKER_DIR", os.path.join(os.environ.get("TEMP", "/tmp"), "chrome-workers"))
    os.environ.setdefault("APPLY_WORKER_DIR", os.path.join(os.environ.get("TEMP", "/tmp"), "apply-workers"))
    os.environ.setdefault("APPLYPILOT_AGENT_TIMEOUT", "300")


def _chrome_slot(worker_id, override=None) -> int:
    """Integer Chrome slot (profile + CDP port + per-run log id) for this worker.

    Multiple apply workers on ONE machine MUST use distinct slots or their Chrome
    instances collide in a single shared browser. Auto-derived from the trailing digits
    of --worker-id (home-0 -> 0, home-1 -> 1), capped to 0-9; --chrome-slot overrides.
    A worker-id with no trailing number falls back to slot 0.
    """
    if override is not None:
        return int(override) % 10
    m = re.search(r"(\d+)\s*$", str(worker_id or ""))
    return (int(m.group(1)) % 10) if m else 0


def classify_apply_channel(tab_urls) -> dict:
    """Record HOW a LinkedIn apply happened, from the browser tabs the agent ended on
    (ZERO LinkedIn scraping -- these tabs were already opened by the apply itself):
      easy_apply -> stayed on linkedin.com (the catastrophe-class on-LinkedIn submit)
      external   -> redirected to an off-LinkedIn ATS (first-party submit, ~no ban risk)
    Returns {'apply_channel': 'easy_apply'|'external'|None, 'apply_external_host': base-host|None}.
    None = no informative tab (can't tell -- record nothing rather than guess)."""
    from urllib.parse import urlparse
    hosts = []
    for u in tab_urls or []:
        p = urlparse(u or "")
        if p.scheme in ("http", "https") and p.hostname:
            h = p.hostname.lower()
            hosts.append(h[4:] if h.startswith("www.") else h)
    if not hosts:
        return {"apply_channel": None, "apply_external_host": None}
    external = [h for h in hosts if not h.endswith("linkedin.com")]
    if external:
        parts = external[0].split(".")
        base = ".".join(parts[-2:]) if len(parts) >= 2 else external[0]
        return {"apply_channel": "external", "apply_external_host": base}
    return {"apply_channel": "easy_apply", "apply_external_host": None}


_KNOWN_EXTERNAL_APPLY_SUFFIXES = (
    "amazon.jobs",
    "ashbyhq.com",
    "greenhouse.io",
    "lever.co",
    "myworkdayjobs.com",
    "myworkdaysite.com",
    "workdayjobs.com",
    "jobvite.com",
    "smartrecruiters.com",
    "icims.com",
    "workable.com",
    "bamboohr.com",
    "paylocity.com",
    "paycomonline.net",
    "ultipro.com",
    "eightfold.ai",
)

_EXTERNAL_TEXT_MARKERS = (
    ("workday", "workday"),
    ("greenhouse", "greenhouse.io"),
    ("ashby", "ashbyhq.com"),
    ("lever", "lever.co"),
)

_HOST_IN_TEXT_RE = re.compile(
    r"(?i)(?<!@)\b((?:[a-z0-9-]+\.)+(?:com|io|co|net|org|jobs|careers|ai))\b"
)


def _known_external_apply_host(host: str | None) -> str | None:
    host = (host or "").strip().lower().strip(".")
    if not host or host.endswith("linkedin.com"):
        return None
    for suffix in _KNOWN_EXTERNAL_APPLY_SUFFIXES:
        if host == suffix or host.endswith(f".{suffix}"):
            return suffix
    return None


def classify_apply_channel_from_text(text: str | None) -> dict:
    """Fallback channel recorder for auth walls after the browser context is gone.

    CDP tab capture is the primary source. Some external ATS auth flows close the
    browser service before we can read `/json`, so use the current agent transcript
    only for known application-host evidence. Generic identity providers such as
    Google SSO are intentionally ignored; those can be true LinkedIn auth walls.
    """
    value = text or ""
    for match in _HOST_IN_TEXT_RE.finditer(value):
        host = _known_external_apply_host(match.group(1))
        if host:
            return {"apply_channel": "external", "apply_external_host": host}

    lowered = value.lower()
    for marker, host in _EXTERNAL_TEXT_MARKERS:
        if marker in lowered:
            return {"apply_channel": "external", "apply_external_host": host}
    return {"apply_channel": None, "apply_external_host": None}


def _cdp_page_urls(port: int) -> list[str]:
    """Best-effort read-only list of open page-tab URLs from Chrome's CDP /json endpoint."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2) as r:
            data = json.loads(r.read().decode("utf-8", "ignore"))
        return [t.get("url", "") for t in data if isinstance(t, dict) and t.get("type") == "page"]
    except Exception:
        return []


_BROWSER_TOOL_RETRY_REASONS = {
    "browser_tool_unavailable",
    "browser_unavailable",
    "browser_service_unavailable",
    "browser_connection_lost",
}


def _browser_tool_retryable(status: str | None) -> bool:
    reason = (status or "").split(":", 1)[-1].strip().lower()
    return reason in _BROWSER_TOOL_RETRY_REASONS


def _prearmed_auth_retryable(status: str | None, stats: dict | None) -> bool:
    normalized = (status or "").strip().lower()
    retryable_status = (
        _browser_tool_retryable(status)
        or normalized.startswith("failed:no_result")
        or normalized.startswith("failed:timeout")
        or normalized.startswith("failed:browser_")
    )
    evidence = stats or {}
    try:
        application_tool_calls = int(evidence.get("application_tool_calls") or 0)
    except (TypeError, ValueError):
        application_tool_calls = 0
    return (
        retryable_status
        and application_tool_calls == 0
        and evidence.get("safe_requeue") is True
    )


_ASSISTED_RETRY_TERMINAL_STATUSES = frozenset({
    "applied",
    "already_applied",
    "dry_run",
    "expired",
    "failed:already_applied",
    "failed:not_eligible_location",
    "failed:not_eligible_work_auth",
    "failed:not_eligible_salary",
    "failed:not_a_job_application",
    "failed:unsafe_permissions",
    "failed:unsafe_verification",
})


def _assisted_retry_is_terminal(status: str | None) -> bool:
    """Accept only final outcomes whose launcher meaning is explicit and bounded."""
    return (status or "").strip().lower() in _ASSISTED_RETRY_TERMINAL_STATUSES


def _should_launch_chrome_headless() -> bool:
    return platform.system() == "Linux" and not bool(os.environ.get("DISPLAY"))


class PgAttemptStore:
    """Bind the durable attempt ledger to one held canonical queue lease."""

    def __init__(self, conn, job: dict, *, worker_id: str):
        self.conn = conn
        self.job = job
        self.worker_id = worker_id

    def create_prepared(self, *, route, route_version, evidence=None):
        from applypilot.fleet import apply_attempts

        return apply_attempts.create_prepared(
            self.conn,
            queue_name="apply_queue",
            url=self.job["url"],
            dedup_key=self.job.get("dedup_key"),
            worker_id=self.worker_id,
            route=route,
            route_version=route_version,
            evidence=evidence,
        )

    def transition(self, attempt_id, **kwargs):
        from applypilot.fleet import apply_attempts

        return apply_attempts.transition(self.conn, attempt_id, **kwargs)

    def mark_browser_interaction(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT public.fleet_worker_mark_browser_interaction() AS marked")
            marked = cur.fetchone()["marked"]
        if not marked:
            self.conn.rollback()
            raise RuntimeError("browser interaction marker rejected for active worker lease")
        self.conn.commit()


def make_apply_fn(
    model: str,
    agent: str,
    slot: int = 0,
    fleet_worker_id: str | None = None,
    *,
    controlled: bool = False,
):
    """Return apply_fn(job) -> {"run_status", "est_cost_usd"} wrapping launcher.run_job.
    Imports launcher LAZILY (after _setup_apply_env).

    Note on chrome API:
      - launch_chrome(worker_id, port=None, ...) -> subprocess.Popen
        (port defaults to BASE_CDP_PORT + worker_id; returns the process, NOT the port)
      - cleanup_worker(worker_id, process) -> bool  (process is the Popen returned above)
    """
    from applypilot.apply import launcher, chrome
    from applypilot.apply.browser_preflight import check_browser_readiness, clear_readiness_cache
    from applypilot.apply.chrome import BASE_CDP_PORT
    from applypilot.apply.lifecycle_fault import (
        enforce_no_lifecycle_faults,
        require_browser_cleanup,
    )
    from applypilot.apply.container_worker import _real_cost
    agent_model = launcher.effective_agent_model_label(agent, model)

    def apply_fn(job: dict, *, attempt_store=None) -> dict:
        emergency_admission.require_allowed(
            emergency_admission.launcher_admission(
                attempt_store.conn if attempt_store is not None else None
            )
        )
        # `slot` keys this worker's Chrome profile + CDP port + per-run logs, so multiple
        # workers on ONE machine (distinct slots) never collide in a shared browser.
        worker_id = slot
        port = BASE_CDP_PORT + worker_id
        previous_fleet_worker_id = os.environ.get("FLEET_WORKER_ID")
        if fleet_worker_id:
            os.environ["FLEET_WORKER_ID"] = str(fleet_worker_id)
        proc = None
        out = None
        try:
            enforce_no_lifecycle_faults()
            launch_kwargs = {"headless": _should_launch_chrome_headless()}
            if controlled:
                launch_kwargs["kill_existing"] = False
            if job.get("session_required"):
                from applypilot.apply import tenant_sessions

                session = tenant_sessions.select_session(
                    job.get("target_host") or job.get("application_url") or job.get("url"),
                    profile_id=job.get("tenant_profile_id"),
                )
                if session["state"] != "ready":
                    out = {
                        "run_status": "auth_required",
                        "est_cost_usd": 0.0,
                        "agent": agent,
                        "agent_model": model,
                        "application_tool_calls": 0,
                        "session_preflight_failure": True,
                        "result_metadata": {"tenant_session": {
                            "host": session["host"],
                            "profile_id": session["profile_id"],
                            "state": session["state"],
                            "reason": session.get("reason"),
                        }},
                    }
                    return out
                launch_kwargs["profile_dir"] = session["profile_dir"]
            proc = chrome.launch_chrome(
                worker_id,
                **launch_kwargs,
            )  # returns Popen; port is implicit BASE_CDP_PORT+slot
            browser_readiness = check_browser_readiness(port)
            if not browser_readiness["ready"]:
                first_readiness = browser_readiness
                require_browser_cleanup(chrome.cleanup_worker, worker_id, proc)
                proc = None
                clear_readiness_cache()
                proc = chrome.launch_chrome(worker_id, **launch_kwargs)
                browser_readiness = check_browser_readiness(port)
            if not browser_readiness["ready"]:
                out = {
                    "run_status": f"failed:browser_preflight:{browser_readiness['reason']}",
                    "est_cost_usd": 0.0,
                    "agent": agent,
                    "agent_model": model,
                    "application_tool_calls": 0,
                    "infrastructure_preflight_failure": True,
                    "infrastructure_restart_attempted": True,
                    "result_metadata": {
                        "browser_preflight": browser_readiness,
                        "browser_preflight_initial": first_readiness,
                    },
                }
                return out
            assisted_retry_count = 0
            assisted_hint_kind = None
            prearmed_request_id = (
                launcher._prearm_inbox_auth_request(
                    job, conn=attempt_store.conn if attempt_store is not None else None
                )
                if launcher._should_prearm_inbox_auth(job)
                else None
            )
            run_kwargs = {"model": model, "agent": agent}
            if attempt_store is not None:
                run_kwargs["attempt_store"] = attempt_store
            status, _dur = launcher.run_job(job, port, worker_id, **run_kwargs)
            first_stats = dict(
                (getattr(launcher, "_last_run_stats", {}) or {}).get(worker_id, {})
            )
            attempt_statuses = [status]
            attempt_stats = [first_stats]
            if prearmed_request_id is not None and (
                launcher._is_auth_required_result(status)
                or _prearmed_auth_retryable(status, first_stats)
            ):
                total_timeout = max(
                    0,
                    int(os.environ.get("APPLYPILOT_INBOX_AUTH_TIMEOUT") or 300),
                )
                postrun_timeout = max(
                    0,
                    int(os.environ.get("APPLYPILOT_INBOX_AUTH_POSTRUN_TIMEOUT") or 45),
                )
                wait_started = time.monotonic()
                inbox_hint = launcher._consume_prearmed_inbox_auth_hint(
                    prearmed_request_id,
                    timeout_seconds=min(total_timeout, postrun_timeout),
                    conn=attempt_store.conn if attempt_store is not None else None,
                )
                if not inbox_hint and launcher._is_auth_required_result(status):
                    elapsed = time.monotonic() - wait_started
                    remaining = max(0, int(total_timeout - elapsed))
                    inbox_hint = launcher._consume_prearmed_inbox_auth_hint(
                        prearmed_request_id,
                        timeout_seconds=remaining,
                        conn=attempt_store.conn if attempt_store is not None else None,
                    )
                if inbox_hint:
                    assisted_retry_count = 1
                    assisted_hint_kind = (
                        "magic_link" if inbox_hint.startswith("magic_link=") else "code"
                    )
                    retry_kwargs = dict(run_kwargs)
                    retry_kwargs["inbox_auth_hint"] = inbox_hint
                    status, _dur = launcher.run_job(
                        job, port, worker_id, **retry_kwargs
                    )
                    attempt_statuses.append(status)
                    attempt_stats.append(
                        dict(
                            (getattr(launcher, "_last_run_stats", {}) or {}).get(
                                worker_id,
                                {},
                            )
                        )
                    )
            stats = attempt_stats[-1]

            def _sum_int(field: str) -> int:
                total = 0
                for item in attempt_stats:
                    try:
                        total += int(item.get(field) or 0)
                    except (TypeError, ValueError):
                        continue
                return total

            attempt_costs = [_real_cost(item, model) for item in attempt_stats]
            total_cost = round(sum(attempt_costs), 4)
            # agent flows to write_apply_result -> llm_usage.provider for per-agent spend.
            out = {
                "run_status": status,
                "est_cost_usd": total_cost,
                "agent": agent,
                "agent_model": agent_model,
                "route": stats.get("route") or "agent",
                "failure_class": stats.get("failure_class"),
                "tool_calls_total": _sum_int("tool_calls_total"),
                "application_tool_calls": _sum_int("application_tool_calls"),
                "last_tool": stats.get("last_tool"),
                "job_log_path": stats.get("job_log_path") or stats.get("job_log"),
                "transcript_digest": stats.get("transcript_digest"),
                "final_result_source": stats.get("final_result_source"),
                "result_metadata": stats.get("result_metadata"),
                "assisted_retry_count": assisted_retry_count,
                "inbox_auth_prearmed": prearmed_request_id is not None,
                "assisted_retry_terminal": (
                    assisted_retry_count == 1
                    and _assisted_retry_is_terminal(status)
                ),
            }
            result_metadata = dict(out.get("result_metadata") or {})
            result_metadata.update({
                "unmapped_required": stats.get("unmapped_required"),
                "unmapped_required_count": stats.get("unmapped_required_count"),
                "attempt_id": stats.get("attempt_id"),
                "route_version": stats.get("route_version"),
                "adapter_version": stats.get("adapter_version"),
                "submit_checkpoint_state": stats.get("submit_checkpoint_state"),
                "verification_status": stats.get("verification_status"),
                "verification_method": stats.get("verification_method"),
                "verification_ref": stats.get("verification_ref"),
                "submission_diagnostics": stats.get("submission_diagnostics"),
                "safe_requeue": stats.get("safe_requeue"),
                "worker_level_failure": stats.get("worker_level_failure"),
                "adapter_name": stats.get("adapter_name"),
                "adapter_plan_ready": stats.get("adapter_plan_ready"),
                "attempt_count": len(attempt_stats),
                "prior_attempt_status": (
                    attempt_statuses[0] if len(attempt_statuses) > 1 else None
                ),
                "prior_attempt_cost_usd": (
                    attempt_costs[0] if len(attempt_costs) > 1 else None
                ),
            })
            result_metadata["inbox_auth"] = {
                "prearmed": prearmed_request_id is not None,
                "assisted_retry_count": assisted_retry_count,
                "hint_kind": assisted_hint_kind,
                "parked_unsupported": (
                    status in {"auth_required", "login_issue"}
                    and assisted_retry_count == 0
                ),
            }
            out["result_metadata"] = result_metadata
            # Record the apply channel from the STILL-OPEN tabs (the finally below kills
            # Chrome). This is needed for non-applied terminal statuses too: an
            # auth_required result on an external ATS is a job-level wall, not a
            # LinkedIn account wall.
            channel = classify_apply_channel(_cdp_page_urls(port))
            transcript_channel = classify_apply_channel_from_text(stats.get("transcript"))
            if status != "applied" and transcript_channel.get("apply_channel") == "external":
                channel = transcript_channel
            out.update(channel)
            return out
        finally:
            if fleet_worker_id:
                if previous_fleet_worker_id is None:
                    os.environ.pop("FLEET_WORKER_ID", None)
                else:
                    os.environ["FLEET_WORKER_ID"] = previous_fleet_worker_id
            cleanup_result = (
                require_browser_cleanup(chrome.cleanup_worker, worker_id, proc)
                if proc is not None
                else False
            )
            cleanup_ok = True if proc is None else (
                cleanup_result is True if controlled else cleanup_result is not False
            )
            if out is not None:
                out["browser_cleanup_ok"] = cleanup_ok
    return apply_fn


def make_log_tail_fn(slot: int, *, n_lines: int = 40):
    """Return a zero-arg callable yielding the LAST ~n_lines of THIS apply worker's rich
    log (the live agent transcript), or None. The launcher writes that file at
    ``config.LOG_DIR / f"worker-{slot}.log"`` (launcher.py:1141, keyed by the Chrome slot
    int). WorkerLoop scrubs + caps the returned text before it is shipped/stored, so this
    only has to read defensively: a missing file (worker hasn't applied yet) -> None, and
    ANY read error -> None (the heartbeat then ships the in-memory event ring instead)."""
    def _tail():
        try:
            from applypilot import config  # the module launcher.run_job writes through
            path = config.LOG_DIR / f"worker-{slot}.log"
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
            return "".join(lines[-n_lines:]) if lines else None
        except Exception:
            return None
    return _tail


def resolve_agent_timeout(conn, *, env_default=None) -> int:
    """Effective apply-agent wall-clock timeout for THIS worker. The FLEET DOCTOR may RAISE
    a bounded override (fleet_config.agent_timeout_override) when a host clusters timeouts; a
    worker PREFERS that override when set, else the env/default (APPLYPILOT_AGENT_TIMEOUT, 300).

    Read-only + best-effort: any DB error falls back to the env/default so a transient blip
    never changes the worker's bound. The override is the Doctor's ONLY conservative timeout
    lever -- it can only ever lengthen the timeout within a ceiling (see doctor._assert_conservative)."""
    default = env_default
    if default is None:
        try:
            default = int(os.environ.get("APPLYPILOT_AGENT_TIMEOUT") or 300)
        except (TypeError, ValueError):
            default = 300
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT public.fleet_worker_admission_snapshot() AS snapshot")
            snapshot = cur.fetchone()["snapshot"] or {}
            row = {"agent_timeout_override": snapshot.get("agent_timeout_override")}
        try:
            conn.rollback()  # read-only
        except Exception:
            pass
        if row is not None:
            ov = row.get("agent_timeout_override") if hasattr(row, "get") else row["agent_timeout_override"]
            if ov is not None:
                # Defensive clamp: never let a bad override SHORTEN the timeout below the
                # default (the Doctor only ever raises it; this guards manual edits too).
                return max(int(default), int(ov))
    except Exception:
        pass
    return int(default)


def _apply_timeout_override(dsn=None, *, conn=None) -> None:
    """If the Doctor set fleet_config.agent_timeout_override, prefer it: assign the launcher's
    module-level AGENT_TIMEOUT_SECONDS (which it reads from APPLYPILOT_AGENT_TIMEOUT at import)
    to the resolved value. No override -> leave the env/default untouched.

    Called BOTH once at startup (with a ``dsn`` -- opens its own short-lived connection) AND on
    EVERY apply tick (with an already-open ``conn`` -- avoids a second connection per tick). The
    per-tick call is what makes a live Doctor timeout_bump actually take effect on a long-lived
    worker: launcher.run_job reads AGENT_TIMEOUT_SECONDS as a module global per-job, so the next
    job after a bump sees the raised value. The 'only reassign if changed' guard keeps it cheap
    and best-effort (a transient DB blip never changes the bound)."""
    try:
        from applypilot.apply import launcher

        def _set_from(c):
            eff = resolve_agent_timeout(c)
            if int(getattr(launcher, "AGENT_TIMEOUT_SECONDS", 0)) != int(eff):
                launcher.AGENT_TIMEOUT_SECONDS = int(eff)

        if conn is not None:
            _set_from(conn)
        else:
            from applypilot.apply import pgqueue
            with pgqueue.connect(dsn) as own:
                _set_from(own)
    except Exception:  # pragma: no cover - best-effort; never block the worker
        logger.debug("could not resolve agent_timeout_override; using env/default", exc_info=True)


def build_apply_loop(*, dsn, worker_id, home_ip, model="sonnet", agent="codex", machine_owner=None, slot=0):
    _setup_apply_env()
    from applypilot.apply.lifecycle_fault import enforce_no_lifecycle_faults

    enforce_no_lifecycle_faults()
    from applypilot.apply import launcher, liveness, pgqueue
    from applypilot.fleet.worker import WorkerLoop
    # Prefer the Doctor's bounded agent_timeout_override when present (else env/default).
    _apply_timeout_override(dsn)
    loop = WorkerLoop(lambda: pgqueue.connect(dsn), worker_id, home_ip=home_ip, role="apply",
                      apply_fn=make_apply_fn(model, agent, slot, fleet_worker_id=worker_id),
                      machine_owner=machine_owner,
                      sw_version=worker_version(),
                      log_tail_fn=make_log_tail_fn(slot),
                      preflight_fn=liveness.probe_url,
                      attempt_store_factory=lambda conn, job: PgAttemptStore(
                          conn, job, worker_id=worker_id
                      ))
    loop.current_agent = agent
    loop.current_model = launcher.effective_agent_model_label(agent, model)
    return loop


# When all agents are usage-limit-walled the worker pauses until the nearer reset. Cap a
# single sleep so a SIGTERM/stop is still honored within a few minutes rather than after a
# multi-hour reset wait.
_PAUSE_CEILING = 300.0


class PgAgentBudget:
    """PG-backed fleet agent budget injected into run_apply. Two jobs, both best-effort
    (run_apply guards every call): (1) share this worker's reactive walls + read other
    workers'/the monitor's blocks via the agent_availability table -- the FLEET-WIDE
    reactive layer; (2) run the throttled predictive spend monitor
    (agent_budget.evaluate_soft_blocks), which pre-emptively blocks an agent over its
    soft cap. With no soft caps the predictive half is inert (fleet-wide reactive still
    works). Blocked-until times cross the boundary as epochs (run_apply speaks epochs)."""

    def __init__(self, *, soft_caps=None, window_seconds=18000.0, cooldown_seconds=1800.0,
                 eval_interval_seconds=120.0, time_fn=None):
        self.soft_caps = soft_caps or {}
        self.window_seconds = float(window_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.eval_interval_seconds = float(eval_interval_seconds)
        self._time = time_fn or time.time
        self._last_eval = None  # None = never evaluated -> first maybe_evaluate runs

    def blocks(self, conn) -> dict:
        from datetime import datetime

        with conn.cursor() as cur:
            cur.execute("SELECT public.fleet_worker_agent_blocks() AS blocks")
            rows = cur.fetchone()["blocks"] or {}
        conn.rollback()
        return {
            agent: datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            for agent, value in rows.items()
        }

    def record_wall(self, conn, agent, blocked_until_epoch) -> None:
        from datetime import datetime, timezone
        with conn.cursor() as cur:
            cur.execute(
                "SELECT public.fleet_worker_record_agent_wall(%s,%s)",
                (agent, datetime.fromtimestamp(blocked_until_epoch, tz=timezone.utc)),
            )
        conn.commit()

    def maybe_evaluate(self, conn) -> None:
        if not any(c and c > 0 for c in self.soft_caps.values()):
            return  # predictive monitor is opt-in (no caps set)
        now = self._time()
        if self._last_eval is not None and now - self._last_eval < self.eval_interval_seconds:
            return
        self._last_eval = now
        with conn.cursor() as cur:
            cur.execute(
                "SELECT public.fleet_worker_evaluate_agent_budget(%s::jsonb,%s,%s) AS blocks",
                (json.dumps(self.soft_caps), int(self.window_seconds), int(self.cooldown_seconds)),
            )
            cur.fetchone()
        conn.commit()


def run_apply(conn_factory, loop, *, max_iterations=None, idle_sleep=5.0,
              switcher=None, rebuild_apply_fn=None, time_fn=None, now_local_fn=None,
              budget=None, model_for_agent=None) -> dict:
    """Drive the apply loop. Before each iteration check should_halt (paused/spend cap)
    and idle when halted. A per-tick error backs off (no hot crash loop). Returns a
    counts dict (testable). Production calls with max_iterations=None (forever).

    Dynamic agent switching (optional): when a ``switcher`` (AgentSwitcher) and
    ``rebuild_apply_fn(agent) -> apply_fn`` are supplied, each tick selects the agent whose
    usage-limit window is open (preferred, else fallback), rebuilding loop.apply_fn when it
    changes. A ``usage_limit`` tick (agent walled, page never touched) records the wall --
    at the reset time parsed from the transcript when present, else a cooldown -- so the
    walled agent is skipped until it resets. When NO agent is open the worker pauses until
    the nearer reset instead of churning the queue. Without a switcher, behavior is
    unchanged."""
    from applypilot.apply import pgqueue
    from applypilot.fleet.agent_switch import parse_reset_at
    _time = time_fn or time.time
    from datetime import datetime
    _now_local = now_local_fn or (lambda: datetime.now().astimezone())
    counts = {"applied": 0, "halted": 0, "idle": 0, "error": 0}
    current_agent = None
    it = 0
    if switcher is not None:
        try:
            loop.agent_chain = ",".join(switcher.agents)
        except Exception:
            pass
    while not _STOP_REQUESTED.is_set() and (max_iterations is None or it < max_iterations):
        it += 1
        try:
            command_handler = (
                getattr(loop, "_handle_commands", None)
                if hasattr(type(loop), "_handle_commands")
                else None
            )
            if callable(command_handler):
                with conn_factory() as cmd_conn:
                    stop = command_handler(cmd_conn)
                    if stop is not None:
                        logger.info(
                            "remote %s command: exiting between jobs before lease gates "
                            "(supervisor respawns)",
                            stop,
                        )
                        break
            # Fleet-wide + predictive layer (optional): pull the shared agent_availability
            # blocks (another worker's reactive wall, or the predictive spend monitor) and
            # feed them into THIS worker's switcher, and run the throttled spend evaluator.
            # Layered on top of the per-worker reactive switch below; never replaces it.
            if budget is not None and switcher is not None:
                now0 = _time()
                with conn_factory() as bconn:
                    try:
                        budget.maybe_evaluate(bconn)
                        switcher.sync_blocks(now0, budget.blocks(bconn))
                    except Exception:  # pragma: no cover - best-effort; never block the worker
                        logger.debug("agent budget sync failed", exc_info=True)
            if switcher is not None:
                now = _time()
                agent = switcher.effective_agent(now)
                reason = (
                    "startup"
                    if current_agent is None
                    else f"switch:{current_agent}->{agent}"
                )
                if agent is None:
                    try:
                        loop.current_agent = None
                        loop.current_model = None
                        loop.last_agent_switch_reason = "all_agents_walled"
                    except Exception:
                        pass
                    # Every agent is walled -> pause until the nearer reset (capped).
                    counts["idle"] += 1
                    beat = getattr(loop, "_beat", None)
                    if callable(beat):
                        try:
                            with conn_factory() as pause_conn:
                                beat(pause_conn, state="paused")
                        except Exception:  # pragma: no cover - never fatal
                            logger.debug("agent-wall heartbeat failed", exc_info=True)
                    if idle_sleep:
                        resume = switcher.resume_at(now)
                        nap = min(max(resume - now, 0.0), _PAUSE_CEILING) if resume is not None else idle_sleep
                        if nap:
                            time.sleep(nap)
                    continue
                if agent != current_agent and rebuild_apply_fn is not None:
                    loop.apply_fn = rebuild_apply_fn(agent)
                    try:
                        loop.current_agent = agent
                        if model_for_agent is not None:
                            loop.current_model = model_for_agent(agent)
                        loop.last_agent_switch_at = _now_local()
                        loop.last_agent_switch_reason = reason
                    except Exception:
                        pass
                    current_agent = agent
                    setter = getattr(loop, "set_agent_telemetry", None)
                    if callable(setter):
                        setter(
                            current_agent=agent,
                            current_model=getattr(loop, "_current_model", None),
                            agent_chain=">".join(getattr(switcher, "agents", [agent])),
                            last_agent_switch_reason=reason,
                        )
            with conn_factory() as conn:
                admission = emergency_admission.worker_tick_admission(conn)
                if not admission.allowed:
                    logger.error(admission.reason)
                    counts["halted"] += 1
                    break
                # Re-resolve the Doctor's agent_timeout_override on EVERY tick (not just at
                # startup): the Doctor sets the override mid-flight while this worker is already
                # running, so a startup-only read would never see a live timeout_bump. The
                # 'only reassign if changed' guard inside makes this cheap, and launcher reads
                # AGENT_TIMEOUT_SECONDS as a module global per-job, so the next job picks it up.
                _apply_timeout_override(conn=conn)
                # H1: the APPLY lane honors the Doctor's ATS-only pause (ats_paused) in addition
                # to the shared kill switch; ats_should_halt OR-s it in. The LinkedIn worker keeps
                # its own linkedin_should_halt(), so a Doctor ATS pause never halts that lane.
                if pgqueue.ats_should_halt(conn):
                    counts["halted"] += 1
                    # Beat while halted: a paused worker that stops beating is indistinguishable
                    # from a dead one -- the watchdog treats it as stuck and the console shows the
                    # fleet as down (live 2026-07-03). state='paused' says "alive, intentionally
                    # idle". Best-effort: a beat write failure must not crash the halt loop.
                    try:
                        loop._beat(conn, state="paused")
                    except Exception:  # pragma: no cover - never fatal
                        logger.debug("halted heartbeat failed", exc_info=True)
                    if idle_sleep:
                        time.sleep(idle_sleep)
                    continue
            res = loop.run_once()
            action = res.get("action")
            if action == "applied":
                counts["applied"] += 1
            elif action == "usage_limit":
                # The current agent hit a usage/session wall (job was re-queued in the loop).
                # Record the wall so the switcher skips this agent until it resets; prefer the
                # reset time parsed from the transcript, else the switcher's cooldown.
                counts["idle"] += 1
                if switcher is not None and current_agent is not None:
                    reset_at = None
                    tail_fn = getattr(loop, "_log_tail_fn", None)
                    tail = tail_fn() if callable(tail_fn) else None
                    dt = parse_reset_at(tail, now_local=_now_local()) if tail else None
                    if dt is not None:
                        reset_at = dt.timestamp()
                    switcher.note_wall(current_agent, _time(), reset_at=reset_at)
                    # Propagate the wall fleet-wide so every worker skips this agent too,
                    # not just the one that discovered it. Use the parsed reset when known,
                    # else the switcher's cooldown (matches the local block).
                    if budget is not None:
                        blocked_epoch = reset_at if reset_at is not None else _time() + switcher.cooldown_seconds
                        try:
                            with conn_factory() as wconn:
                                budget.record_wall(wconn, current_agent, blocked_epoch)
                        except Exception:  # pragma: no cover - best-effort; never block the worker
                            logger.debug("recording fleet wall failed", exc_info=True)
                if idle_sleep:
                    time.sleep(idle_sleep)
            elif action == "stop":
                logger.info("remote %s command: exiting between jobs (supervisor respawns)",
                            res.get("command"))
                break
            elif action in ("idle", "paused"):
                counts["idle"] += 1
                if idle_sleep:
                    time.sleep(idle_sleep)
        except Exception:  # pragma: no cover - logged, backed off, never fatal
            logger.exception("apply tick failed; backing off")
            counts["error"] += 1
            if idle_sleep:
                time.sleep(idle_sleep)
    if _STOP_REQUESTED.is_set():
        logger.info("stop requested (SIGTERM); exiting after current job, before next lease")
    return counts


def enforce_host_identity(machine_owner, *, public_ip, env=None) -> None:
    """Refuse to run a worker that belongs to a DIFFERENT machine than this box.

    Each box declares its own fleet identity in APPLYPILOT_FLEET_LABEL (set once per box:
    'home', 'm2', 'm4'); --machine-owner names WHICH machine's slots a worker fills. If a
    labeled box is asked to run another machine's workers -- e.g. a `-Label m2` agent
    started on the HOME box -- the workers physically run on the wrong host (the live
    2026-07-04 "TARPON/m2 workers spawned on the home box" incident, where home's own
    desired count is 0). When the box is labeled and the two disagree, refuse to start.

    Unknown or incomplete identity is not enrolled and must fail closed.
    """
    require_enrolled_browser_host(
        machine_owner=machine_owner,
        public_ip=public_ip,
        env=env,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="applypilot-fleet-apply")
    p.add_argument("--dsn", default=os.environ.get("FLEET_PG_DSN"))
    p.add_argument("--worker-id", required=True)
    p.add_argument("--home-ip", default=os.environ.get("FLEET_HOME_IP", "0.0.0.0"))
    p.add_argument("--model", default="sonnet")
    # Default agent is Codex (ChatGPT quota pool) so fleet apply stays OFF the
    # Claude Max subscription by default. Pass --agent claude to override, or set
    # --fallback-agent claude to spill over only when Codex hits its wall.
    p.add_argument("--agent", default="codex")
    p.add_argument("--fallback-agent", default=os.environ.get("APPLYPILOT_FALLBACK_AGENT"),
                   help="Comma-separated ordered fallback agents to switch to when --agent "
                        "hits its usage/session limit, e.g. 'claude' (an independent quota "
                        "pool). Omit for none: the worker then pauses until the primary "
                        "agent's window resets.")
    p.add_argument("--machine-owner", default=os.environ.get("FLEET_MACHINE_OWNER"))
    p.add_argument("--chrome-slot", type=int, default=None,
                   help="Browser slot (Chrome profile + CDP port + logs). Auto-derived from "
                        "--worker-id's trailing digits; set explicitly (0,1,2,...) to run "
                        "multiple workers on ONE machine without browser collisions.")
    p.add_argument("--max-iterations", type=int, default=None,
                   help="Stop after this many worker loop ticks. Intended for bounded canaries.")
    p.add_argument("--once", action="store_true",
                   help="Run one worker loop tick, then exit.")
    return p


def main(argv=None) -> int:  # pragma: no cover - long-running
    args = build_parser().parse_args(argv)
    # Defense-in-depth: refuse to physically host another machine's workers (the fleet-agent
    # / run-fleet-worker launchers early-reject too, but this backstops manual + SSH launches).
    enforce_host_identity(args.machine_owner, public_ip=args.home_ip)
    if not args.dsn:
        raise SystemExit("set --dsn or FLEET_PG_DSN")
    slot = _chrome_slot(args.worker_id, args.chrome_slot)
    from applypilot.apply import launcher, pgqueue
    from applypilot.fleet.agent_switch import AgentSwitcher
    from applypilot.fleet import schema as fleet_schema
    with pgqueue.connect_runtime(
        args.dsn, worker_id=args.worker_id, contract="apply"
    ) as schema_conn:
        emergency_admission.require_allowed(
            emergency_admission.worker_admission(
                schema_conn,
                machine_label=os.environ.get("APPLYPILOT_FLEET_LABEL"),
                machine_owner=args.machine_owner,
                worker_id=args.worker_id,
            )
        )
        try:
            fleet_schema.require_apply_result_event_schema(schema_conn)
            fleet_schema.require_apply_attempt_schema(schema_conn)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from None
    loop = build_apply_loop(dsn=args.dsn, worker_id=args.worker_id, home_ip=args.home_ip,
                            model=args.model, agent=args.agent, machine_owner=args.machine_owner,
                            slot=slot)
    # AFTER build_apply_loop: the launcher import inside it installs its own SIGTERM
    # handler (launcher.py); ours must be the LAST writer or the drain kills mid-apply.
    install_stop_handler()
    # Dynamic agent failover: prefer --agent; on its usage/session wall, switch down the
    # --fallback-agent chain (each an independent quota pool) until an earlier agent's window
    # resets. With no fallback the switcher still pauses the worker until the reset (vs
    # churning the queue). Chain order = [primary, *fallbacks].
    try:
        cooldown = float(os.environ.get("APPLYPILOT_USAGE_LIMIT_COOLDOWN_SECONDS") or 3600)
    except (TypeError, ValueError):
        cooldown = 3600.0
    fallbacks = [a.strip() for a in (args.fallback_agent or "").split(",") if a.strip()]
    switcher = AgentSwitcher(agents=[args.agent, *fallbacks], cooldown_seconds=cooldown)

    # Fleet-wide + predictive layer: reactive walls are shared via agent_availability so ONE
    # worker's wall skips the agent fleet-wide; a per-agent soft $ cap (env, opt-in) pre-empts
    # the wall by shifting leases before the agent is exhausted. Spend window + caps from env.
    def _envf(name, default):
        try:
            return float(os.environ.get(name) or default)
        except (TypeError, ValueError):
            return default
    soft_caps = {}
    for a in ("claude", "codex"):
        cap = _envf(f"APPLYPILOT_{a.upper()}_SOFT_CAP_USD", 0.0)
        if cap > 0:
            soft_caps[a] = cap
    budget = PgAgentBudget(
        soft_caps=soft_caps,
        window_seconds=_envf("APPLYPILOT_AGENT_WINDOW_SECONDS", 18000.0),   # 5h rolling
        cooldown_seconds=_envf("APPLYPILOT_AGENT_SOFT_BLOCK_COOLDOWN_SECONDS", 1800.0),
        eval_interval_seconds=_envf("APPLYPILOT_AGENT_EVAL_INTERVAL_SECONDS", 120.0),
    )
    max_iterations = 1 if args.once else args.max_iterations
    run_apply(lambda: pgqueue.connect(args.dsn), loop,
              switcher=switcher,
              rebuild_apply_fn=lambda agent: make_apply_fn(
                  args.model, agent, slot, fleet_worker_id=args.worker_id),
              model_for_agent=lambda agent: launcher.effective_agent_model_label(agent, args.model),
              budget=budget,
              max_iterations=max_iterations)
    return 0
