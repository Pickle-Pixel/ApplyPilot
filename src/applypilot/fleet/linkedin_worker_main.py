"""applypilot-fleet-linkedin: the HOME-BOX LinkedIn apply worker.

Drives WorkerLoop(role='linkedin') against the LinkedIn queue. A single advisory
lock ('applypilot:linkedin_driver') ensures only one LinkedIn driver runs at a time
-- the owner box is the sole origin for LinkedIn applies; having two drivers would
race on session state and inflate the risk of an account ban.

The advisory lock is held via a DEDICATED long-lived connection for the life of the
process. Releasing the connection releases the lock -- so any crash or clean exit
automatically frees the lock for the supervised launcher (Task 7).
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time

from applypilot.fleet.browser_host_guard import require_linkedin_owner_host

logger = logging.getLogger("applypilot.fleet.linkedin_worker_main")

_PAUSE_CEILING = 300.0


def _setup_apply_env() -> None:
    """Mirror apply_worker_main._setup_apply_env (home-box flavored)."""
    repo_app_dir = Path(__file__).resolve().parents[3] / ".applypilot"
    if (repo_app_dir / "profile.json").exists():
        os.environ.setdefault("APPLYPILOT_DIR", str(repo_app_dir))
        for env_name, filename in (
            ("APPLYPILOT_PROFILE_PATH", "profile.json"),
            ("APPLYPILOT_RESUME_PATH", "resume.txt"),
            ("APPLYPILOT_RESUME_PDF_PATH", "resume.pdf"),
            ("APPLYPILOT_RESUME_STRATEGY_PATH", "resume_strategy.yaml"),
            ("APPLYPILOT_PREFERENCE_PROFILE_PATH", "job_preference_profile.json"),
            ("APPLYPILOT_KNOWLEDGE_GRAPH_PROMPT_PATH", "job_knowledge_graph_prompt.md"),
        ):
            path = repo_app_dir / filename
            if path.exists():
                os.environ.setdefault(env_name, str(path))
    os.environ["APPLYPILOT_BASE_RESUME"] = "1"
    # Fleet row selection is lane-filtered at push time; keep worker-side acquire opt-in.
    os.environ.setdefault("APPLYPILOT_LANE_FILTER", "0")
    os.environ.setdefault("APPLYPILOT_DB_PATH", os.path.join(os.environ.get("TEMP", "/tmp"), "fleet_apply_throwaway.db"))
    os.environ.setdefault("CHROME_WORKER_DIR", os.path.join(os.environ.get("TEMP", "/tmp"), "chrome-workers"))
    os.environ.setdefault("APPLY_WORKER_DIR", os.path.join(os.environ.get("TEMP", "/tmp"), "apply-workers"))
    os.environ.setdefault("APPLYPILOT_AGENT_TIMEOUT", "300")

    # Some callers import pgqueue/config before building the LinkedIn loop. Refresh the
    # path constants so load_profile()/resume lookup still sees the env set above.
    cfg = sys.modules.get("applypilot.config")
    if cfg is not None:
        app_dir = Path(os.environ.get("APPLYPILOT_DIR", str(cfg.APP_DIR)))
        cfg.APP_DIR = app_dir
        cfg.DB_PATH = Path(os.environ.get("APPLYPILOT_DB_PATH", str(cfg.DB_PATH)))
        cfg.PROFILE_PATH = Path(os.environ.get("APPLYPILOT_PROFILE_PATH", str(app_dir / "profile.json")))
        cfg.RESUME_PATH = Path(os.environ.get("APPLYPILOT_RESUME_PATH", str(app_dir / "resume.txt")))
        cfg.RESUME_PDF_PATH = Path(os.environ.get("APPLYPILOT_RESUME_PDF_PATH", str(app_dir / "resume.pdf")))
        cfg.RESUME_STRATEGY_PATH = Path(
            os.environ.get("APPLYPILOT_RESUME_STRATEGY_PATH", str(app_dir / "resume_strategy.yaml"))
        )
        cfg.PREFERENCE_PROFILE_PATH = Path(
            os.environ.get("APPLYPILOT_PREFERENCE_PROFILE_PATH", str(app_dir / "job_preference_profile.json"))
        )
        cfg.KNOWLEDGE_GRAPH_PROMPT_PATH = Path(
            os.environ.get("APPLYPILOT_KNOWLEDGE_GRAPH_PROMPT_PATH", str(app_dir / "job_knowledge_graph_prompt.md"))
        )


def acquire_linkedin_interlock(conn) -> bool:
    """Try to acquire the advisory lock for the LinkedIn driver.

    Returns True if acquired (this process is the sole LinkedIn driver), False if
    another session already holds it. The lock is SESSION-level: it is released
    automatically when *conn* is closed, so the caller must keep *conn* open for
    the process lifetime.

    The key string is EXACTLY 'applypilot:linkedin_driver' -- byte-identical with
    the probe in the supervised launcher (Task 7).
    """
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(hashtext('applypilot:linkedin_driver')) AS ok")
        ok = cur.fetchone()["ok"]
    conn.commit()
    return bool(ok)


def build_linkedin_loop(
    *,
    dsn,
    worker_id,
    public_ip,
    owner_ip,
    model="sonnet",
    agent="codex",
    machine_owner=None,
):
    """Construct a WorkerLoop for the linkedin role.

    public_ip is the enrolled node's observed identity and owner_ip is the configured
    owner identity. The entrypoint requires them to match before constructing the loop.
    apply_fn is built via the same make_apply_fn path used by apply_worker_main --
    run_job is URL-agnostic; the LinkedIn-seeded Chrome profile is what
    setup_worker_profile already prefers for li_at sessions. We do NOT force a
    fresh profile here.
    """
    _setup_apply_env()
    from applypilot.apply import pgqueue
    from applypilot.fleet.worker import WorkerLoop
    from applypilot.fleet.apply_worker_main import PgAttemptStore, make_apply_fn, make_log_tail_fn

    from applypilot.apply import launcher
    loop = WorkerLoop(
        lambda: pgqueue.connect(dsn),
        worker_id,
        home_ip=public_ip,
        role="linkedin",
        apply_fn=make_apply_fn(model, agent, fleet_worker_id=worker_id),
        machine_owner=machine_owner,
        public_ip=public_ip,
        owner_ip=owner_ip,
        on_owner_machine=True,
        log_tail_fn=make_log_tail_fn(0),
        attempt_store_factory=lambda conn, job: PgAttemptStore(
            conn, job, worker_id=worker_id
        ),
    )
    loop.current_agent = agent
    loop.current_model = launcher.effective_agent_model_label(agent, model)
    return loop


def run_linkedin(conn_factory, loop, *, max_iterations=None, idle_sleep=5.0,
                 switcher=None, rebuild_apply_fn=None, time_fn=None, now_local_fn=None,
                 budget=None, model_for_agent=None) -> dict:
    """Drive the LinkedIn apply loop (mirrors run_apply from apply_worker_main).

    Before each iteration check the LinkedIn-specific shared kill switch and idle when
    halted. Lease-time gates still enforce the LinkedIn canary, account halt, daily cap,
    and mutex. A per-tick error backs off without crashing. Returns a counts dict
    (testable). Production calls with max_iterations=None (forever).
    """
    from applypilot.apply import pgqueue
    from applypilot.fleet import emergency_admission
    from applypilot.fleet.agent_switch import parse_reset_at
    from datetime import datetime
    _time = time_fn or time.time
    _now_local = now_local_fn or (lambda: datetime.now().astimezone())
    counts = {"applied": 0, "halted": 0, "idle": 0, "error": 0}
    current_agent = None
    it = 0
    if switcher is not None:
        try:
            loop.agent_chain = ",".join(switcher.agents)
        except Exception:
            pass
    while max_iterations is None or it < max_iterations:
        it += 1
        try:
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
                if agent is None:
                    try:
                        loop.current_agent = None
                        loop.current_model = None
                        loop.last_agent_switch_reason = "all_agents_walled"
                    except Exception:
                        pass
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
                        loop.last_agent_switch_reason = "agent_available"
                    except Exception:
                        pass
                    current_agent = agent
            with conn_factory() as conn:
                admission = emergency_admission.linkedin_tick_admission(conn)
                if not admission.allowed:
                    logger.error(admission.reason)
                    counts["halted"] += 1
                    break
                if pgqueue.linkedin_should_halt(conn):
                    counts["halted"] += 1
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
                counts["idle"] += 1
                if switcher is not None and current_agent is not None:
                    reset_at = None
                    tail_fn = getattr(loop, "_log_tail_fn", None)
                    tail = tail_fn() if callable(tail_fn) else None
                    dt = parse_reset_at(tail, now_local=_now_local()) if tail else None
                    if dt is not None:
                        reset_at = dt.timestamp()
                    switcher.note_wall(current_agent, _time(), reset_at=reset_at)
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
            logger.exception("linkedin tick failed; backing off")
            counts["error"] += 1
            if idle_sleep:
                time.sleep(idle_sleep)
    return counts


def main(argv=None) -> int:  # pragma: no cover - long-running
    p = argparse.ArgumentParser(prog="applypilot-fleet-linkedin")
    p.add_argument("--dsn", default=os.environ.get("FLEET_PG_DSN"))
    p.add_argument("--worker-id", required=True)
    p.add_argument("--public-ip", default=os.environ.get("FLEET_HOME_IP", "0.0.0.0"))
    p.add_argument("--owner-ip", default=os.environ.get("FLEET_OWNER_IP", "0.0.0.0"))
    p.add_argument("--model", default="sonnet")
    # Match the apply-lane default: Codex uses the ChatGPT quota pool and avoids
    # burning the Claude Max subscription unless the operator explicitly opts in.
    p.add_argument("--agent", default="codex")
    p.add_argument("--fallback-agent", default=os.environ.get("APPLYPILOT_LINKEDIN_FALLBACK_AGENT")
                   or os.environ.get("APPLYPILOT_FALLBACK_AGENT"),
                   help="Comma-separated ordered fallback agents to switch to when --agent "
                        "hits its usage/session limit. Omit for none: the worker then pauses "
                        "until the primary agent's window resets.")
    p.add_argument("--machine-owner", default=os.environ.get("FLEET_MACHINE_OWNER"))
    args = p.parse_args(argv)
    require_linkedin_owner_host(
        machine_owner=args.machine_owner,
        public_ip=args.public_ip,
        owner_ip=args.owner_ip,
    )
    if not args.dsn:
        raise SystemExit("set --dsn or FLEET_PG_DSN")

    _setup_apply_env()
    from applypilot.apply import pgqueue
    from applypilot.fleet import emergency_admission, schema as fleet_schema

    with pgqueue.connect_runtime(
        args.dsn, worker_id=args.worker_id, contract="linkedin"
    ) as schema_conn:
        emergency_admission.require_allowed(
            emergency_admission.linkedin_worker_admission(schema_conn)
        )
        try:
            fleet_schema.require_apply_result_event_schema(schema_conn)
            fleet_schema.require_apply_attempt_schema(schema_conn)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from None
        with schema_conn.cursor() as cur:
            cur.execute("SELECT public.fleet_worker_admission_snapshot() AS snapshot")
            startup = cur.fetchone()["snapshot"] or {}
        schema_conn.rollback()
    configured_owner_ip = startup.get("linkedin_owner_ip")
    if not configured_owner_ip:
        raise SystemExit("LinkedIn owner IP is not configured in fleet control state")
    require_linkedin_owner_host(
        machine_owner=args.machine_owner,
        public_ip=args.public_ip,
        owner_ip=configured_owner_ip,
    )

    # Open a DEDICATED long-lived connection solely for holding the advisory lock.
    # This connection is kept open for the process life; releasing it releases the lock.
    interlock_conn = pgqueue.connect(args.dsn)
    if not acquire_linkedin_interlock(interlock_conn):
        interlock_conn.close()
        raise SystemExit("another LinkedIn driver holds the interlock")

    logger.info("advisory lock acquired: applypilot:linkedin_driver")

    loop = build_linkedin_loop(
        dsn=args.dsn,
        worker_id=args.worker_id,
        public_ip=args.public_ip,
        owner_ip=configured_owner_ip,
        model=args.model,
        agent=args.agent,
        machine_owner=args.machine_owner,
    )
    from applypilot.fleet.agent_switch import AgentSwitcher
    from applypilot.fleet.apply_worker_main import PgAgentBudget, make_apply_fn

    try:
        cooldown = float(os.environ.get("APPLYPILOT_USAGE_LIMIT_COOLDOWN_SECONDS") or 3600)
    except (TypeError, ValueError):
        cooldown = 3600.0
    fallbacks = [a.strip() for a in (args.fallback_agent or "").split(",") if a.strip()]
    switcher = AgentSwitcher(agents=[args.agent, *fallbacks], cooldown_seconds=cooldown)

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
        window_seconds=_envf("APPLYPILOT_AGENT_WINDOW_SECONDS", 18000.0),
        cooldown_seconds=_envf("APPLYPILOT_AGENT_SOFT_BLOCK_COOLDOWN_SECONDS", 1800.0),
        eval_interval_seconds=_envf("APPLYPILOT_AGENT_EVAL_INTERVAL_SECONDS", 120.0),
    )
    try:
        run_linkedin(
            lambda: pgqueue.connect(args.dsn),
            loop,
            switcher=switcher,
            rebuild_apply_fn=lambda agent: make_apply_fn(
                args.model, agent, fleet_worker_id=args.worker_id),
            budget=budget,
        )
    finally:
        try:
            interlock_conn.close()  # releases the advisory lock
        except Exception:
            pass
    return 0
