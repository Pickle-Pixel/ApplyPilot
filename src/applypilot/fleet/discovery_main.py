"""applypilot-fleet-discovery: a discovery worker for residential machines.
Leases search_tasks from Postgres, runs JobSpy scrapes, and stages postings
back to discovered_postings. No compute, no apply — pure scrape-and-stage.

applypilot-fleet-discovery-home: fills or drains the discovery pipeline from
the home box (expand search config -> search_tasks; pull discovered postings
from PG into the brain).
"""
from __future__ import annotations

import argparse
import json
import os

from applypilot.apply import pgqueue
from applypilot.fleet.discovery_adapter import make_search_fn
from applypilot.fleet import schema as fleet_schema
from applypilot.fleet.emergency_admission import discovery_worker_admission, require_allowed
from applypilot.fleet.worker import WorkerLoop


# ---------------------------------------------------------------------------
# Core builder — returns a single WorkerLoop (no context-version dance needed
# for discovery: the search config is cheap to re-load if it changes).
# ---------------------------------------------------------------------------

def build_discovery_loop(
    *,
    dsn: str,
    worker_id: str,
    home_ip: str,
    results_per_site: int,
    hours_old: int,
    proxy: str | None,
    search_cfg: dict | None = None,
) -> WorkerLoop:
    """Build and return a WorkerLoop configured for the discovery role.

    Unlike build_compute_loop, this returns a single value (the loop) — there
    is no context version to track because the search config is stateless and
    re-loaded on each expand_searches call.
    """
    search_fn = make_search_fn(
        results_per_site=results_per_site,
        hours_old=hours_old,
        proxy=proxy,
        search_cfg=search_cfg,
    )
    loop = WorkerLoop(
        lambda: pgqueue.connect(dsn),
        worker_id,
        home_ip=home_ip,
        role="discovery",
        search_fn=search_fn,
    )
    return loop


# ---------------------------------------------------------------------------
# Home-box thin delegates
# ---------------------------------------------------------------------------

def expand_searches(conn, config: dict) -> int:
    """Thin delegate: expand a search config dict into search_tasks rows."""
    from applypilot.fleet.scheduler import expand_search_config
    return expand_search_config(conn, config)


def pull(*, sqlite_conn=None, pg_conn=None) -> int:
    """Thin delegate: pull discovered postings from PG into the brain."""
    from applypilot.fleet.sync import pull_discovered
    return pull_discovered(sqlite_conn=sqlite_conn, pg_conn=pg_conn)


# ---------------------------------------------------------------------------
# Worker entrypoint
# ---------------------------------------------------------------------------

def main_worker(argv=None) -> int:
    p = argparse.ArgumentParser(prog="applypilot-fleet-discovery")
    p.add_argument("--dsn", default=os.environ.get("FLEET_PG_DSN"))
    p.add_argument("--worker-id", required=True)
    p.add_argument("--home-ip", default=os.environ.get("FLEET_HOME_IP", "0.0.0.0"))
    p.add_argument("--results-per-site", type=int, default=50)
    p.add_argument("--hours-old", type=int, default=72)
    p.add_argument("--proxy", default=os.environ.get("FLEET_PROXY") or None)
    args = p.parse_args(argv)
    if not args.dsn:
        raise SystemExit("set --dsn or FLEET_PG_DSN")

    with pgqueue.connect_runtime(
        args.dsn, worker_id=args.worker_id, contract="discovery"
    ) as conn:
        fleet_schema.require_apply_result_event_schema(conn)
        fleet_schema.require_apply_attempt_schema(conn)
        require_allowed(discovery_worker_admission(conn))
    loop = build_discovery_loop(
        dsn=args.dsn,
        worker_id=args.worker_id,
        home_ip=args.home_ip,
        results_per_site=args.results_per_site,
        hours_old=args.hours_old,
        proxy=args.proxy,
    )
    loop.run_forever()
    return 0  # unreachable; satisfies type-checkers


# ---------------------------------------------------------------------------
# Home entrypoint
# ---------------------------------------------------------------------------

def main_home(argv=None) -> int:
    p = argparse.ArgumentParser(prog="applypilot-fleet-discovery-home")
    p.add_argument("cmd", choices=["expand", "pull"])
    p.add_argument("--dsn", default=os.environ.get("FLEET_PG_DSN"))
    p.add_argument("--config", default=None,
                   help="Path to searches JSON or YAML config (required for expand)")
    args = p.parse_args(argv)

    if not args.dsn:
        raise SystemExit("set --dsn or FLEET_PG_DSN")
    with pgqueue.connect(args.dsn) as schema_conn:
        fleet_schema.ensure_schema_v3(schema_conn)

    if args.cmd == "expand":
        if not args.config:
            raise SystemExit("expand requires --config <path>")
        path = args.config
        if path.endswith(".yaml") or path.endswith(".yml"):
            import yaml
            with open(path, "r", encoding="utf-8") as fh:
                config = yaml.safe_load(fh) or {}
        else:
            with open(path, "r", encoding="utf-8") as fh:
                config = json.load(fh)
        with pgqueue.connect(args.dsn) as conn:
            n = expand_searches(conn, config)
        print("expanded", n)
    else:  # pull
        with pgqueue.connect(args.dsn) as pg:
            n = pull(pg_conn=pg)
        print("pulled", n)
    return 0
