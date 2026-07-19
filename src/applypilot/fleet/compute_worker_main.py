"""applypilot-fleet-compute: a compute worker (score + audit) for owner-controlled
machines. Reads PG DSN + LLM key/provider from the local env, loads the shared
context, and runs the WorkerLoop. Compute is IP-free (no browser, no site traffic)."""
from __future__ import annotations

import argparse
import os

from applypilot.apply import pgqueue
from applypilot.fleet import compute_context as cc
from applypilot.fleet import schema as fleet_schema
from applypilot.fleet.compute_adapters import make_audit_fn, make_score_fn
from applypilot.fleet.emergency_admission import compute_worker_admission, require_allowed
from applypilot.fleet.worker import WorkerLoop

# How many run_once iterations between context-version checks. Keep low enough
# to pick up a re-published resume/KG without being DB-heavy (one extra SELECT
# per N jobs is negligible vs the LLM cost).
_VERSION_CHECK_INTERVAL = 10


def _require_context(ctx, version: str) -> None:
    if not ctx.resume_text.strip():
        detail = "ctx:resume is empty" if version else "ctx:version and ctx:resume are missing"
        raise RuntimeError(
            f"published compute context invalid: {detail}; run "
            "applypilot-fleet-compute-home publish-context on the home machine"
        )


def build_compute_loop(conn, *, dsn, worker_id, home_ip, providers, fallback, ensemble,
                       machine_owner=None) -> tuple[WorkerLoop, str]:
    """Build the WorkerLoop and return (loop, initial_version) so callers can
    implement periodic re-fetch without calling load_context a second time."""
    ctx, version = cc.load_context(conn, providers=providers, fallback=fallback, ensemble=ensemble)
    _require_context(ctx, version)
    fns = {"score": make_score_fn(ctx), "audit": make_audit_fn(ctx)}
    loop = WorkerLoop(lambda: pgqueue.connect(dsn), worker_id, home_ip=home_ip, role="compute",
                      compute_fns=fns, machine_owner=machine_owner)
    return loop, version


def maybe_refresh_context(conn, loop: WorkerLoop, *, current_version: str,
                           providers, fallback, ensemble) -> str:
    """Check whether the published context version has changed.

    If the stored ctx:version differs from *current_version* AND is non-empty,
    rebuild loop.compute_fns in place from the freshly loaded context and return
    the new version string.  Otherwise return *current_version* unchanged (the
    dict object is NOT replaced, preserving identity for the no-op case).
    """
    ctx, new_version = cc.load_context(conn, providers=providers, fallback=fallback,
                                       ensemble=ensemble)
    if new_version and new_version != current_version:
        _require_context(ctx, new_version)
        loop.compute_fns = {"score": make_score_fn(ctx), "audit": make_audit_fn(ctx)}
        return new_version
    return current_version


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="applypilot-fleet-compute")
    p.add_argument("--dsn", default=os.environ.get("FLEET_PG_DSN"))
    p.add_argument("--worker-id", required=True)
    p.add_argument("--home-ip", default=os.environ.get("FLEET_HOME_IP", "0.0.0.0"))
    p.add_argument("--providers", default=os.environ.get("LLM_SCORE_PROVIDER", "deepseek"))
    p.add_argument("--fallback", default=os.environ.get("LLM_SCORE_FALLBACK", ""))
    p.add_argument("--ensemble", action="store_true", default=bool(os.environ.get("FLEET_ENSEMBLE")))
    p.add_argument("--machine-owner", default=os.environ.get("FLEET_MACHINE_OWNER"))
    args = p.parse_args(argv)
    if not args.dsn:
        raise SystemExit("set --dsn or FLEET_PG_DSN")
    providers = [s for s in args.providers.split(",") if s]
    fallback = [s for s in args.fallback.split(",") if s]

    with pgqueue.connect_runtime(
        args.dsn, worker_id=args.worker_id, contract="compute"
    ) as conn:
        fleet_schema.require_apply_result_event_schema(conn)
        fleet_schema.require_apply_attempt_schema(conn)
        require_allowed(compute_worker_admission(conn))
        loop, ctx_version = build_compute_loop(
            conn, dsn=args.dsn, worker_id=args.worker_id, home_ip=args.home_ip,
            providers=providers, fallback=fallback, ensemble=args.ensemble,
            machine_owner=args.machine_owner,
        )

    # Drive run_once ourselves so we can periodically re-check the context version.
    # run_forever is still available for callers that don't need the re-fetch
    # (e.g. very short-lived workers), but the recommended path for long-running
    # workers is this loop.
    import time
    iteration = 0
    while True:  # pragma: no cover
        try:
            res = loop.run_once()
        except Exception as exc:
            try:
                import traceback
                tb = traceback.format_exc()
                loop._last_error = tb[:4000]
                loop._record_event(f"ERROR: {exc}")
                with pgqueue.connect(args.dsn) as conn:
                    loop._beat(conn, state="error")
            except Exception:
                pass
            res = {"action": "error"}
        if res.get("action") == "stop":
            return 0  # remote restart/drain: exit between jobs; supervisor respawns
        if res.get("action") in ("idle", "paused", "error"):
            time.sleep(5.0)
        iteration += 1
        if iteration % _VERSION_CHECK_INTERVAL == 0:
            with pgqueue.connect(args.dsn) as conn:
                ctx_version = maybe_refresh_context(
                    conn, loop, current_version=ctx_version,
                    providers=providers, fallback=fallback, ensemble=args.ensemble,
                )

    return 0  # unreachable; satisfies type-checkers
