"""applypilot-fleet-remediate: autonomously re-queue usage-limit-casualty jobs behind the 3-layer
double-apply guard. --once for a single pass; --interval to loop. ATS only, bounded, reversible."""
from __future__ import annotations
import argparse
import sys
import time

from applypilot.apply import pgqueue
from applypilot.fleet import remediator, schema as fleet_schema


def _one_pass(args) -> None:
    with pgqueue.connect(args.dsn) as conn:
        out = remediator.remediate(conn, max_requeue=args.max_requeue,
                                   max_per_job=args.max_per_job, window_minutes=args.window_minutes,
                                   backfill=args.usage_limit_backfill,
                                   pre_touch_backfill=args.pre_touch_backfill,
                                   dry_run=args.dry_run)
    print(f"[remediate] candidates={out['candidates']} requeued={out['requeued']} "
          f"vetoed_applied_set={out['vetoed_applied_set']} vetoed_email={out['vetoed_email']} "
          f"capped={out['capped']} dry_run={out.get('dry_run', False)}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="applypilot-fleet-remediate",
        description="Re-queue usage-limit-casualty jobs (3-guard double-apply gate). ATS only.")
    p.add_argument("--dsn", default=None,
                   help="explicit mapped fleet DSN (default: FLEET_PG_DSN env; no generic fallback).")
    p.add_argument("--max-requeue", type=int, default=50, dest="max_requeue",
                   help="per-pass blast-radius cap (default 50)")
    p.add_argument("--max-per-job", type=int, default=2, dest="max_per_job",
                   help="max re-queues per job ever (default 2)")
    p.add_argument("--window-minutes", type=int, default=30, dest="window_minutes")
    p.add_argument("--usage-limit-backfill", action="store_true", dest="usage_limit_backfill",
                   help="Phase 2.4/C12: select ALL failed+usage_limit ATS rows with NO time "
                        "window (status-keyed, not update-time-keyed) -- for casualties older "
                        "than any sane --window-minutes. Same 3-guard pipeline as the default "
                        "windowed mode; only the candidate query differs.")
    p.add_argument("--pre-touch-backfill", action="store_true",
                   help="Include failed no_browser_tool rows with durable zero tool calls.")
    p.add_argument("--dry-run", action="store_true",
                   help="Select and guard-check candidates without requeueing them.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true", help="single pass")
    g.add_argument("--interval", type=int, help="loop every N seconds")
    args = p.parse_args(argv)

    with pgqueue.connect(args.dsn) as schema_conn:
        fleet_schema.ensure_schema_v3(schema_conn)

    if args.once:
        _one_pass(args)
        return 0
    while True:                       # --interval loop
        _one_pass(args)
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
