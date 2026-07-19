"""applypilot-fleet-autotriage: bounded autonomous triage for apply failures."""
from __future__ import annotations

import argparse
import sys
import time

from applypilot.apply import pgqueue
from applypilot.fleet import autotriage, schema as fleet_schema


def _one_pass(args) -> dict:
    with pgqueue.connect(args.dsn) as conn:
        return autotriage.run_pass(
            conn,
            brain_path=args.brain_path,
            limit=args.limit,
            window_minutes=args.window_minutes,
            enable_llm=args.enable_llm,
            dry_run=args.dry_run,
            crash_only=args.crash_only,
        )


def _format_summary(out: dict) -> str:
    actions = ",".join(f"{k}={v}" for k, v in sorted((out.get("actions") or {}).items())) or "none"
    statuses = ",".join(f"{k}={v}" for k, v in sorted((out.get("statuses") or {}).items())) or "none"
    return (
        f"[autotriage] contexts={out.get('contexts', 0)} applied={out.get('applied', 0)} "
        f"actions({actions}) statuses({statuses})"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="applypilot-fleet-autotriage",
        description="Autonomous bounded LLM/rule triage for recent apply failures.",
    )
    p.add_argument("--dsn", default=None, help="explicit mapped fleet DSN (default: FLEET_PG_DSN env; no generic fallback).")
    p.add_argument("--brain-path", default=None, help="SQLite brain path for confirming-email vetoes.")
    p.add_argument("--limit", type=int, default=50, help="Max recent failure rows to triage per pass.")
    p.add_argument("--window-minutes", type=int, default=1440, help="Failure lookback window.")
    p.add_argument("--enable-llm", action="store_true", help="Let the LLM choose from the fixed action menu.")
    p.add_argument("--dry-run", action="store_true", help="Write audit rows but do not apply actions.")
    p.add_argument("--crash-only", action="store_true",
                   help="Limit selection to crash_unconfirmed rows; never triage failed/blocked rows.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true", help="Run a single pass.")
    g.add_argument("--interval", type=int, help="Loop every N seconds.")
    args = p.parse_args(argv)

    with pgqueue.connect(args.dsn) as schema_conn:
        fleet_schema.ensure_schema_v3(schema_conn)

    if args.once:
        print(_format_summary(_one_pass(args)), flush=True)
        return 0

    while True:
        print(_format_summary(_one_pass(args)), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
