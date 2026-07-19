"""applypilot-fleet-diagnose: read a worker's log, diagnose the apply-failure root cause,
write an advisory row to fleet_diagnoses, and print it. ADVISORY — takes no fleet actions."""
from __future__ import annotations
import argparse
import sys

from applypilot.apply import pgqueue
from applypilot.fleet import diagnoser, schema as fleet_schema


def _failing_workers(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT worker_id FROM apply_queue WHERE worker_id IS NOT NULL "
                    "AND status IN ('failed','crash_unconfirmed') "
                    "AND updated_at > now() - interval '20 minutes'")
        return [r["worker_id"] for r in cur.fetchall()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="applypilot-fleet-diagnose",
                                description="Advisory log-reading diagnosis of apply failures.")
    p.add_argument("--dsn", default=None,
                   help="explicit mapped fleet DSN (default: FLEET_PG_DSN env; no generic fallback).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--worker", help="diagnose a single worker_id")
    g.add_argument("--all-failing", action="store_true",
                   help="diagnose every worker with recent failures")
    args = p.parse_args(argv)

    with pgqueue.connect(args.dsn) as conn:
        fleet_schema.ensure_schema_v3(conn)
        workers = [args.worker] if args.worker else _failing_workers(conn)
        if not workers:
            print("no failing workers in the last 20 min")
            return 0
        for w in workers:
            ctx = diagnoser.load_worker_ctx(conn, w)
            d = diagnoser.diagnose(ctx)
            diagnoser.write_diagnosis(conn, d)
            print(f"[{w}] {d.root_cause} ({d.source}, conf {d.confidence:.2f}): {d.recommendation}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
