"""fleet-status.py [label-prefix]

Print live fleet status from the home Postgres: worker heartbeats (LIVE if beat < 2.5
min old), and -- when a label prefix is given (e.g. m4) -- that machine's desired apply
workers and the compute_queue depth. Requires FLEET_PG_DSN to be an explicit mapped fleet
DSN (never the postgres admin or the shared fleet_worker role); refuses to run without it."""
import os
import sys

label = sys.argv[1] if len(sys.argv) > 1 else ""
if not os.environ.get("FLEET_PG_DSN"):
    sys.exit(
        "FLEET_PG_DSN is not set: export an explicit mapped fleet DSN "
        "(never the postgres admin or the shared fleet_worker role). Refusing to read fleet status.")

from applypilot.apply import pgqueue  # noqa: E402  (import deferred until FLEET_PG_DSN is verified)

c = pgqueue.connect()
cur = c.cursor()

like = (label + "-%") if label else "%"
title = label or "all"
print(f"=== workers ({title}) — LIVE = beat < 2.5 min ===")
cur.execute(
    "SELECT worker_id, role, state, "
    "round(extract(epoch from (now()-last_beat))/60,1) AS age_min, machine_owner "
    "FROM worker_heartbeat WHERE worker_id LIKE %s ORDER BY worker_id", (like,))
rows = cur.fetchall()
if not rows:
    print("  (no matching heartbeats)")
for r in rows:
    fresh = "LIVE " if float(r["age_min"]) < 2.5 else "stale"
    print(f"  [{fresh}] {r['worker_id']:<15} {r['role']:<9} {r['state']:<11} "
          f"beat {r['age_min']}min ago  owner={r['machine_owner']}")

if label:
    cur.execute("SELECT desired_workers, agent, model FROM fleet_desired_state "
                "WHERE machine_owner=%s", (label,))
    row = cur.fetchone()
    print(f"\n=== {label} desired apply workers ===")
    print("  ", dict(row) if row else "(no row)")

print("\n=== compute_queue (scoring backlog) ===")
cur.execute("SELECT status, count(*) AS n FROM compute_queue GROUP BY status ORDER BY n DESC")
q = [dict(r) for r in cur.fetchall()]
print("  ", q or "empty")
