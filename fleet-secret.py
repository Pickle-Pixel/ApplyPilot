"""fleet-secret.py get|put <name> [value]

Tiny helper over the fleet_assets blob store (same table the cloud worker uses for
profile.json/resume). Lets worker boxes fetch a shared secret (e.g. the DeepSeek key)
from the Postgres they already talk to -- so no per-machine .env copy is needed.

  get <name>          -> prints the stored value (nothing if absent)
  put <name> <value>  -> stores it (use the --stdin form to keep it off the cmdline)
  put <name> --stdin  -> reads the value from stdin (never appears in process args)

Reads FLEET_PG_DSN from the env (an explicit mapped fleet DSN; no generic fallback)."""
import sys

from applypilot.apply import pgqueue


def main(argv):
    if len(argv) < 2:
        print("usage: fleet-secret.py get|put <name> [value|--stdin]", file=sys.stderr)
        return 2
    cmd, name = argv[0], argv[1]
    conn = pgqueue.connect()
    if cmd == "get":
        data = pgqueue.get_asset(conn, name)
        if data:
            sys.stdout.write(data.decode("utf-8", "replace"))
        return 0
    if cmd == "put":
        if len(argv) >= 3 and argv[2] == "--stdin":
            value = sys.stdin.read().strip()
        elif len(argv) >= 3:
            value = argv[2]
        else:
            print("put needs a value or --stdin", file=sys.stderr)
            return 2
        pgqueue.put_asset(conn, name, value.encode("utf-8"))
        print(f"stored {name} ({len(value)} chars)")
        return 0
    print(f"unknown cmd {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
