"""Controller-owned CLI for explicit ApplyPilot fleet schema migration."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from applypilot.apply import pgqueue
from applypilot.fleet import migrator


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_PATH = _REPOSITORY_ROOT / "src" / "applypilot" / "fleet" / "migrations" / "manifest-v1.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="applypilot-fleet-migrate-schema")
    parser.add_argument(
        "--dsn",
        default=None,
        help="explicit mapped fleet owner/migrator Postgres DSN (default: FLEET_PG_DSN; no generic fallback).",
    )
    args = parser.parse_args(argv)

    try:
        dsn = pgqueue.get_dsn(args.dsn)
        manifest = migrator.load_manifest(_MANIFEST_PATH)
        with pgqueue.connect_admin(dsn) as conn:
            result = migrator.apply_manifest(conn, manifest, _REPOSITORY_ROOT)
    except Exception as exc:
        print(f"fleet schema migration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        "fleet schema migration complete: "
        f"applied={len(result.applied)} already_applied={len(result.already_applied)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
