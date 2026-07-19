"""``python -m applypilot.fleet.launch_guard``: fail closed unless FLEET_PG_DSN names
the exact approved unique mapped principal through a complete explicit runtime DSN.

Runtime launch surfaces call this BEFORE any file/task/process/database side effect. It
reuses the single hardened conninfo parser (``pgqueue.assert_mapped_principal``) -- there
is no second, weaker parser -- so a forbidden ``postgres`` / shared ``fleet_worker``
principal is rejected in libpq keyword DSNs, URL DSNs (including percent-encoded and
query-string identities), mixed case / quoted spellings, ambient libpq variables, indirect
``service`` / ``passfile`` forms, role-changing options, incomplete authentication fields,
and unsafe transport settings. It exits non-zero with a redacted message and never prints
the DSN value.
"""

from __future__ import annotations

import sys

from applypilot.apply import pgqueue


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        dsn = pgqueue.get_dsn()
        principal = pgqueue.assert_mapped_principal(dsn)
    except RuntimeError as exc:  # message is already redacted (never carries the DSN)
        sys.stderr.write(f"[launch-guard] {exc}\n")
        return 78
    sys.stderr.write(f"[launch-guard] mapped principal '{principal}' accepted\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
