#!/usr/bin/env python3
"""Verify the Runtime brain command-authority matrix and emit its receipt.

Runs the full fail-closed gate: refuses a dirty tree, pins the exact head/tree,
validates the checked-in matrix (no unsupported/extra/missing/duplicate/
noncanonical data, no secrets/DSNs/credentials/absolute paths/timestamps),
asserts an exact discovery<->matrix bijection with zero unclassified surfaces,
and prints a strict canonical repository receipt (schema version, runtime kind,
head/tree, discovery/matrix digests, counts, clean state, self-hash) to stdout.

Exits non-zero on any gate failure.  Repeated runs are byte-identical.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import brain_command_authority as authority  # noqa: E402


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_default_repo_root(),
        help="repository root to verify (defaults to this checkout)",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help="matrix document to verify (defaults to the checked-in runtime matrix)",
    )
    arguments = parser.parse_args(argv)
    # Relative matrix paths are repository-relative. Keep the default relative
    # too, so a relative --repo-root is not accidentally prefixed twice.
    matrix_path = arguments.matrix or Path(
        "config/brain-command-authority-matrix.runtime.json"
    )

    try:
        receipt = authority.evaluate(arguments.repo_root, matrix_path)
    except authority.AuthorityError as error:
        print(f"verification failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001 - fail closed on any unexpected error
        print(f"verification failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    sys.stdout.write(authority.canonical_json(receipt).decode("ascii") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
