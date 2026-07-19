#!/usr/bin/env python3
"""Deterministic inventory of the Runtime brain command-authority surface.

Discovers the complete Runtime command surface (Python entrypoints and
subcommands, MCP/HTTP registrations, launchers, scheduled tasks and generated
wrappers, containers, services, and launchd definitions) and prints it as
strict canonical JSON. With ``--check`` it also loads the checked-in matrix and
asserts an exact bijection, exiting non-zero on any unclassified, stale, or
drifted surface.

The inventory is byte-identical across runs and free of secrets, DSNs,
credentials, absolute user paths, and timestamps.
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
        help="repository root to discover (defaults to this checkout)",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help="matrix document to check discovery against",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail closed unless discovery and the matrix are an exact bijection",
    )
    arguments = parser.parse_args(argv)

    try:
        discovered = authority.discover(arguments.repo_root)
        if arguments.check:
            matrix_path = arguments.matrix or (
                arguments.repo_root / "config" / "brain-command-authority-matrix.runtime.json"
            )
            matrix, _content = authority.load_matrix_document(matrix_path)
            rows = authority.validate_matrix(matrix)
            authority.check_coverage(discovered, rows)
    except authority.AuthorityError as error:
        print(f"inventory failed: {error}", file=sys.stderr)
        return 1

    sys.stdout.write(authority.canonical_json(discovered).decode("ascii") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
