#!/usr/bin/env python3
"""Bootstrap ApplyPilot with a canned profile (e.g. Gustaf Garnow).

Usage:
    python scripts/bootstrap_profile.py gustaf [--force] [--target /custom/path]

Copies profile.json, resume.txt, searches.yaml, and .env from profiles/<name>
into ~/.applypilot (or --target). Existing files are left untouched unless
--force is provided.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = ROOT / "profiles"
DEFAULT_APP_DIR = Path(os.environ.get("APPLYPILOT_DIR", Path.home() / ".applypilot"))


def copy_file(src: Path, dest: Path, force: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[skip] {dest} already exists (use --force to overwrite)")
        return
    shutil.copy2(src, dest)
    print(f"[ok] {src.name} -> {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap ApplyPilot profile data")
    parser.add_argument("profile", help="Profile folder name under profiles/ (e.g. gustaf)")
    parser.add_argument(
        "--target",
        help="Destination ApplyPilot dir (defaults to ~/.applypilot or $APPLYPILOT_DIR)",
        default=None,
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    profile_dir = PROFILE_ROOT / args.profile
    if not profile_dir.exists():
        print(f"Profile '{args.profile}' not found in {PROFILE_ROOT}")
        return 1

    target_root = Path(args.target).expanduser() if args.target else DEFAULT_APP_DIR
    target_root.mkdir(parents=True, exist_ok=True)

    print(f"Target directory: {target_root}")

    copied_any = False
    for filename in ("profile.json", "resume.txt", "searches.yaml", ".env"):
        src = profile_dir / filename
        if not src.exists():
            continue
        dest = target_root / filename
        copy_file(src, dest, args.force)
        copied_any = True

    if not copied_any:
        print("Nothing to copy — ensure the profile folder contains config files.")
        return 1

    print("Done. Run 'applypilot doctor' to verify the setup.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
