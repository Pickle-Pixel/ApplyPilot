#!/usr/bin/env python3
"""Fail-closed scanner for Runtime launch-principal safety (Task 9B, launch half).

Root cause this guards against: many tracked Runtime launchers silently construct
fleet PostgreSQL DSNs that select ``user=postgres`` (an ambiguous local admin), the
shared ``fleet_worker`` role, ``PGUSER=postgres``, a role-switching fallback, or a
generic admin DSN. Any of those bypasses the mapped per-node/controller/migrator
topology. A production launch surface must instead require an explicitly supplied
``FLEET_PG_DSN`` (or exact command parameter) and fail closed before any task/process
registration or worker/database action.

The scanner is deterministic and depends on NO hand-maintained allowlist of launcher
filenames. It discovers launch surfaces from the git-tracked tree:

  * root-level PowerShell / Python / shell / batch files,
  * deploy/service templates (``deploy/*``, ``*.template``, ``*.plist.template``),
  * launcher wrappers under ``scripts/``, and
  * every ``[project.scripts]`` console-entry module ("installed/direct launch
    wrappers") plus the first-party modules those entry points directly import
    (so reporting/console helpers reached through the ``applypilot`` CLI are covered).

Comments, examples, help/error text, and generated command lines are all part of the
safety surface: a forbidden principal token is an offence wherever it appears.

The receipt is canonical JSON (sorted keys). It carries base/head placeholders, the
discovered-file count, the offending count, and a SHA-256 discovery digest. It never
contains DSN values or environment secrets -- only file paths, line numbers, and the
name of the forbidden pattern.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import urllib.parse
from dataclasses import dataclass
from pathlib import Path


RECEIPT_SCHEMA = "applypilot.runtime-launch-principals.v1"

# The scanner and its focused test are NOT production launch surfaces: they carry the
# forbidden sentinels on purpose (patterns / fixtures). Self-exclusion is a fixed pair
# of paths, not an allowlist of the launchers under audit.
SELF_EXCLUDED = frozenset(
    {
        "scripts/verify-runtime-launch-principals.py",
        "tests/test_runtime_launch_principals.py",
    }
)

# Directories whose contents are never Runtime launch surfaces.
EXCLUDED_TOP_DIRS = frozenset(
    {
        ".agent-run",
        ".git",
        ".github",  # CI workflows use a throwaway CI-only Postgres; not a runtime launcher
        ".staging-evidence",
        ".superpowers",
        "applypilot-webdriver-cost-research",
        "docs",
        "tests",
    }
)

LAUNCH_SUFFIXES = frozenset({".ps1", ".py", ".sh", ".bat"})


# --- forbidden principal patterns -------------------------------------------------
#
# Each is a (name, compiled-regex) pair. A match anywhere in a discovered launch
# surface -- including comments, help text, error strings, and generated command
# lines -- is an offence. The patterns deliberately require a DSN/assignment context
# (``user=``, ``PGUSER=``, a URL principal, or a role switch) so that legitimate
# REJECTION guards such as ``$Role -in @("postgres", "fleet_worker")`` or
# ``params.get("user") in {"postgres", "fleet_worker"}`` are NOT flagged.
FORBIDDEN_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    # user=postgres in a libpq keyword/value DSN (the P0: ambiguous local admin). The optional
    # quote + free spacing catch user='postgres', user = postgres, and URL ?user=postgres query
    # identities; re.IGNORECASE catches POSTGRES / Postgres mixed-case spellings.
    ("dsn_user_postgres", re.compile(r"\buser\s*=\s*['\"]?postgres\b", re.IGNORECASE)),
    # user=fleet_worker: the shared worker role, not a unique mapped per-node login.
    ("dsn_user_shared_fleet_worker", re.compile(r"\buser\s*=\s*['\"]?fleet_worker\b", re.IGNORECASE)),
    # PGUSER=postgres / $env:PGUSER='fleet_worker': the libpq session env selecting admin/shared.
    # The (?<!\$) lookbehind skips PowerShell params spelled like $PgUser (a script parameter,
    # not the ambient libpq variable) while still catching $env:PGUSER / export PGUSER / PGUSER=.
    ("pguser_admin_or_shared", re.compile(r"(?<!\$)\bPGUSER\s*=\s*['\"]?(?:postgres|fleet_worker)\b", re.IGNORECASE)),
    # postgres://postgres@ or postgresql://fleet_worker:... URL DSN principals.
    ("url_admin_or_shared", re.compile(r"\bpostgres(?:ql)?://(?:postgres|fleet_worker)[:@]", re.IGNORECASE)),
    # SET ROLE / SET SESSION AUTHORIZATION postgres|fleet_worker: role-switching fallback.
    (
        "role_switch_to_admin_or_shared",
        re.compile(r"\bSET\s+(?:ROLE|SESSION\s+AUTHORIZATION)\s+(?:postgres|fleet_worker)\b", re.IGNORECASE),
    ),
    # set_role("postgres") style helper switch.
    ("set_role_call_admin_or_shared", re.compile(r"set_role\s*\(\s*['\"](?:postgres|fleet_worker)\b", re.IGNORECASE)),
    # Generic DSN fallback SOURCE: sourcing a fleet DSN from DATABASE_URL / APPLYPILOT_FLEET_DSN
    # (never FLEET_PG_DSN) is a live-worker fallback that bypasses the mapped-role topology.
    # It matches the fallback IDIOM only -- ``X or environ.get("APPLYPILOT_FLEET_DSN")``,
    # ``environ.get("DATABASE_URL") or Y``, and ``default=... environ.get(...)`` -- so a
    # redaction/scrub read (``_sanitize(..., environ.get("APPLYPILOT_FLEET_DSN", ""))``) or a
    # reject-list membership test (``("APPLYPILOT_FLEET_DSN", ...)``) is NOT flagged.
    (
        "generic_dsn_env_fallback",
        re.compile(
            r"\bor\s+(?:os\.)?(?:environ\.get|getenv)\(\s*['\"](?:DATABASE_URL|APPLYPILOT_FLEET_DSN)['\"]"
            r"|(?:environ\.get|getenv)\(\s*['\"](?:DATABASE_URL|APPLYPILOT_FLEET_DSN)['\"][^)\n]*\)\s*or\b"
            r"|default\s*=\s*[^#\n]*(?:environ\.get|getenv)\(\s*['\"](?:DATABASE_URL|APPLYPILOT_FLEET_DSN)['\"]"
        ),
    ),
    # Help/usage text advertising the retired generic DSN fallbacks as a default source.
    (
        "generic_dsn_help_default",
        re.compile(r"(?i)default[^\n]*\b(?:DATABASE_URL|APPLYPILOT_FLEET_DSN)\b"),
    ),
)

# postgres URL DSN whose userinfo we percent-decode before checking (URI-encoded identities
# such as postgresql://%70ostgres:pw@host/db that the literal patterns above cannot see).
_URL_USERINFO_RE = re.compile(r"\bpostgres(?:ql)?://([^/@\s\"']+)@", re.IGNORECASE)


@dataclass(frozen=True)
class Finding:
    """One forbidden-principal match in a discovered launch surface."""

    file: str
    line: int
    pattern: str


def _url_encoded_admin_principal(line: str) -> bool:
    """True if a postgres URL DSN on ``line`` has a percent-decoded admin/shared userinfo."""
    for match in _URL_USERINFO_RE.finditer(line):
        userinfo = match.group(1)
        principal = userinfo.split(":", 1)[0]
        try:
            decoded = urllib.parse.unquote(principal)
        except Exception:  # pragma: no cover - unquote never raises on str, defensive only
            decoded = principal
        if decoded.strip().lower() in {"postgres", "fleet_worker"}:
            return True
    return False


def scan_text(text: str) -> list[Finding]:
    """Return every forbidden-principal match in ``text`` (path filled by caller)."""
    findings: list[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for name, pattern in FORBIDDEN_PATTERNS:
            if pattern.search(line):
                findings.append(Finding(file="", line=lineno, pattern=name))
        if _url_encoded_admin_principal(line):
            findings.append(Finding(file="", line=lineno, pattern="url_encoded_admin_or_shared"))
    return findings


def scan_file(repo: Path, relpath: str) -> list[Finding]:
    """Scan a single discovered file, tagging findings with its repo-relative path."""
    text = (repo / relpath).read_text(encoding="utf-8", errors="replace")
    return [Finding(file=relpath, line=f.line, pattern=f.pattern) for f in scan_text(text)]


def _git_tracked_files(repo: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return [p for p in result.stdout.split("\0") if p]


def _is_path_pattern_surface(relpath: str) -> bool:
    """True for root launchers, deploy/service templates, and scripts/ wrappers."""
    posix = relpath.replace("\\", "/")
    top = posix.split("/", 1)[0]
    if top in EXCLUDED_TOP_DIRS:
        return False
    name = posix.rsplit("/", 1)[-1]
    depth = posix.count("/")

    # Service/deploy templates anywhere they are shipped from (root or deploy/).
    if name.endswith(".template") or name.endswith(".plist.template"):
        return True
    if top == "deploy":
        return True

    # Root-level PowerShell / Python / shell / batch launchers.
    if depth == 0 and Path(name).suffix in LAUNCH_SUFFIXES:
        return True

    # Launcher / setup wrappers shipped under scripts/.
    if top == "scripts" and depth == 1 and Path(name).suffix in LAUNCH_SUFFIXES:
        return True

    return False


def _module_to_relpath(module: str) -> str | None:
    """Map a first-party dotted module to its repo-relative source file (src layout)."""
    if not module.startswith("applypilot"):
        return None
    return "src/" + module.replace(".", "/") + ".py"


def _console_entry_modules(repo: Path) -> list[str]:
    """Return the module files backing every [project.scripts] console entry point."""
    pyproject = repo / "pyproject.toml"
    if not pyproject.exists():
        return []
    text = pyproject.read_text(encoding="utf-8")
    modules: list[str] = []
    in_scripts = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_scripts = stripped == "[project.scripts]"
            continue
        if not in_scripts or "=" not in stripped or stripped.startswith("#"):
            continue
        target = stripped.split("=", 1)[1].strip().strip('"').strip("'")
        dotted = target.split(":", 1)[0]
        rel = _module_to_relpath(dotted)
        if rel is not None:
            modules.append(rel)
    return modules


def _first_party_imports(repo: Path, relpath: str) -> list[str]:
    """One-hop static scan: first-party modules directly imported by ``relpath``."""
    try:
        tree = ast.parse((repo / relpath).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            rel = _module_to_relpath(node.module)
            if rel is not None:
                found.append(rel)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                rel = _module_to_relpath(alias.name)
                if rel is not None:
                    found.append(rel)
    return found


# ``python -m applypilot.fleet.<module>`` invoked from a launch surface. Deterministic and
# allowlist-free: the module target is read straight out of the launcher text (PowerShell,
# shell, or Python), so a reachable live helper started via -m is audited like a console entry.
_DASH_M_MODULE_RE = re.compile(r"-m\s+[\"']?(applypilot(?:\.\w+)+)")


def _python_dash_m_modules(text: str) -> list[str]:
    """First-party modules launched via ``python -m <module>`` from a launcher's text."""
    modules: list[str] = []
    for match in _DASH_M_MODULE_RE.finditer(text):
        rel = _module_to_relpath(match.group(1))
        if rel is not None:
            modules.append(rel)
    return modules


def _add_module_and_imports(repo: Path, module_rel: str, tracked: set[str], surface: set[str]) -> None:
    """Add a first-party module file plus its one-hop first-party imports to the surface."""
    if module_rel not in tracked or module_rel in SELF_EXCLUDED:
        return
    surface.add(module_rel)
    for imported in _first_party_imports(repo, module_rel):
        if imported in tracked and imported not in SELF_EXCLUDED:
            surface.add(imported)


def discover_launch_surfaces(repo: Path) -> list[str]:
    """Return the sorted, de-duplicated repo-relative launch-surface file list."""
    tracked = set(_git_tracked_files(repo))
    surface: set[str] = set()

    for relpath in tracked:
        if relpath in SELF_EXCLUDED:
            continue
        if _is_path_pattern_surface(relpath):
            surface.add(relpath)

    # Installed/direct launch wrappers: console-script modules + the first-party
    # modules they directly import (reporting/console helpers reached via the CLI).
    for module_rel in _console_entry_modules(repo):
        _add_module_and_imports(repo, module_rel, tracked, surface)

    # ``python -m <module>`` targets reached from the path-pattern launchers discovered above
    # (root/scripts launchers, deploy templates). A live helper started via -m -- e.g.
    # ``python -m applypilot.fleet.autotriage_main`` -- is audited even without a console entry.
    for relpath in sorted(surface):
        try:
            text = (repo / relpath).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for module_rel in _python_dash_m_modules(text):
            _add_module_and_imports(repo, module_rel, tracked, surface)

    return sorted(surface)


def discovery_digest(repo: Path, surfaces: list[str]) -> str:
    """SHA-256 over the sorted (relpath, content-hash) pairs of discovered files.

    Captures both WHICH files were discovered and their exact content, so a receipt
    pins the audited surface. Contains no DSN values or secrets.
    """
    hasher = hashlib.sha256()
    for relpath in sorted(surfaces):
        content = (repo / relpath).read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()
        hasher.update(relpath.replace("\\", "/").encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(content_hash.encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def scan_repository(repo: Path) -> dict:
    """Discover launch surfaces, scan them, and build the canonical receipt dict."""
    surfaces = discover_launch_surfaces(repo)
    findings: list[Finding] = []
    for relpath in surfaces:
        findings.extend(scan_file(repo, relpath))

    offenders = [
        {"file": f.file.replace("\\", "/"), "line": f.line, "pattern": f.pattern}
        for f in findings
    ]
    offenders.sort(key=lambda o: (o["file"], o["line"], o["pattern"]))

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "base": os.environ.get("APPLYPILOT_SCAN_BASE", "<base>"),
        "head": os.environ.get("APPLYPILOT_SCAN_HEAD", "<head>"),
        "discovered_file_count": len(surfaces),
        "offending_count": len(offenders),
        "discovery_sha256": discovery_digest(repo, surfaces),
        "offenders": offenders,
    }
    return receipt


def _canonical_json(receipt: dict) -> str:
    return json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parents[1]),
        help="repository root (defaults to the scanner's parent repo)",
    )
    parser.add_argument("--receipt", help="write the canonical JSON receipt to this path")
    parser.add_argument(
        "--print-receipt",
        action="store_true",
        help="print the canonical JSON receipt to stdout",
    )
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    receipt = scan_repository(repo)

    if args.receipt:
        Path(args.receipt).write_text(_canonical_json(receipt) + "\n", encoding="utf-8")
    if args.print_receipt:
        print(_canonical_json(receipt))
    else:
        print(
            f"[runtime-launch-principals] discovered {receipt['discovered_file_count']} launch "
            f"surface(s); {receipt['offending_count']} offending occurrence(s); "
            f"discovery={receipt['discovery_sha256'][:12]}"
        )
        for offender in receipt["offenders"]:
            print(f"  OFFENCE {offender['file']}:{offender['line']} [{offender['pattern']}]")

    return 1 if receipt["offending_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
