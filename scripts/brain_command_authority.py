"""Fail-closed library for the Runtime brain command-authority matrix.

This module owns three responsibilities for the Runtime half of the Task 9A
command-authority discovery/receipt:

* Deterministic, repository-owned, structural discovery of the Python *process
  entrypoints* that make up the Runtime command surface -- console scripts,
  directly runnable modules, directly runnable stand-alone files, and the MCP
  tool registry.  Discovery is not a filename allowlist: a surface is found
  because of what it *is* (a ``[project.scripts]`` entry, an ``if __name__ ==
  '__main__'`` guard, top-level side-effecting code, or an ``@mcp.tool()``
  decorator), so any new/renamed/removed entrypoint changes the discovered set
  and trips the fail-closed gates until the checked-in matrix is refreshed.

* Validation of the checked-in authority matrix against discovery: every
  discovered surface must have exactly one fully classified row and every row
  must bind to a discovered surface.  Unknown / new / missing / duplicate /
  stale / mixed-mode / unclassified surfaces fail closed.

* A strict canonical repository receipt binding the exact head/tree, the sorted
  discovery digest, the sorted matrix digest, the counts, a zero-unclassified
  assertion, the clean-tree state, and a self-hash.  The receipt is explicitly
  scoped to this Runtime shard and makes no global authority / C2 claim.

The matrix grants *no* live authority in this shard: every inventoried surface
is default-deny (``liveEligibility`` false, ``authoritySource`` ``none``).  The
live-writer rules are still enforced so that any attempt to promote a surface to
live is held to the exact role/source/caps/scopes/resources/fail-closed
contract.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Schema constants
# --------------------------------------------------------------------------- #

MATRIX_SCHEMA_VERSION = "applypilot-brain-command-authority-matrix-runtime-v1"
RECEIPT_SCHEMA_VERSION = "applypilot-brain-command-authority-receipt-runtime-v1"
RUNTIME_KIND = "python-process-entrypoints"
REPOSITORY = "applypilot"

INCLUDED_SURFACE_KINDS = (
    "console-script",
    "mcp-tool",
    "python-file",
    "python-module",
)

# Surface kinds that are real but deliberately out of this bounded shard.  They
# are declared (never silently dropped) and deferred to sibling Runtime packets
# and the cross-repo pairing.  The receipt disclaims any global authority so
# this shard is never mistaken for a complete command-authority receipt.
DEFERRED_SURFACE_KINDS = (
    "argparse-subcommand",
    "batch-launcher",
    "console-http-action",
    "container-entrypoint",
    "launchd-plist",
    "powershell-launcher",
    "scheduled-task-identity",
    "service-unit",
    "shell-launcher",
    "typer-subcommand",
)

READ_WRITE_CLASSES = (
    "admin-ddl",
    "control",
    "none",
    "read",
    "write",
)

OWNERS = (
    "brain-admin",
    "fleet-console",
    "fleet-worker",
    "operator",
    "release-engineer",
)

# Principals that a *live* writer may never run as.  Shared or admin principals
# defeat least-privilege attribution, so any live writer bound to one fails
# closed.
SHARED_ADMIN_PRINCIPALS = frozenset(
    {"shared", "admin", "postgres", "root", "superuser", "brain-admin"}
)

PAUSE_SCOPE_VOCABULARY = (
    "apply",
    "brain-admin",
    "compute",
    "control",
    "discovery",
    "email",
    "linkedin",
    "migration",
    "none",
    "otp",
    "release",
)

MATRIX_TOP_LEVEL_KEYS = frozenset(
    {
        "schemaVersion",
        "runtimeKind",
        "repository",
        "noGlobalAuthorityClaim",
        "scope",
        "pauseScopeVocabulary",
        "rows",
    }
)

SCOPE_KEYS = frozenset(
    {"includedSurfaceKinds", "deferredSurfaceKinds", "deferralReason"}
)

# Structural fields are produced verbatim by discovery; the matrix must rebind
# them exactly so a moved/renamed/mixed-mode surface cannot hide behind a stale
# classification.
STRUCTURAL_ROW_KEYS = (
    "id",
    "repository",
    "sourcePath",
    "symbol",
    "surfaceKind",
    "canonicalTarget",
    "aliases",
    "installed",
    "directInvocation",
)

CLASSIFICATION_ROW_KEYS = (
    "owner",
    "readWriteClass",
    "resourcesRead",
    "resourcesWritten",
    "employerMutation",
    "authoritySource",
    "capabilities",
    "pauseScopes",
    "liveEligibility",
    "failClosedBehavior",
)

ROW_KEYS = frozenset(STRUCTURAL_ROW_KEYS) | frozenset(CLASSIFICATION_ROW_KEYS)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_MAIN_GUARD_TARGET = "__main__"


class AuthorityError(RuntimeError):
    """Raised for any fail-closed gate violation."""


# --------------------------------------------------------------------------- #
# Canonical JSON
# --------------------------------------------------------------------------- #


def canonical_json(value: Any) -> bytes:
    """Deterministic, sorted, ASCII canonical JSON bytes."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuthorityError(f"duplicate object key is not permitted: {key}")
        result[key] = value
    return result


def strict_json_loads(content: bytes, label: str) -> Any:
    try:
        return json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorityError(f"{label} is not valid UTF-8 JSON") from exc


def _reject_constant(value: str) -> None:
    raise AuthorityError(f"non-standard JSON constant is not permitted: {value}")


# --------------------------------------------------------------------------- #
# Structural discovery
# --------------------------------------------------------------------------- #


def _read_text(path: Path) -> str:
    # ``utf-8-sig`` tolerates a leading BOM that some checked-in sources carry
    # while still rejecting genuinely non-UTF-8 content.
    return path.read_bytes().decode("utf-8-sig")


def _module_to_source(repo_root: Path, module: str) -> str:
    """Resolve a dotted module to a repo-relative POSIX source path."""
    src_root = repo_root / "src"
    base = src_root / Path(*module.split("."))
    package_init = base / "__init__.py"
    module_file = base.with_suffix(".py")
    if module_file.is_file():
        chosen = module_file
    elif package_init.is_file():
        chosen = package_init
    else:
        # Record the would-be module file so the verifier flags a broken /
        # stale entrypoint rather than silently dropping it.
        chosen = module_file
    return chosen.relative_to(repo_root).as_posix()


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    left, right = test.left, test.comparators[0]
    names = {left, right}
    has_dunder_name = any(isinstance(n, ast.Name) and n.id == "__name__" for n in names)
    has_main = any(
        isinstance(n, ast.Constant) and n.value == _MAIN_GUARD_TARGET for n in names
    )
    return has_dunder_name and has_main


def _has_main_guard(tree: ast.Module) -> bool:
    return any(_is_main_guard(node) for node in tree.body)


def _is_type_checking_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _is_library_module(tree: ast.Module) -> bool:
    """A stand-alone .py is a *library* (not a runnable script) when every
    top-level statement is a declaration: imports, defs, classes, a docstring,
    plain assignments (module constants), ``__all__``, ``pass``, the main guard,
    or a ``TYPE_CHECKING`` guard.  Any bare top-level action (a call statement,
    a loop, a ``with``/``try`` block) marks it runnable.
    """
    for node in tree.body:
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Assign,
                ast.AnnAssign,
                ast.AugAssign,
                ast.Pass,
            ),
        ):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # module docstring / bare literal
        if _is_type_checking_guard(node):
            continue
        if _is_main_guard(node):
            return False  # a main guard makes it runnable
        return False  # a bare action statement makes it runnable
    return True


def _mcp_tool_decorated(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute) and target.attr == "tool":
            value = target.value
            if isinstance(value, ast.Name) and value.id == "mcp":
                return True
            if isinstance(value, ast.Attribute) and value.attr == "mcp":
                return True
    return False


def _iter_python_files(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return
    yield from sorted(directory.rglob("*.py"), key=lambda p: p.as_posix())


def _dotted_module(repo_root: Path, path: Path) -> str:
    relative = path.relative_to(repo_root / "src")
    parts = list(relative.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


def _surface(
    *,
    surface_id: str,
    source_path: str,
    symbol: str,
    surface_kind: str,
    canonical_target: str,
    aliases: Iterable[str],
    installed: bool,
    direct_invocation: bool,
) -> dict[str, Any]:
    return {
        "id": surface_id,
        "repository": REPOSITORY,
        "sourcePath": source_path,
        "symbol": symbol,
        "surfaceKind": surface_kind,
        "canonicalTarget": canonical_target,
        "aliases": sorted(set(aliases)),
        "installed": installed,
        "directInvocation": direct_invocation,
    }


def _discover_console_scripts(repo_root: Path) -> list[dict[str, Any]]:
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        return []
    data = tomllib.loads(_read_text(pyproject))
    scripts = data.get("project", {}).get("scripts", {})
    surfaces: list[dict[str, Any]] = []
    for name in sorted(scripts):
        target = scripts[name]
        module, sep, attr = target.partition(":")
        if not sep or not module or not attr:
            raise AuthorityError(
                f"console script {name!r} has a non-canonical target: {target!r}"
            )
        source_path = _module_to_source(repo_root, module)
        surfaces.append(
            _surface(
                surface_id=f"console-script:{name}",
                source_path=source_path,
                symbol=attr,
                surface_kind="console-script",
                canonical_target=f"{module}:{attr}",
                # The installed command name is the alias. ``python -m <module>``
                # is a *distinct* surface (the module's own ``__main__``) and is
                # owned by the python-module row, not the console entry -- two
                # console scripts can share a module without sharing that form.
                aliases=[name],
                installed=True,
                direct_invocation=True,
            )
        )
    return surfaces


def _discover_python_modules(repo_root: Path) -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    for path in _iter_python_files(repo_root / "src"):
        tree = ast.parse(_read_text(path))
        if not _has_main_guard(tree):
            continue
        dotted = _dotted_module(repo_root, path)
        relative = path.relative_to(repo_root).as_posix()
        surfaces.append(
            _surface(
                surface_id=f"python-module:{dotted}",
                source_path=relative,
                symbol=_MAIN_GUARD_TARGET,
                surface_kind="python-module",
                canonical_target=f"{dotted}:{_MAIN_GUARD_TARGET}",
                aliases=[f"python -m {dotted}", f"python {relative}"],
                installed=False,
                direct_invocation=True,
            )
        )
    return surfaces


def _discover_python_files(repo_root: Path) -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    candidates: list[Path] = sorted(
        (p for p in repo_root.glob("*.py")), key=lambda p: p.as_posix()
    )
    candidates += sorted(
        (p for p in (repo_root / "scripts").glob("*.py")),
        key=lambda p: p.as_posix(),
    )
    for path in candidates:
        tree = ast.parse(_read_text(path))
        if _is_library_module(tree):
            continue
        relative = path.relative_to(repo_root).as_posix()
        symbol = _MAIN_GUARD_TARGET if _has_main_guard(tree) else "<module-body>"
        surfaces.append(
            _surface(
                surface_id=f"python-file:{relative}",
                source_path=relative,
                symbol=symbol,
                surface_kind="python-file",
                canonical_target=relative,
                aliases=[f"python {relative}"],
                installed=False,
                direct_invocation=True,
            )
        )
    return surfaces


def _discover_mcp_tools(repo_root: Path) -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    for path in _iter_python_files(repo_root / "src"):
        tree = ast.parse(_read_text(path))
        dotted = _dotted_module(repo_root, path)
        relative = path.relative_to(repo_root).as_posix()
        for node in tree.body:
            if not _mcp_tool_decorated(node):
                continue
            func = node.name
            surfaces.append(
                _surface(
                    surface_id=f"mcp-tool:{dotted}:{func}",
                    source_path=relative,
                    symbol=func,
                    surface_kind="mcp-tool",
                    canonical_target=f"{dotted}:{func}",
                    aliases=[f"mcp:{func}"],
                    installed=False,
                    direct_invocation=False,
                )
            )
    return surfaces


def discover(repo_root: Path) -> list[dict[str, Any]]:
    """Return the sorted, canonical list of discovered Runtime surfaces.

    Deterministic: the same tree always yields byte-identical output.  Raises
    :class:`AuthorityError` on a duplicate surface id (a collision that must not
    be silently coalesced).
    """
    repo_root = Path(repo_root)
    surfaces: list[dict[str, Any]] = []
    surfaces += _discover_console_scripts(repo_root)
    surfaces += _discover_python_modules(repo_root)
    surfaces += _discover_python_files(repo_root)
    surfaces += _discover_mcp_tools(repo_root)

    seen: dict[str, dict[str, Any]] = {}
    for surface in surfaces:
        surface_id = surface["id"]
        if surface_id in seen:
            raise AuthorityError(f"duplicate discovered surface id: {surface_id}")
        seen[surface_id] = surface
    return [seen[key] for key in sorted(seen)]


def discovery_digest(surfaces: list[dict[str, Any]]) -> str:
    return sha256_hex(canonical_json(sorted(surfaces, key=lambda s: s["id"])))


# --------------------------------------------------------------------------- #
# Forbidden-content scan
# --------------------------------------------------------------------------- #

_FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("postgres DSN", re.compile(r"postgres(?:ql)?://", re.IGNORECASE)),
    ("keyword DSN", re.compile(r"\bdbname\s*=", re.IGNORECASE)),
    ("connection secret", re.compile(r"\bpassword\s*=", re.IGNORECASE)),
    ("credential token", re.compile(r"(?:secret|api[_-]?key|token)\s*[:=]", re.IGNORECASE)),
    ("windows user path", re.compile(r"[A-Za-z]:\\\\?Users\\\\?", re.IGNORECASE)),
    ("windows drive path", re.compile(r"[A-Za-z]:\\\\", re.IGNORECASE)),
    ("posix home path", re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+")),
    ("iso timestamp", re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")),
)


def scan_forbidden_content(value: Any, label: str) -> None:
    """Fail closed on secrets, DSNs, credentials, absolute user paths, and
    nondeterministic timestamps anywhere in ``value``."""
    text = canonical_json(value).decode("utf-8")
    for name, pattern in _FORBIDDEN_PATTERNS:
        if pattern.search(text):
            raise AuthorityError(f"{label} contains forbidden {name}")


# --------------------------------------------------------------------------- #
# Matrix validation
# --------------------------------------------------------------------------- #


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def _sorted_unique_str_list(value: Any, label: str) -> list[str]:
    _require(isinstance(value, list), f"{label} must be a list")
    _require(all(isinstance(item, str) for item in value), f"{label} must be strings")
    _require(value == sorted(value), f"{label} must be sorted canonically")
    _require(len(value) == len(set(value)), f"{label} must not contain duplicates")
    return value


def _validate_scope(scope: Any) -> None:
    _require(isinstance(scope, dict), "scope must be an object")
    _require(set(scope) == SCOPE_KEYS, "scope has unexpected or missing keys")
    included = _sorted_unique_str_list(scope["includedSurfaceKinds"], "includedSurfaceKinds")
    deferred = _sorted_unique_str_list(scope["deferredSurfaceKinds"], "deferredSurfaceKinds")
    _require(
        tuple(included) == INCLUDED_SURFACE_KINDS,
        "scope includedSurfaceKinds must exactly match the shard predicate",
    )
    _require(
        tuple(deferred) == DEFERRED_SURFACE_KINDS,
        "scope deferredSurfaceKinds must exactly match the declared deferrals",
    )
    _require(
        isinstance(scope["deferralReason"], str) and scope["deferralReason"].strip(),
        "scope deferralReason must be a non-empty string",
    )


def _validate_row_shape(row: Any, index: int) -> None:
    _require(isinstance(row, dict), f"row {index} must be an object")
    _require(
        set(row) == ROW_KEYS,
        f"row {index} has unexpected or missing keys: "
        f"{sorted(set(row) ^ ROW_KEYS)}",
    )
    _require(row["repository"] == REPOSITORY, f"row {index} has a foreign repository")
    _require(row["surfaceKind"] in INCLUDED_SURFACE_KINDS, f"row {index} has an unknown surfaceKind")
    _require(row["readWriteClass"] in READ_WRITE_CLASSES, f"row {index} has an unknown readWriteClass")
    _require(row["owner"] in OWNERS, f"row {index} has an unknown owner")
    for boolean_field in ("installed", "directInvocation", "employerMutation", "liveEligibility"):
        _require(isinstance(row[boolean_field], bool), f"row {index} {boolean_field} must be boolean")
    for string_field in (
        "id",
        "sourcePath",
        "symbol",
        "canonicalTarget",
        "authoritySource",
        "failClosedBehavior",
    ):
        _require(
            isinstance(row[string_field], str) and row[string_field].strip(),
            f"row {index} {string_field} must be a non-empty string",
        )
    _sorted_unique_str_list(row["aliases"], f"row {index} aliases")
    _sorted_unique_str_list(row["resourcesRead"], f"row {index} resourcesRead")
    _sorted_unique_str_list(row["resourcesWritten"], f"row {index} resourcesWritten")
    _sorted_unique_str_list(row["capabilities"], f"row {index} capabilities")
    pause_scopes = _sorted_unique_str_list(row["pauseScopes"], f"row {index} pauseScopes")
    _require(
        all(scope in PAUSE_SCOPE_VOCABULARY for scope in pause_scopes),
        f"row {index} pauseScopes must be drawn from the pause-scope vocabulary",
    )


def _validate_authority_rules(row: dict[str, Any], index: int) -> None:
    live = row["liveEligibility"]
    write_class = row["readWriteClass"]
    is_writer = write_class in {"write", "admin-ddl", "control"}
    if live:
        # Live writers require the exact role/source/caps/scopes/resources and a
        # specific fail-closed rule, and may never run as a shared/admin
        # principal.
        _require(
            row["authoritySource"] != "none",
            f"row {index} is live but declares no authoritySource",
        )
        _require(
            row["owner"] not in SHARED_ADMIN_PRINCIPALS
            and row["authoritySource"] not in SHARED_ADMIN_PRINCIPALS,
            f"row {index} is a live writer bound to a shared/admin principal",
        )
        _require(row["capabilities"], f"row {index} is live but declares no capabilities")
        _require(row["pauseScopes"], f"row {index} is live but declares no pauseScopes")
        if is_writer:
            _require(
                row["resourcesWritten"],
                f"row {index} is a live writer but writes no declared resource",
            )
        _require(
            row["failClosedBehavior"].startswith("checkpoint")
            or row["failClosedBehavior"].startswith("gated"),
            f"row {index} is live but lacks a gated/checkpoint fail-closed rule",
        )
    else:
        # This shard grants no live authority: default-deny is explicit.
        _require(
            row["authoritySource"] == "none",
            f"row {index} is not live yet claims an authoritySource",
        )
        _require(
            not row["capabilities"],
            f"row {index} is not live yet claims capabilities",
        )
        _require(
            row["failClosedBehavior"].startswith("deny"),
            f"row {index} must declare an explicit deny fail-closed rule while not live",
        )
    if is_writer and not live:
        # Denied / checkpoint-only writers must say so explicitly.
        _require(
            "writer" in row["failClosedBehavior"] or "checkpoint" in row["failClosedBehavior"],
            f"row {index} is a non-live writer but its fail-closed rule is not explicit",
        )


def validate_matrix(matrix: Any) -> list[dict[str, Any]]:
    """Validate the matrix document and return its rows. Fail closed on any
    unsupported/extra/missing/duplicate/noncanonical data."""
    _require(isinstance(matrix, dict), "matrix must be a JSON object")
    _require(set(matrix) == MATRIX_TOP_LEVEL_KEYS, "matrix has unexpected or missing top-level keys")
    _require(matrix["schemaVersion"] == MATRIX_SCHEMA_VERSION, "matrix schemaVersion is unsupported")
    _require(matrix["runtimeKind"] == RUNTIME_KIND, "matrix runtimeKind is unsupported")
    _require(matrix["repository"] == REPOSITORY, "matrix repository is unexpected")
    _require(matrix["noGlobalAuthorityClaim"] is True, "matrix must disclaim global authority")
    _validate_scope(matrix["scope"])
    vocabulary = _sorted_unique_str_list(matrix["pauseScopeVocabulary"], "pauseScopeVocabulary")
    _require(
        tuple(vocabulary) == PAUSE_SCOPE_VOCABULARY,
        "pauseScopeVocabulary must exactly match the canonical vocabulary",
    )

    rows = matrix["rows"]
    _require(isinstance(rows, list), "matrix rows must be a list")
    _require(len(rows) > 0, "matrix rows must not be empty")

    ids: list[str] = []
    targets: list[tuple[str, str, str]] = []
    alias_owner: dict[str, str] = {}
    for index, row in enumerate(rows):
        _validate_row_shape(row, index)
        _validate_authority_rules(row, index)
        ids.append(row["id"])
        targets.append((row["surfaceKind"], row["canonicalTarget"], row["sourcePath"]))
        for alias in row["aliases"]:
            if alias in alias_owner and alias_owner[alias] != row["id"]:
                raise AuthorityError(
                    f"alias {alias!r} is claimed by multiple rows: "
                    f"{alias_owner[alias]} and {row['id']}"
                )
            alias_owner[alias] = row["id"]

    _require(len(ids) == len(set(ids)), "matrix contains duplicate row ids")
    _require(len(targets) == len(set(targets)), "matrix contains duplicate canonical targets")
    _require(
        [row["id"] for row in rows] == sorted(row["id"] for row in rows),
        "matrix rows must be sorted by id",
    )
    # An alias must never collide with a *different* row's id.
    for alias, owner_id in alias_owner.items():
        for other in ids:
            if alias == other and other != owner_id:
                raise AuthorityError(f"alias {alias!r} collides with row id {other!r}")

    scan_forbidden_content(matrix, "matrix")
    return rows


# --------------------------------------------------------------------------- #
# Coverage / bijection
# --------------------------------------------------------------------------- #


def check_coverage(discovered: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    """Assert an exact bijection between discovered surfaces and matrix rows,
    with every structural field rebinding exactly."""
    discovered_by_id = {surface["id"]: surface for surface in discovered}
    rows_by_id = {row["id"]: row for row in rows}

    missing = sorted(set(discovered_by_id) - set(rows_by_id))
    if missing:
        raise AuthorityError(f"discovered surfaces are unclassified (missing rows): {missing}")
    stale = sorted(set(rows_by_id) - set(discovered_by_id))
    if stale:
        raise AuthorityError(f"matrix has stale rows for absent surfaces: {stale}")

    for surface_id, surface in discovered_by_id.items():
        row = rows_by_id[surface_id]
        for field in STRUCTURAL_ROW_KEYS:
            if row[field] != surface[field]:
                raise AuthorityError(
                    f"row {surface_id} field {field!r} drifted from discovery: "
                    f"{row[field]!r} != {surface[field]!r}"
                )


def count_by_kind(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {kind: 0 for kind in INCLUDED_SURFACE_KINDS}
    for row in rows:
        counts[row["surfaceKind"]] += 1
    return counts


# --------------------------------------------------------------------------- #
# Git identity / clean tree
# --------------------------------------------------------------------------- #


def _git(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_identity(repo_root: Path) -> tuple[str, str]:
    commit = _git(repo_root, "rev-parse", "HEAD^{commit}")
    tree = _git(repo_root, "rev-parse", "HEAD^{tree}")
    if not _COMMIT_RE.fullmatch(commit) or not _COMMIT_RE.fullmatch(tree):
        raise AuthorityError(f"invalid git identity: commit={commit!r} tree={tree!r}")
    return commit, tree


def tracked_tree_is_clean(repo_root: Path) -> bool:
    """Clean means no tracked modification/staging/deletion. Untracked scratch
    (for example the disposable ``.agent-run``) is ignored because the receipt
    binds committed content, not working-tree scratch."""
    porcelain = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    return porcelain == ""


def assert_clean(repo_root: Path) -> None:
    if not tracked_tree_is_clean(repo_root):
        raise AuthorityError("refusing to operate on a dirty tree (tracked changes present)")


# --------------------------------------------------------------------------- #
# Receipt
# --------------------------------------------------------------------------- #


def build_receipt(
    *,
    head: str,
    tree: str,
    matrix: dict[str, Any],
    discovered: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    matrix_path: str,
    clean_state: bool,
) -> dict[str, Any]:
    live_writers = sum(1 for row in rows if row["liveEligibility"])
    receipt = {
        "schemaVersion": RECEIPT_SCHEMA_VERSION,
        "runtimeKind": RUNTIME_KIND,
        "repository": REPOSITORY,
        "noGlobalAuthorityClaim": True,
        "head": head,
        "tree": tree,
        "matrixPath": matrix_path,
        "discoveryDigest": discovery_digest(discovered),
        "matrixDigest": sha256_hex(canonical_json(matrix)),
        "counts": {
            "discovered": len(discovered),
            "rows": len(rows),
            "byKind": count_by_kind(rows),
            "liveWriters": live_writers,
            "unclassified": 0,
            "deferredKinds": len(DEFERRED_SURFACE_KINDS),
        },
        "cleanState": clean_state,
    }
    receipt["selfHash"] = sha256_hex(canonical_json(receipt))
    return receipt


def verify_receipt(receipt: Any) -> None:
    """Fail closed on a forged / drifted receipt."""
    _require(isinstance(receipt, dict), "receipt must be a JSON object")
    _require("selfHash" in receipt, "receipt is missing its self-hash")
    supplied = receipt["selfHash"]
    _require(isinstance(supplied, str), "receipt selfHash must be a string")
    body = {key: value for key, value in receipt.items() if key != "selfHash"}
    expected = sha256_hex(canonical_json(body))
    if supplied != expected:
        raise AuthorityError("receipt self-hash does not match its content (forged or drifted)")
    _require(receipt["schemaVersion"] == RECEIPT_SCHEMA_VERSION, "receipt schemaVersion is unsupported")
    _require(receipt["runtimeKind"] == RUNTIME_KIND, "receipt runtimeKind is unsupported")
    _require(receipt["noGlobalAuthorityClaim"] is True, "receipt must disclaim global authority")
    counts = receipt.get("counts")
    _require(isinstance(counts, dict), "receipt counts must be an object")
    _require(counts.get("unclassified") == 0, "receipt must assert zero unclassified surfaces")
    _require(
        counts.get("discovered") == counts.get("rows"),
        "receipt discovered/rows counts must agree",
    )
    scan_forbidden_content(receipt, "receipt")


def load_matrix_document(path: Path) -> tuple[dict[str, Any], bytes]:
    content = Path(path).read_bytes()
    matrix = strict_json_loads(content, f"matrix {Path(path).name}")
    if not isinstance(matrix, dict):
        raise AuthorityError("matrix document must be a JSON object")
    return matrix, content


def evaluate(repo_root: Path, matrix_path: Path) -> dict[str, Any]:
    """Run the full gate and return the canonical receipt (dict). Raises
    :class:`AuthorityError` on any gate failure."""
    repo_root = Path(repo_root)
    matrix_path = Path(matrix_path)
    assert_clean(repo_root)
    head, tree = git_identity(repo_root)
    matrix, _content = load_matrix_document(matrix_path)
    rows = validate_matrix(matrix)
    discovered = discover(repo_root)
    check_coverage(discovered, rows)
    receipt = build_receipt(
        head=head,
        tree=tree,
        matrix=matrix,
        discovered=discovered,
        rows=rows,
        matrix_path=matrix_path.relative_to(repo_root).as_posix(),
        clean_state=True,
    )
    verify_receipt(receipt)
    return receipt
