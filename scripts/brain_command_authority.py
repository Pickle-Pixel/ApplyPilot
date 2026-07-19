"""Fail-closed library for the Runtime brain command-authority matrix.

This module owns three responsibilities for the Runtime half of the Task 9A
command-authority discovery/receipt:

* Deterministic, repository-owned, structural discovery of the complete Runtime
  command surface: Python entrypoints and subcommands, MCP/HTTP registrations,
  PowerShell/shell/batch launchers, scheduled tasks and generated wrappers,
  containers, services, and launchd definitions. Discovery is structural rather
  than a filename allowlist, so any new/renamed/removed entrypoint changes the
  discovered set and trips the fail-closed gates until the matrix is refreshed.

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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Schema constants
# --------------------------------------------------------------------------- #

MATRIX_SCHEMA_VERSION = "applypilot-brain-command-authority-matrix-runtime-v1"
RECEIPT_SCHEMA_VERSION = "applypilot-brain-command-authority-receipt-runtime-v1"
RUNTIME_KIND = "complete-runtime-command-surface"
REPOSITORY = "applypilot"

INCLUDED_SURFACE_KINDS = (
    "argparse-subcommand",
    "batch-launcher",
    "console-http-action",
    "console-script",
    "container-entrypoint",
    "generated-wrapper",
    "launchd-plist",
    "mcp-tool",
    "powershell-launcher",
    "powershell-module",
    "python-file",
    "python-module",
    "scheduled-task-action",
    "scheduled-task-identity",
    "service-unit",
    "shell-launcher",
    "typer-subcommand",
)

# The complete Runtime inventory has no deferred structural kinds.
DEFERRED_SURFACE_KINDS: tuple[str, ...] = ()

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

AUTHORITY_SOURCES = ("none",)

RESOURCE_VOCABULARY = (
    "backup-artifact",
    "brain-postgres",
    "brain-sqlite",
    "employer-ats",
    "fleet-postgres",
    "fleet-roles",
    "fleet-schema",
    "mailbox",
    "release-evidence",
    "repository-source",
    "secret-store",
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

RECEIPT_KEYS = frozenset(
    {
        "schemaVersion",
        "runtimeKind",
        "repository",
        "noGlobalAuthorityClaim",
        "head",
        "tree",
        "matrixPath",
        "discoveryDigest",
        "matrixDigest",
        "counts",
        "cleanState",
        "selfHash",
    }
)

RECEIPT_COUNT_KEYS = frozenset(
    {"discovered", "rows", "byKind", "liveWriters", "unclassified", "deferredKinds"}
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


def _module_to_source(tree: "_RepoTree", module: str) -> str:
    """Resolve a dotted module to a repo-relative POSIX source path within the
    committed tree.  A missing target is still returned (as the would-be module
    file) so the verifier flags a broken / stale entrypoint rather than silently
    dropping it."""
    base = "src/" + "/".join(module.split("."))
    module_file = f"{base}.py"
    package_init = f"{base}/__init__.py"
    if tree.is_regular(module_file):
        return module_file
    if tree.is_regular(package_init):
        return package_init
    return module_file


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


def _dotted_module(relpath: str) -> str:
    """Dotted module name for a repo-relative POSIX path under ``src/``."""
    parts = relpath.split("/")[1:]  # drop the leading "src"
    if parts[-1] in {"__init__.py", "__main__.py"}:
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


def _excluded(relpath: str, excluded_parts: set[str]) -> bool:
    """True when any path component of ``relpath`` is an excluded directory."""
    return bool(set(relpath.split("/")) & excluded_parts)


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


def _discover_console_scripts(tree: "_RepoTree") -> list[dict[str, Any]]:
    if not tree.is_regular("pyproject.toml"):
        return []
    data = tomllib.loads(tree.read_text("pyproject.toml"))
    scripts = data.get("project", {}).get("scripts", {})
    surfaces: list[dict[str, Any]] = []
    for name in sorted(scripts):
        target = scripts[name]
        module, sep, attr = target.partition(":")
        if not sep or not module or not attr:
            raise AuthorityError(
                f"console script {name!r} has a non-canonical target: {target!r}"
            )
        source_path = _module_to_source(tree, module)
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


def _discover_python_modules(tree: "_RepoTree") -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    for relative in tree.python_paths_under("src"):
        module_ast = ast.parse(tree.read_text(relative))
        name = relative.rsplit("/", 1)[-1]
        if name != "__main__.py" and not _has_main_guard(module_ast):
            continue
        dotted = _dotted_module(relative)
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


def _discover_python_files(tree: "_RepoTree") -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    excluded_parts = {
        ".agent-run",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "docs",
        "node_modules",
        "tests",
        "venv",
    }
    candidates = [
        relative
        for relative in tree.all_python_paths()
        if not _excluded(relative, excluded_parts)
        and relative != "src"
        and not relative.startswith("src/")
    ]
    for relative in candidates:
        module_ast = ast.parse(tree.read_text(relative))
        if _is_library_module(module_ast):
            continue
        symbol = _MAIN_GUARD_TARGET if _has_main_guard(module_ast) else "<module-body>"
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


def _expr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expr_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _constant_string(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _call_name(call: ast.Call) -> str | None:
    return _expr_name(call.func)


def _walk_statements(tree: ast.AST) -> Iterable[ast.AST]:
    return sorted(ast.walk(tree), key=lambda node: (getattr(node, "lineno", -1), getattr(node, "col_offset", -1)))


def _discover_mcp_tools(tree: "_RepoTree") -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    for relative in tree.python_paths_under("src"):
        module_ast = ast.parse(tree.read_text(relative))
        dotted = _dotted_module(relative)
        server_names: set[str] = set()
        for node in _walk_statements(module_ast):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not isinstance(value, ast.Call) or (_call_name(value) or "").split(".")[-1] != "FastMCP":
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            server_names.update(name for target in targets if (name := _expr_name(target)))
        if not server_names:
            continue

        tools: dict[tuple[str, str], tuple[str, int]] = {}

        def record_tool(registered_name: str, symbol: str, line: int) -> None:
            key = (registered_name, symbol)
            if key in tools:
                raise AuthorityError(
                    f"duplicate MCP tool registration in {relative}:{line}: "
                    f"{registered_name}/{symbol}"
                )
            tools[key] = (symbol, line)

        for node in _walk_statements(module_ast):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    target = decorator.func if isinstance(decorator, ast.Call) else decorator
                    if not isinstance(target, ast.Attribute) or target.attr != "tool":
                        continue
                    server = _expr_name(target.value)
                    if server not in server_names:
                        continue
                    registered_name = node.name
                    if isinstance(decorator, ast.Call):
                        explicit_name = decorator.args[0] if decorator.args else next(
                            (keyword.value for keyword in decorator.keywords if keyword.arg == "name"),
                            None,
                        )
                        if explicit_name is not None:
                            registered_name = _constant_string(explicit_name) or ""
                            if not registered_name:
                                raise AuthorityError(
                                    f"dynamic MCP tool name in {relative}:{node.lineno}"
                                )
                    record_tool(registered_name, node.name, node.lineno)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_tool":
                server = _expr_name(node.func.value)
                if server not in server_names:
                    continue
                callable_node = node.args[0] if node.args else next(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg in {"fn", "func", "tool"}
                    ),
                    None,
                )
                symbol = _expr_name(callable_node) if callable_node is not None else None
                if not symbol:
                    raise AuthorityError(f"dynamic MCP add_tool callable in {relative}:{node.lineno}")
                name_node = next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "name"),
                    None,
                )
                registered_name = symbol.split(".")[-1]
                if name_node is not None:
                    registered_name = _constant_string(name_node) or ""
                    if not registered_name:
                        raise AuthorityError(f"dynamic MCP tool name in {relative}:{node.lineno}")
                record_tool(registered_name, symbol, node.lineno)

        for (registered_name, _symbol_key), (func, _line) in sorted(tools.items()):
            surfaces.append(
                _surface(
                    surface_id=f"mcp-tool:{dotted}:{registered_name}:{func}",
                    source_path=relative,
                    symbol=func,
                    surface_kind="mcp-tool",
                    canonical_target=f"{dotted}:{func}#mcp:{registered_name}",
                    aliases=[f"mcp:{registered_name}"],
                    installed=False,
                    direct_invocation=False,
                )
            )
    return surfaces


def _module_or_file_target(relpath: str) -> str:
    if relpath == "src" or relpath.startswith("src/"):
        return _dotted_module(relpath)
    return relpath


def _discover_typer_subcommands(tree: "_RepoTree") -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    for relative in tree.python_paths_under("src"):
        module_ast = ast.parse(tree.read_text(relative))
        typer_vars: set[str] = set()
        for node in _walk_statements(module_ast):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(node.value, ast.Call):
                continue
            if (_call_name(node.value) or "").split(".")[-1] != "Typer":
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            typer_vars.update(name for target in targets if (name := _expr_name(target)))
        if not typer_vars:
            continue

        parents: dict[str, tuple[str, str]] = {}
        for node in _walk_statements(module_ast):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "add_typer":
                continue
            parent = _expr_name(node.func.value)
            child = _expr_name(node.args[0]) if node.args else None
            if parent not in typer_vars or child not in typer_vars:
                continue
            name_node = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "name"),
                None,
            )
            name = child.removesuffix("_app").replace("_", "-")
            if name_node is not None:
                name = _constant_string(name_node) or ""
                if not name:
                    raise AuthorityError(f"dynamic Typer group name in {relative}:{node.lineno}")
            parents[child] = (parent, name)

        def prefix(variable: str) -> list[str]:
            result: list[str] = []
            seen: set[str] = set()
            while variable in parents and variable not in seen:
                seen.add(variable)
                variable, name = parents[variable]
                result.append(name)
            return list(reversed(result))

        target_base = _module_or_file_target(relative)
        for node in _walk_statements(module_ast):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute) or decorator.func.attr != "command":
                    continue
                variable = _expr_name(decorator.func.value)
                if variable not in typer_vars:
                    continue
                command = _constant_string(decorator.args[0]) if decorator.args else None
                name_node = next(
                    (keyword.value for keyword in decorator.keywords if keyword.arg == "name"),
                    None,
                )
                if decorator.args and command is None:
                    raise AuthorityError(f"dynamic Typer command name in {relative}:{node.lineno}")
                if name_node is not None:
                    command = _constant_string(name_node)
                    if command is None:
                        raise AuthorityError(
                            f"dynamic Typer command name in {relative}:{node.lineno}"
                        )
                command = command or node.name.replace("_", "-")
                full_command = " ".join([*prefix(variable), command])
                surfaces.append(
                    _surface(
                        surface_id=f"typer-subcommand:{target_base}:{full_command}:{node.name}",
                        source_path=relative,
                        symbol=node.name,
                        surface_kind="typer-subcommand",
                        canonical_target=f"{target_base}:{node.name}#typer:{full_command}",
                        aliases=[f"typer:{full_command}"],
                        installed=False,
                        direct_invocation=True,
                    )
                )
    return surfaces


def _discover_argparse_subcommands(tree: "_RepoTree") -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    excluded = {".git", ".venv", "venv", "tests", "docs", "node_modules", "__pycache__"}
    for relative in tree.all_python_paths():
        if _excluded(relative, excluded):
            continue
        module_ast = ast.parse(tree.read_text(relative))
        target_base = _module_or_file_target(relative)
        commands: dict[str, int] = {}
        for node in _walk_statements(module_ast):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "add_parser" or not node.args:
                continue
            command = _constant_string(node.args[0])
            if command is None:
                raise AuthorityError(f"dynamic argparse subcommand in {relative}:{node.lineno}")
            commands[command] = commands.get(command, 0) + 1
            ordinal = commands[command]
            suffix = f":{ordinal}" if ordinal > 1 else ""
            surfaces.append(
                _surface(
                    surface_id=f"argparse-subcommand:{target_base}:{command}{suffix}",
                    source_path=relative,
                    symbol=command,
                    surface_kind="argparse-subcommand",
                    canonical_target=f"{target_base}#argparse:{command}{suffix}",
                    aliases=[f"argparse:{target_base}:{command}{suffix}"],
                    installed=False,
                    direct_invocation=True,
                )
            )
    return surfaces


def _discover_http_actions(tree: "_RepoTree") -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    methods = {"delete", "get", "patch", "post", "put"}
    for relative in tree.python_paths_under("src"):
        module_ast = ast.parse(tree.read_text(relative))
        dotted = _dotted_module(relative)
        for node in _walk_statements(module_ast):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                    continue
                method = decorator.func.attr.lower()
                if method not in methods or not decorator.args:
                    continue
                route = _constant_string(decorator.args[0])
                if route is None:
                    raise AuthorityError(f"dynamic HTTP route in {relative}:{node.lineno}")
                surfaces.append(
                    _surface(
                        surface_id=f"console-http-action:{dotted}:{method}:{route}:{node.name}",
                        source_path=relative,
                        symbol=node.name,
                        surface_kind="console-http-action",
                        canonical_target=f"{dotted}:{node.name}#http:{method}:{route}",
                        aliases=[f"http:{method.upper()}:{route}"],
                        installed=False,
                        direct_invocation=False,
                    )
                )
    return surfaces


def _repo_files(tree: "_RepoTree", suffixes: tuple[str, ...]) -> list[str]:
    excluded = {".agent-run", ".git", ".venv", "venv", "node_modules", "__pycache__"}
    return [
        relpath
        for relpath in tree.paths_with_suffix(suffixes)
        if not _excluded(relpath, excluded)
    ]


def _file_surface(relative: str, kind: str, launcher: str) -> dict[str, Any]:
    return _surface(
        surface_id=f"{kind}:{relative}",
        source_path=relative,
        symbol="<file>",
        surface_kind=kind,
        canonical_target=relative,
        aliases=[f"{launcher} {relative}"],
        installed=False,
        direct_invocation=kind != "powershell-module",
    )


def _discover_launchers(tree: "_RepoTree") -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    for relpath in _repo_files(tree, (".ps1",)):
        surfaces.append(_file_surface(relpath, "powershell-launcher", "powershell -File"))
    for relpath in _repo_files(tree, (".psm1",)):
        surfaces.append(_file_surface(relpath, "powershell-module", "Import-Module"))
    for relpath in _repo_files(tree, (".sh",)):
        surfaces.append(_file_surface(relpath, "shell-launcher", "sh"))
    for relpath in _repo_files(tree, (".bat", ".cmd")):
        surfaces.append(_file_surface(relpath, "batch-launcher", "cmd /c"))
    for relpath in _repo_files(tree, (".service", ".timer")):
        surfaces.append(_file_surface(relpath, "service-unit", "service-unit"))
    for relpath in _repo_files(tree, (".plist",)):
        surfaces.append(_file_surface(relpath, "launchd-plist", "launchctl load"))
    return surfaces


_SHELL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_STATIC_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")


def _shell_assignments(text: str, relative: str) -> dict[str, str]:
    """Return unique, single-line shell assignments without evaluating shell.

    Discovery deliberately understands only the small static assignment grammar
    needed to bind committed service installers.  A duplicate assignment is not
    silently resolved by shell ordering: it is ambiguous evidence and fails
    closed when a structural service refers to that name.
    """
    values: dict[str, list[str]] = {}
    pattern = re.compile(
        r"(?m)^\s*(?:local\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)="
        r"(?P<quote>[\"'])(?P<value>[^\r\n]*?)(?P=quote)\s*$"
    )
    for match in pattern.finditer(text):
        values.setdefault(match.group("name"), []).append(match.group("value"))
    return {
        name: candidates[0]
        for name, candidates in values.items()
        if len(candidates) == 1
    }


def _static_shell_value(
    value: str,
    assignments: dict[str, str],
    label: str,
    seen: frozenset[str] = frozenset(),
) -> str:
    """Resolve `${NAME:-literal}` defaults and simple `$NAME` indirection.

    Host-dependent values such as HOME/USER/INSTALL_DIR remain symbolic in the
    binding.  Command substitution, backticks, parameter transforms, and
    recursive or unresolved defaults are rejected rather than guessed.
    """
    if "$(" in value or "`" in value or "${!" in value:
        raise AuthorityError(f"{label} is dynamic or executes shell code")

    whole_reference = re.fullmatch(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", value)
    if whole_reference and whole_reference.group(1) in assignments:
        name = whole_reference.group(1)
        if name in seen:
            raise AuthorityError(f"{label} has recursive shell indirection")
        return _static_shell_value(
            assignments[name], assignments, label, seen | {name}
        )

    default_pattern = re.compile(
        r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^{}$`]+)\}"
    )
    resolved = default_pattern.sub(lambda match: match.group(2), value)
    if "${" in resolved and re.search(r"\$\{[^}]*[:?+=#%/]", resolved):
        raise AuthorityError(f"{label} uses unsupported dynamic parameter expansion")
    return resolved


def _plist_mapping(relative: str, text: str) -> dict[str, ET.Element]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise AuthorityError(f"launchd template {relative!r} is malformed XML") from exc
    if root.tag != "plist" or len(root) != 1 or root[0].tag != "dict":
        raise AuthorityError(f"launchd template {relative!r} must contain one plist dict")
    children = list(root[0])
    if len(children) % 2:
        raise AuthorityError(f"launchd template {relative!r} has an unmatched dict key")
    mapping: dict[str, ET.Element] = {}
    for index in range(0, len(children), 2):
        key, value = children[index], children[index + 1]
        if key.tag != "key" or not key.text or key.text in mapping:
            raise AuthorityError(f"launchd template {relative!r} has a missing or duplicate key")
        mapping[key.text] = value
    return mapping


def _launchd_install_binding(tree: "_RepoTree", relative: str) -> tuple[str, str, str]:
    basename = relative.rsplit("/", 1)[-1]
    bindings: list[tuple[str, str, str]] = []
    for script in _repo_files(tree, (".sh",)):
        text = tree.read_text(script)
        install_lines = [
            line
            for line in text.splitlines()
            if basename in line and re.search(r">\s*[\"']?\$[A-Za-z_][A-Za-z0-9_]*[\"']?\s*$", line)
        ]
        if not install_lines:
            continue
        if len(install_lines) != 1:
            raise AuthorityError(
                f"launchd template {relative!r} has ambiguous install actions in {script!r}"
            )
        destination_match = re.search(
            r">\s*[\"']?\$([A-Za-z_][A-Za-z0-9_]*)[\"']?\s*$", install_lines[0]
        )
        assert destination_match is not None
        destination_name = destination_match.group(1)
        assignments = _shell_assignments(text, script)
        if destination_name not in assignments:
            raise AuthorityError(
                f"launchd template {relative!r} install destination is dynamic in {script!r}"
            )
        destination = _static_shell_value(
            assignments[destination_name], assignments, f"launchd destination in {script}"
        )
        if not destination.endswith(".plist") or not destination.startswith("$HOME/"):
            raise AuthorityError(
                f"launchd template {relative!r} has an unsupported install destination"
            )
        load_pattern = re.compile(
            rf"(?m)^\s*launchctl\s+load\s+(?P<flags>(?:-[A-Za-z]+\s+)*)"
            rf"[\"']?\${re.escape(destination_name)}[\"']?\s*$"
        )
        loads = list(load_pattern.finditer(text))
        if len(loads) != 1:
            raise AuthorityError(
                f"launchd template {relative!r} must have exactly one bound launchctl load action"
            )
        flags = " ".join(loads[0].group("flags").split())
        if "-w" not in flags.split():
            raise AuthorityError(
                f"launchd template {relative!r} load action must persist with -w"
            )
        load = "launchctl load" + (f" {flags}" if flags else "")
        bindings.append((script, destination, load))
    if len(bindings) != 1:
        raise AuthorityError(
            f"launchd template {relative!r} must have exactly one structural installer/load binding"
        )
    return bindings[0]


def _discover_launchd_templates(tree: "_RepoTree") -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    labels: dict[str, str] = {}
    destinations: dict[str, str] = {}
    templates = [
        path for path in tree.regular_paths() if path.lower().endswith(".plist.template")
    ]
    for relative in templates:
        mapping = _plist_mapping(relative, tree.read_text(relative))
        label_node = mapping.get("Label")
        arguments_node = mapping.get("ProgramArguments")
        if (
            label_node is None
            or label_node.tag != "string"
            or not label_node.text
            or not _STATIC_LABEL_RE.fullmatch(label_node.text)
        ):
            raise AuthorityError(f"launchd template {relative!r} has no static Label")
        if arguments_node is None or arguments_node.tag != "array":
            raise AuthorityError(f"launchd template {relative!r} has no ProgramArguments array")
        arguments = [node.text or "" for node in arguments_node]
        if (
            not arguments
            or any(node.tag != "string" or not value for node, value in zip(arguments_node, arguments))
            or any("$(" in value or "`" in value or "${" in value for value in arguments)
        ):
            raise AuthorityError(
                f"launchd template {relative!r} has dynamic or malformed ProgramArguments"
            )
        label = label_node.text
        installer, destination, load = _launchd_install_binding(tree, relative)
        if label in labels:
            raise AuthorityError(
                f"launchd label collision {label!r}: {labels[label]!r} and {relative!r}"
            )
        if destination in destinations:
            raise AuthorityError(
                f"launchd destination collision {destination!r}: "
                f"{destinations[destination]!r} and {relative!r}"
            )
        labels[label] = relative
        destinations[destination] = relative
        encoded_arguments = canonical_json(arguments).decode("ascii")
        surfaces.append(
            _surface(
                surface_id=f"launchd-plist:{relative}",
                source_path=relative,
                symbol=label,
                surface_kind="launchd-plist",
                canonical_target=(
                    f"{relative}#label={label};programArguments={encoded_arguments};"
                    f"install={installer}:{destination};load={load}"
                ),
                aliases=sorted(
                    [
                        f"launchctl:{label}",
                        f"{load} {destination}",
                    ]
                ),
                installed=False,
                direct_invocation=False,
            )
        )
    return surfaces


_HEREDOC_RE = re.compile(
    r"(?ms)^\s*(?:sudo\s+)?tee\s+(?P<destination>[^\s>]+)"
    r"[^\r\n]*?<<(?P<quote>[\"']?)(?P<marker>[A-Za-z_][A-Za-z0-9_]*)(?P=quote)\s*\r?\n"
    r"(?P<body>.*?)^\s*(?P=marker)\s*$"
)


def _one_unit_field(body: str, name: str, relative: str) -> str:
    values = re.findall(rf"(?m)^\s*{re.escape(name)}\s*=\s*(\S(?:.*\S)?)\s*$", body)
    if len(values) != 1:
        raise AuthorityError(
            f"generated systemd unit in {relative!r} must have exactly one {name}"
        )
    value = values[0]
    if "$(" in value or "`" in value or "${" in value:
        raise AuthorityError(f"generated systemd {name} in {relative!r} is dynamically ambiguous")
    return value


def _reject_unsupported_service_generators(tree: "_RepoTree") -> None:
    """Fail closed when a tracked script writes a service by an unbound form.

    The structural parsers deliberately support only the reviewed shell tee
    heredoc and committed launchd-template forms.  Other common writers must
    not silently degrade to an ordinary launcher row.
    """
    shell_writer = re.compile(r"(?i)\b(?:cat|printf|install|cp)\b")
    systemd_destination = re.compile(
        r"(?i)/etc/systemd/system/[^\s\"']+\.(?:service|timer)\b"
    )
    launchd_destination = re.compile(
        r"(?i)(?:\$HOME|~)/Library/LaunchAgents/[^\s\"']+\.plist\b"
    )
    for relative in _repo_files(tree, (".sh",)):
        for line in tree.read_text(relative).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or not shell_writer.search(stripped):
                continue
            if systemd_destination.search(stripped):
                raise AuthorityError(
                    f"unsupported shell systemd generator in {relative!r}"
                )
            if launchd_destination.search(stripped):
                raise AuthorityError(
                    f"unsupported inline launchd generator in {relative!r}"
                )

    powershell_writer = re.compile(
        r"(?i)\b(?:Set-Content|Out-File|WriteAllText)\b"
    )
    for relative in _repo_files(tree, (".ps1", ".psm1")):
        for line in tree.read_text(relative).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or not powershell_writer.search(stripped):
                continue
            if systemd_destination.search(stripped):
                raise AuthorityError(
                    f"unsupported PowerShell systemd generator in {relative!r}"
                )


def _discover_generated_systemd_units(tree: "_RepoTree") -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    unit_names: dict[str, str] = {}
    destinations: dict[str, str] = {}
    for relative in _repo_files(tree, (".sh",)):
        text = tree.read_text(relative)
        assignments = _shell_assignments(text, relative)
        for match in _HEREDOC_RE.finditer(text):
            body = match.group("body")
            has_service = bool(re.search(r"(?m)^\s*\[Service\]\s*$", body))
            has_timer = bool(re.search(r"(?m)^\s*\[Timer\]\s*$", body))
            if not has_service and not has_timer:
                continue
            if has_service and has_timer:
                raise AuthorityError(f"generated systemd unit in {relative!r} mixes service and timer")
            raw_destination = match.group("destination").strip("\"'")
            destination = _static_shell_value(
                raw_destination, assignments, f"systemd destination in {relative}"
            )
            # Resolve a unique default for the variable embedded in a unit path.
            referenced = re.findall(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", destination)
            unit_variable = next((name for name in referenced if name in assignments), None)
            resolved_destination = destination
            if unit_variable is not None:
                unit_stem = _static_shell_value(
                    assignments[unit_variable], assignments, f"systemd unit name in {relative}"
                )
                if not _STATIC_LABEL_RE.fullmatch(unit_stem):
                    raise AuthorityError(f"generated systemd unit name in {relative!r} is dynamic")
                resolved_destination = re.sub(
                    rf"\$\{{?{re.escape(unit_variable)}\}}?", unit_stem, destination
                )
            suffix = ".service" if has_service else ".timer"
            if not resolved_destination.startswith("/etc/systemd/system/") or not resolved_destination.endswith(suffix):
                raise AuthorityError(
                    f"generated systemd unit in {relative!r} has unsupported destination"
                )
            unit_name = resolved_destination.rsplit("/", 1)[-1]
            if not _STATIC_LABEL_RE.fullmatch(unit_name):
                raise AuthorityError(
                    f"generated systemd unit name in {relative!r} is dynamically ambiguous"
                )
            if unit_name in unit_names:
                raise AuthorityError(
                    f"generated systemd unit collision {unit_name!r}: "
                    f"{unit_names[unit_name]!r} and {relative!r}"
                )
            if resolved_destination in destinations:
                raise AuthorityError(
                    f"generated systemd destination collision {resolved_destination!r}"
                )

            target_tokens = {unit_name, unit_name[: -len(suffix)]}
            if unit_variable is not None:
                target_tokens.add(f"${unit_variable}")
            activation_lines = []
            for line in text.splitlines():
                stripped = line.strip()
                if not re.match(r"^(?:sudo\s+)?systemctl\s+enable\b", stripped):
                    continue
                activation_tokens = {
                    token.strip("\"'") for token in stripped.split()
                }
                if target_tokens & activation_tokens:
                    activation_lines.append(" ".join(stripped.split()))
            if len(activation_lines) != 1 or "--now" not in activation_lines[0].split():
                raise AuthorityError(
                    f"generated systemd unit {unit_name!r} must have exactly one bound enable --now action"
                )

            if has_service:
                user = _one_unit_field(body, "User", relative)
                exec_start = _one_unit_field(body, "ExecStart", relative)
                detail = f"User={user};ExecStart={exec_start}"
            else:
                timer_fields = sorted(
                    re.findall(r"(?m)^\s*(On[A-Za-z]+)\s*=\s*(\S(?:.*\S)?)\s*$", body)
                )
                if not timer_fields:
                    raise AuthorityError(
                        f"generated systemd timer in {relative!r} has no static On* schedule"
                    )
                if any("$(" in value or "`" in value or "${" in value for _, value in timer_fields):
                    raise AuthorityError(
                        f"generated systemd timer in {relative!r} has a dynamic schedule"
                    )
                detail = ";".join(f"{name}={value}" for name, value in timer_fields)
            activation = activation_lines[0]
            unit_names[unit_name] = relative
            destinations[resolved_destination] = relative
            surfaces.append(
                _surface(
                    surface_id=f"service-unit:{relative}:generated:{unit_name}",
                    source_path=relative,
                    symbol=unit_name,
                    surface_kind="service-unit",
                    canonical_target=(
                        f"{relative}#generated-systemd:{unit_name};"
                        f"destinationTemplate={destination};"
                        f"resolvedDefaultDestination={resolved_destination};"
                        f"{detail};activation={activation}"
                    ),
                    aliases=sorted(
                        [f"systemd:{unit_name}", f"systemctl enable --now {unit_name}"]
                    ),
                    installed=False,
                    direct_invocation=False,
                )
            )
    return surfaces


def _powershell_command_blocks(
    text: str, command_pattern: re.Pattern[str]
) -> list[tuple[int, str]]:
    """Return non-comment PowerShell command blocks joined across backticks."""
    lines = text.splitlines()
    blocks: list[tuple[int, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        if stripped.startswith("#") or not command_pattern.search(line):
            index += 1
            continue
        start = index
        parts = [line.rstrip()]
        while parts[-1].endswith("`"):
            index += 1
            if index >= len(lines):
                raise AuthorityError("unterminated PowerShell command continuation")
            parts.append(lines[index].rstrip())
        blocks.append((start + 1, " ".join(part[:-1] if part.endswith("`") else part for part in parts)))
        index += 1
    return blocks


_POWERSHELL_VALUE = (
    r"'(?:''|[^'])*'|\"(?:`.|[^\"])*\"|"
    r"\$[A-Za-z_][A-Za-z0-9_:\.\[\]]*|\([^)]*\)|[^\s]+"
)


def _powershell_parameter_binding(
    block: str,
    name: str,
    *,
    relative: str,
    required: bool = False,
) -> str:
    matches = list(
        re.finditer(
            rf"(?i)(?:^|\s)-{re.escape(name)}\s+(?P<value>{_POWERSHELL_VALUE})",
            block,
        )
    )
    if len(matches) > 1:
        raise AuthorityError(
            f"scheduled task command in {relative!r} has duplicate {name} parameter"
        )
    if not matches:
        if required:
            raise AuthorityError(
                f"scheduled task command in {relative!r} has no {name} parameter"
            )
        return "absent"
    expression = matches[0].group("value")
    if expression.startswith("'") and expression.endswith("'"):
        value = expression[1:-1].replace("''", "'")
        return canonical_json(value).decode("ascii")
    if expression.startswith('"') and expression.endswith('"'):
        value = expression[1:-1]
        if "$" not in value and "`" not in value:
            return canonical_json(value).decode("ascii")
    normalized = " ".join(expression.split())
    return f"dynamic-sha256:{sha256_hex(normalized.encode('utf-8'))}"


def _discover_scheduled_tasks_and_wrappers(tree: "_RepoTree") -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    registration_pattern = re.compile(
        r"(?i)\b(?:Register-ScheduledTask|schtasks(?:\.exe)?\s+/Create)\b"
    )
    action_pattern = re.compile(r"(?i)\bNew-ScheduledTaskAction\b")
    for relative in _repo_files(tree, (".ps1", ".psm1")):
        text = tree.read_text(relative)
        registrations = _powershell_command_blocks(text, registration_pattern)
        actions = _powershell_command_blocks(text, action_pattern)
        for kind, blocks in (("scheduled-task-identity", registrations), ("scheduled-task-action", actions)):
            for ordinal, (line, block) in enumerate(blocks, 1):
                if kind == "scheduled-task-action":
                    execute = _powershell_parameter_binding(
                        block, "Execute", relative=relative, required=True
                    )
                    argument = _powershell_parameter_binding(
                        block, "Argument", relative=relative
                    )
                    detail = f"execute={execute};argument={argument}"
                elif re.search(r"(?i)\bschtasks(?:\.exe)?\s+/Create\b", block):
                    command = _powershell_parameter_binding(
                        re.sub(r"(?i)(?<!\S)/", " -", block),
                        "TR",
                        relative=relative,
                        required=True,
                    )
                    user = _powershell_parameter_binding(
                        re.sub(r"(?i)(?<!\S)/", " -", block),
                        "RU",
                        relative=relative,
                    )
                    detail = f"command={command};user={user};principal=absent"
                else:
                    user = _powershell_parameter_binding(
                        block, "User", relative=relative
                    )
                    principal = _powershell_parameter_binding(
                        block, "Principal", relative=relative
                    )
                    if user != "absent" and principal != "absent":
                        raise AuthorityError(
                            f"scheduled task command in {relative!r} has both User and Principal"
                        )
                    detail = f"user={user};principal={principal}"
                surfaces.append(
                    _surface(
                        surface_id=f"{kind}:{relative}:{ordinal}",
                        source_path=relative,
                        symbol=f"line-{line}",
                        surface_kind=kind,
                        canonical_target=f"{relative}#{kind}:{ordinal};{detail}",
                        aliases=[f"{kind}:{relative}:{ordinal}"],
                        installed=False,
                        direct_invocation=False,
                    )
                )
        wrapper_markers = (
            ".ps1" in text
            and ("Set-Content" in text or "WriteAllText" in text)
            and ("wrapper" in text.lower() or "_task-wrappers" in text)
        )
        if wrapper_markers:
            surfaces.append(
                _surface(
                    surface_id=f"generated-wrapper:{relative}",
                    source_path=relative,
                    symbol="<generated-wrapper>",
                    surface_kind="generated-wrapper",
                    canonical_target=f"{relative}#generated-wrapper",
                    aliases=[f"generated-wrapper:{relative}"],
                    installed=False,
                    direct_invocation=False,
                )
            )
    return surfaces


def _discover_container_entrypoints(tree: "_RepoTree") -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    dockerfiles = [
        relpath
        for relpath in tree.regular_paths()
        if relpath.rsplit("/", 1)[-1].startswith("Dockerfile")
        and not _excluded(relpath, {".git"})
    ]
    for relative in dockerfiles:
        instructions = [
            (match.group(1).upper(), " ".join(match.group(2).split()))
            for match in re.finditer(r"(?im)^\s*(ENTRYPOINT|CMD)\s+(.+)$", tree.read_text(relative))
        ]
        for ordinal, (instruction, value) in enumerate(instructions, 1):
            surfaces.append(
                _surface(
                    surface_id=f"container-entrypoint:{relative}:{instruction.lower()}:{ordinal}",
                    source_path=relative,
                    symbol=instruction,
                    surface_kind="container-entrypoint",
                    canonical_target=f"{relative}#{instruction.lower()}:{ordinal}:{value}",
                    aliases=[f"container:{relative}:{instruction.lower()}:{ordinal}"],
                    installed=False,
                    direct_invocation=False,
                )
            )
    return surfaces


_PARSEABLE_EXTENSIONS = (
    ".py",
    ".ps1",
    ".psm1",
    ".sh",
    ".bat",
    ".cmd",
    ".plist",
    ".template",
    ".service",
    ".timer",
)


def _is_parseable_path(relpath: str) -> bool:
    """Committed blobs whose *bytes* discovery must read (not merely enumerate)."""
    name = relpath.rsplit("/", 1)[-1]
    return (
        relpath.endswith(_PARSEABLE_EXTENSIONS)
        or relpath == "pyproject.toml"
        or name.startswith("Dockerfile")
    )


class _RepoTree:
    """A read-only, committed-tree view of one repository at an exact HEAD/tree.

    Every candidate path is enumerated from ``git ls-tree`` and every byte
    discovery parses is the *committed blob* (via ``git cat-file``), so untracked
    or ignored scratch, working-tree caches, symlink targets, filesystem case,
    and checkout-specific files can never enter the discovered surface set.
    """

    def __init__(
        self,
        repo_root: Path,
        head: str,
        tree: str,
        regular: dict[str, str],
        symlinks: dict[str, str],
        blobs: dict[str, bytes],
    ) -> None:
        self.repo_root = Path(repo_root)
        self.head = head
        self.tree = tree
        self._regular = regular  # repo-relative POSIX path -> blob sha (files)
        self._symlinks = symlinks  # repo-relative POSIX path -> blob sha (symlinks)
        self._blobs = blobs  # repo-relative POSIX path -> committed bytes

    def is_regular(self, relpath: str) -> bool:
        return relpath in self._regular

    def is_symlink(self, relpath: str) -> bool:
        return relpath in self._symlinks

    def regular_paths(self) -> list[str]:
        return sorted(self._regular)

    def all_python_paths(self) -> list[str]:
        return [relpath for relpath in sorted(self._regular) if relpath.endswith(".py")]

    def python_paths_under(self, prefix: str) -> list[str]:
        prefix = prefix.rstrip("/") + "/"
        return [relpath for relpath in self.all_python_paths() if relpath.startswith(prefix)]

    def paths_with_suffix(self, suffixes: tuple[str, ...]) -> list[str]:
        return [
            relpath
            for relpath in sorted(self._regular)
            if relpath.lower().endswith(suffixes)
        ]

    def read_bytes(self, relpath: str) -> bytes:
        try:
            return self._blobs[relpath]
        except KeyError as exc:  # missing committed blob -> fail closed
            raise AuthorityError(
                f"committed blob is unavailable for tracked path {relpath!r}"
            ) from exc

    def read_text(self, relpath: str) -> str:
        # ``utf-8-sig`` tolerates a leading BOM that some checked-in sources carry
        # while still rejecting genuinely non-UTF-8 content.
        return self.read_bytes(relpath).decode("utf-8-sig")


def build_repo_tree(repo_root: Path) -> _RepoTree:
    """Enumerate the committed tree at the exact HEAD and read the parseable
    blobs.  Fails closed on head/tree movement, a malformed object listing, a
    non-UTF-8 tracked path, a duplicate path, or any Git command failure."""
    repo_root = Path(repo_root)
    head, tree = git_identity(repo_root)
    raw = _git_bytes(repo_root, "ls-tree", "-r", "-z", tree)
    regular: dict[str, str] = {}
    symlinks: dict[str, str] = {}
    for record in raw.split(b"\x00"):
        if not record:
            continue
        meta, tab, path_bytes = record.partition(b"\t")
        if not tab:
            raise AuthorityError("malformed git ls-tree record (no path separator)")
        fields = meta.decode("ascii", "strict").split(" ")
        if len(fields) != 3:
            raise AuthorityError("malformed git ls-tree metadata")
        mode, obj_type, sha = fields
        try:
            path = path_bytes.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise AuthorityError("non-UTF-8 tracked path is not permitted") from exc
        if obj_type == "blob" and mode in ("100644", "100755"):
            if path in regular or path in symlinks:
                raise AuthorityError(f"duplicate tracked path from git ls-tree: {path}")
            regular[path] = sha
        elif obj_type == "blob" and mode == "120000":
            # A symlink is *not* a command surface: its target must never affect
            # the discovered set.  Record it only so type mismatches are visible.
            if path in regular or path in symlinks:
                raise AuthorityError(f"duplicate tracked path from git ls-tree: {path}")
            symlinks[path] = sha
        # trees / gitlinks / other modes contribute no command surface

    read_shas = {sha for path, sha in regular.items() if _is_parseable_path(path)}
    sha_bytes = _cat_file_batch(repo_root, read_shas)
    blobs = {
        path: sha_bytes[sha]
        for path, sha in regular.items()
        if _is_parseable_path(path)
    }

    # Guard against HEAD/tree movement in between the identity read and the
    # object enumeration: a shifting repository must never yield a receipt.
    head_again, tree_again = git_identity(repo_root)
    if (head, tree) != (head_again, tree_again):
        raise AuthorityError("git HEAD/tree moved during discovery")
    return _RepoTree(repo_root, head, tree, regular, symlinks, blobs)


def discover_from_tree(tree: _RepoTree) -> list[dict[str, Any]]:
    """Return the sorted, canonical list of discovered Runtime surfaces for an
    already-built committed-tree view."""
    surfaces: list[dict[str, Any]] = []
    surfaces += _discover_console_scripts(tree)
    surfaces += _discover_python_modules(tree)
    surfaces += _discover_python_files(tree)
    surfaces += _discover_mcp_tools(tree)
    surfaces += _discover_typer_subcommands(tree)
    surfaces += _discover_argparse_subcommands(tree)
    surfaces += _discover_http_actions(tree)
    surfaces += _discover_launchers(tree)
    surfaces += _discover_launchd_templates(tree)
    _reject_unsupported_service_generators(tree)
    surfaces += _discover_generated_systemd_units(tree)
    surfaces += _discover_scheduled_tasks_and_wrappers(tree)
    surfaces += _discover_container_entrypoints(tree)

    seen: dict[str, dict[str, Any]] = {}
    for surface in surfaces:
        source_path = surface["sourcePath"]
        source_digest = sha256_hex(tree.read_bytes(source_path))
        surface = {
            **surface,
            "canonicalTarget": (
                f"{surface['canonicalTarget']};sourceBlobSha256={source_digest}"
            ),
        }
        surface_id = surface["id"]
        if surface_id in seen:
            raise AuthorityError(f"duplicate discovered surface id: {surface_id}")
        seen[surface_id] = surface
    return [seen[key] for key in sorted(seen)]


def discover(repo_root: Path) -> list[dict[str, Any]]:
    """Return the sorted, canonical list of discovered Runtime surfaces.

    Discovery is a deterministic function of the pinned Git HEAD/tree: candidate
    paths come from ``git ls-tree`` at the exact committed tree and every parsed
    byte is the committed blob, so the same tree always yields byte-identical
    output regardless of untracked/ignored scratch or checkout-specific files.
    Raises :class:`AuthorityError` on a duplicate surface id.
    """
    return discover_from_tree(build_repo_tree(repo_root))


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
    _require(
        row["authoritySource"] in AUTHORITY_SOURCES,
        f"row {index} has an unknown authoritySource",
    )
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
    resources_read = _sorted_unique_str_list(row["resourcesRead"], f"row {index} resourcesRead")
    resources_written = _sorted_unique_str_list(row["resourcesWritten"], f"row {index} resourcesWritten")
    _require(
        all(resource in RESOURCE_VOCABULARY for resource in [*resources_read, *resources_written]),
        f"row {index} resources must be drawn from the resource vocabulary",
    )
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
    _require(not live, f"row {index} attempts to grant live Runtime authority")
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


_MUTATION_TOKENS = re.compile(
    r"(?:^|[-_:/# ])(?:ack|activate|apply|approve|arm|backfill|backup|blackout|blocklist|canary|cleanup|"
    r"clear|contain|create|dedupe|delete|deploy|destroy|discover|drop|erase|grant|halt|import|insert|kill|"
    r"lease|lift|mark|migrate|normalize|park|poll|promote|purge|push|quarantine|reenrich|register|repair|"
    r"reset|resolve|restart|restore|result|resume|retire|retry|revert|revoke|rollback|run|score|set|setup|"
    r"split|stop|supervise|swap|sync|track|truncate|unguard|update|validate|wipe|worker|write)"
    r"(?:$|[-_:/# ])",
    re.IGNORECASE,
)
_READ_ONLY_TOKENS = re.compile(
    r"(?:^|[-_:/# ])(?:audit|check|challenges|doctor|explain|get|inventory|list|readiness|report|status|"
    r"verify|version)(?:$|[-_:/# ])",
    re.IGNORECASE,
)
_EMPLOYER_TOKENS = re.compile(
    r"(?:^|[-_:/# ])(?:apply|approve|employer|lease|linkedin|submit|write[-_]result)(?:$|[-_:/# ])",
    re.IGNORECASE,
)


def _validate_classification_against_surface(row: dict[str, Any], index: int) -> None:
    """Conservatively cross-check high-risk structural names against metadata."""
    if row["surfaceKind"] in {"argparse-subcommand", "typer-subcommand", "mcp-tool", "console-http-action"}:
        signal = f"{row['symbol']} {row['canonicalTarget']}"
    else:
        signal = f"{row['id']} {row['sourcePath']} {row['canonicalTarget']}"
    read_only = bool(_READ_ONLY_TOKENS.search(signal)) and not bool(_MUTATION_TOKENS.search(row["symbol"]))
    mutation = bool(_MUTATION_TOKENS.search(signal)) and not read_only
    if mutation:
        _require(
            row["readWriteClass"] in {"admin-ddl", "control", "write"},
            f"row {index} structurally names a mutation but is classified read-only",
        )
        _require(
            bool(row["resourcesWritten"]),
            f"row {index} structurally names a mutation but declares no written resources",
        )
    employer_mutation = bool(_EMPLOYER_TOKENS.search(signal)) and not read_only
    if employer_mutation:
        _require(
            row["employerMutation"] is True,
            f"row {index} structurally names employer mutation but is not flagged",
        )
    if row["sourcePath"] == "backup-fleet-pg.ps1":
        _require(
            row["readWriteClass"] == "write"
            and row["resourcesRead"] == ["fleet-postgres"]
            and row["resourcesWritten"] == ["backup-artifact"]
            and row["failClosedBehavior"] == "deny-unverified-backup-writer",
            f"row {index} fleet backup surface must bind fleet read to backup-artifact write",
        )
    if row["sourcePath"] == "scripts/backup-railway-postgres.ps1":
        _require(
            row["readWriteClass"] == "write"
            and row["resourcesRead"] == ["brain-postgres"]
            and row["resourcesWritten"] == ["backup-artifact"]
            and row["failClosedBehavior"] == "deny-unverified-backup-writer",
            f"row {index} Railway backup surface must bind brain read to backup-artifact write",
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
        _validate_classification_against_surface(row, index)
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
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # fail closed on Git failure
        raise AuthorityError(f"git {' '.join(arguments)} failed") from exc
    return result.stdout.strip()


def _git_bytes(repo_root: Path, *arguments: str) -> bytes:
    """Like :func:`_git` but returns raw stdout bytes (for NUL-delimited / binary
    plumbing output).  Fails closed on any Git command failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuthorityError(f"git {' '.join(arguments)} failed") from exc
    return result.stdout


def _cat_file_batch(repo_root: Path, shas: Iterable[str]) -> dict[str, bytes]:
    """Read the committed content of every requested blob in a single
    ``git cat-file --batch`` process.  Fails closed on a missing object, a
    malformed batch stream, or any Git command failure."""
    ordered = sorted(set(shas))
    if not ordered:
        return {}
    for sha in ordered:
        if not _COMMIT_RE.fullmatch(sha):
            raise AuthorityError(f"refusing to read non-canonical object id: {sha!r}")
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "--batch"],
            input=("\n".join(ordered) + "\n").encode("ascii"),
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuthorityError("git cat-file --batch failed") from exc

    out = completed.stdout
    result: dict[str, bytes] = {}
    position = 0
    for sha in ordered:
        newline = out.find(b"\n", position)
        if newline == -1:
            raise AuthorityError("malformed git cat-file batch stream")
        header = out[position:newline].decode("ascii", "strict")
        position = newline + 1
        parts = header.split(" ")
        if len(parts) == 2 and parts[1] == "missing":
            raise AuthorityError(f"committed blob is missing: {sha}")
        if len(parts) != 3 or parts[0] != sha or parts[1] != "blob":
            raise AuthorityError(f"unexpected git cat-file header: {header!r}")
        size = int(parts[2])
        content = out[position : position + size]
        if len(content) != size:
            raise AuthorityError("truncated git cat-file batch content")
        position += size
        if out[position : position + 1] != b"\n":
            raise AuthorityError("malformed git cat-file batch record terminator")
        position += 1
        result[sha] = content
    return result


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


def _validate_receipt_shape(receipt: Any) -> None:
    _require(isinstance(receipt, dict), "receipt must be a JSON object")
    _require(set(receipt) == RECEIPT_KEYS, "receipt has unexpected or missing keys")
    supplied = receipt["selfHash"]
    _require(
        isinstance(supplied, str)
        and re.fullmatch(r"[0-9a-f]{64}", supplied) is not None
        and supplied != "0" * 64,
        "receipt selfHash must be a resolved SHA-256 digest",
    )
    body = {key: value for key, value in receipt.items() if key != "selfHash"}
    expected = sha256_hex(canonical_json(body))
    if supplied != expected:
        raise AuthorityError("receipt self-hash does not match its content (forged or drifted)")
    _require(receipt["schemaVersion"] == RECEIPT_SCHEMA_VERSION, "receipt schemaVersion is unsupported")
    _require(receipt["runtimeKind"] == RUNTIME_KIND, "receipt runtimeKind is unsupported")
    _require(receipt["repository"] == REPOSITORY, "receipt repository is unexpected")
    _require(receipt["noGlobalAuthorityClaim"] is True, "receipt must disclaim global authority")
    _require(
        isinstance(receipt.get("head"), str)
        and _COMMIT_RE.fullmatch(receipt["head"])
        and receipt["head"] != "0" * 40,
        "receipt head must be a resolved commit",
    )
    _require(
        isinstance(receipt.get("tree"), str)
        and _COMMIT_RE.fullmatch(receipt["tree"])
        and receipt["tree"] != "0" * 40,
        "receipt tree must be a resolved tree",
    )
    for field in ("discoveryDigest", "matrixDigest"):
        value = receipt.get(field)
        _require(
            isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{64}", value) is not None
            and value != "0" * 64,
            f"receipt {field} must be a resolved SHA-256 digest",
        )
    _require(receipt.get("cleanState") is True, "receipt must bind a clean tree")
    counts = receipt.get("counts")
    _require(isinstance(counts, dict), "receipt counts must be an object")
    _require(set(counts) == RECEIPT_COUNT_KEYS, "receipt counts have unexpected or missing keys")
    _require(
        all(
            isinstance(counts[key], int) and not isinstance(counts[key], bool) and counts[key] >= 0
            for key in RECEIPT_COUNT_KEYS - {"byKind"}
        ),
        "receipt scalar counts must be non-negative integers",
    )
    _require(
        isinstance(counts["byKind"], dict)
        and set(counts["byKind"]) == set(INCLUDED_SURFACE_KINDS)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in counts["byKind"].values()
        ),
        "receipt byKind counts must exactly cover the included kinds",
    )
    _require(counts.get("unclassified") == 0, "receipt must assert zero unclassified surfaces")
    _require(
        counts.get("discovered") == counts.get("rows"),
        "receipt discovered/rows counts must agree",
    )
    _require(
        counts["discovered"] == sum(counts["byKind"].values()),
        "receipt discovered count must equal the byKind total",
    )
    _require(counts["liveWriters"] == 0, "receipt must bind zero live writers")
    _require(
        counts["deferredKinds"] == len(DEFERRED_SURFACE_KINDS),
        "receipt deferredKinds count is inconsistent",
    )
    scan_forbidden_content(receipt, "receipt")


def verify_receipt(
    document: bytes | str,
    *,
    repo_root: Path,
    matrix_path: Path,
) -> dict[str, Any]:
    """Strictly decode and re-bind a receipt to fresh repository facts.

    A self-hash is only corruption detection. Authority verification recomputes
    the exact clean HEAD/tree, discovery, matrix, counts, digests, and canonical
    receipt bytes. Duplicate, reordered, or noncanonical documents are denied.
    """
    raw = document.encode("utf-8") if isinstance(document, str) else document
    _require(isinstance(raw, bytes), "receipt must be supplied as canonical bytes")
    payload = raw[:-1] if raw.endswith(b"\n") else raw
    receipt = strict_json_loads(payload, "receipt")
    _validate_receipt_shape(receipt)
    canonical = canonical_json(receipt)
    _require(payload == canonical, "receipt is reordered or not canonical JSON")
    expected = _evaluate_unverified(Path(repo_root), Path(matrix_path))
    _require(
        canonical == canonical_json(expected),
        "receipt does not re-bind to the exact current repository state",
    )
    return receipt


def load_matrix_document(path: Path) -> tuple[dict[str, Any], bytes]:
    content = Path(path).read_bytes()
    matrix = strict_json_loads(content, f"matrix {Path(path).name}")
    if not isinstance(matrix, dict):
        raise AuthorityError("matrix document must be a JSON object")
    return matrix, content


def _resolve_repository_paths(repo_root: Path, matrix_path: Path) -> tuple[Path, Path]:
    """Resolve the matrix relative to its repository and require containment."""
    try:
        resolved_root = Path(repo_root).resolve(strict=True)
        candidate = Path(matrix_path)
        if not candidate.is_absolute():
            candidate = resolved_root / candidate
        resolved_matrix = candidate.resolve(strict=True)
        resolved_matrix.relative_to(resolved_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise AuthorityError(
            "matrix path must resolve to an existing file inside the repository"
        ) from exc
    return resolved_root, resolved_matrix


def _evaluate_unverified(repo_root: Path, matrix_path: Path) -> dict[str, Any]:
    repo_root, matrix_path = _resolve_repository_paths(repo_root, matrix_path)
    # Gate on a clean tracked tree first: even though discovery binds committed
    # blob bytes, the receipt asserts that the *checkout* matches those bytes, so
    # any tracked modification/staging/deletion/type-change must fail issuance.
    assert_clean(repo_root)
    repo_tree = build_repo_tree(repo_root)
    matrix, _content = load_matrix_document(matrix_path)
    rows = validate_matrix(matrix)
    discovered = discover_from_tree(repo_tree)
    check_coverage(discovered, rows)
    return build_receipt(
        head=repo_tree.head,
        tree=repo_tree.tree,
        matrix=matrix,
        discovered=discovered,
        rows=rows,
        matrix_path=matrix_path.relative_to(repo_root).as_posix(),
        clean_state=True,
    )


def evaluate(repo_root: Path, matrix_path: Path) -> dict[str, Any]:
    """Run the complete gate and self-verify against fresh repository facts."""
    repo_root = Path(repo_root)
    matrix_path = Path(matrix_path)
    receipt = _evaluate_unverified(repo_root, matrix_path)
    verify_receipt(
        canonical_json(receipt),
        repo_root=repo_root,
        matrix_path=matrix_path,
    )
    return receipt
