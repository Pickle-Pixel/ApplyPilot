"""Helpers for selecting and launching the autonomous agent backend."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass

from applypilot import config


@dataclass(frozen=True)
class AgentBackend:
    """Execution backend for the auto-apply agent."""

    name: str
    binary: str
    model: str


def get_agent_backend(preferred: str | None = None, model: str | None = None) -> AgentBackend:
    """Resolve the backend and default model to use."""
    backend = (preferred or config.get_agent_backend()).strip().lower()
    if backend not in config.AGENT_BACKENDS:
        backend = config.get_agent_backend()

    if backend == "codex":
        binary = shutil.which("codex") or "codex"
        resolved_model = model or "gpt-5.4-mini"
    else:
        binary = shutil.which("claude") or "claude"
        resolved_model = model or "haiku"

    return AgentBackend(name=backend, binary=binary, model=resolved_model)


def codex_login_ok() -> tuple[bool, str]:
    """Check Codex login status."""
    return config.codex_login_status()


def build_playwright_override_args(cdp_port: int) -> list[str]:
    """Build `codex exec -c` overrides for the Playwright MCP server."""
    args = [
        "@playwright/mcp@latest",
        f"--cdp-endpoint=http://localhost:{cdp_port}",
        f"--viewport-size={config.DEFAULTS['viewport']}",
    ]
    return [
        "-c",
        'mcp_servers.playwright.command="npx"',
        "-c",
        f"mcp_servers.playwright.args={json.dumps(args)}",
    ]
