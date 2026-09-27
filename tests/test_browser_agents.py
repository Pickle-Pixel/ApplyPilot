"""Tests for browser-agent selection and command construction."""

from pathlib import Path

import pytest

from applypilot import config
from applypilot.apply import launcher


def test_configured_browser_agent_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "load_env", lambda: None)
    monkeypatch.setenv(config.BROWSER_AGENT_ENV, "claude")
    monkeypatch.setattr(launcher.shutil, "which", lambda name: f"/bin/{name}")

    assert config.get_browser_agent() == "claude"
    assert launcher.resolve_agent("auto") == "claude"


def test_auto_agent_falls_back_to_an_installed_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "get_browser_agent", lambda: "auto")
    monkeypatch.setattr(
        launcher.shutil,
        "which",
        lambda name: "/bin/claude" if name == "claude" else None,
    )

    assert launcher.resolve_agent("auto") == "claude"


def test_selected_agent_must_be_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="agy"):
        launcher.resolve_agent("antigravity")


def test_tier_checks_the_configured_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "load_env", lambda: None)
    monkeypatch.setenv("LLM_URL", "http://localhost:8080/v1")
    monkeypatch.setenv(config.BROWSER_AGENT_ENV, "claude")
    monkeypatch.setattr(config, "get_chrome_path", lambda: "/bin/chrome")
    monkeypatch.setattr(
        config.shutil,
        "which",
        lambda name: "/bin/agy" if name == "agy" else None,
    )

    assert config.get_tier() == 2

    monkeypatch.setenv(config.BROWSER_AGENT_ENV, "antigravity")
    assert config.get_tier() == 3


def test_antigravity_command_uses_stream_json() -> None:
    command, model = launcher.build_agent_command(
        "antigravity",
        None,
        "apply now",
        Path("mcp.json"),
    )

    assert command[0] == "agy"
    assert model == "gemini-3.8-flash-medium"
    assert "--output-format" in command
    assert "stream-json" in command
    assert "--print=apply now" in command


def test_claude_command_uses_generated_mcp_config() -> None:
    command, model = launcher.build_agent_command(
        "claude",
        None,
        "apply now",
        Path("mcp.json"),
    )

    assert command[0] == "claude"
    assert model == "haiku"
    assert command[command.index("--mcp-config") + 1] == "mcp.json"
    assert command[-1] == "-"
