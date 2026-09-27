"""Tests for browser-agent setup in the init wizard."""

from pathlib import Path

import pytest

from applypilot import config
from applypilot.wizard import init as init_wizard


def test_set_env_value_preserves_and_updates_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_MODEL=test-model\n", encoding="utf-8")
    monkeypatch.setattr(init_wizard, "ENV_PATH", env_path)

    init_wizard._set_env_value(config.BROWSER_AGENT_ENV, "claude")
    init_wizard._set_env_value(config.BROWSER_AGENT_ENV, "antigravity")

    contents = env_path.read_text(encoding="utf-8")
    assert "LLM_MODEL=test-model" in contents
    assert contents.count(config.BROWSER_AGENT_ENV) == 1
    assert f"{config.BROWSER_AGENT_ENV}=antigravity" in contents


def test_antigravity_mcp_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> None:
        commands.append(command)

    monkeypatch.setattr(init_wizard.subprocess, "run", fake_run)

    assert init_wizard._configure_antigravity_mcp() is True
    assert commands[0][:5] == ["agy", "mcp", "add", "applypilot-playwright", "npx"]
    assert "@playwright/mcp@latest" in commands[0]
    assert commands[1][:5] == ["agy", "mcp", "add", "applypilot-gmail", "npx"]


def test_setup_saves_antigravity_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    confirmations = iter([True, False])
    mcp_configured: list[bool] = []

    monkeypatch.setattr(init_wizard, "ENV_PATH", env_path)
    monkeypatch.setattr(init_wizard.Confirm, "ask", lambda *_args, **_kwargs: next(confirmations))
    monkeypatch.setattr(init_wizard.Prompt, "ask", lambda *_args, **_kwargs: "antigravity")
    monkeypatch.setattr(init_wizard, "get_browser_agent", lambda: "auto")
    monkeypatch.setattr(init_wizard.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        init_wizard,
        "_configure_antigravity_mcp",
        lambda: mcp_configured.append(True) or True,
    )

    init_wizard._setup_auto_apply()

    assert f"{config.BROWSER_AGENT_ENV}=antigravity" in env_path.read_text(encoding="utf-8")
    assert mcp_configured == [True]


def test_setup_saves_claude_choice_without_antigravity_mcp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    confirmations = iter([True, False])
    mcp_configured: list[bool] = []

    monkeypatch.setattr(init_wizard, "ENV_PATH", env_path)
    monkeypatch.setattr(init_wizard.Confirm, "ask", lambda *_args, **_kwargs: next(confirmations))
    monkeypatch.setattr(init_wizard.Prompt, "ask", lambda *_args, **_kwargs: "claude")
    monkeypatch.setattr(init_wizard, "get_browser_agent", lambda: "auto")
    monkeypatch.setattr(init_wizard.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        init_wizard,
        "_configure_antigravity_mcp",
        lambda: mcp_configured.append(True) or True,
    )

    init_wizard._setup_auto_apply()

    assert f"{config.BROWSER_AGENT_ENV}=claude" in env_path.read_text(encoding="utf-8")
    assert mcp_configured == []
