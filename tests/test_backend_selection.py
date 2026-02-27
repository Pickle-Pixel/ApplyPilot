"""Tests for backend selection via get_backend() factory.

@file test_backend_selection.py
@description Validates backend routing: default fallback, explicit selection,
             case-insensitive names, and actionable error on invalid names.
             All tests are offline and deterministic (no subprocess or network).
"""

from __future__ import annotations


import pytest

from applypilot.apply.backends import (
    DEFAULT_BACKEND,
    VALID_BACKENDS,
    AgentBackend,
    ClaudeBackend,
    InvalidBackendError,
    OpenCodeBackend,
    get_available_backends,
    get_backend,
    resolve_default_agent,
    resolve_default_model,
)


# ---------------------------------------------------------------------------
# 1. Default backend when APPLY_BACKEND env var is unset
# ---------------------------------------------------------------------------


class TestDefaultBackend:
    """get_backend(None) with no env var should return the code default."""

    def test_returns_claude_when_env_unset(self):
        """Default backend is 'claude' per DEFAULT_BACKEND constant."""
        backend = get_backend()
        assert isinstance(backend, ClaudeBackend)
        assert backend.name == "claude"

    def test_default_constant_is_claude(self):
        """DEFAULT_BACKEND constant matches expectations."""
        assert DEFAULT_BACKEND == "claude"


# ---------------------------------------------------------------------------
# 2. Explicit backend selection (positional arg)
# ---------------------------------------------------------------------------


class TestExplicitSelection:
    """get_backend('name') should return the matching backend class."""

    def test_select_claude_explicitly(self):
        backend = get_backend("claude")
        assert isinstance(backend, ClaudeBackend)
        assert backend.name == "claude"

    def test_select_opencode_explicitly(self):
        backend = get_backend("opencode")
        assert isinstance(backend, OpenCodeBackend)
        assert backend.name == "opencode"


# ---------------------------------------------------------------------------
# 3. Case-insensitive and whitespace-trimmed selection
# ---------------------------------------------------------------------------


class TestCaseInsensitive:
    """Backend names are normalized to lowercase and stripped."""

    def test_uppercase_claude(self):
        backend = get_backend("CLAUDE")
        assert isinstance(backend, ClaudeBackend)

    def test_mixed_case_opencode(self):
        backend = get_backend("OpenCode")
        assert isinstance(backend, OpenCodeBackend)

    def test_padded_whitespace(self):
        backend = get_backend("  claude  ")
        assert isinstance(backend, ClaudeBackend)

    def test_uppercase_with_whitespace(self):
        backend = get_backend("  OPENCODE  ")
        assert isinstance(backend, OpenCodeBackend)


# ---------------------------------------------------------------------------
# 4. APPLY_BACKEND environment variable
# ---------------------------------------------------------------------------


class TestEnvVarSelection:
    """get_backend(None) reads APPLY_BACKEND from env."""

    def test_env_var_claude(self, monkeypatch):
        monkeypatch.setenv("APPLY_BACKEND", "claude")
        backend = get_backend()
        assert isinstance(backend, ClaudeBackend)

    def test_env_var_opencode(self, monkeypatch):
        monkeypatch.setenv("APPLY_BACKEND", "opencode")
        backend = get_backend()
        assert isinstance(backend, OpenCodeBackend)

    def test_env_var_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("APPLY_BACKEND", "OpenCode")
        backend = get_backend()
        assert isinstance(backend, OpenCodeBackend)

    def test_explicit_arg_overrides_env(self, monkeypatch):
        """Explicit argument takes priority over env var."""
        monkeypatch.setenv("APPLY_BACKEND", "opencode")
        backend = get_backend("claude")
        assert isinstance(backend, ClaudeBackend)


# ---------------------------------------------------------------------------
# 5. Invalid backend raises InvalidBackendError with actionable message
# ---------------------------------------------------------------------------


class TestInvalidBackend:
    """Unsupported names produce clear, helpful errors."""

    def test_raises_invalid_backend_error(self):
        with pytest.raises(InvalidBackendError) as exc_info:
            get_backend("nonexistent")
        err = exc_info.value
        assert err.backend == "nonexistent"
        assert err.available == VALID_BACKENDS

    def test_error_message_includes_name(self):
        with pytest.raises(InvalidBackendError, match="nonexistent"):
            get_backend("nonexistent")

    def test_error_message_lists_supported_backends(self):
        with pytest.raises(InvalidBackendError, match="claude") as exc_info:
            get_backend("gpt4")
        msg = str(exc_info.value)
        assert "opencode" in msg
        assert "claude" in msg

    def test_error_message_mentions_env_var(self):
        with pytest.raises(InvalidBackendError, match="APPLY_BACKEND"):
            get_backend("wrong")

    def test_empty_string_raises(self):
        with pytest.raises(InvalidBackendError):
            get_backend("")

    def test_env_var_invalid_raises(self, monkeypatch):
        monkeypatch.setenv("APPLY_BACKEND", "llama")
        with pytest.raises(InvalidBackendError, match="llama"):
            get_backend()


# ---------------------------------------------------------------------------
# 6. Backend interface contract
# ---------------------------------------------------------------------------


class TestBackendInterface:
    """All backends satisfy the AgentBackend abstract interface."""

    @pytest.mark.parametrize("name", sorted(VALID_BACKENDS))
    def test_all_valid_backends_instantiate(self, name):
        backend = get_backend(name)
        assert isinstance(backend, AgentBackend)
        assert backend.name == name

    def test_get_available_backends_matches_valid(self):
        assert get_available_backends() == VALID_BACKENDS

    def test_claude_has_required_methods(self):
        b = ClaudeBackend()
        assert callable(b.run_job)
        assert callable(b.get_active_proc)
        assert hasattr(b, "name")

    def test_opencode_has_required_methods(self):
        b = OpenCodeBackend()
        assert callable(b.run_job)
        assert callable(b.get_active_proc)
        assert hasattr(b, "name")

    def test_active_proc_none_by_default(self):
        """No process active until run_job called."""
        for name in VALID_BACKENDS:
            b = get_backend(name)
            assert b.get_active_proc(0) is None
            assert b.get_active_proc(99) is None


class TestBackendDefaults:
    """Backend-aware model and agent default resolution."""

    def test_default_model_for_claude(self, monkeypatch):
        monkeypatch.delenv("APPLY_CLAUDE_MODEL", raising=False)
        assert resolve_default_model("claude") == "haiku"

    def test_default_model_for_claude_env_override(self, monkeypatch):
        monkeypatch.setenv("APPLY_CLAUDE_MODEL", "sonnet")
        assert resolve_default_model("claude") == "sonnet"

    def test_default_model_for_opencode_uses_llm_model(self, monkeypatch):
        monkeypatch.delenv("APPLY_OPENCODE_MODEL", raising=False)
        monkeypatch.setenv("LLM_MODEL", "gh/claude-sonnet-4.5")
        assert resolve_default_model("opencode") == "gh/claude-sonnet-4.5"

    def test_default_model_for_opencode_env_override(self, monkeypatch):
        monkeypatch.setenv("APPLY_OPENCODE_MODEL", "o4-mini")
        monkeypatch.setenv("LLM_MODEL", "ignored")
        assert resolve_default_model("opencode") == "o4-mini"

    def test_default_agent_for_opencode_env(self, monkeypatch):
        monkeypatch.setenv("APPLY_OPENCODE_AGENT", "coder")
        assert resolve_default_agent("opencode") == "coder"

    def test_default_agent_for_claude(self):
        assert resolve_default_agent("claude") is None
