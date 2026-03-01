"""Tests for output parsing and runner edge behavior in backends.

@file test_parser_edge_cases.py
@description Validates that malformed subprocess output, non-JSON lines,
             partial RESULT: lines, and process failures produce controlled
             status strings (not crashes). Uses mock subprocess to stay offline.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from applypilot.apply.backends import BackendError, ClaudeBackend, OpenCodeBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(title: str = "Test Engineer", site: str = "testco") -> dict:
    """Build a minimal job dict for backend.run_job()."""
    return {
        "url": "https://example.com/jobs/1",
        "title": title,
        "site": site,
        "application_url": "https://example.com/apply/1",
        "tailored_resume_path": None,
        "fit_score": 8,
        "location": "Remote",
        "full_description": "Test job description",
        "cover_letter_path": None,
    }


def _fake_popen(stdout_lines: list[str], returncode: int = 0):
    """Create a mock Popen that yields the given stdout lines.

    Returns a mock that behaves like subprocess.Popen:
    - stdin.write/close are no-ops
    - stdout iterates over provided lines
    - wait() returns immediately
    - returncode set as specified
    - poll() returns returncode
    """
    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.stdin = MagicMock()
    mock_proc.stdout = iter(line + "\n" for line in stdout_lines)
    mock_proc.returncode = returncode
    mock_proc.wait.return_value = returncode
    mock_proc.poll.return_value = returncode
    mock_proc.pid = 12345
    return mock_proc


# Patch targets — dashboard and config are side-effect-heavy; mock them out
_DASHBOARD_PATCHES = {
    "applypilot.apply.dashboard.update_state": MagicMock(),
    "applypilot.apply.dashboard.add_event": MagicMock(),
    "applypilot.apply.dashboard.get_state": MagicMock(return_value=None),
}


def _run_claude_with_output(
    stdout_lines: list[str],
    returncode: int = 0,
    tmp_path: Path | None = None,
) -> tuple[str, int]:
    """Run ClaudeBackend.run_job() with mocked subprocess output."""
    backend = ClaudeBackend()
    worker_dir = tmp_path or Path("/tmp/test-worker")
    mcp_config = worker_dir / "mcp.json" if tmp_path else Path("/tmp/mcp.json")

    mock_proc = _fake_popen(stdout_lines, returncode)

    with (
        patch("subprocess.Popen", return_value=mock_proc),
        patch.dict("applypilot.apply.backends.__dict__", {}),
        patch("applypilot.apply.dashboard.update_state"),
        patch("applypilot.apply.dashboard.add_event"),
        patch("applypilot.apply.dashboard.get_state", return_value=None),
    ):
        return backend.run_job(
            job=_make_job(),
            port=9222,
            worker_id=0,
            model="test-model",
            agent=None,
            dry_run=True,
            prompt="test prompt",
            mcp_config_path=mcp_config,
            worker_dir=worker_dir,
        )


def _run_opencode_with_output(
    stdout_lines: list[str],
    returncode: int = 0,
    tmp_path: Path | None = None,
) -> tuple[str, int]:
    """Run OpenCodeBackend.run_job() with mocked subprocess output."""
    backend = OpenCodeBackend()
    worker_dir = tmp_path or Path("/tmp/test-worker")
    mcp_config = worker_dir / "mcp.json" if tmp_path else Path("/tmp/mcp.json")

    mock_proc = _fake_popen(stdout_lines, returncode)

    with (
        patch("subprocess.Popen", return_value=mock_proc),
        patch.object(backend, "_find_binary", return_value="/usr/bin/opencode"),
        patch("applypilot.apply.dashboard.update_state"),
        patch("applypilot.apply.dashboard.add_event"),
        patch("applypilot.apply.dashboard.get_state", return_value=None),
    ):
        return backend.run_job(
            job=_make_job(),
            port=9222,
            worker_id=0,
            model="test-model",
            agent=None,
            dry_run=True,
            prompt="test prompt",
            mcp_config_path=mcp_config,
            worker_dir=worker_dir,
        )


# ---------------------------------------------------------------------------
# Claude Backend: Output parsing edge cases
# ---------------------------------------------------------------------------


class TestClaudeMalformedOutput:
    """ClaudeBackend should never crash on bad output; return controlled status."""

    def test_empty_output_returns_no_result(self, tmp_path):
        """No output at all => failed:no_result_line."""
        status, duration = _run_claude_with_output([], tmp_path=tmp_path)
        assert status == "failed:no_result_line"
        assert duration >= 0

    def test_only_garbage_lines(self, tmp_path):
        """Non-JSON garbage output => failed:no_result_line (not crash)."""
        lines = [
            "this is not json",
            "neither is this!!!",
            "{{broken json{",
        ]
        status, _ = _run_claude_with_output(lines, tmp_path=tmp_path)
        assert status == "failed:no_result_line"

    def test_valid_json_but_no_result_marker(self, tmp_path):
        """Valid JSON messages but no RESULT: in text => failed:no_result_line."""
        lines = [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "Working on it..."}]},
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "Done, I think."}]},
                }
            ),
        ]
        status, _ = _run_claude_with_output(lines, tmp_path=tmp_path)
        assert status == "failed:no_result_line"

    def test_result_applied_parsed(self, tmp_path):
        """RESULT:APPLIED in output => 'applied' status."""
        lines = [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "RESULT:APPLIED"}]},
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                    "total_cost_usd": 0.01,
                    "num_turns": 3,
                    "result": "RESULT:APPLIED",
                }
            ),
        ]
        status, _ = _run_claude_with_output(lines, tmp_path=tmp_path)
        assert status == "applied"

    def test_result_expired_parsed(self, tmp_path):
        """RESULT:EXPIRED in output => 'expired' status."""
        lines = [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "RESULT:EXPIRED"}]},
                }
            ),
        ]
        status, _ = _run_claude_with_output(lines, tmp_path=tmp_path)
        assert status == "expired"

    def test_result_captcha_parsed(self, tmp_path):
        """RESULT:CAPTCHA in output => 'captcha' status."""
        lines = [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "RESULT:CAPTCHA"}]},
                }
            ),
        ]
        status, _ = _run_claude_with_output(lines, tmp_path=tmp_path)
        assert status == "captcha"

    def test_result_login_issue_parsed(self, tmp_path):
        """RESULT:LOGIN_ISSUE in output => 'login_issue' status."""
        lines = [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "RESULT:LOGIN_ISSUE"}]},
                }
            ),
        ]
        status, _ = _run_claude_with_output(lines, tmp_path=tmp_path)
        assert status == "login_issue"

    def test_result_failed_with_reason(self, tmp_path):
        """RESULT:FAILED:some_reason => 'failed:some_reason'."""
        lines = [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "RESULT:FAILED:form_not_found"},
                        ]
                    },
                }
            ),
        ]
        status, _ = _run_claude_with_output(lines, tmp_path=tmp_path)
        assert status == "failed:form_not_found"

    def test_result_failed_reason_promoted_to_captcha(self, tmp_path):
        """RESULT:FAILED:captcha promotes to 'captcha' status."""
        lines = [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "RESULT:FAILED:captcha"},
                        ]
                    },
                }
            ),
        ]
        status, _ = _run_claude_with_output(lines, tmp_path=tmp_path)
        assert status == "captcha"

    def test_mixed_json_and_garbage(self, tmp_path):
        """Mix of valid JSON and garbage lines — garbage appended, no crash."""
        lines = [
            "garbage before json",
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "RESULT:APPLIED"}]},
                }
            ),
            "trailing garbage",
        ]
        status, _ = _run_claude_with_output(lines, tmp_path=tmp_path)
        assert status == "applied"

    def test_negative_returncode_means_skipped(self, tmp_path):
        """Negative return code (signal kill) => 'skipped'."""
        lines = ["some output"]
        status, _ = _run_claude_with_output(lines, returncode=-9, tmp_path=tmp_path)
        assert status == "skipped"

    def test_result_failed_no_colon_after_failed(self, tmp_path):
        """RESULT:FAILED without trailing colon => 'failed:unknown'."""
        lines = [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "RESULT:FAILED"},
                        ]
                    },
                }
            ),
        ]
        status, _ = _run_claude_with_output(lines, tmp_path=tmp_path)
        assert status.startswith("failed:")

    def test_result_failed_reason_cleaned_of_markdown(self, tmp_path):
        """Trailing markdown chars (*`\") stripped from failure reason."""
        lines = [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": 'RESULT:FAILED:timeout_error**"`'},
                        ]
                    },
                }
            ),
        ]
        status, _ = _run_claude_with_output(lines, tmp_path=tmp_path)
        assert status == "failed:timeout_error"


# ---------------------------------------------------------------------------
# OpenCode Backend: Output parsing edge cases
# ---------------------------------------------------------------------------


class TestOpenCodeMalformedOutput:
    """OpenCodeBackend should handle bad output identically to Claude."""

    def test_empty_output_returns_no_result(self, tmp_path):
        status, duration = _run_opencode_with_output([], tmp_path=tmp_path)
        assert status == "failed:no_result_line"
        assert duration >= 0

    def test_only_garbage_lines(self, tmp_path):
        lines = ["not json at all", "!!!broken!!!"]
        status, _ = _run_opencode_with_output(lines, tmp_path=tmp_path)
        assert status == "failed:no_result_line"

    def test_opencode_text_event_with_result(self, tmp_path):
        """OpenCode 'text' event type with RESULT:APPLIED."""
        lines = [
            json.dumps(
                {
                    "type": "text",
                    "part": {"text": "RESULT:APPLIED"},
                }
            ),
        ]
        status, _ = _run_opencode_with_output(lines, tmp_path=tmp_path)
        assert status == "applied"

    def test_opencode_step_finish_token_stats(self, tmp_path):
        """step_finish event parsed without crash even with no RESULT."""
        lines = [
            json.dumps(
                {
                    "type": "step_finish",
                    "part": {
                        "tokens": {"input": 500, "output": 200, "cache": {"read": 10, "write": 5}},
                        "cost": 0.005,
                    },
                }
            ),
        ]
        status, _ = _run_opencode_with_output(lines, tmp_path=tmp_path)
        assert status == "failed:no_result_line"

    def test_opencode_failed_with_reason(self, tmp_path):
        lines = [
            json.dumps(
                {
                    "type": "text",
                    "part": {"text": "RESULT:FAILED:page_load_error"},
                }
            ),
        ]
        status, _ = _run_opencode_with_output(lines, tmp_path=tmp_path)
        assert status == "failed:page_load_error"

    def test_opencode_negative_returncode(self, tmp_path):
        status, _ = _run_opencode_with_output(["output"], returncode=-15, tmp_path=tmp_path)
        assert status == "skipped"

    def test_opencode_tool_use_event_no_crash(self, tmp_path):
        """tool_use events processed without crash, even with minimal data."""
        lines = [
            json.dumps(
                {
                    "type": "tool_use",
                    "part": {"name": "mcp__playwright__navigate", "input": {"url": "https://example.com"}},
                }
            ),
            json.dumps(
                {
                    "type": "text",
                    "part": {"text": "RESULT:APPLIED"},
                }
            ),
        ]
        status, _ = _run_opencode_with_output(lines, tmp_path=tmp_path)
        assert status == "applied"

    def test_opencode_mixed_garbage_and_events(self, tmp_path):
        """Non-JSON lines mixed with valid events — no crash."""
        lines = [
            "warning: something",
            json.dumps({"type": "text", "part": {"text": "Working..."}}),
            "another warning",
            json.dumps({"type": "text", "part": {"text": "RESULT:EXPIRED"}}),
        ]
        status, _ = _run_opencode_with_output(lines, tmp_path=tmp_path)
        assert status == "expired"


# ---------------------------------------------------------------------------
# Process failure edge cases (shared patterns)
# ---------------------------------------------------------------------------


class TestProcessFailureEdgeCases:
    """Backend run_job handles subprocess exceptions gracefully."""

    def test_claude_popen_exception(self, tmp_path):
        """If Popen raises, run_job returns failed: with error text."""
        backend = ClaudeBackend()
        with (
            patch("subprocess.Popen", side_effect=OSError("binary not found")),
            patch("applypilot.apply.dashboard.update_state"),
            patch("applypilot.apply.dashboard.add_event"),
            patch("applypilot.apply.dashboard.get_state", return_value=None),
        ):
            status, duration = backend.run_job(
                job=_make_job(),
                port=9222,
                worker_id=0,
                model="test",
                agent=None,
                dry_run=True,
                prompt="test",
                mcp_config_path=tmp_path / "mcp.json",
                worker_dir=tmp_path,
            )
        assert status.startswith("failed:")
        assert "binary not found" in status

    def test_opencode_missing_binary(self):
        """OpenCodeBackend raises BackendError when opencode not on PATH."""
        from applypilot.apply.backends import BackendError

        backend = OpenCodeBackend()
        with patch("shutil.which", return_value=None):
            with pytest.raises(BackendError, match="OpenCode CLI not found"):
                backend._find_binary()

    def test_opencode_popen_exception(self, tmp_path):
        """If Popen raises on opencode, run_job returns failed: error."""
        backend = OpenCodeBackend()
        with (
            patch("subprocess.Popen", side_effect=OSError("spawn failed")),
            patch.object(backend, "_find_binary", return_value="/usr/bin/opencode"),
            patch("applypilot.apply.dashboard.update_state"),
            patch("applypilot.apply.dashboard.add_event"),
            patch("applypilot.apply.dashboard.get_state", return_value=None),
        ):
            status, duration = backend.run_job(
                job=_make_job(),
                port=9222,
                worker_id=0,
                model="test",
                agent=None,
                dry_run=True,
                prompt="test",
                mcp_config_path=tmp_path / "mcp.json",
                worker_dir=tmp_path,
            )
        assert status.startswith("failed:")
        assert "spawn failed" in status


class TestOpenCodeMcpParity:
    """OpenCode backend enforces MCP baseline parity with Claude flow."""

    def test_build_command_includes_agent_when_set(self):
        backend = OpenCodeBackend()
        with patch.object(backend, "_find_binary", return_value="/usr/bin/opencode"):
            cmd = backend._build_command("o4-mini", Path("/tmp/w"), "coder")
        assert "--agent" in cmd
        assert "coder" in cmd

    def test_missing_required_mcp_servers_raises(self):
        backend = OpenCodeBackend()
        with patch.object(backend, "_list_mcp_servers", return_value={"search"}):
            with pytest.raises(BackendError, match="Missing server"):
                backend._ensure_required_mcp_servers(["playwright", "gmail"])


class TestPromptParity:
    """Both backends receive the exact same launcher-built prompt string."""

    def test_claude_prompt_forwarded_to_stdin(self, tmp_path):
        backend = ClaudeBackend()
        prompt = "PROMPT_PAYLOAD_CLAUDE"
        mock_proc = _fake_popen(["RESULT:FAILED:manual"], returncode=0)
        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch("applypilot.apply.dashboard.update_state"),
            patch("applypilot.apply.dashboard.add_event"),
            patch("applypilot.apply.dashboard.get_state", return_value=None),
        ):
            backend.run_job(
                job=_make_job(),
                port=9222,
                worker_id=0,
                model="haiku",
                agent=None,
                dry_run=True,
                prompt=prompt,
                mcp_config_path=tmp_path / "mcp.json",
                worker_dir=tmp_path,
                required_mcp_servers=["playwright", "gmail"],
            )
        mock_proc.stdin.write.assert_called_once_with(prompt)

    def test_opencode_prompt_forwarded_to_stdin(self, tmp_path):
        backend = OpenCodeBackend()
        prompt = "PROMPT_PAYLOAD_OPENCODE"
        mock_proc = _fake_popen(["RESULT:FAILED:manual"], returncode=0)
        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(backend, "_find_binary", return_value="/usr/bin/opencode"),
            patch.object(backend, "_list_mcp_servers", return_value={"playwright", "gmail"}),
            patch("applypilot.apply.dashboard.update_state"),
            patch("applypilot.apply.dashboard.add_event"),
            patch("applypilot.apply.dashboard.get_state", return_value=None),
        ):
            backend.run_job(
                job=_make_job(),
                port=9222,
                worker_id=0,
                model="gh/claude-sonnet-4.5",
                agent="coder",
                dry_run=True,
                prompt=prompt,
                mcp_config_path=tmp_path / "mcp.json",
                worker_dir=tmp_path,
                required_mcp_servers=["playwright", "gmail"],
            )
        mock_proc.stdin.write.assert_called_once_with(prompt)
