"""Agent backend abstraction for auto-apply launcher.

This module provides a strategy/adapter pattern for switching between
different agent backends (Claude CLI, OpenCode, etc.) while preserving
the existing output parsing and status taxonomy.

@file backends.py
@description Backend abstraction layer for agent execution.
             Integrates with launcher.py to enable multi-backend support.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

# Supported backend identifiers
VALID_BACKENDS: frozenset[str] = frozenset({"claude", "opencode"})
DEFAULT_BACKEND: str = "claude"
DEFAULT_CLAUDE_MODEL: str = "haiku"
DEFAULT_OPENCODE_MODEL: str = "gpt-4o-mini"
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")


class BackendError(Exception):
    """Raised when backend operations fail or invalid backend is requested."""

    pass


class InvalidBackendError(BackendError):
    """Raised when an unsupported backend identifier is provided."""

    def __init__(self, backend: str, available: frozenset[str]) -> None:
        self.backend = backend
        self.available = available
        super().__init__(
            f"Invalid backend '{backend}'. "
            f"Supported backends: {', '.join(sorted(available))}. "
            f"Set via APPLY_BACKEND environment variable or backend config option."
        )


class AgentBackend(ABC):
    """Abstract base class for agent execution backends.

    Implementations must provide a run_job method that executes the agent
    with the given prompt and configuration, returning status and duration.
    """

    @abstractmethod
    def run_job(
        self,
        job: dict[str, Any],
        port: int,
        worker_id: int,
        model: str,
        agent: str | None,
        dry_run: bool,
        prompt: str,
        mcp_config_path: Path,
        worker_dir: Path,
        required_mcp_servers: Sequence[str] | None = None,
        update_callback: Any | None = None,
    ) -> tuple[str, int]:
        """Execute the agent for a single job application.

        Args:
            job: Job dictionary with url, title, site, etc.
            port: CDP port for browser connection.
            worker_id: Numeric worker identifier.
            model: Model name for the backend.
            dry_run: If True, don't actually submit applications.
            prompt: The full agent prompt text.
            mcp_config_path: Path to MCP configuration file.
            worker_dir: Working directory for the agent.
            update_callback: Optional callback for status updates.

        Returns:
            Tuple of (status_string, duration_ms). Status is one of:
            'applied', 'expired', 'captcha', 'login_issue',
            'failed:reason', or 'skipped'.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend identifier name."""
        ...

    @abstractmethod
    def get_active_proc(self, worker_id: int) -> subprocess.Popen | None:
        """Get the active process for a worker (for signal handling).

        Args:
            worker_id: Numeric worker identifier.

        Returns:
            The active subprocess.Popen instance, or None if no process is active.
        """
        ...


class ClaudeBackend(AgentBackend):
    """Claude Code CLI backend implementation.

    Spawns Claude Code CLI subprocess with Playwright MCP integration.
    Parses stream-json output to extract result status and token usage.
    """

    def __init__(self) -> None:
        self._active_procs: dict[int, subprocess.Popen] = {}

    @property
    def name(self) -> str:
        return "claude"

    def _build_command(
        self,
        model: str,
        mcp_config_path: Path,
    ) -> list[str]:
        """Build the Claude CLI command arguments."""
        return [
            "claude",
            "--model",
            model,
            "-p",
            "--mcp-config",
            str(mcp_config_path),
            "--permission-mode",
            "bypassPermissions",
            "--no-session-persistence",
            "--disallowedTools",
            (
                "mcp__gmail__draft_email,mcp__gmail__modify_email,"
                "mcp__gmail__delete_email,mcp__gmail__download_attachment,"
                "mcp__gmail__batch_modify_emails,mcp__gmail__batch_delete_emails,"
                "mcp__gmail__create_label,mcp__gmail__update_label,"
                "mcp__gmail__delete_label,mcp__gmail__get_or_create_label,"
                "mcp__gmail__list_email_labels,mcp__gmail__create_filter,"
                "mcp__gmail__list_filters,mcp__gmail__get_filter,"
                "mcp__gmail__delete_filter"
            ),
            "--output-format",
            "stream-json",
            "--verbose",
            "-",
        ]

    def _prepare_environment(self) -> dict[str, str]:
        """Prepare clean environment for Claude process."""
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        env.pop("CLAUDE_CODE_ENTRYPOINT", None)
        return env

    def run_job(
        self,
        job: dict[str, Any],
        port: int,
        worker_id: int,
        model: str,
        agent: str | None,
        dry_run: bool,
        prompt: str,
        mcp_config_path: Path,
        worker_dir: Path,
        required_mcp_servers: Sequence[str] | None = None,
        update_callback: Any | None = None,
    ) -> tuple[str, int]:
        """Execute Claude Code for a single job application.

        This implementation preserves the existing launcher.py behavior
        for Claude CLI execution, including output parsing and status
        taxonomy (APPLIED, EXPIRED, CAPTCHA, LOGIN_ISSUE, FAILED).
        """
        import re
        import time
        from datetime import datetime, timedelta

        from applypilot import config
        from applypilot.apply.dashboard import add_event, get_state, update_state

        cmd = self._build_command(model, mcp_config_path)
        env = self._prepare_environment()

        update_state(
            worker_id,
            status="applying",
            job_title=job["title"],
            company=job.get("site", ""),
            score=job.get("fit_score", 0),
            start_time=time.time(),
            actions=0,
            last_action="starting",
        )
        add_event(f"[W{worker_id}] Starting: {job['title'][:40]} @ {job.get('site', '')}")

        worker_log = config.LOG_DIR / f"worker-{worker_id}.log"

        # Set up JSONL audit file
        audit_path = config.LOG_DIR / f"worker-{worker_id}.events.jsonl"
        audit_file = open(audit_path, "a", encoding="utf-8")

        # Clean up old audit files (30-day retention)
        cutoff = datetime.utcnow() - timedelta(days=30)
        for old_file in config.LOG_DIR.glob("worker-*.events.jsonl"):
            try:
                if datetime.utcfromtimestamp(old_file.stat().st_mtime) < cutoff:
                    old_file.unlink()
            except Exception:
                pass

        ts_header = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_header = (
            f"\n{'=' * 60}\n"
            f"[{ts_header}] {job['title']} @ {job.get('site', '')}\n"
            f"URL: {job.get('application_url') or job['url']}\n"
            f"Score: {job.get('fit_score', 'N/A')}/10\n"
            f"{'=' * 60}\n"
        )

        start = time.time()
        stats: dict = {}
        proc = None

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                cwd=str(worker_dir),
            )
            self._active_procs[worker_id] = proc
            if proc.stdin is None or proc.stdout is None:
                raise BackendError("Claude backend process streams unavailable")
            stdin = proc.stdin
            stdout = proc.stdout

            stdin.write(prompt)
            stdin.close()

            text_parts: list[str] = []
            with open(worker_log, "a", encoding="utf-8") as lf:
                lf.write(log_header)

                # Instrumentation for timing analysis
                last_msg_time = None
                tool_start_time = None
                current_tool = None
                turn_count = 0
                perf_log_path = config.LOG_DIR / f"worker-{worker_id}.perf.jsonl"
                perf_file = open(perf_log_path, "a", encoding="utf-8")
                
                for line in stdout:
                    line = line.strip()
                    if not line:
                        continue
                    
                    msg_received_time = time.time()
                    
                    # Calculate gap from last message
                    if last_msg_time:
                        gap_ms = (msg_received_time - last_msg_time) * 1000
                    else:
                        gap_ms = 0
                    
                    try:
                        msg = json.loads(line)
                        msg_type = msg.get("type")
                        
                        # Log message timing
                        if gap_ms > 100:  # Only log significant gaps (>100ms)
                            perf_entry = {
                                "timestamp": datetime.utcnow().isoformat(),
                                "event": "message_received",
                                "msg_type": msg_type,
                                "gap_ms": round(gap_ms, 2),
                                "turn": turn_count,
                            }
                            perf_file.write(json.dumps(perf_entry) + "\n")
                            perf_file.flush()
                        
                        # Track step start
                        if msg_type == "step_start":
                            turn_count += 1
                            step_start_time = time.time()
                            if turn_count == 1:
                                add_event(f"[W{worker_id}] ✓ Agent active - starting application...")
                            perf_entry = {
                                "timestamp": datetime.utcnow().isoformat(),
                                "event": "llm_turn_start",
                                "turn": turn_count,
                            }
                            perf_file.write(json.dumps(perf_entry) + "\n")
                            perf_file.flush()
                            
                            # Log if this was a long wait
                            if gap_ms > 5000:  # >5 seconds
                                lf.write(f"\n[PERF] Long gap before turn {turn_count}: {gap_ms/1000:.1f}s (likely API latency + LLM thinking)\n")
                                lf.flush()
                        
                        # Track tool execution
                        if msg_type == "tool_use":
                            tool_start_time = time.time()
                            # Extract tool name (simplified)
                            part = msg.get("part", {})
                            current_tool = part.get("tool", part.get("name", "unknown"))
                        if msg_type == "assistant":
                            for block in msg.get("message", {}).get("content", []):
                                bt = block.get("type")
                                if bt == "text":
                                    text_parts.append(block["text"])
                                    lf.write(block["text"] + "\n")
                                elif bt == "tool_use":
                                    name = (
                                        block.get("name", "")
                                        .replace("mcp__playwright__", "")
                                        .replace("mcp__gmail__", "gmail:")
                                    )
                                    inp = block.get("input", {})
                                    if "url" in inp:
                                        desc = f"{name} {inp['url'][:60]}"
                                    elif "ref" in inp:
                                        desc = f"{name} {inp.get('element', inp.get('text', ''))}"[:50]
                                    elif "fields" in inp:
                                        desc = f"{name} ({len(inp['fields'])} fields)"
                                    elif "paths" in inp:
                                        desc = f"{name} upload"
                                    else:
                                        desc = name

                                    # Write enhanced log with timestamp
                                    ts = datetime.utcnow().isoformat()
                                    lf.write(f"{ts} >> {desc}\n")

                                    # Add dashboard event
                                    add_event(f"[W{worker_id}] {desc[:80]}")

                                    # Update state
                                    ws = get_state(worker_id)
                                    cur_actions = ws.actions if ws else 0
                                    update_state(worker_id, actions=cur_actions + 1, last_action=desc[:35])

                                    # Write JSONL audit entry
                                    audit_entry = {
                                        "timestamp": ts,
                                        "worker_id": worker_id,
                                        "type": "tool_use",
                                        "tool": name,
                                        "input": inp,  # Full input payload
                                        "tab_id": inp.get("tab_id"),
                                        "url": inp.get("url"),
                                    }
                                    audit_file.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
                                    audit_file.flush()
                        elif msg_type == "result":
                            stats = {
                                "input_tokens": msg.get("usage", {}).get("input_tokens", 0),
                                "output_tokens": msg.get("usage", {}).get("output_tokens", 0),
                                "cache_read": msg.get("usage", {}).get("cache_read_input_tokens", 0),
                                "cache_create": msg.get("usage", {}).get("cache_creation_input_tokens", 0),
                                "cost_usd": msg.get("total_cost_usd", 0),
                                "turns": msg.get("num_turns", 0),
                            }
                            text_parts.append(msg.get("result", ""))
                    except json.JSONDecodeError:
                        text_parts.append(line)
                        lf.write(line + "\n")

            proc.wait(timeout=300)
            returncode = proc.returncode
            proc = None

            if returncode and returncode < 0:
                return "skipped", int((time.time() - start) * 1000)

            output = "\n".join(text_parts)
            elapsed = int(time.time() - start)
            duration_ms = int((time.time() - start) * 1000)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_log = config.LOG_DIR / f"claude_{ts}_w{worker_id}_{job.get('site', 'unknown')[:20]}.txt"
            job_log.write_text(output, encoding="utf-8")

            if stats:
                cost = stats.get("cost_usd", 0)
                ws = get_state(worker_id)
                prev_cost = ws.total_cost if ws else 0.0
                update_state(worker_id, total_cost=prev_cost + cost)

            def _clean_reason(s: str) -> str:
                return re.sub(r'[*`"]+$', "", s).strip()

            for result_status in ["APPLIED", "EXPIRED", "CAPTCHA", "LOGIN_ISSUE"]:
                if f"RESULT:{result_status}" in output:
                    add_event(f"[W{worker_id}] {result_status} ({elapsed}s): {job['title'][:30]}")
                    update_state(worker_id, status=result_status.lower(), last_action=f"{result_status} ({elapsed}s)")
                    return result_status.lower(), duration_ms

            if "RESULT:FAILED" in output:
                for out_line in output.split("\n"):
                    if "RESULT:FAILED" in out_line:
                        reason = (
                            out_line.split("RESULT:FAILED:")[-1].strip()
                            if ":" in out_line[out_line.index("FAILED") + 6 :]
                            else "unknown"
                        )
                        reason = _clean_reason(reason)
                        PROMOTE_TO_STATUS = {"captcha", "expired", "login_issue"}
                        if reason in PROMOTE_TO_STATUS:
                            add_event(f"[W{worker_id}] {reason.upper()} ({elapsed}s): {job['title'][:30]}")
                            update_state(worker_id, status=reason, last_action=f"{reason.upper()} ({elapsed}s)")
                            return reason, duration_ms
                        add_event(f"[W{worker_id}] FAILED ({elapsed}s): {reason[:30]}")
                        update_state(worker_id, status="failed", last_action=f"FAILED: {reason[:25]}")
                        return f"failed:{reason}", duration_ms
                return "failed:unknown", duration_ms

            add_event(f"[W{worker_id}] NO RESULT ({elapsed}s)")
            update_state(worker_id, status="failed", last_action=f"no result ({elapsed}s)")
            return "failed:no_result_line", duration_ms

        except subprocess.TimeoutExpired:
            duration_ms = int((time.time() - start) * 1000)
            elapsed = int(time.time() - start)
            add_event(f"[W{worker_id}] TIMEOUT ({elapsed}s)")
            update_state(worker_id, status="failed", last_action=f"TIMEOUT ({elapsed}s)")
            return "failed:timeout", duration_ms
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            add_event(f"[W{worker_id}] ERROR: {str(e)[:40]}")
            update_state(worker_id, status="failed", last_action=f"ERROR: {str(e)[:25]}")
            return f"failed:{str(e)[:100]}", duration_ms
        finally:
            audit_file.close()
            self._active_procs.pop(worker_id, None)
            if proc is not None and proc.poll() is None:
                self._kill_process_tree(proc.pid)

    def _kill_process_tree(self, pid: int) -> None:
        """Kill a process and all its children."""
        import signal

        try:
            import psutil

            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            _, alive = psutil.wait_procs([parent], timeout=3)
            for p in alive:
                p.kill()
        except ImportError:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass

    def get_active_proc(self, worker_id: int) -> subprocess.Popen | None:
        """Get the active process for a worker (for signal handling)."""
        return self._active_procs.get(worker_id)

    def kill_all(self) -> None:
        """Kill all active Claude processes."""
        for worker_id, proc in list(self._active_procs.items()):
            if proc.poll() is None:
                self._kill_process_tree(proc.pid)
        self._active_procs.clear()


class OpenCodeBackend(AgentBackend):
    """OpenCode CLI backend implementation.

    Spawns OpenCode CLI subprocess in non-interactive 'run' mode with
    Playwright MCP integration. Parses JSON event stream to extract
    result status and token usage.

    OpenCode MCP servers must be pre-configured (via `opencode mcp add`)
    since OpenCode does not accept per-invocation MCP config files.
    """

    def __init__(self, use_global_config: bool | None = None) -> None:
        self._active_procs: dict[int, subprocess.Popen] = {}
        if use_global_config is None:
            use_global_config = os.environ.get("APPLY_OPENCODE_USE_GLOBAL_CONFIG", "").lower() in ("1", "true", "yes")
        self._use_global_config = use_global_config

    @property
    def name(self) -> str:
        return "opencode"

    def _find_binary(self) -> str:
        """Locate the opencode binary, raising clear errors if missing."""
        import shutil

        # Check PATH first
        binary = shutil.which("opencode")
        if binary:
            return binary

        # Check default installation location
        default_path = Path.home() / ".opencode" / "bin" / "opencode"
        if default_path.exists():
            return str(default_path)

        raise BackendError(
            "OpenCode CLI not found on PATH. "
            "Install it from https://opencode.ai or run: "
            "curl -fsSL https://opencode.ai/install | bash\n"
            "Then configure MCP servers: opencode mcp add playwright"
        )

    def _build_command(
        self,
        model: str,
        worker_dir: Path,
        agent: str | None,
        prompt: str,
    ) -> list[str]:
        """Build the OpenCode CLI command arguments.

        Note: OpenCode manages MCP servers via its own config system,
        not per-invocation config files like Claude CLI. Playwright MCP
        must be pre-configured via `opencode mcp add`.
        """
        binary = self._find_binary()
        cmd = [
            binary,
            "run",
            "--format",
            "json",
            "--dir",
            str(worker_dir),
        ]
        if model:
            cmd.extend(["--model", model])
        if agent:
            cmd.extend(["--agent", agent])
        # Add --variant for faster reasoning (minimal = fastest, high = most thorough)
        variant = os.environ.get("APPLY_OPENCODE_VARIANT", "minimal")
        if variant:
            cmd.extend(["--variant", variant])
        # OpenCode expects the prompt as positional argument(s)
        cmd.append(prompt)
        return cmd

    def _list_mcp_servers(self) -> set[str]:
        from applypilot import config
        binary = self._find_binary()
        proc = subprocess.run(
            [binary, "mcp", "list"],
            capture_output=True,
            text=True,
            check=False,
            env=self._prepare_environment(),
            cwd=str(config.APP_DIR),
        )
        output = proc.stdout or ""
        cleaned = ANSI_ESCAPE_RE.sub("", output)
        servers: set[str] = set()
        for line in cleaned.splitlines():
            m = re.match(
                r"^\s*[●*]?\s*[✓x]?\s*([A-Za-z0-9_-]+)\s+(connected|disconnected|error)\b",
                line.strip(),
            )
            if m:
                servers.add(m.group(1))
        return servers

    def _ensure_required_mcp_servers(self, required: Sequence[str] | None) -> None:
        if not required:
            return
        configured = self._list_mcp_servers()
        missing = [name for name in required if name not in configured]
        if not missing:
            return
        raise BackendError(
            "OpenCode MCP baseline mismatch. Missing server(s): "
            + ", ".join(missing)
            + ". Configure matching MCP servers before apply. "
            "Expected baseline: " + ", ".join(required) + ". Example: `opencode mcp add <name> -- <command>`"
        )

    def _validate_agent_config(self, agent: str | None) -> None:
        """Validate that the agent exists in the resolved opencode config.

        Uses `opencode debug agent <name>` so the CLI loads and merges all
        config files itself — no manual JSONC parsing needed.

        Args:
            agent: Agent name to validate.
        Raises:
            BackendError: If the agent is not found or the CLI check fails.
        """
        if not agent:
            return
        import json

        binary = self._find_binary()
        env = self._prepare_environment()
        from applypilot import config as ap_config
        proc = subprocess.run(
            [binary, "debug", "agent", agent],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=str(ap_config.APP_DIR),
        )
        if proc.returncode != 0:
            err = (proc.stdout or proc.stderr or "").strip()
            raise BackendError(
                f"Agent '{agent}' not found in opencode config. "
                f"opencode said: {err}. "
                "Make sure the agent is defined in ~/.applypilot/.opencode/opencode.jsonc "
                "and named 'applypilot-apply'."
            )
        # Parse the returned JSON to surface any prompt file issues early.
        # Failures here are non-fatal: the agent exists, run_job will catch issues.
        try:
            json.loads(ANSI_ESCAPE_RE.sub("", proc.stdout))
        except json.JSONDecodeError:
            pass


    def _prepare_environment(self) -> dict[str, str]:
        """Prepare environment for OpenCode process.

        By default, redirects XDG_CONFIG_HOME so opencode only loads the
        applypilot project config (~/.applypilot/.opencode/opencode.jsonc)
        and ignores the user's global ~/.config/opencode/ directory.

        Set APPLY_OPENCODE_USE_GLOBAL_CONFIG=true to opt into loading the
        global opencode config alongside the project config.
        """
        env = os.environ.copy()
        # Remove OpenCode Desktop environment variables.
        # When running inside the desktop app, these vars cause the CLI to
        # attach to the existing desktop session instead of running standalone.
        # XDG_STATE_HOME is also stripped: the desktop app points it at
        # /Library/Application Support/ai.opencode.desktop/ which is wrong
        # for standalone CLI usage; letting it fall back to the default
        # (~/.local/state/opencode/) is correct.
        desktop_vars = [
            "OPENCODE_CLIENT",
            "OPENCODE_SERVER_PASSWORD",
            "OPENCODE_SERVER_USERNAME",
            "OPENCODE",
            "OPENCODE_EXPERIMENTAL_FILEWATCHER",
            "OPENCODE_EXPERIMENTAL_ICON_DISCOVERY",
            "__CFBundleIdentifier",
            "XDG_STATE_HOME",
        ]
        for key in desktop_vars:
            env.pop(key, None)
        # Ensure PATH includes the opencode binary directory so the CLI is
        # always found even when not on the user's normal PATH.
        opencode_bin = str(Path.home() / ".opencode" / "bin")
        current_path = env.get("PATH", "")
        if opencode_bin not in current_path:
            env["PATH"] = f"{opencode_bin}:{current_path}"
        # Set TERM if absent (needed for proper terminal handling).
        if "TERM" not in env or not env["TERM"]:
            env["TERM"] = "xterm-256color"
        if not self._use_global_config:
            # Redirect XDG_CONFIG_HOME so opencode looks for its "global" config
            # at ~/.applypilot/opencode/ (which does not exist) instead of
            # ~/.config/opencode/. This prevents the user's personal opencode
            # config — including plugins like OMO and personal instructions —
            # from being loaded.
            #
            # Why not use "plugin": [] to suppress plugins? opencode's config
            # merge function (mergeConfigConcatArrays) CONCATENATES arrays across
            # config files — setting an empty array in project config does not
            # clear entries from global config. XDG_CONFIG_HOME redirection is the
            # only reliable way to prevent global config from loading.
            #
            # XDG_CONFIG_HOME only affects config file discovery. Auth credentials
            # (~/.opencode/auth.json), the session DB (~/.local/share/opencode/),
            # and state (~/.local/state/opencode/) use different XDG vars and are
            # unaffected by this redirect.
            from applypilot import config as ap_config
            env["XDG_CONFIG_HOME"] = str(ap_config.APP_DIR)
        return env

    def run_job(
        self,
        job: dict[str, Any],
        port: int,
        worker_id: int,
        model: str,
        agent: str | None,
        dry_run: bool,
        prompt: str,
        mcp_config_path: Path,
        worker_dir: Path,
        required_mcp_servers: Sequence[str] | None = None,
        update_callback: Any | None = None,
    ) -> tuple[str, int]:
        """Execute OpenCode for a single job application.

        Mirrors the status taxonomy of ClaudeBackend: APPLIED, EXPIRED,
        CAPTCHA, LOGIN_ISSUE, FAILED. Parses OpenCode's JSON event stream
        (step_start, text, tool_use, step_finish) to extract results.
        """
        import re
        import time
        from datetime import datetime, timedelta

        from applypilot import config
        from applypilot.apply.dashboard import add_event, get_state, update_state

        self._ensure_required_mcp_servers(required_mcp_servers)
        self._validate_agent_config(agent)
        cmd = self._build_command(model, worker_dir, agent, prompt)
        env = self._prepare_environment()

        update_state(
            worker_id,
            status="applying",
            job_title=job["title"],
            company=job.get("site", ""),
            score=job.get("fit_score", 0),
            start_time=time.time(),
            actions=0,
            last_action="starting (opencode)",
        )
        add_event(f"[W{worker_id}] Starting (opencode): {job['title'][:40]} @ {job.get('site', '')}")
        
        # Let user know initialization takes time
        add_event(f"[W{worker_id}] ⏳ OpenCode initializing (15-20s)...")

        worker_log = config.LOG_DIR / f"worker-{worker_id}.log"

        # Set up JSONL audit file
        audit_path = config.LOG_DIR / f"worker-{worker_id}.events.jsonl"
        audit_file = open(audit_path, "a", encoding="utf-8")

        # Clean up old audit files (30-day retention)
        cutoff = datetime.utcnow() - timedelta(days=30)
        for old_file in config.LOG_DIR.glob("worker-*.events.jsonl"):
            try:
                if datetime.utcfromtimestamp(old_file.stat().st_mtime) < cutoff:
                    old_file.unlink()
            except Exception:
                pass

        ts_header = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_header = (
            f"\n{'=' * 60}\n"
            f"[{ts_header}] [opencode] {job['title']} @ {job.get('site', '')}\n"
            f"URL: {job.get('application_url') or job['url']}\n"
            f"Score: {job.get('fit_score', 'N/A')}/10\n"
            f"{'=' * 60}\n"
        )

        start = time.time()
        stats: dict = {}
        proc = None

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                cwd=str(config.APP_DIR),
            )
            self._active_procs[worker_id] = proc
            if proc.stdin is None or proc.stdout is None:
                raise BackendError("OpenCode backend process streams unavailable")
            stdin = proc.stdin
            stdout = proc.stdout

            # Prompt passed as positional arg to opencode, not via stdin
            stdin.close()

            text_parts: list[str] = []
            with open(worker_log, "a", encoding="utf-8") as lf:
                lf.write(log_header)

                # Instrumentation for timing analysis
                last_msg_time = None
                tool_start_time = None
                current_tool = None
                turn_count = 0
                perf_log_path = config.LOG_DIR / f"worker-{worker_id}.perf.jsonl"
                perf_file = open(perf_log_path, "a", encoding="utf-8")
                
                for line in stdout:
                    line = line.strip()
                    if not line:
                        continue
                    
                    msg_received_time = time.time()
                    
                    # Calculate gap from last message
                    if last_msg_time:
                        gap_ms = (msg_received_time - last_msg_time) * 1000
                    else:
                        gap_ms = 0
                    
                    try:
                        msg = json.loads(line)
                        msg_type = msg.get("type")
                        
                        # Log message timing
                        if gap_ms > 100:  # Only log significant gaps (>100ms)
                            perf_entry = {
                                "timestamp": datetime.utcnow().isoformat(),
                                "event": "message_received",
                                "msg_type": msg_type,
                                "gap_ms": round(gap_ms, 2),
                                "turn": turn_count,
                            }
                            perf_file.write(json.dumps(perf_entry) + "\n")
                            perf_file.flush()
                        
                        # Track step start
                        if msg_type == "step_start":
                            turn_count += 1
                            step_start_time = time.time()
                            if turn_count == 1:
                                add_event(f"[W{worker_id}] ✓ Agent active - starting application...")
                            perf_entry = {
                                "timestamp": datetime.utcnow().isoformat(),
                                "event": "llm_turn_start",
                                "turn": turn_count,
                            }
                            perf_file.write(json.dumps(perf_entry) + "\n")
                            perf_file.flush()
                            
                            # Log if this was a long wait
                            if gap_ms > 5000:  # >5 seconds
                                lf.write(f"\n[PERF] Long gap before turn {turn_count}: {gap_ms/1000:.1f}s (likely API latency + LLM thinking)\n")
                                lf.flush()
                        
                        # Track tool execution
                        if msg_type == "tool_use":
                            tool_start_time = time.time()
                            # Extract tool name (simplified)
                            part = msg.get("part", {})
                            current_tool = part.get("tool", part.get("name", "unknown"))

                        if msg_type == "text":
                            text_content = msg.get("part", {}).get("text", "")
                            if text_content:
                                text_parts.append(text_content)
                                lf.write(text_content + "\n")

                        elif msg_type == "tool_use":
                            # Try multiple locations for tool name and input
                            # OpenCode may output these at top level or in "part" sub-object
                            name = msg.get("name", "")
                            if not name:
                                part = msg.get("part", {})
                                name = part.get("name", "")
                            if not name:
                                part = msg.get("part", {})
                                name = part.get("tool", "")  # OpenCode uses 'tool' field
                            inp = msg.get("input", {})
                            if not inp:
                                part = msg.get("part", {})
                                inp = part.get("input", {})
                            if not inp:
                                part = msg.get("part", {})
                                state = part.get("state", {})
                                inp = state.get("input", {})  # OpenCode nests input in state
                            # Clean up MCP prefixes
                            name = name.replace("mcp__playwright__", "").replace("mcp__gmail__", "gmail:")

                            # Debug: log structure to help diagnose format issues
                            if not name:
                                logger.debug(f"Tool use message has empty name. Keys: {list(msg.keys())}")
                            if "url" in inp:
                                desc = f"{name} {inp['url'][:60]}"
                            elif "ref" in inp:
                                desc = (f"{name} {inp.get('element', inp.get('text', ''))}")[:50]
                            elif "fields" in inp:
                                desc = f"{name} ({len(inp['fields'])} fields)"
                            elif "paths" in inp:
                                desc = f"{name} upload"
                            else:
                                desc = name

                            # Write enhanced log with timestamp
                            ts = datetime.utcnow().isoformat()
                            lf.write(f"{ts} >> {desc}\n")

                            # Add dashboard event
                            add_event(f"[W{worker_id}] {desc[:80]}")

                            # Update state
                            ws = get_state(worker_id)
                            cur_actions = ws.actions if ws else 0
                            update_state(
                                worker_id,
                                actions=cur_actions + 1,
                                last_action=desc[:35],
                            )

                            # Calculate tool timing if we have start time
                            tool_exec_time = None
                            if tool_start_time:
                                tool_exec_time = time.time() - tool_start_time
                            
                            # Write JSONL audit entry
                            audit_entry = {
                                "timestamp": ts,
                                "worker_id": worker_id,
                                "type": "tool_use",
                                "tool": name,
                                "input": inp,  # Full input payload
                                "tab_id": inp.get("tab_id"),
                                "url": inp.get("url"),
                                "tool_exec_time": round(tool_exec_time, 3) if tool_exec_time else None,
                                "turn": turn_count,
                            }
                            audit_file.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
                            audit_file.flush()
                            
                            # Log performance data
                            if tool_exec_time:
                                perf_entry = {
                                    "timestamp": ts,
                                    "event": "tool_executed",
                                    "tool": name,
                                    "exec_time_ms": round(tool_exec_time * 1000, 2),
                                    "turn": turn_count,
                                }
                                perf_file.write(json.dumps(perf_entry) + "\n")
                                perf_file.flush()

                        elif msg_type == "step_finish":
                            step_end_time = time.time()
                            part = msg.get("part", {})
                            tokens = part.get("tokens", {})
                            cache = tokens.get("cache", {})
                            
                            # Log turn completion
                            perf_entry = {
                                "timestamp": datetime.utcnow().isoformat(),
                                "event": "llm_turn_complete",
                                "turn": turn_count,
                                "input_tokens": tokens.get("input", 0),
                                "output_tokens": tokens.get("output", 0),
                            }
                            perf_file.write(json.dumps(perf_entry) + "\n")
                            perf_file.flush()
                            stats = {
                                "input_tokens": tokens.get("input", 0),
                                "output_tokens": tokens.get("output", 0),
                                "cache_read": cache.get("read", 0),
                                "cache_create": cache.get("write", 0),
                                "cost_usd": part.get("cost", 0),
                                "turns": 1,
                            }

                    except json.JSONDecodeError:
                        text_parts.append(line)
                        lf.write(line + "\n")

            proc.wait(timeout=300)
            returncode = proc.returncode
            proc = None

            if returncode and returncode < 0:
                return "skipped", int((time.time() - start) * 1000)

            output = "\n".join(text_parts)
            elapsed = int(time.time() - start)
            duration_ms = int((time.time() - start) * 1000)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_log = config.LOG_DIR / f"opencode_{ts}_w{worker_id}_{job.get('site', 'unknown')[:20]}.txt"
            job_log.write_text(output, encoding="utf-8")

            if stats:
                cost = stats.get("cost_usd", 0)
                ws = get_state(worker_id)
                prev_cost = ws.total_cost if ws else 0.0
                update_state(worker_id, total_cost=prev_cost + cost)

            def _clean_reason(s: str) -> str:
                return re.sub(r'[*`"]+$', "", s).strip()

            for result_status in ["APPLIED", "EXPIRED", "CAPTCHA", "LOGIN_ISSUE"]:
                if f"RESULT:{result_status}" in output:
                    add_event(f"[W{worker_id}] {result_status} ({elapsed}s): {job['title'][:30]}")
                    update_state(
                        worker_id,
                        status=result_status.lower(),
                        last_action=f"{result_status} ({elapsed}s)",
                    )
                    return result_status.lower(), duration_ms

            if "RESULT:FAILED" in output:
                for out_line in output.split("\n"):
                    if "RESULT:FAILED" in out_line:
                        reason = (
                            out_line.split("RESULT:FAILED:")[-1].strip()
                            if ":" in out_line[out_line.index("FAILED") + 6 :]
                            else "unknown"
                        )
                        reason = _clean_reason(reason)
                        PROMOTE_TO_STATUS = {"captcha", "expired", "login_issue"}
                        if reason in PROMOTE_TO_STATUS:
                            add_event(f"[W{worker_id}] {reason.upper()} ({elapsed}s): {job['title'][:30]}")
                            update_state(
                                worker_id,
                                status=reason,
                                last_action=f"{reason.upper()} ({elapsed}s)",
                            )
                            return reason, duration_ms
                        add_event(f"[W{worker_id}] FAILED ({elapsed}s): {reason[:30]}")
                        update_state(
                            worker_id,
                            status="failed",
                            last_action=f"FAILED: {reason[:25]}",
                        )
                        return f"failed:{reason}", duration_ms
                return "failed:unknown", duration_ms

            add_event(f"[W{worker_id}] NO RESULT ({elapsed}s)")
            update_state(worker_id, status="failed", last_action=f"no result ({elapsed}s)")
            return "failed:no_result_line", duration_ms

        except BackendError:
            # Re-raise backend errors (e.g. missing binary) without wrapping
            raise
        except subprocess.TimeoutExpired:
            duration_ms = int((time.time() - start) * 1000)
            elapsed = int(time.time() - start)
            add_event(f"[W{worker_id}] TIMEOUT ({elapsed}s)")
            update_state(worker_id, status="failed", last_action=f"TIMEOUT ({elapsed}s)")
            return "failed:timeout", duration_ms
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            add_event(f"[W{worker_id}] ERROR: {str(e)[:40]}")
            update_state(
                worker_id,
                status="failed",
                last_action=f"ERROR: {str(e)[:25]}",
            )
            return f"failed:{str(e)[:100]}", duration_ms
        finally:
            audit_file.close()
            self._active_procs.pop(worker_id, None)
            if proc is not None and proc.poll() is None:
                self._kill_process_tree(proc.pid)

    def _kill_process_tree(self, pid: int) -> None:
        """Kill a process and all its children."""
        import signal

        try:
            import psutil

            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            _, alive = psutil.wait_procs([parent], timeout=3)
            for p in alive:
                p.kill()
        except ImportError:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass

    def get_active_proc(self, worker_id: int) -> subprocess.Popen | None:
        """Get the active process for a worker (for signal handling)."""
        return self._active_procs.get(worker_id)

    def kill_all(self) -> None:
        """Kill all active OpenCode processes."""
        for worker_id, proc in list(self._active_procs.items()):
            if proc.poll() is None:
                self._kill_process_tree(proc.pid)
        self._active_procs.clear()


def get_backend(backend_name: str | None = None) -> AgentBackend:
    """Factory function to get the appropriate backend instance.
    Args:
        backend_name: Backend identifier ("claude", "opencode", or None for default).
                     Reads from APPLY_BACKEND env var if not provided.
        An AgentBackend instance.
        InvalidBackendError: If the backend identifier is not supported.
    """
    backend_name = resolve_backend_name(backend_name)
    if backend_name not in VALID_BACKENDS:
        raise InvalidBackendError(backend_name, VALID_BACKENDS)
    if backend_name == "claude":
        return ClaudeBackend()
    if backend_name == "opencode":
        return OpenCodeBackend()
    # This should never happen due to the check above
    raise InvalidBackendError(backend_name, VALID_BACKENDS)


def get_available_backends() -> frozenset[str]:
    """Return the set of available backend identifiers."""
    return VALID_BACKENDS


def resolve_backend_name(backend_name: str | None = None) -> str:
    if backend_name is None:
        backend_name = os.environ.get("APPLY_BACKEND", DEFAULT_BACKEND)
    return backend_name.lower().strip()


def resolve_default_model(backend_name: str) -> str:
    if backend_name == "opencode":
        return os.environ.get("APPLY_OPENCODE_MODEL") or os.environ.get("LLM_MODEL") or DEFAULT_OPENCODE_MODEL
    return os.environ.get("APPLY_CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL


def resolve_default_agent(backend_name: str) -> str | None:
    if backend_name == "opencode":
        # Default to "applypilot-apply" agent defined in opencode.jsonc, unless overridden by env var
        return os.environ.get("APPLY_OPENCODE_AGENT") or "applypilot-apply"
    return None
