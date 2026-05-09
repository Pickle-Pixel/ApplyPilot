"""Apply orchestration: acquire jobs, spawn Claude Code sessions, track results.

This is the main entry point for the apply pipeline. It pulls jobs from
the database, launches Chrome + Claude Code for each one, parses the
result, and updates the database. Supports parallel workers via --workers.
"""

import atexit
import json
import logging
import os
import platform
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.live import Live

from applypilot import config
from applypilot.database import get_connection
from applypilot.apply import prompt as prompt_mod
from applypilot.apply.chrome import (
    launch_chrome,
    cleanup_worker,
    kill_all_chrome,
    reset_worker_dir,
    cleanup_on_exit,
    _kill_process_tree,
    BASE_CDP_PORT,
)
from applypilot.apply.dashboard import (
    init_worker,
    update_state,
    add_event,
    get_state,
    render_full,
    get_totals,
)

logger = logging.getLogger(__name__)


ApplyResultLike = str
RESULT_TOKEN_MARKERS = {
    "APPLIED": "applied",
    "DRY_RUN_READY": "dry_run_ready",
    "EXPIRED": "expired",
    "CAPTCHA": "captcha",
    "LOGIN_ISSUE": "login_issue",
}
RESULT_FAILED_TOKEN = "FAILED"


def _normalize_failure_reason(raw_reason: str) -> str:
    """Normalize free-form failure detail into a DB-safe reason token."""
    if not raw_reason:
        return "unknown"
    normalized = raw_reason.strip().lower()
    normalized = normalized.split("\n", 1)[0].strip()
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_:-]", "", normalized)
    return normalized or "unknown"


def _extract_result_token(output: str) -> str | None:
    # Example inputs:
    # - RESULT:APPLIED
    # - RESULT:FAILED
    # - RESULT:FAILED: reason text
    match = re.search(r"(?im)^[ \t]*RESULT:\s*([A-Za-z0-9_]+)(?::\s*(.*))?\s*$", output)
    if not match:
        return None
    token = match.group(1).upper()
    if token == RESULT_FAILED_TOKEN and match.group(2):
        return f"{token}:{match.group(2).strip()}"
    return token

# Blocked sites loaded from config/sites.yaml
def _load_blocked():
    from applypilot.config import load_blocked_sites

    return load_blocked_sites()


# How often to poll the DB when the queue is empty (seconds)
POLL_INTERVAL = config.DEFAULTS["poll_interval"]

# Thread-safe shutdown coordination
_stop_event = threading.Event()

# Track active Claude Code processes for skip (Ctrl+C) handling
_claude_procs: dict[int, subprocess.Popen] = {}
_claude_lock = threading.Lock()

# Register cleanup on exit
atexit.register(cleanup_on_exit)
if platform.system() != "Windows":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


# ---------------------------------------------------------------------------
# MCP config
# ---------------------------------------------------------------------------


def _make_mcp_config(cdp_port: int) -> dict:
    """Build MCP config dict for a specific CDP port."""
    return {
        "mcpServers": {
            "playwright": {
                "command": "npx",
                "args": [
                    "-y",
                    "@playwright/mcp@latest",
                    f"--cdp-endpoint=http://localhost:{cdp_port}",
                    f"--viewport-size={config.DEFAULTS['viewport']}",
                ],
            },
        }
    }


def _build_claude_command(model: str, mcp_config_path: Path) -> list[str]:
    """Build the Claude command for apply workers without blanket bypass."""
    return [
        "claude",
        "--model", model,
        "-p",
        "--mcp-config", str(mcp_config_path),
        "--no-session-persistence",
        "--output-format", "stream-json",
        "--verbose", "-",
    ]


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------


def acquire_job(target_url: str | None = None, min_score: int = 7, worker_id: int = 0) -> dict | None:
    """Atomically acquire the next job to apply to.

    Args:
        target_url: Apply to a specific URL instead of picking from queue.
        min_score: Minimum fit_score threshold.
        worker_id: Worker claiming this job (for tracking).

    Returns:
        Job dict or None if the queue is empty.
    """
    import sqlite3 as _sqlite3

    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        blocked_sites, blocked_patterns = _load_blocked()
        lane_filter = "AND COALESCE(job_lane, 'primary') = 'primary'"
        readiness_filters = """
            AND tailored_resume_path IS NOT NULL
            AND tailored_resume_path <> ''
            AND cover_letter_path IS NOT NULL
            AND cover_letter_path <> ''
            AND fit_score >= ?
            AND (COALESCE(apply_status, '') = ''
                 OR apply_status = 'failed'
                 OR apply_status LIKE 'failed:%')
            AND (apply_attempts IS NULL OR apply_attempts < ?)
        """
        blocked_site_clause = ""
        params_suffix: list = []
        if blocked_sites:
            placeholders = ",".join("?" * len(blocked_sites))
            blocked_site_clause = f" AND site NOT IN ({placeholders})"
            params_suffix.extend(blocked_sites)
        blocked_url_clause = ""
        if blocked_patterns:
            blocked_url_clause = " ".join("AND url NOT LIKE ?" for _ in blocked_patterns)
            params_suffix.extend(blocked_patterns)

        if target_url:
            norm_url = target_url.split('?')[0].rstrip('/')
            row = conn.execute("""
                SELECT url, title, site, company, application_url, tailored_resume_path,
                       fit_score, location, full_description, cover_letter_path
                FROM jobs
                WHERE (url = ? OR application_url = ? OR url = ? OR application_url = ?)
                  {lane_filter}
                  {readiness_filters}
                  {blocked_site_clause}
                  {blocked_url_clause}
                LIMIT 1
            """.format(
                lane_filter=lane_filter,
                readiness_filters=readiness_filters,
                blocked_site_clause=blocked_site_clause,
                blocked_url_clause=blocked_url_clause,
            ), (
                target_url,
                target_url,
                norm_url,
                norm_url,
                min_score,
                config.DEFAULTS["max_apply_attempts"],
                *params_suffix,
            )).fetchone()
        else:
            row = conn.execute(f"""
                SELECT url, title, site, company, application_url, tailored_resume_path,
                       fit_score, location, full_description, cover_letter_path
                FROM jobs
                WHERE 1=1
                  {lane_filter}
                  {readiness_filters}
                  {blocked_site_clause}
                  {blocked_url_clause}
                ORDER BY fit_score DESC, url
                LIMIT 1
            """.format(
                lane_filter=lane_filter,
                readiness_filters=readiness_filters,
                blocked_site_clause=blocked_site_clause,
                blocked_url_clause=blocked_url_clause,
            ), [min_score, config.DEFAULTS["max_apply_attempts"]] + params_suffix).fetchone()

        if not row:
            conn.rollback()
            return None

        # Hard-exclude AVOID-list companies. Belt-and-suspenders on top of the
        # ingest-time filter and the pack-export filter — last line of defense
        # before an actual (or dry-run) submission.
        from applypilot.config import is_avoid_company, is_manual_ats
        candidate_name = (row["company"] if "company" in row.keys() else None) or row["site"] or ""
        if candidate_name and is_avoid_company(candidate_name):
            conn.execute(
                "UPDATE jobs SET apply_status = 'avoid', apply_error = 'AVOID list' WHERE url = ?",
                (row["url"],),
            )
            conn.commit()
            logger.info("Skipping AVOID-list company: %s", candidate_name[:80])
            return None

        # Skip manual ATS sites (unsolvable CAPTCHAs)
        apply_url = row["application_url"] or row["url"]
        if is_manual_ats(apply_url):
            conn.execute(
                "UPDATE jobs SET apply_status = 'manual', apply_error = 'manual ATS' WHERE url = ?",
                (row["url"],),
            )
            conn.commit()
            logger.info("Skipping manual ATS: %s", row["url"][:80])
            return None

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            UPDATE jobs SET apply_status = 'in_progress',
                           agent_id = ?,
                           last_attempted_at = ?
            WHERE url = ?
        """,
            (f"worker-{worker_id}", now, row["url"]),
        )
        conn.commit()

        return dict(row)
    except _sqlite3.OperationalError as e:
        conn.rollback()
        if "locked" in str(e).lower():
            logger.warning("DB locked in acquire_job, retrying after backoff...")
            time.sleep(2)
            return acquire_job(target_url, min_score, worker_id)
        raise
    except Exception:
        conn.rollback()
        raise


def mark_result(
    url: str,
    status: str,
    error: str | None = None,
    permanent: bool = False,
    duration_ms: int | None = None,
    task_id: str | None = None,
) -> None:
    """Update a job's apply status in the database."""
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    if status == "applied":
        conn.execute(
            """
            UPDATE jobs SET apply_status = 'applied', applied_at = ?,
                           apply_error = NULL, agent_id = NULL,
                           apply_duration_ms = ?, apply_task_id = ?
            WHERE url = ?
        """,
            (now, duration_ms, task_id, url),
        )
    else:
        if permanent:
            conn.execute(
                """
                UPDATE jobs SET apply_status = ?, apply_error = ?,
                               apply_attempts = 99, agent_id = NULL,
                               apply_duration_ms = ?, apply_task_id = ?
                WHERE url = ?
            """,
                (status, error or "unknown", duration_ms, task_id, url),
            )
        else:
            conn.execute(
                """
                UPDATE jobs SET apply_status = ?, apply_error = ?,
                               apply_attempts = COALESCE(apply_attempts, 0) + 1, agent_id = NULL,
                               apply_duration_ms = ?, apply_task_id = ?
                WHERE url = ?
            """,
                (status, error or "unknown", duration_ms, task_id, url),
            )
    conn.commit()


def release_lock(url: str, release_reason: str | None = None) -> None:
    """Release the in_progress lock without changing status."""
    conn = get_connection()
    if release_reason:
        conn.execute(
            """
            UPDATE jobs
            SET agent_id = NULL,
                apply_error = COALESCE(?, apply_error)
            WHERE url = ? AND apply_status = 'in_progress'
            """,
            (release_reason, url),
        )
    else:
        conn.execute(
            """
            UPDATE jobs
            SET agent_id = NULL
            WHERE url = ? AND apply_status = 'in_progress'
            """,
            (url,),
        )
    conn.commit()


def release_generated_prompt_lock(url: str) -> None:
    """Undo the temporary claim taken only to generate a manual prompt."""
    conn = get_connection()
    conn.execute(
        """
        UPDATE jobs
        SET apply_status = NULL,
            apply_error = NULL,
            agent_id = NULL
        WHERE url = ? AND apply_status = 'in_progress'
        """,
        (url,),
    )
    conn.commit()


def release_stale_locks(
    max_age_minutes: int = 60,
    status: str = "manual_review",
    reason: str = "stale_in_progress_orphaned",
) -> int:
    """Move old in-progress rows to manual review instead of silently retrying."""
    conn = get_connection()
    cursor = conn.execute("""
        UPDATE jobs
        SET apply_status = ?,
            apply_error = ?,
            agent_id = NULL
        WHERE apply_status = 'in_progress'
          AND datetime(last_attempted_at) < datetime('now', ?)
    """, (status, reason, f"-{max_age_minutes} minutes"))
    conn.commit()
    return cursor.rowcount


# ---------------------------------------------------------------------------
# Utility modes (--gen, --mark-applied, --mark-failed, --reset-failed)
# ---------------------------------------------------------------------------


def gen_prompt(target_url: str, min_score: int = 7, model: str = "sonnet", worker_id: int = 0) -> Path | None:
    """Generate a prompt file and print the Claude CLI command for manual debugging.

    Returns:
        Path to the generated prompt file, or None if no job found.
    """
    job = acquire_job(target_url=target_url, min_score=min_score, worker_id=worker_id)
    if not job:
        return None

    # Read resume text
    resume_path = job.get("tailored_resume_path")
    txt_path = Path(resume_path).with_suffix(".txt") if resume_path else None
    resume_text = ""
    if txt_path and txt_path.exists():
        resume_text = txt_path.read_text(encoding="utf-8")

    prompt = prompt_mod.build_prompt(job=job, tailored_resume=resume_text, dry_run=True)

    # Release the lock so the job stays available
    release_generated_prompt_lock(job["url"])

    # Write prompt file
    config.ensure_dirs()
    site_slug = (job.get("site") or "unknown")[:20].replace(" ", "_")
    prompt_file = config.LOG_DIR / f"prompt_{site_slug}_{job['title'][:30].replace(' ', '_')}.txt"
    prompt_file.write_text(prompt, encoding="utf-8")

    # Write MCP config for reference
    port = BASE_CDP_PORT + worker_id
    mcp_path = config.APP_DIR / f".mcp-apply-{worker_id}.json"
    mcp_path.write_text(json.dumps(_make_mcp_config(port)), encoding="utf-8")

    return prompt_file


def mark_job(url: str, status: str, reason: str | None = None) -> None:
    """Manually mark a job's apply status in the database.

    Args:
        url: Job URL to mark.
        status: Either 'applied' or 'failed'.
        reason: Failure reason (only for status='failed').
    """
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    if status == "applied":
        conn.execute(
            """
            UPDATE jobs SET apply_status = 'applied', applied_at = ?,
                           apply_error = NULL, agent_id = NULL
            WHERE url = ?
        """,
            (now, url),
        )
    else:
        conn.execute(
            """
            UPDATE jobs SET apply_status = 'failed', apply_error = ?,
                           apply_attempts = 99, agent_id = NULL
            WHERE url = ?
        """,
            (reason or "manual", url),
        )
    conn.commit()


def parse_apply_result(output: str) -> str:
    """Classify Claude apply output into ApplyPilot status strings."""
    if not output:
        return "failed:no_result_line"

    lower = output.lower()
    result_marker = _extract_result_token(output)
    if result_marker:
        marker = result_marker.upper()
        for token, normalized in RESULT_TOKEN_MARKERS.items():
            if marker == token:
                return normalized
        if marker.startswith(RESULT_FAILED_TOKEN):
            if marker == RESULT_FAILED_TOKEN:
                raw_reason = "unknown"
            else:
                raw_reason = result_marker.split(":", 1)[-1]
            raw_reason = raw_reason.strip() if raw_reason else "unknown"
            if not raw_reason:
                raw_reason = "unknown"
            reason = _normalize_failure_reason(raw_reason or "unknown")
            return f"failed:{reason}"

    submitted_markers = [
        "application successfully submitted",
        "application has been completed and submitted",
        "thank you for completing",
        "thank you for applying",
        "/apply/thanks",
        "appreciate your interest in working with our company",
    ]
    if any(marker in lower for marker in submitted_markers):
        return "applied"

    expired_markers = [
        "no longer accepting applications",
        "no longer open",
        "no longer available",
        "page you are looking for doesn't exist",
        "page you are looking for does not exist",
        "job posting is no longer available",
        "job posting has expired",
    ]
    if any(marker in lower for marker in expired_markers):
        return "expired"

    manual_approval_markers = [
        "manual review required",
        "requires manual review",
        "needs manual follow-up",
        "manual intervention",
        "requires human review",
        "manual review",
    ]
    if any(marker in lower for marker in manual_approval_markers):
        return "failed:manual_review_required"

    captcha_markers = ["captcha", "recaptcha", "hcaptcha", "turnstile", "cloudflare"]
    if any(marker in lower for marker in captcha_markers):
        return "captcha"

    login_issue_markers = [
        "could not sign in",
        "unable to sign in",
        "could not log in",
        "unable to log in",
        "please log in to",
        "please sign in to",
        "login required",
        "sign in required",
        "must log in",
        "must sign in",
        "sign in to",
        "login to",
        "log in to",
        "cannot log in",
        "can't log in",
        "account setup",
        "create an account",
    ]
    if any(marker in lower for marker in login_issue_markers):
        return "login_issue"

    browser_lost_markers = [
        "localhost:9222 is being refused",
        "browser backend",
        "connection refused",
        "cdp",
    ]
    if any(marker in lower for marker in browser_lost_markers):
        return "failed:browser_backend_lost"

    return "failed:no_result_line"


def reset_failed() -> int:
    """Reset all failed jobs so they can be retried.

    Returns:
        Number of jobs reset.
    """
    conn = get_connection()
    cursor = conn.execute("""
        UPDATE jobs SET apply_status = NULL, apply_error = NULL,
                       apply_attempts = 0, agent_id = NULL
        WHERE apply_status = 'failed'
           OR apply_status LIKE 'failed:%'
    """)
    conn.commit()
    return cursor.rowcount


# ---------------------------------------------------------------------------
# Per-job execution
# ---------------------------------------------------------------------------


def run_job(job: dict, port: int, worker_id: int = 0, model: str = "sonnet", dry_run: bool = False) -> tuple[str, int]:
    """Spawn a Claude Code session for one job application.

    Returns:
        Tuple of (status_string, duration_ms). Status is one of:
        'applied', 'dry_run_ready', 'expired', 'captcha', 'login_issue',
        'failed:<reason>'.
    """
    # Read tailored resume text
    resume_path = job.get("tailored_resume_path")
    txt_path = Path(resume_path).with_suffix(".txt") if resume_path else None
    resume_text = ""
    if txt_path and txt_path.exists():
        resume_text = txt_path.read_text(encoding="utf-8")

    # Build the prompt
    agent_prompt = prompt_mod.build_prompt(
        job=job,
        tailored_resume=resume_text,
        dry_run=dry_run,
    )

    # Write per-worker MCP config
    mcp_config_path = config.APP_DIR / f".mcp-apply-{worker_id}.json"
    mcp_config_path.write_text(json.dumps(_make_mcp_config(port)), encoding="utf-8")

    cmd = _build_claude_command(model=model, mcp_config_path=mcp_config_path)

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)

    profile = config.load_profile()
    pwd = profile.get("personal", {}).get("password", "")
    if pwd:
        env["APPLYPILOT_SITE_PASSWORD"] = pwd

    worker_dir = reset_worker_dir(worker_id)

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
        with _claude_lock:
            _claude_procs[worker_id] = proc

        proc.stdin.write(agent_prompt)
        proc.stdin.close()

        text_parts: list[str] = []
        with open(worker_log, "a", encoding="utf-8") as lf:
            lf.write(log_header)

            line_queue: queue.Queue[str | None] = queue.Queue()

            def _read_stdout() -> None:
                try:
                    if proc.stdout is not None:
                        for raw_line in proc.stdout:
                            line_queue.put(raw_line)
                finally:
                    line_queue.put(None)

            reader = threading.Thread(target=_read_stdout, daemon=True)
            reader.start()
            deadline = start + config.DEFAULTS["apply_timeout"]

            while True:
                if time.time() > deadline:
                    raise subprocess.TimeoutExpired(cmd, config.DEFAULTS["apply_timeout"])
                try:
                    line = line_queue.get(timeout=0.5)
                except queue.Empty:
                    if proc.poll() is not None and not reader.is_alive():
                        break
                    continue
                if line is None:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    msg_type = msg.get("type")
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

                                lf.write(f"  >> {desc}\n")
                                ws = get_state(worker_id)
                                cur_actions = ws.actions if ws else 0
                                update_state(worker_id, actions=cur_actions + 1, last_action=desc[:35])
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

        proc.wait(timeout=10)
        returncode = proc.returncode
        proc = None

        if returncode and returncode < 0:
            return "failed:launcher_terminated", int((time.time() - start) * 1000)

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

        result = parse_apply_result(output)
        if result in {"applied", "expired", "captcha", "login_issue", "dry_run_ready"}:
            add_event(f"[W{worker_id}] {result.upper()} ({elapsed}s): {job['title'][:30]}")
            update_state(worker_id, status=result, last_action=f"{result.upper()} ({elapsed}s)")
        else:
            reason = result.split(":", 1)[-1] if ":" in result else result
            add_event(f"[W{worker_id}] FAILED ({elapsed}s): {reason[:30]}")
            update_state(worker_id, status="failed", last_action=f"FAILED: {reason[:25]}")
        return result, duration_ms

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
        with _claude_lock:
            _claude_procs.pop(worker_id, None)
        if proc is not None and proc.poll() is None:
            _kill_process_tree(proc.pid)


# ---------------------------------------------------------------------------
# Permanent failure classification
# ---------------------------------------------------------------------------

PERMANENT_FAILURES: set[str] = {
    "expired", "captcha", "login_issue", "browser_backend_lost",
    "not_eligible_location", "not_eligible_salary",
    "already_applied", "account_required",
    "not_a_job_application", "unsafe_permissions",
    "unsafe_verification", "sso_required",
    "site_blocked", "cloudflare_blocked", "blocked_by_cloudflare",
    "manual_review_required", "page_error", "stale_in_progress_orphaned",
}

PERMANENT_PREFIXES: tuple[str, ...] = ("site_blocked", "cloudflare", "blocked_by")


def _is_permanent_failure(result: str) -> bool:
    """Determine if a failure should never be retried."""
    reason = result.split(":", 1)[-1] if ":" in result else result
    return (
        result in PERMANENT_FAILURES
        or reason in PERMANENT_FAILURES
        or any(reason.startswith(p) for p in PERMANENT_PREFIXES)
    )


def _to_db_state(parsed_result: ApplyResultLike) -> tuple[str, str | None, bool]:
    """Map parser output to DB status, error, and permanence."""
    if parsed_result == "applied":
        return "applied", None, False
    if parsed_result == "dry_run_ready":
        return "dry_run_ready", None, False
    if parsed_result in {"expired", "captcha", "login_issue", "browser_backend_lost"}:
        return parsed_result, parsed_result, True
    if parsed_result.startswith("failed:"):
        reason = _normalize_failure_reason(parsed_result.split(":", 1)[-1])
        return "failed", reason, _is_permanent_failure(reason) or _is_permanent_failure(parsed_result)
    return "failed", _normalize_failure_reason(parsed_result), _is_permanent_failure(parsed_result)


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------


def worker_loop(
    worker_id: int = 0,
    limit: int = 1,
    target_url: str | None = None,
    min_score: int = 7,
    headless: bool = False,
    model: str = "sonnet",
    dry_run: bool = False,
) -> tuple[int, int]:
    """Run jobs sequentially until limit is reached or queue is empty.

    Args:
        worker_id: Numeric worker identifier.
        limit: Max jobs to process (0 = continuous).
        target_url: Apply to a specific URL.
        min_score: Minimum fit_score threshold.
        headless: Run Chrome headless.
        model: Claude model name.
        dry_run: Don't click Submit.

    Returns:
        Tuple of (applied_count, failed_count).
    """
    applied = 0
    failed = 0
    continuous = limit == 0
    jobs_done = 0
    empty_polls = 0
    port = BASE_CDP_PORT + worker_id

    while not _stop_event.is_set():
        if not continuous and jobs_done >= limit:
            break

        update_state(worker_id, status="idle", job_title="", company="", last_action="waiting for job", actions=0)

        job = acquire_job(target_url=target_url, min_score=min_score, worker_id=worker_id)
        if not job:
            if not continuous:
                add_event(f"[W{worker_id}] Queue empty")
                update_state(worker_id, status="done", last_action="queue empty")
                break
            empty_polls += 1
            update_state(worker_id, status="idle", last_action=f"polling ({empty_polls})")
            if empty_polls == 1:
                add_event(f"[W{worker_id}] Queue empty, polling every {POLL_INTERVAL}s...")
            # Use Event.wait for interruptible sleep
            if _stop_event.wait(timeout=POLL_INTERVAL):
                break  # Stop was requested during wait
            continue

        empty_polls = 0

        chrome_proc = None
        try:
            add_event(f"[W{worker_id}] Launching Chrome...")
            chrome_proc = launch_chrome(worker_id, port=port, headless=headless)

            result, duration_ms = run_job(job, port=port, worker_id=worker_id,
                                            model=model, dry_run=dry_run)
            db_status, db_reason, db_permanent = _to_db_state(result)
            if dry_run and db_status == "applied":
                db_status = "dry_run_ready"
                db_reason = None
                db_permanent = False

            mark_result(
                job["url"],
                db_status,
                db_reason,
                permanent=db_permanent,
                duration_ms=duration_ms,
            )
            if db_status in {"applied", "dry_run_ready"}:
                applied += 1
                if dry_run and result == "applied":
                    add_event(f"[W{worker_id}] Dry run complete: {job['title'][:30]}")
                else:
                    add_event(f"[W{worker_id}] {result.upper()} ({(duration_ms // 1000)}s): {job['title'][:30]}")
                update_state(worker_id, jobs_applied=applied,
                             jobs_done=applied + failed)
            else:
                failed += 1
                add_event(f"[W{worker_id}] result: {result[:40]}")
                update_state(worker_id, jobs_failed=failed,
                             jobs_done=applied + failed)

        except KeyboardInterrupt:
            release_lock(job["url"], release_reason="worker_interrupt")
            if _stop_event.is_set():
                break
            add_event(f"[W{worker_id}] Job skipped (Ctrl+C)")
            continue
        except Exception as e:
            logger.exception("Worker %d launcher error", worker_id)
            add_event(f"[W{worker_id}] Launcher error: {str(e)[:40]}")
            mark_result(
                job["url"],
                "failed",
                _normalize_failure_reason(str(e)[:120]),
                permanent=False,
            )
            failed += 1
            update_state(worker_id, jobs_failed=failed)
        finally:
            if chrome_proc:
                cleanup_worker(worker_id, chrome_proc)

        jobs_done += 1
        if target_url:
            break

    update_state(worker_id, status="done", last_action="finished")
    return applied, failed


# ---------------------------------------------------------------------------
# Main entry point (called from cli.py)
# ---------------------------------------------------------------------------


def main(
    limit: int = 1,
    target_url: str | None = None,
    min_score: int = 7,
    headless: bool = False,
    model: str = "sonnet",
    dry_run: bool = False,
    continuous: bool = False,
    poll_interval: int = 60,
    workers: int = 1,
) -> None:
    """Launch the apply pipeline.

    Args:
        limit: Max jobs to apply to (0 or with continuous=True means run forever).
        target_url: Apply to a specific URL.
        min_score: Minimum fit_score threshold.
        headless: Run Chrome in headless mode.
        model: Claude model name.
        dry_run: Don't click Submit.
        continuous: Run forever, polling for new jobs.
        poll_interval: Seconds between DB polls when queue is empty.
        workers: Number of parallel workers (default 1).
    """
    global POLL_INTERVAL
    POLL_INTERVAL = poll_interval
    _stop_event.clear()

    config.ensure_dirs()
    stale_count = release_stale_locks(max_age_minutes=60)
    console = Console()

    if continuous:
        effective_limit = 0
        mode_label = "continuous"
    else:
        effective_limit = limit
        mode_label = f"{limit} jobs"

    # Initialize dashboard for all workers
    for i in range(workers):
        init_worker(i)

    worker_label = f"{workers} worker{'s' if workers > 1 else ''}"
    console.print(f"Launching apply pipeline ({mode_label}, {worker_label}, poll every {POLL_INTERVAL}s)...")
    if stale_count:
        console.print(f"[yellow]Moved {stale_count} stale in-progress job(s) to manual review.[/yellow]")
    console.print("[dim]Ctrl+C = skip current job(s) | Ctrl+C x2 = stop[/dim]")

    # Double Ctrl+C handler
    _ctrl_c_count = 0

    def _sigint_handler(sig, frame):
        nonlocal _ctrl_c_count
        _ctrl_c_count += 1
        if _ctrl_c_count == 1:
            console.print("\n[yellow]Skipping current job(s)... (Ctrl+C again to STOP)[/yellow]")
            # Kill all active Claude processes to skip current jobs
            with _claude_lock:
                for wid, cproc in list(_claude_procs.items()):
                    if cproc.poll() is None:
                        _kill_process_tree(cproc.pid)
        else:
            console.print("\n[red bold]STOPPING[/red bold]")
            _stop_event.set()
            with _claude_lock:
                for wid, cproc in list(_claude_procs.items()):
                    if cproc.poll() is None:
                        _kill_process_tree(cproc.pid)
            kill_all_chrome()
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        with Live(render_full(), console=console, refresh_per_second=2) as live:
            # Daemon thread for display refresh only (no business logic)
            _dashboard_running = True

            def _refresh():
                while _dashboard_running:
                    live.update(render_full())
                    time.sleep(0.5)

            refresh_thread = threading.Thread(target=_refresh, daemon=True)
            refresh_thread.start()

            if workers == 1:
                # Single worker — run directly in main thread
                total_applied, total_failed = worker_loop(
                    worker_id=0,
                    limit=effective_limit,
                    target_url=target_url,
                    min_score=min_score,
                    headless=headless,
                    model=model,
                    dry_run=dry_run,
                )
            else:
                # Multi-worker — distribute limit across workers
                if effective_limit:
                    base = effective_limit // workers
                    extra = effective_limit % workers
                    limits = [base + (1 if i < extra else 0) for i in range(workers)]
                else:
                    limits = [0] * workers  # continuous mode

                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="apply-worker") as executor:
                    futures = {
                        executor.submit(
                            worker_loop,
                            worker_id=i,
                            limit=limits[i],
                            target_url=target_url,
                            min_score=min_score,
                            headless=headless,
                            model=model,
                            dry_run=dry_run,
                        ): i
                        for i in range(workers)
                    }

                    results: list[tuple[int, int]] = []
                    for future in as_completed(futures):
                        wid = futures[future]
                        try:
                            results.append(future.result())
                        except Exception:
                            logger.exception("Worker %d crashed", wid)
                            results.append((0, 0))

                total_applied = sum(r[0] for r in results)
                total_failed = sum(r[1] for r in results)

            _dashboard_running = False
            refresh_thread.join(timeout=2)
            live.update(render_full())

        totals = get_totals()
        console.print(f"\n[bold]Done: {total_applied} applied, {total_failed} failed (${totals['cost']:.3f})[/bold]")
        console.print(f"Logs: {config.LOG_DIR}")

    except KeyboardInterrupt:
        pass
    finally:
        _stop_event.set()
        kill_all_chrome()
