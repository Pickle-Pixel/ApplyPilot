"""PURE fleet dead-man detector -- read-only watcher for the autonomous apply lane.

The owner removed the lifetime spend cap; the rolling ``cost_cap_daily_usd`` is the
only remaining throttle. This module is the safety net that notices when the fleet
goes silent, its queue stalls, its self-healer (Doctor/Watchdog) itself dies, or it
starts running hot against the daily cap.

``deadman_check`` is a PURE function: it issues SELECT-only queries against the given
connection, never writes/commits, and never calls ``datetime.now()`` -- the caller
(Task 2's persistent-loop wrapper) injects ``now`` so this module stays fully testable
and deterministic. The caller is also responsible for persisting ``new_hot_streak``
across invocations and passing it back in as ``prev_hot_streak``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import imaplib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from applypilot import config, inbox_auth
from applypilot.mail_source import ImapMailSource, MailSourceError, MailSourceOverflowError

# ---------------------------------------------------------------------------
# Thresholds (module constants -- tune here, not inline).
# ---------------------------------------------------------------------------
STALE_MIN = 30          # silent_death / selfheal_dead: heartbeat staleness (minutes)
STALL_HOURS = 3         # stalled_queue: how long without an 'applied' row counts as stalled
HOT_FRACTION = 0.95     # running_hot: fraction of cost_cap_daily_usd that counts as "hot"
HOT_STREAK_MIN = 2      # running_hot: consecutive hot checks required before alerting
OWNER_INBOX_WINDOW_MIN = 30
OWNER_INBOX_STALE_HOURS = 24
OTP_STALL_SECONDS_DEFAULT = 120
OTP_STALL_SECONDS_MIN = 30

_WATCHDOG_OR_LINKEDIN_RE = re.compile(r"watchdog|linkedin")
_WATCHDOG_RE = re.compile(r"watchdog")


@dataclass
class Alert:
    kind: str
    severity: str
    detail: str


def _fleet_config(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT paused, ats_paused, cost_cap_daily_usd, doctor_last_pass_at "
            "FROM fleet_config WHERE id=1;"
        )
        row = cur.fetchone()
    return dict(row) if row else {"paused": True, "ats_paused": True,
                                  "cost_cap_daily_usd": 0, "doctor_last_pass_at": None}


def _is_armed(cfg: dict) -> bool:
    return cfg["paused"] is False and cfg["ats_paused"] is False


def _max_last_beat(conn, pattern_sql: str, negate: bool) -> dt.datetime | None:
    op = "!~" if negate else "~"
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT MAX(last_beat) AS max_beat FROM worker_heartbeat WHERE worker_id {op} %s;",
            (pattern_sql,),
        )
        row = cur.fetchone()
    return row["max_beat"] if row else None


def _check_silent_death(conn, cfg: dict, now: dt.datetime) -> Alert | None:
    if not _is_armed(cfg):
        return None
    max_beat = _max_last_beat(conn, "watchdog|linkedin", negate=True)
    stale_before = now - dt.timedelta(minutes=STALE_MIN)
    if max_beat is None or max_beat < stale_before:
        detail = (
            "no apply-worker heartbeat" if max_beat is None
            else f"last apply-worker heartbeat at {max_beat.isoformat()}"
        )
        return Alert(kind="silent_death", severity="critical", detail=detail)
    return None


def _check_stalled_queue(conn, cfg: dict, now: dt.datetime) -> Alert | None:
    if not _is_armed(cfg):
        return None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM apply_queue WHERE status='queued' "
            "AND approved_batch IS NOT NULL) AS has_backlog;"
        )
        has_backlog = cur.fetchone()["has_backlog"]
    if not has_backlog:
        return None
    stall_before = now - dt.timedelta(hours=STALL_HOURS)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM apply_queue WHERE status='applied' "
            "AND updated_at > %s) AS has_recent_apply;",
            (stall_before,),
        )
        has_recent_apply = cur.fetchone()["has_recent_apply"]
    if has_recent_apply:
        return None
    return Alert(
        kind="stalled_queue", severity="critical",
        detail=f"approved backlog queued but no 'applied' row in the last {STALL_HOURS}h",
    )


def _check_selfheal_dead(conn, cfg: dict, now: dt.datetime) -> Alert | None:
    """Fires if EITHER self-healer is down. There are two: the Watchdog (Layer A
    deterministic reclaim/breakers -- beats worker_heartbeat worker_id='watchdog') AND the
    Fleet Doctor (the primary 5-min self-fixer -- stamps fleet_config.doctor_last_pass_at,
    doctor.py H18). Both must be alive; a dead self-healer is critical regardless of pause
    state (nothing heals a stuck fleet), so this is intentionally NOT armed-gated."""
    stale_before = now - dt.timedelta(minutes=STALE_MIN)
    down: list[str] = []
    wd_beat = _max_last_beat(conn, "watchdog", negate=False)
    if wd_beat is None or wd_beat < stale_before:
        down.append("watchdog (" + ("no heartbeat" if wd_beat is None else wd_beat.isoformat()) + ")")
    doc_pass = cfg.get("doctor_last_pass_at")
    if doc_pass is None or doc_pass < stale_before:
        down.append("doctor (" + ("no pass recorded" if doc_pass is None else doc_pass.isoformat()) + ")")
    if down:
        return Alert(kind="selfheal_dead", severity="critical",
                     detail="self-healer down: " + "; ".join(down))
    return None


def _check_running_hot(
    conn, cfg: dict, now: dt.datetime, prev_hot_streak: int
) -> tuple[Alert | None, int]:
    daily_cap = cfg["cost_cap_daily_usd"]
    if not daily_cap or daily_cap <= 0:
        return None, 0
    window_start = now - dt.timedelta(hours=24)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) AS spend FROM llm_usage WHERE ts >= %s;",
            (window_start,),
        )
        spend = cur.fetchone()["spend"]
    spend = float(spend)
    daily_cap = float(daily_cap)
    if spend >= HOT_FRACTION * daily_cap:
        streak = prev_hot_streak + 1
        if streak >= HOT_STREAK_MIN:
            return Alert(
                kind="running_hot", severity="warning",
                detail=f"rolling-24h spend ${spend:.2f} >= {HOT_FRACTION:.0%} of "
                       f"${daily_cap:.2f} daily cap (streak={streak})",
            ), streak
        return None, streak
    return None, 0


def _otp_stall_seconds() -> int:
    raw = os.environ.get("APPLYPILOT_OTP_STALL_SECONDS")
    if raw is None:
        return OTP_STALL_SECONDS_DEFAULT
    try:
        configured = int(raw)
    except (TypeError, ValueError):
        return OTP_STALL_SECONDS_DEFAULT
    return max(configured, OTP_STALL_SECONDS_MIN)


def _active_otp_demand(conn, now: dt.datetime) -> tuple[int, dt.datetime | None]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS pending_count, "
            "MIN(wait_started_at) AS oldest_wait_started_at "
            "FROM otp_request "
            "WHERE code IS NULL AND consumed_at IS NULL "
            "AND wait_started_at IS NOT NULL "
            "AND requested_at <= %s AND expires_at > %s",
            (now, now),
        )
        row = cur.fetchone()
    if not row:
        return 0, None
    return int(row["pending_count"]), row["oldest_wait_started_at"]


def _check_otp_relay(
    conn, now: dt.datetime, gmail_token_ok: bool | None
) -> list[Alert]:
    """Report relay health and delivery stalls without exposing request data."""
    alerts: list[Alert] = []
    relay_down_reasons: list[str] = []
    if gmail_token_ok is False:
        relay_down_reasons.append("mail source authentication failed")

    pending_count, oldest_wait_started_at = _active_otp_demand(conn, now)
    otp_beat = _max_last_beat(conn, "otp_responder", negate=False)
    stale_before = now - dt.timedelta(minutes=STALE_MIN)
    if otp_beat is None and pending_count:
        relay_down_reasons.append(
            f"responder heartbeat absent with {pending_count} active wait(s)"
        )
    elif otp_beat is not None and otp_beat < stale_before:
        stale_seconds = max(0, int((now - otp_beat).total_seconds()))
        relay_down_reasons.append(
            f"responder heartbeat stale by {stale_seconds}s"
        )

    if relay_down_reasons:
        alerts.append(
            Alert(
                kind="otp_relay_down",
                severity="critical",
                detail="OTP relay down: " + "; ".join(relay_down_reasons),
            )
        )

    responder_fresh = otp_beat is not None and otp_beat >= stale_before
    if (
        pending_count
        and oldest_wait_started_at is not None
        and responder_fresh
        and gmail_token_ok is True
    ):
        wait_age = max(dt.timedelta(0), now - oldest_wait_started_at)
        wait_age_seconds = int(wait_age.total_seconds())
        if wait_age.total_seconds() > _otp_stall_seconds():
            alerts.append(
                Alert(
                    kind="otp_delivery_stalled",
                    severity="critical",
                    detail=(
                        f"{pending_count} active wait(s); oldest age "
                        f"{wait_age_seconds}s; responder and mail source healthy"
                    ),
                )
            )

    return alerts


def _check_owner_inbox_backlog(conn, cfg: dict, now: dt.datetime) -> Alert | None:
    """Alert when the fleet quietly shifts into a manual-auth backlog.

    ``owner_inbox`` challenges do not generate OTP emails; they park rows for manual
    triage in the console. A sudden burst can look like "OTP stopped working" to the
    operator even though the relay is healthy. Promote that burst into the same
    DeadMan/banner channel used for other lane-stall conditions.
    """
    if not _is_armed(cfg):
        return None
    window_start = now - dt.timedelta(minutes=OWNER_INBOX_WINDOW_MIN)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT kind, COUNT(*) AS n "
            "FROM auth_challenge "
            "WHERE resolved_at IS NULL AND route='owner_inbox' AND raised_at >= %s "
            "GROUP BY kind ORDER BY n DESC, kind ASC",
            (window_start,),
        )
        rows = cur.fetchall()
        cur.execute(
            "SELECT COUNT(*) AS n FROM otp_request WHERE requested_at >= %s",
            (window_start,),
        )
        otp_recent = int(cur.fetchone()["n"])
    total = sum(int(row["n"]) for row in rows)
    if total <= 0 or otp_recent > 0:
        return None
    kind_parts = [f"{row['kind']}={int(row['n'])}" for row in rows if row.get("kind")]
    detail = (
        f"{total} fresh owner_inbox challenge(s) in the last {OWNER_INBOX_WINDOW_MIN}m"
        f"; {otp_recent} fresh otp_request(s)"
        + (f" ({', '.join(kind_parts)})" if kind_parts else "")
    )
    return Alert(
        kind="owner_inbox_backlog",
        severity="critical",
        detail=detail,
    )


def _check_stale_owner_inbox(conn, cfg: dict, now: dt.datetime) -> Alert | None:
    """Alert when owner-gated ATS leases remain frozen beyond the review window."""
    if not _is_armed(cfg):
        return None
    stale_before = now - dt.timedelta(hours=OWNER_INBOX_STALE_HOURS)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS n, MIN(a.raised_at) AS oldest
            FROM auth_challenge a
            JOIN apply_queue q ON q.url = a.url AND q.lane = 'ats'
            WHERE a.resolved_at IS NULL AND a.route = 'owner_inbox'
              AND q.status = 'leased' AND q.apply_status = 'challenge_pending'
              AND a.raised_at < %s
            """,
            (stale_before,),
        )
        row = cur.fetchone()
    count = int(row["n"] or 0)
    if count <= 0:
        return None
    oldest = row["oldest"]
    age_hours = int(max(dt.timedelta(0), now - oldest).total_seconds() // 3600) if oldest else 0
    return Alert(
        kind="owner_inbox_stale",
        severity="critical",
        detail=(
            f"{count} owner_inbox ATS lease(s) older than {OWNER_INBOX_STALE_HOURS}h; "
            f"oldest age {age_hours}h"
        ),
    )


def deadman_check(
    conn, *, now: dt.datetime, prev_hot_streak: int = 0, gmail_token_ok: bool | None = None
) -> tuple[list[Alert], int]:
    """Read-only dead-man check. Never writes/commits; ``now`` is always injected.

    Returns (alerts, new_hot_streak) -- the caller is responsible for persisting
    ``new_hot_streak`` and passing it back in as ``prev_hot_streak`` on the next call.
    """
    # Robustness over strictness for a safety monitor: a naive `now` (an easy caller
    # mistake -- datetime.now() vs datetime.now(timezone.utc)) would otherwise raise a
    # bare TypeError deep in an aware-vs-naive comparison and crash the whole watcher --
    # strictly worse than a missed alert. All repo timestamps are UTC, so coerce.
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)

    cfg = _fleet_config(conn)
    alerts: list[Alert] = []

    silent = _check_silent_death(conn, cfg, now)
    if silent:
        alerts.append(silent)

    stalled = _check_stalled_queue(conn, cfg, now)
    if stalled:
        alerts.append(stalled)

    selfheal = _check_selfheal_dead(conn, cfg, now)
    if selfheal:
        alerts.append(selfheal)

    alerts.extend(_check_otp_relay(conn, now, gmail_token_ok))

    owner_inbox = _check_owner_inbox_backlog(conn, cfg, now)
    if owner_inbox:
        alerts.append(owner_inbox)

    stale_owner_inbox = _check_stale_owner_inbox(conn, cfg, now)
    if stale_owner_inbox:
        alerts.append(stale_owner_inbox)

    hot_alert, new_streak = _check_running_hot(conn, cfg, now, prev_hot_streak)
    if hot_alert:
        alerts.append(hot_alert)

    return alerts, new_streak


# ---------------------------------------------------------------------------
# Task 2: persistence + delivery wrapper around the pure check above.
# ---------------------------------------------------------------------------

ALERT_FILENAME = "fleet-ALERT.txt"


def _summarize(alerts: list[Alert]) -> str:
    return " | ".join(f"{a.kind}: {a.detail}" for a in alerts)


def gmail_token_alive() -> bool | None:
    """Best-effort: does the home-box Gmail OAuth token still refresh? Returns True/False, or
    None if it can't be checked (google-auth libs or token file absent, or a network error).
    Attempts an IN-MEMORY refresh -- no Gmail data is read and the token is NOT persisted.
    Never raises. Feeds run_deadman -> deadman_check(gmail_token_ok=...) so the fleet-wide OTP
    relay (which dies every ~7 days on a Testing-mode OAuth app) surfaces as an alert."""
    try:
        import json
        from google.auth.exceptions import RefreshError
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from applypilot.config import APP_DIR
    except Exception:
        return None
    tok = Path(APP_DIR) / "gmail_token.json"
    if not tok.exists():
        return None
    try:
        info = json.loads(tok.read_text(encoding="utf-8"))
        creds = Credentials.from_authorized_user_info(info, info.get("scopes"))
        creds.refresh(Request())  # in-memory only; never saved
        return True
    except RefreshError:
        return False       # invalid_grant -> the token backing the relay is dead
    except Exception:
        return None        # network/parse error -> unknown, don't false-alarm


def mail_source_alive() -> bool | None:
    """Best-effort: can the fleet's mail relay still read Gmail? Prefers the IMAP
    app-password path (permanent, replaces the 7-day OAuth token): if an app password
    is configured, attempts a real login+select+search via ImapMailSource.fetch(). Returns
    True on success, False on a MailSourceError/imaplib login failure (bad password or IMAP
    disabled in Gmail settings -- the relay is actually dead), or None on any other exception
    (network/unknown -- don't false-alarm). If NO app password is configured, falls back to
    the legacy gmail_token_alive() OAuth-refresh probe so nothing regresses for OAuth-only
    setups. Never raises. Feeds run_deadman -> deadman_check(gmail_token_ok=...)."""
    try:
        creds = config.load_gmail_app_password()
    except Exception:
        creds = None

    if not creds:
        return gmail_token_alive()

    try:
        ImapMailSource(creds[0], creds[1]).fetch(
            since_days=1,
            max_messages=1,
            gmail_raw_query=inbox_auth.AUTH_GMAIL_RAW_QUERY,
        )
        return True
    except MailSourceOverflowError:
        # The bounded probe already authenticated and found candidates; overflow
        # is evidence of a busy inbox, not evidence that the relay is down.
        return True
    except MailSourceError:
        return False
    except imaplib.IMAP4.error:
        return False
    except Exception:
        return None


def _send_toast(summary: str) -> None:
    """Best-effort Windows toast via BurntToast. Callers MUST wrap this in
    try/except -- a missing module / non-Windows host / no PowerShell is a
    silent no-op, not a failure of the monitor."""
    ps_cmd = (
        "Import-Module BurntToast -ErrorAction Stop; "
        "New-BurntToastNotification -Text 'ApplyPilot fleet dead-man alert', "
        f"'{summary}'"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_cmd],
        capture_output=True, timeout=15, check=False,
    )


def run_deadman(conn, *, now: dt.datetime, alert_dir: Path,
                gmail_token_ok: bool | None = None) -> list[Alert]:
    """Persistence + delivery wrapper around ``deadman_check``.

    Reads/persists the running_hot streak in ``fleet_config.deadman_hot_streak``,
    writes the current alert summary to ``fleet_config.deadman_alert`` (+ _at) and
    an ``alert_dir/fleet-ALERT.txt`` file when alerts are active, clears both when
    healthy, and best-effort attempts a Windows toast notification. Commits its
    own writes. Never raises on delivery failures (toast / file I/O around the
    toast is isolated); returns the list of active alerts.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT deadman_hot_streak FROM fleet_config WHERE id=1;")
        row = cur.fetchone()
    prev_streak = row["deadman_hot_streak"] if row else 0

    alerts, new_streak = deadman_check(
        conn, now=now, prev_hot_streak=prev_streak, gmail_token_ok=gmail_token_ok)

    alert_dir = Path(alert_dir)
    alert_path = alert_dir / ALERT_FILENAME

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE fleet_config SET deadman_hot_streak=%s WHERE id=1;",
            (new_streak,),
        )
        if alerts:
            summary = _summarize(alerts)
            cur.execute(
                "UPDATE fleet_config SET deadman_alert=%s, deadman_alert_at=%s WHERE id=1;",
                (summary, now),
            )
        else:
            cur.execute(
                "UPDATE fleet_config SET deadman_alert=NULL, deadman_alert_at=NULL WHERE id=1;",
            )
    conn.commit()

    if alerts:
        summary = _summarize(alerts)
        try:
            alert_dir.mkdir(parents=True, exist_ok=True)
            lines = [f"ApplyPilot fleet dead-man alert -- {now.isoformat()}", ""]
            for a in alerts:
                lines.append(f"[{a.severity}] {a.kind}: {a.detail}")
            alert_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            pass  # best-effort delivery; the DB flag is the source of truth.
        try:
            _send_toast(summary)
        except Exception:
            pass  # best-effort delivery; never let a toast failure crash the monitor.
    else:
        try:
            if alert_path.exists():
                alert_path.unlink()
        except OSError:
            pass

    return alerts


def _write_monitor_failure_alert(alert_dir: Path, *, now: dt.datetime) -> None:
    """Persist a credential-free fallback when the database monitor cannot run."""
    try:
        alert_dir = Path(alert_dir)
        alert_dir.mkdir(parents=True, exist_ok=True)
        (alert_dir / ALERT_FILENAME).write_text(
            f"ApplyPilot fleet dead-man alert -- {now.isoformat()}\n\n"
            "[critical] monitor_failure: fleet health check could not complete\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def main(argv=None) -> int:  # pragma: no cover - thin CLI glue
    p = argparse.ArgumentParser(
        prog="applypilot-fleet-deadman",
        description="Read-only fleet dead-man monitor: alerts on silent workers, "
                    "a stalled queue, a dead self-healer, or a hot cost streak.",
    )
    p.add_argument(
        "--dsn",
        default=os.environ.get("FLEET_PG_DSN"),
        help="explicit mapped fleet DSN (default: FLEET_PG_DSN). Never the postgres admin "
             "or shared fleet_worker role; no generic DATABASE_URL/APPLYPILOT_FLEET_DSN fallback.",
    )
    p.add_argument(
        "--alert-dir",
        default=os.path.join(os.environ.get("LOCALAPPDATA", ""), "ApplyPilot"),
    )
    p.add_argument(
        "--ensure-schema",
        action="store_true",
        help="Apply the fleet schema before checking (use during bootstrap, not on every scheduled pass).",
    )
    p.add_argument("--once", action="store_true", default=True,
                    help="run a single pass and exit (default; the scheduled task "
                         "triggers this once per cadence)")
    args = p.parse_args(argv)

    if not args.dsn:
        print("[fleet-deadman] no --dsn / FLEET_PG_DSN set", flush=True)
        return 1

    from applypilot.apply import pgqueue
    from applypilot.fleet import schema as fleet_schema

    now = dt.datetime.now(dt.timezone.utc)
    try:
        with pgqueue.connect(args.dsn) as conn:
            if args.ensure_schema:
                fleet_schema.ensure_schema_v3(conn)
            alerts = run_deadman(
                conn, now=now, alert_dir=Path(args.alert_dir),
                gmail_token_ok=mail_source_alive(),  # network probe stays in the untested CLI glue
            )
    except Exception:  # pragma: no cover - exercised through CLI tests
        _write_monitor_failure_alert(Path(args.alert_dir), now=now)
        print("[fleet-deadman] check failed; see local alert", flush=True)
        return 2

    if alerts:
        print(f"[fleet-deadman] ALERT: {_summarize(alerts)}", flush=True)
    else:
        print("[fleet-deadman] healthy", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
