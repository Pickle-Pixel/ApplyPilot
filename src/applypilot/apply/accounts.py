"""ATS account registry: remember which employer portals already have an account.

Workday (and iCIMS, Taleo, Oracle, SuccessFactors) make you create a separate
account per employer tenant, verified by an email link or code. Tracking which
tenants we've already registered lets the apply agent sign in directly instead
of trying "Create Account" again (which fails with "email already in use").

The key is the tenant host, e.g. ``adobe.wd5.myworkdayjobs.com``.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from urllib.parse import urlsplit

from applypilot.database import get_connection

# Hosts where every employer runs its own account silo
_ACCOUNT_ATS = {
    "myworkdayjobs.com": "workday",
    "myworkday.com": "workday",
    "icims.com": "icims",
    "taleo.net": "taleo",
    "oraclecloud.com": "oracle",
    "successfactors.com": "successfactors",
    "successfactors.eu": "successfactors",
    "brassring.com": "brassring",
}

# Lines the agent prints to report account state (parsed by the launcher)
ACCOUNT_LINE = re.compile(r"ACCOUNT:(CREATED|SIGNED_IN|EXISTS_BAD_PASSWORD|VERIFY_PENDING)", re.I)


def _ensure_table() -> None:
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ats_accounts (
            host         TEXT PRIMARY KEY,
            ats          TEXT,
            email        TEXT,
            status       TEXT,
            created_at   TEXT,
            last_used_at TEXT
        )
    """)
    conn.commit()


def ats_for_url(url: str | None) -> tuple[str | None, str | None]:
    """Return (ats_name, tenant_host) for an account-based ATS URL, else (None, None)."""
    if not url:
        return None, None
    host = (urlsplit(url).hostname or "").lower()
    for suffix, name in _ACCOUNT_ATS.items():
        if host == suffix or host.endswith("." + suffix):
            return name, host
    return None, None


def get_account(host: str) -> dict | None:
    _ensure_table()
    row = get_connection().execute("SELECT * FROM ats_accounts WHERE host = ?", (host,)).fetchone()
    return dict(row) if row else None


def record_account(host: str, ats: str, email: str, status: str) -> None:
    """Upsert the account status for a tenant host."""
    _ensure_table()
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    conn.execute("""
        INSERT INTO ats_accounts (host, ats, email, status, created_at, last_used_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(host) DO UPDATE SET status = excluded.status,
                                        email = excluded.email,
                                        last_used_at = excluded.last_used_at
    """, (host, ats, email, status, now, now))
    conn.commit()


def record_from_output(url: str | None, email: str, output: str) -> str | None:
    """Parse ACCOUNT:* lines from agent output and persist the latest one."""
    ats, host = ats_for_url(url)
    if not host:
        return None
    matches = ACCOUNT_LINE.findall(output)
    if not matches:
        return None
    status = matches[-1].lower()
    record_account(host, ats, email, status)
    return status


def list_accounts() -> list[dict]:
    _ensure_table()
    rows = get_connection().execute("SELECT * FROM ats_accounts ORDER BY last_used_at DESC").fetchall()
    return [dict(r) for r in rows]
