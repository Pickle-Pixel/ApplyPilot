"""Handshake job discovery: scrapes jobs from app.joinhandshake.com.

Uses Playwright for authentication (supports university SSO) and Handshake's
internal JSON API for job listings.

Configuration via .env:
  HANDSHAKE_EMAIL    - your university email address
  HANDSHAKE_PASSWORD - your password (optional; leave blank for manual SSO login)

Session state is cached at ~/.applypilot/handshake_session.json so you only
need to log in once. Delete that file to force a fresh login.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from applypilot import config
from applypilot.database import get_connection, init_db, store_jobs

log = logging.getLogger(__name__)

_SESSION_PATH = Path.home() / ".applypilot" / "handshake_session.json"
_BASE_URL = "https://app.joinhandshake.com"
_LOGIN_URL = f"{_BASE_URL}/login"
_JOBS_API = f"{_BASE_URL}/stu/postings"


# ---------------------------------------------------------------------------
# Location filter (same pattern as other discovery modules)
# ---------------------------------------------------------------------------

def _location_ok(location: str | None, accept: list[str], reject: list[str]) -> bool:
    if not location:
        return True
    loc = location.lower()
    if any(r in loc for r in ("remote", "anywhere", "work from home", "wfh", "distributed")):
        return True
    for r in reject:
        if r.lower() in loc:
            return False
    for a in accept:
        if a.lower() in loc:
            return True
    return False


# ---------------------------------------------------------------------------
# Login helpers
# ---------------------------------------------------------------------------

def _do_login(page, email: str, password: str) -> None:
    """Attempt automated login. Falls back to manual if SSO is detected."""
    log.info("Navigating to Handshake login page...")
    page.goto(_LOGIN_URL, wait_until="networkidle")

    # Fill in email
    try:
        page.fill('input[type="email"], input[name="email"], input[id="email"]', email)
        page.click('button[type="submit"], input[type="submit"]')
        page.wait_for_timeout(2000)
    except Exception:
        pass

    # If we're still on a login-like page, check for SSO redirect
    current_url = page.url
    if "joinhandshake.com" not in current_url or "login" in current_url or "sso" in current_url.lower():
        if password:
            # Try filling password if there's a password field
            try:
                page.fill('input[type="password"]', password)
                page.click('button[type="submit"], input[type="submit"]')
                page.wait_for_timeout(3000)
            except Exception:
                pass

    # Check if we landed on SSO (university login page) — need manual input
    current_url = page.url
    if "joinhandshake.com/stu" not in current_url:
        log.warning(
            "Automatic login could not complete (likely university SSO). "
            "Please complete the login manually in the browser window. "
            "The script will wait up to 3 minutes..."
        )
        # Wait for the user to log in manually (up to 3 minutes)
        try:
            page.wait_for_url("**/stu/**", timeout=180_000)
            log.info("Manual login detected — continuing.")
        except Exception:
            raise RuntimeError(
                "Timed out waiting for Handshake login. "
                "Set HANDSHAKE_EMAIL and HANDSHAKE_PASSWORD in .env, "
                "or complete the SSO login within 3 minutes."
            )


def _ensure_logged_in(playwright) -> tuple:
    """Return (browser, context, page) with a valid Handshake session.

    Loads cached session if available. Re-authenticates if session is expired.
    """
    email = os.environ.get("HANDSHAKE_EMAIL", "")
    password = os.environ.get("HANDSHAKE_PASSWORD", "")

    browser = playwright.chromium.launch(headless=False)  # visible for SSO

    # Try to restore saved session
    if _SESSION_PATH.exists():
        log.info("Loading saved Handshake session from %s", _SESSION_PATH)
        context = browser.new_context(storage_state=str(_SESSION_PATH))
        page = context.new_page()
        page.goto(f"{_BASE_URL}/stu/postings", wait_until="networkidle")

        if "/stu/" in page.url:
            log.info("Session still valid.")
            return browser, context, page

        log.info("Session expired, re-authenticating...")
        context.close()

    # Fresh login
    context = browser.new_context()
    page = context.new_page()
    _do_login(page, email, password)

    # Save session for next run
    _SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    context.storage_state(path=str(_SESSION_PATH))
    log.info("Session saved to %s", _SESSION_PATH)

    return browser, context, page


# ---------------------------------------------------------------------------
# Job fetching via internal API
# ---------------------------------------------------------------------------

def _fetch_jobs_page(page, query: str, page_num: int, per_page: int = 25) -> dict:
    """Call Handshake's internal postings JSON API via the logged-in browser."""
    params = {
        "ajax": "true",
        "page": str(page_num),
        "per_page": str(per_page),
        "sort_direction": "desc",
        "sort_column": "default",
        "query": query,
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{_JOBS_API}?{qs}"

    response = page.evaluate(
        """async (url) => {
            const resp = await fetch(url, {
                headers: { "Accept": "application/json", "X-Requested-With": "XMLHttpRequest" }
            });
            return { status: resp.status, body: await resp.text() };
        }""",
        url,
    )

    if response["status"] != 200:
        raise RuntimeError(f"Handshake API returned HTTP {response['status']} for: {url}")

    return json.loads(response["body"])


def _parse_posting(posting: dict) -> dict | None:
    """Convert a Handshake posting dict to the ApplyPilot job format."""
    posting_id = posting.get("id")
    if not posting_id:
        return None

    url = f"{_BASE_URL}/stu/postings/{posting_id}"

    # Employer name
    employer = posting.get("employer", {}) or {}
    company = employer.get("name", "")

    # Location: prefer remote label, fall back to city/state
    location = ""
    if posting.get("is_remote"):
        location = "Remote"
    else:
        city = posting.get("city", "") or ""
        state = posting.get("state", "") or ""
        location = f"{city}, {state}".strip(", ")

    # Salary
    salary = ""
    pay_min = posting.get("compensation_min")
    pay_max = posting.get("compensation_max")
    pay_type = posting.get("compensation_period", "")
    if pay_min:
        salary = f"${int(pay_min):,}"
        if pay_max:
            salary += f"-${int(pay_max):,}"
        if pay_type:
            salary += f"/{pay_type}"

    return {
        "url": url,
        "title": posting.get("title", ""),
        "location": location,
        "salary": salary,
        "description": posting.get("description", ""),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_handshake_discovery(workers: int = 1) -> None:
    """Discover jobs from Handshake and store them in the ApplyPilot DB.

    Reads search queries from searches.yaml (same queries as JobSpy).
    Requires HANDSHAKE_EMAIL in .env; HANDSHAKE_PASSWORD is optional.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
        return

    search_cfg = config.load_search_config()
    queries = [q["query"] for q in search_cfg.get("queries", [])]
    accept_locs = search_cfg.get("location_accept", [])
    reject_locs = search_cfg.get("location_reject_non_remote", [])

    if not queries:
        log.warning("No queries defined in searches.yaml — skipping Handshake.")
        return

    email = os.environ.get("HANDSHAKE_EMAIL", "")
    if not email:
        log.warning("HANDSHAKE_EMAIL not set in .env — skipping Handshake discovery.")
        return

    init_db()
    conn = get_connection()
    total_new = 0
    total_dupes = 0

    with sync_playwright() as playwright:
        browser, context, page = _ensure_logged_in(playwright)

        try:
            for query in queries:
                log.info("Handshake: searching for '%s'", query)
                new = 0
                dupes = 0
                page_num = 1

                while True:
                    try:
                        data = _fetch_jobs_page(page, query, page_num)
                    except Exception as e:
                        log.error("Handshake API error (page %d): %s", page_num, e)
                        break

                    postings = data.get("postings", [])
                    if not postings:
                        break

                    jobs = []
                    for posting in postings:
                        job = _parse_posting(posting)
                        if job and _location_ok(job["location"], accept_locs, reject_locs):
                            jobs.append(job)

                    n, d = store_jobs(conn, jobs, site="handshake", strategy="handshake_api")
                    new += n
                    dupes += d

                    # Stop if we've reached the last page
                    total_count = data.get("total_count", 0)
                    per_page = data.get("per_page", 25)
                    if page_num * per_page >= total_count:
                        break

                    page_num += 1
                    time.sleep(1)  # be polite

                log.info("Handshake ['%s']: %d new, %d dupes", query, new, dupes)
                total_new += new
                total_dupes += dupes

        finally:
            context.close()
            browser.close()

    log.info("Handshake discovery complete: %d new | %d dupes", total_new, total_dupes)
