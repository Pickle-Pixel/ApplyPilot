"""Greenhouse ATS discovery: fetches jobs from Greenhouse Job Board API.

Greenhouse is used by ~60% of AI/ML startups (OpenAI, Anthropic, Scale AI, etc.).
Uses the official public Job Board API: https://boards-api.greenhouse.io/v1/boards/{token}/jobs
"""

import logging
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import httpx
import yaml

from applypilot import config
from applypilot.config import APP_DIR, CONFIG_DIR
from applypilot.database import get_connection

log = logging.getLogger(__name__)

# Greenhouse Job Board API endpoint
GREENHOUSE_API_BASE = "https://boards-api.greenhouse.io/v1/boards"


def load_employers() -> dict:
    """Load Greenhouse employer registry.

    Tries user config first (~/.applypilot/greenhouse.yaml),
    falls back to package config.
    """
    # Try user config
    user_path = APP_DIR / "greenhouse.yaml"
    if user_path.exists():
        log.info("Loading user Greenhouse config from %s", user_path)
        try:
            data = yaml.safe_load(user_path.read_text(encoding="utf-8"))
            if data and "employers" in data:
                return data.get("employers", {})
        except Exception as e:
            log.warning("Failed to load user config: %s", e)

    # Fall back to package config
    package_path = CONFIG_DIR / "greenhouse.yaml"
    if not package_path.exists():
        log.warning("greenhouse.yaml not found at %s", package_path)
        return {}

    try:
        data = yaml.safe_load(package_path.read_text(encoding="utf-8"))
        return data.get("employers", {})
    except Exception as e:
        log.error("Failed to load package config: %s", e)
        return {}


def _load_location_filter(search_cfg: dict | None = None):
    """Load location accept/reject lists from search config."""
    if search_cfg is None:
        search_cfg = config.load_search_config()

    accept = search_cfg.get("location_accept", [])
    reject = search_cfg.get("location_reject_non_remote", [])
    return accept, reject


def _location_ok(location: str | None, accept: list[str], reject: list[str]) -> bool:
    """Check if a job location passes the user's location filter."""
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


def _title_matches_query(title: str, query: str) -> bool:
    """Check if job title matches search query (simple keyword matching)."""
    if not query:
        return True

    title_lower = title.lower()
    query_terms = query.lower().split()

    # Match if any query term appears in title
    return any(term in title_lower for term in query_terms)


def _strip_html(html_content: str) -> str:
    """Strip HTML tags from content to get plain text."""
    if not html_content:
        return ""

    # Simple regex to remove HTML tags
    text = re.sub(r"<[^>]+>", "", html_content)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_jobs_api(board_token: str, content: bool = True) -> dict | None:
    """Fetch jobs from Greenhouse Job Board API.

    Args:
        board_token: The company slug (e.g., "stripe", "robinhood")
        content: If True, include full job description in response

    Returns:
        API response dict with "jobs" and "meta" keys, or None on error
    """
    url = f"{GREENHOUSE_API_BASE}/{board_token}/jobs"
    params = {"content": "true"} if content else {}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(url, headers=headers, params=params)

            if resp.status_code == 404:
                log.debug("Board not found: %s", board_token)
                return None
            elif resp.status_code == 429:
                log.warning("Rate limited for %s, retrying...", board_token)
                time.sleep(2)
                resp = client.get(url, headers=headers, params=params)
                resp.raise_for_status()
            else:
                resp.raise_for_status()

            return resp.json()

    except httpx.HTTPStatusError as e:
        log.warning("HTTP error for %s: %s", board_token, e)
        return None
    except Exception as e:
        log.warning("Failed to fetch %s: %s", board_token, e)
        return None


def parse_api_response(data: dict, company_name: str, query: str = "") -> list[dict]:
    """Parse job listings from Greenhouse API response.

    Args:
        data: API response dict with "jobs" key
        company_name: Display name of the company
        query: Optional query string to filter jobs

    Returns:
        List of job dicts with standardized fields
    """
    jobs = []
    job_list = data.get("jobs", [])

    for job_data in job_list:
        try:
            title = job_data.get("title", "")
            if not title:
                continue

            # Filter by query
            if query and not _title_matches_query(title, query):
                continue

            # Extract location
            location_obj = job_data.get("location", {})
            location = location_obj.get("name", "") if isinstance(location_obj, dict) else str(location_obj)

            # Extract department
            departments = job_data.get("departments", [])
            department = departments[0].get("name", "") if departments else ""

            # Extract offices
            offices = job_data.get("offices", [])
            office_names = [office.get("name", "") for office in offices if office.get("name")]

            # Get full description and strip HTML
            html_content = job_data.get("content", "")
            description = _strip_html(html_content)

            # Build job dict
            job = {
                "title": title,
                "company": company_name,
                "location": location,
                "department": department,
                "offices": office_names,
                "url": job_data.get("absolute_url", ""),
                "strategy": "greenhouse",
                # New fields from API
                "job_id": job_data.get("id"),
                "internal_job_id": job_data.get("internal_job_id"),
                "description": description,
                "updated_at": job_data.get("updated_at"),
            }

            jobs.append(job)

        except Exception as e:
            log.debug("Error parsing job: %s", e)
            continue

    return jobs


def search_employer(
    employer_key: str,
    employer: dict,
    search_text: str,
    location_filter: bool = True,
    accept_locs: list[str] | None = None,
    reject_locs: list[str] | None = None,
) -> list[dict]:
    """Search a single Greenhouse employer via API."""
    log.info('%s: searching "%s"...', employer["name"], search_text)

    # Fetch from API
    api_data = fetch_jobs_api(employer_key, content=True)
    if not api_data:
        return []

    jobs = parse_api_response(api_data, employer["name"], search_text)

    # Apply location filter
    if location_filter and (accept_locs or reject_locs):
        filtered = []
        for job in jobs:
            if _location_ok(job.get("location"), accept_locs or [], reject_locs or []):
                filtered.append(job)
        jobs = filtered

    log.info("%s: found %d jobs", employer["name"], len(jobs))
    return jobs


def search_all(
    search_text: str,
    workers: int = 4,
    location_filter: bool = True,
    _employers_override: dict | None = None,
) -> tuple[int, int]:
    """Search all configured Greenhouse employers via API.

    Returns (new_jobs_count, existing_jobs_count).
    """
    employers = _employers_override if _employers_override else load_employers()
    if not employers:
        log.warning("No Greenhouse employers configured")
        return 0, 0

    accept_locs, reject_locs = _load_location_filter()

    log.info('Greenhouse API search: %d employers, "%s", workers=%d', len(employers), search_text, workers)

    all_jobs = []
    errors = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                search_employer,
                key,
                emp,
                search_text,
                location_filter,
                accept_locs,
                reject_locs,
            ): key
            for key, emp in employers.items()
        }

        for future in as_completed(futures):
            key = futures[future]
            try:
                jobs = future.result()
                all_jobs.extend(jobs)
            except Exception as e:
                log.error("Error searching %s: %s", key, e)
                errors += 1

    log.info(
        "Greenhouse API search complete: %d total jobs from %d employers (%d errors)",
        len(all_jobs),
        len(employers),
        errors,
    )

    # Store in database
    return _store_jobs(all_jobs)


def _store_jobs(jobs: list[dict]) -> tuple[int, int]:
    """Store discovered jobs in the database. Returns (new, existing)."""
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    new = 0
    existing = 0

    for job in jobs:
        try:
            conn.execute(
                "INSERT INTO jobs (url, title, salary, description, location, site, strategy, "
                "discovered_at, full_description, application_url, detail_scraped_at, detail_error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job["url"],
                    job["title"],
                    None,  # salary not provided by API
                    job.get("description", ""),  # Now we have full description!
                    job.get("location", ""),
                    job["company"],
                    "greenhouse",
                    now,
                    job.get("description"),  # full_description
                    job["url"],  # application_url
                    now,  # detail_scraped_at (we got it from API)
                    None,
                ),
            )
            new += 1
        except sqlite3.IntegrityError:
            existing += 1

    conn.commit()
    return new, existing


def run_all_searches(
    searches: list[dict],
    workers: int = 4,
) -> dict:
    """Run multiple search queries across all Greenhouse employers.

    Args:
        searches: List of search configs with 'query' key
        workers: Number of parallel threads

    Returns:
        Dict with total new/existing counts and per-query breakdown
    """
    total_new = 0
    total_existing = 0
    per_query = []

    for search in searches:
        query = search.get("query", "")
        log.info('Greenhouse API search: "%s"', query)

        new, existing = search_all(query, workers=workers)
        total_new += new
        total_existing += existing

        per_query.append(
            {
                "query": query,
                "new": new,
                "existing": existing,
            }
        )

    return {
        "total_new": total_new,
        "total_existing": total_existing,
        "per_query": per_query,
    }


# Legacy functions for backward compatibility (deprecated)
def fetch_greenhouse_board(company_slug: str) -> str | None:
    """DEPRECATED: Use fetch_jobs_api() instead."""
    log.warning("fetch_greenhouse_board() is deprecated, use fetch_jobs_api()")
    return None


def parse_greenhouse_jobs(html: str, company_name: str, query: str = "") -> list[dict]:
    """DEPRECATED: Use parse_api_response() instead."""
    log.warning("parse_greenhouse_jobs() is deprecated, use parse_api_response()")
    return []
