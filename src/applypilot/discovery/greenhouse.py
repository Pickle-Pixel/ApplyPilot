"""Greenhouse ATS discovery: scrapes jobs from Greenhouse boards.

Greenhouse is used by ~60% of AI/ML startups (OpenAI, Anthropic, Scale AI, etc.).
Uses HTML scraping with BeautifulSoup since Greenhouse doesn't have a public JSON API.
"""

import logging
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import urljoin

import httpx
import yaml
from bs4 import BeautifulSoup

from applypilot import config
from applypilot.config import CONFIG_DIR
from applypilot.database import get_connection, init_db

log = logging.getLogger(__name__)

# Greenhouse board URL pattern
GREENHOUSE_BASE_URL = "https://job-boards.greenhouse.io/{company}"


def load_employers() -> dict:
    """Load Greenhouse employer registry from config/greenhouse.yaml."""
    path = CONFIG_DIR / "greenhouse.yaml"
    if not path.exists():
        log.warning("greenhouse.yaml not found at %s", path)
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("employers", {})


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


def fetch_greenhouse_board(company_slug: str) -> str | None:
    """Fetch the HTML content of a Greenhouse board."""
    url = GREENHOUSE_BASE_URL.format(company=company_slug)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.text
    except Exception as e:
        log.warning("Failed to fetch %s: %s", url, e)
        return None


def parse_greenhouse_jobs(html: str, company_name: str, query: str = "") -> list[dict]:
    """Parse job listings from Greenhouse board HTML."""
    jobs = []
    soup = BeautifulSoup(html, "html.parser")

    # Find all job posting rows
    job_rows = soup.find_all("tr", class_="job-post")

    for row in job_rows:
        try:
            # Extract job link and title
            link_elem = row.find("a", href=True)
            if not link_elem:
                continue

            href = link_elem.get("href", "")
            job_url = str(href) if href else ""
            if job_url and not job_url.startswith("http"):
                job_url = urljoin("https://job-boards.greenhouse.io", job_url)

            # Extract title
            title_elem = link_elem.find("p", class_="body--medium")
            title = title_elem.get_text(strip=True) if title_elem else ""

            # Extract location
            location_elem = link_elem.find("p", class_="body--metadata")
            location = location_elem.get_text(strip=True) if location_elem else ""

            # Extract department (from parent department section)
            dept_section = row.find_parent("div", class_="job-posts--table--department")
            department = ""
            if dept_section:
                dept_header = dept_section.find("h3", class_="section-header")
                if dept_header:
                    department = dept_header.get_text(strip=True)

            # Filter by query
            if query and not _title_matches_query(title, query):
                continue

            jobs.append(
                {
                    "title": title,
                    "company": company_name,
                    "location": location,
                    "department": department,
                    "url": job_url,
                    "strategy": "greenhouse",
                }
            )

        except Exception as e:
            log.debug("Error parsing job row: %s", e)
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
    """Search a single Greenhouse employer."""
    log.info('%s: searching "%s"...', employer["name"], search_text)

    html = fetch_greenhouse_board(employer_key)
    if not html:
        return []

    jobs = parse_greenhouse_jobs(html, employer["name"], search_text)

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
    """Search all configured Greenhouse employers.

    Returns (new_jobs_count, existing_jobs_count).
    """
    employers = _employers_override if _employers_override else load_employers()
    if not employers:
        log.warning("No Greenhouse employers configured")
        return 0, 0

    accept_locs, reject_locs = _load_location_filter()

    log.info('Greenhouse search: %d employers, "%s", workers=%d', len(employers), search_text, workers)

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
        "Greenhouse search complete: %d total jobs from %d employers (%d errors)", len(all_jobs), len(employers), errors
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
                    None,
                    job.get("department", ""),
                    job.get("location", ""),
                    job["company"],
                    "greenhouse",
                    now,
                    None,
                    job["url"],
                    None,
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
        log.info('Greenhouse search: "%s"', query)

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
