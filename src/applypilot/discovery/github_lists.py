"""GitHub job-list discovery: import curated postings from community repos.

Repos like SimplifyJobs/New-Grad-Positions and SimplifyJobs/Summer2027-Internships
publish hand-picked postings with direct application links. This module pulls
them into the jobs table so they flow through enrich > score > tailor > apply.

Two formats are supported:
  1. Simplify-style ``listings.json`` (structured; preferred when present)
  2. README tables -- HTML ``<table>`` or markdown pipe tables (speedyapply, etc.)

Configured via the ``github_lists`` section of searches.yaml:

    github_lists:
      repos:
        - SimplifyJobs/New-Grad-Positions
        - SimplifyJobs/Summer2027-Internships
      max_age_days: 14
      include_titles: ["software", "engineer"]
      exclude_titles: ["senior"]
      locations: ["Remote", "VA", "Virginia"]
      categories: []            # listings.json only, e.g. ["Software", "AI/ML/Data"]
      skip_no_sponsorship: false
      skip_citizenship_required: false
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup

from applypilot import config
from applypilot.database import get_connection, init_db

log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_REPOS = [
    "SimplifyJobs/New-Grad-Positions",
    "SimplifyJobs/Summer2027-Internships",
]

_BRANCHES = ("dev", "main", "master")
_LISTINGS_PATHS = (".github/scripts/listings.json", "listings.json")
_RAW = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"

# Tracking params added by list maintainers -- stripped so URLs dedupe cleanly
_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term", "ref", "src"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_url(url: str) -> str:
    """Strip tracking query params (utm_*, ref=Simplify) from a URL."""
    parts = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if k.lower() not in _TRACKING_PARAMS]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def _parse_age_days(text: str) -> float | None:
    """Parse age strings like '3d', '2w', '1mo', '5h', or 'Sep 12' into days."""
    text = (text or "").strip().lower()
    m = re.fullmatch(r"(\d+)\s*(h|d|w|mo|m|y)", text)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        return {"h": n / 24, "d": n, "w": n * 7, "mo": n * 30, "m": n * 30, "y": n * 365}[unit]
    for fmt in ("%b %d", "%B %d", "%b %d, %Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text.title() if "%b" in fmt or "%B" in fmt else text, fmt)
        except ValueError:
            continue
        now = datetime.now()
        if dt.year == 1900:
            dt = dt.replace(year=now.year)
            if dt > now:
                dt = dt.replace(year=now.year - 1)
        return max((now - dt).days, 0)
    return None


def _http_get(client: httpx.Client, url: str) -> httpx.Response | None:
    try:
        resp = client.get(url)
    except httpx.HTTPError as e:
        log.debug("GET %s failed: %s", url, e)
        return None
    return resp if resp.status_code == 200 else None


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _fetch_listings_json(client: httpx.Client, repo: str) -> list[dict] | None:
    for branch in _BRANCHES:
        for path in _LISTINGS_PATHS:
            resp = _http_get(client, _RAW.format(repo=repo, branch=branch, path=path))
            if resp is None:
                continue
            try:
                data = resp.json()
            except ValueError:
                continue
            if isinstance(data, list):
                log.info("%s: using %s@%s (%d listings)", repo, path, branch, len(data))
                return data
    return None


def _fetch_readme(client: httpx.Client, repo: str) -> str | None:
    for branch in _BRANCHES:
        for name in ("README.md", "readme.md"):
            resp = _http_get(client, _RAW.format(repo=repo, branch=branch, path=name))
            if resp is not None:
                return resp.text
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _from_listings_json(data: list[dict], repo: str) -> list[dict]:
    jobs = []
    now = time.time()
    for item in data:
        if not item.get("active", True) or not item.get("is_visible", True):
            continue
        url = item.get("url")
        if not url:
            continue
        posted = item.get("date_posted") or item.get("date_updated")
        age = (now - posted) / 86400 if isinstance(posted, (int, float)) else None
        jobs.append({
            "company": (item.get("company_name") or "").strip(),
            "title": (item.get("title") or "").strip(),
            "locations": item.get("locations") or [],
            "url": url,
            "age_days": age,
            "category": item.get("category") or "",
            "sponsorship": item.get("sponsorship") or "",
            "terms": item.get("terms") or [],
            "repo": repo,
        })
    return jobs


def _pick_apply_link(cell: BeautifulSoup) -> str | None:
    """Pick the employer's apply link from a cell, preferring non-aggregator links."""
    hrefs = [a.get("href", "") for a in cell.find_all("a") if a.get("href", "").startswith("http")]
    if not hrefs:
        return None
    direct = [h for h in hrefs if "simplify.jobs/p/" not in h]
    return (direct or hrefs)[0]


def _map_columns(headers: list[str]) -> dict[str, int]:
    cols: dict[str, int] = {}
    for i, h in enumerate(headers):
        h = h.lower()
        if "company" in h and "company" not in cols:
            cols["company"] = i
        elif any(k in h for k in ("role", "position", "title")) and "title" not in cols:
            cols["title"] = i
        elif "location" in h and "location" not in cols:
            cols["location"] = i
        elif any(k in h for k in ("application", "apply", "link", "posting")) and "apply" not in cols:
            cols["apply"] = i
        elif any(k in h for k in ("age", "date", "posted")) and "age" not in cols:
            cols["age"] = i
    return cols


def _rows_from_cells(rows: list[list[BeautifulSoup]], headers: list[str], repo: str) -> list[dict]:
    cols = _map_columns(headers)
    if not {"company", "title", "apply"} <= cols.keys():
        return []
    jobs = []
    last_company = ""
    for cells in rows:
        if len(cells) <= max(cols.values()):
            continue
        row_text = " ".join(c.get_text(" ", strip=True) for c in cells)
        if "🔒" in row_text:
            continue
        company = cells[cols["company"]].get_text(" ", strip=True)
        if company.startswith("↳") or not company:
            company = last_company
        else:
            last_company = company
        url = _pick_apply_link(cells[cols["apply"]])
        if not url:
            continue
        loc_cell = cells[cols["location"]] if "location" in cols else None
        locations = []
        if loc_cell is not None:
            details = loc_cell.find("details")
            raw = details.get_text("\n", strip=True) if details else loc_cell.get_text("\n", strip=True)
            locations = [part.strip() for part in re.split(r"\n|;|</br>", raw) if part.strip()
                         and not re.fullmatch(r"\*?\*?\d+ locations\*?\*?", part.strip())]
        age = _parse_age_days(cells[cols["age"]].get_text(strip=True)) if "age" in cols else None
        jobs.append({
            "company": company.replace("🔥", "").strip(),
            "title": cells[cols["title"]].get_text(" ", strip=True).replace("🛂", "").replace("🇺🇸", "").strip(),
            "locations": locations,
            "url": url,
            "age_days": age,
            "category": "",
            "sponsorship": "Does Not Offer Sponsorship" if "🛂" in row_text else "",
            "citizenship": "🇺🇸" in row_text,
            "terms": [],
            "repo": repo,
        })
    return jobs


def _from_readme(text: str, repo: str) -> list[dict]:
    jobs: list[dict] = []

    # HTML tables
    soup = BeautifulSoup(text, "html.parser")
    for table in soup.find_all("table"):
        header_cells = table.find_all("th")
        headers = [th.get_text(" ", strip=True) for th in header_cells]
        rows = [tr.find_all("td") for tr in table.find_all("tr")]
        jobs.extend(_rows_from_cells([r for r in rows if r], headers, repo))

    # Markdown pipe tables
    headers: list[str] | None = None
    md_rows: list[list[BeautifulSoup]] = []

    def flush():
        if headers and md_rows:
            jobs.extend(_rows_from_cells(md_rows, headers, repo))

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            flush()
            headers, md_rows = None, []
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if headers is None:
            headers = [BeautifulSoup(c, "html.parser").get_text(" ", strip=True) for c in cells]
        elif all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        else:
            md_rows.append([BeautifulSoup(c.replace("</br>", "\n").replace("<br>", "\n"), "html.parser")
                            for c in cells])
    flush()
    return jobs


def fetch_repo(repo: str, client: httpx.Client | None = None) -> list[dict]:
    """Fetch and parse all open postings from one GitHub repo ('owner/name')."""
    own_client = client is None
    if own_client:
        headers = {"User-Agent": "applypilot"}
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        client = httpx.Client(timeout=30, follow_redirects=True, headers=headers)
    try:
        data = _fetch_listings_json(client, repo)
        if data is not None:
            return _from_listings_json(data, repo)
        readme = _fetch_readme(client, repo)
        if readme is None:
            log.warning("%s: no listings.json or README found", repo)
            return []
        jobs = _from_readme(readme, repo)
        log.info("%s: parsed %d postings from README", repo, len(jobs))
        return jobs
    finally:
        if own_client:
            client.close()


# ---------------------------------------------------------------------------
# Filtering + storage
# ---------------------------------------------------------------------------

def _matches_any(text: str, patterns: list[str]) -> bool:
    text = text.lower()
    return any(p.lower() in text for p in patterns)


def filter_jobs(jobs: list[dict], cfg: dict) -> list[dict]:
    max_age = cfg.get("max_age_days")
    include = cfg.get("include_titles") or []
    exclude = cfg.get("exclude_titles") or []
    locations = cfg.get("locations") or []
    exclude_locations = cfg.get("exclude_locations") or []
    categories = cfg.get("categories") or []
    skip_no_sponsor = cfg.get("skip_no_sponsorship", False)
    skip_citizen = cfg.get("skip_citizenship_required", False)

    out = []
    for j in jobs:
        if max_age is not None and j["age_days"] is not None and j["age_days"] > max_age:
            continue
        if include and not _matches_any(j["title"], include):
            continue
        if exclude and _matches_any(j["title"], exclude):
            continue
        if locations and not _matches_any(" | ".join(j["locations"]), locations):
            continue
        # Drop only when EVERY listed location is excluded (e.g. "London, UK" but not "London, UK; NYC")
        if exclude_locations and j["locations"] and all(_matches_any(loc, exclude_locations) for loc in j["locations"]):
            continue
        if categories and j["category"] and not _matches_any(j["category"], categories):
            continue
        if skip_no_sponsor and "not offer" in j["sponsorship"].lower():
            continue
        if skip_citizen and (j.get("citizenship") or "citizen" in j["sponsorship"].lower()):
            continue
        out.append(j)
    # Newest first; unknown age last
    out.sort(key=lambda j: j["age_days"] if j["age_days"] is not None else float("inf"))
    return out


def store(jobs: list[dict]) -> tuple[int, int]:
    """Insert jobs; the apply link is both the job URL and application_url."""
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    new = dup = 0
    for j in jobs:
        url = clean_url(j["url"])
        loc = "; ".join(j["locations"])
        desc = f"{j['title']} at {j['company']} ({loc or 'location n/a'}). Curated by github.com/{j['repo']}."
        # The same posting often appears in several lists under slightly different URLs
        # (boards.greenhouse.io vs job-boards.greenhouse.io); skip it if another list has it.
        seen_elsewhere = conn.execute(
            "SELECT 1 FROM jobs WHERE site = ? COLLATE NOCASE AND title = ? COLLATE NOCASE "
            "AND strategy LIKE 'github:%' AND strategy != ? LIMIT 1",
            (j["company"], j["title"], f"github:{j['repo']}"),
        ).fetchone()
        if seen_elsewhere:
            dup += 1
            continue
        cur = conn.execute(
            "INSERT OR IGNORE INTO jobs (url, title, description, location, site, strategy, "
            "discovered_at, application_url) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (url, j["title"], desc, loc, j["company"], f"github:{j['repo']}", now, url),
        )
        if cur.rowcount:
            new += 1
        else:
            dup += 1
    conn.commit()
    return new, dup


def load_github_config() -> dict:
    cfg = (config.load_search_config() or {}).get("github_lists") or {}
    cfg.setdefault("repos", DEFAULT_REPOS)
    cfg.setdefault("max_age_days", 14)
    return cfg


def run_github_discovery(repos: list[str] | None = None, overrides: dict | None = None,
                         dry_run: bool = False) -> dict:
    """Import postings from the configured GitHub repos.

    Returns:
        {"fetched": n, "kept": n, "new": n, "duplicates": n, "by_repo": {repo: kept}}
    """
    cfg = load_github_config()
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    repo_list = repos or cfg["repos"]

    init_db()
    totals = {"fetched": 0, "kept": 0, "new": 0, "duplicates": 0, "by_repo": {}, "jobs": []}
    for entry in repo_list:
        repo = entry["repo"] if isinstance(entry, dict) else entry
        repo = repo.removeprefix("https://github.com/").strip("/")
        raw = fetch_repo(repo)
        kept = filter_jobs(raw, cfg)
        totals["fetched"] += len(raw)
        totals["kept"] += len(kept)
        totals["by_repo"][repo] = len(kept)
        totals["jobs"].extend(kept)
        if not dry_run:
            new, dup = store(kept)
            totals["new"] += new
            totals["duplicates"] += dup
        log.info("%s: %d fetched, %d after filters", repo, len(raw), len(kept))
    return totals
