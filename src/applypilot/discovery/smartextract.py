"""AI-powered smart extraction: discovers jobs from arbitrary websites.

Two-phase approach:
  Phase 1: Lightweight intelligence (JSON-LD, API responses, data-testids, DOM stats)
           -> LLM picks the best extraction strategy
  Phase 2: Only for CSS selectors -- Playwright finds repeating card elements,
           extracts 2-3 examples, sends focused HTML to LLM for selector generation.

JSON-LD and API strategies execute directly from stored data -- no LLM needed.

Sites are loaded from config/sites.yaml, with {query_encoded} and {location_encoded}
placeholders replaced from the user's search configuration.
"""

import json
import logging
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import quote_plus

import yaml
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from applypilot import config
from applypilot.config import CONFIG_DIR
from applypilot.database import init_db, get_stats
from applypilot.llm import get_client

log = logging.getLogger(__name__)

# Fix Windows encoding -- prevents charmap errors on emoji/unicode in job titles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


# -- Location filtering -------------------------------------------------------


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


# -- Site configuration from YAML --------------------------------------------


def load_sites() -> list[dict]:
    """Load scraping target sites from config/sites.yaml."""
    path = CONFIG_DIR / "sites.yaml"
    if not path.exists():
        log.warning("sites.yaml not found at %s", path)
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("sites", [])


def _store_jobs_filtered(
    conn: sqlite3.Connection,
    jobs: list[dict],
    site: str,
    strategy: str,
    accept_locs: list[str],
    reject_locs: list[str],
) -> tuple[int, int]:
    """Store jobs with location filtering. Returns (new, existing)."""
    now = datetime.now(timezone.utc).isoformat()
    new = 0
    existing = 0
    filtered = 0

    for job in jobs:
        url = job.get("url")
        if not url:
            continue
        if not _location_ok(job.get("location"), accept_locs, reject_locs):
            filtered += 1
            continue
        try:
            conn.execute(
                "INSERT INTO jobs (url, title, salary, description, location, site, strategy, discovered_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    url,
                    job.get("title"),
                    job.get("salary"),
                    job.get("description"),
                    job.get("location"),
                    site,
                    strategy,
                    now,
                ),
            )
            new += 1
        except sqlite3.IntegrityError:
            existing += 1

    if filtered:
        log.info("Filtered %d jobs (wrong location)", filtered)
    conn.commit()
    return new, existing


# -- Page intelligence collector ---------------------------------------------


def collect_page_intelligence(url: str, headless: bool = True) -> dict:
    """Load a page with Playwright and collect every signal a scraping engineer
    would look at in DevTools. Returns a structured intelligence report."""
    intel: dict = {
        "url": url,
        "json_ld": [],
        "api_responses": [],
        "data_testids": [],
        "page_title": "",
        "dom_stats": {},
        "card_candidates": [],
    }

    captured_responses: list[dict] = []

    def on_response(response):
        ct = response.headers.get("content-type", "")
        rurl = response.url
        if any(ext in rurl for ext in [".js", ".css", ".png", ".jpg", ".svg", ".woff", ".ico", ".gif", ".webp"]):
            return
        if "json" in ct or "/api/" in rurl or "algolia" in rurl or "graphql" in rurl:
            try:
                body = response.text()
                try:
                    data = json.loads(body)
                except Exception:
                    data = None
                captured_responses.append(
                    {
                        "url": rurl,
                        "status": response.status,
                        "size": len(body),
                        "data": data,
                    }
                )
            except Exception:
                pass

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(user_agent=UA)
        page.on("response", on_response)

        page.goto(url, timeout=60000)
        page.wait_for_load_state("networkidle")

        intel["page_title"] = page.title()

        # 1. JSON-LD
        for el in page.query_selector_all('script[type="application/ld+json"]'):
            try:
                data = json.loads(el.inner_text())
                intel["json_ld"].append(data)
            except Exception:
                pass

        # 2. __NEXT_DATA__
        next_data = page.query_selector("script#__NEXT_DATA__")
        if next_data:
            try:
                intel["next_data"] = json.loads(next_data.inner_text())
            except Exception:
                pass

        # 3. data-testid attributes
        intel["data_testids"] = page.evaluate("""
            () => {
                const els = document.querySelectorAll('[data-testid]');
                const results = [];
                els.forEach(el => {
                    results.push({
                        testid: el.getAttribute('data-testid'),
                        tag: el.tagName.toLowerCase(),
                        text: el.innerText?.slice(0, 80) || ''
                    });
                });
                return results.slice(0, 50);
            }
        """)

        # 4. DOM stats
        intel["dom_stats"] = page.evaluate("""
            () => {
                const body = document.body;
                return {
                    total_elements: body.querySelectorAll('*').length,
                    links: body.querySelectorAll('a[href]').length,
                    headings: body.querySelectorAll('h1,h2,h3,h4').length,
                    lists: body.querySelectorAll('ul,ol').length,
                    tables: body.querySelectorAll('table').length,
                    articles: body.querySelectorAll('article').length,
                    has_data_ids: body.querySelectorAll('[data-id]').length,
                };
            }
        """)

        # 5. Find repeating card-like elements
        intel["card_candidates"] = page.evaluate("""
            () => {
                const candidates = [];
                const allParents = document.querySelectorAll('*');

                for (const parent of allParents) {
                    const children = Array.from(parent.children);
                    if (children.length < 3) continue;

                    const tagCounts = {};
                    children.forEach(c => {
                        const key = c.tagName;
                        tagCounts[key] = (tagCounts[key] || 0) + 1;
                    });

                    const dominant = Object.entries(tagCounts).sort((a,b) => b[1]-a[1])[0];
                    if (!dominant || dominant[1] < 3) continue;

                    const repeatingChildren = children.filter(c => c.tagName === dominant[0]);
                    const withText = repeatingChildren.filter(c => c.innerText?.trim().length > 20);
                    if (withText.length < 3) continue;

                    const withLinks = withText.filter(c => c.querySelector('a[href]'));
                    const score = withLinks.length * 2 + withText.length;

                    const parentId = parent.id ? '#' + parent.id : '';
                    const parentClasses = Array.from(parent.classList).filter(c => c.length < 30).slice(0, 3).join('.');
                    const parentTag = parent.tagName.toLowerCase();
                    const parentSelector = parentTag + (parentId || (parentClasses ? '.' + parentClasses : ''));

                    const childTag = dominant[0].toLowerCase();
                    const sampleChild = withText[0];
                    const childClasses = Array.from(sampleChild.classList).filter(c => c.length < 30).slice(0, 3).join('.');
                    const childSelector = childTag + (childClasses ? '.' + childClasses : '');

                    const examples = withText.slice(0, 3).map(c => {
                        const clone = c.cloneNode(true);
                        clone.querySelectorAll('script,style,svg,noscript').forEach(el => el.remove());
                        const html = clone.outerHTML;
                        return html.length > 5000 ? html.slice(0, 5000) + '...' : html;
                    });

                    candidates.push({
                        parent_selector: parentSelector,
                        child_selector: childSelector,
                        child_tag: childTag,
                        total_children: repeatingChildren.length,
                        with_text: withText.length,
                        with_links: withLinks.length,
                        score: score,
                        examples: examples,
                    });
                }

                candidates.sort((a,b) => b.score - a.score);
                return candidates.slice(0, 3);
            }
        """)

        # Capture full rendered HTML
        intel["full_html"] = page.content()

        browser.close()

    # Process API responses
    for resp in captured_responses:
        summary: dict = {
            "url": resp["url"][:200],
            "status": resp["status"],
            "size": resp["size"],
            "_raw_data": resp.get("data"),
        }
        data = resp.get("data")
        if data:
            if isinstance(data, list) and data:
                summary["type"] = f"array[{len(data)}]"
                if isinstance(data[0], dict):
                    summary["first_item_keys"] = list(data[0].keys())[:20]
                    summary["first_item_sample"] = {k: str(v)[:100] for k, v in list(data[0].items())[:8]}
            elif isinstance(data, dict):
                summary["type"] = "object"
                summary["keys"] = list(data.keys())[:20]

                def _explore_nested(obj, path_prefix, depth=0):
                    if depth > 3 or not isinstance(obj, dict):
                        return
                    for key in list(obj.keys())[:15]:
                        val = obj[key]
                        path = f"{path_prefix}.{key}" if path_prefix else key
                        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                            info = {
                                "count": len(val),
                                "first_item_keys": list(val[0].keys())[:20],
                                "first_item_sample": {k: str(v)[:200] for k, v in list(val[0].items())[:8]},
                            }
                            for subkey in list(val[0].keys())[:10]:
                                subval = val[0][subkey]
                                if isinstance(subval, list) and len(subval) > 0 and isinstance(subval[0], dict):
                                    info[f"first_item.{subkey}"] = {
                                        "count": len(subval),
                                        "first_item_keys": list(subval[0].keys())[:15],
                                        "first_item_sample": {k: str(v)[:100] for k, v in list(subval[0].items())[:8]},
                                    }
                                elif isinstance(subval, dict):
                                    info[f"first_item.{subkey}"] = {
                                        "type": "object",
                                        "keys": list(subval.keys())[:15],
                                        "sample": {k: str(v)[:150] for k, v in list(subval.items())[:8]},
                                    }
                            summary[f"nested_{path}"] = info
                        elif isinstance(val, dict) and depth < 3:
                            _explore_nested(val, path, depth + 1)

                _explore_nested(data, "")
        intel["api_responses"].append(summary)

    return intel


# -- Judge: filter API responses ---------------------------------------------

JUDGE_PROMPT = """You are filtering intercepted API responses from a job listings website.
Decide if this API response contains actual job listing data (titles, companies, locations, etc).

API Response Summary:
  URL: {url}
  Status: {status}
  Size: {size} chars
  Type: {type}
  Keys/Fields: {fields}
  Sample: {sample}

Is this job listing data? Answer in under 10 words. Return ONLY valid JSON:
{{"relevant": true, "reason": "job objects with title/company"}}
or
{{"relevant": false, "reason": "auth endpoint"}}

No explanation, no markdown, no thinking."""


def judge_api_responses(api_responses: list[dict]) -> list[dict]:
    """Use the LLM to filter API responses, keeping only job-relevant ones."""
    if not api_responses:
        return []

    client = get_client()
    relevant: list[dict] = []

    for resp in api_responses:
        fields = ""
        sample = ""
        resp_type = resp.get("type", "unknown")
        if "first_item_keys" in resp:
            fields = str(resp["first_item_keys"])
            sample = json.dumps(resp.get("first_item_sample", {}), indent=2)[:500]
        elif "keys" in resp:
            fields = str(resp["keys"])
            for k, v in resp.items():
                if k.startswith("nested_"):
                    fields += f"\n  .{k.replace('nested_', '')}: {v.get('count', '?')} items, keys={v.get('first_item_keys', '?')}"
                    sample = json.dumps(v.get("first_item_sample", {}), indent=2)[:500]
        else:
            fields = "no structured data"

        prompt = JUDGE_PROMPT.format(
            url=resp.get("url", "?")[:200],
            status=resp.get("status", "?"),
            size=resp.get("size", "?"),
            type=resp_type,
            fields=fields,
            sample=sample or "n/a",
        )

        try:
            raw = client.chat([{"role": "user", "content": prompt}], max_output_tokens=1024)
            verdict = extract_json(raw)
            is_relevant = verdict.get("relevant", False)
            reason = verdict.get("reason", "?")
            log.info("Judge: %s -> %s (%s)", resp.get("url", "?")[:80], "KEEP" if is_relevant else "DROP", reason)
            if is_relevant:
                relevant.append(resp)
        except Exception as e:
            log.warning("Judge ERROR for %s: %s -- keeping", resp.get("url", "?")[:80], e)
            relevant.append(resp)

    return relevant


# -- Phase 1: strategy selection ---------------------------------------------


def format_strategy_briefing(intel: dict) -> str:
    """Lightweight briefing for strategy selection. No raw DOM."""
    sections: list[str] = []
    sections.append(f"PAGE: {intel['url']}")
    sections.append(f"TITLE: {intel['page_title']}")

    # JSON-LD
    if intel["json_ld"]:
        job_postings = [j for j in intel["json_ld"] if isinstance(j, dict) and j.get("@type") == "JobPosting"]
        other = [j for j in intel["json_ld"] if not (isinstance(j, dict) and j.get("@type") == "JobPosting")]
        if job_postings:
            sections.append(f"\nJSON-LD: {len(job_postings)} JobPosting entries found (usable!)")
            sections.append(f"First JobPosting:\n{json.dumps(job_postings[0], indent=2)[:3000]}")
        else:
            sections.append("\nJSON-LD: NO JobPosting entries (json_ld strategy will NOT work)")
        if other:
            types = [j.get("@type", "?") if isinstance(j, dict) else "?" for j in other]
            sections.append(f"Other JSON-LD types (NOT job data): {types}")
    else:
        sections.append("\nJSON-LD: none")

    # API responses
    if intel["api_responses"]:
        sections.append(f"\nAPI RESPONSES INTERCEPTED: {len(intel['api_responses'])} calls")
        for resp in intel["api_responses"]:
            sections.append(f"\n  URL: {resp['url']}")
            sections.append(
                f"  Status: {resp['status']} | Size: {resp['size']:,} chars | Type: {resp.get('type', '?')}"
            )
            if "first_item_keys" in resp:
                sections.append(f"  Item keys: {resp['first_item_keys']}")
                sections.append(f"  Sample: {json.dumps(resp.get('first_item_sample', {}), indent=2)[:1000]}")
            if "keys" in resp:
                sections.append(f"  Object keys: {resp['keys']}")
            for k, v in resp.items():
                if k.startswith("nested_"):
                    arr_name = k.replace("nested_", "")
                    sections.append(f"  .{arr_name}: array of {v['count']} items")
                    sections.append(f"    Item keys: {v['first_item_keys']}")
                    sections.append(f"    Sample: {json.dumps(v.get('first_item_sample', {}), indent=2)[:1000]}")
                    for sk, sv in v.items():
                        if sk.startswith("first_item.") and isinstance(sv, dict):
                            sub_name = sk.replace("first_item.", "")
                            if "count" in sv:
                                sections.append(f"    .{arr_name}[0].{sub_name}: array of {sv['count']} items")
                                sections.append(f"      Item keys: {sv['first_item_keys']}")
                                sections.append(
                                    f"      Sample: {json.dumps(sv.get('first_item_sample', {}), indent=2)[:1500]}"
                                )
                            elif "keys" in sv:
                                sections.append(f"    .{arr_name}[0].{sub_name}: object with keys {sv['keys']}")
                                sections.append(f"      Sample: {json.dumps(sv.get('sample', {}), indent=2)[:1500]}")
    else:
        sections.append("\nAPI RESPONSES: none intercepted")

    # data-testid
    if intel["data_testids"]:
        sections.append(f"\nDATA-TESTID ATTRIBUTES: {len(intel['data_testids'])} elements")
        for dt in intel["data_testids"][:15]:
            text_preview = dt["text"].replace("\n", " ")[:60]
            sections.append(f'  <{dt["tag"]} data-testid="{dt["testid"]}"> {text_preview}')
    else:
        sections.append("\nDATA-TESTID: none found")

    # DOM stats
    stats = intel.get("dom_stats", {})
    sections.append(
        f"\nDOM STATS: {stats.get('total_elements', '?')} elements, "
        f"{stats.get('links', '?')} links, {stats.get('headings', '?')} headings, "
        f"{stats.get('tables', '?')} tables, {stats.get('articles', '?')} articles, "
        f"{stats.get('has_data_ids', '?')} data-id elements"
    )

    # Card candidates
    if intel["card_candidates"]:
        sections.append(f"\nREPEATING ELEMENTS DETECTED: {len(intel['card_candidates'])} candidate groups")
        for i, cand in enumerate(intel["card_candidates"]):
            sections.append(
                f"  [{i}] parent={cand['parent_selector']} child={cand['child_selector']} "
                f"count={cand['total_children']} with_text={cand['with_text']} with_links={cand['with_links']}"
            )
    else:
        sections.append("\nREPEATING ELEMENTS: none detected")

    return "\n".join(sections)


STRATEGY_PROMPT = """You are analyzing a job listings page to pick the best extraction strategy.

Below is a lightweight intelligence briefing -- JSON-LD data, intercepted API responses, data-testid attributes, and DOM statistics. NO raw DOM HTML is included.

Pick the BEST strategy:

1. "json_ld" -- ONLY if briefing shows JobPosting JSON-LD entries (it will say "usable!")
2. "api_response" -- ONLY if an intercepted API response has job-like fields (name, title, salary, description, location, slug)
3. "css_selectors" -- when neither JSON-LD nor API data has job data

HOW TO THINK:
- If the briefing says "JSON-LD: NO JobPosting entries" or "json_ld strategy will NOT work", do NOT pick json_ld.
- For api_response: "url_pattern" must be a substring that matches one of the INTERCEPTED API URLs listed above (not the page URL!). Copy a unique part of the API URL.
- For api_response: "items_path" must point to the ARRAY of items, not a single item. Use dot notation with [n] ONLY for traversing into a specific index to reach an inner array. Example: if data is {{"results": [{{"hits": [...]}}]}}, items_path is "results[0].hits" to reach the hits array.
- For api_response: field paths (title, salary, etc.) are RELATIVE TO EACH ITEM in the array. If items are nested objects like {{"_source": {{"Title": "..."}}}}, use "_source.Title" for the title field.
- For css_selectors: just return {{"strategy":"css_selectors","reasoning":"...","extraction":{{}}}} -- selectors will be generated in a separate focused step.

Return ONLY valid JSON:

For json_ld:
{{"strategy":"json_ld","reasoning":"...","extraction":{{"title":"title","salary":"baseSalary_path_or_null","description":"description","location":"jobLocation[0].address.addressCountry","url":"url_field"}}}}

For api_response:
{{"strategy":"api_response","reasoning":"...","extraction":{{"url_pattern":"actual.url.substring","items_path":"path.to.the.array","title":"field_in_each_item","salary":"salary_field_or_null","description":"description_field_or_null","location":"location_path","url":"url_field"}}}}

For css_selectors:
{{"strategy":"css_selectors","reasoning":"...","extraction":{{}}}}

Keep reasoning under 20 words. No explanation, no markdown, no code fences.

INTELLIGENCE BRIEFING:
{briefing}"""


# -- Card HTML cleaning (allowlist approach) ----------------------------------

_ALLOWED_ATTRS = {
    "id",
    "href",
    "data-testid",
    "data-id",
    "data-type",
    "data-slug",
    "role",
    "aria-label",
    "aria-labelledby",
    "type",
    "name",
    "for",
}
_ALLOWED_PREFIXES = ("data-", "aria-")
_UTILITY_CLASS_RE = re.compile(
    r"^("
    r"[a-z]{1,2}-\d+|"
    r"[a-z]{1,3}-[a-z]{1,3}-\d+|"
    r"col-\d+|"
    r"d-\w+|"
    r"align-\w+|justify-\w+|"
    r"flex-\w+|order-\d+|"
    r"text-\w+|font-\w+|"
    r"bg-\w+|border-\w+|"
    r"rounded-?\w*|shadow-?\w*|"
    r"w-\d+|h-\d+|"
    r"position-\w+|overflow-\w+|"
    r"float-\w+|clearfix|"
    r"visible-\w+|invisible|"
    r"sr-only|"
    r"css-[a-z0-9]+|"
    r"sc-[a-zA-Z]+|"
    r"sc-[a-f0-9]+-\d+"
    r")$"
)


def clean_card_html(html: str) -> str:
    """Strip layout noise from card HTML, keep only what the LLM needs for selectors."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(True):
        new_attrs: dict = {}
        for attr, val in list(tag.attrs.items()):
            if attr in _ALLOWED_ATTRS or any(attr.startswith(p) for p in _ALLOWED_PREFIXES):
                new_attrs[attr] = val
            elif attr == "class":
                classes = val if isinstance(val, list) else val.split()
                kept = [c for c in classes if not _UTILITY_CLASS_RE.match(c)]
                if kept:
                    new_attrs["class"] = kept
        tag.attrs = new_attrs

    return str(soup)


def clean_page_html(html: str, max_chars: int = 150_000) -> str:
    """Strip full page HTML to essential structure for LLM card detection."""
    soup = BeautifulSoup(html, "html.parser")

    main = soup.find("main") or soup.find(attrs={"role": "main"})
    if main and len(str(main)) > 1000:
        soup = BeautifulSoup(str(main), "html.parser")

    for tag in soup.find_all(["script", "style", "svg", "noscript", "iframe", "link", "meta", "head", "footer", "nav"]):
        tag.decompose()

    for tag in soup.find_all(True):
        new_attrs: dict = {}
        for attr, val in list(tag.attrs.items()):
            if attr in _ALLOWED_ATTRS or any(attr.startswith(p) for p in _ALLOWED_PREFIXES):
                new_attrs[attr] = val
            elif attr == "class":
                classes = val if isinstance(val, list) else val.split()
                kept = [c for c in classes if not _UTILITY_CLASS_RE.match(c)]
                if kept:
                    new_attrs["class"] = kept
        tag.attrs = new_attrs

    for tag in soup.find_all(True):
        if not tag.get_text(strip=True) and not tag.find("img") and not tag.find("a"):
            tag.decompose()

    result = str(soup)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n<!-- TRUNCATED -->"
    return result


# -- Phase 2: CSS selector generation ----------------------------------------

FULL_PAGE_SELECTOR_PROMPT = """You are a senior web scraping engineer. Below is the cleaned HTML of a job listings page.

Your task:
1. Find the repeating HTML elements that represent individual job listings
2. Generate CSS selectors to extract data from them

Return a JSON object:
- "job_card": CSS selector matching each job card (MUST match ALL cards on the page)
- "title": selector RELATIVE to the card for the job title
- "salary": selector relative to card for salary, or null
- "description": selector relative to card for description snippet, or null
- "location": selector relative to card for location, or null
- "url": selector relative to card for the link (<a> tag) to the job detail page

Selector rules:
- SIMPLEST wins. A single attribute selector like [data-testid="job-card"] is better than a multi-level path like li > div > [data-testid="job-card"]. Do NOT add parent/ancestor selectors unless the target is ambiguous without them.
- For data-testid/data-id with DYNAMIC values (e.g. data-testid="card-123"), use prefix matching: [data-testid^="card-"]
- For data-testid with STATIC values (e.g. data-testid="job-card"), use exact: [data-testid="job-card"]
- Prefer semantic HTML: article, section, h2, h3 over div
- NEVER use hashed/generated classes: sc-*, css-*, random 5-8 char strings like "fJyWhK"
- Max 2 levels deep. One level is best.
- The "url" selector should target an <a> element (we extract its href attribute)
- If the page has NO job listings visible, return {{"error": "no job listings found"}}

Return ONLY valid JSON, no explanation, no markdown.

PAGE HTML:
{page_html}"""


# -- LLM helpers -------------------------------------------------------------


def ask_llm(prompt: str) -> tuple[str, float, dict]:
    """Send prompt to LLM. Returns (response_text, seconds_taken, metadata)."""
    client = get_client()
    t0 = time.time()
    text = client.chat([{"role": "user", "content": prompt}], max_output_tokens=4096)
    elapsed = time.time() - t0
    meta = {
        "finish_reason": "stop",
        "prompt_chars": len(prompt),
        "response_chars": len(text),
    }
    return text, elapsed, meta


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling think tags and code fences."""
    if "<think>" in text:
        after = text.split("</think>")[-1].strip()
        if after:
            text = after
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    text = text.strip()
    text = re.sub(r'\\([^"\\\/bfnrtu])', r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    while text.endswith("}") or text.endswith("]"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            text = text[:-1].rstrip()
    raise json.JSONDecodeError("Could not parse JSON", text, 0)


# -- JSON path resolution ---------------------------------------------------


def resolve_json_path_raw(data, path: str):
    """Navigate a JSON path and return whatever is there (including lists/dicts)."""
    if not path or not data:
        return None
    try:
        current = data
        for part in path.replace("[", ".[").split("."):
            if not part:
                continue
            if part.startswith("[") and part.endswith("]"):
                idx = int(part[1:-1])
                current = current[idx]
            else:
                current = current[part]
        return current
    except (KeyError, IndexError, TypeError):
        return None


def resolve_json_path(data, path: str):
    """Simple JSON path resolver with type coercion for display."""
    if not path or not data:
        return None
    try:
        current = data
        for part in path.replace("[", ".[").split("."):
            if not part:
                continue
            if part.startswith("[") and part.endswith("]"):
                idx = int(part[1:-1])
                current = current[idx]
            else:
                current = current[part]
        if isinstance(current, (str, int, float)):
            return str(current) if not isinstance(current, str) else current
        elif isinstance(current, dict):
            return current.get("name", current.get("text", str(current)[:100]))
        elif isinstance(current, list):
            if current and isinstance(current[0], dict):
                return ", ".join(str(item.get("name", item.get("text", ""))) for item in current[:3])
            return ", ".join(str(x) for x in current[:3])
        return str(current) if current else None
    except (KeyError, IndexError, TypeError):
        return None


# -- Extraction executors ----------------------------------------------------


def execute_json_ld(intel: dict, plan: dict) -> list[dict]:
    """Extract jobs from JSON-LD JobPosting entries."""
    ext = plan["extraction"]
    jobs: list[dict] = []
    for entry in intel["json_ld"]:
        if not isinstance(entry, dict) or entry.get("@type") != "JobPosting":
            continue
        job: dict = {}
        for field in ["title", "salary", "description", "location", "url"]:
            path = ext.get(field)
            if not path or path == "null":
                job[field] = None
                continue
            job[field] = resolve_json_path(entry, path)
        jobs.append(job)
    return jobs


def execute_api_response(intel: dict, plan: dict) -> list[dict]:
    """Extract jobs from intercepted API response data."""
    ext = plan["extraction"]
    url_pattern = ext.get("url_pattern", "")

    target_data = None
    for resp in intel["api_responses"]:
        if url_pattern in resp.get("url", ""):
            target_data = resp.get("_raw_data")
            break

    if not target_data:
        log.warning("Could not find stored API response matching: %s", url_pattern)
        return []

    items_path = ext.get("items_path", "")
    items = resolve_json_path_raw(target_data, items_path)
    if not isinstance(items, list):
        log.warning("items_path '%s' did not resolve to a list (got %s)", items_path, type(items).__name__)
        return []

    jobs: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        job: dict = {}
        for field in ["title", "salary", "description", "location", "url"]:
            path = ext.get(field)
            if not path or path == "null":
                job[field] = None
                continue
            job[field] = resolve_json_path(item, path)
        jobs.append(job)
    return jobs


def execute_css_selectors(intel: dict) -> tuple[dict, list[dict]]:
    """Phase 2: Send full cleaned page HTML to LLM for card detection + selector generation.
    Returns (selectors, jobs)."""
    full_html = intel.get("full_html", "")
    if not full_html:
        log.warning("No page HTML captured")
        return {}, []

    cleaned = clean_page_html(full_html)
    log.info("Page HTML: %s -> %s chars", f"{len(full_html):,}", f"{len(cleaned):,}")

    prompt = FULL_PAGE_SELECTOR_PROMPT.format(page_html=cleaned)

    try:
        raw, elapsed, meta = ask_llm(prompt)
    except Exception as e:
        log.error("LLM_ERROR in Phase 2: %s", e)
        return {}, []

    log.info("Phase 2 LLM: %d chars, %.1fs", meta["response_chars"], elapsed)

    try:
        selectors = extract_json(raw)
    except Exception as e:
        log.error("PARSE_ERROR in Phase 2: %s | raw: %s", e, raw[:500])
        return {}, []

    if "error" in selectors:
        log.warning("LLM: %s", selectors["error"])
        return selectors, []

    log.info("Selectors: %s", selectors)

    # Apply selectors to the ORIGINAL full_html
    soup = BeautifulSoup(full_html, "html.parser")
    card_sel = selectors.get("job_card", "NONE")
    try:
        cards = soup.select(card_sel)
    except Exception as e:
        log.error("Invalid card selector '%s': %s", card_sel, e)
        return selectors, []

    log.info("Matched %d cards", len(cards))

    jobs: list[dict] = []
    for card in cards:
        job: dict = {}
        for field in ["title", "salary", "description", "location", "url"]:
            sel = selectors.get(field)
            if not sel or sel == "null":
                job[field] = None
                continue
            try:
                el = card.select_one(sel)
            except Exception:
                job[field] = None
                continue
            if el:
                job[field] = el.get("href") if field == "url" else el.get_text(strip=True)
            else:
                job[field] = None
        jobs.append(job)
    return selectors, jobs


# -- Main per-site extraction ------------------------------------------------


def _run_one_site(name: str, url: str) -> dict:
    """Run full smart extraction pipeline on one site URL."""
    log.info("=" * 60)
    log.info("%s: %s", name, url)

    # Step 1: Collect intelligence
    log.info("[1] Collecting page intelligence...")
    t0 = time.time()
    intel = collect_page_intelligence(url)
    collect_time = time.time() - t0
    log.info(
        "Done in %.1fs | JSON-LD: %d | API: %d | testids: %d | cards: %d",
        collect_time,
        len(intel["json_ld"]),
        len(intel["api_responses"]),
        len(intel["data_testids"]),
        len(intel["card_candidates"]),
    )

    # Headful retry if page content is tiny
    full_html = intel.get("full_html", "")
    cleaned_check = clean_page_html(full_html) if full_html else ""
    _captcha_signals = [
        "captcha",
        "are you a human",
        "verify you",
        "unusual requests",
        "access denied",
        "please verify",
        "bot detection",
    ]
    _is_captcha = any(s in full_html.lower() for s in _captcha_signals) if full_html else False
    if len(cleaned_check) < 5000 and full_html and not _is_captcha:
        log.info("Cleaned HTML only %s chars -- retrying headful...", f"{len(cleaned_check):,}")
        intel = collect_page_intelligence(url, headless=False)
        collect_time = time.time() - t0
        log.info(
            "Headful done in %.1fs | JSON-LD: %d | API: %d",
            collect_time,
            len(intel["json_ld"]),
            len(intel["api_responses"]),
        )
    elif _is_captcha:
        log.warning("CAPTCHA/rate-limit detected -- skipping headful retry")

    # Step 1.5: Judge filters API responses
    if intel["api_responses"]:
        log.info("[1.5] Judge filtering API responses...")
        intel["api_responses"] = judge_api_responses(intel["api_responses"])
        log.info("Kept %d relevant responses", len(intel["api_responses"]))

    # Step 2: Strategy selection
    briefing = format_strategy_briefing(intel)
    log.info("[2] Phase 1: Strategy selection (%s chars briefing)", f"{len(briefing):,}")

    prompt = STRATEGY_PROMPT.format(briefing=briefing)
    try:
        raw, elapsed, meta = ask_llm(prompt)
    except Exception as e:
        log.error("LLM_ERROR: %s", e)
        return {"name": name, "status": "LLM_ERROR", "error": str(e)}

    log.info("LLM: %d chars, %.1fs", meta["response_chars"], elapsed)

    try:
        plan = extract_json(raw)
    except Exception as e:
        log.error("PARSE_ERROR: %s | raw: %s", e, raw[:500])
        return {"name": name, "status": "PARSE_ERROR", "error": str(e), "raw": raw}

    strategy = plan.get("strategy", "?")
    reasoning = plan.get("reasoning", "?")
    log.info("Strategy: %s | Reasoning: %s", strategy, reasoning)

    # Step 3: Execute
    log.info("[3] Executing %s...", strategy)
    try:
        if strategy == "json_ld":
            log.info("Extraction plan: %s", json.dumps(plan.get("extraction", {}))[:300])
            jobs = execute_json_ld(intel, plan)
        elif strategy == "api_response":
            log.info("Extraction plan: %s", json.dumps(plan.get("extraction", {}))[:300])
            jobs = execute_api_response(intel, plan)
        elif strategy == "css_selectors":
            log.info("-> Phase 2: Generating selectors from card examples...")
            selectors, jobs = execute_css_selectors(intel)
            plan["extraction"] = selectors
        else:
            log.warning("Unknown strategy: %s", strategy)
            jobs = []
    except Exception as e:
        log.error("EXECUTION_ERROR: %s", e)
        return {"name": name, "status": "EXEC_ERROR", "error": str(e), "plan": plan}

    # Step 4: Report
    titles = sum(1 for j in jobs if j.get("title"))
    total = len(jobs)
    status = "PASS" if total > 0 and titles / max(total, 1) >= 0.8 else "FAIL" if total == 0 else "PARTIAL"

    urls = sum(1 for j in jobs if j.get("url"))
    salaries = sum(1 for j in jobs if j.get("salary"))
    descs = sum(1 for j in jobs if j.get("description"))
    log.info(
        "RESULT: %s -- %d jobs, %d titles, %d urls, %d salaries, %d descriptions",
        status,
        total,
        titles,
        urls,
        salaries,
        descs,
    )

    for j in jobs[:3]:
        log.info(
            "  - %s | loc: %s | salary: %s",
            str(j.get("title") or "?")[:55],
            str(j.get("location") or "?")[:25],
            str(j.get("salary") or "-")[:20],
        )

    return {
        "name": name,
        "status": status,
        "strategy": strategy,
        "total": total,
        "titles": titles,
        "plan": plan,
        "jobs": jobs,
        "sample": jobs[:5],
    }


# -- Target building --------------------------------------------------------


def build_scrape_targets(
    sites: list[dict] | None = None,
    search_cfg: dict | None = None,
) -> list[dict]:
    """Build the full list of (name, url) targets from sites + search config queries.

    - "search" sites get expanded: 1 URL per query from search config
    - "static" sites get scraped once as-is

    Placeholders in URLs:
      {query_encoded} -> URL-encoded search query
      {location_encoded} -> URL-encoded location
      {query} -> raw search query (for simple substitution)
    """
    if sites is None:
        sites = load_sites()
    if search_cfg is None:
        search_cfg = config.load_search_config()

    queries_cfg = search_cfg.get("queries", [])
    queries = [q["query"] for q in queries_cfg]
    locs = search_cfg.get("locations", [])
    default_location = locs[0]["location"] if locs else ""

    targets: list[dict] = []

    for site in sites:
        site_url = site.get("url", "")
        site_name = site.get("name", "Unknown")
        site_type = site.get("type", "static")

        if site_type == "search" and queries:
            for query in queries:
                expanded_url = site_url
                expanded_url = expanded_url.replace("{query_encoded}", quote_plus(query))
                expanded_url = expanded_url.replace("{query}", quote_plus(query))
                expanded_url = expanded_url.replace("{location_encoded}", quote_plus(default_location))
                targets.append(
                    {
                        "name": site_name,
                        "url": expanded_url,
                        "query": query,
                    }
                )
        else:
            expanded_url = site_url
            expanded_url = expanded_url.replace("{location_encoded}", quote_plus(default_location))
            targets.append(
                {
                    "name": site_name,
                    "url": expanded_url,
                    "query": None,
                }
            )

    return targets


# -- Run all sites -----------------------------------------------------------


def _run_all(
    targets: list[dict],
    accept_locs: list[str],
    reject_locs: list[str],
    workers: int = 1,
) -> dict:
    """Run smart extract on all targets.

    Sequential by default. When workers > 1, scrapes multiple sites in parallel
    using ThreadPoolExecutor. DB storage is still serialized after each result.
    """
    conn = init_db()
    pre_stats = get_stats(conn)
    log.info(
        "Database: %d jobs already stored, %d pending detail scrape", pre_stats["total"], pre_stats["pending_detail"]
    )

    results: list[dict] = []
    total_new = 0
    total_existing = 0

    def _process_result(r: dict, target: dict) -> None:
        nonlocal total_new, total_existing
        jobs = r.get("jobs", [])
        if jobs:
            new, existing = _store_jobs_filtered(
                conn, jobs, target["name"], r.get("strategy", "?"), accept_locs, reject_locs
            )
            total_new += new
            total_existing += existing
            log.info("DB: +%d new, %d already existed", new, existing)

    if workers > 1 and len(targets) > 1:
        # Parallel mode
        with ThreadPoolExecutor(max_workers=min(workers, len(targets))) as pool:
            future_to_target = {pool.submit(_run_one_site, target["name"], target["url"]): target for target in targets}
            for future in as_completed(future_to_target):
                target = future_to_target[future]
                r = future.result()
                results.append(r)
                _process_result(r, target)
    else:
        # Sequential mode (default)
        for i, target in enumerate(targets):
            label = target["name"]
            if target.get("query"):
                label = f"{target['name']} [{target['query']}]"
            log.info("[%d/%d] %s", i + 1, len(targets), label)

            r = _run_one_site(target["name"], target["url"])
            results.append(r)
            _process_result(r, target)

    # Summary
    for r in results:
        strategy = r.get("strategy", "?")
        if r["status"] in ("PASS", "PARTIAL", "FAIL"):
            detail = f"{r['total']} jobs, {r['titles']} titles, strategy={strategy}"
        else:
            detail = r.get("error", "")[:60]
        log.info("%-10s | %-25s | %s", r["status"], r["name"], detail)

    passed = sum(1 for r in results if r["status"] == "PASS")
    log.info("%d/%d PASS", passed, len(results))

    return {"total_new": total_new, "total_existing": total_existing, "passed": passed, "total": len(results)}


# -- Public entry point ------------------------------------------------------


def run_smart_extract(
    sites: list[dict] | None = None,
    workers: int = 1,
) -> dict:
    """Main entry point for AI-powered smart extraction.

    Loads sites from config/sites.yaml and search queries from the user's
    search config, then runs the extraction pipeline on all targets.

    Args:
        sites: Override the site list. If None, loads from YAML.
        workers: Number of parallel threads for site scraping. Default 1 (sequential).

    Returns:
        Dict with stats: total_new, total_existing, passed, total.
    """
    search_cfg = config.load_search_config()
    accept_locs, reject_locs = _load_location_filter(search_cfg)

    targets = build_scrape_targets(sites=sites, search_cfg=search_cfg)

    if not targets:
        log.warning("No scrape targets configured. Create config/sites.yaml and searches.yaml.")
        return {"total_new": 0, "total_existing": 0, "passed": 0, "total": 0}

    search_sites = sum(1 for s in (sites or load_sites()) if s.get("type") == "search")
    static_sites = sum(1 for s in (sites or load_sites()) if s.get("type") != "search")
    log.info(
        "Sites: %d searchable, %d static | Total targets: %d (workers=%d)",
        search_sites,
        static_sites,
        len(targets),
        workers,
    )

    return _run_all(targets, accept_locs, reject_locs, workers=workers)
