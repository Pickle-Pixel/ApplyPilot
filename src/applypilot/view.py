"""ApplyPilot HTML Dashboard Generator.

Generates a self-contained HTML dashboard with:
  - Summary stats (total, enriched, scored, high-fit)
  - Score distribution bar chart
  - Jobs-by-source breakdown
  - Filterable job cards grouped by score
  - Client-side search and score filtering
"""

from __future__ import annotations

import os
import webbrowser
from datetime import datetime
from html import escape
from pathlib import Path

from rich.console import Console

from applypilot.config import APP_DIR, DB_PATH
from applypilot.database import get_connection

console = Console()


def generate_dashboard(output_path: str | None = None) -> str:
    """Generate an HTML dashboard of all jobs with fit scores.

    Args:
        output_path: Where to write the HTML file. Defaults to ~/.applypilot/dashboard.html.

    Returns:
        Absolute path to the generated HTML file.
    """
    out = Path(output_path) if output_path else APP_DIR / "dashboard.html"

    conn = get_connection()

    # Stats
    total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    ready = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE full_description IS NOT NULL AND application_url IS NOT NULL"
    ).fetchone()[0]
    scored = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE fit_score IS NOT NULL"
    ).fetchone()[0]
    high_fit = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE fit_score >= 7"
    ).fetchone()[0]

    # Score distribution
    score_dist: dict[int, int] = {}
    if scored:
        rows = conn.execute(
            "SELECT fit_score, COUNT(*) FROM jobs "
            "WHERE fit_score IS NOT NULL "
            "GROUP BY fit_score ORDER BY fit_score DESC"
        ).fetchall()
        for r in rows:
            score_dist[r[0]] = r[1]

    # Site stats
    site_stats = conn.execute("""
        SELECT site,
               COUNT(*) as total,
               SUM(CASE WHEN fit_score >= 7 THEN 1 ELSE 0 END) as high_fit,
               SUM(CASE WHEN fit_score BETWEEN 5 AND 6 THEN 1 ELSE 0 END) as mid_fit,
               SUM(CASE WHEN fit_score < 5 AND fit_score IS NOT NULL THEN 1 ELSE 0 END) as low_fit,
               SUM(CASE WHEN fit_score IS NULL THEN 1 ELSE 0 END) as unscored,
               ROUND(AVG(fit_score), 1) as avg_score
        FROM jobs GROUP BY site ORDER BY high_fit DESC, total DESC
    """).fetchall()

    # All scored jobs (5+), ordered by score desc
    jobs = conn.execute("""
        SELECT url, title, salary, description, location, site, strategy,
               full_description, application_url, detail_error,
               fit_score, score_reasoning
        FROM jobs
        WHERE fit_score >= 5
        ORDER BY fit_score DESC, site, title
    """).fetchall()

    # Successfully submitted applications
    applied_jobs = conn.execute("""
        SELECT url, title, site, location, fit_score,
               applied_at, apply_duration_ms, application_url
        FROM jobs
        WHERE apply_status = 'applied' AND applied_at IS NOT NULL
        ORDER BY applied_at DESC
    """).fetchall()

    # Failed applications (attempted but not successfully applied)
    failed_jobs = conn.execute("""
        SELECT url, title, site, location, fit_score,
               apply_status, apply_error, apply_attempts, last_attempted_at,
               application_url
        FROM jobs
        WHERE apply_status IS NOT NULL AND apply_status != 'applied'
          AND apply_attempts > 0
        ORDER BY last_attempted_at DESC
    """).fetchall()

    applied_count = len(applied_jobs)
    failed_count = len(failed_jobs)

    # Color map per site
    colors = {
        "RemoteOK": "#10b981", "WelcomeToTheJungle": "#f59e0b",
        "Job Bank Canada": "#3b82f6", "CareerJet Canada": "#8b5cf6",
        "Hacker News Jobs": "#ff6600", "BuiltIn Remote": "#ec4899",
        "TD Bank": "#00a651", "CIBC": "#c41f3e", "RBC": "#003168",
        "indeed": "#2164f3", "linkedin": "#0a66c2",
        "Dice": "#eb1c26", "Glassdoor": "#0caa41",
    }

    # Score distribution bar chart
    score_bars = ""
    max_count = max(score_dist.values()) if score_dist else 1
    for s in range(10, 0, -1):
        count = score_dist.get(s, 0)
        pct = (count / max_count * 100) if max_count else 0
        score_color = "#10b981" if s >= 7 else ("#f59e0b" if s >= 5 else "#ef4444")
        score_bars += f"""
        <div class="score-row">
          <span class="score-label">{s}</span>
          <div class="score-bar-track">
            <div class="score-bar-fill" style="width:{pct}%;background:{score_color}"></div>
          </div>
          <span class="score-count">{count}</span>
        </div>"""

    # Site stats rows
    site_rows = ""
    for s in site_stats:
        site = s["site"] or "?"
        color = colors.get(site, "#6b7280")
        avg = s["avg_score"] or 0
        site_rows += f"""
        <div class="site-row">
          <div class="site-name" style="color:{color}">{escape(site)}</div>
          <div class="site-nums">{s['total']} jobs &middot; {s['high_fit']} strong fit &middot; avg score {avg}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{s['high_fit']/max(s['total'],1)*100}%;background:{color}"></div>
            <div class="bar-fill" style="width:{s['mid_fit']/max(s['total'],1)*100}%;background:{color}66"></div>
          </div>
        </div>"""

    # Job cards grouped by score
    job_sections = ""
    current_score = None
    for j in jobs:
        score = j["fit_score"] or 0
        if score != current_score:
            if current_score is not None:
                job_sections += "</div>"
            score_color = "#10b981" if score >= 7 else "#f59e0b"
            score_label = {
                10: "Perfect Match", 9: "Excellent Fit", 8: "Strong Fit",
                7: "Good Fit", 6: "Moderate+", 5: "Moderate",
            }.get(score, f"Score {score}")
            count_at_score = score_dist.get(score, 0)
            job_sections += f"""
            <h2 class="score-header" style="border-color:{score_color}">
              <span class="score-badge" style="background:{score_color}">{score}</span>
              {score_label} ({count_at_score} jobs)
            </h2>
            <div class="job-grid">"""
            current_score = score

        title = escape(j["title"] or "Untitled")
        url = escape(j["url"] or "")
        salary = escape(j["salary"] or "")
        location = escape(j["location"] or "")
        site = escape(j["site"] or "")
        site_color = colors.get(j["site"] or "", "#6b7280")
        apply_url = escape(j["application_url"] or "")

        # Parse keywords and reasoning from score_reasoning
        reasoning_raw = j["score_reasoning"] or ""
        reasoning_lines = reasoning_raw.split("\n")
        keywords = reasoning_lines[0][:120] if reasoning_lines else ""
        reasoning = reasoning_lines[1][:200] if len(reasoning_lines) > 1 else ""

        desc_preview = escape(j["full_description"] or "")[:300]
        full_desc_html = escape(j["full_description"] or "").replace("\n", "<br>")
        desc_len = len(j["full_description"] or "")

        meta_parts = []
        meta_parts.append(
            f'<span class="meta-tag site-tag" style="background:{site_color}33;color:{site_color}">{site}</span>'
        )
        if salary:
            meta_parts.append(f'<span class="meta-tag salary">{salary}</span>')
        if location:
            meta_parts.append(f'<span class="meta-tag location">{location[:40]}</span>')
        meta_html = " ".join(meta_parts)

        apply_html = ""
        if apply_url:
            apply_html = f'<a href="{apply_url}" class="apply-link" target="_blank">Apply</a>'

        job_sections += f"""
        <div class="job-card" data-score="{score}" data-site="{escape(j['site'] or '')}" data-location="{location.lower()}">
          <div class="card-header">
            <span class="score-pill" style="background:{'#10b981' if score >= 7 else '#f59e0b'}">{score}</span>
            <a href="{url}" class="job-title" target="_blank">{title}</a>
          </div>
          <div class="meta-row">{meta_html}</div>
          {f'<div class="keywords-row">{escape(keywords)}</div>' if keywords else ''}
          {f'<div class="reasoning-row">{escape(reasoning)}</div>' if reasoning else ''}
          <p class="desc-preview">{desc_preview}...</p>
          {"<details class='full-desc-details'><summary class='expand-btn'>Full Description (" + f'{desc_len:,}' + " chars)</summary><div class='full-desc'>" + full_desc_html + "</div></details>" if j["full_description"] else ""}
          <div class="card-footer">{apply_html}</div>
        </div>"""

    if current_score is not None:
        job_sections += "</div>"

    # --- Submitted applications table ---
    def _fmt_date(iso: str | None) -> str:
        """Format ISO timestamp to a readable local date/time string."""
        if not iso:
            return "—"
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return escape(iso[:16])

    def _fmt_duration(ms: int | None) -> str:
        if not ms:
            return "—"
        if ms < 60_000:
            return f"{ms // 1000}s"
        return f"{ms // 60_000}m {(ms % 60_000) // 1000}s"

    def _score_chip(score: int | None) -> str:
        if score is None:
            return '<span style="color:#475569">—</span>'
        color = "#10b981" if score >= 7 else ("#f59e0b" if score >= 5 else "#ef4444")
        return f'<span class="score-chip" style="background:{color}">{score}</span>'

    def _status_chip(status: str | None) -> str:
        if not status:
            return "—"
        css = f"status-{status}" if status in (
            "applied", "expired", "captcha", "login_issue", "failed"
        ) else "status-default"
        return f'<span class="status-chip {css}">{escape(status.replace("_", " "))}</span>'

    if applied_jobs:
        applied_rows = ""
        for j in applied_jobs:
            title = escape(j["title"] or "Untitled")
            url = escape(j["url"] or "#")
            app_url = escape(j["application_url"] or "")
            site = escape(j["site"] or "")
            location = escape(j["location"] or "—")
            applied_rows += f"""
            <tr>
              <td>{_fmt_date(j['applied_at'])}</td>
              <td><a href="{url}" class="job-link" target="_blank">{title}</a></td>
              <td>{_score_chip(j['fit_score'])}</td>
              <td>{site}</td>
              <td>{location}</td>
              <td>{_fmt_duration(j['apply_duration_ms'])}</td>
              <td>{"<a href='" + app_url + "' class='apply-btn' target='_blank'>View</a>" if app_url else "—"}</td>
            </tr>"""
        applied_table_html = f"""
        <div class="app-table-wrap">
          <table class="app-table">
            <thead><tr>
              <th>Date Submitted</th><th>Job Title</th><th>Score</th>
              <th>Source</th><th>Location</th><th>Duration</th><th>Posting</th>
            </tr></thead>
            <tbody>{applied_rows}</tbody>
          </table>
        </div>"""
    else:
        applied_table_html = '<p class="empty-state">No submitted applications yet.</p>'

    # --- Failed applications table ---
    if failed_jobs:
        failed_rows = ""
        for j in failed_jobs:
            title = escape(j["title"] or "Untitled")
            url = escape(j["url"] or "#")
            app_url = escape(j["application_url"] or "")
            site = escape(j["site"] or "")
            reason = escape(j["apply_error"] or "")
            attempts = j["apply_attempts"] or 0
            failed_rows += f"""
            <tr>
              <td>{_fmt_date(j['last_attempted_at'])}</td>
              <td><a href="{url}" class="job-link" target="_blank">{title}</a></td>
              <td>{_score_chip(j['fit_score'])}</td>
              <td>{_status_chip(j['apply_status'])}</td>
              <td class="fail-reason">{reason or "—"}</td>
              <td style="text-align:center">{attempts}</td>
              <td>{site}</td>
              <td>{"<a href='" + app_url + "' class='apply-btn' target='_blank'>View</a>" if app_url else "—"}</td>
            </tr>"""
        failed_table_html = f"""
        <div class="app-table-wrap">
          <table class="app-table">
            <thead><tr>
              <th>Last Attempted</th><th>Job Title</th><th>Score</th>
              <th>Status</th><th>Failure Reason</th><th>Tries</th>
              <th>Source</th><th>Posting</th>
            </tr></thead>
            <tbody>{failed_rows}</tbody>
          </table>
        </div>"""
    else:
        failed_table_html = '<p class="empty-state">No failed applications.</p>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ApplyPilot Dashboard</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }}

  h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; }}
  .subtitle {{ color: #94a3b8; margin-bottom: 2rem; }}

  /* Summary cards */
  .summary {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 2.5rem; }}
  .stat-card {{ background: #1e293b; border-radius: 12px; padding: 1.25rem; }}
  .stat-num {{ font-size: 2rem; font-weight: 700; }}
  .stat-label {{ color: #94a3b8; font-size: 0.85rem; margin-top: 0.25rem; }}
  .stat-ok .stat-num {{ color: #10b981; }}
  .stat-scored .stat-num {{ color: #60a5fa; }}
  .stat-high .stat-num {{ color: #f59e0b; }}
  .stat-total .stat-num {{ color: #e2e8f0; }}

  /* Filters */
  .filters {{ background: #1e293b; border-radius: 12px; padding: 1.25rem; margin-bottom: 2rem; display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; }}
  .filter-label {{ color: #94a3b8; font-size: 0.85rem; font-weight: 600; }}
  .filter-btn {{ background: #334155; border: none; color: #94a3b8; padding: 0.4rem 0.8rem; border-radius: 6px; cursor: pointer; font-size: 0.8rem; transition: all 0.15s; }}
  .filter-btn:hover {{ background: #475569; color: #e2e8f0; }}
  .filter-btn.active {{ background: #60a5fa; color: #0f172a; font-weight: 600; }}
  .search-input {{ background: #334155; border: 1px solid #475569; color: #e2e8f0; padding: 0.4rem 0.8rem; border-radius: 6px; font-size: 0.8rem; width: 200px; }}
  .search-input::placeholder {{ color: #64748b; }}

  /* Score distribution */
  .score-section {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2.5rem; }}
  .score-dist {{ background: #1e293b; border-radius: 12px; padding: 1.5rem; }}
  .score-dist h3 {{ font-size: 1rem; margin-bottom: 1rem; color: #94a3b8; }}
  .score-row {{ display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem; }}
  .score-label {{ width: 1.5rem; text-align: right; font-size: 0.85rem; font-weight: 600; }}
  .score-bar-track {{ flex: 1; height: 14px; background: #334155; border-radius: 4px; overflow: hidden; }}
  .score-bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
  .score-count {{ width: 2.5rem; font-size: 0.8rem; color: #94a3b8; }}

  /* Site bars */
  .sites-section {{ background: #1e293b; border-radius: 12px; padding: 1.5rem; }}
  .sites-section h3 {{ font-size: 1rem; margin-bottom: 1rem; color: #94a3b8; }}
  .site-row {{ margin-bottom: 0.8rem; }}
  .site-name {{ font-weight: 600; font-size: 0.9rem; }}
  .site-nums {{ color: #94a3b8; font-size: 0.75rem; margin: 0.15rem 0; }}
  .bar-track {{ height: 8px; background: #334155; border-radius: 4px; display: flex; overflow: hidden; }}
  .bar-fill {{ height: 100%; transition: width 0.3s; }}

  /* Score group headers */
  .score-header {{ font-size: 1.2rem; font-weight: 600; margin: 2.5rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 3px solid; display: flex; align-items: center; gap: 0.75rem; }}
  .score-badge {{ display: inline-flex; align-items: center; justify-content: center; width: 2rem; height: 2rem; border-radius: 8px; color: #0f172a; font-weight: 700; font-size: 1rem; }}

  /* Job grid */
  .job-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 1rem; }}

  .job-card {{ background: #1e293b; border-radius: 10px; padding: 1rem; border-left: 3px solid #334155; transition: all 0.15s; }}
  .job-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px #00000044; }}
  .job-card[data-score="9"], .job-card[data-score="10"] {{ border-left-color: #10b981; }}
  .job-card[data-score="8"] {{ border-left-color: #34d399; }}
  .job-card[data-score="7"] {{ border-left-color: #60a5fa; }}
  .job-card[data-score="6"] {{ border-left-color: #f59e0b; }}
  .job-card[data-score="5"] {{ border-left-color: #f59e0b88; }}

  .card-header {{ display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; }}
  .score-pill {{ display: inline-flex; align-items: center; justify-content: center; min-width: 1.6rem; height: 1.6rem; border-radius: 6px; color: #0f172a; font-weight: 700; font-size: 0.8rem; flex-shrink: 0; }}

  .job-title {{ color: #e2e8f0; text-decoration: none; font-weight: 600; font-size: 0.95rem; }}
  .job-title:hover {{ color: #60a5fa; }}

  .meta-row {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.4rem; }}
  .meta-tag {{ font-size: 0.72rem; padding: 0.15rem 0.5rem; border-radius: 4px; background: #334155; color: #94a3b8; }}
  .meta-tag.salary {{ background: #064e3b; color: #6ee7b7; }}
  .meta-tag.location {{ background: #1e3a5f; color: #93c5fd; }}

  .keywords-row {{ font-size: 0.75rem; color: #10b981; margin-bottom: 0.3rem; line-height: 1.4; }}
  .reasoning-row {{ font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.5rem; font-style: italic; line-height: 1.4; }}

  .desc-preview {{ font-size: 0.8rem; color: #64748b; line-height: 1.5; margin-bottom: 0.75rem; max-height: 3.6em; overflow: hidden; }}

  .card-footer {{ display: flex; justify-content: flex-end; }}
  .apply-link {{ font-size: 0.8rem; color: #60a5fa; text-decoration: none; padding: 0.3rem 0.8rem; border: 1px solid #60a5fa33; border-radius: 6px; font-weight: 500; }}
  .apply-link:hover {{ background: #60a5fa22; }}

  /* Expandable full description */
  .full-desc-details {{ margin-bottom: 0.75rem; }}
  .expand-btn {{ font-size: 0.8rem; color: #60a5fa; cursor: pointer; list-style: none; padding: 0.3rem 0; }}
  .expand-btn::-webkit-details-marker {{ display: none; }}
  .expand-btn:hover {{ color: #93c5fd; }}
  .full-desc {{ font-size: 0.8rem; color: #cbd5e1; line-height: 1.6; margin-top: 0.5rem; padding: 0.75rem; background: #0f172a; border-radius: 8px; max-height: 400px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }}

  .hidden {{ display: none !important; }}
  .job-count {{ color: #94a3b8; font-size: 0.85rem; margin-bottom: 1rem; }}

  /* Application tables */
  .app-section {{ margin-bottom: 3rem; }}
  .app-section h2 {{ font-size: 1.3rem; font-weight: 700; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #334155; display: flex; align-items: center; gap: 0.75rem; }}
  .app-section h2 .count-badge {{ background: #334155; color: #94a3b8; font-size: 0.75rem; padding: 0.15rem 0.5rem; border-radius: 99px; font-weight: 600; }}
  .app-table-wrap {{ overflow-x: auto; }}
  .app-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  .app-table th {{ text-align: left; padding: 0.6rem 0.75rem; color: #94a3b8; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; border-bottom: 1px solid #334155; white-space: nowrap; }}
  .app-table td {{ padding: 0.65rem 0.75rem; border-bottom: 1px solid #1e293b; vertical-align: top; }}
  .app-table tr:last-child td {{ border-bottom: none; }}
  .app-table tr:hover td {{ background: #1e293b44; }}
  .app-table .job-link {{ color: #e2e8f0; text-decoration: none; font-weight: 600; }}
  .app-table .job-link:hover {{ color: #60a5fa; }}
  .app-table .apply-btn {{ color: #60a5fa; text-decoration: none; font-size: 0.78rem; padding: 0.2rem 0.6rem; border: 1px solid #60a5fa33; border-radius: 5px; white-space: nowrap; }}
  .app-table .apply-btn:hover {{ background: #60a5fa22; }}
  .score-chip {{ display: inline-flex; align-items: center; justify-content: center; min-width: 1.6rem; height: 1.4rem; border-radius: 5px; font-weight: 700; font-size: 0.78rem; color: #0f172a; }}
  .status-chip {{ display: inline-block; font-size: 0.72rem; padding: 0.15rem 0.5rem; border-radius: 4px; font-weight: 600; white-space: nowrap; }}
  .status-applied {{ background: #064e3b; color: #6ee7b7; }}
  .status-expired {{ background: #1c1917; color: #78716c; }}
  .status-captcha {{ background: #3b0764; color: #d8b4fe; }}
  .status-login_issue {{ background: #450a0a; color: #fca5a5; }}
  .status-failed {{ background: #431407; color: #fdba74; }}
  .status-default {{ background: #1e293b; color: #94a3b8; }}
  .fail-reason {{ color: #f87171; font-size: 0.78rem; max-width: 300px; }}
  .empty-state {{ color: #475569; font-style: italic; padding: 2rem; text-align: center; }}
  .stat-applied .stat-num {{ color: #6ee7b7; }}
  .stat-failed .stat-num {{ color: #f87171; }}

  @media (max-width: 768px) {{
    .summary {{ grid-template-columns: repeat(2, 1fr); }}
    .score-section {{ grid-template-columns: 1fr; }}
    .job-grid {{ grid-template-columns: 1fr; }}
    body {{ padding: 1rem; }}
  }}
</style>
</head>
<body>

<h1>ApplyPilot Dashboard</h1>
<p class="subtitle">{total} jobs &middot; {scored} scored &middot; {high_fit} strong matches (7+)</p>

<div class="summary">
  <div class="stat-card stat-total"><div class="stat-num">{total}</div><div class="stat-label">Total Jobs</div></div>
  <div class="stat-card stat-ok"><div class="stat-num">{ready}</div><div class="stat-label">Ready (desc + URL)</div></div>
  <div class="stat-card stat-scored"><div class="stat-num">{scored}</div><div class="stat-label">Scored by LLM</div></div>
  <div class="stat-card stat-high"><div class="stat-num">{high_fit}</div><div class="stat-label">Strong Fit (7+)</div></div>
  <div class="stat-card stat-applied"><div class="stat-num">{applied_count}</div><div class="stat-label">Submitted</div></div>
  <div class="stat-card stat-failed"><div class="stat-num">{failed_count}</div><div class="stat-label">Failed</div></div>
</div>

<div class="filters">
  <span class="filter-label">Score:</span>
  <button class="filter-btn active" onclick="filterScore(0)">All 5+</button>
  <button class="filter-btn" onclick="filterScore(7)">7+ Strong</button>
  <button class="filter-btn" onclick="filterScore(8)">8+ Excellent</button>
  <button class="filter-btn" onclick="filterScore(9)">9+ Perfect</button>
  <span class="filter-label" style="margin-left:1rem">Search:</span>
  <input type="text" class="search-input" placeholder="Filter by title, site..." oninput="filterText(this.value)">
</div>

<div class="score-section">
  <div class="score-dist">
    <h3>Score Distribution</h3>
    {score_bars}
  </div>
  <div class="sites-section">
    <h3>By Source</h3>
    {site_rows}
  </div>
</div>

<div class="app-section">
  <h2 style="color:#6ee7b7">Submitted Applications <span class="count-badge">{applied_count}</span></h2>
  {applied_table_html}
</div>

<div class="app-section">
  <h2 style="color:#f87171">Failed Applications <span class="count-badge">{failed_count}</span></h2>
  {failed_table_html}
</div>

<div id="job-count" class="job-count"></div>

{job_sections}

<script>
let minScore = 0;
let searchText = '';

function filterScore(min) {{
  minScore = min;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  applyFilters();
}}

function filterText(text) {{
  searchText = text.toLowerCase();
  applyFilters();
}}

function applyFilters() {{
  let shown = 0;
  let total = 0;
  document.querySelectorAll('.job-card').forEach(card => {{
    total++;
    const score = parseInt(card.dataset.score) || 0;
    const text = card.textContent.toLowerCase();
    const scoreMatch = score >= (minScore || 5);
    const textMatch = !searchText || text.includes(searchText);
    if (scoreMatch && textMatch) {{
      card.classList.remove('hidden');
      shown++;
    }} else {{
      card.classList.add('hidden');
    }}
  }});
  document.getElementById('job-count').textContent = `Showing ${{shown}} of ${{total}} jobs`;

  // Hide empty score groups
  document.querySelectorAll('.score-header').forEach(header => {{
    const grid = header.nextElementSibling;
    if (grid && grid.classList.contains('job-grid')) {{
      const visible = grid.querySelectorAll('.job-card:not(.hidden)').length;
      header.style.display = visible ? '' : 'none';
      grid.style.display = visible ? '' : 'none';
    }}
  }});
}}

applyFilters();
</script>

</body>
</html>"""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    abs_path = str(out.resolve())
    console.print(f"[green]Dashboard written to {abs_path}[/green]")
    return abs_path


def open_dashboard(output_path: str | None = None) -> None:
    """Generate the dashboard and open it in the default browser.

    Args:
        output_path: Where to write the HTML file. Defaults to ~/.applypilot/dashboard.html.
    """
    path = generate_dashboard(output_path)
    console.print("[dim]Opening in browser...[/dim]")
    webbrowser.open(f"file:///{path}")
