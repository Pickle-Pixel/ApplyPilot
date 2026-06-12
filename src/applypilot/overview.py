"""Cross-workspace overview dashboard.

Aggregates the databases of every ~/.applypilot* workspace (one per target
role, e.g. MLE / DS / AIE) into a single self-contained HTML page:

  - Jobs discovered in the last 1h / 24h / 72h, per role
  - Source platform breakdown (Indeed, LinkedIn, Workday, ...)
  - Application site (ATS) breakdown (Workday, Greenhouse, Lever, ...)
  - Applied / pending status, with a filterable recent-jobs table
"""

from __future__ import annotations

import json
import sqlite3
import webbrowser
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from urllib.parse import urlparse

from applypilot.config import APP_DIR
from applypilot.database import classify_region

WINDOWS_H = (1, 24, 72)

# Known ATS domains -> friendly names
_ATS_NAMES = {
    "myworkdayjobs": "Workday",
    "workday": "Workday",
    "greenhouse": "Greenhouse",
    "lever": "Lever",
    "icims": "iCIMS",
    "ashbyhq": "Ashby",
    "smartrecruiters": "SmartRecruiters",
    "taleo": "Taleo",
    "oraclecloud": "Oracle",
    "jobvite": "Jobvite",
    "bamboohr": "BambooHR",
    "workable": "Workable",
    "successfactors": "SuccessFactors",
    "adp": "ADP",
    "indeed": "Indeed",
    "linkedin": "LinkedIn",
}

_ROLE_COLORS = ["#4263eb", "#0ca678", "#e8590c", "#9c36b5", "#e03131"]


def _ats_name(url: str | None) -> str:
    """Map an application URL to a friendly ATS / site name."""
    if not url:
        return "—"
    host = urlparse(url).netloc.lower()
    for key, name in _ATS_NAMES.items():
        if key in host:
            return name
    # Fall back to the registrable part of the domain
    parts = host.replace("www.", "").split(".")
    return parts[-2].capitalize() if len(parts) >= 2 else (host or "—")


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def discover_workspaces() -> list[tuple[str, Path]]:
    """Find all ~/.applypilot* workspaces containing a database.

    Returns:
        List of (role_name, db_path), the current APP_DIR first.
    """
    found: list[tuple[str, Path]] = []
    for d in sorted(Path.home().glob(".applypilot*")):
        db = d / "applypilot.db"
        if db.exists():
            name = d.name.replace(".applypilot-", "").replace(".applypilot", "default")
            found.append((name, db))
    # If APPLYPILOT_DIR points somewhere custom, include it too
    app_db = APP_DIR / "applypilot.db"
    if app_db.exists() and not any(db == app_db for _, db in found):
        found.insert(0, (APP_DIR.name.lstrip("."), app_db))
    return found


def collect_stats(role: str, db_path: Path, now: datetime) -> dict:
    """Aggregate one workspace database into dashboard-ready stats."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        # Read-only connection: tolerate columns missing from older databases
        cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        sel_posted = "date_posted" if "date_posted" in cols else "NULL AS date_posted"
        sel_short = "shortlisted" if "shortlisted" in cols else "NULL AS shortlisted"
        hidden_clause = "WHERE COALESCE(hidden, 0) != 1" if "hidden" in cols else ""
        rows = conn.execute(
            f"SELECT url, title, site, location, application_url, fit_score, "
            f"       discovered_at, {sel_posted}, tailored_resume_path, applied_at, "
            f"       apply_status, {sel_short} "
            f"FROM jobs {hidden_clause}"
        ).fetchall()
    finally:
        conn.close()

    cutoffs = {h: now - timedelta(hours=h) for h in WINDOWS_H}
    window_counts = {h: 0 for h in WINDOWS_H}
    window_sites: dict[int, dict[str, int]] = {h: {} for h in WINDOWS_H}
    sites: dict[str, int] = {}
    ats: dict[str, int] = {}
    recent: list[dict] = []

    total = len(rows)
    scored = tailored = applied = 0
    applied_days: list[str] = []  # local dates of every application, for history charts

    for r in rows:
        site = r["site"] or "unknown"
        sites[site] = sites.get(site, 0) + 1
        if r["fit_score"] is not None:
            scored += 1
        if r["tailored_resume_path"]:
            tailored += 1
        if r["applied_at"]:
            applied += 1
            applied_ts = _parse_ts(r["applied_at"])
            if applied_ts:
                applied_days.append(applied_ts.astimezone().date().isoformat())
        if r["application_url"]:
            name = _ats_name(r["application_url"])
            ats[name] = ats.get(name, 0) + 1

        ts = _parse_ts(r["discovered_at"])
        if ts is None:
            continue
        for h in WINDOWS_H:
            if ts >= cutoffs[h]:
                window_counts[h] += 1
                window_sites[h][site] = window_sites[h].get(site, 0) + 1
        if ts >= cutoffs[72]:
            age_h = (now - ts).total_seconds() / 3600
            # date_posted is day-granularity (boards don't expose the hour).
            # Encode as full days so "posted today" (0h) matches any sub-24h
            # filter window instead of being penalized by a midnight offset.
            posted = _parse_ts(r["date_posted"])
            posted_h = None
            if posted:
                days_ago = (now.astimezone().date() - posted.date()).days
                posted_h = max(0, days_ago) * 24.0
            recent.append({
                "key": r["url"],  # DB primary key, used by the hide endpoint
                "posted_h": posted_h,
                "east": classify_region(r["location"]) == "east",
                "title": r["title"] or "",
                "site": site,
                "location": r["location"] or "",
                "ats": _ats_name(r["application_url"]),
                "url": r["application_url"] or r["url"],
                "score": r["fit_score"],
                "status": ("applied" if r["applied_at"]
                           else (r["apply_status"] or
                                 ("tailored" if r["tailored_resume_path"] else "new"))),
                "shortlisted": bool(r["shortlisted"]),
                "age_h": round(age_h, 1),
            })

    recent.sort(key=lambda j: j["age_h"])
    return {
        "role": role,
        "total": total,
        "scored": scored,
        "tailored": tailored,
        "applied": applied,
        "windows": window_counts,
        "window_sites": window_sites,
        "sites": sites,
        "ats": ats,
        "recent": recent,
        "applied_days": applied_days,
    }


# ── HTML rendering ───────────────────────────────────────────────────────

def _bar_rows(counts: dict[str, int], color: str) -> str:
    if not counts:
        return '<div class="muted">no data</div>'
    top = sorted(counts.items(), key=lambda kv: -kv[1])
    peak = top[0][1] or 1
    out = ""
    for name, n in top:
        pct = max(2, round(n / peak * 100))
        out += (
            f'<div class="bar-row"><span class="bar-label">{escape(str(name))}</span>'
            f'<span class="bar-track"><span class="bar-fill" '
            f'style="width:{pct}%;background:{color}"></span></span>'
            f'<span class="bar-num">{n}</span></div>'
        )
    return out


def _applied_history_html(all_stats: list[dict], colors: dict[str, str],
                          now: datetime) -> str:
    """Stacked per-role bars: applications per day (14d) and per week (8w)."""
    local_today = now.astimezone().date()

    # role -> {date_iso: count}
    by_role: dict[str, dict[str, int]] = {}
    for s in all_stats:
        counts: dict[str, int] = {}
        for d in s["applied_days"]:
            counts[d] = counts.get(d, 0) + 1
        by_role[s["role"]] = counts

    def stacked_rows(buckets: list[tuple[str, dict[str, int]]]) -> str:
        peak = max((sum(per_role.values()) for _, per_role in buckets), default=0) or 1
        out = ""
        for label, per_role in buckets:
            total_n = sum(per_role.values())
            segs = ""
            for role, n in per_role.items():
                if n:
                    pct = round(n / peak * 100)
                    segs += (f'<span class="bar-fill" title="{escape(role)}: {n}" '
                             f'style="width:{pct}%;background:{colors[role]}"></span>')
            out += (f'<div class="bar-row"><span class="bar-label">{label}</span>'
                    f'<span class="bar-track stacked">{segs}</span>'
                    f'<span class="bar-num">{total_n or ""}</span></div>')
        return out

    # Daily: last 14 days
    daily: list[tuple[str, dict[str, int]]] = []
    for i in range(13, -1, -1):
        day = local_today - timedelta(days=i)
        key = day.isoformat()
        daily.append((day.strftime("%m-%d"),
                      {role: counts.get(key, 0) for role, counts in by_role.items()}))

    # Weekly: last 8 weeks (Monday-anchored)
    monday = local_today - timedelta(days=local_today.weekday())
    weekly: list[tuple[str, dict[str, int]]] = []
    for i in range(7, -1, -1):
        wk_start = monday - timedelta(weeks=i)
        days = [(wk_start + timedelta(days=d)).isoformat() for d in range(7)]
        weekly.append((f"wk {wk_start.strftime('%m-%d')}",
                       {role: sum(counts.get(d, 0) for d in days)
                        for role, counts in by_role.items()}))

    total_applied = sum(sum(c.values()) for c in by_role.values())
    return f"""
<h2>Applications over time <span class="muted" style="text-transform:none">({total_applied} total)</span></h2>
<div class="cols">
  <div class="col"><div class="col-title">Per day (last 14 days)</div>{stacked_rows(daily)}</div>
  <div class="col"><div class="col-title">Per week (last 8 weeks)</div>{stacked_rows(weekly)}</div>
</div>"""


def build_html(all_stats: list[dict], now: datetime) -> str:
    colors = {s["role"]: _ROLE_COLORS[i % len(_ROLE_COLORS)] for i, s in enumerate(all_stats)}

    # Role summary cards
    cards = ""
    for s in all_stats:
        c = colors[s["role"]]
        cards += f"""
<div class="card" style="border-top:3px solid {c}">
  <div class="card-title" style="color:{c}">{escape(s['role'].upper())}</div>
  <div class="windows">
    <div><span class="big">{s['windows'][1]}</span><span class="lbl">1h</span></div>
    <div><span class="big">{s['windows'][24]}</span><span class="lbl">24h</span></div>
    <div><span class="big">{s['windows'][72]}</span><span class="lbl">72h</span></div>
  </div>
  <div class="totals">
    {s['total']} total &middot; {s['scored']} scored &middot;
    {s['tailored']} tailored &middot; <b>{s['applied']} applied</b>
  </div>
</div>"""

    # Per-role platform breakdown (with window tabs handled client-side)
    platform_cols = ""
    for s in all_stats:
        c = colors[s["role"]]
        blocks = ""
        for h in WINDOWS_H:
            blocks += (
                f'<div class="win win-{h}" style="display:none">'
                + _bar_rows(s["window_sites"][h], c) + "</div>"
            )
        blocks += f'<div class="win win-all">{_bar_rows(s["sites"], c)}</div>'
        platform_cols += (
            f'<div class="col"><div class="col-title" style="color:{c}">'
            f'{escape(s["role"].upper())}</div>{blocks}</div>'
        )

    # ATS breakdown
    ats_cols = ""
    for s in all_stats:
        c = colors[s["role"]]
        ats_cols += (
            f'<div class="col"><div class="col-title" style="color:{c}">'
            f'{escape(s["role"].upper())}</div>{_bar_rows(s["ats"], c)}</div>'
        )

    # Recent jobs table (72h), data embedded as JSON for client-side filtering
    table_rows = []
    for s in all_stats:
        for j in s["recent"]:
            table_rows.append({**j, "role": s["role"], "color": colors[s["role"]]})
    table_json = json.dumps(table_rows, ensure_ascii=False).replace("</", "<\\/")

    role_buttons = "".join(
        f'<button class="fbtn rbtn" data-role="{escape(s["role"])}" '
        f'style="--c:{colors[s["role"]]}">{escape(s["role"].upper())}</button>'
        for s in all_stats
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ApplyPilot Overview</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,'Segoe UI',Arial,sans-serif; background:#faf9f7; color:#1f2328; padding:28px 32px; }}
h1 {{ font-size:21px; margin-bottom:4px; font-weight:700; letter-spacing:-0.2px; }}
h2 {{ font-size:12px; text-transform:uppercase; letter-spacing:1.5px; color:#a39e93; margin:32px 0 14px; font-weight:600; }}
.muted {{ color:#8a8578; font-size:12px; }}
.cards {{ display:flex; gap:16px; flex-wrap:wrap; margin-top:16px; }}
.card {{ background:#fff; border:1px solid #eceae4; border-radius:12px; padding:18px 22px;
        min-width:230px; flex:1; box-shadow:0 1px 3px rgba(31,35,40,.04); }}
.card-title {{ font-weight:700; font-size:15px; margin-bottom:12px; }}
.windows {{ display:flex; gap:26px; }}
.windows .big {{ font-size:27px; font-weight:700; display:block; letter-spacing:-0.5px; }}
.windows .lbl {{ font-size:11px; color:#a39e93; }}
.totals {{ margin-top:12px; font-size:12px; color:#6e6a60; }}
.cols {{ display:flex; gap:18px; flex-wrap:wrap; }}
.col {{ background:#fff; border:1px solid #eceae4; border-radius:12px; padding:16px 20px;
       flex:1; min-width:260px; box-shadow:0 1px 3px rgba(31,35,40,.04); }}
.col-title {{ font-weight:700; font-size:13px; margin-bottom:12px; }}
.bar-row {{ display:flex; align-items:center; gap:8px; margin:5px 0; font-size:12px; color:#3d4148; }}
.bar-label {{ width:110px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.bar-track {{ flex:1; background:#f1efe9; border-radius:4px; height:14px; overflow:hidden; }}
.bar-fill {{ display:block; height:100%; border-radius:4px; }}
.bar-track.stacked {{ display:flex; }}
.bar-track.stacked .bar-fill {{ border-radius:0; }}
.hide-btn {{ background:none; border:none; color:#c2bdb2; cursor:pointer; font-size:14px;
            padding:2px 6px; border-radius:4px; }}
.hide-btn:hover {{ background:#fdeaea; color:#c03030; }}
.actions {{ white-space:nowrap; }}
.mark-btn {{ background:none; border:none; color:#c2bdb2; cursor:pointer; font-size:14px;
            padding:2px 6px; border-radius:4px; }}
.mark-btn:hover {{ background:#d9f2e3; color:#1b7a43; }}
th.sortable {{ cursor:pointer; user-select:none; }}
th.sortable:hover {{ color:#1f2328; }}
th.sortable .arrow {{ font-size:9px; }}
.bar-num {{ width:40px; text-align:right; color:#8a8578; }}
.fbar {{ display:flex; gap:8px; margin:10px 0 16px; flex-wrap:wrap; }}
.fbtn {{ background:#fff; color:#6e6a60; border:1px solid #ddd9d1; border-radius:7px;
        padding:5px 14px; font-size:12px; cursor:pointer; transition:all .12s; }}
.fbtn:hover {{ border-color:#b8b3a8; color:#1f2328; }}
.fbtn.active {{ background:var(--c,#1f2328); color:#fff; border-color:transparent; }}
table {{ width:100%; border-collapse:collapse; font-size:12.5px; background:#fff;
        border:1px solid #eceae4; border-radius:12px; overflow:hidden;
        box-shadow:0 1px 3px rgba(31,35,40,.04); }}
th {{ text-align:left; padding:10px 12px; color:#a39e93; font-size:10.5px; text-transform:uppercase;
     letter-spacing:.8px; border-bottom:1px solid #eceae4; background:#fcfbf9; }}
td {{ padding:9px 12px; border-bottom:1px solid #f4f2ed; }}
tr:last-child td {{ border-bottom:none; }}
tr:hover td {{ background:#faf9f6; }}
td a {{ color:#3b5bdb; text-decoration:none; }}
td a:hover {{ text-decoration:underline; }}
.pill {{ display:inline-block; padding:2px 9px; border-radius:10px; font-size:11px; font-weight:600; }}
.pill.applied {{ background:#d9f2e3; color:#1b7a43; }}
.pill.tailored {{ background:#e0ebfb; color:#1c5fae; }}
.pill.failed {{ background:#fde3e3; color:#c03030; }}
.pill.in_progress {{ background:#fcf0d3; color:#946d00; }}
.pill.new, .pill.manual {{ background:#efede8; color:#6e6a60; }}
.pill.east {{ background:#d9f0f4; color:#0b7285; }}
.role-tag {{ font-weight:700; font-size:11px; }}
.star {{ color:#e8a912; }}
</style>
</head>
<body>
<h1>ApplyPilot Overview</h1>
<div class="muted">Generated {now.astimezone().strftime('%Y-%m-%d %H:%M %Z')} &middot;
{len(all_stats)} workspace(s)</div>

<h2>Discovered jobs by role</h2>
<div class="cards">{cards}</div>

<h2>Source platforms</h2>
<div class="fbar" id="winbar">
  <button class="fbtn" data-win="1">1h</button>
  <button class="fbtn" data-win="24">24h</button>
  <button class="fbtn" data-win="72">72h</button>
  <button class="fbtn active" data-win="all">All time</button>
</div>
<div class="cols">{platform_cols}</div>

<h2>Application sites (ATS)</h2>
<div class="cols">{ats_cols}</div>
{_applied_history_html(all_stats, colors, now)}

<h2>Jobs from the last 72h</h2>
<div class="fbar">
  <button class="fbtn active" data-tw="72">72h</button>
  <button class="fbtn" data-tw="24">24h</button>
  <button class="fbtn" data-tw="1">1h</button>
  <input type="number" id="customh" min="1" max="72" placeholder="custom h"
         style="width:84px;background:#fff;color:#1f2328;border:1px solid #ddd9d1;
                border-radius:7px;padding:5px 10px;font-size:12px">
  <span style="width:14px"></span>
  <span class="muted" style="align-self:center">by</span>
  <button class="fbtn active" data-basis="age_h">Discovered</button>
  <button class="fbtn" data-basis="posted_h">Posted</button>
  <span style="width:14px"></span>
  <button class="fbtn rbtn active" data-role="__all__">ALL ROLES</button>
  {role_buttons}
  <span style="width:14px"></span>
  <button class="fbtn" id="appliedonly">Applied only</button>
  <button class="fbtn" id="eastonly">East Coast only</button>
  <span style="width:14px"></span>
  <button class="fbtn" id="resetsort" title="Back to default: freshest discovered first">↺ Reset sort</button>
</div>
<table>
<thead><tr><th>Role</th><th class="sortable" data-sort="age_h">Discovered <span class="arrow"></span></th>
<th class="sortable" data-sort="posted_h">Posted <span class="arrow"></span></th>
<th>Title</th><th>Platform</th><th>ATS</th>
<th class="sortable" data-sort="score">Score <span class="arrow"></span></th>
<th>Status</th><th></th></tr></thead>
<tbody id="jobs"></tbody>
</table>
<div class="muted" id="count" style="margin-top:8px"></div>

<script>
const DATA = {table_json};
let win = 72, role = "__all__", appliedOnly = false, eastOnly = false, basis = "age_h";
// Multi-sort chain in CLICK ORDER: first clicked column = primary sort,
// each new column clicked is appended as the next tiebreaker. Clicking a
// column already in the chain only flips its direction (position is kept,
// so the existing ordering doesn't reshuffle). Empty chain = default
// (discovered, freshest first). dir: 1 = asc, -1 = desc.
let sortChain = [];
const DEFAULT_CHAIN = [{{key: "age_h", dir: 1}}];

// Hidden jobs: server mode writes to the DB; file:// mode falls back to localStorage
const SERVED = location.protocol.startsWith("http");
const lsHidden = new Set(JSON.parse(localStorage.getItem("apHidden") || "[]"));

function markApplied(key, jrole) {{
  if (!SERVED) {{
    alert("标记申请需要 serve 模式：applypilot overview --serve");
    return;
  }}
  const j = DATA.find(x => x.key === key && x.role === jrole);
  if (j) j.status = "applied";
  fetch("/api/applied", {{method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({{url: key, role: jrole}})}})
    .catch(() => alert("Failed to persist -- is the server still running?"));
  render();
}}

function hideJob(key, jrole) {{
  const i = DATA.findIndex(j => j.key === key && j.role === jrole);
  if (i >= 0) DATA.splice(i, 1);
  if (SERVED) {{
    fetch("/api/hide", {{method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify({{url: key, role: jrole}})}})
      .catch(() => alert("Failed to persist -- is the server still running?"));
  }} else {{
    lsHidden.add(jrole + "|" + key);
    localStorage.setItem("apHidden", JSON.stringify([...lsHidden]));
  }}
  render();
}}

function pill(s) {{
  const cls = ["applied","tailored","failed","in_progress"].includes(s) ? s : "new";
  return `<span class="pill ${{cls}}">${{s}}</span>`;
}}
function fmtAge(h) {{
  if (h == null) return "—";
  return h < 1 ? Math.round(h*60)+"m" : (h < 24 ? h.toFixed(1)+"h" : Math.round(h/24)+"d");
}}
function fmtPosted(h) {{  // day-granularity: boards only expose the date
  if (h == null) return "—";
  return h < 24 ? "today" : Math.round(h/24)+"d";
}}

function render() {{
  const rows = DATA.filter(j => !lsHidden.has(j.role + "|" + j.key))
    .filter(j => j[basis] != null && j[basis] <= win)
    .filter(j => role === "__all__" || j.role === role)
    .filter(j => !appliedOnly || j.status === "applied")
    .filter(j => !eastOnly || j.east);
  rows.sort((a, b) => {{
    for (const {{key, dir}} of (sortChain.length ? sortChain : DEFAULT_CHAIN)) {{
      const av = a[key], bv = b[key];
      if (av == null && bv == null) continue;
      if (av == null) return 1;   // nulls always last
      if (bv == null) return -1;
      if (av !== bv) return (av - bv) * dir;
    }}
    return 0;
  }});
  document.getElementById("jobs").innerHTML = rows.map(j => `
    <tr>
      <td><span class="role-tag" style="color:${{j.color}}">${{j.role.toUpperCase()}}</span></td>
      <td class="muted">${{fmtAge(j.age_h)}}</td>
      <td class="muted">${{fmtPosted(j.posted_h)}}</td>
      <td>${{j.shortlisted ? '<span class="star">★</span> ' : ''}}<a href="${{j.url}}" target="_blank">${{j.title}}</a>
          <div class="muted">${{j.location}}${{j.east ? ' <span class="pill east">美东</span>' : ''}}</div></td>
      <td>${{j.site}}</td>
      <td>${{j.ats}}</td>
      <td>${{j.score ?? "—"}}</td>
      <td>${{pill(j.status)}}</td>
      <td class="actions">${{j.status === "applied" ? "" :
          `<button class="mark-btn" title="Mark as applied"
            data-key="${{encodeURIComponent(j.key)}}" data-role="${{j.role}}">✓</button>`}}<button
          class="hide-btn" title="Remove from dashboard (kept in DB)"
          data-key="${{encodeURIComponent(j.key)}}" data-role="${{j.role}}">✕</button></td>
    </tr>`).join("");
  let note = rows.length + " job(s)";
  if (basis === "posted_h") {{
    const unknown = DATA.filter(j => j.posted_h == null).length;
    if (unknown) note += ` (${{unknown}} hidden: no posting date)`;
  }}
  document.getElementById("count").textContent = note;
}}

document.querySelectorAll("[data-tw]").forEach(b => b.onclick = () => {{
  document.querySelectorAll("[data-tw]").forEach(x => x.classList.remove("active"));
  b.classList.add("active"); win = +b.dataset.tw;
  document.getElementById("customh").value = ""; render();
}});
document.getElementById("customh").oninput = (e) => {{
  const v = +e.target.value;
  if (v >= 1) {{
    document.querySelectorAll("[data-tw]").forEach(x => x.classList.remove("active"));
    win = v; render();
  }}
}};
document.querySelectorAll("[data-basis]").forEach(b => b.onclick = () => {{
  document.querySelectorAll("[data-basis]").forEach(x => x.classList.remove("active"));
  b.classList.add("active"); basis = b.dataset.basis; render();
}});
document.querySelectorAll("[data-role]").forEach(b => b.onclick = () => {{
  document.querySelectorAll("[data-role]").forEach(x => x.classList.remove("active"));
  b.classList.add("active"); role = b.dataset.role; render();
}});
document.getElementById("appliedonly").onclick = (e) => {{
  appliedOnly = !appliedOnly; e.target.classList.toggle("active"); render();
}};
document.getElementById("eastonly").onclick = (e) => {{
  eastOnly = !eastOnly; e.target.classList.toggle("active"); render();
}};
document.getElementById("jobs").addEventListener("click", (e) => {{
  const hide = e.target.closest(".hide-btn");
  if (hide) {{ hideJob(decodeURIComponent(hide.dataset.key), hide.dataset.role); return; }}
  const mark = e.target.closest(".mark-btn");
  if (mark) markApplied(decodeURIComponent(mark.dataset.key), mark.dataset.role);
}});
function updateArrows() {{
  document.querySelectorAll("th.sortable").forEach(th => {{
    const i = sortChain.findIndex(s => s.key === th.dataset.sort);
    th.querySelector(".arrow").textContent =
      i < 0 ? "" : (sortChain[i].dir === 1 ? "▲" : "▼") + (sortChain.length > 1 ? (i + 1) : "");
  }});
}}
document.querySelectorAll("th.sortable").forEach(th => th.onclick = () => {{
  const key = th.dataset.sort;
  const existing = sortChain.findIndex(s => s.key === key);
  if (existing >= 0) {{
    sortChain[existing].dir = -sortChain[existing].dir;  // flip in place
  }} else {{
    sortChain.push({{key, dir: key === "score" ? -1 : 1}});  // append as tiebreaker
  }}
  updateArrows();
  render();
}});
document.getElementById("resetsort").onclick = () => {{
  sortChain = [];
  updateArrows();
  render();
}};
updateArrows();
document.querySelectorAll("#winbar [data-win]").forEach(b => b.onclick = () => {{
  document.querySelectorAll("#winbar [data-win]").forEach(x => x.classList.remove("active"));
  b.classList.add("active");
  const w = b.dataset.win;
  document.querySelectorAll(".win").forEach(el => el.style.display = "none");
  document.querySelectorAll(".win-" + w).forEach(el => el.style.display = "block");
}});
render();
</script>
</body>
</html>"""


def generate_overview(output_path: str | None = None) -> str:
    """Generate the cross-workspace overview dashboard.

    Returns:
        Absolute path to the generated HTML file.
    """
    now = datetime.now(timezone.utc)
    workspaces = discover_workspaces()
    if not workspaces:
        raise FileNotFoundError("No workspace with an applypilot.db found under ~/.applypilot*")

    all_stats = [collect_stats(role, db, now) for role, db in workspaces]
    html = build_html(all_stats, now)

    out = Path(output_path) if output_path else APP_DIR / "overview.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)


def open_overview(output_path: str | None = None) -> None:
    """Generate the overview and open it in the default browser."""
    path = generate_overview(output_path)
    webbrowser.open(f"file://{path}")


# ── Serve mode (live dashboard with working hide buttons) ────────────────

def _hide_job(db_path: Path, url: str) -> None:
    """Mark a job hidden in its workspace database (adds the column if missing)."""
    conn = sqlite3.connect(db_path)
    try:
        try:
            conn.execute("UPDATE jobs SET hidden = 1 WHERE url = ?", (url,))
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE jobs ADD COLUMN hidden INTEGER")
            conn.execute("UPDATE jobs SET hidden = 1 WHERE url = ?", (url,))
        conn.commit()
    finally:
        conn.close()


def _mark_applied(db_path: Path, url: str) -> None:
    """Mark a job as manually applied (same effect as `apply --mark-applied`)."""
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE jobs SET applied_at = ?, apply_status = 'applied' WHERE url = ?",
            (now, url),
        )
        conn.commit()
    finally:
        conn.close()


def serve_overview(port: int = 8765) -> None:
    """Serve the dashboard on localhost with a working hide endpoint.

    GET /          -> freshly generated dashboard (stats recomputed each reload)
    POST /api/hide -> {"url": ..., "role": ...} marks the job hidden in its DB
    """
    import http.server
    import logging as _logging

    log_srv = _logging.getLogger(__name__)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            now = datetime.now(timezone.utc)
            workspaces = discover_workspaces()
            stats = [collect_stats(role, db, now) for role, db in workspaces]
            body = build_html(stats, now).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            actions = {"/api/hide": _hide_job, "/api/applied": _mark_applied}
            if self.path not in actions:
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(length))
                role, url = data["role"], data["url"]
                db_map = dict(discover_workspaces())
                if role not in db_map:
                    raise KeyError(f"unknown workspace: {role}")
                actions[self.path](db_map[role], url)
                log_srv.info("%s [%s] %s", self.path, role, url[:80])
                body = b'{"ok": true}'
                self.send_response(200)
            except Exception as e:
                body = json.dumps({"ok": False, "error": str(e)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence default request logging
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    webbrowser.open(f"http://127.0.0.1:{port}")
    print(f"Dashboard: http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
