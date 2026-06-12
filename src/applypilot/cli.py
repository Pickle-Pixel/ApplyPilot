"""ApplyPilot CLI — the main entry point."""

from __future__ import annotations

import logging
import re
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from applypilot import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

app = typer.Typer(
    name="applypilot",
    help="AI-powered end-to-end job application pipeline.",
    no_args_is_help=True,
)
console = Console()
log = logging.getLogger(__name__)

# Valid pipeline stages (in execution order)
VALID_STAGES = ("discover", "enrich", "score", "tailor", "cover", "pdf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    """Common setup: load env, create dirs, init DB."""
    from applypilot.config import load_env, ensure_dirs
    from applypilot.database import init_db, update_regions

    load_env()
    ensure_dirs()
    conn = init_db()
    update_regions(conn)  # tag new rows with region (e.g. US East Coast)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold]applypilot[/bold] {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """ApplyPilot — AI-powered end-to-end job application pipeline."""


@app.command()
def init() -> None:
    """Run the first-time setup wizard (profile, resume, search config)."""
    from applypilot.wizard.init import run_wizard

    run_wizard()


@app.command()
def run(
    stages: Optional[list[str]] = typer.Argument(
        None,
        help=(
            "Pipeline stages to run. "
            f"Valid: {', '.join(VALID_STAGES)}, all. "
            "Defaults to 'all' if omitted."
        ),
    ),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for tailor/cover stages."),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel threads for discovery/enrichment stages."),
    stream: bool = typer.Option(False, "--stream", help="Run stages concurrently (streaming mode)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview stages without executing."),
    validation: str = typer.Option(
        "normal",
        "--validation",
        help=(
            "Validation strictness for tailor/cover stages. "
            "strict: banned words = errors, judge must pass. "
            "normal: banned words = warnings only (default, recommended for Gemini free tier). "
            "lenient: banned words ignored, LLM judge skipped (fastest, fewest API calls)."
        ),
    ),
    job_type: Optional[str] = typer.Option(
        None, "--job-type", "-t",
        help="Filter tailor stage by job type (e.g. full-time, internship).",
    ),
    eligible_only: bool = typer.Option(
        False, "--eligible-only",
        help="Only tailor jobs explicitly marked eligible=1.",
    ),
    shortlist_only: bool = typer.Option(
        False, "--shortlist-only",
        help="Only tailor hand-picked jobs (see `applypilot shortlist`).",
    ),
) -> None:
    """Run pipeline stages: discover, enrich, score, tailor, cover, pdf."""
    _bootstrap()

    from applypilot.pipeline import run_pipeline

    stage_list = stages if stages else ["all"]

    # Validate stage names
    for s in stage_list:
        if s != "all" and s not in VALID_STAGES:
            console.print(
                f"[red]Unknown stage:[/red] '{s}'. "
                f"Valid stages: {', '.join(VALID_STAGES)}, all"
            )
            raise typer.Exit(code=1)

    # Gate AI stages behind Tier 2
    llm_stages = {"score", "tailor", "cover"}
    if any(s in stage_list for s in llm_stages) or "all" in stage_list:
        from applypilot.config import check_tier
        check_tier(2, "AI scoring/tailoring")

    # Validate the --validation flag value
    valid_modes = ("strict", "normal", "lenient")
    if validation not in valid_modes:
        console.print(
            f"[red]Invalid --validation value:[/red] '{validation}'. "
            f"Choose from: {', '.join(valid_modes)}"
        )
        raise typer.Exit(code=1)

    result = run_pipeline(
        stages=stage_list,
        min_score=min_score,
        dry_run=dry_run,
        stream=stream,
        workers=workers,
        validation_mode=validation,
        job_type=job_type,
        eligible_only=eligible_only,
        shortlist_only=shortlist_only,
    )

    if result.get("errors"):
        raise typer.Exit(code=1)


@app.command()
def apply(
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Max applications to submit."),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel browser workers."),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for job selection."),
    model: str = typer.Option("haiku", "--model", "-m", help="Claude model name."),
    continuous: bool = typer.Option(False, "--continuous", "-c", help="Run forever, polling for new jobs."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without submitting."),
    headless: bool = typer.Option(False, "--headless", help="Run browsers in headless mode."),
    url: Optional[str] = typer.Option(None, "--url", help="Apply to a specific job URL."),
    gen: bool = typer.Option(False, "--gen", help="Generate prompt file for manual debugging instead of running."),
    mark_applied: Optional[str] = typer.Option(None, "--mark-applied", help="Manually mark a job URL as applied."),
    mark_failed: Optional[str] = typer.Option(None, "--mark-failed", help="Manually mark a job URL as failed (provide URL)."),
    fail_reason: Optional[str] = typer.Option(None, "--fail-reason", help="Reason for --mark-failed."),
    reset_failed: bool = typer.Option(False, "--reset-failed", help="Reset all failed jobs for retry."),
    shortlist_only: bool = typer.Option(
        False, "--shortlist-only",
        help="Only apply to hand-picked jobs (see `applypilot shortlist`).",
    ),
) -> None:
    """Launch auto-apply to submit job applications."""
    _bootstrap()

    from applypilot.config import check_tier, PROFILE_PATH as _profile_path
    from applypilot.database import get_connection

    # --- Utility modes (no Chrome/Claude needed) ---

    if mark_applied:
        from applypilot.apply.launcher import mark_job
        mark_job(mark_applied, "applied")
        console.print(f"[green]Marked as applied:[/green] {mark_applied}")
        return

    if mark_failed:
        from applypilot.apply.launcher import mark_job
        mark_job(mark_failed, "failed", reason=fail_reason)
        console.print(f"[yellow]Marked as failed:[/yellow] {mark_failed} ({fail_reason or 'manual'})")
        return

    if reset_failed:
        from applypilot.apply.launcher import reset_failed as do_reset
        count = do_reset()
        console.print(f"[green]Reset {count} failed job(s) for retry.[/green]")
        return

    # --- Full apply mode ---

    # Check 1: Tier 3 required (Claude Code CLI + Chrome)
    check_tier(3, "auto-apply")

    # Check 2: Profile exists
    if not _profile_path.exists():
        console.print(
            "[red]Profile not found.[/red]\n"
            "Run [bold]applypilot init[/bold] to create your profile first."
        )
        raise typer.Exit(code=1)

    # Check 3: Tailored resumes exist (skip for --gen with --url)
    if not (gen and url):
        conn = get_connection()
        ready = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL AND applied_at IS NULL"
        ).fetchone()[0]
        if ready == 0:
            console.print(
                "[red]No tailored resumes ready.[/red]\n"
                "Run [bold]applypilot run score tailor[/bold] first to prepare applications."
            )
            raise typer.Exit(code=1)

    if gen:
        from applypilot.apply.launcher import gen_prompt, BASE_CDP_PORT
        target = url or ""
        if not target:
            console.print("[red]--gen requires --url to specify which job.[/red]")
            raise typer.Exit(code=1)
        prompt_file = gen_prompt(target, min_score=min_score, model=model)
        if not prompt_file:
            console.print("[red]No matching job found for that URL.[/red]")
            raise typer.Exit(code=1)
        mcp_path = _profile_path.parent / ".mcp-apply-0.json"
        console.print(f"[green]Wrote prompt to:[/green] {prompt_file}")
        console.print(f"\n[bold]Run manually:[/bold]")
        console.print(
            f"  claude --model {model} -p "
            f"--mcp-config {mcp_path} "
            f"--permission-mode bypassPermissions < {prompt_file}"
        )
        return

    from applypilot.apply.launcher import main as apply_main

    effective_limit = limit if limit is not None else (0 if continuous else 1)

    console.print("\n[bold blue]Launching Auto-Apply[/bold blue]")
    console.print(f"  Limit:    {'unlimited' if continuous else effective_limit}")
    console.print(f"  Workers:  {workers}")
    console.print(f"  Model:    {model}")
    console.print(f"  Headless: {headless}")
    console.print(f"  Dry run:  {dry_run}")
    if url:
        console.print(f"  Target:   {url}")
    console.print()

    apply_main(
        limit=effective_limit,
        target_url=url,
        min_score=min_score,
        headless=headless,
        model=model,
        dry_run=dry_run,
        continuous=continuous,
        workers=workers,
        shortlist_only=shortlist_only,
    )


@app.command()
def status() -> None:
    """Show pipeline statistics from the database."""
    _bootstrap()

    from applypilot.database import get_stats

    stats = get_stats()

    console.print("\n[bold]ApplyPilot Pipeline Status[/bold]\n")

    # Summary table
    summary = Table(title="Pipeline Overview", show_header=True, header_style="bold cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Count", justify="right")

    summary.add_row("Total jobs discovered", str(stats["total"]))
    summary.add_row("With full description", str(stats["with_description"]))
    summary.add_row("Pending enrichment", str(stats["pending_detail"]))
    summary.add_row("Enrichment errors", str(stats["detail_errors"]))
    summary.add_row("Scored by LLM", str(stats["scored"]))
    summary.add_row("Pending scoring", str(stats["unscored"]))
    summary.add_row("Tailored resumes", str(stats["tailored"]))
    summary.add_row("Pending tailoring (7+)", str(stats["untailored_eligible"]))
    summary.add_row("Cover letters", str(stats["with_cover_letter"]))
    summary.add_row("Ready to apply", str(stats["ready_to_apply"]))
    summary.add_row("Applied", str(stats["applied"]))
    summary.add_row("Apply errors", str(stats["apply_errors"]))

    console.print(summary)

    # Score distribution
    if stats["score_distribution"]:
        dist_table = Table(title="\nScore Distribution", show_header=True, header_style="bold yellow")
        dist_table.add_column("Score", justify="center")
        dist_table.add_column("Count", justify="right")
        dist_table.add_column("Bar")

        max_count = max(count for _, count in stats["score_distribution"]) or 1
        for score, count in stats["score_distribution"]:
            bar_len = int(count / max_count * 30)
            if score >= 7:
                color = "green"
            elif score >= 5:
                color = "yellow"
            else:
                color = "red"
            bar = f"[{color}]{'=' * bar_len}[/{color}]"
            dist_table.add_row(str(score), str(count), bar)

        console.print(dist_table)

    # By site
    if stats["by_site"]:
        site_table = Table(title="\nJobs by Source", show_header=True, header_style="bold magenta")
        site_table.add_column("Site")
        site_table.add_column("Count", justify="right")

        for site, count in stats["by_site"]:
            site_table.add_row(site or "Unknown", str(count))

        console.print(site_table)

    console.print()


@app.command()
def dashboard() -> None:
    """Generate and open the HTML dashboard in your browser."""
    _bootstrap()

    from applypilot.view import open_dashboard

    open_dashboard()


@app.command()
def doctor() -> None:
    """Check your setup and diagnose missing requirements."""
    import shutil
    from applypilot.config import (
        load_env, PROFILE_PATH, RESUME_PATH, RESUME_PDF_PATH,
        SEARCH_CONFIG_PATH, ENV_PATH, get_chrome_path,
    )

    load_env()

    ok_mark = "[green]OK[/green]"
    fail_mark = "[red]MISSING[/red]"
    warn_mark = "[yellow]WARN[/yellow]"

    results: list[tuple[str, str, str]] = []  # (check, status, note)

    # --- Tier 1 checks ---
    # Profile
    if PROFILE_PATH.exists():
        results.append(("profile.json", ok_mark, str(PROFILE_PATH)))
    else:
        results.append(("profile.json", fail_mark, "Run 'applypilot init' to create"))

    # Resume
    if RESUME_PATH.exists():
        results.append(("resume.txt", ok_mark, str(RESUME_PATH)))
    elif RESUME_PDF_PATH.exists():
        results.append(("resume.txt", warn_mark, "Only PDF found — plain-text needed for AI stages"))
    else:
        results.append(("resume.txt", fail_mark, "Run 'applypilot init' to add your resume"))

    # Search config
    if SEARCH_CONFIG_PATH.exists():
        results.append(("searches.yaml", ok_mark, str(SEARCH_CONFIG_PATH)))
    else:
        results.append(("searches.yaml", warn_mark, "Will use example config — run 'applypilot init'"))

    # jobspy (discovery dep installed separately)
    try:
        import jobspy  # noqa: F401
        results.append(("python-jobspy", ok_mark, "Job board scraping available"))
    except ImportError:
        results.append(("python-jobspy", warn_mark,
                        "pip install --no-deps python-jobspy && pip install pydantic tls-client requests markdownify regex"))

    # --- Tier 2 checks ---
    import os
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_local = bool(os.environ.get("LLM_URL"))
    if has_gemini:
        model = os.environ.get("LLM_MODEL", "gemini-2.0-flash")
        results.append(("LLM API key", ok_mark, f"Gemini ({model})"))
    elif has_openai:
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        results.append(("LLM API key", ok_mark, f"OpenAI ({model})"))
    elif has_local:
        results.append(("LLM API key", ok_mark, f"Local: {os.environ.get('LLM_URL')}"))
    else:
        results.append(("LLM API key", fail_mark,
                        "Set GEMINI_API_KEY in ~/.applypilot/.env (run 'applypilot init')"))

    # --- Tier 3 checks ---
    # Claude Code CLI
    claude_bin = shutil.which("claude")
    if claude_bin:
        results.append(("Claude Code CLI", ok_mark, claude_bin))
    else:
        results.append(("Claude Code CLI", fail_mark,
                        "Install from https://claude.ai/code (needed for auto-apply)"))

    # Chrome
    try:
        chrome_path = get_chrome_path()
        results.append(("Chrome/Chromium", ok_mark, chrome_path))
    except FileNotFoundError:
        results.append(("Chrome/Chromium", fail_mark,
                        "Install Chrome or set CHROME_PATH env var (needed for auto-apply)"))

    # Node.js / npx (for Playwright MCP)
    npx_bin = shutil.which("npx")
    if npx_bin:
        results.append(("Node.js (npx)", ok_mark, npx_bin))
    else:
        results.append(("Node.js (npx)", fail_mark,
                        "Install Node.js 18+ from nodejs.org (needed for auto-apply)"))

    # CapSolver (optional)
    capsolver = os.environ.get("CAPSOLVER_API_KEY")
    if capsolver:
        results.append(("CapSolver API key", ok_mark, "CAPTCHA solving enabled"))
    else:
        results.append(("CapSolver API key", "[dim]optional[/dim]",
                        "Set CAPSOLVER_API_KEY in .env for CAPTCHA solving"))

    # --- Render results ---
    console.print()
    console.print("[bold]ApplyPilot Doctor[/bold]\n")

    col_w = max(len(r[0]) for r in results) + 2
    for check, status, note in results:
        pad = " " * (col_w - len(check))
        console.print(f"  {check}{pad}{status}  [dim]{note}[/dim]")

    console.print()

    # Tier summary
    from applypilot.config import get_tier, TIER_LABELS
    tier = get_tier()
    console.print(f"[bold]Current tier: Tier {tier} — {TIER_LABELS[tier]}[/bold]")

    if tier == 1:
        console.print("[dim]  → Tier 2 unlocks: scoring, tailoring, cover letters (needs LLM API key)[/dim]")
        console.print("[dim]  → Tier 3 unlocks: auto-apply (needs Claude Code CLI + Chrome + Node.js)[/dim]")
    elif tier == 2:
        console.print("[dim]  → Tier 3 unlocks: auto-apply (needs Claude Code CLI + Chrome + Node.js)[/dim]")

    console.print()


@app.command()
def overview(
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output HTML path."),
    no_open: bool = typer.Option(False, "--no-open", help="Generate without opening the browser."),
    serve: bool = typer.Option(False, "--serve", help="Serve live on localhost (hide buttons persist to DB)."),
    port: int = typer.Option(8765, "--port", help="Port for --serve mode."),
) -> None:
    """Cross-workspace dashboard: jobs per role over 1h/24h/72h, platforms, ATS, applied status."""
    from applypilot.overview import generate_overview, open_overview, serve_overview

    if serve:
        serve_overview(port)
    elif no_open:
        path = generate_overview(output)
        console.print(f"[green]Overview generated:[/green] {path}")
    else:
        open_overview(output)
        console.print("[green]Overview opened in browser.[/green]")


@app.command()
def shortlist(
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for the picker list."),
    limit: int = typer.Option(50, "--limit", "-l", help="Max jobs to show in the picker."),
    show: bool = typer.Option(False, "--list", help="Show the current shortlist and exit."),
    clear: bool = typer.Option(False, "--clear", help="Clear all shortlist marks and exit."),
    east_only: bool = typer.Option(False, "--east-only", help="Only show US East Coast jobs."),
) -> None:
    """Hand-pick which jobs to tailor/apply (use with --shortlist-only)."""
    _bootstrap()

    from applypilot.database import get_connection

    conn = get_connection()

    if clear:
        n = conn.execute("UPDATE jobs SET shortlisted = NULL WHERE shortlisted = 1").rowcount
        conn.commit()
        console.print(f"[green]Cleared {n} shortlist mark(s).[/green]")
        return

    if show:
        rows = conn.execute(
            "SELECT fit_score, title, site, location, url FROM jobs "
            "WHERE shortlisted = 1 ORDER BY fit_score DESC"
        ).fetchall()
        if not rows:
            console.print("[yellow]Shortlist is empty.[/yellow] Run `applypilot shortlist` to pick jobs.")
            return
        table = Table(title=f"Shortlist ({len(rows)} jobs)")
        table.add_column("Score", justify="right")
        table.add_column("Title", max_width=45)
        table.add_column("Site")
        table.add_column("Location", max_width=25)
        for score, title, site, location, _url in rows:
            table.add_row(str(score), title or "", site or "", location or "")
        console.print(table)
        return

    # --- Interactive picker ---
    east_clause = "AND region = 'east' " if east_only else ""
    rows = conn.execute(
        "SELECT url, fit_score, title, site, location, shortlisted, region, "
        "       tailored_resume_path, applied_at "
        "FROM jobs "
        f"WHERE fit_score >= ? AND applied_at IS NULL "
        f"AND COALESCE(hidden, 0) != 1 {east_clause}"
        "ORDER BY fit_score DESC, (region = 'east') DESC, discovered_at DESC LIMIT ?",
        (min_score, limit),
    ).fetchall()

    if not rows:
        console.print(f"[yellow]No scored jobs with fit_score >= {min_score}.[/yellow] "
                      "Run `applypilot run score` first.")
        return

    selected = {i for i, r in enumerate(rows) if r["shortlisted"] == 1}

    def _render() -> None:
        table = Table(title=f"Scored jobs (score >= {min_score}, top {len(rows)})")
        table.add_column("#", justify="right")
        table.add_column("Pick")
        table.add_column("Score", justify="right")
        table.add_column("Title", max_width=45)
        table.add_column("Site")
        table.add_column("Location", max_width=22)
        table.add_column("Region")
        table.add_column("Tailored")
        for i, r in enumerate(rows):
            mark = "[green]★[/green]" if i in selected else ""
            tailored = "✓" if r["tailored_resume_path"] else ""
            region = "[cyan]美东[/cyan]" if r["region"] == "east" else ""
            style = "bold" if i in selected else None
            table.add_row(str(i + 1), mark, str(r["fit_score"]), r["title"] or "",
                          r["site"] or "", r["location"] or "", region, tailored, style=style)
        console.print(table)

    _render()
    console.print(
        "[dim]Toggle picks: numbers/ranges (e.g. 1 3 5-8) | a=all | n=none | "
        "p=reprint list | Enter=save & exit[/dim]"
    )

    while True:
        try:
            raw = console.input("[bold]shortlist>[/bold] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\nAborted, nothing saved.")
            return

        if not raw:
            break
        if raw == "a":
            selected = set(range(len(rows)))
        elif raw == "n":
            selected = set()
        elif raw == "p":
            _render()
            continue
        else:
            ok = True
            for tok in raw.replace(",", " ").split():
                m = re.fullmatch(r"(\d+)(?:-(\d+))?", tok)
                if not m:
                    console.print(f"[red]Bad input: {tok}[/red]")
                    ok = False
                    break
                lo = int(m.group(1))
                hi = int(m.group(2) or lo)
                for n in range(lo, hi + 1):
                    if not 1 <= n <= len(rows):
                        console.print(f"[red]Out of range: {n}[/red]")
                        ok = False
                        break
                    i = n - 1
                    selected.symmetric_difference_update({i})
                if not ok:
                    break
            if not ok:
                continue
        console.print(f"  [green]{len(selected)}[/green] selected")

    # Persist: only touch the rows that were shown
    for i, r in enumerate(rows):
        conn.execute(
            "UPDATE jobs SET shortlisted = ? WHERE url = ?",
            (1 if i in selected else None, r["url"]),
        )
    conn.commit()
    console.print(
        f"[bold green]Saved: {len(selected)} job(s) shortlisted.[/bold green] "
        "Next: [bold]applypilot run tailor --shortlist-only[/bold]"
    )


@app.command()
def check_dupes(
    workspaces: Optional[list[str]] = typer.Argument(
        None,
        help="Workspace directories to compare. Defaults to all ~/.applypilot* dirs containing a database.",
    ),
    applied_only: bool = typer.Option(
        False, "--applied-only",
        help="Only report jobs applied to from more than one workspace.",
    ),
) -> None:
    """Find jobs that appear in multiple workspaces (cross-role duplicate check)."""
    import sqlite3
    from pathlib import Path

    if workspaces:
        dirs = [Path(w).expanduser() for w in workspaces]
    else:
        dirs = sorted(p for p in Path.home().glob(".applypilot*") if (p / "applypilot.db").exists())

    db_paths = [(d, d / "applypilot.db") for d in dirs if (d / "applypilot.db").exists()]
    if len(db_paths) < 2:
        console.print("[yellow]Need at least 2 workspaces with a database to compare.[/yellow]")
        for d in dirs:
            mark = "[green]✓[/green]" if (d / "applypilot.db").exists() else "[red]✗ no applypilot.db[/red]"
            console.print(f"  {d}  {mark}")
        raise typer.Exit()

    # url -> list of (workspace_name, title, applied_at)
    seen: dict[str, list[tuple[str, str, str | None]]] = {}
    for d, db in db_paths:
        name = d.name.replace(".applypilot-", "").replace(".applypilot", "default")
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT url, title, applied_at FROM jobs").fetchall()
        finally:
            conn.close()
        for url, title, applied_at in rows:
            seen.setdefault(url, []).append((name, title or "", applied_at))

    dupes = {u: hits for u, hits in seen.items() if len(hits) > 1}
    if applied_only:
        dupes = {u: hits for u, hits in dupes.items()
                 if sum(1 for _, _, a in hits if a) > 1}

    console.print(f"\nCompared {len(db_paths)} workspaces: "
                  + ", ".join(d.name for d, _ in db_paths))

    if not dupes:
        console.print("[bold green]No cross-workspace duplicates found.[/bold green]")
        return

    double_applied = {u: hits for u, hits in dupes.items()
                      if sum(1 for _, _, a in hits if a) > 1}

    table = Table(title=f"{len(dupes)} jobs in multiple workspaces")
    table.add_column("Title", max_width=50)
    table.add_column("Workspaces")
    table.add_column("Applied from")
    table.add_column("URL", overflow="fold", max_width=60)
    for url, hits in sorted(dupes.items(), key=lambda kv: -len(kv[1])):
        title = hits[0][1]
        ws = ", ".join(h[0] for h in hits)
        applied = ", ".join(h[0] for h in hits if h[2]) or "-"
        style = "red" if url in double_applied else None
        table.add_row(title, ws, applied, url, style=style)
    console.print(table)

    if double_applied:
        console.print(f"\n[bold red]⚠ {len(double_applied)} jobs were applied to from more than one workspace![/bold red]")
    else:
        console.print("\n[green]No job has been applied to twice.[/green] "
                      "[dim]Rows above are only discovered in multiple workspaces — "
                      "the apply stage will catch them if you check before running apply.[/dim]")


@app.command()
def reset_db(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt."),
) -> None:
    """Delete the jobs database and all WAL files to start fresh."""
    from applypilot.config import load_env, DB_PATH

    load_env()

    files = [DB_PATH, DB_PATH.with_suffix(".db-shm"), DB_PATH.with_suffix(".db-wal")]
    existing = [f for f in files if f.exists()]

    if not existing:
        console.print("[yellow]No database files found, nothing to delete.[/yellow]")
        return

    if not force:
        console.print("[red]This will delete:[/red]")
        for f in existing:
            console.print(f"  {f}")
        confirm = typer.confirm("Are you sure?")
        if not confirm:
            console.print("Aborted.")
            raise typer.Exit()

    for f in existing:
        f.unlink()
        console.print(f"[green]Deleted:[/green] {f}")

    console.print("[bold green]Database reset complete. Run 'applypilot run' to start fresh.[/bold green]")


if __name__ == "__main__":
    app()
