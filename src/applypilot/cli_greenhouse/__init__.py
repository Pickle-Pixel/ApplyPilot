"""Greenhouse CLI commands for managing Greenhouse ATS employers."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import typer
import yaml
from rich.console import Console
from rich.table import Table

console = Console()

# API endpoint templates
API_BASE = "https://boards-api.greenhouse.io/v1/boards"
API_TEMPLATE = f"{API_BASE}/{{slug}}/jobs"

# Known slug fixes
KNOWN_FIXES = {"notion": "notionhq"}

app = typer.Typer(
    name="greenhouse",
    help="Manage Greenhouse ATS employers and verify configurations.",
    no_args_is_help=True,
)


def _load_config(config_path: Optional[Path] = None) -> dict:
    """Load greenhouse.yaml configuration."""
    if config_path is None:
        # Try user config first, then package config
        from applypilot.config import APP_DIR, CONFIG_DIR

        user_path = APP_DIR / "greenhouse.yaml"
        if user_path.exists():
            config_path = user_path
        else:
            config_path = CONFIG_DIR / "greenhouse.yaml"

    if not config_path.exists():
        console.print(f"[red]Config not found:[/red] {config_path}")
        raise typer.Exit(code=1)

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data.get("employers", {})


def _check_slug(slug: str) -> Tuple[bool, Optional[int], Optional[str]]:
    """Check if a slug is valid via Greenhouse API."""
    url = API_TEMPLATE.format(slug=slug)

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers={"Accept": "application/json"})
    except httpx.RequestError as e:
        return False, None, f"Request error: {e}"

    if resp.status_code == 200:
        try:
            data = resp.json()
            jobs = data.get("jobs", [])
            total = len(jobs)
            return True, total, None
        except ValueError:
            return True, None, "Invalid JSON"
    elif resp.status_code == 404:
        return False, None, "Not found"
    elif resp.status_code == 429:
        return False, None, "Rate limited"
    else:
        return False, None, f"HTTP {resp.status_code}"


def _generate_variations(name: str) -> List[str]:
    """Generate slug variations for a company name."""
    name = name.lower().strip()
    variations = [name]

    # No spaces
    no_spaces = name.replace(" ", "")
    if no_spaces != name:
        variations.append(no_spaces)

    # Dashes
    dash = name.replace(" ", "-")
    if dash != name:
        variations.append(dash)

    # Underscores
    underscore = name.replace(" ", "_")
    if underscore != name:
        variations.append(underscore)

    # First word only
    first_word = name.split()[0] if name else ""
    if first_word and first_word not in variations:
        variations.append(first_word)

    # Suffixes
    for suffix in ["careers", "jobs"]:
        plain = f"{name}{suffix}"
        dash_suf = f"{name}-{suffix}"
        if plain not in variations:
            variations.append(plain)
        if dash_suf not in variations:
            variations.append(dash_suf)

    # Deduplicate while preserving order
    seen = set()
    return [v for v in variations if not (v in seen or seen.add(v))]


@app.command()
def verify(
    slug: str = typer.Argument(..., help="Company slug to verify"),
    try_variations: bool = typer.Option(
        True, "--variations/--no-variations", help="Try common slug variations if not found"
    ),
) -> None:
    """Verify a Greenhouse company slug exists."""
    console.print(f"Verifying [bold]{slug}[/bold]...")

    is_valid, total, error = _check_slug(slug)

    if is_valid:
        console.print(f"[green]✓[/green] {slug}: {total or 'jobs found'}")
        raise typer.Exit(code=0)

    console.print(f"[red]✗[/red] {slug}: {error}")

    if not try_variations:
        raise typer.Exit(code=1)

    # Try variations
    console.print("\nTrying variations...")
    variations = _generate_variations(slug)

    for i, variant in enumerate(variations[1:], 1):  # Skip original
        time.sleep(1)  # Polite delay
        is_valid, total, error = _check_slug(variant)

        if is_valid:
            console.print(f"[green]✓[/green] {variant}: {total or 'jobs found'}")
            raise typer.Exit(code=0)
        else:
            console.print(f"[red]✗[/red] {variant}: {error}")

    console.print("\n[yellow]No valid slug found[/yellow]")
    raise typer.Exit(code=1)


@app.command()
def discover(
    name: Optional[str] = typer.Argument(None, help="Company name to search for"),
    url: Optional[str] = typer.Option(None, "--url", help="Career page URL to scrape"),
) -> None:
    """Discover Greenhouse slugs from company name or career URL."""
    if not name and not url:
        console.print("[red]Error:[/red] Provide either a company name or --url")
        raise typer.Exit(code=1)

    if url:
        # Scrape URL for Greenhouse references
        console.print(f"Analyzing URL: {url}")

        try:
            with httpx.Client(timeout=15.0, follow_redirects=True) as client:
                resp = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                html = resp.text
        except Exception as e:
            console.print(f"[red]Error fetching URL:[/red] {e}")
            raise typer.Exit(code=1)

        # Extract slugs from HTML
        patterns = [
            r"boards\.greenhouse\.io/(\w+)",
            r"job-boards\.greenhouse\.io/(\w+)",
            r"api\.greenhouse\.io/v1/boards/(\w+)",
            r"greenhouse\.io/embed/job_board\?for=(\w+)",
        ]

        slugs = set()
        for pattern in patterns:
            matches = re.findall(pattern, html)
            slugs.update(matches)

        if not slugs:
            # Try hostname as fallback
            hostname = urlparse(str(resp.url)).hostname or ""
            if hostname:
                base = hostname.replace("careers.", "").replace("jobs.", "").replace("www.", "").split(".")[0]
                slugs.add(base)

        candidates = list(slugs)
    else:
        # Generate from name
        console.print(f"Trying variations of [bold]{name}[/bold]...")
        candidates = _generate_variations(name)

    # Verify candidates
    console.print(f"\nChecking {len(candidates)} candidates...\n")

    for i, slug in enumerate(candidates, 1):
        console.print(f"  ({i}/{len(candidates)}) {slug}...", end=" ")
        is_valid, total, error = _check_slug(slug)

        if is_valid:
            console.print(f"[green]✓ {total} jobs[/green]")
        else:
            console.print(f"[red]✗ {error}[/red]")

        if i < len(candidates):
            time.sleep(1)  # Polite delay

    raise typer.Exit(code=0)


@app.command()
def validate(
    fix: bool = typer.Option(False, "--fix", help="Auto-fix known slug issues"),
    config_path: Optional[Path] = typer.Option(None, "--config", help="Path to greenhouse.yaml"),
) -> None:
    """Validate all companies in greenhouse.yaml configuration."""
    employers = _load_config(config_path)
    slugs = list(employers.keys())
    total = len(slugs)

    if total == 0:
        console.print("[yellow]No employers found in configuration[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"Validating [bold]{total}[/bold] companies...\n")

    valid_count = 0
    invalid = {}
    fixed = []

    for i, slug in enumerate(slugs, 1):
        console.print(f"Checking {i}/{total}...", end="\r")

        is_valid, total_jobs, error = _check_slug(slug)

        if is_valid:
            valid_count += 1
            console.print(f"[green]✓[/green] {slug}: {total_jobs} jobs")
        else:
            # Try auto-fix
            if fix and slug in KNOWN_FIXES:
                new_slug = KNOWN_FIXES[slug]
                is_valid2, total2, _ = _check_slug(new_slug)

                if is_valid2:
                    console.print(f"[yellow]✗[/yellow] {slug}: {error}")
                    console.print(f"  [green]→ Fixed:[/green] {slug} → {new_slug} ({total2} jobs)")
                    fixed.append((slug, new_slug))
                    valid_count += 1
                    continue

            invalid[slug] = error
            console.print(f"[red]✗[/red] {slug}: {error}")

        time.sleep(0.5)  # Polite delay

    # Summary
    console.print(f"\n[bold]Summary:[/bold] {valid_count}/{total} valid")

    if invalid:
        console.print(f"\n[red]Invalid:[/red] {', '.join(invalid.keys())}")

    if fixed:
        console.print(f"\n[green]Fixed:[/green] {len(fixed)} issue(s)")

    if valid_count == total:
        console.print("\n[green]All companies valid![/green]")
        raise typer.Exit(code=0)
    else:
        raise typer.Exit(code=1)


@app.command()
def list_employers(
    config_path: Optional[Path] = typer.Option(None, "--config", help="Path to greenhouse.yaml"),
) -> None:
    """List all configured Greenhouse employers."""
    employers = _load_config(config_path)

    if not employers:
        console.print("[yellow]No employers configured[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="Greenhouse Employers", show_header=True)
    table.add_column("Slug", style="cyan")
    table.add_column("Name", style="green")

    for slug, data in sorted(employers.items()):
        name = data.get("name", slug)
        table.add_row(slug, name)

    console.print(table)
    console.print(f"\nTotal: {len(employers)} employers")


@app.command()
def add_job(
    url: str = typer.Argument(..., help="Greenhouse job URL to add"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without saving to database"),
) -> None:
    """Add a specific Greenhouse job from URL and display structured data."""
    import json
    from rich.panel import Panel
    from rich.json import JSON
    
    console.print(f"🔗 Processing URL: {url}")
    console.print()
    
    # Extract company slug and job ID from URL
    match = re.search(r'greenhouse\.io/(\w+)/jobs/(\d+)', url)
    if not match:
        console.print("[red]✗[/red] Invalid Greenhouse URL format")
        console.print("[dim]Expected: https://boards.greenhouse.io/{company}/jobs/{job_id}[/dim]")
        raise typer.Exit(code=1)
    
    company_slug = match.group(1)
    job_id = match.group(2)
    
    console.print(f"📍 Company: {company_slug}")
    console.print(f"🆔 Job ID: {job_id}")
    console.print()
    
    # Fetch all jobs for this company
    console.print("⬇️  Fetching job data...")
    from applypilot.discovery.greenhouse import fetch_jobs_api, parse_api_response, _store_jobs
    
    data = fetch_jobs_api(company_slug, content=True)
    
    if not data:
        console.print("[red]✗[/red] Failed to fetch jobs from API")
        raise typer.Exit(code=1)
    
    # Find the specific job
    jobs = parse_api_response(data, company_slug.replace('-', ' ').title(), '')
    job = next((j for j in jobs if str(j.get('job_id')) == job_id), None)
    
    if not job:
        console.print(f"[red]✗[/red] Job {job_id} not found")
        raise typer.Exit(code=1)
    
    # Display structured data
    console.print("=" * 70)
    console.print("[bold green]✓ Job Found[/bold green]")
    console.print("=" * 70)
    console.print()
    
    # Basic info table
    info_table = Table(show_header=False, box=None)
    info_table.add_column("Field", style="cyan", width=15)
    info_table.add_column("Value", style="white")
    
    info_table.add_row("Job ID", str(job.get('job_id', 'N/A')))
    info_table.add_row("Title", job.get('title', 'N/A'))
    info_table.add_row("Company", job.get('company', 'N/A'))
    info_table.add_row("Location", job.get('location', 'N/A'))
    info_table.add_row("Department", job.get('department', 'N/A'))
    info_table.add_row("Strategy", job.get('strategy', 'N/A'))
    info_table.add_row("URL", job.get('url', 'N/A')[:60] + "...")
    info_table.add_row("Updated", job.get('updated_at', 'N/A'))
    
    console.print(Panel(info_table, title="📋 Job Information", border_style="green"))
    console.print()
    
    # Description panel
    desc = job.get('description', '')
    if desc:
        if len(desc) > 800:
            desc = desc[:800] + "..."
        console.print(Panel(desc, title="📝 Description", border_style="blue"))
        console.print()
    
    # Full structured data (JSON)
    console.print("📊 Full Structured Data (as stored in database):")
    
    # Create a copy with limited fields for cleaner display
    display_job = {
        "job_id": job.get("job_id"),
        "title": job.get("title"),
        "company": job.get("company"),
        "location": job.get("location"),
        "department": job.get("department"),
        "description": job.get("description", "")[:200] + "..." if len(job.get("description", "")) > 200 else job.get("description"),
        "url": job.get("url"),
        "strategy": job.get("strategy"),
        "updated_at": job.get("updated_at"),
    }
    console.print(JSON(json.dumps(display_job, indent=2, default=str)))
    console.print()
    
    if dry_run:
        console.print("[yellow]🏃 Dry run mode - job NOT saved to database[/yellow]")
    else:
        # Store in database
        console.print("💾 Saving to database...")
        try:
            new, existing = _store_jobs([job])
            if new:
                console.print(f"[green]✓[/green] Job saved successfully (new)")
            elif existing:
                console.print(f"[yellow]⚠[/yellow] Job already exists in database")
            else:
                console.print(f"[green]✓[/green] Job processed")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to save: {e}")
            raise typer.Exit(code=1)
    
    console.print()
    console.print("=" * 70)
    console.print("[dim]Next: Run 'applypilot run enrich score' to process this job[/dim]")
