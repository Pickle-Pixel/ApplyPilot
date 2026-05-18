"""ApplyPilot first-time setup wizard.

Interactive flow that creates ~/.applypilot/ with:
  - resume.txt (and optionally resume.pdf)
  - profile.json
  - searches.yaml
  - .env (LLM API key)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from applypilot.config import (
    APP_DIR,
    ENV_PATH,
    PROFILE_PATH,
    RESUME_PATH,
    RESUME_PDF_PATH,
    SEARCH_CONFIG_PATH,
    ensure_dirs,
)

console = Console()


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def _setup_resume() -> None:
    """Prompt for resume file and copy into APP_DIR."""
    console.print(Panel("[bold]Step 1: Resume[/bold]\nPoint to your master resume file (.txt or .pdf)."))

    while True:
        path_str = Prompt.ask("Resume file path")
        src = Path(path_str.strip().strip('"').strip("'")).expanduser().resolve()

        if not src.exists():
            console.print(f"[red]File not found:[/red] {src}")
            continue

        suffix = src.suffix.lower()
        if suffix not in (".txt", ".pdf"):
            console.print("[red]Unsupported format.[/red] Provide a .txt or .pdf file.")
            continue

        if suffix == ".txt":
            shutil.copy2(src, RESUME_PATH)
            console.print(f"[green]Copied to {RESUME_PATH}[/green]")
        elif suffix == ".pdf":
            shutil.copy2(src, RESUME_PDF_PATH)
            console.print(f"[green]Copied to {RESUME_PDF_PATH}[/green]")

            # Also ask for a plain-text version for LLM consumption
            txt_path_str = Prompt.ask(
                "Plain-text version of your resume (.txt)",
                default="",
            )
            if txt_path_str.strip():
                txt_src = Path(txt_path_str.strip().strip('"').strip("'")).expanduser().resolve()
                if txt_src.exists():
                    shutil.copy2(txt_src, RESUME_PATH)
                    console.print(f"[green]Copied to {RESUME_PATH}[/green]")
                else:
                    console.print("[yellow]File not found, skipping plain-text copy.[/yellow]")
        break


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

def _setup_profile() -> dict:
    """Walk through profile questions and return a nested profile dict."""
    console.print(Panel("[bold]Step 2: Profile[/bold]\nTell ApplyPilot about yourself. This powers scoring, tailoring, and auto-fill."))

    profile: dict = {}

    # -- Personal --
    console.print("\n[bold cyan]Personal Information[/bold cyan]")
    full_name = Prompt.ask("Full name")
    profile["personal"] = {
        "full_name": full_name,
        "preferred_name": Prompt.ask("Preferred/nickname (leave blank to use first name)", default=""),
        "email": Prompt.ask("Email address"),
        "phone": Prompt.ask("Phone number", default=""),
        "city": Prompt.ask("City"),
        "province_state": Prompt.ask("Province/State (e.g. Ontario, California)", default=""),
        "country": Prompt.ask("Country"),
        "postal_code": Prompt.ask("Postal/ZIP code", default=""),
        "address": Prompt.ask("Street address (optional, used for form auto-fill)", default=""),
        "linkedin_url": Prompt.ask("LinkedIn URL", default=""),
        "github_url": Prompt.ask("GitHub URL (optional)", default=""),
        "portfolio_url": Prompt.ask("Portfolio URL (optional)", default=""),
        "website_url": Prompt.ask("Personal website URL (optional)", default=""),
        "password": Prompt.ask("Job site password (used for login walls during auto-apply)", password=True, default=""),
    }

    # -- Work Authorization --
    console.print("\n[bold cyan]Work Authorization[/bold cyan]")
    profile["work_authorization"] = {
        "legally_authorized_to_work": Confirm.ask("Are you legally authorized to work in your target country?"),
        "require_sponsorship": Confirm.ask("Will you now or in the future need sponsorship?"),
        "work_permit_type": Prompt.ask("Work permit type (e.g. Citizen, PR, Open Work Permit — leave blank if N/A)", default=""),
    }

    # -- Compensation --
    console.print("\n[bold cyan]Compensation[/bold cyan]")
    console.print("[dim]Warehouse roles are typically quoted hourly (e.g. 22 CAD/hour).[/dim]")
    salary = Prompt.ask("Expected wage (hourly rate, number only)", default="22")
    salary_currency = Prompt.ask("Currency", default="CAD")
    salary_unit = Prompt.ask("Unit", choices=["hour", "year"], default="hour")
    salary_range = Prompt.ask("Acceptable range (e.g. 20-28)", default="")
    range_parts = salary_range.split("-") if "-" in salary_range else [salary, salary]
    profile["compensation"] = {
        "salary_expectation": salary,
        "salary_currency": salary_currency,
        "salary_unit": salary_unit,
        "salary_range_min": range_parts[0].strip(),
        "salary_range_max": range_parts[1].strip() if len(range_parts) > 1 else range_parts[0].strip(),
    }

    # -- Experience --
    console.print("\n[bold cyan]Experience[/bold cyan]")
    current_title = Prompt.ask("Current/most recent job title", default="Warehouse Associate")
    target_role = Prompt.ask(
        "Target role (what you're applying for, e.g. 'Order Picker', 'Warehouse Worker')",
        default=current_title,
    )
    profile["experience"] = {
        "years_of_experience_total": Prompt.ask("Years of warehouse/logistics experience", default="0"),
        "education_level": Prompt.ask("Highest education (e.g. High School Diploma, GED, College)", default="High School Diploma"),
        "current_title": current_title,
        "target_role": target_role,
    }

    # -- Skills Boundary --
    console.print("\n[bold cyan]Skills[/bold cyan] (comma-separated)")
    equipment = Prompt.ask(
        "Equipment you can operate (e.g. pallet jack, RF scanner, forklift, reach truck)",
        default="",
    )
    certifications = Prompt.ask(
        "Certifications (e.g. Forklift, WHMIS 2015, First Aid, OFA Level 1)",
        default="",
    )
    systems = Prompt.ask(
        "Warehouse systems you've used (e.g. SAP WM, Manhattan WMS, Oracle WMS)",
        default="",
    )
    physical = Prompt.ask(
        "Physical capabilities (e.g. lift 50 lbs, stand full shift, cooler/freezer work)",
        default="",
    )
    profile["skills_boundary"] = {
        "equipment": [s.strip() for s in equipment.split(",") if s.strip()],
        "certifications": [s.strip() for s in certifications.split(",") if s.strip()],
        "systems": [s.strip() for s in systems.split(",") if s.strip()],
        "physical": [s.strip() for s in physical.split(",") if s.strip()],
    }

    # -- Resume Facts (preserved truths for tailoring) --
    console.print("\n[bold cyan]Resume Facts[/bold cyan]")
    console.print("[dim]These are preserved exactly during resume tailoring — the AI will never change them.[/dim]")
    companies = Prompt.ask("Past employers to always keep (comma-separated)", default="")
    projects = Prompt.ask(
        "Notable achievements to always keep (e.g. 'Led night-shift pick team of 6')",
        default="",
    )
    school = Prompt.ask("School name(s) to preserve", default="")
    metrics = Prompt.ask(
        "Real metrics to preserve (e.g. '150 orders/shift, 99% pick accuracy, zero safety incidents')",
        default="",
    )
    profile["resume_facts"] = {
        "preserved_companies": [s.strip() for s in companies.split(",") if s.strip()],
        "preserved_projects": [s.strip() for s in projects.split(",") if s.strip()],
        "preserved_school": school.strip(),
        "real_metrics": [s.strip() for s in metrics.split(",") if s.strip()],
    }

    # -- EEO Voluntary (defaults) --
    profile["eeo_voluntary"] = {
        "gender": "Decline to self-identify",
        "race_ethnicity": "Decline to self-identify",
        "veteran_status": "Decline to self-identify",
        "disability_status": "Decline to self-identify",
    }

    # -- Availability --
    profile["availability"] = {
        "earliest_start_date": Prompt.ask("Earliest start date", default="Immediately"),
    }

    # Save
    PROFILE_PATH.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[green]Profile saved to {PROFILE_PATH}[/green]")
    return profile


# ---------------------------------------------------------------------------
# Search config
# ---------------------------------------------------------------------------

def _setup_searches() -> None:
    """Generate a searches.yaml from user input."""
    console.print(Panel("[bold]Step 3: Job Search Config[/bold]\nDefine what you're looking for."))

    location = Prompt.ask(
        "Target location (e.g. 'Vancouver, BC', 'Surrey, BC', 'Burnaby, BC')",
        default="Vancouver, BC",
    )
    distance_str = Prompt.ask("Search radius in km (e.g. 25 for Metro Vancouver)", default="25")
    try:
        distance = int(distance_str)
    except ValueError:
        distance = 25

    roles_raw = Prompt.ask(
        "Target job titles (comma-separated, e.g. 'Warehouse Associate, Order Picker, Warehouse Worker')",
        default="Warehouse Associate, Order Picker, Warehouse Worker, Order Selector, Warehouse Operator, Warehouse Assistant, Warehouse Person, Warehouse Stocker, Packager",
    )
    roles = [r.strip() for r in roles_raw.split(",") if r.strip()]

    if not roles:
        console.print("[yellow]No roles provided. Using the default warehouse set.[/yellow]")
        roles = [
            "Warehouse Associate",
            "Order Picker",
            "Warehouse Worker",
            "Order Selector",
            "Warehouse Operator",
            "Warehouse Assistant",
            "Warehouse Person",
            "Warehouse Stocker",
            "Packager",
        ]

    # Greater Vancouver municipalities — used for both the locations list
    # and the accept_patterns. Warehouse jobs are on-site, so we hit every
    # Metro Vancouver city directly rather than relying on a single radius
    # search (boards inconsistently honour distance for hourly roles).
    gvrd_locations = [
        "Vancouver, BC",
        "Burnaby, BC",
        "Richmond, BC",
        "Surrey, BC",
        "Delta, BC",
        "Langley, BC",
        "Coquitlam, BC",
        "Port Coquitlam, BC",
        "New Westminster, BC",
        "North Vancouver, BC",
        "Port Moody, BC",
        "Pitt Meadows, BC",
        "Maple Ridge, BC",
    ]
    # Make sure the user's chosen location appears first (and only once).
    locations_list = [location] + [c for c in gvrd_locations if c != location]

    accept_patterns = [
        "Vancouver", "Burnaby", "Richmond", "Surrey", "Delta", "Tsawwassen",
        "Ladner", "Langley", "Cloverdale", "Coquitlam", "Port Coquitlam",
        "Port Moody", "New Westminster", "North Vancouver", "West Vancouver",
        "Pitt Meadows", "Maple Ridge", "White Rock", "Greater Vancouver",
        "Metro Vancouver", "Lower Mainland", "GVRD", "BC", "British Columbia",
    ]
    reject_patterns = [
        "Toronto", "Ontario", "Alberta", "Calgary", "Edmonton", "Quebec",
        "Montreal", "Halifax", "United States", "USA", "Remote",
        "Victoria, BC", "Nanaimo", "Kelowna", "Kamloops", "Abbotsford",
        "Chilliwack", "Mission, BC", "Squamish", "Whistler",
    ]
    exclude_titles = [
        "manager", "supervisor", "lead ", "team lead", "director", "VP",
        "vice president", "chief", "coordinator", "planner", "analyst",
        "specialist", "engineer", "developer", "buyer", "merchandiser",
        "category", "sales", "account", "marketing", "controller",
        "accountant", "Class 1 driver", "Class 3 driver", "truck driver",
        "delivery driver", "owner operator", "millwright", "electrician",
        "mechanic", "technician", "machinist", "welder", "intern",
        "internship", "co-op", "student",
    ]

    # Build YAML content
    lines: list[str] = [
        "# ApplyPilot search configuration — Warehousing & Logistics, Greater Vancouver",
        "# Edit this file to refine your job search queries.",
        "",
        "country: \"Canada\"",
        "",
        "defaults:",
        f'  location: "{location}"',
        f"  distance: {distance}",
        "  hours_old: 168",
        "  results_per_site: 100",
        "",
        "boards:",
        "  - indeed",
        "  - linkedin",
        "  - zip_recruiter",
        "  - google",
        "",
        "locations:",
    ]
    for loc in locations_list:
        lines.append(f'  - location: "{loc}"')
        lines.append("    remote: false")

    lines.append("")
    lines.append("queries:")
    for i, role in enumerate(roles):
        lines.append(f'  - query: "{role}"')
        lines.append(f"    tier: {min(i + 1, 3)}")

    lines.append("")
    lines.append("location:")
    lines.append("  accept_patterns:")
    for p in accept_patterns:
        lines.append(f'    - "{p}"')
    lines.append("  reject_patterns:")
    for p in reject_patterns:
        lines.append(f'    - "{p}"')

    lines.append("")
    lines.append("exclude_titles:")
    for t in exclude_titles:
        lines.append(f'  - "{t}"')

    SEARCH_CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[green]Search config saved to {SEARCH_CONFIG_PATH}[/green]")


# ---------------------------------------------------------------------------
# AI Features
# ---------------------------------------------------------------------------

def _setup_ai_features() -> None:
    """Ask about AI scoring/tailoring — optional LLM configuration."""
    console.print(Panel(
        "[bold]Step 4: AI Features (optional)[/bold]\n"
        "An LLM powers job scoring, resume tailoring, and cover letters.\n"
        "Without this, you can still discover and enrich jobs."
    ))

    if not Confirm.ask("Enable AI scoring and resume tailoring?", default=True):
        console.print("[dim]Discovery-only mode. You can configure AI later with [bold]applypilot init[/bold].[/dim]")
        return

    console.print("Supported providers: [bold]Gemini[/bold] (recommended, free tier), OpenAI, local (Ollama/llama.cpp)")
    provider = Prompt.ask(
        "Provider",
        choices=["gemini", "openai", "local"],
        default="gemini",
    )

    env_lines = ["# ApplyPilot configuration", ""]

    if provider == "gemini":
        api_key = Prompt.ask("Gemini API key (from aistudio.google.com)")
        model = Prompt.ask("Model", default="gemini-2.0-flash")
        env_lines.append(f"GEMINI_API_KEY={api_key}")
        env_lines.append(f"LLM_MODEL={model}")
    elif provider == "openai":
        api_key = Prompt.ask("OpenAI API key")
        model = Prompt.ask("Model", default="gpt-4o-mini")
        env_lines.append(f"OPENAI_API_KEY={api_key}")
        env_lines.append(f"LLM_MODEL={model}")
    elif provider == "local":
        url = Prompt.ask("Local LLM endpoint URL", default="http://localhost:8080/v1")
        model = Prompt.ask("Model name", default="local-model")
        env_lines.append(f"LLM_URL={url}")
        env_lines.append(f"LLM_MODEL={model}")

    env_lines.append("")
    ENV_PATH.write_text("\n".join(env_lines), encoding="utf-8")
    console.print(f"[green]AI configuration saved to {ENV_PATH}[/green]")


# ---------------------------------------------------------------------------
# Auto-Apply
# ---------------------------------------------------------------------------

def _setup_auto_apply() -> None:
    """Configure autonomous job application (requires Claude Code CLI)."""
    console.print(Panel(
        "[bold]Step 5: Auto-Apply (optional)[/bold]\n"
        "ApplyPilot can autonomously fill and submit job applications\n"
        "using Claude Code as the browser agent."
    ))

    if not Confirm.ask("Enable autonomous job applications?", default=True):
        console.print("[dim]You can apply manually using the tailored resumes ApplyPilot generates.[/dim]")
        return

    # Check for Claude Code CLI
    if shutil.which("claude"):
        console.print("[green]Claude Code CLI detected.[/green]")
    else:
        console.print(
            "[yellow]Claude Code CLI not found on PATH.[/yellow]\n"
            "Install it from: [bold]https://claude.ai/code[/bold]\n"
            "Auto-apply won't work until Claude Code is installed."
        )

    # Optional: CapSolver for CAPTCHAs
    console.print("\n[dim]Some job sites use CAPTCHAs. CapSolver can handle them automatically.[/dim]")
    if Confirm.ask("Configure CapSolver API key? (optional)", default=False):
        capsolver_key = Prompt.ask("CapSolver API key")
        # Append to existing .env or create
        if ENV_PATH.exists():
            existing = ENV_PATH.read_text(encoding="utf-8")
            if "CAPSOLVER_API_KEY" not in existing:
                ENV_PATH.write_text(
                    existing.rstrip() + f"\nCAPSOLVER_API_KEY={capsolver_key}\n",
                    encoding="utf-8",
                )
        else:
            ENV_PATH.write_text(f"# ApplyPilot configuration\nCAPSOLVER_API_KEY={capsolver_key}\n", encoding="utf-8")
        console.print("[green]CapSolver key saved.[/green]")
    else:
        console.print("[dim]Skipped. Add CAPSOLVER_API_KEY to .env later if needed.[/dim]")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_wizard() -> None:
    """Run the full interactive setup wizard."""
    console.print()
    console.print(
        Panel.fit(
            "[bold green]ApplyPilot Setup Wizard[/bold green]\n\n"
            "This will create your configuration at:\n"
            f"  [cyan]{APP_DIR}[/cyan]\n\n"
            "You can re-run this anytime with [bold]applypilot init[/bold].",
            border_style="green",
        )
    )

    ensure_dirs()
    console.print(f"[dim]Created {APP_DIR}[/dim]\n")

    # Step 1: Resume
    _setup_resume()
    console.print()

    # Step 2: Profile
    _setup_profile()
    console.print()

    # Step 3: Search config
    _setup_searches()
    console.print()

    # Step 4: AI features (optional LLM)
    _setup_ai_features()
    console.print()

    # Step 5: Auto-apply (Claude Code detection)
    _setup_auto_apply()
    console.print()

    # Done — show tier status
    from applypilot.config import get_tier, TIER_LABELS, TIER_COMMANDS

    tier = get_tier()

    tier_lines: list[str] = []
    for t in range(1, 4):
        label = TIER_LABELS[t]
        cmds = ", ".join(f"[bold]{c}[/bold]" for c in TIER_COMMANDS[t])
        if t <= tier:
            tier_lines.append(f"  [green]✓ Tier {t} — {label}[/green]  ({cmds})")
        elif t == tier + 1:
            tier_lines.append(f"  [yellow]→ Tier {t} — {label}[/yellow]  ({cmds})")
        else:
            tier_lines.append(f"  [dim]✗ Tier {t} — {label}  ({cmds})[/dim]")

    unlock_hint = ""
    if tier == 1:
        unlock_hint = "\n[dim]To unlock Tier 2: configure an LLM API key (re-run [bold]applypilot init[/bold]).[/dim]"
    elif tier == 2:
        unlock_hint = "\n[dim]To unlock Tier 3: install Claude Code CLI + Chrome.[/dim]"

    console.print(
        Panel.fit(
            "[bold green]Setup complete![/bold green]\n\n"
            f"[bold]Your tier: Tier {tier} — {TIER_LABELS[tier]}[/bold]\n\n"
            + "\n".join(tier_lines)
            + unlock_hint,
            border_style="green",
        )
    )
