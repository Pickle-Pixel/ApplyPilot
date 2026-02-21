"""ApplyPilot first-time setup wizard.

LLM-first interactive flow that creates ~/.applypilot/ with:
  - resume.txt (and optionally resume.pdf)
  - .env (LLM API key) - REQUIRED
  - profile.json (extracted from resume + user input)
  - searches.yaml
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

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
# Step 1: Resume
# ---------------------------------------------------------------------------

def _setup_resume() -> str | None:
    """Prompt for resume file, copy into APP_DIR, return extracted text."""
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

        resume_text = ""

        if suffix == ".txt":
            shutil.copy2(src, RESUME_PATH)
            resume_text = src.read_text(encoding="utf-8")
            console.print(f"[green]Copied to {RESUME_PATH}[/green]")

        elif suffix == ".pdf":
            shutil.copy2(src, RESUME_PDF_PATH)
            console.print(f"[green]Copied to {RESUME_PDF_PATH}[/green]")

            # Extract text from PDF
            if HAS_PYPDF:
                try:
                    reader = PdfReader(src)
                    pages_text = [page.extract_text() or "" for page in reader.pages]
                    resume_text = "\n\n".join(pages_text).strip()
                except Exception as e:
                    console.print(f"[yellow]PDF text extraction failed: {e}[/yellow]")

            if resume_text and len(resume_text) > 100:
                RESUME_PATH.write_text(resume_text, encoding="utf-8")
                console.print(f"[green]Auto-extracted text to {RESUME_PATH}[/green]")
            else:
                if not HAS_PYPDF:
                    console.print("[yellow]pypdf not installed - cannot auto-extract text.[/yellow]")
                else:
                    console.print("[yellow]Could not extract sufficient text from PDF.[/yellow]")

                txt_path_str = Prompt.ask("Plain-text version of your resume (.txt)", default="")
                if txt_path_str.strip():
                    txt_src = Path(txt_path_str.strip().strip('"').strip("'")).expanduser().resolve()
                    if txt_src.exists():
                        shutil.copy2(txt_src, RESUME_PATH)
                        resume_text = txt_src.read_text(encoding="utf-8")
                        console.print(f"[green]Copied to {RESUME_PATH}[/green]")
                    else:
                        console.print("[yellow]File not found.[/yellow]")

        return resume_text if resume_text else None


# ---------------------------------------------------------------------------
# Step 2: LLM Setup (Required)
# ---------------------------------------------------------------------------

def _setup_llm() -> bool:
    """Configure LLM provider - REQUIRED for ApplyPilot."""
    console.print(Panel(
        "[bold]Step 2: LLM Setup (Required)[/bold]\n"
        "ApplyPilot needs an LLM for scoring, tailoring, and profile extraction.\n"
        "Gemini offers a free tier (15 requests/min)."
    ))

    console.print("Supported providers: [bold]Gemini[/bold] (recommended, free), OpenAI, local (Ollama/llama.cpp)")
    provider = Prompt.ask(
        "Provider",
        choices=["gemini", "openai", "local"],
        default="gemini",
    )

    env_lines = ["# ApplyPilot configuration", ""]
    api_key = ""
    model = ""
    endpoint = ""

    while True:
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
            endpoint = Prompt.ask("Local LLM endpoint URL", default="http://localhost:8080/v1")
            model = Prompt.ask("Model name", default="local-model")
            api_key = Prompt.ask("API key (leave blank if none)", default="")
            env_lines.append(f"LLM_URL={endpoint}")
            env_lines.append(f"LLM_MODEL={model}")
            if api_key:
                env_lines.append(f"LLM_API_KEY={api_key}")

        # Validate the API key
        console.print("[dim]Validating API key...[/dim]")
        from applypilot.llm import validate_api_key
        is_valid, error = validate_api_key(provider, api_key, model, endpoint)

        if is_valid:
            console.print("[green]API key validated successfully![/green]")
            break
        else:
            console.print(f"[red]Validation failed: {error}[/red]")
            if not Confirm.ask("Try again?", default=True):
                console.print("[red]LLM configuration is required. Exiting setup.[/red]")
                return False
            env_lines = ["# ApplyPilot configuration", ""]  # Reset

    env_lines.append("")
    ENV_PATH.write_text("\n".join(env_lines), encoding="utf-8")
    console.print(f"[green]LLM configuration saved to {ENV_PATH}[/green]")

    # Set env vars for this session so extraction works
    if provider == "gemini":
        os.environ["GEMINI_API_KEY"] = api_key
    elif provider == "openai":
        os.environ["OPENAI_API_KEY"] = api_key
    elif provider == "local":
        os.environ["LLM_URL"] = endpoint
        if api_key:
            os.environ["LLM_API_KEY"] = api_key
    os.environ["LLM_MODEL"] = model

    return True


# ---------------------------------------------------------------------------
# Step 3: Profile Extraction + Review
# ---------------------------------------------------------------------------

def _extract_and_review_profile(resume_text: str) -> dict:
    """Extract profile from resume using LLM, then let user review and fill gaps."""
    console.print(Panel(
        "[bold]Step 3: Profile Setup[/bold]\n"
        "Extracting your profile from your resume..."
    ))

    # Extract from resume
    from applypilot.wizard.resume_parser import extract_resume_data, extracted_to_profile

    console.print("[dim]Analyzing resume with AI...[/dim]")
    extracted, metadata = extract_resume_data(resume_text)

    profile: dict = {}

    if extracted and metadata["success"]:
        console.print("[green]Profile data extracted![/green]")
        if metadata["warnings"]:
            for warning in metadata["warnings"]:
                console.print(f"[yellow]  - {warning}[/yellow]")

        # Convert to profile format
        profile = extracted_to_profile(extracted)

        # Show extracted data
        console.print("\n[bold cyan]Extracted from your resume:[/bold cyan]")
        _display_extracted_profile(profile)

        # Ask for confirmation
        if not Confirm.ask("\nIs this information correct?", default=True):
            console.print("[dim]You can edit specific fields below.[/dim]")
            profile = _edit_profile(profile)
    else:
        console.print("[yellow]Could not extract profile from resume.[/yellow]")
        if metadata["errors"]:
            for error in metadata["errors"]:
                console.print(f"[red]  - {error}[/red]")
        console.print("[dim]You'll need to enter profile information manually.[/dim]")
        profile = _manual_profile_entry()

    # Fill in fields that CAN'T be extracted from resume
    console.print("\n[bold cyan]Additional Information[/bold cyan]")
    console.print("[dim]These fields aren't typically in resumes.[/dim]\n")

    profile = _fill_non_resume_fields(profile)

    # Save profile
    PROFILE_PATH.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[green]Profile saved to {PROFILE_PATH}[/green]")

    return profile


def _display_extracted_profile(profile: dict) -> None:
    """Display extracted profile data in a nice table."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="dim")
    table.add_column("Value")

    personal = profile.get("personal", {})
    experience = profile.get("experience", {})
    skills = profile.get("skills_boundary", {})
    facts = profile.get("resume_facts", {})

    # Personal
    if personal.get("full_name"):
        table.add_row("Name", personal["full_name"])
    if personal.get("email"):
        table.add_row("Email", personal["email"])
    if personal.get("phone"):
        table.add_row("Phone", personal["phone"])
    if personal.get("city") or personal.get("country"):
        location = ", ".join(filter(None, [personal.get("city"), personal.get("province_state"), personal.get("country")]))
        table.add_row("Location", location)
    if personal.get("linkedin_url"):
        table.add_row("LinkedIn", personal["linkedin_url"])
    if personal.get("github_url"):
        table.add_row("GitHub", personal["github_url"])

    # Experience
    if experience.get("current_title"):
        table.add_row("Current Title", experience["current_title"])
    if experience.get("years_of_experience_total"):
        table.add_row("Experience", f"{experience['years_of_experience_total']} years")
    if experience.get("education_level"):
        table.add_row("Education", experience["education_level"])

    # Skills
    if skills.get("programming_languages"):
        table.add_row("Languages", ", ".join(skills["programming_languages"]))
    if skills.get("frameworks"):
        table.add_row("Frameworks", ", ".join(skills["frameworks"]))
    if skills.get("tools"):
        table.add_row("Tools", ", ".join(skills["tools"]))

    # Resume facts
    if facts.get("preserved_companies"):
        table.add_row("Companies", ", ".join(facts["preserved_companies"]))

    console.print(table)


def _edit_profile(profile: dict) -> dict:
    """Let user edit specific extracted fields."""
    personal = profile.get("personal", {})
    experience = profile.get("experience", {})
    skills = profile.get("skills_boundary", {})

    console.print("\n[dim]Press Enter to keep current value, or type new value.[/dim]\n")

    # Editable fields
    personal["full_name"] = Prompt.ask("Full name", default=personal.get("full_name", ""))
    personal["email"] = Prompt.ask("Email", default=personal.get("email", ""))
    personal["phone"] = Prompt.ask("Phone", default=personal.get("phone", ""))
    experience["current_title"] = Prompt.ask("Current title", default=experience.get("current_title", ""))
    experience["years_of_experience_total"] = Prompt.ask("Years of experience", default=experience.get("years_of_experience_total", ""))

    langs = Prompt.ask("Programming languages (comma-separated)", default=", ".join(skills.get("programming_languages", [])))
    skills["programming_languages"] = [s.strip() for s in langs.split(",") if s.strip()]

    frameworks = Prompt.ask("Frameworks (comma-separated)", default=", ".join(skills.get("frameworks", [])))
    skills["frameworks"] = [s.strip() for s in frameworks.split(",") if s.strip()]

    tools = Prompt.ask("Tools (comma-separated)", default=", ".join(skills.get("tools", [])))
    skills["tools"] = [s.strip() for s in tools.split(",") if s.strip()]

    profile["personal"] = personal
    profile["experience"] = experience
    profile["skills_boundary"] = skills

    return profile


def _manual_profile_entry() -> dict:
    """Fallback: manual profile entry when extraction fails."""
    profile: dict = {}

    console.print("\n[bold cyan]Personal Information[/bold cyan]")
    profile["personal"] = {
        "full_name": Prompt.ask("Full name"),
        "preferred_name": "",
        "email": Prompt.ask("Email address"),
        "phone": Prompt.ask("Phone number", default=""),
        "city": "",
        "province_state": "",
        "country": "",
        "postal_code": "",
        "address": "",
        "linkedin_url": Prompt.ask("LinkedIn URL", default=""),
        "github_url": Prompt.ask("GitHub URL", default=""),
        "portfolio_url": "",
        "website_url": "",
        "password": "",
    }

    console.print("\n[bold cyan]Experience[/bold cyan]")
    current_title = Prompt.ask("Current/most recent job title", default="")
    profile["experience"] = {
        "years_of_experience_total": Prompt.ask("Years of professional experience", default=""),
        "education_level": Prompt.ask("Highest education (Bachelor's, Master's, PhD, etc.)", default=""),
        "current_title": current_title,
        "target_role": current_title,
    }

    console.print("\n[bold cyan]Skills[/bold cyan] (comma-separated)")
    langs = Prompt.ask("Programming languages", default="")
    frameworks = Prompt.ask("Frameworks & libraries", default="")
    tools = Prompt.ask("Tools & platforms", default="")
    profile["skills_boundary"] = {
        "programming_languages": [s.strip() for s in langs.split(",") if s.strip()],
        "frameworks": [s.strip() for s in frameworks.split(",") if s.strip()],
        "tools": [s.strip() for s in tools.split(",") if s.strip()],
    }

    profile["resume_facts"] = {
        "preserved_companies": [],
        "preserved_projects": [],
        "preserved_school": "",
        "real_metrics": [],
    }

    return profile


def _fill_non_resume_fields(profile: dict) -> dict:
    """Ask for fields that can't be extracted from a resume."""
    personal = profile.get("personal", {})
    experience = profile.get("experience", {})

    # Location (often not on resumes or incomplete)
    if not personal.get("city"):
        personal["city"] = Prompt.ask("City", default="")
    if not personal.get("province_state"):
        personal["province_state"] = Prompt.ask("Province/State", default="")
    if not personal.get("country"):
        personal["country"] = Prompt.ask("Country", default="")

    # Target role
    current_title = experience.get("current_title", "")
    experience["target_role"] = Prompt.ask(
        "Target role (what you're applying for)",
        default=current_title
    )

    # Work authorization
    console.print("\n[bold cyan]Work Authorization[/bold cyan]")
    profile["work_authorization"] = {
        "legally_authorized_to_work": Confirm.ask("Are you legally authorized to work in your target country?"),
        "require_sponsorship": Confirm.ask("Will you need sponsorship now or in the future?"),
        "work_permit_type": Prompt.ask("Work permit type (Citizen, PR, Work Permit, etc.)", default=""),
    }

    # Compensation
    console.print("\n[bold cyan]Compensation[/bold cyan]")
    salary = Prompt.ask("Expected annual salary (number)", default="")
    salary_currency = Prompt.ask("Currency", default="USD")
    salary_range = Prompt.ask("Acceptable range (e.g. 80000-120000)", default="")
    range_parts = salary_range.split("-") if "-" in salary_range else [salary, salary]
    profile["compensation"] = {
        "salary_expectation": salary,
        "salary_currency": salary_currency,
        "salary_range_min": range_parts[0].strip() if range_parts else "",
        "salary_range_max": range_parts[1].strip() if len(range_parts) > 1 else range_parts[0].strip() if range_parts else "",
    }

    # Availability
    profile["availability"] = {
        "earliest_start_date": Prompt.ask("Earliest start date", default="Immediately"),
    }

    # Password for job sites
    personal["password"] = Prompt.ask(
        "Job site password (for auto-apply login walls)",
        password=True,
        default=""
    )

    # EEO defaults
    profile["eeo_voluntary"] = {
        "gender": "Decline to self-identify",
        "race_ethnicity": "Decline to self-identify",
        "veteran_status": "Decline to self-identify",
        "disability_status": "Decline to self-identify",
    }

    profile["personal"] = personal
    profile["experience"] = experience

    return profile


# ---------------------------------------------------------------------------
# Step 4: Search Config
# ---------------------------------------------------------------------------

def _setup_searches(profile: dict) -> None:
    """Generate a searches.yaml from user input."""
    console.print(Panel("[bold]Step 4: Job Search Config[/bold]\nDefine what you're looking for."))

    target_role = profile.get("experience", {}).get("target_role", "Software Engineer")

    location = Prompt.ask("Target location (e.g. 'Remote', 'Canada', 'New York, NY')", default="Remote")
    distance_str = Prompt.ask("Search radius in miles (0 for remote-only)", default="0")
    try:
        distance = int(distance_str)
    except ValueError:
        distance = 0

    roles_raw = Prompt.ask(
        "Target job titles (comma-separated)",
        default=target_role
    )
    roles = [r.strip() for r in roles_raw.split(",") if r.strip()]

    if not roles:
        roles = ["Software Engineer"]

    # Build YAML content
    lines = [
        "# ApplyPilot search configuration",
        "# Edit this file to refine your job search queries.",
        "",
        "defaults:",
        f'  location: "{location}"',
        f"  distance: {distance}",
        "  hours_old: 72",
        "  results_per_site: 50",
        "",
        "locations:",
        f'  - location: "{location}"',
        f"    remote: {str(distance == 0).lower()}",
        "",
        "queries:",
    ]
    for i, role in enumerate(roles):
        lines.append(f'  - query: "{role}"')
        lines.append(f"    tier: {min(i + 1, 3)}")

    SEARCH_CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[green]Search config saved to {SEARCH_CONFIG_PATH}[/green]")


# ---------------------------------------------------------------------------
# Step 5: Auto-Apply (Optional)
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
        if ENV_PATH.exists():
            existing = ENV_PATH.read_text(encoding="utf-8")
            if "CAPSOLVER_API_KEY" not in existing:
                ENV_PATH.write_text(
                    existing.rstrip() + f"\nCAPSOLVER_API_KEY={capsolver_key}\n",
                    encoding="utf-8",
                )
        else:
            ENV_PATH.write_text(f"CAPSOLVER_API_KEY={capsolver_key}\n", encoding="utf-8")
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
    resume_text = _setup_resume()
    console.print()

    if not resume_text:
        console.print("[yellow]Warning: No resume text available. Profile extraction will be limited.[/yellow]")
        resume_text = ""

    # Step 2: LLM setup (REQUIRED)
    if not _setup_llm():
        raise typer.Exit(1)
    console.print()

    # Step 3: Extract profile from resume + fill gaps
    profile = _extract_and_review_profile(resume_text)
    console.print()

    # Step 4: Search config
    _setup_searches(profile)
    console.print()

    # Step 5: Auto-apply (optional)
    _setup_auto_apply()
    console.print()

    # Done - show tier status
    from applypilot.config import get_tier, TIER_LABELS, TIER_COMMANDS

    tier = get_tier()

    tier_lines: list[str] = []
    for t in range(1, 4):
        label = TIER_LABELS[t]
        cmds = ", ".join(f"[bold]{c}[/bold]" for c in TIER_COMMANDS[t])
        if t <= tier:
            tier_lines.append(f"  [green]* Tier {t} - {label}[/green]  ({cmds})")
        elif t == tier + 1:
            tier_lines.append(f"  [yellow]-> Tier {t} - {label}[/yellow]  ({cmds})")
        else:
            tier_lines.append(f"  [dim]x Tier {t} - {label}  ({cmds})[/dim]")

    unlock_hint = ""
    if tier == 1:
        unlock_hint = "\n[dim]To unlock Tier 2: configure an LLM API key (re-run [bold]applypilot init[/bold]).[/dim]"
    elif tier == 2:
        unlock_hint = "\n[dim]To unlock Tier 3: install Claude Code CLI + Chrome.[/dim]"

    console.print(
        Panel.fit(
            "[bold green]Setup complete![/bold green]\n\n"
            f"[bold]Your tier: Tier {tier} - {TIER_LABELS[tier]}[/bold]\n\n"
            + "\n".join(tier_lines)
            + unlock_hint,
            border_style="green",
        )
    )
