"""ApplyPilot configuration: paths, platform detection, user data."""

import os
import platform
import shutil
from pathlib import Path


def _path_is_within(path: Path, parent: Path) -> bool:
    """Return True when path is equal to or nested under parent."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _running_in_wsl() -> bool:
    """Detect WSL so we can prefer the Windows profile when appropriate."""
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    return "microsoft" in platform.release().lower()


def _default_app_dir() -> Path:
    """Choose the best default data directory for the current environment."""
    override = os.environ.get("APPLYPILOT_DIR")
    if override:
        return Path(override).expanduser()

    if _running_in_wsl():
        userprofile = os.environ.get("USERPROFILE")
        if userprofile:
            windows_home = Path(userprofile)
            windows_app_dir = windows_home / ".applypilot"
            try:
                cwd = Path.cwd()
            except OSError:
                cwd = None

            # In WSL, prefer the Windows profile when the project is being run
            # from the Windows side or when the user already has data there.
            if windows_app_dir.exists() or (cwd and _path_is_within(cwd, windows_home)):
                return windows_app_dir

    return Path.home() / ".applypilot"


# User data directory — all user-specific files live here
APP_DIR = _default_app_dir()

# Core paths
DB_PATH = APP_DIR / "applypilot.db"
PROFILE_PATH = APP_DIR / "profile.json"
RESUME_PATH = APP_DIR / "resume.txt"
RESUME_PDF_PATH = APP_DIR / "resume.pdf"
SEARCH_CONFIG_PATH = APP_DIR / "searches.yaml"
ENV_PATH = APP_DIR / ".env"

# Generated output
TAILORED_DIR = APP_DIR / "tailored_resumes"
COVER_LETTER_DIR = APP_DIR / "cover_letters"
APPLICATION_PACK_DIR = APP_DIR / "application_packs"
LOG_DIR = APP_DIR / "logs"

# Chrome worker isolation
CHROME_WORKER_DIR = APP_DIR / "chrome-workers"
APPLY_WORKER_DIR = APP_DIR / "apply-workers"

# Package-shipped config (YAML registries)
PACKAGE_DIR = Path(__file__).parent
CONFIG_DIR = PACKAGE_DIR / "config"


def get_chrome_path() -> str:
    """Auto-detect Chrome/Chromium executable path, cross-platform.

    Override with CHROME_PATH environment variable.
    """
    env_path = os.environ.get("CHROME_PATH")
    if env_path:
        env_candidate = Path(env_path).expanduser()
        if env_candidate.is_file():
            return str(env_candidate)
        if env_candidate.exists():
            raise FileNotFoundError(
                f"CHROME_PATH points to '{env_candidate}', which is not a Chrome executable file."
            )

    system = platform.system()

    if system == "Windows":
        candidates = [
            Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
            / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
        ]
    elif system == "Darwin":
        candidates = [
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        ]
    else:  # Linux
        candidates = []
        for name in ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium"):
            found = shutil.which(name)
            if found:
                candidates.append(Path(found))

    for c in candidates:
        if c and c.exists():
            return str(c)

    # Fall back to PATH search
    for name in ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            return found

    raise FileNotFoundError("Chrome/Chromium not found. Install Chrome or set CHROME_PATH environment variable.")


def get_chrome_user_data() -> Path:
    """Default Chrome user data directory, cross-platform."""
    system = platform.system()
    if system == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"
    elif system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
    else:
        return Path.home() / ".config" / "google-chrome"


def ensure_dirs():
    """Create all required directories."""
    for d in [
        APP_DIR,
        TAILORED_DIR,
        COVER_LETTER_DIR,
        APPLICATION_PACK_DIR,
        LOG_DIR,
        CHROME_WORKER_DIR,
        APPLY_WORKER_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def load_profile() -> dict:
    """Load user profile from ~/.applypilot/profile.json."""
    import json

    if not PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"Profile not found at {PROFILE_PATH}. Run `applypilot init` first."
        )
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8-sig"))


def load_search_config() -> dict:
    """Load search configuration from ~/.applypilot/searches.yaml."""
    import yaml

    if not SEARCH_CONFIG_PATH.exists():
        # Fall back to package-shipped example
        example = CONFIG_DIR / "searches.example.yaml"
        if example.exists():
            return yaml.safe_load(example.read_text(encoding="utf-8")) or {}
        return {}
    return yaml.safe_load(SEARCH_CONFIG_PATH.read_text(encoding="utf-8")) or {}


def load_sites_config() -> dict:
    """Load sites.yaml configuration (sites list, manual_ats, blocked, etc.)."""
    import yaml

    path = CONFIG_DIR / "sites.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def is_manual_ats(url: str | None) -> bool:
    """Check if a URL routes through an ATS that requires manual application."""
    if not url:
        return False
    sites_cfg = load_sites_config()
    domains = sites_cfg.get("manual_ats", [])
    url_lower = url.lower()
    return any(domain in url_lower for domain in domains)


def load_blocked_sites() -> tuple[set[str], list[str]]:
    """Load blocked sites and URL patterns from sites.yaml.

    Returns:
        (blocked_site_names, blocked_url_patterns)
    """
    cfg = load_sites_config()
    blocked = cfg.get("blocked", {})
    sites = set(blocked.get("sites", []))
    patterns = blocked.get("url_patterns", [])
    return sites, patterns


def load_blocked_sso() -> list[str]:
    """Load blocked SSO domains from sites.yaml."""
    cfg = load_sites_config()
    return cfg.get("blocked_sso", [])


def load_avoid_companies() -> list[dict]:
    """Load the user-curated AVOID list from ~/.applypilot/avoid_companies.yaml.

    Companies returned here are treated as hard exclusions — skipped during
    curated list ingestion and filtered out of application pack exports.

    Returns a list of dicts: {"name": str, "aliases": list[str], "reason": str}.
    Missing file -> [].
    """
    import yaml
    path = APP_DIR / "avoid_companies.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("avoid", []) or []
    out = []
    for entry in entries:
        if isinstance(entry, str):
            out.append({"name": entry, "aliases": [], "reason": ""})
        elif isinstance(entry, dict) and entry.get("name"):
            out.append({
                "name": str(entry["name"]),
                "aliases": list(entry.get("aliases") or []),
                "reason": str(entry.get("reason") or ""),
            })
    return out


def is_avoid_company(candidate: str, avoid: list[dict] | None = None) -> bool:
    """Return True if `candidate` (a company or site name) matches the AVOID list.

    Matching rules:
      - Case-insensitive.
      - A needle >= 4 chars matches as a case-insensitive substring
        (e.g. "ibex" inside "IBEX Global Services").
      - A needle <= 3 chars (e.g. short acronyms like "TP") ONLY matches as
        a whole word bounded by non-word characters or string ends. This
        prevents "TP" from matching "HostPapa", "TalentPop", etc. via stray
        substrings.
    """
    if not candidate:
        return False
    import re as _re
    entries = avoid if avoid is not None else load_avoid_companies()
    cand = str(candidate).strip()
    cand_l = cand.lower()
    for entry in entries:
        needles = [entry.get("name", "")] + list(entry.get("aliases", []))
        for needle in needles:
            nd = str(needle).strip()
            if not nd:
                continue
            if len(nd) >= 4:
                if nd.lower() in cand_l:
                    return True
            else:
                if _re.search(rf"(?:^|\W){_re.escape(nd)}(?:\W|$)",
                              cand, flags=_re.IGNORECASE):
                    return True
    return False


def load_base_urls() -> dict[str, str | None]:
    """Load site base URLs for URL resolution from sites.yaml."""
    cfg = load_sites_config()
    return cfg.get("base_urls", {})


# ---------------------------------------------------------------------------
# Default values — referenced across modules instead of magic numbers
# ---------------------------------------------------------------------------

DEFAULTS = {
    "min_score": 7,
    "max_apply_attempts": 3,
    "max_tailor_attempts": 5,
    "poll_interval": 60,
    "apply_timeout": 300,
    "viewport": "1280x900",
}


def load_env():
    """Load environment variables from ~/.applypilot/.env if it exists."""
    from dotenv import dotenv_values, load_dotenv

    provider_keys = (
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "LLM_URL",
        "LLM_MODEL",
        "LLM_API_KEY",
    )

    if ENV_PATH.exists():
        env_values = {
            key: value
            for key, value in dotenv_values(ENV_PATH).items()
            if value is not None
        }

        # Treat ~/.applypilot/.env as the source of truth for LLM provider
        # selection so removing a key there actually removes it at runtime.
        for key in provider_keys:
            if key not in env_values:
                os.environ.pop(key, None)

        load_dotenv(ENV_PATH, override=True)
    else:
        # Also try CWD .env as fallback only when the ApplyPilot-local env
        # does not exist. Once ~/.applypilot/.env exists, it is the source of
        # truth for provider selection and should not be silently overridden.
        load_dotenv()


# ---------------------------------------------------------------------------
# Tier system — feature gating by installed dependencies
# ---------------------------------------------------------------------------

TIER_LABELS = {
    1: "Discovery",
    2: "AI Scoring & Tailoring",
    3: "Full Auto-Apply",
}

TIER_COMMANDS: dict[int, list[str]] = {
    1: ["init", "run discover", "run enrich", "status", "dashboard"],
    2: ["run score", "run tailor", "run cover", "run pdf", "run"],
    3: ["apply"],
}


def get_tier() -> int:
    """Detect the current tier based on available dependencies.

    Tier 1 (Discovery):            Python + pip
    Tier 2 (AI Scoring & Tailoring): + LLM API key
    Tier 3 (Full Auto-Apply):       + Claude Code CLI + Chrome
    """
    load_env()

    has_llm = any(os.environ.get(k) for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_URL"))
    if not has_llm:
        return 1

    has_claude = shutil.which("claude") is not None
    try:
        get_chrome_path()
        has_chrome = True
    except FileNotFoundError:
        has_chrome = False

    if has_claude and has_chrome:
        return 3

    return 2


def check_tier(required: int, feature: str) -> None:
    """Raise SystemExit with a clear message if the current tier is too low.

    Args:
        required: Minimum tier needed (1, 2, or 3).
        feature: Human-readable description of the feature being gated.
    """
    current = get_tier()
    if current >= required:
        return

    from rich.console import Console

    _console = Console(stderr=True)

    missing: list[str] = []
    if required >= 2 and not any(os.environ.get(k) for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_URL")):
        missing.append("LLM API key — run [bold]applypilot init[/bold] or set GEMINI_API_KEY")
    if required >= 3:
        if not shutil.which("claude"):
            missing.append("Claude Code CLI — install from [bold]https://claude.ai/code[/bold]")
        try:
            get_chrome_path()
        except FileNotFoundError:
            missing.append("Chrome/Chromium — install or set CHROME_PATH")

    _console.print(
        f"\n[red]'{feature}' requires {TIER_LABELS.get(required, f'Tier {required}')} (Tier {required}).[/red]\n"
        f"Current tier: {TIER_LABELS.get(current, f'Tier {current}')} (Tier {current})."
    )
    if missing:
        _console.print("\n[yellow]Missing:[/yellow]")
        for m in missing:
            _console.print(f"  - {m}")
    _console.print()
    raise SystemExit(1)
