from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_daily_unguard_requires_explicit_force_and_scheduled_wrapper_is_safe():
    script = (ROOT / "fleet-daily-unguard.ps1").read_text(encoding="utf-8")

    assert "[switch]$Force" in script
    assert "if (-not $Force)" in script
    assert "explicit -Force is required" in script
    wrapper_invocation = next(
        line for line in script.splitlines() if "& '$scriptPath' -Reason" in line
    )
    assert "-Force" not in wrapper_invocation
    assert "-Dsn" not in wrapper_invocation
    assert "& '$python' -m applypilot.fleet.launch_guard" in script
