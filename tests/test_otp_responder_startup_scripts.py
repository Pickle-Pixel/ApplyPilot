from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_otp_responder_launcher_keeps_dsn_out_of_argv_and_logs() -> None:
    script = (REPO / "run-otp-responder.ps1").read_text(encoding="utf-8")

    assert "applypilot-fleet-otp-home.exe" in script
    assert "$env:FLEET_PG_DSN = $Dsn" in script
    assert '-ArgumentList @("--machine-owner", $MachineOwner)' in script
    assert "--dsn" not in script.lower()
    assert "-RedirectStandardOutput $OutLog" in script
    assert "-RedirectStandardError $ErrLog" in script
    assert "-WindowStyle Hidden" in script
    assert "Stop-StaleOtpResponderProcesses" in script


def test_register_otp_responder_startup_falls_back_to_user_startup() -> None:
    script = (REPO / "register-otp-responder-startup.ps1").read_text(encoding="utf-8")

    assert "Register-ScheduledTask" in script
    assert "Register-ScheduledTask -TaskName $TaskName" in script
    assert "Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop" in script
    assert "-Force -ErrorAction Stop" in script
    assert "Install-StartupShortcut" in script
    assert "[Environment]::GetFolderPath('Startup')" in script
    assert "ApplyPilotFleet-OtpResponder.lnk" in script
    assert "run-otp-responder.ps1" in script
    assert "-Supervise" in script
    assert "Task Scheduler registration denied; installing Startup shortcut fallback" in script
