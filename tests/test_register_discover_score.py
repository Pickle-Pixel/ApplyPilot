from pathlib import Path
import re


def _script() -> str:
    return Path("register-discover-score.ps1").read_text(encoding="utf-8")


def test_register_discover_score_uses_run_applypilot_file_wrappers():
    script = _script()

    assert "run-applypilot.ps1" in script
    assert "-File `\"$WrapperPath`\"" in script
    assert "-Command" not in script


def test_register_discover_score_commands_are_capped_and_fast():
    script = _script()

    assert "run discover enrich --workers 1 --discover-mode fast" in script
    assert "score-jobs --limit 400 --workers 1" in script
    assert "run audit" in script


def test_register_discover_score_trigger_times_are_owner_present_hours():
    script = _script()
    times = re.findall(r"New-ScheduledTaskTrigger\s+-Daily\s+-At\s+\"?([0-2]?\d:[0-5]\d)\"?", script)

    assert times
    for raw in times:
        hour, minute = map(int, raw.split(":"))
        assert 8 <= hour < 20, raw
        assert 0 <= minute <= 59


def test_register_discover_score_registers_exactly_two_tasks():
    script = _script()

    assert script.count("Register-CadenceTask -Name") == 2
    assert "${TaskPrefix}DiscoverEnrich" in script
    assert "${TaskPrefix}ScoreAudit" in script


def test_run_applypilot_backs_up_score_jobs_command():
    script = Path("run-applypilot.ps1").read_text(encoding="utf-8")

    assert script.count('"score-jobs"') == 1


def test_compute_ingest_wrapper_includes_unscored_push():
    script = Path("register-fleet-tasks.ps1").read_text(encoding="utf-8")

    assert "& '$computeIngestPs1' -Once -IncludeUnscored" in script


def test_home_task_registration_includes_otp_responder():
    script = Path("register-fleet-tasks.ps1").read_text(encoding="utf-8")

    assert "${TaskPrefix}OtpResponder" in script
    assert "run-otp-responder.ps1" in script
    wrapper_invocation = next(
        line for line in script.splitlines() if "& '$otpLauncher'" in line
    )
    assert wrapper_invocation.strip() == "& '$otpLauncher' -Supervise"
    assert "-Dsn" not in wrapper_invocation


def test_apply_cycle_does_not_stage_advisory_research_scores():
    script = Path("register-apply-cycle.ps1").read_text(encoding="utf-8")

    assert "--include-research" not in script


def test_m4_compute_score_defaults_to_max_parallelism():
    script = Path("register-fleet-tasks.ps1").read_text(encoding="utf-8")

    assert "[ValidateRange(1,16)][int]$ComputeWorkers = 16" in script
    assert "& '$computeScorePs1' -Label $Machine -Workers $ComputeWorkers" in script
