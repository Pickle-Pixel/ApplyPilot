from pathlib import Path


def test_fleet_worker_launcher_never_uses_default_zero_home_ip():
    script = Path("run-fleet-worker.ps1").read_text(encoding="utf-8")

    assert "Resolve-FleetHomeIp" in script
    assert "Refusing to start worker" in script
    assert "--home-ip" in script
    assert "--machine-owner" in script
    assert "FLEET_MACHINE_OWNER = $Label" in script
    assert "--worker-id \"$WorkerId\" --chrome-slot $Slot --agent $Agent @margs @fargs" not in script


def test_fleet_worker_launcher_can_detect_tailscale_ip_when_env_missing():
    script = Path("run-fleet-worker.ps1").read_text(encoding="utf-8")

    assert "tailscale.exe" in script
    assert "ip -4" in script
    assert "100.*" in script


def test_multi_worker_launcher_supports_non_overlapping_slot_ranges():
    script = Path("run-fleet-workers.ps1").read_text(encoding="utf-8")

    assert "[int]$StartSlot = 0" in script
    assert "$slotPattern" in script
    assert "$StartSlot..($StartSlot + $Count - 1)" in script
    assert "slots {1}..{2}" in script


def test_remote_multi_worker_launcher_requires_explicit_fleet_dsn():
    script = Path("run-fleet-workers.ps1").read_text(encoding="utf-8")

    # Fail closed for EVERY label: no home localhost fallback, no admin/shared principal.
    assert 'throw "FLEET_PG_DSN is not set for worker label' in script
    assert "user=postgres" not in script
    assert "user=fleet_worker" not in script
    assert "host=localhost" not in script


def test_remote_fleet_agent_requires_explicit_fleet_dsn():
    script = Path("fleet-agent.ps1").read_text(encoding="utf-8")

    # Fail closed for EVERY label: fleet-agent no longer defaults home to a localhost admin DSN.
    assert 'throw "FLEET_PG_DSN is not set for fleet-agent label' in script
    assert "user=postgres" not in script
    assert "user=fleet_worker" not in script


def test_fleet_agent_applies_m4_work_hours_blackout_before_reconcile():
    script = Path("fleet-agent.ps1").read_text(encoding="utf-8")

    assert "applypilot.fleet.work_hours" in script
    assert "APPLYPILOT_ALLOW_WORK_HOURS_APPLY" in script
    assert "$want = 0" in script
    assert "work-hours blackout" in script


def test_worker_launcher_probes_fleet_pg_before_starting_apply_loop():
    script = Path("run-fleet-worker.ps1").read_text(encoding="utf-8")

    assert "fleet-agent-query.py" in script
    assert "Cannot reach fleet Postgres over FLEET_PG_DSN" in script
    assert "before starting worker" in script


def test_direct_worker_launcher_enforces_lifecycle_fault_gate_before_exec():
    script = Path("run-fleet-worker.ps1").read_text(encoding="utf-8")

    gate = script.index("enforce_no_lifecycle_faults")
    launch = script.index("& $exe --dsn")
    assert gate < launch
    assert "operator reconciliation is required" in script


def test_windows_apply_worker_forces_inbox_relay_not_direct_gmail():
    script = Path("run-fleet-worker.ps1").read_text(encoding="utf-8")

    assert 'APPLYPILOT_INBOX_AUTH = "1"' in script
    assert 'APPLYPILOT_INBOX_AUTH_MODE = "relay"' in script
    assert 'APPLYPILOT_ENABLE_GMAIL_MCP = "0"' in script
    assert "hydrate-gmail.py" not in script


def test_worker_launcher_enables_greenhouse_shadow_and_defaults_submit_sentinel_off():
    script = Path("run-fleet-worker.ps1").read_text(encoding="utf-8")

    assert '$env:APPLYPILOT_GREENHOUSE_ADAPTER = "1"' in script
    assert "greenhouse-submit.enabled" in script
    assert "Test-Path -LiteralPath $greenhouseSubmitFlag" in script
    assert '$env:APPLYPILOT_GREENHOUSE_ADAPTER_SUBMIT = "1"' in script
    assert '$env:APPLYPILOT_GREENHOUSE_ADAPTER_SUBMIT = "0"' in script


def test_worker_launcher_enables_ashby_shadow_and_defaults_submit_sentinel_off():
    script = Path("run-fleet-worker.ps1").read_text(encoding="utf-8")

    assert '$env:APPLYPILOT_ASHBY_ADAPTER = "1"' in script
    assert "ashby-submit.enabled" in script
    assert "Test-Path -LiteralPath $ashbySubmitFlag" in script
    assert '$env:APPLYPILOT_ASHBY_ADAPTER_SUBMIT = "1"' in script
    assert '$env:APPLYPILOT_ASHBY_ADAPTER_SUBMIT = "0"' in script
