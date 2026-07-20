from __future__ import annotations

import base64
import hashlib
import json
import os
import runpy
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT = ROOT / "scripts" / "emergency-contain-legacy-authority.ps1"


ADAPTER = r"""
function Read-FakeState { Get-Content -LiteralPath $env:FAKE_STATE -Raw | ConvertFrom-Json }
function Write-FakeState($State) { $State | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $env:FAKE_STATE -Encoding utf8 }
function Get-LegacyTasks {
  if ($env:FAKE_ENUM_FAIL -eq 'scheduled_tasks') { throw 'task enumeration leaked-secret' }
  $state = Read-FakeState
  return @($state.tasks | ForEach-Object {
    [pscustomobject]@{
      TaskName=$_.name; TaskPath='\'; State=$_.state
      Principal=[pscustomobject]@{UserId=$_.principal}
      Actions=@([pscustomobject]@{Execute='pwsh.exe'; Arguments=$_.arguments; WorkingDirectory='C:\secret'})
    }
  })
}
function Stop-LegacyTask($Task) {
  if ($env:FAKE_FAIL_ACTION -eq 'stop_task') { throw 'injected stop failure' }
  if ($env:FAKE_NOOP_ACTIONS -eq '1') { return }
  $state=Read-FakeState; foreach($item in $state.tasks){if($item.name -eq $Task.TaskName){$item.state='Ready'}}; Write-FakeState $state
}
function Disable-LegacyTask($Task) {
  if ($env:FAKE_FAIL_ACTION -eq 'disable_task') { throw 'injected disable failure' }
  if ($env:FAKE_NOOP_ACTIONS -eq '1') { return }
  $state=Read-FakeState; foreach($item in $state.tasks){if($item.name -eq $Task.TaskName){$item.state='Disabled'}}; Write-FakeState $state
}
function Get-LegacyServices {
  if ($env:FAKE_ENUM_FAIL -eq 'services') { throw 'service enumeration leaked-secret' }
  $state=Read-FakeState
  return @($state.services | ForEach-Object { [pscustomobject]@{Name=$_.name; Status=$_.status; StartType=$_.start_type} })
}
function Stop-LegacyService($Service) {
  if ($env:FAKE_FAIL_ACTION -eq 'stop_service') { throw 'injected service failure' }
  if ($env:FAKE_NOOP_ACTIONS -eq '1') { return }
  $state=Read-FakeState; foreach($item in $state.services){if($item.name -eq $Service.Name){$item.status='Stopped'}}; Write-FakeState $state
}
function Disable-LegacyService($Service) {
  if ($env:FAKE_FAIL_ACTION -eq 'disable_service') { throw 'injected service failure' }
  if ($env:FAKE_NOOP_ACTIONS -eq '1') { return }
  $state=Read-FakeState; foreach($item in $state.services){if($item.name -eq $Service.Name){$item.start_type='Disabled'}}; Write-FakeState $state
}
function Get-LegacyProcesses {
  if ($env:FAKE_ENUM_FAIL -eq 'processes') { throw 'process enumeration leaked-secret' }
  $state=Read-FakeState
  return @($state.processes | ForEach-Object { [pscustomobject]@{ProcessId=$_.id; Name=$_.name; ExecutablePath=$_.path; CommandLine=$_.command} })
}
function Stop-LegacyProcess($Process) {
  if ($env:FAKE_FAIL_ACTION -eq 'stop_process') { throw 'injected process failure' }
  if ($env:FAKE_NOOP_ACTIONS -eq '1') { return }
  $state=Read-FakeState; $state.processes=@($state.processes | Where-Object {$_.id -ne $Process.ProcessId}); Write-FakeState $state
}
function Write-KnownEvidenceBytes($Stream, [byte[]]$Content) {
  if ($env:FAKE_NOOP_ACTIONS -eq '1') { return }
  $Stream.Position = 0
  $Stream.SetLength(0)
  $Stream.Write($Content, 0, $Content.Length)
}
function Flush-KnownEvidenceStream($Stream) {
  if ($env:FAKE_NOOP_ACTIONS -eq '1') { return }
  $Stream.Flush($true)
}
function Write-KnownWrapperDenyStub($Stream, [byte[]]$Content) {
  if ($env:FAKE_NOOP_ACTIONS -eq '1') { return }
  $Stream.Position = 0
  $Stream.SetLength(0)
  $Stream.Write($Content, 0, $Content.Length)
  $Stream.Flush($true)
}
function Get-ControlEvidence {
  return [ordered]@{
    admission_state=[ordered]@{
      available=$true; authority_source='fleet_postgres'; dsn_reference='FLEET_PG_DSN'
      fields=[ordered]@{paused=$true; ats_paused=$false; ats_apply_mode='canary'; canary_enabled=$false; linkedin_apply_mode='paused'}
    }
    unresolved_attempt_counts=[ordered]@{
      available=$true; authority_source='fleet_postgres'; dsn_reference='FLEET_PG_DSN'
      queues=[ordered]@{apply_queue=[ordered]@{leased=2; unresolved=3}; linkedin_queue=[ordered]@{leased=1; unresolved=1}}
    }
  }
}
function Get-SupplementaryLocalAttemptCounts {
  return [ordered]@{available=$true; authority_source='local_sqlite_supplementary'; ambiguous=9; in_progress=4}
}
"""


def _fixture(tmp_path: Path) -> dict[str, Path]:
    secret = "postgresql://operator:super-secret@db.invalid/fleet"
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "name": "ApplyPilot ApplyCycle",
                        "state": "Running",
                        "principal": "DOMAIN\\secret-operator",
                        "arguments": f"-dsn {secret}",
                    }
                ],
                "services": [
                    {
                        "name": "ApplyPilotWorkday",
                        "status": "Running",
                        "start_type": "Automatic",
                    }
                ],
                "processes": [
                    {
                        "id": 4242,
                        "name": "applypilot-workday-rollout.exe",
                        "path": "C:\\secret\\applypilot-workday-rollout.exe",
                        "command": "applypilot-workday-rollout canary --approval-token secret-token",
                    }
                ],
                "evidence": ["must-survive"],
            }
        ),
        encoding="utf-8",
    )
    adapter = tmp_path / "adapter.ps1"
    adapter.write_text(ADAPTER, encoding="utf-8")
    probe = tmp_path / "applypilot-probe.ps1"
    probe.write_text(
        "Write-Output 'APPLYPILOT_ADMISSION_DENIED:EMERGENCY_HOLD verified'; exit 78\n",
        encoding="utf-8",
    )
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    known = wrappers / "apply-cycle-task.ps1"
    known.write_text(f"$env:FLEET_PG_DSN = '{secret}'\n# evidence remains\n", encoding="utf-8")
    known_cmd = wrappers / "apply-worker-m2.cmd"
    known_cmd.write_text(
        "\n".join(
            [
                f"set FLEET_PG_DSN={secret}",
                f'set "DATABASE_URL={secret}/database-url"',
                "set APPLYPILOT_OPENAI_API_KEY=raw-wrapper-secret",
                "REM evidence remains",
            ]
        ),
        encoding="utf-8",
    )
    known_bat = wrappers / "linkedin-m2.bat"
    known_bat.write_text(
        f'set "APPLYPILOT_FLEET_DSN={secret}/legacy"\nREM linkedin evidence remains\n',
        encoding="utf-8",
    )
    outside = wrappers / "unrelated-maintenance.ps1"
    outside.write_text(f"# unrelated {secret}\n", encoding="utf-8")
    nested = wrappers / "nested"
    nested.mkdir()
    nested_outside = nested / "workday-unrelated.ps1"
    nested_outside.write_text(f"# nested unrelated {secret}\n", encoding="utf-8")
    return {
        "state": state,
        "adapter": adapter,
        "probe": probe,
        "wrappers": wrappers,
        "known": known,
        "known_cmd": known_cmd,
        "known_bat": known_bat,
        "outside": outside,
        "nested_outside": nested_outside,
    }


def _run(
    mode: str,
    fixture: dict[str, Path],
    *,
    fail_action: str | None = None,
    noop_actions: bool = False,
    probe=None,
    inject_adapter: bool = True,
    inject_console: bool = True,
):
    env = os.environ.copy()
    env.update(
        {
            "FAKE_STATE": str(fixture["state"]),
            "FLEET_PG_DSN": "postgresql://operator:super-secret@db.invalid/fleet",
            "APPLYPILOT_OPENAI_API_KEY": "raw-api-secret",
        }
    )
    if not inject_adapter:
        env.pop("FLEET_PG_DSN", None)
        env.pop("APPLYPILOT_FLEET_DSN", None)
    if fail_action:
        env["FAKE_FAIL_ACTION"] = fail_action
    if noop_actions:
        env["FAKE_NOOP_ACTIONS"] = "1"
    command = ["pwsh", "-NoProfile", "-File", str(SCRIPT), f"-{mode}"]
    if inject_adapter:
        command.extend(["-AdapterPath", str(fixture["adapter"])])
    command.extend(["-WrapperRoot", str(fixture["wrappers"])])
    if inject_console:
        command.extend(["-ConsoleCommand", str(probe or fixture["probe"])])
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return result, payload


def _ps_quote(value: Path | str) -> str:
    return str(value).replace("'", "''")


def _read_path_bytes(path: Path) -> bytes:
    absolute = os.path.abspath(path)
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        absolute = "\\\\?\\" + absolute
    return Path(absolute).read_bytes()


def _run_wrapper_rewrite_harness(
    tmp_path: Path,
    wrapper: Path,
    overrides: str = "",
) -> tuple[subprocess.CompletedProcess[str], dict]:
    harness = tmp_path / "wrapper-rewrite-harness.ps1"
    harness.write_text(
        "\n".join(
            [
                f". '{_ps_quote(SCRIPT)}' -DefinitionImport -WrapperRoot '{_ps_quote(wrapper.parent)}'",
                "$expectedWrappers = @(Get-WrapperSnapshot)",
                overrides,
                "$script:Failures.Clear()",
                "Remove-EmbeddedWrapperDsns -ExpectedWrappers $expectedWrappers",
                "$snapshotAvailable = $true",
                "try { $wrappers = @(Get-WrapperSnapshot) } catch { "
                "$wrappers = @(); $snapshotAvailable = $false }",
                "$snapshot = [pscustomobject]@{scheduled_tasks=@(); services=@(); "
                "process_identities=@(); wrapper_hashes=$wrappers}",
                "[ordered]@{failures=@($script:Failures); expected_wrappers=$expectedWrappers; wrappers=$wrappers; "
                "snapshot_available=$snapshotAvailable; "
                "unresolved=@(Get-UnresolvedAfterState $snapshot)} | "
                "ConvertTo-Json -Depth 8 -Compress",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(harness)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, json.loads(result.stdout.strip().splitlines()[-1])


def _run_wrapper_snapshot_harness(
    tmp_path: Path,
    wrapper_root: Path,
    overrides: str = "",
) -> tuple[subprocess.CompletedProcess[str], dict]:
    harness = tmp_path / "wrapper-snapshot-harness.ps1"
    harness.write_text(
        "\n".join(
            [
                f". '{_ps_quote(SCRIPT)}' -DefinitionImport -WrapperRoot '{_ps_quote(wrapper_root)}'",
                overrides,
                "$snapshot = @()",
                "$available = $true",
                "$errorType = $null",
                "try { $snapshot = @(Get-WrapperSnapshot) } catch { "
                "$available = $false; $errorType = $_.Exception.GetType().Name }",
                "[ordered]@{available=$available; error_type=$errorType; wrappers=$snapshot} | "
                "ConvertTo-Json -Depth 8 -Compress",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(harness)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, json.loads(result.stdout.strip().splitlines()[-1])


def _run_evidence_crash_harness(
    tmp_path: Path,
    wrapper: Path,
    boundary: str,
) -> subprocess.CompletedProcess[str]:
    overrides = {
        "post_create": "\n".join(
            [
                "function Write-KnownEvidenceBytes($Stream, [byte[]]$Content) {",
                "  Stop-Process -Id $PID -Force",
                "}",
            ]
        ),
        "pre_hardlink": "\n".join(
            [
                "function Publish-NewKnownEvidenceByHandle($Lease) {",
                "  Stop-Process -Id $PID -Force",
                "}",
            ]
        ),
        "post_hardlink": "\n".join(
            [
                "$script:OriginalEvidencePublisher = ${function:Publish-NewKnownEvidenceByHandle}",
                "function Publish-NewKnownEvidenceByHandle($Lease) {",
                "  & $script:OriginalEvidencePublisher $Lease",
                "  Stop-Process -Id $PID -Force",
                "}",
            ]
        ),
    }
    harness = tmp_path / f"crash-evidence-{boundary}.ps1"
    harness.write_text(
        "\n".join(
            [
                f". '{_ps_quote(SCRIPT)}' -DefinitionImport -WrapperRoot '{_ps_quote(wrapper.parent)}'",
                "$expectedWrappers = @(Get-WrapperSnapshot)",
                overrides[boundary],
                "Remove-EmbeddedWrapperDsns -ExpectedWrappers $expectedWrappers",
            ]
        ),
        encoding="utf-8",
    )
    return subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(harness)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_wrapper_candidate_harness(
    tmp_path: Path,
    wrapper_root: Path,
    candidate: str,
) -> tuple[subprocess.CompletedProcess[str], dict]:
    harness = tmp_path / "wrapper-candidate-harness.ps1"
    harness.write_text(
        "\n".join(
            [
                f". '{_ps_quote(SCRIPT)}' -DefinitionImport -WrapperRoot '{_ps_quote(wrapper_root)}'",
                "$lease = $null",
                "$opened = $false",
                "$errorType = $null",
                f"try {{ $lease = Open-KnownWrapperSnapshotExclusive '{_ps_quote(candidate)}'; "
                "$opened = $true } catch { $errorType = $_.Exception.GetType().Name } finally { "
                "if ($null -ne $lease) { $lease.Dispose() } }",
                "[ordered]@{opened=$opened; error_type=$errorType} | ConvertTo-Json -Compress",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(harness)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, json.loads(result.stdout.strip().splitlines()[-1])


def _run_orchestration(
    mode: str,
    fixture: dict[str, Path],
    *,
    fail_action: str | None = None,
    noop_actions: bool = False,
    enumeration_failure: str | None = None,
    probe_verified: bool = True,
    include_disposition: bool = False,
):
    harness = fixture["state"].parent / f"orchestration-{mode}.ps1"
    harness.write_text(
        "\n".join(
            [
                f". '{_ps_quote(SCRIPT)}' -DefinitionImport -WrapperRoot '{_ps_quote(fixture['wrappers'])}'",
                ADAPTER,
                "function Invoke-RejectionProbe {",
                (
                    "  return [ordered]@{status='verified'; verified=$true; decision='deny'; "
                    "exit_code=78; output_digest='probe-digest'}"
                    if probe_verified
                    else "  return [ordered]@{status='unverified'; verified=$false; decision=$null; "
                    "exit_code=1; output_digest='probe-digest'}"
                ),
                "}",
                f"$result = Invoke-ContainmentOrchestration -Operation '{mode.lower()}'",
                (
                    "$disposition = Get-ContainmentDisposition -Core $result; "
                    "[ordered]@{core=$result; disposition=$disposition} | ConvertTo-Json -Depth 12 -Compress"
                    if include_disposition
                    else "$result | ConvertTo-Json -Depth 12 -Compress"
                ),
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "FAKE_STATE": str(fixture["state"]),
            "FLEET_PG_DSN": "postgresql://operator:super-secret@db.invalid/fleet",
            "APPLYPILOT_OPENAI_API_KEY": "raw-api-secret",
        }
    )
    if fail_action:
        env["FAKE_FAIL_ACTION"] = fail_action
    if noop_actions:
        env["FAKE_NOOP_ACTIONS"] = "1"
    if enumeration_failure:
        env["FAKE_ENUM_FAIL"] = enumeration_failure
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(harness)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return result, payload


@pytest.mark.parametrize("mode", ["Inspect", "Contain"])
def test_injected_seams_are_rejected_without_touching_fixture(mode, tmp_path):
    fixture = _fixture(tmp_path)
    state_before = fixture["state"].read_bytes()
    known_before = fixture["known"].read_bytes()
    outside_before = fixture["outside"].read_bytes()
    nested_before = fixture["nested_outside"].read_bytes()

    result, payload = _run(mode, fixture)

    assert result.returncode != 0
    assert payload == {
        "schema_version": 3,
        "mode": "test",
        "operation": mode.lower(),
        "operational": False,
        "non_operational_reasons": [
            "adapter_injected",
            "console_command_injected",
            "wrapper_root_injected",
        ],
        "success": False,
        "rejection": "injected_execution_seam_disabled",
        "evidence_deleted": False,
    }
    assert len(result.stdout.strip().splitlines()) == 1
    assert "super-secret" not in result.stdout
    assert "raw-api-secret" not in result.stdout
    assert "secret-token" not in result.stdout
    assert fixture["state"].read_bytes() == state_before
    assert fixture["known"].read_bytes() == known_before
    assert fixture["outside"].read_bytes() == outside_before
    assert fixture["nested_outside"].read_bytes() == nested_before


def test_pure_after_state_evaluation_reports_unresolved_targets_and_is_nonoperational():
    snapshot = {
        "scheduled_tasks": [{"state": "Ready", "target_digest": "task-digest"}],
        "services": [
            {"status": "Running", "start_type": "Automatic", "target_digest": "service-digest"}
        ],
        "process_identities": [{"target_digest": "process-digest"}],
        "wrapper_hashes": [{"embedded_dsn": True, "path_digest": "wrapper-digest"}],
    }
    result = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-File",
            str(SCRIPT),
            "-EvaluateAfterStateJson",
            json.dumps(snapshot, separators=(",", ":")),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    payload = json.loads(result.stdout.strip())
    assert payload["mode"] == "test"
    assert payload["operation"] == "evaluate_after_state"
    assert payload["operational"] is False
    assert payload["success"] is False
    assert payload["postconditions_satisfied"] is False
    assert {item["kind"] for item in payload["unresolved_targets"]} == {
        "task",
        "service",
        "process",
        "wrapper",
    }


def test_definition_import_direct_invocation_is_nonoperational_and_nonzero():
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-DefinitionImport"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert json.loads(result.stdout.strip()) == {
        "schema_version": 3,
        "mode": "test",
        "operation": "definition_import",
        "operational": False,
        "success": False,
        "rejection": "definition_import_requires_dot_source",
        "evidence_deleted": False,
    }


def test_shared_orchestration_executes_complete_containment_and_is_idempotent(tmp_path):
    fixture = _fixture(tmp_path)
    outside_before = fixture["outside"].read_bytes()
    nested_before = fixture["nested_outside"].read_bytes()
    wrapper_preimages = {
        path.name: path.read_bytes()
        for path in [fixture["known"], fixture["known_cmd"], fixture["known_bat"]]
    }

    inspect_result, inspected = _run_orchestration("Inspect", fixture)
    assert inspect_result.returncode == 0, inspect_result.stderr
    assert inspected["operation"] == "inspect"
    assert "operational" not in inspected
    assert "success" not in inspected
    assert inspected["before_rejection_probe"]["status"] == "verified"
    assert inspected["before"]["pause_admission_state"]["authority_source"] == "fleet_postgres"
    assert inspected["before"]["unresolved_attempt_counts"]["authority_source"] == "fleet_postgres"

    first_result, first = _run_orchestration("Contain", fixture)
    assert first_result.returncode == 0, first_result.stderr
    assert first["operation"] == "contain"
    assert "operational" not in first
    assert "success" not in first
    assert first["postconditions_satisfied"] is True
    assert first["failures"] == []
    assert first["after_rejection_probe"]["decision"] == "deny"
    evidence_after_first = {}
    for wrapper in [fixture["known"], fixture["known_cmd"], fixture["known_bat"]]:
        stub = wrapper.read_text(encoding="utf-8")
        assert "acquisition denied: emergency containment" in stub.lower()
        assert "78" in stub
        assert "FLEET_PG_DSN" not in stub
        assert "APPLYPILOT_FLEET_DSN" not in stub
        assert "DATABASE_URL" not in stub
        assert "APPLYPILOT_OPENAI_API_KEY" not in stub
        assert "super-secret" not in stub
        assert "raw-wrapper-secret" not in stub
        evidence = list(wrapper.parent.glob(f"{wrapper.name}.emergency-containment-evidence-*"))
        assert len(evidence) == 1
        assert evidence[0].read_bytes() == wrapper_preimages[wrapper.name]
        evidence_after_first[wrapper.name] = (evidence[0].name, evidence[0].read_bytes())
    assert "super-secret" not in first_result.stdout
    assert "raw-wrapper-secret" not in first_result.stdout
    assert fixture["outside"].read_bytes() == outside_before
    assert fixture["nested_outside"].read_bytes() == nested_before
    state = json.loads(fixture["state"].read_text(encoding="utf-8-sig"))
    assert state["evidence"] == ["must-survive"]
    assert state["tasks"][0]["state"] == "Disabled"
    assert state["services"][0]["status"] == "Stopped"
    assert state["services"][0]["start_type"] == "Disabled"
    assert state["processes"] == []

    known_after_first = fixture["known"].read_bytes()
    second_result, second = _run_orchestration("Contain", fixture)
    assert second_result.returncode == 0, second_result.stderr
    assert second["postconditions_satisfied"] is True
    assert second["failures"] == []
    assert "operational" not in second
    assert "success" not in second
    assert fixture["known"].read_bytes() == known_after_first
    for wrapper in [fixture["known"], fixture["known_cmd"], fixture["known_bat"]]:
        evidence = list(wrapper.parent.glob(f"{wrapper.name}.emergency-containment-evidence-*"))
        assert [(item.name, item.read_bytes()) for item in evidence] == [
            evidence_after_first[wrapper.name]
        ]


def test_sensitive_wrapper_identifiers_force_evidence_backup_and_deny_stub(tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    contents = {
        "apply-cycle-task.ps1": (
            "${env:FLEET_PG_DSN} = 'raw-secret-one'\n# powershell evidence\n"
        ),
        "fleet-agent-task.ps1": (
            "[Environment]::SetEnvironmentVariable( 'DATABASE_URL', "
            "'raw-secret-two' )\n# environment evidence\n"
        ),
        "apply-worker-m2.cmd": (
            '@set "APPLYPILOT_FLEET_DSN=raw-secret-three"\r\n'
            "REM cmd evidence\r\n"
        ),
        "linkedin-m2.bat": (
            '  @SeT   "anthropic_api_key=raw-secret-four"\n'
            "REM bat evidence\n"
        ),
    }
    preimages = {}
    for name, content in contents.items():
        path = wrappers / name
        path.write_text(
            content,
            encoding="utf-16" if name == "fleet-agent-task.ps1" else "utf-8",
            newline="",
        )
        preimages[name] = path.read_bytes()

    harness = tmp_path / "sanitize-sensitive-wrappers.ps1"
    harness.write_text(
        "\n".join(
            [
                f". '{_ps_quote(SCRIPT)}' -DefinitionImport -WrapperRoot '{_ps_quote(wrappers)}'",
                "$before = @(Get-WrapperSnapshot)",
                "Remove-EmbeddedWrapperDsns -ExpectedWrappers $before",
                "$after = @(Get-WrapperSnapshot)",
                "[ordered]@{before=$before; after=$after} | ConvertTo-Json -Depth 8 -Compress",
            ]
        ),
        encoding="utf-8",
    )

    first = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(harness)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert first.returncode == 0, first.stderr
    payload = json.loads(first.stdout.strip())
    assert all(item["embedded_dsn"] for item in payload["before"])
    assert not any(item["embedded_dsn"] for item in payload["after"])
    for secret in [
        "raw-secret-one",
        "raw-secret-two",
        "raw-secret-three",
        "raw-secret-four",
    ]:
        assert secret not in first.stdout
        assert secret not in first.stderr

    sensitive_identifiers = [
        "FLEET_PG_DSN",
        "APPLYPILOT_FLEET_DSN",
        "DATABASE_URL",
        "APPLYPILOT_OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]
    first_stubs = {}
    first_evidence = {}
    for name, preimage in preimages.items():
        wrapper = wrappers / name
        stub = wrapper.read_text(encoding="utf-8")
        first_stubs[name] = wrapper.read_bytes()
        assert "emergency containment" in stub.lower()
        assert "78" in stub
        assert all(identifier.lower() not in stub.lower() for identifier in sensitive_identifiers)
        evidence = list(wrappers.glob(f"{name}.emergency-containment-evidence-*"))
        assert len(evidence) == 1
        assert evidence[0].read_bytes() == preimage
        first_evidence[name] = (evidence[0].name, evidence[0].read_bytes())

    second = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(harness)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert second.returncode == 0, second.stderr
    second_payload = json.loads(second.stdout.strip())
    assert not any(item["embedded_dsn"] for item in second_payload["before"])
    assert not any(item["embedded_dsn"] for item in second_payload["after"])
    for name, stub in first_stubs.items():
        assert (wrappers / name).read_bytes() == stub
        evidence = list(wrappers.glob(f"{name}.emergency-containment-evidence-*"))
        assert [(item.name, item.read_bytes()) for item in evidence] == [first_evidence[name]]


def test_wrapper_replacement_before_mutation_open_fails_identity_closed(tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    original = b"$env:FLEET_PG_DSN = 'original-secret'\n"
    wrapper.write_bytes(original)
    replacement = b"$env:FLEET_PG_DSN = 'replacement-secret'\n# replacement bytes\n"
    replacement_path = wrappers / "replacement.tmp"
    replacement_path.write_bytes(replacement)
    displaced_path = wrappers / "original.displaced"
    overrides = "\n".join(
        [
            "$script:WrapperBeforeLeafOpenHook = {",
            "  $script:WrapperBeforeLeafOpenHook = $null",
            f"  Move-Item -LiteralPath '{_ps_quote(wrapper)}' -Destination '{_ps_quote(displaced_path)}'",
            f"  Move-Item -LiteralPath '{_ps_quote(replacement_path)}' -Destination '{_ps_quote(wrapper)}'",
            "}",
        ]
    )

    result, payload = _run_wrapper_rewrite_harness(tmp_path, wrapper, overrides)

    assert result.returncode == 0, result.stderr
    assert [item["action"] for item in payload["failures"]] == ["rewrite_wrapper"]
    assert list(wrappers.glob(f"{wrapper.name}.emergency-containment-evidence-*")) == []
    assert wrapper.read_bytes() == replacement
    assert displaced_path.read_bytes() == original
    assert payload["wrappers"][0]["embedded_dsn"] is True
    assert payload["expected_wrappers"][0]["identity_token"]
    assert payload["wrappers"][0]["identity_token"]
    assert (
        payload["expected_wrappers"][0]["identity_token"]
        != payload["wrappers"][0]["identity_token"]
    )


def test_wrapper_replacement_attempt_is_blocked_while_exclusive_handle_is_held(tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"$env:FLEET_PG_DSN = 'held-secret'\n# held preimage\n"
    wrapper.write_bytes(preimage)
    marker = tmp_path / "replacement-attempt.txt"
    replacement = base64.b64encode(b"replacement-without-evidence").decode("ascii")
    overrides = "\n".join(
        [
            "function Write-KnownEvidenceBytes($Stream, [byte[]]$Content) {",
            "  try {",
            f"    [IO.File]::WriteAllBytes('{_ps_quote(wrapper)}', [Convert]::FromBase64String('{replacement}'))",
            f"    Set-Content -LiteralPath '{_ps_quote(marker)}' -Value 'replacement_succeeded'",
            "  } catch {",
            f"    Set-Content -LiteralPath '{_ps_quote(marker)}' -Value 'replacement_blocked'",
            "  }",
            "  $Stream.Position = 0",
            "  $Stream.SetLength(0)",
            "  $Stream.Write($Content, 0, $Content.Length)",
            "}",
        ]
    )

    result, payload = _run_wrapper_rewrite_harness(tmp_path, wrapper, overrides)

    assert result.returncode == 0, result.stderr
    assert payload["failures"] == []
    assert marker.read_text(encoding="utf-8").strip() == "replacement_blocked"
    evidence = list(wrappers.glob(f"{wrapper.name}.emergency-containment-evidence-*"))
    assert len(evidence) == 1
    assert _read_path_bytes(evidence[0]) == preimage
    assert b"emergency containment" in wrapper.read_bytes().lower()


def test_hardlink_creation_during_mutation_is_detected_and_preimage_is_unchanged(tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"$env:FLEET_PG_DSN = 'held-hardlink-secret'\n"
    wrapper.write_bytes(preimage)
    hardlink = tmp_path / "outside-hardlink.ps1"
    marker = tmp_path / "hardlink-attempt.txt"
    overrides = "\n".join(
        [
            "$script:WrapperReadCount = 0",
            "function Read-KnownWrapperStreamBytes($Stream) {",
            "  $script:WrapperReadCount++",
            "  if ($script:WrapperReadCount -eq 1) {",
            f"    try {{ New-Item -ItemType HardLink -Path '{_ps_quote(hardlink)}' "
            f"-Target '{_ps_quote(wrapper)}' -ErrorAction Stop | Out-Null; "
            f"Set-Content -LiteralPath '{_ps_quote(marker)}' -Value 'hardlink_succeeded' }} "
            f"catch {{ Set-Content -LiteralPath '{_ps_quote(marker)}' -Value 'hardlink_blocked' }}",
            "  }",
            "  $Stream.Position = 0",
            "  $buffer = [IO.MemoryStream]::new()",
            "  try { $Stream.CopyTo($buffer); return ,$buffer.ToArray() } finally { $buffer.Dispose() }",
            "}",
        ]
    )

    result, payload = _run_wrapper_rewrite_harness(tmp_path, wrapper, overrides)

    assert result.returncode == 0, result.stderr
    assert [item["action"] for item in payload["failures"]] == ["rewrite_wrapper"]
    assert marker.read_text(encoding="utf-8").strip() == "hardlink_succeeded"
    assert hardlink.read_bytes() == preimage
    assert wrapper.read_bytes() == preimage
    assert payload["snapshot_available"] is False


def test_wrapper_leaf_symlink_is_rejected_without_reading_target(tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    secret = "leaf-symlink-target-secret"
    target = tmp_path / "outside.ps1"
    target.write_text(f"$env:FLEET_PG_DSN = '{secret}'\n", encoding="utf-8")
    wrapper = wrappers / "apply-cycle-task.ps1"
    try:
        wrapper.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"file symlink creation unavailable: {exc}")

    result, payload = _run_wrapper_snapshot_harness(tmp_path, wrappers)

    assert result.returncode == 0, result.stderr
    assert payload["available"] is False
    assert payload["wrappers"] == []
    assert secret not in result.stdout
    assert str(target) not in result.stdout


def test_wrapper_root_junction_is_rejected_without_reading_target(tmp_path):
    real_root = tmp_path / "outside-root"
    real_root.mkdir()
    secret = "junction-target-secret"
    (real_root / "apply-cycle-task.ps1").write_text(
        f"$env:FLEET_PG_DSN = '{secret}'\n", encoding="utf-8"
    )
    junction = tmp_path / "wrappers"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(real_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip(f"junction creation unavailable: {created.stderr or created.stdout}")

    result, payload = _run_wrapper_snapshot_harness(tmp_path, junction)

    assert result.returncode == 0, result.stderr
    assert payload["available"] is False
    assert payload["wrappers"] == []
    assert secret not in result.stdout
    assert str(real_root) not in result.stdout


def test_higher_ancestor_junction_above_wrapper_root_is_rejected(tmp_path):
    real_parent = tmp_path / "real-parent"
    wrappers = real_parent / "ApplyPilot" / ".fleet-logs" / "_task-wrappers"
    wrappers.mkdir(parents=True)
    secret = "higher-junction-target-secret"
    (wrappers / "apply-cycle-task.ps1").write_text(
        f"$env:FLEET_PG_DSN = '{secret}'\n", encoding="utf-8"
    )
    junction_parent = tmp_path / "junction-parent"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction_parent), str(real_parent)],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip(f"junction creation unavailable: {created.stderr or created.stdout}")
    redirected_root = (
        junction_parent / "ApplyPilot" / ".fleet-logs" / "_task-wrappers"
    )

    result, payload = _run_wrapper_snapshot_harness(tmp_path, redirected_root)

    assert result.returncode == 0, result.stderr
    assert payload["available"] is False
    assert payload["wrappers"] == []
    assert secret not in result.stdout
    assert str(real_parent) not in result.stdout


def test_higher_ancestor_substitution_during_component_open_fails_closed(tmp_path):
    container = tmp_path / "container"
    original_parent = container / "ApplyPilot"
    wrappers = original_parent / ".fleet-logs" / "_task-wrappers"
    wrappers.mkdir(parents=True)
    original = wrappers / "apply-cycle-task.ps1"
    original_bytes = b"$env:FLEET_PG_DSN = 'higher-original-secret'\n"
    original.write_bytes(original_bytes)

    replacement_staging = container / "replacement-staging"
    replacement_root = replacement_staging / ".fleet-logs" / "_task-wrappers"
    replacement_root.mkdir(parents=True)
    replacement = replacement_root / original.name
    replacement_bytes = b"$env:FLEET_PG_DSN = 'higher-replacement-secret'\n"
    replacement.write_bytes(replacement_bytes)
    moved_parent = container / "moved-original"
    marker = tmp_path / "higher-ancestor-hook-ran"
    overrides = "\n".join(
        [
            "$script:WrapperAfterComponentOpenHook = {",
            "  param([string]$ComponentPath)",
            f"  if ($ComponentPath -ieq '{_ps_quote(original_parent)}') {{",
            "    $script:WrapperAfterComponentOpenHook = $null",
            f"    Move-Item -LiteralPath '{_ps_quote(original_parent)}' -Destination '{_ps_quote(moved_parent)}'",
            f"    Move-Item -LiteralPath '{_ps_quote(replacement_staging)}' -Destination '{_ps_quote(original_parent)}'",
            f"    Set-Content -LiteralPath '{_ps_quote(marker)}' -Value ran",
            "  }",
            "}",
        ]
    )

    result, payload = _run_wrapper_snapshot_harness(tmp_path, wrappers, overrides)

    assert result.returncode == 0, result.stderr
    assert marker.exists()
    assert payload["available"] is False
    assert payload["wrappers"] == []
    assert (moved_parent / ".fleet-logs" / "_task-wrappers" / original.name).read_bytes() == original_bytes
    assert (original_parent / ".fleet-logs" / "_task-wrappers" / original.name).read_bytes() == replacement_bytes
    assert "higher-original-secret" not in result.stdout
    assert "higher-replacement-secret" not in result.stdout


def test_wrapper_root_substitution_during_open_fails_closed_with_held_ancestor(tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    original = wrappers / "apply-cycle-task.ps1"
    original_bytes = b"$env:FLEET_PG_DSN = 'original-ancestor-secret'\n"
    original.write_bytes(original_bytes)
    replacement_root = tmp_path / "replacement-root"
    replacement_root.mkdir()
    replacement = replacement_root / original.name
    replacement_bytes = b"$env:FLEET_PG_DSN = 'replacement-ancestor-secret'\n"
    replacement.write_bytes(replacement_bytes)
    moved_root = tmp_path / "moved-original-root"
    marker = tmp_path / "ancestor-hook-ran"
    overrides = "\n".join(
        [
            "$script:WrapperBeforeLeafOpenHook = {",
            f"  Move-Item -LiteralPath '{_ps_quote(wrappers)}' -Destination '{_ps_quote(moved_root)}'",
            f"  Move-Item -LiteralPath '{_ps_quote(replacement_root)}' -Destination '{_ps_quote(wrappers)}'",
            f"  Set-Content -LiteralPath '{_ps_quote(marker)}' -Value ran",
            "}",
        ]
    )

    result, payload = _run_wrapper_snapshot_harness(tmp_path, wrappers, overrides)

    assert result.returncode == 0, result.stderr
    assert marker.exists()
    assert payload["available"] is False
    assert payload["wrappers"] == []
    assert (moved_root / original.name).read_bytes() == original_bytes
    assert (wrappers / original.name).read_bytes() == replacement_bytes
    assert "original-ancestor-secret" not in result.stdout
    assert "replacement-ancestor-secret" not in result.stdout


def test_wrapper_hardlink_substitution_is_rejected_without_reading_target(tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    secret = "hardlink-target-secret"
    target = tmp_path / "outside.ps1"
    target.write_text(f"$env:FLEET_PG_DSN = '{secret}'\n", encoding="utf-8")
    wrapper = wrappers / "apply-cycle-task.ps1"
    os.link(target, wrapper)

    result, payload = _run_wrapper_snapshot_harness(tmp_path, wrappers)

    assert result.returncode == 0, result.stderr
    assert payload["available"] is False
    assert payload["wrappers"] == []
    assert secret not in result.stdout
    assert str(target) not in result.stdout


@pytest.mark.parametrize(
    "alias_kind",
    ["ads", "trailing_dot", "trailing_space", "dot_component", "extended_path"],
)
def test_wrapper_win32_aliases_are_rejected_before_open(alias_kind, tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    wrapper.write_text("Write-Output 'authorized wrapper'\n", encoding="utf-8")
    if alias_kind == "ads":
        candidate = f"{wrapper}:payload"
        Path(candidate).write_text("alternate stream secret", encoding="utf-8")
    elif alias_kind == "trailing_dot":
        candidate = f"{wrapper}."
    elif alias_kind == "trailing_space":
        candidate = f"{wrapper} "
    elif alias_kind == "dot_component":
        candidate = f"{wrappers}\\.\\{wrapper.name}"
    else:
        candidate = rf"\\?\{wrapper}"

    result, payload = _run_wrapper_candidate_harness(tmp_path, wrappers, candidate)

    assert result.returncode == 0, result.stderr
    assert payload["opened"] is False
    assert str(wrapper) not in result.stdout


def test_read_only_wrapper_is_inspected_but_mutation_remains_unresolved(tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"$env:FLEET_PG_DSN = 'read-only-secret'\n"
    wrapper.write_bytes(preimage)
    wrapper.chmod(0o444)
    try:
        snapshot_result, snapshot = _run_wrapper_snapshot_harness(tmp_path, wrappers)
        rewrite_result, rewritten = _run_wrapper_rewrite_harness(tmp_path, wrapper)
    finally:
        wrapper.chmod(0o666)

    assert snapshot_result.returncode == 0, snapshot_result.stderr
    assert snapshot["available"] is True
    assert snapshot["wrappers"][0]["embedded_dsn"] is True
    assert rewrite_result.returncode == 0, rewrite_result.stderr
    assert rewritten["snapshot_available"] is True
    assert [item["action"] for item in rewritten["failures"]] == ["rewrite_wrapper"]
    assert rewritten["unresolved"][0]["conditions"] == [
        "embedded_dsn_present",
        "exact_deny_stub_absent",
    ]
    assert wrapper.read_bytes() == preimage


def test_acl_read_only_wrapper_is_inspected_but_mutation_remains_unresolved(tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"$env:FLEET_PG_DSN = 'acl-secret'\n"
    wrapper.write_bytes(preimage)
    whoami = subprocess.run(
        ["whoami", "/user", "/fo", "csv", "/nh"],
        capture_output=True,
        text=True,
        check=False,
    )
    if whoami.returncode != 0:
        pytest.skip("current Windows SID unavailable")
    sid = whoami.stdout.strip().split(",")[-1].strip().strip('"')
    denied = subprocess.run(
        ["icacls", str(wrapper), "/deny", f"*{sid}:(WD,AD,WEA,WA)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if denied.returncode != 0:
        pytest.skip(f"ACL deny unavailable: {denied.stderr or denied.stdout}")
    try:
        snapshot_result, snapshot = _run_wrapper_snapshot_harness(tmp_path, wrappers)
        rewrite_result, rewritten = _run_wrapper_rewrite_harness(tmp_path, wrapper)
    finally:
        subprocess.run(
            ["icacls", str(wrapper), "/remove:d", f"*{sid}"],
            capture_output=True,
            text=True,
            check=False,
        )

    assert snapshot_result.returncode == 0, snapshot_result.stderr
    assert snapshot["available"] is True
    assert rewrite_result.returncode == 0, rewrite_result.stderr
    assert rewritten["snapshot_available"] is True
    assert [item["action"] for item in rewritten["failures"]] == ["rewrite_wrapper"]
    assert rewritten["unresolved"]
    assert wrapper.read_bytes() == preimage


def test_exclusively_locked_wrapper_fails_snapshot_closed_and_redacts_path(tmp_path):
    wrappers = tmp_path / "locked-wrapper-root-secret"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    wrapper.write_text("$env:FLEET_PG_DSN = 'locked-content-secret'\n", encoding="utf-8")
    ready = tmp_path / "lock-ready"
    release = tmp_path / "lock-release"
    locker = tmp_path / "locker.ps1"
    locker.write_text(
        "\n".join(
            [
                f"$stream = [IO.FileStream]::new('{_ps_quote(wrapper)}', [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)",
                f"Set-Content -LiteralPath '{_ps_quote(ready)}' -Value ready",
                f"while (-not (Test-Path -LiteralPath '{_ps_quote(release)}')) {{ Start-Sleep -Milliseconds 20 }}",
                "$stream.Dispose()",
            ]
        ),
        encoding="utf-8",
    )
    process = subprocess.Popen(
        ["pwsh", "-NoProfile", "-File", str(locker)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for _ in range(250):
            if ready.exists():
                break
            if process.poll() is not None:
                pytest.fail(f"locker exited early: {process.communicate()}")
            import time

            time.sleep(0.02)
        else:
            pytest.fail("locker did not acquire wrapper handle")
        result, payload = _run_wrapper_snapshot_harness(tmp_path, wrappers)
    finally:
        release.write_text("release", encoding="utf-8")
        process.communicate(timeout=10)

    assert result.returncode == 0, result.stderr
    assert payload["available"] is False
    assert payload["wrappers"] == []
    assert "locked-wrapper-root-secret" not in result.stdout
    assert "locked-content-secret" not in result.stdout


def test_evidence_delete_and_replace_are_blocked_until_wrapper_and_reverify_finish(tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"$env:FLEET_PG_DSN = 'held-evidence-secret'\n# evidence lock\n"
    wrapper.write_bytes(preimage)
    marker = tmp_path / "evidence-race.txt"
    evidence_path = wrappers / (
        f"{wrapper.name}.emergency-containment-evidence-{hashlib.sha256(preimage).hexdigest()}"
    )
    replacement = base64.b64encode(b"unverified replacement bytes").decode("ascii")
    overrides = "\n".join(
        [
            "$script:OriginalEvidencePublisher = ${function:Publish-NewKnownEvidenceByHandle}",
            "function Publish-NewKnownEvidenceByHandle($Lease) {",
            "  & $script:OriginalEvidencePublisher $Lease",
            f"  try {{ [IO.File]::Delete('{_ps_quote(evidence_path)}'); Add-Content "
            f"-LiteralPath '{_ps_quote(marker)}' -Value 'delete_succeeded' }} "
            f"catch {{ Add-Content -LiteralPath '{_ps_quote(marker)}' -Value 'delete_blocked' }}",
            f"  try {{ [IO.File]::WriteAllBytes('{_ps_quote(evidence_path)}', "
            f"[Convert]::FromBase64String('{replacement}')); Add-Content "
            f"-LiteralPath '{_ps_quote(marker)}' -Value 'replace_succeeded' }} "
            f"catch {{ Add-Content -LiteralPath '{_ps_quote(marker)}' -Value 'replace_blocked' }}",
            "}",
        ]
    )

    result, payload = _run_wrapper_rewrite_harness(tmp_path, wrapper, overrides)

    assert result.returncode == 0, result.stderr
    assert payload["failures"] == []
    assert marker.read_text(encoding="utf-8").splitlines() == [
        "delete_blocked",
        "replace_blocked",
    ]
    evidence = list(wrappers.glob(f"{wrapper.name}.emergency-containment-evidence-*"))
    assert len(evidence) == 1
    assert evidence[0].read_bytes() == preimage
    assert b"emergency containment" in wrapper.read_bytes().lower()
    assert "held-evidence-secret" not in result.stdout
    assert "held-evidence-secret" not in result.stderr


def test_post_wrapper_flush_evidence_reverification_failure_is_reported(tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"$env:FLEET_PG_DSN = 'reverify-secret'\n# preserved\n"
    wrapper.write_bytes(preimage)
    overrides = "\n".join(
        [
            "$script:EvidenceVerificationCount = 0",
            "function Assert-KnownEvidenceStreamDigest($Stream, [string]$ExpectedDigest) {",
            "  $script:EvidenceVerificationCount++",
            "  if ($script:EvidenceVerificationCount -gt 1) { throw 'post-flush verify failed' }",
            "  $bytes = Read-KnownWrapperStreamBytes $Stream",
            "  if ((Get-KnownWrapperByteDigest $bytes) -cne $ExpectedDigest) { throw 'digest mismatch' }",
            "}",
        ]
    )

    result, payload = _run_wrapper_rewrite_harness(tmp_path, wrapper, overrides)

    assert result.returncode == 0, result.stderr
    assert any(item["action"] == "rewrite_wrapper" for item in payload["failures"])
    assert b"emergency containment" in wrapper.read_bytes().lower()
    evidence = list(wrappers.glob(f"{wrapper.name}.emergency-containment-evidence-*"))
    assert len(evidence) == 1
    assert evidence[0].read_bytes() == preimage
    assert "reverify-secret" not in result.stdout
    assert "reverify-secret" not in result.stderr


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_idempotent_deny_stub_requires_verified_preimage_evidence(tmp_path, damage):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"$env:FLEET_PG_DSN = 'idempotent-secret'\n# original\n"
    wrapper.write_bytes(preimage)

    first_result, first_payload = _run_wrapper_rewrite_harness(tmp_path, wrapper)
    assert first_result.returncode == 0, first_result.stderr
    assert first_payload["failures"] == []
    deny_stub = wrapper.read_bytes()
    evidence = list(wrappers.glob(f"{wrapper.name}.emergency-containment-evidence-*"))
    assert len(evidence) == 1
    if damage == "missing":
        evidence[0].unlink()
    else:
        evidence[0].write_bytes(b"corrupt evidence without secret")

    second_result, second_payload = _run_wrapper_rewrite_harness(tmp_path, wrapper)

    assert second_result.returncode == 0, second_result.stderr
    assert any(item["action"] == "rewrite_wrapper" for item in second_payload["failures"])
    assert second_payload["unresolved"] == [
        {
            "kind": "wrapper",
            "target_digest": second_payload["wrappers"][0]["path_digest"],
            "conditions": ["preserved_evidence_unverified"],
        }
    ]
    assert wrapper.read_bytes() == deny_stub
    assert "idempotent-secret" not in second_result.stdout
    assert "idempotent-secret" not in second_result.stderr


def test_idempotent_deny_stub_with_verified_evidence_remains_stable(tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"$env:FLEET_PG_DSN = 'stable-secret'\n# stable\n"
    wrapper.write_bytes(preimage)

    first_result, first_payload = _run_wrapper_rewrite_harness(tmp_path, wrapper)
    assert first_result.returncode == 0, first_result.stderr
    assert first_payload["failures"] == []
    deny_stub = wrapper.read_bytes()
    evidence = list(wrappers.glob(f"{wrapper.name}.emergency-containment-evidence-*"))
    assert len(evidence) == 1

    second_result, second_payload = _run_wrapper_rewrite_harness(tmp_path, wrapper)

    assert second_result.returncode == 0, second_result.stderr
    assert second_payload["failures"] == []
    assert second_payload["unresolved"] == []
    assert wrapper.read_bytes() == deny_stub
    assert evidence[0].read_bytes() == preimage
    assert "stable-secret" not in second_result.stdout
    assert "stable-secret" not in second_result.stderr


@pytest.mark.parametrize(
    ("name", "fragment"),
    [
        ("apply-cycle-task.ps1", b"Write-Output 'benign fragment'\n"),
        ("apply-worker-m2.cmd", b"@echo off\r\necho incomplete\r\n"),
        ("linkedin-m2.bat", b""),
        ("fleet-agent-task.ps1", b"\xff\xfe\x00partial"),
    ],
)
def test_every_existing_authorized_wrapper_converges_to_exact_deny_stub(
    tmp_path, name, fragment
):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / name
    wrapper.write_bytes(fragment)

    result, payload = _run_wrapper_rewrite_harness(tmp_path, wrapper)

    assert result.returncode == 0, result.stderr
    assert payload["failures"] == []
    assert payload["unresolved"] == []
    assert payload["wrappers"][0]["deny_stub"] is True
    evidence = list(wrappers.glob(f"{name}.emergency-containment-evidence-*"))
    assert len(evidence) == 1
    assert _read_path_bytes(evidence[0]) == fragment


def test_authorized_wrapper_evidence_supports_extended_length_path(tmp_path):
    wrappers = tmp_path / ("wrappers-" + ("x" * 120))
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"Write-Output 'extended length evidence'\n"
    wrapper.write_bytes(preimage)
    evidence_path = Path(
        f"{wrapper}.emergency-containment-evidence-{hashlib.sha256(preimage).hexdigest()}"
    )
    assert len(str(evidence_path)) > 260

    result, payload = _run_wrapper_rewrite_harness(tmp_path, wrapper)

    assert result.returncode == 0, result.stderr
    assert payload["failures"] == []
    assert payload["unresolved"] == []
    assert _read_path_bytes(evidence_path) == preimage


def test_partial_wrapper_write_is_rolled_back_through_same_handle(tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"Write-Output 'nonsensitive original'\n"
    wrapper.write_bytes(preimage)
    overrides = "\n".join(
        [
            "function Write-KnownWrapperDenyStub($Stream, [byte[]]$Content) {",
            "  $Stream.Position = 0; $Stream.SetLength(0)",
            "  $Stream.Write($Content, 0, 11); $Stream.Flush($true)",
            "  throw 'partial stub write'",
            "}",
        ]
    )

    result, payload = _run_wrapper_rewrite_harness(tmp_path, wrapper, overrides)

    assert result.returncode == 0, result.stderr
    assert any(item["action"] == "rewrite_wrapper" for item in payload["failures"])
    assert wrapper.read_bytes() == preimage
    assert payload["wrappers"][0]["deny_stub"] is False
    assert payload["unresolved"][0]["conditions"] == ["exact_deny_stub_absent"]
    evidence = list(wrappers.glob(f"{wrapper.name}.emergency-containment-evidence-*"))
    assert len(evidence) == 1
    assert _read_path_bytes(evidence[0]) == preimage


def test_restore_failure_stays_unresolved_and_clean_retry_preserves_fragment(tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"Write-Output 'original bytes'\n"
    wrapper.write_bytes(preimage)
    overrides = "\n".join(
        [
            "function Write-KnownWrapperDenyStub($Stream, [byte[]]$Content) {",
            "  $Stream.Position = 0; $Stream.SetLength(0)",
            "  $Stream.Write($Content, 0, 9); $Stream.Flush($true)",
            "  throw 'partial stub write'",
            "}",
            "function Restore-KnownWrapperBytes($Stream, [byte[]]$Content) { throw 'restore failed' }",
        ]
    )

    first_result, first_payload = _run_wrapper_rewrite_harness(tmp_path, wrapper, overrides)

    assert first_result.returncode == 0, first_result.stderr
    assert any(item["action"] == "rewrite_wrapper" for item in first_payload["failures"])
    fragment = wrapper.read_bytes()
    assert fragment != preimage
    assert first_payload["unresolved"][0]["conditions"] == ["exact_deny_stub_absent"]

    second_result, second_payload = _run_wrapper_rewrite_harness(tmp_path, wrapper)

    assert second_result.returncode == 0, second_result.stderr
    assert second_payload["failures"] == []
    assert second_payload["unresolved"] == []
    assert second_payload["wrappers"][0]["deny_stub"] is True
    evidence = list(wrappers.glob(f"{wrapper.name}.emergency-containment-evidence-*"))
    assert sorted(_read_path_bytes(item) for item in evidence) == sorted([preimage, fragment])


def test_failed_evidence_creation_is_retryable_and_never_accepts_fragment(tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"Write-Output 'retryable evidence'\n"
    wrapper.write_bytes(preimage)
    overrides = "function Flush-KnownEvidenceStream($Stream) { throw 'flush failed' }"

    first_result, first_payload = _run_wrapper_rewrite_harness(tmp_path, wrapper, overrides)
    assert first_result.returncode == 0, first_result.stderr
    assert any(item["action"] == "rewrite_wrapper" for item in first_payload["failures"])
    assert wrapper.read_bytes() == preimage
    assert first_payload["unresolved"][0]["conditions"] == ["exact_deny_stub_absent"]

    second_result, second_payload = _run_wrapper_rewrite_harness(tmp_path, wrapper)

    assert second_result.returncode == 0, second_result.stderr
    assert second_payload["failures"] == []
    assert second_payload["unresolved"] == []
    evidence = list(wrappers.glob(f"{wrapper.name}.emergency-containment-evidence-*"))
    assert len(evidence) == 1
    assert _read_path_bytes(evidence[0]) == preimage


def test_preexisting_corrupt_evidence_is_immutable_and_blocks_rewrite(tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"Write-Output 'immutable evidence preimage'\n"
    corrupt = b"preexisting corrupt evidence"
    wrapper.write_bytes(preimage)
    digest = hashlib.sha256(preimage).hexdigest()
    evidence = wrappers / f"{wrapper.name}.emergency-containment-evidence-{digest}"
    evidence.write_bytes(corrupt)

    result, payload = _run_wrapper_rewrite_harness(tmp_path, wrapper)

    assert result.returncode == 0, result.stderr
    assert any(item["action"] == "rewrite_wrapper" for item in payload["failures"])
    assert wrapper.read_bytes() == preimage
    assert evidence.read_bytes() == corrupt
    assert payload["unresolved"][0]["conditions"] == ["exact_deny_stub_absent"]


@pytest.mark.parametrize("failure_stage", ["partial_write", "flush", "verify"])
def test_new_partial_evidence_is_removed_through_held_handle(tmp_path, failure_stage):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"Write-Output 'cleanup new evidence'\n"
    wrapper.write_bytes(preimage)
    overrides_by_stage = {
        "partial_write": "\n".join(
            [
                "function Write-KnownEvidenceBytes($Stream, [byte[]]$Content) {",
                "  $Stream.Write($Content, 0, 7)",
                "  throw 'partial evidence write'",
                "}",
            ]
        ),
        "flush": "function Flush-KnownEvidenceStream($Stream) { throw 'evidence flush failed' }",
        "verify": (
            "function Assert-KnownEvidenceStreamDigest($Stream, [string]$ExpectedDigest) "
            "{ throw 'evidence verify failed' }"
        ),
    }

    result, payload = _run_wrapper_rewrite_harness(
        tmp_path, wrapper, overrides_by_stage[failure_stage]
    )

    assert result.returncode == 0, result.stderr
    assert any(item["action"] == "rewrite_wrapper" for item in payload["failures"])
    assert wrapper.read_bytes() == preimage
    assert list(wrappers.glob(f"{wrapper.name}.emergency-containment-evidence-*")) == []
    assert payload["unresolved"][0]["conditions"] == ["exact_deny_stub_absent"]


def test_failed_evidence_publication_deletes_partial_and_allows_clean_retry(tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"Write-Output 'cleanup failure preimage'\n"
    wrapper.write_bytes(preimage)
    overrides = "function Publish-NewKnownEvidenceByHandle($Stream) { throw 'publish failed' }"

    first_result, first_payload = _run_wrapper_rewrite_harness(tmp_path, wrapper, overrides)

    assert first_result.returncode == 0, first_result.stderr
    assert any(item["action"] == "rewrite_wrapper" for item in first_payload["failures"])
    evidence = list(wrappers.glob(f"{wrapper.name}.emergency-containment-evidence-*"))
    assert evidence == []
    assert wrapper.read_bytes() == preimage

    second_result, second_payload = _run_wrapper_rewrite_harness(tmp_path, wrapper)

    assert second_result.returncode == 0, second_result.stderr
    assert second_payload["failures"] == []
    assert second_payload["unresolved"] == []
    assert b"emergency containment" in wrapper.read_bytes().lower()
    evidence = list(wrappers.glob(f"{wrapper.name}.emergency-containment-evidence-*"))
    assert len(evidence) == 1
    assert evidence[0].read_bytes() == preimage


@pytest.mark.parametrize("boundary", ["post_create", "pre_hardlink"])
def test_process_termination_before_evidence_hardlink_is_cleanly_retryable(tmp_path, boundary):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"Write-Output 'crash-idempotent evidence'\n"
    wrapper.write_bytes(preimage)
    crashed = _run_evidence_crash_harness(tmp_path, wrapper, boundary)

    assert crashed.returncode != 0
    assert wrapper.read_bytes() == preimage
    assert list(wrappers.glob(f"{wrapper.name}.emergency-containment-evidence-*")) == []
    assert list(wrappers.glob(".applypilot-emergency-containment-stage-*")) == []

    retry_result, retry_payload = _run_wrapper_rewrite_harness(tmp_path, wrapper)

    assert retry_result.returncode == 0, retry_result.stderr
    assert retry_payload["failures"] == []
    assert retry_payload["unresolved"] == []
    evidence = list(wrappers.glob(f"{wrapper.name}.emergency-containment-evidence-*"))
    assert len(evidence) == 1
    assert evidence[0].read_bytes() == preimage


def test_process_termination_after_evidence_hardlink_preserves_exact_final_and_retries(
    tmp_path,
):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"Write-Output 'post-hardlink crash evidence'\n"
    wrapper.write_bytes(preimage)
    digest = hashlib.sha256(preimage).hexdigest()
    final_evidence = wrappers / f"{wrapper.name}.emergency-containment-evidence-{digest}"

    crashed = _run_evidence_crash_harness(tmp_path, wrapper, "post_hardlink")

    assert crashed.returncode != 0
    assert wrapper.read_bytes() == preimage
    assert final_evidence.read_bytes() == preimage
    assert list(wrappers.glob(".applypilot-emergency-containment-stage-*")) == []

    retry_result, retry_payload = _run_wrapper_rewrite_harness(tmp_path, wrapper)

    assert retry_result.returncode == 0, retry_result.stderr
    assert retry_payload["failures"] == []
    assert retry_payload["unresolved"] == []
    assert final_evidence.read_bytes() == preimage
    assert list(wrappers.glob(f"{wrapper.name}.emergency-containment-evidence-*")) == [
        final_evidence
    ]
    assert b"emergency containment" in wrapper.read_bytes().lower()


def test_wrapper_stream_write_failure_preserves_preimage_and_reports_failure(tmp_path):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"$env:FLEET_PG_DSN = 'write-failure-secret'\n# retain exactly\n"
    wrapper.write_bytes(preimage)
    overrides = "function Write-KnownWrapperDenyStub($Stream, [byte[]]$Content) { throw 'write failed' }"

    result, payload = _run_wrapper_rewrite_harness(tmp_path, wrapper, overrides)

    assert result.returncode == 0, result.stderr
    assert any(item["action"] == "rewrite_wrapper" for item in payload["failures"])
    assert wrapper.read_bytes() == preimage
    evidence = list(wrappers.glob(f"{wrapper.name}.emergency-containment-evidence-*"))
    assert len(evidence) == 1
    assert evidence[0].read_bytes() == preimage
    assert "write-failure-secret" not in result.stdout
    assert "write-failure-secret" not in result.stderr


@pytest.mark.parametrize(
    "overrides",
    [
        "function Open-KnownWrapperMutationExclusive([string]$Path, [string]$ExpectedIdentityToken) { throw 'open failed' }",
        "function Read-KnownWrapperStreamBytes($Stream) { throw 'read failed' }",
        "function Open-KnownEvidenceExclusive([string]$Path) { throw 'evidence open failed' }",
        "function Write-KnownEvidenceBytes($Stream, [byte[]]$Content) { throw 'evidence write failed' }",
        "function Flush-KnownEvidenceStream($Stream) { throw 'evidence flush failed' }",
        "function Assert-KnownEvidenceStreamDigest($Stream, [string]$ExpectedDigest) { "
        "throw 'evidence verify failed' }",
    ],
    ids=[
        "exclusive-open",
        "stream-read",
        "evidence-open",
        "evidence-write",
        "evidence-flush",
        "evidence-verify",
    ],
)
def test_wrapper_stream_stage_failure_is_recorded_without_destroying_preimage(
    tmp_path, overrides
):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    wrapper = wrappers / "apply-cycle-task.ps1"
    preimage = b"$env:FLEET_PG_DSN = 'stage-failure-secret'\n# must survive\n"
    wrapper.write_bytes(preimage)

    result, payload = _run_wrapper_rewrite_harness(tmp_path, wrapper, overrides)

    assert result.returncode == 0, result.stderr
    assert any(item["action"] == "rewrite_wrapper" for item in payload["failures"])
    if payload["snapshot_available"]:
        assert payload["wrappers"][0]["embedded_dsn"] is True
    assert wrapper.read_bytes() == preimage
    evidence = list(wrappers.glob(f"{wrapper.name}.emergency-containment-evidence-*"))
    assert len(evidence) <= 1
    if evidence and evidence[0].read_bytes() != preimage:
        assert evidence[0].read_bytes() == b""
    assert "stage-failure-secret" not in result.stdout
    assert "stage-failure-secret" not in result.stderr


def test_shared_orchestration_reports_action_failure(tmp_path):
    fixture = _fixture(tmp_path)
    result, payload = _run_orchestration("Contain", fixture, fail_action="disable_task")

    assert result.returncode == 0, result.stderr
    assert payload["postconditions_satisfied"] is False
    assert any(failure["action"] == "disable_task" for failure in payload["failures"])
    assert "operational" not in payload
    assert "success" not in payload
    assert "super-secret" not in result.stdout


def test_shared_orchestration_rejects_successful_noop_after_snapshot(tmp_path):
    fixture = _fixture(tmp_path)
    result, payload = _run_orchestration("Contain", fixture, noop_actions=True)

    assert result.returncode == 0, result.stderr
    assert payload["postconditions_satisfied"] is False
    assert {item["kind"] for item in payload["unresolved_targets"]} == {
        "task",
        "service",
        "process",
        "wrapper",
    }
    assert all(item["target_digest"] for item in payload["unresolved_targets"])
    assert "operational" not in payload
    assert "success" not in payload


@pytest.mark.parametrize("source", ["scheduled_tasks", "services", "processes"])
def test_enumeration_failure_is_sanitized_and_cannot_succeed(source, tmp_path):
    fixture = _fixture(tmp_path)
    result, payload = _run_orchestration(
        "Inspect",
        fixture,
        enumeration_failure=source,
        include_disposition=True,
    )

    assert result.returncode == 0, result.stderr
    core = payload["core"]
    assert core["postconditions_satisfied"] is False
    assert core["enumeration_failures"] == [
        {"source": source, "error": "enumeration_unavailable"}
    ]
    assert payload["disposition"] == {"success": False, "exit_code": 1}
    assert "leaked-secret" not in result.stdout


def test_unverified_inspect_disposition_is_unsuccessful_and_nonzero(tmp_path):
    fixture = _fixture(tmp_path)
    result, payload = _run_orchestration(
        "Inspect",
        fixture,
        probe_verified=False,
        include_disposition=True,
    )

    assert result.returncode == 0, result.stderr
    assert payload["core"]["after_rejection_probe"]["verified"] is False
    assert payload["disposition"] == {"success": False, "exit_code": 1}
    assert "operational" not in payload


def test_control_query_reads_dsn_from_environment_not_child_argv(tmp_path):
    argv_capture = tmp_path / "python-argv.json"
    harness = tmp_path / "dsn-argv-harness.ps1"
    harness.write_text(
        "\n".join(
            [
                f". '{_ps_quote(SCRIPT)}' -DefinitionImport",
                "function py {",
                "  [IO.File]::WriteAllText($env:ARGV_CAPTURE, (@($args) | ConvertTo-Json -Compress))",
                "  $global:LASTEXITCODE = 0",
                "  Write-Output '{\"state\":{\"paused\":true},\"counts\":{}}'",
                "}",
                "$result = Get-ControlEvidence",
                "$result | ConvertTo-Json -Depth 8 -Compress",
            ]
        ),
        encoding="utf-8",
    )
    secret_dsn = "postgresql://operator:argv-secret@db.invalid/fleet"
    env = os.environ.copy()
    env["FLEET_PG_DSN"] = secret_dsn
    env.pop("APPLYPILOT_FLEET_DSN", None)
    env["ARGV_CAPTURE"] = str(argv_capture)

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(harness)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["admission_state"]["available"] is True
    argv_text = argv_capture.read_text(encoding="utf-8-sig")
    assert secret_dsn not in argv_text
    assert "argv-secret" not in argv_text


def test_authority_inventory_enumerates_all_and_survivors_are_unresolved(tmp_path):
    harness = tmp_path / "authority-inventory.ps1"
    harness.write_text(
        "\n".join(
            [
                f". '{_ps_quote(SCRIPT)}' -DefinitionImport",
                "$global:FakeTasks = @(",
                "  [pscustomobject]@{TaskName='ApplyPilot ApplyCycle'; TaskPath='\\'; State='Ready'; Principal=[pscustomobject]@{UserId='user'}; Actions=@([pscustomobject]@{Execute='cmd.exe'; Arguments='/c echo benign'; WorkingDirectory='C:\\repo'})},",
                "  [pscustomobject]@{TaskName='Neutral Exec'; TaskPath='\\'; State='Ready'; Principal=[pscustomobject]@{UserId='user'}; Actions=@([pscustomobject]@{Execute='C:\\ApplyPilot\\applypilot-fleet-apply.exe'; Arguments='--worker-id m2-0'; WorkingDirectory='C:\\repo'})},",
                "  [pscustomobject]@{TaskName='Neutral Task'; TaskPath='\\'; State='Ready'; Principal=[pscustomobject]@{UserId='user'}; Actions=@([pscustomobject]@{Execute='pwsh.exe'; Arguments='-File C:\\repo\\run-fleet-worker.ps1'; WorkingDirectory='C:\\repo'})},",
                "  [pscustomobject]@{TaskName='Program Files PowerShell Action'; TaskPath='\\'; State='Ready'; Principal=[pscustomobject]@{UserId='user'}; Actions=@([pscustomobject]@{Execute='C:\\Program Files\\PowerShell\\7\\pwsh.exe'; Arguments='-NoProfile -File run-fleet-workers.ps1 -Count 2'; WorkingDirectory='C:\\repo'})},",
                "  [pscustomobject]@{TaskName='Program Files Python Action'; TaskPath='\\'; State='Ready'; Principal=[pscustomobject]@{UserId='user'}; Actions=@([pscustomobject]@{Execute='C:\\Program Files\\Python312\\python.exe'; Arguments='-I -m applypilot.fleet.apply_worker_main --worker-id m2-0'; WorkingDirectory='C:\\repo'})},",
                "  [pscustomobject]@{TaskName='Program Files PowerShell Benign'; TaskPath='\\'; State='Ready'; Principal=[pscustomobject]@{UserId='user'}; Actions=@([pscustomobject]@{Execute='C:\\Program Files\\PowerShell\\7\\pwsh.exe'; Arguments='-Command benign -File run-fleet-workers.ps1'; WorkingDirectory='C:\\repo'})},",
                "  [pscustomobject]@{TaskName='Program Files Python Benign'; TaskPath='\\'; State='Ready'; Principal=[pscustomobject]@{UserId='user'}; Actions=@([pscustomobject]@{Execute='C:\\Program Files\\Python312\\python.exe'; Arguments='-c pass -m applypilot.fleet.apply_worker_main'; WorkingDirectory='C:\\repo'})},",
                "  [pscustomobject]@{TaskName='ApplyPilot LinkedIn Report'; TaskPath='\\'; State='Ready'; Principal=[pscustomobject]@{UserId='user'}; Actions=@([pscustomobject]@{Execute='C:\\ApplyPilot\\applypilot-workday-report.exe'; Arguments='--read-only'; WorkingDirectory='C:\\repo'})},",
                "  [pscustomobject]@{TaskName='ApplyPilot Workday Discovery'; TaskPath='\\'; State='Ready'; Principal=[pscustomobject]@{UserId='user'}; Actions=@([pscustomobject]@{Execute='cmd.exe'; Arguments='/c echo discovery'; WorkingDirectory='C:\\repo'})},",
                "  [pscustomobject]@{TaskName='ApplyPilot LinkedIn Monitor'; TaskPath='\\'; State='Ready'; Principal=[pscustomobject]@{UserId='user'}; Actions=@([pscustomobject]@{Execute='cmd.exe'; Arguments='/c echo monitor'; WorkingDirectory='C:\\repo'})},",
                "  [pscustomobject]@{TaskName='Benign Task'; TaskPath='\\'; State='Ready'; Principal=[pscustomobject]@{UserId='user'}; Actions=@([pscustomobject]@{Execute='pwsh.exe'; Arguments='-File C:\\repo\\run-fleet-worker-report.ps1'; WorkingDirectory='C:\\repo'})}",
                ")",
                "$global:FakeServices = @(",
                "  [pscustomobject]@{Name='ApplyPilotWorkday'; State='Running'; StartMode='Auto'; PathName='cmd.exe /c echo benign'},",
                "  [pscustomobject]@{Name='NeutralService'; State='Running'; StartMode='Auto'; PathName='pwsh.exe -File C:\\repo\\run-tarpon-linkedin.ps1'},",
                "  [pscustomobject]@{Name='ApplyPilotWorkdayReport'; State='Running'; StartMode='Auto'; PathName='C:\\ApplyPilot\\applypilot-workday-report.exe --read-only'},",
                "  [pscustomobject]@{Name='ApplyPilot LinkedIn Report'; State='Running'; StartMode='Auto'; PathName='cmd.exe /c echo report'},",
                "  [pscustomobject]@{Name='ApplyPilot Workday Discovery'; State='Running'; StartMode='Auto'; PathName='cmd.exe /c echo discovery'},",
                "  [pscustomobject]@{Name='ApplyPilotLinkedInMonitor'; State='Running'; StartMode='Auto'; PathName='cmd.exe /c echo monitor'},",
                "  [pscustomobject]@{Name='BenignService'; State='Running'; StartMode='Auto'; PathName='pwsh.exe -File C:\\repo\\run-tarpon-linkedin-report.ps1'}",
                ")",
                "$global:FakeProcesses = @(",
                "  [pscustomobject]@{ProcessId=9001; Name='py.exe'; ExecutablePath='C:\\Windows\\py.exe'; CommandLine='py.exe -3.12 -X utf8 -m applypilot.fleet.apply_worker_main --worker-id m2-0'},",
                "  [pscustomobject]@{ProcessId=9002; Name='py.exe'; ExecutablePath='C:\\Windows\\py.exe'; CommandLine='py.exe -m applypilot.fleet.apply_worker_main_helper --read-only'}",
                ")",
                "function Get-ScheduledTask { [CmdletBinding()] param([string]$TaskName); if ($PSBoundParameters.ContainsKey('TaskName')) { throw 'name-prefiltered' }; $global:FakeTasks }",
                "function Get-CimInstance { [CmdletBinding()] param([Parameter(Position=0)][string]$ClassName); if ($ClassName -eq 'Win32_Service') { $global:FakeServices } elseif ($ClassName -eq 'Win32_Process') { $global:FakeProcesses } }",
                "$tasks = @(Get-TaskSnapshot)",
                "$services = @(Get-ServiceSnapshot)",
                "$processes = @(Get-ProcessSnapshot)",
                "$snapshot = [pscustomobject]@{scheduled_tasks=$tasks; services=$services; process_identities=$processes; wrapper_hashes=@()}",
                "$unresolved = @(Get-UnresolvedAfterState $snapshot)",
                "[ordered]@{tasks=$tasks; services=$services; processes=$processes; unresolved=$unresolved} | ConvertTo-Json -Depth 10 -Compress",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(harness)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert [item["name"] for item in payload["tasks"]] == [
        "ApplyPilot ApplyCycle",
        "Neutral Exec",
        "Neutral Task",
        "Program Files PowerShell Action",
        "Program Files Python Action",
        "Program Files Python Benign",
    ]
    assert payload["tasks"][-1]["classification"] == "ambiguous"
    assert [item["name"] for item in payload["services"]] == [
        "ApplyPilotWorkday",
        "NeutralService",
    ]
    assert [item["process_id"] for item in payload["processes"]] == [9001]
    assert {item["kind"] for item in payload["unresolved"]} == {
        "task",
        "service",
        "process",
    }
    assert all(item["target_digest"] for item in payload["unresolved"])


def test_generated_wrapper_inventory_uses_only_exact_authorized_basenames(tmp_path):
    authorized = [
        "apply-cycle-task.ps1",
        "fleet-agent-task.ps1",
        "apply-worker-m2.cmd",
        "linkedin-m2.bat",
    ]
    benign = [
        "linkedin-report.ps1",
        "workday-discovery.ps1",
        "workday-monitor.cmd",
        "fleet-agent-monitor.ps1",
    ]
    for name in authorized + benign:
        (tmp_path / name).write_text("# fixture\n", encoding="utf-8")
    harness = tmp_path / "wrapper-inventory.ps1"
    harness.write_text(
        "\n".join(
            [
                f". '{_ps_quote(SCRIPT)}' -DefinitionImport -WrapperRoot '{_ps_quote(tmp_path)}'",
                "@(Get-KnownWrapperPaths | ForEach-Object Name) | ConvertTo-Json -Compress",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(harness)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert sorted(json.loads(result.stdout.strip())) == sorted(authorized)


@pytest.mark.parametrize("mode", ["Inspect", "Contain"])
def test_operational_wrapper_root_is_rejected_before_enumeration(mode, tmp_path):
    secret = "wrapper-root-must-not-be-enumerated"
    wrapper_root = tmp_path / secret
    wrapper_root.mkdir()
    (wrapper_root / "apply-cycle-task.ps1").write_text(
        "$env:FLEET_PG_DSN = 'must-not-be-read'\n", encoding="utf-8"
    )

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), f"-{mode}", "-WrapperRoot", str(wrapper_root)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert len(result.stdout.strip().splitlines()) == 1
    assert secret not in result.stdout
    assert "must-not-be-read" not in result.stdout
    payload = json.loads(result.stdout.strip())
    assert payload["operational"] is False
    assert payload["success"] is False
    assert payload["rejection"] == "injected_execution_seam_disabled"
    assert payload["non_operational_reasons"] == ["wrapper_root_injected"]


def test_localappdata_environment_cannot_redirect_authorized_wrapper_root(tmp_path):
    redirected_local = tmp_path / "mutable-localappdata-secret"
    redirected_root = redirected_local / "ApplyPilot" / "_task-wrappers"
    redirected_root.mkdir(parents=True)
    (redirected_root / "apply-cycle-task.ps1").write_text(
        "$env:FLEET_PG_DSN = 'redirected-wrapper-secret'\n", encoding="utf-8"
    )
    harness = tmp_path / "native-known-folder-root.ps1"
    harness.write_text(
        "\n".join(
            [
                f". '{_ps_quote(SCRIPT)}' -DefinitionImport",
                "$roots = @(Get-KnownAuthorizedWrapperRoots)",
                "$candidates = @(Get-KnownWrapperPaths)",
                f"[ordered]@{{redirected_root_authorized=($roots -contains '{_ps_quote(redirected_root)}'); "
                f"redirected_candidate=(@($candidates | Where-Object FullName -eq '{_ps_quote(redirected_root / 'apply-cycle-task.ps1')}').Count -gt 0)}} | "
                "ConvertTo-Json -Compress",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["LOCALAPPDATA"] = str(redirected_local)

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(harness)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip()) == {
        "redirected_root_authorized": False,
        "redirected_candidate": False,
    }
    assert "mutable-localappdata-secret" not in result.stdout
    assert "redirected-wrapper-secret" not in result.stdout


def test_ambiguous_authorities_remain_unresolved_and_are_never_acted_on(tmp_path):
    harness = tmp_path / "ambiguous-authorities.ps1"
    harness.write_text(
        "\n".join(
            [
                f". '{_ps_quote(SCRIPT)}' -DefinitionImport",
                "$global:ActionCalled = $false",
                "$task = [pscustomobject]@{TaskName='Neutral Task'; TaskPath='\\'; State='Ready'; Principal=[pscustomobject]@{UserId='user'}; Actions=@([pscustomobject]@{Execute='python.exe'; Arguments='--unknown=applypilot.fleet.apply_worker_main'; WorkingDirectory='C:\\repo'})}",
                "$service = [pscustomobject]@{Name='NeutralService'; State='Running'; StartMode='Auto'; PathName='pwsh.exe -Unknown=run-fleet-workers.ps1'}",
                "$process = [pscustomobject]@{ProcessId=4300; Name='python.exe'; ExecutablePath='C:\\Python312\\python.exe'; CreationDate=[datetime]'2026-07-13T16:00:00Z'; CommandLine='python --unknown=applypilot.fleet.apply_worker_main'}",
                "function Get-ScheduledTask { @($task) }",
                "function Get-CimInstance { [CmdletBinding()] param([Parameter(Position=0)][string]$ClassName); if ($ClassName -eq 'Win32_Service') { @($service) } elseif ($ClassName -eq 'Win32_Process') { @($process) } }",
                "function Get-KnownWrapperPaths { @() }",
                "function Stop-LegacyTask { $global:ActionCalled = $true }",
                "function Disable-LegacyTask { $global:ActionCalled = $true }",
                "function Stop-LegacyService { $global:ActionCalled = $true }",
                "function Disable-LegacyService { $global:ActionCalled = $true }",
                "function Open-LegacyProcessHandle { $global:ActionCalled = $true; throw 'must not open' }",
                "function Get-ControlEvidence { [ordered]@{admission_state=[ordered]@{available=$true; authority_source='fleet_postgres'; fields=[ordered]@{paused=$true}}; unresolved_attempt_counts=[ordered]@{available=$true; authority_source='fleet_postgres'; queues=[ordered]@{}}} }",
                "function Get-SupplementaryLocalAttemptCounts { [ordered]@{available=$false; authority_source='local_sqlite_supplementary'} }",
                "function Invoke-RejectionProbe { [ordered]@{status='verified'; verified=$true; decision='deny'; exit_code=78; output_digest='probe-digest'} }",
                "$core = Invoke-ContainmentOrchestration -Operation 'contain'",
                "$disposition = Get-ContainmentDisposition -Core $core",
                "[ordered]@{core=$core; disposition=$disposition; action_called=$global:ActionCalled} | ConvertTo-Json -Depth 12 -Compress",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(harness)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert payload["action_called"] is False
    assert payload["disposition"] == {"success": False, "exit_code": 1}
    assert {item["kind"] for item in payload["core"]["unresolved_targets"]} == {
        "task",
        "service",
        "process",
    }
    assert all(
        item["conditions"] == ["ambiguous_command"]
        for item in payload["core"]["unresolved_targets"]
    )
    assert all(
        item["classification"] == "ambiguous"
        for item in payload["core"]["after"]["scheduled_tasks"]
        + payload["core"]["after"]["services"]
        + payload["core"]["after"]["process_identities"]
    )


def test_interpreter_payload_ambiguity_blocks_containment_success(tmp_path):
    encoded = base64.b64encode(
        "run-fleet-workers.ps1 || Write-Output done".encode("utf-16le")
    ).decode("ascii")
    commands = [
        "python -c \"import runpy; runpy.run_module('applypilot.fleet.apply_worker_main')\"",
        'python -c "print(\'opaque but apparently benign\')"',
        "python -cpass",
        "python",
        "python.exe",
        "py",
        "python -",
        "pwsh",
        "powershell.exe",
        "pwsh -",
        "powershell.exe -",
        "pwsh -SSHServerMode",
        "powershell.exe -sshs",
        "cmd",
        "cmd.exe /d",
        "pwsh -File -",
        "pwsh -Command -",
        "pwsh -Unknown benign -Command -",
        'pwsh -Command "run-fleet-workers.ps1; Write-Output done"',
        'pwsh -Command "run-fleet-workers.ps1 && Write-Output done"',
        'pwsh -Command "run-fleet-workers.ps1 &"',
        'pwsh -Command "run-fleet-workers.ps1 -Count 2" ; Write-Output done',
        'pwsh -Command "run-fleet-workers.ps1 -Count 2" && Write-Output done',
        'pwsh -Command "Invoke-Expression \'run-fleet-workers.ps1\'"',
        'pwsh -Command "Start-Process run-fleet-workers.ps1"',
        'pwsh -Command Start-Process -FilePath "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
        'pwsh -Command Start-Process "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
        'pwsh -Command saps -FilePath "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
        'pwsh -Command Start-Process -FilePath:"C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
        'pwsh -Command saps -FilePath="C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
        f"pwsh -EncodedCommand {encoded}",
        "pwsh -EncodedCommand not-valid-base64!",
        'pwsh -Command "& $env:APPLYPILOT_TARGET"',
        'pwsh -Command "Start-Process -FilePath $env:APPLYPILOT_TARGET"',
        'pwsh -Command "Invoke-Expression $env:APPLYPILOT_COMMAND"',
        "cmd /c(run-fleet-worker.cmd)",
        "cmd /c^run-fleet-worker.cmd",
        'cmd /c"run-fleet-worker.cmd & echo done"',
        "cmd /c %APPLYPILOT_COMMAND%",
        "cmd /v:on/k!APPLYPILOT_COMMAND!",
        "cmd /c echo first & %APPLYPILOT_COMMAND%",
        "cmd /c echo first & run-fleet-^worker.cmd",
    ]
    process_rows = ",".join(
        (
            "[pscustomobject]@{ProcessId="
            f"{4400 + index}; Name='fixture.exe'; ExecutablePath='C:\\fixture.exe'; "
            "CreationDate=[datetime]'2026-07-13T16:00:00Z'; CommandLine='"
            f"{_ps_quote(command)}'"
            "}"
        )
        for index, command in enumerate(commands)
    )
    harness = tmp_path / "interpreter-payload-ambiguity.ps1"
    harness.write_text(
        "\n".join(
            [
                f". '{_ps_quote(SCRIPT)}' -DefinitionImport",
                "$global:ActionCalled = $false",
                f"$global:FixtureProcesses = @({process_rows})",
                "function Get-ScheduledTask { @() }",
                "function Get-CimInstance { [CmdletBinding()] param([Parameter(Position=0)][string]$ClassName); if ($ClassName -eq 'Win32_Service') { @() } elseif ($ClassName -eq 'Win32_Process') { @($global:FixtureProcesses) } }",
                "function Get-KnownWrapperPaths { @() }",
                "function Open-LegacyProcessHandle { $global:ActionCalled = $true; throw 'must not open' }",
                "function Get-ControlEvidence { [ordered]@{admission_state=[ordered]@{available=$true; authority_source='fleet_postgres'; fields=[ordered]@{paused=$true}}; unresolved_attempt_counts=[ordered]@{available=$true; authority_source='fleet_postgres'; queues=[ordered]@{}}} }",
                "function Get-SupplementaryLocalAttemptCounts { [ordered]@{available=$false; authority_source='local_sqlite_supplementary'} }",
                "function Invoke-RejectionProbe { [ordered]@{status='verified'; verified=$true; decision='deny'; exit_code=78; output_digest='probe-digest'} }",
                "$core = Invoke-ContainmentOrchestration -Operation 'contain'",
                "$disposition = Get-ContainmentDisposition -Core $core",
                "[ordered]@{core=$core; disposition=$disposition; action_called=$global:ActionCalled} | ConvertTo-Json -Depth 12 -Compress",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(harness)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert payload["action_called"] is False
    assert payload["disposition"] == {"success": False, "exit_code": 1}
    assert len(payload["core"]["unresolved_targets"]) == len(commands)
    assert all(
        item["conditions"] == ["ambiguous_command"]
        for item in payload["core"]["unresolved_targets"]
    )


def test_interpreter_execution_modes_cannot_disappear_from_inspect_snapshot(tmp_path):
    encoded = base64.b64encode(
        "run-fleet-workers.ps1 -Count 2".encode("utf-16le")
    ).decode("ascii")
    commands = [
        ("python -- apply_worker_main.py --worker-id m2-0", "acquisition"),
        (
            "python -c \"import runpy; runpy.run_module('applypilot.fleet.apply_worker_main')\"",
            "ambiguous",
        ),
        ('pwsh -Command "run-fleet-workers.ps1 -Count 2"', "acquisition"),
        (
            'pwsh -Command "run-fleet-workers.ps1; Write-Output done"',
            "ambiguous",
        ),
        (f"pwsh -EncodedCommand {encoded}", "acquisition"),
        ("pwsh -EncodedCommand not-valid-base64!", "ambiguous"),
    ]
    process_rows = ",".join(
        (
            "[pscustomobject]@{ProcessId="
            f"{4500 + index}; Name='fixture.exe'; ExecutablePath='C:\\fixture.exe'; "
            "CreationDate=[datetime]'2026-07-13T16:00:00Z'; CommandLine='"
            f"{_ps_quote(command)}'"
            "}"
        )
        for index, (command, _) in enumerate(commands)
    )
    harness = tmp_path / "interpreter-execution-mode-snapshot.ps1"
    harness.write_text(
        "\n".join(
            [
                f". '{_ps_quote(SCRIPT)}' -DefinitionImport",
                f"$global:FixtureProcesses = @({process_rows})",
                "function Get-ScheduledTask { @() }",
                "function Get-CimInstance { [CmdletBinding()] param([Parameter(Position=0)][string]$ClassName); if ($ClassName -eq 'Win32_Service') { @() } elseif ($ClassName -eq 'Win32_Process') { @($global:FixtureProcesses) } }",
                "function Get-KnownWrapperPaths { @() }",
                "function Get-ControlEvidence { [ordered]@{admission_state=[ordered]@{available=$true; authority_source='fleet_postgres'; fields=[ordered]@{paused=$true}}; unresolved_attempt_counts=[ordered]@{available=$true; authority_source='fleet_postgres'; queues=[ordered]@{}}} }",
                "function Get-SupplementaryLocalAttemptCounts { [ordered]@{available=$false; authority_source='local_sqlite_supplementary'} }",
                "function Invoke-RejectionProbe { [ordered]@{status='verified'; verified=$true; decision='deny'; exit_code=78; output_digest='probe-digest'} }",
                "$core = Invoke-ContainmentOrchestration -Operation 'inspect'",
                "$disposition = Get-ContainmentDisposition -Core $core",
                "[ordered]@{core=$core; disposition=$disposition} | ConvertTo-Json -Depth 12 -Compress",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(harness)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert payload["disposition"] == {"success": False, "exit_code": 1}
    assert [
        item["classification"]
        for item in payload["core"]["after"]["process_identities"]
    ] == [expected for _, expected in commands]
    assert len(payload["core"]["unresolved_targets"]) == len(commands)


def _run_process_handle_scenario(
    tmp_path, scenario: str, replacement_command: str = "", native_tick_delta: int = 0
):
    harness = tmp_path / "process-identity-race.ps1"
    replacement = (
        f"[pscustomobject]@{{ProcessId=4242; Name='reused.exe'; "
        "ExecutablePath='C:\\Windows\\reused.exe'; "
        "CreationDate=[datetime]'2026-07-13T16:01:00Z'; "
        f"CommandLine='{_ps_quote(replacement_command)}'}}"
        if scenario == "reuse"
        else "$null"
    )
    harness.write_text(
        "\n".join(
            [
                f". '{_ps_quote(SCRIPT)}' -DefinitionImport",
                "$global:ProcessEnumCount = 0",
                "$global:PidStopCalled = $false",
                "$global:OpenPid = $null",
                "$global:TerminatedToken = $null",
                "$global:HandleDisposed = $false",
                "$global:OriginalProcess = [pscustomobject]@{ProcessId=4242; Name='python.exe'; ExecutablePath='C:\\Python312\\python.exe'; CreationDate=[datetime]'2026-07-13T16:00:00Z'; CommandLine='python.exe -m applypilot.fleet.apply_worker_main --worker-id original'}",
                f"$global:CurrentProcess = {replacement}",
                "function Get-LegacyTasks { @() }",
                "function Get-LegacyServices { @() }",
                "function Get-KnownWrapperPaths { @() }",
                "function Get-CimInstance {",
                "  [CmdletBinding()] param([Parameter(Position=0)][string]$ClassName, [string]$Filter)",
                "  if ($ClassName -eq 'Win32_Service') { return @() }",
                "  if ($ClassName -ne 'Win32_Process') { return @() }",
                "  if ($PSBoundParameters.ContainsKey('Filter')) { return @($global:CurrentProcess) }",
                "  $global:ProcessEnumCount += 1",
                "  if ($global:ProcessEnumCount -le 2) { return $global:OriginalProcess }",
                "  return @($global:CurrentProcess)",
                "}",
                "function Open-LegacyProcessHandle([int]$ProcessId) {",
                "  $global:OpenPid = $ProcessId",
                ("  throw 'simulated open failure'" if scenario == "open_failure" else "  $handle = [pscustomobject]@{Token='validated-handle'}"),
                *(
                    []
                    if scenario == "open_failure"
                    else [
                        "  $handle | Add-Member -MemberType ScriptMethod -Name Dispose -Value { $global:HandleDisposed = $true }",
                        "  return $handle",
                    ]
                ),
                "}",
                "function Get-LegacyProcessHandleIdentity($Handle) {",
                "  if ($Handle.Token -ne 'validated-handle') { throw 'wrong handle' }",
                *(
                    ["  throw 'simulated identity failure'"]
                    if scenario == "identity_failure"
                    else []
                ),
                (
                    "  return [ordered]@{creation_file_time_utc=$global:CurrentProcess.CreationDate.ToUniversalTime().ToFileTimeUtc(); executable_path=$global:CurrentProcess.ExecutablePath}"
                    if scenario == "reuse"
                    else f"  return [ordered]@{{creation_file_time_utc=$global:OriginalProcess.CreationDate.ToUniversalTime().ToFileTimeUtc() + {native_tick_delta}; executable_path=$global:OriginalProcess.ExecutablePath}}"
                ),
                "}",
                "function Stop-LegacyProcessHandle($Handle) {",
                "  if ($Handle.Token -ne 'validated-handle') { throw 'wrong handle' }",
                "  $global:TerminatedToken = $Handle.Token",
                *( ["  throw 'simulated termination failure'"] if scenario == "terminate_failure" else [] ),
                "}",
                "function Stop-Process { [CmdletBinding()] param([int]$Id, [switch]$Force); $global:PidStopCalled = $true }",
                "function Get-ControlEvidence { return [ordered]@{admission_state=[ordered]@{available=$true; authority_source='fleet_postgres'; fields=[ordered]@{paused=$true}}; unresolved_attempt_counts=[ordered]@{available=$true; authority_source='fleet_postgres'; queues=[ordered]@{}}} }",
                "function Get-SupplementaryLocalAttemptCounts { return [ordered]@{available=$false; authority_source='local_sqlite_supplementary'} }",
                "function Invoke-RejectionProbe { return [ordered]@{status='verified'; verified=$true; decision='deny'; exit_code=78; output_digest='probe-digest'} }",
                "$core = Invoke-ContainmentOrchestration -Operation 'contain'",
                "$disposition = Get-ContainmentDisposition -Core $core",
                "[ordered]@{core=$core; disposition=$disposition; pid_stop_called=$global:PidStopCalled; open_pid=$global:OpenPid; terminated_token=$global:TerminatedToken; handle_disposed=$global:HandleDisposed} | ConvertTo-Json -Depth 12 -Compress",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(harness)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip())


@pytest.mark.parametrize(
    ("reused_command", "expected_success", "expected_unresolved"),
    [
        ("notepad.exe report.txt", True, 0),
        ("python.exe -m applypilot.fleet.apply_worker_main --worker-id reused", False, 1),
    ],
)
def test_process_reuse_is_skipped_and_after_snapshot_decides(
    tmp_path, reused_command, expected_success, expected_unresolved
):
    payload = _run_process_handle_scenario(tmp_path, "reuse", reused_command)

    assert payload["pid_stop_called"] is False
    assert payload["open_pid"] == 4242
    assert payload["terminated_token"] is None
    assert payload["handle_disposed"] is True
    assert payload["disposition"] == {
        "success": expected_success,
        "exit_code": 0 if expected_success else 1,
    }
    assert len(payload["core"]["unresolved_targets"]) == expected_unresolved
    assert payload["core"]["failures"] == []
    assert len(payload["core"]["skipped_actions"]) == 1
    skipped = payload["core"]["skipped_actions"][0]
    assert skipped["action"] == "stop_process"
    assert skipped["result"] == "identity_changed"
    assert skipped["target_digest"]


@pytest.mark.parametrize(
    ("scenario", "expected_skip", "expected_terminated", "expected_disposed"),
    [
        ("open_failure", "handle_open_failed", None, False),
        ("identity_failure", "identity_unavailable", None, True),
        ("terminate_failure", "termination_failed", "validated-handle", True),
        ("successful_termination", None, "validated-handle", True),
    ],
)
def test_process_handle_lifecycle_never_falls_back_to_pid_stop(
    tmp_path, scenario, expected_skip, expected_terminated, expected_disposed
):
    payload = _run_process_handle_scenario(tmp_path, scenario)

    assert payload["pid_stop_called"] is False
    assert payload["open_pid"] == 4242
    assert payload["terminated_token"] == expected_terminated
    assert payload["handle_disposed"] is expected_disposed
    assert payload["disposition"] == {"success": True, "exit_code": 0}
    assert payload["core"]["unresolved_targets"] == []
    skipped = payload["core"]["skipped_actions"]
    if expected_skip is None:
        assert skipped == []
    else:
        assert len(skipped) == 1
        assert skipped[0]["result"] == expected_skip
        assert skipped[0]["target_digest"]


@pytest.mark.parametrize(
    ("native_tick_delta", "expected_match"),
    [(1, True), (9, True), (10, False)],
)
def test_process_creation_filetime_allows_only_sub_microsecond_cim_loss(
    tmp_path, native_tick_delta, expected_match
):
    payload = _run_process_handle_scenario(
        tmp_path, "precision", native_tick_delta=native_tick_delta
    )

    assert payload["pid_stop_called"] is False
    assert payload["terminated_token"] == (
        "validated-handle" if expected_match else None
    )
    assert payload["handle_disposed"] is True
    if expected_match:
        assert payload["core"]["skipped_actions"] == []
    else:
        assert payload["core"]["skipped_actions"][0]["result"] == "identity_changed"


@pytest.mark.parametrize(
    ("inject_adapter", "inject_console", "reason"),
    [(True, False, "adapter_injected"), (False, True, "console_command_injected")],
)
def test_each_injected_seam_marks_receipt_non_operational(
    tmp_path, inject_adapter, inject_console, reason
):
    fixture = _fixture(tmp_path)
    result, payload = _run(
        "Inspect",
        fixture,
        inject_adapter=inject_adapter,
        inject_console=inject_console,
    )

    assert result.returncode != 0
    assert payload["mode"] == "test"
    assert payload["operational"] is False
    assert payload["success"] is False
    assert payload["non_operational_reasons"] == [reason, "wrapper_root_injected"]
    assert payload["rejection"] == "injected_execution_seam_disabled"


@pytest.mark.parametrize(
    ("command_line", "expected"),
    [
        ('"C:\\Program Files\\ApplyPilot\\applypilot.exe" apply --url https://example.invalid/job', "acquisition"),
        ('C:\\ApplyPilot\\applypilot.exe apply --url https://example.invalid/job', "acquisition"),
        ('"C:\\Python312\\python.exe" -m applypilot apply --url https://example.invalid/job', "acquisition"),
        ('python -bb -m applypilot.fleet.apply_worker_main', "acquisition"),
        ('python --check-hash-based-pycs always -m applypilot.fleet.apply_worker_main', "acquisition"),
        ('python -vv -m applypilot.fleet.apply_worker_main', "acquisition"),
        ('python -vvv -OO -m applypilot.fleet.apply_worker_main', "acquisition"),
        ('"C:\\Python312\\python.exe" "C:\\Program Files\\ApplyPilot\\src\\applypilot\\apply\\launcher.py" --limit 1', "acquisition"),
        ('"C:\\Program Files\\ApplyPilot\\applypilot-workday-onboard.exe" --limit 1', "acquisition"),
        ('"C:\\Program Files\\ApplyPilot\\applypilot-fleet-apply-home.exe" status', "acquisition"),
        ('C:\\ApplyPilot\\applypilot-fleet-apply-home.exe readiness --strict', "acquisition"),
        ('"C:\\Program Files\\ApplyPilot\\applypilot-fleet-linkedin-home.exe" run', "acquisition"),
        ('C:\\ApplyPilot\\applypilot-fleet-linkedin-home.exe status', "acquisition"),
        ('"C:\\Python312\\python.exe" -m applypilot.fleet.apply_home_main run', "acquisition"),
        ('python -m applypilot.fleet.linkedin_home_main status', "acquisition"),
        ('py.exe -3.12 -X utf8 -m applypilot.fleet.apply_worker_main --worker-id m2-0', "acquisition"),
        ('"C:\\Windows\\py.exe" -3 -I -m applypilot apply --url https://example.invalid/job', "acquisition"),
        ('python -W ignore -m applypilot.fleet.linkedin_worker_main --worker-id m2-0', "acquisition"),
        ('python apply_worker_main.py --worker-id m2-0', "acquisition"),
        ('python "apply_worker_main.py" --worker-id m2-0', "acquisition"),
        ('python -- apply_worker_main.py --worker-id m2-0', "acquisition"),
        ('pwsh.exe -File C:\\ApplyPilot\\run-fleet-worker.ps1 -Slot 0', "acquisition"),
        ('pwsh.exe -NoProfile run-fleet-workers.ps1', "acquisition"),
        ('pwsh.exe -NoProfile -Fi run-fleet-workers.ps1', "acquisition"),
        ('pwsh.exe -NoProfile -Fil run-fleet-workers.ps1', "acquisition"),
        ('pwsh.exe -NoExit -NoP run-fleet-workers.ps1', "acquisition"),
        ('pwsh.exe -File run-fleet-workers.ps1 -Count 2', "acquisition"),
        ('pwsh.exe -NoProfile -File "run-fleet-workers.ps1" -Count 2', "acquisition"),
        ('powershell.exe -NoProfile -File "C:\\ApplyPilot\\run-fleet-workers.ps1" -Count 2', "acquisition"),
        ('pwsh.exe -File "C:\\ApplyPilot\\run-tarpon-linkedin.ps1"', "acquisition"),
        ('powershell.exe -File C:\\ApplyPilot\\keepalive-apply.ps1', "acquisition"),
        ('pwsh.exe -File C:\\ApplyPilot\\load-canary-remote.ps1 -Count 1', "acquisition"),
        ('cmd /d /s /c run-fleet-worker.cmd', "acquisition"),
        ('cmd /c call run-fleet-worker.cmd', "acquisition"),
        ('cmd /V:ON /R run-fleet-worker.cmd', "acquisition"),
        ('cmd /c "call run-fleet-worker.cmd"', "acquisition"),
        ('pwsh.exe -Command run-fleet-workers.ps1', "acquisition"),
        ('pwsh.exe -Command "run-fleet-workers.ps1 -Count 2"', "acquisition"),
        ('pwsh.exe -Command "run-fleet-workers.ps1" -Count 2', "acquisition"),
        (
            'pwsh.exe -Command "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
            "acquisition",
        ),
        (
            'pwsh.exe -Command & "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
            "acquisition",
        ),
        ('pwsh.exe -Command "& run-fleet-workers.ps1"', "acquisition"),
        ('pwsh.exe -Command "& \'run-fleet-workers.ps1\'"', "acquisition"),
        ('pwsh.exe -Command "& \\"run-fleet-workers.ps1\\""', "acquisition"),
        (
            'pwsh.exe -EncodedCommand '
            + base64.b64encode("run-fleet-workers.ps1 -Count 2".encode("utf-16le")).decode("ascii"),
            "acquisition",
        ),
        (
            'pwsh.exe -EncodedCommand '
            + base64.b64encode("& 'run-fleet-workers.ps1'".encode("utf-16le")).decode("ascii"),
            "acquisition",
        ),
        (
            'pwsh.exe -EncodedCommand '
            + base64.b64encode('& "run-fleet-workers.ps1"'.encode("utf-16le")).decode("ascii"),
            "acquisition",
        ),
        (
            'pwsh.exe -EncodedCommand '
            + base64.b64encode(
                '& "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"'.encode(
                    "utf-16le"
                )
            ).decode("ascii"),
            "acquisition",
        ),
        ('"C:\\Python312\\python.exe" "C:\\Program Files\\ApplyPilot\\src\\applypilot\\fleet\\workday_rollout_main.py" canary', "acquisition"),
        ('"C:\\Program Files\\ApplyPilot\\applypilot-workday-report.exe" --read-only', "benign"),
        ('"C:\\Program Files\\ApplyPilot\\applypilot-fleet-apply-home-helper.exe" status', "benign"),
        ('python -m applypilot.fleet.apply_home_main_helper run', "benign"),
        ('py.exe -m applypilot.fleet.apply_worker_main_helper --read-only', "benign"),
        ('python -c pass -m applypilot.fleet.apply_worker_main', "ambiguous"),
        ('python -c "print(1)" -m applypilot apply --url https://example.invalid/job', "ambiguous"),
        ('python -- -m applypilot.fleet.apply_worker_main', "benign"),
        ('python -- benign.py applypilot.fleet.apply_worker_main', "benign"),
        ('python -- --unknown=apply_worker_main.py', "ambiguous"),
        (
            'python -c "import runpy; runpy.run_module(\'applypilot.fleet.apply_worker_main\', run_name=\'__main__\')"',
            "ambiguous",
        ),
        (
            'python -c "import runpy; runpy.run_module(\'applypilot.fleet.apply_worker_main_helper\')"',
            "ambiguous",
        ),
        ('python --unknown-option -m applypilot.fleet.apply_worker_main', "ambiguous"),
        ('python --unknown=applypilot.fleet.apply_worker_main', "ambiguous"),
        ('python --unknown=applypilot.fleet.apply_worker_main_helper', "benign"),
        ('python -bbb -m applypilot.fleet.apply_worker_main', "ambiguous"),
        ('python -OOO -m applypilot.fleet.apply_worker_main', "ambiguous"),
        ('python "apply_worker_main.py', "ambiguous"),
        ('pwsh.exe -Command "Write-Output benign" -File run-fleet-workers.ps1', "benign"),
        ('pwsh.exe -Command "run-fleet-workers.ps1; Write-Output done"', "ambiguous"),
        ('pwsh.exe -Command "run-fleet-workers.ps1 && Write-Output done"', "ambiguous"),
        ('pwsh.exe -Command "run-fleet-workers.ps1 || Write-Output done"', "ambiguous"),
        ('pwsh.exe -Command "run-fleet-workers.ps1 &"', "ambiguous"),
        ('pwsh.exe -Command "run-fleet-workers.ps1 | Out-Null"', "ambiguous"),
        ('pwsh.exe -Command "run-fleet-workers.ps1 > output.txt"', "ambiguous"),
        (
            'pwsh.exe -Command "run-fleet-workers.ps1 -Count 2" ; Write-Output done',
            "ambiguous",
        ),
        (
            'pwsh.exe -Command "run-fleet-workers.ps1 -Count 2" && Write-Output done',
            "ambiguous",
        ),
        (
            'pwsh.exe -Command & "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1" && Write-Output done',
            "ambiguous",
        ),
        (
            'pwsh.exe -Command "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1" ; Write-Output done',
            "ambiguous",
        ),
        ('pwsh.exe -Command "run-fleet-workers.ps1.backup"', "benign"),
        ('pwsh.exe -Command "& \'run-fleet-workers.ps1.backup\'"', "benign"),
        (
            'pwsh.exe -Command "& \'run-fleet-workers.ps1.backup\' && Write-Output done"',
            "benign",
        ),
        ('pwsh.exe -Command "Write-Output benign" run-fleet-workers.ps1', "benign"),
        ('pwsh.exe -Command "Write-Output \'run-fleet-workers.ps1\'"', "benign"),
        (
            'pwsh.exe -Command "Write-Output benign" ; Write-Output run-fleet-workers.ps1',
            "ambiguous",
        ),
        (
            'pwsh.exe -Command "run-fleet-workers.ps1.backup" && Write-Output done',
            "benign",
        ),
        (
            'pwsh.exe -Command & "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1.backup"',
            "benign",
        ),
        ('pwsh.exe -Command "Invoke-Expression \'run-fleet-workers.ps1\'"', "ambiguous"),
        ('pwsh.exe -Command "iex \'run-fleet-workers.ps1\'"', "ambiguous"),
        ('pwsh.exe -Command "Start-Process run-fleet-workers.ps1"', "ambiguous"),
        ('pwsh.exe -Command "saps -FilePath run-fleet-workers.ps1"', "ambiguous"),
        (
            'pwsh.exe -Command "Start-Process -FilePath \'C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1\'"',
            "ambiguous",
        ),
        (
            'pwsh.exe -Command Start-Process -FilePath "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
            "ambiguous",
        ),
        (
            'pwsh.exe -Command Start-Process "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
            "ambiguous",
        ),
        (
            'pwsh.exe -Command saps -FilePath "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
            "ambiguous",
        ),
        (
            'pwsh.exe -Command Start-Process -FilePath:"C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
            "ambiguous",
        ),
        (
            'pwsh.exe -Command saps -FilePath:"C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
            "ambiguous",
        ),
        (
            'pwsh.exe -Command Start-Process -FilePath="C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
            "ambiguous",
        ),
        (
            'pwsh.exe -Command saps -FilePath="C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
            "ambiguous",
        ),
        (
            'pwsh.exe -Command "Invoke-Command -ScriptBlock { run-fleet-workers.ps1 }"',
            "ambiguous",
        ),
        (
            'pwsh.exe -Command "Start-Process notepad.exe -ArgumentList \'run-fleet-workers.ps1\'"',
            "benign",
        ),
        (
            'pwsh.exe -Command Start-Process notepad.exe -ArgumentList "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
            "benign",
        ),
        (
            'pwsh.exe -Command Start-Process notepad.exe -ArgumentList:"C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
            "benign",
        ),
        (
            'pwsh.exe -Command Start-Process -FilePath:notepad.exe -ArgumentList:"C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
            "benign",
        ),
        ('pwsh.exe -Command "iex \'run-fleet-workers.ps1.backup\'"', "benign"),
        (
            'pwsh.exe -Command "Start-Process run-fleet-workers.ps1.backup"',
            "benign",
        ),
        (
            'pwsh.exe -Command Start-Process -FilePath "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1.backup"',
            "benign",
        ),
        (
            'pwsh.exe -Command Start-Process -FilePath:"C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1.backup"',
            "benign",
        ),
        (
            'pwsh.exe -Command saps -FilePath="C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1.backup"',
            "benign",
        ),
        (
            'pwsh.exe -EncodedCommand '
            + base64.b64encode(
                "run-fleet-workers.ps1; Write-Output done".encode("utf-16le")
            ).decode("ascii"),
            "ambiguous",
        ),
        (
            'pwsh.exe -EncodedCommand '
            + base64.b64encode(
                "run-fleet-workers.ps1 && Write-Output done".encode("utf-16le")
            ).decode("ascii"),
            "ambiguous",
        ),
        (
            'pwsh.exe -EncodedCommand '
            + base64.b64encode(
                "run-fleet-workers.ps1 || Write-Output done".encode("utf-16le")
            ).decode("ascii"),
            "ambiguous",
        ),
        (
            'pwsh.exe -EncodedCommand '
            + base64.b64encode("run-fleet-workers.ps1 &".encode("utf-16le")).decode("ascii"),
            "ambiguous",
        ),
        (
            'pwsh.exe -EncodedCommand '
            + base64.b64encode("run-fleet-workers.ps1.backup".encode("utf-16le")).decode("ascii"),
            "benign",
        ),
        (
            'pwsh.exe -EncodedCommand '
            + base64.b64encode(
                "& 'run-fleet-workers.ps1.backup' && Write-Output done".encode("utf-16le")
            ).decode("ascii"),
            "benign",
        ),
        (
            'pwsh.exe -EncodedCommand '
            + base64.b64encode(
                "Invoke-Expression 'run-fleet-workers.ps1'".encode("utf-16le")
            ).decode("ascii"),
            "ambiguous",
        ),
        (
            'pwsh.exe -EncodedCommand '
            + base64.b64encode(
                "Start-Process run-fleet-workers.ps1".encode("utf-16le")
            ).decode("ascii"),
            "ambiguous",
        ),
        (
            'pwsh.exe -EncodedCommand '
            + base64.b64encode(
                "Start-Process -FilePath:'C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1'".encode(
                    "utf-16le"
                )
            ).decode("ascii"),
            "ambiguous",
        ),
        (
            'pwsh.exe -EncodedCommand '
            + base64.b64encode(
                "saps -FilePath='C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1'".encode(
                    "utf-16le"
                )
            ).decode("ascii"),
            "ambiguous",
        ),
        (
            'pwsh.exe -EncodedCommand '
            + base64.b64encode(
                "Start-Process -FilePath:'C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1.backup'".encode(
                    "utf-16le"
                )
            ).decode("ascii"),
            "benign",
        ),
        (
            'pwsh.exe -EncodedCommand '
            + base64.b64encode(
                "saps -FilePath='C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1.backup'".encode(
                    "utf-16le"
                )
            ).decode("ascii"),
            "benign",
        ),
        (
            'pwsh.exe -EncodedCommand '
            + base64.b64encode(
                "Start-Process -FilePath:notepad.exe -ArgumentList:'C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1'".encode(
                    "utf-16le"
                )
            ).decode("ascii"),
            "benign",
        ),
        ('pwsh.exe -EncodedCommand not-valid-base64!', "ambiguous"),
        ('pwsh.exe -EncodedCommand YQ==', "ambiguous"),
        ('pwsh.exe -EncodedCommand ANg=', "ambiguous"),
        ('pwsh.exe -- -File run-fleet-workers.ps1', "benign"),
        ('pwsh.exe -Unknown run-fleet-workers.ps1', "ambiguous"),
        ('pwsh.exe -Unknown=run-fleet-workers.ps1', "ambiguous"),
        ('pwsh.exe -Unknown=run-fleet-workers.ps1.backup', "benign"),
        ('pwsh.exe -File C:\\ApplyPilot\\run-fleet-worker-report.ps1', "benign"),
        ('pwsh.exe -File C:\\ApplyPilot\\run-tarpon-linkedin-report.ps1', "benign"),
        ('pwsh.exe -File linkedin-report.ps1', "benign"),
        ('pwsh.exe -File workday-discovery.ps1', "benign"),
        ('cmd /c workday-monitor.cmd', "benign"),
        ('cmd /V:ON /R "call workday-monitor.cmd"', "benign"),
        ('pwsh.exe -File fleet-agent-monitor.ps1', "benign"),
        ('"C:\\Tools\\notapplypilot.exe" apply --url https://example.invalid/job', "benign"),
        ('python -m unrelated.applypilot apply --url https://example.invalid/job', "benign"),
        ('notepad.exe "C:\\Program Files\\ApplyPilot\\src\\applypilot\\fleet\\workday_rollout_main.py"', "benign"),
    ],
)
def test_acquisition_command_line_matching_is_bounded_and_quote_aware(command_line, expected):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command_line],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert payload == {
        "mode": "test",
        "operational": False,
        "classification": expected,
        "matched": expected == "acquisition",
    }


@pytest.mark.parametrize(
    "command",
    [
        r"%APPLYPILOT_WRAPPER% --limit 1",
        r"!APPLYPILOT_WRAPPER! --limit 1",
        r"%APPLYPILOT_WRAPPER --limit 1",
        r"$env:APPLYPILOT_WRAPPER --limit 1",
        r"${env:APPLYPILOT_WRAPPER} --limit 1",
        r"$env:APPLYPILOT_WRAPPER.ps1 --limit 1",
        r"pwsh -File $env:APPLYPILOT_WRAPPER",
        r"cmd /c %APPLYPILOT_WRAPPER%",
        r"cmd /k !APPLYPILOT_WRAPPER!",
    ],
)
def test_environment_selected_executable_basename_is_ambiguous_and_not_expanded(command):
    secret = "expanded-wrapper-secret"
    env = os.environ.copy()
    env["APPLYPILOT_WRAPPER"] = f"run-fleet-workers.ps1-{secret}"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert secret not in result.stdout
    assert command not in result.stdout
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


@pytest.mark.parametrize(
    "command",
    [
        r"%APPLYPILOT_ROOT%\run-fleet-workers.ps1 --limit 1",
        r"!APPLYPILOT_ROOT!\run-fleet-workers.ps1 --limit 1",
        r"$env:APPLYPILOT_ROOT\run-fleet-workers.ps1 --limit 1",
        r"${env:APPLYPILOT_ROOT}\run-fleet-workers.ps1 --limit 1",
        r"pwsh -File $env:APPLYPILOT_ROOT\run-fleet-workers.ps1",
        r"cmd /c %APPLYPILOT_ROOT%\run-fleet-worker.cmd",
    ],
)
def test_environment_reference_in_directory_retains_literal_basename_classification(command):
    secret = "directory-expansion-must-not-occur"
    env = os.environ.copy()
    env["APPLYPILOT_ROOT"] = rf"C:\sensitive\{secret}"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert secret not in result.stdout
    assert command not in result.stdout
    assert json.loads(result.stdout.strip())["classification"] == "acquisition"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("python -mapplypilot.fleet.apply_worker_main", "acquisition"),
        ("python -mapplypilot apply --url https://example.invalid/job", "acquisition"),
        ("python -mapplypilot.fleet.apply_worker_main_helper", "benign"),
        ("python -mapplypilot.fleet.apply_worker_main.backup", "benign"),
        ("cmd /crun-fleet-worker.cmd", "acquisition"),
        ("cmd /krun-fleet-worker.cmd", "acquisition"),
        ("cmd /crun-fleet-worker.cmd.backup", "benign"),
        ("cmd /kworkday-monitor.cmd", "benign"),
    ],
)
def test_compact_launcher_forms_are_bounded(command, expected):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == expected


@pytest.mark.parametrize(
    "command",
    [
        "python3.exe -m applypilot.fleet.apply_worker_main",
        "python3.9.exe -m applypilot.fleet.apply_worker_main",
        "python3.13.exe -m applypilot.fleet.apply_worker_main",
        "python2.7 -m applypilot.fleet.apply_worker_main",
        "pythonw3.exe -m applypilot.fleet.apply_worker_main",
        "pythonw3.13.exe -m applypilot.fleet.apply_worker_main",
        "pyw.exe -m applypilot.fleet.apply_worker_main",
        "pyw.exe -3.13 -m applypilot.fleet.apply_worker_main",
        r'"C:\Python313\python3.13.exe" -m applypilot.fleet.apply_worker_main',
        r"python3.exe C:\ApplyPilot\apply_worker_main.py --worker-id m2-0",
        r"pythonw3.13.exe -- C:\ApplyPilot\linkedin_worker_main.py",
    ],
)
def test_valid_python_launcher_aliases_preserve_acquisition_markers(command):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "acquisition"


@pytest.mark.parametrize(
    "command",
    [
        "cmd /c python3.exe -m applypilot.fleet.apply_worker_main",
        "cmd /k pyw.exe -3.13 -m applypilot.fleet.apply_worker_main",
        'pwsh.exe -Command "python3.13.exe -m applypilot.fleet.apply_worker_main"',
        'powershell.exe -Command "pythonw3.13.exe C:\\ApplyPilot\\apply_worker_main.py"',
    ],
)
def test_python_launcher_aliases_remain_authoritative_through_shell_wrappers(command):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "acquisition"


@pytest.mark.parametrize(
    "command",
    [
        "python3.exe -m http.server 8000",
        "python3.13.exe --version",
        "pythonw3.13.exe C:\\tools\\report.py",
        "pyw.exe -3.13 -m unrelated.worker",
        "cmd /c python3.exe -m unrelated.worker",
        'pwsh.exe -Command "pythonw3.13.exe -m unrelated.worker"',
    ],
)
def test_valid_python_launcher_aliases_without_acquisition_markers_are_benign(command):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "benign"


@pytest.mark.parametrize(
    "launcher",
    [
        "python3x.exe",
        "python3.13-helper.exe",
        "python313.exe",
        "pythonw3x.exe",
        "pythonw3.13.old.exe",
        "pywx.exe",
    ],
)
def test_python_launcher_alias_normalization_rejects_near_match_stems(launcher):
    command = f"{launcher} -m applypilot.fleet.apply_worker_main"
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "benign"


@pytest.mark.parametrize(
    "command",
    [
        r"%PYTHON_LAUNCHER% -m applypilot.fleet.apply_worker_main",
        r"!PYTHON_LAUNCHER! -m applypilot.fleet.apply_worker_main",
        r"$env:PYTHON_LAUNCHER -m applypilot.fleet.apply_worker_main",
        r"${env:PYTHON_LAUNCHER}.exe -m applypilot.fleet.apply_worker_main",
    ],
)
def test_environment_selected_python_launcher_alias_remains_ambiguous(command):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


@pytest.mark.parametrize(
    "command",
    [
        "python3.13t.exe -m applypilot.fleet.apply_worker_main",
        "pythonw3.13t.exe -m applypilot.fleet.linkedin_worker_main",
        r"python3.14t.exe C:\ApplyPilot\apply_worker_main.py",
        "python4.0.exe -m applypilot.fleet.apply_worker_main",
        r"python4.1t.exe C:\ApplyPilot\linkedin_worker_main.py",
        "cmd /c python3.13t.exe -m applypilot.fleet.apply_worker_main",
        "cmd /k python4.0.exe -m applypilot.fleet.linkedin_worker_main",
        'pwsh.exe -Command "pythonw3.13t.exe -m applypilot.fleet.apply_worker_main"',
        'powershell.exe -Command "python4.1t.exe C:\\ApplyPilot\\apply_worker_main.py"',
        "py.exe -3.13t -m applypilot.fleet.apply_worker_main",
        r"pyw.exe -3.13t C:\ApplyPilot\linkedin_worker_main.py",
        "cmd /c pyw.exe -3.13t -m applypilot.fleet.apply_worker_main",
        'pwsh.exe -Command "py.exe -3.13t -m applypilot.fleet.linkedin_worker_main"',
    ],
)
def test_exact_acquisition_payloads_are_authoritative_after_safe_parsing(command):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "acquisition"


@pytest.mark.parametrize(
    "command",
    [
        "python3.13t.exe -m http.server",
        "pythonw3.13t.exe C:\\tools\\report.py",
        "py.exe -3.13t -m unrelated.worker",
        "pyw.exe -3.13t C:\\tools\\report.py",
        "python4.0.exe -m unrelated.worker",
        r"future-python.exe C:\tools\report.py",
        "cmd /c python3.13t.exe -m unrelated.worker",
        'pwsh.exe -Command "pythonw3.13t.exe C:\\tools\\report.py"',
        "python3.13t.exe -m applypilot.fleet.apply_worker_main_extra",
        r"pythonw3.13t.exe C:\ApplyPilot\apply_worker_main.py.bak",
        "py.exe -3.13t -m applypilot.fleet.linkedin_worker_main_extra",
    ],
)
def test_free_threaded_and_future_launchers_without_exact_payloads_are_benign(command):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "benign"


@pytest.mark.parametrize(
    "command",
    [
        r"%PYTHON313T% -m applypilot.fleet.apply_worker_main",
        r"cmd /c %PYTHON313T% -m applypilot.fleet.apply_worker_main",
        r"$env:PYTHON313T -m applypilot.fleet.apply_worker_main",
        'pwsh.exe -Command "& $env:PYTHON313T -m applypilot.fleet.apply_worker_main"',
        'python3.13t.exe -m "applypilot.fleet.apply_worker_main',
    ],
)
def test_free_threaded_environment_and_malformed_targets_remain_ambiguous(command):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


@pytest.mark.parametrize(
    "command",
    [
        "cmd /rrun-fleet-worker.cmd",
        "cmd /d/crun-fleet-worker.cmd",
        "cmd /q/krun-fleet-worker.cmd",
        "cmd /v:on/crun-fleet-worker.cmd",
        "cmd /d/s/rrun-fleet-worker.cmd",
        "cmd /d /s /c run-fleet-worker.cmd",
        "cmd /q /k @call run-fleet-worker.cmd",
        'cmd /d/c"C:\\Program Files\\ApplyPilot\\run-fleet-worker.cmd"',
    ],
)
def test_cmd_switch_grammar_finds_terminal_mode_and_exact_wrapper(command):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "acquisition"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cmd /d/z/crun-fleet-worker.cmd", "ambiguous"),
        ("cmd /v:maybe/crun-fleet-worker.cmd", "ambiguous"),
        ("cmd /d/c(run-fleet-worker.cmd)", "ambiguous"),
        ("cmd /d/crun-fleet-worker.cmd.backup", "benign"),
        ("cmd /v:on/cworkday-monitor.cmd", "benign"),
        ("cmd /c %APPLYPILOT_COMMAND%", "ambiguous"),
        ("cmd /v:on/k!APPLYPILOT_COMMAND!", "ambiguous"),
        ("cmd /c echo %APPLYPILOT_COMMAND%", "ambiguous"),
    ],
)
def test_cmd_switch_grammar_is_fail_closed_and_bounded(command, expected):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cmd /c echo first & %TARGET%", "ambiguous"),
        ("cmd /v:on /c echo first & !TARGET!", "ambiguous"),
        ("cmd /c echo first & run-fleet-^worker.cmd", "ambiguous"),
        ("cmd /z /c %TARGET%", "ambiguous"),
        ("cmd /z /c run-fleet-worker.cmd", "ambiguous"),
        ("cmd /x /y /t:0A /d /c run-fleet-worker.cmd", "acquisition"),
        ("cmd /x/y/t:0a/crun-fleet-worker.cmd", "acquisition"),
        ("cmd /c echo first & run-fleet-^worker-report.cmd", "benign"),
        ("cmd /t:GG /c run-fleet-worker.cmd", "ambiguous"),
    ],
)
def test_cmd_whole_payload_and_canonical_switch_invariants(command, expected):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cmd /c cmd /c run-fleet-worker.cmd", "acquisition"),
        ("cmd /c cmd.exe /d /c run-fleet-worker.cmd", "acquisition"),
        ("cmd /c cmd /c run-fleet-^^worker.cmd", "ambiguous"),
        ("cmd /c cmd /c cmd /c run-fleet-^^^^worker.cmd", "ambiguous"),
        ("cmd /c cmd /c run-fleet-worker-report.cmd", "benign"),
        ("cmd /c cmd /c run-fleet-^^worker-report.cmd", "benign"),
        ("cmd /c cmd /c %TARGET%", "ambiguous"),
        (
            "cmd /c cmd /c cmd /c cmd /c cmd /c cmd /c run-fleet-worker-report.cmd",
            "ambiguous",
        ),
    ],
)
def test_nested_cmd_interpreters_are_bounded_and_apply_one_caret_layer(command, expected):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == expected


@pytest.mark.parametrize("command", ["python -c pass", "python -cpass", "python -c print(1)"])
def test_python_code_execution_is_always_opaque_and_ambiguous(command):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


@pytest.mark.parametrize(
    "command",
    [
        "python",
        "python.exe",
        "py",
        "python -",
        "python -I -",
        "pwsh",
        "powershell.exe",
        "pwsh -",
        "powershell.exe -",
        "pwsh -SSHServerMode",
        "powershell.exe -sshs",
        "pwsh -File -",
        "pwsh -F -",
        "pwsh -Command -",
        "pwsh -C -",
        "cmd",
        "cmd.exe /d",
    ],
)
def test_interpreter_stdin_execution_modes_are_ambiguous(command):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("pwsh -nopr -ex Bypass -c run-fleet-workers.ps1", "acquisition"),
        ("pwsh -nol -noe -noni -c run-fleet-workers.ps1", "acquisition"),
        ("pwsh -inp Text -out Text -c run-fleet-workers.ps1", "acquisition"),
        ("pwsh -cwa run-fleet-workers.ps1", "acquisition"),
        ("pwsh -CommandWithArgs run-fleet-workers.ps1", "acquisition"),
        ("pwsh -Unknown benign -Command -", "ambiguous"),
        ("pwsh -Unknown benign -File -", "ambiguous"),
        ("pwsh -Unknown benign -File run-fleet-workers.ps1", "ambiguous"),
        ("pwsh -Unknown benign -Command Write-Output safe", "benign"),
        ("pwsh -executor benign.ps1", "benign"),
    ],
)
def test_powershell_option_metadata_and_late_terminal_invariants(command, expected):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == expected


@pytest.mark.parametrize("mode", ["c", "k"])
def test_compact_quoted_cmd_preserves_single_wrapper_token(mode):
    command = f'cmd /{mode}"C:\\Program Files\\ApplyPilot\\run-fleet-worker.cmd"'

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "acquisition"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            'cmd /c"C:\\Program Files\\ApplyPilot\\run-fleet-worker.cmd.backup"',
            "benign",
        ),
        (
            'cmd /k"C:\\Program Files\\ApplyPilot\\run-fleet-worker-report.cmd"',
            "benign",
        ),
        (
            'cmd /c"C:\\Program Files\\ApplyPilot\\run-fleet-worker.cmd & echo done"',
            "ambiguous",
        ),
        (
            'cmd /k"C:\\Program Files\\ApplyPilot\\run-fleet-worker.cmd" && echo done',
            "ambiguous",
        ),
    ],
)
def test_compact_quoted_cmd_is_bounded_and_compound_aware(command, expected):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == expected


def test_compact_quoted_cmd_probe_redacts_compound_payload():
    marker = "SECRET_COMPACT_CMD_MARKER"
    command = (
        'cmd /c"C:\\Program Files\\ApplyPilot\\run-fleet-worker.cmd '
        f'& echo {marker}"'
    )

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker not in result.stdout
    assert marker not in result.stderr
    assert command not in result.stdout
    assert command not in result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


@pytest.mark.parametrize("mode", ["c", "k"])
@pytest.mark.parametrize(
    "command_tail",
    [
        "@ call run-fleet-worker.cmd",
        "@call run-fleet-worker.cmd",
        "@run-fleet-worker.cmd",
        '@ call "C:\\Program Files\\ApplyPilot\\run-fleet-worker.cmd"',
        '@"C:\\Program Files\\ApplyPilot\\run-fleet-worker.cmd"',
    ],
)
def test_cmd_post_mode_optional_echo_and_call_prefixes_are_acquisition(mode, command_tail):
    command = f"cmd /{mode} {command_tail}"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "acquisition"


@pytest.mark.parametrize("mode", ["c", "k"])
@pytest.mark.parametrize(
    "attached_tail",
    [
        "@call run-fleet-worker.cmd",
        "@run-fleet-worker.cmd",
        '@call "C:\\Program Files\\ApplyPilot\\run-fleet-worker.cmd"',
        '@"C:\\Program Files\\ApplyPilot\\run-fleet-worker.cmd"',
    ],
)
def test_compact_cmd_optional_echo_and_call_prefixes_are_acquisition(mode, attached_tail):
    command = f"cmd /{mode}{attached_tail}"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "acquisition"


@pytest.mark.parametrize("mode", ["c", "k"])
@pytest.mark.parametrize(
    "syntax",
    [
        "(run-fleet-worker.cmd)",
        "^run-fleet-worker.cmd",
        "run-fleet-worker.cmd & echo done",
        "run-fleet-worker.cmd && echo done",
        "run-fleet-worker.cmd || echo done",
        "run-fleet-worker.cmd | more",
        "run-fleet-worker.cmd > output.txt",
        "start /wait run-fleet-worker.cmd",
        "echo run-fleet-worker.cmd",
    ],
)
def test_cmd_unfamiliar_exact_wrapper_syntax_is_ambiguous(mode, syntax):
    command = f'cmd /{mode}"{syntax}"'

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c"(run-fleet-worker.cmd.backup)"',
        'cmd /k"^run-fleet-worker-report.cmd"',
        'cmd /c"echo run-fleet-worker.cmd.backup"',
        'cmd /k @call run-fleet-worker.cmd.backup',
    ],
)
def test_cmd_post_mode_near_matches_remain_benign(command):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "benign"


def test_cmd_unfamiliar_syntax_probe_redacts_payload():
    marker = "SECRET_CMD_TRI_STATE_MARKER"
    command = f'cmd /c"(run-fleet-worker.cmd) & echo {marker}"'

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker not in result.stdout
    assert marker not in result.stderr
    assert command not in result.stdout
    assert command not in result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            "pwsh.exe -Command Start-Process -FilePath:'$env:APPLYPILOT_TARGET'",
            "benign",
        ),
        (
            'pwsh.exe -Command Start-Process -FilePath:"$env:APPLYPILOT_TARGET"',
            "ambiguous",
        ),
        (
            "pwsh.exe -Command Start-Process -FilePath:$env:APPLYPILOT_TARGET",
            "ambiguous",
        ),
    ],
)
def test_attached_powershell_filepath_preserves_quote_semantics(command, expected):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == expected


@pytest.mark.parametrize(
    ("decoded", "expected"),
    [
        ("Start-Process -FilePath:'$env:APPLYPILOT_TARGET'", "benign"),
        ('Start-Process -FilePath:"$env:APPLYPILOT_TARGET"', "ambiguous"),
        ("Start-Process -FilePath:$env:APPLYPILOT_TARGET", "ambiguous"),
    ],
)
def test_encoded_powershell_filepath_quote_semantics_agree(decoded, expected):
    encoded = base64.b64encode(decoded.encode("utf-16le")).decode("ascii")
    command = f"pwsh.exe -enc {encoded}"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert decoded not in result.stdout
    assert decoded not in result.stderr
    assert encoded not in result.stdout
    assert encoded not in result.stderr
    assert json.loads(result.stdout.strip())["classification"] == expected


@pytest.mark.parametrize(
    "option",
    [f"-{('encodedcommand')[:length]}" for length in range(1, len("encodedcommand") + 1)]
    + ["-ec"],
)
def test_encoded_command_unique_abbreviations_decode_and_classify(option):
    decoded = "run-fleet-workers.ps1 -Count 2"
    encoded = base64.b64encode(decoded.encode("utf-16le")).decode("ascii")
    command = f"pwsh.exe {option} {encoded}"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert decoded not in result.stdout
    assert decoded not in result.stderr
    assert encoded not in result.stdout
    assert encoded not in result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "acquisition"


@pytest.mark.parametrize(
    "option",
    [f"-{('command')[:length]}" for length in range(1, len("command") + 1)],
)
def test_command_unique_abbreviations_classify_complete_payload(option):
    command = f"pwsh.exe {option} run-fleet-workers.ps1 -Count 2"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "acquisition"


@pytest.mark.parametrize(
    "option",
    ["-configurationname", "-executionpolicy", "-connect", "-encodedoutput"],
)
def test_powershell_unrelated_options_are_not_bound_as_command_prefixes(option):
    command = f"pwsh.exe {option} run-fleet-workers.ps1"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] != "acquisition"


def test_encoded_command_abbreviation_redacts_ambiguous_payload():
    marker = "SECRET_ENCODED_ABBREVIATION_MARKER"
    decoded = f"run-fleet-workers.ps1; Write-Output {marker}"
    encoded = base64.b64encode(decoded.encode("utf-16le")).decode("ascii")
    command = f"pwsh.exe -en {encoded}"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker not in result.stdout
    assert marker not in result.stderr
    assert encoded not in result.stdout
    assert encoded not in result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


@pytest.mark.parametrize(
    "payload",
    [
        "& ('run-fleet-' + 'workers.ps1')",
        "$x = 'run-fleet-workers.ps1'; & $x",
        ". ('run-fleet-' + 'workers.ps1')",
        "Start-Process -FilePath $env:APPLYPILOT_TARGET",
        "Start-Process -FilePath:$env:APPLYPILOT_TARGET",
        "Start-Process -FilePath=$env:APPLYPILOT_TARGET",
        "Start-Process ($env:APPLYPILOT_TARGET)",
        "Invoke-Expression $env:APPLYPILOT_COMMAND",
        "Write-Output $(& $env:APPLYPILOT_TARGET)",
        "Write-Output @(& $env:APPLYPILOT_TARGET)",
        "Write-Output $(Start-Process $env:APPLYPILOT_TARGET)",
        "& $env:APPLYPILOT_TARGET",
    ],
)
def test_dynamic_powershell_executable_targets_are_ambiguous(payload):
    command = f"pwsh.exe -Command {payload}"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("& 'notepad.exe'", "benign"),
        (". 'benign-profile.ps1'", "benign"),
        ("Start-Process -FilePath notepad.exe", "benign"),
        ("Invoke-Expression 'Write-Output benign'", "benign"),
        ("Write-Output $env:APPLYPILOT_TARGET", "benign"),
        (
            '& "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
            "acquisition",
        ),
    ],
)
def test_dynamic_target_guard_preserves_static_commands_and_arguments(payload, expected):
    command = f"pwsh.exe -Command {payload}"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == expected


@pytest.mark.parametrize(
    "decoded",
    [
        "& ('run-fleet-' + 'workers.ps1')",
        "$x = 'run-fleet-workers.ps1'; & $x",
        ". ('run-fleet-' + 'workers.ps1')",
        "Start-Process -FilePath $env:APPLYPILOT_TARGET",
        "Invoke-Expression $env:APPLYPILOT_COMMAND",
        "Write-Output $(& $env:APPLYPILOT_TARGET)",
        "Write-Output @(. $env:APPLYPILOT_TARGET)",
    ],
)
def test_encoded_dynamic_powershell_targets_are_ambiguous_and_redacted(decoded):
    encoded = base64.b64encode(decoded.encode("utf-16le")).decode("ascii")
    command = f"pwsh.exe -enc {encoded}"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert decoded not in result.stdout
    assert decoded not in result.stderr
    assert encoded not in result.stdout
    assert encoded not in result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


@pytest.mark.parametrize("operator", [";", "&&", "|"])
def test_split_command_composition_with_acquisition_indicator_is_ambiguous(operator):
    command = (
        "pwsh.exe -Command Start-Process -FilePath:notepad.exe "
        f'{operator} & "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"'
    )

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            'pwsh.exe -Command Start-Process -FilePath:notepad.exe -ArgumentList "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
            "benign",
        ),
        (
            'pwsh.exe -Command Write-Output "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
            "benign",
        ),
        (
            'pwsh.exe -Command Start-Process -FilePath:notepad.exe ; & "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1.backup"',
            "benign",
        ),
        (
            'pwsh.exe -Command & "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
            "acquisition",
        ),
        (
            'pwsh.exe -Command "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
            "acquisition",
        ),
    ],
)
def test_split_command_composition_guard_preserves_bounded_simple_cases(command, expected):
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == expected


@pytest.mark.parametrize("operator", [";", "&&", "|"])
def test_encoded_command_composition_with_acquisition_indicator_is_ambiguous(operator):
    decoded = (
        "Start-Process -FilePath:notepad.exe "
        f'{operator} & "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"'
    )
    encoded = base64.b64encode(decoded.encode("utf-16le")).decode("ascii")
    command = f"pwsh.exe -EncodedCommand {encoded}"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert decoded not in result.stdout
    assert decoded not in result.stderr
    assert encoded not in result.stdout
    assert encoded not in result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


def test_encoded_command_composition_near_match_remains_benign():
    decoded = (
        "Start-Process -FilePath:notepad.exe ; & "
        '"C:\\Program Files\\Neutral\\run-fleet-workers.ps1.backup"'
    )
    encoded = base64.b64encode(decoded.encode("utf-16le")).decode("ascii")
    command = f"pwsh.exe -EncodedCommand {encoded}"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert decoded not in result.stdout
    assert decoded not in result.stderr
    assert encoded not in result.stdout
    assert encoded not in result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "benign"


def test_split_command_composition_probe_redacts_all_payload_tokens():
    marker = "SECRET_SPLIT_COMPOSITION_MARKER"
    command = (
        "pwsh.exe -Command Start-Process -FilePath:notepad.exe ; & "
        f'"C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1" "{marker}"'
    )

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker not in result.stdout
    assert marker not in result.stderr
    assert command not in result.stdout
    assert command not in result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


@pytest.mark.parametrize(
    "payload",
    [
        '. "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
        'Write-Output before ; . "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
        'Write-Output $(& "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1")',
        "Write-Output $(python -m applypilot.fleet.apply_worker_main)",
        'Write-Output before $(& "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1") after',
        "Write-Output before $(python -m applypilot.fleet.apply_worker_main) after",
    ],
)
def test_plain_dot_and_compact_subexpression_acquisition_is_ambiguous(payload):
    command = f"pwsh.exe -Command {payload}"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            '& "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1"',
            "acquisition",
        ),
        (
            "Write-Output 'literal $(' 'run-fleet-workers.ps1'",
            "benign",
        ),
        (
            'Write-Output $(& "C:\\Program Files\\Neutral\\run-fleet-workers.ps1.backup")',
            "benign",
        ),
        (
            "Write-Output $(python -m applypilot.fleet.apply_worker_main_helper)",
            "benign",
        ),
    ],
)
def test_plain_subexpression_guard_preserves_simple_and_bounded_cases(payload, expected):
    command = f"pwsh.exe -Command {payload}"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == expected


@pytest.mark.parametrize(
    ("decoded", "expected"),
    [
        (
            ". 'C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1'",
            "ambiguous",
        ),
        (
            "Write-Output before; . 'C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1'",
            "ambiguous",
        ),
        (
            "Write-Output $(& 'C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1')",
            "ambiguous",
        ),
        (
            "Write-Output $(python -m applypilot.fleet.apply_worker_main)",
            "ambiguous",
        ),
        ("Write-Output 'literal $(' 'run-fleet-workers.ps1'", "benign"),
        (
            "Write-Output $(python -m applypilot.fleet.apply_worker_main_helper)",
            "benign",
        ),
    ],
)
def test_encoded_dot_and_subexpression_classification_is_bounded(decoded, expected):
    encoded = base64.b64encode(decoded.encode("utf-16le")).decode("ascii")
    command = f"pwsh.exe -EncodedCommand {encoded}"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert decoded not in result.stdout
    assert decoded not in result.stderr
    assert encoded not in result.stdout
    assert encoded not in result.stderr
    assert json.loads(result.stdout.strip())["classification"] == expected


def test_compact_subexpression_probe_redacts_payload():
    marker = "SECRET_COMPACT_SUBEXPRESSION_MARKER"
    command = (
        "pwsh.exe -Command Write-Output $(& "
        f'"C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1" "{marker}")'
    )

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker not in result.stdout
    assert marker not in result.stderr
    assert command not in result.stdout
    assert command not in result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


@pytest.mark.parametrize(
    "payload",
    [
        'Write-Output @(& "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1")',
        'Write-Output @(. "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1")',
        'Write-Output before @(& "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1") after',
        'Write-Output before @(. "C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1") after',
    ],
)
def test_plain_array_subexpression_acquisition_is_ambiguous(payload):
    command = f"pwsh.exe -Command {payload}"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


@pytest.mark.parametrize(
    "payload",
    [
        "Write-Output 'literal @(' 'run-fleet-workers.ps1'",
        'Write-Output @(& "C:\\Program Files\\Neutral\\run-fleet-workers.ps1.backup")',
        "Write-Output @(python -m applypilot.fleet.apply_worker_main_helper)",
    ],
)
def test_plain_array_subexpression_guard_preserves_literals_and_near_matches(payload):
    command = f"pwsh.exe -Command {payload}"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "benign"


@pytest.mark.parametrize(
    ("decoded", "expected"),
    [
        (
            "Write-Output @(& 'C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1')",
            "ambiguous",
        ),
        (
            "Write-Output @(. 'C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1')",
            "ambiguous",
        ),
        (
            "Write-Output before @(& 'C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1') after",
            "ambiguous",
        ),
        ("Write-Output 'literal @(' 'run-fleet-workers.ps1'", "benign"),
        (
            "Write-Output @(& 'C:\\Program Files\\Neutral\\run-fleet-workers.ps1.backup')",
            "benign",
        ),
        (
            "Write-Output @(python -m applypilot.fleet.apply_worker_main_helper)",
            "benign",
        ),
    ],
)
def test_encoded_array_subexpression_classification_is_bounded(decoded, expected):
    encoded = base64.b64encode(decoded.encode("utf-16le")).decode("ascii")
    command = f"pwsh.exe -EncodedCommand {encoded}"

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert decoded not in result.stdout
    assert decoded not in result.stderr
    assert encoded not in result.stdout
    assert encoded not in result.stderr
    assert json.loads(result.stdout.strip())["classification"] == expected


def test_array_subexpression_probe_redacts_payload():
    marker = "SECRET_ARRAY_SUBEXPRESSION_MARKER"
    command = (
        "pwsh.exe -Command Write-Output @(& "
        f'"C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1" "{marker}")'
    )

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker not in result.stdout
    assert marker not in result.stderr
    assert command not in result.stdout
    assert command not in result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


@pytest.mark.parametrize("operator", [";", "&&", "||", "|", ">"])
def test_encoded_command_probe_never_emits_raw_or_decoded_payload(operator):
    marker = f"SECRET_PAYLOAD_MARKER_{operator}"
    decoded = f"run-fleet-workers.ps1 {operator} Write-Output {marker}"
    encoded = base64.b64encode(decoded.encode("utf-16le")).decode("ascii")

    result = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-File",
            str(SCRIPT),
            "-CommandLineProbe",
            f"pwsh -EncodedCommand {encoded}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker not in result.stdout
    assert marker not in result.stderr
    assert encoded not in result.stdout
    assert encoded not in result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


def test_plain_command_probe_never_emits_raw_payload():
    marker = "SECRET_PLAIN_COMMAND_MARKER"
    command = (
        'pwsh -Command "run-fleet-workers.ps1 -Count 2" '
        f"&& Write-Output {marker}"
    )

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker not in result.stdout
    assert marker not in result.stderr
    assert command not in result.stdout
    assert command not in result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


def test_indirect_command_probe_never_emits_raw_payload():
    marker = "SECRET_INDIRECT_COMMAND_MARKER"
    command = (
        "pwsh -Command \"Invoke-Expression "
        f"'run-fleet-workers.ps1; Write-Output {marker}'\""
    )

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker not in result.stdout
    assert marker not in result.stderr
    assert command not in result.stdout
    assert command not in result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


def test_split_start_process_probe_never_emits_raw_payload():
    marker = "SECRET_SPLIT_START_PROCESS_MARKER"
    command = (
        'pwsh -Command Start-Process -FilePath '
        '"C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1" '
        f'-ArgumentList "{marker}"'
    )

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker not in result.stdout
    assert marker not in result.stderr
    assert command not in result.stdout
    assert command not in result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


def test_attached_start_process_probe_never_emits_raw_payload():
    marker = "SECRET_ATTACHED_START_PROCESS_MARKER"
    command = (
        'pwsh -Command Start-Process '
        '-FilePath:"C:\\Program Files\\ApplyPilot\\run-fleet-workers.ps1" '
        f'-ArgumentList:"{marker}"'
    )

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-CommandLineProbe", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker not in result.stdout
    assert marker not in result.stderr
    assert command not in result.stdout
    assert command not in result.stderr
    assert json.loads(result.stdout.strip())["classification"] == "ambiguous"


def test_real_adapters_execute_in_safe_operational_inspect_mode():
    env = os.environ.copy()
    env.pop("FLEET_PG_DSN", None)
    env.pop("APPLYPILOT_FLEET_DSN", None)
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(SCRIPT), "-Inspect"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["mode"] == "operational"
    assert payload["operation"] == "inspect"
    assert payload["operational"] is True
    assert payload["evidence_deleted"] is False
    assert result.returncode == (0 if payload["success"] else 1)
    if not payload["success"]:
        assert (
            payload["unresolved_targets"]
            or payload["enumeration_failures"]
            or not payload["after_rejection_probe"]["verified"]
        )


def test_applypilot_import_is_pinned_to_this_checkout():
    import applypilot

    imported = Path(applypilot.__file__).resolve()
    assert Path(sys.path[0]).resolve() == SRC.resolve()
    assert imported.is_relative_to(SRC.resolve())


@pytest.mark.parametrize("mode", ["Inspect", "Contain"])
def test_hostile_adapter_is_rejected_before_execution(mode, tmp_path):
    fixture = _fixture(tmp_path)
    sentinel = tmp_path / "adapter-executed.txt"
    fixture["adapter"].write_text(
        "\n".join(
            [
                "$operational = $true",
                "$nonOperationalReasons = @()",
                "Set-Variable -Scope Script -Name AdapterSeamInjected -Value $false -Force",
                f"Set-Content -LiteralPath '{sentinel}' -Value 'executed'",
                "Write-Output 'HOSTILE_ADAPTER_EXECUTED'",
            ]
        ),
        encoding="utf-8",
    )

    result, payload = _run(mode, fixture, inject_console=False)

    assert result.returncode != 0
    assert not sentinel.exists()
    assert "HOSTILE_ADAPTER_EXECUTED" not in result.stdout
    assert payload == {
        "schema_version": 3,
        "mode": "test",
        "operation": mode.lower(),
        "operational": False,
        "non_operational_reasons": ["adapter_injected", "wrapper_root_injected"],
        "success": False,
        "rejection": "injected_execution_seam_disabled",
        "evidence_deleted": False,
    }


def test_containment_script_parses_as_powershell():
    quoted_path = str(SCRIPT).replace("'", "''")
    command = (
        "$errors = $null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile("
        f"(Resolve-Path '{quoted_path}'), [ref]$null, [ref]$errors); "
        "if ($errors.Count) { $errors | ForEach-Object { Write-Error $_ }; exit 1 }"
    )
    result = subprocess.run(["pwsh", "-NoProfile", "-Command", command], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_script_has_no_destructive_evidence_deletion_and_matches_installed_workday_processes():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "Remove-Item" not in text
    assert "DROP TABLE" not in text.upper()
    assert "DELETE FROM" not in text.upper()
    assert "applypilot-workday-rollout" in text
    assert "applypilot-workday-onboard" in text
    assert "Get-ChildItem -LiteralPath $root -File -ErrorAction Stop" in text
    assert "-Recurse" not in text


def test_evidence_publication_uses_delete_on_close_staging_and_verified_final_copy():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "FileAttributeTemporary = 0x00000100" in text
    assert "FileFlagDeleteOnClose = 0x04000000" in text
    assert "GenericRead | GenericWrite | Delete" in text
    assert "CreateNew" in text
    assert "public static FileStream PublishEvidence(string finalPath, byte[] content)" in text
    assert "stream.Flush(true)" in text
    assert "CryptographicOperations.FixedTimeEquals(copied, content)" in text
    assert "CreateHardLinkW" not in text
    assert "FileAttributeTemporary | FileFlagDeleteOnClose" in text
    assert "[guid]::NewGuid().ToString('N')" in text
    assert ".applypilot-emergency-containment-stage-" in text
    assert "SetFileInformationByHandle" in text
    assert "SafeFileHandle openedHandle = stream != null ? stream.SafeFileHandle : handle" in text
    assert "DeleteOpenedEvidence(openedHandle)" in text
    assert "File.Delete(extendedFinalPath)" not in text
    assert "MoveFile" not in text
    assert "ReplaceFile" not in text
    assert "File.Move" not in text


def test_control_database_failure_returns_stop_not_keep(monkeypatch, capsys):
    from applypilot.apply import pgqueue

    def fail_connect(_dsn):
        raise RuntimeError("postgresql://user:secret@example.invalid/control")

    monkeypatch.setattr(pgqueue, "connect", fail_connect)
    monkeypatch.setattr(sys, "argv", ["fleet-agent-query.py", "m2"])
    runpy.run_path(str(ROOT / "fleet-agent-query.py"), run_name="__main__")
    captured = capsys.readouterr()
    assert captured.out.strip() == "STOP|||"
    assert "secret" not in captured.err
    assert "postgresql://" not in captured.err


def test_healthy_mapped_control_database_returns_desired_worker_contract(monkeypatch, capsys):
    from applypilot.apply import pgqueue
    from fleet_agent_env import LIBPQ_SESSION_ENV_VARS

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def execute(self, *_args): return None
        def fetchone(self):
            return {"state": {
                "machine_owner": "m2", "desired_workers": 8, "desired_agent": "codex",
                "desired_model": "", "generation": 39, "admission_allowed": True,
            }}

    class Connection:
        def cursor(self): return Cursor()
        def rollback(self): return None

    monkeypatch.setattr(pgqueue, "connect", lambda _dsn: Connection())
    monkeypatch.setattr(sys, "argv", ["fleet-agent-query.py", "m2"])
    for variable in LIBPQ_SESSION_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("FLEET_PG_DSN", "fleet-test-dsn")
    monkeypatch.delenv("APPLYPILOT_FLEET_DSN", raising=False)
    runpy.run_path(str(ROOT / "fleet-agent-query.py"), run_name="__main__")
    assert capsys.readouterr().out.strip() == "8|codex||39"


def test_worker_admission_denies_unenrolled_worker_without_mutation():
    from applypilot.fleet import emergency_admission

    class Cursor:
        def __init__(self): self.statements = []
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def execute(self, statement, params=None): self.statements.append((statement, params))
        def fetchone(self): return {"snapshot": None}

    class Connection:
        def __init__(self): self.cursor_value = Cursor()
        def cursor(self): return self.cursor_value

    conn = Connection()
    result = emergency_admission.worker_admission(conn, machine_label="m2", machine_owner="m2", worker_id="m2-1")
    assert result.decision.value == "deny"
    assert "does not match worker identity" in result.reason
    assert all(
        not statement.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"))
        for statement, _params in conn.cursor_value.statements
    )


def test_worker_admission_denies_stale_desired_state():
    from applypilot.fleet import emergency_admission

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def execute(self, *_args): return None
        def fetchone(self):
            return {"snapshot": {
                "worker_id": "m2-1", "machine_owner": "m2", "validated": True,
                "revoked_at": None, "desired_workers": 1, "generation": 7,
                "desired_updated_at": datetime.now(timezone.utc) - timedelta(minutes=6),
                "admission_allowed": False, "admission_reason": "desired_state_stale",
            }}

    class Connection:
        def cursor(self): return Cursor()

    result = emergency_admission.worker_admission(Connection(), machine_label="m2", machine_owner="m2", worker_id="m2-1")
    assert result.decision.value == "deny"
    assert "stale" in result.reason


def test_worker_tick_denial_stops_before_worker_loop_access():
    from applypilot.fleet import apply_worker_main

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def execute(self, statement, params=None):
            assert "fleet_worker_admission_snapshot" in statement
        def fetchone(self):
            return {"snapshot": {"contract": "apply", "admission_allowed": False,
                                  "admission_reason": "ats_stopped"}}

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def cursor(self): return Cursor()
        def rollback(self): return None

    class Loop:
        def run_once(self): raise AssertionError("denied worker reached queue lease")

    result = apply_worker_main.run_apply(lambda: Connection(), Loop(), max_iterations=1, idle_sleep=0)
    assert result == {"applied": 0, "halted": 1, "idle": 0, "error": 0}


def test_admission_decision_type_has_only_allow_and_deny():
    from applypilot.fleet.emergency_admission import AdmissionDecision

    assert {decision.value for decision in AdmissionDecision} == {"allow", "deny"}


def test_status_and_readiness_do_not_run_schema_ddl(fleet_db, monkeypatch, capsys):
    from applypilot.fleet import apply_home_main, schema

    monkeypatch.setattr(schema, "ensure_schema_v3", lambda *_a, **_k: pytest.fail("status/readiness attempted schema DDL"))
    assert apply_home_main.main(["--dsn", fleet_db, "status"]) == 0
    assert apply_home_main.main(["--dsn", fleet_db, "readiness", "--strict"]) == 2
    assert "emergency_acquisition_hold" in capsys.readouterr().out
