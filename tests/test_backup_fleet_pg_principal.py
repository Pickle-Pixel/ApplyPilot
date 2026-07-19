from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "backup-fleet-pg.ps1"
MATRIX = REPO / "config" / "brain-command-authority-matrix.runtime.json"
RAILWAY_SCRIPT = REPO / "scripts" / "backup-railway-postgres.ps1"


def _script() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_backup_requires_a_separated_protected_explicit_channel() -> None:
    text = _script()
    assert '[Parameter(Mandatory = $true)][string]$PgHost' in text
    assert '[Parameter(Mandatory = $true)][string]$PgPassFile' in text
    assert "PgUser -cne 'fleet_backup'" in text
    assert '[string]$PgUser = "postgres"' not in text
    assert 'Join-Path $env:APPDATA "postgresql\\pgpass.conf"' not in text
    assert "Assert-ProtectedCredentialFile" in text
    assert "$connectionArgs" in text
    assert "--no-password" in text
    assert "-U $PgUser" not in text
    assert "PGPASSWORD" in text and "Remove-Item" in text


def test_registration_binds_channel_without_persisting_a_secret() -> None:
    text = _script()
    action_start = text.index("$arguments =")
    action_end = text.index("New-ScheduledTaskTrigger", action_start)
    action = text[action_start:action_end]
    for token in (
        "-BackupDir",
        "-PgHost",
        "-PgPort",
        "-PgDb",
        "-PgUser",
        "-PgPassFile",
        "-ExpectedPsqlSha256",
        "-ExpectedPgRestoreSha256",
    ):
        assert token in action
    assert "PGPASSWORD" not in action
    assert "password=" not in action.casefold()
    assert "-PgDsn" not in action


def test_backup_preflight_rejects_elevated_or_write_capable_login() -> None:
    text = _script()
    for token in (
        "rolsuper",
        "rolcreatedb",
        "rolcreaterole",
        "rolreplication",
        "rolbypassrls",
        "has_database_privilege",
        "has_schema_privilege",
        "has_table_privilege",
        "has_sequence_privilege",
        "pg_auth_members",
        "prosecdef",
        "relrowsecurity",
        "proowner",
        "typowner",
    ):
        assert token in text
    assert text.index("Assert-BackupDatabasePrincipal") < text.index("[IO.File]::Open($partial")


def test_backup_pins_runtime_and_hardens_persisted_artifacts() -> None:
    text = _script()
    for token in (
        "$ExpectedPgDumpSha256",
        "$ExpectedPsqlSha256",
        "$ExpectedPgRestoreSha256",
        "$ExpectedRepoHead",
        "$ExpectedRepoTree",
        "$ExpectedScriptSha256",
        "PostgreSQL\\) 18",
        "Get-FileHash",
        "Set-ProtectedFileAcl",
        "pg_restore",
        "[IO.File]::Move",
        "New-ScheduledTaskPrincipal",
        "-LogonType Interactive",
        "-RunLevel Limited",
        "protected-launchers",
        "Assert-ProtectedDirectory",
        "CreateNew",
        "Flush($true)",
        "pg_largeobject_metadata",
        "installed backup script hash mismatch",
    ):
        assert token in text
    assert "IsPathFullyQualified" not in text
    assert "[IO.File]::Move($installTemp,$launcher,$true)" not in text


def test_backup_scripts_parse_and_selected_file_apis_execute_in_windows_powershell_51() -> None:
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        pytest.skip("Windows PowerShell 5.1 is unavailable")

    script_paths = ",".join("'" + str(path).replace("'", "''") + "'" for path in (SCRIPT, RAILWAY_SCRIPT))
    probe = rf"""
$ErrorActionPreference = 'Stop'
foreach ($path in @({script_paths})) {{
  $tokens = $null; $errors = $null
  [void][Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$errors)
  if ($errors.Count -ne 0) {{ throw ($errors | Out-String) }}
}}
if ([IO.Path]::GetPathRoot('C:\safe\artifact.dump') -ne 'C:\') {{ throw 'GetPathRoot probe failed' }}
$dir = Join-Path ([IO.Path]::GetTempPath()) ('applypilot-ps51-' + [guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($dir) | Out-Null
try {{
  $source = Join-Path $dir 'source.tmp'; $dest = Join-Path $dir 'dest.tmp'; $backup = Join-Path $dir 'dest.bak'
  [IO.File]::WriteAllText($source,'one'); [IO.File]::Move($source,$dest)
  [IO.File]::WriteAllText($source,'two'); [IO.File]::Replace($source,$dest,$backup,$true)
  if ([IO.File]::ReadAllText($dest) -cne 'two') {{ throw 'atomic replacement probe failed' }}
}} finally {{ [IO.Directory]::Delete($dir,$true) }}
"""
    completed = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", probe],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("bad_tool", ["psql", "pg_restore"])
def test_windows_powershell_51_rejects_unpinned_companion_before_execution(
    tmp_path: Path, bad_tool: str
) -> None:
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        pytest.skip("Windows PowerShell 5.1 is unavailable")

    tool_paths = {
        "pg_dump": tmp_path / "pg_dump.exe",
        "psql": tmp_path / "psql.exe",
        "pg_restore": tmp_path / "pg_restore.exe",
    }
    for name, path in tool_paths.items():
        path.write_bytes((f"hostile-{name}-must-never-execute\n").encode())
    digests = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in tool_paths.items()}
    expected = dict(digests)
    expected[bad_tool] = "0" * 64

    def ps_literal(value: str | Path) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    expected_message = f"{bad_tool} hash pin mismatch."
    probe = f"""
$ErrorActionPreference = 'Stop'
Import-Module Microsoft.PowerShell.Utility -ErrorAction Stop
$tokens=$null; $errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile({ps_literal(SCRIPT)},[ref]$tokens,[ref]$errors)
if ($errors.Count) {{ throw ($errors | Out-String) }}
$definition=$ast.Find({{param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Assert-PinnedRuntime'}},$true)
if ($null -eq $definition) {{ throw 'Assert-PinnedRuntime function missing' }}
Invoke-Expression $definition.Extent.Text
$ExpectedPgDumpSha256={ps_literal(expected['pg_dump'])}
$ExpectedPsqlSha256={ps_literal(expected['psql'])}
$ExpectedPgRestoreSha256={ps_literal(expected['pg_restore'])}
$ExpectedScriptSha256={'0' * 64!r}
$ExpectedRepoHead={'0' * 40!r}
$ExpectedRepoTree={'0' * 40!r}
try {{
  Assert-PinnedRuntime -PgDump {ps_literal(tool_paths['pg_dump'])} -Psql {ps_literal(tool_paths['psql'])} -PgRestore {ps_literal(tool_paths['pg_restore'])}
  throw 'unpinned companion was accepted'
}} catch {{
  if ($_.Exception.Message -cne {ps_literal(expected_message)}) {{ throw }}
  Write-Output {ps_literal(expected_message)}
}}
"""
    completed = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", probe],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert expected_message in completed.stdout


def test_backup_matrix_rows_bind_database_read_and_backup_artifact_write() -> None:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    rows = [row for row in matrix["rows"] if row["sourcePath"] == "backup-fleet-pg.ps1"]
    assert {row["surfaceKind"] for row in rows} == {
        "powershell-launcher",
        "scheduled-task-action",
        "scheduled-task-identity",
    }
    for row in rows:
        assert row["readWriteClass"] == "write"
        assert row["resourcesRead"] == ["fleet-postgres"]
        assert row["resourcesWritten"] == ["backup-artifact"]
        assert row["failClosedBehavior"] == "deny-unverified-backup-writer"


def test_railway_backup_uses_explicit_pinned_protected_channel() -> None:
    text = RAILWAY_SCRIPT.read_text(encoding="utf-8")
    assert "DATABASE_PUBLIC_URL" not in text
    for token in (
        "$PgHost",
        "$PgPassFile",
        "$PgUser",
        "$PgDb",
        "$ExpectedSystemIdentifier",
        "$ExpectedDatabaseOid",
        "$ExpectedDatabaseIncarnationId",
        "Assert-ProtectedCredentialFile",
        "pg_control_system()",
        "applypilot_database_identity",
        "session_user",
        "current_user",
        "pg_auth_members",
        "prosecdef",
        "relrowsecurity",
        "ExpectedPgDumpSha256",
        "ExpectedPsqlSha256",
        "ExpectedPgRestoreSha256",
        "ExpectedRepoHead",
        "ExpectedRepoTree",
        "ExpectedScriptSha256",
        "Set-ProtectedFileAcl",
        "[IO.File]::Move",
        "ValidateSet('verify-full')",
        "ExpectedPgSslRootCertSha256",
        "Assert-ProtectedDirectory",
        "CreateNew",
        "Flush($true)",
        "pg_largeobject_metadata",
    ):
        assert token in text
    assert text.index("Assert-RailwayBackupTarget") < text.index("[IO.File]::Open($partial")


def test_companion_hash_and_pg18_version_gates_precede_security_critical_use() -> None:
    fleet = _script()
    railway = RAILWAY_SCRIPT.read_text(encoding="utf-8")
    for text, preflight_call in (
        (fleet, "\nAssert-BackupDatabasePrincipal\n"),
        (railway, "\nAssert-RailwayBackupTarget\n"),
    ):
        for token in (
            "psql hash pin mismatch.",
            "pg_restore hash pin mismatch.",
            "^psql \\(PostgreSQL\\) 18",
            "^pg_restore \\(PostgreSQL\\) 18",
        ):
            assert token in text
        assert text.index("psql hash pin mismatch.") < text.index(preflight_call)
        assert text.index("pg_restore hash pin mismatch.") < text.index(preflight_call)
        assert text.index("pg_restore must be approved PostgreSQL major 18.") < text.index(
            "& $pgRestore --list"
        )


def test_railway_backup_matrix_binding_is_database_read_to_artifact_write() -> None:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    row = next(
        row
        for row in matrix["rows"]
        if row["sourcePath"] == "scripts/backup-railway-postgres.ps1"
    )
    assert row["readWriteClass"] == "write"
    assert row["resourcesRead"] == ["brain-postgres"]
    assert row["resourcesWritten"] == ["backup-artifact"]
    assert row["failClosedBehavior"] == "deny-unverified-backup-writer"
