<#
Offline break-glass mapped-role rotation and first-match-safe pg_hba reconciliation.
The provider-admin DSN and evidence-authentication key exist only in this
operator process and are never installed as worker runtime state.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$NodeId,
    [Parameter(Mandatory = $true)][ValidateSet('apply', 'linkedin', 'compute', 'discovery')]
    [string]$Contract,
    [Parameter(Mandatory = $true)][string]$Role,
    [Parameter(Mandatory = $true)][string]$RegrantManifest,
    [string]$TailnetCidr = "100.64.0.0/10",
    [string]$Database = "applypilot_fleet",
    [string]$ReceiptPath = ".\deployment-receipts\fleet-role-receipt.json",
    [string]$RollbackSql = ".\deployment-receipts\fleet-role-rollback.sql",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

function Assert-SafeScalar {
    param([Parameter(Mandatory = $true)][string]$Name, [Parameter(Mandatory = $true)][string]$Value)
    if (-not $Value -or $Value.IndexOfAny([char[]](0..31 + 127)) -ge 0) {
        throw "$Name contains empty or control-character data"
    }
}

function Get-ApplyPilotHbaLines {
    param([Parameter(Mandatory = $true)][string]$HbaPath)
    $lines = @(Get-Content -LiteralPath $HbaPath)
    foreach ($line in $lines) {
        if ($line.IndexOf([char]0) -ge 0) { throw "pg_hba.conf contains a NUL control character" }
        $trimmed = $line.Trim()
        if ($trimmed -and -not $trimmed.StartsWith("#") -and $trimmed -match '^include(_if_exists|_dir)?\s') {
            throw "pg_hba.conf include directives are unsupported; flatten and review includes before rollout"
        }
    }
    return ,$lines
}

function New-ApplyPilotManagedHba {
    param(
        [Parameter(Mandatory = $true)][string[]]$Lines,
        [Parameter(Mandatory = $true)][string]$Database,
        [Parameter(Mandatory = $true)][string]$Role,
        [Parameter(Mandatory = $true)][string]$TailnetCidr
    )
    $begin = "# BEGIN APPLYPILOT MAPPED ROLE $Role"
    $end = "# END APPLYPILOT MAPPED ROLE $Role"
    $backupBegin = "# BEGIN APPLYPILOT BACKUP ROLE fleet_backup"
    $backupEnd = "# END APPLYPILOT BACKUP ROLE fleet_backup"
    $withoutManaged = [Collections.Generic.List[string]]::new()
    $inside = $false
    $expectedEnd = $null
    foreach ($line in $Lines) {
        if ($line -eq $begin -or $line -eq $backupBegin) {
            if ($inside) { throw "nested ApplyPilot pg_hba managed blocks" }
            $inside = $true
            $expectedEnd = if ($line -eq $begin) { $end } else { $backupEnd }
            continue
        }
        if ($line -eq $end -or $line -eq $backupEnd) {
            if (-not $inside) { throw "unmatched ApplyPilot pg_hba managed block end" }
            if ($line -ne $expectedEnd) { throw "mismatched ApplyPilot pg_hba managed block end" }
            $inside = $false
            $expectedEnd = $null
            continue
        }
        if (-not $inside) { $withoutManaged.Add($line) }
    }
    if ($inside) { throw "unterminated ApplyPilot pg_hba managed block" }

    $firstHost = $withoutManaged.Count
    for ($index = 0; $index -lt $withoutManaged.Count; $index++) {
        $trimmed = $withoutManaged[$index].Trim()
        if ($trimmed -and -not $trimmed.StartsWith("#") -and $trimmed -match '^(host|hostssl|hostnossl)\s') {
            $firstHost = $index
            break
        }
    }
    $managed = @(
        $backupBegin,
        "host $Database fleet_backup 127.0.0.1/32 scram-sha-256",
        "host $Database fleet_backup ::1/128 scram-sha-256",
        "host $Database fleet_backup 0.0.0.0/0 reject",
        "host $Database fleet_backup ::0/0 reject",
        $backupEnd,
        $begin,
        "host $Database $Role $TailnetCidr scram-sha-256",
        "host $Database $Role 0.0.0.0/0 reject",
        "host $Database $Role ::0/0 reject",
        $end
    )
    $candidate = @()
    if ($firstHost -gt 0) { $candidate += @($withoutManaged[0..($firstHost - 1)]) }
    $candidate += $managed
    if ($firstHost -lt $withoutManaged.Count) {
        $candidate += @($withoutManaged[$firstHost..($withoutManaged.Count - 1)])
    }
    return ,$candidate
}

function Assert-ApplyPilotHbaEffectiveOrder {
    param(
        [Parameter(Mandatory = $true)][string[]]$Lines,
        [Parameter(Mandatory = $true)][string]$Database,
        [Parameter(Mandatory = $true)][string]$Role,
        [Parameter(Mandatory = $true)][string]$TailnetCidr
    )
    $expected = @(
        "host $Database fleet_backup 127.0.0.1/32 scram-sha-256",
        "host $Database fleet_backup ::1/128 scram-sha-256",
        "host $Database fleet_backup 0.0.0.0/0 reject",
        "host $Database fleet_backup ::0/0 reject",
        "host $Database $Role $TailnetCidr scram-sha-256",
        "host $Database $Role 0.0.0.0/0 reject",
        "host $Database $Role ::0/0 reject"
    )
    $hostIndices = @()
    for ($index = 0; $index -lt $Lines.Count; $index++) {
        $trimmed = $Lines[$index].Trim()
        if ($trimmed -and -not $trimmed.StartsWith("#") -and $trimmed -match '^(host|hostssl|hostnossl)\s') {
            $hostIndices += $index
        }
    }
    if ($hostIndices.Count -lt 7) { throw "managed backup and mapped-role pg_hba allow/deny rules are missing" }
    $first = $hostIndices[0]
    for ($offset = 0; $offset -lt $expected.Count; $offset++) {
        if ($Lines[$hostIndices[$offset]].Trim() -ne $expected[$offset]) {
            throw "managed backup and mapped-role allow/deny rules do not precede every effective host rule"
        }
    }
    foreach ($later in $hostIndices | Select-Object -Skip 7) {
        if ($later -le $hostIndices[6]) { throw "broad host rule precedes the managed deny boundary" }
    }
}

function Assert-ApplyPilotOutputPaths {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Paths,
        [Parameter(Mandatory = $true)][string[]]$MustNotExist
    )
    $normalized = @{}
    foreach ($name in $Paths.Keys) {
        Assert-SafeScalar -Name $name -Value ([string]$Paths[$name])
        $full = [IO.Path]::GetFullPath([string]$Paths[$name])
        $key = $full.ToUpperInvariant()
        if ($normalized.ContainsKey($key)) {
            throw "path collision: $name and $($normalized[$key]) resolve to $full"
        }
        $normalized[$key] = $name
    }
    foreach ($name in $MustNotExist) {
        if (Test-Path -LiteralPath $Paths[$name]) { throw "$name already exists: $($Paths[$name])" }
    }
}

function Write-ApplyPilotDurableAtomicBytes {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][byte[]]$Payload,
        [scriptblock]$BeforeReplace
    )
    $fullPath = [IO.Path]::GetFullPath($Path)
    $directory = [IO.Path]::GetDirectoryName($fullPath)
    $temporary = [IO.Path]::Combine(
        $directory,
        (".{0}.{1}.tmp" -f [IO.Path]::GetFileName($fullPath), [guid]::NewGuid().ToString("N"))
    )
    $replacementBackup = "$temporary.previous"
    $destinationAcl = Get-Acl -LiteralPath $fullPath
    try {
        $stream = [IO.FileStream]::new(
            $temporary,
            [IO.FileMode]::CreateNew,
            [IO.FileAccess]::Write,
            [IO.FileShare]::None,
            4096,
            [IO.FileOptions]::WriteThrough
        )
        try {
            $stream.Write($Payload, 0, $Payload.Length)
            $stream.Flush($true)
        }
        finally { $stream.Dispose() }
        Set-Acl -LiteralPath $temporary -AclObject $destinationAcl
        $metadataStream = [IO.FileStream]::new(
            $temporary, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::Read
        )
        try { $metadataStream.Flush($true) }
        finally { $metadataStream.Dispose() }
        if ($BeforeReplace) { & $BeforeReplace }
        [IO.File]::Replace($temporary, $fullPath, $replacementBackup, $true)
        Set-Acl -LiteralPath $fullPath -AclObject $destinationAcl
        $committedStream = [IO.FileStream]::new(
            $fullPath, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::Read
        )
        try { $committedStream.Flush($true) }
        finally { $committedStream.Dispose() }
    }
    finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
        if (Test-Path -LiteralPath $replacementBackup) { Remove-Item -LiteralPath $replacementBackup -Force }
    }
}

function Invoke-ApplyPilotHbaReplacement {
    param(
        [Parameter(Mandatory = $true)][string]$HbaPath,
        [Parameter(Mandatory = $true)][string[]]$CandidateLines,
        [Parameter(Mandatory = $true)][string]$Database,
        [Parameter(Mandatory = $true)][string]$Role,
        [Parameter(Mandatory = $true)][string]$TailnetCidr,
        [Parameter(Mandatory = $true)][string]$ReceiptPath,
        [Parameter(Mandatory = $true)][string]$RollbackSql,
        [Parameter(Mandatory = $true)][string]$PreflightBackup,
        [Parameter(Mandatory = $true)][string]$ReplaceBackup,
        [Parameter(Mandatory = $true)][string]$CandidatePath,
        [Parameter(Mandatory = $true)][string]$RestorePath,
        [Parameter(Mandatory = $true)][scriptblock]$ValidateAndReload,
        [Parameter(Mandatory = $true)][scriptblock]$ReloadOriginal,
        [Parameter(Mandatory = $true)][scriptblock]$FinalizeOutput
    )
    $paths = @{
        HbaPath = $HbaPath; ReceiptPath = $ReceiptPath; RollbackSql = $RollbackSql
        PreflightBackup = $PreflightBackup; ReplaceBackup = $ReplaceBackup
        CandidatePath = $CandidatePath; RestorePath = $RestorePath
    }
    Assert-ApplyPilotOutputPaths -Paths $paths -MustNotExist @(
        "PreflightBackup", "ReplaceBackup", "CandidatePath", "RestorePath"
    )
    $original = [IO.File]::ReadAllBytes($HbaPath)
    $receiptOriginal = [IO.File]::ReadAllBytes($ReceiptPath)
    $rollbackOriginal = [IO.File]::ReadAllBytes($RollbackSql)
    $replaced = $false
    try {
        Copy-Item -LiteralPath $HbaPath -Destination $PreflightBackup
        [IO.File]::WriteAllLines($CandidatePath, $CandidateLines, [Text.UTF8Encoding]::new($false))
        [System.IO.File]::Replace($CandidatePath, $HbaPath, $ReplaceBackup, $true)
        $replaced = $true
        $effective = Get-ApplyPilotHbaLines -HbaPath $HbaPath
        Assert-ApplyPilotHbaEffectiveOrder -Lines $effective -Database $Database -Role $Role -TailnetCidr $TailnetCidr
        & $ValidateAndReload
        & $FinalizeOutput
    }
    catch {
        $failure = $_
        $reloadFailure = $null
        if ($replaced) {
            [IO.File]::WriteAllBytes($RestorePath, $original)
            [System.IO.File]::Replace($RestorePath, $HbaPath, $CandidatePath, $true)
            try { & $ReloadOriginal }
            catch { $reloadFailure = $_ }
        }
        Write-ApplyPilotDurableAtomicBytes -Path $ReceiptPath -Payload $receiptOriginal
        Write-ApplyPilotDurableAtomicBytes -Path $RollbackSql -Payload $rollbackOriginal
        if ($reloadFailure) {
            throw "pg_hba restored but reload failed: $($reloadFailure.Exception.Message); original failure: $($failure.Exception.Message)"
        }
        throw $failure
    }
}

if ($MyInvocation.InvocationName -eq '.') { return }

foreach ($item in @{
    NodeId = $NodeId; Contract = $Contract; Role = $Role; RegrantManifest = $RegrantManifest
    TailnetCidr = $TailnetCidr; Database = $Database; ReceiptPath = $ReceiptPath
    RollbackSql = $RollbackSql; Python = $Python
}.GetEnumerator()) {
    Assert-SafeScalar -Name $item.Key -Value ([string]$item.Value)
}
if ($Role -in @("postgres", "fleet_worker")) { throw "Role must be a unique per-node login role" }
if ($Role -notmatch '^[A-Za-z_][A-Za-z0-9_.-]{0,62}$') { throw "Role contains unsupported characters" }
if ($Database -notmatch '^[A-Za-z_][A-Za-z0-9_.-]{0,62}$') { throw "Database contains unsupported characters" }
if ($TailnetCidr -notmatch '^[0-9A-Fa-f:.]+/[0-9]{1,3}$') { throw "TailnetCidr must be an explicit CIDR" }
if (-not $env:APPLYPILOT_ADMIN_PG_DSN) { throw "APPLYPILOT_ADMIN_PG_DSN is required for offline mapped-role reconciliation" }
if (-not $env:APPLYPILOT_ROLLBACK_HMAC_KEY_HEX) { throw "APPLYPILOT_ROLLBACK_HMAC_KEY_HEX is required for authenticated rollback evidence" }
if (-not $env:APPLYPILOT_ROLLBACK_HMAC_KEY_ID) { throw "APPLYPILOT_ROLLBACK_HMAC_KEY_ID is required for authenticated rollback evidence" }
if (-not (Test-Path -LiteralPath $RegrantManifest -PathType Leaf)) { throw "RegrantManifest does not exist: $RegrantManifest" }

$hbaInventory = @'
import json, os, psycopg
from psycopg.rows import dict_row
with psycopg.connect(os.environ["APPLYPILOT_ADMIN_PG_DSN"], row_factory=dict_row) as conn:
    rows = conn.execute(
        "SELECT line_number,type,database,user_name,address,auth_method,error "
        "FROM pg_hba_file_rules ORDER BY line_number"
    ).fetchall()
    if any(row["error"] for row in rows):
        raise SystemExit("existing pg_hba_file_rules contains parse errors")
    hba = conn.execute("SHOW hba_file").fetchone()["hba_file"]
print(json.dumps({"hba_file": hba, "rules": rows}, default=str))
'@
$inventoryJson = & $Python -c $hbaInventory
if ($LASTEXITCODE -ne 0) { throw "pg_hba inventory failed" }
$inventory = $inventoryJson | ConvertFrom-Json
$HbaPath = [string]$inventory.hba_file
$lines = Get-ApplyPilotHbaLines -HbaPath $HbaPath
$candidateLines = New-ApplyPilotManagedHba -Lines $lines -Database $Database -Role $Role -TailnetCidr $TailnetCidr
Assert-ApplyPilotHbaEffectiveOrder -Lines $candidateLines -Database $Database -Role $Role -TailnetCidr $TailnetCidr

$receiptFull = [IO.Path]::GetFullPath($ReceiptPath)
$rollbackFull = [IO.Path]::GetFullPath($RollbackSql)
$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fffffff"
$preflightBackup = "$HbaPath.applypilot-$stamp.preflight.bak"
$replaceBackup = "$HbaPath.applypilot-$stamp.replace.bak"
$candidatePath = "$HbaPath.applypilot-$stamp.candidate"
$restorePath = "$HbaPath.applypilot-$stamp.restore"
$preflightPaths = @{
    HbaPath = $HbaPath; ReceiptPath = $receiptFull; RollbackSql = $rollbackFull
    PreflightBackup = $preflightBackup; ReplaceBackup = $replaceBackup
    CandidatePath = $candidatePath; RestorePath = $restorePath
}
Assert-ApplyPilotOutputPaths -Paths $preflightPaths -MustNotExist @(
    "ReceiptPath", "RollbackSql", "PreflightBackup", "ReplaceBackup", "CandidatePath", "RestorePath"
)
New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($receiptFull)) | Out-Null
New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($rollbackFull)) | Out-Null

$secret = Read-Host -AsSecureString "New SCRAM password for mapped role '$Role'"
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToGlobalAllocUnicode($secret)
try { $env:APPLYPILOT_PG_ROLE_PW = [Runtime.InteropServices.Marshal]::PtrToStringUni($ptr) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeGlobalAllocUnicode($ptr) }
$env:APPLYPILOT_PG_ROLE = $Role
$env:APPLYPILOT_PG_NODE = $NodeId
$env:APPLYPILOT_PG_CONTRACT = $Contract
$env:APPLYPILOT_REGRANT_MANIFEST = [IO.Path]::GetFullPath($RegrantManifest)
$env:APPLYPILOT_RECEIPT_PATH = $receiptFull
$env:APPLYPILOT_ROLLBACK_SQL = $rollbackFull

$roleReconcile = @'
import base64, hashlib, json, os
from dataclasses import asdict
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from applypilot.fleet.pg_roles import (
    AclRegrant,
    DurableEvidencePaths,
    RegrantManifest,
    authenticate_evidence_receipt,
    ensure_fleet_worker_role,
    verify_evidence_receipt,
)

with open(os.environ["APPLYPILOT_REGRANT_MANIFEST"], encoding="utf-8") as stream:
    raw = json.load(stream)
allowed_keys = {"database_owner_role","controller_roles","verifier_roles","retired_admin_roles","backup_role","migrator_role","infrastructure_superuser_roles","expected_service_roles","regrants"}
unknown = set(raw) - allowed_keys
if unknown or "regrant_sql" in raw:
    raise SystemExit(f"unsupported regrant manifest fields: {sorted(unknown | ({'regrant_sql'} & set(raw)))}")
manifest = RegrantManifest(
    database_owner_role=raw["database_owner_role"],
    backup_role=raw.get("backup_role", "fleet_backup"),
    migrator_role=raw.get("migrator_role", "brain_schema_migrator"),
    controller_roles=tuple(raw["controller_roles"]),
    verifier_roles=tuple(raw["verifier_roles"]),
    retired_admin_roles=tuple(raw["retired_admin_roles"]),
    infrastructure_superuser_roles=tuple(raw["infrastructure_superuser_roles"]),
    expected_service_roles=tuple(raw.get("expected_service_roles", ())),
    regrants=tuple(
        AclRegrant(
            object_kind=item["object_kind"], qualified_name=item["qualified_name"],
            privileges=tuple(item["privileges"]), grantee=item["grantee"]
        )
        for item in raw.get("regrants", ())
        if not (set(item) - {"object_kind", "qualified_name", "privileges", "grantee"})
    ),
)
if len(manifest.regrants) != len(raw.get("regrants", ())):
    raise SystemExit("structured regrant contains unsupported fields")

try:
    authentication_key = bytes.fromhex(os.environ["APPLYPILOT_ROLLBACK_HMAC_KEY_HEX"])
except (KeyError, ValueError):
    authentication_key = b""
authentication_key_id = os.environ.get("APPLYPILOT_ROLLBACK_HMAC_KEY_ID", "")
if len(authentication_key) < 32 or not authentication_key_id:
    raise SystemExit("trusted rollback HMAC key and key id are required")
receipt_path = Path(os.environ["APPLYPILOT_RECEIPT_PATH"]).resolve()
rollback_path = Path(os.environ["APPLYPILOT_ROLLBACK_SQL"]).resolve()
evidence_paths = DurableEvidencePaths(
    preparation_receipt_path=receipt_path,
    rollback_sql_path=rollback_path,
    authentication_key=authentication_key,
    authentication_key_id=authentication_key_id,
)

with psycopg.connect(os.environ["APPLYPILOT_ADMIN_PG_DSN"], row_factory=dict_row) as conn:
    receipt = ensure_fleet_worker_role(
        conn, os.environ["APPLYPILOT_PG_ROLE_PW"], role=os.environ["APPLYPILOT_PG_ROLE"],
        worker_id=os.environ["APPLYPILOT_PG_NODE"], contract=os.environ["APPLYPILOT_PG_CONTRACT"],
        regrant_manifest=manifest, evidence_paths=evidence_paths,
    )

with receipt_path.open(encoding="utf-8-sig") as stream:
    prepared = json.load(stream)
verify_evidence_receipt(
    prepared,
    authentication_key=authentication_key,
    expected_key_id=authentication_key_id,
)
rollback_bytes = rollback_path.read_bytes()
inventory = dict(receipt.inventory)
target_identity = {
    "database_name": inventory.get("database_name"),
    "system_identifier": inventory.get("system_identifier"),
    "database_oid": inventory.get("database_oid"),
}
if (
    prepared.get("status") != "prepared_before_database_mutation"
    or prepared.get("rollback_mode") != "topology_exact"
    or Path(prepared.get("rollback_sql_path", "")).resolve(strict=True) != rollback_path
    or prepared.get("rollback_sql_sha256") != hashlib.sha256(rollback_bytes).hexdigest()
    or any(prepared.get(name) != value for name, value in target_identity.items())
):
    raise SystemExit("prepared rollback evidence is not bound to the reconciled database target")
data = asdict(receipt)
rollback = data.pop("rollback_sql")
encoded = rollback.encode("utf-8")
if encoded != rollback_bytes:
    raise SystemExit("returned rollback SQL differs from durable prepared rollback SQL")
data.update(
    target_identity,
    rollback_mode="topology_exact",
    rollback_sql_sha256=hashlib.sha256(encoded).hexdigest(),
    rollback_sql_path=str(rollback_path),
    status="database_reconciled",
    escalation_required=True,
    in_doubt=True,
)
data = authenticate_evidence_receipt(
    data,
    authentication_key=authentication_key,
    authentication_key_id=authentication_key_id,
)
verify_evidence_receipt(
    data,
    authentication_key=authentication_key,
    expected_key_id=authentication_key_id,
)
payload = (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
print(base64.b64encode(payload).decode("ascii"))
'@
$verifyReceipt = @'
import hashlib, json, os
from pathlib import Path
from applypilot.fleet.pg_roles import verify_evidence_receipt

try:
    authentication_key = bytes.fromhex(os.environ["APPLYPILOT_ROLLBACK_HMAC_KEY_HEX"])
except (KeyError, ValueError):
    authentication_key = b""
authentication_key_id = os.environ.get("APPLYPILOT_ROLLBACK_HMAC_KEY_ID", "")
if len(authentication_key) < 32 or not authentication_key_id:
    raise SystemExit("trusted rollback HMAC key and key id are required")
receipt_path = Path(os.environ["APPLYPILOT_RECEIPT_PATH"]).resolve(strict=True)
rollback_path = Path(os.environ["APPLYPILOT_ROLLBACK_SQL"]).resolve(strict=True)
with receipt_path.open(encoding="utf-8-sig") as stream:
    receipt = json.load(stream)
verify_evidence_receipt(
    receipt,
    authentication_key=authentication_key,
    expected_key_id=authentication_key_id,
)
expected_status = os.environ["APPLYPILOT_EXPECTED_RECEIPT_STATUS"]
expected_flags = {
    "database_reconciled": (True, True),
    "deployment_committed": (False, False),
}
if expected_status not in expected_flags or receipt.get("status") != expected_status:
    raise SystemExit("authenticated receipt status does not match the expected deployment stage")
if (receipt.get("escalation_required"), receipt.get("in_doubt")) != expected_flags[expected_status]:
    raise SystemExit("authenticated receipt uncertainty flags do not match the deployment stage")
inventory = receipt.get("inventory")
if not isinstance(inventory, dict) or receipt.get("rollback_mode") != "topology_exact":
    raise SystemExit("authenticated receipt does not identify topology-exact rollback")
for name in ("database_name", "system_identifier", "database_oid"):
    if receipt.get(name) != inventory.get(name):
        raise SystemExit("authenticated receipt target identity pins are incomplete or inconsistent")
if (
    not receipt.get("database_name")
    or not str(receipt.get("system_identifier", "")).isdigit()
    or type(receipt.get("database_oid")) is not int
    or receipt["database_oid"] <= 0
):
    raise SystemExit("authenticated receipt is missing exact target identity pins")
signed_rollback_path = Path(receipt.get("rollback_sql_path", ""))
if not signed_rollback_path.is_absolute() or signed_rollback_path.resolve(strict=True) != rollback_path:
    raise SystemExit("rollback SQL path does not match the authenticated receipt")
if receipt.get("rollback_sql_sha256") != hashlib.sha256(rollback_path.read_bytes()).hexdigest():
    raise SystemExit("rollback SQL SHA-256 does not match the authenticated receipt")
'@
try {
    $reconciledPayload = & $Python -c $roleReconcile
    if ($LASTEXITCODE -ne 0) { throw "role reconciliation failed" }
    $reconciledBytes = [Convert]::FromBase64String(($reconciledPayload -join ""))
    $preparedBytes = [IO.File]::ReadAllBytes($receiptFull)
    try {
        Write-ApplyPilotDurableAtomicBytes -Path $receiptFull -Payload $reconciledBytes
        $env:APPLYPILOT_EXPECTED_RECEIPT_STATUS = "database_reconciled"
        & $Python -c $verifyReceipt
        if ($LASTEXITCODE -ne 0) { throw "authenticated database reconciliation receipt verification failed" }
    }
    catch {
        Write-ApplyPilotDurableAtomicBytes -Path $receiptFull -Payload $preparedBytes
        throw
    }
}
finally { $env:APPLYPILOT_PG_ROLE_PW = "" }

$validateAndReload = {
    $code = @'
import os, psycopg
from psycopg.rows import dict_row
with psycopg.connect(os.environ["APPLYPILOT_ADMIN_PG_DSN"], row_factory=dict_row) as conn:
    errors = conn.execute("SELECT line_number,error FROM pg_hba_file_rules WHERE error IS NOT NULL ORDER BY line_number").fetchall()
    if errors: raise SystemExit(f"candidate pg_hba invalid: {errors}")
    if not conn.execute("SELECT pg_reload_conf() AS reloaded").fetchone()["reloaded"]:
        raise SystemExit("pg_reload_conf returned false")
'@
    & $Python -c $code
    if ($LASTEXITCODE -ne 0) { throw "pg_hba validation or reload failed" }
}
$reloadOriginal = {
    & $Python -c "import os,psycopg; c=psycopg.connect(os.environ['APPLYPILOT_ADMIN_PG_DSN']); c.execute('SELECT pg_reload_conf()'); c.close()"
    if ($LASTEXITCODE -ne 0) { throw "restored pg_hba reload failed" }
}
$finalizeOutput = {
    $env:APPLYPILOT_HBA_PATH = $HbaPath
    $env:APPLYPILOT_HBA_BACKUP_PATH = $preflightBackup
    $env:APPLYPILOT_HBA_RULES_JSON = ($inventory.rules | ConvertTo-Json -Depth 16 -Compress)
    $env:APPLYPILOT_HBA_BACKUP_POLICY_JSON = ([ordered]@{
        role = "fleet_backup"
        allowed_sources = @("127.0.0.1/32", "::1/128")
        authentication = "scram-sha-256"
        all_other_ipv4 = "reject"
        all_other_ipv6 = "reject"
    } | ConvertTo-Json -Depth 8 -Compress)
    $code = @'
import base64, hashlib, json, os, stat
from pathlib import Path
from applypilot.fleet.pg_roles import authenticate_evidence_receipt, verify_evidence_receipt

try:
    authentication_key = bytes.fromhex(os.environ["APPLYPILOT_ROLLBACK_HMAC_KEY_HEX"])
except (KeyError, ValueError):
    authentication_key = b""
authentication_key_id = os.environ.get("APPLYPILOT_ROLLBACK_HMAC_KEY_ID", "")
if len(authentication_key) < 32 or not authentication_key_id:
    raise SystemExit("trusted rollback HMAC key and key id are required")
receipt_path = Path(os.environ["APPLYPILOT_RECEIPT_PATH"]).resolve(strict=True)
with receipt_path.open(encoding="utf-8-sig") as stream:
    receipt = json.load(stream)
verify_evidence_receipt(
    receipt,
    authentication_key=authentication_key,
    expected_key_id=authentication_key_id,
)
if receipt.get("status") != "database_reconciled" or not receipt.get("in_doubt"):
    raise SystemExit("HBA finalization requires an authenticated database-reconciled receipt")
live_hba = Path(os.environ["APPLYPILOT_HBA_PATH"]).resolve(strict=True)
backup_hba = Path(os.environ["APPLYPILOT_HBA_BACKUP_PATH"]).resolve(strict=True)
if live_hba == backup_hba or live_hba.parent != backup_hba.parent:
    raise SystemExit("HBA backup must be a distinct file in the live HBA directory")
for label, path in (("live HBA", live_hba), ("HBA backup", backup_hba)):
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(f"{label} must be a regular non-symlink file")
live_bytes = live_hba.read_bytes()
backup_bytes = backup_hba.read_bytes()
live_stat = live_hba.stat()
backup_stat = backup_hba.stat()
if (live_stat.st_uid, live_stat.st_gid) != (backup_stat.st_uid, backup_stat.st_gid):
    raise SystemExit("HBA backup ownership differs from live HBA")
receipt.update(
    hba_rules=json.loads(os.environ["APPLYPILOT_HBA_RULES_JSON"]),
    hba_path=str(live_hba),
    hba_backup=str(backup_hba),
    backup_hba_policy=json.loads(os.environ["APPLYPILOT_HBA_BACKUP_POLICY_JSON"]),
    hba_restore={
        "format": "applypilot-hba-restore-v2",
        "live_hba_path": str(live_hba),
        "backup_path": str(backup_hba),
        "expected_target_sha256": hashlib.sha256(live_bytes).hexdigest(),
        "backup_sha256": hashlib.sha256(backup_bytes).hexdigest(),
        "backup_size": len(backup_bytes),
        "backup_mode": stat.S_IMODE(backup_stat.st_mode),
        "owner": {"uid": live_stat.st_uid, "gid": live_stat.st_gid},
    },
    status="deployment_committed",
    escalation_required=False,
    in_doubt=False,
)
receipt = authenticate_evidence_receipt(
    receipt,
    authentication_key=authentication_key,
    authentication_key_id=authentication_key_id,
)
verify_evidence_receipt(
    receipt,
    authentication_key=authentication_key,
    expected_key_id=authentication_key_id,
)
payload = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
print(base64.b64encode(payload).decode("ascii"))
'@
    $committedPayload = & $Python -c $code
    if ($LASTEXITCODE -ne 0) { throw "authenticated HBA receipt finalization failed" }
    $committedBytes = [Convert]::FromBase64String(($committedPayload -join ""))
    Write-ApplyPilotDurableAtomicBytes -Path $receiptFull -Payload $committedBytes
    $env:APPLYPILOT_EXPECTED_RECEIPT_STATUS = "deployment_committed"
    & $Python -c $verifyReceipt
    if ($LASTEXITCODE -ne 0) { throw "authenticated deployment receipt verification failed" }
}
Invoke-ApplyPilotHbaReplacement -HbaPath $HbaPath -CandidateLines $candidateLines `
    -Database $Database -Role $Role -TailnetCidr $TailnetCidr -ReceiptPath $receiptFull `
    -RollbackSql $rollbackFull -PreflightBackup $preflightBackup -ReplaceBackup $replaceBackup `
    -CandidatePath $candidatePath -RestorePath $restorePath -ValidateAndReload $validateAndReload `
    -ReloadOriginal $reloadOriginal -FinalizeOutput $finalizeOutput

Write-Host "[pg-tailscale] mapped role and first-match pg_hba policy reconciled"
Write-Host "[pg-tailscale] receipt: $receiptFull"
Write-Host "[pg-tailscale] rollback SQL: $rollbackFull"
