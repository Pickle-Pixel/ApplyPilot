<#
Controller-only mapped-role rotation and first-match-safe pg_hba reconciliation.
The admin DSN exists only in this migration process and is never installed as
worker runtime state.
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

function Write-ApplyPilotDurableAtomicJson {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][object]$Value,
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
    $payload = [Text.UTF8Encoding]::new($false).GetBytes(
        (($Value | ConvertTo-Json -Depth 16) + [Environment]::NewLine)
    )
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
            $stream.Write($payload, 0, $payload.Length)
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
        [IO.File]::WriteAllBytes($ReceiptPath, $receiptOriginal)
        [IO.File]::WriteAllBytes($RollbackSql, $rollbackOriginal)
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
if (-not $env:APPLYPILOT_CONTROLLER_PG_DSN) { throw "APPLYPILOT_CONTROLLER_PG_DSN is required for mapped-role reconciliation" }
if (-not (Test-Path -LiteralPath $RegrantManifest -PathType Leaf)) { throw "RegrantManifest does not exist: $RegrantManifest" }

$hbaInventory = @'
import json, os, psycopg
from psycopg.rows import dict_row
with psycopg.connect(os.environ["APPLYPILOT_CONTROLLER_PG_DSN"], row_factory=dict_row) as conn:
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
import hashlib, json, os, tempfile
from dataclasses import asdict
import psycopg
from psycopg.rows import dict_row
from applypilot.fleet.pg_roles import AclRegrant, RegrantManifest, ensure_fleet_worker_role

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

def fsync_parent(path):
    if os.name == "nt":
        return
    descriptor = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def write_exclusive_durable(path, payload):
    with open(path, "xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_parent(path)

def replace_durable(path, payload):
    directory = os.path.dirname(path) or "."
    descriptor, temporary = tempfile.mkstemp(prefix=".applypilot-receipt-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_parent(path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def write_precommit_evidence(inventory, rollback):
    encoded = rollback.encode("utf-8")
    prepared = {
        "status": "prepared_before_database_mutation",
        "inventory": inventory,
        "rollback_sql_sha256": hashlib.sha256(encoded).hexdigest(),
        "rollback_sql_path": os.environ["APPLYPILOT_ROLLBACK_SQL"],
        "escalation_required": True,
        "in_doubt": True,
    }
    write_exclusive_durable(os.environ["APPLYPILOT_ROLLBACK_SQL"], encoded)
    write_exclusive_durable(
        os.environ["APPLYPILOT_RECEIPT_PATH"],
        (json.dumps(prepared, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )

with psycopg.connect(os.environ["APPLYPILOT_CONTROLLER_PG_DSN"], row_factory=dict_row) as conn:
    receipt = ensure_fleet_worker_role(
        conn, os.environ["APPLYPILOT_PG_ROLE_PW"], role=os.environ["APPLYPILOT_PG_ROLE"],
        worker_id=os.environ["APPLYPILOT_PG_NODE"], contract=os.environ["APPLYPILOT_PG_CONTRACT"],
        regrant_manifest=manifest, evidence_writer=write_precommit_evidence,
    )
data = asdict(receipt)
rollback = data.pop("rollback_sql")
encoded = rollback.encode("utf-8")
data["rollback_sql_sha256"] = hashlib.sha256(encoded).hexdigest()
data["rollback_sql_path"] = os.environ["APPLYPILOT_ROLLBACK_SQL"]
data["status"] = "database_reconciled"
data["escalation_required"] = True
data["in_doubt"] = True
replace_durable(
    os.environ["APPLYPILOT_RECEIPT_PATH"],
    (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8"),
)
'@
try {
    & $Python -c $roleReconcile
    if ($LASTEXITCODE -ne 0) { throw "role reconciliation failed" }
}
finally { $env:APPLYPILOT_PG_ROLE_PW = "" }

$validateAndReload = {
    $code = @'
import os, psycopg
from psycopg.rows import dict_row
with psycopg.connect(os.environ["APPLYPILOT_CONTROLLER_PG_DSN"], row_factory=dict_row) as conn:
    errors = conn.execute("SELECT line_number,error FROM pg_hba_file_rules WHERE error IS NOT NULL ORDER BY line_number").fetchall()
    if errors: raise SystemExit(f"candidate pg_hba invalid: {errors}")
    if not conn.execute("SELECT pg_reload_conf() AS reloaded").fetchone()["reloaded"]:
        raise SystemExit("pg_reload_conf returned false")
'@
    & $Python -c $code
    if ($LASTEXITCODE -ne 0) { throw "pg_hba validation or reload failed" }
}
$reloadOriginal = {
    & $Python -c "import os,psycopg; c=psycopg.connect(os.environ['APPLYPILOT_CONTROLLER_PG_DSN']); c.execute('SELECT pg_reload_conf()'); c.close()"
    if ($LASTEXITCODE -ne 0) { throw "restored pg_hba reload failed" }
}
$finalizeOutput = {
    $receipt = Get-Content -LiteralPath $receiptFull -Raw | ConvertFrom-Json
    $receipt | Add-Member -NotePropertyName hba_rules -NotePropertyValue $inventory.rules
    $receipt | Add-Member -NotePropertyName hba_path -NotePropertyValue $HbaPath
    $receipt | Add-Member -NotePropertyName hba_backup -NotePropertyValue $preflightBackup
    $receipt | Add-Member -NotePropertyName backup_hba_policy -NotePropertyValue ([ordered]@{
        role = "fleet_backup"
        allowed_sources = @("127.0.0.1/32", "::1/128")
        authentication = "scram-sha-256"
        all_other_ipv4 = "reject"
        all_other_ipv6 = "reject"
    })
    $receipt | Add-Member -NotePropertyName rollback_sql_path -NotePropertyValue $rollbackFull
    $receipt | Add-Member -Force -NotePropertyName status -NotePropertyValue "deployment_committed"
    $receipt | Add-Member -Force -NotePropertyName escalation_required -NotePropertyValue $false
    $receipt | Add-Member -Force -NotePropertyName in_doubt -NotePropertyValue $false
    Write-ApplyPilotDurableAtomicJson -Path $receiptFull -Value $receipt
    $verified = Get-Content -LiteralPath $receiptFull -Raw | ConvertFrom-Json
    if ($verified.hba_path -ne $HbaPath -or $verified.hba_backup -ne $preflightBackup -or
        $verified.backup_hba_policy.role -ne "fleet_backup" -or
        $verified.status -ne "deployment_committed" -or $verified.escalation_required -or $verified.in_doubt) {
        throw "deployment receipt verification failed"
    }
}
Invoke-ApplyPilotHbaReplacement -HbaPath $HbaPath -CandidateLines $candidateLines `
    -Database $Database -Role $Role -TailnetCidr $TailnetCidr -ReceiptPath $receiptFull `
    -RollbackSql $rollbackFull -PreflightBackup $preflightBackup -ReplaceBackup $replaceBackup `
    -CandidatePath $candidatePath -RestorePath $restorePath -ValidateAndReload $validateAndReload `
    -ReloadOriginal $reloadOriginal -FinalizeOutput $finalizeOutput

Write-Host "[pg-tailscale] mapped role and first-match pg_hba policy reconciled"
Write-Host "[pg-tailscale] receipt: $receiptFull"
Write-Host "[pg-tailscale] rollback SQL: $rollbackFull"
