# Nightly fleet PostgreSQL backup through the fixed least-privilege fleet_backup login.
# This script never provisions roles. Before registration, an operator must create
# fleet_backup LOGIN NOINHERIT with CONNECTION LIMIT 1; grant only database CONNECT,
# application-schema USAGE, and SELECT on all application tables/sequences; configure
# matching ALTER DEFAULT PRIVILEGES for brain_schema_migrator; and grant no ownership,
# role membership, write/DDL privilege, or elevated attribute.
param(
    [int]$KeepBackups = 14,
    [switch]$RegisterTask,
    [string]$BackupDir = (Join-Path $env:LOCALAPPDATA "ApplyPilot\pg-backups"),
    [Parameter(Mandatory = $true)][string]$PgHost,
    [Parameter(Mandatory = $true)][int]$PgPort,
    [Parameter(Mandatory = $true)][string]$PgDb,
    [Parameter(Mandatory = $true)][string]$PgUser,
    [Parameter(Mandatory = $true)][string]$PgPassFile,
    [Parameter(Mandatory = $true)][string]$PgDumpPath,
    [Parameter(Mandatory = $true)][ValidateSet('disable','require','verify-ca','verify-full')][string]$PgSslMode,
    [string]$PgSslRootCert,
    [Parameter(Mandatory = $true)][string]$ExpectedSystemIdentifier,
    [Parameter(Mandatory = $true)][uint32]$ExpectedDatabaseOid,
    [Parameter(Mandatory = $true)][guid]$ExpectedDatabaseIncarnationId,
    [Parameter(Mandatory = $true)][string]$ExpectedPgDumpSha256,
    [Parameter(Mandatory = $true)][string]$ExpectedPsqlSha256,
    [Parameter(Mandatory = $true)][string]$ExpectedPgRestoreSha256,
    [Parameter(Mandatory = $true)][string]$ExpectedRepoHead,
    [Parameter(Mandatory = $true)][string]$ExpectedRepoTree,
    [Parameter(Mandatory = $true)][string]$ExpectedScriptSha256,
    [Parameter(Mandatory = $true)][string]$RepoPath
)
$ErrorActionPreference = "Stop"

function Test-FullyQualifiedPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    try { $root = [IO.Path]::GetPathRoot($Path) } catch { return $false }
    return ($root -match '^[A-Za-z]:[\\/]$' -or $root -match '^\\\\[^\\]+\\[^\\]+[\\/]?$')
}

function Assert-ProtectedCredentialFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-FullyQualifiedPath -Path $Path)) { throw "PgPassFile path must be absolute." }
    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    $item = Get-Item -LiteralPath $resolved -Force
    if ($item.PSIsContainer -or (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
        throw "PgPassFile must be a regular non-reparse file."
    }
    $acl = Get-Acl -LiteralPath $resolved
    if (-not $acl.AreAccessRulesProtected) { throw "PgPassFile ACL must not inherit permissions." }
    $currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $allowed = @($currentSid, "S-1-5-18", "S-1-5-32-544")
    $owner = $acl.Owner
    try { $owner = ([Security.Principal.NTAccount]$acl.Owner).Translate([Security.Principal.SecurityIdentifier]).Value } catch {}
    if ($owner -notin $allowed) { throw "PgPassFile owner is outside the protected operator set." }
    foreach ($rule in $acl.Access) {
        if ($rule.AccessControlType -ne [Security.AccessControl.AccessControlType]::Allow) { continue }
        try { $sid = $rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value }
        catch { throw "PgPassFile has an unresolvable allowed ACL identity." }
        if ($sid -notin $allowed) { throw "PgPassFile grants access outside the protected operator set." }
    }
    return $resolved
}

function Set-ProtectedFileAcl {
    param([Parameter(Mandatory = $true)][string]$Path)
    $currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User
    $acl = [Security.AccessControl.FileSecurity]::new()
    $acl.SetOwner($currentSid)
    $acl.SetAccessRuleProtection($true,$false)
    foreach ($sidText in @($currentSid.Value,'S-1-5-18','S-1-5-32-544')) {
        $sid = [Security.Principal.SecurityIdentifier]::new($sidText)
        $rule = [Security.AccessControl.FileSystemAccessRule]::new($sid,[Security.AccessControl.FileSystemRights]::FullControl,[Security.AccessControl.AccessControlType]::Allow)
        [void]$acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $Path -AclObject $acl
}

function Assert-ProtectedDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-FullyQualifiedPath -Path $Path)) { throw "directory path must be absolute." }
    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    $item = Get-Item -LiteralPath $resolved -Force
    if (-not $item.PSIsContainer -or (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) { throw "directory must be non-reparse." }
    $acl = Get-Acl -LiteralPath $resolved
    if (-not $acl.AreAccessRulesProtected) { throw "directory ACL must be protected before first byte is written." }
    $currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $allowed = @($currentSid,'S-1-5-18','S-1-5-32-544')
    $owner = $acl.Owner
    try { $owner = ([Security.Principal.NTAccount]$acl.Owner).Translate([Security.Principal.SecurityIdentifier]).Value } catch {}
    if ($owner -notin $allowed) { throw "directory owner is outside the protected operator set." }
    foreach ($rule in $acl.Access) {
        if ($rule.AccessControlType -ne [Security.AccessControl.AccessControlType]::Allow) { continue }
        $sid = $rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
        if ($sid -notin $allowed) { throw "directory grants access outside the protected operator set." }
    }
    return $resolved
}

function Get-Sha256Hex {
    param([Parameter(Mandatory = $true)][string]$Path)
    $stream = [IO.File]::Open(
        $Path,
        [IO.FileMode]::Open,
        [IO.FileAccess]::Read,
        [IO.FileShare]::Read
    )
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha256.ComputeHash($stream))).Replace('-','').ToLowerInvariant()
    } finally {
        $sha256.Dispose()
        $stream.Dispose()
    }
}

function Assert-PinnedRuntime {
    param(
        [Parameter(Mandatory = $true)][string]$PgDump,
        [Parameter(Mandatory = $true)][string]$Psql,
        [Parameter(Mandatory = $true)][string]$PgRestore
    )
    foreach ($digest in @($ExpectedPgDumpSha256,$ExpectedPsqlSha256,$ExpectedPgRestoreSha256,$ExpectedScriptSha256)) {
        if ($digest -notmatch '^[0-9a-f]{64}$') { throw "Expected SHA-256 pin is invalid." }
    }
    foreach ($commit in @($ExpectedRepoHead,$ExpectedRepoTree)) {
        if ($commit -notmatch '^[0-9a-f]{40}$') { throw "Expected Git pin is invalid." }
    }
    if ((Get-Sha256Hex -Path $PgDump) -cne $ExpectedPgDumpSha256) { throw "pg_dump hash pin mismatch." }
    if ((Get-Sha256Hex -Path $Psql) -cne $ExpectedPsqlSha256) { throw "psql hash pin mismatch." }
    if ((Get-Sha256Hex -Path $PgRestore) -cne $ExpectedPgRestoreSha256) { throw "pg_restore hash pin mismatch." }
    $version = (& $PgDump --version 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $version -notmatch '^pg_dump \(PostgreSQL\) 18(?:\.|$)') { throw "pg_dump must be approved PostgreSQL major 18." }
    $version = (& $Psql --version 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $version -notmatch '^psql \(PostgreSQL\) 18(?:\.|$)') { throw "psql must be approved PostgreSQL major 18." }
    $version = (& $PgRestore --version 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $version -notmatch '^pg_restore \(PostgreSQL\) 18(?:\.|$)') { throw "pg_restore must be approved PostgreSQL major 18." }
    $repo = (Resolve-Path -LiteralPath $RepoPath -ErrorAction Stop).Path
    if ((& git -C $repo rev-parse 'HEAD^{commit}').Trim() -cne $ExpectedRepoHead -or
        (& git -C $repo rev-parse 'HEAD^{tree}').Trim() -cne $ExpectedRepoTree) { throw "repository exact-head pin mismatch." }
    if ((Get-Sha256Hex -Path $PSCommandPath) -cne $ExpectedScriptSha256) { throw "backup script hash pin mismatch." }
}

if ($PgUser -cne 'fleet_backup') { throw "PgUser must be the fixed dedicated fleet_backup login." }
if ($KeepBackups -lt 1 -or $KeepBackups -gt 365) { throw "KeepBackups must be between 1 and 365." }
if ([string]::IsNullOrWhiteSpace($PgHost) -or $PgHost -match '[\r\n]') { throw "PgHost is invalid." }
if ($PgPort -lt 1 -or $PgPort -gt 65535) { throw "PgPort is invalid." }
if ($PgDb -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') { throw "PgDb is invalid." }
if ($ExpectedSystemIdentifier -notmatch '^[1-9][0-9]{5,24}$') { throw "ExpectedSystemIdentifier is invalid." }
if ($PgSslMode -in @('verify-ca','verify-full')) {
    if (-not (Test-FullyQualifiedPath -Path $PgSslRootCert) -or -not (Test-Path -LiteralPath $PgSslRootCert -PathType Leaf)) { throw "PgSslRootCert must be an absolute file for verified TLS." }
    $PgSslRootCert = (Resolve-Path -LiteralPath $PgSslRootCert).Path
}
$protectedPassFile = Assert-ProtectedCredentialFile -Path $PgPassFile
$BackupDir = Assert-ProtectedDirectory -Path $BackupDir
$pgDump = (Resolve-Path -LiteralPath $PgDumpPath -ErrorAction Stop).Path
$toolDir = Split-Path -Parent $pgDump
$psql = Join-Path $toolDir 'psql.exe'
$pgRestore = Join-Path $toolDir 'pg_restore.exe'
foreach ($tool in @($psql,$pgRestore)) {
    if (-not (Test-Path -LiteralPath $tool -PathType Leaf)) { throw "Required PostgreSQL tool is missing beside the approved pg_dump." }
}
$psql = (Resolve-Path -LiteralPath $psql -ErrorAction Stop).Path
$pgRestore = (Resolve-Path -LiteralPath $pgRestore -ErrorAction Stop).Path
Assert-PinnedRuntime -PgDump $pgDump -Psql $psql -PgRestore $pgRestore

if ($RegisterTask) {
    $script = $MyInvocation.MyCommand.Path
    foreach ($value in @($script,$protectedPassFile,$PgHost,$pgDump,$PgSslRootCert,$RepoPath,$BackupDir)) {
        if ($value -match '["\r\n]') { throw "Scheduled-task argument contains unsupported characters." }
    }
    $arguments = "-KeepBackups $KeepBackups -BackupDir `"$BackupDir`" " +
        "-PgHost `"$PgHost`" -PgPort $PgPort -PgDb $PgDb -PgUser fleet_backup " +
        "-PgPassFile `"$protectedPassFile`" -PgDumpPath `"$pgDump`" -PgSslMode $PgSslMode " +
        $(if ($PgSslRootCert) { "-PgSslRootCert `"$PgSslRootCert`" " } else { "" }) +
        "-ExpectedSystemIdentifier $ExpectedSystemIdentifier -ExpectedDatabaseOid $ExpectedDatabaseOid " +
        "-ExpectedDatabaseIncarnationId $($ExpectedDatabaseIncarnationId.ToString()) " +
        "-ExpectedPgDumpSha256 $ExpectedPgDumpSha256 -ExpectedPsqlSha256 $ExpectedPsqlSha256 " +
        "-ExpectedPgRestoreSha256 $ExpectedPgRestoreSha256 -ExpectedRepoHead $ExpectedRepoHead " +
        "-ExpectedRepoTree $ExpectedRepoTree -ExpectedScriptSha256 $ExpectedScriptSha256 " +
        "-RepoPath `"$RepoPath`""
    $launcherDir = Assert-ProtectedDirectory -Path (Join-Path $env:LOCALAPPDATA "ApplyPilot\protected-launchers")
    $launcher = Join-Path $launcherDir "backup-fleet-pg.ps1"
    $installTemp = Join-Path $launcherDir (".backup-fleet-pg-{0}.tmp" -f [guid]::NewGuid().ToString('N'))
    $installBackup = Join-Path $launcherDir (".backup-fleet-pg-{0}.bak" -f [guid]::NewGuid().ToString('N'))
    $sourceBytes = [IO.File]::ReadAllBytes($script)
    try {
        $installStream = [IO.File]::Open($installTemp,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
        try { $installStream.Flush($true) } finally { $installStream.Dispose() }
        Set-ProtectedFileAcl -Path $installTemp
        $installStream = [IO.File]::Open($installTemp,[IO.FileMode]::Open,[IO.FileAccess]::Write,[IO.FileShare]::None)
        try { $installStream.Write($sourceBytes,0,$sourceBytes.Length); $installStream.Flush($true) } finally { $installStream.Dispose() }
        if (Test-Path -LiteralPath $launcher -PathType Leaf) {
            [IO.File]::Replace($installTemp,$launcher,$installBackup,$true)
        } else {
            # The two-argument overload is PowerShell 5.1 compatible and fails closed
            # if another process wins the destination-creation race.
            [IO.File]::Move($installTemp,$launcher)
        }
    } finally {
        Remove-Item -LiteralPath $installTemp -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $installBackup -Force -ErrorAction SilentlyContinue
    }
    Assert-ProtectedCredentialFile -Path $launcher | Out-Null
    if ((Get-Sha256Hex -Path $launcher) -cne $ExpectedScriptSha256) { throw "installed backup script hash mismatch." }
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$launcher`" $arguments"
    $trigger = New-ScheduledTaskTrigger -Daily -At 3:30am
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1)
    $principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value) -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName "ApplyPilot fleet PG backup" -Action $action -Trigger $trigger `
        -Settings $settings -Principal $principal -Description "Nightly least-privilege pg_dump of fleet PostgreSQL" -Force | Out-Null
    return
}

foreach ($name in @('PGHOST','PGPORT','PGDATABASE','PGUSER','PGPASSWORD','PGPASSFILE','PGSERVICE','PGSERVICEFILE','PGOPTIONS','PGSSLMODE','PGSSLROOTCERT','PGSSLCERT','PGSSLKEY','PGTARGETSESSIONATTRS','PGCHANNELBINDING','PGCONNECT_TIMEOUT')) {
    Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
}
$env:PGPASSFILE = $protectedPassFile
$env:PGSSLMODE = $PgSslMode
if ($PgSslRootCert) { $env:PGSSLROOTCERT = $PgSslRootCert }
$connectionArgs = @('-h',$PgHost,'-p',$PgPort.ToString(),'-U','fleet_backup','-d',$PgDb)

function Assert-BackupDatabasePrincipal {
    $sql = @"
SELECT concat_ws(E'\t',session_user,current_user,current_database(),
 (pg_control_system()).system_identifier::text,
 (SELECT oid::text FROM pg_database WHERE datname=current_database()),
 (SELECT database_incarnation_id::text FROM public.applypilot_database_identity WHERE singleton_id=1),
 (NOT (r.rolinherit OR r.rolsuper OR r.rolcreatedb OR r.rolcreaterole OR r.rolreplication OR r.rolbypassrls) AND r.rolconnlimit=1)::text,
 (NOT EXISTS (SELECT 1 FROM pg_auth_members WHERE member=r.oid OR roleid=r.oid))::text,
 (NOT EXISTS (SELECT 1 FROM pg_database WHERE datdba=r.oid) AND NOT EXISTS (SELECT 1 FROM pg_namespace WHERE nspowner=r.oid) AND NOT EXISTS (SELECT 1 FROM pg_class WHERE relowner=r.oid) AND NOT EXISTS (SELECT 1 FROM pg_proc WHERE proowner=r.oid) AND NOT EXISTS (SELECT 1 FROM pg_type WHERE typowner=r.oid))::text,
 (has_database_privilege(current_user,current_database(),'CONNECT') AND NOT has_database_privilege(current_user,current_database(),'CREATE'))::text,
 (NOT EXISTS (SELECT 1 FROM pg_namespace n WHERE n.nspname !~ '^pg_' AND n.nspname<>'information_schema' AND (NOT has_schema_privilege(current_user,n.oid,'USAGE') OR has_schema_privilege(current_user,n.oid,'CREATE'))))::text,
 (EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname !~ '^pg_' AND n.nspname<>'information_schema' AND c.relkind IN ('r','p','m','f')) AND NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname !~ '^pg_' AND n.nspname<>'information_schema' AND c.relkind IN ('r','p','m','f') AND (c.relrowsecurity OR NOT has_table_privilege(current_user,c.oid,'SELECT') OR has_table_privilege(current_user,c.oid,'INSERT') OR has_table_privilege(current_user,c.oid,'UPDATE') OR has_table_privilege(current_user,c.oid,'DELETE') OR has_table_privilege(current_user,c.oid,'TRUNCATE') OR has_table_privilege(current_user,c.oid,'REFERENCES') OR has_table_privilege(current_user,c.oid,'TRIGGER'))) AND NOT EXISTS (SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname !~ '^pg_' AND n.nspname<>'information_schema' AND p.prosecdef AND has_function_privilege(current_user,p.oid,'EXECUTE')))::text,
 (NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname !~ '^pg_' AND n.nspname<>'information_schema' AND c.relkind='S' AND (NOT has_sequence_privilege(current_user,c.oid,'SELECT') OR has_sequence_privilege(current_user,c.oid,'USAGE') OR has_sequence_privilege(current_user,c.oid,'UPDATE'))))::text,
 (NOT EXISTS (SELECT 1 FROM pg_largeobject_metadata))::text)
FROM pg_roles r WHERE r.rolname=current_user;
"@
    $result = & $psql @connectionArgs -X -A -t --no-password -v ON_ERROR_STOP=1 -c $sql 2>&1
    if ($LASTEXITCODE -ne 0) { throw "backup-principal preflight failed." }
    $fields = (($result | Out-String).Trim()) -split "`t"
    $incarnation = $ExpectedDatabaseIncarnationId.ToString().ToLowerInvariant()
    if ($fields.Count -ne 14 -or $fields[0] -cne 'fleet_backup' -or $fields[1] -cne 'fleet_backup' -or
        $fields[2] -cne $PgDb -or $fields[3] -cne $ExpectedSystemIdentifier -or
        $fields[4] -cne $ExpectedDatabaseOid.ToString() -or $fields[5] -cne $incarnation -or
        ($fields[6..13] | Where-Object { $_ -cne 'true' }).Count -ne 0) {
        throw "backup principal or database identity violates the fixed least-privilege contract."
    }
}

Assert-BackupDatabasePrincipal
$freeGB = [math]::Round((Get-PSDrive ($BackupDir.Substring(0,1))).Free / 1GB, 1)
if ($freeGB -lt 2) { throw "Only ${freeGB}GB free; skipping backup." }
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$dest = Join-Path $BackupDir "applypilot_fleet-$stamp.dump"
$partial = Join-Path $BackupDir (".applypilot_fleet-{0}.partial" -f [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $dest) { throw "backup destination already exists." }
$reserve = [IO.File]::Open($partial,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try { $reserve.Flush($true) } finally { $reserve.Dispose() }
Set-ProtectedFileAcl -Path $partial
& $pgDump @connectionArgs -Fc --no-password -f $partial
if ($LASTEXITCODE -ne 0) {
    if (Test-Path $partial) { Remove-Item $partial -Force }
    throw "pg_dump failed (exit $LASTEXITCODE)."
}
$size = (Get-Item $partial).Length
if ($size -lt 10KB) { Remove-Item $partial -Force; throw "pg_dump output is implausibly small." }
& $pgRestore --list $partial | Out-Null
if ($LASTEXITCODE -ne 0) { Remove-Item $partial -Force; throw "pg_restore archive validation failed." }
Set-ProtectedFileAcl -Path $partial
$flush = [IO.File]::Open($partial,[IO.FileMode]::Open,[IO.FileAccess]::ReadWrite,[IO.FileShare]::Read)
try { $flush.Flush($true) } finally { $flush.Dispose() }
[IO.File]::Move($partial,$dest)
$artifactHash = Get-Sha256Hex -Path $dest
Assert-ProtectedCredentialFile -Path $dest | Out-Null
Get-ChildItem $BackupDir -Filter "applypilot_fleet-*.dump" | Sort-Object LastWriteTime -Descending |
    Select-Object -Skip $KeepBackups | Remove-Item -Force -ErrorAction SilentlyContinue
Write-Host ("[pg-backup] OK -> {0} ({1:N1} MB); sha256={2}; keeping newest {3}." -f $dest,($size/1MB),$artifactHash,$KeepBackups)
