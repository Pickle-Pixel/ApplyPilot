[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [Parameter(Mandatory = $true)][string]$PgHost,
    [Parameter(Mandatory = $true)][int]$PgPort,
    [Parameter(Mandatory = $true)][string]$PgDb,
    [Parameter(Mandatory = $true)][string]$PgUser,
    [Parameter(Mandatory = $true)][string]$PgPassFile,
    [Parameter(Mandatory = $true)][ValidateSet('verify-full')][string]$PgSslMode,
    [Parameter(Mandatory = $true)][string]$PgSslRootCert,
    [Parameter(Mandatory = $true)][string]$ExpectedPgSslRootCertSha256,
    [Parameter(Mandatory = $true)][string]$ExpectedSystemIdentifier,
    [Parameter(Mandatory = $true)][uint32]$ExpectedDatabaseOid,
    [Parameter(Mandatory = $true)][guid]$ExpectedDatabaseIncarnationId,
    [Parameter(Mandatory = $true)][string]$PgDumpPath,
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
    if ($item.PSIsContainer -or (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) { throw "PgPassFile must be a regular non-reparse file." }
    $acl = Get-Acl -LiteralPath $resolved
    if (-not $acl.AreAccessRulesProtected) { throw "PgPassFile ACL must not inherit permissions." }
    $currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $allowed = @($currentSid,"S-1-5-18","S-1-5-32-544")
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
    $acl.SetOwner($currentSid); $acl.SetAccessRuleProtection($true,$false)
    foreach ($sidText in @($currentSid.Value,'S-1-5-18','S-1-5-32-544')) {
        $rule = [Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new($sidText),[Security.AccessControl.FileSystemRights]::FullControl,[Security.AccessControl.AccessControlType]::Allow)
        [void]$acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $Path -AclObject $acl
}

function Assert-ProtectedDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-FullyQualifiedPath -Path $Path)) { throw "output directory path must be absolute." }
    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    $item = Get-Item -LiteralPath $resolved -Force
    if (-not $item.PSIsContainer -or (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) { throw "output directory must be non-reparse." }
    $acl = Get-Acl -LiteralPath $resolved
    if (-not $acl.AreAccessRulesProtected) { throw "output directory ACL must be protected before first byte is written." }
    $currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $allowed = @($currentSid,'S-1-5-18','S-1-5-32-544')
    $owner = $acl.Owner
    try { $owner = ([Security.Principal.NTAccount]$acl.Owner).Translate([Security.Principal.SecurityIdentifier]).Value } catch {}
    if ($owner -notin $allowed) { throw "output directory owner is outside the protected operator set." }
    foreach ($rule in $acl.Access) {
        if ($rule.AccessControlType -eq [Security.AccessControl.AccessControlType]::Allow -and $rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value -notin $allowed) { throw "output directory grants access outside the protected operator set." }
    }
    return $resolved
}

if ($PgUser -cne 'fleet_backup') { throw "PgUser must be the fixed dedicated fleet_backup login." }
if ([string]::IsNullOrWhiteSpace($PgHost) -or $PgHost -match '[\r\n]') { throw "PgHost is invalid." }
if ($PgPort -lt 1 -or $PgPort -gt 65535) { throw "PgPort is invalid." }
if ($PgDb -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') { throw "PgDb is invalid." }
if ($ExpectedSystemIdentifier -notmatch '^[1-9][0-9]{5,24}$') { throw "ExpectedSystemIdentifier is invalid." }
if (-not (Test-FullyQualifiedPath -Path $PgSslRootCert) -or -not (Test-Path -LiteralPath $PgSslRootCert -PathType Leaf)) { throw "PgSslRootCert must be an absolute file for verify-full TLS." }
$PgSslRootCert = Assert-ProtectedCredentialFile -Path $PgSslRootCert
if ($ExpectedPgSslRootCertSha256 -notmatch '^[0-9a-f]{64}$' -or (Get-FileHash -Algorithm SHA256 -LiteralPath $PgSslRootCert).Hash.ToLowerInvariant() -cne $ExpectedPgSslRootCertSha256) { throw "TLS root certificate hash pin mismatch." }
$protectedPassFile = Assert-ProtectedCredentialFile -Path $PgPassFile
if (-not (Test-Path -LiteralPath $PgDumpPath -PathType Leaf)) { throw "pg_dump was not found." }
$PgDumpPath = (Resolve-Path -LiteralPath $PgDumpPath).Path
$toolDir = Split-Path -Parent $PgDumpPath
$pgRestore = Join-Path $toolDir "pg_restore.exe"
$psql = Join-Path $toolDir "psql.exe"
foreach ($tool in @($pgRestore,$psql)) { if (-not (Test-Path -LiteralPath $tool -PathType Leaf)) { throw "Required PostgreSQL tool is missing beside the approved pg_dump." } }
$pgRestore = (Resolve-Path -LiteralPath $pgRestore -ErrorAction Stop).Path
$psql = (Resolve-Path -LiteralPath $psql -ErrorAction Stop).Path
foreach ($digest in @($ExpectedPgDumpSha256,$ExpectedPsqlSha256,$ExpectedPgRestoreSha256,$ExpectedScriptSha256)) { if ($digest -notmatch '^[0-9a-f]{64}$') { throw "Expected SHA-256 pin is invalid." } }
foreach ($commit in @($ExpectedRepoHead,$ExpectedRepoTree)) { if ($commit -notmatch '^[0-9a-f]{40}$') { throw "Expected Git pin is invalid." } }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $PgDumpPath).Hash.ToLowerInvariant() -cne $ExpectedPgDumpSha256) { throw "pg_dump hash pin mismatch." }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $psql).Hash.ToLowerInvariant() -cne $ExpectedPsqlSha256) { throw "psql hash pin mismatch." }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $pgRestore).Hash.ToLowerInvariant() -cne $ExpectedPgRestoreSha256) { throw "pg_restore hash pin mismatch." }
$version = (& $PgDumpPath --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $version -notmatch '^pg_dump \(PostgreSQL\) 18(?:\.|$)') { throw "pg_dump must be approved PostgreSQL major 18." }
$version = (& $psql --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $version -notmatch '^psql \(PostgreSQL\) 18(?:\.|$)') { throw "psql must be approved PostgreSQL major 18." }
$version = (& $pgRestore --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $version -notmatch '^pg_restore \(PostgreSQL\) 18(?:\.|$)') { throw "pg_restore must be approved PostgreSQL major 18." }
$repo = (Resolve-Path -LiteralPath $RepoPath -ErrorAction Stop).Path
if ((& git -C $repo rev-parse 'HEAD^{commit}').Trim() -cne $ExpectedRepoHead -or (& git -C $repo rev-parse 'HEAD^{tree}').Trim() -cne $ExpectedRepoTree) { throw "repository exact-head pin mismatch." }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $MyInvocation.MyCommand.Path).Hash.ToLowerInvariant() -cne $ExpectedScriptSha256) { throw "backup script hash pin mismatch." }
foreach ($name in @('PGHOST','PGPORT','PGDATABASE','PGUSER','PGPASSWORD','PGPASSFILE','PGSERVICE','PGSERVICEFILE','PGOPTIONS','PGSSLMODE','PGSSLROOTCERT','PGSSLCERT','PGSSLKEY','PGTARGETSESSIONATTRS','PGCHANNELBINDING','PGCONNECT_TIMEOUT')) {
    Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
}
$env:PGPASSFILE = $protectedPassFile
$env:PGSSLMODE = $PgSslMode
if ($PgSslRootCert) { $env:PGSSLROOTCERT = $PgSslRootCert }
$connectionArgs = @('-h',$PgHost,'-p',$PgPort.ToString(),'-U','fleet_backup','-d',$PgDb)

function Assert-RailwayBackupTarget {
    $sql = @"
SELECT concat_ws(E'\t',session_user,current_user,current_database(),(pg_control_system()).system_identifier::text,
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
    if ($LASTEXITCODE -ne 0) { throw "Railway backup target preflight failed." }
    $fields = (($result | Out-String).Trim()) -split "`t"
    $incarnation = $ExpectedDatabaseIncarnationId.ToString().ToLowerInvariant()
    if ($fields.Count -ne 14 -or $fields[0] -cne 'fleet_backup' -or $fields[1] -cne 'fleet_backup' -or $fields[2] -cne $PgDb -or
        $fields[3] -cne $ExpectedSystemIdentifier -or $fields[4] -cne $ExpectedDatabaseOid.ToString() -or $fields[5] -cne $incarnation -or
        ($fields[6..13] | Where-Object { $_ -cne 'true' }).Count -ne 0) { throw "Railway backup target violates the exact fixed backup contract." }
}

Assert-RailwayBackupTarget
$parent = Split-Path -Parent $OutputPath
if (-not $parent) { throw "OutputPath must include a parent directory." }
$resolvedParent = Assert-ProtectedDirectory -Path $parent
$resolvedOutput = Join-Path $resolvedParent (Split-Path -Leaf $OutputPath)
$partial = Join-Path $resolvedParent (".railway-backup-{0}.partial" -f [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $resolvedOutput) { throw "backup destination already exists." }
try {
    $reserve=[IO.File]::Open($partial,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try {$reserve.Flush($true)} finally {$reserve.Dispose()}
    Set-ProtectedFileAcl -Path $partial
    & $PgDumpPath @connectionArgs --format=custom --no-password --file=$partial
    if ($LASTEXITCODE -ne 0) { throw "pg_dump failed with exit $LASTEXITCODE." }
    $item = Get-Item -LiteralPath $partial
    if ($item.Length -le 0) { throw "pg_dump produced an empty file." }
    & $pgRestore --list $partial | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "pg_restore could not read the generated archive." }
    Set-ProtectedFileAcl -Path $partial
    $flush=[IO.File]::Open($partial,[IO.FileMode]::Open,[IO.FileAccess]::ReadWrite,[IO.FileShare]::Read)
    try {$flush.Flush($true)} finally {$flush.Dispose()}
    [IO.File]::Move($partial,$resolvedOutput)
    $final = Get-Item -LiteralPath $resolvedOutput
    Assert-ProtectedCredentialFile -Path $resolvedOutput | Out-Null
    $stream = [IO.File]::OpenRead($resolvedOutput)
    try { $sha256=[Security.Cryptography.SHA256]::Create(); try {$digest=$sha256.ComputeHash($stream)} finally {$sha256.Dispose()} } finally {$stream.Dispose()}
    [pscustomobject]@{path=$final.FullName;bytes=$final.Length;sha256=[BitConverter]::ToString($digest).Replace('-','').ToLowerInvariant()} | ConvertTo-Json -Compress
} finally { Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue }
