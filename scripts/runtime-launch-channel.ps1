Set-StrictMode -Version Latest

function Get-ApplyPilotRuntimePython {
  param([Parameter(Mandatory = $true)][string]$RepoRoot)
  foreach ($candidate in @(
    (Join-Path $RepoRoot ".conda-env\python.exe"),
    (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
    (Join-Path $RepoRoot ".conda-env\Scripts\python.exe")
  )) {
    if (Test-Path -LiteralPath $candidate) { return $candidate }
  }
  throw "No project Python is available for Runtime launch-channel validation."
}

function Assert-ApplyPilotRuntimeChannel {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [string]$OverrideDsn,
    [switch]$RequirePersistedUserChannel
  )

  $userDsn = [Environment]::GetEnvironmentVariable("FLEET_PG_DSN", "User")
  $userPrincipal = [Environment]::GetEnvironmentVariable("APPLYPILOT_PG_PRINCIPAL", "User")
  if ($RequirePersistedUserChannel) {
    if (-not $userDsn -or -not $userPrincipal) {
      throw "The persisted User Runtime channel is incomplete; run mapped-role worker setup before registration."
    }
    if ($OverrideDsn -and -not [string]::Equals($OverrideDsn, $userDsn, [StringComparison]::Ordinal)) {
      throw "-Dsn does not match the exact persisted User Runtime channel that the future task will inherit."
    }
    $effectiveDsn = $userDsn
    $effectivePrincipal = $userPrincipal
    $channel = "User"
  } else {
    $effectiveDsn = if ($OverrideDsn) { $OverrideDsn } else { $env:FLEET_PG_DSN }
    $effectivePrincipal = if ($env:APPLYPILOT_PG_PRINCIPAL) {
      $env:APPLYPILOT_PG_PRINCIPAL
    } else {
      $userPrincipal
    }
    $channel = "Process"
    if (-not $effectiveDsn -or -not $effectivePrincipal) {
      throw "The current Runtime channel is incomplete; FLEET_PG_DSN and APPLYPILOT_PG_PRINCIPAL are required."
    }
  }

  $python = Get-ApplyPilotRuntimePython -RepoRoot $RepoRoot
  $previousDsn = $env:FLEET_PG_DSN
  $previousPrincipal = $env:APPLYPILOT_PG_PRINCIPAL
  $previousPythonPath = $env:PYTHONPATH
  try {
    $env:FLEET_PG_DSN = $effectiveDsn
    $env:APPLYPILOT_PG_PRINCIPAL = $effectivePrincipal
    $env:PYTHONPATH = Join-Path $RepoRoot "src"
    & $python -m applypilot.fleet.launch_guard
    if ($LASTEXITCODE -ne 0) {
      throw "The exact $channel Runtime channel failed offline principal validation."
    }
  } finally {
    if ($null -eq $previousDsn) { Remove-Item Env:FLEET_PG_DSN -ErrorAction SilentlyContinue }
    else { $env:FLEET_PG_DSN = $previousDsn }
    if ($null -eq $previousPrincipal) { Remove-Item Env:APPLYPILOT_PG_PRINCIPAL -ErrorAction SilentlyContinue }
    else { $env:APPLYPILOT_PG_PRINCIPAL = $previousPrincipal }
    if ($null -eq $previousPythonPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
    else { $env:PYTHONPATH = $previousPythonPath }
  }

  return [pscustomobject]@{
    Dsn = $effectiveDsn
    Principal = $effectivePrincipal
    Channel = $channel
  }
}

function Write-ApplyPilotLaunchRolloutEvidence {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)]$RuntimeChannel,
    [Parameter(Mandatory = $true)][string[]]$WrapperPath,
    [string[]]$TaskName = @(),
    [string[]]$ShortcutPath = @()
  )

  $wrapperEvidence = foreach ($item in $WrapperPath) {
    if (-not (Test-Path -LiteralPath $item -PathType Leaf)) {
      throw "Expected generated wrapper is missing: $item"
    }
    $content = [IO.File]::ReadAllText($item)
    if ($content.Contains([string]$RuntimeChannel.Dsn)) {
      throw "Generated wrapper contains the Runtime DSN and was not accepted."
    }
    [ordered]@{
      path = [IO.Path]::GetFullPath($item)
      sha256 = (Get-FileHash -LiteralPath $item -Algorithm SHA256).Hash.ToLowerInvariant()
      dsn_embedded = $false
    }
  }

  $taskEvidence = foreach ($name in $TaskName) {
    $task = Get-ScheduledTask -TaskName $name -ErrorAction Stop
    $actionText = @($task.Actions | ForEach-Object {
      ([string]$_.Execute) + "`n" + ([string]$_.Arguments)
    }) -join "`n"
    if ($actionText.Contains([string]$RuntimeChannel.Dsn)) {
      throw "Scheduled task action contains the Runtime DSN and was not accepted."
    }
    [ordered]@{ name = $name; dsn_embedded = $false }
  }

  $shortcutEvidence = foreach ($item in $ShortcutPath) {
    if (-not (Test-Path -LiteralPath $item -PathType Leaf)) {
      throw "Expected Startup shortcut is missing: $item"
    }
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($item)
    $shortcutText = ([string]$shortcut.TargetPath) + "`n" + ([string]$shortcut.Arguments)
    if ($shortcutText.Contains([string]$RuntimeChannel.Dsn)) {
      throw "Startup shortcut contains the Runtime DSN and was not accepted."
    }
    [ordered]@{
      path = [IO.Path]::GetFullPath($item)
      sha256 = (Get-FileHash -LiteralPath $item -Algorithm SHA256).Hash.ToLowerInvariant()
      dsn_embedded = $false
    }
  }

  $receipt = [ordered]@{
    schema = "applypilot.runtime-launch-rollout.v1"
    channel = [string]$RuntimeChannel.Channel
    principal = [string]$RuntimeChannel.Principal
    generated_at = [DateTimeOffset]::UtcNow.ToString("o")
    wrappers = @($wrapperEvidence)
    tasks = @($taskEvidence)
    shortcuts = @($shortcutEvidence)
  }
  $parent = Split-Path -Parent $Path
  if (-not (Test-Path -LiteralPath $parent)) {
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
  }
  [IO.File]::WriteAllText(
    $Path,
    (($receipt | ConvertTo-Json -Depth 8) + "`n"),
    [Text.UTF8Encoding]::new($false)
  )
}
