# Invoke-FleetMakeCurrent.ps1 - pin this checkout and reconcile the reachable fleet.
#
# This is the normal operator command after publishing a tested fleet branch. It
# pins the current tree version in fleet_config, repairs/reconciles Windows boxes
# over Tailscale SSH, reconciles Paloma only when SSH is reachable, then runs the
# local health report. SSH remains the bootstrap/repair transport; the durable
# convergence path is the pinned tree version plus fleet-agent.ps1 -AutoUpdate.
[CmdletBinding()]
param(
  [string]$PublicBranch = "codex/fleet-applier-hardening",
  [string]$MacBranch = "applypilot-hardening-and-brainstorm-integration",
  [string]$Dsn = $env:FLEET_PG_DSN,
  [switch]$SkipPaloma,
  [switch]$RequirePaloma,
  [int]$SshTimeoutSeconds = 20,
  [int]$RemoteCommandTimeoutSeconds = 700
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
if (-not $repo) { $repo = Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location $repo

if (-not $Dsn) {
  throw "No fleet DSN: pass -Dsn or set FLEET_PG_DSN to an explicit mapped fleet DSN (never the postgres admin or the shared fleet_worker role). Refusing to pin/reconcile the fleet without it."
}
if ($Dsn -match '(?i)\buser\s*=\s*(?:postgres|fleet_worker)\b') {
  throw "Invoke-FleetMakeCurrent: fleet DSN selects an admin/shared principal. Supply an explicit mapped fleet DSN."
}

function Write-Section([string]$Title) {
  Write-Host ""
  Write-Host "================================================================================" -ForegroundColor DarkCyan
  Write-Host $Title -ForegroundColor Cyan
  Write-Host "================================================================================" -ForegroundColor DarkCyan
}

function Resolve-RepoPython {
  foreach ($candidate in @(".\.conda-env\python.exe", ".\.venv\Scripts\python.exe", "python.exe")) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
      return (Get-Command $candidate).Source
    }
    if (Test-Path $candidate) {
      return (Resolve-Path $candidate).Path
    }
  }
  throw "No repo Python found (.conda-env, .venv, or python.exe)."
}

function Invoke-Checked([string]$Label, [scriptblock]$Command) {
  Write-Section $Label
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "$Label failed with exit $LASTEXITCODE"
  }
}

function Set-FleetPinnedVersion([string]$PythonPath) {
  $env:APPLYPILOT_FLEET_DSN = $Dsn
  $env:FLEET_PG_DSN = $Dsn
  $code = @'
from applypilot.apply import pgqueue
from applypilot.fleet.config import set_pinned_version
from applypilot.fleet.software_version import current_sw_version

version = current_sw_version()
if not version.startswith("0.3.0+git.tree"):
    raise SystemExit(f"refusing to pin non-tree software version: {version}")
if version.endswith(".dirty"):
    raise SystemExit(f"refusing to pin dirty working tree version: {version}")
with pgqueue.connect(None) as conn:
    set_pinned_version(conn, version)
print(version)
'@
  $version = ($code | & $PythonPath -).Trim()
  if ($LASTEXITCODE -ne 0) {
    throw "set_pinned_version failed"
  }
  Write-Host "Pinned fleet to $version" -ForegroundColor Green
  return $version
}

function Test-PalomaSsh {
  try {
    return [bool](Test-NetConnection -ComputerName palomas-macbook-air -Port 22 -InformationLevel Quiet -WarningAction SilentlyContinue)
  } catch {
    return $false
  }
}

$python = Resolve-RepoPython
Write-Section "Pin Fleet Version"
$version = Set-FleetPinnedVersion $python
if ($version -notlike "0.3.0+git.tree*") {
  throw "Pinned version must use 0.3.0+git.tree identity, got $version"
}

$reconcile = Join-Path $repo "Invoke-FleetReconcile.ps1"
if (-not (Test-Path $reconcile)) { throw "Missing $reconcile" }

Invoke-Checked "Reconcile Windows Fleet" {
  & $reconcile -Only Tarpon,GGGTower -Apply -Branch $PublicBranch -SshTimeoutSeconds $SshTimeoutSeconds -RemoteCommandTimeoutSeconds $RemoteCommandTimeoutSeconds
}

if (-not $SkipPaloma) {
  if (Test-PalomaSsh) {
    Invoke-Checked "Reconcile Paloma" {
      & $reconcile -Only Paloma -Apply -Branch $MacBranch -SshTimeoutSeconds $SshTimeoutSeconds -RemoteCommandTimeoutSeconds $RemoteCommandTimeoutSeconds
    }
  } else {
    Write-Warning "Paloma is not reachable over Tailscale SSH; skipping Mac reconcile."
    if ($RequirePaloma) {
      throw "Paloma is not reachable and -RequirePaloma was passed."
    }
  }
}

$health = Join-Path $repo "fleet-health.ps1"
if (Test-Path $health) {
  Invoke-Checked "Fleet Health" {
    & $health -SkipRemote
  }
}

Write-Section "Done"
Write-Host "Fleet make-current finished for pinned version $version." -ForegroundColor Green
