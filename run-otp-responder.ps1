# run-otp-responder.ps1 [-Supervise] [-Dsn "..."] [-MachineOwner home]
# Starts the home-box OTP responder with quoting that survives Windows process
# launch boundaries. Use -Supervise for logon/startup persistence.
param(
  [Parameter(Mandatory = $true)][string]$Dsn,
  [string]$MachineOwner = "home",
  [switch]$Supervise,
  [int]$RestartDelaySeconds = 10
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# Fail closed on an admin/shared principal: the OTP responder must run under an explicit
# mapped fleet DSN, never the postgres admin or the shared fleet_worker role.
if ($Dsn -match '(?i)\buser\s*=\s*(?:postgres|fleet_worker)\b') {
  throw "run-otp-responder: -Dsn selects an admin/shared principal. Supply an explicit mapped fleet DSN."
}

$LogDir = Join-Path $ProjectRoot ".fleet-logs"
if (-not (Test-Path -LiteralPath $LogDir)) {
  New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}
$OutLog = Join-Path $LogDir "otp-responder.out.log"
$ErrLog = Join-Path $LogDir "otp-responder.err.log"

$env:FLEET_PG_DSN = $Dsn
$env:APPLYPILOT_FLEET_DSN = $Dsn
$env:FLEET_MACHINE_OWNER = $MachineOwner
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Resolve-OtpResponderExe {
  foreach ($d in @(".\.conda-env\Scripts", ".\.venv\Scripts")) {
    $cand = Join-Path $ProjectRoot (Join-Path $d "applypilot-fleet-otp-home.exe")
    if (Test-Path -LiteralPath $cand) {
      return (Resolve-Path -LiteralPath $cand).Path
    }
  }
  throw "applypilot-fleet-otp-home.exe not found in .conda-env or .venv -- run 'pip install -e .' first."
}

function Stop-StaleOtpResponderProcesses {
  $self = $PID
  Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
      $_.ProcessId -ne $self -and $_.CommandLine -and
      $_.CommandLine -like "*$ProjectRoot*" -and
      ($_.CommandLine -like "*applypilot-fleet-otp-home.exe*" -or
       $_.CommandLine -like "*run-otp-responder.ps1*" -or
       $_.CommandLine -like "*otp-responder-task.ps1*")
    } |
    ForEach-Object {
      Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

$Exe = Resolve-OtpResponderExe
Stop-StaleOtpResponderProcesses

if ($Supervise) {
  while ($true) {
    & $Exe --dsn $Dsn --machine-owner $MachineOwner >> $OutLog 2>> $ErrLog
    $exit = $LASTEXITCODE
    Add-Content -LiteralPath $ErrLog -Value "[run-otp-responder] responder exited code=$exit; restarting in $RestartDelaySeconds seconds"
    Start-Sleep -Seconds ([Math]::Max(1, $RestartDelaySeconds))
  }
}

$ArgumentList = "--dsn `"$Dsn`" --machine-owner `"$MachineOwner`""
$process = Start-Process -FilePath $Exe -WindowStyle Hidden -ArgumentList $ArgumentList `
  -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog `
  -WorkingDirectory $ProjectRoot -PassThru
Write-Host "[run-otp-responder] started pid=$($process.Id)"
