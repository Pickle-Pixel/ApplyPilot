# register-otp-responder-startup.ps1
# Registers the home-box OTP responder. Task Scheduler is preferred, but this
# script falls back to the current user's Startup folder when registration is
# denied by local policy/elevation.
param(
  [switch]$NoStart
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Launcher = Join-Path $ProjectRoot "run-otp-responder.ps1"

. (Join-Path $ProjectRoot "scripts\runtime-launch-channel.ps1")
$runtimeChannel = Assert-ApplyPilotRuntimeChannel -RepoRoot $ProjectRoot -RequirePersistedUserChannel
$TaskName = "ApplyPilotFleet-OtpResponder"
$ShortcutName = "ApplyPilotFleet-OtpResponder.lnk"

if (-not (Test-Path -LiteralPath $Launcher)) {
  throw "OTP launcher not found: $Launcher"
}

# Idempotently remove any legacy action/shortcut that may still contain a historical DSN.
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
  Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
}
$legacyShortcut = Join-Path ([Environment]::GetFolderPath('Startup')) $ShortcutName
if (Test-Path -LiteralPath $legacyShortcut) {
  Remove-Item -LiteralPath $legacyShortcut -Force
}

function Install-StartupShortcut {
  $startup = [Environment]::GetFolderPath('Startup')
  if (-not (Test-Path -LiteralPath $startup)) {
    New-Item -ItemType Directory -Path $startup -Force | Out-Null
  }
  $shortcutPath = Join-Path $startup $ShortcutName
  $shell = New-Object -ComObject WScript.Shell
  $shortcut = $shell.CreateShortcut($shortcutPath)
  $shortcut.TargetPath = "powershell.exe"
  $shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Launcher`" -Supervise"
  $shortcut.WorkingDirectory = $ProjectRoot
  $shortcut.WindowStyle = 7
  $shortcut.Description = "ApplyPilot home OTP responder"
  $shortcut.Save()
  return $shortcutPath
}

function Register-OtpScheduledTask {
  $argument = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Launcher`" -Supervise"
  $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argument -WorkingDirectory $ProjectRoot
  $trigger = New-ScheduledTaskTrigger -AtLogOn
  $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 10 -RestartInterval (New-TimeSpan -Minutes 1)
  $taskUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
  $principal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
  Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description "ApplyPilot home OTP responder." -Force -ErrorAction Stop | Out-Null
}

try {
  Register-OtpScheduledTask
  Write-ApplyPilotLaunchRolloutEvidence `
    -Path (Join-Path $ProjectRoot ".fleet-logs\runtime-launch-rollout-otp.json") `
    -RuntimeChannel $runtimeChannel `
    -WrapperPath @($Launcher) `
    -TaskName @($TaskName)
  Write-Host "[otp-startup] registered Task Scheduler task $TaskName"
  if (-not $NoStart) {
    Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    Write-Host "[otp-startup] started Task Scheduler task $TaskName"
  }
} catch {
  Write-Host "[otp-startup] Task Scheduler registration denied; installing Startup shortcut fallback: $($_.Exception.Message)" -ForegroundColor Yellow
  $shortcutPath = Install-StartupShortcut
  Write-ApplyPilotLaunchRolloutEvidence `
    -Path (Join-Path $ProjectRoot ".fleet-logs\runtime-launch-rollout-otp.json") `
    -RuntimeChannel $runtimeChannel `
    -WrapperPath @($Launcher) `
    -ShortcutPath @($shortcutPath)
  Write-Host "[otp-startup] installed Startup shortcut $shortcutPath"
  if (-not $NoStart) {
    # No -Dsn: the launcher reads the mapped DSN from the FLEET_PG_DSN environment channel.
    & $Launcher
  }
}
