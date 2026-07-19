Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot ".fleet-logs") | Out-Null

if (-not $env:FLEET_PG_DSN) {
  throw "FLEET_PG_DSN is not set. Export an explicit mapped per-node fleet DSN (never the postgres admin or the shared fleet_worker role) before launching the TARPON LinkedIn worker."
}
$env:PYTHONPATH = "src"
$env:APPLYPILOT_FLEET_LABEL = "m2"
$env:FLEET_MACHINE_OWNER = "m2"
$env:FLEET_OWNER_IP = "100.77.65.8"
$env:APPLYPILOT_LINKEDIN_FALLBACK_AGENT = "codex"
$env:APPLYPILOT_AGENT_TIMEOUT = "900"
$env:APPLYPILOT_USAGE_LIMIT_COOLDOWN_SECONDS = "3600"

$exe = Join-Path $PSScriptRoot ".venv\Scripts\applypilot-fleet-linkedin.exe"
if (-not (Test-Path -LiteralPath $exe)) {
  throw "Missing LinkedIn worker executable: $exe"
}

& $exe `
  --worker-id "tarpon-linkedin-0" `
  --owner-ip "100.77.65.8" `
  --machine-owner "m2" `
  --agent "claude" `
  --fallback-agent "codex" `
  *>> (Join-Path $PSScriptRoot ".fleet-logs\tarpon-linkedin-task.log")
