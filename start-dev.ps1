param(
  [int]$FrontendPort = 5173,
  [int]$BackendPort = 8000
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendPath = Join-Path $repoRoot "frontend"
$backendPath = Join-Path $repoRoot "backend"
$discordbotPath = Join-Path $repoRoot "discordbot"
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$backendPython = if (Test-Path $venvPython) { $venvPython } else { "python" }

if (!(Test-Path $frontendPath)) {
  throw "Frontend directory not found: $frontendPath"
}

if (!(Test-Path $backendPath)) {
  throw "Backend directory not found: $backendPath"
}

if (!(Test-Path $discordbotPath)) {
  Write-Host "Warning: Discord bot directory not found: $discordbotPath"
}

$backendCommand = @"
cd "$repoRoot"
$backendPython -m uvicorn backend.app:app --reload --host 127.0.0.1 --port $BackendPort
"@

$frontendCommand = @"
cd "$frontendPath"
npm run dev -- --host 127.0.0.1 --port $FrontendPort
"@

$discordbotCommand = @"
cd "$repoRoot"
$backendPython discordbot\discord_bot.py
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCommand | Out-Null
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCommand | Out-Null
$discordTokenConfigured = $env:DISCORD_BOT_TOKEN
if (-not $discordTokenConfigured) {
  $envFile = Join-Path $repoRoot ".env"
  if (Test-Path $envFile) {
    $discordTokenConfigured = Select-String -Path $envFile -Pattern '^DISCORD_BOT_TOKEN=\S+' -Quiet
  }
}
if ($discordTokenConfigured) {
  Start-Process powershell -ArgumentList "-NoExit", "-Command", $discordbotCommand | Out-Null
  Write-Host "Discord Bot:  Running in separate terminal"
} else {
  Write-Host "Discord Bot:  Skipped (DISCORD_BOT_TOKEN not set)"
}

Write-Host "Started backend and frontend in separate terminals."
Write-Host "Backend:      http://127.0.0.1:$BackendPort"
Write-Host "Frontend:     http://127.0.0.1:$FrontendPort"
Write-Host "Discord Bot:  Running in separate terminal"
if (!(Test-Path $venvPython)) {
  Write-Host "Note: .venv not found. Services are using global Python. If dependencies are missing, run:"
  Write-Host "  pip install -r backend\requirements.txt"
  Write-Host "  pip install -r discordbot\requirements.txt"
}
