#!/usr/bin/env powershell
<#
.SYNOPSIS
Automated setup script for ionerdstechfusion development environment

.DESCRIPTION
Initializes the development environment for both backend and frontend

.PARAMETER SkipVenv
Skip Python virtual environment creation

.PARAMETER SkipNpm
Skip npm installation

.EXAMPLE
./setup.ps1
./setup.ps1 -SkipVenv
#>

param(
    [switch]$SkipVenv,
    [switch]$SkipNpm
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 ionerdstechfusion Team Setup" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "✓ Checking prerequisites..." -ForegroundColor Green
$checks = @{
    "Python" = "python --version"
    "Node.js" = "npm --version"
    "Git" = "git --version"
}

foreach ($tool in $checks.Keys) {
    try {
        $output = Invoke-Expression $checks[$tool] 2>&1
        Write-Host "  ✓ $tool: $output" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ $tool: Not found or not in PATH" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Python venv setup
if (-not $SkipVenv) {
    Write-Host "📦 Setting up Python environment..." -ForegroundColor Green
    
    if (Test-Path .venv) {
        Write-Host "  • Virtual environment already exists" -ForegroundColor Yellow
    } else {
        Write-Host "  • Creating virtual environment..."
        python -m venv .venv
        Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
    }
    
    Write-Host "  • Activating virtual environment..."
    .\.venv\Scripts\Activate.ps1
    
    Write-Host "  • Installing backend dependencies..."
    pip install -q -r backend\requirements.txt
    
    Write-Host "  • Installing ML dependencies..."
    pip install -q -r ml\requirements.txt

    Write-Host "  • Installing Discord bot dependencies..."
    pip install -q -r discordbot\requirements.txt
    
    Write-Host "  ✓ Python environment ready" -ForegroundColor Green
} else {
    Write-Host "⊘ Skipping Python virtual environment setup" -ForegroundColor Yellow
}
Write-Host ""

# Frontend setup
if (-not $SkipNpm) {
    Write-Host "📦 Setting up Frontend environment..." -ForegroundColor Green
    
    if (Test-Path frontend\node_modules) {
        Write-Host "  • node_modules already exists" -ForegroundColor Yellow
    } else {
        Write-Host "  • Installing npm dependencies..."
        Push-Location frontend
        npm install
        Pop-Location
        Write-Host "  ✓ Frontend dependencies installed" -ForegroundColor Green
    }
} else {
    Write-Host "⊘ Skipping npm setup" -ForegroundColor Yellow
}
Write-Host ""

# Environment file
Write-Host "⚙️  Checking environment configuration..." -ForegroundColor Green
if (-not (Test-Path .env)) {
    if (Test-Path .env.example) {
        Write-Host "  • Copying .env.example to .env..."
        Copy-Item .env.example .env
        Write-Host "  ✓ Created .env file (update with your settings)" -ForegroundColor Green
    }
} else {
    Write-Host "  • .env file already exists" -ForegroundColor Yellow
}
Write-Host ""

# Directory structure verification
Write-Host "📁 Verifying project structure..." -ForegroundColor Green
$dirs = @("backend", "frontend", "ml", "backend\data")
foreach ($dir in $dirs) {
    if (Test-Path $dir) {
        Write-Host "  ✓ $dir/" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $dir/ - MISSING" -ForegroundColor Red
    }
}
Write-Host ""

# Summary
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Review and update .env with your settings"
Write-Host "2. Read TEAM_SETUP.md for development workflow"
Write-Host "3. Create your feature branch:"
Write-Host "   git checkout -b feature/your-feature develop"
Write-Host ""
Write-Host "To start development:"
Write-Host "  Terminal 1: uvicorn backend.app:app --reload --port 8000"
Write-Host "  Terminal 2: cd frontend && npm run dev"
Write-Host "  Terminal 3: python backend\replay.py --mode http --url http://localhost:8000/ingest"
Write-Host ""
Write-Host "Happy coding! 🚀" -ForegroundColor Cyan
