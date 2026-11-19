Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Compiling SynCura IEEE Research Paper with Local CLI     " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$paperDir = $PSScriptRoot
Set-Location -Path $paperDir

# Refresh Environment Path in current session in case MiKTeX was just installed
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

$pdfLatex = Get-Command pdflatex -ErrorAction SilentlyContinue
if (-not $pdfLatex) {
    # Check common MiKTeX paths directly
    $commonPaths = @(
        "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe",
        "$env:ProgramFiles\MiKTeX\miktex\bin\x64\pdflatex.exe",
        "$env:ProgramFiles\MiKTeX 2.9\miktex\bin\x64\pdflatex.exe",
        "C:\texlive\2026\bin\windows\pdflatex.exe",
        "C:\texlive\2025\bin\windows\pdflatex.exe"
    )
    foreach ($p in $commonPaths) {
        if (Test-Path $p) {
            $pdfLatex = Get-Item $p
            $binDir = Split-Path $p
            $env:Path = "$binDir;$env:Path"
            break
        }
    }
}

if (-not $pdfLatex) {
    Write-Host "[ERROR] pdflatex command was not found in PATH." -ForegroundColor Red
    Write-Host "Please ensure MiKTeX or TeX Live is installed." -ForegroundColor Yellow
    Exit 1
}

Write-Host "[1/4] Running Pass 1: pdflatex..." -ForegroundColor Green
& pdflatex -interaction=nonstopmode syncura_ieee_paper.tex

Write-Host "[2/4] Running BibTeX for bibliography citations..." -ForegroundColor Green
& bibtex syncura_ieee_paper

Write-Host "[3/4] Running Pass 2: pdflatex..." -ForegroundColor Green
& pdflatex -interaction=nonstopmode syncura_ieee_paper.tex

Write-Host "[4/4] Running Pass 3: pdflatex (resolving cross-references)..." -ForegroundColor Green
& pdflatex -interaction=nonstopmode syncura_ieee_paper.tex

if (Test-Path "syncura_ieee_paper.pdf") {
    $size = (Get-Item "syncura_ieee_paper.pdf").Length / 1KB
    Write-Host "`n[SUCCESS] PDF generated successfully!" -ForegroundColor Green
    Write-Host "Output: $paperDir\syncura_ieee_paper.pdf ($([Math]::Round($size, 1)) KB)" -ForegroundColor Cyan
} else {
    Write-Host "`n[FAIL] syncura_ieee_paper.pdf was not produced. Please inspect syncura_ieee_paper.log." -ForegroundColor Red
}
