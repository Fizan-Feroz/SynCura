@echo off
cd /d "%~dp0"
echo ==========================================================
echo  Compiling SynCura IEEE Research Paper with Local CLI
echo ==========================================================

REM Check if pdflatex is in path, or add standard MiKTeX path
where pdflatex >nul 2>nul
if %errorlevel% neq 0 (
    if exist "%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" (
        set "PATH=%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64;%PATH%"
    ) else if exist "%ProgramFiles%\MiKTeX\miktex\bin\x64\pdflatex.exe" (
        set "PATH=%ProgramFiles%\MiKTeX\miktex\bin\x64;%PATH%"
    )
)

where pdflatex >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] pdflatex not found in PATH or standard installation directories.
    echo Please install MiKTeX or TeX Live.
    pause
    exit /b 1
)

echo [1/4] Running Pass 1: pdflatex...
pdflatex -interaction=nonstopmode syncura_ieee_paper.tex

echo [2/4] Running BibTeX...
bibtex syncura_ieee_paper

echo [3/4] Running Pass 2: pdflatex...
pdflatex -interaction=nonstopmode syncura_ieee_paper.tex

echo [4/4] Running Pass 3: pdflatex...
pdflatex -interaction=nonstopmode syncura_ieee_paper.tex

if exist syncura_ieee_paper.pdf (
    echo.
    echo [SUCCESS] Generated: syncura_ieee_paper.pdf
) else (
    echo.
    echo [FAIL] Compilation failed. Check syncura_ieee_paper.log for details.
)

pause
