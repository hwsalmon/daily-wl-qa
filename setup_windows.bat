@echo off
REM Daily WL QA — Windows setup
REM Downloads an embeddable Python 3.12 runtime and installs all
REM dependencies into the app folder.  No admin rights required.
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
set "RUNTIME=%ROOT%python_runtime"
set "PY_VERSION=3.12.8"
set "PY_ZIP=python-%PY_VERSION%-embed-amd64.zip"
set "PY_URL=https://www.python.org/ftp/python/%PY_VERSION%/%PY_ZIP%"
set "GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py"

echo ============================================
echo   Daily WL QA — Windows Setup
echo ============================================
echo.

REM ── If python_runtime already exists, skip straight to deps ──────────────────
if exist "%RUNTIME%\python.exe" (
    echo Python runtime already found — skipping download.
    goto :install_deps
)

REM ── Download embeddable Python ────────────────────────────────────────────────
echo Downloading Python %PY_VERSION% ^(embeddable, ~8 MB^)...
if not exist "%RUNTIME%" mkdir "%RUNTIME%"

powershell -Command "Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%RUNTIME%\%PY_ZIP%' -UseBasicParsing"
if errorlevel 1 (
    echo.
    echo ERROR: Could not download Python.
    echo   Check your internet connection and try again.
    pause
    exit /b 1
)

REM ── Extract ───────────────────────────────────────────────────────────────────
echo Extracting Python...
powershell -Command "Expand-Archive -Path '%RUNTIME%\%PY_ZIP%' -DestinationPath '%RUNTIME%' -Force"
if errorlevel 1 (
    echo ERROR: Extraction failed.
    pause
    exit /b 1
)
del "%RUNTIME%\%PY_ZIP%"

REM ── Patch ._pth to enable pip / site-packages ─────────────────────────────────
echo Enabling pip support...
for %%f in ("%RUNTIME%\python*._pth") do (
    powershell -Command "(Get-Content '%%f') -replace '#import site','import site' | Set-Content '%%f'"
)

REM ── Install pip ───────────────────────────────────────────────────────────────
echo Installing pip...
powershell -Command "Invoke-WebRequest -Uri '%GET_PIP_URL%' -OutFile '%RUNTIME%\get-pip.py' -UseBasicParsing"
if errorlevel 1 (
    echo ERROR: Could not download pip installer.
    pause
    exit /b 1
)
"%RUNTIME%\python.exe" "%RUNTIME%\get-pip.py" --no-warn-script-location --quiet
del "%RUNTIME%\get-pip.py"

:install_deps
REM ── Install / update WL QA dependencies ──────────────────────────────────────
echo.
echo Installing dependencies ^(first run may take a few minutes^)...
"%RUNTIME%\python.exe" -m pip install --upgrade --no-warn-script-location --quiet ^
    PySide6 ^
    pydicom ^
    opencv-python ^
    scipy ^
    reportlab ^
    matplotlib ^
    Pillow

if errorlevel 1 (
    echo.
    echo ERROR: Dependency installation failed.
    echo   Check your internet connection and try again.
    pause
    exit /b 1
)

REM ── Generate icon ─────────────────────────────────────────────────────────────
echo Generating app icon...
"%RUNTIME%\python.exe" -c "
import sys
sys.path.insert(0, r'%ROOT%')
exec(open(r'%ROOT%install.py').read().split('# -- 4.')[0])
" 2>nul

echo.
echo ============================================
echo   Setup complete!
echo   Double-click run_wl_qa.bat to launch.
echo ============================================
pause
