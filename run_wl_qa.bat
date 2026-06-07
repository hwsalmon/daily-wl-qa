@echo off
REM Daily WL QA — Windows launcher
REM Uses the local python_runtime if setup_windows.bat has been run,
REM otherwise falls back to system Python.
setlocal

set "ROOT=%~dp0"
set "LOCAL_PY=%ROOT%python_runtime\python.exe"

if exist "%LOCAL_PY%" (
    "%LOCAL_PY%" "%ROOT%wl_qa_tool.py"
) else (
    python "%ROOT%wl_qa_tool.py"
)

if errorlevel 1 (
    echo.
    echo Error launching app.
    echo Run setup_windows.bat first if you have not already done so.
    pause
)
