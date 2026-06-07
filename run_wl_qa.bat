@echo off
REM Winston-Lutz Daily QA Tool — Windows launcher
REM Double-click this file, or run from a terminal.

cd /d "%~dp0"
python wl_qa_tool.py
if errorlevel 1 (
    echo.
    echo Error launching app. Make sure dependencies are installed:
    echo   pip install -r requirements.txt
    pause
)
