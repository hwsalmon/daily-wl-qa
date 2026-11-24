@echo off
rem Daily QA - WL - DLG - Picket Fence: portable start (no installation, no admin rights, no internet needed).
setlocal
cd /d "%~dp0app"
set PYTHONNOUSERSITE=1
set MPLBACKEND=Agg
..\python\python.exe -W ignore wl_qa_tool.py
if errorlevel 1 (
    echo.
    echo The application stopped with an error ^(see above^).
    pause
)
