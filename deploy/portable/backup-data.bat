@echo off
rem Copies the history database and the settings to a dated backup folder next to this one.
setlocal
cd /d "%~dp0app"
set "D=%~dp0backup-%date:~-4%%date:~4,2%%date:~7,2%"
mkdir "%D%" 2>nul
copy /y wl_qa_history.db "%D%\" 2>nul
copy /y wl_qa_config.json "%D%\" 2>nul
echo Backed up to %D%
pause
