@echo off
rem Smoke test of the Claude Code route (Claude subscription, no API key):
rem one tiny request, no evaluation runs, no database changes.
setlocal
cd /d "%~dp0.."
if not exist venv\Scripts\python.exe (
  echo Run eval\prepare.bat first.
  pause
  exit /b 1
)
venv\Scripts\python.exe manage.py eval_claude_smoke %*
echo.
echo The result is also saved in eval_data\claude_code_smoke_log.txt
pause
