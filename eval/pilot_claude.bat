@echo off
rem Separate two-site pilot with Claude Opus 5.5 (own database and folder:
rem eval_data\pilot_claude). The main 20-site experiment is not touched.
setlocal
cd /d "%~dp0.."
if not exist venv\Scripts\python.exe (
  echo Run eval\prepare.bat first.
  pause
  exit /b 1
)
venv\Scripts\python.exe eval\pilot_claude.py
echo.
echo The full log is in eval_data\pilot_claude\pilot_log.txt
pause
