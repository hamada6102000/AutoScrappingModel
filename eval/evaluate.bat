@echo off
rem Run the evaluation. "evaluate.bat offline" runs only the modules that need no LLM.
setlocal
cd /d "%~dp0.."
if not exist venv\Scripts\python.exe (
  echo Run eval\prepare.bat first.
  pause
  exit /b 1
)
venv\Scripts\python.exe eval\evaluate.py %*
echo.
echo Finished. The full log is in eval_data\evaluate_log.txt
pause
