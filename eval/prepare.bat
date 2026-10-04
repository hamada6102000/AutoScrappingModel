@echo off
rem Prepare the 20-site evaluation (no LLM runs). Run from anywhere.
setlocal
cd /d "%~dp0.."
if not exist venv\Scripts\python.exe (
  echo Creating virtual environment...
  python -m venv venv
  if errorlevel 1 (
    echo Could not create the virtual environment. Is "python --version" working?
    exit /b 1
  )
)
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe eval\prepare.py
echo.
echo Finished. The full log is in eval_data\prepare_log.txt
pause
