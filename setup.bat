@echo off
REM One-time setup for running the de-identification (no training packages).
REM Needs Python 3.12 on PATH. Run by double-click or:  setup.bat
cd /d "%~dp0"
if not exist .venv (python -m venv .venv || goto :fail)
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
".venv\Scripts\python.exe" inference\check_setup.py || goto :fail
echo.
echo Setup finished. Next: put the model folder in models\ and run deidentify.bat your_file.csv
pause
exit /b 0
:fail
echo.
echo SETUP FAILED - see the messages above.
pause
exit /b 1
