@echo off
rem Run the model on the fake sample files; results go to inference\dummy_results\. Does NOT touch git.
rem Uses the python of the active environment (or .venv). Usage: inference\run_dummy.bat [model_name]
cd /d "%~dp0.."
set PY=python
if exist ".venv\pyvenv.cfg" if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
set MODELARG=
if not "%~1"=="" set MODELARG=--model "%~1"
for %%F in (test test_long) do (
  "%PY%" -X utf8 inference\de_identify.py --input inference\dummy_data\%%F.csv --output-dir inference\dummy_results %MODELARG% || exit /b 1
)
