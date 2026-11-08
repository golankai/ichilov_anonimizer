@echo off
rem Run the model on the fake test set (inference\dummy_data\*.csv); results go to inference\dummy_results\ and are
rem scored against the ground truth. Does NOT touch git.
rem Uses the python of the active environment (or .venv). Usage: inferenceun_dummy.bat [model_name]
cd /d "%~dp0.."
set PY=python
if exist ".venv\pyvenv.cfg" if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
set MODELARG=
if not "%~1"=="" set MODELARG=--model "%~1"
for %%F in (inference\dummy_data\*.csv) do (
  "%PY%" -X utf8 inference\de_identify.py --input "%%F" --output-dir inference\dummy_results %MODELARG% || exit /b 1
)
echo.
echo ===== scoring against the ground truth (inference\dummy_data\*.truth.jsonl) =====
"%PY%" -X utf8 inference\score_dummy.py
