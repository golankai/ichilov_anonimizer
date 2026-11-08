@echo off
rem MAINTAINERS ONLY. Commits and pushes inference\dummy_results\ (fake sample outputs) and nothing else.
rem Run inference\run_dummy.bat first. A guard refuses to push anything that is not an output of inference\dummy_data.
cd /d "%~dp0.."
set PY=python
if exist ".venv\pyvenv.cfg" if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
git add inference\dummy_results
"%PY%" -X utf8 inference\check_dummy_results.py || (git reset -q inference\dummy_results & exit /b 1)
git commit -m "inference: dummy results" -- inference\dummy_results
git pull --rebase
git push
