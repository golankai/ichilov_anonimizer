@echo off
rem Duplicate / leakage check of the annotated data. Prints COUNTS ONLY (no text, no ids); nothing is modified.
rem The report is also saved to training\duplicates_report.txt (git-ignored) so it can be pasted in one go.
rem Extra arguments are passed on, e.g.:  training\run_check_duplicates.bat --data-dir "D:\De ID All\NER_anon_dataset\cons"
cd /d "%~dp0.."
python -X utf8 training\check_duplicates.py %* > training\duplicates_report.txt
if errorlevel 1 (type training\duplicates_report.txt & exit /b 1)
type training\duplicates_report.txt
echo.
echo ===== saved to training\duplicates_report.txt (counts only) =====
