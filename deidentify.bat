@echo off
REM De-identify a CSV (needs columns: id, text).   deidentify.bat path\to\file.csv [label|mask|x]
REM Output is written next to the input file. The model is read from HeMed_NER_baseline\ in this folder.
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: deidentify.bat path\to\file.csv [label^|mask^|x]
  exit /b 1
)
set MODE=%~2
if "%MODE%"=="" set MODE=label
".venv\Scripts\python.exe" -X utf8 inference\de_identify.py --input "%~f1" --mode %MODE%
