@echo off
REM De-identify a CSV (needs columns: id, text).
REM   deidentify.bat path\to\file.csv [label^|mask^|x] [model_name]
REM Output is written next to the input file. Models are the folders inside models\ ; with no model name the
REM default (models\HeMed_Anonimizer_Model, or the only model there) is used. Output files of other models
REM carry the model name, so you can run several models on the same file and compare.
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: deidentify.bat path\to\file.csv [label^|mask^|x] [model_name]
  exit /b 1
)
set MODE=%~2
if "%MODE%"=="" set MODE=label
set MODELARG=
if not "%~3"=="" set MODELARG=--model "%~3"
REM Uses .venv if setup.bat made one; otherwise the python of the active environment (e.g. a conda env).
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
"%PY%" -X utf8 inference\de_identify.py --input "%~f1" --mode %MODE% %MODELARG%
