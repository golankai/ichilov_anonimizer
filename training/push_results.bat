@echo off
rem Commit and push the metadata of all training runs (training\runs\*). Models are git-ignored and never pushed.
cd /d "%~dp0.."
git add training\runs
git commit -m "training runs: metadata" 
git pull --rebase
git push
