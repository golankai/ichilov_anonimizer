@echo off
rem MAINTAINERS ONLY. Commits and pushes inference\dummy_results\ (fake sample outputs) and nothing else.
rem Run inferenceun_dummy.bat first. Never run this where real patient data was processed into that folder.
cd /d "%~dp0.."
git add inference\dummy_results
git commit -m "inference: dummy results" -- inference\dummy_results
git pull --rebase
git push
