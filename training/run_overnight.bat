@echo off
rem OVERNIGHT CAMPAIGN: several training recipes x 3 seeds, each model also scored on the full inference pipeline.
rem First a smoke test of every training mode on tiny data (a few minutes); if that fails nothing else starts.
rem Models:  training\models\<campaign>\<recipe>\seed_<n>\   (git-ignored)     Results: training\runs\<campaign>\
rem Does NOT touch git. Resume an interrupted night with:  training\run_overnight.bat --campaign <campaign id>
rem Other options:  --only A_real B_mix_full   --seeds 42   --max-hours 9   --list   --smoke-only
rem When it is done, run training\push_results.bat and tell Claude the campaign id (printed at the end, also in registry.md).
cd /d "%~dp0.."
python -X utf8 training\run_campaign.py %*
echo.
echo Exit code %ERRORLEVEL%  (0 = finished, 2 = preflight failed, 3 = smoke test failed). Next: training\push_results.bat
