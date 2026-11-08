@echo off
rem Train the base model: 6 epochs max, early stopping, 3 seeds (42 43 44), then the summary table.
rem Scores the real test split and the synthetic sets (training\synth_eval) for every seed. Does NOT touch git.
rem Extra arguments are passed on, e.g.:  training\run_train.bat --seeds 42     or     training\run_train.bat --model "D:\path\model_anon"
rem When it is done, run training\push_results.bat to publish the metadata (counts and metrics only).
cd /d "%~dp0.."
python -X utf8 training\train.py train %*
