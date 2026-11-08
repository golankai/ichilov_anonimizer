# RUN_ME

Updated 2026-11-09. Goal: train the production model (real data + half the synthetic set, 6 epochs).

## GPU machine

1. `cd "D:\DeID KAI\ichilov_anonimizer"`
2. `git pull`
3. `python -X utf8 training\train.py train-all --epochs 6 --extra-train training\synth_train\synth_train_all.jsonl --extra-fraction 0.5 --seed 42` (running now)
4. When it finishes: `training\push_results.bat`
5. Tell Claude it is done. The model stays in `training\models\<name>_<timestamp>\` (not pushed).
