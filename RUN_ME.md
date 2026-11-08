# RUN_ME: what Kai does next

Claude keeps this file current. Open it on whichever machine you are on, copy the block, run it.
Every block says where to run it (LOCAL = laptop, GPU = hospital machine) and what to do with outputs.

## Now (2026-11-09): train the production model  [GPU]

Campaign `night-20261108-1846` is analysed (3 seeds per recipe). Pick: **C_mix_half** = real data + half the synthetic set, mixed.
Best test F1 among the mixed recipes, best pipeline masked (0.956) and fewest leaks (32), lowest extra predictions (1.3), small seed spread.
Production uses **6 epochs, fixed** (`train-all` has no validation split, so no early stopping; epoch 6 was the best epoch in most campaign runs). The production synthetic file is the all-words variant (no held-out words).

```
cd "D:\DeID KAI\ichilov_anonimizer"
git pull
python -X utf8 training\train.py train-all --epochs 6 --extra-train training\synth_train\synth_train_all.jsonl --extra-fraction 0.5 --seed 42
```

- Output: `training\models\<name>_<timestamp>\` (git-ignored, stays on the GPU machine; hand over from there).
- Metadata `training_run.json` lands in `training\runs\`. Publish it with `training\push_results.bat` (counts only, no content).
- Tell Claude when it is done.

## Open question for Claude, no action for you yet
- Close call between C_mix_half and D_staged_1ep / F_staged_2ep on real test F1 (0.931 / 0.932 / 0.936 vs C 0.931); C wins on synthetic robustness and pipeline.
