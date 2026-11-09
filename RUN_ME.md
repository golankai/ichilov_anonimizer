# RUN_ME

Updated 2026-11-09. Goal: see why medical organisations fail (diagnostic only, nothing is sent anywhere).

## GPU machine

1. `cd "D:\DeID KAI\ichilov_anonimizer"`
2. `git pull`
3. `python -X utf8 training\diagnose_medorg.py training\models\model_non_anon_20261109-0144`
   (fake set only; prints a summary and writes `training\runs\medorg-diagnostic\medorg_fake.txt`)
4. `training\push_results.bat`  (pushes only the fake-text listing)
5. Tell Claude it is done.

Optional, for the real data (counts only, no text): add `--real-split test`. Caution: the production model was trained on that split, so the numbers will look too good. Skip it unless you have a model trained without the test split.

## Colleagues' e-mail

Not sent yet. The Hebrew draft is `EMAIL_DRAFT_HE.md` (for you only); the colleagues' instructions are `DEID_INSTRUCTIONS_RUN.md`. The new model is already in `D:\De ID All\NER_anon_models\real_plus_half_synth_all_data`.
