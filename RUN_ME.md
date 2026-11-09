# RUN_ME

Updated 2026-11-09. Goal: measure the new rules (F8 repeated names, phones, age 90+) on the production model, using the fake pipeline test set. No patient text is involved.

## GPU machine

1. `cd "D:\DeID KAI\ichilov_anonimizer"`
2. `git pull`
3. `python -X utf8 training\eval_production.py training\models\model_non_anon_20261109-0144` (three variants: strict, lenient, no rules; a few minutes)
4. When it finishes: `training\push_results.bat` (pushes only the scores in `training\runs\production-eval-20261109\`)
5. Tell Claude it is done.

## Afterwards (for later)

- Copy the model folder `training\models\model_non_anon_20261109-0144` to the server, and send me the server path (it goes into `FOR_COLLEAGUES.md`).
- The e-mail draft for the colleagues is `EMAIL_DRAFT_HE.md` (Hebrew, for you only). The instructions the colleagues follow are `FOR_COLLEAGUES.md` (English only).
