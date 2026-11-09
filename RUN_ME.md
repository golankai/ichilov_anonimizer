# RUN_ME

Updated 2026-11-09. Goal: put the new production model next to the old one in the shared models folder, so the colleagues can test it.

## GPU machine

1. `cd "D:\DeID KAI\ichilov_anonimizer"`
2. `git pull`
3. `python -X utf8 training\publish_model.py` (copies only the needed files of `training\models\model_non_anon_20261109-0144` to `D:\De ID All\NER_anon_models\real_plus_half_synth_all_data`, next to `real_only_all_data`; refuses to overwrite an existing folder)
4. Check that the new folder has `config.json`, `model.safetensors`, the tokenizer files and `MODEL_INFO.txt`.
5. Tell Claude it is done.

## Then (on any computer)

- Send the e-mail. The Hebrew draft is `EMAIL_DRAFT_HE.md` (for you only). The instructions the colleagues follow are `DEID_INSTRUCTIONS_RUN.md` (English only). Both already name the server folder `D:\De ID All\NER_anon_models\`.
