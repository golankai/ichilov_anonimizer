# RUN_ME

Updated 2026-11-09. Nothing to run on the GPU machine right now.

Done this session: the organisation diagnostic showed that most "partial/wider" MED_ORG cases were scorer artifacts (see `training/runs/medorg-diagnostic/medorg_fake.txt`). The scorer now ignores whitespace between predicted pieces, and a new deterministic rule (`rule_med_org_lexicon`, list in `inference/lexicons/med_orgs.tsv`) masks well-known hospitals and health funds the model misses. Ambiguous words (`כללית`, `מאיר`, `גהה`) are masked only after an institution word. Extend the list by adding lines to the TSV.

Rule-only change: colleagues get it with `git pull`; the model in `D:\De ID All\NER_anon_models\real_plus_half_synth_all_data` is unchanged and needs no re-publish.

## Colleagues' e-mail

Not sent yet: Hebrew draft `EMAIL_DRAFT_HE.md` (for you only); instructions `DEID_INSTRUCTIONS_RUN.md`.
