# Next session: pick-up list (written 2026-11-08)

## Update (2026-11-09, night): more rules + message for colleagues
Added to `inference/utils/rules.py` (tests in `inference/tests/test_rules.py`, 134 pass): **F8** document consistency (repeats of a masked name/city/org are masked, prefix tolerant, skips `lexicons/ambiguous_words.tsv`; switch `consistency=False` / `--no-consistency`), **F13** Israeli phone numbers incl. `+972-54-...`, **F14** age 90-120 after בן/בת/גיל. Reason: in the campaign, 13 of ~33 pipeline leaks per seed were a first name repeated with a Hebrew prefix. On the fake truth F8 recovers all repeats that are not ambiguous words and adds nothing outside the truth; the ambiguous-word repeats (about 22% of repeats in the fake set) are left to the model by decision. **Not yet measured with the model**: re-run the pipeline eval on the production model (`run_dummy.bat` / campaign pipeline eval) and compare leaks and traps with the campaign numbers.
INNER_ID: dropped on purpose. Real data has only a handful of examples and the one seen is annotated as separate word pieces plus a number, so there is no reliable shape; the model labels case numbers CONTACT, text is masked either way. A relabel rule (F15) was written and removed.
`FOR_COLLEAGUES.md` (English only, for the colleagues who test the new version): the user must put the server path of the model into it before sending it. The Hebrew e-mail draft for the user is `EMAIL_DRAFT_HE.md` (not for colleagues to follow). Both reach others only after a push (ask first). Production model: `training/runs/model_non_anon_20261109-0144` (weights stay on the GPU machine); `training/eval_production.py` measures F8/F13/F14 on the fake set (strict / lenient / norules).

## Update (2026-11-09, evening): production model training
Campaign night-20261108-1846 analysed (report in `training/runs/night-20261108-1846/`). Recipe chosen: **C_mix_half** (real + half the synthetic, mixed); mixing beats staged on ambiguous/rare words, staged is slightly higher on real test F1; full synthetic was a bit worse than half. Production run started on the GPU box (`train-all --epochs 6 --extra-train synth_train_all.jsonl --extra-fraction 0.5`, fixed 6 epochs, no early stopping since `train-all` has no validation). Added `--extra-fraction` to `train-all`.
Instructions for the user now live in `RUN_ME.md` (keep it short and current). When the user reports done: `git pull` (results come via `training\push_results.bat`), check `training/runs/` for the new `training_run.json`, then hand-over of the model folder. Do not commit/push locally before that pull unless needed; the GPU push is `pull --rebase` + push.

## Update (2026-11-09): overnight campaign ready
See `training/TODO.md` ("Update 2026-11-09"). After the night: `training\push_results.bat`, then analyse `training/runs/<campaign>/report.md` + `registry.md`. Decide the recipe, then train the production model with `train-all --extra-train training/synth_train/synth_train_all.jsonl`.

## Update (2026-11-08, evening, training side)
Pushed to main at 14948d9. Details and open points: `training/TODO.md` ("Update 2026-11-08, evening", "Open points", "What belongs where").
Short version: duplicate check clean; real dates are DAY/MONTH/YEAR pieces and the synthetic sets now match; `train.py` scores rare_hard,
long and ambiguous by default; next on the GPU box is a 1-epoch re-check (last line must read `rare_hard (586 docs ...)`), then the full
3-seed run and `training\push_results.bat`. Base-model eval and pipeline eval stay separate; long documents mainly belong to the pipeline
set (`t5_long` has only 4 docs, to be enlarged). Still unverified: `second_` ids = same patient across splits? (ask the annotator).

## Update (end of 2026-11-08, second session): deterministic rules are in
Implemented in `inference/utils/rules.py` (on by default, `--no-rules` = old stages), tests in `inference/tests/test_rules.py`:
F1/F2 digit runs + ID checksum, F3 e-mail as one span, F4/F5 dates (numeric, written, ISO, Hebrew calendar, months),
F7 hyphenated surnames. The old date/ID stages no longer rewrite the text before the model (offsets fit the text).
Rules alone vs fake truth: 225/226 date/email/ID/case-number spans exact, 0 spans outside truth. **Not yet measured with
the model: run `inference\run_dummy.bat` and compare with the 2026-11-08 first run.**
Decisions: F6 dropped (prefix inside the masked span is fine; score `*-prefixed` tags leniently); months alone are masked
(policy, hospital counts them); F8 will propagate only names NOT in `inference/lexicons/ambiguous_words.tsv`.
Still open from section 2 below: F8 (+ strict/lenient switch), F9, F10, F11, F12.
Done after that: `training/synth_eval_ambiguous/` (809 docs: name / plain / mixed use of every word of the list; own folder so
`training/synth_eval` and its tests are untouched; score it with `train.py --synth-eval-dir training/synth_eval_ambiguous`;
decide whether `train.py` should load both folders). List now has ORG, `phi_head`, `plain_example`.
Note for the pipeline side: the real corpus annotates a date as DAY/MONTH/YEAR pieces (see `training/synth_eval/README.md`), the
model therefore emits pieces; `inference/utils/rules.py` merges them into ONE DATE span (right for masking: `[DATE]`), but
`inference/dummy_data` truth and `score_dummy.py` still assume one span per date, so "exact boundaries" on DATE will not match
until the scorer accepts either (decide).
Open for later, not started: the same words in the pipeline test set `inference/dummy_data`, and in synthetic training data;
corpus counts for the list; batching (O1).

Original note (before the rules): nothing in the pipeline had been fixed yet. Everything below is planned.

## 0. Results to bring back first (from the GPU box, `pii` env, after `git pull`)
1. `inference\run_dummy.bat` -> paste the score tables (baseline per tier / label / situation). Untested in real batch.
2. `python -X utf8 training\check_duplicates.py` -> paste the counts only (the printed report holds no text or ids).
   Key numbers: test/validation documents that also occur in train (inflated scores?), conflicting-label duplicates,
   `second_*` ids, near duplicates, ids of `ignorable_ids.txt` present.
3. `training\run_train.bat` (3 seeds, 6 epochs max with early stopping, about 25 min; `--seeds 42 --epochs 2` first for a quick check),
   then `training\push_results.bat`. Read `training/runs/<experiment>/summary.md` (per label, synthetic sets, mean ± std).

## 1. Decide on data cleaning (after the counts)
`check_duplicates.py --write-clean DIR`: conflicting groups `--conflicts drop|first`; `--drop-near-from-train`.
Originals are never modified. The default `train` mode ignores `ignorable_ids.txt`; decide whether it should apply it.

## 2. Fix the pipeline (main work), in `inference/utils/pipeline_stages.py`, each with a regression test
Full list: `inference/docs/HOSPITAL_FEEDBACK_AND_FIXES.md` (issues H1-H12, P1-P4, M1-M2, D1-D5; fixes F1-F12;
data/model T1-T5; operations O1-O3). Suggested order:
1. P1 email prefix leak -> email as one unit (F3).
2. P2 split / garbled dates -> merge date pieces, stop rewriting written dates to ISO (F4), Hebrew date grammar (F5).
3. H CaseNum / ID -> digit-run rule (`(\d-?){4,}\d`) and Israeli ID checksum (F1, F2), ideas from HebSafeHarbor.
4. ~~P3 prefix absorbed (F6)~~ dropped by decision; P4 hyphenated surnames (F7) done.
5. Repeated values / document-level consistency, propagating only names not in the ambiguous list (F8); medical stop-list (F9); city gazetteer (F10).
6. Birth vs medical date by context (F11, F12).
Measure each with `inference\run_dummy.bat`; compare against the baseline from step 0.

## 3. Things to verify
- Convention-tagged truth spans (`policy`: month alone as date, street address as one location, passport as ID)
  against the real annotation guidelines.
- Offset drift (D1): JSON offsets are in the normalised text; the scorer only uses the "survive" check there.
- The hospital's metric is recall only and counts ordinary words (אושר, אדר, ניסן, סיון, אייר); keep that in mind
  when reading their numbers. Correct reading of the 8 Sep table: the new pipeline is marginally better than the old on ID/org.

## Working rules (standing)
- Never push without asking. Commit locally, then ask.
- Real patient data is never committed; reports contain metadata/counts only.
- Dummy-result pushing is maintainer-only and guarded (`push_dummy_results.bat`).

## Update 2026-11-08 evening: overnight campaign launched (where we stopped)

- Pushed to origin: 6dd8a2a (campaign runner, synthetic training set, bare-name MED_ORG/ORG convention).
- GPU machine (`D:\DeID KAI\ichilov_anonimizer`) is running campaign **night-20261108-1846** (started 18:46, smoke test passed 18:48).
  6 recipes x seeds 42/43/44 (A_real, B_mix_full, D_staged_1ep, C_mix_half, E_mix_quarter, F_staged_2ep), up to 6 epochs, pipeline eval after each model.
- Models stay on the GPU machine: `training/models/night-20261108-1846/<recipe>/seed_<n>/`. Log: `.../campaign.log`.
- Resume if interrupted: `training\run_overnight.bat --campaign night-20261108-1846`.
- User must run `training\push_results.bat` on the GPU machine when the campaign is done, then tell me the campaign id. (Local repo is intentionally NOT committed/pushed further, so the remote push stays fast-forward.)

### Next session
1. `git pull`; read `training/runs/night-20261108-1846/report.md`, `registry.md` (check errors.txt for failed runs).
2. Compare recipes: real test F1 (must not drop vs A_real), ambig_heldout vs ambig_seen, rare_hard/long, pipeline scores with rules vs norules (first seed only).
3. Pick recipe; production model: `train-all --extra-train training\synth_train\synth_train_all.jsonl` (all real + all synthetic words).
4. Still open: ask annotator about `second_` ids; verify real CONTACT `-`-only spans; enlarge pipeline long set (t5_long has 4 docs); hospital feedback F8-F12.
- Privacy note: example strings the user pasted earlier remain in the session transcript (~/.claude/projects/.../a44546ad-...jsonl); not in repo.
