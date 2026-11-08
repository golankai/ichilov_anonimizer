# Next session: pick-up list (written 2026-11-08)

## Update (end of 2026-11-08, second session): deterministic rules are in
Implemented in `inference/utils/rules.py` (on by default, `--no-rules` = old stages), tests in `inference/tests/test_rules.py`:
F1/F2 digit runs + ID checksum, F3 e-mail as one span, F4/F5 dates (numeric, written, ISO, Hebrew calendar, months),
F7 hyphenated surnames. The old date/ID stages no longer rewrite the text before the model (offsets fit the text).
Rules alone vs fake truth: 225/226 date/email/ID/case-number spans exact, 0 spans outside truth. **Not yet measured with
the model: run `inference\run_dummy.bat` and compare with the 2026-11-08 first run.**
Decisions: F6 dropped (prefix inside the masked span is fine; score `*-prefixed` tags leniently); months alone are masked
(policy, hospital counts them); F8 will propagate only names NOT in `inference/lexicons/ambiguous_words.tsv`.
Still open from section 2 below: F8 (+ strict/lenient switch), F9, F10, F11, F12.
Open for later, not started: ambiguous-word cases in the fake test set (traps tier) and in synthetic training data;
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
