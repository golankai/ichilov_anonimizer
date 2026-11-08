# Next session: pick-up list (written 2026-11-08)

Nothing in the pipeline has been fixed yet. Everything below is planned, not done.

## 0. Results to bring back first (from the GPU box, `pii` env, after `git pull`)
1. `inference\run_dummy.bat` -> paste the score tables (baseline per tier / label / situation). Untested in real batch.
2. `python -X utf8 training\check_duplicates.py` -> paste the counts only (the printed report holds no text or ids).
   Key numbers: test/validation documents that also occur in train (inflated scores?), conflicting-label duplicates,
   `second_*` ids, near duplicates, ids of `ignorable_ids.txt` present.
3. Optional: `python -X utf8 training\train.py train` (about 13 min), or `--epochs 2` first.

## 1. Decide on data cleaning (after the counts)
`check_duplicates.py --write-clean DIR`: conflicting groups `--conflicts drop|first`; `--drop-near-from-train`.
Originals are never modified. The default `train` mode ignores `ignorable_ids.txt`; decide whether it should apply it.

## 2. Fix the pipeline (main work), in `inference/utils/pipeline_stages.py`, each with a regression test
Full list: `inference/docs/HOSPITAL_FEEDBACK_AND_FIXES.md` (issues H1-H12, P1-P4, M1-M2, D1-D5; fixes F1-F12;
data/model T1-T5; operations O1-O3). Suggested order:
1. P1 email prefix leak -> email as one unit (F3).
2. P2 split / garbled dates -> merge date pieces, stop rewriting written dates to ISO (F4), Hebrew date grammar (F5).
3. H CaseNum / ID -> digit-run rule (`(\d-?){4,}\d`) and Israeli ID checksum (F1, F2), ideas from HebSafeHarbor.
4. P3 prefix absorbed, P4 hyphenated surnames (F6, F7).
5. Repeated values / document-level consistency with a stop-list (F8); medical stop-list (F9); city gazetteer (F10).
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
