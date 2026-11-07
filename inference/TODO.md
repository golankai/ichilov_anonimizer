# Inference: status, to-dos, ideas

## Status (2026-11-07)
- Deployer flow: `setup.bat`, put the model in `HeMed_Anonimizer_Model\`, `deidentify.bat file.csv [label|mask|x]`.
- Output is now **CSV and TXT (tab-separated) plus JSON** by default (`--output-format csv|txt|both`).
- Input is read with `utf-8-sig`; TABs in the text become spaces; row progress and start/end time are printed.
- The model folder was renamed from `HeMed_NER_baseline` to `HeMed_Anonimizer_Model`. Anyone who already placed the
  model under the old name must rename the folder (or pass `--model`).
- Tests: `pytest inference/tests`. Fake sample data: `inference/dummy_data/` (committed).

## To do
1. **Edge-margin A/B test.** Training masks 64 tokens at inner window edges; check whether inference should do the same
   (`--edge-margin`): run both on the same documents and compare entities found.
2. **NLTK clean-up (docs and code).** `inference/README.md` still says "Python 3.9 or higher" and that long text is split
   sentence by sentence with NLTK. The pipeline now cuts overlapping token windows (`utils/chunking.py`), and the setup
   needs Python 3.12. Find out whether NLTK is still needed for the sentence boundaries; if yes, fix the wording; if no,
   drop `setup_nltk()` and the dependency (it also avoids a download on machines without internet).
3. Deployer batch: `deidentify.bat` does not expose `--output-format`; decide if it needs a third argument.
4. Reconcile with the deployer's own branch when it is uploaded.
5. Update the stale examples in `README_Dev.md` (e.g. `test_deid_label_2.csv`, `dicta-il/dictabert-ner`).

## Ideas
- Compare outputs of several models on the same CSV (`--model` accepts any folder), e.g. old vs new fine-tune,
  and diff the `identified_entities` columns.
- A regression gold file per model version, so model changes show up as a diff in the tests.
