# Training: status, to-dos, ideas

## Status (2026-11-07)
- Training code ported from Nminsker/MedNER (`train`, `train-all`, `cv`, `final`, `split`); every run writes `training_run.json`.
- GPU machine verified: conda env `PII` (Python 3.10, torch 2.6.0+cu124, transformers 4.57.0), A40 46 GB. Data, base model and GPU all reachable.
- Smoke run (1 epoch, `regular`, base `model_non_anon`): 76 s, validation F1 0.921, test F1 0.898 (recall 0.916).
  That only proves the pipeline works; it is not a result to compare against.

## Update (2026-11-08, later)
- First baseline (10 epochs, 1 seed): validation F1 0.940 (epoch 4), test P 0.933 / R 0.920 / F1 0.927 / F2 0.934; overfits after epoch 4.
- Done since: 6 epochs + early stopping (patience 2), 3 seeds per model with `summary.md`, per-label P/R/F1 everywhere, synthetic
  base-model evaluation sets (`synth_eval/`, separate from the pipeline fake set), `run_train.bat`, `run_check_duplicates.bat`.
  Not yet run on the GPU machine: next is `training\run_train.bat` (about 3 x 8 min) and reading `summary.md`.
- ID and INNER_ID have no examples in the real validation/test splits (23 and 3 in train): the synthetic `rare_hard` set is the only measure for them,
  and its INNER_ID convention is a guess (see `synth_eval/README.md`).

## Update (2026-11-08, evening): what was found and changed
- **Duplicates**: `check_duplicates.py` clean (no exact duplicates, no leakage between splits). 3456 ids carry a second_/copy_/dup_ prefix
  and 30 base ids exist in several forms, all with different texts. Whether `second_` means the same patient across splits (patient-level
  leakage) is unverified; only the annotator can say. Ask them.
- **Dates are three spans in the real corpus** (DAY/MONTH/YEAR, renamed DATE on load; separators unlabeled; 2097 spans in train, median length 2,
  1062 directly adjacent, no 8-digit spans). The synthetic sets used one span per date, so synthetic DATE scored badly for a reason that
  is not the model. Fixed in `make_synth_eval.py` (`split_date_spans`); tool: `check_date_layout.py` (counts only).
- `train.py` now scores **three** synthetic sets by default: `rare_hard` (586 docs), `long` (24), `ambiguous` (809, from `synth_eval_ambiguous/`).
  `--synth-eval-dir DIR [DIR ...]`. Still never used to pick the epoch.
- The 1-epoch run of 2026-11-08 was a plumbing check only (ORG scored 0 on real data too after 1 epoch). Its synthetic numbers came from the OLD
  generator (846 docs): ignore them. After `git pull` the last line must say `rare_hard (586 docs, 1076 entities ...)`.

## Open points (not done, in order of value)
1. Re-run `run_train.bat --seeds 42 --epochs 1` on the new code; DATE on synthetic should be near the real 0.9+.
2. Full run `run_train.bat` (3 seeds, up to 6 epochs): the first baseline with the final recipe (the only earlier baseline is 10 epochs, 1 seed,
   and it was before any of: early stopping, per-label metrics, seeds). It gives mean ± std, which is what any later model is compared against.
3. Synthetic ORG and MED_ORG scored 0 with the old code; MED_ORG recall on real data is 0.77, so the synthetic convention may differ from
   the real one. Look at real MED_ORG/ORG span shapes (counts only, like `check_date_layout.py`) before believing the synthetic numbers.
4. ID / INNER_ID: only 23 / 3 real train examples and none in val/test; synthetic is the only measure, and its INNER_ID convention is a guess.
5. `make_ambiguous_eval.py` has its own copy of `split_date_spans`: keep the two in step if the date convention changes.
6. Long documents: see "What belongs where" below.

## What belongs where (base model vs pipeline)
- **Base-model evaluation** (`train.py`, `training/synth_eval*`): one 512-token window at a time. Question: how good is the NER model on each label,
  including rare labels and ambiguous words. Real val/test + `rare_hard` + `ambiguous`.
- **Pipeline evaluation** (`inference/dummy_data`, `score_dummy.py`): chunking, rules (dates, IDs, e-mail), propagation, masking. Question: does a whole
  document come out clean. Long documents are mainly a pipeline question (chunking, window edges, repeated names, memory/time). `t5_long` has
  only 4 documents (700 to 5000 words): too few; `training/synth_eval/long.jsonl` (24 docs) tests mostly the window logic in the training
  evaluation, not the model. Proposal (not yet done, needs a decision): keep `long` in train.py as a small window-edge sanity check reported
  separately, and put the weight on a larger pipeline long set (e.g. 20+ documents of several lengths).
- The two never share texts, seeds or numbers. Pipeline date truth is one span per date, the model emits pieces: `score_dummy.py` must accept both.

## To do (in this order)
0. **Check the data for duplicates before trusting any score** (the original annotator confirmed copies with different annotations remain, e.g. `second_<id>.txt`): `python -X utf8 training\check_duplicates.py`, read the counts, decide on `--write-clean`. If test documents have twins in train, the numbers of every earlier run are inflated.
1. **Train normally and check the results are good.** On the GPU machine, in `PII`:
   `python -X utf8 training\train.py train` (10 epochs, train split only, best epoch by validation, test scored once).
   Look at `training_run.json`: validation curve, best epoch, test precision/recall/F1/F2. Recall matters most here
   (a missed name is a leak), so watch recall and F2, not only F1.
2. Compare with the previous model (the student's `HeMed_Anonimizer_Model`): score both on the same test split with
   `data_utils.evaluate_char_spans` (exact character spans), not only token metrics.
3. Production model: `train-all --epochs <best epoch>` from step 1, then hand over the folder (see "Where models live").
4. Check the base model choice: `model_non_anon` (default) vs `model_anon`, same recipe, compare test numbers.

## Ideas / open questions
- `regular` vs `cons` dataset: which one is the right train/test? (`cons` has a much bigger test split, 1000 documents.)
- A small `compare` command: score several model folders on one test set and print one table.
- Reuse `inference/utils/chunking.py` for the training windows so train and inference cut text the same way.
- Bad annotations: `ignorable_ids.txt` has the known ones; `cv` can surface more (documents that are always wrong).
- Class balance / rare entities (`AGE_ABOVE_89`, `INNER_ID`): look at per-label results before tuning anything.
- Try other base models (heBERT, DictaBERT, HeRo, AlephBERT) with the same split.

## Where models live
- Training writes to `training/models/<base>_<timestamp>/` (git-ignored). Nothing there is ever committed.
- Good models are copied by hand to a general folder **outside the repo** on the GPU machine. Keep the run's
  `training_run.json` next to the model; it says exactly how the model was made.
- To let colleagues try models, they clone the repo and copy model folders into `models/` (git-ignored, one sub-folder per
  model; `models/HeMed_Anonimizer_Model` is the default). Then `deidentify.bat file.csv label <model_name>` runs one of them,
  and the output files carry the model name so several runs can be compared.

## Remote machine notes (hospital network)
- `pip` (PyPI) works. `uv` and `download.pytorch.org` / `download-r2.pytorch.org` do not (proxy certificate / 503):
  do not reinstall torch there. Use `PII`; `hemed` (torch 2.1 + cu121) also has CUDA. `deid` and `base` are CPU-only.
- `tools/remote_check.ps1` is the one-command check (`-Install`, `-Smoke`, `-ScanEnvs`); it writes `remote_report.txt`.
- Patient and training data are never committed; the tools print metadata only.
