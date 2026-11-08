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
