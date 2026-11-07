# Training

Fine-tunes a Hebrew BERT (HeMed, heBERT, DictaBERT, HeRo, AlephBERT, ...) for token-classification NER
over 10 entity types: `FIRST_NAME LAST_NAME DATE AGE_ABOVE_89 LOC ORG MED_ORG CONTACT ID INNER_ID`.
The output is a standard Hugging Face model directory that `inference/de_identify.py --model <dir>` loads directly.

Ported from [Nminsker/MedNER](https://github.com/Nminsker/MedNER); see "Differences from MedNER" below.

## Setup

```bash
pip install -r requirements-training.txt   # needs a CUDA build of torch for the GPU
```

## Data

One JSON object per line; `end` is exclusive:

```json
{"id": "12651673.txt", "text": "...", "label": [{"start": 0, "end": 4, "label": "FIRST_NAME"}]}
```

Real data goes under `data/` (git-ignored; it contains patient text and must never be committed).
On load, labels are cleaned (`AGE\_ABOVE\_89` -> `AGE_ABOVE_89`), `DAY/MONTH/YEAR` become `DATE`, any label outside the 10 above
is dropped, and documents listed in `ignorable_ids.txt` are skipped by the `--data` modes (`cv`, `final`, `train-all --data`). If that file is absent, nothing is skipped. Pre-made split folders (`train`) are never filtered.

## Quick start on the GPU machine (replicates MedNER's last run)

```powershell
cd "D:\DeID KAI\ichilov_anonimizer"
C:\ProgramDatanaconda3\envs\PII\python.exe -X utf8 training	rain.py train
```

(`PII` is the conda env on that machine with CUDA torch 2.6.0+cu124. The hospital proxy blocks `download.pytorch.org`, so do not try to reinstall torch there.)

No GPU or no real data? Try the pipeline on the tiny fake sample: `python training/train.py train --data-dir training/dummy_data --model <any BERT> --epochs 1`.
`tools/remote_check.ps1 -Smoke` runs the tests and a 1-epoch training in one command and writes `remote_report.txt`.

Defaults: base model `D:\HeMed All\HeMed Final Model\model_non_anon`, data `D:\De ID All\NER_anon_dataset\regular`
(a ready-made train/validation/test split; `...\cons` is the other variant), output `models\<model>_<timestamp>`.
For the anonymized-corpus base model add `--model "D:\HeMed All\HeMed Final Model\model_anon"`.
After training, the best epoch (by validation F1) is also scored on the test split.

## Development vs production

1. **Develop** with `train`: it trains on the *train* split only, picks the best epoch on *validation*, and scores
   *test* once at the end. Change one thing at a time and compare the test/validation numbers in `training_run.json`.
2. **Produce** the final model with `train-all --epochs <best epoch from step 1>`: it merges train + validation + test
   and trains with no held-out data. Nothing selects the epoch there, so reuse the best epoch from development.
   Its run file says so (`notes`), and no honest score exists for it: judge it by the development run of the same recipe.

## Run documentation

Every run writes **`training_run.json`** into its output folder, next to the model: exact command, git commit
(and whether `training/` had uncommitted changes), base model, data files with SHA-256 / sizes / per-split document and
per-label entity counts, labels, windowing settings, all hyperparameters, Python/package/GPU info, and the results
(per-epoch validation metrics, best epoch, test metrics, duration). It is written when training starts and finalized at
the end (`status`: `running` -> `completed` or `failed`, with the error). It contains counts only, never document text.

## Workflow

```bash
# 1. stratified 80/10/10 split -> data/splits/{train,validation,test}.jsonl
python training/train.py split --data data/annotated.jsonl --out data/splits

# 2. train, keeping the best epoch by validation F1
python training/train.py train --model <hf-id-or-path> --data-dir data/splits --output-dir models/HeMed_NER

# optional: k-fold CV to spot noisy annotations / unstable results
python training/train.py cv --model <...> --data data/annotated.jsonl -k 5 --output-dir models

# 3. production model: 95/5 split with early stopping, or all data with no validation
python training/train.py final     --model <...> --data data/annotated.jsonl --output-dir models/Final
python training/train.py train-all --model <...> --data data/annotated.jsonl --output-dir models/Final_all
```

Hyperparameters (defaults match MedNER): `--lr 5e-5 --batch-size 8 --epochs 10 --weight-decay 0.01`.
To compare several base models: `for m in avichr/heBERT dicta-il/dictabert HeNLP/HeRo; do python training/train.py train --model $m --data-dir data/splits --output-dir models/$(basename $m); done`

Evaluate any pipeline on exact character spans with `data_utils.evaluate_char_spans(records, pipe)`.

## Layout

| File | Role |
|------|------|
| `run_record.py` | Writes `training_run.json` (config + results) beside every model |
| `train.py` | CLI and training loops (`split`, `train`, `train-all`, `cv`, `final`) |
| `dataset.py` | JSONL loading, label cleaning, stratified split, sliding-window tokenization + label alignment |
| `labels.py` | Label set, BIO maps, label cleaning |
| `metrics.py` | Trainer metrics: seqeval P/R/F1 and token-level F2 (recall-weighted) |
| `data_utils.py` | Span merging, word-level IOB, character-span evaluation |
| `ignorable_ids.txt` | Documents with known-bad annotations |
| `tests/` | Offline tests (toy tokenizer, no downloads): `pytest training/tests` |

Long documents are cut into 512-token windows with a 128-token overlap. Inside multi-window documents the 64 tokens at each inner
window edge get label `-100`, so every token is trained and scored once, with context on both sides.

## Differences from MedNER

- **No hardcoded `D:\...` paths**: everything is a CLI argument; the CLI replaces editing the `__main__` block.
- **transformers 5 compatibility**: `Trainer(tokenizer=...)` became `processing_class=...`.
- **Dependencies**: `scikit-multilearn>=0.2.2` does not exist (latest is 0.2.0); `jsonl`/`jsonlines` is replaced by the stdlib `json`.
- **Clean label names**: backslashes are stripped on load instead of hardcoding `AGE\\_ABOVE\\_89` in the label list.
- **Split fixes**: the "no overlap" assertion used `and` (it almost never fired) and now checks every pair; the split is seeded.
  Splits are written to disk (the dump was commented out).
- **`evaluate_char_spans`** now accepts the dataset's dict spans; the original only worked with `[start, end, tag]` triples, so it failed on real records.
- **Removed as dead code**: `tokenize_and_align_labels` and `tokenize_and_align_labels_sliding` (superseded by `..._sliding_window`),
  `compute_metrics_sliding` (assumes margins that the labels already mask), and the unused `FULL_DATASET`/`get_all_labels` plumbing.
  The "original" (non-mandatory) label mode was dropped too, since it only existed behind a commented-out flag.
