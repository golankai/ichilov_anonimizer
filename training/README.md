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
training\run_train.bat                    # = python -X utf8 training\train.py train  (6 epochs max, early stopping, seeds 42 43 44)
training\run_train.bat --seeds 42         # one seed only
training\push_results.bat                 # publish the metadata (counts and metrics only) when it is done
```
(Run it in the `PII` conda env, the one with CUDA torch.)

(`PII` is the conda env on that machine with CUDA torch 2.6.0+cu124. The hospital proxy blocks `download.pytorch.org`, so do not try to reinstall torch there.)

No GPU or no real data? Try the pipeline on the tiny fake sample: `python training/train.py train --data-dir training/dummy_data --model <any BERT> --epochs 1`.
`tools/remote_check.ps1 -Smoke` runs the tests and a 1-epoch training in one command and writes `remote_report.txt`.

Defaults: base model `D:\HeMed All\HeMed Final Model\model_non_anon`, data `D:\De ID All\NER_anon_dataset\regular`
(a ready-made train/validation/test split; `...\cons` is the other variant), output `models\<model>_<timestamp>`.
For the anonymized-corpus base model add `--model "D:\HeMed All\HeMed Final Model\model_anon"`.
Each seed trains up to 6 epochs, stops early after 2 epochs without a better validation F1 (`--patience`, 0 = off), keeps the
best epoch by validation F1, and is then scored on the real test split and on the synthetic sets (below).

## Checking the data for duplicates (counts only)

The split code compares ids, never texts. The same document can exist twice under different ids (e.g. `second_<id>.txt`),
sometimes with different annotations; copies in train and test inflate the test score, and copies that disagree teach the
model contradictions. `check_duplicates.py` reports this without printing any text or id:

```
training\run_check_duplicates.bat                                  # same as the next line, and saves the report to training\duplicates_report.txt
python -X utf8 training\check_duplicates.py                        # default split folder of the GPU machine
python -X utf8 training\check_duplicates.py --data-dir <folder>    # or --data one.jsonl
```

It prints: documents per split, ids with a `second_`/`copy_` prefix, exact duplicates (after whitespace normalisation) split
into identical and **conflicting** annotations, how many test/validation documents also occur in train, near duplicates
(5-word shingles, Jaccard >= 0.9), how many ids of `ignorable_ids.txt` are present, and what a cleaning would leave.
Optional: `--groups f.json` (ids of each group, keep out of git), `--write-clean <dir>` (cleaned copy; originals are never
touched; identical copies keep the evaluated one, conflicting ones are dropped, `--conflicts first` keeps one,
`--drop-near-from-train` also removes train near-twins of validation/test documents). Note that the default `train` mode
reads pre-made split folders and does not apply `ignorable_ids.txt`.

## Two kinds of evaluation (keep them apart)

| | Base model | Full pipeline |
|---|---|---|
| What is scored | the fine-tuned token classifier on pre-tokenized windows, as trained | `inference/de_identify.py` on raw text: chunking, model, rules, post-processing |
| Where | `train.py` after every run | `inference/run_dummy.bat` / `inference/score_dummy.py` |
| Data | real validation/test splits + the synthetic sets in `training/synth_eval/` | `inference/dummy_data/` (5 tiers, with ground truth) |
| Metric | entity-level seqeval P/R/F1 (overall and per label), token-level F2 | character coverage per tier/label/situation, leaks, over-masking |

The two fake sets are generated separately (different seeds and templates, no shared text; a test checks that). A pipeline problem (a split
date, an email prefix) does not show in the base-model numbers, and a weak label in the base model does not show as such in the pipeline numbers.

**Synthetic sets for the base model** (`training/synth_eval/`, see its README): `rare_hard` (586 short documents with 40+ examples of every rare
label, name/word collisions, prefixes and look-alike numbers that must stay unmasked) and `long` (24 documents of 450-2500 words, several windows).
They are scored after training, reported separately, and **never select the epoch**. Regenerate with `python training/make_synth_eval.py`;
skip them with `--no-synth-eval`; use another folder with `--synth-eval-dir`.

## Seeds and the summary table

`train` repeats the whole run for each seed (default `42 43 44`; change with `--seeds`) and writes

```
training/models/<model>_<timestamp>/seed_42/ ... seed_43/ ... seed_44/     the models (git-ignored), each with training_run.json
training/models/<model>_<timestamp>/summary.json, summary.md              mean ± std over seeds
training/runs/<model>_<timestamp>/seed_<n>/training_run.json              committed copies (metadata only)
training/runs/<model>_<timestamp>/summary.json, summary.md
```

`summary.md` has: overall P/R/F1/F2 for validation (best epoch), test and each synthetic set (mean ± sample std, min-max, value per seed);
a per-label table for each of them; and one row per seed (best epoch, epochs run, early stop, minutes, commit). A label that never occurs in a set has
support 0 and no recall/F1 (undefined, not 0). With three seeds the std is itself rough: treat differences smaller than about two std as noise.

Re-create or compare summaries at any time, e.g. on the dev machine after `git pull`:
`python training/aggregate_runs.py training/runs/<experiment> [training/runs/<other experiment> ...]` (several folders also print a comparison table).
Per-label results are recorded for every epoch, the test split and the synthetic sets (`results.per_epoch_validation[].per_label`, `results.test.per_label`,
`results.synthetic.<set>.per_label` in each `training_run.json`).

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
A copy goes to `training/runs/<run name>/training_run.json`, which **is** committed (models are not). After a run on the GPU machine, double-click or run `training\push_results.bat` to commit and push the metadata, then `git pull` here to see it.

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

Hyperparameters: `--lr 5e-5 --batch-size 8 --epochs 6 --patience 2 --weight-decay 0.01` (MedNER used 10 epochs and no early stopping; the first
baseline here peaked at epoch 4 and overfit afterwards).
To compare several base models: `for m in avichr/heBERT dicta-il/dictabert HeNLP/HeRo; do python training/train.py train --model $m --data-dir data/splits --output-dir models/$(basename $m); done`

Evaluate any pipeline on exact character spans with `data_utils.evaluate_char_spans(records, pipe)`.

## Layout

| File | Role |
|------|------|
| `run_record.py` | Writes `training_run.json` (config + results) beside every model |
| `train.py` | CLI and training loops (`split`, `train`, `train-all`, `cv`, `final`) |
| `aggregate_runs.py` | Mean/std over the seeds of one experiment -> `summary.json` / `summary.md`; compares experiments |
| `make_synth_eval.py`, `synth_eval/` | Generator and files of the synthetic evaluation sets for the base model |
| `run_train.bat`, `run_check_duplicates.bat`, `push_results.bat` | One-command wrappers for the GPU machine |
| `dataset.py` | JSONL loading, label cleaning, stratified split, sliding-window tokenization + label alignment |
| `labels.py` | Label set, BIO maps, label cleaning |
| `metrics.py` | Trainer metrics: seqeval P/R/F1 (overall and per label) and token-level F2 (recall-weighted) |
| `data_utils.py` | Span merging, word-level IOB, character-span evaluation |
| `ignorable_ids.txt` | Documents with known-bad annotations |
| `check_duplicates.py` | Duplicate / leakage report (counts only) and optional cleaned copy of the splits |
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
