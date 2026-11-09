# Ichilov Hebrew Medical Text Anonymizer

Removes personal details (names, dates, places, IDs, contacts, ...) from Hebrew medical text, using regex rules plus a
fine-tuned BERT NER model. **Everything runs locally; no text leaves the machine.**

## For deployers: get it running (3 steps)

You received this repository (or a zip of it) and the model on a USB stick.

1. **Install once.** Needs Python 3.12. Double-click `setup.bat` (Linux/macOS: `./setup.sh`).
2. **Place the model.** Copy the model folder from the USB so that this exists:
   `models\HeMed_Anonimizer_Model\config.json` (next to `model.safetensors` and the tokenizer files). The `models` folder can hold several models, each in its own sub-folder.
3. **Run.** Prepare a CSV with the columns `id` and `text` (UTF-8), then:
   ```
   deidentify.bat C:\path\to\your_file.csv
   ```
   Linux/macOS: `./deidentify.sh /path/to/your_file.csv`.
   Results are written next to your input file: a CSV (original columns plus `de_identified_text` and
   `identified_entities`) and a JSON with exact entity positions.

Optional second argument: `label` (default, `[FIRST_NAME]`), `mask` (`[REDACTED]`), or `x` (`XXXX`), e.g. `deidentify.bat file.csv mask`.
Optional third argument: the name of a model folder inside `models\`, e.g. `deidentify.bat file.csv label MyOtherModel`. The result files of that run carry the model name, so you can run several models on the same file and compare them.
Try it first on the sample: `deidentify.bat inference\dummy_data\test.csv`. Without `setup.bat` (e.g. in a conda env) it uses the `python` of the active environment, so `python inference/de_identify.py --input file.csv` works too. `inference\run_dummy.bat` runs the model on the samples and commits the results to `inference/dummy_results/` for review. Full step-by-step instructions for testers: [`DEID_INSTRUCTIONS_RUN.md`](DEID_INSTRUCTIONS_RUN.md).

A GPU is used automatically if present; otherwise it runs on CPU (slower, same results).
If something fails, run `setup.bat` again: it prints what is missing. Details and advanced options: [`inference/README.md`](inference/README.md).

## For developers

The repository has two independent parts that meet at one artifact, the trained **model folder**. Deployers can ignore `training/`.

```
.
├── setup.bat / deidentify.bat   deployer entry points (.sh equivalents too)
├── inference/                   the de-identification pipeline  (README.md, README_Dev.md)
├── training/                    fine-tuning a model, ported from Nminsker/MedNER  (README.md)
├── tools/                       machine/model inspection scripts
└── data/, models/               git-ignored: real annotated data and trained models (tiny fake samples: inference/dummy_data, training/dummy_data)
```

| I want to...                      | Go to |
|-----------------------------------|-------|
| Extend the pipeline (new stage)   | [`inference/README_Dev.md`](inference/README_Dev.md) |
| Train / evaluate / compare models | [`training/README.md`](training/README.md) |

Two separate Python environments, one per job (each is just `pip install -r ...`):

| Environment | Install | Used for |
|---|---|---|
| run | `pip install -r requirements.txt` (what `setup.bat` does) | `inference/` |
| train / develop | `pip install -r requirements-training.txt` (with a CUDA torch) | `training/`, tests |

Run scripts by path
(`python training/train.py`, `python inference/de_identify.py`), not with `-m`.
Training writes models to `training/models/` (git-ignored); good ones are copied by hand outside the repo. For others to try several models, they clone the repo and put each model in its own folder under `models/` (git-ignored; `models/HeMed_Anonimizer_Model` is the default one). Pick one with `--model <folder name>`.

To-do lists: [`training/TODO.md`](training/TODO.md), [`inference/TODO.md`](inference/TODO.md).
