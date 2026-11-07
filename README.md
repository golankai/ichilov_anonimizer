# Ichilov Hebrew Medical Text Anonymizer

De-identification of Hebrew medical text with a mix of regex rules and a fine-tuned BERT NER model.
The repository has two independent parts that meet at one artifact, the **trained model directory**.

```
.
├── training/    Fine-tune a Hebrew BERT on annotated data  ->  produces a model directory
├── inference/   Run the de-identification pipeline         <-  consumes a model directory
├── pyproject.toml / requirements.txt / requirements-training.txt
└── data/, models/   (git-ignored) your annotated data and trained models
```

| I want to...                         | Go to                                      |
|--------------------------------------|--------------------------------------------|
| De-identify a CSV of documents       | [`inference/README.md`](inference/README.md) |
| Extend the pipeline (new stage)      | [`inference/README_Dev.md`](inference/README_Dev.md) |
| Train / evaluate / compare models    | [`training/README.md`](training/README.md)   |

## Setup

```bash
uv sync                      # inference only
uv sync --group training     # + training dependencies (datasets, seqeval, scikit-learn, ...)
```

(`pip install -r requirements.txt`, plus `-r requirements-training.txt` for training, also works.)

## Quick start

```bash
# train (see training/README.md for data format and splitting)
uv run --group training python training/train.py train \
    --model <hf-id-or-path> --data-dir data/splits --output-dir models/my_model

# de-identify with the result
uv run python inference/de_identify.py --input inference/dummy_data/test.csv --model models/my_model --mode label
```

Both parts use top-level imports relative to their own folder, so run the scripts by path
(`python training/train.py`, `python inference/de_identify.py`) rather than `python -m`.
Training originated from [Nminsker/MedNER](https://github.com/Nminsker/MedNER).
