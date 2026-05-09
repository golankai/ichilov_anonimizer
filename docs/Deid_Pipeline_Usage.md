### De-identification pipeline – how to run and test

This document explains **how to run and test** `de_identify.py` from the project root with `uv`, both **with** and **without** logging.

All commands assume:

- **Working directory**: `ichilov_anonimizer` project root.
- **Shell**: Windows PowerShell.
- **Input file**: `.\dummy_data\test_deid_label_2.csv` (has at least `id` and `text` columns).
- **Model**: `dicta-il/dictabert-ner` (or any compatible NER model / local path).

If you are using **bash / Linux / macOS** instead of PowerShell:

- Replace `.\de_identify.py` with `./de_identify.py`.
- Replace `.\dummy_data\test_deid_label_2.csv` with `./dummy_data/test_deid_label_2.csv`.
- Use `\` for line continuation instead of PowerShell's backtick, for example:
  ```bash
  uv run ./de_identify.py \
    --input ./dummy_data/test_deid_label_2.csv \
    --model dicta-il/dictabert-ner \
    --mode label
  ```

---

### 1. Basic run (no logging)

This is the simplest way to run the new pipeline and verify everything works end‑to‑end.

```powershell
uv run .\de_identify.py `
  --input .\dummy_data\test_deid_label_2.csv `
  --model dicta-il/dictabert-ner `
  --mode label
```

- **What it does**:
  - Runs the full pipeline: `normalize_ids` → `normalize_dates` → `null` → `ner_mask` → `label_clean`.
  - Produces:
    - A CSV: `dummy_data/test_deid_label_2_deid_label_pipeline.csv`.
    - A JSON: `dummy_data/test_deid_label_2_deid_label_pipeline.json`.
- **No extra logging** beyond basic INFO output and any warnings from libraries.

---

### 2. Per‑stage summaries (`--observe-stages`)

Use this when you want to see how the document evolves **after each pipeline stage** (length and entity count).

```powershell
uv run .\de_identify.py `
  --input .\dummy_data\test_deid_label_2.csv `
  --model dicta-il/dictabert-ner `
  --mode label `
  --observe-stages
```

- **Logs (INFO)**:
  - One line per stage, e.g.:
    - `[stage=normalize_ids] text_len=... entities=...`
    - `[stage=normalize_dates] text_len=... entities=...`
    - `[stage=ner_mask] text_len=... entities=...`

To see a short text snippet and the full entities after each stage, add `--verbose`:

```powershell
uv run .\de_identify.py `
  --input .\dummy_data\test_deid_label_2.csv `
  --model dicta-il/dictabert-ner `
  --mode label `
  --observe-stages `
  --verbose
```

- **Additional DEBUG logs**:
  - For each stage, prints:
    - A 200‑character text snippet (newlines collapsed).
    - The full `entities` list at that point.

---

### 3. Detailed replacement logs (`--log-replacements`) – classic test

This is the **classic test** for verifying that replacements are fired correctly from all relevant stages.

```powershell
uv run .\de_identify.py `
  --input .\dummy_data\test_deid_label_2.csv `
  --model dicta-il/dictabert-ner `
  --mode label `
  --log-replacements
```

- **What it logs (INFO)**:
  - One line per replacement, for example:
    - `[stage=normalize_ids] Found string 'ת.ז. 345678901' at [8, 20), label=ID, replace with '[ID]'`
    - `[stage=normalize_dates] Found string '15.03.2024' at [32, 42), label=DATE, replace with '15.03.2024'`
    - `[stage=ner_mask] Found string 'דוד כהן' at [7, 14), label=PER, replace with '[PER]'`
- This is useful to **validate labeling, spans, and masking tokens** without diffing full texts.

You can freely combine this with `--observe-stages` and/or `--verbose` (see next section).

---

### 4. Maximum observability (stages + replacements + verbose)

Use this when debugging tricky cases or validating a new stage:

```powershell
uv run .\de_identify.py `
  --input .\dummy_data\test_deid_label_2.csv `
  --model dicta-il/dictabert-ner `
  --mode label `
  --observe-stages `
  --log-replacements `
  --verbose
```

- **You get all of**:
  - Per‑stage INFO summaries (length + entity count).
  - Per‑stage DEBUG snippets and full entity lists.
  - Per‑replacement INFO logs (original string, span, label, replacement token).

This is the recommended command when you want to understand exactly **how** a specific record is being transformed.

---

### 5. Trying different masking modes (`--mode`)

The `--mode` flag controls **how** entities are masked in the final text:

- `**--mode label`** (default):
  - Replaces each entity with its label in brackets, e.g. `[PER]`, `[DATE]`.
- `**--mode x**`:
  - Replaces each entity with a string of `X` characters matching its original length.
- `**--mode mask**` (or any other value):
  - Falls back to a generic `[REDACTED]` token.

Examples:

```powershell
# Label-based masking
uv run .\de_identify.py `
  --input .\dummy_data\test_deid_label_2.csv `
  --model dicta-il/dictabert-ner `
  --mode label

# Fixed-width X masking
uv run .\de_identify.py `
  --input .\dummy_data\test_deid_label_2.csv `
  --model dicta-il/dictabert-ner `
  --mode x

# Generic [REDACTED] masking
uv run .\de_identify.py `
  --input .\dummy_data\test_deid_label_2.csv `
  --model dicta-il/dictabert-ner `
  --mode mask
```

You can combine any of these with the logging flags from the previous sections.

---

### 6. Quick checklist when something looks wrong

- **No output files?**
  - Check that the input CSV has both `id` and `text` columns.
  - Confirm `--input` path is correct and the file exists.
- **No entities are being found?**
  - Ensure the `--model` path or name is valid and the environment can download/use it.
  - Try running with `--observe-stages` to see if earlier stages are accidentally deleting text.
- **Spans or labels look off in logs?**
  - Run with `--log-replacements --verbose` and inspect:
    - The replacement lines (for spans and labels).
    - The DEBUG snippets around the problematic area.

