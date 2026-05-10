## De-identification Regression Tests – Design

### Goals

- **Protect core de-identification behavior**: Detect regressions whenever the NER model, tokenization, or masking logic changes.
- **Support multiple tools and configurations**: Allow testing different NER backends or masking modes, both individually and as a pipeline.
- **Stay lightweight by default**: Make fast, model-free tests the default, with optional slow tests that run full models when available.

### Current Pipeline Surfaces Under Test

- **`apply_ner_mask(text, ner_pipe, mode)`** (in `de_identify.py`):
  - Takes a raw document, splits it into sentences (NLTK), calls a Hugging Face-style NER pipeline, and:
    - Builds a de-identified text string by applying replacements with global character offsets.
    - Produces a structured list of entities with global `start`/`end` offsets and labels.
- **`save_output(df, original_path, deid_col, ent_col, mode)`** (in `de_identify.py`):
  - Adds `de_identified_text` and `identified_entities` to the original dataframe and writes:
    - A CSV (`*_deid_<mode>.csv`) suitable for human grading.
    - A JSON (`*_deid_<mode>.json`) for programmatic use.

These functions are pure with respect to the model choice and I/O side effects, making them suitable anchors for regression tests.

### Test Directory Layout

- **Location**: All tests live under a dedicated `tests/` directory (pytest best practice).
- **Pilot file**: `tests/test_deidentification_regression.py`
  - Uses `dummy_data/test_deid_label_2.csv` and `dummy_data/test_deid_label_2_deid_label.json` as curated gold examples.
  - Additional tests for other tools or configurations can be added in the same directory as separate `test_*.py` files.

### Pilot Regression Tests (Implemented)

- **1. `test_apply_ner_mask_matches_gold_text_and_entities`**
  - **Input**: Each row from `dummy_data/test_deid_label_2_deid_label.json`:
    - `text`: original document.
    - `deid text`: expected fully de-identified string.
    - `entities`: list of entities with global `start`, `end`, `text`, and `label`.
  - **Technique**:
    - A `FakeNerPipeline` is constructed per example, replaying the gold `entities` but exposing them through the same interface as the Hugging Face aggregation pipeline:
      - For each sentence slice, it returns entities with sentence-relative `start`/`end` and an `entity_group` field.
    - `apply_ner_mask` is invoked with the original text and the fake pipeline in `"label"` mode.
  - **Assertions**:
    - The returned de-identified string exactly equals the gold `deid text`.
    - The returned structured entity list exactly matches the gold `entities` (same global offsets and labels).
  - **What this protects**:
    - Sentence boundary handling (NLTK chunking and span reconstruction).
    - Global offset computation and replacement order.
    - Label cleaning and the text-masking logic.

- **2. `test_save_output_produces_consistent_csv_and_json`**
  - **Input**: The same gold examples, projected into:
    - A synthetic in-memory dataframe with `id` and `text`.
    - Lists of expected de-identified texts and entity structures.
  - **Execution**:
    - Writes an input CSV into a `tmp_path` (pytest’s temporary directory).
    - Calls `save_output` to produce `*_deid_label.csv` and `*_deid_label.json`.
  - **Assertions**:
    - The output CSV exists and its `de_identified_text` column matches the expected de-identified strings.
    - For the output JSON, each entity’s `text` matches `row["text"][start:end]` (offset consistency check).
  - **What this protects**:
    - CSV/JSON schema and field naming.
    - The human-readable `identified_entities` column content (indirectly, via use of the same inputs).
    - Offset correctness in the JSON output.

### Future Extensions

- **Model-backed end-to-end tests (optional, slow)**
  - Add tests that:
    - Load the actual NER model from a configurable path (e.g. `HEMED_MODEL_PATH` env var or a default like `HeMed_NER_baseline`).
    - Run `run_processing_loop` and `save_output` on a small subset of `dummy_data` inputs.
    - Compare the produced de-identified texts and entities against the curated gold with a tolerance policy (e.g. allow minor label changes while enforcing correct masking of PHI).
  - Mark these tests with `@pytest.mark.slow` and/or skip automatically when the model directory is missing.

- **Multiple tools in a pipeline**
  - Introduce an abstraction (e.g. a simple pipeline configuration object) describing:
    - Tokenization/sentence splitter (NLTK or alternatives).
    - NER engine (local model, remote API, rule-based anonymizer, safe-harbor rules, etc.).
    - Post-processing steps (label mapping, conflict resolution).
  - For each tool:
    - Unit tests on its own behavior using small, hand-crafted fixtures.
  - For combinations:
    - Scenario tests where a given configuration is run end-to-end on the same input, and performance is compared:
      - Quantitative: precision/recall/F1 on labels, per-entity type, vs. gold entities in the JSON.
      - Qualitative: how many PHI tokens are missed or over-masked.

- **Metric-focused regression**
  - Add helpers that take:
    - A gold JSON file (like `test_deid_label_2_deid_label.json`).
    - A predicted JSON file from a given run.
  - Compute:
    - Exact-span metrics (exact match on `start`, `end`, `label`).
    - Token-level metrics (overlapping spans).
  - Use thresholds to guard against regressions:
    - For example, “overall F1 on PERSON-like labels must stay ≥ X on the curated set”.

### How to Run the Tests

- From the project root (`ichilov_anonimizer`):

```bash
pytest
```

- To run just the de-identification regression tests:

```bash
pytest tests/test_deidentification_regression.py
```

Or (for `uv` environements)
```bash
uv run pytest tests/test_deidentification_regression.py
```
For verbose:
```bash
uv run python -m pytest -v tests/test_deidentification_regression.py
```






OR for full non-slow:
```bash
uv run python -m pytest -m "not slow"
```

- In future, slow tests that require the full NER model will be:
  - Marked with `@pytest.mark.slow`.
  - Skipped automatically if the model folder or configuration is not available.

