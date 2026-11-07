# De-identification regression tests

File: `inference/tests/test_deidentification_regression.py`. Run: `pytest inference/tests`.

There is a single implementation (`inference/de_identify.py` + `inference/utils/`). The tests need no model: the
Hugging Face NER pipeline is replaced by small fakes, so they cover everything around the model.

## Fakes
- `NoNer` - returns nothing (used to get the text as normalized by the regex stages).
- `SubstringNer({"John": "FIRST_NAME"})` - tags every occurrence of the given substrings.
- `ReplayNer(full_text, entities)` - replays gold entities (global offsets) sentence by sentence.

## Tests
- **`test_pipeline_matches_gold`** (one per record of `dummy_data/test_deid_label_pipeline.json`): the full pipeline with
  ID/date normalization reproduces the curated `deid text` and `entities`.
- **`test_masking_modes`**: `label` -> `[LABEL]`, `mask` -> `[REDACTED]`, `x` -> `XXXX`.
- **`test_entities_have_global_offsets_across_sentences`**: sentence-relative offsets are mapped back to document offsets.
- **`test_bio_prefixes_are_stripped_from_labels`**, **`test_numeric_date_is_normalized_before_ner`**,
  **`test_ner_hits_inside_normalization_placeholders_are_ignored`**, **`test_non_string_text_is_coerced`**.
- **`test_run_processing_loop_returns_one_result_per_row`** and
  **`test_save_output_writes_consistent_csv_and_json`** (outputs are named `<input>_deid_<mode>_pipeline.{csv,json}`).

## Known quirk
Entity offsets are not in one coordinate system. Entities from the model point into the *normalized* text, while the
regex ID stage records its entity with offsets into the text as it was when that stage ran (so `entities[i].start/end`
do not always index the original `text`). `test_pipeline_matches_gold` documents and works around this.
