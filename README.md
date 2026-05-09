# BERT NER DE-IDENTIFICATION PIPELINE

## OVERVIEW

This project provides a complete pipeline for de-identifying medical text in Hebrew using a mix of **classic regex-based methods** and a **neural NER model**.

The de-identification is implemented as a **linear, stage-based pipeline**:

- Regex stages: normalize IDs → normalize dates.
- Network stage: NER masking over sentences.
- Post-processing: label cleanup.

This makes each step a small, testable building block while still leveraging a powerful NER model. The pipeline is designed to handle long documents by chunking the text into individual sentences using NLTK, rather than processing them all at once, which avoids the token-length limits of standard BERT models.

## IDENTIFIED ENTITIES

Our local model (`HeMed_NER_baseline`) is trained to identify and mask the following entities:

- `FIRST_NAME`
- `LAST_NAME`
- `DATE`
- `AGE_ABOVE_89`
- `LOC` (Location)
- `ORG` (Organization)
- `MED_ORG` (Medical Organization)
- `CONTACT` (Contact Information)
- `ID` (ID number)
- `INNER_ID` (Hospital internal ID)

## INSTALLATION

1. Install Python 3.9 or higher.
2. Install dependencies:
  ```bash
   pip install -r requirements.txt
  ```
   Or:
  ```bash
    uv sync
  ```
3. Download NLTK data (Required for sentence splitting):
  The script will attempt this automatically, but you can run:
   *Note: 'punkt_tab' is the required resource for modern NLTK versions.

## FILE STRUCTURE

- `de_identify.py` : Main CLI script (processes CSVs and masks text via the pipeline).
- `utils/doc_pipeline.py` : Pipeline composition logic (how stages are wired together).
- `utils/pipeline_stages.py` : Reusable pipeline stages (date/ID normalization, NER masking, label cleanup).
- `docs/Deid_Pipeline_Usage.MD` : Brief usage guide with concrete CLI examples and logging options.
- `requirements.txt` : List of Python dependencies.

## DE-IDENTIFICATION (de_identify.py)

Run the script from the command line to process a CSV file. The input CSV **must** contain both `id` and `text` columns.

### OUTPUT FORMATS

The pipeline automatically generates two matched output files in your directory:

1. **CSV Output (`*_deid_<mode>_pipeline.csv`)**: Retains 100% of your original data columns (including `id` and `text`), seamlessly appending a `de_identified_text` column and an `identified_entities` column formatted for human grading (e.g., `John(FIRST_NAME)`).
2. **JSON Output (`*_deid_<mode>_pipeline.json`)**: A machine-readable structured JSON array mapping out the `id`, original `text`, and a list of entities perfectly tracking the global document `start` and `end` character offsets along with clean base labels (ignoring BIO tag groupings).

Usage (basic run):

```bash
python de_identify.py --input <path_to_csv> --model <path_to_model> --mode <mode>
```

Arguments:

- `--input`            : Path to your input CSV file. (**Required**)
- `--model`            : Path to your trained model folder. (**Required**, e.g., `HeMed_NER_baseline` or a HF model name like `dicta-il/dictabert-ner`)
- `--mode`             : (Optional, default: `label`) How to mask identified entities:
      - `label` -> replaces with `[ENTITY_TYPE]` (e.g., `[PER]`)
      - `mask`  -> replaces with `[REDACTED]`
      - `x`     -> replaces with a string of `X` characters
- `--observe-stages`   : (Optional) Log a short summary after each pipeline stage (text length + entity count).
- `--log-replacements` : (Optional) Log each string replacement applied by any stage (original text, span, label, replacement token).
- `-v`, `--verbose`    : (Optional) Enable DEBUG-level logging for this tool (e.g., text snippets and full entity lists per stage).

Usage examples:

```bash
python de_identify.py --input dummy_data/test.csv --model HeMed_NER_baseline --mode label
```

- **With full observability (research / debugging)**:
  ```bash
  python de_identify.py \
    --input dummy_data/test.csv \
    --model HeMed_NER_baseline \
    --mode label \
    --observe-stages \
    --log-replacements \
    --verbose
  ```
  This logs per-stage summaries, detailed per-stage snippets/entities, and every replacement that any stage applies.

For more concrete examples (including logging and different masking modes), see `docs/Deid_Pipeline_Usage.MD`.

## TROUBLESHOOTING

*Q*: I get a LookupError for 'tokenizers/punkt'?  
*A*: This means the NLTK data is missing or in the wrong path. 
   Ensure you have run the 'nltk.download' commands mentioned in the 
   Installation section.

## CONTACT

- [nminsker@gmail.com](mailto:nminsker@gmail.com)
- [kai.golanhashiloni@post.runi.ac.il](mailto:kai.golanhashiloni@post.runi.ac.il)
- [assaf@razon-family.com](mailto:assaf@razon-family.com)

