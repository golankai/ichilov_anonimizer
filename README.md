# BERT NER DE-IDENTIFICATION PIPELINE

## OVERVIEW
This project provides a complete pipeline for de-identifying medical text in Hebrew using Named Entity Recognition (NER) model.

The pipeline is designed to handle long documents by chunking the text
into individual sentences using NLTK, rather than processing them all at once. 
This avoids the token-length limits of standard BERT models.

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

3. Download NLTK data (Required for sentence splitting):
   The script will attempt this automatically, but you can run:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
   ```

   *Note: 'punkt_tab' is the required resource for modern NLTK versions.

## FILE STRUCTURE
- `de_identify.py` : Inference script (Processes CSVs and masks text).
- `requirements.txt` : List of Python dependencies.

## DE-IDENTIFICATION (de_identify.py)
Run the script from the command line to process a CSV file. The input CSV **must** contain both `id` and `text` columns.

### OUTPUT FORMATS
The pipeline automatically generates two matched output files in your directory:
1. **CSV Output (`*_deid_<mode>.csv`)**: Retains 100% of your original data columns (including `id` and `text`), seamlessly appending a `de_identified_text` column and an `identified_entities` column formatted for human grading (e.g., `John(FIRST_NAME)`).
2. **JSON Output (`*_deid_<mode>.json`)**: A machine-readable structured JSON array mapping out the `id`, original `text`, and a list of entities perfectly tracking the global document `start` and `end` character offsets along with clean base labels (ignoring BIO tag groupings).

Usage:
```bash
python de_identify.py --input <path_to_csv> --model <path_to_model> --mode <mode>
```

Arguments:
- `--input`  : Path to your input CSV file.
- `--model`  : (Optional) Path to your trained model folder (e.g., `HeMed_NER_baseline`).
- `--mode`   : (Optional) 
           'label'    -> Replaces with [ENTITY_TYPE] (e.g., [PER])
           'mask'     -> Replaces with [REDACTED]
           'x'        -> Replaces with 'XXXX'

Usage Example:
```bash
python de_identify.py --input test.csv --model HeMed_NER_baseline --mode label
```

## TROUBLESHOOTING
Q: I get a LookupError for 'tokenizers/punkt'?
A: This means the NLTK data is missing or in the wrong path. 
   Ensure you have run the 'nltk.download' commands mentioned in the 
   Installation section.

## CONTACT
- nminsker@gmail.com
- kai.golanhashiloni@post.runi.ac.il
