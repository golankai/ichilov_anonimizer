# BERT NER DE-IDENTIFICATION PIPELINE

## OVERVIEW
This project provides a complete pipeline for de-identifying medical text in Hebrew using Named Entity Recognition (NER) model.

The pipeline is designed to handle long documents by chunking the text
into individual sentences using NLTK, rather than processing them all at once. 
This avoids the token-length limits of standard BERT models.

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
Run the script from the command line to process a CSV file. The CSV must have a column named "text".

Usage:
```bash
python de_identify.py --input <path_to_csv> --model <path_to_model> --mode <mode>
```

Arguments:
- `--input`  : Path to your input CSV file.
- `--model`  : (Optional) Path to your trained model folder or Hugging Face repository. 
             Note that `HeMed_NER_baseline` is our custom model, however, one can use 
             any BERT-like Hebrew speaking model from Hugging Face.
             Examples:
             - `dicta-il/dictabert`
             - `onlplab/alephbert-base`
             - `imvladikon/alephbertgimmel-base-512`
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
