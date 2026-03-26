import pandas as pd
import os
import argparse
import nltk
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
from typing import List, Tuple, Any

def setup_nltk() -> None:
    """
    Ensures all required NLTK components are downloaded and available.
    Specifically checks for 'punkt' and 'punkt_tab'.
    """
    # 'punkt' is the legacy resource, 'punkt_tab' is the new standard
    for package in ['punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            print(f"--- Downloading NLTK resource: {package} ---")
            nltk.download(package)

def get_pipeline(model_path: str) -> Any:
    """
    Detects hardware and initializes the NER pipeline.

    Args:
        model_path (str): The path to the Hugging Face model directory or repository name.

    Returns:
        Any: The initialized Hugging Face token classification pipeline.
    """
    device: Any
    if torch.cuda.is_available():
        device = 0
        print("--- Using GPU (CUDA) ---")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("--- Using Apple Silicon GPU (MPS) ---")
    else:
        device = -1
        print("--- Using CPU ---")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    return pipeline(
        "ner", 
        model=model, 
        tokenizer=tokenizer, 
        aggregation_strategy="simple",
        device=device
    )

def apply_ner_mask(text: str, ner_pipe: Any, mode: str) -> Tuple[str, str]:
    """
    Processes a single string: splits into sentences, runs NER, and masks the text.

    Args:
        text (str): The original text to be de-identified.
        ner_pipe (Any): The Hugging Face NER pipeline.
        mode (str): The masking mode ('label', 'mask', or 'x').

    Returns:
        Tuple[str, str]: A tuple containing the de-identified text and a log of found entities.
    """
    sentences = nltk.sent_tokenize(str(text))
    processed_sentences = []
    entities_found = []

    for sent in sentences:
        results = ner_pipe(sent)
        # Sort reverse to avoid index shifting during string manipulation
        sorted_results = sorted(results, key=lambda x: x['start'], reverse=True)
        
        for ent in sorted_results:
            entities_found.append(f"{ent['word']}({ent['entity_group']})")
            start, end = ent['start'], ent['end']
            
            if mode == "label":
                rep = f"[{ent['entity_group']}]"
            elif mode == "x":
                rep = "X" * (end - start)
            else:
                rep = "[REDACTED]"
            
            sent = sent[:start] + rep + sent[end:]
        processed_sentences.append(sent)

    return " ".join(processed_sentences), " | ".join(entities_found)

def run_processing_loop(df: pd.DataFrame, ner_pipe: Any, mode: str) -> Tuple[List[str], List[str]]:
    """
    Iterates through the dataframe and returns processed columns for all records.

    Args:
        df (pd.DataFrame): The input dataframe containing a 'text' column.
        ner_pipe (Any): The initialized Hugging Face NER pipeline.
        mode (str): The chosen masking mode ('label', 'mask', or 'x').

    Returns:
        Tuple[List[str], List[str]]: Two lists corresponding to the de-identified texts and entity logs.
    """
    deid_texts = []
    ent_logs = []

    # tqdm provides the progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc="De-identifying"):
        clean_text, log = apply_ner_mask(row['text'], ner_pipe, mode)
        deid_texts.append(clean_text)
        ent_logs.append(log)

    return deid_texts, ent_logs

def save_output(df: pd.DataFrame, original_path: str, deid_col: List[str], ent_col: List[str], mode: str) -> None:
    """
    Constructs the output path and saves the dataframe with new columns to a CSV file.

    Args:
        df (pd.DataFrame): The original input dataframe.
        original_path (str): The file path of the input CSV.
        deid_col (List[str]): List of processed text strings.
        ent_col (List[str]): List of entity log strings.
        mode (str): The masking mode used, which acts as a suffix in the output filename.
    """
    output_df = pd.DataFrame({
        'de_identified_text': deid_col,
        'identified_entities': ent_col
    })
    
    base, ext = os.path.splitext(original_path)
    output_path = f"{base}_deid_{mode}{ext}"
    output_df.to_csv(output_path, index=False)
    print(f"\nSuccess! File saved to: {output_path}")


def run(input_file: str, model_path: str = "HeMed_NER_baseline", mode: str = "label") -> None:
    """
    The main execution flow: sets up dependencies, loads model and data, processes, and saves.

    Args:
        input_file (str): Path to the input CSV file containing a 'text' column.
        model_path (str): Path to the model or Hugging Face repository.
        mode (str): Masking mode to use ('label', 'mask', or 'x').
    """
    setup_nltk()
    df = pd.read_csv(input_file)
    
    if 'text' not in df.columns:
        print("Error: Input CSV must have a 'text' column.")
        return
    
    # 2. Load Pipeline
    ner_pipe = get_pipeline(model_path)

    # 3. Core Logic
    deid_col, ent_col = run_processing_loop(df, ner_pipe, mode)

    # 4. Save
    save_output(df, input_file, deid_col, ent_col, mode)


def main():

    # Run from command line
    parser = argparse.ArgumentParser(description="NER De-identification Tool")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--mode", choices=["label", "mask", "x"], default="label", help="How to mask identified entities")
    args = parser.parse_args()

    run(args.input, args.model, args.mode)

    # Example on how to run directly from script 
    # run(r"D:\DeID NOA\DeIdentification\ICHI\test.csv")


if __name__ == "__main__":
    main()