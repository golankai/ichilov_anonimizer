import pandas as pd
import os
import argparse
import nltk
import torch
import json
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
from typing import List, Tuple, Any, Dict

def setup_nltk() -> None:
    """
    Ensures all required NLTK components are downloaded and available.
    Specifically checks for 'punkt' and 'punkt_tab'.
    """
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
        model_path (str): The path to the trained model directory.

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

def clean_label(label: str) -> str:
    """Strips B- and I- prefixes from labels if they exist."""
    if label.startswith("B-") or label.startswith("I-"):
        return label[2:]
    return label

def apply_ner_mask(text: str, ner_pipe: Any, mode: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Processes a single string: finds sentences, runs NER, tracks global offsets, and masks the text.

    Args:
        text (str): The original text to be de-identified.
        ner_pipe (Any): The Hugging Face NER pipeline.
        mode (str): The masking mode ('label', 'mask', or 'x').

    Returns:
        Tuple[str, List[Dict[str, Any]]]: 
            - The full de-identified document string.
            - A list of entity dictionaries with exact global starts and ends.
    """
    if not isinstance(text, str):
        return str(text), []
        
    tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    spans = list(tokenizer.span_tokenize(text))
    
    entities_found: List[Dict[str, Any]] = []
    
    # We will build the de-identified text exactly by keeping track of masks
    # The safest way is to apply edits from the back to the front of the whole text.
    # So we collect all edits across all sentences first.
    
    edits = []

    for start_idx, end_idx in spans:
        sent = text[start_idx:end_idx]
        results = ner_pipe(sent)
        
        for ent in results:
            ent_start_global = start_idx + ent['start']
            ent_end_global = start_idx + ent['end']
            clean_ent_label = clean_label(ent['entity_group'])
            
            # Store structured entity for JSON
            entities_found.append({
                "start": ent_start_global,
                "end": ent_end_global,
                "text": ent['word'],
                "label": clean_ent_label
            })
            
            # Determine replacement string
            if mode == "label":
                rep = f"[{clean_ent_label}]"
            elif mode == "x":
                rep = "X" * (ent['end'] - ent['start'])
            else:
                rep = "[REDACTED]"
                
            edits.append((ent_start_global, ent_end_global, rep))
            
    # Apply all edits from the end of the document to the beginning to avoid shifting
    edits.sort(key=lambda x: x[0], reverse=True)
    
    deid_text = text
    for start, end, rep in edits:
        deid_text = deid_text[:start] + rep + deid_text[end:]

    return deid_text, entities_found

def run_processing_loop(df: pd.DataFrame, ner_pipe: Any, mode: str) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
    """
    Iterates through the dataframe and returns processed columns for all records.

    Args:
        df (pd.DataFrame): The input dataframe containing a 'text' column.
        ner_pipe (Any): The initialized Hugging Face NER pipeline.
        mode (str): The chosen masking mode ('label', 'mask', or 'x').

    Returns:
        Tuple[List[str], List[List[Dict[str, Any]]]]: Two lists corresponding to the de-identified texts and entity structured lists.
    """
    deid_texts: List[str] = []
    ent_logs: List[List[Dict[str, Any]]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="De-identifying"):
        clean_text, log = apply_ner_mask(row['text'], ner_pipe, mode)
        deid_texts.append(clean_text)
        ent_logs.append(log)

    return deid_texts, ent_logs

def save_output(df: pd.DataFrame, original_path: str, deid_col: List[str], ent_col: List[List[Dict[str, Any]]], mode: str) -> None:
    """
    Constructs the output path, saves the original dataframe with new columns to a CSV file,
    and also generates a structured JSON output file.

    Args:
        df (pd.DataFrame): The original input dataframe.
        original_path (str): The file path of the input CSV.
        deid_col (List[str]): List of processed text strings.
        ent_col (List[List[Dict[str, Any]]]): List of parsed entity dictionaries.
        mode (str): The masking mode used, which acts as a suffix in the output filename.
    """
    # Keep original dataframe columns, add the new ones
    output_df = df.copy()
    output_df['de_identified_text'] = deid_col
    # Serialize the list of dicts to a JSON string for the CSV column
    output_df['identified_entities'] = [json.dumps(ents, ensure_ascii=False) for ents in ent_col]
    
    base, ext = os.path.splitext(original_path)
    output_csv_path = f"{base}_deid_{mode}{ext}"
    output_df.to_csv(output_csv_path, index=False)
    print(f"\nSuccess! CSV saved to:   {output_csv_path}")
    
    # Generate the JSON list
    json_data = []
    for idx, row in output_df.iterrows():
        json_data.append({
            "id": row['id'],
            "text": row['text'],
            "deid text": deid_col[idx],
            "entities": ent_col[idx]
        })
        
    output_json_path = f"{base}_deid_{mode}.json"
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"Success! JSON saved to:  {output_json_path}")


def run(input_file: str, model_path: str = "HeMed_NER_baseline", mode: str = "label") -> None:
    """
    The main execution flow: sets up dependencies, loads model and data, processes, and saves.

    Args:
        input_file (str): Path to the input CSV file containing 'id' and 'text' columns.
        model_path (str): Path to the trained model directory.
        mode (str): Masking mode to use ('label', 'mask', or 'x').
    """
    setup_nltk()
    df = pd.read_csv(input_file)
    
    if 'id' not in df.columns:
        print("Error: Input CSV must have an 'id' column.")
        return
        
    if 'text' not in df.columns:
        print("Error: Input CSV must have a 'text' column.")
        return
    
    # 2. Load Pipeline
    ner_pipe = get_pipeline(model_path)

    # 3. Core Logic
    deid_col, ent_col = run_processing_loop(df, ner_pipe, mode)

    # 4. Save
    save_output(df, input_file, deid_col, ent_col, mode)


def main() -> None:
    """
    Entry point for CLI execution. Parses arguments and calls the run orchestrator.
    """
    parser = argparse.ArgumentParser(description="NER De-identification Tool")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--mode", choices=["label", "mask", "x"], default="label", help="How to mask identified entities")
    args = parser.parse_args()

    run(args.input, args.model, args.mode)


if __name__ == "__main__":
    main()