import os
import argparse
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from utils.nlp_utils import get_pipeline, setup_nltk
from utils.pipeline_utils import (
    DeidDoc,
    make_stage_observer,
    make_replacement_observer,
    make_pipeline_start_stage,
)
from utils.pipeline_stages import ReplacementObserver
from utils.doc_pipeline import build_doc_pipeline


logger = logging.getLogger(__name__)


def run_processing_loop(
    df: pd.DataFrame,
    ner_pipe: Any,
    mode: str,
    observer: Optional[Callable[[str, DeidDoc], None]] = None,
    replacement_observer: Optional[ReplacementObserver] = None,
    header_stage: Optional[Callable[[DeidDoc], DeidDoc]] = None,
) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
    """
    Iterates through the dataframe and returns processed columns for all records,
    using the composed document pipeline.
    """
    doc_pipeline = build_doc_pipeline(
        ner_pipe,
        mode,
        observer=observer,
        replacement_observer=replacement_observer,
        normalize_dates=True,
        normalize_ids=True,
        header_stage=header_stage,
    )

    deid_texts: List[str] = []
    ent_logs: List[List[Dict[str, Any]]] = []

    # for _, row in tqdm(df.iterrows(), total=len(df), desc="De-identifying (pipeline)"):
    for _, row in df.iterrows():
        doc = DeidDoc(text=row["text"])
        doc = doc_pipeline(doc)
        deid_texts.append(doc.text)
        ent_logs.append(doc.entities)

    return deid_texts, ent_logs


def save_output(
    df: pd.DataFrame,
    original_path: str,
    deid_col: List[str],
    ent_col: List[List[Dict[str, Any]]],
    mode: str,
) -> None:
    """
    Constructs the output path, saves the original dataframe with new columns to a CSV file,
    and also generates a structured JSON output file.
    """
    output_df = df.copy()
    output_df["de_identified_text"] = deid_col
    output_df["identified_entities"] = [
        " | ".join([f"{ent['text']}({ent['label']})" for ent in ents]) for ents in ent_col
    ]

    base, ext = os.path.splitext(original_path)
    output_csv_path = f"{base}_deid_{mode}_pipeline{ext}"
    output_df.to_csv(output_csv_path, index=False)
    print(f"\nSuccess! CSV saved to:   {output_csv_path}")

    json_data = []
    for idx, row in output_df.iterrows():
        json_data.append(
            {
                "id": row["id"],
                "text": row["text"],
                "deid text": deid_col[idx],
                "entities": ent_col[idx],
            }
        )

    output_json_path = f"{base}_deid_{mode}_pipeline.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"Success! JSON saved to:  {output_json_path}")


def run(
    input_file: str,
    model_path: str = "HeMed_NER_baseline",
    mode: str = "label",
    observe_stages: bool = False,
    log_replacements: bool = False,
) -> None:
    """
    The main execution flow for the pipeline-based variant.
    """
    setup_nltk()
    df = pd.read_csv(input_file)

    if "id" not in df.columns:
        print("Error: Input CSV must have an 'id' column.")
        return

    if "text" not in df.columns:
        print("Error: Input CSV must have a 'text' column.")
        return

    ner_pipe = get_pipeline(model_path)

    # Install observers based on CLI flags.
    stage_observer = make_stage_observer(logger, observe_stages)
    replacement_observer = make_replacement_observer(logger, log_replacements)
    # Always log a per-document banner, independent of --observe-stages.
    header = make_pipeline_start_stage(logger)

    deid_col, ent_col = run_processing_loop(
        df,
        ner_pipe,
        mode,
        observer=stage_observer,
        replacement_observer=replacement_observer,
        header_stage=header,
    )
    save_output(df, input_file, deid_col, ent_col, mode)


def main() -> None:
    """
    Entry point for CLI execution of the pipeline-based tool.
    """
    parser = argparse.ArgumentParser(description="NER De-identification Tool (pipeline version)")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument(
        "--mode",
        choices=["label", "mask", "x"],
        default="label",
        help="How to mask identified entities",
    )
    parser.add_argument(
        "--observe-stages",
        action="store_true",
        help=(
            "If set, logs information about the output of each "
            "pipeline stage (more detail with --verbose)."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging for this tool.",
    )
    parser.add_argument(
        "--log-replacements",
        action="store_true",
        help=(
            "If set, logs each string replacement applied by any "
            "pipeline stage that performs text replacements, including "
            "original text, location, label (if available), and "
            "replacement token."
        ),
    )
    args = parser.parse_args()

    # Global default: only WARNING+ from all libraries
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    # Our logger: promote to INFO/DEBUG based on --verbose
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    run(
        args.input,
        args.model,
        args.mode,
        observe_stages=args.observe_stages,
        log_replacements=args.log_replacements,
    )


if __name__ == "__main__":
    main()

