import os
import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
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
from utils.chunking import DEFAULT_OVERLAP
from utils.doc_pipeline import build_doc_pipeline


logger = logging.getLogger(__name__)

# Models live side by side in <repo>/models/<name>/ (git-ignored); `--model <name>` picks one.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_NAME = "HeMed_Anonimizer_Model"


def available_models() -> List[str]:
    if not MODELS_DIR.is_dir():
        return []
    return sorted(p.name for p in MODELS_DIR.iterdir() if (p / "config.json").is_file())


def resolve_model(arg: Optional[str]) -> Optional[str]:
    """None = default model; a name = models/<name>; an existing folder or a Hugging Face id is used as is."""
    if arg is None:
        names = available_models()
        if DEFAULT_MODEL_NAME in names:
            return str(MODELS_DIR / DEFAULT_MODEL_NAME)
        return str(MODELS_DIR / names[0]) if len(names) == 1 else None
    if Path(arg).is_dir():
        return arg
    if (MODELS_DIR / arg).is_dir():
        return str(MODELS_DIR / arg)
    if re.fullmatch(r"[\w.-]+/[\w.-]+", arg):
        return arg  # Hugging Face model id
    return None


def run_processing_loop(
    df: pd.DataFrame,
    ner_pipe: Any,
    mode: str,
    observer: Optional[Callable[[str, DeidDoc], None]] = None,
    replacement_observer: Optional[ReplacementObserver] = None,
    header_stage: Optional[Callable[[DeidDoc], DeidDoc]] = None,
    max_tokens: Optional[int] = None,
    overlap_tokens: int = DEFAULT_OVERLAP,
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
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    deid_texts: List[str] = []
    ent_logs: List[List[Dict[str, Any]]] = []

    # for _, row in tqdm(df.iterrows(), total=len(df), desc="De-identifying (pipeline)"):
    total = len(df)
    for n, (_, row) in enumerate(df.iterrows(), 1):
        print(f"Row {n}/{total}", flush=True)
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
    output_format: str = "both",
    output_tag: str = "",
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
    if output_format in ("csv", "both"):
        output_csv_path = f"{base}_deid_{mode}_pipeline{output_tag}{ext}"
        output_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print(f"\nSuccess! CSV saved to:   {output_csv_path}")
    if output_format in ("txt", "both"):
        # Tab-separated; TABs inside values were already replaced by spaces on load.
        output_txt_path = f"{base}_deid_{mode}_pipeline{output_tag}.txt"
        output_df.to_csv(output_txt_path, index=False, sep="\t", encoding="utf-8-sig")
        print(f"Success! TXT saved to:   {output_txt_path}")

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

    output_json_path = f"{base}_deid_{mode}_pipeline{output_tag}.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"Success! JSON saved to:  {output_json_path}")


def run(
    input_file: str,
    model_path: str = str(MODELS_DIR / DEFAULT_MODEL_NAME),
    mode: str = "label",
    observe_stages: bool = False,
    log_replacements: bool = False,
    max_tokens: Optional[int] = None,
    overlap_tokens: int = DEFAULT_OVERLAP,
    output_format: str = "both",
    output_tag: str = "",
) -> None:
    """
    The main execution flow for the pipeline-based variant.
    """
    started = time.time()
    print(f"Start time: {datetime.now():%Y-%m-%d %H:%M:%S}")
    setup_nltk()
    # utf-8-sig reads files with or without a BOM (e.g. saved from Excel).
    df = pd.read_csv(input_file, encoding="utf-8-sig")

    if "id" not in df.columns:
        print("Error: Input CSV must have an 'id' column.")
        return

    if "text" not in df.columns:
        print("Error: Input CSV must have a 'text' column.")
        return

    # TABs would break tab-separated output and add nothing to the NER: use a space.
    df["text"] = df["text"].astype(str).str.replace("\t", " ", regex=False)

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
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    save_output(df, input_file, deid_col, ent_col, mode, output_format, output_tag)
    print(f"End time:   {datetime.now():%Y-%m-%d %H:%M:%S} (took {time.time() - started:.0f}s)")


def main() -> None:
    """
    Entry point for CLI execution of the pipeline-based tool.
    """
    parser = argparse.ArgumentParser(description="NER De-identification Tool (pipeline version)")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name (a folder in models/), a path to a model folder, or a Hugging Face model id. "
            f"Default: models/{DEFAULT_MODEL_NAME}, or the only model in models/."
        ),
    )
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
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=None,
        help=(
            "Max text tokens per NER window. Default: the model limit minus "
            "special tokens and a small margin (e.g. 502 for a 512 model). "
            "Lower it if the model was trained on short inputs."
        ),
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="Tokens of shared context between neighbouring windows (default %(default)s).",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "txt", "both"],
        default="both",
        help="Main output file(s): csv, tab-separated txt, or both (default). A JSON file is always written too.",
    )
    args = parser.parse_args()

    model = resolve_model(args.model)
    if model is None:
        found = ", ".join(available_models()) or "none yet"
        parser.error(
            f"model not found: {args.model or '(default)'}\n"
            f"Put each model in its own folder under '{MODELS_DIR}' (it contains config.json and "
            f"model.safetensors) and pick one with --model <folder name>. Models found there: {found}."
        )
    # Output files of non-default models carry the model name, so runs with several models do not overwrite each other.
    name = re.sub(r"[^\w.-]", "_", Path(model).name)
    output_tag = "" if name == DEFAULT_MODEL_NAME else f"_{name}"
    if not Path(args.input).is_file():
        parser.error(f"input file not found: {args.input}")

    # Global default: only WARNING+ from all libraries
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    # Our logger: promote to INFO/DEBUG based on --verbose
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    run(
        args.input,
        model,
        args.mode,
        observe_stages=args.observe_stages,
        log_replacements=args.log_replacements,
        max_tokens=args.max_chunk_tokens,
        overlap_tokens=args.chunk_overlap,
        output_format=args.output_format,
        output_tag=output_tag,
    )


if __name__ == "__main__":
    main()

