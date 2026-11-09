import os
import argparse
import hashlib
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


def output_base(original_path: str, output_dir: Optional[str]) -> str:
    base, _ = os.path.splitext(original_path)
    if output_dir:  # default: next to the input file
        base = os.path.join(output_dir, os.path.basename(base))
    return base


def input_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha1()
    for i, t in zip(df["id"].astype(str), df["text"]):
        h.update(i.encode("utf-8"))
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


def load_checkpoint(path: Path, meta: Optional[Dict[str, Any]], df: pd.DataFrame) -> Dict[int, Tuple[str, list]]:
    """Rows finished by an earlier, interrupted run: {row index: (de-identified text, entities)}.
    Only used if input, model and options are exactly the same; otherwise the old file is set aside."""
    if not path.is_file():
        return {}
    ids = df["id"].astype(str).tolist()
    done: Dict[int, Tuple[str, list]] = {}
    try:
        with open(path, encoding="utf-8") as f:
            first = json.loads(f.readline())
            if first.get("meta") != meta:
                raise ValueError("input, model or options changed")
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # half-written line of the crashed run
                if 0 <= r["i"] < len(ids) and ids[r["i"]] == r["id"]:
                    done[r["i"]] = (r["t"], r["e"])
    except (ValueError, KeyError, OSError) as exc:
        stale = path.with_suffix(".stale")
        print(f"Ignoring the earlier progress file ({exc}); moved to {stale.name}. Starting from the beginning.")
        os.replace(path, stale)
        return {}
    return done


def open_checkpoint(path: Path, meta: Optional[Dict[str, Any]], append: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a" if append else "w", encoding="utf-8")
    if append:
        f.write("\n")  # a half-written last line must not swallow the next record
    else:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        f.flush()
    return f


def run_processing_loop(
    df: pd.DataFrame,
    ner_pipe: Any,
    mode: str,
    observer: Optional[Callable[[str, DeidDoc], None]] = None,
    replacement_observer: Optional[ReplacementObserver] = None,
    header_stage: Optional[Callable[[DeidDoc], DeidDoc]] = None,
    max_tokens: Optional[int] = None,
    overlap_tokens: int = DEFAULT_OVERLAP,
    use_rules: bool = True,
    consistency: bool = True,
    checkpoint_path: Optional[Path] = None,
    checkpoint_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[List[Dict[str, Any]]], List[Any]]:
    """
    Iterates through the dataframe and returns processed columns for all records,
    using the composed document pipeline.
    """
    doc_pipeline = build_doc_pipeline(
        ner_pipe,
        mode,
        observer=observer,
        replacement_observer=replacement_observer,
        # With the rules the text is never rewritten before the model (offsets stay in the original text).
        normalize_dates=not use_rules,
        normalize_ids=not use_rules,
        header_stage=header_stage,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        use_rules=use_rules,
        consistency=consistency,
    )

    total = len(df)
    done = load_checkpoint(checkpoint_path, checkpoint_meta, df) if checkpoint_path else {}
    if done:
        print(f"Resuming: {len(done)} of {total} documents were already processed in the earlier run ({checkpoint_path.name}).")
    ckpt = open_checkpoint(checkpoint_path, checkpoint_meta, append=bool(done)) if checkpoint_path else None

    deid_texts: List[str] = []
    ent_logs: List[List[Dict[str, Any]]] = []
    failed: List[Any] = []
    bar = tqdm(total=total, initial=len(done), desc="De-identifying", unit="doc", smoothing=0.05, mininterval=2, dynamic_ncols=True)
    try:
        for n, (_, row) in enumerate(df.iterrows()):
            if n in done:
                deid_texts.append(done[n][0])
                ent_logs.append(done[n][1])
                continue
            try:
                doc = doc_pipeline(DeidDoc(text=row["text"]))
                deid_texts.append(doc.text)
                ent_logs.append(doc.entities)
                if ckpt:
                    ckpt.write(json.dumps({"i": n, "id": str(row["id"]), "t": doc.text, "e": doc.entities}, ensure_ascii=False) + "\n")
                    ckpt.flush()
            except Exception as exc:  # one bad document must not stop the run; it is fully masked, never left in clear text
                failed.append(row["id"])
                tqdm.write(f"WARNING: document id={row['id']} (row {n + 1}) failed: {type(exc).__name__}: {str(exc)[:200]}")
                deid_texts.append("[PROCESSING_ERROR]")
                ent_logs.append([])
            bar.update(1)
    finally:
        bar.close()
        if ckpt:
            ckpt.close()
    if failed:
        print(f"\n{len(failed)} document(s) failed and were replaced by [PROCESSING_ERROR] (ids: {failed[:20]}{' ...' if len(failed) > 20 else ''}). "
              f"Run the same command again to retry only these.")
    return deid_texts, ent_logs, failed


def save_output(
    df: pd.DataFrame,
    original_path: str,
    deid_col: List[str],
    ent_col: List[List[Dict[str, Any]]],
    mode: str,
    output_format: str = "both",
    output_tag: str = "",
    output_dir: Optional[str] = None,
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

    _, ext = os.path.splitext(original_path)
    base = output_base(original_path, output_dir)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
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
    output_dir: Optional[str] = None,
    use_rules: bool = True,
    consistency: bool = True,
    restart: bool = False,
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

    # Progress file: every finished document is appended, so a crash / Ctrl+C resumes instead of restarting.
    checkpoint_path = Path(f"{output_base(input_file, output_dir)}_deid_{mode}_pipeline{output_tag}.progress.jsonl")
    if restart and checkpoint_path.exists():
        checkpoint_path.unlink()
    checkpoint_meta = {
        "v": 1, "input": input_fingerprint(df), "n": len(df), "model": str(model_path), "mode": mode,
        "rules": use_rules, "consistency": consistency, "max_tokens": max_tokens, "overlap": overlap_tokens,
    }

    deid_col, ent_col, failed = run_processing_loop(
        df,
        ner_pipe,
        mode,
        observer=stage_observer,
        replacement_observer=replacement_observer,
        header_stage=header,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        use_rules=use_rules,
        consistency=consistency,
        checkpoint_path=checkpoint_path,
        checkpoint_meta=checkpoint_meta,
    )
    save_output(df, input_file, deid_col, ent_col, mode, output_format, output_tag, output_dir)
    if not failed:  # keep the progress file if some documents failed: the next run retries only those
        checkpoint_path.unlink(missing_ok=True)
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
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder for the result files (created if missing). Default: next to the input file.",
    )
    parser.add_argument(
        "--no-rules",
        action="store_true",
        help="Switch off the deterministic rules (utils/rules.py) and use the old date/ID rewriting stages.",
    )
    parser.add_argument(
        "--no-consistency",
        action="store_true",
        help="Lenient mode: do not mask the repeated occurrences of a name / city / organisation the model masked "
        "(rule F8). Default (strict): they are masked, except words in lexicons/ambiguous_words.tsv.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore the progress file of an interrupted run and start from the beginning. "
        "Default: an interrupted run resumes where it stopped (same input, model and options).",
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
    # One INFO banner per document would bury the progress bar on large files: only with the debugging flags.
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO if (args.observe_stages or args.log_replacements) else logging.WARNING)

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
        output_dir=args.output_dir,
        use_rules=not args.no_rules,
        consistency=not args.no_consistency,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()

