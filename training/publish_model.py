r"""Copy a trained model to the shared models folder, with only the files inference needs.

Default: the production model (real data + half the synthetic set) -> D:\De ID All\NER_anon_models\real_plus_half_synth_all_data
Usage:   python -X utf8 training\publish_model.py
         python -X utf8 training\publish_model.py --src training\models\<run> --dest "D:\De ID All\NER_anon_models\<name>"

Copies the top-level files of the model folder (config, weights, tokenizer) and skips training leftovers
(checkpoint-* folders, optimizer/scheduler state, training_args.bin, trainer_state.json). Writes MODEL_INFO.txt
(training recipe, no data). Refuses to overwrite an existing destination unless --force. Does not touch git.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "training" / "models" / "model_non_anon_20261109-0144"
DEFAULT_DEST = Path(r"D:\De ID All\NER_anon_models\real_plus_half_synth_all_data")
SKIP = {"training_args.bin", "trainer_state.json", "optimizer.pt", "scheduler.pt", "scaler.pt"}
INFO = """real_plus_half_synth_all_data

Hebrew medical de-identification (NER) model, 10 entity types (FIRST_NAME LAST_NAME DATE AGE_ABOVE_89 LOC ORG MED_ORG
CONTACT ID INNER_ID). Fine-tuned from the HeMed base model.

Training: ALL real annotated documents (train + dev + test) mixed with half (1,500 of 3,000) of the synthetic documents,
6 epochs, seed 42, 2026-11-09. There is no held-out set for this model.
Compared with real_only_all_data (the model first sent for testing): same real data, plus the synthetic documents
(ambiguous words, organisation names, etc.).

Use it with the current code (git pull): deidentify.bat file.csv label real_plus_half_synth_all_data
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--force", action="store_true", help="overwrite files in an existing destination")
    a = ap.parse_args()
    src, dest = Path(a.src), Path(a.dest)
    for need in ("config.json", "model.safetensors"):
        if not (src / need).is_file():
            sys.exit(f"missing {need} in {src}")
    if dest.exists() and any(dest.iterdir()) and not a.force:
        sys.exit(f"{dest} already exists and is not empty (use --force to overwrite)")
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for f in sorted(src.iterdir()):
        if f.is_file() and f.name not in SKIP and not f.name.startswith("checkpoint"):
            shutil.copy2(f, dest / f.name)
            total += f.stat().st_size
            print(f"copied {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    (dest / "MODEL_INFO.txt").write_text(INFO, encoding="utf-8")
    for name in ("config.json", "model.safetensors"):
        assert (dest / name).stat().st_size == (src / name).stat().st_size, f"size mismatch: {name}"
    print(f"\nDone: {total / 1e6:.0f} MB copied to {dest}")


if __name__ == "__main__":
    main()
