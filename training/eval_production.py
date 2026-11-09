"""Pipeline evaluation of a trained model on the FAKE pipeline test set (inference/dummy_data), three variants:
  strict    rules + document consistency (F8)   = what the colleagues will run
  lenient   rules, no F8 (--no-consistency)
  norules   model only
Writes only scores (fake text, no patient data) to training/runs/<name>/ ; the bulky outputs are deleted afterwards.

Usage:  python -X utf8 training/eval_production.py <model_folder> [--name production-eval-20261109]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DUMMY = ROOT / "inference" / "dummy_data"
VARIANTS = {"strict": [], "lenient": ["--no-consistency"], "norules": ["--no-rules"]}


def run(cmd, log):
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        return subprocess.run([str(c) for c in cmd], cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--name", default="production-eval-20261109")
    a = ap.parse_args()
    model = Path(a.model)
    if not (model / "config.json").is_file():
        sys.exit(f"not a model folder (no config.json): {model}")
    out = ROOT / "training" / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    tiers = sorted(p.stem for p in DUMMY.glob("*.csv"))
    for variant, flags in VARIANTS.items():
        res = out / f"outputs_{variant}"
        shutil.rmtree(res, ignore_errors=True)
        log = out / f"{variant}.log"
        print(f"[{variant}] de-identifying {len(tiers)} fake files ...", flush=True)
        for t in tiers:
            cmd = [sys.executable, "-X", "utf8", ROOT / "inference" / "de_identify.py", "--input", DUMMY / f"{t}.csv",
                   "--output-dir", res, "--model", model, "--output-format", "csv", *flags]
            if run(cmd, log) != 0:
                sys.exit(f"de_identify failed on {t} ({variant}); see {log}")
        cmd = [sys.executable, "-X", "utf8", ROOT / "inference" / "score_dummy.py", "--results-dir", res,
               "--json-out", out / f"{variant}.json"]
        if run(cmd, out / f"{variant}_scores.txt") != 0:
            sys.exit(f"scoring failed ({variant}); see {out / (variant + '_scores.txt')}")
        shutil.rmtree(res, ignore_errors=True)
    (out / "model.txt").write_text(f"model folder: {model.name}\n", encoding="utf-8")
    print(f"done. Scores in {out}. Now run training\push_results.bat")


if __name__ == "__main__":
    main()
