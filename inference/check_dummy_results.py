"""Guard for push_dummy_results.bat: exit 0 only if inference/dummy_results/ holds nothing but outputs of the fake
samples in inference/dummy_data/ (same ids and texts), and nothing outside that folder is staged. Prints no text."""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES, DATA = ROOT / "inference" / "dummy_results", ROOT / "inference" / "dummy_data"
bad = []

for f in sorted(RES.glob("*")):
    if f.name.startswith("."):
        continue
    stem = f.name.split("_deid_")[0]
    src = DATA / f"{stem}.csv"
    if "_deid_" not in f.name or not src.is_file():
        bad.append(f"{f.name}: not an output of a dummy_data file")
        continue
    if f.suffix == ".csv":
        a, b = pd.read_csv(src), pd.read_csv(f)
        if list(a["id"]) != list(b["id"]) or list(a["text"]) != list(b["text"]):
            bad.append(f"{f.name}: ids/texts differ from dummy_data/{stem}.csv (real data?)")

staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, capture_output=True, text=True).stdout.split()
bad += [f"staged outside dummy_results: {p}" for p in staged if not p.startswith("inference/dummy_results/")]

if bad:
    print("REFUSING TO PUSH:\n  " + "\n  ".join(bad))
    sys.exit(1)
print("dummy_results check passed")
