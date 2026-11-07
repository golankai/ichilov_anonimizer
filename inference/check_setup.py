"""Post-install sanity check for deployers: imports, NLTK data, model folder. Prints plain-language results."""
import sys
from pathlib import Path

MODEL = Path(__file__).resolve().parent.parent / "HeMed_NER_baseline"
ok = True
for mod in ("torch", "transformers", "pandas", "nltk"):
    try:
        __import__(mod)
        print(f"[ok]      {mod}")
    except Exception as e:
        ok = False
        print(f"[MISSING] {mod}: {e}")
try:
    import torch
    print(f"[info]    GPU available: {torch.cuda.is_available()} (CPU also works, just slower)")
except Exception:
    pass
if (MODEL / "config.json").is_file():
    print(f"[ok]      model folder: {MODEL}")
else:
    print(f"[todo]    model not found yet. Copy the model files into: {MODEL}")
sys.exit(0 if ok else 1)
