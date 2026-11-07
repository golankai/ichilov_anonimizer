"""Post-install sanity check for deployers: imports, NLTK data, model folder. Prints plain-language results."""
import sys
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"
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
found = sorted(p.name for p in MODELS.glob("*") if (p / "config.json").is_file()) if MODELS.is_dir() else []
if found:
    print(f"[ok]      models in {MODELS}: {', '.join(found)}")
else:
    print(f"[todo]    no model yet. Copy a model folder (config.json, model.safetensors, ...) into: {MODELS}")
sys.exit(0 if ok else 1)
