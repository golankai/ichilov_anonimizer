"""Writes ``training_run.json`` next to every trained model.

The file documents everything needed to understand (and repeat) a run: the exact command,
code version, base model, data (file hashes, sizes, per-label counts), labels, windowing,
all hyperparameters, hardware/software, and the results (per-epoch metrics, best epoch,
test metrics). It is written when training *starts* (status ``running``) and rewritten
at the end (``completed`` or ``failed``), so a crashed run still documents itself.

It contains counts and metadata only - never document text.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import dataset as ds
from labels import ALL_LABELS, LABEL_MAPPING, MANDATORY_LABELS

RECORD_NAME = "training_run.json"
SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent.parent


def _json_default(o):
    try:  # numpy scalars
        return o.item()
    except Exception:
        return str(o)


def _git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def code_version():
    status = _git("status", "--porcelain", "--", "training")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # True if training/ has uncommitted changes, i.e. the commit alone does not reproduce the code
        "training_code_uncommitted_changes": bool(status) if status is not None else None,
    }


def environment_info():
    from importlib.metadata import PackageNotFoundError, version

    pkgs = {}
    for name in ("torch", "transformers", "datasets", "accelerate", "seqeval", "scikit-learn", "tokenizers"):
        try:
            pkgs[name] = version(name)
        except PackageNotFoundError:
            pkgs[name] = None
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "machine": os.environ.get("COMPUTERNAME") or platform.node(), "packages": pkgs}
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = {"name": p.name, "memory_gb": round(p.total_memory / 1024**3, 1),
                           "torch_cuda": torch.version.cuda, "count": torch.cuda.device_count()}
    except Exception:
        pass
    return info


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def describe_records(records):
    """Counts only: documents, entities, entities per label (after label cleaning/filtering)."""
    per_label = Counter(s["label"] for r in records for s in r["label"])
    return {"n_documents": len(records), "n_entities": sum(per_label.values()),
            "entities_per_label": dict(sorted(per_label.items()))}


def describe_split(path, records):
    """`describe_records` plus the source file's path, size and SHA-256."""
    path = Path(path)
    return {"file": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": _sha256(path), **describe_records(records)}


def base_model_info(model_name):
    info = {"name_or_path": str(model_name)}
    cfg = Path(str(model_name)) / "config.json"
    if cfg.exists():
        try:
            c = json.loads(cfg.read_text(encoding="utf-8"))
            info.update({k: c.get(k) for k in ("model_type", "architectures", "vocab_size", "hidden_size",
                                                "num_hidden_layers", "max_position_embeddings")})
            info["pretrained_head_labels"] = len(c.get("id2label", {}))
        except Exception:
            pass
    return info


class RunRecord:
    def __init__(self, output_dir, *, kind, base_model, data=None, notes=None):
        self.path = Path(output_dir) / RECORD_NAME
        self.t0 = time.time()
        self.doc = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "run_kind": kind,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "working_directory": os.getcwd(),
            "code": code_version(),
            "base_model": base_model_info(base_model),
            "data": data or {},
            "labels": {"scheme": "BIO", "entity_types": MANDATORY_LABELS, "num_labels": len(ALL_LABELS),
                       "label_mapping_applied": LABEL_MAPPING},
            "windowing": {"max_length_tokens": ds.MAX_LENGTH, "stride_overlap_tokens": ds.STRIDE,
                          "edge_margin_tokens": ds.MARGIN,
                          "edge_margin_note": "tokens within this distance of an inner window edge get label -100 "
                                              "(ignored by loss and metrics) in multi-window documents"},
            "notes": notes,
        }

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.doc, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        tmp.replace(self.path)

    def start(self, training_args):
        self.doc["hyperparameters"] = training_args.to_dict()
        self.doc["environment"] = environment_info()
        self._write()

    def finish(self, trainer, test_metrics=None):
        state = trainer.state
        history = state.log_history
        self.doc["results"] = {
            "epochs_completed": state.epoch,
            "global_step": state.global_step,
            "best_metric_f1": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
            "per_epoch_validation": [h for h in history if "eval_f1" in h],
            "train_loss_log": [h for h in history if "loss" in h],
            "train_summary": next((h for h in reversed(history) if "train_runtime" in h), None),
            "test": test_metrics,
        }
        self._close("completed")

    def fail(self, exc):
        self.doc["error"] = f"{type(exc).__name__}: {exc}"
        self._close("failed")

    def _close(self, status):
        self.doc["status"] = status
        self.doc["finished_at"] = datetime.now().isoformat(timespec="seconds")
        self.doc["duration_seconds"] = round(time.time() - self.t0, 1)
        self._write()
