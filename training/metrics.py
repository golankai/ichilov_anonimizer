"""Trainer metrics: entity-level seqeval P/R/F1 (overall and per label) plus a token-level F2.

Per-label numbers are returned as flat keys ``label_<LABEL>_{precision,recall,f1,support}`` because the Trainer only
handles flat dicts of numbers. ``run_record.py`` folds them back into a nested ``per_label`` block.
A label that never occurs in the gold data gets no recall/F1 (undefined, not 0) but keeps ``support`` 0 and its
precision, so spurious predictions of a label that should not occur still show up.
"""

from __future__ import annotations

import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import fbeta_score

from labels import ALL_LABELS

LABEL_PREFIX = "label_"
PER_LABEL_FIELDS = ("precision", "recall", "f1", "support")


def per_label_metrics(gold_tags, pred_tags):
    """``{label: {precision, recall, f1, support}}`` (entity level) for every label present in gold or predictions."""
    report = classification_report(gold_tags, pred_tags, output_dict=True, zero_division=0)
    out = {}
    for label, r in report.items():
        if label in ("micro avg", "macro avg", "weighted avg"):
            continue
        support = int(r["support"])
        out[label] = {
            "precision": float(r["precision"]),
            "recall": float(r["recall"]) if support else None,
            "f1": float(r["f1-score"]) if support else None,
            "support": support,
        }
    return out


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Drop -100 positions (special tokens and window margins).
    pred_tags = [[ALL_LABELS[p] for p, l in zip(pr, lb) if l != -100]
                 for pr, lb in zip(predictions, labels)]
    gold_tags = [[ALL_LABELS[l] for p, l in zip(pr, lb) if l != -100]
                 for pr, lb in zip(predictions, labels)]

    # F2 (recall weighted 2x) on "is this token any entity", the metric that
    # matters most for de-identification: a missed entity is a leak.
    flat_pred = [int(t != "O") for seq in pred_tags for t in seq]
    flat_gold = [int(t != "O") for seq in gold_tags for t in seq]

    metrics = {
        "precision": precision_score(gold_tags, pred_tags),
        "recall": recall_score(gold_tags, pred_tags),
        "f1": f1_score(gold_tags, pred_tags),
        "f2": fbeta_score(flat_gold, flat_pred, beta=2),
    }
    for label, r in per_label_metrics(gold_tags, pred_tags).items():
        for field in PER_LABEL_FIELDS:
            if r[field] is not None:
                metrics[f"{LABEL_PREFIX}{label}_{field}"] = r[field]
    return metrics


def nest_metrics(flat, prefix):
    """Split a Trainer result dict (keys like ``test_f1``, ``test_label_DATE_recall``) into
    ``{"overall": {...}, "per_label": {LABEL: {...}}, "other": {...}}`` for one prefix (``eval``, ``test``, ...).

    ``other`` holds loss, runtime and the like. Labels with undefined recall/F1 get ``None``.
    """
    overall, per_label, other = {}, {}, {}
    p = prefix + "_"
    for key, value in flat.items():
        if not key.startswith(p):
            continue
        k = key[len(p):]
        if k.startswith(LABEL_PREFIX):
            label, _, field = k[len(LABEL_PREFIX):].rpartition("_")
            per_label.setdefault(label, {})[field] = value
        elif k in ("precision", "recall", "f1", "f2"):
            overall[k] = value
        else:
            other[k] = value
    for fields in per_label.values():
        for field in PER_LABEL_FIELDS:
            fields.setdefault(field, None)
    return {"overall": overall, "per_label": dict(sorted(per_label.items())), "other": other}
