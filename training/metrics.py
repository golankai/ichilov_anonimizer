"""Trainer metrics: entity-level seqeval P/R/F1 plus a token-level F2."""

from __future__ import annotations

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import fbeta_score

from labels import ALL_LABELS


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

    return {
        "precision": precision_score(gold_tags, pred_tags),
        "recall": recall_score(gold_tags, pred_tags),
        "f1": f1_score(gold_tags, pred_tags),
        "f2": fbeta_score(flat_gold, flat_pred, beta=2),
    }
