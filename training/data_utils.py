"""Span utilities and character-level evaluation.

Ported from MedNER (`data_utils.py`). Spans may be given either as
``{"start": s, "end": e, "label": t}`` dicts (the dataset format) or as
``[start, end, tag]`` triples (what `pipe_output_to_spans` produces); `end` is
always exclusive.
"""

from __future__ import annotations

import re

# Splits text into word tokens and individual punctuation/symbols.
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_with_offsets(text):
    """Return a list of ``(token, start_char, end_char)`` for `text`."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


def _as_span(span):
    """Accept a dict or a ``(start, end, tag)`` sequence; return a fresh dict."""
    if isinstance(span, dict):
        return {"start": int(span["start"]), "end": int(span["end"]), "tag": span["label"]}
    start, end, tag = span
    return {"start": int(start), "end": int(end), "tag": tag}


def normalize_and_merge_spans(label_list, text):
    """Sort spans and merge same-tag neighbours separated only by whitespace."""
    spans = sorted((_as_span(s) for s in label_list), key=lambda s: s["start"])

    merged = []
    for span in spans:
        if merged and merged[-1]["tag"] == span["tag"]:
            if text[merged[-1]["end"]:span["start"]].strip() == "":
                merged[-1]["end"] = span["end"]
                continue
        merged.append(span)
    return merged


# Kept as an alias: the evaluation code historically had its own copy.
normalize_spans_char = normalize_and_merge_spans


def find_covering_span(spans, token_start, token_end):
    """Return the span overlapping ``[token_start, token_end)``, else None.

    Assumes `spans` is sorted and non-overlapping.
    """
    for span in spans:
        if span["start"] >= token_end:
            break
        if token_start < span["end"] and token_end > span["start"]:
            return span
    return None


def record_to_iob(record):
    """Word-level IOB tagging of ``{"text", "label"}`` using regex tokens."""
    text = record["text"]
    spans = normalize_and_merge_spans(record["label"], text)

    iob_tokens = []
    for tok, start, end in tokenize_with_offsets(text):
        span = find_covering_span(spans, start, end)
        if span is None:
            iob_tokens.append((tok, "O"))
        else:
            prefix = "B-" if start == span["start"] else "I-"
            iob_tokens.append((tok, prefix + span["tag"]))
    return iob_tokens


def preprocess_single_record(record):
    tokens, labels = tuple(zip(*record_to_iob(record)))
    return {"tokens": list(tokens), "labels": list(labels)}


# --- evaluation -------------------------------------------------------------

def pipe_output_to_spans(pipe_output):
    """HF pipeline output -> ``[[start, end, tag], ...]``."""
    return [[int(e["start"]), int(e["end"]), e["entity_group"]] for e in pipe_output]


def span_set(spans):
    return {(s["start"], s["end"], s["tag"]) for s in spans}


def evaluate_char_spans(test_set, pipe):
    """Exact-match, character-span precision/recall/F1.

    test_set: records with ``"text"`` and ``"label"`` (gold spans).
    pipe: HF token-classification pipeline (``aggregation_strategy`` set) or any
          callable ``text -> list of entity dicts``.
    """
    tp = fp = fn = 0
    for ex in test_set:
        text = ex["text"]
        gold = span_set(normalize_and_merge_spans(ex["label"], text))
        pred = span_set(normalize_and_merge_spans(pipe_output_to_spans(pipe(text)), text))
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
