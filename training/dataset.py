"""Loading, splitting and tokenizing the annotated JSONL data.

Record format (one JSON object per line)::

    {"id": "...", "text": "...", "label": [{"start": 0, "end": 4, "label": "FIRST_NAME"}, ...]}

`end` is exclusive. Long documents are cut into overlapping windows
(`MAX_LENGTH` tokens, `STRIDE` overlap).
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np

from labels import LABEL_2_ID, MANDATORY_LABELS, clean_label

MAX_LENGTH = 512
STRIDE = 128
# In multi-window documents, tokens within MARGIN of an inner window edge get
# label -100 (ignored by loss and metrics) so each token is judged with context
# on both sides.
MARGIN = STRIDE // 2

SPLIT_NAMES = ("train", "validation", "test")
# Same file names Hugging Face's json loader recognizes for each split.
SPLIT_ALIASES = {
    "train": ("train",),
    "validation": ("validation", "valid", "dev", "val"),
    "test": ("test",),
}
DEFAULT_IGNORE_FILE = Path(__file__).parent / "ignorable_ids.txt"


# --- reading ----------------------------------------------------------------

def load_jsonl(path):
    """Read JSON Lines (or a plain JSON array, which Hugging Face's loader also accepts)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def dump_jsonl(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_ignorable_ids(path=DEFAULT_IGNORE_FILE):
    """Document ids excluded from training until their annotations are fixed."""
    path = Path(path)
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}


def normalize_record(record, keep_labels=MANDATORY_LABELS):
    """Clean/map label names and drop labels outside `keep_labels`."""
    record = copy.deepcopy(record)
    spans = []
    for span in record["label"]:
        span["label"] = clean_label(span["label"])
        if span["label"] in keep_labels:
            spans.append(span)
    record["label"] = spans
    return record


def load_records(path, ignore_ids=None, keep_labels=MANDATORY_LABELS):
    """Load a JSONL file, drop ignorable documents, normalize labels."""
    ignore_ids = load_ignorable_ids() if ignore_ids is None else ignore_ids
    return [normalize_record(r, keep_labels) for r in load_jsonl(path)
            if r["id"] not in ignore_ids]


# --- splitting --------------------------------------------------------------

def split_sets(dataset, seed=42):
    """Label-stratified 80/10/10 train/test/dev split (iterative stratification)."""
    from skmultilearn.model_selection import iterative_train_test_split

    # iterative_train_test_split draws from numpy's global RNG
    np.random.seed(seed)

    ids = [r["id"] for r in dataset]
    unique_tags = sorted({s["label"] for r in dataset for s in r["label"]})
    tag_index = {t: i for i, t in enumerate(unique_tags)}

    # presence/absence matrix: documents x entity types
    y = np.zeros((len(ids), len(unique_tags)))
    for i, r in enumerate(dataset):
        for s in r["label"]:
            y[i, tag_index[s["label"]]] = 1

    x = np.array(ids).reshape(-1, 1)
    x_train, y_train, x_temp, y_temp = iterative_train_test_split(x, y, test_size=0.2)
    x_test, _, x_dev, _ = iterative_train_test_split(x_temp, y_temp, test_size=0.5)

    train_ids, test_ids, dev_ids = (set(a.reshape(-1).tolist()) for a in (x_train, x_test, x_dev))
    assert not (train_ids & test_ids or train_ids & dev_ids or dev_ids & test_ids), \
        "A record ended up in more than one split"
    print(f"Train: {len(train_ids)}, Dev: {len(dev_ids)}, Test: {len(test_ids)}")

    pick = lambda keep: [r for r in dataset if r["id"] in keep]
    return pick(train_ids), pick(test_ids), pick(dev_ids)


def write_splits(records, out_dir, seed=42):
    """Split `records` and write train/validation/test.jsonl into `out_dir`."""
    train, test, dev = split_sets(records, seed=seed)
    out_dir = Path(out_dir)
    dump_jsonl(train, out_dir / "train.jsonl")
    dump_jsonl(dev, out_dir / "validation.jsonl")
    dump_jsonl(test, out_dir / "test.jsonl")
    return out_dir


def find_split_file(data_dir, name):
    """Locate e.g. validation.jsonl / dev.json / valid.jsonl in `data_dir`, or None."""
    for alias in SPLIT_ALIASES[name]:
        for ext in (".jsonl", ".json"):
            candidate = Path(data_dir) / f"{alias}{ext}"
            if candidate.exists():
                return candidate
    return None


def load_splits(data_dir, ignore_ids=None, keep_labels=MANDATORY_LABELS):
    """Return ``{"train": [...], "validation": [...], "test": [...]}`` (those present).

    `ignore_ids`: None applies `ignorable_ids.txt`; pass ``set()`` to skip filtering.
    """
    splits = {}
    for name in SPLIT_NAMES:
        path = find_split_file(data_dir, name)
        if path is not None:
            splits[name] = load_records(path, ignore_ids, keep_labels)
    return splits


# --- tokenization / label alignment ----------------------------------------

def merge_adjacent_labels(text, labels):
    """Merge same-label spans separated only by whitespace into one span."""
    if not labels:
        return []

    sorted_labels = sorted(labels, key=lambda x: x["start"])
    merged = []
    current = sorted_labels[0]
    for nxt in sorted_labels[1:]:
        gap = text[current["end"]:nxt["start"]]
        if current["label"] == nxt["label"] and gap.strip() == "":
            current["end"] = nxt["end"]
            current["span"] = text[current["start"]:current["end"]]
        else:
            merged.append(current)
            current = nxt
    merged.append(current)
    return merged


def tokenize_and_align_labels_sliding_window(examples, tokenizer):
    """Batched `datasets.map` function: tokenize into overlapping windows and
    align BIO label ids to subword tokens.

    Requires a *fast* tokenizer (offset mappings). Special tokens and window
    margins get -100.
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        is_split_into_words=False,
        padding=False,  # padding is done by the data collator
    )

    sample_map = tokenized["overflow_to_sample_mapping"]
    encodings = tokenized.encodings
    window_counts = Counter(sample_map)
    window_seen = Counter()
    all_labels = []

    for window_idx in range(len(tokenized["input_ids"])):
        doc_idx = sample_map[window_idx]
        is_first = window_seen[doc_idx] == 0
        is_last = window_seen[doc_idx] == window_counts[doc_idx] - 1
        window_seen[doc_idx] += 1

        merged = merge_adjacent_labels(
            examples["text"][doc_idx], copy.deepcopy(examples["label"][doc_idx])
        )
        encoding = encodings[window_idx]
        label_ids = []

        for i, (start_char, end_char) in enumerate(encoding.offsets):
            if encoding.special_tokens_mask[i]:
                label_ids.append(-100)
                continue
            if not is_first and i < MARGIN:
                label_ids.append(-100)
                continue
            if not is_last and i >= MAX_LENGTH - MARGIN:
                label_ids.append(-100)
                continue

            token_label = "O"
            for span in merged:
                if start_char <= span["start"] < end_char:
                    token_label = f"B-{span['label']}"
                    break
                if span["start"] < start_char < span["end"]:
                    token_label = f"I-{span['label']}"
                    break
            label_ids.append(LABEL_2_ID.get(token_label, 0))  # unknown -> "O"

        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    # helper columns, not model inputs
    tokenized.pop("offset_mapping")
    tokenized.pop("overflow_to_sample_mapping")
    return tokenized


def prepare_dataset(records, tokenizer):
    """List of records -> tokenized `datasets.Dataset` ready for the Trainer."""
    from datasets import Dataset

    ds = Dataset.from_list(records)
    return ds.map(
        lambda x: tokenize_and_align_labels_sliding_window(x, tokenizer),
        batched=True,
        remove_columns=["id", "text", "label"],
    )
