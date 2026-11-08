"""Tests for the ambiguous-words evaluation set (training/make_ambiguous_eval.py). No model needed."""
import json
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(TRAINING.parent / "inference"))

import dataset as ds  # noqa: E402
import make_ambiguous_eval as amb  # noqa: E402
from labels import MANDATORY_LABELS  # noqa: E402

OUT = TRAINING / "synth_eval_ambiguous"


def test_committed_files_match_the_generator():
    data, meta = amb.serialize(amb.render())
    assert (OUT / "ambiguous.jsonl").read_bytes() == data, "run training/make_ambiguous_eval.py"
    assert (OUT / "ambiguous.meta.json").read_bytes() == meta, "run training/make_ambiguous_eval.py"


def _records():
    return [json.loads(line) for line in (OUT / "ambiguous.jsonl").read_text(encoding="utf-8").splitlines()]


def test_spans_are_inside_text_and_labels_are_known():
    for r in _records():
        for s in r["label"]:
            assert 0 <= s["start"] < s["end"] <= len(r["text"])
            assert s["label"] in MANDATORY_LABELS


def test_plain_documents_have_no_entities_and_name_documents_have_some():
    meta = json.loads((OUT / "ambiguous.meta.json").read_text(encoding="utf-8"))
    kinds = {}
    for r in _records():
        m = meta[str(r["id"])]
        kinds.setdefault(m["kind"], []).append(r)
        if m["kind"] == "plain":
            assert r["label"] == []
        else:
            assert r["label"]
    assert set(kinds) == {"name", "plain", "mixed"}
    assert all(len(v) > 50 for v in kinds.values())


def test_mixed_documents_mask_the_name_but_not_the_plain_use():
    meta = json.loads((OUT / "ambiguous.meta.json").read_text(encoding="utf-8"))
    n = 0
    for r in _records():
        m = meta[str(r["id"])]
        if m["kind"] != "mixed":
            continue
        n += 1
        w, t = m["word"], r["text"]
        covered = [t[s["start"]:s["end"]] for s in r["label"]]
        assert w in covered  # the name occurrence is labelled
        # the word occurs at least once more in the text, and that occurrence is not labelled
        assert t.count(w) > covered.count(w)
    assert n > 50


def test_loader_used_by_train_py_accepts_the_folder():
    sets = ds.load_extra_eval_sets(OUT)
    assert set(sets) == {"ambiguous"}
    assert len(sets["ambiguous"]) == len(_records())


def test_every_label_the_lexicon_promises_is_covered():
    from collections import Counter
    c = Counter(s["label"] for r in _records() for s in r["label"])
    for lab in ("FIRST_NAME", "LAST_NAME", "DATE", "LOC", "MED_ORG", "ORG"):
        assert c[lab] > 0, lab


def test_rules_alone_do_not_mask_plain_use():
    """The deterministic rules (inference/utils/rules.py) must leave ordinary uses of the words readable."""
    from utils.rules import apply_rules
    meta = json.loads((OUT / "ambiguous.meta.json").read_text(encoding="utf-8"))
    bad = []
    for r in _records():
        if meta[str(r["id"])]["kind"] == "plain":
            bad += [(r["text"], x["text"]) for x in apply_rules(r["text"], [])]
    assert bad == []
