"""Tests for the synthetic TRAINING set (training/make_synth_train.py). No model needed."""
import json
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(TRAINING.parent / "inference"))

import make_synth_train as mst  # noqa: E402
from labels import MANDATORY_LABELS  # noqa: E402

OUT = TRAINING / "synth_train"


def _records(name="synth_train.jsonl"):
    return [json.loads(l) for l in (OUT / name).read_text(encoding="utf-8").splitlines()]


def test_committed_files_match_the_generator():
    for all_words, stem in ((False, "synth_train"), (True, "synth_train_all")):
        data, meta = mst.serialize(mst.render(all_words=all_words))
        assert (OUT / f"{stem}.jsonl").read_bytes() == data, "run training/make_synth_train.py [--all-words]"
        assert (OUT / f"{stem}.meta.json").read_bytes() == meta


def test_spans_are_valid_and_labels_known():
    for r in _records():
        spans = sorted((s["start"], s["end"]) for s in r["label"])
        assert all(0 <= a < b <= len(r["text"]) for a, b in spans)
        assert all(b <= c for (_, b), (c, _) in zip(spans, spans[1:]))
        assert all(s["label"] in MANDATORY_LABELS for s in r["label"])


def test_heldout_words_never_appear_in_the_experiment_set_but_do_in_the_production_set():
    split = mst.word_split()
    meta = json.loads((OUT / "synth_train.meta.json").read_text(encoding="utf-8"))
    used = {m["word"] for m in meta.values() if m["word"]}
    assert used and not used & set(split["heldout"]) and used <= set(split["seen"])
    meta_all = json.loads((OUT / "synth_train_all.meta.json").read_text(encoding="utf-8"))
    assert {m["word"] for m in meta_all.values() if m["word"]} & set(split["heldout"])


def test_names_are_disjoint_from_the_evaluation_vocabulary():
    first, last = mst.name_pools()
    import make_dummy_data as md
    assert not set(first) & (set(md.FIRST) | set(md.LAST)) and not set(last) & (set(md.FIRST) | set(md.LAST))
    assert len(first) > 60 and len(last) > 60


def test_no_training_text_equals_an_evaluation_document():
    ev = set()
    for p in list((TRAINING / "synth_eval").glob("*.jsonl")) + list((TRAINING / "synth_eval_ambiguous").glob("*.jsonl")):
        ev |= {json.loads(l)["text"] for l in p.read_text(encoding="utf-8").splitlines()}
    assert not {r["text"] for r in _records() if r["label"]} & ev   # filler-only documents may coincide


def test_org_spans_are_the_bare_name():
    for r in _records():
        for s in r["label"]:
            if s["label"] in ("MED_ORG", "ORG"):
                span = r["text"][s["start"]:s["end"]]
                assert not span.startswith(("בית החולים", "ביה", "קופ", "חברת", "בנק", "משרד")), span


def test_dates_are_pieces():
    for r in _records():
        for s in r["label"]:
            if s["label"] == "DATE":
                assert not any(c in r["text"][s["start"]:s["end"]] for c in "./ -")
