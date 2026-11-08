"""Per-label metrics, nesting of Trainer results, seed aggregation and the synthetic evaluation sets (all offline)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "inference"))

import aggregate_runs as agg
import dataset as ds
import make_synth_eval as synth
from labels import LABEL_2_ID, MANDATORY_LABELS
from metrics import compute_metrics, nest_metrics

TRAINING = Path(__file__).parent.parent


def logits_for(tag_rows):
    """Fake logits whose argmax is the given tag ids (rows of equal length)."""
    ids = np.array(tag_rows)
    out = np.zeros(ids.shape + (len(LABEL_2_ID),))
    for idx in np.ndindex(ids.shape):
        out[idx + (ids[idx],)] = 1.0
    return out


def test_per_label_recall_precision_and_undefined_labels():
    t = LABEL_2_ID
    gold = [[t["B-DATE"], t["O"], t["B-FIRST_NAME"], t["O"], t["B-ID"]]]
    pred = [[t["B-DATE"], t["O"], t["O"], t["O"], t["O"]]]            # name and ID missed
    m = compute_metrics((logits_for(pred), np.array(gold)))
    assert m["label_DATE_recall"] == 1.0 and m["label_DATE_precision"] == 1.0
    assert m["label_FIRST_NAME_recall"] == 0.0 and m["label_FIRST_NAME_support"] == 1
    assert m["label_ID_recall"] == 0.0
    assert "label_ORG_recall" not in m                                  # never in gold or predictions
    assert m["recall"] == pytest.approx(1 / 3)


def test_spurious_prediction_of_absent_label_keeps_precision_but_no_recall():
    t = LABEL_2_ID
    gold = [[t["O"], t["O"]]]
    pred = [[t["B-ORG"], t["O"]]]
    m = compute_metrics((logits_for(pred), np.array(gold)))
    assert m["label_ORG_support"] == 0 and m["label_ORG_precision"] == 0.0
    assert "label_ORG_recall" not in m and "label_ORG_f1" not in m


def test_nest_metrics_roundtrip_with_underscore_labels():
    flat = {"test_precision": 0.9, "test_recall": 0.8, "test_f1": 0.85, "test_f2": 0.82, "test_loss": 0.01,
            "test_runtime": 2.0, "test_label_FIRST_NAME_recall": 0.7, "test_label_FIRST_NAME_support": 10,
            "test_label_FIRST_NAME_precision": 0.9, "test_label_FIRST_NAME_f1": 0.78, "other_f1": 1.0}
    n = nest_metrics(flat, "test")
    assert n["overall"] == {"precision": 0.9, "recall": 0.8, "f1": 0.85, "f2": 0.82}
    assert n["per_label"]["FIRST_NAME"]["support"] == 10 and n["per_label"]["FIRST_NAME"]["recall"] == 0.7
    assert n["other"]["loss"] == 0.01 and "other_f1" not in json.dumps(n)


# ------------------------------------------------------------------------------------------------ aggregation
def fake_record(seed, f1, recall):
    pl = {"DATE": {"precision": 0.9, "recall": recall, "f1": f1, "support": 100},
          "ID": {"precision": 0.5, "recall": None, "f1": None, "support": 0}}
    overall = {"precision": 0.9, "recall": recall, "f1": f1, "f2": recall}
    return {"schema_version": 2, "status": "completed", "seed": seed, "duration_seconds": 600,
            "code": {"git_commit": "abcdef1234", "training_code_uncommitted_changes": False},
            "hyperparameters": {"learning_rate": 5e-5, "num_train_epochs": 6},
            "early_stopping_patience": 2, "base_model": {"name_or_path": "m"}, "data": {},
            "results": {"best_epoch": 3.0, "epochs_completed": 5.0, "stopped_early": True,
                        "per_epoch_validation": [{"epoch": 3.0, "step": 30, "overall": overall, "per_label": pl}],
                        "test": {"overall": overall, "per_label": pl, "other": {}},
                        "synthetic": {"rare_hard": {"overall": overall, "per_label": pl, "other": {}}}}}


def test_summary_mean_std_and_markdown(tmp_path):
    for seed, f1 in ((42, 0.90), (43, 0.92), (44, 0.94)):
        d = tmp_path / f"seed_{seed}"
        d.mkdir()
        (d / agg.RECORD_NAME).write_text(json.dumps(fake_record(seed, f1, f1)), encoding="utf-8")
    summ = agg.summarize_experiment(tmp_path)
    st = summ["overall"]["test"]["f1"]["stats"]
    assert st["mean"] == pytest.approx(0.92) and st["std"] == pytest.approx(0.02) and st["n"] == 3
    assert summ["seeds"] == [42, 43, 44]
    assert summ["per_label"]["test"]["ID"]["support"] == 0
    assert summ["per_label"]["test"]["ID"]["recall"]["stats"] is None      # undefined, not 0
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "0.920 ± 0.020" in md and "synthetic: rare_hard" in md and "support 0" in md
    assert (tmp_path / "summary.json").exists()


def test_single_run_has_no_std(tmp_path):
    (tmp_path / agg.RECORD_NAME).write_text(json.dumps(fake_record(42, 0.9, 0.9)), encoding="utf-8")
    summ = agg.summarize_experiment(tmp_path)
    assert summ["overall"]["test"]["f1"]["stats"]["std"] is None
    assert "(n=1)" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_incomplete_runs_are_skipped(tmp_path):
    bad = fake_record(42, 0.9, 0.9)
    bad["status"] = "failed"
    d = tmp_path / "seed_42"
    d.mkdir()
    (d / agg.RECORD_NAME).write_text(json.dumps(bad), encoding="utf-8")
    assert agg.summarize_experiment(tmp_path) is None


def test_old_schema_baseline_record_can_be_summarised():
    p = TRAINING / "runs" / "model_non_anon_20261108-1648"
    if not (p / agg.RECORD_NAME).exists():
        pytest.skip("baseline record not present")
    summ = agg.aggregate(agg.load_experiment(p))
    assert summ["overall"]["test"]["f1"]["stats"]["mean"] == pytest.approx(0.9265770, abs=1e-6)
    assert summ["overall"]["validation (best epoch)"]["f1"]["stats"]["mean"] == pytest.approx(0.9401544, abs=1e-6)


# ------------------------------------------------------------------------------------------------ synthetic sets
def test_committed_synthetic_sets_match_the_generator():
    for name, (_fn, seed) in synth.SETS.items():
        data, _ = synth.render(name, seed)
        assert (TRAINING / "synth_eval" / f"{name}.jsonl").read_bytes() == data, f"{name}: run make_synth_eval.py"


def test_synthetic_sets_are_valid_and_cover_the_rare_labels():
    sets = ds.load_extra_eval_sets(TRAINING / "synth_eval")
    assert set(sets) == {"rare_hard", "long"}
    for name, records in sets.items():
        ids = [r["id"] for r in records]
        assert len(ids) == len(set(ids))
        for r in records:
            assert set(r) == {"id", "text", "label"}                      # nothing that breaks tokenisation
            for s in r["label"]:
                assert set(s) == {"start", "end", "label"} and s["label"] in MANDATORY_LABELS
                assert 0 <= s["start"] < s["end"] <= len(r["text"])
    counts = {lab: sum(s["label"] == lab for r in sets["rare_hard"] for s in r["label"]) for lab in MANDATORY_LABELS}
    for lab in ("ID", "INNER_ID", "AGE_ABOVE_89", "ORG", "MED_ORG", "CONTACT", "LOC"):
        assert counts[lab] >= 40, (lab, counts)
    assert min(len(r["text"].split()) for r in sets["long"]) >= 450       # several 512-token windows
    assert any(len(r["text"].split()) >= 2500 for r in sets["long"])


def test_synthetic_sets_are_disjoint_from_the_pipeline_fake_set():
    inf = TRAINING.parent / "inference" / "dummy_data"
    pipeline_texts = set()
    for p in inf.glob("*.truth.jsonl"):
        pipeline_texts |= {json.loads(l)["text"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
    for records in ds.load_extra_eval_sets(TRAINING / "synth_eval").values():
        assert not pipeline_texts & {r["text"] for r in records}
