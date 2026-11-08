"""The fake test set and its scorer: ground truth is consistent, reproducible, and the scorer behaves on known inputs."""
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

INFERENCE_ROOT = Path(__file__).parent.parent
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

import make_dummy_data
import score_dummy

DATA = INFERENCE_ROOT / "dummy_data"
LABELS = {"FIRST_NAME", "LAST_NAME", "DATE", "AGE_ABOVE_89", "LOC", "ORG", "MED_ORG", "CONTACT", "ID", "INNER_ID"}
TIERS = make_dummy_data.TIERS


def truth(tier):
    return score_dummy.load_jsonl(DATA / f"{tier}.truth.jsonl")


def id_checksum_ok(s):
    s = s.zfill(9)
    total = 0
    for i, ch in enumerate(s):
        v = int(ch) * (1 if i % 2 == 0 else 2)
        total += v if v < 10 else v - 9
    return total % 10 == 0


@pytest.mark.parametrize("tier", TIERS)
def test_truth_is_consistent(tier):
    docs = truth(tier)
    assert docs
    for d in docs:
        t, spans = d["text"], sorted(d["label"] + d["keep"], key=lambda x: x["start"])
        for a, b in zip(spans, spans[1:]):
            assert a["end"] <= b["start"], (tier, d["id"], "overlap")
        for s in d["label"]:
            assert s["label"] in LABELS
            assert 0 <= s["start"] < s["end"] <= len(t)
            assert t[s["start"]:s["end"]].strip() == t[s["start"]:s["end"]], "entity with edge whitespace"
            if s["label"] == "ID" and s["tag"] in ("id-with-context", "id-no-context", "id-eight-digits"):
                assert id_checksum_ok(t[s["start"]:s["end"]]), "generated ID must have a valid check digit"


def test_generation_is_reproducible(tmp_path):
    make_dummy_data.main(tmp_path)
    for tier in TIERS:
        for ext in (".csv", ".truth.jsonl"):
            assert (tmp_path / f"{tier}{ext}").read_bytes() == (DATA / f"{tier}{ext}").read_bytes(), (tier, ext)


def fake_outputs(tmp_path, predictor):
    for tier in TIERS:
        outs = []
        for d in truth(tier):
            ents = predictor(d)
            text = d["text"]
            deid = text
            for e in sorted(ents, key=lambda e: e["start"], reverse=True):
                deid = deid[:e["start"]] + f"[{e['label']}]" + deid[e["end"]:]
            outs.append({"id": d["id"], "text": text, "deid text": deid, "entities": ents})
        (tmp_path / f"{tier}_deid_label_pipeline.json").write_text(json.dumps(outs, ensure_ascii=False), encoding="utf-8")


def run_scorer(tmp_path):
    buf = io.StringIO()
    with redirect_stdout(buf):
        score_dummy.main(["--results-dir", str(tmp_path)])
    return buf.getvalue()


def test_perfect_predictor_scores_clean(tmp_path):
    fake_outputs(tmp_path, lambda d: [{"start": s["start"], "end": s["end"], "label": s["label"],
                                       "text": d["text"][s["start"]:s["end"]]} for s in d["label"]])
    out = run_scorer(tmp_path)
    assert "LEAKS: 0 " in out
    assert "declared traps hit: 0 " in out and "extra predictions on non-PHI text: 0" in out
    assert "OFFSET DRIFT" not in out


def test_silent_predictor_leaks_everything(tmp_path):
    fake_outputs(tmp_path, lambda d: [])
    total = sum(len(d["label"]) for t in TIERS for d in truth(t))
    assert f"LEAKS: {total} " in run_scorer(tmp_path)


def test_overmasking_is_reported(tmp_path):
    def everything(d):  # masks every declared trap as well
        return [{"start": s["start"], "end": s["end"], "label": "LOC", "text": d["text"][s["start"]:s["end"]]}
                for s in d["label"] + d["keep"]]
    fake_outputs(tmp_path, everything)
    traps = sum(len(d["keep"]) for t in TIERS for d in truth(t))
    assert f"declared traps hit: {traps} of {traps}" in run_scorer(tmp_path)
