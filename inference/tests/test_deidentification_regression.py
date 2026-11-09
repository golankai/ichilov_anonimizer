"""Regression tests for the de-identification pipeline (no model required).

The real Hugging Face NER model is replaced by small fakes, so these tests check
everything *around* the model: normalization, sentence offsets, masking modes,
placeholder handling and output files.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# `inference/` holds de_identify.py and utils/, so it must be importable.
INFERENCE_ROOT = Path(__file__).parent.parent
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from de_identify import run_processing_loop, save_output
from utils.doc_pipeline import build_doc_pipeline
from utils.nlp_utils import setup_nltk
from utils.pipeline_utils import DeidDoc

DATA_DIR = INFERENCE_ROOT / "dummy_data"
GOLD_JSON_PATH = Path(__file__).parent / "gold" / "test_deid_label_pipeline.json"


@pytest.fixture(scope="module", autouse=True)
def _nltk():
    setup_nltk()


def run_pipeline(text, ner, mode="label", normalize=True):
    pipe = build_doc_pipeline(ner, mode, normalize_dates=normalize, normalize_ids=normalize)
    return pipe(DeidDoc(text=text))


class NoNer:
    def __call__(self, sentence):
        return []


class SubstringNer:
    """Tags every occurrence of the given {substring: label} in a sentence."""

    def __init__(self, targets):
        self.targets = targets

    def __call__(self, sentence):
        out = []
        for sub, label in self.targets.items():
            i = sentence.find(sub)
            while i != -1:
                out.append({"start": i, "end": i + len(sub), "entity_group": label})
                i = sentence.find(sub, i + len(sub))
        return sorted(out, key=lambda e: e["start"])


class ReplayNer:
    """Replays gold entities (global offsets into `full_text`) for each sentence."""

    def __init__(self, full_text, entities):
        self.full_text, self.entities = full_text, entities

    def __call__(self, sentence):
        start = self.full_text.find(sentence)
        assert start != -1, "sentence not found in full text"
        end = start + len(sentence)
        return [
            {"start": e["start"] - start, "end": e["end"] - start, "entity_group": e["label"]}
            for e in self.entities
            if e["start"] >= start and e["end"] <= end
        ]


def load_gold_examples():
    with GOLD_JSON_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("example", load_gold_examples(), ids=lambda e: str(e["id"]))
def test_pipeline_matches_gold(example):
    """Full pipeline (with ID/date normalization) reproduces the curated output.

    NOTE (known quirk): entity offsets are not in one coordinate system. Entities
    from the model point into the *normalized* text, while entities recorded by
    the regex ID stage keep offsets into the text as it was when that stage ran.
    So the fake model replays only the entities that are consistent with the
    normalized text; the rest are produced by the regex stages themselves.
    """
    normalized = run_pipeline(example["text"], NoNer()).text
    model_entities = [
        e for e in example["entities"] if normalized[e["start"]:e["end"]] == e["text"]
    ]
    out = run_pipeline(example["text"], ReplayNer(normalized, model_entities))

    assert out.text == example["deid text"]
    assert out.entities == example["entities"]


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("label", "[FIRST_NAME] [LAST_NAME] was admitted."),
        ("mask", "[REDACTED] [REDACTED] was admitted."),
        ("x", "XXXX XXXXX was admitted."),
    ],
)
def test_masking_modes(mode, expected):
    ner = SubstringNer({"John": "FIRST_NAME", "Smith": "LAST_NAME"})
    assert run_pipeline("John Smith was admitted.", ner, mode, normalize=False).text == expected


def test_entities_have_global_offsets_across_sentences():
    text = "First sentence here. Then John arrived."
    out = run_pipeline(text, SubstringNer({"John": "FIRST_NAME"}), normalize=False)
    (ent,) = out.entities
    assert text[ent["start"]:ent["end"]] == "John" == ent["text"]
    assert out.text == "First sentence here. Then [FIRST_NAME] arrived."


def test_bio_prefixes_are_stripped_from_labels():
    out = run_pipeline("John", SubstringNer({"John": "B-FIRST_NAME"}), normalize=False)
    assert out.entities[0]["label"] == "FIRST_NAME"


def test_numeric_date_is_normalized_before_ner():
    seen = []

    class Spy:
        def __call__(self, sentence):
            seen.append(sentence)
            return []

    run_pipeline("Admitted on 5/3/2024.", Spy())
    assert "05.03.2024" in seen[0]


def test_ner_hits_inside_normalization_placeholders_are_ignored():
    # The ID stage replaces the number with "[ID]" and records the ID entity itself.
    # A model "finding" something inside that placeholder must not add a second one.
    out = run_pipeline("ת.ז. 123456789 John", SubstringNer({"ID": "ID", "John": "FIRST_NAME"}))
    assert [(e["label"], e["text"]) for e in out.entities].count(("ID", "ID")) == 0
    assert sorted(e["label"] for e in out.entities) == ["FIRST_NAME", "ID"]
    assert out.text == "ת.ז. [ID] [FIRST_NAME]"


def test_non_string_text_is_coerced():
    out = run_pipeline(12345, NoNer(), normalize=False)
    assert out.text == "12345" and out.entities == []


def test_run_processing_loop_returns_one_result_per_row():
    df = pd.DataFrame({"id": [1, 2], "text": ["John", "nothing here"]})
    texts, ents, failed = run_processing_loop(df, SubstringNer({"John": "FIRST_NAME"}), "label")
    assert texts == ["[FIRST_NAME]", "nothing here"]
    assert [len(e) for e in ents] == [1, 0]
    assert failed == []


class CountingNer(SubstringNer):
    """Counts calls and can crash on a chosen text, to test resume."""

    def __init__(self, mapping, crash_on=None):
        super().__init__(mapping)
        self.calls = 0
        self.crash_on = crash_on

    def __call__(self, sentence):
        self.calls += 1
        if self.crash_on and self.crash_on in sentence:
            raise RuntimeError("boom")
        return super().__call__(sentence)


def _loop(df, ner, ckpt, meta=None):
    return run_processing_loop(df, ner, "label", checkpoint_path=ckpt, checkpoint_meta=meta or {"v": 1})


def test_failed_document_is_masked_not_leaked_and_retried_on_rerun(tmp_path):
    df = pd.DataFrame({"id": [1, 2, 3], "text": ["John a", "BAD John", "John c"]})
    ckpt = tmp_path / "x.progress.jsonl"
    texts, _, failed = _loop(df, CountingNer({"John": "FIRST_NAME"}, crash_on="BAD"), ckpt)
    assert failed == [2] and texts[1] == "[PROCESSING_ERROR]" and "John" not in " ".join(texts)
    ner = CountingNer({"John": "FIRST_NAME"})
    texts, _, failed = _loop(df, ner, ckpt)  # only the failed row is processed again
    assert failed == [] and ner.calls >= 1 and texts[1] == "BAD [FIRST_NAME]"
    assert texts[0] == "[FIRST_NAME] a"


def test_resume_skips_finished_documents(tmp_path):
    df = pd.DataFrame({"id": [1, 2, 3], "text": ["John a", "John b", "John c"]})
    ckpt = tmp_path / "x.progress.jsonl"
    first = CountingNer({"John": "FIRST_NAME"})
    full, ents_full, _ = _loop(df, first, ckpt)
    # simulate a crash after two documents: keep the header and two records, plus a half-written line
    lines = ckpt.read_text(encoding="utf-8").splitlines()
    ckpt.write_text("\n".join(lines[:3]) + '\n{"i": 2, "id": "3", "t": "[FIRS', encoding="utf-8")
    second = CountingNer({"John": "FIRST_NAME"})
    texts, ents, failed = _loop(df, second, ckpt)
    assert texts == full and ents == ents_full and failed == []
    assert 0 < second.calls < first.calls  # only the third document was processed


def test_changed_input_or_options_ignores_old_progress(tmp_path):
    ckpt = tmp_path / "x.progress.jsonl"
    _loop(pd.DataFrame({"id": [1], "text": ["John"]}), CountingNer({"John": "FIRST_NAME"}), ckpt, {"v": 1, "input": "a"})
    ner = CountingNer({"John": "FIRST_NAME"})
    texts, _, _ = _loop(pd.DataFrame({"id": [1], "text": ["John"]}), ner, ckpt, {"v": 1, "input": "b"})
    assert ner.calls >= 1 and texts == ["[FIRST_NAME]"]
    assert ckpt.with_suffix(".stale").exists()


def test_save_output_writes_consistent_csv_and_json(tmp_path):
    examples = load_gold_examples()
    df = pd.DataFrame({"id": [e["id"] for e in examples], "text": [e["text"] for e in examples]})
    input_csv = tmp_path / "input.csv"
    df.to_csv(input_csv, index=False)

    deid_col = [e["deid text"] for e in examples]
    ent_col = [e["entities"] for e in examples]
    save_output(df, str(input_csv), deid_col, ent_col, mode="label")

    out_csv = tmp_path / "input_deid_label_pipeline.csv"
    out_json = tmp_path / "input_deid_label_pipeline.json"
    assert out_csv.exists() and out_json.exists()
    assert list(pd.read_csv(out_csv)["de_identified_text"]) == deid_col

    rows = json.loads(out_json.read_text(encoding="utf-8"))
    assert [r["deid text"] for r in rows] == deid_col
