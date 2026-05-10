import json
import sys
from pathlib import Path

import pandas as pd
import pytest


# Ensure the project root (where `de_identify.py` lives) is importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from de_identify import apply_ner_mask, save_output
from utils.pipeline_utils import DeidDoc
from utils.doc_pipeline import build_doc_pipeline


DATA_DIR = Path(__file__).parent.parent / "dummy_data"
GOLD_JSON_PATH = DATA_DIR / "test_deid_label_2_deid_label.json"


class FakeNerPipeline:
    """
    Minimal stand-in for the Hugging Face NER pipeline.

    It uses the gold entities (with global start/end offsets) from the
    reference JSON and, for each sentence slice that `apply_ner_mask`
    passes in, returns entities shifted to sentence-relative offsets,
    in the same shape as the real pipeline.
    """

    def __init__(self, full_text, entities):
        self.full_text = full_text
        self.entities = entities

    def __call__(self, sentence):
        # Find where this sentence lives inside the full text
        full = self.full_text
        start_idx = full.find(sentence)
        if start_idx == -1:
            raise AssertionError("Sentence not found in full text while stubbing NER pipeline")

        end_idx = start_idx + len(sentence)

        results = []
        for ent in self.entities:
            ent_start = ent["start"]
            ent_end = ent["end"]

            # Keep only entities fully contained in this sentence slice
            if ent_start >= start_idx and ent_end <= end_idx:
                results.append(
                    {
                        "start": ent_start - start_idx,
                        "end": ent_end - start_idx,
                        "entity_group": ent["label"],
                    }
                )

        results.sort(key=lambda e: e["start"])
        return results


def load_gold_examples():
    with GOLD_JSON_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("example", load_gold_examples())
def test_apply_ner_mask_matches_gold_text_and_entities(example):
    """
    Regression test for the masking + offset logic.

    For each gold example:
    - We build a fake NER pipeline that replays the entities from the
      curated JSON (`test_deid_label_2_deid_label.json`).
    - We run `apply_ner_mask` and expect:
        * The de-identified text to match the gold `deid text`.
        * The structured entities returned by the function to match
          the gold `entities` list exactly (including global offsets).
    """
    text = example["text"]
    expected_deid = example["deid text"]
    expected_entities = example["entities"]

    ner = FakeNerPipeline(text, expected_entities)
    deid_text, entities = apply_ner_mask(text, ner, mode="label")

    assert deid_text == expected_deid
    assert entities == expected_entities


def test_save_output_produces_consistent_csv_and_json(tmp_path):
    """
    Regression test for the I/O layer around `save_output`.

    We construct a tiny in-memory dataset from the same gold examples,
    call `save_output`, and then verify:
    - The CSV contains the expected `de_identified_text` values.
    - The JSON's entity offsets point to the correct substrings.
    """
    examples = load_gold_examples()

    df = pd.DataFrame(
        {
            "id": [ex["id"] for ex in examples],
            "text": [ex["text"] for ex in examples],
        }
    )

    input_csv = tmp_path / "input.csv"
    df.to_csv(input_csv, index=False)

    deid_col = [ex["deid text"] for ex in examples]
    ent_col = [ex["entities"] for ex in examples]

    save_output(df, str(input_csv), deid_col, ent_col, mode="label")

    # CSV regression: de-identified text column should match exactly.
    out_csv = tmp_path / "input_deid_label.csv"
    assert out_csv.exists()
    out_df = pd.read_csv(out_csv)
    assert list(out_df["de_identified_text"]) == deid_col

    # JSON regression: each entity's text must match the slice at [start:end].
    out_json = tmp_path / "input_deid_label.json"
    assert out_json.exists()
    with out_json.open(encoding="utf-8") as f:
        data = json.load(f)

    for row in data:
        text = row["text"]
        for ent in row["entities"]:
            start = ent["start"]
            end = ent["end"]
            expected = ent["text"]
            assert text[start:end] == expected


@pytest.mark.parametrize("example", load_gold_examples())
def test_pipeline_matches_baseline_for_each_example(example):
    """
    Compare the original implementation (`de_identify.py`) with the
    pipeline-based implementation (`de_identify_pipeline.py`).

    For each gold example we:
    - Run `apply_ner_mask` (baseline).
    - Run the document pipeline built by `build_doc_pipeline` on the
      same text and fake NER pipeline.
    - Assert that both implementations produce identical de-identified
      text and structured entities, and that they match the gold data.
    """
    text = example["text"]
    expected_deid = example["deid text"]
    expected_entities = example["entities"]

    # Baseline path
    baseline_ner = FakeNerPipeline(text, expected_entities)
    baseline_deid, baseline_entities = apply_ner_mask(
        text, baseline_ner, mode="label"
    )

    # Pipeline path
    pipeline_ner = FakeNerPipeline(text, expected_entities)
    doc_pipeline = build_doc_pipeline(ner_pipe=pipeline_ner, mode="label")
    doc = DeidDoc(text=text)
    out_doc = doc_pipeline(doc)

    # Compare to baseline and to the curated gold
    assert out_doc.text == baseline_deid == expected_deid
    assert out_doc.entities == baseline_entities == expected_entities

