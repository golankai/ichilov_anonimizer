"""Label alignment tests using a tiny offline WordPiece tokenizer (no downloads)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

import dataset as ds
from data_utils import evaluate_char_spans, normalize_and_merge_spans, record_to_iob
from labels import ALL_LABELS, ID_2_LABEL, LABEL_2_ID, clean_label


@pytest.fixture(scope="module")
def tok():
    words = ["john", "smith", "was", "admitted", "on", "2023", "-", "01", "15", "to", "ward", "a"]
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + words)}
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    from tokenizers.processors import TemplateProcessing
    t.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    return PreTrainedTokenizerFast(tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]")


def decode(label_ids):
    return [ID_2_LABEL[i] if i != -100 else None for i in label_ids]


def test_clean_label_strips_backslashes_and_maps_dates():
    assert clean_label("AGE\\_ABOVE\\_89") == "AGE_ABOVE_89"
    assert clean_label("MONTH") == "DATE"


def test_label_maps_are_bio():
    assert ALL_LABELS[0] == "O" and LABEL_2_ID["B-DATE"] + 1 == LABEL_2_ID["I-DATE"]


def test_single_window_alignment(tok):
    text = "john smith was admitted on 2023-01-15"
    labels = [{"start": 0, "end": 4, "label": "FIRST_NAME"},
              {"start": 5, "end": 10, "label": "LAST_NAME"},
              {"start": 27, "end": 37, "label": "DATE"}]
    out = ds.tokenize_and_align_labels_sliding_window({"text": [text], "label": [labels]}, tok)
    assert len(out["input_ids"]) == 1
    tags = decode(out["labels"][0])
    assert tags[0] is None and tags[-1] is None            # [CLS]/[SEP]
    assert tags[1:6] == ["B-FIRST_NAME", "B-LAST_NAME", "O", "O", "O"]
    assert tags[6] == "B-DATE" and set(tags[7:-1]) == {"I-DATE"}
    assert "offset_mapping" not in out and "overflow_to_sample_mapping" not in out


def test_adjacent_same_label_spans_merge():
    text = "john smith"
    merged = ds.merge_adjacent_labels(text, [{"start": 0, "end": 4, "label": "X"},
                                             {"start": 5, "end": 10, "label": "X"}])
    assert [(m["start"], m["end"]) for m in merged] == [(0, 10)]


def test_long_document_windows_mask_inner_margins(tok):
    text = " ".join(["john"] * 1200)
    out = ds.tokenize_and_align_labels_sliding_window({"text": [text], "label": [[]]}, tok)
    n = len(out["input_ids"])
    assert n > 1
    first, last = out["labels"][0], out["labels"][-1]
    assert first[1] != -100                                  # first window keeps its head
    assert first[ds.MAX_LENGTH - ds.MARGIN] == -100          # ...but masks its tail
    assert out["labels"][1][ds.MARGIN - 1] == -100           # later windows mask their head
    assert out["labels"][1][ds.MARGIN] != -100
    assert last[-2] != -100                                  # last window keeps its tail


def test_normalize_record_drops_unknown_labels():
    r = {"id": "1", "text": "x", "label": [{"start": 0, "end": 1, "label": "YEAR"},
                                           {"start": 0, "end": 1, "label": "FAMILY_STATUS"}]}
    assert [s["label"] for s in ds.normalize_record(r)["label"]] == ["DATE"]


def test_ignorable_ids_loaded():
    assert "11822165.txt" in ds.load_ignorable_ids()


def test_missing_ignore_file_means_no_filtering(tmp_path):
    assert ds.load_ignorable_ids(tmp_path / "nope.txt") == set()


def test_data_utils_accept_dict_and_triple_spans():
    text = "john smith"
    assert normalize_and_merge_spans([{"start": 0, "end": 4, "label": "N"}, [5, 10, "N"]], text)[0]["end"] == 10
    assert record_to_iob({"text": text, "label": [[0, 10, "N"]]}) == [("john", "B-N"), ("smith", "I-N")]


def test_evaluate_char_spans_with_dict_gold():
    rec = {"text": "john smith", "label": [{"start": 0, "end": 4, "label": "FIRST_NAME"}]}
    perfect = lambda t: [{"start": 0, "end": 4, "entity_group": "FIRST_NAME"}]
    assert evaluate_char_spans([rec], perfect)["f1"] == 1.0
    assert evaluate_char_spans([rec], lambda t: [])["recall"] == 0.0


def test_load_splits_accepts_hf_style_names_and_skips_ignore_list(tmp_path):
    rec = lambda i: {"id": i, "text": "john", "label": [{"start": 0, "end": 4, "label": "FIRST_NAME"}]}
    ds.dump_jsonl([rec("11822165.txt")], tmp_path / "train.jsonl")      # an id from ignorable_ids.txt
    (tmp_path / "dev.json").write_text(__import__("json").dumps([rec("v")]), encoding="utf-8")  # JSON array
    splits = ds.load_splits(tmp_path, ignore_ids=set())
    assert [r["id"] for r in splits["train"]] == ["11822165.txt"]
    assert [r["id"] for r in splits["validation"]] == ["v"] and "test" not in splits
