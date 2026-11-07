"""Tests for context-window chunking (utils/chunking.py) and its use in the NER stage.

No trained model is needed. A real (cached) tokenizer is used when available so
token counts are realistic; those tests are skipped if it is not on disk.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

INFERENCE_ROOT = Path(__file__).parent.parent
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from utils.chunking import (
    chunk_text,
    plan_chunks,
    resolve_token_budget,
    sentence_starts,
    token_offsets,
)
from utils.doc_pipeline import build_doc_pipeline
from utils.pipeline_utils import DeidDoc

DATA_DIR = INFERENCE_ROOT / "dummy_data"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "bert-base-multilingual-cased", local_files_only=True
        )
    except Exception:  # not cached on this machine
        pytest.skip("bert-base-multilingual-cased tokenizer not available offline")


@pytest.fixture(scope="module")
def long_text():
    """~2k+ tokens of realistic Hebrew clinical text (all long test records)."""
    df = pd.read_csv(DATA_DIR / "test_long.csv")
    return "\n".join(df["text"])


def boundary_texts(text):
    """The first few characters after every detected sentence start."""
    return [text[b : b + 12] for b in sentence_starts(text)]


# --------------------------------------------------------------------------- #
# Sentence boundaries
# --------------------------------------------------------------------------- #
def test_id_abbreviation_is_not_a_boundary():
    text = "המטופל דוד כהן ת.ז. 123456789 אושפז אתמול. הוא שוחרר."
    assert boundary_texts(text) == ["הוא שוחרר."]


def test_dates_are_not_split_but_can_end_a_sentence():
    text = "נולד ב-10.05.1980 בחיפה. אושפז ב-07.02.2024. שוחרר אתמול."
    assert boundary_texts(text) == ["אושפז ב-07.0", "שוחרר אתמול."]


def test_list_markers_and_initials_are_not_boundaries():
    assert sentence_starts("סעיף א. המטופל הגיע. בדיקה ב. נמצאה תקינה.") == [
        len("סעיף א. המטופל הגיע. ")
    ]


def test_newlines_are_boundaries():
    text = "שם: דוד כהן\nת.ז: 123\n\nאושפז בפנימית"
    assert sentence_starts(text) == [text.index("ת.ז"), text.index("אושפז")]


def test_email_and_question_marks_end_sentences():
    text = "כתובת a.b@clalit.org.il. מה קרה? כלום."
    assert boundary_texts(text) == ["מה קרה? כלום", "כלום."]


# --------------------------------------------------------------------------- #
# Window planning
# --------------------------------------------------------------------------- #
def test_short_text_is_a_single_window(tokenizer):
    text = "המטופל דוד כהן אושפז אתמול."
    (chunk,) = chunk_text(text, tokenizer, max_tokens=500)
    assert (chunk.start, chunk.end, chunk.core_start, chunk.core_end) == (0, len(text), 0, len(text))


@pytest.mark.parametrize("budget, overlap", [(64, 16), (128, 32), (200, 64), (60, 30)])
def test_windows_respect_budget_and_cover_everything(tokenizer, long_text, budget, overlap):
    # `model_limit` makes chunk_text re-tokenize each window and shrink the plan if
    # re-tokenizing a substring costs an extra token (as the real stage does).
    chunks = chunk_text(long_text, tokenizer, budget, overlap, model_limit=budget + 2)
    assert len(chunks) > 1

    # 1. every window fits the token budget (re-tokenized, as the model will see it)
    for c in chunks:
        n = len(tokenizer(long_text[c.start : c.end], add_special_tokens=False)["input_ids"])
        assert n <= budget

    # 2. cores tile the text exactly: no gaps, no overlap
    assert chunks[0].core_start == 0 and chunks[-1].core_end == len(long_text)
    for a, b in zip(chunks, chunks[1:]):
        assert a.core_end == b.core_start
        assert a.start < b.start and a.end < b.end  # forward progress
        assert b.start <= a.end  # neighbours overlap

    # 3. each core lies inside its own window
    for c in chunks:
        assert c.start <= c.core_start < c.core_end <= c.end


def test_windows_are_cut_at_sentence_boundaries_when_possible(tokenizer, long_text):
    boundaries = set(sentence_starts(long_text))
    chunks = chunk_text(long_text, tokenizer, 128, 32)
    inner = chunks[1:]
    at_boundary = sum(1 for c in inner if c.start in boundaries)
    assert at_boundary >= len(inner) * 0.8  # nearly all starts snap to a sentence start


def test_windows_never_cut_between_id_label_and_digits(tokenizer, long_text):
    # The old NLTK splitter produced "...ת.ז." | "301234567 ..." pairs.
    for c in chunk_text(long_text, tokenizer, 96, 24):
        assert not long_text[: c.end].rstrip().endswith("ת.ז.")
        assert not long_text[c.start :].startswith(("301234567", "205551234"))


def test_windows_never_cut_inside_a_word(tokenizer, long_text):
    for c in chunk_text(long_text, tokenizer, 100, 20):
        assert c.start == 0 or long_text[c.start - 1].isspace() or not long_text[c.start].isalnum()
        assert c.end == len(long_text) or long_text[c.end].isspace() or not long_text[c.end - 1].isalnum()


def test_text_without_any_punctuation_is_still_chunked(tokenizer):
    text = " ".join(["מילה"] * 1500)  # no periods, no newlines
    chunks = chunk_text(text, tokenizer, 100, 20)
    assert len(chunks) > 10
    for c in chunks:
        assert len(tokenizer(text[c.start : c.end], add_special_tokens=False)["input_ids"]) <= 100


def test_overlap_larger_than_half_the_window_is_clamped(tokenizer, long_text):
    chunks = chunk_text(long_text, tokenizer, 64, overlap_tokens=1000)
    assert chunks[-1].end == len(long_text) and len(chunks) > 1


def test_fallback_without_tokenizer_still_chunks(long_text):
    chunks = chunk_text(long_text, None, 80, 16)
    assert len(chunks) > 1 and chunks[-1].core_end == len(long_text)


def test_resolve_budget_uses_model_limit_minus_specials(tokenizer):
    class Pipe:
        pass

    pipe = Pipe()
    pipe.tokenizer = tokenizer
    _, budget = resolve_token_budget(pipe)
    assert budget == tokenizer.model_max_length - 2 - 8  # [CLS] + [SEP] + margin


def test_resolve_budget_falls_back_when_no_tokenizer():
    tok, budget = resolve_token_budget(object())
    assert tok is None and 0 < budget < 512


# --------------------------------------------------------------------------- #
# End to end through the NER stage
# --------------------------------------------------------------------------- #
class TruncatingNer:
    """Imitates the Hugging Face NER pipeline: anything past the model's
    maximum length is silently dropped. Tags the given substrings."""

    def __init__(self, tokenizer, targets, max_len=512):
        self.tokenizer, self.targets, self.max_len = tokenizer, targets, max_len
        self.calls = []

    def __call__(self, text):
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_len, return_offsets_mapping=True
        )
        visible_end = max((e for _, e in enc["offset_mapping"]), default=0)
        self.calls.append((len(text), len(enc["input_ids"])))
        out = []
        for sub, label in self.targets.items():
            i = text.find(sub)
            while i != -1:
                if i + len(sub) <= visible_end:
                    out.append({"start": i, "end": i + len(sub), "entity_group": label})
                i = text.find(sub, i + len(sub))
        return sorted(out, key=lambda e: e["start"])


def run(text, ner, **kwargs):
    pipe = build_doc_pipeline(ner, "label", **kwargs)
    return pipe(DeidDoc(text=text))


def test_long_text_without_punctuation_is_fully_covered(tokenizer):
    # One 1,500-word "sentence": the old per-sentence loop would hand all of it to
    # the model at once and lose everything after token 512.
    words = ["מילה"] * 1500
    for i in (10, 700, 1400):
        words[i] = "דוד"
    text = " ".join(words)

    ner = TruncatingNer(tokenizer, {"דוד": "FIRST_NAME"})
    out = run(text, ner)

    assert [e["text"] for e in out.entities] == ["דוד"] * 3
    assert out.text.count("[FIRST_NAME]") == 3 and "דוד" not in out.text
    assert all(tokens <= 512 for _, tokens in ner.calls)  # nothing was truncated
    assert len(ner.calls) > 1


def test_entities_in_overlap_regions_are_reported_exactly_once(tokenizer, long_text):
    targets = {"יוסף": "FIRST_NAME", "ברק": "LAST_NAME", "ת.ז.": "ID_PREFIX"}
    expected = sum(long_text.count(s) for s in targets)

    ner = TruncatingNer(tokenizer, targets)
    out = run(long_text, ner, max_tokens=96, overlap_tokens=48)  # heavy overlap

    assert len(ner.calls) > 5
    assert len(out.entities) == expected
    starts = [e["start"] for e in out.entities]
    assert starts == sorted(set(starts))  # no duplicates, ordered
    for e in out.entities:
        assert long_text[e["start"] : e["end"]] == e["text"]


def test_entity_spanning_a_window_cut_is_not_lost_or_duplicated(tokenizer):
    # Place a multi-word entity at many positions so some fall across planned cuts.
    entity = "בית החולים איכילוב"
    filler = " ".join(["מילה"] * 37)
    text = (" " + filler + " ").join([entity] * 40)

    ner = TruncatingNer(tokenizer, {entity: "HOSPITAL"})
    out = run(text, ner, max_tokens=80, overlap_tokens=30)

    assert len(out.entities) == 40
    assert out.text.count("[HOSPITAL]") == 40


def test_stage_never_calls_model_with_more_than_its_limit(tokenizer, long_text):
    ner = TruncatingNer(tokenizer, {"יוסף": "FIRST_NAME"})
    run(long_text, ner)  # default budget derived from the tokenizer
    assert ner.calls and all(tokens <= 512 for _, tokens in ner.calls)
    # and it uses the available context: windows are large, not per-sentence
    assert max(tokens for _, tokens in ner.calls) > 400
