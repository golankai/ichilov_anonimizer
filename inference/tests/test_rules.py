"""Tests for the deterministic span rules (utils/rules.py). No model needed."""
import json
import sys
from pathlib import Path

import pytest

INFERENCE_ROOT = Path(__file__).parent.parent
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from utils.rules import apply_rules, israeli_id_valid
from utils.doc_pipeline import build_doc_pipeline
from utils.pipeline_utils import DeidDoc

DATA_DIR = INFERENCE_ROOT / "dummy_data"


def labels(text, spans=()):
    return [(s["label"], text[s["start"]:s["end"]]) for s in apply_rules(text, list(spans))]


def span(text, sub, label):
    i = text.index(sub)
    return {"start": i, "end": i + len(sub), "text": sub, "label": label}


# ---------------------------------------------------------------- email (F3 / P1)
def test_email_prefix_is_not_left_behind():
    t = "לכתוב ל genetics.clinic@ichilov.org.il בלבד"
    # model only found the tail, as in the first GPU run
    got = labels(t, [span(t, "clinic@ichilov.org.il", "CONTACT")])
    assert got == [("CONTACT", "genetics.clinic@ichilov.org.il")]


def test_email_found_without_model():
    assert labels("mail: a.b-c@x.co.il.") == [("CONTACT", "a.b-c@x.co.il")]


# ---------------------------------------------------------------- dates (F4 / P2)
@pytest.mark.parametrize(
    "date",
    ["15.03.2024", "5/3/2024", "15-03-2024", "22.03.50", "1949-07-18", "2 במרץ 2024", "15 בספטמבר 2023", "3 מאי 1940"],
)
def test_date_is_one_span(date):
    t = f"אושפז בתאריך {date} ושוחרר"
    assert labels(t) == [("DATE", date)]


def test_date_pieces_from_the_model_are_replaced_by_one_span():
    t = "בוצע 15.03.2024 בבוקר"
    pieces = [span(t, "15", "DATE"), span(t, "03", "DATE"), span(t, "2024", "DATE")]
    assert labels(t, pieces) == [("DATE", "15.03.2024")]


def test_adjacent_date_pieces_are_merged():
    t = "תאריך 15 / 03 / 2024"
    pieces = [span(t, "15", "DATE"), span(t, "03", "DATE"), span(t, "2024", "DATE")]
    assert labels(t, pieces) == [("DATE", "15 / 03 / 2024")]


def test_hebrew_prefix_stays_outside_numeric_date():
    t = "הבדיקה בוצעה ב-17.10.1984 במרפאה"
    assert labels(t) == [("DATE", "17.10.1984")]


@pytest.mark.parametrize("not_date", ["לחץ דם 130/80, דופק 88", "ציון 12/15 בסולם", "בשעה 08:30", "32.13.2024", "CRP 125"])
def test_not_a_date(not_date):
    assert labels(not_date) == []


def test_hebrew_calendar_and_months():
    assert labels("נולד בכ״ה בניסן תשפ״ד והתגורר") == [("DATE", "כ״ה בניסן תשפ״ד")]
    assert labels("נולד בג׳ באדר תשפ״ג") == [("DATE", "ג׳ באדר תשפ״ג")]
    assert labels("תסמינים חדשים ביוני ולכן") == [("DATE", "יוני")]
    assert labels("נבדק לאחרונה במאי 2020 ונמצא") == [("DATE", "מאי 2020")]
    assert labels("בחודש אוגוסט") == [("DATE", "אוגוסט")]


def test_month_like_names_without_date_shape_are_left_alone():
    # `אדר`, `ניסן`, `יוני` are also first names; no day, year or prefix means no rule fires.
    assert labels("ניסן כהן הגיע") == []
    assert labels("יוני אמר שלום") == []
    assert labels("אושר ואדר הגיעו") == []


def test_month_rule_does_not_override_the_model():
    t = "הגיע יוני לבדיקה ב-יוני"
    # model marked "יוני" as a first name; the rule must not add a DATE on top of it
    got = labels(t, [span(t, "יוני", "FIRST_NAME")])
    assert ("FIRST_NAME", "יוני") in got


# ---------------------------------------------------------------- ids (F1 / F2)
def test_israeli_id_checksum():
    assert israeli_id_valid("123456782")
    assert israeli_id_valid("12345678-2")
    assert not israeli_id_valid("123456789")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ת.ז. 123456782", ("ID", "123456782")),
        ("תז: 12345678-2", ("ID", "12345678-2")),
        ("מספר דרכון 12345678", ("ID", "12345678")),
        ("מספר תיק 987654", ("INNER_ID", "987654")),
        ("מס' מקרה 1234567", ("INNER_ID", "1234567")),
        ("מספר מטופל H-4945180", ("INNER_ID", "4945180")),
        ("ראו 54321098 בקובץ", ("INNER_ID", "54321098")),
        ("זוהה לפי מספר 123456782", ("ID", "123456782")),
    ],
)
def test_digit_run_gets_masked(text, expected):
    (got,) = labels(text)
    assert got == expected


@pytest.mark.parametrize(
    "text",
    [
        "מינון הפרין 10000 יחידות.",
        "ספירת תאים לבנים 15000 למיקרוליטר.",
        "נפח השתן 12000 מ\"ל ביממה.",
        "מספר טסיות 250000 למיקרוליטר.",
        "רמת פריטין 24500 ננוגרם.",
        "חום 38.5 ודופק 88",
        "גודל 12345.67 מ\"מ",
    ],
)
def test_lab_values_and_doses_stay_readable(text):
    assert labels(text) == []


def test_digit_run_already_found_by_model_is_not_duplicated():
    t = "מספר תיק 987654"
    got = labels(t, [span(t, "987654", "CONTACT")])
    assert got == [("CONTACT", "987654")]  # label is the model's business; no second span


# ---------------------------------------------------------------- hyphenated surname (F7 / P4)
def test_hyphenated_surname_is_one_span():
    t = "הגב' דנה אלמוג-כץ הגיעה"
    pieces = [span(t, "דנה", "FIRST_NAME"), span(t, "אלמוג", "LAST_NAME"), span(t, "כץ", "LAST_NAME")]
    assert labels(t, pieces) == [("FIRST_NAME", "דנה"), ("LAST_NAME", "אלמוג-כץ")]


def test_different_labels_are_not_joined():
    t = "דוד-כהן"
    pieces = [span(t, "דוד", "FIRST_NAME"), span(t, "כהן", "LAST_NAME")]
    assert len(labels(t, pieces)) == 2


# ---------------------------------------------------------------- whole pipeline
class FakeNer:
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


def test_pipeline_with_rules_keeps_offsets_in_the_original_text():
    text = "נולד ב-2 במרץ 2024. ת.ז. 123456782. דוד כהן, מספר תיק 987654."
    pipe = build_doc_pipeline(FakeNer({"דוד": "FIRST_NAME", "כהן": "LAST_NAME"}), "label", use_rules=True)
    doc = pipe(DeidDoc(text=text))
    assert doc.text == "נולד ב-[DATE]. ת.ז. [ID]. [FIRST_NAME] [LAST_NAME], מספר תיק [INNER_ID]."
    for e in doc.entities:  # no offset drift
        assert text[e["start"]:e["end"]] == e["text"]


def test_pipeline_does_not_rewrite_text_before_the_model():
    seen = []

    class Spy:
        def __call__(self, sentence):
            seen.append(sentence)
            return []

    text = "ת.ז. 123456782 ביום 2 במרץ 2024"
    build_doc_pipeline(Spy(), "label", use_rules=True)(DeidDoc(text=text))
    assert seen == [text]


# ---------------------------------------------------------------- against the fake ground truth
def _truth_rows():
    for f in sorted(DATA_DIR.glob("*.truth.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            yield json.loads(line)


def test_rules_never_touch_declared_traps_or_invent_spans():
    """On every fake document the rules alone (empty model output) may only produce spans that overlap truth."""
    bad = []
    for d in _truth_rows():
        for s in apply_rules(d["text"], []):
            if not any(s["start"] < g["end"] and g["start"] < s["end"] for g in d["label"]):
                bad.append((d["tier"], d["id"], d["text"][s["start"]:s["end"]]))
    assert bad == []


def test_rules_alone_cover_formats_they_are_responsible_for():
    """Dates, emails, case numbers and IDs are found exactly (span and label) without any model."""
    wanted = ("date-", "email-", "inner-id", "id-", "passport")
    total = hit = 0
    misses = []
    for d in _truth_rows():
        got = apply_rules(d["text"], [])
        for g in d["label"]:
            if not g["tag"].startswith(wanted) or g.get("policy"):
                continue
            total += 1
            ok = any(s["start"] == g["start"] and s["end"] == g["end"] and s["label"] == g["label"] for s in got)
            hit += ok
            if not ok:
                misses.append((g["tag"], d["text"][g["start"]:g["end"]]))
    assert total > 150
    assert misses == []
