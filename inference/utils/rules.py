"""Deterministic span rules applied on top of the NER model's predictions.

All functions work on character spans in the ORIGINAL text (nothing is rewritten before the model runs), so the
offsets in the output always fit the input. A span is {"start", "end", "text", "label"}, end exclusive.

Rules (see docs/HOSPITAL_FEEDBACK_AND_FIXES.md):
  F3  e-mail address is one CONTACT span (fixes `genetics.[CONTACT]`)
  F4  numeric and written dates are one DATE span; neighbouring DATE pieces are merged
  F2  Israeli ID checksum, used to choose ID vs INNER_ID for digit runs
  F1  digit runs the model left visible are masked (case numbers, IDs), with guards for lab values and doses
  F7  two surnames (or first names) joined by a hyphen are one span
  --  `ת.ז. 123456789` style IDs with context (what the old normalize_ids stage did, without rewriting the text)
"""
import re
from typing import Any, Dict, List

Span = Dict[str, Any]

# ---------------------------------------------------------------- patterns
EMAIL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%+\-]*@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")

# 15.03.2024, 15/3/2024, 15-03-2024, "15 . 03 . 2024"; two-digit years (22.03.50) only with dots/slashes.
# A hyphen right after a letter is a Hebrew prefix (`ב-17.10.1984`); a hyphen after a digit is not.
NUMERIC_DATE_RE = re.compile(
    r"(?<![\d.,/])(?<!\d-)(\d{1,2})\s?([./\-])\s?(\d{1,2})\s?\2\s?(\d{4}|(?<!-)\d{2})(?![\d])"
)
ISO_DATE_RE = re.compile(r"(?<![\d\-])((?:19|20)\d{2})-(\d{2})-(\d{2})(?![\d])")
# 28022015 only with a date word right before it (otherwise it is indistinguishable from a case number).
COMPACT_DATE_RE = re.compile(r"(?<=תאריך )(\d{2})(\d{2})((?:19|20)\d{2})(?!\d)")

# Hebrew spellings of the Gregorian months (with the common variants).
_MONTHS = [
    "ינואר", "פברואר", "מרץ", "מרס", "אפריל", "מאי", "יוני", "יולי",
    "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר",
]
# "2 במרץ 2024", "2 מרץ 2024", "2 ב-מרץ 2024", "2.3.2024" is handled above.
WRITTEN_DATE_RE = re.compile(
    r"(?<![\d.,/])(?<!\d-)(\d{1,2})\s+[בלמה]?-?(?:" + "|".join(_MONTHS) + r")\s+(\d{4})(?!\d)"
)

_HEB = "א-ת"
_MONTH_ALT = "|".join(_MONTHS)
# Month with a year ("ביוני 2022"), or month alone with an attached prefix ("ביוני", "מאוגוסט") or after "חודש".
# The prefix letter stays outside the span (annotation convention). Unprefixed "יוני" alone is not matched:
# it is also a first name.
MONTH_YEAR_RE = re.compile(
    rf"(?<![{_HEB}])(?:ו?[בלמה]-?|עד\s)?(?P<m>(?:{_MONTH_ALT}))(?:\s+(?P<y>\d{{4}}))?(?![{_HEB}])"
)
MONTH_AFTER_WORD_RE = re.compile(rf"חודש\s+(?P<m>(?:{_MONTH_ALT}))(?![{_HEB}])")

# Hebrew calendar: "י״א בתשרי תשפ״ד", "ג׳ באדר", "בניסן תשפ״ד". A date needs a day or a year, because months such
# as אדר, ניסן, אב, שבט are also names and ordinary words.
_HEB_MONTHS = r"(?:תשרי|חשוון|חשון|כסלו|טבת|שבט|אדר(?:\s[אב][׳'])?|ניסן|אייר|סיוון|סיון|תמוז|אב|אלול)"
_HEB_DAY = rf"(?:[{_HEB}][״\"][{_HEB}]|[{_HEB}][׳'])"
_HEB_YEAR = rf"ה?תש[{_HEB}]?[״\"]?[{_HEB}](?![{_HEB}])"
HEBREW_CAL_RE = re.compile(
    rf"(?<![{_HEB}])(?:[בלמ])?(?P<d>{_HEB_DAY})\s+[בלמ]?-?{_HEB_MONTHS}(?:\s+{_HEB_YEAR})?(?![{_HEB}])"
    rf"|(?<![{_HEB}])[בלמ]?-?{_HEB_MONTHS}\s+{_HEB_YEAR}"
)

# Digits after an ID context word: "ת.ז. 123456789", "תז: 12345678-9", "מספר דרכון 12345678"
ID_CONTEXT_RE = re.compile(
    r"(?:ת\.?\s?ז\.?|תעודת\s+זהות|מספר\s+דרכון|דרכון)\s*[:\-]?\s*(\d{7,9}(?:-\d)?)"
)

# 5+ digits, dashes allowed between groups: 1234567, 123-4567-89
DIGIT_RUN_RE = re.compile(r"(?<![\d.,/:])(?<!\d-)\d(?:-?\d){4,}(?![\w]|[.,]\d|/\d|:\d)")

# Words that, shortly before a number, say it is an identifier or case number.
INNER_ID_WORDS = re.compile(r"(?:תיק|מקרה|אירוע|פנימי|case)\s*[:\-]?\s*$", re.I)
ID_CONTEXT_WORDS = re.compile(
    r"(?:מספר|פנימי|מס'|מס\.|תיק|מקרה|אירוע|מטופל|קבלה|הפניה|ביקור|case|id|no\.?|#)\s*[:\-]?\s*$", re.I
)

# A number followed by one of these is a measurement, not an identifier.
UNIT_AFTER_RE = re.compile(
    r"^\s*(?:mg|mcg|µg|ug|g|kg|ml|cc|l|iu|u|units?|cells?|mmol|mmhg|bpm|"
    r"מ\"ג|מג|מ\"ל|מל|ג'|גרם|יח'|יחידות|יחידה|למיקרוליטר|מיקרוליטר|ננוגרם|מיקרוגרם|"
    r"לליטר|ליטר|לדקה|דקות|שעות|ימים|שקל|ש\"ח|₪|%)",
    re.I,
)
# 5-6 digit numbers look like lab values when the word right before them is a measurement.
MEASURE_BEFORE_RE = re.compile(
    r"(?:מינון|נפח|רמת|ספירת|מספר\s+טסיות|טסיות|לויקוציטים|תאים|קריאטינין|גלוקוז|פריטין|crp|esr|wbc|plt|hb)\s*\S*\s*$",
    re.I,
)

HYPHEN_JOIN_LABELS = {"LAST_NAME", "FIRST_NAME"}


# ---------------------------------------------------------------- helpers
def _span(text: str, start: int, end: int, label: str) -> Span:
    return {"start": start, "end": end, "text": text[start:end], "label": label}


def _overlaps(a: Span, start: int, end: int) -> bool:
    return a["start"] < end and start < a["end"]


def _replace_overlapping(spans: List[Span], new: Span) -> List[Span]:
    """Drop every span that overlaps `new`, then add it."""
    kept = [s for s in spans if not _overlaps(s, new["start"], new["end"])]
    kept.append(new)
    return kept


def _is_covered(spans: List[Span], start: int, end: int) -> bool:
    return any(_overlaps(s, start, end) for s in spans)


def israeli_id_valid(digits: str) -> bool:
    """Checksum of an Israeli ID (8 digits are zero-padded to 9). Dashes are ignored."""
    d = digits.replace("-", "")
    if not d.isdigit() or not (5 <= len(d) <= 9):
        return False
    d = d.zfill(9)
    total = 0
    for i, ch in enumerate(d):
        x = int(ch) * (1 if i % 2 == 0 else 2)
        total += x - 9 if x > 9 else x
    return total % 10 == 0


# ---------------------------------------------------------------- rules
def rule_emails(text: str, spans: List[Span]) -> List[Span]:
    for m in EMAIL_RE.finditer(text):
        spans = _replace_overlapping(spans, _span(text, m.start(), m.end(), "CONTACT"))
    return spans


def rule_dates(text: str, spans: List[Span]) -> List[Span]:
    for pattern in (NUMERIC_DATE_RE, WRITTEN_DATE_RE, ISO_DATE_RE, COMPACT_DATE_RE):
        for m in pattern.finditer(text):
            if pattern is NUMERIC_DATE_RE:
                day, month = int(m.group(1)), int(m.group(3))
            elif pattern is ISO_DATE_RE:
                day, month = int(m.group(3)), int(m.group(2))
            elif pattern is COMPACT_DATE_RE:
                day, month = int(m.group(1)), int(m.group(2))
            else:
                day, month = int(m.group(1)), 1
            if not (1 <= day <= 31 and 1 <= month <= 12):
                continue
            spans = _replace_overlapping(spans, _span(text, m.start(), m.end(), "DATE"))
    return spans


def rule_hebrew_calendar(text: str, spans: List[Span]) -> List[Span]:
    for m in HEBREW_CAL_RE.finditer(text):
        start = m.start("d") if m.group("d") else m.start()
        # drop a one-letter prefix (ב ל מ) glued to the first word
        if not m.group("d"):
            while start < m.end() and text[start] in "בלמ-" and not text[start:].startswith(("בתשרי", "אב")) and                     re.match(rf"[בלמ]-?(?:{_HEB_MONTHS})", text[start:m.end()]):
                start += 1 + (1 if text[start + 1] == "-" else 0)
                break
        spans = _replace_overlapping(spans, _span(text, start, m.end(), "DATE"))
    return spans


def rule_months(text: str, spans: List[Span]) -> List[Span]:
    """Month names the model left visible ("ביוני", "במאי 2020", "חודש אוגוסט"): hospital counts them as PHI.

    Added only where no model span exists, and only with a prefix, a year or the word "חודש" (see MONTH_YEAR_RE).
    """
    for m in MONTH_YEAR_RE.finditer(text):
        prefixed = m.start("m") > m.start()
        if not (prefixed or m.group("y")):
            continue
        end = m.end("y") if m.group("y") else m.end("m")
        if not _is_covered(spans, m.start("m"), end):
            spans.append(_span(text, m.start("m"), end, "DATE"))
    for m in MONTH_AFTER_WORD_RE.finditer(text):
        if not _is_covered(spans, m.start("m"), m.end("m")):
            spans.append(_span(text, m.start("m"), m.end("m"), "DATE"))
    return spans


def rule_merge_dates(text: str, spans: List[Span]) -> List[Span]:
    """DATE pieces separated only by . / - , or spaces become one DATE span."""
    spans = sorted(spans, key=lambda s: s["start"])
    out: List[Span] = []
    for s in spans:
        prev = out[-1] if out else None
        if (
            prev
            and prev["label"] == "DATE"
            and s["label"] == "DATE"
            and re.fullmatch(r"[\s./\-,]{0,3}", text[prev["end"]:s["start"]])
        ):
            out[-1] = _span(text, prev["start"], s["end"], "DATE")
        else:
            out.append(s)
    return out


def rule_context_ids(text: str, spans: List[Span]) -> List[Span]:
    for m in ID_CONTEXT_RE.finditer(text):
        spans = _replace_overlapping(spans, _span(text, m.start(1), m.end(1), "ID"))
    return spans


def rule_digit_runs(text: str, spans: List[Span]) -> List[Span]:
    """Mask long digit runs the model left visible.

    7+ digits: always (unless followed by a unit). 5-6 digits: only with an identifier word right before
    (`מספר תיק 123456`), because lab values such as `250000 למיקרוליטר` have the same shape.
    """
    for m in DIGIT_RUN_RE.finditer(text):
        start, end = m.start(), m.end()
        if _is_covered(spans, start, end):
            continue
        digits = re.sub(r"\D", "", m.group(0))
        before = text[max(0, start - 20):start]
        after = text[end:end + 20]
        if UNIT_AFTER_RE.match(after):
            continue
        has_context = bool(ID_CONTEXT_WORDS.search(before))
        if len(digits) < 7 and not has_context:
            continue
        if len(digits) < 7 and MEASURE_BEFORE_RE.search(before):
            continue
        if INNER_ID_WORDS.search(before):
            label = "INNER_ID"
        elif len(digits) in (8, 9) and israeli_id_valid(digits):
            label = "ID"
        elif re.match(r"0\d{8,9}$", digits) and not has_context:
            label = "CONTACT"  # phone number shape
        else:
            label = "INNER_ID"
        spans.append(_span(text, start, end, label))
    return spans


def rule_hyphen_names(text: str, spans: List[Span]) -> List[Span]:
    spans = sorted(spans, key=lambda s: s["start"])
    out: List[Span] = []
    for s in spans:
        prev = out[-1] if out else None
        if (
            prev
            and prev["label"] == s["label"]
            and s["label"] in HYPHEN_JOIN_LABELS
            and text[prev["end"]:s["start"]] in ("-", "־", "–")
        ):
            out[-1] = _span(text, prev["start"], s["end"], s["label"])
        else:
            out.append(s)
    return out


def resolve_overlaps(spans: List[Span]) -> List[Span]:
    """Keep the longest span when two overlap (earlier wins ties), return sorted by start."""
    kept: List[Span] = []
    for s in sorted(spans, key=lambda s: (s["start"], -(s["end"] - s["start"]))):
        if kept and s["start"] < kept[-1]["end"]:
            continue
        kept.append(s)
    return kept


def apply_rules(text: str, spans: List[Span]) -> List[Span]:
    """Run all rules over the model's spans and return the final, non-overlapping, sorted spans."""
    spans = [dict(s) for s in spans]
    spans = rule_emails(text, spans)
    spans = rule_dates(text, spans)
    spans = rule_hebrew_calendar(text, spans)
    spans = rule_months(text, spans)
    spans = rule_context_ids(text, spans)
    spans = rule_merge_dates(text, spans)
    spans = rule_hyphen_names(text, spans)
    spans = rule_digit_runs(text, spans)
    return resolve_overlaps(spans)
