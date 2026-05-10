import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.nlp_utils import get_sentence_tokenizer
from utils.pipeline_utils import DeidDoc


logger = logging.getLogger(__name__)

# Type alias for the shared replacement observer hook used by stages.
ReplacementObserver = Callable[[str, str, int, int, str, str], None]

# Simple patterns and mappings for preprocessing normalization
NUMERIC_DATE_PATTERN = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")
HEBREW_MONTHS = {
    "ינואר": 1,
    "פברואר": 2,
    "מרץ": 3,
    "אפריל": 4,
    "מאי": 5,
    "יוני": 6,
    "יולי": 7,
    "אוגוסט": 8,
    "ספטמבר": 9,
    "אוקטובר": 10,
    "נובמבר": 11,
    "דצמבר": 12,
}
HEBREW_TEXTUAL_DATE_PATTERN = re.compile(
    r"\b(\d{1,2})\s+ב?(" + "|".join(HEBREW_MONTHS.keys()) + r")\s+(\d{4})\b"
)

# ID-like patterns (Israeli ID, passport number, etc.)
ID_PREFIX_PATTERN = re.compile(
    r"(ת\.?ז\.?|תז|מספר\s+דרכון)\s*[:\-]?\s*([0-9]{7,9})"
)

# Matches placeholder tokens inserted by normalization stages (e.g. [ID])
# Used to prevent the NER model from tagging text inside these markers.
PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z_]+\]")


def _normalize_numeric_date_match(match: re.Match) -> str:
    """
    Normalize numeric dates like 15.03.2024 or 10/05/1980 to YYYY-MM-DD.
    Assumes day-first ordering (Israeli / European style).
    """
    day_s, month_s, year_s = match.groups()
    try:
        day = int(day_s)
        month = int(month_s)
        year = int(year_s)
    except ValueError:
        return match.group(0)

    if not (1 <= day <= 31 and 1 <= month <= 12):
        return match.group(0)

    # result = f"{year:04d}-{month:02d}-{day:02d}"
    # result = f"{year:04d}/{month:02d}/{day:02d}"
    result = f"{day:02d}.{month:02d}.{year:04d}"

    return result


def _normalize_hebrew_textual_date_match(match: re.Match) -> str:
    """
    Normalize textual Hebrew dates like '15 במאי 1948' to YYYY-MM-DD.
    """
    day_s, month_name, year_s = match.groups()
    try:
        day = int(day_s)
        year = int(year_s)
    except ValueError:
        return match.group(0)

    month = HEBREW_MONTHS.get(month_name)
    if month is None or not (1 <= day <= 31):
        return match.group(0)

    return f"{year:04d}-{month:02d}-{day:02d}"


def normalize_dates_stage(
    doc: DeidDoc,
    replacement_observer: Optional[ReplacementObserver] = None,
) -> DeidDoc:
    """
    Pipeline stage that normalizes date strings in the document text to
    a consistent ISO-like format (YYYY-MM-DD).

    This is intended as an early preprocessing step to make downstream
    detection (e.g. by an LLM) easier and more robust.
    """
    text = doc.text
    if not isinstance(text, str) or not text:
        return doc

    stage_name = "normalize_dates"

    # Numeric dates first (e.g. 15.03.2024, 10/05/1980)
    def repl_numeric(match: re.Match) -> str:
        original = match.group(0)
        normalized = _normalize_numeric_date_match(match)
        if replacement_observer and normalized != original:
            start, end = match.span()
            replacement_observer(
                stage_name,
                original,
                start,
                end,
                normalized,
                "DATE",
            )
        return normalized

    text = NUMERIC_DATE_PATTERN.sub(repl_numeric, text)

    # Textual Hebrew dates (e.g. 15 במאי 1948)
    def repl_textual(match: re.Match) -> str:
        original = match.group(0)
        normalized = _normalize_hebrew_textual_date_match(match)
        if replacement_observer and normalized != original:
            start, end = match.span()
            replacement_observer(
                stage_name,
                original,
                start,
                end,
                normalized,
                "DATE",
            )
        return normalized

    text = HEBREW_TEXTUAL_DATE_PATTERN.sub(repl_textual, text)

    doc.text = text
    return doc


def normalize_ids_stage(
    doc: DeidDoc,
    replacement_observer: Optional[ReplacementObserver] = None,
) -> DeidDoc:
    """
    Pipeline stage that masks personal identifier digits while preserving the
    Hebrew prefix (e.g. ת.ז., מספר דרכון):

        'ת.ז. 345678901' -> 'ת.ז. [ID]'

    Processes matches right-to-left so recorded positions stay valid across
    multiple IDs in the same document. Detected entities are appended to
    doc.entities so they survive the downstream NER stage.
    """
    text = doc.text
    if not isinstance(text, str) or not text:
        return doc

    stage_name = "normalize_ids"
    new_entities: List[Dict[str, Any]] = []

    for match in reversed(list(ID_PREFIX_PATTERN.finditer(text))):
        prefix = match.group(1)
        digits = match.group(2)
        between = text[match.end(1):match.start(2)]  # separator between prefix and digits

        entity_start = match.start(2)
        entity_end = match.end(2)

        new_entities.append(
            {"start": entity_start, "end": entity_end, "text": digits, "label": "ID"}
        )

        if replacement_observer:
            replacement_observer(stage_name, digits, entity_start, entity_end, "[ID]", "ID")

        text = text[:match.start()] + prefix + between + "[ID]" + text[match.end():]

    doc.text = text
    doc.entities = doc.entities + list(reversed(new_entities))
    return doc


def clean_label(label: str) -> str:
    """
    Strips B- and I- prefixes and removes backslashes from labels.
    Some NER models export label maps with backslash-escaped underscores
    (e.g. AGE\_ABOVE\_89); this normalises them to AGE_ABOVE_89.
    """
    if label.startswith("B-") or label.startswith("I-"):
        label = label[2:]
    return label.replace("\\", "")


def ner_mask_stage(
    mode: str,
    ner_pipe: Any,
    replacement_observer: Optional[ReplacementObserver] = None,
):
    """
    Factory returning a stage that:
    - runs sentence-level NER using the HF pipeline,
    - collects structured entities,
    - applies masking to the document text.

    If `replacement_observer` is provided, it will be called once for
    each replacement with:
        (stage_name, original_text, start, end, replacement_text, label)
    """

    def stage(doc: DeidDoc) -> DeidDoc:
        if not isinstance(doc.text, str):
            # Keep behaviour similar to original apply_ner_mask
            doc.text = str(doc.text)
            doc.entities = []
            return doc

        text = doc.text
        stage_name = "ner_mask"

        tokenizer = get_sentence_tokenizer("nltk")
        sentences = tokenizer(text)
        spans: List[Tuple[int, int]] = []

        # Reliably find global start/end character offsets for each sentence
        # without relying on internal NLTK pickle loading which crashes on some environments.
        current_idx = 0
        for sent in sentences:
            start_idx = text.find(sent, current_idx)
            if start_idx == -1:
                # Fallback if NLTK somehow mutated the sentence
                start_idx = current_idx

            end_idx = start_idx + len(sent)
            spans.append((start_idx, end_idx))
            current_idx = end_idx

        # Collect spans already occupied by placeholders from prior stages
        # (e.g. [ID] inserted by normalize_ids). NER predictions that fall
        # inside these regions are artifacts of the placeholder text, not
        # real entities, so we discard them.
        placeholder_spans = [
            (m.start(), m.end()) for m in PLACEHOLDER_PATTERN.finditer(text)
        ]

        def _in_placeholder(start: int, end: int) -> bool:
            return any(ps <= start and end <= pe for ps, pe in placeholder_spans)

        entities_found: List[Dict[str, Any]] = []
        edits: List[Tuple[int, int, str]] = []

        for start_idx, end_idx in spans:
            sent = text[start_idx:end_idx]
            results = ner_pipe(sent)

            for ent in results:
                ent_start_global = start_idx + ent["start"]
                ent_end_global = start_idx + ent["end"]

                if _in_placeholder(ent_start_global, ent_end_global):
                    continue

                clean_ent_label = clean_label(ent["entity_group"])

                actual_text = text[ent_start_global:ent_end_global]
                entities_found.append(
                    {
                        "start": ent_start_global,
                        "end": ent_end_global,
                        "text": actual_text,
                        "label": clean_ent_label,
                    }
                )

                if mode == "label":
                    rep = f"[{clean_ent_label}]"
                elif mode == "x":
                    rep = "X" * (ent["end"] - ent["start"])
                else:
                    rep = "[REDACTED]"

                if replacement_observer:
                    replacement_observer(
                        stage_name,
                        actual_text,
                        ent_start_global,
                        ent_end_global,
                        rep,
                        clean_ent_label,
                    )

                edits.append((ent_start_global, ent_end_global, rep))

        edits.sort(key=lambda x: x[0], reverse=True)

        deid_text = text
        for start, end, rep in edits:
            deid_text = deid_text[:start] + rep + deid_text[end:]

        doc.text = deid_text
        doc.entities = sorted(doc.entities + entities_found, key=lambda e: e["start"])
        return doc

    return stage


def label_clean_stage(doc: DeidDoc) -> DeidDoc:
    """
    Pipeline stage that normalizes entity labels in-place using `clean_label`.

    This is primarily useful when upstream detectors may emit BIO-prefixed
    labels (e.g. 'B-PER', 'I-PER'). It is idempotent: applying it multiple
    times will not change already-clean labels.
    """
    if not doc.entities:
        return doc

    for ent in doc.entities:
        label = ent.get("label")
        if isinstance(label, str):
            ent["label"] = clean_label(label)
    return doc


def pipeline_end_stage(doc: DeidDoc) -> DeidDoc:
    """
    A no-op stage that currently serves as a logical end marker for
    the document pipeline.
    """
    return doc

