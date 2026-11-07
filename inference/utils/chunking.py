"""Context-window chunking for token-classification (NER) models.

Why this exists
---------------
A Hugging Face NER pipeline silently truncates its input at the model's maximum
length, and a sentence-by-sentence loop throws away the context that tells the
model, for example, that "יוסף" in sentence 4 is the same patient introduced as
"יוסף ברק ת.ז. ..." in sentence 1.

So instead of cutting by sentence we cut by *token budget*:

1. Tokenize the whole text once (with offsets) to know where every token sits.
2. Fill each window with as many tokens as the model can take.
3. End the window at the last sentence boundary that still leaves it reasonably
   full (never in the middle of a word, and never right after an abbreviation
   such as ת.ז.). If there is no usable boundary, cut at a word boundary.
4. Start the next window ``overlap_tokens`` earlier, again snapped to a sentence
   start, so text near a cut is always seen with context on both sides.
5. Each window "owns" a core region (the middle of its overlaps with its
   neighbours). Entities are kept from the window that owns their centre, so
   there are no duplicates and each entity is read with the most context.

Everything here works on character offsets into the original text.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Tokens of slack kept below the model limit: re-tokenizing a substring can add
# a token or two compared to tokenizing the full text (e.g. SentencePiece "▁").
DEFAULT_MARGIN = 8
DEFAULT_OVERLAP = 64
# A window is never closed before it is this full (fraction of the budget),
# unless the text ends. Prevents tiny windows when a boundary comes very early.
MIN_FILL = 0.5
# Used only when no real tokenizer is available (e.g. test fakes): words and
# punctuation are counted as tokens, so keep the budget conservative.
FALLBACK_MAX_TOKENS = 200

_SENTINEL_LENGTH = 100_000  # HF uses ~1e30 when model_max_length is unknown

# Abbreviations whose trailing period does not end a sentence.
ABBREVIATIONS = frozenset(
    {
        "ת.ז.", "ת.ד.", "פרופ.", "גב.", "עו.ד.", "בע.מ.", "וכו.", "וכד.", "כו.",
        "מס.", "טל.", "נ.צ.", "ע.י.", "א.ק.ג.", "ד.נ.", "ת.ד", "dr.", "mr.",
        "mrs.", "ms.", "prof.", "vs.", "etc.", "e.g.", "i.e.",
    }
)

_END_PUNCT = ".!?…"
_CLOSERS = "\"')]}”’»"
# 12.03.2024  /  3.5  /  1,000.  -> numeric tokens may legitimately end a sentence
_NUMERIC_TOKEN = re.compile(r"^\d[\d./,:-]*\d?$")
_ENDS_WITH_DATE = re.compile(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$")  # e.g. "ב-07.02.2024"
_SENTENCE_END = re.compile(
    r"([.!?…]+)([\"')\]}”’»]*)(\s+)(?=\S)"
)
_NEWLINE_RUN = re.compile(r"[ \t\r]*\n\s*(?=\S)")


# Labels that must stay glued to the value that follows them.
_ID_LABEL = re.compile(r"(ת\.?ז\.?|ת\"ז|תז|דרכון|מס'|מספר)[:\-]?$")


def cut_allowed_before(text: str, pos: int) -> bool:
    """False if cutting the text at ``pos`` would separate an abbreviation or an
    ID label (ת.ז., מספר דרכון, ...) from what follows it."""
    if pos <= 0 or pos >= len(text):
        return True
    head = text[:pos].rstrip()
    token_start = len(head)
    while token_start > 0 and not head[token_start - 1].isspace():
        token_start -= 1
    token = head[token_start:]
    if _ID_LABEL.match(token):
        return False
    if token.endswith(".") and not _is_real_sentence_end(head, len(head) - 1, "."):
        return False
    return True


@dataclass(frozen=True)
class Chunk:
    """A window of the text plus the region of it whose entities we keep."""

    start: int
    end: int
    core_start: int
    core_end: int

    def owns(self, ent_start: int, ent_end: int) -> bool:
        centre = (ent_start + ent_end) / 2
        return self.core_start <= centre < self.core_end


# --------------------------------------------------------------------------- #
# Sentence boundaries
# --------------------------------------------------------------------------- #
def _is_real_sentence_end(text: str, period_idx: int, punct: str) -> bool:
    """Decide whether the end punctuation at ``period_idx`` closes a sentence.

    Being wrong in the "not a boundary" direction only costs a slightly less
    tidy cut; being wrong the other way cuts an ID/abbreviation from its
    context. So we are deliberately conservative.
    """
    if punct != "." and not punct.endswith("."):
        return True  # ? ! … always end a sentence
    token_start = period_idx
    while token_start > 0 and not text[token_start - 1].isspace():
        token_start -= 1
    token = text[token_start : period_idx + len(punct)]
    core = token.rstrip(_END_PUNCT).strip(_CLOSERS + "([{\"'")
    if not core:
        return True
    if token.lower() in ABBREVIATIONS or (core + ".").lower() in ABBREVIATIONS:
        return False
    if "@" in core:
        return True  # e-mail address at the end of a sentence
    if _ENDS_WITH_DATE.search(core):
        return True  # "... ב-07.02.2024. אשתו ..."
    if _NUMERIC_TOKEN.match(core):
        # "1." / "א." style list markers are not sentence ends, but dates and
        # longer numbers ("ביום 12.03.2024.") are.
        return len(core) > 2 or "." in core
    if len(core) == 1 and core.isalpha():
        return False  # "סעיף א." / "בדיקה ב." / initials
    if "." in core:
        return False  # dotted acronym: א.ק.ג.  ת.ז.  U.S.
    return True


def sentence_starts(text: str) -> List[int]:
    """Character offsets at which a new sentence (or line) may begin."""
    starts = set()
    for m in _SENTENCE_END.finditer(text):
        punct_start = m.start(1)
        if _is_real_sentence_end(text, punct_start, m.group(1)):
            starts.add(m.end())
    for m in _NEWLINE_RUN.finditer(text):
        starts.add(m.end())
    return sorted(starts)


# --------------------------------------------------------------------------- #
# Tokenizer helpers
# --------------------------------------------------------------------------- #
def resolve_token_budget(ner_pipe: Any, margin: int = DEFAULT_MARGIN) -> Tuple[Optional[Any], int]:
    """Return ``(tokenizer, content_budget)`` for a NER pipeline.

    ``content_budget`` is the number of text tokens a window may hold, i.e. the
    model limit minus special tokens minus a safety margin. If the pipeline has
    no usable fast tokenizer (test fakes), returns ``(None, FALLBACK_MAX_TOKENS)``.
    """
    tokenizer = getattr(ner_pipe, "tokenizer", None)
    if tokenizer is None or not getattr(tokenizer, "is_fast", False):
        return None, FALLBACK_MAX_TOKENS

    max_len = getattr(tokenizer, "model_max_length", None)
    if not isinstance(max_len, int) or max_len <= 0 or max_len > _SENTINEL_LENGTH:
        config = getattr(getattr(ner_pipe, "model", None), "config", None)
        max_len = getattr(config, "max_position_embeddings", None) or 512
    specials = tokenizer.num_special_tokens_to_add(pair=False)
    return tokenizer, max(16, max_len - specials - margin)


def token_offsets(text: str, tokenizer: Optional[Any]) -> List[Tuple[int, int]]:
    """Character span of every token, in order. Zero-width tokens are dropped."""
    if tokenizer is None:
        return [m.span() for m in re.finditer(r"\w+|[^\w\s]", text)]
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )
    return [(s, e) for s, e in enc["offset_mapping"] if e > s]


def _token_count(text: str, tokenizer: Any) -> int:
    ids = tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
    return len(ids)


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def plan_chunks(
    text: str,
    offsets: Sequence[Tuple[int, int]],
    max_tokens: int,
    overlap_tokens: int = DEFAULT_OVERLAP,
    boundaries: Optional[Sequence[int]] = None,
) -> List[Chunk]:
    """Split ``text`` into overlapping windows of at most ``max_tokens`` tokens."""
    if max_tokens < 2:
        raise ValueError("max_tokens must be at least 2")
    overlap_tokens = max(0, min(overlap_tokens, max_tokens // 2))

    n = len(offsets)
    if n <= max_tokens:
        return [Chunk(0, len(text), 0, len(text))]

    if boundaries is None:
        boundaries = sentence_starts(text)

    # token index of the first token starting at/after each boundary
    starts_chars = [s for s, _ in offsets]
    from bisect import bisect_left

    boundary_tokens = sorted({bisect_left(starts_chars, b) for b in boundaries if b < len(text)})
    boundary_tokens = [t for t in boundary_tokens if 0 < t < n]

    def is_word_start(t: int) -> bool:
        """True if token t begins a new word (whitespace or punctuation before it)."""
        if t <= 0 or t >= n:
            return True
        s = offsets[t][0]
        if not (offsets[t - 1][1] < s or text[s - 1].isspace()):
            return False
        return cut_allowed_before(text, s)

    def last_boundary_at_or_before(limit: int, lower: int) -> Optional[int]:
        from bisect import bisect_right

        idx = bisect_right(boundary_tokens, limit) - 1
        if idx >= 0 and boundary_tokens[idx] >= lower:
            return boundary_tokens[idx]
        return None

    def last_word_start_at_or_before(limit: int, lower: int) -> int:
        for t in range(limit, lower - 1, -1):
            if is_word_start(t):
                return t
        return limit  # hard cut: no word boundary in range

    spans: List[Tuple[int, int]] = []  # token index ranges [i, j)
    i = 0
    while True:
        j = min(i + max_tokens, n)
        if j >= n:
            spans.append((i, n))
            break

        min_end = i + max(1, int(max_tokens * MIN_FILL))
        prev_end = spans[-1][1] if spans else 0
        # a window must reach past the previous one, or it adds no new text
        cut = last_boundary_at_or_before(j, max(min_end, prev_end + 1))
        if cut is None:
            cut = last_word_start_at_or_before(j, max(min_end, prev_end + 1))
        j = cut
        spans.append((i, j))

        # next window starts `overlap_tokens` before this one ends, but always
        # moves forward by at least a third of what we just consumed
        target = max(j - overlap_tokens, i + max(1, (j - i) // 3))
        lower = i + max(1, (j - i) // 3)
        nxt = last_boundary_at_or_before(target, lower)
        if nxt is None:
            nxt = last_word_start_at_or_before(target, lower)
        i = max(nxt, i + 1)

    # convert token spans to character windows + cores
    windows: List[Tuple[int, int]] = []
    for k, (a, b) in enumerate(spans):
        start = 0 if k == 0 else offsets[a][0]
        end = len(text) if k == len(spans) - 1 else offsets[b - 1][1]
        windows.append((start, end))

    chunks: List[Chunk] = []
    for k, (start, end) in enumerate(windows):
        core_start = 0 if k == 0 else (windows[k][0] + windows[k - 1][1]) // 2
        core_end = len(text) if k == len(windows) - 1 else (windows[k + 1][0] + windows[k][1]) // 2
        chunks.append(Chunk(start, end, core_start, core_end))
    return chunks


def chunk_text(
    text: str,
    tokenizer: Optional[Any],
    max_tokens: int,
    overlap_tokens: int = DEFAULT_OVERLAP,
    model_limit: Optional[int] = None,
) -> List[Chunk]:
    """Plan windows for ``text`` and verify each one really fits the model.

    With a real tokenizer, every window is re-tokenized (including special
    tokens). If any exceeds ``model_limit`` the budget is reduced and the plan
    is rebuilt, so nothing is ever silently truncated downstream.
    """
    offsets = token_offsets(text, tokenizer)
    boundaries = sentence_starts(text)
    budget = max_tokens
    for _ in range(6):
        chunks = plan_chunks(text, offsets, budget, overlap_tokens, boundaries)
        if tokenizer is None or model_limit is None:
            return chunks
        specials = tokenizer.num_special_tokens_to_add(pair=False)
        worst = max(_token_count(text[c.start : c.end], tokenizer) for c in chunks) + specials
        if worst <= model_limit:
            return chunks
        budget = max(8, budget - (worst - model_limit) - 4)
        logger.debug("chunk too long (%d > %d); retrying with budget %d", worst, model_limit, budget)
    logger.warning("could not fit all chunks within %s tokens; input may be truncated", model_limit)
    return chunks
