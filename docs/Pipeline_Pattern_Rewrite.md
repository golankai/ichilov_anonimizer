### De-identification Pipeline Pattern (Rewrite)

This document captures the current design, assumptions, and usage patterns for the new pipeline-based de-identification flow implemented in `de_identify_pipeline.py`.

For concrete CLI commands and step‑by‑step run/test recipes (with and without logging), see `docs/Deid_Pipeline_Usage.MD`.

---

### 1. High-level design

- **Core idea**: a **linear pipeline of small stages**, each taking and returning a `DeidDoc`:
  - **`DeidDoc`**: simple container
    - `text: str` – current working text.
    - `entities: list[dict]` – structured entities (e.g. from NER).
- **Stages are pure functions**: `Stage = Callable[[DeidDoc], DeidDoc]`.
- **Composition**:
  - Stages are composed using `toolz.compose_left` via `pipeline_utils.build_pipeline_with_optional_observers`.
  - The builder automatically wraps stages to support a per-stage observer hook.
- **Entry point**: CLI script `de_identify_pipeline.py` which:
  1. Loads NLTK, model, and data.
  2. Builds the pipeline with configured stages.
  3. Runs it over all rows.
  4. Writes CSV and JSON outputs with de-identified text and entities.

---

### 2. Current stages and ordering (CLI default)

The default CLI pipeline (in `run_processing_loop`) currently uses:

1. **`normalize_ids`** (preprocessing)
   - Detects ID-like patterns (Israeli ID, passport) and rewrites them to a neutral token.
2. **`normalize_dates`** (preprocessing)
   - Normalizes date strings to ISO-like `YYYY-MM-DD` format or other.
3. **`ner_mask`**
   - Runs HF NER over the text (sentence by sentence).
   - Builds the de-identified text by applying replacements based on entities.
   - Populates `doc.entities` with global spans and labels.
4. **`label_clean`**
   - Normalizes labels (e.g. strips `B-` / `I-` prefixes) in `doc.entities`.

Order is constructed in `build_doc_pipeline` as:

- Optional prefix stages (currently `normalize_ids`, `normalize_dates`), then
- Core stages (`ner_mask`, `label_clean`, plus a no-op `pipeline_end_stage` that currently acts as a logical end marker).

---

### 3. Preprocessing stages and assumptions

#### 3.1 `normalize_ids_stage`

- **Goal**: Reduce variation in how personal identifiers are written so downstream detectors have an easier, more uniform pattern to work with.
- **Patterns handled** (case-sensitive, Hebrew):
  - `ת.ז. 345678901`
  - `ת"ז 345678901`
  - `תז 345678901`
  - `מספר דרכון 12345678`
  - Variants with optional `:`, `-`, and whitespace.
- **Regex**:
  - `ID_PREFIX_PATTERN = r"(ת\.?ז\.?|תז|מספר\s+דרכון)\s*[:\-]?\s*([0-9]{7,9})"`.
- **Transform**:
  - Always rewrites to: `" ID_<digits>"`, for example:
    - `ת.ז. 345678901` → ` ID_345678901`.
    - `מספר דרכון 12345678` → ` ID_12345678`.
- **Label used in observers**: `"ID"`.

#### 3.2 `normalize_dates_stage`

- **Goal**: Normalize dates into a standard, LLM-friendly and regex-friendly format (`YYYY-MM-DD`), reducing ambiguity and improving downstream detection.
- **Assumption**: Numeric dates are **day-first** (Israeli / European style), i.e. `DD/MM/YYYY`.
- **Numeric patterns**:
  - Examples: `15.03.2024`, `10/05/1980`, `01-02-2024`.
  - Regex: `NUMERIC_DATE_PATTERN = r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b"`.
  - Transform: `DD.MM.YYYY` → `YYYY-MM-DD`, e.g. `15.03.2024` → `2024-03-15`.
- **Textual Hebrew patterns**:
  - Examples: `15 במאי 1948`, `1 ינואר 2000`.
  - Uses a month name map (`ינואר`, `פברואר`, …, `דצמבר`) and regex:
    - `HEBREW_TEXTUAL_DATE_PATTERN = r"\b(\d{1,2})\s+ב?(<MONTH_NAME>)\s+(\d{4})\b"`.
  - Transform: `15 במאי 1948` → `1948-05-15`.
- **Validation**:
  - Only normalizes when day ∈ [1,31] and month ∈ [1,12]; otherwise leaves original text.
- **Label used in observers**: `"DATE"`.

---

### 4. Stage-level observability

Two orthogonal observability mechanisms are supported:

#### 4.1 Stage observer (`--observe-stages`)

- **Interface**: `observer(stage_name, doc) -> None`.
- Implemented in `build_pipeline_with_optional_observers` (in `pipeline_utils.py`):
  - Wraps each stage so the observer is called **after** the stage runs.
- In `de_identify_pipeline.py`, when `--observe-stages` is passed, we install:

```python
def stage_observer(name: str, doc: DeidDoc) -> None:
    logger.info(
        "[stage=%s] text_len=%d entities=%d",
        name,
        len(doc.text),
        len(doc.entities),
    )
    if logger.isEnabledFor(logging.DEBUG):
        snippet = doc.text[:200].replace("\n", " ")
        logger.debug(
            "[stage=%s] text_snippet=%r entities=%s",
            name,
            snippet,
            doc.entities,
        )
```

- **Logging levels**:
  - INFO: per-stage text length and entity count.
  - DEBUG (with `--verbose`): text snippet and full entities.

#### 4.2 Replacement observer (`--log-replacements`)

- **Interface**: `ReplacementObserver(stage_name, original, start, end, replacement, label)`.
  - `stage_name`: which stage emitted this change (e.g. `"normalize_dates"`, `"normalize_ids"`, `"ner_mask"`).
  - `original`: substring before replacement.
  - `start`, `end`: character offsets in the current text.
  - `replacement`: new string applied.
  - `label`: semantic label for the replacement (e.g. `"DATE"`, `"ID"`, NER entity label for `ner_mask`).
- When `--log-replacements` is set, `run` installs:

```python
def replacement_observer(
    stage_name: str,
    original: str,
    start: int,
    end: int,
    replacement: str,
    label: str,
) -> None:
    logger.info(
        "[stage=%s] Found string %r at [%d, %d), label=%s, replace with %r",
        stage_name,
        original,
        start,
        end,
        label,
        replacement,
    )
```

- Stages that currently use this hook:
  - `normalize_ids_stage` (label `"ID"`).
  - `normalize_dates_stage` (label `"DATE"`).
  - `ner_mask_stage` (NER labels, e.g. `"PER"`, `"TIMEX"`, `"FAC"`, etc.).

This mechanism is **stage-aware and semantically precise** (no diffing), at the cost of explicit instrumentation in stages that perform replacements.

---

#### 4.3 How to add a new stage (with observers)

When you add a new pipeline stage, you usually want **two things for free**:

- It should participate in **per-stage observation** (`--observe-stages`).
- If it changes `doc.text`, it should optionally report **fine-grained replacements** (`--log-replacements`).

Thanks to `build_pipeline_with_optional_observers`, the first part is automatic as soon as you register the stage in `build_doc_pipeline`. The second part just requires calling the shared `replacement_observer` from inside your stage where you actually perform replacements. Both observers are **always defined as no-ops by default**, so stages can call them unconditionally.

**Example**: adding a toy `normalize_whitespace` stage that collapses multiple spaces into one.

```python
def normalize_whitespace_stage(
    doc: DeidDoc,
    replacement_observer: Optional[ReplacementObserver] = None,
) -> DeidDoc:
    import re

    text = doc.text
    if not isinstance(text, str) or not text:
        return doc

    stage_name = "normalize_whitespace"

    def repl(match: re.Match) -> str:
        original = match.group(0)
        replacement = " "
        if replacement != original:
            start, end = match.span()
            replacement_observer(
                stage_name,
                original,
                start,
                end,
                replacement,
                "WHITESPACE",
            )
        return replacement

    text = re.sub(r"\s{2,}", repl, text)
    doc.text = text
    return doc
```

To insert it into the pipeline (for example, as a preprocessing stage before IDs and dates), wire it into `build_doc_pipeline` alongside the other prefix stages, passing through the shared `replacement_observer`. We use a small helper `stage_with_replacements` to keep this wiring concise and data-driven:

```python
prefix_stages: list[tuple[str, Callable[[DeidDoc], DeidDoc]]] = []

optional_prefixes = [
    ("normalize_whitespace", normalize_whitespace, normalize_whitespace_stage),
    ("normalize_ids", normalize_ids, normalize_ids_stage),
    ("normalize_dates", normalize_dates, normalize_dates_stage),
]

for name, enabled, fn in optional_prefixes:
    if not enabled:
        continue
    prefix_stages.append(
        (name, stage_with_replacements(fn, replacement_observer))
    )

stages = prefix_stages + core_stages
```

After that:

- `--observe-stages` will include `stage=normalize_whitespace` automatically.
- `--log-replacements` will output lines such as:
  - `[stage=normalize_whitespace] Found string '   ' at [10, 13), label=WHITESPACE, replace with ' '`.

---

### 5. CLI usage examples (Windows / PowerShell)

Assuming current working directory is the project root (`ichilov_anonimizer`) and `uv` is available.

#### 5.1 Basic run (no observability)

```powershell
uv run .\de_identify_pipeline.py `
  --input .\dummy_data\test_deid_label_2.csv `
  --model dicta-il/dictabert-ner `
  --mode label
```

This will:
- Run the full pipeline with `normalize_ids` + `normalize_dates` + NER masking.
- Produce:
  - `dummy_data/test_deid_label_2_deid_label_pipeline.csv`
  - `dummy_data/test_deid_label_2_deid_label_pipeline.json`

#### 5.2 With per-stage summaries

```powershell
uv run .\de_identify_pipeline.py `
  --input .\dummy_data\test_deid_label_2.csv `
  --model dicta-il/dictabert-ner `
  --mode label `
  --observe-stages
```

Logs (INFO) will include lines like:

- `[stage=normalize_ids] text_len=... entities=...`
- `[stage=normalize_dates] text_len=... entities=...`
- `[stage=ner_mask] text_len=... entities=...`

Add `--verbose` to see text snippets and entity lists at DEBUG level.

#### 5.3 With detailed replacement logs

```powershell
uv run .\de_identify_pipeline.py `
  --input .\dummy_data\test_deid_label_2.csv `
  --model dicta-il/dictabert-ner `
  --mode label `
  --log-replacements
```

Example log lines:

- `[stage=normalize_ids] Found string 'ת.ז. 345678901' at [8, 20), label=ID, replace with ' ID_345678901'`.
- `[stage=normalize_dates] Found string '15.03.2024' at [32, 42), label=DATE, replace with '2024-03-15'`.
- `[stage=ner_mask] Found string 'דוד כהן' at [7, 14), label=PER, replace with '[PER]'`.

You can combine this with `--observe-stages` and `--verbose` for maximal tracing.

---

### 6. Logging policy

Global logging is configured in `main` of `de_identify_pipeline.py`:

- Default: only WARNING+ messages from libraries (`httpx`, `transformers`, etc.).
- The module logger (`__name__`) is set to `INFO` by default, or `DEBUG` when `--verbose` is enabled.

This ensures that:

- Stage / replacement logs are visible when requested.
- External library noise is suppressed unless explicitly needed.
