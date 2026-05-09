## Developer notes – de-identification pipeline

This document supplements `README.md` and `docs/Deid_Pipeline_Usage.MD` with a concise, developer‑oriented view of the pipeline design and extension points.

---

### 1. High-level architecture

- **Core document type**: `DeidDoc`
  - Small dataclass holding the current working `text` and accumulated `entities`.
  - Gives every stage the same simple signature (`DeidDoc -> DeidDoc`), which makes pipelines easy to compose, test, and extend.
- **Entry point**: `de_identify.py`
  - CLI, argument parsing, logging configuration.
  - Loads data, model, builds a document pipeline, saves CSV/JSON outputs.
- **Pipeline wiring**: `utils/doc_pipeline.py`
  - `build_doc_pipeline(ner_pipe, mode, observer, replacement_observer, normalize_dates, normalize_ids)` returns a `DeidDoc -> DeidDoc` callable.
  - Composes named stages into a linear pipeline and plugs in the per‑stage observer.
- **Stage implementations**: `utils/pipeline_stages.py`
  - Regex-based stages: `normalize_ids_stage`, `normalize_dates_stage`.
  - Network-based stage: `ner_mask_stage` (HF pipeline).
  - Post-processing: `label_clean_stage`, plus helper `clean_label`.
- **Infrastructure**: `utils/pipeline_utils.py`
  - `DeidDoc` dataclass and `build_pipeline_with_optional_observers`, which wraps stages with an observer tap while preserving left‑to‑right semantics.

---

### 2. Pipeline shape and ordering

The default document pipeline (see `build_doc_pipeline`) is:

1. `pipeline_start` (logging-only stage when observers are enabled; marks the start of a new record's pipeline run)
2. `normalize_ids_stage` (optional, enabled by default in `run_processing_loop`)
3. `normalize_dates_stage` (optional, enabled by default)
4. `ner_mask_stage` (NER over sentences, builds `doc.text` and `doc.entities`)
5. `label_clean_stage` (normalizes BIO-style labels)
6. `pipeline_end_stage` (no-op logical end marker for the pipeline)

Stages are named and passed as `(name, fn)` pairs into `build_pipeline_with_optional_observers`, which inserts a tap after each stage when an observer is provided.

---

### 3. Observability hooks

**Motivation**: we want rich visibility into *what each stage is doing* (and where it changed the text) without hard‑coding logging into every function or maintaining multiple ad‑hoc debug versions of the pipeline.

To achieve this, two orthogonal hooks are available to all stages:

- **Stage observer** (`observer(name, doc)`):
  - Conceptually: a **coarse, per-stage “stethoscope”** over the evolving `DeidDoc`.
  - What it sees:
    - Exactly **one snapshot per stage**: the full `DeidDoc` *after* that stage has finished.
    - It does not know which specific characters changed, only the overall result.
  - When it runs:
    - Injected automatically by `build_pipeline_with_optional_observers`.
    - Called for **every stage**, including ones that do not modify the text.
  - Typical uses:
    - Track how text length and entity count evolve across stages.
    - Inspect snippets and entity lists at different points in the pipeline (e.g. “what did NER see after normalization?”).
    - Compare different pipeline configurations in a reproducible way.
  - Implementation:
    - Installed in `de_identify.py` as `stage_observer`.
    - Wired by `build_doc_pipeline` / `build_pipeline_with_optional_observers`.
    - Activated from the CLI with `--observe-stages` (plus `--verbose` for DEBUG‑level detail).
    - Logs are prefixed with `[S]` (for “stage”), for example:
      - `[S] [stage=ner_mask][after] text_len_after=... entities_after=...`

- **Replacement observer** (`ReplacementObserver` type alias):
  - Conceptually: a **fine-grained change log** for any stage that edits the document text (`DeidDoc.text`).
  - What it sees:
    - **Each individual replacement** a stage applies: original substring, `[start, end)` offsets, label, replacement string.
    - A single stage can trigger this hook many times for one document.
  - When it runs:
    - Called explicitly **inside** stages at the exact point where a replacement is applied.
    - Only used by stages that actually edit text (`normalize_ids_stage`, `normalize_dates_stage`, `ner_mask_stage`).
  - Typical uses:
    - Audit which substrings were replaced, with what labels, and at which character offsets.
    - Debug mismatch between expected vs. actual masking (e.g. wrong span or label).
    - Build external analytics around replacement patterns without modifying stage logic.
  - Implementation:
    - Installed in `de_identify.py` as `replacement_observer`.
    - Passed into stages via `build_doc_pipeline` (e.g. `normalize_ids_stage`, `normalize_dates_stage`, `ner_mask_stage`).
    - Activated from the CLI with `--log-replacements`.
    - Logs are prefixed with an indented `[R]` (for “replacement”), for example:
      - ` [R][stage=ner_mask] original='יעל כץ' span=[0,6) label=PER replacement='[PER]'`

Both observers are always defined as callables; they become no‑ops when the corresponding flag is off, so stages can call them unconditionally.

---

### 4. Example: observe stages without per-replacement logs
When you want to see how each stage affects the document **in aggregate** (length and entity count), but **do not** want a line per replacement, run:
```bash
uv run .\de_identify.py `
  --input .\dummy_data\test_deid_label_2.csv `
  --model dicta-il/dictabert-ner `
  --mode label `
  --observe-stages
```
This will:
- Enable the stage observer (`[S]` lines like `[S] [stage=ner_mask       ][after] text_len_after=... entities_after=...`).
- **Not** enable the replacement observer (no `[R]` lines), so you only see the net effect of each stage on length and entity count.
Add `--verbose` if you also want the per-stage snippets and full entity lists.
---

### 5. Adding a new stage
To add a new pipeline stage that can be reused across tools:

1. **Implement the stage** in `pipeline_stages.py`, following one of these patterns:
   - Pure stage: `def my_stage(doc: DeidDoc) -> DeidDoc: ...`
   - Replacement‑reporting stage: `def my_stage(doc: DeidDoc, replacement_observer: Optional[ReplacementObserver] = None) -> DeidDoc: ...`
     - Call `replacement_observer(...)` whenever you apply a replacement (it may be `None`, so check before calling).
2. **Register the stage in the pipeline** in `utils/doc_pipeline.py`:
   - For a simple stage, add it to `core_stages` or `prefix_stages` as `(name, fn)`.
   - For a replacement‑reporting stage, route it through `stage_with_replacements` so it receives the shared `replacement_observer`.
3. **Optionally expose toggles**:
   - Add boolean flags to `build_doc_pipeline` (similar to `normalize_ids` / `normalize_dates`) and/or to the CLI if you want to enable/disable the stage from the command line.

After this, the stage will automatically:

- Participate in `--observe-stages` logging.
- Emit fine‑grained replacement logs when `--log-replacements` is used.

---

### 6. Using the pipeline programmatically

You can bypass the CLI and reuse the pipeline from your own code:

```python
from utils.nlp_utils import get_pipeline, setup_nltk
from utils.pipeline_utils import DeidDoc
from utils.doc_pipeline import build_doc_pipeline

setup_nltk()
ner_pipe = get_pipeline("HeMed_NER_baseline")

doc_pipeline = build_doc_pipeline(
    ner_pipe,
    mode="label",
    observer=None,
    replacement_observer=None,
    normalize_dates=True,
    normalize_ids=True,
)

doc = DeidDoc(text="Some Hebrew medical text ...")
result = doc_pipeline(doc)
print(result.text, result.entities)
```

For concrete CLI commands and logging combinations, use this alongside `docs/Deid_Pipeline_Usage.MD`.

