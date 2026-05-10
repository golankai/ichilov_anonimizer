from functools import partial
from typing import Any, Callable, List, Optional, Tuple

from utils.pipeline_utils import DeidDoc, build_pipeline_with_optional_observers
from utils.pipeline_stages import (
    ReplacementObserver,
    normalize_dates_stage,
    normalize_ids_stage,
    ner_mask_stage,
    label_clean_stage,
    pipeline_end_stage,
)


Stage = Callable[[DeidDoc], DeidDoc]


def stage_with_replacements(
    fn: Callable[[DeidDoc, ReplacementObserver], DeidDoc],
    replacement_observer: Optional[ReplacementObserver],
) -> Stage:
    """
    Helper to bind the shared `replacement_observer` into a stage that
    accepts (doc, replacement_observer), returning a simple
    `DeidDoc -> DeidDoc` callable suitable for the pipeline builder.
    """
    return partial(fn, replacement_observer=replacement_observer)


def build_doc_pipeline(
    ner_pipe: Any,
    mode: str,
    observer: Optional[Callable[[str, DeidDoc], None]] = None,
    replacement_observer: Optional[ReplacementObserver] = None,
    normalize_dates: bool = False,
    normalize_ids: bool = False,
    header_stage: Optional[Stage] = None,
) -> Stage:
    """
    Builds a document-level pipeline.

    If `observer` is provided, it will be invoked after each stage with
    the stage name and the current `DeidDoc` instance.

    If `replacement_observer` is provided, it will be called from the
    NER masking stage for each replacement applied.

    If `header_stage` is provided, it is inserted as the first stage
    in the pipeline. This is useful for logging a document header or
    other per-document setup without affecting the rest of the stages.
    """

    # Compute a common width for stage names so that logs from both
    # observers can align `[stage=...]` nicely in the output.
    stage_names: List[str] = []
    if header_stage is not None:
        stage_names.append("pipeline_start")
    if normalize_ids:
        stage_names.append("normalize_ids")
    if normalize_dates:
        stage_names.append("normalize_dates")
    stage_names.extend(["pipeline_start", "normalize_ids", "normalize_dates", "ner_mask", "label_clean", "pipeline_end"])
    name_width = max((len(n) for n in stage_names), default=0)

    # Wrap observers to pad stage names for aligned logging.
    padded_observer: Optional[Callable[[str, DeidDoc], None]]
    if observer is not None and name_width > 0:

        def padded_observer(name: str, doc: DeidDoc) -> None:
            observer(name.ljust(name_width), doc)

    else:
        padded_observer = observer

    padded_replacement: Optional[ReplacementObserver]
    if replacement_observer is not None and name_width > 0:

        def padded_replacement(
            stage_name: str,
            original: str,
            start: int,
            end: int,
            replacement: str,
            label: str,
        ) -> None:
            replacement_observer(
                stage_name.ljust(name_width),
                original,
                start,
                end,
                replacement,
                label,
            )

    else:
        padded_replacement = replacement_observer

    ner_stage = ner_mask_stage(
        mode=mode,
        ner_pipe=ner_pipe,
        replacement_observer=padded_replacement,
    )

    core_stages: List[Tuple[str, Stage]] = [
        ("ner_mask", ner_stage),
        ("label_clean", label_clean_stage),
        ("pipeline_end", pipeline_end_stage),
    ]

    prefix_stages: List[Tuple[str, Stage]] = []

    # Register optional prefix stages in a compact, data-driven way.
    optional_prefixes = [
        ("normalize_ids", normalize_ids, normalize_ids_stage),
        ("normalize_dates", normalize_dates, normalize_dates_stage),
    ]

    for name, enabled, fn in optional_prefixes:
        if not enabled:
            continue
        prefix_stages.append(
            (
                name,
                stage_with_replacements(fn, padded_replacement),
            )
        )

    stages: List[Tuple[str, Stage]] = []

    if header_stage is not None:
        stages.append(("pipeline_start", header_stage))

    stages.extend(prefix_stages)
    stages.extend(core_stages)

    return build_pipeline_with_optional_observers(stages, observer=padded_observer)

