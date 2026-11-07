import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from toolz import compose_left


Stage = Callable[[Any], Any]
Observer = Callable[[str, Any], None]


@dataclass
class DeidDoc:
    """
    Shared document container passed between pipeline stages.

    - text:  current working text (may be progressively modified)
    - entities: structured entity list collected by NER / other detectors
    """

    text: str
    entities: List[Dict[str, Any]] = field(default_factory=list)


def build_pipeline_with_optional_observers(
    stages: Sequence[Tuple[str, Stage]],
    observer: Optional[Observer] = None,
) -> Stage:
    """
    Build a left-to-right pipeline of stages, optionally inserting
    observer taps after each stage.

    - `stages`: sequence of (name, stage_fn) pairs.
    - `observer`: called as `observer(name, value)` after each stage,
      with the stage's name and its output value. If None, a no-op
      observer is used.

    The returned callable is equivalent to:
        compose_left(stage_n, ..., stage_2, stage_1)
    optionally interleaved with tap functions between stages.
    """
    if not stages:
        return lambda x: x

    # Normalize observer to a no-op so that downstream code can always
    # assume it is callable.
    if observer is None:

        def observer(_name: str, _value: Any) -> None:  # type: ignore[no-redef]
            return None

    pipeline_parts: List[Stage] = []

    for name, stage in stages:
        pipeline_parts.append(stage)

        def make_tap(label: str) -> Stage:
            def tap(x: Any) -> Any:
                observer(label, x)
                return x

            return tap

        pipeline_parts.append(make_tap(name))

    # Stages (and their taps) are applied left-to-right.
    return compose_left(*pipeline_parts)


def make_stage_observer(
    logger: logging.Logger,
    enabled: bool,
) -> Observer:
    """
    Factory for a standard per-stage observer used by the CLI.

    - Logs a short INFO-level summary (text length and entity count).
    - When the logger is in DEBUG, also logs a 200-character snippet and
      the full entities list.
    """

    def observer(name: str, doc: Any) -> None:
        if not enabled:
            return

        text = getattr(doc, "text", "")
        entities = getattr(doc, "entities", [])

        logger.info(
            "[S] [stage=%s][after] text_len_after=%d entities_after=%d",
            name,
            len(text),
            len(entities),
        )

        if logger.isEnabledFor(logging.DEBUG):
            snippet = str(text)[:200].replace("\n", " ")
            logger.debug(
                "[S] [stage=%s][after] text_snippet_after=%r entities_after=%s",
                name,
                snippet,
                entities,
            )

    return observer


def make_replacement_observer(
    logger: logging.Logger,
    enabled: bool,
) -> Callable[[str, str, int, int, str, str], None]:
    """
    Factory for a standard replacement observer used by the CLI.

    Logs each replacement with:
        stage name, original substring, [start, end) span, label, replacement text.
    """

    def replacement_observer(
        stage_name: str,
        original: str,
        start: int,
        end: int,
        replacement: str,
        label: str,
    ) -> None:
        if not enabled:
            return

        logger.info(
            " [R][stage=%s] original=%r span=[%d,%d) label=%s replacement=%r",
            stage_name,
            original,
            start,
            end,
            label,
            replacement,
        )

    return replacement_observer


def make_pipeline_start_stage(
    logger: logging.Logger,
) -> Callable[[DeidDoc], DeidDoc]:
    """
    Factory for a simple "pipeline start" stage that logs a visual
    separator at the beginning of processing a new record and then
    returns the document unchanged.
    """

    def pipeline_start_stage(doc: DeidDoc) -> DeidDoc:
        logger.info("========== PROCESSING NEW DOCUMENT ==========")
        return doc

    return pipeline_start_stage

