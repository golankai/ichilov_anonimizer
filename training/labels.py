"""Label schema for the NER model.

Source annotations come from an annotation tool that exports some labels with
backslash-escaped underscores (e.g. ``AGE\\_ABOVE\\_89``). We strip backslashes
on load so the schema below is the single clean spelling. The inference side
(`inference/utils/pipeline_stages.py::clean_label`) tolerates both spellings.
"""

from __future__ import annotations

# The entities the production model is trained to mask.
MANDATORY_LABELS = [
    "FIRST_NAME",
    "LAST_NAME",
    "DATE",
    "AGE_ABOVE_89",
    "LOC",
    "ORG",
    "MED_ORG",
    "CONTACT",
    "ID",
    "INNER_ID",
]

# Fine-grained date labels collapsed into DATE.
LABEL_MAPPING = {
    "DAY": "DATE",
    "MONTH": "DATE",
    "YEAR": "DATE",
}


def clean_label(label: str) -> str:
    """Remove annotation-tool escaping and apply `LABEL_MAPPING`."""
    label = label.replace("\\", "")
    return LABEL_MAPPING.get(label, label)


def build_label_maps(labels: list[str] | None = None):
    """Return ``(all_labels, label2id, id2label)`` for a BIO scheme over `labels`."""
    labels = labels if labels is not None else MANDATORY_LABELS
    all_labels = ["O"] + [f"{tag}-{label}" for label in labels for tag in ("B", "I")]
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}
    return all_labels, label2id, id2label


ALL_LABELS, LABEL_2_ID, ID_2_LABEL = build_label_maps()
