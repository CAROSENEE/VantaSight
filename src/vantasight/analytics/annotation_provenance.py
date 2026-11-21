"""Guardrails that distinguish manually reviewed labels from provisional annotations."""

from __future__ import annotations

from typing import Any


def annotation_provenance(
    record: dict[str, Any], *, allow_unreviewed: bool, source: str, line_number: int
) -> str:
    """Return label provenance or reject an unreviewed record by default."""

    manually_reviewed = record.get("reviewed_manually") is True and not record.get("review_note")
    if manually_reviewed:
        return "manually_reviewed"
    if not allow_unreviewed:
        raise ValueError(
            f"Unreviewed annotation at {source}:{line_number}; create labels with the interactive "
            "annotation tool or pass --allow-unreviewed-labels for exploratory metrics"
        )
    return "unreviewed_exploratory"
