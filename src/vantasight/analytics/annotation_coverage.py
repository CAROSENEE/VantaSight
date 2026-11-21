"""Coverage checks for planned manual annotation samples."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from typing import Any


def parse_frame_indexes(value: str) -> tuple[int, ...]:
    """Parse unique nonnegative comma-separated source frame indexes."""
    try:
        frames = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated frame indexes, e.g. 0,600,1200") from exc
    if not frames or any(frame < 0 for frame in frames):
        raise argparse.ArgumentTypeError("Expected frame indexes must be zero or greater")
    if len(set(frames)) != len(frames):
        raise argparse.ArgumentTypeError("Expected frame indexes must not repeat")
    return frames


def validate_annotation_coverage(
    records: Sequence[dict[str, Any]], expected_frames: Iterable[int]
) -> None:
    """Require annotation records to cover exactly the expected source frames."""
    expected = set(expected_frames)
    observed = {record["frame_index"] for record in records}
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing frames: {missing}")
        if unexpected:
            details.append(f"unexpected frames: {unexpected}")
        raise ValueError("Annotation frame coverage mismatch (" + "; ".join(details) + ")")
