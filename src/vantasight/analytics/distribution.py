"""Small descriptive summaries for offline measurement series."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


def summarize_distribution(values: Sequence[float]) -> dict[str, int | float | None]:
    """Return count, min, mean, median, nearest-rank p95, and max for finite values."""

    if any(not math.isfinite(value) for value in values):
        raise ValueError("Distribution values must be finite")
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "max": ordered[-1],
    }


def summarize_body_region_responses(frames: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int | float | None]]:
    """Summarize raw pose heatmap responses by estimated body region."""
    values: dict[str, list[float]] = {}
    for frame in frames:
        for detection in frame.get("detections", []):
            for region in detection.get("body_regions", []):
                values.setdefault(region["name"], []).append(float(region["mean_keypoint_score"]))
    return {name: summarize_distribution(scores) for name, scores in sorted(values.items())}
