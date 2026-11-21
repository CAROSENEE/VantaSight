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


def summarize_temporary_tracks(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe observed temporary track IDs without asserting identity or continuity."""
    observations: dict[int, list[float]] = {}
    for frame in frames:
        timestamp = float(frame["timestamp_seconds"])
        for detection in frame.get("detections", []):
            observations.setdefault(int(detection["track_id"]), []).append(timestamp)
    counts = [len(times) for times in observations.values()]
    spans = [max(times) - min(times) for times in observations.values()]
    return {
        "track_count": len(observations),
        "single_observation_tracks": sum(count == 1 for count in counts),
        "observations_per_track": summarize_distribution(counts),
        "observed_span_seconds_per_track": summarize_distribution(spans),
    }


def summarize_zero_candidate_runs(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe adjacent sampled frames with no model candidates, not enemy absence."""
    runs: list[list[Mapping[str, Any]]] = []
    current_run: list[Mapping[str, Any]] = []
    for frame in frames:
        if frame.get("detections"):
            if current_run:
                runs.append(current_run)
                current_run = []
        else:
            current_run.append(frame)
    if current_run:
        runs.append(current_run)

    longest = max(runs, key=lambda run: (len(run), -int(run[0]["frame_index"])), default=None)
    zero_count = sum(len(run) for run in runs)
    return {
        "sampled_frame_count": len(frames),
        "zero_candidate_sample_count": zero_count,
        "zero_candidate_sample_fraction": zero_count / len(frames) if frames else None,
        "zero_candidate_run_count": len(runs),
        "longest_zero_candidate_run": (
            {
                "sample_count": len(longest),
                "start_frame_index": int(longest[0]["frame_index"]),
                "end_frame_index": int(longest[-1]["frame_index"]),
                "start_timestamp_seconds": float(longest[0]["timestamp_seconds"]),
                "end_timestamp_seconds": float(longest[-1]["timestamp_seconds"]),
            }
            if longest
            else None
        ),
    }
