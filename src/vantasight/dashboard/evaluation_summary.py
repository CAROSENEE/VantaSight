"""Summaries for manual-label evaluation CSVs displayed in the local dashboard."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import statistics

from vantasight.analytics import DetectionCounts, calculate_metrics


@dataclass(frozen=True)
class DetectorEvaluationSummary:
    scored_frames: int
    counts: DetectionCounts
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class PoseRegionSummary:
    region_name: str
    annotated_points: int
    estimated_points: int
    coverage: float
    median_error_pixels: float | None
    median_error_normalized: float | None


def load_detector_evaluation(path: Path, video_name: str) -> DetectorEvaluationSummary | None:
    """Aggregate per-frame TP/FP/FN for one source video from its evaluator CSV."""

    if not path.is_file():
        return None
    totals = [0, 0, 0]
    frames = 0
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"video", "true_positives", "false_positives", "false_negatives"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Detector evaluation CSV is missing required columns: {path}")
        for row in reader:
            if row.get("video") != video_name:
                continue
            try:
                counts = [int(row[key]) for key in ("true_positives", "false_positives", "false_negatives")]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid detector counts in {path}") from exc
            if any(value < 0 for value in counts):
                raise ValueError(f"Negative detector counts in {path}")
            totals = [total + value for total, value in zip(totals, counts)]
            frames += 1
    if not frames:
        return None
    counts = DetectionCounts(*totals)
    metrics = calculate_metrics(counts)
    return DetectorEvaluationSummary(frames, counts, metrics.precision, metrics.recall, metrics.f1)


def load_pose_evaluation(path: Path, video_name: str) -> list[PoseRegionSummary] | None:
    """Aggregate per-point coverage and median localization error by region."""

    if not path.is_file():
        return None
    values: dict[str, list[tuple[float | None, float | None]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"video", "region", "error_pixels"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Pose evaluation CSV is missing required columns: {path}")
        for row in reader:
            if row.get("video") != video_name:
                continue
            name = row.get("region", "")
            if not name:
                raise ValueError(f"Empty region name in {path}")
            raw_error = row.get("error_pixels", "").strip()
            error = None
            normalized = None
            if raw_error:
                try:
                    error = float(raw_error)
                except ValueError as exc:
                    raise ValueError(f"Invalid pose error in {path}") from exc
                if not math.isfinite(error) or error < 0:
                    raise ValueError(f"Invalid pose error in {path}")
                raw_normalized = row.get("error_normalized", "").strip()
                if raw_normalized:
                    try:
                        normalized = float(raw_normalized)
                    except ValueError as exc:
                        raise ValueError(f"Invalid normalized pose error in {path}") from exc
                    if not math.isfinite(normalized) or normalized < 0:
                        raise ValueError(f"Invalid normalized pose error in {path}")
            values.setdefault(name, []).append((error, normalized))
    if not values:
        return None
    summaries: list[PoseRegionSummary] = []
    for name, rows in sorted(values.items()):
        estimated = [row[0] for row in rows if row[0] is not None]
        normalized = [row[1] for row in rows if row[1] is not None]
        summaries.append(
            PoseRegionSummary(
                region_name=name,
                annotated_points=len(rows),
                estimated_points=len(estimated),
                coverage=len(estimated) / len(rows),
                median_error_pixels=statistics.median(estimated) if estimated else None,
                median_error_normalized=statistics.median(normalized) if normalized else None,
            )
        )
    return summaries
