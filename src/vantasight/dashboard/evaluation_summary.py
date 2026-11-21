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


@dataclass(frozen=True)
class ThresholdSweepPoint:
    confidence_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None
    scored_frames: int
    positive_frames: int
    negative_frames: int
    ground_truth_boxes: int
    label_provenance: str
    iou_threshold: float
    model_initialization_seconds: float | None = None
    inference_calls: int | None = None
    detector_latency_mean_ms: float | None = None
    sampled_analysis_elapsed_seconds: float | None = None
    sampled_analysis_frames_per_second: float | None = None


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


def load_threshold_sweep(path: Path, video_name: str) -> list[ThresholdSweepPoint] | None:
    """Load and validate confidence-sweep metrics for one source video."""
    if not path.is_file():
        return None
    points: list[ThresholdSweepPoint] = []
    contexts: set[tuple[object, ...]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "video", "confidence_threshold", "scored_frames", "true_positives", "false_positives",
            "false_negatives", "precision", "recall", "f1", "label_provenance",
            "unreviewed_scored_frames", "iou_threshold", "positive_frames", "negative_frames",
            "ground_truth_boxes",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Threshold sweep CSV is missing required columns: {path}")
        timing_columns = {
            "model_initialization_seconds", "inference_calls", "detector_latency_mean_ms",
            "sampled_analysis_elapsed_seconds", "sampled_analysis_frames_per_second",
        }
        has_timing = timing_columns.issubset(reader.fieldnames)
        if bool(timing_columns.intersection(reader.fieldnames)) and not has_timing:
            raise ValueError(f"Threshold sweep CSV has incomplete timing columns: {path}")
        for row in reader:
            if row.get("video") != video_name:
                continue
            try:
                threshold = float(row["confidence_threshold"])
                scored_frames = int(row["scored_frames"])
                positive_frames = int(row["positive_frames"])
                negative_frames = int(row["negative_frames"])
                ground_truth_boxes = int(row["ground_truth_boxes"])
                counts = [int(row[key]) for key in ("true_positives", "false_positives", "false_negatives")]
                unreviewed_frames = int(row["unreviewed_scored_frames"])
                iou_threshold = float(row["iou_threshold"])
                raw_metrics = [row[name].strip() for name in ("precision", "recall", "f1")]
                metrics = [float(value) if value else None for value in raw_metrics]
                timing = None
                if has_timing:
                    timing = (
                        float(row["model_initialization_seconds"]),
                        int(row["inference_calls"]),
                        float(row["detector_latency_mean_ms"]),
                        float(row["sampled_analysis_elapsed_seconds"]),
                        float(row["sampled_analysis_frames_per_second"]),
                    )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid threshold sweep values in {path}") from exc
            provenance = row.get("label_provenance", "")
            if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError(f"Invalid confidence threshold in {path}")
            if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
                raise ValueError(f"Invalid IoU threshold in {path}")
            if (
                scored_frames < 1
                or any(value < 0 for value in counts)
                or min(positive_frames, negative_frames, ground_truth_boxes) < 0
                or positive_frames + negative_frames != scored_frames
                or ground_truth_boxes < positive_frames
            ):
                raise ValueError(f"Invalid threshold sweep counts in {path}")
            if not 0 <= unreviewed_frames <= scored_frames:
                raise ValueError(f"Invalid unreviewed frame count in {path}")
            if provenance not in {"manually_reviewed", "unreviewed_exploratory"}:
                raise ValueError(f"Invalid label provenance in {path}")
            if provenance == "manually_reviewed" and unreviewed_frames:
                raise ValueError(f"Manual threshold sweep cannot contain unreviewed frames in {path}")
            if timing is not None and (
                any(not math.isfinite(value) or value < 0 for value in (timing[0], timing[2], timing[3], timing[4]))
                or timing[1] != scored_frames
                or timing[3] <= 0
                or timing[4] <= 0
            ):
                raise ValueError(f"Invalid threshold sweep timing metadata in {path}")
            contexts.add((
                scored_frames, positive_frames, negative_frames, ground_truth_boxes,
                provenance, unreviewed_frames, iou_threshold, timing,
            ))
            expected_metrics = calculate_metrics(DetectionCounts(*counts))
            for actual, expected in zip(metrics, (expected_metrics.precision, expected_metrics.recall, expected_metrics.f1)):
                if actual is None and expected is None:
                    continue
                if actual is None or expected is None:
                    raise ValueError(f"Threshold sweep metric does not match counts in {path}")
                if not math.isfinite(actual) or not 0 <= actual <= 1 or not math.isclose(actual, expected, abs_tol=1e-9):
                    raise ValueError(f"Threshold sweep metric does not match counts in {path}")
            points.append(
                ThresholdSweepPoint(
                    confidence_threshold=threshold,
                    true_positives=counts[0],
                    false_positives=counts[1],
                    false_negatives=counts[2],
                    precision=metrics[0],
                    recall=metrics[1],
                    f1=metrics[2],
                    scored_frames=scored_frames,
                    positive_frames=positive_frames,
                    negative_frames=negative_frames,
                    ground_truth_boxes=ground_truth_boxes,
                    label_provenance=provenance,
                    iou_threshold=iou_threshold,
                    model_initialization_seconds=timing[0] if timing else None,
                    inference_calls=timing[1] if timing else None,
                    detector_latency_mean_ms=timing[2] if timing else None,
                    sampled_analysis_elapsed_seconds=timing[3] if timing else None,
                    sampled_analysis_frames_per_second=timing[4] if timing else None,
                )
            )
    if not points:
        return None
    points.sort(key=lambda point: point.confidence_threshold)
    if len({point.confidence_threshold for point in points}) != len(points):
        raise ValueError(f"Duplicate confidence threshold in {path}")
    if len(contexts) != 1:
        raise ValueError(f"Inconsistent threshold sweep sample or provenance in {path}")
    truth_count = points[0].true_positives + points[0].false_negatives
    if truth_count != points[0].ground_truth_boxes:
        raise ValueError(f"Ground-truth box total does not match TP+FN in {path}")
    for previous, current in zip(points, points[1:]):
        if current.true_positives > previous.true_positives:
            raise ValueError(f"True positives increase with confidence threshold in {path}")
        if current.false_positives > previous.false_positives:
            raise ValueError(f"False positives increase with confidence threshold in {path}")
        if current.false_negatives < previous.false_negatives:
            raise ValueError(f"False negatives decrease with confidence threshold in {path}")
        if current.true_positives + current.false_negatives != truth_count:
            raise ValueError(f"Ground-truth count changes across thresholds in {path}")
    return points
