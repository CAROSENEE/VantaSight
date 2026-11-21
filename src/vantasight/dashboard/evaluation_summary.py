"""Summaries for manual-label evaluation CSVs displayed in the local dashboard."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
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
    device: str | None = None
    max_side: int | None = None
    preserve_source_resolution: bool | None = None
    target_height_groups: dict[str, dict[str, int | float | None]] | None = None
    source_sha256: str | None = None


@dataclass(frozen=True)
class AveragePrecisionPoint:
    iou_threshold: float
    average_precision: float | None
    mean_average_precision: float
    scored_frames: int
    positive_frames: int
    negative_frames: int
    ground_truth_boxes: int
    candidate_detections: int
    label_provenance: str
    model_architecture: str | None = None
    model_weights: str | None = None
    confidence_floor: float | None = None
    max_side: int | None = None
    device: str | None = None
    excluded_region_count: int | None = None
    ap_method: str | None = None
    preserve_source_resolution: bool | None = None
    source_sha256: str | None = None


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


def load_threshold_sweep(
    path: Path, video_name: str, expected_source_sha256: str | None = None
) -> list[ThresholdSweepPoint] | None:
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
        resize_columns = {"device", "max_side", "preserve_source_resolution"}
        has_resize_metadata = resize_columns.issubset(reader.fieldnames)
        if bool(resize_columns.intersection(reader.fieldnames)) and not has_resize_metadata:
            raise ValueError(f"Threshold sweep CSV has incomplete inference resize metadata: {path}")
        has_source_fingerprint = "source_sha256" in reader.fieldnames
        group_names = ("tiny", "small", "medium", "large")
        group_metrics = ("ground_truth_boxes", "matched_boxes", "missed_boxes", "recall")
        size_columns = {"target_height_group_basis"} | {
            f"{group}_{metric}" for group in group_names for metric in group_metrics
        }
        has_size_metrics = size_columns.issubset(reader.fieldnames)
        if bool(size_columns.intersection(reader.fieldnames)) and not has_size_metrics:
            raise ValueError(f"Threshold sweep CSV has incomplete target-height metrics: {path}")
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
                preserve_source_resolution = None
                resize_metadata = None
                if has_resize_metadata:
                    raw_resolution_mode = row["preserve_source_resolution"].strip().lower()
                    if raw_resolution_mode not in {"true", "false"}:
                        raise ValueError("preserve_source_resolution must be true or false")
                    preserve_source_resolution = raw_resolution_mode == "true"
                    resize_metadata = (row["device"].strip(), int(row["max_side"]), preserve_source_resolution)
                target_height_groups = None
                if has_size_metrics:
                    expected_basis = "tiny:<2%;small:2-6%;medium:6-16%;large:>=16% of source frame height"
                    if row["target_height_group_basis"].strip() != expected_basis:
                        raise ValueError("Unsupported target-height group basis")
                    target_height_groups = {}
                    for group in group_names:
                        group_counts = {
                            metric: int(row[f"{group}_{metric}"])
                            for metric in group_metrics[:3]
                        }
                        raw_recall = row[f"{group}_recall"].strip()
                        group_recall = float(raw_recall) if raw_recall else None
                        if (
                            min(group_counts.values()) < 0
                            or group_counts["ground_truth_boxes"]
                            != group_counts["matched_boxes"] + group_counts["missed_boxes"]
                            or (
                                group_recall is None
                                and group_counts["ground_truth_boxes"] != 0
                            )
                            or (
                                group_recall is not None
                                and (
                                    group_counts["ground_truth_boxes"] == 0
                                    or not math.isfinite(group_recall)
                                    or not 0 <= group_recall <= 1
                                    or not math.isclose(
                                        group_recall,
                                        group_counts["matched_boxes"] / group_counts["ground_truth_boxes"],
                                        abs_tol=1e-9,
                                    )
                                )
                            )
                        ):
                            raise ValueError(f"Invalid {group} target-height metrics")
                        target_height_groups[group] = {**group_counts, "recall": group_recall}
                source_sha256 = None
                if has_source_fingerprint:
                    source_sha256 = row["source_sha256"].strip()
                    if re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
                        raise ValueError("source_sha256 must be a lowercase 64-character SHA-256 digest")
                    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
                        raise ValueError("threshold sweep source fingerprint does not match the tracking report")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid threshold sweep values in {path}: {exc}") from exc
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
            if resize_metadata is not None and (not resize_metadata[0] or resize_metadata[1] < 1):
                raise ValueError(f"Invalid threshold sweep inference resize metadata in {path}")
            if target_height_groups is not None and (
                sum(group["ground_truth_boxes"] for group in target_height_groups.values()) != ground_truth_boxes
                or sum(group["matched_boxes"] for group in target_height_groups.values()) != counts[0]
                or sum(group["missed_boxes"] for group in target_height_groups.values()) != counts[2]
            ):
                raise ValueError(f"Target-height metrics do not match total counts in {path}")
            if timing is not None and (
                any(not math.isfinite(value) or value < 0 for value in (timing[0], timing[2], timing[3], timing[4]))
                or timing[1] != scored_frames
                or timing[3] <= 0
                or timing[4] <= 0
                or not math.isclose(
                    timing[4], scored_frames / timing[3], rel_tol=1e-5, abs_tol=1e-6
                )
                or timing[1] * timing[2] / 1000 > timing[3] + 0.001
            ):
                raise ValueError(f"Invalid threshold sweep timing metadata in {path}")
            contexts.add((
                scored_frames, positive_frames, negative_frames, ground_truth_boxes,
                provenance, unreviewed_frames, iou_threshold, timing, resize_metadata,
                row.get("target_height_group_basis") if has_size_metrics else None,
                source_sha256,
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
                    device=resize_metadata[0] if resize_metadata else None,
                    max_side=resize_metadata[1] if resize_metadata else None,
                    preserve_source_resolution=preserve_source_resolution,
                    target_height_groups=target_height_groups,
                    source_sha256=source_sha256,
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


def load_iou_average_precision(
    path: Path, video_name: str, expected_source_sha256: str | None = None
) -> list[AveragePrecisionPoint] | None:
    """Load 101-point AP values across IoU thresholds for one source video."""
    if not path.is_file():
        return None
    points: list[AveragePrecisionPoint] = []
    contexts: set[tuple[object, ...]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "video", "scored_frames", "positive_frames", "negative_frames", "ground_truth_boxes",
            "candidate_detections", "iou_threshold", "average_precision_101_point",
            "mean_ap_over_reported_iou_thresholds", "label_provenance", "unreviewed_scored_frames",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"IoU AP CSV is missing required columns: {path}")
        metadata_columns = {
            "model_architecture", "model_weights", "confidence_floor", "max_side", "device",
            "excluded_region_count", "ap_method",
        }
        has_metadata = metadata_columns.issubset(reader.fieldnames)
        if metadata_columns.intersection(reader.fieldnames) and not has_metadata:
            raise ValueError(f"IoU AP CSV has incomplete model metadata: {path}")
        has_resolution_metadata = "preserve_source_resolution" in reader.fieldnames
        has_source_fingerprint = "source_sha256" in reader.fieldnames
        for row in reader:
            if row.get("video") != video_name:
                continue
            try:
                scored_frames = int(row["scored_frames"])
                positive_frames = int(row["positive_frames"])
                negative_frames = int(row["negative_frames"])
                ground_truth_boxes = int(row["ground_truth_boxes"])
                candidate_detections = int(row["candidate_detections"])
                unreviewed_frames = int(row["unreviewed_scored_frames"])
                threshold = float(row["iou_threshold"])
                mean_ap = float(row["mean_ap_over_reported_iou_thresholds"])
                raw_ap = row["average_precision_101_point"].strip()
                average_precision = float(raw_ap) if raw_ap else None
                metadata = None
                if has_metadata:
                    metadata = (
                        row["model_architecture"].strip(),
                        row["model_weights"].strip(),
                        float(row["confidence_floor"]),
                        int(row["max_side"]),
                        row["device"].strip(),
                        int(row["excluded_region_count"]),
                        row["ap_method"].strip(),
                    )
                preserve_source_resolution = None
                if has_resolution_metadata:
                    raw_resolution_mode = row["preserve_source_resolution"].strip().lower()
                    if raw_resolution_mode not in {"true", "false"}:
                        raise ValueError("preserve_source_resolution must be true or false")
                    preserve_source_resolution = raw_resolution_mode == "true"
                source_sha256 = None
                if has_source_fingerprint:
                    source_sha256 = row["source_sha256"].strip()
                    if re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
                        raise ValueError("source_sha256 must be a lowercase 64-character SHA-256 digest")
                    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
                        raise ValueError("IoU AP source fingerprint does not match the tracking report")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid IoU AP values in {path}: {exc}") from exc
            provenance = row.get("label_provenance", "")
            if not math.isfinite(threshold) or not 0 < threshold <= 1:
                raise ValueError(f"Invalid IoU threshold in {path}")
            if not math.isfinite(mean_ap) or not 0 <= mean_ap <= 1:
                raise ValueError(f"Invalid mean AP in {path}")
            if average_precision is not None and (
                not math.isfinite(average_precision) or not 0 <= average_precision <= 1
            ):
                raise ValueError(f"Invalid AP value in {path}")
            if (
                scored_frames < 1
                or min(positive_frames, negative_frames, ground_truth_boxes, candidate_detections) < 0
                or positive_frames + negative_frames != scored_frames
                or ground_truth_boxes < positive_frames
                or (ground_truth_boxes > 0 and average_precision is None)
                or (ground_truth_boxes == 0 and average_precision is not None)
                or not 0 <= unreviewed_frames <= scored_frames
                or provenance not in {"manually_reviewed", "unreviewed_exploratory"}
                or (provenance == "manually_reviewed" and unreviewed_frames != 0)
            ):
                raise ValueError(f"Invalid IoU AP sample metadata in {path}")
            if metadata is not None and (
                not metadata[0]
                or not metadata[1]
                or not math.isfinite(metadata[2])
                or not 0 <= metadata[2] <= 1
                or metadata[3] < 1
                or not metadata[4]
                or metadata[5] < 0
                or metadata[6] != "101_point_interpolated_greedy_one_to_one"
            ):
                raise ValueError(f"Invalid IoU AP model metadata in {path}")
            contexts.add((
                scored_frames, positive_frames, negative_frames, ground_truth_boxes,
                candidate_detections, mean_ap, provenance, unreviewed_frames, metadata,
                preserve_source_resolution,
                source_sha256,
            ))
            points.append(AveragePrecisionPoint(
                iou_threshold=threshold,
                average_precision=average_precision,
                mean_average_precision=mean_ap,
                scored_frames=scored_frames,
                positive_frames=positive_frames,
                negative_frames=negative_frames,
                ground_truth_boxes=ground_truth_boxes,
                candidate_detections=candidate_detections,
                label_provenance=provenance,
                model_architecture=metadata[0] if metadata else None,
                model_weights=metadata[1] if metadata else None,
                confidence_floor=metadata[2] if metadata else None,
                max_side=metadata[3] if metadata else None,
                device=metadata[4] if metadata else None,
                excluded_region_count=metadata[5] if metadata else None,
                ap_method=metadata[6] if metadata else None,
                preserve_source_resolution=preserve_source_resolution,
                source_sha256=source_sha256,
            ))
    if not points:
        return None
    points.sort(key=lambda point: point.iou_threshold)
    if len({point.iou_threshold for point in points}) != len(points):
        raise ValueError(f"Duplicate IoU threshold in {path}")
    if len(contexts) != 1:
        raise ValueError(f"Inconsistent IoU AP sample or provenance in {path}")
    available_ap = [point.average_precision for point in points if point.average_precision is not None]
    calculated_mean = sum(available_ap) / len(available_ap) if available_ap else None
    if calculated_mean is None or not math.isclose(calculated_mean, points[0].mean_average_precision, abs_tol=1e-8):
        raise ValueError(f"Mean AP does not match per-IoU values in {path}")
    return points
