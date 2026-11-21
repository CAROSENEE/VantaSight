"""Box matching and metrics for offline candidate detector evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import math
from typing import Any

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectionCounts:
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class DetectionMetrics:
    precision: float | None
    recall: float | None
    f1: float | None


def box_iou(first: Box, second: Box) -> float:
    """Calculate intersection over union for xyxy boxes."""

    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(predictions: list[Box], ground_truth: list[Box], iou_threshold: float) -> DetectionCounts:
    """Find a maximum-cardinality one-to-one match above the IoU threshold."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    adjacency: list[list[int]] = []
    for prediction in predictions:
        candidates = [
            (box_iou(prediction, truth), truth_index)
            for truth_index, truth in enumerate(ground_truth)
        ]
        adjacency.append(
            [truth_index for overlap, truth_index in sorted(candidates, reverse=True)
             if overlap >= iou_threshold]
        )
    truth_to_prediction: dict[int, int] = {}

    def assign(prediction_index: int, visited_truths: set[int]) -> bool:
        for truth_index in adjacency[prediction_index]:
            if truth_index in visited_truths:
                continue
            visited_truths.add(truth_index)
            previous = truth_to_prediction.get(truth_index)
            if previous is None or assign(previous, visited_truths):
                truth_to_prediction[truth_index] = prediction_index
                return True
        return False

    for prediction_index in range(len(predictions)):
        assign(prediction_index, set())

    true_positives = len(truth_to_prediction)
    return DetectionCounts(
        true_positives=true_positives,
        false_positives=len(predictions) - true_positives,
        false_negatives=len(ground_truth) - true_positives,
    )


def calculate_metrics(counts: DetectionCounts) -> DetectionMetrics:
    """Calculate precision, recall, and F1; undefined empty denominators stay None."""

    precision_denominator = counts.true_positives + counts.false_positives
    recall_denominator = counts.true_positives + counts.false_negatives
    precision = counts.true_positives / precision_denominator if precision_denominator else None
    recall = counts.true_positives / recall_denominator if recall_denominator else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return DetectionMetrics(precision=precision, recall=recall, f1=f1)


def evaluate_threshold_sweep(
    samples: Sequence[tuple[Sequence[tuple[float, Box]], Sequence[Box]]],
    thresholds: Sequence[float],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Evaluate confidence cutoffs over cached per-frame predictions."""
    if not thresholds or any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError("thresholds must be a nonempty sequence between 0 and 1")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("thresholds must not repeat")
    results: list[dict[str, Any]] = []
    for threshold in thresholds:
        total_tp = total_fp = total_fn = 0
        for scored_predictions, ground_truth in samples:
            boxes = [box for score, box in scored_predictions if score >= threshold]
            counts = match_boxes(boxes, list(ground_truth), iou_threshold)
            total_tp += counts.true_positives
            total_fp += counts.false_positives
            total_fn += counts.false_negatives
        counts = DetectionCounts(total_tp, total_fp, total_fn)
        metrics = calculate_metrics(counts)
        results.append(
            {
                "confidence_threshold": threshold,
                "true_positives": total_tp,
                "false_positives": total_fp,
                "false_negatives": total_fn,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
            }
        )
    return results


def evaluate_average_precision(
    samples: Sequence[tuple[Sequence[tuple[float, Box]], Sequence[Box]]],
    iou_threshold: float,
) -> float | None:
    """Calculate 101-point interpolated AP using confidence-ranked detections.

    A prediction is matched greedily to the highest-IoU unmatched ground-truth
    box in its own frame. Returns None when the sample has no ground-truth boxes.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be greater than 0 and at most 1")
    ground_truth_total = sum(len(ground_truth) for _, ground_truth in samples)
    if ground_truth_total == 0:
        return None

    ranked_predictions: list[tuple[float, int, int, Box]] = []
    for sample_index, (predictions, _) in enumerate(samples):
        for prediction_index, (score, box) in enumerate(predictions):
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("prediction confidence scores must be finite and between 0 and 1")
            ranked_predictions.append((score, sample_index, prediction_index, box))
    ranked_predictions.sort(key=lambda item: (-item[0], item[1], item[2]))

    matched_by_frame: dict[int, set[int]] = {}
    true_positives: list[int] = []
    false_positives: list[int] = []
    for _, sample_index, _, prediction in ranked_predictions:
        ground_truth = samples[sample_index][1]
        matched = matched_by_frame.setdefault(sample_index, set())
        candidates = [
            (box_iou(prediction, truth), truth_index)
            for truth_index, truth in enumerate(ground_truth)
            if truth_index not in matched
        ]
        best = max(candidates, default=(0.0, -1))
        if best[0] >= iou_threshold:
            matched.add(best[1])
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    cumulative_tp = cumulative_fp = 0
    precision_recall: list[tuple[float, float]] = []
    for true_positive, false_positive in zip(true_positives, false_positives):
        cumulative_tp += true_positive
        cumulative_fp += false_positive
        precision_recall.append((
            cumulative_tp / ground_truth_total,
            cumulative_tp / (cumulative_tp + cumulative_fp),
        ))
    return sum(
        max((precision for recall, precision in precision_recall if recall >= target_recall), default=0.0)
        for target_recall in (index / 100 for index in range(101))
    ) / 101
