"""Box matching and metrics for offline candidate detector evaluation."""

from __future__ import annotations

from dataclasses import dataclass

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
