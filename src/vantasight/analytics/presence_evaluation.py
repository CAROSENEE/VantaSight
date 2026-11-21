"""Binary presence metrics for candidate detections on labeled media frames."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PresenceCounts:
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def total(self) -> int:
        return self.true_positives + self.false_positives + self.true_negatives + self.false_negatives


@dataclass(frozen=True)
class PresenceMetrics:
    accuracy: float | None
    precision: float | None
    recall: float | None
    specificity: float | None
    f1: float | None


def calculate_presence_metrics(counts: PresenceCounts) -> PresenceMetrics:
    """Calculate binary metrics, keeping undefined empty-denominator values as None."""

    accuracy = (
        (counts.true_positives + counts.true_negatives) / counts.total
        if counts.total
        else None
    )
    precision_denominator = counts.true_positives + counts.false_positives
    recall_denominator = counts.true_positives + counts.false_negatives
    specificity_denominator = counts.true_negatives + counts.false_positives
    precision = counts.true_positives / precision_denominator if precision_denominator else None
    recall = counts.true_positives / recall_denominator if recall_denominator else None
    specificity = counts.true_negatives / specificity_denominator if specificity_denominator else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return PresenceMetrics(accuracy, precision, recall, specificity, f1)
