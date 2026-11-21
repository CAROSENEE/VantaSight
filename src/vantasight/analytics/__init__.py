"""Geometric analytics for offline visual analysis."""

from vantasight.analytics.alignment import (
    AlignmentMeasurement,
    center_crosshair,
    measure_alignment,
)
from vantasight.analytics.detection_evaluation import (
    DetectionCounts,
    DetectionMetrics,
    box_iou,
    calculate_metrics,
    match_boxes,
)
from vantasight.analytics.tracking_report import write_tracking_report

__all__ = [
    "AlignmentMeasurement",
    "DetectionCounts",
    "DetectionMetrics",
    "box_iou",
    "calculate_metrics",
    "center_crosshair",
    "match_boxes",
    "measure_alignment",
    "write_tracking_report",
]
