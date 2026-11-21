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
from vantasight.analytics.pose_association import associate_pose_observations
from vantasight.analytics.pose_evaluation import (
    RegionLocalization,
    RegionLocalizationSummary,
    compare_body_region_points,
    summarize_region_localizations,
)
from vantasight.analytics.tracking_report import write_tracking_report

__all__ = [
    "AlignmentMeasurement",
    "DetectionCounts",
    "DetectionMetrics",
    "RegionLocalization",
    "RegionLocalizationSummary",
    "box_iou",
    "associate_pose_observations",
    "calculate_metrics",
    "center_crosshair",
    "compare_body_region_points",
    "match_boxes",
    "measure_alignment",
    "summarize_region_localizations",
    "write_tracking_report",
]
