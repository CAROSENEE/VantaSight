"""Geometric analytics for offline visual analysis."""

from vantasight.analytics.alignment import (
    AlignmentMeasurement,
    center_crosshair,
    measure_alignment,
)
from vantasight.analytics.annotation_provenance import annotation_provenance
from vantasight.analytics.annotation_coverage import parse_frame_indexes, validate_annotation_coverage
from vantasight.analytics.detection_evaluation import (
    DetectionCounts,
    DetectionMetrics,
    box_iou,
    calculate_metrics,
    match_boxes,
)
from vantasight.analytics.distribution import (
    summarize_body_region_responses,
    summarize_distribution,
    summarize_temporary_tracks,
)
from vantasight.analytics.pose_association import associate_pose_observations
from vantasight.analytics.presence_evaluation import (
    PresenceCounts,
    PresenceMetrics,
    calculate_presence_metrics,
)
from vantasight.analytics.pose_evaluation import (
    RegionLocalization,
    RegionLocalizationSummary,
    compare_body_region_points,
    summarize_region_localizations,
)
from vantasight.analytics.tracking_report import write_tracking_report

__all__ = [
    "AlignmentMeasurement",
    "annotation_provenance",
    "parse_frame_indexes",
    "validate_annotation_coverage",
    "DetectionCounts",
    "DetectionMetrics",
    "RegionLocalization",
    "RegionLocalizationSummary",
    "PresenceCounts",
    "PresenceMetrics",
    "box_iou",
    "associate_pose_observations",
    "calculate_metrics",
    "calculate_presence_metrics",
    "center_crosshair",
    "compare_body_region_points",
    "match_boxes",
    "measure_alignment",
    "summarize_region_localizations",
    "summarize_distribution",
    "summarize_body_region_responses",
    "summarize_temporary_tracks",
    "write_tracking_report",
]
