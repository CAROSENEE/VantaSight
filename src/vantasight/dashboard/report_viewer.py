"""Load and validate a VantaSight tracking report for offline viewing."""

from __future__ import annotations

import json
import math
from pathlib import Path, PureWindowsPath
import re
import statistics
from typing import Any


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_summary_distribution(value: object, label: str, *, nonnegative: bool) -> int:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    count = value.get("count")
    if type(count) is not int or count < 0:
        raise ValueError(f"{label} count must be a nonnegative integer")
    for field in ("min", "mean", "median", "p95", "max"):
        number = value.get(field)
        if count == 0:
            if number is not None:
                raise ValueError(f"Empty {label} values must be null")
        elif not _finite_number(number) or (nonnegative and number < 0):
            qualifier = "finite and nonnegative" if nonnegative else "finite"
            raise ValueError(f"{label} {field} must be {qualifier}")
    return count


def load_tracking_report(path: Path) -> dict[str, Any]:
    """Read a version 1 tracking JSON report and validate required sections."""

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read tracking report {path}: {exc}") from exc
    if (
        not isinstance(report, dict)
        or type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
    ):
        raise ValueError("Unsupported or invalid tracking report schema")
    source = report.get("source")
    summary = report.get("summary")
    frames = report.get("frames")
    if not isinstance(source, dict) or not isinstance(summary, dict) or not isinstance(frames, list):
        raise ValueError("Tracking report must contain source, summary, and frames")
    file_name = source.get("file_name")
    if (
        not isinstance(file_name, str)
        or not file_name
        or Path(file_name).name != file_name
        or PureWindowsPath(file_name).name != file_name
    ):
        raise ValueError("Source file_name must be a simple file name without a path")
    for dimension in ("width", "height"):
        if type(source.get(dimension)) is not int or source[dimension] < 1:
            raise ValueError(f"Source {dimension} must be a positive integer")
    source_sha256 = source.get("source_sha256")
    if source_sha256 is not None and (
        not isinstance(source_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
    ):
        raise ValueError("Source source_sha256 must be a lowercase 64-character SHA-256 digest")
    if not _finite_number(source.get("fps")) or source["fps"] < 0:
        raise ValueError("Source fps must be a finite nonnegative number")
    for field in ("sampled_frames", "candidate_detections"):
        if type(summary.get(field)) is not int or summary[field] < 0:
            raise ValueError(f"Summary {field} must be a nonnegative integer")
    for field in (
        "sampled_frames_per_second",
        "detector_latency_mean_ms",
        "detector_latency_p95_ms",
    ):
        if not _finite_number(summary.get(field)) or summary[field] < 0:
            raise ValueError(f"Summary {field} must be a finite nonnegative number")
    for field in ("pose_latency_mean_ms", "pose_latency_p95_ms"):
        value = summary.get(field)
        if value is not None and (not _finite_number(value) or value < 0):
            raise ValueError(f"Summary {field} must be null or a finite nonnegative number")
    elapsed = summary.get("analysis_elapsed_seconds")
    if elapsed is not None and (not _finite_number(elapsed) or elapsed < 0):
        raise ValueError("Summary analysis_elapsed_seconds must be null or finite and nonnegative")
    confidence_distribution = summary.get("detector_confidence")
    if confidence_distribution is not None:
        if not isinstance(confidence_distribution, dict):
            raise ValueError("Summary detector_confidence must be an object")
        count = confidence_distribution.get("count")
        if type(count) is not int or count < 0:
            raise ValueError("Detector confidence count must be a nonnegative integer")
        for field in ("min", "mean", "median", "p95", "max"):
            value = confidence_distribution.get(field)
            if count == 0:
                if value is not None:
                    raise ValueError("Empty detector confidence distribution values must be null")
            elif not _finite_number(value) or not 0 <= value <= 1:
                raise ValueError(f"Detector confidence {field} must be between 0 and 1")
        if count != summary["candidate_detections"]:
            raise ValueError("Detector confidence count must match candidate_detections")
    region_distributions = summary.get("body_region_heatmap_response")
    if region_distributions is not None:
        if not isinstance(region_distributions, dict):
            raise ValueError("Summary body_region_heatmap_response must be an object")
        for region_name, distribution in region_distributions.items():
            if region_name not in {"head", "chest", "lower_torso", "legs"}:
                raise ValueError("Body-region response summary has an unsupported region")
            _validate_summary_distribution(
                distribution, f"Body-region {region_name} heatmap response", nonnegative=False
            )
    track_summary = summary.get("temporary_tracks")
    if track_summary is not None:
        if not isinstance(track_summary, dict):
            raise ValueError("Summary temporary_tracks must be an object")
        track_count = track_summary.get("track_count")
        single_count = track_summary.get("single_observation_tracks")
        if type(track_count) is not int or not 0 <= track_count <= summary["candidate_detections"]:
            raise ValueError("Temporary track count must fit candidate_detections")
        if type(single_count) is not int or not 0 <= single_count <= track_count:
            raise ValueError("Single-observation track count must fit track_count")
        observations_count = _validate_summary_distribution(
            track_summary.get("observations_per_track"), "Observations per track", nonnegative=True
        )
        span_count = _validate_summary_distribution(
            track_summary.get("observed_span_seconds_per_track"),
            "Observed span per track",
            nonnegative=True,
        )
        if observations_count != track_count or span_count != track_count:
            raise ValueError("Temporary track distributions must match track_count")

    previous_frame_index = -1
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get("detections"), list):
            raise ValueError("Each report frame must contain a detections list")
        if type(frame.get("frame_index")) is not int or frame["frame_index"] < 0:
            raise ValueError("Each report frame must have a nonnegative integer frame_index")
        if frame["frame_index"] <= previous_frame_index:
            raise ValueError("Report frames must be ordered by strictly increasing frame_index")
        previous_frame_index = frame["frame_index"]
        if not _finite_number(frame.get("timestamp_seconds")) or frame["timestamp_seconds"] < 0:
            raise ValueError("Each report frame must have a finite nonnegative timestamp")
        if type(frame.get("detection_count")) is not int or frame["detection_count"] != len(frame["detections"]):
            raise ValueError("detection_count must match the frame detections list")
        for detection in frame["detections"]:
            if not isinstance(detection, dict):
                raise ValueError("Each frame detection must be an object")
            confidence = detection.get("detector_score")
            if not _finite_number(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Each frame detection must have a detector_score between 0 and 1")
            if type(detection.get("track_id")) is not int or detection["track_id"] < 1:
                raise ValueError("Each frame detection must have a positive temporary track_id")
            box = detection.get("box_xyxy")
            if not isinstance(box, list) or len(box) != 4 or not all(_finite_number(value) for value in box):
                raise ValueError("Each frame detection must contain a four-coordinate box_xyxy")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= source["width"] and 0 <= y1 < y2 <= source["height"]):
                raise ValueError("Detection box coordinates must form a valid box inside the source frame")
            if "body_regions" in detection and not isinstance(detection["body_regions"], list):
                raise ValueError("Detection body_regions must be a list")
            for region in detection.get("body_regions", []):
                if not isinstance(region, dict):
                    raise ValueError("Each body region must be an object")
                region_name = region.get("name")
                if not isinstance(region_name, str) or region_name not in {"head", "chest", "lower_torso", "legs"}:
                    raise ValueError("Body-region name is not supported")
                point = region.get("center_xy")
                if not isinstance(point, list) or len(point) != 2 or not all(_finite_number(v) for v in point):
                    raise ValueError("Each body region must have a finite center_xy point")
                if not (0 <= point[0] <= source["width"] and 0 <= point[1] <= source["height"]):
                    raise ValueError("Body-region centers must lie inside the source frame")
                if not _finite_number(region.get("mean_keypoint_score")):
                    raise ValueError("Body-region mean_keypoint_score must be finite")
                alignment = region.get("alignment")
                if alignment is not None:
                    if not isinstance(alignment, dict):
                        raise ValueError("Body-region alignment must be an object")
                    if not _finite_number(alignment.get("distance_pixels")) or not _finite_number(
                        alignment.get("distance_normalized")
                    ):
                        raise ValueError("Alignment distances must be finite numbers")
                    if alignment["distance_pixels"] < 0 or alignment["distance_normalized"] < 0:
                        raise ValueError("Alignment distances must be nonnegative")
    return report


def summarize_alignment_errors(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize recorded geometric alignment errors by estimated body region."""

    errors: dict[str, list[tuple[float, float]]] = {}
    for frame in report["frames"]:
        for detection in frame["detections"]:
            for region in detection.get("body_regions", []):
                alignment = region.get("alignment")
                if alignment is None:
                    continue
                errors.setdefault(region["name"], []).append(
                    (alignment["distance_pixels"], alignment["distance_normalized"])
                )

    summaries = []
    for name, values in sorted(errors.items()):
        pixel_values = sorted(value[0] for value in values)
        normalized_values = sorted(value[1] for value in values)
        p95_index = max(0, math.ceil(0.95 * len(values)) - 1)
        summaries.append(
            {
                "region_name": name,
                "measurements": len(values),
                "median_error_pixels": statistics.median(pixel_values),
                "p95_error_pixels": pixel_values[p95_index],
                "median_error_normalized": statistics.median(normalized_values),
                "p95_error_normalized": normalized_values[p95_index],
            }
        )
    return summaries
