"""Load and validate a VantaSight tracking report for offline viewing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_tracking_report(path: Path) -> dict[str, Any]:
    """Read a version 1 tracking JSON report and validate required sections."""

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read tracking report {path}: {exc}") from exc
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("Unsupported or invalid tracking report schema")
    source = report.get("source")
    summary = report.get("summary")
    frames = report.get("frames")
    if not isinstance(source, dict) or not isinstance(summary, dict) or not isinstance(frames, list):
        raise ValueError("Tracking report must contain source, summary, and frames")
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get("detections"), list):
            raise ValueError("Each report frame must contain a detections list")
        for detection in frame["detections"]:
            if not isinstance(detection, dict):
                raise ValueError("Each frame detection must be an object")
            box = detection.get("box_xyxy")
            if not isinstance(box, list) or len(box) != 4 or not all(
                isinstance(value, (int, float)) for value in box
            ):
                raise ValueError("Each frame detection must contain a four-coordinate box_xyxy")
            if "body_regions" in detection and not isinstance(detection["body_regions"], list):
                raise ValueError("Detection body_regions must be a list")
    return report
