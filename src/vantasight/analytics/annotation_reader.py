"""Read and validate manually labeled video opponent annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vantasight.analytics.annotation_provenance import annotation_provenance
from vantasight.analytics.detection_evaluation import Box


def read_video_annotations(
    path: Path, video_name: str, *, allow_unreviewed: bool = False
) -> list[dict[str, Any]]:
    """Load one video's JSONL annotations and enforce provenance and schema rules."""
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    records: list[dict[str, Any]] = []
    seen_frames: set[int] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            if record.get("video") != video_name:
                continue
            provenance = annotation_provenance(
                record, allow_unreviewed=allow_unreviewed, source=str(path), line_number=line_number
            )

            frame_index = record.get("frame_index")
            if type(frame_index) is not int or frame_index < 0:
                raise ValueError(f"Invalid frame_index at {path}:{line_number}")
            if frame_index in seen_frames:
                raise ValueError(f"Duplicate annotation for frame {frame_index} in {path}")
            seen_frames.add(frame_index)
            record["_label_provenance"] = provenance

            status = record.get("status")
            if status not in {"opponent_visible", "no_opponent_visible", "uncertain"}:
                raise ValueError(f"Invalid status at {path}:{line_number}: {status!r}")
            boxes = record.get("opponent_boxes_xyxy")
            if not isinstance(boxes, list):
                raise ValueError(f"Expected opponent_boxes_xyxy list at {path}:{line_number}")
            if status == "opponent_visible" and not boxes:
                raise ValueError(f"Visible-opponent frame {frame_index} has no boxes")
            if status == "no_opponent_visible" and boxes:
                raise ValueError(f"Negative frame {frame_index} must not contain boxes")
            records.append(record)

    if not records:
        raise ValueError(
            f"No annotations for {video_name} in {path}. First run the annotation tool; "
            "see the M4 instructions in docs/WORKFLOW.md."
        )
    return sorted(records, key=lambda item: item["frame_index"])


def validated_opponent_boxes(
    record: dict[str, Any], width: int, height: int
) -> list[Box]:
    """Return validated in-frame opponent boxes for one annotation record."""
    source_width, source_height = record.get("source_width"), record.get("source_height")
    if source_width != width or source_height != height:
        raise ValueError(
            f"Annotation size {source_width}x{source_height} does not match video {width}x{height} "
            f"at frame {record['frame_index']}"
        )

    boxes: list[Box] = []
    for index, box in enumerate(record["opponent_boxes_xyxy"]):
        if not isinstance(box, list) or len(box) != 4 or not all(type(value) is int for value in box):
            raise ValueError(f"Invalid box {index} at frame {record['frame_index']}")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Out-of-frame box {box} at frame {record['frame_index']}")
        boxes.append((x1, y1, x2, y2))
    return boxes
