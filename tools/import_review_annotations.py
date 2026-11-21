"""Validate and copy manually reviewed browser labels into the canonical data folder."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from vantasight.analytics import (
    read_video_annotations,
    validate_annotation_coverage,
    validated_opponent_boxes,
)
from vantasight.input import VideoReader, file_sha256, verify_frame_fingerprint

_BODY_REGIONS = {"head", "chest", "lower_torso", "legs"}


def _read_shortlist(path: Path, video: Path) -> set[int]:
    video_hash = file_sha256(video)
    required = {"source_video", "source_sha256", "frame_index"}
    expected: set[int] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Shortlist must contain columns: {', '.join(sorted(required))}")
        for line_number, row in enumerate(reader, start=2):
            if row.get("source_video") != video.name:
                raise ValueError(f"Shortlist video mismatch at line {line_number}")
            if row.get("source_sha256", "").lower() != video_hash:
                raise ValueError(f"Shortlist source SHA-256 mismatch at line {line_number}")
            try:
                frame_index = int(row["frame_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid frame index at shortlist line {line_number}") from exc
            if frame_index < 0 or frame_index in expected:
                raise ValueError(f"Negative or duplicate shortlist frame index: {frame_index}")
            expected.add(frame_index)
    if not expected:
        raise ValueError(f"Shortlist contains no frames: {path}")
    return expected


def _require_single_video(path: Path, video_name: str) -> None:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict) or record.get("video") != video_name:
                raise ValueError(f"Expected only {video_name} records at {path}:{line_number}")


def _validate_region_points(record: dict[str, Any]) -> None:
    targets = record.get("opponent_body_regions_xy", [])
    if not isinstance(targets, list):
        raise ValueError(f"Invalid body-region list at frame {record['frame_index']}")
    boxes = record["opponent_boxes_xyxy"]
    seen_box_indexes: set[int] = set()
    for target in targets:
        if not isinstance(target, dict) or type(target.get("box_index")) is not int:
            raise ValueError(f"Invalid body-region target at frame {record['frame_index']}")
        box_index = target["box_index"]
        if not 0 <= box_index < len(boxes) or box_index in seen_box_indexes:
            raise ValueError(f"Invalid or duplicate body-region box index at frame {record['frame_index']}")
        seen_box_indexes.add(box_index)
        points = target.get("points")
        if not isinstance(points, dict) or not points:
            raise ValueError(f"Body-region target has no points at frame {record['frame_index']}")
        x1, y1, x2, y2 = boxes[box_index]
        for name, point in points.items():
            if name not in _BODY_REGIONS or not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"Invalid body-region point {name!r} at frame {record['frame_index']}")
            if not all(type(value) is int for value in point):
                raise ValueError(f"Body-region points must be integer pixels at frame {record['frame_index']}")
            if not (x1 <= point[0] <= x2 and y1 <= point[1] <= y2):
                raise ValueError(f"Body-region point lies outside its opponent box at frame {record['frame_index']}")


def _write_atomically(path: Path, records: list[dict[str, Any]], *, replace: bool) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"Output already exists; pass --replace to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps({key: value for key, value in record.items() if key != "_label_provenance"}, separators=(",", ":"))
        + "\n"
        for record in records
    )
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temp_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        if path.exists() and not replace:
            raise FileExistsError(f"Output already exists; pass --replace to replace it: {path}")
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate manually reviewed video labels and copy them to data/annotations."
    )
    parser.add_argument("video", type=Path, help="Original source video")
    parser.add_argument("review_jsonl", type=Path, help="Downloaded browser-review JSONL")
    parser.add_argument("--shortlist-csv", type=Path, required=True, help="Fingerprint-bound reviewed frame list")
    parser.add_argument("--output", type=Path, help="Destination; defaults to data/annotations/<video>_user_verified.jsonl")
    parser.add_argument("--replace", action="store_true", help="Replace an existing destination after validation")
    args = parser.parse_args()

    video = args.video.expanduser()
    source = args.review_jsonl.expanduser()
    shortlist = args.shortlist_csv.expanduser()
    output = args.output.expanduser() if args.output else Path("data/annotations") / f"{video.stem}_user_verified.jsonl"
    try:
        if not video.is_file():
            raise FileNotFoundError(f"Source video not found: {video}")
        if not source.is_file():
            raise FileNotFoundError(f"Review JSONL not found: {source}")
        if not shortlist.is_file():
            raise FileNotFoundError(f"Shortlist CSV not found: {shortlist}")

        _require_single_video(source, video.name)
        records = read_video_annotations(source, video.name)
        expected_frames = _read_shortlist(shortlist, video)
        validate_annotation_coverage(records, expected_frames)

        with VideoReader(video) as reader:
            if reader.info is None:
                raise RuntimeError("Could not read source-video metadata")
            width, height = reader.info.width, reader.info.height
            for record in records:
                frame_index = record["frame_index"]
                if record.get("reviewed_manually") is not True:
                    raise ValueError(f"Frame {frame_index} is not marked as manually reviewed")
                if record.get("provisional_prefill_used") is not False:
                    raise ValueError(f"Frame {frame_index} must come from a blank, non-prefilled review")
                if not isinstance(record.get("source_frame_sha256"), str):
                    raise ValueError(f"Frame {frame_index} has no source-frame fingerprint")
                frame = reader.read_frame(frame_index)
                verify_frame_fingerprint(record, frame.pixels, frame_index)
                validated_opponent_boxes(record, width, height)
                _validate_region_points(record)

        _write_atomically(output, records, replace=args.replace)
        print(f"Imported {len(records)} manually reviewed frames to {output.resolve()}")
        print("Source video, shortlist coverage, frame fingerprints, boxes, and optional body points were validated.")
        return 0
    except (FileNotFoundError, FileExistsError, IndexError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"annotation import: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
