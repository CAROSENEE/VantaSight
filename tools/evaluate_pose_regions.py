"""Evaluate COCO body-region centers against manual offline point labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vantasight.analytics import (
    associate_pose_observations,
    compare_body_region_points,
    summarize_region_localizations,
)
from vantasight.cli import _load_excluded_regions
from vantasight.input import VideoReader
from vantasight.vision.body_region_estimator import TorchvisionBodyRegionEstimator

_REGIONS = {"head", "chest", "lower_torso", "legs"}


def _load_region_annotations(path: Path, video_name: str) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    records: list[dict] = []
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
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            if record.get("video") != video_name:
                continue
            frame_index = record.get("frame_index")
            if type(frame_index) is not int or frame_index < 0 or frame_index in seen_frames:
                raise ValueError(f"Invalid or duplicate frame index at {path}:{line_number}")
            seen_frames.add(frame_index)
            if record.get("status") != "opponent_visible":
                continue
            boxes = record.get("opponent_boxes_xyxy")
            targets = record.get("opponent_body_regions_xy", [])
            if not isinstance(boxes, list) or not isinstance(targets, list):
                raise ValueError(f"Invalid boxes or body-region labels at {path}:{line_number}")
            labeled_box_indexes: set[int] = set()
            for target in targets:
                if not isinstance(target, dict) or type(target.get("box_index")) is not int:
                    raise ValueError(f"Invalid body-region target at {path}:{line_number}")
                box_index = target["box_index"]
                if not 0 <= box_index < len(boxes):
                    raise ValueError(f"Body-region box index out of range at {path}:{line_number}")
                if box_index in labeled_box_indexes:
                    raise ValueError(f"Duplicate body-region target for box {box_index} at {path}:{line_number}")
                labeled_box_indexes.add(box_index)
                box = boxes[box_index]
                points = target.get("points")
                if (
                    not isinstance(box, list)
                    or len(box) != 4
                    or not all(type(value) is int for value in box)
                    or not isinstance(points, dict)
                    or not points
                ):
                    raise ValueError(f"Invalid body-region points at {path}:{line_number}")
                x1, y1, x2, y2 = box
                for name, point in points.items():
                    if name not in _REGIONS or not isinstance(point, list) or len(point) != 2:
                        raise ValueError(f"Invalid {name!r} point at {path}:{line_number}")
                    if not all(type(value) is int for value in point):
                        raise ValueError(f"Body-region coordinates must be integer pixels at {path}:{line_number}")
                    if not (x1 <= point[0] <= x2 and y1 <= point[1] <= y2):
                        raise ValueError(f"Body-region point {point} is outside box {box}")
            if targets:
                records.append(record)
    if not records:
        raise ValueError(
            f"No body-region point labels found for {video_name}; run the annotation tool and mark points with 1–4."
        )
    return sorted(records, key=lambda item: item["frame_index"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure experimental body-region localization against manual video point labels."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "annotations",
        type=Path,
        nargs="?",
        default=Path("data/annotations/manual_enemy_boxes.jsonl"),
    )
    parser.add_argument("--iou-threshold", type=float, default=0.10)
    parser.add_argument("--keypoint-score-threshold", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-side", type=int, default=1920)
    parser.add_argument("--regions-file", type=Path, default=Path("data/annotations/source_regions.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/pose_region_evaluation.csv"))
    args = parser.parse_args()
    if not 0.0 < args.iou_threshold <= 1.0 or args.max_side < 1 or args.keypoint_score_threshold < 0:
        parser.error("IoU threshold must be in (0,1], max-side positive, and keypoint score cutoff nonnegative")

    video = args.video.expanduser()
    try:
        records = _load_region_annotations(args.annotations.expanduser(), video.name)
        regions = _load_excluded_regions(args.regions_file.expanduser(), video.name)
        estimator = TorchvisionBodyRegionEstimator(
            device=args.device,
            max_size=args.max_side,
            min_size=min(1080, args.max_side),
            keypoint_score_threshold=args.keypoint_score_threshold,
            excluded_regions=regions,
        )
        rows: list[dict] = []
        localizations = []
        unmatched_targets = 0
        with VideoReader(video) as reader:
            if reader.info is None:
                raise RuntimeError("Could not read video metadata")
            width, height = reader.info.width, reader.info.height
            for record in records:
                if record.get("source_width") != width or record.get("source_height") != height:
                    raise ValueError(f"Annotation dimensions do not match frame {record['frame_index']}")
                frame = reader.read_frame(record["frame_index"])
                boxes: list[tuple[int, int, int, int]] = []
                for box in record["opponent_boxes_xyxy"]:
                    if not isinstance(box, list) or len(box) != 4 or not all(type(v) is int for v in box):
                        raise ValueError(f"Invalid opponent box on frame {record['frame_index']}")
                    x1, y1, x2, y2 = box
                    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                        raise ValueError(f"Opponent box outside frame on {record['frame_index']}: {box}")
                    boxes.append(tuple(box))
                observations = estimator.estimate(frame.pixels)
                matches = associate_pose_observations(boxes, observations, args.iou_threshold)
                by_region = {i: {region.name: region.center_xy for region in pose.body_regions}
                             for i, pose in matches.items()}
                for target in record["opponent_body_regions_xy"]:
                    box_index = target["box_index"]
                    estimates = by_region.get(box_index, {})
                    if box_index not in matches:
                        unmatched_targets += 1
                    expected = {name: tuple(point) for name, point in target["points"].items()}
                    comparisons = compare_body_region_points(expected, estimates, width, height)
                    localizations.extend(comparisons)
                    for result in comparisons:
                        rows.append(
                            {
                                "video": video.name,
                                "frame_index": record["frame_index"],
                                "timestamp_seconds": f"{frame.timestamp_seconds:.6f}",
                                "opponent_box_index": box_index,
                                "region": result.region_name,
                                "pose_matched": box_index in matches,
                                "ground_truth_x": result.ground_truth_xy[0],
                                "ground_truth_y": result.ground_truth_xy[1],
                                "estimate_x": "" if result.estimate_xy is None else result.estimate_xy[0],
                                "estimate_y": "" if result.estimate_xy is None else result.estimate_xy[1],
                                "error_pixels": "" if result.error_pixels is None else f"{result.error_pixels:.3f}",
                                "error_normalized": "" if result.error_normalized is None else f"{result.error_normalized:.6f}",
                                "pose_iou_threshold": args.iou_threshold,
                                "keypoint_score_threshold": args.keypoint_score_threshold,
                            }
                        )

        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "video", "frame_index", "timestamp_seconds", "opponent_box_index", "region", "pose_matched",
            "ground_truth_x", "ground_truth_y", "estimate_x", "estimate_y", "error_pixels", "error_normalized",
            "pose_iou_threshold", "keypoint_score_threshold",
        ]
        with output.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Device: {estimator.device}")
        print(f"Labeled region points: {len(localizations)}; target boxes without a matched pose: {unmatched_targets}")
        for summary in summarize_region_localizations(localizations):
            coverage = "n/a" if summary.coverage is None else f"{summary.coverage:.3f}"
            median = "n/a" if summary.median_error_pixels is None else f"{summary.median_error_pixels:.1f}px"
            print(
                f"{summary.region_name}: estimates={summary.estimated_points}/{summary.annotated_points} "
                f"coverage={coverage} median_error={median}"
            )
        print(f"Per-point CSV: {output.resolve()}")
        print("Errors describe only the manually labeled frames and this model/threshold configuration.")
        return 0
    except (FileNotFoundError, IndexError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"pose evaluation: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
