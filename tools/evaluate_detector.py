"""Evaluate offline person candidates against manually labeled opponent boxes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vantasight.analytics import (
    DetectionCounts,
    annotation_provenance,
    calculate_metrics,
    match_boxes,
    parse_frame_indexes,
    validate_annotation_coverage,
)
from vantasight.cli import _load_excluded_regions
from vantasight.input import VideoReader
from vantasight.vision import TorchvisionPersonDetector


def _read_annotations(path: Path, video_name: str, *, allow_unreviewed: bool = False) -> list[dict]:
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


def _validated_boxes(record: dict, width: int, height: int) -> list[tuple[int, int, int, int]]:
    source_width, source_height = record.get("source_width"), record.get("source_height")
    if source_width != width or source_height != height:
        raise ValueError(
            f"Annotation size {source_width}x{source_height} does not match video {width}x{height} "
            f"at frame {record['frame_index']}"
        )

    boxes: list[tuple[int, int, int, int]] = []
    for index, box in enumerate(record["opponent_boxes_xyxy"]):
        if not isinstance(box, list) or len(box) != 4 or not all(type(value) is int for value in box):
            raise ValueError(f"Invalid box {index} at frame {record['frame_index']}")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Out-of-frame box {box} at frame {record['frame_index']}")
        boxes.append((x1, y1, x2, y2))
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure offline person-candidate boxes against manual opponent annotations."
    )
    parser.add_argument("video", type=Path, help="Recorded video used to make the annotations")
    parser.add_argument(
        "annotations",
        type=Path,
        nargs="?",
        default=Path("data/annotations/manual_enemy_boxes.jsonl"),
        help="JSON Lines from the annotation tool",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument(
        "--expected-frames",
        type=parse_frame_indexes,
        help="Require exactly these comma-separated annotated source frame indexes",
    )
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, or cuda:0")
    parser.add_argument("--max-side", type=int, default=1920)
    parser.add_argument(
        "--allow-unreviewed-labels",
        action="store_true",
        help="Allow provisional/legacy labels for exploratory metrics and mark their provenance in the CSV",
    )
    parser.add_argument(
        "--regions-file",
        type=Path,
        default=Path("data/annotations/source_regions.json"),
        help="Optional per-source exclusion regions JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/detector_evaluation.csv"),
        help="Per-frame metric CSV output",
    )
    args = parser.parse_args()

    if not 0.0 < args.iou_threshold <= 1.0:
        parser.error("--iou-threshold must be greater than 0 and at most 1")
    if args.max_side < 1:
        parser.error("--max-side must be positive")

    video = args.video.expanduser()
    annotation_path = args.annotations.expanduser()
    output = args.output.expanduser()
    try:
        records = _read_annotations(annotation_path, video.name, allow_unreviewed=args.allow_unreviewed_labels)
        if args.expected_frames is not None:
            validate_annotation_coverage(records, args.expected_frames)
        regions = _load_excluded_regions(args.regions_file, video.name)
        detector = TorchvisionPersonDetector(
            confidence_threshold=args.confidence_threshold,
            device=args.device,
            min_size=min(1080, args.max_side),
            max_size=args.max_side,
            excluded_regions=regions,
        )

        rows: list[dict] = []
        total_tp = total_fp = total_fn = 0
        with VideoReader(video) as reader:
            if reader.info is None:
                raise RuntimeError("Could not read video metadata")
            for record in records:
                frame_index = record["frame_index"]
                frame = reader.read_frame(frame_index)
                ground_truth = _validated_boxes(record, reader.info.width, reader.info.height)
                if record["status"] == "uncertain":
                    continue

                detections = detector.detect(frame.pixels)
                predictions = [item.box_xyxy for item in detections]
                counts = match_boxes(predictions, ground_truth, args.iou_threshold)
                total_tp += counts.true_positives
                total_fp += counts.false_positives
                total_fn += counts.false_negatives
                rows.append(
                    {
                        "video": video.name,
                        "frame_index": frame_index,
                        "timestamp_seconds": f"{frame.timestamp_seconds:.6f}",
                        "label": record["status"],
                        "label_provenance": record["_label_provenance"],
                        "ground_truth_boxes": len(ground_truth),
                        "predicted_boxes": len(predictions),
                        "true_positives": counts.true_positives,
                        "false_positives": counts.false_positives,
                        "false_negatives": counts.false_negatives,
                        "confidence_threshold": args.confidence_threshold,
                        "iou_threshold": args.iou_threshold,
                    }
                )

        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "video", "frame_index", "timestamp_seconds", "label", "ground_truth_boxes",
            "label_provenance",
            "predicted_boxes", "true_positives", "false_positives", "false_negatives",
            "confidence_threshold", "iou_threshold",
        ]
        with output.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        metrics = calculate_metrics(DetectionCounts(total_tp, total_fp, total_fn))
        precision, recall, f1 = metrics.precision, metrics.recall, metrics.f1
        print(f"Device: {detector.device}")
        print(f"Scored frames: {len(rows)} (uncertain frames excluded)")
        unreviewed_frames = sum(row["label_provenance"] != "manually_reviewed" for row in rows)
        print(f"Unreviewed exploratory frames: {unreviewed_frames}")
        print(f"TP={total_tp} FP={total_fp} FN={total_fn}")
        print(f"Precision: {precision:.3f}" if precision is not None else "Precision: n/a")
        print(f"Recall: {recall:.3f}" if recall is not None else "Recall: n/a")
        print(f"F1: {f1:.3f}" if f1 is not None else "F1: n/a")
        print(f"Per-frame CSV: {output.resolve()}")
        print("These metrics describe only the labeled frames and this confidence/IoU setting.")
        if unreviewed_frames:
            print("WARNING: Some labels are unreviewed; these exploratory metrics are not a manual ground-truth benchmark.")
        return 0
    except (FileNotFoundError, IndexError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"evaluation: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
