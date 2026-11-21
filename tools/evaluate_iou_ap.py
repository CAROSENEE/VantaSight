"""Evaluate offline detector box quality across multiple IoU thresholds."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from vantasight.analytics import (
    evaluate_average_precision,
    parse_frame_indexes,
    read_video_annotations,
    validate_annotation_coverage,
    validated_opponent_boxes,
)
from vantasight.cli import _load_excluded_regions
from vantasight.input import VideoReader, verify_frame_fingerprint
from vantasight.vision import TorchvisionPersonDetector


def _parse_iou_thresholds(value: str) -> tuple[float, ...]:
    try:
        thresholds = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated IoU thresholds, e.g. 0.50,0.75") from exc
    if not thresholds or any(not math.isfinite(item) or not 0.0 < item <= 1.0 for item in thresholds):
        raise argparse.ArgumentTypeError("IoU thresholds must be greater than 0 and at most 1")
    if len(set(thresholds)) != len(thresholds) or tuple(sorted(thresholds)) != thresholds:
        raise argparse.ArgumentTypeError("IoU thresholds must be unique and in ascending order")
    return thresholds


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure confidence-ranked offline box AP across multiple IoU thresholds."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument(
        "--iou-thresholds",
        type=_parse_iou_thresholds,
        default=_parse_iou_thresholds("0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95"),
    )
    parser.add_argument("--expected-frames", type=parse_frame_indexes)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-side", type=int, default=1920)
    parser.add_argument("--allow-unreviewed-labels", action="store_true")
    parser.add_argument("--regions-file", type=Path, default=Path("data/annotations/source_regions.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/detector_iou_ap.csv"))
    args = parser.parse_args()
    if args.max_side < 1:
        parser.error("--max-side must be positive")

    video = args.video.expanduser()
    annotation_path = args.annotations.expanduser()
    output = args.output.expanduser()
    try:
        records = read_video_annotations(
            annotation_path, video.name, allow_unreviewed=args.allow_unreviewed_labels
        )
        if args.expected_frames is not None:
            validate_annotation_coverage(records, args.expected_frames)
        regions = _load_excluded_regions(args.regions_file.expanduser(), video.name)
        detector = TorchvisionPersonDetector(
            confidence_threshold=0.0,
            device=args.device,
            min_size=min(1080, args.max_side),
            max_size=args.max_side,
            excluded_regions=regions,
        )

        samples: list[tuple[list[tuple[float, tuple[int, int, int, int]]], list[tuple[int, int, int, int]]]] = []
        unreviewed_scored_frames = 0
        prediction_count = 0
        with VideoReader(video) as reader:
            if reader.info is None:
                raise RuntimeError("Could not read video metadata")
            for record in records:
                frame = reader.read_frame(record["frame_index"])
                verify_frame_fingerprint(record, frame.pixels, frame.index)
                ground_truth = validated_opponent_boxes(record, reader.info.width, reader.info.height)
                if record["status"] == "uncertain":
                    continue
                if record["_label_provenance"] != "manually_reviewed":
                    unreviewed_scored_frames += 1
                detections = detector.detect(frame.pixels)
                predictions = [(item.confidence, item.box_xyxy) for item in detections]
                prediction_count += len(predictions)
                samples.append((predictions, ground_truth))
        if not samples:
            raise ValueError("No scored frames remain after excluding uncertain annotations")

        average_precisions = [
            evaluate_average_precision(samples, threshold) for threshold in args.iou_thresholds
        ]
        if all(value is None for value in average_precisions):
            raise ValueError("The scored sample contains no ground-truth opponent boxes")
        mean_ap = sum(value for value in average_precisions if value is not None) / sum(
            value is not None for value in average_precisions
        )
        positive_frames = sum(bool(truth) for _, truth in samples)
        negative_frames = len(samples) - positive_frames
        ground_truth_boxes = sum(len(truth) for _, truth in samples)
        provenance = "unreviewed_exploratory" if unreviewed_scored_frames else "manually_reviewed"

        output.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "video", "scored_frames", "positive_frames", "negative_frames", "ground_truth_boxes",
            "candidate_detections", "iou_threshold", "average_precision_101_point",
            "mean_ap_over_reported_iou_thresholds", "label_provenance", "unreviewed_scored_frames",
            "model_architecture", "model_weights", "confidence_floor", "max_side", "device",
            "excluded_region_count", "ap_method",
        ]
        with output.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            for threshold, average_precision in zip(args.iou_thresholds, average_precisions):
                writer.writerow({
                    "video": video.name,
                    "scored_frames": len(samples),
                    "positive_frames": positive_frames,
                    "negative_frames": negative_frames,
                    "ground_truth_boxes": ground_truth_boxes,
                    "candidate_detections": prediction_count,
                    "iou_threshold": threshold,
                    "average_precision_101_point": "" if average_precision is None else f"{average_precision:.9f}",
                    "mean_ap_over_reported_iou_thresholds": f"{mean_ap:.9f}",
                    "label_provenance": provenance,
                    "unreviewed_scored_frames": unreviewed_scored_frames,
                    "model_architecture": "torchvision_fasterrcnn_mobilenet_v3_large_fpn",
                    "model_weights": "COCO_V1",
                    "confidence_floor": "0.0",
                    "max_side": args.max_side,
                    "device": str(detector.device),
                    "excluded_region_count": len(regions),
                    "ap_method": "101_point_interpolated_greedy_one_to_one",
                })

        print(f"Device: {detector.device}")
        print(f"Scored frames: {len(samples)}; candidates: {prediction_count}; ground-truth boxes: {ground_truth_boxes}")
        print(f"Annotation provenance: {provenance}; unreviewed scored frames: {unreviewed_scored_frames}")
        for threshold, average_precision in zip(args.iou_thresholds, average_precisions):
            value = "n/a" if average_precision is None else f"{average_precision:.3f}"
            print(f"IoU {threshold:.2f}: AP={value}")
        print(f"Mean AP over reported IoU thresholds: {mean_ap:.3f}")
        print(f"AP CSV: {output.resolve()}")
        if unreviewed_scored_frames:
            print("WARNING: these exploratory metrics are not user-reviewed ground truth.")
        return 0
    except (FileNotFoundError, IndexError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"IoU AP evaluation: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
