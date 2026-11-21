"""Evaluate multiple detector confidence cutoffs from one offline inference pass."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

from vantasight.analytics import (
    evaluate_threshold_sweep,
    parse_frame_indexes,
    read_video_annotations,
    validate_annotation_coverage,
    validated_opponent_boxes,
)
from vantasight.cli import _load_excluded_regions
from vantasight.input import VideoReader, verify_frame_fingerprint
from vantasight.vision import TorchvisionPersonDetector



def _parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        thresholds = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated confidence thresholds, e.g. 0.25,0.50,0.75") from exc
    if not thresholds or any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise argparse.ArgumentTypeError("Confidence thresholds must be between 0 and 1")
    if len(set(thresholds)) != len(thresholds):
        raise argparse.ArgumentTypeError("Confidence thresholds must not repeat")
    if tuple(sorted(thresholds)) != thresholds:
        raise argparse.ArgumentTypeError("Confidence thresholds must be in ascending order")
    return thresholds


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep person-candidate confidence thresholds using one offline inference pass."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=_parse_thresholds("0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95"),
    )
    parser.add_argument("--expected-frames", type=parse_frame_indexes)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-side", type=int, default=1920)
    parser.add_argument("--preserve-source-resolution", action="store_true", help="Run each frame at original pixel dimensions (higher memory use)")
    parser.add_argument("--allow-unreviewed-labels", action="store_true")
    parser.add_argument("--regions-file", type=Path, default=Path("data/annotations/source_regions.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/detector_threshold_sweep.csv"))
    args = parser.parse_args()

    if not 0.0 < args.iou_threshold <= 1.0:
        parser.error("--iou-threshold must be greater than 0 and at most 1")
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
        regions = _load_excluded_regions(args.regions_file, video.name)
        initialization_started = time.perf_counter()
        detector = TorchvisionPersonDetector(
            confidence_threshold=args.thresholds[0],
            device=args.device,
            min_size=min(1080, args.max_side),
            max_size=args.max_side,
            excluded_regions=regions,
            preserve_source_resolution=args.preserve_source_resolution,
        )
        model_initialization_seconds = time.perf_counter() - initialization_started

        samples = []
        unreviewed_frames = 0
        inference_seconds = []
        analysis_started = time.perf_counter()
        with VideoReader(video) as reader:
            if reader.info is None:
                raise RuntimeError("Could not read video metadata")
            for record in records:
                frame_index = record["frame_index"]
                frame = reader.read_frame(frame_index)
                verify_frame_fingerprint(record, frame.pixels, frame_index)
                ground_truth = validated_opponent_boxes(record, reader.info.width, reader.info.height)
                if record["_label_provenance"] != "manually_reviewed":
                    unreviewed_frames += record["status"] != "uncertain"
                if record["status"] == "uncertain":
                    continue
                inference_started = time.perf_counter()
                detections = detector.detect(frame.pixels)
                inference_seconds.append(time.perf_counter() - inference_started)
                samples.append((
                    [(item.confidence, item.box_xyxy) for item in detections],
                    ground_truth,
                ))
        if not samples:
            raise ValueError("No scored frames remain after excluding uncertain annotations")
        sampled_analysis_elapsed_seconds = time.perf_counter() - analysis_started
        detector_latency_mean_ms = sum(inference_seconds) / len(inference_seconds) * 1000
        sampled_analysis_frames_per_second = len(samples) / sampled_analysis_elapsed_seconds

        results = evaluate_threshold_sweep(samples, args.thresholds, args.iou_threshold)
        positive_frames = sum(bool(ground_truth) for _, ground_truth in samples)
        negative_frames = len(samples) - positive_frames
        ground_truth_boxes = sum(len(ground_truth) for _, ground_truth in samples)
        provenance = "unreviewed_exploratory" if unreviewed_frames else "manually_reviewed"
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "video", "confidence_threshold", "scored_frames", "positive_frames", "negative_frames",
            "ground_truth_boxes", "true_positives", "false_positives",
            "false_negatives", "precision", "recall", "f1", "iou_threshold",
            "label_provenance", "unreviewed_scored_frames",
            "model_initialization_seconds", "inference_calls", "detector_latency_mean_ms",
            "sampled_analysis_elapsed_seconds", "sampled_analysis_frames_per_second",
            "device", "max_side", "preserve_source_resolution",
        ]
        with output.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            for result in results:
                writer.writerow({
                    **result,
                    "video": video.name,
                    "scored_frames": len(samples),
                    "positive_frames": positive_frames,
                    "negative_frames": negative_frames,
                    "ground_truth_boxes": ground_truth_boxes,
                    "iou_threshold": args.iou_threshold,
                    "label_provenance": provenance,
                    "unreviewed_scored_frames": unreviewed_frames,
                    "model_initialization_seconds": f"{model_initialization_seconds:.6f}",
                    "inference_calls": len(inference_seconds),
                    "detector_latency_mean_ms": f"{detector_latency_mean_ms:.3f}",
                    "sampled_analysis_elapsed_seconds": f"{sampled_analysis_elapsed_seconds:.6f}",
                    "sampled_analysis_frames_per_second": f"{sampled_analysis_frames_per_second:.6f}",
                    "device": str(detector.device),
                    "max_side": args.max_side,
                    "preserve_source_resolution": args.preserve_source_resolution,
                })

        print(f"Device: {detector.device}")
        print(f"Scored frames: {len(samples)}; threshold values: {len(results)}")
        print(f"Detector inference calls: {len(inference_seconds)}; mean latency: {detector_latency_mean_ms:.1f} ms")
        print(
            f"Model initialization: {model_initialization_seconds:.2f} s; "
            f"sample analysis: {sampled_analysis_elapsed_seconds:.2f} s "
            f"({sampled_analysis_frames_per_second:.2f} scored frames/s)"
        )
        print(f"Annotation provenance: {provenance}; unreviewed scored frames: {unreviewed_frames}")
        for result in results:
            precision = "n/a" if result["precision"] is None else f"{result['precision']:.3f}"
            recall = "n/a" if result["recall"] is None else f"{result['recall']:.3f}"
            f1 = "n/a" if result["f1"] is None else f"{result['f1']:.3f}"
            print(
                f"{result['confidence_threshold']:.2f}: TP={result['true_positives']} "
                f"FP={result['false_positives']} FN={result['false_negatives']} "
                f"P={precision} R={recall} F1={f1}"
            )
        print(f"Threshold sweep CSV: {output.resolve()}")
        if unreviewed_frames:
            print("WARNING: these are exploratory metrics, not user-reviewed ground truth.")
        return 0
    except (FileNotFoundError, IndexError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"threshold sweep: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
