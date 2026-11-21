"""Analyze one screenshot offline and export candidate detections as CSV/JSON."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from vantasight.analytics import (
    associate_pose_observations,
    center_crosshair,
    measure_alignment,
    summarize_body_region_responses,
    summarize_distribution,
    write_tracking_report,
)
from vantasight.cli import IMAGE_SUFFIXES, _load_excluded_regions, _parse_point
from vantasight.input import file_sha256, load_image
from vantasight.vision import TorchvisionBodyRegionEstimator, TorchvisionPersonDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a screenshot offline and export candidate boxes.")
    parser.add_argument("image", type=Path, help="Path to a PNG/JPG screenshot")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, or cuda:0")
    parser.add_argument("--max-side", type=int, default=1920)
    parser.add_argument("--preserve-source-resolution", action="store_true", help="Disable model-side resizing for this image (higher memory use)")
    parser.add_argument("--estimate-body-regions", action="store_true")
    parser.add_argument("--measure-alignment", action="store_true")
    parser.add_argument("--crosshair", type=_parse_point, help="Reference point in source pixels as x,y")
    parser.add_argument("--keypoint-score-threshold", type=float, default=0.0)
    parser.add_argument("--pose-person-threshold", type=float, default=0.25)
    parser.add_argument("--pose-match-iou-threshold", type=float, default=0.10)
    parser.add_argument("--regions-file", type=Path, default=Path("data/annotations/source_regions.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/image_analysis.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("reports/image_analysis.json"))
    args = parser.parse_args()

    image_path = args.image.expanduser()
    csv_path = args.output.expanduser()
    json_path = args.json_output.expanduser()
    if image_path.suffix.lower() not in IMAGE_SUFFIXES:
        parser.error("image must use a supported image suffix: PNG, JPG, JPEG, BMP, or WEBP")
    if args.max_side < 1:
        parser.error("--max-side must be positive")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        parser.error("--confidence-threshold must be between 0 and 1")
    if not 0.0 <= args.pose_person_threshold <= 1.0:
        parser.error("--pose-person-threshold must be between 0 and 1")
    if args.keypoint_score_threshold < 0:
        parser.error("--keypoint-score-threshold must be zero or greater")
    if not 0.0 < args.pose_match_iou_threshold <= 1.0:
        parser.error("--pose-match-iou-threshold must be greater than 0 and at most 1")
    if args.measure_alignment and not args.estimate_body_regions:
        parser.error("--measure-alignment requires --estimate-body-regions")
    if args.crosshair is not None and not args.measure_alignment:
        parser.error("--crosshair requires --measure-alignment")
    if csv_path.resolve() == json_path.resolve():
        parser.error("--output and --json-output must use different file paths")

    try:
        image = load_image(image_path)
        height, width = image.pixels.shape[:2]
        source_sha256 = file_sha256(image.path)
        excluded_regions = _load_excluded_regions(args.regions_file.expanduser(), image.path.name)
        detector = TorchvisionPersonDetector(
            confidence_threshold=args.confidence_threshold,
            device=args.device,
            min_size=min(1080, args.max_side),
            max_size=args.max_side,
            excluded_regions=excluded_regions,
            preserve_source_resolution=args.preserve_source_resolution,
        )
        pose_estimator = (
            TorchvisionBodyRegionEstimator(
                person_confidence_threshold=args.pose_person_threshold,
                device=args.device,
                min_size=min(1080, args.max_side),
                max_size=args.max_side,
                keypoint_score_threshold=args.keypoint_score_threshold,
                excluded_regions=excluded_regions,
                preserve_source_resolution=args.preserve_source_resolution,
            )
            if args.estimate_body_regions
            else None
        )

        started_at = time.perf_counter()
        detector_started = time.perf_counter()
        detections = detector.detect(image.pixels)
        detector_latency_ms = (time.perf_counter() - detector_started) * 1000
        pose_latency_ms: float | None = None
        pose_observations = []
        if pose_estimator is not None:
            pose_started = time.perf_counter()
            pose_observations = pose_estimator.estimate(image.pixels)
            pose_latency_ms = (time.perf_counter() - pose_started) * 1000
        pose_by_detection = associate_pose_observations(
            [item.box_xyxy for item in detections],
            pose_observations,
            iou_threshold=args.pose_match_iou_threshold,
        ) if pose_estimator else {}

        reference_xy = (
            (args.crosshair or center_crosshair(width, height))
            if args.measure_alignment
            else None
        )
        if reference_xy is not None and not (
            0 <= reference_xy[0] <= width and 0 <= reference_xy[1] <= height
        ):
            raise ValueError("Crosshair coordinates must lie inside the source image")
        detection_records: list[dict] = []
        detector_scores = [item.confidence for item in detections]
        rows: list[dict] = []
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.box_xyxy
            record = {
                "track_id": index + 1,
                "class_name": detection.class_name,
                "detector_score": detection.confidence,
                "box_xyxy": [x1, y1, x2, y2],
                "center_xy": [(x1 + x2) / 2, (y1 + y2) / 2],
                "velocity_xy_pixels_per_second": None,
            }
            if pose_estimator:
                pose = pose_by_detection.get(index)
                record["pose_status"] = "matched" if pose else "unmatched"
                if pose:
                    record["pose_box_xyxy"] = list(pose.box_xyxy)
                    record["pose_person_score"] = pose.person_confidence
                    record["body_regions"] = []
                    for region in pose.body_regions:
                        region_record = {
                            "name": region.name,
                            "center_xy": list(region.center_xy),
                            "mean_keypoint_score": region.mean_keypoint_score,
                            "supporting_keypoints": list(region.supporting_keypoints),
                        }
                        if reference_xy is not None:
                            measurement = measure_alignment(
                                tuple(float(value) for value in region.center_xy),
                                reference_xy,
                                width,
                                height,
                                region_name=region.name,
                            )
                            region_record["alignment"] = {
                                "crosshair_xy": list(measurement.crosshair_xy),
                                "distance_pixels": measurement.distance_pixels,
                                "distance_normalized": measurement.distance_normalized,
                            }
                        record["body_regions"].append(region_record)
            detection_records.append(record)
            rows.append(
                {
                    "video": image.path.name,
                    "frame_index": 0,
                    "timestamp_seconds": "0.000000",
                    "frame_detection_count": len(detections),
                    "track_id": index + 1,
                    "class_name": detection.class_name,
                    "confidence": f"{detection.confidence:.6f}",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": (x1 + x2) / 2,
                    "center_y": (y1 + y2) / 2,
                    "velocity_x_px_per_s": "",
                    "velocity_y_px_per_s": "",
                }
            )
        if not detections:
            rows.append(
                {
                    "video": image.path.name,
                    "frame_index": 0,
                    "timestamp_seconds": "0.000000",
                    "frame_detection_count": 0,
                }
            )

        elapsed = time.perf_counter() - started_at
        if file_sha256(image.path) != source_sha256:
            raise RuntimeError("Source screenshot changed during analysis; refusing to write a mismatched report")
        report = {
            "schema_version": 1,
            "analysis_type": "offline_single_image_person_candidate_analysis",
            "interpretation_limits": [
                "Detections are COCO person candidates, not verified enemies or teams.",
                "Single-image candidate IDs are local to this image and do not imply tracking.",
                "Alignment is a geometric estimate, not hit probability or aim accuracy.",
                "This report contains no game input or live gameplay data.",
            ],
            "source": {
                "file_name": image.path.name,
                "source_sha256": source_sha256,
                "width": width,
                "height": height,
                "fps": 0.0,
                "frame_count": 1,
                "duration_seconds": 0.0,
            },
            "settings": {
                "confidence_threshold": args.confidence_threshold,
                "device": str(detector.device),
                "preserve_source_resolution": args.preserve_source_resolution,
                "body_region_estimation": args.estimate_body_regions,
                "pose_person_threshold": args.pose_person_threshold if pose_estimator else None,
                "alignment_measurement": args.measure_alignment,
                "crosshair_xy": list(reference_xy) if reference_xy else None,
                "keypoint_score_threshold": args.keypoint_score_threshold,
                "pose_match_iou_threshold": args.pose_match_iou_threshold,
            },
            "summary": {
                "sampled_frames": 1,
                "candidate_detections": len(detections),
                "analysis_elapsed_seconds": elapsed,
                "sampled_frames_per_second": 1 / elapsed if elapsed > 0 else 0.0,
                "detector_latency_mean_ms": detector_latency_ms,
                "detector_latency_p95_ms": detector_latency_ms,
                "detector_confidence": summarize_distribution(detector_scores),
                "body_region_heatmap_response": summarize_body_region_responses([
                    {"detections": detection_records}
                ]),
                "pose_latency_mean_ms": pose_latency_ms,
                "pose_latency_p95_ms": pose_latency_ms,
            },
            "frames": [
                {
                    "frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "detection_count": len(detection_records),
                    "detections": detection_records,
                }
            ],
        }
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as destination:
            fieldnames = [
                "video", "frame_index", "timestamp_seconds", "frame_detection_count", "track_id",
                "class_name", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y",
                "velocity_x_px_per_s", "velocity_y_px_per_s",
            ]
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        write_tracking_report(json_path, report)
        print(f"Device: {detector.device}")
        print(f"Source image: {width}x{height}")
        print(f"Candidates: {len(detections)}")
        print(f"Detector latency: {detector_latency_ms:.1f} ms")
        score_summary = summarize_distribution(detector_scores)
        if score_summary["count"]:
            print(
                "Detector confidence: "
                f"median={score_summary['median']:.3f} p95={score_summary['p95']:.3f} "
                f"min={score_summary['min']:.3f} max={score_summary['max']:.3f}"
            )
        else:
            print("Detector confidence: no candidates")
        print(f"CSV: {csv_path.resolve()}")
        print(f"JSON report: {json_path.resolve()}")
        return 0
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"image analysis: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
