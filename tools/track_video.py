"""Run sampled, offline person-candidate detection and short-term tracking."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from pathlib import Path

from vantasight.cli import _load_excluded_regions
from vantasight.analytics import (
    associate_pose_observations,
    center_crosshair,
    measure_alignment,
    write_tracking_report,
)
from vantasight.input import VideoReader
from vantasight.tracking import IoUTracker
from vantasight.vision.body_region_estimator import TorchvisionBodyRegionEstimator
from vantasight.vision import TorchvisionPersonDetector
from vantasight.visualization.detection_renderer import save_annotated_image
from vantasight.visualization.pose_renderer import render_pose_estimates
from vantasight.visualization.tracking_renderer import render_tracks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a recorded video offline and export temporary candidate tracks."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--frame-step", type=int, default=30, help="Analyze every Nth source frame")
    parser.add_argument("--start-frame", type=int, default=0, help="Inclusive first frame index")
    parser.add_argument("--stop-frame", type=int, help="Exclusive last frame index")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--max-missed-updates", type=int, default=1)
    parser.add_argument("--max-side", type=int, default=1920)
    parser.add_argument("--estimate-body-regions", action="store_true")
    parser.add_argument("--measure-alignment", action="store_true")
    parser.add_argument("--crosshair", type=str, help="Reference point in source pixels as x,y")
    parser.add_argument("--pose-person-threshold", type=float, default=0.25)
    parser.add_argument("--keypoint-score-threshold", type=float, default=0.0)
    parser.add_argument("--pose-match-iou-threshold", type=float, default=0.10)
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, or cuda:0")
    parser.add_argument(
        "--regions-file",
        type=Path,
        default=Path("data/annotations/source_regions.json"),
    )
    parser.add_argument("--preview-frame", type=int, help="Save a tracking preview for this sampled frame")
    parser.add_argument(
        "--pause-file",
        type=Path,
        help="Wait while this local sentinel file exists (used by the offline dashboard)",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/tracks.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("reports/tracks.json"))
    args = parser.parse_args()

    if args.frame_step < 1 or args.start_frame < 0 or args.max_side < 1:
        parser.error("--frame-step and --max-side must be positive; --start-frame must be nonnegative")
    if args.stop_frame is not None and args.stop_frame <= args.start_frame:
        parser.error("--stop-frame must be greater than --start-frame")
    if args.preview_frame is not None and args.preview_frame < 0:
        parser.error("--preview-frame must be nonnegative")
    if args.measure_alignment and not args.estimate_body_regions:
        parser.error("--measure-alignment requires --estimate-body-regions")
    if args.crosshair and not args.measure_alignment:
        parser.error("--crosshair requires --measure-alignment")
    if args.keypoint_score_threshold < 0:
        parser.error("--keypoint-score-threshold must be zero or greater")
    if not 0.0 <= args.pose_person_threshold <= 1.0:
        parser.error("--pose-person-threshold must be between 0 and 1")
    if not 0.0 < args.pose_match_iou_threshold <= 1.0:
        parser.error("--pose-match-iou-threshold must be greater than 0 and at most 1")
    crosshair_override = None
    if args.crosshair:
        try:
            x_text, y_text = args.crosshair.split(",", maxsplit=1)
            crosshair_override = (float(x_text), float(y_text))
        except (ValueError, TypeError):
            parser.error("--crosshair must use the format x,y")

    video = args.video.expanduser()
    output = args.output.expanduser()
    json_output = args.json_output.expanduser()
    if output.resolve() == json_output.resolve():
        parser.error("--output and --json-output must use different file paths")
    try:
        regions = _load_excluded_regions(args.regions_file.expanduser(), video.name)
        detector = TorchvisionPersonDetector(
            confidence_threshold=args.confidence_threshold,
            device=args.device,
            min_size=min(1080, args.max_side),
            max_size=args.max_side,
            excluded_regions=regions,
        )
        tracker = IoUTracker(
            iou_threshold=args.iou_threshold,
            max_missed_updates=args.max_missed_updates,
        )
        pose_estimator = (
            TorchvisionBodyRegionEstimator(
                person_confidence_threshold=args.pose_person_threshold,
                device=args.device,
                min_size=min(1080, args.max_side),
                max_size=args.max_side,
                keypoint_score_threshold=args.keypoint_score_threshold,
                excluded_regions=regions,
            )
            if args.estimate_body_regions
            else None
        )
        rows: list[dict] = []
        frame_records: list[dict] = []
        latencies_ms: list[float] = []
        pose_latencies_ms: list[float] = []
        sampled_frames = 0
        total_detections = 0
        preview_saved = False
        paused_seconds = 0.0
        pause_file = args.pause_file.expanduser() if args.pause_file else None
        started_at = time.perf_counter()

        with VideoReader(video) as reader:
            if reader.info is None:
                raise RuntimeError("Could not read video metadata")
            reference_xy = (
                crosshair_override
                if crosshair_override is not None
                else center_crosshair(reader.info.width, reader.info.height)
            )
            if args.measure_alignment and not (
                0 <= reference_xy[0] <= reader.info.width
                and 0 <= reference_xy[1] <= reader.info.height
            ):
                raise ValueError("Crosshair reference must lie within the source frame")
            for frame in reader.frames(start_frame=args.start_frame):
                if args.stop_frame is not None and frame.index >= args.stop_frame:
                    break
                if (frame.index - args.start_frame) % args.frame_step:
                    continue

                pause_started: float | None = None
                while pause_file is not None and pause_file.exists():
                    if pause_started is None:
                        pause_started = time.perf_counter()
                    time.sleep(0.1)
                if pause_started is not None:
                    paused_seconds += time.perf_counter() - pause_started

                inference_started = time.perf_counter()
                detections = detector.detect(frame.pixels)
                latencies_ms.append((time.perf_counter() - inference_started) * 1000)
                tracked = tracker.update(detections, frame.index, frame.timestamp_seconds)
                if pose_estimator:
                    pose_started = time.perf_counter()
                    pose_observations = pose_estimator.estimate(frame.pixels)
                    pose_latencies_ms.append((time.perf_counter() - pose_started) * 1000)
                else:
                    pose_observations = []
                pose_by_candidate = associate_pose_observations(
                    [item.detection.box_xyxy for item in tracked],
                    pose_observations,
                    iou_threshold=args.pose_match_iou_threshold,
                ) if pose_estimator else {}
                sampled_frames += 1
                total_detections += len(detections)

                frame_detections = []
                for item in tracked:
                    velocity = item.velocity_xy_pixels_per_second
                    x1, y1, x2, y2 = item.detection.box_xyxy
                    frame_detections.append(
                        {
                            "track_id": item.track_id,
                            "class_name": item.detection.class_name,
                            "detector_score": item.detection.confidence,
                            "box_xyxy": [x1, y1, x2, y2],
                            "center_xy": [(x1 + x2) / 2, (y1 + y2) / 2],
                            "velocity_xy_pixels_per_second": None if velocity is None else list(velocity),
                        }
                    )
                    candidate_record = frame_detections[-1]
                    if pose_estimator:
                        pose = pose_by_candidate.get(len(frame_detections) - 1)
                        candidate_record["pose_status"] = "matched" if pose else "unmatched"
                        if pose:
                            candidate_record["pose_box_xyxy"] = list(pose.box_xyxy)
                            candidate_record["pose_person_score"] = pose.person_confidence
                            candidate_record["body_regions"] = []
                            for region in pose.body_regions:
                                region_record = {
                                    "name": region.name,
                                    "center_xy": list(region.center_xy),
                                    "mean_keypoint_score": region.mean_keypoint_score,
                                    "supporting_keypoints": list(region.supporting_keypoints),
                                }
                                if args.measure_alignment:
                                    measurement = measure_alignment(
                                        tuple(float(v) for v in region.center_xy),
                                        reference_xy,
                                        reader.info.width,
                                        reader.info.height,
                                        region_name=region.name,
                                    )
                                    region_record["alignment"] = {
                                        "crosshair_xy": list(measurement.crosshair_xy),
                                        "distance_pixels": measurement.distance_pixels,
                                        "distance_normalized": measurement.distance_normalized,
                                    }
                                candidate_record["body_regions"].append(region_record)
                frame_records.append(
                    {
                        "frame_index": frame.index,
                        "timestamp_seconds": frame.timestamp_seconds,
                        "detection_count": len(frame_detections),
                        "detections": frame_detections,
                    }
                )

                if args.preview_frame == frame.index:
                    preview = Path("data/previews") / f"{video.stem}_frame_{frame.index:06d}_tracks.png"
                    preview_image = render_tracks(frame.pixels, tracked)
                    if pose_estimator:
                        preview_image = render_pose_estimates(
                            preview_image,
                            pose_observations,
                            crosshair_xy=reference_xy if args.measure_alignment else None,
                        )
                    save_annotated_image(preview_image, preview)
                    preview_saved = True

                if not tracked:
                    rows.append(
                        {
                            "video": video.name,
                            "frame_index": frame.index,
                            "timestamp_seconds": f"{frame.timestamp_seconds:.6f}",
                            "frame_detection_count": 0,
                        }
                    )
                for item in tracked:
                    velocity = item.velocity_xy_pixels_per_second
                    rows.append(
                        {
                            "video": video.name,
                            "frame_index": frame.index,
                            "timestamp_seconds": f"{frame.timestamp_seconds:.6f}",
                            "frame_detection_count": len(tracked),
                            "track_id": item.track_id,
                            "class_name": item.detection.class_name,
                            "confidence": f"{item.detection.confidence:.6f}",
                            "x1": item.detection.box_xyxy[0],
                            "y1": item.detection.box_xyxy[1],
                            "x2": item.detection.box_xyxy[2],
                            "y2": item.detection.box_xyxy[3],
                            "center_x": (item.detection.box_xyxy[0] + item.detection.box_xyxy[2]) / 2,
                            "center_y": (item.detection.box_xyxy[1] + item.detection.box_xyxy[3]) / 2,
                            "velocity_x_px_per_s": "" if velocity is None else f"{velocity[0]:.3f}",
                            "velocity_y_px_per_s": "" if velocity is None else f"{velocity[1]:.3f}",
                        }
                    )

        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "video", "frame_index", "timestamp_seconds", "frame_detection_count", "track_id",
            "class_name", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y",
            "velocity_x_px_per_s", "velocity_y_px_per_s",
        ]
        with output.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        elapsed = max(0.0, time.perf_counter() - started_at - paused_seconds)
        throughput = sampled_frames / elapsed if elapsed > 0 else 0.0
        mean_latency = statistics.mean(latencies_ms) if latencies_ms else 0.0
        p95_latency = sorted(latencies_ms)[math.ceil(0.95 * len(latencies_ms)) - 1] if latencies_ms else 0.0
        mean_pose_latency = statistics.mean(pose_latencies_ms) if pose_latencies_ms else None
        p95_pose_latency = (
            sorted(pose_latencies_ms)[math.ceil(0.95 * len(pose_latencies_ms)) - 1]
            if pose_latencies_ms
            else None
        )
        report = {
            "schema_version": 1,
            "analysis_type": "offline_sampled_person_candidate_tracking",
            "interpretation_limits": [
                "Detections are COCO person candidates, not verified enemies or teams.",
                "Track IDs are temporary box associations, not player identities.",
                "Velocity is measured in source-image pixels per second.",
                "This report contains no game input or live gameplay data.",
            ],
            "source": {
                "file_name": video.name,
                "width": reader.info.width,
                "height": reader.info.height,
                "fps": reader.info.fps,
                "frame_count": reader.info.frame_count,
                "duration_seconds": reader.info.duration_seconds,
            },
            "settings": {
                "frame_step": args.frame_step,
                "start_frame": args.start_frame,
                "stop_frame_exclusive": args.stop_frame,
                "confidence_threshold": args.confidence_threshold,
                "iou_threshold": args.iou_threshold,
                "max_missed_updates": args.max_missed_updates,
                "device": str(detector.device),
                "body_region_estimation": args.estimate_body_regions,
                "pose_person_threshold": args.pose_person_threshold if pose_estimator else None,
                "alignment_measurement": args.measure_alignment,
                "crosshair_xy": list(reference_xy) if args.measure_alignment else None,
                "keypoint_score_threshold": args.keypoint_score_threshold,
                "pose_match_iou_threshold": args.pose_match_iou_threshold,
            },
            "summary": {
                "sampled_frames": sampled_frames,
                "candidate_detections": total_detections,
                "analysis_elapsed_seconds": elapsed,
                "analysis_paused_seconds": paused_seconds,
                "sampled_frames_per_second": throughput,
                "detector_latency_mean_ms": mean_latency,
                "detector_latency_p95_ms": p95_latency,
                "pose_latency_mean_ms": mean_pose_latency,
                "pose_latency_p95_ms": p95_pose_latency,
            },
            "frames": frame_records,
        }
        write_tracking_report(json_output, report)
        print(f"Device: {detector.device}")
        print(f"Source FPS: {reader.info.fps:.3f}")
        print(f"Sampled frames: {sampled_frames}; detections: {total_detections}")
        print(f"Analysis throughput: {throughput:.3f} sampled frames/s")
        if paused_seconds:
            print(f"User-paused time (excluded from throughput): {paused_seconds:.1f}s")
        print(f"Detector latency: mean={mean_latency:.1f} ms p95={p95_latency:.1f} ms")
        if pose_latencies_ms:
            print(f"Pose latency: mean={mean_pose_latency:.1f} ms p95={p95_pose_latency:.1f} ms")
        print(f"Track rows: {output.resolve()}")
        print(f"Structured JSON report: {json_output.resolve()}")
        if args.preview_frame is not None:
            print("Tracking preview saved" if preview_saved else "Requested preview frame was not sampled")
        print("Track IDs are temporary box associations; they are not player identity or enemy/team labels.")
        return 0
    except (FileNotFoundError, IndexError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"tracking: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
