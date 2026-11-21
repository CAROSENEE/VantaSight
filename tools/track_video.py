"""Run sampled, offline person-candidate detection and short-term tracking."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from pathlib import Path

from vantasight.cli import _load_excluded_regions
from vantasight.analytics import write_tracking_report
from vantasight.input import VideoReader
from vantasight.tracking import IoUTracker
from vantasight.vision import TorchvisionPersonDetector
from vantasight.visualization.detection_renderer import save_annotated_image
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
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, or cuda:0")
    parser.add_argument(
        "--regions-file",
        type=Path,
        default=Path("data/annotations/source_regions.json"),
    )
    parser.add_argument("--preview-frame", type=int, help="Save a tracking preview for this sampled frame")
    parser.add_argument("--output", type=Path, default=Path("reports/tracks.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("reports/tracks.json"))
    args = parser.parse_args()

    if args.frame_step < 1 or args.start_frame < 0 or args.max_side < 1:
        parser.error("--frame-step and --max-side must be positive; --start-frame must be nonnegative")
    if args.stop_frame is not None and args.stop_frame <= args.start_frame:
        parser.error("--stop-frame must be greater than --start-frame")
    if args.preview_frame is not None and args.preview_frame < 0:
        parser.error("--preview-frame must be nonnegative")

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
        rows: list[dict] = []
        frame_records: list[dict] = []
        latencies_ms: list[float] = []
        sampled_frames = 0
        total_detections = 0
        preview_saved = False
        started_at = time.perf_counter()

        with VideoReader(video) as reader:
            if reader.info is None:
                raise RuntimeError("Could not read video metadata")
            for frame in reader.frames(start_frame=args.start_frame):
                if args.stop_frame is not None and frame.index >= args.stop_frame:
                    break
                if (frame.index - args.start_frame) % args.frame_step:
                    continue

                inference_started = time.perf_counter()
                detections = detector.detect(frame.pixels)
                latencies_ms.append((time.perf_counter() - inference_started) * 1000)
                tracked = tracker.update(detections, frame.index, frame.timestamp_seconds)
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
                    save_annotated_image(render_tracks(frame.pixels, tracked), preview)
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

        elapsed = time.perf_counter() - started_at
        throughput = sampled_frames / elapsed if elapsed > 0 else 0.0
        mean_latency = statistics.mean(latencies_ms) if latencies_ms else 0.0
        p95_latency = sorted(latencies_ms)[math.ceil(0.95 * len(latencies_ms)) - 1] if latencies_ms else 0.0
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
            },
            "summary": {
                "sampled_frames": sampled_frames,
                "candidate_detections": total_detections,
                "analysis_elapsed_seconds": elapsed,
                "sampled_frames_per_second": throughput,
                "detector_latency_mean_ms": mean_latency,
                "detector_latency_p95_ms": p95_latency,
            },
            "frames": frame_records,
        }
        write_tracking_report(json_output, report)
        print(f"Device: {detector.device}")
        print(f"Source FPS: {reader.info.fps:.3f}")
        print(f"Sampled frames: {sampled_frames}; detections: {total_detections}")
        print(f"Analysis throughput: {throughput:.3f} sampled frames/s")
        print(f"Detector latency: mean={mean_latency:.1f} ms p95={p95_latency:.1f} ms")
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
