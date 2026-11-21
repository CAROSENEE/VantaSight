"""Command-line input inspector for VantaSight."""

import argparse
import json
from pathlib import Path

from vantasight.analytics import center_crosshair, measure_alignment
from vantasight.input import VideoReader, load_image
from vantasight.visualization.frame_preview import save_frame_preview
from vantasight.visualization.pose_renderer import render_pose_estimates
from vantasight.vision import TorchvisionBodyRegionEstimator, TorchvisionPersonDetector
from vantasight.visualization.detection_renderer import render_detections, save_annotated_image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _format_duration(seconds: float | None) -> str:
    return "unknown" if seconds is None else f"{seconds:.3f} s"


def _parse_region(value: str) -> tuple[int, int, int, int]:
    """Parse an exclusion rectangle written as x1,y1,x2,y2."""

    try:
        coordinates = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected four comma-separated integers: x1,y1,x2,y2") from exc
    if len(coordinates) != 4:
        raise argparse.ArgumentTypeError("Expected four comma-separated integers: x1,y1,x2,y2")
    x1, y1, x2, y2 = coordinates
    if x1 >= x2 or y1 >= y2:
        raise argparse.ArgumentTypeError("Region must have x1 < x2 and y1 < y2")
    return x1, y1, x2, y2


def _parse_point(value: str) -> tuple[float, float]:
    """Parse a crosshair point written as x,y in source-image pixels."""

    try:
        coordinates = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected two comma-separated coordinates: x,y") from exc
    if len(coordinates) != 2:
        raise argparse.ArgumentTypeError("Expected two comma-separated coordinates: x,y")
    return coordinates


def _detect_and_save(
    pixels,
    args: argparse.Namespace,
    output_path: Path,
    asset_name: str,
) -> None:
    configured_regions = _load_excluded_regions(args.regions_file, asset_name)
    detector = TorchvisionPersonDetector(
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        min_size=min(1080, args.max_side),
        max_size=args.max_side,
        excluded_regions=(*configured_regions, *(args.exclude_region or ())),
        preserve_source_resolution=args.preserve_source_resolution,
    )
    detections = detector.detect(pixels)
    print(f"Device: {detector.device}")
    print("Class: COCO person candidates (not enemy/team identity)")
    print(f"Candidates: {len(detections)}")
    for item in detections:
        print(
            f"  {item.class_name} confidence={item.confidence:.3f} "
            f"box={item.box_xyxy}"
        )
    annotated = render_detections(pixels, detections)
    saved_path = save_annotated_image(annotated, output_path)
    print(f"Annotated preview: {saved_path}")


def _estimate_body_regions_and_save(
    pixels,
    args: argparse.Namespace,
    output_path: Path,
    asset_name: str,
) -> None:
    configured_regions = _load_excluded_regions(args.regions_file, asset_name)
    estimator = TorchvisionBodyRegionEstimator(
        person_confidence_threshold=args.confidence_threshold,
        keypoint_score_threshold=args.keypoint_score_threshold,
        device=args.device,
        min_size=min(1080, args.max_side),
        max_size=args.max_side,
        excluded_regions=(*configured_regions, *(args.exclude_region or ())),
        preserve_source_resolution=args.preserve_source_resolution,
    )
    observations = estimator.estimate(pixels)
    height, width = pixels.shape[:2]
    print(f"Device: {estimator.device}")
    print("Model: COCO Keypoint R-CNN (not trained for Valorant footage)")
    print(f"Person candidates: {len(observations)}")
    reference = None
    if args.measure_alignment:
        reference = args.crosshair or center_crosshair(width, height)
        if not (0 <= reference[0] <= width and 0 <= reference[1] <= height):
            raise ValueError("Crosshair coordinates must lie inside the source frame")
        print(f"Crosshair reference (source pixels): ({reference[0]:.1f}, {reference[1]:.1f})")
    for index, observation in enumerate(observations, start=1):
        print(
            f"  Candidate {index}: confidence={observation.person_confidence:.3f} "
            f"box={observation.box_xyxy} keypoints={len(observation.keypoints)}"
        )
        for region in observation.body_regions:
            print(
                f"    {region.name} estimate center={region.center_xy} "
                f"mean-keypoint-heatmap-score={region.mean_keypoint_score:.3f} "
                f"from={','.join(region.supporting_keypoints)}"
            )
            if reference is not None:
                measurement = measure_alignment(
                    point_xy=region.center_xy,
                    crosshair_xy=reference,
                    width=width,
                    height=height,
                    region_name=region.name,
                )
                print(
                    f"      alignment distance={measurement.distance_pixels:.2f}px "
                    f"normalized={measurement.distance_normalized:.6f}"
                )
        if not observation.body_regions:
            print("    Body regions: uncertain (no keypoints passed the threshold)")
    saved_path = save_annotated_image(
        render_pose_estimates(pixels, observations, crosshair_xy=reference), output_path
    )
    print(f"Body-region preview: {saved_path}")


def _load_excluded_regions(
    regions_file: Path,
    asset_name: str,
) -> tuple[tuple[int, int, int, int], ...]:
    """Load optional per-input exclusion rectangles from a local JSON file."""

    if not regions_file.is_file():
        return ()
    try:
        document = json.loads(regions_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read exclusion regions from {regions_file}: {exc}") from exc

    if not isinstance(document, dict):
        raise ValueError(f"Expected an object at the top level of {regions_file}")
    asset_regions = document.get(asset_name, [])
    if not isinstance(asset_regions, list):
        raise ValueError(f"Expected a list of regions for {asset_name} in {regions_file}")

    regions: list[tuple[int, int, int, int]] = []
    for index, record in enumerate(asset_regions):
        box = record.get("box_xyxy") if isinstance(record, dict) else None
        if not isinstance(box, list) or len(box) != 4 or not all(type(n) is int for n in box):
            raise ValueError(
                f"Region {index} for {asset_name} must have integer box_xyxy: [x1, y1, x2, y2]"
            )
        x1, y1, x2, y2 = box
        if x1 < x2 and y1 < y2:
            regions.append((x1, y1, x2, y2))
        else:
            raise ValueError(f"Region {index} for {asset_name} has invalid coordinates: {box}")
    return tuple(regions)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="vantasight",
        description="Inspect a screenshot or recorded video for offline analysis.",
    )
    parser.add_argument("input", type=Path, help="Path to a screenshot or video file")
    parser.add_argument(
        "--frame-index",
        type=int,
        help="Extract this zero-based frame from a video and save a preview image",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output preview path (default: data/previews/<input>_preview.png)",
    )
    parser.add_argument(
        "--detect-persons",
        action="store_true",
        help="Run the optional COCO person-candidate detector on the image or selected video frame",
    )
    parser.add_argument(
        "--estimate-body-regions",
        action="store_true",
        help="Run the experimental COCO Keypoint R-CNN on an image or selected video frame",
    )
    parser.add_argument(
        "--keypoint-score-threshold",
        type=float,
        default=0.0,
        help="Minimum raw keypoint heatmap score, not a probability (default: 0.0)",
    )
    parser.add_argument(
        "--pose-output",
        type=Path,
        help="Body-region preview path (default: data/previews/<input>_body_regions.png)",
    )
    parser.add_argument(
        "--measure-alignment",
        action="store_true",
        help="Measure region-center distance from the frame-center crosshair or --crosshair point",
    )
    parser.add_argument(
        "--crosshair",
        type=_parse_point,
        help="Crosshair reference as source-pixel x,y (default: geometric frame center)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Minimum model confidence for a person candidate (default: 0.25)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, or cuda:0 (default: auto)",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1920,
        help="Maximum internal detector image side in pixels; output boxes map to source resolution",
    )
    parser.add_argument(
        "--preserve-source-resolution",
        action="store_true",
        help="Run vision inference at the input frame's original pixel dimensions (higher memory use)",
    )
    parser.add_argument(
        "--exclude-region",
        type=_parse_region,
        action="append",
        help="Ignore detections centered inside x1,y1,x2,y2 source-pixel rectangle; may be repeated",
    )
    parser.add_argument(
        "--regions-file",
        type=Path,
        default=Path("data/annotations/source_regions.json"),
        help="Optional per-input exclusion regions JSON (default: data/annotations/source_regions.json)",
    )
    args = parser.parse_args()
    path: Path = args.input

    try:
        if path.suffix.lower() in IMAGE_SUFFIXES:
            if args.frame_index is not None:
                parser.error("--frame-index can only be used with a video")
            if args.output is not None and not args.detect_persons:
                parser.error("--output requires --detect-persons for image input")
            if args.exclude_region and not (args.detect_persons or args.estimate_body_regions):
                parser.error("--exclude-region requires --detect-persons or --estimate-body-regions")
            if args.pose_output is not None and not args.estimate_body_regions:
                parser.error("--pose-output requires --estimate-body-regions")
            if args.measure_alignment and not args.estimate_body_regions:
                parser.error("--measure-alignment requires --estimate-body-regions")
            if args.crosshair is not None and not args.measure_alignment:
                parser.error("--crosshair requires --measure-alignment")
            image = load_image(path)
            print(f"Type: image\nPath: {image.path}\nResolution: {image.width}x{image.height}")
            if args.detect_persons:
                output = args.output or Path("data/previews") / f"{path.stem}_detections.png"
                _detect_and_save(image.pixels, args, output, path.name)
            if args.estimate_body_regions:
                output = args.pose_output or Path("data/previews") / f"{path.stem}_body_regions.png"
                _estimate_body_regions_and_save(image.pixels, args, output, path.name)
            return 0

        if args.output is not None and args.frame_index is None:
            parser.error("--output requires --frame-index for video input")
        if args.pose_output is not None and args.frame_index is None:
            parser.error("--pose-output requires --frame-index for video input")
        if args.output is not None and not args.detect_persons:
            parser.error("--output requires --detect-persons")
        if args.pose_output is not None and not args.estimate_body_regions:
            parser.error("--pose-output requires --estimate-body-regions")
        if args.measure_alignment and not args.estimate_body_regions:
            parser.error("--measure-alignment requires --estimate-body-regions")
        if args.crosshair is not None and not args.measure_alignment:
            parser.error("--crosshair requires --measure-alignment")
        if args.exclude_region and not (args.detect_persons or args.estimate_body_regions):
            parser.error("--exclude-region requires --detect-persons or --estimate-body-regions")
        if (args.detect_persons or args.estimate_body_regions) and args.frame_index is None:
            parser.error("Detection or body-region estimation on a video requires --frame-index")

        with VideoReader(path) as reader:
            assert reader.info is not None
            info = reader.info
            print(
                "\n".join(
                    [
                        "Type: video",
                        f"Path: {info.path}",
                        f"Resolution: {info.width}x{info.height}",
                        f"FPS: {info.fps:.3f}" if info.fps > 0 else "FPS: unknown",
                        f"Frame count: {info.frame_count}" if info.frame_count > 0 else "Frame count: unknown",
                        f"Duration: {_format_duration(info.duration_seconds)}",
                    ]
                )
            )
            if args.frame_index is not None:
                frame = reader.read_frame(args.frame_index)
                output = args.output or Path("data/previews") / (
                    f"{path.stem}_frame_{args.frame_index:06d}.png"
                )
                if args.detect_persons:
                    if args.output is None:
                        output = Path("data/previews") / (
                            f"{path.stem}_frame_{args.frame_index:06d}_detections.png"
                        )
                    _detect_and_save(frame.pixels, args, output, path.name)
                if args.estimate_body_regions:
                    pose_output = args.pose_output or Path("data/previews") / (
                        f"{path.stem}_frame_{args.frame_index:06d}_body_regions.png"
                    )
                    _estimate_body_regions_and_save(frame.pixels, args, pose_output, path.name)
                if not (args.detect_persons or args.estimate_body_regions):
                    saved_path = save_frame_preview(frame, output)
                    print(
                        f"Preview: {saved_path}\n"
                        f"Frame index: {frame.index}\n"
                        f"Timestamp: {frame.timestamp_seconds:.3f} s"
                    )
        return 0
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"vantasight: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
