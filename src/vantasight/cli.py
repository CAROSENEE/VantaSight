"""Command-line input inspector for VantaSight."""

import argparse
from pathlib import Path

from vantasight.input import VideoReader, load_image
from vantasight.visualization.frame_preview import save_frame_preview
from vantasight.vision import TorchvisionPersonDetector
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


def _detect_and_save(
    pixels,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    detector = TorchvisionPersonDetector(
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        min_size=min(1080, args.max_side),
        max_size=args.max_side,
        excluded_regions=args.exclude_region or (),
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
        "--exclude-region",
        type=_parse_region,
        action="append",
        help="Ignore detections centered inside x1,y1,x2,y2 source-pixel rectangle; may be repeated",
    )
    args = parser.parse_args()
    path: Path = args.input

    try:
        if path.suffix.lower() in IMAGE_SUFFIXES:
            if args.frame_index is not None:
                parser.error("--frame-index can only be used with a video")
            if (args.output is not None or args.exclude_region) and not args.detect_persons:
                parser.error("--output and --exclude-region require --detect-persons")
            image = load_image(path)
            print(f"Type: image\nPath: {image.path}\nResolution: {image.width}x{image.height}")
            if args.detect_persons:
                output = args.output or Path("data/previews") / f"{path.stem}_detections.png"
                _detect_and_save(image.pixels, args, output)
            return 0

        if args.output is not None and args.frame_index is None:
            parser.error("--output requires --frame-index for video input")
        if args.exclude_region and not args.detect_persons:
            parser.error("--exclude-region requires --detect-persons")
        if args.detect_persons and args.frame_index is None:
            parser.error("--detect-persons on a video requires --frame-index")

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
                    _detect_and_save(frame.pixels, args, output)
                else:
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
