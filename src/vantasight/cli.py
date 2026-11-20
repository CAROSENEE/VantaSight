"""Command-line input inspector for VantaSight."""

import argparse
from pathlib import Path

from vantasight.input import VideoReader, load_image
from vantasight.visualization.frame_preview import save_frame_preview

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _format_duration(seconds: float | None) -> str:
    return "unknown" if seconds is None else f"{seconds:.3f} s"


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
        help="Preview image path (default: data/previews/<video>_frame_<index>.png)",
    )
    args = parser.parse_args()
    path: Path = args.input

    try:
        if path.suffix.lower() in IMAGE_SUFFIXES:
            if args.frame_index is not None or args.output is not None:
                parser.error("--frame-index and --output can only be used with a video")
            image = load_image(path)
            print(f"Type: image\nPath: {image.path}\nResolution: {image.width}x{image.height}")
            return 0

        if args.output is not None and args.frame_index is None:
            parser.error("--output requires --frame-index")

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
