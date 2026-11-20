"""Save decoded frames as standalone preview images."""

from pathlib import Path

import cv2

from vantasight.input.video_loader import VideoFrame


def save_frame_preview(frame: VideoFrame, output_path: str | Path) -> Path:
    """Write a video frame as an image, including when the output path has Unicode."""

    destination = Path(output_path).expanduser()
    extension = destination.suffix.lower()
    if extension not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        raise ValueError(f"Unsupported preview image extension: {extension or '(missing)'}")

    success, encoded = cv2.imencode(extension, frame.pixels)
    if not success:
        raise OSError(f"Could not encode preview image: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded.tofile(destination)
    except OSError as exc:
        raise OSError(f"Could not save preview image: {destination}") from exc
    return destination.resolve()
