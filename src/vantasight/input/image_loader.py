"""Load still images while handling Windows paths with non-ASCII characters."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageFrame:
    """An image and its source metadata. Pixels use OpenCV BGR channel order."""

    path: Path
    pixels: np.ndarray
    width: int
    height: int


def load_image(path: str | Path) -> ImageFrame:
    """Read a supported image or raise a useful error if it cannot be decoded."""

    image_path = Path(path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    try:
        encoded = np.fromfile(image_path, dtype=np.uint8)
    except OSError as exc:
        raise OSError(f"Could not read image file: {image_path}") from exc

    pixels = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if pixels is None:
        raise ValueError(f"Unsupported or corrupt image: {image_path}")

    height, width = pixels.shape[:2]
    return ImageFrame(path=image_path.resolve(), pixels=pixels, width=width, height=height)
