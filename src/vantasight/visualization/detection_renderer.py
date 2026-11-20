"""Draw neutral person-candidate annotations on a copy of the source image."""

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from vantasight.vision.person_detector import Detection


def render_detections(image_bgr: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    """Draw candidate boxes without changing the source image or claiming enemy identity."""

    rendered = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.box_xyxy
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (50, 220, 80), 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        text_y = max(18, y1 - 8)
        cv2.putText(
            rendered,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (50, 220, 80),
            2,
            cv2.LINE_AA,
        )
    return rendered


def save_annotated_image(image_bgr: np.ndarray, output_path: str | Path) -> Path:
    """Save an annotated image safely when output paths contain Unicode characters."""

    destination = Path(output_path).expanduser()
    extension = destination.suffix.lower()
    if extension not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        raise ValueError(f"Unsupported output image extension: {extension or '(missing)'}")

    success, encoded = cv2.imencode(extension, image_bgr)
    if not success:
        raise OSError(f"Could not encode annotated image: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded.tofile(destination)
    except OSError as exc:
        raise OSError(f"Could not save annotated image: {destination}") from exc
    return destination.resolve()
