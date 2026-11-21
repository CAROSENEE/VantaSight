"""Render temporary tracking IDs over offline video frames."""

import cv2
import numpy as np

from vantasight.tracking import TrackedDetection


def render_tracks(image_bgr: np.ndarray, tracks: list[TrackedDetection]) -> np.ndarray:
    """Draw candidate boxes and temporary IDs on a copy of the frame."""

    rendered = image_bgr.copy()
    for tracked in tracks:
        x1, y1, x2, y2 = tracked.detection.box_xyxy
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (40, 220, 80), 2)
        label = f"ID {tracked.track_id} | person {tracked.detection.confidence:.2f}"
        if tracked.velocity_xy_pixels_per_second is not None:
            vx, vy = tracked.velocity_xy_pixels_per_second
            label += f" | v=({vx:.0f},{vy:.0f}) px/s"
        cv2.putText(
            rendered,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 220, 80),
            1,
            cv2.LINE_AA,
        )
    return rendered
