"""Render experimental keypoints and coarse body-region estimates."""

import cv2
import numpy as np

from vantasight.vision.body_region_estimator import PoseObservation


_REGION_COLORS = {
    "head": (255, 190, 0),
    "chest": (0, 180, 255),
    "lower_torso": (0, 240, 240),
    "legs": (255, 80, 180),
}


def render_pose_estimates(
    image_bgr: np.ndarray,
    observations: list[PoseObservation],
    crosshair_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    """Draw source-coordinate person boxes, keypoints, region centers, and reference."""

    rendered = image_bgr.copy()
    for observation in observations:
        x1, y1, x2, y2 = observation.box_xyxy
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (40, 210, 240), 2)
        cv2.putText(
            rendered,
            f"person candidate {observation.person_confidence:.2f}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 210, 240),
            1,
            cv2.LINE_AA,
        )
        for keypoint in observation.keypoints:
            cv2.circle(rendered, (keypoint.x, keypoint.y), 4, (230, 60, 220), -1)
        for region in observation.body_regions:
            rx, ry = region.center_xy
            cv2.drawMarker(
                rendered,
                (rx, ry),
                _REGION_COLORS.get(region.name, (255, 255, 255)),
                cv2.MARKER_CROSS,
                14,
                2,
            )
    if observations:
        cv2.rectangle(rendered, (8, 8), (min(image_bgr.shape[1] - 8, 980), 38), (20, 20, 20), -1)
        cv2.putText(
            rendered,
            "Magenta dots: keypoints | region centers:",
            (16, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        x = 342
        for region_name, color in _REGION_COLORS.items():
            cv2.circle(rendered, (x, 23), 5, color, -1)
            cv2.putText(
                rendered,
                region_name,
                (x + 8, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            x += 105 if region_name != "lower_torso" else 130
    if crosshair_xy is not None:
        reference = (round(crosshair_xy[0]), round(crosshair_xy[1]))
        cv2.drawMarker(rendered, reference, (40, 40, 240), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(
            rendered,
            "crosshair reference",
            (reference[0] + 10, max(18, reference[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 40, 240),
            1,
            cv2.LINE_AA,
        )
    return rendered
