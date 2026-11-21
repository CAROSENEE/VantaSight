"""Render experimental keypoints and coarse body-region estimates."""

import cv2
import numpy as np

from vantasight.vision.body_region_estimator import PoseObservation


def render_pose_estimates(image_bgr: np.ndarray, observations: list[PoseObservation]) -> np.ndarray:
    """Draw source-coordinate person boxes, keypoints, and region centers."""

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
            cv2.putText(
                rendered,
                f"{keypoint.name}:{keypoint.confidence:.2f}",
                (keypoint.x + 4, max(12, keypoint.y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        for region in observation.body_regions:
            rx, ry = region.center_xy
            cv2.drawMarker(rendered, (rx, ry), (0, 170, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(
                rendered,
                f"{region.name}_est:{region.confidence:.2f}",
                (rx + 8, ry + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 170, 255),
                1,
                cv2.LINE_AA,
            )
    return rendered
