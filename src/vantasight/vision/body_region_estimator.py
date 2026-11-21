"""Experimental COCO keypoint estimates for offline human-shaped candidates."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class KeypointEstimate:
    """One COCO keypoint and its raw heatmap response in source-image pixels."""

    name: str
    x: int
    y: int
    heatmap_score: float


@dataclass(frozen=True)
class BodyRegionEstimate:
    """A coarse center computed from the available keypoints for one region."""

    name: str
    center_xy: tuple[int, int]
    mean_keypoint_score: float
    supporting_keypoints: tuple[str, ...]


@dataclass(frozen=True)
class PoseObservation:
    """One COCO person candidate with source-coordinate keypoints."""

    box_xyxy: tuple[int, int, int, int]
    person_confidence: float
    keypoints: tuple[KeypointEstimate, ...]
    body_regions: tuple[BodyRegionEstimate, ...]


_REGION_KEYPOINTS = {
    "head": ("nose", "left_eye", "right_eye", "left_ear", "right_ear"),
    "chest": ("left_shoulder", "right_shoulder"),
    "lower_torso": ("left_hip", "right_hip"),
    "legs": ("left_knee", "right_knee", "left_ankle", "right_ankle"),
}


class TorchvisionBodyRegionEstimator:
    """Run COCO Keypoint R-CNN; returned regions are estimates, not game labels."""

    def __init__(
        self,
        person_confidence_threshold: float = 0.25,
        keypoint_score_threshold: float = 0.0,
        device: str = "auto",
        min_size: int = 1080,
        max_size: int = 1920,
        excluded_regions: Sequence[tuple[int, int, int, int]] = (),
        preserve_source_resolution: bool = False,
    ) -> None:
        if not 0.0 <= person_confidence_threshold <= 1.0:
            raise ValueError("person_confidence_threshold must be between 0 and 1")
        if keypoint_score_threshold < 0.0:
            raise ValueError("keypoint_score_threshold must be zero or greater")
        if min_size < 1 or max_size < min_size:
            raise ValueError("Expected 1 <= min_size <= max_size")

        project_root = Path(__file__).resolve().parents[3]
        os.environ.setdefault("TORCH_HOME", str(project_root / "models" / "cache"))
        try:
            import torch
            from torchvision.models.detection import (
                KeypointRCNN_ResNet50_FPN_Weights,
                keypointrcnn_resnet50_fpn,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Body-region estimation requires the optional 'torch' and 'torchvision' "
                "packages. Follow the vision setup instructions in README.md."
            ) from exc

        if device == "auto":
            resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
            if resolved_device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")

        self._torch = torch
        self.device = torch.device(resolved_device)
        self.person_confidence_threshold = person_confidence_threshold
        self.keypoint_score_threshold = keypoint_score_threshold
        self.excluded_regions = tuple(excluded_regions)
        self.preserve_source_resolution = preserve_source_resolution
        self._weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
        self.keypoint_names = tuple(self._weights.meta["keypoint_names"])

        try:
            self._model = keypointrcnn_resnet50_fpn(
                weights=self._weights,
                min_size=min_size,
                max_size=max_size,
                box_score_thresh=0.0,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load the TorchVision Keypoint R-CNN COCO_V1 weights. "
                "Check the model cache and network access; details: " + str(exc)
            ) from exc
        self._model.to(self.device)
        self._model.eval()
        self._person_class_id = self._weights.meta["categories"].index("person")

    def estimate(self, image_bgr: np.ndarray) -> list[PoseObservation]:
        """Return person keypoints and coarse region centers in source pixels."""

        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR image")
        height, width = image_bgr.shape[:2]
        if self.preserve_source_resolution:
            self._model.transform.min_size = (min(height, width),)
            self._model.transform.max_size = max(height, width)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = (
            self._torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .to(dtype=self._torch.float32)
            .div_(255.0)
        )

        with self._torch.inference_mode():
            prediction = self._model([image_tensor.to(self.device)])[0]

        boxes = prediction["boxes"].detach().cpu().numpy()
        scores = prediction["scores"].detach().cpu().numpy()
        labels = prediction["labels"].detach().cpu().numpy()
        keypoints = prediction["keypoints"].detach().cpu().numpy()
        keypoint_scores_tensor = prediction.get("keypoints_scores")
        if keypoint_scores_tensor is None:
            raise RuntimeError(
                "This TorchVision build did not return raw keypoint heatmap scores; "
                "refusing to treat the keypoint visibility flag as confidence."
            )
        keypoint_scores = keypoint_scores_tensor.detach().cpu().numpy()

        observations: list[PoseObservation] = []
        for box, person_score, label, points, point_scores in zip(
            boxes, scores, labels, keypoints, keypoint_scores
        ):
            person_confidence = float(person_score)
            if int(label) != self._person_class_id or person_confidence < self.person_confidence_threshold:
                continue

            x1 = max(0, min(width, int(round(float(box[0])))))
            y1 = max(0, min(height, int(round(float(box[1])))))
            x2 = max(0, min(width, int(round(float(box[2])))))
            y2 = max(0, min(height, int(round(float(box[3])))))
            if x2 <= x1 or y2 <= y1:
                continue
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            if any(rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2
                   for rx1, ry1, rx2, ry2 in self.excluded_regions):
                continue

            estimates: list[KeypointEstimate] = []
            for name, point, confidence in zip(self.keypoint_names, points, point_scores):
                score = float(confidence)
                if score < self.keypoint_score_threshold:
                    continue
                px = max(0, min(width, int(round(float(point[0])))))
                py = max(0, min(height, int(round(float(point[1])))))
                estimates.append(KeypointEstimate(name=name, x=px, y=py, heatmap_score=score))

            by_name = {point.name: point for point in estimates}
            regions: list[BodyRegionEstimate] = []
            for region_name, names in _REGION_KEYPOINTS.items():
                available = [by_name[name] for name in names if name in by_name]
                if not available:
                    continue
                center = (
                    int(round(sum(point.x for point in available) / len(available))),
                    int(round(sum(point.y for point in available) / len(available))),
                )
                regions.append(
                    BodyRegionEstimate(
                        name=region_name,
                        center_xy=center,
                        mean_keypoint_score=sum(point.heatmap_score for point in available) / len(available),
                        supporting_keypoints=tuple(point.name for point in available),
                    )
                )
            observations.append(
                PoseObservation(
                    box_xyxy=(x1, y1, x2, y2),
                    person_confidence=person_confidence,
                    keypoints=tuple(estimates),
                    body_regions=tuple(regions),
                )
            )
        return observations
