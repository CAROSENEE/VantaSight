"""COCO-pretrained person-candidate detector using TorchVision Faster R-CNN."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """One person-class candidate in original source-image pixel coordinates."""

    class_name: str
    confidence: float
    box_xyxy: tuple[int, int, int, int]


class TorchvisionPersonDetector:
    """Run a COCO-pretrained detector; it reports people, not enemy/team identity."""

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        device: str = "auto",
        min_size: int = 1080,
        max_size: int = 1920,
        excluded_regions: Sequence[tuple[int, int, int, int]] = (),
        preserve_source_resolution: bool = False,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if min_size < 1 or max_size < min_size:
            raise ValueError("Expected 1 <= min_size <= max_size")

        project_root = Path(__file__).resolve().parents[3]
        os.environ.setdefault("TORCH_HOME", str(project_root / "models" / "cache"))

        try:
            import torch
            from torchvision.models.detection import (
                FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                fasterrcnn_mobilenet_v3_large_fpn,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Person detection requires the optional 'torch' and 'torchvision' "
                "packages. Follow the detector setup instructions in README.md."
            ) from exc

        if device == "auto":
            resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
            if resolved_device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")

        self._torch = torch
        self.device = torch.device(resolved_device)
        self.confidence_threshold = confidence_threshold
        self.excluded_regions = tuple(excluded_regions)
        self.preserve_source_resolution = preserve_source_resolution
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        self._person_class_id = weights.meta["categories"].index("person")

        try:
            self._model = fasterrcnn_mobilenet_v3_large_fpn(
                weights=weights,
                min_size=min_size,
                max_size=max_size,
                box_score_thresh=0.0,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load the TorchVision COCO_V1 weights. Check the model "
                "cache and network access; details: " + str(exc)
            ) from exc

        self._model.to(self.device)
        self._model.eval()

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Return person candidates, mapped to the dimensions of the input frame."""

        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR image")
        height, width = image_bgr.shape[:2]
        if self.preserve_source_resolution:
            # TorchVision normally rescales to configured min_size/max_size. For
            # small distant targets, allow inference at the original pixel size.
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

        results: list[Detection] = []
        for box, score, label in zip(
            prediction["boxes"].detach().cpu().numpy(),
            prediction["scores"].detach().cpu().numpy(),
            prediction["labels"].detach().cpu().numpy(),
        ):
            if int(label) != self._person_class_id or float(score) < self.confidence_threshold:
                continue

            x1 = max(0, min(width, int(round(float(box[0])))))
            y1 = max(0, min(height, int(round(float(box[1])))))
            x2 = max(0, min(width, int(round(float(box[2])))))
            y2 = max(0, min(height, int(round(float(box[3])))))
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            if any(rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2
                   for rx1, ry1, rx2, ry2 in self.excluded_regions):
                continue
            if x2 <= x1 or y2 <= y1:
                continue

            results.append(
                Detection(
                    class_name="person candidate",
                    confidence=float(score),
                    box_xyxy=(x1, y1, x2, y2),
                )
            )

        return results
