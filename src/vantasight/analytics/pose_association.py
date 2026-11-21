"""One-to-one association between detector boxes and pose observations."""

from __future__ import annotations

from vantasight.vision.body_region_estimator import PoseObservation

Box = tuple[int, int, int, int]


def _iou(first: Box, second: Box) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def associate_pose_observations(
    candidate_boxes: list[Box],
    observations: list[PoseObservation],
    iou_threshold: float = 0.10,
) -> dict[int, PoseObservation]:
    """Map candidate index to pose observation index with max-cardinality IoU matching."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be greater than 0 and at most 1")
    possible: list[list[int]] = []
    for candidate_box in candidate_boxes:
        overlaps = [(_iou(candidate_box, observation.box_xyxy), index)
                    for index, observation in enumerate(observations)]
        possible.append(
            [index for overlap, index in sorted(overlaps, reverse=True) if overlap >= iou_threshold]
        )

    observation_to_candidate: dict[int, int] = {}

    def assign(candidate_index: int, visited: set[int]) -> bool:
        for observation_index in possible[candidate_index]:
            if observation_index in visited:
                continue
            visited.add(observation_index)
            previous = observation_to_candidate.get(observation_index)
            if previous is None or assign(previous, visited):
                observation_to_candidate[observation_index] = candidate_index
                return True
        return False

    for candidate_index in range(len(candidate_boxes)):
        assign(candidate_index, set())
    return {
        candidate_index: observations[observation_index]
        for observation_index, candidate_index in observation_to_candidate.items()
    }
