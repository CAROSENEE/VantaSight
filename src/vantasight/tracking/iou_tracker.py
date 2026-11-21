"""Short-term IoU association for person candidates in sampled video frames."""

from dataclasses import dataclass

from vantasight.vision.person_detector import Detection


@dataclass
class _TrackState:
    track_id: int
    box_xyxy: tuple[int, int, int, int]
    center_xy: tuple[float, float]
    last_frame_index: int
    last_timestamp_seconds: float
    missed_updates: int = 0


@dataclass(frozen=True)
class TrackedDetection:
    """A current detection with a temporary track ID and optional image velocity."""

    track_id: int
    detection: Detection
    velocity_xy_pixels_per_second: tuple[float, float] | None


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


class IoUTracker:
    """Associate nearby boxes over time; IDs are temporary and not identity claims."""

    def __init__(self, iou_threshold: float = 0.30, max_missed_updates: int = 1) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be greater than 0 and at most 1")
        if max_missed_updates < 0:
            raise ValueError("max_missed_updates must be zero or greater")
        self.iou_threshold = iou_threshold
        self.max_missed_updates = max_missed_updates
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        timestamp_seconds: float,
    ) -> list[TrackedDetection]:
        """Match this frame's detections and estimate center velocity in px/s."""

        track_ids = list(self._tracks)
        pairs = sorted(
            (
                (_iou(self._tracks[track_id].box_xyxy, detection.box_xyxy), track_id, index)
                for track_id in track_ids
                for index, detection in enumerate(detections)
            ),
            reverse=True,
        )
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        assignments: dict[int, tuple[int, tuple[float, float] | None]] = {}

        for overlap, track_id, detection_index in pairs:
            if overlap < self.iou_threshold:
                break
            if track_id in matched_tracks or detection_index in matched_detections:
                continue

            track = self._tracks[track_id]
            detection = detections[detection_index]
            current_center = _center(detection.box_xyxy)
            delta_t = timestamp_seconds - track.last_timestamp_seconds
            velocity = None
            if delta_t > 0:
                velocity = (
                    (current_center[0] - track.center_xy[0]) / delta_t,
                    (current_center[1] - track.center_xy[1]) / delta_t,
                )
            assignments[detection_index] = (track_id, velocity)
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            track.box_xyxy = detection.box_xyxy
            track.center_xy = current_center
            track.last_frame_index = frame_index
            track.last_timestamp_seconds = timestamp_seconds
            track.missed_updates = 0

        for track_id in track_ids:
            if track_id in matched_tracks:
                continue
            track = self._tracks[track_id]
            track.missed_updates += 1
            if track.missed_updates > self.max_missed_updates:
                del self._tracks[track_id]

        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            center = _center(detection.box_xyxy)
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                box_xyxy=detection.box_xyxy,
                center_xy=center,
                last_frame_index=frame_index,
                last_timestamp_seconds=timestamp_seconds,
            )
            assignments[index] = (track_id, None)

        return [
            TrackedDetection(
                track_id=assignments[index][0],
                detection=detection,
                velocity_xy_pixels_per_second=assignments[index][1],
            )
            for index, detection in enumerate(detections)
        ]
