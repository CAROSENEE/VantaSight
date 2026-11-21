"""Read video metadata and decode frames with source timestamps."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Metadata reported by the video container/decoder."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float | None


@dataclass(frozen=True)
class VideoFrame:
    """A decoded video frame with zero-based index and presentation timestamp."""

    index: int
    timestamp_seconds: float
    pixels: np.ndarray


class VideoReader:
    """Context-managed video reader. Always releases the decoder on exit."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._capture: cv2.VideoCapture | None = None
        self.info: VideoInfo | None = None

    def __enter__(self) -> "VideoReader":
        if not self.path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {self.path}")

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Unsupported or corrupt video: {self.path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 and frame_count > 0 else None

        self._capture = capture
        self.info = VideoInfo(
            path=self.path.resolve(),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
        )
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def frames(self, start_frame: int = 0) -> Iterator[VideoFrame]:
        """Yield decoded frames from a zero-based start index with decoder timestamps."""

        if self._capture is None:
            raise RuntimeError("Use VideoReader inside a 'with' block before reading frames")
        if start_frame < 0:
            raise ValueError("Start frame must be zero or greater")
        if self.info is not None and self.info.frame_count > 0 and start_frame >= self.info.frame_count:
            raise IndexError(
                f"Start frame {start_frame} is outside the video ({self.info.frame_count} frames)"
            )

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        index = start_frame
        while True:
            success, pixels = self._capture.read()
            if not success:
                break
            timestamp = self._capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield VideoFrame(index=index, timestamp_seconds=timestamp, pixels=pixels)
            index += 1

    def read_frame(self, frame_index: int) -> VideoFrame:
        """Read one zero-based frame by index, seeking instead of decoding earlier frames."""

        if self._capture is None:
            raise RuntimeError("Use VideoReader inside a 'with' block before reading frames")
        if frame_index < 0:
            raise ValueError("Frame index must be zero or greater")
        if self.info is not None and self.info.frame_count > 0 and frame_index >= self.info.frame_count:
            raise IndexError(
                f"Frame index {frame_index} is outside the video ({self.info.frame_count} frames)"
            )

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, pixels = self._capture.read()
        if not success:
            raise IndexError(f"Could not decode video frame {frame_index}")

        timestamp = self._capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return VideoFrame(index=frame_index, timestamp_seconds=timestamp, pixels=pixels)

    def sampled_frames(
        self, frame_step: int, start_frame: int = 0, stop_frame: int | None = None
    ) -> Iterator[VideoFrame]:
        """Seek directly to selected frames; stop_frame is exclusive, like range()."""

        if self._capture is None:
            raise RuntimeError("Use VideoReader inside a 'with' block before reading frames")
        if frame_step < 1:
            raise ValueError("Frame step must be a positive integer")
        if start_frame < 0:
            raise ValueError("Start frame must be zero or greater")
        if stop_frame is not None and stop_frame <= start_frame:
            raise ValueError("Stop frame must be greater than start frame")
        if self.info is not None and self.info.frame_count > 0 and start_frame >= self.info.frame_count:
            raise IndexError(
                f"Start frame {start_frame} is outside the video ({self.info.frame_count} frames)"
            )

        index = start_frame
        while stop_frame is None or index < stop_frame:
            try:
                yield self.read_frame(index)
            except IndexError:
                break
            index += frame_step
