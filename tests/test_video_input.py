import unittest
from pathlib import Path

import cv2
import numpy as np

from vantasight.input import VideoReader, frame_sha256, verify_frame_fingerprint


class VideoReaderSamplingTests(unittest.TestCase):
    def _make_video(self, path: Path, frame_count: int = 12) -> None:
        writer = cv2.VideoWriter(
            str(path), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
        )
        if not writer.isOpened():
            writer.release()
            self.skipTest("OpenCV MJPG video writer is unavailable")
        try:
            for index in range(frame_count):
                frame = np.full((24, 32, 3), index * 15, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

    def test_seeks_sampled_indexes_and_preserves_timestamps(self):
        video = Path("reports/.test_video_reader_sampling.avi")
        self.addCleanup(lambda: video.unlink(missing_ok=True))
        self._make_video(video)
        with VideoReader(video) as reader:
            frames = list(reader.sampled_frames(frame_step=3, start_frame=2, stop_frame=10))
        self.assertEqual([frame.index for frame in frames], [2, 5, 8])
        for frame in frames:
            self.assertAlmostEqual(frame.timestamp_seconds, frame.index / 10.0, delta=0.15)
            expected = frame.index * 15
            self.assertAlmostEqual(float(frame.pixels.mean()), expected, delta=8.0)

    def test_rejects_invalid_sampling_parameters(self):
        video = Path("reports/.test_video_reader_validation.avi")
        self.addCleanup(lambda: video.unlink(missing_ok=True))
        self._make_video(video)
        with VideoReader(video) as reader:
            with self.assertRaises(ValueError):
                list(reader.sampled_frames(frame_step=0))
            with self.assertRaises(ValueError):
                list(reader.sampled_frames(frame_step=1, start_frame=-1))
            with self.assertRaises(ValueError):
                list(reader.sampled_frames(frame_step=1, start_frame=2, stop_frame=2))

    def test_annotation_fingerprint_matches_redecoded_video_frame(self):
        video = Path("reports/.test_video_reader_fingerprint.avi")
        self.addCleanup(lambda: video.unlink(missing_ok=True))
        self._make_video(video)
        with VideoReader(video) as reader:
            labeled_frame = reader.read_frame(5)
            same_frame = reader.read_frame(5)
            different_frame = reader.read_frame(6)
        record = {"source_frame_sha256": frame_sha256(labeled_frame.pixels)}
        verify_frame_fingerprint(record, same_frame.pixels, 5)
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            verify_frame_fingerprint(record, different_frame.pixels, 5)


if __name__ == "__main__":
    unittest.main()
