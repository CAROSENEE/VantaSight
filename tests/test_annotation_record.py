import unittest

import numpy as np

from tools.annotate_video_frames import _build_annotation_record
from vantasight.input import VideoFrame, frame_sha256


class AnnotationRecordTests(unittest.TestCase):
    def test_record_preserves_manual_labels_and_binds_them_to_frame(self):
        pixels = np.zeros((108, 192, 3), dtype=np.uint8)
        frame = VideoFrame(index=600, timestamp_seconds=5.0, pixels=pixels)
        box = (20, 15, 90, 100)
        regions = {0: {"head": (54, 25), "chest": (54, 46)}}
        suggestion = {"opponent_boxes_xyxy": [list(box)]}

        record = _build_annotation_record(
            frame, "vd-1.mp4", "opponent_visible", [box], regions, suggestion
        )

        self.assertEqual(record["frame_index"], 600)
        self.assertEqual(record["source_frame_sha256"], frame_sha256(pixels))
        self.assertEqual(record["status"], "opponent_visible")
        self.assertEqual(record["opponent_boxes_xyxy"], [[20, 15, 90, 100]])
        self.assertTrue(record["reviewed_manually"])
        self.assertTrue(record["provisional_prefill_used"])
        self.assertEqual(
            record["opponent_body_regions_xy"],
            [{"box_index": 0, "points": {"head": [54, 25], "chest": [54, 46]}}],
        )


if __name__ == "__main__":
    unittest.main()
