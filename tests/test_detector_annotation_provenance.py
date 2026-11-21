import json
import unittest
from pathlib import Path

from tools.evaluate_detector import _read_annotations


class DetectorAnnotationProvenanceTests(unittest.TestCase):
    def test_rejects_unreviewed_labels_by_default(self):
        path = Path("reports/.test_detector_annotations.jsonl")
        record = {
            "video": "vd-1.mp4",
            "frame_index": 0,
            "status": "no_opponent_visible",
            "opponent_boxes_xyxy": [],
        }
        try:
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unreviewed annotation"):
                _read_annotations(path, "vd-1.mp4")
            exploratory = _read_annotations(path, "vd-1.mp4", allow_unreviewed=True)
            self.assertEqual(exploratory[0]["_label_provenance"], "unreviewed_exploratory")
        finally:
            path.unlink(missing_ok=True)

    def test_accepts_interactive_tool_review_marker(self):
        path = Path("reports/.test_detector_annotations.jsonl")
        record = {
            "video": "vd-1.mp4",
            "frame_index": 0,
            "status": "no_opponent_visible",
            "opponent_boxes_xyxy": [],
            "reviewed_manually": True,
        }
        try:
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            rows = _read_annotations(path, "vd-1.mp4")
            self.assertEqual(rows[0]["_label_provenance"], "manually_reviewed")
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
