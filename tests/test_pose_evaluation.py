import unittest
import json
from pathlib import Path

from vantasight.analytics import compare_body_region_points, summarize_region_localizations
from tools.evaluate_pose_regions import _load_region_annotations


class PoseEvaluationTests(unittest.TestCase):
    def test_computes_pixel_and_normalized_localization_error(self) -> None:
        rows = compare_body_region_points(
            {"head": (0.0, 0.0)}, {"head": (3.0, 4.0)}, width=3, height=4
        )
        self.assertEqual(rows[0].error_pixels, 5.0)
        self.assertEqual(rows[0].error_normalized, 1.0)

    def test_missing_model_region_reduces_coverage_without_fake_error(self) -> None:
        rows = compare_body_region_points(
            {"head": (10.0, 10.0), "chest": (10.0, 20.0)}, {"head": (10.0, 10.0)}, 100, 100
        )
        summaries = {item.region_name: item for item in summarize_region_localizations(rows)}
        self.assertEqual(summaries["head"].coverage, 1.0)
        self.assertEqual(summaries["head"].median_error_pixels, 0.0)
        self.assertEqual(summaries["chest"].coverage, 0.0)
        self.assertIsNone(summaries["chest"].median_error_pixels)

    def test_rejects_ground_truth_point_outside_frame(self) -> None:
        with self.assertRaises(ValueError):
            compare_body_region_points({"head": (11.0, 5.0)}, {}, 10, 10)

    def test_reads_optional_body_points_from_manual_annotation_jsonl(self) -> None:
        path = Path("reports/.test_pose_annotations.jsonl")
        record = {
            "video": "vd-1.mp4",
            "frame_index": 3000,
            "status": "opponent_visible",
            "opponent_boxes_xyxy": [[10, 20, 30, 60]],
            "opponent_body_regions_xy": [{"box_index": 0, "points": {"head": [20, 25]}}],
        }
        try:
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            rows = _load_region_annotations(path, "vd-1.mp4")
            self.assertEqual(rows[0]["opponent_body_regions_xy"][0]["points"]["head"], [20, 25])
        finally:
            path.unlink(missing_ok=True)

    def test_rejects_body_point_outside_manual_opponent_box(self) -> None:
        path = Path("reports/.test_pose_annotations.jsonl")
        record = {
            "video": "vd-1.mp4",
            "frame_index": 3000,
            "status": "opponent_visible",
            "opponent_boxes_xyxy": [[10, 20, 30, 60]],
            "opponent_body_regions_xy": [{"box_index": 0, "points": {"head": [35, 25]}}],
        }
        try:
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _load_region_annotations(path, "vd-1.mp4")
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
