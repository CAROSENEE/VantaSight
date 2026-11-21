import json
import unittest
from pathlib import Path

from vantasight.dashboard import load_tracking_report, summarize_alignment_errors


class ReportViewerTests(unittest.TestCase):
    def test_loads_supported_tracking_report(self) -> None:
        path = Path("reports/.test_dashboard_report.json")
        report = {
            "schema_version": 1,
            "source": {"file_name": "vd-1.mp4", "width": 1920, "height": 1080, "fps": 120.0},
            "summary": {
                "sampled_frames": 1,
                "candidate_detections": 0,
                "sampled_frames_per_second": 30.0,
                "detector_latency_mean_ms": 12.0,
                "detector_latency_p95_ms": 15.0,
                "pose_latency_mean_ms": None,
                "pose_latency_p95_ms": None,
            },
            "frames": [{"frame_index": 0, "timestamp_seconds": 0.0, "detection_count": 0, "detections": []}],
        }
        try:
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(load_tracking_report(path), report)
        finally:
            path.unlink(missing_ok=True)

    def test_rejects_unsupported_report_schema(self) -> None:
        path = Path("reports/.test_dashboard_report.json")
        try:
            path.write_text('{"schema_version": 2}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_tracking_report(path)
        finally:
            path.unlink(missing_ok=True)

    def test_rejects_confidence_summary_with_wrong_count(self) -> None:
        path = Path("reports/.test_dashboard_report.json")
        report = {
            "schema_version": 1,
            "source": {"file_name": "ss-1.png", "width": 10, "height": 10, "fps": 0.0},
            "summary": {
                "sampled_frames": 1,
                "candidate_detections": 1,
                "sampled_frames_per_second": 1.0,
                "detector_latency_mean_ms": 1.0,
                "detector_latency_p95_ms": 1.0,
                "detector_confidence": {
                    "count": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None
                },
            },
            "frames": [{"frame_index": 0, "timestamp_seconds": 0.0, "detection_count": 1, "detections": [
                {"track_id": 1, "detector_score": 0.5, "box_xyxy": [1, 1, 2, 2]}
            ]}],
        }
        try:
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must match candidate_detections"):
                load_tracking_report(path)
        finally:
            path.unlink(missing_ok=True)

    def test_rejects_source_path_traversal(self) -> None:
        path = Path("reports/.test_dashboard_report.json")
        report = {
            "schema_version": 1,
            "source": {"file_name": "..\\outside.mp4", "width": 10, "height": 10, "fps": 30.0},
            "summary": {
                "sampled_frames": 0,
                "candidate_detections": 0,
                "sampled_frames_per_second": 0.0,
                "detector_latency_mean_ms": 0.0,
                "detector_latency_p95_ms": 0.0,
            },
            "frames": [],
        }
        try:
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_tracking_report(path)
        finally:
            path.unlink(missing_ok=True)

    def test_summarizes_alignment_error_distribution_by_region(self) -> None:
        report = {
            "frames": [
                {"detections": [{"body_regions": [{"name": "head", "alignment": {
                    "distance_pixels": 10.0, "distance_normalized": 0.01
                }}]}]},
                {"detections": [{"body_regions": [{"name": "head", "alignment": {
                    "distance_pixels": 30.0, "distance_normalized": 0.03
                }}]}]},
            ]
        }
        summary = summarize_alignment_errors(report)
        self.assertEqual(summary[0]["measurements"], 2)
        self.assertEqual(summary[0]["median_error_pixels"], 20.0)
        self.assertEqual(summary[0]["p95_error_pixels"], 30.0)


if __name__ == "__main__":
    unittest.main()
