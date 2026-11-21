import json
import unittest
from pathlib import Path

from vantasight.dashboard import load_tracking_report


class ReportViewerTests(unittest.TestCase):
    def test_loads_supported_tracking_report(self) -> None:
        path = Path("reports/.test_dashboard_report.json")
        report = {
            "schema_version": 1,
            "source": {"file_name": "vd-1.mp4", "width": 1920, "height": 1080, "fps": 120.0},
            "summary": {},
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

    def test_rejects_source_path_traversal(self) -> None:
        path = Path("reports/.test_dashboard_report.json")
        report = {
            "schema_version": 1,
            "source": {"file_name": "..\\outside.mp4", "width": 10, "height": 10, "fps": 30.0},
            "summary": {},
            "frames": [],
        }
        try:
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_tracking_report(path)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
