import json
import unittest
from pathlib import Path

from vantasight.analytics import write_tracking_report


class TrackingReportTests(unittest.TestCase):
    def test_writes_readable_utf8_json(self) -> None:
        report = {"schema_version": 1, "note": "offline analysis", "frames": []}
        path = Path("reports/.test_tracking_report.json")
        try:
            write_tracking_report(path, report)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), report)
        finally:
            path.unlink(missing_ok=True)

    def test_rejects_non_finite_values(self) -> None:
        path = Path("reports/.test_tracking_report.json")
        try:
            with self.assertRaises(ValueError):
                write_tracking_report(path, {"score": float("nan")})
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
