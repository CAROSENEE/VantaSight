import unittest
from pathlib import Path

from vantasight.dashboard.evaluation_summary import load_detector_evaluation, load_pose_evaluation


class EvaluationSummaryTests(unittest.TestCase):
    def test_detector_summary_aggregates_only_the_selected_video(self) -> None:
        path = Path("reports/.test_detector_summary.csv")
        try:
            path.write_text(
                "video,frame_index,true_positives,false_positives,false_negatives\n"
                "vd-1.mp4,1,2,1,3\nvd-2.mp4,2,10,0,0\n",
                encoding="utf-8",
            )
            summary = load_detector_evaluation(path, "vd-1.mp4")
            self.assertIsNotNone(summary)
            assert summary is not None
            self.assertEqual(summary.scored_frames, 1)
            self.assertEqual(summary.counts.true_positives, 2)
            self.assertEqual(summary.precision, 2 / 3)
            self.assertEqual(summary.recall, 2 / 5)
        finally:
            path.unlink(missing_ok=True)

    def test_pose_summary_reports_coverage_and_median_error(self) -> None:
        path = Path("reports/.test_pose_summary.csv")
        try:
            path.write_text(
                "video,region,error_pixels,error_normalized\n"
                "vd-1.mp4,head,10,0.01\nvd-1.mp4,head,,\nvd-1.mp4,chest,20,0.02\n",
                encoding="utf-8",
            )
            summaries = load_pose_evaluation(path, "vd-1.mp4")
            self.assertIsNotNone(summaries)
            assert summaries is not None
            head = next(item for item in summaries if item.region_name == "head")
            self.assertEqual(head.coverage, 0.5)
            self.assertEqual(head.median_error_pixels, 10.0)
            self.assertEqual(head.median_error_normalized, 0.01)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
