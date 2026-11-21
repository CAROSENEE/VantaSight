import unittest
from pathlib import Path

from vantasight.dashboard.evaluation_summary import (
    load_detector_evaluation,
    load_pose_evaluation,
    load_threshold_sweep,
)


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

    def test_threshold_sweep_loads_only_selected_video_and_keeps_provenance(self) -> None:
        path = Path("reports/.test_threshold_sweep.csv")
        try:
            path.write_text(
                "video,confidence_threshold,scored_frames,true_positives,false_positives,false_negatives,"
                "precision,recall,f1,iou_threshold,label_provenance,unreviewed_scored_frames\n"
                "vd-1.mp4,0.25,8,1,1,0,0.5,1,0.6666666666666666,0.5,unreviewed_exploratory,8\n"
                "vd-2.mp4,0.25,8,8,0,0,1,1,1,0.5,manually_reviewed,0\n",
                encoding="utf-8",
            )
            points = load_threshold_sweep(path, "vd-1.mp4")
            self.assertIsNotNone(points)
            assert points is not None
            self.assertEqual(len(points), 1)
            self.assertEqual(points[0].scored_frames, 8)
            self.assertEqual(points[0].label_provenance, "unreviewed_exploratory")
            self.assertAlmostEqual(points[0].f1, 2 / 3)
        finally:
            path.unlink(missing_ok=True)

    def test_threshold_sweep_rejects_metrics_that_do_not_match_counts(self) -> None:
        path = Path("reports/.test_threshold_sweep_invalid.csv")
        try:
            path.write_text(
                "video,confidence_threshold,scored_frames,true_positives,false_positives,false_negatives,"
                "precision,recall,f1,iou_threshold,label_provenance,unreviewed_scored_frames\n"
                "vd-1.mp4,0.25,1,1,1,0,0.9,1,0.95,0.5,unreviewed_exploratory,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match counts"):
                load_threshold_sweep(path, "vd-1.mp4")
        finally:
            path.unlink(missing_ok=True)

    def test_threshold_sweep_rejects_non_monotonic_detection_counts(self) -> None:
        path = Path("reports/.test_threshold_sweep_non_monotonic.csv")
        try:
            path.write_text(
                "video,confidence_threshold,scored_frames,true_positives,false_positives,false_negatives,"
                "precision,recall,f1,iou_threshold,label_provenance,unreviewed_scored_frames\n"
                "vd-1.mp4,0.25,1,1,0,1,1,0.5,0.6666666666666666,0.5,unreviewed_exploratory,1\n"
                "vd-1.mp4,0.50,1,2,0,0,1,1,1,0.5,unreviewed_exploratory,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "increase with confidence threshold"):
                load_threshold_sweep(path, "vd-1.mp4")
        finally:
            path.unlink(missing_ok=True)

    def test_threshold_sweep_rejects_mixed_iou_settings(self) -> None:
        path = Path("reports/.test_threshold_sweep_mixed_iou.csv")
        try:
            path.write_text(
                "video,confidence_threshold,scored_frames,true_positives,false_positives,false_negatives,"
                "precision,recall,f1,iou_threshold,label_provenance,unreviewed_scored_frames\n"
                "vd-1.mp4,0.25,1,1,0,0,1,1,1,0.50,unreviewed_exploratory,1\n"
                "vd-1.mp4,0.50,1,1,0,0,1,1,1,0.75,unreviewed_exploratory,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Inconsistent threshold sweep"):
                load_threshold_sweep(path, "vd-1.mp4")
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
