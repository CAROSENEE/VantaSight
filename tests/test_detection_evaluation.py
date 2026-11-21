import unittest

from vantasight.analytics import DetectionCounts, calculate_metrics, match_boxes


class DetectionEvaluationTests(unittest.TestCase):
    def test_matching_maximizes_true_positives_not_only_best_single_overlap(self) -> None:
        predictions = [(0, 0, 9, 10), (0, 0, 8, 10)]
        ground_truth = [(0, 0, 10, 10), (2, 0, 12, 10)]
        counts = match_boxes(predictions, ground_truth, 0.55)
        self.assertEqual(counts, DetectionCounts(2, 0, 0))

    def test_extra_prediction_and_unmatched_truth_are_counted(self) -> None:
        counts = match_boxes([(0, 0, 10, 10), (20, 20, 30, 30)], [(0, 0, 10, 10), (40, 40, 50, 50)], 0.5)
        self.assertEqual(counts, DetectionCounts(1, 1, 1))

    def test_empty_negative_sample_counts_false_positives(self) -> None:
        counts = match_boxes([(0, 0, 10, 10)], [], 0.5)
        self.assertEqual(counts, DetectionCounts(0, 1, 0))
        self.assertEqual(calculate_metrics(counts).precision, 0.0)
        self.assertIsNone(calculate_metrics(counts).recall)

    def test_empty_perfectly_negative_sample_keeps_metrics_undefined(self) -> None:
        metrics = calculate_metrics(DetectionCounts(0, 0, 0))
        self.assertIsNone(metrics.precision)
        self.assertIsNone(metrics.recall)
        self.assertIsNone(metrics.f1)


if __name__ == "__main__":
    unittest.main()
