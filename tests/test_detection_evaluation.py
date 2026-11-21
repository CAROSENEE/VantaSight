import unittest

from vantasight.analytics import (
    DetectionCounts,
    associate_pose_observations,
    calculate_metrics,
    evaluate_threshold_sweep,
    match_boxes,
)
from vantasight.vision.body_region_estimator import PoseObservation


class DetectionEvaluationTests(unittest.TestCase):
    def test_threshold_sweep_reuses_predictions_and_updates_counts(self):
        samples = [
            ([(0.9, (0, 0, 10, 10)), (0.4, (20, 20, 30, 30))], [(0, 0, 10, 10)])
        ]
        low, high = evaluate_threshold_sweep(samples, [0.25, 0.75], 0.5)
        self.assertEqual((low["true_positives"], low["false_positives"], low["false_negatives"]), (1, 1, 0))
        self.assertEqual((high["true_positives"], high["false_positives"], high["false_negatives"]), (1, 0, 0))
        self.assertEqual(high["f1"], 1.0)

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

    def test_pose_association_keeps_maximum_number_of_candidate_matches(self) -> None:
        candidates = [(0, 0, 9, 10), (0, 0, 8, 10)]
        observations = [
            PoseObservation((0, 0, 10, 10), 0.9, (), ()),
            PoseObservation((2, 0, 12, 10), 0.8, (), ()),
        ]
        matched = associate_pose_observations(candidates, observations, 0.55)
        self.assertEqual(len(matched), 2)
        self.assertEqual(len({id(value) for value in matched.values()}), 2)


if __name__ == "__main__":
    unittest.main()
