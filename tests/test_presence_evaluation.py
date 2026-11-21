import unittest

from vantasight.analytics import PresenceCounts, calculate_presence_metrics


class PresenceEvaluationTests(unittest.TestCase):
    def test_calculates_binary_presence_metrics(self):
        metrics = calculate_presence_metrics(PresenceCounts(1, 1, 0, 0))
        self.assertEqual(metrics.accuracy, 0.5)
        self.assertEqual(metrics.precision, 0.5)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.specificity, 0.0)
        self.assertAlmostEqual(metrics.f1, 2 / 3)

    def test_empty_sample_keeps_metrics_undefined(self):
        metrics = calculate_presence_metrics(PresenceCounts(0, 0, 0, 0))
        self.assertIsNone(metrics.accuracy)
        self.assertIsNone(metrics.precision)
        self.assertIsNone(metrics.recall)
        self.assertIsNone(metrics.specificity)
        self.assertIsNone(metrics.f1)

    def test_one_class_sample_keeps_missing_rate_undefined(self):
        metrics = calculate_presence_metrics(PresenceCounts(0, 0, 2, 0))
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertIsNone(metrics.precision)
        self.assertIsNone(metrics.recall)
        self.assertEqual(metrics.specificity, 1.0)
        self.assertIsNone(metrics.f1)


if __name__ == "__main__":
    unittest.main()
