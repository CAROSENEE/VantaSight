import unittest

from vantasight.analytics import summarize_distribution


class DistributionSummaryTests(unittest.TestCase):
    def test_summarizes_scores(self):
        summary = summarize_distribution([0.2, 0.4, 0.8, 1.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["min"], 0.2)
        self.assertAlmostEqual(summary["mean"], 0.6)
        self.assertAlmostEqual(summary["median"], 0.6)
        self.assertEqual(summary["p95"], 1.0)
        self.assertEqual(summary["max"], 1.0)

    def test_empty_distribution_is_explicit(self):
        self.assertEqual(
            summarize_distribution([]),
            {"count": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None},
        )

    def test_rejects_non_finite_values(self):
        with self.assertRaises(ValueError):
            summarize_distribution([float("nan")])


if __name__ == "__main__":
    unittest.main()
