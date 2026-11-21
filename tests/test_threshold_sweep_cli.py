import argparse
import unittest

from tools.evaluate_threshold_sweep import _parse_thresholds


class ThresholdSweepCliTests(unittest.TestCase):
    def test_parses_ascending_unique_thresholds(self):
        self.assertEqual(_parse_thresholds("0.25, 0.50,0.95"), (0.25, 0.5, 0.95))

    def test_rejects_duplicate_descending_and_out_of_range_thresholds(self):
        for value in ("0.25,0.25", "0.75,0.50", "-0.1,0.5", "0.5,1.1", ""):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                _parse_thresholds(value)


if __name__ == "__main__":
    unittest.main()
