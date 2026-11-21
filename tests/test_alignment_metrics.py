import math
import unittest

from vantasight.analytics import center_crosshair, measure_alignment


class AlignmentMetricsTests(unittest.TestCase):
    def test_center_crosshair_uses_geometric_frame_center(self) -> None:
        self.assertEqual(center_crosshair(1920, 1080), (960.0, 540.0))
        self.assertEqual(center_crosshair(1917, 1073), (958.5, 536.5))

    def test_alignment_uses_euclidean_distance_and_frame_diagonal(self) -> None:
        result = measure_alignment((3, 4), (0, 0), 100, 100, "head")
        self.assertEqual(result.distance_pixels, 5.0)
        self.assertAlmostEqual(result.distance_normalized, 5 / math.hypot(100, 100))
        self.assertEqual(result.region_name, "head")

    def test_normalized_distance_is_resolution_independent(self) -> None:
        low = measure_alignment((60, 40), (50, 50), 100, 100)
        high = measure_alignment((120, 80), (100, 100), 200, 200)
        self.assertAlmostEqual(low.distance_normalized, high.distance_normalized)

    def test_rejects_invalid_dimensions_and_out_of_frame_points(self) -> None:
        with self.assertRaises(ValueError):
            center_crosshair(0, 100)
        with self.assertRaises(ValueError):
            measure_alignment((-1, 10), (20, 20), 100, 100)


if __name__ == "__main__":
    unittest.main()
