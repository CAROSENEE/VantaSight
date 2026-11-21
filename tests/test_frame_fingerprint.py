import unittest

import numpy as np

from vantasight.input import frame_sha256, verify_frame_fingerprint


class FrameFingerprintTests(unittest.TestCase):
    def test_fingerprint_is_stable_and_sensitive_to_pixels_and_shape(self):
        pixels = np.zeros((3, 4, 3), dtype=np.uint8)
        self.assertEqual(frame_sha256(pixels), frame_sha256(pixels.copy()))
        changed_pixels = pixels.copy()
        changed_pixels[0, 0, 0] = 1
        self.assertNotEqual(frame_sha256(pixels), frame_sha256(changed_pixels))
        self.assertNotEqual(frame_sha256(pixels), frame_sha256(pixels.reshape(4, 3, 3)))

    def test_verify_supports_legacy_and_checks_fingerprint(self):
        pixels = np.zeros((2, 2, 3), dtype=np.uint8)
        verify_frame_fingerprint({}, pixels, 12)
        verify_frame_fingerprint({"source_frame_sha256": frame_sha256(pixels)}, pixels, 12)
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            verify_frame_fingerprint({"source_frame_sha256": "0" * 64}, pixels, 12)

    def test_rejects_malformed_fingerprint(self):
        with self.assertRaisesRegex(ValueError, "Invalid source_frame_sha256"):
            verify_frame_fingerprint({"source_frame_sha256": "not-a-hash"}, np.zeros((1, 1)), 0)


if __name__ == "__main__":
    unittest.main()
