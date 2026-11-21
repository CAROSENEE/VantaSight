import unittest

from vantasight.tracking import IoUTracker
from vantasight.vision import Detection


def detection(box: tuple[int, int, int, int]) -> Detection:
    return Detection(class_name="person candidate", confidence=0.9, box_xyxy=box)


class IoUTrackerTests(unittest.TestCase):
    def test_overlapping_boxes_keep_temporary_id_and_get_velocity(self) -> None:
        tracker = IoUTracker(iou_threshold=0.3)
        first = tracker.update([detection((0, 0, 10, 10))], 0, 0.0)
        second = tracker.update([detection((1, 0, 11, 10))], 1, 1.0)
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(second[0].velocity_xy_pixels_per_second, (1.0, 0.0))

    def test_track_survives_configured_missed_update_then_expires(self) -> None:
        tracker = IoUTracker(max_missed_updates=1)
        first = tracker.update([detection((0, 0, 10, 10))], 0, 0.0)[0]
        self.assertEqual(tracker.update([], 1, 1.0), [])
        resumed = tracker.update([detection((0, 0, 10, 10))], 2, 2.0)[0]
        self.assertEqual(resumed.track_id, first.track_id)
        tracker.update([], 3, 3.0)
        tracker.update([], 4, 4.0)
        new_track = tracker.update([detection((0, 0, 10, 10))], 5, 5.0)[0]
        self.assertNotEqual(new_track.track_id, first.track_id)

    def test_no_detection_does_not_invent_an_observation(self) -> None:
        tracker = IoUTracker()
        self.assertEqual(tracker.update([], 0, 0.0), [])


if __name__ == "__main__":
    unittest.main()
