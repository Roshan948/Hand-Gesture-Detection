"""Automated tests for the logic-only test cases (no webcam / MediaPipe needed).

Run from the project root:   python -m unittest discover -s tests -v
Maps to the plan's test cases: TC04-TC10 (TC01-TC03, TC11, TC12 need a camera).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand_tracking import HandTracker                     # noqa: E402
from swipe_detection import (                             # noqa: E402
    DOWN, LEFT, RIGHT, UP, SwipeDetector, classify_movement,
)


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def feed(tracker, points):
    for p in points:
        tracker.update(p)


def line(start, end, steps):
    (x1, y1), (x2, y2) = start, end
    return [(x1 + (x2 - x1) * i // steps, y1 + (y2 - y1) * i // steps)
            for i in range(steps + 1)]


class TestClassification(unittest.TestCase):
    def test_four_directions(self):
        self.assertEqual(classify_movement(200, 5, 120), RIGHT)    # TC05
        self.assertEqual(classify_movement(-200, 5, 120), LEFT)    # TC04
        self.assertEqual(classify_movement(5, -200, 120), UP)      # TC06
        self.assertEqual(classify_movement(5, 200, 120), DOWN)     # TC07

    def test_small_movement_is_not_a_swipe(self):                  # TC08
        self.assertIsNone(classify_movement(30, 10, 120))

    def test_diagonal_is_ambiguous(self):
        self.assertIsNone(classify_movement(150, 140, 120, dominance_ratio=1.5))


class TestDetector(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.detector = SwipeDetector(threshold=120, min_frames=4,
                                      dominance_ratio=1.5, cooldown=1.0,
                                      clock=self.clock)
        self.tracker = HandTracker(max_frames=12, max_lost_frames=5)

    def test_swipe_right_detected(self):
        feed(self.tracker, line((100, 200), (350, 205), 8))
        self.assertEqual(self.detector.detect(self.tracker), RIGHT)

    def test_swipe_up_detected(self):
        feed(self.tracker, line((300, 400), (305, 150), 8))
        self.assertEqual(self.detector.detect(self.tracker), UP)

    def test_too_few_frames_no_swipe(self):
        feed(self.tracker, [(100, 200), (400, 200)])
        self.assertIsNone(self.detector.detect(self.tracker))

    def test_slow_drift_is_ignored(self):
        # 100 px spread over 60 frames: window only holds 12 -> never > 120.
        feed(self.tracker, line((100, 200), (200, 200), 60))
        self.assertIsNone(self.detector.detect(self.tracker))

    def test_cooldown_blocks_repeats(self):
        feed(self.tracker, line((100, 200), (350, 200), 8))
        self.assertEqual(self.detector.detect(self.tracker), RIGHT)
        self.tracker.reset()
        feed(self.tracker, line((100, 200), (350, 200), 8))
        self.assertIsNone(self.detector.detect(self.tracker))
        self.clock.t += 1.5
        self.assertEqual(self.detector.detect(self.tracker), RIGHT)


class TestTracker(unittest.TestCase):
    def test_hand_disappears_resets_tracking(self):                # TC09
        t = HandTracker(max_frames=12, max_lost_frames=3)
        feed(t, line((0, 0), (50, 0), 5))
        results = [t.mark_lost() for _ in range(4)]
        self.assertEqual(results, [False, False, False, True])
        self.assertFalse(t.is_tracking)

    def test_invalid_coordinates_ignored(self):
        t = HandTracker(max_frames=12, max_lost_frames=3)
        self.assertFalse(t.update(None))
        self.assertFalse(t.update((float("nan"), 3)))
        self.assertEqual(t.frame_count, 0)


if __name__ == "__main__":
    unittest.main()
