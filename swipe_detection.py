"""Swipe detection module.

Responsibility: turn movement (dx, dy) into LEFT / RIGHT / UP / DOWN using
dominance + threshold rules, plus frame-count confirmation and cooldown.

Image coordinates: X grows to the right, Y grows DOWNWARD, so
    dx > 0 -> RIGHT     dx < 0 -> LEFT
    dy > 0 -> DOWN      dy < 0 -> UP
"""
import time
from typing import Callable, Optional

import config
from hand_tracking import HandTracker

LEFT = "SWIPE LEFT"
RIGHT = "SWIPE RIGHT"
UP = "SWIPE UP"
DOWN = "SWIPE DOWN"


def classify_movement(
    dx: float,
    dy: float,
    threshold: float,
    dominance_ratio: float = 1.0,
) -> Optional[str]:
    """Pure function: return a swipe label, or None if not a swipe."""
    adx, ady = abs(dx), abs(dy)

    if adx > ady and adx >= dominance_ratio * ady:      # horizontal dominates
        if dx > threshold:
            return RIGHT
        if dx < -threshold:
            return LEFT
    elif ady > adx and ady >= dominance_ratio * adx:    # vertical dominates
        if dy > threshold:
            return DOWN
        if dy < -threshold:
            return UP
    return None


class SwipeDetector:
    def __init__(
        self,
        threshold: float = config.MOVEMENT_THRESHOLD,
        min_frames: int = config.MIN_FRAMES,
        dominance_ratio: float = config.DOMINANCE_RATIO,
        cooldown: float = config.COOLDOWN_TIME,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.threshold = threshold
        self.min_frames = min_frames
        self.dominance_ratio = dominance_ratio
        self.cooldown = cooldown
        self._clock = clock
        self._last_swipe_time: Optional[float] = None

    def in_cooldown(self) -> bool:
        if self._last_swipe_time is None or self.cooldown <= 0:
            return False
        return (self._clock() - self._last_swipe_time) < self.cooldown

    def detect(self, tracker: HandTracker) -> Optional[str]:
        """Return a confirmed swipe label, or None."""
        if self.in_cooldown():
            return None
        if tracker.frame_count < self.min_frames:      # confirmation: min frames
            return None
        movement = tracker.displacement()
        if movement is None:
            return None

        direction = classify_movement(
            movement[0], movement[1], self.threshold, self.dominance_ratio
        )
        if direction is not None:
            self._last_swipe_time = self._clock()
        return direction
