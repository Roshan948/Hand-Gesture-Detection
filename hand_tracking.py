"""Hand tracking module.

Responsibility: remember where the hand WAS and where it IS NOW across
frames. It answers: where was the hand, where is it now, how far did it move.
It does not decide whether a swipe happened.
"""
import math
from collections import deque
from typing import Deque, List, Optional, Tuple

Point = Tuple[int, int]


class HandTracker:
    def __init__(self, max_frames: int, max_lost_frames: int) -> None:
        self._positions: Deque[Point] = deque(maxlen=max_frames)
        self._max_lost = max_lost_frames
        self._lost_frames = 0

    # ----------------------------------------------------------- updates
    def update(self, position: Optional[Point]) -> bool:
        """Add the current hand position. Returns False if it was invalid."""
        if (
            position is None
            or len(position) != 2
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in position)
        ):
            return False
        self._positions.append((int(position[0]), int(position[1])))
        self._lost_frames = 0
        return True

    def mark_lost(self) -> bool:
        """Call on frames with no hand.

        Returns True exactly once when a movement in progress is abandoned
        (hand gone longer than the allowed grace frames) -> "Tracking Lost".
        """
        if not self._positions:
            return False
        self._lost_frames += 1
        if self._lost_frames > self._max_lost:
            had_movement = len(self._positions) >= 2
            self.reset()
            return had_movement
        return False

    def reset(self) -> None:
        self._positions.clear()
        self._lost_frames = 0

    # ----------------------------------------------------------- queries
    @property
    def frame_count(self) -> int:
        return len(self._positions)

    @property
    def positions(self) -> List[Point]:
        return list(self._positions)

    @property
    def is_tracking(self) -> bool:
        return len(self._positions) > 0

    def displacement(self) -> Optional[Tuple[int, int]]:
        """(dx, dy) from the oldest to the newest position in the window."""
        if len(self._positions) < 2:
            return None
        x1, y1 = self._positions[0]
        x2, y2 = self._positions[-1]
        return x2 - x1, y2 - y1
