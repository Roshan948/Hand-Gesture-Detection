"""Drawing helpers: landmarks, bounding box, trail, arrow, status text."""
from typing import List, Optional, Tuple

import cv2

Point = Tuple[int, int]

GREEN = (0, 220, 0)
RED = (0, 0, 255)
YELLOW = (0, 220, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (0, 140, 255)


def draw_status(frame, text: str, color=WHITE, line: int = 0) -> None:
    """Small status line in the top-left corner (line 0, 1, 2 ...)."""
    y = 28 + line * 28
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_hand(frame, bbox: Optional[Tuple[int, int, int, int]],
              landmarks: List[Point], position: Optional[Point]) -> None:
    if bbox is not None:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), GREEN, 2)
    for p in landmarks:
        cv2.circle(frame, p, 3, YELLOW, -1)
    if position is not None:
        cv2.circle(frame, position, 8, RED, -1)


def draw_trail(frame, positions: List[Point]) -> None:
    for a, b in zip(positions[:-1], positions[1:]):
        cv2.line(frame, a, b, ORANGE, 2, cv2.LINE_AA)


def draw_arrow(frame, start: Point, end: Point) -> None:
    if start != end:
        cv2.arrowedLine(frame, start, end, RED, 3, cv2.LINE_AA, tipLength=0.25)


def draw_swipe_result(frame, text: str) -> None:
    """Large centred banner for the final SWIPE result."""
    height, width = frame.shape[:2]
    scale, thickness = 1.4, 3
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = (width - tw) // 2, height - 40
    cv2.rectangle(frame, (x - 15, y - th - 15), (x + tw + 15, y + 15), BLACK, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, GREEN,
                thickness, cv2.LINE_AA)
