"""Hand detection module (MediaPipe Tasks - HandLandmarker).

Responsibility: take a BGR frame, find ONE primary hand, and return a stable
(x, y) pixel position plus data needed for drawing. It knows nothing about
tracking or swipes.
"""
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

import config

Point = Tuple[int, int]

# Wrist + the base knuckle of each finger: a stable "palm centre" that does
# not jitter as individual fingers move.
PALM_LANDMARK_IDS = (0, 5, 9, 13, 17)


@dataclass
class HandResult:
    detected: bool = False
    position: Optional[Point] = None                  # palm centre (pixels)
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    landmarks: List[Point] = field(default_factory=list)
    hand_count: int = 0


def ensure_model() -> str:
    """Return the model path, downloading it on first run if missing."""
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.MODEL_DIR)
    path = os.path.join(folder, config.MODEL_FILE)
    if os.path.isfile(path) and os.path.getsize(path) > 100_000:
        return path

    os.makedirs(folder, exist_ok=True)
    print("Downloading hand model (~7 MB), first run only...")
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(config.MODEL_URL, tmp)
        os.replace(tmp, path)
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"Could not download the hand model ({exc}).\n"
            f"Download it manually from:\n  {config.MODEL_URL}\n"
            f"and save it as:\n  {path}"
        ) from exc
    return path


class HandDetector:
    def __init__(self) -> None:
        model_path = ensure_model()
        with open(model_path, "rb") as f:        # buffer avoids Windows path issues
            model_bytes = f.read()

        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.MAX_NUM_HANDS,
            min_hand_detection_confidence=config.DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = 0

    def detect(self, frame: np.ndarray) -> HandResult:
        """Return the primary (largest) hand, or HandResult() if none."""
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode needs strictly increasing timestamps (milliseconds).
        ts_ms = max(int(time.monotonic() * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        output = self._landmarker.detect_for_video(mp_image, ts_ms)

        if not output.hand_landmarks:
            return HandResult()

        best: Optional[HandResult] = None
        best_area = -1
        valid_hands = 0
        for hand in output.hand_landmarks:
            points = self._to_pixels(hand, width, height)
            if points is None:          # invalid coordinates -> ignore hand
                continue
            valid_hands += 1
            bbox = self._bounding_box(points)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > best_area:
                best_area = area
                best = HandResult(
                    detected=True,
                    position=self._palm_centre(points),
                    bbox=bbox,
                    landmarks=points,
                )

        if best is None:
            return HandResult()
        best.hand_count = valid_hands
        return best

    def close(self) -> None:
        self._landmarker.close()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _to_pixels(hand, width: int, height: int) -> Optional[List[Point]]:
        points: List[Point] = []
        for lm in hand:
            if not (np.isfinite(lm.x) and np.isfinite(lm.y)):
                return None
            x = int(min(max(lm.x, 0.0), 1.0) * (width - 1))
            y = int(min(max(lm.y, 0.0), 1.0) * (height - 1))
            points.append((x, y))
        return points

    @staticmethod
    def _bounding_box(points: List[Point]) -> Tuple[int, int, int, int]:
        arr = np.array(points)
        x1, y1 = arr.min(axis=0)
        x2, y2 = arr.max(axis=0)
        return int(x1), int(y1), int(x2), int(y2)

    @staticmethod
    def _palm_centre(points: List[Point]) -> Point:
        palm = np.array([points[i] for i in PALM_LANDMARK_IDS])
        cx, cy = palm.mean(axis=0)
        return int(cx), int(cy)
