"""Hand Swipe Direction Detection App - main application loop.

Pipeline (one pass per frame):
    webcam -> validate frame -> detect hand -> track positions
           -> calculate dx/dy -> threshold + confirm -> classify -> display
Press Q to quit.
"""

import sys
import time

import cv2

import config
from hand_tracking import HandTracker
from swipe_detection import SwipeDetector
from utils import drawing


def open_camera():
    """Open the webcam; return None if it is unavailable."""
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def is_valid_frame(frame) -> bool:
    return frame is not None and getattr(frame, "size", 0) > 0 and frame.ndim == 3


def main() -> int:
    # MediaPipe is imported here so a missing install gives a friendly message.
    try:
        from hand_detection import HandDetector
    except ImportError as exc:
        print(f"[ERROR] Missing dependency: {exc}")
        print("Run:  pip install -r requirements.txt")
        return 1

    cap = open_camera()
    if cap is None:
        print(
            f"[CAMERA ERROR] Cannot open webcam (CAMERA_INDEX={config.CAMERA_INDEX})."
        )
        print(
            "Check that it is connected, not used by another app, or change CAMERA_INDEX in config.py."
        )
        return 1

    try:
        detector = HandDetector()
    except Exception as exc:
        print(f"[ERROR] Could not start the hand detector:\n{exc}")
        cap.release()
        return 1
    tracker = HandTracker(config.MAX_TRACK_FRAMES, config.MAX_LOST_FRAMES)
    swipes = SwipeDetector()

    invalid_frames = 0
    result_text = None
    result_until = 0.0
    lost_until = 0.0
    exit_code = 0

    try:
        while True:
            ok, frame = cap.read()

            # ---- Frame validation
            if not ok or not is_valid_frame(frame):
                invalid_frames += 1
                if invalid_frames >= config.MAX_INVALID_FRAMES:
                    print("[ERROR] Too many invalid frames from the webcam. Stopping.")
                    exit_code = 1
                    break
                time.sleep(0.01)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
                continue
            invalid_frames = 0

            if config.MIRROR_VIEW:
                frame = cv2.flip(frame, 1)

            now = time.monotonic()

            # ---- Detect hand
            hand = detector.detect(frame)

            status = "No Hand Detected"
            status_color = drawing.WHITE

            if swipes.in_cooldown():
                # Ignore motion (e.g. the hand returning) right after a swipe.
                tracker.reset()
                status, status_color = "Cooldown...", drawing.YELLOW
            elif hand.detected and tracker.update(hand.position):
                status, status_color = "Tracking...", drawing.GREEN
                if hand.hand_count > 1:
                    status = "Tracking... (multiple hands: using primary)"
                    status_color = drawing.ORANGE

                # ---- Swipe calculation + confirmation + classification
                swipe = swipes.detect(tracker)
                if swipe is not None:
                    result_text = swipe
                    result_until = now + config.RESULT_DISPLAY_TIME
                    print(swipe)
                    tracker.reset()  # one motion = one swipe
            else:
                if tracker.mark_lost():
                    lost_until = now + config.TRACKING_LOST_DISPLAY_TIME

            if now < lost_until and not tracker.is_tracking:
                status, status_color = "Tracking Lost", drawing.RED

            # ---- Display
            if hand.detected:
                drawing.draw_hand(frame, hand.bbox, hand.landmarks, hand.position)
            positions = tracker.positions
            if len(positions) >= 2:
                drawing.draw_trail(frame, positions)
                drawing.draw_arrow(frame, positions[0], positions[-1])

            drawing.draw_status(frame, status, status_color)
            if result_text and now < result_until:
                drawing.draw_swipe_result(frame, result_text)

            cv2.imshow(config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        pass
    except Exception as exc:  # basic safety net: never leave the camera locked
        print(f"[ERROR] Unexpected error: {exc}")
        exit_code = 1
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
