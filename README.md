# Hand Swipe Direction Detection App

A local Python webcam app that detects whether your hand swiped **LEFT, RIGHT, UP, or DOWN**
by comparing hand positions across several frames (temporal movement detection, not static
hand position). No internet or API is needed.

**Tech:** Python 3.9+, OpenCV, NumPy, MediaPipe Hand Landmarker (landmarks only, no model training).

## Project structure

```
hand_swipe_detection/
├── main.py              # webcam loop: capture -> detect -> track -> classify -> display
├── hand_detection.py    # MediaPipe: finds the primary hand, returns palm-centre (x, y)
├── hand_tracking.py     # stores previous/current positions, handles lost hand
├── swipe_detection.py   # dx/dy, dominance + threshold rules, confirmation, cooldown
├── config.py            # ALL tunable values (threshold, frames, cooldown, camera)
├── requirements.txt
├── README.md
├── utils/
│   └── drawing.py       # landmarks, box, trail, arrow, status text
├── models/              # hand_landmarker.task is auto-downloaded here on first run
├── tests/
│   └── test_swipe_logic.py   # automated logic tests (no camera needed)
└── screenshots/         # put your demo screenshots here
```

## Setup

```bash
cd hand_swipe_detection
python -m venv venv
# Windows:        venv\Scripts\activate
# macOS / Linux:  source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The first run downloads a small hand model (~7 MB) into `models/`, so it needs internet once.
Hold one hand in front of the webcam and swipe. Press **Q** to quit.

| Screen text        | Meaning                                                  |
|--------------------|----------------------------------------------------------|
| No Hand Detected   | nothing to track (normal condition)                      |
| Tracking...        | hand found, movement is being observed                   |
| SWIPE LEFT/RIGHT/UP/DOWN | confirmed swipe (also printed in the terminal)     |
| Cooldown...        | short pause after a swipe so one motion = one swipe      |
| Tracking Lost      | hand vanished mid-movement; tracking was reset           |

The view is mirrored, so moving your hand to **your** right shows SWIPE RIGHT
(set `MIRROR_VIEW = False` in `config.py` to disable).

## How it works

1. Each frame, MediaPipe finds the hand; the **palm centre** (average of wrist + finger-base
   landmarks) is used as a stable position. With several hands, the largest is used.
2. `HandTracker` keeps the last `MAX_TRACK_FRAMES` positions.
3. `dx = newest_x - oldest_x`, `dy = newest_y - oldest_y` over that window.
4. The dominant axis wins (`|dx|` vs `|dy|`, using `DOMINANCE_RATIO`). Image Y grows downward,
   so `dy > 0` = DOWN and `dy < 0` = UP.
5. A swipe is confirmed only if the movement beats `MOVEMENT_THRESHOLD` pixels and at least
   `MIN_FRAMES` frames were tracked. Slow drift never reaches the threshold inside the short window.
6. After a swipe, tracking resets and `COOLDOWN_TIME` seconds are ignored.

## Configuration (`config.py`)

| Setting | Default | Purpose |
|---|---|---|
| `MOVEMENT_THRESHOLD` | 120 | pixels the hand must travel (at 640x480) |
| `MIN_FRAMES` | 4 | frames required before confirming |
| `MAX_TRACK_FRAMES` | 12 | length of the tracking window |
| `DOMINANCE_RATIO` | 1.5 | ignore diagonal movement |
| `COOLDOWN_TIME` | 1.0 | seconds between swipes |
| `CAMERA_INDEX` | 0 | which webcam |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | 640 / 480 | camera resolution |

### Threshold tuning procedure

1. Start conservative (default 120).
2. Test slow and fast swipes in all four directions.
3. Test small movements that should **not** trigger.
4. Test at different distances from the webcam.
5. Adjust `MOVEMENT_THRESHOLD` / `MIN_FRAMES` until normal motion is reliable.
6. Record your final values here:

```
MOVEMENT_THRESHOLD = ____   MIN_FRAMES = ____   COOLDOWN_TIME = ____
```

Too many missed swipes -> lower the threshold. Too many false swipes -> raise it or raise `DOMINANCE_RATIO`.

## Testing

Automated logic tests (no webcam needed):

```bash
python -m unittest discover -s tests -v
```

| ID | Test | Expected | How |
|----|------|----------|-----|
| TC01 | Webcam unavailable | Camera Error | manual: set `CAMERA_INDEX = 9` |
| TC02 | Webcam available | Webcam Feed | manual |
| TC03 | No hand | No Hand Detected | manual |
| TC04-TC07 | Move left/right/up/down | SWIPE LEFT/RIGHT/UP/DOWN | automated + manual |
| TC08 | Very small movement | No Swipe | automated + manual |
| TC09 | Hand disappears | Tracking Lost | automated + manual |
| TC10 | Multiple hands | Primary Hand / Warning | manual |
| TC11 | Invalid frame | Handled safely | built into `main.py` (skips, stops after `MAX_INVALID_FRAMES`) |
| TC12 | Press Q | Clean exit | manual |

## Troubleshooting

- **Model download fails** - download the file manually from
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
  and save it as `models/hand_landmarker.task`.
- **Camera error** - close other apps using the webcam, or try `CAMERA_INDEX = 1`.
- **Wrong direction** - toggle `MIRROR_VIEW` in `config.py`.

## Optional next step (only after the core works)

`Swipe Result -> API/LLM -> text/action`, e.g. "User swiped to the right." The core app does not need it.
