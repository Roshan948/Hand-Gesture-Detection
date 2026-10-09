"""Central configuration for the Hand Swipe Direction Detection App.

All tunable values live here so the application logic never needs editing
when you tune the system for your camera, distance, and hand speed.
"""

# ---------------------------------------------------------------- Camera
CAMERA_INDEX = 0  # 0 = default webcam; try 1, 2 ... for others
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIRROR_VIEW = True  # flip frame so your RIGHT hand movement = SWIPE RIGHT
WINDOW_NAME = "Hand Swipe Direction Detection  (press Q to quit)"

# ----------------------------------------------------- Hand detection
MAX_NUM_HANDS = 2  # detect up to 2 so we can warn / pick primary
DETECTION_CONFIDENCE = 0.4
TRACKING_CONFIDENCE = 0.4

# ------------------------------------------------------------- Swipe
# Minimum distance (in PIXELS, at FRAME_WIDTH x FRAME_HEIGHT) the hand must
# travel inside the tracking window. Tune for distance / camera / hand size.
MOVEMENT_THRESHOLD = 120

# Minimum number of consecutive tracked frames before a swipe can be confirmed.
MIN_FRAMES = 4

# Tracking window length (frames). A swipe must reach MOVEMENT_THRESHOLD
# inside this window, so slow drifting is ignored.
MAX_TRACK_FRAMES = 12

# Dominant axis must be at least this many times larger than the other axis.
# 1.0 = simple |dx| > |dy| rule; higher = ignore diagonal movement.
DOMINANCE_RATIO = 1.5

# Seconds to ignore new swipes after a confirmed swipe (0 disables cooldown).
COOLDOWN_TIME = 1.0

# ----------------------------------------------------- Error handling
MAX_INVALID_FRAMES = 30  # consecutive bad frames before giving up
MAX_LOST_FRAMES = 12  # frames a hand may vanish before tracking resets

# ---------------------------------------------------------------- UI
RESULT_DISPLAY_TIME = 1.5  # seconds the SWIPE result stays on screen
TRACKING_LOST_DISPLAY_TIME = 1.0

# -------------------------------------------------------- Hand model
# MediaPipe Tasks needs a small model file (~7 MB). It is downloaded
# automatically on the first run into the models/ folder.
MODEL_DIR = "models"
MODEL_FILE = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
