import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (5,6),(6,7),(7,8),
    (9,10),(10,11),(11,12),
    (13,14),(14,15),(15,16),
    (17,18),(18,19),(19,20),
    (0,5),(5,9),(9,13),(13,17),(0,17),
]


def _ensure_model() -> None:
    if not os.path.exists(_MODEL_PATH):
        print("Κατέβασμα μοντέλου (~8MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Έτοιμο.")


class HandTracker:
    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.7):
        _ensure_model()
        base = python.BaseOptions(model_asset_path=_MODEL_PATH)
        opts = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(opts)
        self._ts_ms = 0

    def process(self, frame_bgr):
        """Return list of 21 NormalizedLandmark for first hand, or None."""
        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33
        result   = self._detector.detect_for_video(mp_image, self._ts_ms)
        if result.hand_landmarks:
            return result.hand_landmarks[0]
        return None

    def draw(self, frame, landmarks) -> None:
        if not landmarks:
            return
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in _CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 200), 1, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1)
            cv2.circle(frame, pt, 4, (0, 150, 255), 1)
