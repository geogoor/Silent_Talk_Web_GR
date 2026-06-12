"""
auto_extract.py — Αυτόματη εξαγωγή screenshots + landmarks από το training video.

Τρέξε μία φορά:
  python3 auto_extract.py
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import gesture_matcher as gm
from progress_tracker import GREEK_LETTERS

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets", "letters")
_VIDEO_PATH = os.path.join(os.path.dirname(__file__), "data", "training_video.mp4")
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Detected letter start times (seconds) — auto-detected from video
LETTER_STARTS = [
    13.8, 18.0, 22.0, 26.0, 30.2, 34.0, 37.8, 41.8,
    45.6, 49.2, 52.8, 57.0, 61.0, 64.8, 68.6, 72.2,
    76.0, 79.8, 86.4, 90.4, 94.4, 98.4, 102.2, 105.8,
]

FPS              = 25.0
SAMPLES          = 40      # landmark samples per letter
SCREENSHOT_DELAY = 1.5     # seconds after letter start for screenshot
SAMPLE_WINDOW    = (1.0, 3.5)  # seconds after start to collect landmarks


def _ensure_model() -> None:
    if not os.path.exists(_MODEL_PATH):
        import urllib.request
        print("Κατέβασμα μοντέλου (~8MB)…")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)


def _make_detector():
    _ensure_model()
    base = python.BaseOptions(model_asset_path=_MODEL_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)


def _detect(detector, frame_bgr, ts_ms: int):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return detector.detect_for_video(img, ts_ms)


def _read_frame(cap, t_sec: float):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * FPS))
    ret, frame = cap.read()
    return frame if ret else None


def extract() -> None:
    os.makedirs(_ASSETS_DIR, exist_ok=True)
    cap      = cv2.VideoCapture(_VIDEO_PATH)
    detector = _make_detector()
    ts_ms    = 0

    print(f"Εξαγωγή για {len(GREEK_LETTERS)} γράμματα…\n")

    for i, (letter, t_start) in enumerate(zip(GREEK_LETTERS, LETTER_STARTS)):
        print(f"[{i+1:2d}/24]  {letter}  (t={t_start:.1f}s)", end="  ")

        # ── Screenshot ─────────────────────────────────────────────────────
        shot_path = os.path.join(_ASSETS_DIR, f"{letter}.jpg")
        frame_shot = _read_frame(cap, t_start + SCREENSHOT_DELAY)
        if frame_shot is not None:
            cv2.imwrite(shot_path, frame_shot)
            print("📸", end="  ")

        # ── Landmarks ──────────────────────────────────────────────────────
        t_end = LETTER_STARTS[i + 1] if i + 1 < len(LETTER_STARTS) else t_start + 4.0
        win_start = t_start + SAMPLE_WINDOW[0]
        win_end   = min(t_start + SAMPLE_WINDOW[1], t_end - 0.3)

        # Sample evenly within the window
        n_frames    = max(1, int((win_end - win_start) * FPS))
        step        = max(1, n_frames // SAMPLES)
        saved       = 0

        for offset_f in range(0, n_frames, step):
            t = win_start + offset_f / FPS
            frame = _read_frame(cap, t)
            if frame is None:
                continue
            ts_ms += 40
            result = _detect(detector, frame, ts_ms)
            if result.hand_landmarks:
                vec = gm.normalize(result.hand_landmarks[0])
                gm.save_sample(letter, vec)
                saved += 1
            if saved >= SAMPLES:
                break

        # If window gave few samples, try additional offsets
        if saved < 10:
            for extra_t in np.linspace(win_start, win_end, SAMPLES * 2):
                if saved >= SAMPLES:
                    break
                frame = _read_frame(cap, float(extra_t))
                if frame is None:
                    continue
                ts_ms += 40
                result = _detect(detector, frame, ts_ms)
                if result.hand_landmarks:
                    vec = gm.normalize(result.hand_landmarks[0])
                    gm.save_sample(letter, vec)
                    saved += 1

        total = gm.sample_count(letter)
        status = "✓" if total >= 20 else "⚠"
        print(f"{status} {total} landmarks")

    cap.release()

    # Summary
    print("\n─── Αποτελέσματα ───────────────────────")
    ok, warn = 0, 0
    for letter in GREEK_LETTERS:
        sc   = gm.sample_count(letter)
        shot = os.path.exists(os.path.join(_ASSETS_DIR, f"{letter}.jpg"))
        if sc >= 20 and shot:
            ok += 1
        else:
            warn += 1
            print(f"  ⚠  {letter}: {sc} samples, shot={'✓' if shot else '✗'}")
    print(f"\n✓ Έτοιμα: {ok}/24  |  ⚠ Χρειάζονται προσοχή: {warn}/24")
    if warn:
        print("→ Χρησιμοποίησε video_trainer.py για να διορθώσεις τα problematic γράμματα.")
    else:
        print("→ Όλα έτοιμα! Τρέξε main.py για να παίξεις.")


if __name__ == "__main__":
    extract()
