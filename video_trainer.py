#!/usr/bin/env python3
"""
video_trainer.py — Εξαγωγή training data από βίντεο ΕΝΓ

Χρήση:
  python3 video_trainer.py "https://youtube.com/..."   # κατεβάζει και ανοίγει
  python3 video_trainer.py path/to/video.mp4           # τοπικό αρχείο

Έλεγχοι:
  ←  /  →        : -1 / +1 frame
  A  /  D         : -30 / +30 frames  (~1s)
  SPACE           : play / pause
  ENTER           : αποθήκευση screenshot + 30 landmark samples για τρέχον γράμμα
  N               : επόμενο γράμμα για label
  B               : προηγούμενο γράμμα για label
  R               : reset samples για τρέχον γράμμα
  Q / ESC         : αποθήκευση και έξοδος
"""

import os
import sys
import time
import argparse

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import gesture_matcher as gm
from progress_tracker import GREEK_LETTERS

# ── Paths ─────────────────────────────────────────────────────────────────────
_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets", "letters")
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(9,10),(10,11),(11,12),
    (13,14),(14,15),(15,16),(17,18),(18,19),(19,20),
    (0,5),(5,9),(9,13),(13,17),(0,17),
]
SAMPLES_PER_LETTER = 30   # landmark samples to save per letter


# ── Download ──────────────────────────────────────────────────────────────────
def download(url: str) -> str:
    """Download YouTube video with yt-dlp; return local path."""
    import yt_dlp
    out = os.path.join(os.path.dirname(__file__), "data", "training_video.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        print(f"Το βίντεο υπάρχει ήδη: {out}")
        return out
    print("Κατέβασμα βίντεο…")
    ydl_opts = {
        "format":    "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "outtmpl":   out,
        "quiet":     True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"Αποθηκεύτηκε: {out}")
    return out


# ── MediaPipe setup ───────────────────────────────────────────────────────────
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


def _draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 220), 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)
        cv2.circle(frame, pt, 4, (0, 140, 255), 1)


# ── Sidebar renderer ──────────────────────────────────────────────────────────
def _draw_sidebar(sidebar, letter: str, letter_idx: int,
                  has_shot: bool, sample_count: int, hint: str) -> None:
    sidebar[:] = (18, 18, 30)
    h, w = sidebar.shape[:2]

    # Header
    cv2.rectangle(sidebar, (0, 0), (w, 55), (40, 30, 80), -1)
    cv2.putText(sidebar, "VIDEO TRAINER", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 160, 255), 1)
    cv2.putText(sidebar, f"{letter_idx+1}/{len(GREEK_LETTERS)}",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 180), 1)

    # Big letter (OpenCV doesn't support Greek natively — use placeholder box)
    cv2.rectangle(sidebar, (w//2-55, 65), (w//2+55, 155), (50, 40, 80), -1)
    cv2.rectangle(sidebar, (w//2-55, 65), (w//2+55, 155), (100, 80, 160), 2)
    cv2.putText(sidebar, letter.encode('ascii', 'replace').decode(),
                (w//2-30, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 210, 50), 3)

    # Status
    y = 175
    shot_col = (0, 200, 100) if has_shot else (80, 80, 100)
    cv2.putText(sidebar, f"Screenshot: {'OK' if has_shot else '---'}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, shot_col, 1)
    y += 28
    sc_col = (0, 200, 100) if sample_count >= SAMPLES_PER_LETTER else (200, 150, 50)
    cv2.putText(sidebar, f"Landmarks:  {sample_count}/{SAMPLES_PER_LETTER}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sc_col, 1)

    # Progress bar for samples
    bw = w - 20
    cv2.rectangle(sidebar, (10, y+10), (10+bw, y+22), (50, 50, 70), -1)
    fill = int(bw * min(sample_count / SAMPLES_PER_LETTER, 1.0))
    if fill:
        cv2.rectangle(sidebar, (10, y+10), (10+fill, y+22), sc_col, -1)
    y += 40

    # Progress dots (all 24 letters)
    cols = 6
    for i, l in enumerate(GREEK_LETTERS):
        col_  = i % cols
        row_  = i // cols
        cx = 14 + col_ * 36
        cy = y + row_ * 22
        has = gm.has_reference(l)
        shot = os.path.exists(os.path.join(_ASSETS_DIR, f"{l}.jpg"))
        color = (0, 180, 80) if (has and shot) else (
                (0, 120, 50) if (has or shot) else (50, 50, 70))
        cv2.circle(sidebar, (cx, cy), 8 if i == letter_idx else 6, color, -1)
        if i == letter_idx:
            cv2.circle(sidebar, (cx, cy), 10, (255, 210, 50), 2)
    y += 90

    # Controls
    controls = [
        ("←/→",  "1 frame"),
        ("A/D",  "1 sec"),
        ("SPC",  "Play/Pause"),
        ("ENTER","Save shot+LM"),
        ("N/B",  "Next/Prev"),
        ("R",    "Reset LM"),
        ("Q",    "Quit"),
    ]
    cv2.putText(sidebar, "── Controls ──", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 150), 1)
    y += 18
    for key, desc in controls:
        cv2.putText(sidebar, f"{key:<7} {desc}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 200), 1)
        y += 18

    # Hint
    if hint:
        cv2.rectangle(sidebar, (5, h-45), (w-5, h-5), (40, 40, 80), -1)
        cv2.putText(sidebar, hint[:28], (8, h-28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 160), 1)


# ── Main interactive loop ─────────────────────────────────────────────────────
def label_video(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Αδύνατο άνοιγμα: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    detector     = _make_detector()
    os.makedirs(_ASSETS_DIR, exist_ok=True)

    frame_idx    = 0
    letter_idx   = 0
    playing      = False
    hint         = "Navigare στο σωστό frame και πάτα ENTER"
    last_frame   = None
    last_lms     = None
    ts_ms        = 0

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(idx, total_frames - 1)))
        ret, frm = cap.read()
        return frm if ret else last_frame

    while True:
        if playing:
            frame_idx = min(frame_idx + 1, total_frames - 1)

        frame = read_frame(frame_idx)
        if frame is None:
            break
        frame = cv2.flip(frame, 1)

        ts_ms += 33
        result  = _detect(detector, frame, ts_ms)
        lms     = result.hand_landmarks[0] if result.hand_landmarks else None
        last_lms = lms

        display = frame.copy()
        if lms:
            _draw_landmarks(display, lms)

        # Timestamp overlay
        secs = frame_idx / fps
        cv2.putText(display, f"{int(secs//60):02d}:{secs%60:05.2f}  frame {frame_idx}/{total_frames}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(display, f"{int(secs//60):02d}:{secs%60:05.2f}  frame {frame_idx}/{total_frames}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 255), 1)

        letter = GREEK_LETTERS[letter_idx]
        has_shot = os.path.exists(os.path.join(_ASSETS_DIR, f"{letter}.jpg"))
        sc       = gm.sample_count(letter)

        # Sidebar
        sw      = 220
        sidebar = np.zeros((display.shape[0], sw, 3), dtype=np.uint8)
        _draw_sidebar(sidebar, letter, letter_idx, has_shot, sc, hint)

        combined = np.hstack([display, sidebar])
        cv2.imshow("Video Trainer — Sign Language GR", combined)

        delay = max(1, int(1000 / fps)) if playing else 30
        key   = cv2.waitKey(delay) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == 81 or key == ord("a"):    # left / A
            frame_idx = max(0, frame_idx - (30 if key == ord("a") else 1))
            playing = False
        elif key == 83 or key == ord("d"):    # right / D
            frame_idx = min(total_frames - 1, frame_idx + (30 if key == ord("d") else 1))
            playing = False
        elif key == ord("n"):
            letter_idx = (letter_idx + 1) % len(GREEK_LETTERS)
            hint = f"Επόμενο: {GREEK_LETTERS[letter_idx]}"
        elif key == ord("b"):
            letter_idx = (letter_idx - 1) % len(GREEK_LETTERS)
            hint = f"Προηγούμενο: {GREEK_LETTERS[letter_idx]}"
        elif key == ord("r"):
            gm.delete_reference(letter)
            hint = f"Reset: {letter}"
        elif key == 13:  # ENTER — save screenshot + landmarks
            letter = GREEK_LETTERS[letter_idx]
            saved_any = False

            # Screenshot
            shot_path = os.path.join(_ASSETS_DIR, f"{letter}.jpg")
            cv2.imwrite(shot_path, frame)
            saved_any = True

            # Landmarks (collect multiple samples around current frame)
            if lms is not None:
                vec = gm.normalize(lms)
                # Save SAMPLES_PER_LETTER samples from nearby frames
                for offset in range(-SAMPLES_PER_LETTER // 2, SAMPLES_PER_LETTER // 2):
                    f2 = read_frame(frame_idx + offset)
                    if f2 is None:
                        continue
                    ts_ms += 33
                    r2 = _detect(detector, f2, ts_ms)
                    if r2.hand_landmarks:
                        v2 = gm.normalize(r2.hand_landmarks[0])
                        gm.save_sample(letter, v2)

                count = gm.sample_count(letter)
                hint  = f"✓ {letter}: shot + {count} samples"
            else:
                hint = f"✓ {letter}: screenshot (χέρι δεν εντοπίστηκε)"

            # Auto-advance to next unfinished letter
            for step in range(1, len(GREEK_LETTERS)):
                nxt = (letter_idx + step) % len(GREEK_LETTERS)
                l   = GREEK_LETTERS[nxt]
                if not gm.has_reference(l) or not os.path.exists(
                        os.path.join(_ASSETS_DIR, f"{l}.jpg")):
                    letter_idx = nxt
                    break

    cap.release()
    cv2.destroyAllWindows()

    done = sum(
        1 for l in GREEK_LETTERS
        if gm.has_reference(l) and os.path.exists(os.path.join(_ASSETS_DIR, f"{l}.jpg"))
    )
    print(f"\nΟλοκληρώθηκε! {done}/{len(GREEK_LETTERS)} γράμματα έτοιμα.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video trainer για Sign Language GR")
    parser.add_argument("source", help="YouTube URL ή τοπικό αρχείο βίντεο")
    args = parser.parse_args()

    src = args.source
    if src.startswith("http"):
        src = download(src)

    if not os.path.exists(src):
        print(f"Αρχείο δεν βρέθηκε: {src}")
        sys.exit(1)

    label_video(src)
