#!/usr/bin/env python3
"""
train_self.py — Καταγραφή των δικών σου gestures για τα 24 γράμματα ΕΝΓ.

Αντικαθιστά τα training data του βίντεο με δικά σου δείγματα.
Μετά την καταγραφή, εκπαιδεύει αυτόματα τον KNN classifier.

Controls:
  SPACE  : ξεκίνα καταγραφή για τρέχον γράμμα (3" countdown)
  N      : επόμενο γράμμα
  B      : προηγούμενο γράμμα
  R      : διαγραφή samples για τρέχον γράμμα
  T      : εκπαίδευση classifier (μετά από όλα τα γράμματα)
  Q/ESC  : αποθήκευση + έξοδος
"""

import os
import sys
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont

import gesture_matcher as gm
from progress_tracker import GREEK_LETTERS

# ── Config ────────────────────────────────────────────────────────────────────
WIN_W, WIN_H   = 1280, 720
HALF           = WIN_W // 2
ASSETS_DIR     = os.path.join(os.path.dirname(__file__), "assets", "letters")
MODEL_PATH     = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL      = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
SAMPLES_NEEDED = 60       # samples to capture per letter
COUNTDOWN_SECS = 3.0      # seconds of countdown before capture starts

_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(9,10),(10,11),(11,12),
    (13,14),(14,15),(15,16),(17,18),(18,19),(19,20),
    (0,5),(5,9),(9,13),(13,17),(0,17),
]
_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _font(size):
    for p in _FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def _pil_text(canvas, msg, pos, size=28, color=(255,255,255), anchor="lt"):
    pil  = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, msg, font=_font(size), fill=(color[2],color[1],color[0]),
              anchor=anchor)
    canvas[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def _bar(canvas, x, y, w, h, ratio, fg=(0,200,100), bg=(40,40,60)):
    ratio = max(0.0, min(1.0, ratio))
    cv2.rectangle(canvas, (x,y), (x+w,y+h), bg, -1)
    if ratio > 0:
        cv2.rectangle(canvas, (x,y), (x+int(w*ratio),y+h), fg, -1)
    cv2.rectangle(canvas, (x,y), (x+w,y+h), (80,80,100), 1)

def _panel(canvas, x, y, w, h, color=(15,15,28), alpha=0.88):
    ov = canvas.copy()
    cv2.rectangle(ov, (x,y), (x+w,y+h), color, -1)
    cv2.addWeighted(ov, alpha, canvas, 1-alpha, 0, canvas)

def _load_ref(letter, w, h):
    p = os.path.join(ASSETS_DIR, f"{letter}.jpg")
    if os.path.exists(p):
        img = cv2.imread(p)
        if img is not None:
            return cv2.resize(img, (w, h))
    ph = np.full((h, w, 3), (28,28,45), dtype=np.uint8)
    _pil_text(ph, "?", (w//2, h//2), size=100, color=(80,80,120), anchor="mm")
    return ph

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        print("Κατέβασμα μοντέλου…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def _make_detector():
    _ensure_model()
    base = python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return vision.HandLandmarker.create_from_options(opts)

def _detect(detector, frame_bgr, ts_ms):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return detector.detect_for_video(img, ts_ms)

# ── States ────────────────────────────────────────────────────────────────────
IDLE       = "idle"
COUNTDOWN  = "countdown"
CAPTURING  = "capturing"
DONE       = "done"

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    cap      = cv2.VideoCapture(0)
    detector = _make_detector()
    ts_ms    = 0

    letter_idx  = 0
    state       = IDLE
    state_start = 0.0
    captured    = 0
    status_msg  = ""
    status_t    = 0.0
    ref_cache   = {}

    cv2.namedWindow("Self Trainer — Sign Language GR", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Self Trainer — Sign Language GR", WIN_W, WIN_H)

    print("Self Trainer ξεκίνησε. Πάτα SPACE για καταγραφή κάθε γράμματος.")
    print("Μετά από όλα τα γράμματα, πάτα T για εκπαίδευση.")

    while True:
        ret, cam = cap.read()
        if not ret:
            break
        cam  = cv2.flip(cam, 1)
        ts_ms += 33
        result = _detect(detector, cam, ts_ms)
        lms    = result.hand_landmarks[0] if result.hand_landmarks else None

        letter = GREEK_LETTERS[letter_idx]
        now    = time.monotonic()

        # ── State machine ─────────────────────────────────────────────────
        if state == COUNTDOWN:
            elapsed = now - state_start
            if elapsed >= COUNTDOWN_SECS:
                state       = CAPTURING
                state_start = now
                captured    = 0

        elif state == CAPTURING:
            if lms is not None:
                vec = gm.normalize(lms)
                gm.save_sample(letter, vec)
                captured += 1
            if captured >= SAMPLES_NEEDED:
                status_msg = f"✓ {letter}: {captured} samples αποθηκεύτηκαν!"
                status_t   = now
                # auto-advance to next undone letter
                for step in range(1, len(GREEK_LETTERS)+1):
                    nxt = (letter_idx + step) % len(GREEK_LETTERS)
                    if gm.sample_count(GREEK_LETTERS[nxt]) < SAMPLES_NEEDED:
                        letter_idx = nxt
                        break
                state = IDLE

        # ── Draw ──────────────────────────────────────────────────────────
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)

        # Left: reference image
        ref_key = letter
        if ref_key not in ref_cache:
            ref_cache[ref_key] = _load_ref(letter, HALF, WIN_H - 120)
        ref_img = ref_cache[ref_key]
        canvas[0:WIN_H-120, :HALF] = ref_img

        # Right: webcam
        cam_r = cv2.resize(cam, (HALF, WIN_H - 120))
        canvas[0:WIN_H-120, HALF:] = cam_r

        # Draw landmarks on right side
        if lms:
            h_c, w_c = cam.shape[:2]
            sx = HALF / w_c; sy = (WIN_H-120) / h_c
            for a, b in _CONNECTIONS:
                pa = (int(lms[a].x*w_c*sx)+HALF, int(lms[a].y*h_c*sy))
                pb = (int(lms[b].x*w_c*sx)+HALF, int(lms[b].y*h_c*sy))
                cv2.line(canvas, pa, pb, (0,200,220), 1, cv2.LINE_AA)
            for lm in lms:
                pt = (int(lm.x*w_c*sx)+HALF, int(lm.y*h_c*sy))
                cv2.circle(canvas, pt, 5, (255,255,255), -1)
                cv2.circle(canvas, pt, 5, (0,140,255), 1)

        # Letter label overlay on reference image
        _pil_text(canvas, letter, (20, WIN_H-180), size=90, color=(255,215,50))
        _pil_text(canvas, f"{letter_idx+1}/{len(GREEK_LETTERS)}",
                  (HALF-15, 15), size=20, color=(200,200,200), anchor="rt")

        # Bottom bar
        bar_y = WIN_H - 120
        canvas[bar_y:] = (18, 18, 30)
        cv2.line(canvas, (0, bar_y), (WIN_W, bar_y), (50,50,70), 2)
        cv2.line(canvas, (HALF,0), (HALF, WIN_H-120), (50,50,70), 2)

        # Progress dots
        dot_y = WIN_H - 95
        step = (WIN_W - 60) // len(GREEK_LETTERS)
        for i, l in enumerate(GREEK_LETTERS):
            cx = 30 + i*step + step//2
            sc = gm.sample_count(l)
            done = sc >= SAMPLES_NEEDED
            col  = (0,200,80) if done else (55,55,75)
            r    = 9 if i==letter_idx else 5
            cv2.circle(canvas, (cx, dot_y), r, col, -1)
            if i == letter_idx:
                cv2.circle(canvas, (cx, dot_y), r+3, (255,215,50), 2)

        # State display
        sc = gm.sample_count(letter)
        sc_done = sc >= SAMPLES_NEEDED

        if state == COUNTDOWN:
            remain = COUNTDOWN_SECS - (now - state_start)
            msg    = f"Ετοιμάσου...  {remain:.1f}"
            col    = (60, 180, 255)
            # Flash overlay
            ov = canvas.copy()
            cv2.rectangle(ov, (HALF,0),(WIN_W,WIN_H-120),(40,80,40),-1)
            cv2.addWeighted(ov, 0.15, canvas, 0.85, 0, canvas)
            _pil_text(canvas, msg, (HALF+HALF//2, WIN_H-52),
                      size=30, color=col, anchor="mm")

        elif state == CAPTURING:
            ratio = captured / SAMPLES_NEEDED
            msg   = f"● REC  {captured}/{SAMPLES_NEEDED}  — κράτα το χέρι σου σταθερό"
            # Flash
            ov = canvas.copy()
            cv2.rectangle(ov,(HALF,0),(WIN_W,WIN_H-120),(0,80,0),-1)
            cv2.addWeighted(ov, 0.20, canvas, 0.80, 0, canvas)
            _pil_text(canvas, msg, (HALF+HALF//2, WIN_H-65),
                      size=26, color=(80,255,120), anchor="mm")
            _bar(canvas, HALF+40, WIN_H-40, HALF-80, 20, ratio,
                 fg=(0,220,80))

        else:
            # Show status message if recent
            if now - status_t < 2.5 and status_msg:
                _pil_text(canvas, status_msg, (WIN_W//2, WIN_H-52),
                          size=24, color=(80,255,120), anchor="mm")
            else:
                done_count = sum(1 for l in GREEK_LETTERS
                                 if gm.sample_count(l) >= SAMPLES_NEEDED)
                if sc_done:
                    hint = f"✓ {letter} OK ({sc} samples)   [N] Επόμενο  [R] Επανάλαβε"
                    _pil_text(canvas, hint, (WIN_W//2, WIN_H-52),
                              size=20, color=(80,255,120), anchor="mm")
                else:
                    hint = f"[SPACE] Ξεκίνα καταγραφή για  {letter}  ({sc}/{SAMPLES_NEEDED})"
                    _pil_text(canvas, hint, (WIN_W//2, WIN_H-52),
                              size=24, color=(255,215,50), anchor="mm")
                if done_count == len(GREEK_LETTERS):
                    _pil_text(canvas, "✓ Όλα έτοιμα! Πάτα [T] για εκπαίδευση",
                              (WIN_W//2, WIN_H-22), size=20, color=(100,255,160), anchor="mm")
                else:
                    _pil_text(canvas, f"[N] Επόμενο  [B] Πίσω  [R] Reset  [T] Train  [Q] Έξοδος",
                              (WIN_W//2, WIN_H-22), size=17, color=(120,120,150), anchor="mm")

        # Hand detection indicator (right side top)
        dot_col = (0,220,80) if lms else (60,60,80)
        cv2.circle(canvas, (WIN_W-30, 20), 10, dot_col, -1)
        _pil_text(canvas, "χέρι" if lms else "---",
                  (WIN_W-50, 20), size=15, color=(150,150,150), anchor="rm")

        cv2.imshow("Self Trainer — Sign Language GR", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" ") and state == IDLE:
            state       = COUNTDOWN
            state_start = now
        elif key == ord("n"):
            letter_idx  = (letter_idx + 1) % len(GREEK_LETTERS)
            state       = IDLE
            ref_cache.clear()
        elif key == ord("b"):
            letter_idx  = (letter_idx - 1) % len(GREEK_LETTERS)
            state       = IDLE
            ref_cache.clear()
        elif key == ord("r"):
            gm.delete_reference(GREEK_LETTERS[letter_idx])
            status_msg = f"Διαγράφηκαν samples για {GREEK_LETTERS[letter_idx]}"
            status_t   = now
            state      = IDLE
        elif key == ord("t"):
            # Train classifier
            done = sum(1 for l in GREEK_LETTERS if gm.sample_count(l) >= 10)
            if done < 5:
                status_msg = f"Χρειάζεσαι τουλάχιστον 5 γράμματα με ≥10 samples"
                status_t   = now
            else:
                print(f"\nΕκπαίδευση με {done} γράμματα…")
                acc = gm.train_classifier()
                status_msg = f"✓ Classifier εκπαιδεύτηκε! Ακρίβεια: {acc:.0%}"
                status_t   = now
                print(status_msg)

    cap.release()
    cv2.destroyAllWindows()

    done = sum(1 for l in GREEK_LETTERS if gm.sample_count(l) >= SAMPLES_NEEDED)
    print(f"\nΟλοκληρώθηκε! {done}/{len(GREEK_LETTERS)} γράμματα με δείγματα.")
    if done >= 5:
        print("Εκπαίδευση classifier…")
        acc = gm.train_classifier()
        print(f"✓ Ακρίβεια: {acc:.0%}  — Τρέξε python3 main.py")

if __name__ == "__main__":
    run()
