import os
import random
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gesture_matcher as gm
from hand_tracker import HandTracker
from score_tracker import ScoreTracker
from progress_tracker import GREEK_LETTERS

# ── Constants ─────────────────────────────────────────────────────────────────
WIN_W, WIN_H   = 1280, 720
HALF           = WIN_W // 2
MATCH_THRESHOLD  = 0.85
HOLD_SECONDS     = 2.0   # seconds to hold gesture before it counts
GAME_COOLDOWN    = 2.0   # seconds after correct/wrong before scoring resumes
LEARN_COOLDOWN   = 2.5   # seconds after auto-advance in learn mode
RECORD_BURST     = 20
GAME_TIMEOUT     = 8.0
MAX_LIVES       = 3
ASSETS_DIR      = os.path.join(os.path.dirname(__file__), "assets", "letters")

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(9,10),(10,11),(11,12),
    (13,14),(14,15),(15,16),(17,18),(18,19),(19,20),
    (0,5),(5,9),(9,13),(13,17),(0,17),
]

# Modes
MENU = "menu"; LEARN = "learn"; GAME = "game"
WIN  = "win";  GAMEOVER = "gameover"; SCORES = "scores"

# ── Design system (BGR) ───────────────────────────────────────────────────────
C_BG      = (15,  10,  10)    # #0A0A0F  near-black
C_SURFACE = (30,  25,  25)    # slightly lighter surface
C_TEXT    = (255, 255, 255)   # white
C_DIM     = (150, 145, 140)   # secondary text
C_ACCENT  = (247, 142,  79)   # #4F8EF7  electric blue
C_AMBER   = ( 79, 184, 247)   # #F7B84F  amber gold (for letters)
C_SUCCESS = (100, 210,  90)   # green
C_ERROR   = ( 80,  80, 210)   # red
C_BAR_BG  = ( 38,  33,  33)   # thin bar background

_LETTER_NAMES = {
    'Α':'Άλφα',  'Β':'Βήτα',    'Γ':'Γάμα',    'Δ':'Δέλτα',
    'Ε':'Έψιλον','Ζ':'Ζήτα',    'Η':'Ήτα',     'Θ':'Θήτα',
    'Ι':'Ιώτα',  'Κ':'Κάπα',    'Λ':'Λάμδα',   'Μ':'Μι',
    'Ν':'Νι',    'Ξ':'Ξι',      'Ο':'Όμικρον', 'Π':'Πι',
    'Ρ':'Ρο',    'Σ':'Σίγμα',   'Τ':'Ταυ',     'Υ':'Ύψιλον',
    'Φ':'Φι',    'Χ':'Χι',      'Ψ':'Ψι',      'Ω':'Ωμέγα',
}

# ── Font helper ───────────────────────────────────────────────────────────────
def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

# ── Drawing helpers ───────────────────────────────────────────────────────────
def _pil_text(canvas, msg: str, pos: tuple, size=28,
              color=(255,255,255), anchor="lt") -> None:
    pil  = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, msg, font=_font(size), fill=(color[2],color[1],color[0]),
              anchor=anchor)
    canvas[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def _pil_on(canvas, x0, y0, w, h, fn):
    """Apply a Pillow drawing function on a sub-region of canvas."""
    roi = canvas[y0:y0+h, x0:x0+w].copy()
    pil  = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    fn(draw, pil, w, h)
    canvas[y0:y0+h, x0:x0+w] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def _bar(canvas, x, y, w, h, ratio,
         fg=(0,200,100), bg=(40,40,60), radius=4):
    ratio = max(0.0, min(1.0, ratio))
    cv2.rectangle(canvas, (x,y), (x+w,y+h), bg, -1)
    if ratio > 0:
        cv2.rectangle(canvas, (x,y), (x+int(w*ratio),y+h), fg, -1)
    cv2.rectangle(canvas, (x,y), (x+w,y+h), (80,80,100), 1)

def _panel(canvas, x, y, w, h, color=(15,15,28), alpha=0.82):
    ov = canvas.copy()
    cv2.rectangle(ov, (x,y), (x+w,y+h), color, -1)
    cv2.addWeighted(ov, alpha, canvas, 1-alpha, 0, canvas)

def _flash(canvas, color, alpha=0.30):
    ov = canvas.copy()
    h, w = canvas.shape[:2]
    cv2.rectangle(ov, (0,0), (w,h), color, -1)
    cv2.addWeighted(ov, alpha, canvas, 1-alpha, 0, canvas)

def _heart(canvas, cx, cy, size, filled):
    c = (60,60,220) if filled else (55,55,75)
    cv2.circle(canvas, (cx-size//4, cy), size//3, c, -1)
    cv2.circle(canvas, (cx+size//4, cy), size//3, c, -1)
    pts = np.array([[cx-size//2,cy],[cx,cy+int(size*.75)],[cx+size//2,cy]], np.int32)
    cv2.fillPoly(canvas, [pts], c)

def _load_ref(letter: str, w: int, h: int):
    p = os.path.join(ASSETS_DIR, f"{letter}.jpg")
    if os.path.exists(p):
        img = cv2.imread(p)
        if img is not None:
            # Letterbox: fit within (w,h) preserving aspect, center on C_BG bars
            ih, iw = img.shape[:2]
            scale  = min(w / iw, h / ih)
            nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas  = np.full((h, w, 3), C_BG, dtype=np.uint8)
            x0, y0  = (w - nw) // 2, (h - nh) // 2
            canvas[y0:y0+nh, x0:x0+nw] = resized
            return canvas
    # Placeholder
    ph = np.full((h, w, 3), (28,28,45), dtype=np.uint8)
    _pil_text(ph, "?", (w//2, h//2), size=120,
              color=(80,80,120), anchor="mm")
    return ph

# ── App ───────────────────────────────────────────────────────────────────────
class App:
    def __init__(self):
        self.cap      = cv2.VideoCapture(0)
        self.tracker  = HandTracker()
        self.scores   = ScoreTracker()
        self.mode       = MENU
        self._fs        = True
        self._last_vec  = None
        self._hover_idx = -1
        self._quit      = False

        # Learn
        self._learn_idx        = 0
        self._learn_hold_since = 0.0
        self._learn_cooldown_t = 0.0   # timestamp of last auto-advance

        # Game
        self._game_letters: list[str] = []
        self._game_idx      = 0
        self._lives         = MAX_LIVES
        self._score         = 0
        self._game_start    = 0.0
        self._letter_start  = 0.0
        self._hold_since    = 0.0
        self._game_cooldown_t = 0.0   # timestamp of last correct answer

        # Feedback
        self._fb_msg   = ""; self._fb_color = (0,200,0); self._fb_t = 0.0

        # End-game
        self._final_score = 0; self._final_letters = 0
        self._final_time  = 0.0; self._rank = 0

        # Ref image cache
        self._ref_cache: dict[str,np.ndarray] = {}

    # ── Window setup ──────────────────────────────────────────────────────────
    def _init_window(self):
        cv2.namedWindow("Sign Language GR", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Sign Language GR",
                              cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback("Sign Language GR", self._on_mouse)

    def _toggle_fullscreen(self):
        self._fs = not self._fs
        prop = cv2.WINDOW_FULLSCREEN if self._fs else cv2.WINDOW_NORMAL
        cv2.setWindowProperty("Sign Language GR", cv2.WND_PROP_FULLSCREEN, prop)

    # ── Mouse helpers ─────────────────────────────────────────────────────────
    def _canvas_xy(self, mx, my):
        """Convert screen mouse coords → canvas (1280×720) coords."""
        try:
            r = cv2.getWindowImageRect("Sign Language GR")
            if r[2] > 0 and r[3] > 0:
                return (int((mx - r[0]) * WIN_W / r[2]),
                        int((my - r[1]) * WIN_H / r[3]))
        except Exception:
            pass
        return mx, my

    def _menu_hit(self, cx, cy):
        """Return menu item index 0-3 if (cx,cy) is over a menu item, else -1."""
        scx = WIN_W // 2
        for i in range(4):
            iy = 305 + i * 70
            if scx - 190 < cx < scx + 210 and iy - 30 < cy < iy + 30:
                return i
        return -1

    def _on_mouse(self, event, mx, my, flags, param):
        cx, cy = self._canvas_xy(mx, my)
        if self.mode != MENU:
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self._hover_idx = self._menu_hit(cx, cy)
        elif event == cv2.EVENT_LBUTTONDOWN:
            idx = self._menu_hit(cx, cy)
            if idx >= 0:
                self._handle_key([ord("1"),ord("2"),ord("3"),ord("q")][idx],
                                 self._last_vec)

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        self._init_window()
        print("Sign Language GR  |  F=Fullscreen  Q=Quit")

        while True:
            ret, cam = self.cap.read()
            if not ret:
                break
            cam = cv2.flip(cam, 1)
            lms = self.tracker.process(cam)
            vec = gm.normalize(lms) if lms else None

            self._last_vec = vec
            canvas = self._compose(cam, lms, vec)
            cv2.imshow("Sign Language GR", canvas)

            key = cv2.waitKey(1) & 0xFF
            if self._quit or (key in (ord("q"), 27) and self.mode == MENU):
                break
            self._handle_key(key, vec)

        self.cap.release()
        cv2.destroyAllWindows()

    # ── Key handler ───────────────────────────────────────────────────────────
    def _handle_key(self, key, vec):
        if key == ord("f"):
            self._toggle_fullscreen(); return

        if self.mode == MENU:
            if key == ord("1"):
                self._learn_idx=0; self._learn_hold_since=0.0; self.mode=LEARN
            elif key == ord("2"):
                self._start_game()
            elif key == ord("3"): self.mode=SCORES
            elif key in (ord("q"), 27): self._quit=True

        elif self.mode == LEARN:
            if key in (83, ord("n")):   # right arrow or N
                self._learn_idx=(self._learn_idx+1)%len(GREEK_LETTERS)
                self._learn_hold_since=0.0; self._ref_cache.clear()
            elif key in (81, ord("b")): # left arrow or B
                self._learn_idx=(self._learn_idx-1)%len(GREEK_LETTERS)
                self._learn_hold_since=0.0; self._ref_cache.clear()
            elif key == 27: self.mode=MENU

        elif self.mode in (WIN, GAMEOVER):
            if key == ord("2"):   self._start_game()
            elif key == ord("3"): self.mode=SCORES
            elif key == 27:       self.mode=MENU

        elif self.mode == SCORES:
            if key in (27, ord(" ")): self.mode=MENU

        elif self.mode == GAME:
            if key == 27: self.mode=MENU

    # ── Game helpers ──────────────────────────────────────────────────────────
    def _start_game(self, _=None):
        letters = GREEK_LETTERS[:]
        random.shuffle(letters)
        self._game_letters = letters
        self._game_idx=0; self._lives=MAX_LIVES; self._score=0
        self._game_start=self._letter_start=time.monotonic()
        self._hold_since=0.0; self.mode=GAME

    def _next_letter(self):
        self._game_idx+=1; self._hold_since=0.0
        now = time.monotonic()
        self._game_cooldown_t = now
        # Timer starts after cooldown so player doesn't lose time unfairly
        self._letter_start = now + GAME_COOLDOWN
        if self._game_idx>=len(self._game_letters):
            self._end_game(won=True)

    def _wrong(self):
        self._lives-=1; self._hold_since=0.0
        self._game_cooldown_t = time.monotonic()
        self._letter_start = time.monotonic() + GAME_COOLDOWN
        if self._lives<=0: self._end_game(won=False)

    def _end_game(self, won):
        elapsed=time.monotonic()-self._game_start
        self._final_score=self._score; self._final_letters=self._game_idx
        self._final_time=elapsed
        self._rank=self.scores.add_score(self._score, self._game_idx, elapsed)
        self.mode=WIN if won else GAMEOVER

    def _set_fb(self, msg, color=(0,200,0)):
        self._fb_msg=msg; self._fb_color=color; self._fb_t=time.monotonic()

    # ── Ref image ─────────────────────────────────────────────────────────────
    def _ref(self, letter, w, h):
        key = f"{letter}_{w}_{h}"
        if key not in self._ref_cache:
            self._ref_cache[key] = _load_ref(letter, w, h)
        return self._ref_cache[key]

    # ── Composition ───────────────────────────────────────────────────────────
    def _compose(self, cam, lms, vec) -> np.ndarray:
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)

        # ── Right half: camera ────────────────────────────────────────────
        cam_r = cv2.resize(cam, (HALF, WIN_H))
        canvas[:, HALF:] = cam_r

        if lms:
            h_c, w_c = cam.shape[:2]
            sx = HALF / w_c; sy = WIN_H / h_c
            for a, b in _CONNECTIONS:
                pa = (int(lms[a].x*w_c*sx)+HALF, int(lms[a].y*h_c*sy))
                pb = (int(lms[b].x*w_c*sx)+HALF, int(lms[b].y*h_c*sy))
                cv2.line(canvas, pa, pb, C_ACCENT, 1, cv2.LINE_AA)
            for lm in lms:
                pt = (int(lm.x*w_c*sx)+HALF, int(lm.y*h_c*sy))
                cv2.circle(canvas, pt, 4, C_TEXT, -1)
                cv2.circle(canvas, pt, 4, C_ACCENT, 1)

        # ── Left half: dark background ────────────────────────────────────
        canvas[:, :HALF] = C_BG

        # ── Mode-specific rendering ───────────────────────────────────────
        if   self.mode == MENU:     self._draw_menu(canvas, vec)
        elif self.mode == LEARN:    self._draw_learn(canvas, vec)
        elif self.mode == GAME:     self._draw_game(canvas, vec)
        elif self.mode in (WIN, GAMEOVER): self._draw_end(canvas)
        elif self.mode == SCORES:   self._draw_scores(canvas)

        # ── F key hint (bottom-right, very subtle) ────────────────────────
        _pil_text(canvas, "F", (WIN_W-20, WIN_H-16),
                  size=13, color=(60,58,55), anchor="rm")

        return canvas

    # ─────────────────────────────────────────────────────────────────────────
    # MENU
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_menu(self, canvas, vec):
        # Full-canvas background: reference image with heavy dark overlay
        bg = self._ref(GREEK_LETTERS[0], WIN_W, WIN_H)
        cv2.addWeighted(np.full_like(bg, C_BG, dtype=np.uint8), 0.82,
                        bg, 0.18, 0, canvas)

        cx = WIN_W // 2

        # App title
        _pil_text(canvas, "Sign Language GR", (cx, 175),
                  size=46, color=C_TEXT, anchor="mm")
        _pil_text(canvas, "Ελληνική Νοηματική Γλώσσα", (cx, 222),
                  size=19, color=C_DIM, anchor="mm")

        # Thin accent separator
        cv2.line(canvas, (cx-70, 250), (cx+70, 250), C_ACCENT, 1)

        # Menu items
        items = [("1", "Εκμάθηση"), ("2", "Παιχνίδι"),
                 ("3", "Βαθμολογίες"), ("Q", "Έξοδος")]
        for i, (key, label) in enumerate(items):
            y = 305 + i * 70
            col = C_DIM if key == "Q" else C_TEXT
            # Hover highlight
            if self._hover_idx == i:
                ov = canvas.copy()
                cv2.rectangle(ov, (cx-192, y-28), (cx+212, y+28), C_SURFACE, -1)
                cv2.addWeighted(ov, 0.55, canvas, 0.45, 0, canvas)
                cv2.rectangle(canvas, (cx-192, y-28), (cx+212, y+28), C_ACCENT, 1)
            _pil_text(canvas, key,   (cx - 130, y), size=22, color=C_ACCENT, anchor="mm")
            _pil_text(canvas, label, (cx - 100, y), size=26, color=col,      anchor="lm")

        # Bottom stats
        rec  = len(gm.recorded_letters())
        best = self.scores.best_score()
        _pil_text(canvas, f"{rec} / 24 γράμματα  ·  Ρεκόρ: {best} pts",
                  (cx, WIN_H - 28), size=15, color=C_DIM, anchor="mm")

        # Live detection preview (right, subtle overlay)
        if vec is not None:
            letter, score = gm.best_match(vec, gm.recorded_letters())
            if letter and score > 0.65:
                ov = canvas.copy()
                cv2.rectangle(ov, (HALF, 0), (WIN_W, 72), C_SURFACE, -1)
                cv2.addWeighted(ov, 0.75, canvas, 0.25, 0, canvas)
                _pil_text(canvas, f"{letter}",
                          (HALF + 36, 36), size=32, color=C_AMBER, anchor="lm")
                _pil_text(canvas, f"{int(score*100)}%",
                          (HALF + 90, 36), size=20, color=C_DIM, anchor="lm")

    # ─────────────────────────────────────────────────────────────────────────
    # LEARN
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_learn(self, canvas, vec):
        letter      = GREEK_LETTERS[self._learn_idx]
        now         = time.monotonic()
        in_cooldown = (now - self._learn_cooldown_t) < LEARN_COOLDOWN
        score       = 0.0
        if not in_cooldown and vec is not None:
            score = gm.match_score(vec, letter)

        # ── Auto-advance ──────────────────────────────────────────────────
        if not in_cooldown:
            if score >= MATCH_THRESHOLD:
                if self._learn_hold_since == 0.0:
                    self._learn_hold_since = now
                elif now - self._learn_hold_since >= HOLD_SECONDS:
                    self._learn_hold_since = 0.0
                    self._learn_cooldown_t = now
                    self._learn_idx = (self._learn_idx + 1) % len(GREEK_LETTERS)
                    self._ref_cache.clear()
                    letter = GREEK_LETTERS[self._learn_idx]
                    score  = 0.0
                    _flash(canvas, C_SUCCESS, alpha=0.18)
            else:
                self._learn_hold_since = 0.0

        # ── Left: reference image, full-bleed ────────────────────────────
        PROGRESS_H = 4
        img_h = WIN_H - PROGRESS_H
        img   = self._ref(letter, HALF, img_h)
        canvas[:img_h, :HALF] = img

        # Dark gradient on bottom of image
        GRAD = 180
        for dy in range(GRAD):
            a = (dy / GRAD) ** 1.4
            y = img_h - GRAD + dy
            row = canvas[y, :HALF].astype(np.float32)
            dark = np.array(C_BG, dtype=np.float32)
            canvas[y, :HALF] = (row*(1-a) + dark*a).astype(np.uint8)

        # Letter + counter overlaid on image
        _pil_text(canvas, letter,
                  (22, img_h - 18), size=80, color=C_AMBER, anchor="lb")
        _pil_text(canvas, f"{self._learn_idx+1} / {len(GREEK_LETTERS)}",
                  (HALF-16, img_h - 18), size=18, color=C_DIM, anchor="rb")
        _pil_text(canvas, "← →",
                  (16, 20), size=15, color=(65, 60, 58), anchor="lt")

        # Thin progress bar at very bottom
        canvas[img_h:, :HALF] = C_BG
        ratio = self._learn_idx / max(1, len(GREEK_LETTERS) - 1)
        filled = max(2, int(HALF * ratio))
        cv2.rectangle(canvas, (0, img_h), (filled, WIN_H), C_ACCENT, -1)
        cv2.rectangle(canvas, (filled, img_h), (HALF, WIN_H), C_BAR_BG, -1)

        # ── Right: feedback overlay at bottom ────────────────────────────
        FB_H = 90
        fb_y = WIN_H - FB_H
        ov = canvas.copy()
        cv2.rectangle(ov, (HALF, fb_y), (WIN_W, WIN_H), (8, 6, 6), -1)
        cv2.addWeighted(ov, 0.82, canvas, 0.18, 0, canvas)

        PX = HALF + 28;  PW = HALF - 56

        if in_cooldown:
            remain = LEARN_COOLDOWN - (now - self._learn_cooldown_t)
            _pil_text(canvas, f"Ετοιμάσου…  {remain:.1f}s",
                      (HALF + HALF//2, fb_y + 34), size=20, color=C_DIM, anchor="mm")
            _bar(canvas, PX, fb_y + 54, PW, 3, remain/LEARN_COOLDOWN,
                 fg=C_ACCENT, bg=C_BAR_BG)
        elif vec is None:
            _pil_text(canvas, "Δείξε το χέρι σου",
                      (HALF + HALF//2, fb_y + 44), size=19,
                      color=(70, 65, 62), anchor="mm")
        else:
            bar_col = C_SUCCESS if score >= MATCH_THRESHOLD else C_ACCENT
            _pil_text(canvas, f"{int(score*100)}%",
                      (PX, fb_y + 28), size=24, color=bar_col, anchor="lm")
            _bar(canvas, PX, fb_y + 46, PW, 3, score, fg=bar_col, bg=C_BAR_BG)

            if score >= MATCH_THRESHOLD and self._learn_hold_since:
                hold_r = (now - self._learn_hold_since) / HOLD_SECONDS
                _pil_text(canvas, "Κράτα…",
                          (PX, fb_y + 62), size=17, color=C_AMBER, anchor="lm")
                _bar(canvas, PX, fb_y + 76, PW, 3, hold_r, fg=C_AMBER, bg=C_BAR_BG)
            elif score >= MATCH_THRESHOLD:
                _pil_text(canvas, "✓  Κράτα για να προχωρήσεις",
                          (HALF + HALF//2, fb_y + 68), size=17,
                          color=C_SUCCESS, anchor="mm")
            else:
                _pil_text(canvas, "Αντέγραψε την χειρονομία από μνήμη",
                          (HALF + HALF//2, fb_y + 68), size=17,
                          color=(72, 68, 65), anchor="mm")

    # ─────────────────────────────────────────────────────────────────────────
    # GAME
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_game(self, canvas, vec):
        if not self._game_letters:
            self.mode=MENU; return

        letter      = self._game_letters[self._game_idx % len(self._game_letters)]
        now         = time.monotonic()
        in_cooldown = (now - self._game_cooldown_t) < GAME_COOLDOWN
        score       = 0.0 if in_cooldown or vec is None else gm.match_score(vec, letter)
        elapsed     = max(0.0, now - self._letter_start)   # doesn't count cooldown time
        time_left   = max(0.0, GAME_TIMEOUT - elapsed)

        # Hold detection (only outside cooldown)
        if not in_cooldown:
            if score >= MATCH_THRESHOLD:
                if self._hold_since == 0.0:
                    self._hold_since = now
                elif now - self._hold_since >= HOLD_SECONDS:
                    pts = 10 + (5 if elapsed < 3.0 else 0)
                    self._score += pts
                    self._set_fb(f"✓  ΣΩΣΤΟ!  +{pts} pts", (0,220,80))
                    self._hold_since = 0.0
                    self._next_letter()
                    return
            else:
                self._hold_since = 0.0

            if time_left <= 0:
                self._set_fb("✗  ΛΑΘΟΣ!", (60,60,220))
                self._wrong(); return

        # ── Feedback flash ────────────────────────────────────────────────
        if now - self._fb_t < 0.4:
            _flash(canvas, self._fb_color, alpha=0.18)

        # ── Left: score + lives ───────────────────────────────────────────
        _pil_text(canvas, str(self._score),
                  (28, 38), size=32, color=C_TEXT, anchor="lm")
        _pil_text(canvas, "pts",
                  (28 + len(str(self._score))*19 + 4, 42), size=15,
                  color=C_DIM, anchor="lm")
        for i in range(MAX_LIVES):
            _heart(canvas, HALF-18-i*30, 38, 13, filled=(i < self._lives))
        cv2.line(canvas, (28, 62), (HALF-28, 62), C_SURFACE, 1)

        # ── Letter display (center) ───────────────────────────────────────
        cx = HALF // 2
        cy = (62 + WIN_H - 88) // 2
        _pil_text(canvas, f"{self._game_idx+1}  /  {len(self._game_letters)}",
                  (cx, cy - 105), size=17, color=C_DIM, anchor="mm")
        _pil_text(canvas, letter,
                  (cx, cy), size=150, color=C_AMBER, anchor="mm")
        _pil_text(canvas, _LETTER_NAMES.get(letter, ""),
                  (cx, cy + 95), size=22, color=C_DIM, anchor="mm")

        # ── Bottom: timer + hold / cooldown ──────────────────────────────
        by = WIN_H - 88
        cv2.line(canvas, (28, by), (HALF-28, by), C_SURFACE, 1)

        if in_cooldown:
            cd_r = (GAME_COOLDOWN - (now - self._game_cooldown_t)) / GAME_COOLDOWN
            _pil_text(canvas, "Ετοιμάσου…",
                      (28, by + 30), size=18, color=C_DIM, anchor="lm")
            _bar(canvas, 28, by + 48, HALF-56, 3, cd_r,
                 fg=C_ACCENT, bg=C_BAR_BG)
        else:
            t_col = C_ACCENT if time_left > 3 else C_ERROR
            _pil_text(canvas, f"{time_left:.1f}s",
                      (28, by + 26), size=20, color=t_col, anchor="lm")
            _bar(canvas, 28, by + 38, HALF-56, 3,
                 time_left / GAME_TIMEOUT, fg=t_col, bg=C_BAR_BG)
            hold_ratio = (now-self._hold_since)/HOLD_SECONDS if self._hold_since else 0.0
            _pil_text(canvas, "Κράτα",
                      (28, by + 60), size=17, color=C_DIM, anchor="lm")
            _bar(canvas, 28, by + 72, HALF-56, 3, hold_ratio,
                 fg=C_AMBER, bg=C_BAR_BG)

        _pil_text(canvas, "ESC",
                  (HALF-16, WIN_H-10), size=12, color=(55,52,50), anchor="rb")

        # ── Right: feedback overlay ───────────────────────────────────────
        FB_H = 80
        fb_y = WIN_H - FB_H
        ov   = canvas.copy()
        cv2.rectangle(ov, (HALF, fb_y), (WIN_W, WIN_H), (8, 6, 6), -1)
        cv2.addWeighted(ov, 0.82, canvas, 0.18, 0, canvas)

        PX = HALF + 28;  PW = HALF - 56
        bar_col = C_SUCCESS if score >= MATCH_THRESHOLD else C_ACCENT

        if now - self._fb_t < 1.8 and self._fb_msg:
            fb_col = C_SUCCESS if "ΣΩΣΤΟ" in self._fb_msg else C_ERROR
            _pil_text(canvas, self._fb_msg,
                      (HALF + HALF//2, fb_y + 40), size=26, color=fb_col, anchor="mm")
        else:
            _pil_text(canvas, f"{int(score*100)}%",
                      (PX, fb_y + 26), size=22, color=bar_col, anchor="lm")
            _bar(canvas, PX, fb_y + 42, PW, 3, score, fg=bar_col, bg=C_BAR_BG)
            if self._hold_since and score >= MATCH_THRESHOLD:
                hold_r = (now - self._hold_since) / HOLD_SECONDS
                _pil_text(canvas, "Κράτα…",
                          (PX, fb_y + 58), size=16, color=C_AMBER, anchor="lm")
                _bar(canvas, PX, fb_y + 68, PW, 3, hold_r, fg=C_AMBER, bg=C_BAR_BG)
            else:
                hint = "✓" if score >= MATCH_THRESHOLD else "Κάνε το gesture από μνήμη"
                _pil_text(canvas, hint,
                          (HALF + HALF//2, fb_y + 60), size=16,
                          color=C_SUCCESS if score >= MATCH_THRESHOLD else (65,62,60),
                          anchor="mm")

    # ─────────────────────────────────────────────────────────────────────────
    # WIN / GAMEOVER
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_end(self, canvas):
        won = (self.mode == WIN)
        cx  = WIN_W // 2

        # Dark overlay over full canvas
        ov = np.full_like(canvas, C_BG, dtype=np.uint8)
        cv2.addWeighted(ov, 0.88, canvas, 0.12, 0, canvas)

        # Result
        if won:
            _pil_text(canvas, "Νίκη", (cx, WIN_H//2 - 140),
                      size=72, color=C_AMBER, anchor="mm")
        else:
            _pil_text(canvas, "Game Over", (cx, WIN_H//2 - 140),
                      size=60, color=C_ERROR, anchor="mm")

        # Thin separator
        cv2.line(canvas, (cx-100, WIN_H//2-95), (cx+100, WIN_H//2-95), C_SURFACE, 1)

        # Stats
        _pil_text(canvas, f"{self._final_score} pts",
                  (cx, WIN_H//2 - 52), size=38, color=C_TEXT, anchor="mm")
        _pil_text(canvas,
                  f"{self._final_letters} γράμματα  ·  {self._final_time:.1f}s",
                  (cx, WIN_H//2 + 4), size=20, color=C_DIM, anchor="mm")

        if self._rank == 1:
            _pil_text(canvas, "Νέο ρεκόρ",
                      (cx, WIN_H//2 + 52), size=22, color=C_ACCENT, anchor="mm")
        elif 1 < self._rank <= 5:
            _pil_text(canvas, f"#{self._rank} στην κατάταξη",
                      (cx, WIN_H//2 + 52), size=22, color=C_DIM, anchor="mm")

        # Actions
        cv2.line(canvas, (cx-100, WIN_H//2+86), (cx+100, WIN_H//2+86), C_SURFACE, 1)
        actions = [("2", "Ξανά"), ("3", "Βαθμολογίες"), ("ESC", "Μενού")]
        total_w = 380
        x0 = cx - total_w // 2
        for key, label in actions:
            _pil_text(canvas, key,   (x0, WIN_H//2+116), size=18, color=C_ACCENT, anchor="lm")
            _pil_text(canvas, label, (x0+34, WIN_H//2+116), size=18, color=C_DIM, anchor="lm")
            x0 += total_w // len(actions)

    # ─────────────────────────────────────────────────────────────────────────
    # SCORES
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_scores(self, canvas):
        cx = WIN_W // 2
        ov = np.full_like(canvas, C_BG, dtype=np.uint8)
        cv2.addWeighted(ov, 0.92, canvas, 0.08, 0, canvas)

        _pil_text(canvas, "Βαθμολογίες", (cx, 80),
                  size=40, color=C_TEXT, anchor="mm")
        cv2.line(canvas, (cx-120, 108), (cx+120, 108), C_ACCENT, 1)

        sc = self.scores.high_scores()
        if not sc:
            _pil_text(canvas, "Δεν υπάρχουν βαθμολογίες ακόμα.",
                      (cx, WIN_H//2), size=22, color=C_DIM, anchor="mm")
        else:
            ranks = ["01", "02", "03", "04", "05"]
            for i, e in enumerate(sc):
                y = 148 + i * 88
                col = C_AMBER if i == 0 else C_DIM
                _pil_text(canvas, ranks[i],
                          (cx - 260, y), size=18, color=col, anchor="lm")
                _pil_text(canvas, f"{e['score']}",
                          (cx - 220, y), size=34, color=C_TEXT, anchor="lm")
                _pil_text(canvas, "pts",
                          (cx - 220 + len(str(e['score']))*20, y+10),
                          size=14, color=C_DIM, anchor="lm")
                _pil_text(canvas,
                          f"{e['letters']} γράμματα  ·  {e['time']}s  ·  {e.get('date','')}",
                          (cx - 220, y + 36), size=16, color=C_DIM, anchor="lm")
                if i < len(sc)-1:
                    cv2.line(canvas, (cx-260, y+62), (cx+260, y+62), C_SURFACE, 1)

        _pil_text(canvas, "ESC  Πίσω",
                  (cx, WIN_H - 36), size=16, color=(65, 62, 60), anchor="mm")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().run()
