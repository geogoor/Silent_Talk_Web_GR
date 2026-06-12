import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# ── Text ──────────────────────────────────────────────────────────────────────

def text(frame, msg: str, pos: tuple, size: int = 28,
         color: tuple = (255, 255, 255)) -> np.ndarray:
    pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, msg, font=_font(size), fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def centered_text(frame, msg: str, y: int, size: int = 28,
                  color: tuple = (255, 255, 255)) -> np.ndarray:
    pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    f    = _font(size)
    bbox = draw.textbbox((0, 0), msg, font=f)
    x    = (frame.shape[1] - (bbox[2] - bbox[0])) // 2
    draw.text((x, y), msg, font=f, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ── Shapes ────────────────────────────────────────────────────────────────────

def bar(frame, x: int, y: int, w: int, h: int, ratio: float,
        fg=(0, 200, 100), bg=(40, 40, 55)) -> None:
    ratio = max(0.0, min(1.0, ratio))
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
    if ratio > 0:
        cv2.rectangle(frame, (x, y), (x + int(w * ratio), y + h), fg, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 120), 1)


def panel(frame, x: int, y: int, w: int, h: int,
          color=(15, 15, 28), alpha: float = 0.75) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def flash(frame, color: tuple, alpha: float = 0.28) -> None:
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def heart(frame, cx: int, cy: int, size: int, filled: bool) -> None:
    color = (60, 60, 220) if filled else (60, 60, 60)
    cv2.circle(frame, (cx - size // 4, cy), size // 3, color, -1)
    cv2.circle(frame, (cx + size // 4, cy), size // 3, color, -1)
    pts = np.array([
        [cx - size // 2, cy],
        [cx,             cy + int(size * 0.75)],
        [cx + size // 2, cy],
    ], np.int32)
    cv2.fillPoly(frame, [pts], color)
