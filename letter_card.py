import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets", "letters")


def _screenshot_path(letter: str) -> str:
    return os.path.join(_ASSETS_DIR, f"{letter}.jpg")


def has_screenshot(letter: str) -> bool:
    return os.path.exists(_screenshot_path(letter))

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


def make_card(letter: str, sample_count: int,
              width: int = 280, height: int = 380) -> np.ndarray:
    """Return a BGR numpy array with the letter reference card.
    If a screenshot from the training video exists, it is shown as the
    reference image; otherwise a placeholder card is generated."""

    shot_path = _screenshot_path(letter)

    if os.path.exists(shot_path):
        # ── Photo card (from training video) ──────────────────────────────
        photo = cv2.imread(shot_path)
        # Crop to portrait and resize
        ph, pw = photo.shape[:2]
        crop_h = min(ph, int(pw * 1.1))
        y0 = max(0, (ph - crop_h) // 2)
        photo = photo[y0:y0 + crop_h, :]
        photo = cv2.resize(photo, (width, height - 70))

        card = np.zeros((height, width, 3), dtype=np.uint8)
        card[:height - 70] = photo

        # Bottom status strip
        card[height - 70:] = (18, 18, 32)
        pil  = Image.fromarray(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)

        # ΕΝΓ badge overlay on photo
        draw.rounded_rectangle([width - 58, 6, width - 6, 30],
                                radius=6, fill=(50, 80, 160, 200))
        draw.text((width - 32, 18), "ΕΝΓ", font=_font(14),
                  fill=(200, 220, 255), anchor="mm")

        # Letter overlay (bottom-left of photo)
        draw.text((10, height - 120), letter, font=_font(64),
                  fill=(255, 215, 50),
                  stroke_width=2, stroke_fill=(0, 0, 0))

        # Status strip
        if sample_count >= 20:
            draw.rounded_rectangle([10, height - 58, width - 10, height - 10],
                                   radius=6, fill=(20, 100, 50))
            draw.text((width // 2, height - 34), f"✓ Έτοιμο! ({sample_count})",
                      font=_font(17), fill=(120, 255, 160), anchor="mm")
        else:
            draw.text((width // 2, height - 50), f"Δείγματα: {sample_count}/20",
                      font=_font(16), fill=(255, 180, 60), anchor="mm")
            bw = width - 40
            draw.rectangle([20, height - 28, 20 + bw, height - 14],
                           fill=(50, 50, 70))
            draw.rectangle([20, height - 28,
                            20 + int(bw * min(sample_count / 20, 1.0)), height - 14],
                           fill=(255, 140, 30))

        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    else:
        # ── Placeholder card (no video screenshot yet) ────────────────────
        img  = Image.new("RGB", (width, height), (18, 18, 32))
        draw = ImageDraw.Draw(img)

        for y in range(60):
            a = int(80 * (1 - y / 60))
            draw.line([(0, y), (width, y)], fill=(60, 40, a + 20))

        draw.rounded_rectangle([width//2 - 36, 10, width//2 + 36, 38],
                                radius=8, fill=(50, 80, 160))
        draw.text((width // 2, 24), "ΕΝΓ", font=_font(18),
                  fill=(200, 220, 255), anchor="mm")

        draw.text((width // 2, 130), letter, font=_font(130),
                  fill=(255, 215, 50), anchor="mm")

        draw.line([(30, 210), (width - 30, 210)], fill=(60, 60, 80), width=1)
        draw.text((width // 2, 240), "Τρέξε video_trainer.py", font=_font(14),
                  fill=(160, 160, 190), anchor="mm")
        draw.text((width // 2, 262), "για να προσθέσεις εικόνα.", font=_font(14),
                  fill=(160, 160, 190), anchor="mm")
        draw.line([(30, 290), (width - 30, 290)], fill=(60, 60, 80), width=1)

        if sample_count >= 20:
            draw.rounded_rectangle([width//2 - 70, 300, width//2 + 70, 330],
                                   radius=8, fill=(20, 120, 60))
            draw.text((width // 2, 315), f"✓ Έτοιμο! ({sample_count})",
                      font=_font(17), fill=(120, 255, 160), anchor="mm")
        elif sample_count > 0:
            draw.text((width // 2, 305), f"Δείγματα: {sample_count}/20",
                      font=_font(16), fill=(255, 180, 60), anchor="mm")
            bw = width - 60
            draw.rectangle([30, 330, 30 + bw, 342], fill=(50, 50, 70))
            draw.rectangle([30, 330, 30 + int(bw * sample_count / 20), 342],
                           fill=(255, 150, 30))
        else:
            draw.text((width // 2, 315), "[SPACE] Καταγραφή",
                      font=_font(16), fill=(200, 200, 80), anchor="mm")

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
