#!/usr/bin/env python3
"""
export_references.py — Export the self-recorded landmark samples for the web app.

Dumps EVERY sample from data/references/*.npy (originals + augmentation) into
web/public/references.json, so the in-browser KNN is trained on the exact same
data as the desktop classifier (gesture_matcher.train_classifier). The browser
adds x-mirror augmentation at load, mirroring the desktop pipeline.

Run from the project root:
    python3 web/export_references.py
"""

import os
import json
import numpy as np

# project root = parent of this file's directory (web/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys; sys.path.insert(0, ROOT)
from progress_tracker import GREEK_LETTERS  # noqa: E402

REFS_DIR = os.path.join(ROOT, "data", "references")
OUT_PATH = os.path.join(ROOT, "web", "public", "references.json")
DECIMALS = 4   # plenty for normalized landmark coords; keeps the file small


def main():
    out = {}
    total = 0
    for letter in GREEK_LETTERS:
        p = os.path.join(REFS_DIR, f"{letter}.npy")
        if not os.path.exists(p):
            print(f"  {letter}: no samples — skipped")
            continue
        a = np.load(p).astype(np.float32)
        out[letter] = [[round(float(v), DECIMALS) for v in s] for s in a]
        total += len(a)
        print(f"  {letter}: {len(a)} samples")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

    kb = os.path.getsize(OUT_PATH) / 1024
    print(f"\n✓ {len(out)} letters, {total} samples → {os.path.relpath(OUT_PATH, ROOT)} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
