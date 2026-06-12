"""
augment_data.py — Εμπλουτισμός υπαρχόντων landmark samples με augmentation.

Τρέξε μία φορά:
  python3 augment_data.py

Για κάθε γράμμα, παίρνει τα υπάρχοντα δείγματα και προσθέτει:
  - μικρό Gaussian noise
  - x-mirror (για να καλύψει το camera-flip vs training-video mismatch)
  - μικρή κλιμάκωση (±10%)
  - μικρή 3D περιστροφή (±10°)
"""

import os
import numpy as np
from progress_tracker import GREEK_LETTERS

_REFS_DIR = os.path.join(os.path.dirname(__file__), "data", "references")
TARGET_SAMPLES = 120   # samples per letter after augmentation


def _mirror_x(vec: np.ndarray) -> np.ndarray:
    v = vec.copy()
    v[0::3] *= -1
    return v


def _add_noise(vec: np.ndarray, sigma=0.015) -> np.ndarray:
    return vec + np.random.randn(*vec.shape).astype(np.float32) * sigma


def _scale(vec: np.ndarray) -> np.ndarray:
    s = np.random.uniform(0.90, 1.10)
    return vec * s


def _rotate_y(vec: np.ndarray, max_deg=10) -> np.ndarray:
    """Rotate all 3D points around Y axis."""
    angle = np.radians(np.random.uniform(-max_deg, max_deg))
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    pts = vec.reshape(21, 3)
    pts = pts @ R.T
    return pts.flatten()


def _path(letter: str) -> str:
    return os.path.join(_REFS_DIR, f"{letter}.npy")


def augment() -> None:
    os.makedirs(_REFS_DIR, exist_ok=True)
    print(f"Augmentation → στόχος {TARGET_SAMPLES} δείγματα ανά γράμμα\n")

    total_added = 0
    for letter in GREEK_LETTERS:
        p = _path(letter)
        if not os.path.exists(p):
            print(f"  {letter}: ΔΕΝ υπάρχουν δείγματα — παράλειψη")
            continue

        orig = np.load(p)
        n    = len(orig)
        need = max(0, TARGET_SAMPLES - n)

        if need == 0:
            print(f"  {letter}: {n} δείγματα  (αρκετά, παράλειψη)")
            continue

        augmented = []

        # Pass 1: x-mirror of all originals (fixes camera flip mismatch)
        for vec in orig:
            augmented.append(_mirror_x(vec))
            if len(augmented) >= need:
                break

        # Pass 2: noise
        while len(augmented) < need:
            vec = orig[np.random.randint(len(orig))]
            augmented.append(_add_noise(vec))

        # Pass 3: scale
        while len(augmented) < need:
            vec = orig[np.random.randint(len(orig))]
            augmented.append(_scale(vec))

        # Pass 4: rotation
        while len(augmented) < need:
            vec = orig[np.random.randint(len(orig))]
            augmented.append(_rotate_y(vec))

        # Combine and save
        extra = np.array(augmented[:need], dtype=np.float32)
        data  = np.vstack([orig, extra])
        np.save(p, data)
        total_added += need
        print(f"  {letter}: {n} → {len(data)} δείγματα  (+{need})")

    print(f"\n✓ Ολοκληρώθηκε!  Προστέθηκαν {total_added} νέα δείγματα.")


if __name__ == "__main__":
    augment()
