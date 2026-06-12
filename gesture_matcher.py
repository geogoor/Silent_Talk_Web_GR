import os
import pickle
import numpy as np

_REFS_DIR  = os.path.join(os.path.dirname(__file__), "data", "references")
_CLF_PATH  = os.path.join(os.path.dirname(__file__), "data", "classifier.pkl")

# Cached classifier (loaded once)
_clf   = None
_labels = None


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize(landmarks) -> np.ndarray:
    """Position- and scale-invariant landmark vector (shape: 63,)."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9])
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


def _mirror_x(vec: np.ndarray) -> np.ndarray:
    """Flip x-axis (handles camera-flip vs unflipped training video)."""
    v = vec.copy()
    v[0::3] *= -1
    return v


# ── Storage helpers ───────────────────────────────────────────────────────────

def _path(letter: str) -> str:
    return os.path.join(_REFS_DIR, f"{letter}.npy")


def has_reference(letter: str) -> bool:
    return os.path.exists(_path(letter))


def recorded_letters() -> list[str]:
    os.makedirs(_REFS_DIR, exist_ok=True)
    return sorted(f[:-4] for f in os.listdir(_REFS_DIR) if f.endswith(".npy"))


def save_sample(letter: str, vec: np.ndarray) -> int:
    os.makedirs(_REFS_DIR, exist_ok=True)
    p = _path(letter)
    if os.path.exists(p):
        data = np.vstack([np.load(p), vec.reshape(1, -1)])
    else:
        data = vec.reshape(1, -1)
    np.save(p, data)
    return len(data)


def delete_reference(letter: str) -> None:
    if os.path.exists(_path(letter)):
        os.remove(_path(letter))


def sample_count(letter: str) -> int:
    if not has_reference(letter):
        return 0
    return len(np.load(_path(letter)))


# ── KNN Classifier ────────────────────────────────────────────────────────────

def train_classifier() -> float:
    """Train a KNN classifier on all available samples. Returns cross-val accuracy."""
    global _clf, _labels
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import normalize as sk_normalize
    from sklearn.model_selection import cross_val_score

    letters = recorded_letters()
    X, y = [], []
    for letter in letters:
        p = _path(letter)
        if not os.path.exists(p):
            continue
        data = np.load(p)
        for vec in data:
            X.append(vec)
            y.append(letter)
            # Include x-mirror as extra sample to handle camera flip
            X.append(_mirror_x(vec))
            y.append(letter)

    if len(X) < 10:
        return 0.0

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    # L2-normalize for cosine KNN
    X_norm = sk_normalize(X)

    k = min(7, max(1, len(X) // (len(letters) * 3)))
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    clf.fit(X_norm, y)

    # Cross-val accuracy
    try:
        cv_scores = cross_val_score(clf, X_norm, y, cv=min(5, len(letters)), scoring="accuracy")
        acc = float(cv_scores.mean())
    except Exception:
        acc = 0.0

    # Save
    os.makedirs(os.path.dirname(_CLF_PATH), exist_ok=True)
    with open(_CLF_PATH, "wb") as f:
        pickle.dump({"clf": clf, "labels": letters}, f)

    _clf    = clf
    _labels = letters
    return acc


def _load_classifier():
    global _clf, _labels
    if _clf is not None:
        return True
    if not os.path.exists(_CLF_PATH):
        return False
    try:
        from sklearn.preprocessing import normalize as sk_normalize  # noqa
        with open(_CLF_PATH, "rb") as f:
            obj = pickle.load(f)
        _clf    = obj["clf"]
        _labels = obj["labels"]
        return True
    except Exception:
        return False


def _predict_knn(vec: np.ndarray) -> tuple[str | None, float]:
    """Return (letter, confidence) using trained KNN. Uses both normal and mirrored."""
    from sklearn.preprocessing import normalize as sk_normalize

    def _query(v):
        vn = sk_normalize(v.reshape(1, -1))
        proba = _clf.predict_proba(vn)[0]
        idx   = int(np.argmax(proba))
        return _clf.classes_[idx], float(proba[idx])

    l1, s1 = _query(vec)
    l2, s2 = _query(_mirror_x(vec))
    return (l1, s1) if s1 >= s2 else (l2, s2)


# ── Cosine fallback ───────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _cosine_score(live_vec: np.ndarray, letter: str) -> float:
    refs     = np.load(_path(letter))
    mirrored = _mirror_x(live_vec)
    scores   = []
    for r in refs:
        scores.append(_cosine(live_vec, r))
        scores.append(_cosine(mirrored, r))
    scores.sort(reverse=True)
    top_k = scores[:min(5, len(scores))]
    return float(np.mean(top_k))


# ── Public API ────────────────────────────────────────────────────────────────

def match_score(live_vec: np.ndarray, letter: str) -> float:
    """Similarity score in [0, 1] for one letter. Uses KNN if trained."""
    if not has_reference(letter):
        return 0.0
    if _load_classifier() and letter in (_labels or []):
        pred_letter, conf = _predict_knn(live_vec)
        if pred_letter == letter:
            return conf
        # Also compute raw cosine to avoid sudden drops
        return max(conf * 0.5, _cosine_score(live_vec, letter) * 0.8)
    return _cosine_score(live_vec, letter)


def best_match(live_vec: np.ndarray, letters: list[str]) -> tuple[str | None, float]:
    """Return (letter, score) for the best matching letter."""
    if _load_classifier():
        pred_letter, conf = _predict_knn(live_vec)
        if pred_letter in letters:
            return pred_letter, conf
    best_l, best_s = None, 0.0
    for l in letters:
        s = _cosine_score(live_vec, l)
        if s > best_s:
            best_s, best_l = s, l
    return best_l, best_s
