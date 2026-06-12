# Sign Language GR — Ελληνική Νοηματική Γλώσσα

**English** · [Ελληνικά](README.el.md)

An interactive learning game for the **Greek Sign Language (ΕΝΓ) alphabet**, built
from scratch — including a **custom-trained gesture recognition model** using my own
hand as training data. It ships in two flavours:

- 🌐 **Web edition** (`web/`) — runs entirely in the browser, no install. Hand
  tracking via MediaPipe WASM, the KNN classifier ported to JavaScript. Deployable
  to Vercel as a static site.
- 🖥️ **Desktop edition** (Python + OpenCV) — the original app and the full
  data-collection / self-training toolchain.

> **Portfolio project** — No pre-existing ΕΝΓ dataset exists publicly. Every
> component, from data collection to the trained classifier, was designed and built
> by hand.

---

## Try it online

The web edition needs only a webcam and a modern browser (Chrome/Edge/Firefox).
Camera access requires HTTPS — Vercel provides it automatically, and `localhost`
works for local dev.

```bash
cd web
npm install
npm run dev        # open the printed localhost URL, allow camera
```

Everything runs **client-side** — no server, no data leaves your device.

---

## How It Works

The app uses your webcam to detect your hand in real time, extracts 21 3D landmarks
via **MediaPipe**, normalizes them into a scale- and position-invariant vector, and
classifies the gesture using a **KNN model trained on my own hand**.

```
Webcam → MediaPipe (21 landmarks) → Normalize → KNN Classifier → Match %
```

The exact same pipeline runs natively in Python (`gesture_matcher.py`) and in the
browser (`web/src/matcher.js`) — the JS port is a faithful re-implementation of the
normalization + KNN/cosine logic, including x-mirror augmentation.

---

## The Game

Three modes, controlled with keyboard or mouse:

#### 🎓 Learn Mode `[1]`
- Reference illustration of the sign on the left, live webcam with hand landmarks
  on the right.
- **Hold the correct gesture for 2 seconds** → auto-advances to the next letter.
- Short cooldown after each advance to avoid false positives on transitions.

#### 🎮 Game Mode `[2]`
- All **24 letters in random order**, no reference image — test your memory.
- Each letter shown large with its Greek name (Άλφα, Βήτα…).
- 8-second timer per letter, 3 lives.
- Scoring: +10 pts per correct answer, +5 bonus if answered in under 3 seconds.

#### 🏆 Scoreboard `[3]`
- Top 5 scores saved locally (desktop: JSON file · web: `localStorage`).

`[F]` toggles fullscreen.

---

## How the Model Was Built

### Phase 1 — The Problem: No Dataset Exists
Greek Sign Language has no public hand-gesture dataset. The only option was to build
one from scratch — record my own hand performing all 24 letters.

### Phase 2 — Self-Recording the Data
`train_self.py` is a guided recording tool: it shows the reference sign, captures 60
landmark samples per letter after a countdown, and progresses through all 24 letters.

### Phase 3 — Augmentation
`augment_data.py` expands the dataset with x-mirror (camera flip vs reference
orientation), Gaussian noise, scale variation (±10%), and 3D rotation (±10°).

### Phase 4 — Training the KNN Classifier
Built into `gesture_matcher.py`:
- **Model:** K-Nearest Neighbors (scikit-learn) with cosine distance, `weights="distance"`.
- **Features:** 63-dimensional L2-normalized vector (21 landmarks × xyz).
- **Normalization:** subtract wrist, scale by wrist→middle-MCP distance → invariant to
  hand position and distance from camera.
- **Mirror augmentation at inference:** both orientations tested on every prediction.

For the web edition, the self-recorded samples are exported from
`data/references/*.npy` to `web/public/references.json` (the full set —
originals plus augmentation, so the in-browser KNN trains on exactly the same
data as the desktop classifier); the browser rebuilds the KNN dataset from it on
load, adding x-mirror augmentation just like the desktop pipeline.

---

## Technical Stack

| Component | Desktop | Web |
|---|---|---|
| Hand detection | MediaPipe HandLandmarker (Tasks API) | MediaPipe Tasks Vision (WASM) |
| Gesture classification | scikit-learn KNeighborsClassifier | KNN/cosine ported to JS |
| Camera & UI | OpenCV | Canvas 2D + getUserMedia |
| Greek text | Pillow (PIL) | native Canvas text |
| Build / host | Python | Vite → Vercel (static) |

---

## Deploy the Web Edition to Vercel

The web app lives in `web/`. In the Vercel project settings:

- **Framework Preset:** Vite
- **Root Directory:** `web`
- Build command `npm run build`, output `dist` (Vite defaults — auto-detected)

Static site, no env vars or serverless functions needed. See [`web/README.md`](web/README.md).

---

## Desktop Installation

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # mediapipe, opencv-python, numpy, pillow, scikit-learn

python3 main.py                   # MediaPipe model auto-downloads on first run
```

### Train on Your Own Hand (Recommended)
The included model was trained on **my hand**. For best accuracy, record your own:

```bash
python3 train_self.py             # hold each sign ~3s; press [T] to train when done
```

---

## Project Structure

```
.
├── main.py              # Desktop app (UI, game logic)
├── gesture_matcher.py   # Normalization, KNN classifier, cosine matching
├── hand_tracker.py      # MediaPipe HandLandmarker wrapper
├── train_self.py        # Guided self-recording training tool
├── augment_data.py      # Data augmentation
├── score_tracker.py     # High-score persistence
├── progress_tracker.py  # Greek alphabet definition
├── data/
│   ├── references/      # Per-letter .npy landmark arrays (self-recorded)
│   └── classifier.pkl   # Trained KNN model
├── assets/letters/      # Reference illustrations (Α.jpg … Ω.jpg)
└── web/                 # Browser edition (Vite + MediaPipe WASM + JS KNN)
    ├── src/             # main.js, matcher.js, hand.js, scores.js, state.js
    └── public/          # references.json (exported samples) + letters/
```

---

## License

MIT License — see `LICENSE` for details.
