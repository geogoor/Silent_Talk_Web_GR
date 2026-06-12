# Sign Language GR — Ελληνική Νοηματική Γλώσσα

An interactive learning game for the **Greek Sign Language (ΕΝΓ) alphabet**, built entirely from scratch — including a **custom-trained gesture recognition model** using my own hand as training data.

> **Portfolio project** — No pre-existing ΕΝΓ dataset exists publicly. Every component, from data collection to the trained classifier, was designed and built by hand.

---

## Demo

| Learn Mode | Game Mode |
|---|---|
| Reference image of sign · Live webcam · Auto-advance on correct gesture | Letter prompt from memory · Timer · Lives · Score |

---

## How It Works

The app uses your webcam to detect your hand in real time, extracts 21 3D landmarks via **MediaPipe**, normalizes them into a scale- and position-invariant vector, and classifies the gesture using a **KNN model trained on my own hand**.

```
Webcam → MediaPipe (21 landmarks) → Normalize → KNN Classifier → Match %
```

---

## Project Phases

### Phase 1 — The Problem: No Dataset Exists

Greek Sign Language has no public hand gesture dataset. The only option was to build one from scratch.

**Approach:** Find a reference video of the ΕΝΓ alphabet, extract reference images from it, then **record my own gestures** as the actual training data.

---

### Phase 2 — Automated Data Extraction from Video

Built `auto_extract.py` — a script that:
- Takes a YouTube video of the ΕΝΓ alphabet
- Detects each letter's timestamp automatically
- Extracts a reference screenshot per letter (used in Learn mode)
- Collects 40 landmark samples per letter from the video

```bash
python3 auto_extract.py
```

Also built `video_trainer.py` — an interactive frame-by-frame tool for manual correction of timestamps.

---

### Phase 3 — Self-Training the Model

This is the core of the project. Since gesture recognition is highly personal (hand shape, size, and angle vary between people), **I recorded my own hand performing all 24 Greek letters**.

Built `train_self.py` — a guided recording tool that:
- Shows the reference image of each sign on the left
- Shows the live webcam on the right
- Captures **60 samples per letter** after a 3-second countdown
- Progresses automatically through all 24 letters

```bash
python3 train_self.py
```

**Total self-recorded samples:** 60 × 24 = **1,440 personal gesture samples**

Then ran `augment_data.py` to expand the dataset:
- X-axis mirror (handles camera flip vs reference video orientation)
- Gaussian noise
- Scale variation (±10%)
- 3D rotation (±10°)

**Final dataset:** 120 samples per letter = **2,880 augmented samples**

---

### Phase 4 — Training the KNN Classifier

Built the classifier directly into `gesture_matcher.py`:

```python
python3 -c "import gesture_matcher as gm; acc = gm.train_classifier(); print(f'{acc:.1%}')"
```

**Model:** K-Nearest Neighbors (scikit-learn) with cosine distance metric  
**Features:** 63-dimensional L2-normalized vector (21 landmarks × xyz)  
**Cross-validation accuracy:** ~100% on personal training data  
**Saved to:** `data/classifier.pkl`

Key technical decisions:
- **Normalization:** subtract wrist position, scale by wrist→middle-MCP distance → invariant to hand position and distance from camera
- **Mirror augmentation at inference time:** training video was unflipped; webcam feed is horizontally flipped → both orientations are tested at every prediction
- **KNN with `weights="distance"`:** closer neighbors count more, reduces noise from borderline samples

---

### Phase 5 — The Game

Three modes, controlled with keyboard or mouse:

#### 🎓 Learn Mode (`[1]`)
- Left panel: reference photo of the sign
- Right panel: live webcam with hand landmarks
- **Hold the correct gesture for 2 seconds** → auto-advances to next letter
- 2.5-second cooldown after each advance (prevents false positives on transition)
- Thin progress bar at bottom showing completion

#### 🎮 Game Mode (`[2]`)
- All **24 letters in random order**, no reference image — test your memory
- Each letter shown as large text with its Greek name (Άλφα, Βήτα...)
- 8-second timer per letter, 3 lives
- Scoring: +10 pts per correct answer, +5 bonus if answered in under 3 seconds
- 2-second cooldown after each answer

#### 🏆 Scoreboard (`[3]`)
- Top 5 scores saved locally with letter count and time

---

## Technical Stack

| Component | Technology |
|---|---|
| Hand detection | MediaPipe HandLandmarker (Tasks API) |
| Gesture classification | scikit-learn KNeighborsClassifier |
| Camera & UI | OpenCV |
| Greek text rendering | Pillow (PIL) |
| Data augmentation | NumPy |
| Video download | yt-dlp |

---

## Installation

```bash
git clone https://github.com/F0rgiv3n/SilentTalkGr
cd Sign_Language_GR

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install mediapipe opencv-python numpy pillow scikit-learn yt-dlp
```

Download the MediaPipe hand landmark model (auto-downloaded on first run):

```bash
python3 main.py
```

---

## Train on Your Own Hand (Recommended)

The included model was trained on **my hand**. For best accuracy, record your own:

```bash
source venv/bin/activate
python3 train_self.py
```

Follow the on-screen instructions — hold each sign for ~3 seconds, the app captures 60 samples and moves to the next letter automatically. After all 24 letters, press `[T]` to train and save the classifier.

---

## Project Structure

```
Sign_Language_GR/
├── main.py              # Main app (UI, game logic)
├── gesture_matcher.py   # Normalization, KNN classifier, cosine matching
├── hand_tracker.py      # MediaPipe HandLandmarker wrapper
├── train_self.py        # Guided self-recording training tool
├── auto_extract.py      # Automated extraction from training video
├── video_trainer.py     # Manual frame-by-frame labeling tool
├── augment_data.py      # Data augmentation (mirror, noise, scale, rotation)
├── score_tracker.py     # High score persistence
├── progress_tracker.py  # Greek alphabet definition
├── data/
│   ├── references/      # Per-letter .npy landmark arrays
│   └── classifier.pkl   # Trained KNN model
└── assets/
    └── letters/         # Reference screenshots (Α.jpg … Ω.jpg)
```

---

## Key Challenges & Solutions

**Challenge:** No ΕΝΓ dataset exists publicly  
**Solution:** Built a self-recording pipeline → trained on personal gesture data

**Challenge:** Training video was unflipped; webcam feed is horizontally mirrored  
**Solution:** Mirror augmentation at inference time — both orientations tested on every prediction

**Challenge:** Gesture recognition too sensitive (false positives on transitions)  
**Solution:** 2-second hold requirement + 2-second cooldown after each recognition event

**Challenge:** OpenCV cannot render Greek Unicode text  
**Solution:** Pillow (PIL) used for all text rendering, converted back to BGR for OpenCV

---

## License

MIT License — see `LICENSE` for details.
