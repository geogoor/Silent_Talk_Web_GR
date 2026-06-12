# Sign Language GR — Ελληνική Νοηματική Γλώσσα

[English](README.md) · **Ελληνικά**

Ένα διαδραστικό παιχνίδι εκμάθησης του **αλφαβήτου της Ελληνικής Νοηματικής Γλώσσας
(ΕΝΓ)**, φτιαγμένο από το μηδέν — μαζί με ένα **custom μοντέλο αναγνώρισης
χειρονομιών** εκπαιδευμένο με το δικό μου χέρι ως training data. Διατίθεται σε δύο
εκδόσεις:

- 🌐 **Web έκδοση** (`web/`) — τρέχει εξ ολοκλήρου στον browser, χωρίς εγκατάσταση.
  Hand tracking μέσω MediaPipe WASM, με τον KNN classifier μεταφερμένο σε JavaScript.
  Ανεβαίνει στο Vercel ως static site.
- 🖥️ **Desktop έκδοση** (Python + OpenCV) — η αρχική εφαρμογή και όλη η αλυσίδα
  συλλογής δεδομένων / self-training.

> **Portfolio project** — Δεν υπάρχει δημόσιο dataset ΕΝΓ. Κάθε κομμάτι, από τη
> συλλογή δεδομένων μέχρι τον εκπαιδευμένο classifier, σχεδιάστηκε και χτίστηκε στο
> χέρι.

---

## Δοκίμασέ το online

Η web έκδοση χρειάζεται μόνο webcam και έναν σύγχρονο browser (Chrome/Edge/Firefox).
Η πρόσβαση στην κάμερα απαιτεί HTTPS — το Vercel το παρέχει αυτόματα, και τοπικά το
`localhost` επιτρέπεται.

```bash
cd web
npm install
npm run dev        # άνοιξε το localhost URL που τυπώνεται, δώσε άδεια κάμερας
```

Όλα τρέχουν **client-side** — κανένας server, κανένα δεδομένο δεν φεύγει από τη
συσκευή σου.

---

## Πώς λειτουργεί

Η εφαρμογή εντοπίζει το χέρι σου σε πραγματικό χρόνο μέσω webcam, εξάγει 21 3D
landmarks με **MediaPipe**, τα κανονικοποιεί σε ένα vector ανεξάρτητο από κλίμακα και
θέση, και ταξινομεί τη χειρονομία με ένα **KNN μοντέλο εκπαιδευμένο στο δικό μου
χέρι**.

```
Webcam → MediaPipe (21 landmarks) → Normalize → KNN Classifier → Match %
```

Το ίδιο ακριβώς pipeline τρέχει native σε Python (`gesture_matcher.py`) και στον
browser (`web/src/matcher.js`) — το JS port είναι πιστή αναπαραγωγή της λογικής
normalization + KNN/cosine, μαζί με το x-mirror augmentation.

---

## Το παιχνίδι

Τρία modes, με πληκτρολόγιο ή ποντίκι:

#### 🎓 Εκμάθηση `[1]`
- Εικονογράφηση αναφοράς του σημείου αριστερά, ζωντανή κάμερα με τα landmarks του
  χεριού δεξιά.
- **Κράτα τη σωστή χειρονομία για 2 δευτερόλεπτα** → προχωράει αυτόματα στο επόμενο
  γράμμα.
- Μικρό cooldown μετά από κάθε προχώρημα ώστε να αποφεύγονται false positives στις
  μεταβάσεις.

#### 🎮 Παιχνίδι `[2]`
- Και τα **24 γράμματα σε τυχαία σειρά**, χωρίς εικόνα αναφοράς — δοκίμασε τη μνήμη σου.
- Κάθε γράμμα εμφανίζεται μεγάλο με το ελληνικό του όνομα (Άλφα, Βήτα…).
- Χρονόμετρο 8 δευτερολέπτων ανά γράμμα, 3 ζωές.
- Βαθμολογία: +10 πόντοι ανά σωστή απάντηση, +5 bonus αν απαντηθεί κάτω από 3
  δευτερόλεπτα.

#### 🏆 Βαθμολογίες `[3]`
- Top 5 σκορ αποθηκευμένα τοπικά (desktop: αρχείο JSON · web: `localStorage`).

`[F]` εναλλαγή πλήρους οθόνης.

---

## Πώς χτίστηκε το μοντέλο

### Φάση 1 — Το πρόβλημα: δεν υπάρχει dataset
Η Ελληνική Νοηματική Γλώσσα δεν έχει δημόσιο dataset χειρονομιών. Η μόνη επιλογή ήταν
να φτιαχτεί από το μηδέν — να καταγράψω το δικό μου χέρι σε όλα τα 24 γράμματα.

### Φάση 2 — Καταγραφή δεδομένων
Το `train_self.py` είναι ένα καθοδηγούμενο εργαλείο καταγραφής: δείχνει το σημείο
αναφοράς, καταγράφει 60 δείγματα landmarks ανά γράμμα μετά από countdown, και προχωράει
σε όλα τα 24 γράμματα.

### Φάση 3 — Augmentation
Το `augment_data.py` εμπλουτίζει το dataset με x-mirror (camera flip vs φορά
αναφοράς), Gaussian noise, μεταβολή κλίμακας (±10%) και 3D περιστροφή (±10°).

### Φάση 4 — Εκπαίδευση του KNN classifier
Ενσωματωμένο στο `gesture_matcher.py`:
- **Μοντέλο:** K-Nearest Neighbors (scikit-learn) με cosine distance, `weights="distance"`.
- **Χαρακτηριστικά:** 63-διάστατο L2-normalized vector (21 landmarks × xyz).
- **Normalization:** αφαίρεση καρπού, κλιμάκωση με απόσταση καρπού→μέσης-MCP →
  ανεξαρτησία από θέση και απόσταση χεριού από την κάμερα.
- **Mirror augmentation στο inference:** και οι δύο προσανατολισμοί ελέγχονται σε κάθε
  πρόβλεψη.

Για τη web έκδοση, τα self-recorded samples εξάγονται από τα `data/references/*.npy`
στο `web/public/references.json` (ολόκληρο το σετ — originals συν το augmentation,
ώστε το KNN στον browser να εκπαιδεύεται στα ίδια ακριβώς δεδομένα με τον desktop
classifier)· ο browser ξαναχτίζει το KNN dataset από αυτό κατά τη φόρτωση,
προσθέτοντας x-mirror augmentation όπως και το desktop pipeline.

---

## Τεχνολογίες

| Στοιχείο | Desktop | Web |
|---|---|---|
| Εντοπισμός χεριού | MediaPipe HandLandmarker (Tasks API) | MediaPipe Tasks Vision (WASM) |
| Ταξινόμηση χειρονομίας | scikit-learn KNeighborsClassifier | KNN/cosine port σε JS |
| Κάμερα & UI | OpenCV | Canvas 2D + getUserMedia |
| Ελληνικά κείμενα | Pillow (PIL) | native Canvas text |
| Build / host | Python | Vite → Vercel (static) |

---

## Deploy της web έκδοσης στο Vercel

Η web εφαρμογή βρίσκεται στο `web/`. Στις ρυθμίσεις του Vercel project:

- **Framework Preset:** Vite
- **Root Directory:** `web`
- Build command `npm run build`, output `dist` (defaults του Vite — auto-detected)

Static site, χωρίς env vars ή serverless functions. Δες το [`web/README.md`](web/README.md).

---

## Εγκατάσταση Desktop

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # mediapipe, opencv-python, numpy, pillow, scikit-learn

python3 main.py                   # το μοντέλο MediaPipe κατεβαίνει αυτόματα στην πρώτη εκτέλεση
```

### Εκπαίδευσε στο δικό σου χέρι (συνιστάται)
Το μοντέλο που περιλαμβάνεται εκπαιδεύτηκε στο **δικό μου χέρι**. Για καλύτερη
ακρίβεια, κατάγραψε το δικό σου:

```bash
python3 train_self.py             # κράτα κάθε σημείο ~3δ· πάτα [T] για εκπαίδευση στο τέλος
```

---

## Δομή του project

```
.
├── main.py              # Desktop app (UI, λογική παιχνιδιού)
├── gesture_matcher.py   # Normalization, KNN classifier, cosine matching
├── hand_tracker.py      # Wrapper του MediaPipe HandLandmarker
├── train_self.py        # Εργαλείο καθοδηγούμενης self-recording
├── augment_data.py      # Data augmentation
├── score_tracker.py     # Αποθήκευση high scores
├── progress_tracker.py  # Ορισμός ελληνικού αλφαβήτου
├── data/
│   ├── references/      # .npy landmark arrays ανά γράμμα (self-recorded)
│   └── classifier.pkl   # Εκπαιδευμένο KNN μοντέλο
├── assets/letters/      # Εικονογραφήσεις αναφοράς (Α.jpg … Ω.jpg)
└── web/                 # Browser έκδοση (Vite + MediaPipe WASM + JS KNN)
    ├── src/             # main.js, matcher.js, hand.js, scores.js, state.js
    └── public/          # references.json (εξαγμένα δείγματα) + letters/
```

---

## Άδεια

MIT License — δες το `LICENSE` για λεπτομέρειες.
