// Port of gesture_matcher.py — normalization + KNN/cosine matching, all in-browser.

let DATA = null;        // { letter: Float32Array[] }  raw normalized 63-d samples
let KNN  = null;        // { vecs: Float32Array (L2-normed, flattened), labels: string[] }
const DIM = 63;

// ── Normalization (position- & scale-invariant 63-d vector) ────────────────────
export function normalize(landmarks) {
  const pts = new Float32Array(DIM);
  for (let i = 0; i < 21; i++) {
    pts[i*3]   = landmarks[i].x;
    pts[i*3+1] = landmarks[i].y;
    pts[i*3+2] = landmarks[i].z;
  }
  const wx = pts[0], wy = pts[1], wz = pts[2];      // wrist
  for (let i = 0; i < 21; i++) {
    pts[i*3]   -= wx; pts[i*3+1] -= wy; pts[i*3+2] -= wz;
  }
  // scale = ||middle-MCP (idx 9)||
  const mx = pts[27], my = pts[28], mz = pts[29];
  const scale = Math.hypot(mx, my, mz);
  if (scale > 1e-6) for (let i = 0; i < DIM; i++) pts[i] /= scale;
  return pts;
}

function mirrorX(v) {
  const o = new Float32Array(DIM);
  for (let i = 0; i < DIM; i++) o[i] = v[i];
  for (let i = 0; i < DIM; i += 3) o[i] = -o[i];
  return o;
}

function l2norm(v) {
  let n = 0; for (let i = 0; i < DIM; i++) n += v[i]*v[i];
  n = Math.sqrt(n);
  const o = new Float32Array(DIM);
  if (n > 1e-8) for (let i = 0; i < DIM; i++) o[i] = v[i]/n;
  return o;
}

function dot(a, b, bOff = 0) {
  let s = 0; for (let i = 0; i < DIM; i++) s += a[i]*b[bOff+i];
  return s;
}

// ── Load exported reference samples and build the KNN dataset ───────────────────
export async function loadData(url) {
  const raw = await (await fetch(url)).json();
  DATA = {};
  const vecChunks = [];
  const labels = [];
  for (const letter of Object.keys(raw)) {
    const arr = raw[letter].map((s) => Float32Array.from(s));
    DATA[letter] = arr;
    for (const v of arr) {
      vecChunks.push(l2norm(v));          labels.push(letter);
      vecChunks.push(l2norm(mirrorX(v))); labels.push(letter);   // mirror augmentation
    }
  }
  const flat = new Float32Array(vecChunks.length * DIM);
  vecChunks.forEach((v, i) => flat.set(v, i*DIM));
  KNN = { vecs: flat, labels };
}

export const recordedLetters = () => (DATA ? Object.keys(DATA) : []);

// ── Weighted KNN over cosine similarity ────────────────────────────────────────
function queryKNN(qn, k = 7) {
  const n = KNN.labels.length;
  // find k highest similarities
  const bestSim = new Array(k).fill(-2);
  const bestIdx = new Array(k).fill(-1);
  for (let i = 0; i < n; i++) {
    const s = dot(qn, KNN.vecs, i*DIM);
    if (s > bestSim[k-1]) {
      let j = k-1;
      while (j > 0 && bestSim[j-1] < s) { bestSim[j]=bestSim[j-1]; bestIdx[j]=bestIdx[j-1]; j--; }
      bestSim[j] = s; bestIdx[j] = i;
    }
  }
  const weights = {};
  let total = 0;
  for (let j = 0; j < k; j++) {
    if (bestIdx[j] < 0) continue;
    const dist = 1 - bestSim[j];
    const w = 1 / (dist + 1e-6);
    const lab = KNN.labels[bestIdx[j]];
    weights[lab] = (weights[lab] || 0) + w;
    total += w;
  }
  let label = null, best = 0;
  for (const lab in weights) if (weights[lab] > best) { best = weights[lab]; label = lab; }
  return { label, conf: total > 0 ? best/total : 0 };
}

function predictKNN(vec) {
  const a = queryKNN(l2norm(vec));
  const b = queryKNN(l2norm(mirrorX(vec)));
  return a.conf >= b.conf ? a : b;
}

// Top-5 mean cosine to a letter's raw samples (both orientations) — fallback.
function cosineScore(vec, letter) {
  const refs = DATA[letter]; if (!refs) return 0;
  const v = l2norm(vec), vm = l2norm(mirrorX(vec));
  const scores = [];
  for (const r of refs) {
    const rn = l2norm(r);
    scores.push(dot(v, rn)); scores.push(dot(vm, rn));
  }
  scores.sort((x, y) => y - x);
  const top = scores.slice(0, Math.min(5, scores.length));
  return top.reduce((a, b) => a + b, 0) / (top.length || 1);
}

// ── Public API (mirrors gesture_matcher.py) ────────────────────────────────────
export function matchScore(vec, letter) {
  if (!DATA || !DATA[letter]) return 0;
  const { label, conf } = predictKNN(vec);
  if (label === letter) return conf;
  return Math.max(conf * 0.5, cosineScore(vec, letter) * 0.8);
}

export function bestMatch(vec, letters) {
  if (!KNN) return { letter: null, score: 0 };
  const { label, conf } = predictKNN(vec);
  if (letters.includes(label)) return { letter: label, score: conf };
  let bl = null, bs = 0;
  for (const l of letters) { const s = cosineScore(vec, l); if (s > bs) { bs = s; bl = l; } }
  return { letter: bl, score: bs };
}
