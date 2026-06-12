// ── Greek alphabet + display names ─────────────────────────────────────────────
export const GREEK_LETTERS = [
  "Α","Β","Γ","Δ","Ε","Ζ","Η","Θ","Ι","Κ","Λ","Μ",
  "Ν","Ξ","Ο","Π","Ρ","Σ","Τ","Υ","Φ","Χ","Ψ","Ω",
];

export const LETTER_NAMES = {
  "Α":"Άλφα","Β":"Βήτα","Γ":"Γάμα","Δ":"Δέλτα",
  "Ε":"Έψιλον","Ζ":"Ζήτα","Η":"Ήτα","Θ":"Θήτα",
  "Ι":"Ιώτα","Κ":"Κάπα","Λ":"Λάμδα","Μ":"Μι",
  "Ν":"Νι","Ξ":"Ξι","Ο":"Όμικρον","Π":"Πι",
  "Ρ":"Ρο","Σ":"Σίγμα","Τ":"Ταυ","Υ":"Ύψιλον",
  "Φ":"Φι","Χ":"Χι","Ψ":"Ψι","Ω":"Ωμέγα",
};

// ── Tunables (mirrors the Python app) ──────────────────────────────────────────
export const WIN_W = 1280, WIN_H = 720, HALF = WIN_W / 2;
export const MATCH_THRESHOLD = 0.85;
export const HOLD_SECONDS    = 2.0;
export const GAME_COOLDOWN   = 2.0;
export const LEARN_COOLDOWN  = 2.5;
export const GAME_TIMEOUT    = 8.0;
export const MAX_LIVES       = 3;

// ── Palette (RGB) ──────────────────────────────────────────────────────────────
export const C = {
  bg:      "rgb(10,10,15)",
  surface: "rgb(30,25,25)",
  text:    "rgb(255,255,255)",
  dim:     "rgb(150,145,140)",
  accent:  "rgb(79,142,247)",   // electric blue
  amber:   "rgb(247,184,79)",   // amber gold
  success: "rgb(90,210,100)",
  error:   "rgb(210,80,80)",
  barbg:   "rgb(38,33,33)",
};

// 21-landmark hand topology (MediaPipe)
export const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[5,6],[6,7],[7,8],[9,10],[10,11],[11,12],
  [13,14],[14,15],[15,16],[17,18],[18,19],[19,20],
  [0,5],[5,9],[9,13],[13,17],[0,17],
];
