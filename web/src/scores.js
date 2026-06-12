// High-score persistence in localStorage (browser equivalent of score_tracker.py)
const KEY = "stgr_high_scores";
const MAX = 5;

function load() {
  try { return JSON.parse(localStorage.getItem(KEY)) || []; }
  catch { return []; }
}

export function highScores() { return load(); }

export function bestScore() {
  const s = load();
  return s.length ? s[0].score : 0;
}

export function addScore(score, letters, time) {
  const list = load();
  const entry = {
    score, letters,
    time: Math.round(time * 10) / 10,
    date: new Date().toLocaleDateString("el-GR"),
  };
  list.push(entry);
  list.sort((a, b) => (b.score - a.score) || (a.time - b.time));
  const trimmed = list.slice(0, MAX);
  localStorage.setItem(KEY, JSON.stringify(trimmed));
  const rank = trimmed.indexOf(entry) + 1;
  return rank; // 0 if it didn't make the top list
}
