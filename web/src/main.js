import {
  GREEK_LETTERS, LETTER_NAMES, WIN_W, WIN_H, HALF, C, CONNECTIONS,
  MATCH_THRESHOLD, HOLD_SECONDS, GAME_COOLDOWN, LEARN_COOLDOWN, GAME_TIMEOUT, MAX_LIVES,
} from "./state.js";
import { createHandTracker } from "./hand.js";
import * as gm from "./matcher.js";
import * as scores from "./scores.js";

const BASE = import.meta.env.BASE_URL || "/";
const now  = () => performance.now() / 1000;   // seconds, monotonic

// ── Canvas + drawing helpers ───────────────────────────────────────────────────
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const FONT = '700 SIZEpx system-ui, "Segoe UI", Arial, sans-serif';
const fontStr = (s) => FONT.replace("SIZE", s);

function text(msg, x, y, { size = 28, color = C.text, align = "left", base = "alphabetic" } = {}) {
  ctx.font = fontStr(size); ctx.fillStyle = color;
  ctx.textAlign = align; ctx.textBaseline = base;
  ctx.fillText(msg, x, y);
}
function rect(x, y, w, h, color) { ctx.fillStyle = color; ctx.fillRect(x, y, w, h); }
function roundRectPath(x, y, w, h, r) {
  ctx.beginPath();
  if (ctx.roundRect) { ctx.roundRect(x, y, w, h, r); return; }
  ctx.moveTo(x+r, y);
  ctx.arcTo(x+w, y, x+w, y+h, r); ctx.arcTo(x+w, y+h, x, y+h, r);
  ctx.arcTo(x, y+h, x, y, r);     ctx.arcTo(x, y, x+w, y, r);
  ctx.closePath();
}
function bar(x, y, w, h, ratio, fg, bg = C.barbg) {
  ratio = Math.max(0, Math.min(1, ratio));
  rect(x, y, w, h, bg);
  if (ratio > 0) rect(x, y, w * ratio, h, fg);
}
function panel(x, y, w, h, color = "rgb(15,15,28)", alpha = 0.82) {
  ctx.save(); ctx.globalAlpha = alpha; rect(x, y, w, h, color); ctx.restore();
}
function flash(color, alpha = 0.3) {
  ctx.save(); ctx.globalAlpha = alpha; rect(0, 0, WIN_W, WIN_H, color); ctx.restore();
}
function heart(cx, cy, size, filled) {
  ctx.fillStyle = filled ? "rgb(220,60,60)" : "rgb(75,55,55)";
  ctx.beginPath();
  ctx.arc(cx - size/4, cy, size/3, 0, Math.PI*2);
  ctx.arc(cx + size/4, cy, size/3, 0, Math.PI*2);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(cx - size/2, cy); ctx.lineTo(cx, cy + size*0.75); ctx.lineTo(cx + size/2, cy);
  ctx.closePath(); ctx.fill();
}

// ── Reference-image cache with cover (fill + center-crop) draw ───────────────────
const imgCache = {};
function refImage(letter) {
  if (!(letter in imgCache)) {
    const im = new Image();
    im.src = `${BASE}letters/${encodeURIComponent(letter)}.jpg`;
    imgCache[letter] = im;
  }
  return imgCache[letter];
}
function drawCover(im, x, y, w, h) {
  rect(x, y, w, h, C.bg);
  if (!im.complete || !im.naturalWidth) return;
  // Cover: fill (w,h) preserving aspect, center-crop the overflow
  const iw = im.naturalWidth, ih = im.naturalHeight;
  const sw = Math.min(iw, ih * (w / h));
  const sh = Math.min(ih, iw * (h / w));
  ctx.drawImage(im, (iw - sw) / 2, (ih - sh) / 2, sw, sh, x, y, w, h);
}

// ── App ─────────────────────────────────────────────────────────────────────────
const MENU="menu", LEARN="learn", GAME="game", WIN="win", GAMEOVER="gameover", SCORES="scores";

class App {
  constructor(tracker, video, detCanvas) {
    this.tracker = tracker; this.video = video; this.det = detCanvas;
    this.dctx = detCanvas.getContext("2d");
    this.mode = MENU; this.lastVec = null; this.hoverIdx = -1; this.homeHover = false;
    this._links = []; this._hoverLink = -1;

    this.learnIdx = 0; this.learnHold = 0; this.learnCool = 0;
    this.gameLetters = []; this.gameIdx = 0; this.lives = MAX_LIVES; this.score = 0;
    this.gameStart = 0; this.letterStart = 0; this.hold = 0; this.gameCool = 0;
    this.fbMsg = ""; this.fbColor = C.success; this.fbT = 0;
    this.finalScore = 0; this.finalLetters = 0; this.finalTime = 0; this.rank = 0;
    this.tipShowing = false; this.tipSeen = false;   // one-time camera tips on first practice
    this._okRect = null; this._okHover = false;

    window.addEventListener("keydown", (e) => this.onKey(e));
    canvas.addEventListener("mousemove", (e) => this.onMouse(e, false));
    canvas.addEventListener("click", (e) => this.onMouse(e, true));
  }

  // ── Input ─────────────────────────────────────────────────────────────────────
  canvasXY(e) {
    const r = canvas.getBoundingClientRect();
    return [ (e.clientX - r.left) * WIN_W / r.width, (e.clientY - r.top) * WIN_H / r.height ];
  }
  menuHit(cx, cy) {
    const scx = WIN_W / 2;
    for (let i = 0; i < 4; i++) {
      const iy = 305 + i * 70;
      if (cx > scx - 190 && cx < scx + 210 && cy > iy - 30 && cy < iy + 30) return i;
    }
    return -1;
  }
  homeHit(cx, cy) { return cx >= 8 && cx <= 60 && cy >= 8 && cy <= 56; }
  onMouse(e, click) {
    const [cx, cy] = this.canvasXY(e);
    if (this.tipShowing) {                                  // modal: only the OK button is live
      const r = this._okRect;
      this._okHover = !!r && cx >= r.x && cx <= r.x + r.w && cy >= r.y && cy <= r.y + r.h;
      canvas.style.cursor = this._okHover ? "pointer" : "default";
      if (click && this._okHover) this.tipShowing = false;
      return;
    }
    if (this.mode !== MENU) {
      const overHome = this.homeHit(cx, cy);
      this.homeHover = overHome;
      canvas.style.cursor = overHome ? "pointer" : "default";
      if (click && overHome) this.mode = MENU;
      return;
    }
    const idx = this.menuHit(cx, cy);
    const linkIdx = this._links.findIndex(l =>
      cx >= l.x && cx <= l.x + l.w && cy >= l.y && cy <= l.y + l.h);
    if (click) {
      if (idx >= 0) this.handleKey(["1","2","3","f"][idx]);
      else if (linkIdx >= 0) window.open(this._links[linkIdx].url, "_blank", "noopener");
    } else {
      this.hoverIdx = idx; this._hoverLink = linkIdx;
      canvas.style.cursor = (idx >= 0 || linkIdx >= 0) ? "pointer" : "default";
    }
  }
  drawHome() {
    const hov = this.homeHover;
    ctx.save(); ctx.globalAlpha = hov ? 0.95 : 0.78;
    ctx.fillStyle = C.surface; roundRectPath(12, 12, 44, 40, 10); ctx.fill();
    ctx.restore();
    ctx.strokeStyle = hov ? C.accent : "rgb(72,68,64)"; ctx.lineWidth = 1.5;
    roundRectPath(12, 12, 44, 40, 10); ctx.stroke();
    text("←", 34, 33, { size: 24, color: hov ? C.accent : C.text, align: "center", base: "middle" });
  }
  onKey(e) {
    const k = e.key;
    if (k === "Escape") return this.handleKey("Escape");
    if (k === "ArrowRight") return this.handleKey("ArrowRight");
    if (k === "ArrowLeft")  return this.handleKey("ArrowLeft");
    this.handleKey(k.toLowerCase());
  }
  maybeTip() {            // show the camera tips once, on first practice
    if (!this.tipSeen) { this.tipSeen = true; this.tipShowing = true; }
  }
  handleKey(k) {
    if (this.tipShowing) { this.tipShowing = false; return; }   // any key dismisses the tip
    if (k === "f") return toggleFullscreen();
    if (this.mode === MENU) {
      if (k === "1") { this.learnIdx = 0; this.learnHold = 0; this.mode = LEARN; this.maybeTip(); }
      else if (k === "2") this.startGame();
      else if (k === "3") this.mode = SCORES;
    } else if (this.mode === LEARN) {
      if (k === "ArrowRight" || k === "n") { this.learnIdx = (this.learnIdx+1)%24; this.learnHold = 0; }
      else if (k === "ArrowLeft" || k === "b") { this.learnIdx = (this.learnIdx+23)%24; this.learnHold = 0; }
      else if (k === "Escape") this.mode = MENU;
    } else if (this.mode === WIN || this.mode === GAMEOVER) {
      if (k === "2") this.startGame();
      else if (k === "3") this.mode = SCORES;
      else if (k === "Escape") this.mode = MENU;
    } else if (this.mode === SCORES) {
      if (k === "Escape" || k === " ") this.mode = MENU;
    } else if (this.mode === GAME) {
      if (k === "Escape") this.mode = MENU;
    }
  }

  // ── Game flow ──────────────────────────────────────────────────────────────────
  startGame() {
    this.gameLetters = [...GREEK_LETTERS].sort(() => Math.random() - 0.5);
    this.gameIdx = 0; this.lives = MAX_LIVES; this.score = 0;
    this.gameStart = this.letterStart = now(); this.hold = 0; this.mode = GAME;
    this.maybeTip();
  }
  nextLetter() {
    this.gameIdx++; this.hold = 0;
    const t = now(); this.gameCool = t; this.letterStart = t + GAME_COOLDOWN;
    if (this.gameIdx >= this.gameLetters.length) this.endGame(true);
  }
  wrong() {
    this.lives--; this.hold = 0;
    const t = now(); this.gameCool = t; this.letterStart = t + GAME_COOLDOWN;
    if (this.lives <= 0) this.endGame(false);
  }
  endGame(won) {
    const elapsed = now() - this.gameStart;
    this.finalScore = this.score; this.finalLetters = this.gameIdx; this.finalTime = elapsed;
    this.rank = scores.addScore(this.score, this.gameIdx, elapsed);
    this.mode = won ? WIN : GAMEOVER;
  }
  setFb(msg, color) { this.fbMsg = msg; this.fbColor = color; this.fbT = now(); }

  // ── Per-frame ───────────────────────────────────────────────────────────────────
  frame() {
    // Mirror the webcam into the detection canvas, then detect (matches training).
    const dc = this.det, d = this.dctx;
    d.save(); d.translate(dc.width, 0); d.scale(-1, 1);
    d.drawImage(this.video, 0, 0, dc.width, dc.height); d.restore();

    let lms = null;
    try {
      lms = this.tracker.detect(dc, Math.round(performance.now()));
    } catch (err) {
      // Throttle: a failing detector would otherwise flood the console every frame.
      if (now() - (this._lastDetectWarn || 0) > 3) {
        this._lastDetectWarn = now();
        console.warn("hand detection failed:", err);
      }
    }
    const vec = lms ? gm.normalize(lms) : null;
    this.lastVec = vec;

    ctx.clearRect(0, 0, WIN_W, WIN_H);

    // Right half: mirrored webcam (already in `dc`) + landmarks
    ctx.drawImage(dc, HALF, 0, HALF, WIN_H);
    if (lms) {
      ctx.strokeStyle = C.accent; ctx.lineWidth = 1.5;
      for (const [a, b] of CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(lms[a].x*HALF + HALF, lms[a].y*WIN_H);
        ctx.lineTo(lms[b].x*HALF + HALF, lms[b].y*WIN_H);
        ctx.stroke();
      }
      for (const lm of lms) {
        ctx.beginPath(); ctx.arc(lm.x*HALF + HALF, lm.y*WIN_H, 4, 0, Math.PI*2);
        ctx.fillStyle = C.text; ctx.fill();
        ctx.strokeStyle = C.accent; ctx.lineWidth = 1; ctx.stroke();
      }
    }

    // Left half base
    rect(0, 0, HALF, WIN_H, C.bg);

    // While the camera-tips overlay is up, keep the game clock frozen.
    const tipUp = this.tipShowing;
    if (tipUp && this.mode === GAME) this.gameStart = this.letterStart = now();

    if (this.mode === MENU) this.drawMenu(vec);
    else if (this.mode === LEARN) this.drawLearn(vec);
    else if (this.mode === GAME) this.drawGame(vec);
    else if (this.mode === WIN || this.mode === GAMEOVER) this.drawEnd();
    else if (this.mode === SCORES) this.drawScores();

    if (this.mode !== MENU) this.drawHome();
    text("F", WIN_W - 20, WIN_H - 12, { size: 13, color: "rgb(60,58,55)", align: "right", base: "middle" });

    if (tipUp) this.drawTip();
  }

  // ── First-use camera tips (modal; dismissed with the OK button or any key) ───────
  drawTip() {
    panel(0, 0, WIN_W, WIN_H, C.bg, 0.9);
    const cx = WIN_W/2;
    text("📷  Πριν ξεκινήσεις", cx, 178, { size: 34, color: C.amber, align: "center", base: "middle" });
    const tips = [
      "Το χέρι σου να φαίνεται καθαρά, ολόκληρο μέσα στο κάδρο",
      "Απόφυγε έντονο φως ή κόντρα φωτισμό πίσω σου — μπερδεύει την αναγνώριση",
      "Προτίμησε ουδέτερο φόντο και δείξε ένα χέρι τη φορά",
      "Κράτα το χέρι ~30–50 cm από την κάμερα",
    ];
    tips.forEach((t, i) => {
      const y = 256 + i*52;
      text("•", cx-330, y, { size: 22, color: C.accent, align: "center", base: "middle" });
      text(t, cx-308, y, { size: 21, color: C.text, align: "left", base: "middle" });
    });
    // OK button
    const bw = 160, bh = 52, bx = cx - bw/2, by = WIN_H - 132;
    this._okRect = { x: bx, y: by, w: bw, h: bh };
    const hov = this._okHover;
    ctx.save(); ctx.globalAlpha = hov ? 1 : 0.92;
    ctx.fillStyle = C.accent; roundRectPath(bx, by, bw, bh, 12); ctx.fill();
    ctx.restore();
    text("OK", cx, by + bh/2, { size: 24, color: "rgb(8,10,16)", align: "center", base: "middle" });
  }

  // ── MENU ─────────────────────────────────────────────────────────────────────
  drawMenu(vec) {
    const im = refImage(GREEK_LETTERS[0]);
    if (im.complete && im.naturalWidth) { ctx.save(); ctx.globalAlpha = 0.18; ctx.drawImage(im, 0, 0, WIN_W, WIN_H); ctx.restore(); }
    panel(0, 0, WIN_W, WIN_H, C.bg, 0.82);
    const cx = WIN_W / 2;
    text("Sign Language GR", cx, 175, { size: 46, align: "center", base: "middle" });
    text("Ελληνική Νοηματική Γλώσσα", cx, 222, { size: 19, color: C.dim, align: "center", base: "middle" });
    ctx.strokeStyle = C.accent; ctx.beginPath(); ctx.moveTo(cx-70, 250); ctx.lineTo(cx+70, 250); ctx.stroke();

    const items = [["1","Εκμάθηση"],["2","Παιχνίδι"],["3","Βαθμολογίες"],["F","Πλήρης οθόνη"]];
    items.forEach(([key, label], i) => {
      const y = 305 + i * 70;
      if (this.hoverIdx === i) {
        panel(cx-192, y-28, 404, 56, C.surface, 0.55);
        ctx.strokeStyle = C.accent; ctx.strokeRect(cx-192, y-28, 404, 56);
      }
      text(key, cx-130, y, { size: 22, color: C.accent, align: "center", base: "middle" });
      text(label, cx-100, y, { size: 26, color: C.text, align: "left", base: "middle" });
    });

    const rec = gm.recordedLetters().length, best = scores.bestScore();
    text(`${rec} / 24 γράμματα  ·  Ρεκόρ: ${best} pts`, cx, WIN_H-86, { size: 15, color: C.dim, align: "center", base: "middle" });

    // Credits — both lines are clickable (open a mailto); ✉ signals the affordance
    this._links = [];
    const sz = 13;
    const creditLines = [
      { text: "Δημιουργία: Γουρζιώτης Γιώργος  ·  ✉ georgegourziotis@gmail.com",
        url: "mailto:georgegourziotis@gmail.com" },
      { text: "Φωτογραφίες: Γιάννης Παπαβασιλείου  ·  ✉ gpapava@gmail.com",
        url: "mailto:gpapava@gmail.com" },
    ];
    creditLines.forEach((c, i) => {
      ctx.font = fontStr(sz);
      const cw = ctx.measureText(c.text).width;
      const cyC = WIN_H - 52 + i * 22;
      const hover = this._hoverLink === i;
      this._links.push({ x: cx - cw/2, y: cyC - 11, w: cw, h: 22, url: c.url });
      text(c.text, cx, cyC, { size: sz, color: hover ? C.accent : C.dim, align: "center", base: "middle" });
    });

    // Live recognition — a tidy labelled card, bottom-right (only when a hand reads)
    if (vec) {
      const { letter, score } = gm.bestMatch(vec, gm.recordedLetters());
      if (letter && score > 0.65) {
        const bw = 196, bh = 64, bx = WIN_W - bw - 18, by = WIN_H - bh - 26;
        panel(bx, by, bw, bh, C.surface, 0.9);
        ctx.strokeStyle = "rgb(60,56,52)"; ctx.lineWidth = 1; ctx.strokeRect(bx, by, bw, bh);
        text("ΣΕ ΑΝΑΓΝΩΡΙΖΕΙ", bx+16, by+17, { size: 11, color: C.dim, base: "middle" });
        text(letter, bx+16, by+43, { size: 28, color: C.amber, base: "middle" });
        text(`${Math.round(score*100)}%`, bx+58, by+44, { size: 20, color: C.dim, base: "middle" });
      }
    }
  }

  // ── LEARN ────────────────────────────────────────────────────────────────────
  drawLearn(vec) {
    let letter = GREEK_LETTERS[this.learnIdx];
    const t = now();
    const inCool = (t - this.learnCool) < LEARN_COOLDOWN;
    let score = (!inCool && vec) ? gm.matchScore(vec, letter) : 0;

    if (!inCool) {
      if (score >= MATCH_THRESHOLD) {
        if (this.learnHold === 0) this.learnHold = t;
        else if (t - this.learnHold >= HOLD_SECONDS) {
          this.learnHold = 0; this.learnCool = t;
          this.learnIdx = (this.learnIdx + 1) % 24;
          letter = GREEK_LETTERS[this.learnIdx]; score = 0;
          flash(C.success, 0.18);
        }
      } else this.learnHold = 0;
    }

    const PROGRESS_H = 4, imgH = WIN_H - PROGRESS_H;
    drawCover(refImage(letter), 0, 0, HALF, imgH);

    // bottom gradient
    const g = ctx.createLinearGradient(0, imgH-180, 0, imgH);
    g.addColorStop(0, "rgba(10,10,15,0)"); g.addColorStop(1, "rgba(10,10,15,1)");
    ctx.fillStyle = g; ctx.fillRect(0, imgH-180, HALF, 180);

    text(letter, 22, imgH-18, { size: 80, color: C.amber, base: "alphabetic" });
    ctx.font = fontStr(80);
    const glyphW = ctx.measureText(letter).width;
    text(LETTER_NAMES[letter] || "", 30 + glyphW, imgH-30, { size: 27, color: C.text, base: "alphabetic" });
    text(`${this.learnIdx+1} / 24`, HALF-16, imgH-18, { size: 18, color: C.dim, align: "right" });
    text("←   →", HALF/2, imgH-26, { size: 16, color: "rgb(120,116,112)", align: "center", base: "alphabetic" });

    rect(0, imgH, HALF, PROGRESS_H, C.bg);
    const ratio = this.learnIdx / 23, filled = Math.max(2, HALF*ratio);
    rect(0, imgH, filled, PROGRESS_H, C.accent);

    // right feedback overlay
    const FB = 90, fy = WIN_H - FB;
    panel(HALF, fy, HALF, FB, "rgb(8,6,6)", 0.82);
    const PX = HALF + 28, PW = HALF - 56;
    if (inCool) {
      const remain = LEARN_COOLDOWN - (t - this.learnCool);
      text(`Ετοιμάσου…  ${remain.toFixed(1)}s`, HALF + HALF/2, fy+34, { size: 20, color: C.dim, align: "center", base: "middle" });
      bar(PX, fy+54, PW, 3, remain/LEARN_COOLDOWN, C.accent);
    } else if (!vec) {
      text("Δείξε το χέρι σου", HALF + HALF/2, fy+44, { size: 19, color: "rgb(70,65,62)", align: "center", base: "middle" });
    } else {
      const col = score >= MATCH_THRESHOLD ? C.success : C.accent;
      text(`${Math.round(score*100)}%`, PX, fy+28, { size: 24, color: col, base: "middle" });
      bar(PX, fy+46, PW, 3, score, col);
      if (score >= MATCH_THRESHOLD && this.learnHold) {
        text("Κράτα…", PX, fy+62, { size: 17, color: C.amber, base: "middle" });
        bar(PX, fy+76, PW, 3, (t-this.learnHold)/HOLD_SECONDS, C.amber);
      } else if (score >= MATCH_THRESHOLD) {
        text("✓  Κράτα για να προχωρήσεις", HALF+HALF/2, fy+68, { size: 17, color: C.success, align: "center", base: "middle" });
      } else {
        text("Μιμήσου το σημείο αριστερά", HALF+HALF/2, fy+68, { size: 16, color: "rgb(110,105,100)", align: "center", base: "middle" });
      }
    }
  }

  // ── GAME ─────────────────────────────────────────────────────────────────────
  drawGame(vec) {
    if (!this.gameLetters.length) { this.mode = MENU; return; }
    const letter = this.gameLetters[this.gameIdx % this.gameLetters.length];
    const t = now();
    const inCool = (t - this.gameCool) < GAME_COOLDOWN;
    const score = (inCool || !vec) ? 0 : gm.matchScore(vec, letter);
    const elapsed = Math.max(0, t - this.letterStart);
    const timeLeft = Math.max(0, GAME_TIMEOUT - elapsed);

    if (!inCool) {
      if (score >= MATCH_THRESHOLD) {
        if (this.hold === 0) this.hold = t;
        else if (t - this.hold >= HOLD_SECONDS) {
          const pts = 10 + (elapsed < 3 ? 5 : 0);
          this.score += pts; this.setFb(`✓  ΣΩΣΤΟ!  +${pts} pts`, C.success);
          this.hold = 0; this.nextLetter(); return;
        }
      } else this.hold = 0;
      if (timeLeft <= 0) { this.setFb("✗  ΛΑΘΟΣ!", C.error); this.wrong(); return; }
    }

    if (now() - this.fbT < 0.4) flash(this.fbColor, 0.18);

    text(String(this.score), 64, 42, { size: 32, base: "middle" });
    text("pts", 64 + String(this.score).length*19 + 6, 44, { size: 15, color: C.dim, base: "middle" });
    for (let i = 0; i < MAX_LIVES; i++) heart(HALF-18-i*30, 38, 13, i < this.lives);
    ctx.strokeStyle = C.surface; ctx.beginPath(); ctx.moveTo(28,62); ctx.lineTo(HALF-28,62); ctx.stroke();

    const cx = HALF/2, cy = (62 + WIN_H - 88) / 2;
    text(`${this.gameIdx+1}  /  ${this.gameLetters.length}`, cx, cy-105, { size: 17, color: C.dim, align: "center", base: "middle" });
    text(letter, cx, cy, { size: 150, color: C.amber, align: "center", base: "middle" });
    text(LETTER_NAMES[letter] || "", cx, cy+95, { size: 22, color: C.dim, align: "center", base: "middle" });

    const by = WIN_H - 88;
    ctx.strokeStyle = C.surface; ctx.beginPath(); ctx.moveTo(28,by); ctx.lineTo(HALF-28,by); ctx.stroke();
    if (inCool) {
      const cdr = (GAME_COOLDOWN - (t - this.gameCool)) / GAME_COOLDOWN;
      text("Ετοιμάσου…", 28, by+30, { size: 18, color: C.dim, base: "middle" });
      bar(28, by+48, HALF-56, 3, cdr, C.accent);
    } else {
      const tcol = timeLeft > 3 ? C.accent : C.error;
      text(`${timeLeft.toFixed(1)}s`, 28, by+26, { size: 20, color: tcol, base: "middle" });
      bar(28, by+38, HALF-56, 3, timeLeft/GAME_TIMEOUT, tcol);
      const hr = this.hold ? (t-this.hold)/HOLD_SECONDS : 0;
      text("Κράτα", 28, by+60, { size: 17, color: C.dim, base: "middle" });
      bar(28, by+72, HALF-56, 3, hr, C.amber);
    }

    const FB = 80, fy = WIN_H - FB;
    panel(HALF, fy, HALF, FB, "rgb(8,6,6)", 0.82);
    const PX = HALF + 28, PW = HALF - 56;
    const col = score >= MATCH_THRESHOLD ? C.success : C.accent;
    if (now() - this.fbT < 1.8 && this.fbMsg) {
      const fc = this.fbMsg.includes("ΣΩΣΤΟ") ? C.success : C.error;
      text(this.fbMsg, HALF+HALF/2, fy+40, { size: 26, color: fc, align: "center", base: "middle" });
    } else {
      text(`${Math.round(score*100)}%`, PX, fy+26, { size: 22, color: col, base: "middle" });
      bar(PX, fy+42, PW, 3, score, col);
      if (this.hold && score >= MATCH_THRESHOLD) {
        text("Κράτα…", PX, fy+58, { size: 16, color: C.amber, base: "middle" });
        bar(PX, fy+68, PW, 3, (t-this.hold)/HOLD_SECONDS, C.amber);
      } else {
        const hint = score >= MATCH_THRESHOLD ? "✓" : "Κάνε το gesture από μνήμη";
        text(hint, HALF+HALF/2, fy+60, { size: 16, color: score >= MATCH_THRESHOLD ? C.success : "rgb(65,62,60)", align: "center", base: "middle" });
      }
    }
  }

  // ── WIN / GAMEOVER ─────────────────────────────────────────────────────────────
  drawEnd() {
    const won = this.mode === WIN, cx = WIN_W/2;
    panel(0, 0, WIN_W, WIN_H, C.bg, 0.88);
    if (won) text("Νίκη", cx, WIN_H/2-140, { size: 72, color: C.amber, align: "center", base: "middle" });
    else text("Game Over", cx, WIN_H/2-140, { size: 60, color: C.error, align: "center", base: "middle" });
    text(`${this.finalScore} pts`, cx, WIN_H/2-52, { size: 38, align: "center", base: "middle" });
    text(`${this.finalLetters} γράμματα  ·  ${this.finalTime.toFixed(1)}s`, cx, WIN_H/2+4, { size: 20, color: C.dim, align: "center", base: "middle" });
    if (this.rank === 1) text("Νέο ρεκόρ", cx, WIN_H/2+52, { size: 22, color: C.accent, align: "center", base: "middle" });
    else if (this.rank > 1 && this.rank <= 5) text(`#${this.rank} στην κατάταξη`, cx, WIN_H/2+52, { size: 22, color: C.dim, align: "center", base: "middle" });

    const actions = [["2","Ξανά"],["3","Βαθμολογίες"],["ESC","Μενού"]];
    let x0 = cx - 190;
    for (const [k, l] of actions) {
      text(k, x0, WIN_H/2+116, { size: 18, color: C.accent, base: "middle" });
      text(l, x0+34, WIN_H/2+116, { size: 18, color: C.dim, base: "middle" });
      x0 += 380/3;
    }
  }

  // ── SCORES ───────────────────────────────────────────────────────────────────
  drawScores() {
    const cx = WIN_W/2;
    panel(0, 0, WIN_W, WIN_H, C.bg, 0.92);
    text("Βαθμολογίες", cx, 80, { size: 40, align: "center", base: "middle" });
    ctx.strokeStyle = C.accent; ctx.beginPath(); ctx.moveTo(cx-120,108); ctx.lineTo(cx+120,108); ctx.stroke();
    const sc = scores.highScores();
    if (!sc.length) {
      text("Δεν υπάρχουν βαθμολογίες ακόμα.", cx, WIN_H/2, { size: 22, color: C.dim, align: "center", base: "middle" });
    } else {
      sc.forEach((e, i) => {
        const y = 148 + i*88, col = i === 0 ? C.amber : C.dim;
        text(String(i+1).padStart(2,"0"), cx-260, y, { size: 18, color: col, base: "middle" });
        text(`${e.score}`, cx-220, y, { size: 34, base: "middle" });
        text("pts", cx-220 + String(e.score).length*20, y+10, { size: 14, color: C.dim, base: "middle" });
        text(`${e.letters} γράμματα  ·  ${e.time}s  ·  ${e.date || ""}`, cx-220, y+36, { size: 16, color: C.dim, base: "middle" });
      });
    }
    text("ESC  Πίσω", cx, WIN_H-36, { size: 16, color: "rgb(65,62,60)", align: "center", base: "middle" });
  }
}

// ── Fullscreen ───────────────────────────────────────────────────────────────────
function toggleFullscreen() {
  if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
  else document.exitFullscreen?.();
}

// ── Bootstrap ────────────────────────────────────────────────────────────────────
const gate = document.getElementById("gate");
const startBtn = document.getElementById("start");
const errEl = document.getElementById("err");
const video = document.getElementById("cam");

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true; errEl.textContent = "Φόρτωση…";
  try {
    const [stream] = await Promise.all([
      navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false }),
    ]);
    video.srcObject = stream;
    await video.play();

    errEl.textContent = "Φόρτωση μοντέλου…";
    const [tracker] = await Promise.all([ createHandTracker(), gm.loadData(`${BASE}references.json`) ]);

    const det = document.createElement("canvas");
    det.width = video.videoWidth || 1280; det.height = video.videoHeight || 720;

    const app = new App(tracker, video, det);
    gate.style.display = "none";

    const loop = () => { app.frame(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  } catch (e) {
    console.error(e);
    startBtn.disabled = false;
    errEl.textContent = "Σφάλμα: " + (e?.message || e) + " — έλεγξε ότι έδωσες άδεια κάμερας.";
  }
});
