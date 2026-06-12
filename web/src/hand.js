// MediaPipe HandLandmarker (Tasks Vision, WASM) — browser equivalent of hand_tracker.py
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

// WASM runtime: pinned CDN (jsdelivr is reliable and version-locked).
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
// Model: self-hosted copy ships with the app; CDN is only a last-resort fallback.
const BASE = import.meta.env.BASE_URL || "/";
const MODEL_LOCAL = `${BASE}models/hand_landmarker.task`;
const MODEL_CDN =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/" +
  "hand_landmarker/float16/1/hand_landmarker.task";

export async function createHandTracker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);

  // Try GPU then CPU, and the local model then the CDN — first combo that builds wins.
  const make = (modelAssetPath, delegate) =>
    HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath, delegate },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.7,
      minHandPresenceConfidence: 0.7,
      minTrackingConfidence: 0.7,
    });

  let landmarker, lastErr;
  for (const model of [MODEL_LOCAL, MODEL_CDN]) {
    for (const delegate of ["GPU", "CPU"]) {
      try { landmarker = await make(model, delegate); break; }
      catch (e) { lastErr = e; }
    }
    if (landmarker) break;
  }
  if (!landmarker) throw lastErr || new Error("HandLandmarker init failed");

  return {
    /** Returns array of 21 {x,y,z} normalized landmarks, or null. */
    detect(source, timestampMs) {
      const res = landmarker.detectForVideo(source, timestampMs);
      return res.landmarks && res.landmarks.length ? res.landmarks[0] : null;
    },
  };
}
