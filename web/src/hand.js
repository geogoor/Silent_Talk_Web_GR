// MediaPipe HandLandmarker (Tasks Vision, WASM) — browser equivalent of hand_tracker.py
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/" +
  "hand_landmarker/float16/1/hand_landmarker.task";

export async function createHandTracker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  let landmarker;
  try {
    landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.7,
      minHandPresenceConfidence: 0.7,
      minTrackingConfidence: 0.7,
    });
  } catch {
    // Fall back to CPU if GPU delegate is unavailable
    landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
      runningMode: "VIDEO",
      numHands: 1,
    });
  }

  return {
    /** Returns array of 21 {x,y,z} normalized landmarks, or null. */
    detect(source, timestampMs) {
      const res = landmarker.detectForVideo(source, timestampMs);
      return res.landmarks && res.landmarks.length ? res.landmarks[0] : null;
    },
  };
}
