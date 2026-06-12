# Sign Language GR — Web (browser edition)

Fully client-side rewrite of the ΕΝΓ learning game. Hand tracking runs in the
browser via **MediaPipe Tasks Vision (WASM)**; the gesture classifier (normalize
+ KNN/cosine) is ported to JavaScript and runs entirely on the device. No server,
no data leaves the browser.

## Local dev

```bash
cd web
npm install
npm run dev        # open the printed localhost URL, allow camera
```

## Build

```bash
npm run build      # → web/dist (static)
npm run preview    # serve the production build locally
```

## Deploy to Vercel

This app lives in the `web/` subfolder of the repo. In the Vercel project settings:

- **Framework Preset:** Vite
- **Root Directory:** `web`
- Build command `npm run build`, output `dist` (Vite defaults — auto-detected)

That's it — it's a static site, no env vars or serverless functions needed.

## How the data was ported

`web/public/references.json` is exported from the Python project's
`data/references/*.npy` (the self-recorded hand samples), cleaned (dominant
orientation cluster) and trimmed to ~50 samples per letter. The browser rebuilds
the KNN dataset from it on load, adding x-mirror augmentation — identical logic to
`gesture_matcher.py`.

Reference images in `web/public/letters/` are the same per-letter illustrations
used by the desktop app.
