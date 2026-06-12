import { defineConfig } from "vite";

// Fully static, client-side app — deploys to Vercel with zero config.
// (Set `base` only if hosting under a sub-path; root deploy needs nothing.)
export default defineConfig({
  build: { target: "es2020", outDir: "dist" },
});
