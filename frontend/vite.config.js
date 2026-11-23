import fs from "node:fs";
import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

function satarkPrecachePlugin() {
  return {
    name: "satark-precache-sw",
    closeBundle() {
      const distDir = path.resolve(__dirname, "dist");
      const assetsDir = path.join(distDir, "assets");
      const swPath = path.join(distDir, "sw.js");
      if (!fs.existsSync(swPath) || !fs.existsSync(assetsDir)) return;

      const assetFiles = fs
        .readdirSync(assetsDir)
        .filter((f) => /\.(js|css)$/i.test(f))
        .map((f) => `"./assets/${f}"`)
        .join(",\n  ");

      const swSrc = fs.readFileSync(swPath, "utf8");
      const updated = swSrc.replace("/* __PRECACHE_ASSETS__ */", assetFiles);
      fs.writeFileSync(swPath, updated, "utf8");
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), satarkPrecachePlugin()],
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
