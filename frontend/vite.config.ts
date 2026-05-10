import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      // /api/* → http://localhost:8000/* (strip the /api prefix).
      // SSE 走 chunked HTTP；不需要 ws。
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        ws: false,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
