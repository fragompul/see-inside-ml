import { fileURLToPath, URL } from 'node:url';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  // Relative base keeps the build deployable on GitHub Pages under any repo name.
  base: './',
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  // The 3D scene (three.js) is a lazy chunk of about 930 kB; tools/check-bundle.mjs holds the budget instead.
  build: { chunkSizeWarningLimit: 1000 },
  server: { port: 5173, open: false },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    css: false,
    // some lab tests train several models; give them room on slow CI runners
    testTimeout: 300_000,
  },
});
