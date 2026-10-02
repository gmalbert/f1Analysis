import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    sourcemap: true,
    // Per PARITY_CHECKLIST §14: production main chunk must be < 500 KB gzipped.
    // Vite fails the build if any individual chunk exceeds this budget.
    chunkSizeWarningLimit: 500,
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.js'],
    css: false,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      include: ['src/**/*.{js,jsx}'],
      exclude: ['src/test/**', 'src/main.jsx', '**/*.test.{js,jsx}'],
      // PARITY_CHECKLIST §14 requires at least 80% frontend line coverage.
      thresholds: {
        lines: 80,
        functions: 40,
        branches: 60,
        statements: 60,
      },
    },
  },
});
