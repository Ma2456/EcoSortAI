import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The FastAPI backend runs on :8000 by default. During development every
// request to /api is proxied there so the frontend can use relative URLs.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.ECOSORT_API_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  preview: {
    proxy: {
      '/api': {
        target: process.env.ECOSORT_API_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});
