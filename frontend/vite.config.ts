import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

const agenticApiTarget = process.env.VITE_AGENTIC_API_ORIGIN || 'http://localhost:8000'
const agenticWsTarget = process.env.VITE_AGENTIC_WS_ORIGIN || agenticApiTarget.replace(/^http/i, 'ws')

// https://vitejs.dev/config/
export default defineConfig({
  base: process.env.VITE_BASE_PATH || '/',
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: agenticApiTarget,
        changeOrigin: true,
      },
      '/agentic-api': {
        target: agenticApiTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/agentic-api/, ''),
      },
      '/ws': {
        target: agenticWsTarget,
        ws: true,
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
  },
})