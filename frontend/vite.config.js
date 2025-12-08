import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
  plugins: [react()],
  publicDir: 'public',
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/chat': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/status': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/health': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/static': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/login': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/logout': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/auth': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/admin': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/reload': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/extract-pdf': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/extract-xlsx': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/enhance-pdf': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/approve-summary': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/approve-xlsx': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/add-text': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/upload': {
        target: 'http://localhost:5000',
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: '../app/static/react',
    emptyOutDir: true,
    copyPublicDir: true
  }
})
