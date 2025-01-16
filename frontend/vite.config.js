import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: import.meta.env.PORT || 5173
  },
  define: {
    'process.env.VITE_API_URL': JSON.stringify(import.meta.env.VITE_API_URL)
  }
});