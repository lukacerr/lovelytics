import { defineConfig } from 'vite'
import { tanstackRouter } from '@tanstack/router-plugin/vite'
import viteReact from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// SPA only — no SSR, no server functions. The build output (`dist/`) is a
// static bundle suitable for Cloudflare Pages (see AGENTS.md §11).
export default defineConfig({
  plugins: [
    // Router plugin MUST come before viteReact() — generates routeTree.gen.ts
    // and does auto code splitting per route.
    tanstackRouter({ target: 'react', autoCodeSplitting: true }),
    viteReact(),
    tailwindcss(),
  ],
  server: { port: 3000 },
  preview: { port: 3000 },
})
