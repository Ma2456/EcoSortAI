/**
 * Single-file build: `npm run build:single` → dist-single/EcoSortAI.html
 * Inlines all JS, CSS, images AND the park video (as data URIs), so the page
 * opens directly from disk (double-click) with no server.
 */
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { viteSingleFile } from 'vite-plugin-singlefile';
import fs from 'node:fs';
import path from 'node:path';

const MEDIA = path.resolve(__dirname, 'public/media');
const mime = (f) => (f.endsWith('.mp4') ? 'video/mp4' : f.endsWith('.webm') ? 'video/webm' : f.endsWith('.png') ? 'image/png' : 'image/jpeg');
const cache = new Map();
const dataUri = (file) => {
  if (!cache.has(file)) cache.set(file, `data:${mime(file)};base64,${fs.readFileSync(path.join(MEDIA, file)).toString('base64')}`);
  return cache.get(file);
};

// Which embedded file each public URL resolves to (one video per format to keep size down).
const MAP = {
  '/media/park-720.mp4': 'park-720.mp4',
  '/media/park-480.mp4': 'park-720.mp4',
  '/media/original/Camera_journey_through_eco-park_20261015162017.mp4': 'park-720.mp4',
  '/media/park-720.webm': 'park-480.webm',
  '/media/park-480.webm': 'park-480.webm',
  '/media/poster.jpg': 'poster.jpg',
  '/media/end.jpg': 'end.jpg',
  '/media/sample-metal.jpg': 'sample-metal.jpg',
};

function inlineMedia() {
  return {
    name: 'inline-media',
    enforce: 'pre',
    transform(code, id) {
      if (!/\.(jsx?|css)$/.test(id.split('?')[0]) || !code.includes('/media/')) return null;
      if (id.split('?')[0].endsWith('.css')) {
        return { code: code.replace(/url\((['"]?)(\/media\/[^'")]+)\1\)/g, (m, q, p) => (MAP[p] ? `url("${dataUri(MAP[p])}")` : m)), map: null };
      }
      // JS: declare each embedded file once per module, reference it by name.
      const decls = new Map();
      const out = code.replace(/(['"])(\/media\/[^'"]+)\1/g, (m, q, p) => {
        if (!MAP[p]) return m;
        const file = MAP[p];
        if (!decls.has(file)) decls.set(file, `__MEDIA_${decls.size}`);
        return decls.get(file);
      });
      const head = [...decls].map(([file, v]) => `const ${v} = ${JSON.stringify(dataUri(file))};`).join('\n');
      return { code: `${head}\n${out}`, map: null };
    },
  };
}

export default defineConfig({
  plugins: [react(), inlineMedia(), viteSingleFile({ removeViteModuleLoader: true })],
  publicDir: false,
  build: {
    outDir: 'dist-single',
    emptyOutDir: true,
    assetsInlineLimit: 100_000_000,
    chunkSizeWarningLimit: 100_000,
    cssCodeSplit: false,
  },
});
