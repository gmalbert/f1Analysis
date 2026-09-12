// Capture screenshots of the React/FastAPI app for visual diffing against
// the Streamlit reference. Requires the FastAPI backend to be running
// on http://127.0.0.1:8000 and the Vite dev server (or `vite preview`)
// on http://127.0.0.1:5173.
//
// Usage:
//   npm run capture:react
//   # or:
//   node parity_evidence/capture_react.mjs
//
// Outputs PNGs into parity_evidence/screenshots/react/ at the configured
// viewports. Run npm run capture:streamlit next, then npm run capture:diff
// to compute the pixel-difference percentages.

import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT = join(__dirname, 'screenshots', 'react');
const VIEWS = [
  { name: 'desktop', width: 1280, height: 800 },
  { name: 'tablet', width: 768, height: 1024 },
];

const PAGES = [
  { name: 'home', hash: '#/Data%20Explorer' },
  { name: 'data-explorer', hash: '#/Data%20Explorer' },
  { name: 'analytics', hash: '#/Analytics' },
  { name: 'current-season', hash: '#/Current%20Season' },
  { name: 'next-race', hash: '#/Next%20Race' },
  { name: 'models', hash: '#/Predictive%20Models' },
  { name: 'raw-data', hash: '#/Raw%20Data' },
  { name: 'betting-research', hash: '#/Betting%20Research' },
];

const BASE = process.env.REACT_BASE_URL || 'http://127.0.0.1:5173';
const WAIT_MS = parseInt(process.env.REACT_WAIT_MS || '3000', 10);

async function run() {
  await mkdir(OUT, { recursive: true });
  const browser = await chromium.launch();
  try {
    for (const view of VIEWS) {
      const context = await browser.newContext({ viewport: { width: view.width, height: view.height } });
      const page = await context.newPage();
      for (const target of PAGES) {
        const url = `${BASE}/${target.hash}`;
        console.log(`[${view.name}] ${url}`);
        // goto() only changes the fragment between routes (same-document
        // navigation), which does not remount the React app, so force a real
        // load to apply the hash on mount.
        await page.goto(url, { waitUntil: 'networkidle', timeout: 30_000 });
        await page.reload({ waitUntil: 'networkidle', timeout: 30_000 });
        // Disable transitions and wait for charts to settle
        await page.addStyleTag({ content: '*{transition:none!important;animation:none!important;}' });
        await page.waitForTimeout(WAIT_MS);
        const out = join(OUT, `${view.name}-${target.name}.png`);
        await page.screenshot({ path: out });
        console.log(`  -> ${out}`);
      }
      await context.close();
    }
  } finally {
    await browser.close();
  }
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
