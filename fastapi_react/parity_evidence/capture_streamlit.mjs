// Capture screenshots of the Streamlit reference app for visual diffing
// against the React/FastAPI parity build.
//
// Usage:
//   1. Start the Streamlit app on port 8501:
//        streamlit run raceAnalysis.py --server.port 8501 --server.headless true
//   2. Run:
//        node parity_evidence/capture_streamlit.mjs
//
// Outputs PNGs into parity_evidence/screenshots/streamlit/ at the same
// viewports as capture_react.mjs. The page-name mapping is approximate:
// Streamlit uses one continuous page with tabs/sidebar, so we drive
// the URL hash (Streamlit supports a query-param route via
// ?tab=<index> in some forks; the default Streamlit page is the only
// one we screenshot here). The "matching" between React pages and
// Streamlit sections is recorded in PARITY_CHECKLIST.md \u00A713.

import { createRequire } from 'node:module';
import { mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const requireFromFrontend = createRequire(join(__dirname, '../frontend/package.json'));
const { chromium } = requireFromFrontend('playwright');
const SNAPSHOT_DIR = process.env.PARITY_SCREENSHOT_DIR || join(__dirname, 'visual');
const OUT = join(SNAPSHOT_DIR, 'streamlit');
const VIEWS = [
  { name: 'desktop', width: 1280, height: 800 },
  { name: 'tablet', width: 768, height: 1024 },
  { name: 'mobile', width: 390, height: 844 },
];

const SECTIONS = [
  { name: 'home', label: null },
  { name: 'data-explorer', label: /Data Explorer/ },
  { name: 'analytics', label: /Analytics & Visualizations/ },
  { name: 'current-season', label: /Schedule/ },
  { name: 'next-race', label: /Next Race/ },
  { name: 'models', label: /Predictive Models/ },
  { name: 'raw-data', label: /Data & Debug/ },
  { name: 'betting-research', label: /Betting Research/ },
];

const BASE = process.env.STREAMLIT_BASE_URL || 'http://127.0.0.1:8501';
const WAIT_MS = parseInt(process.env.STREAMLIT_WAIT_MS || '5000', 10);

async function wakeIfNeeded(page) {
  const wake = page.getByRole('button', { name: /wake|back up|get this app|yes/i }).first();
  if (await wake.count()) {
    try {
      await wake.click({ timeout: 5000 });
      await page.waitForTimeout(8000);
    } catch {
      // The page may already be waking or the control may disappear mid-click.
    }
  }
}

async function clickStreamlitTab(page, label) {
  const candidates = [
    page.getByRole('tab', { name: label }).first(),
    page.locator('button[data-baseweb="tab"]').filter({ hasText: label }).first(),
    page.locator('[data-testid="stTabs"] button').filter({ hasText: label }).first(),
    page.getByText(label).first(),
  ];
  for (const candidate of candidates) {
    if (await candidate.count()) {
      try {
        await candidate.click({ timeout: 8000, force: true });
        return;
      } catch {
        // Try the next DOM shape; Streamlit changes markup across releases.
      }
    }
  }
  const bodyText = (await page.locator('body').innerText()).slice(0, 5000);
  throw new Error(`Could not find Streamlit tab ${label}. Page text: ${bodyText}`);
}

async function run() {
  await mkdir(OUT, { recursive: true });
  const browser = await chromium.launch();
  try {
    for (const view of VIEWS) {
      const context = await browser.newContext({ viewport: { width: view.width, height: view.height } });
      const page = await context.newPage();
      console.log(`[${view.name}] loading ${BASE}/`);
      await page.goto(BASE + '/', { waitUntil: 'domcontentloaded', timeout: 60_000 });
      await wakeIfNeeded(page);
      await page.addStyleTag({ content: '*{font-family:"Segoe UI",system-ui,sans-serif!important;transition:none!important;animation:none!important;} [data-testid="stToolbar"],[data-testid="stStatusWidget"]{visibility:hidden!important;} img[alt="Betting Oracle Logo"]{visibility:hidden!important;}' });
      await page.waitForTimeout(WAIT_MS);
      const normalizeDynamicText = async () => {
        const lastUpdated = page.getByText(/^Last updated:/).first();
        const deployed = page.getByText(/^Code deployed at:/).first();
        if (await lastUpdated.count()) await lastUpdated.evaluate(el => { el.textContent = "Last updated: 2026-09-30 09:00 PM"; });
        if (await deployed.count()) await deployed.evaluate(el => { el.textContent = "Code deployed at: 2026-09-30 21:00:00 UTC"; });
      };
      await normalizeDynamicText();
      for (const section of SECTIONS) {
        if (section.name === 'analytics') {
          await clickStreamlitTab(page, /Data Explorer/);
          const filterToggle = page.getByRole('checkbox', { name: 'Filter Results' });
          if (!(await filterToggle.isChecked())) await filterToggle.check({ force: true });
          await page.waitForTimeout(WAIT_MS);
          await clickStreamlitTab(page, /Analytics & Visualizations/);
          await page.waitForTimeout(WAIT_MS);
        } else if (section.label) {
          await clickStreamlitTab(page, section.label);
          await page.waitForTimeout(WAIT_MS);
        }
        await normalizeDynamicText();
        const out = join(OUT, `${view.name}-${section.name}.png`);
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
