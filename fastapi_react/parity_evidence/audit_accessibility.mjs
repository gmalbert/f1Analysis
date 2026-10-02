import { createRequire } from 'node:module';
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const requireFromFrontend = createRequire(join(__dirname, '../frontend/package.json'));
const { chromium } = requireFromFrontend('playwright');
const OUT_DIR = join(__dirname);
const OUT = join(OUT_DIR, 'accessibility.json');
const BASE = process.env.REACT_BASE_URL || 'http://127.0.0.1:5173';
const PAGES = [
  ['Data Explorer', '#/Data%20Explorer'],
  ['Analytics', '#/Analytics'],
  ['Current Season', '#/Current%20Season'],
  ['Next Race', '#/Next%20Race'],
  ['Predictive Models', '#/Predictive%20Models'],
  ['Raw Data', '#/Raw%20Data'],
  ['Betting Research', '#/Betting%20Research'],
];
const axePath = join(__dirname, '..', 'frontend', 'node_modules', 'axe-core', 'axe.min.js');

async function run() {
  await mkdir(OUT_DIR, { recursive: true });
  const browser = await chromium.launch();
  const results = [];
  try {
    for (const [name, hash] of PAGES) {
      const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
      await page.goto(`${BASE}/${hash}`, { waitUntil: 'networkidle', timeout: 60_000 });
      await page.waitForTimeout(1200);
      await page.addScriptTag({ path: axePath });
      const audit = await page.evaluate(async () => {
        const result = await window.axe.run(document, {
          runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa', 'best-practice'] },
        });
        return {
          passes: result.passes.length,
          violations: result.violations.map(violation => ({
            id: violation.id,
            impact: violation.impact,
            description: violation.description,
            nodes: violation.nodes.map(node => ({ target: node.target, summary: node.failureSummary })),
          })),
        };
      });
      results.push({ page: name, ...audit });
      await page.close();
    }
  } finally {
    await browser.close();
  }
  const summary = {
    generated_at: new Date().toISOString(),
    base: BASE,
    pages: results,
    violation_count: results.reduce((sum, page) => sum + page.violations.length, 0),
  };
  await writeFile(OUT, JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary, null, 2));
  if (summary.violation_count) process.exitCode = 1;
}

run().catch(error => {
  console.error(error);
  process.exit(1);
});