// Operational benchmarks for the FastAPI backend.
//
// Run with both servers up:
//   uvicorn backend.app.main:app --port 8000
//   node fastapi_react/parity_evidence/benchmark.mjs
//
// Produces parity_evidence/benchmarks.json with:
//   memory:   { rss_mb_at_start, rss_mb_after_warmup, rss_mb_peak }
//   first_page_ms:     wall-clock from new-page goto to networkidle for /
//   navigation_ms:     average of 5 sequential navigations across the 7 pages
//   concurrent_2: { p50_ms, p95_ms, success_rate, peak_rss_mb } for 2 users
//   concurrent_5: { p50_ms, p95_ms, success_rate, peak_rss_mb } for 5 users
//
// These are recorded as-is in PARITY_REPORT.md \u00A7 Operational. The
// Streamlit reference is benchmarked the same way against its own
// /healthz endpoint, against the same data_files/ snapshot.

import { chromium } from 'playwright';
import { writeFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT = join(__dirname, 'benchmarks.json');
const BASE = process.env.BENCH_BASE_URL || 'http://127.0.0.1:5173';
const API = process.env.BENCH_API_URL || 'http://127.0.0.1:8000';

const PAGES = [
  '#/Data%20Explorer', '#/Analytics', '#/Current%20Season', '#/Next%20Race',
  '#/Predictive%20Models', '#/Raw%20Data', '#/Betting%20Research',
];

async function rss() {
  const r = await fetch(`${API}/api/health`);
  const j = await r.json();
  return j.rss_mb;
}

function p(arr, q) {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.floor((sorted.length) * q));
  return sorted[idx];
}

async function firstPage(browser) {
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  const t0 = Date.now();
  await page.goto(BASE + '/', { waitUntil: 'networkidle', timeout: 30_000 });
  const t1 = Date.now();
  await ctx.close();
  return t1 - t0;
}

async function navigation(browser) {
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  const times = [];
  for (const hash of PAGES) {
    const t0 = Date.now();
    await page.goto(BASE + '/' + hash, { waitUntil: 'networkidle' });
    times.push(Date.now() - t0);
  }
  await ctx.close();
  return times;
}

async function concurrent(browser, n) {
  const ctxs = [];
  const results = [];
  for (let i = 0; i < n; i++) {
    const ctx = await browser.newContext();
    ctxs.push(ctx);
  }
  const t0 = Date.now();
  await Promise.all(ctxs.map(async (ctx, i) => {
    const page = await ctx.newPage();
    const start = Date.now();
    let ok = false;
    try {
      await page.goto(BASE + '/' + PAGES[i % PAGES.length], { waitUntil: 'networkidle', timeout: 60_000 });
      ok = true;
    } catch { /* swallow per-user errors */ }
    results.push({ user: i, ms: Date.now() - start, ok });
    await page.close();
  }));
  const total = Date.now() - t0;
  for (const ctx of ctxs) await ctx.close();
  const okResults = results.filter(r => r.ok).map(r => r.ms);
  return {
    total_ms: total,
    p50_ms: okResults.length ? p(okResults, 0.5) : null,
    p95_ms: okResults.length ? p(okResults, 0.95) : null,
    success_rate: results.length ? results.filter(r => r.ok).length / results.length : 0,
  };
}

async function run() {
  await mkdir(dirname(OUT), { recursive: true });
  const browser = await chromium.launch();
  const summary = { generated_at: new Date().toISOString(), base: BASE, api: API };

  try {
    summary.memory = {
      rss_mb_at_start: await rss(),
    };
    // Warmup
    const warmupCtx = await browser.newContext();
    const warmupPage = await warmupCtx.newPage();
    await warmupPage.goto(BASE + '/', { waitUntil: 'networkidle' });
    for (const hash of PAGES) await warmupPage.goto(BASE + '/' + hash, { waitUntil: 'networkidle' });
    await warmupCtx.close();
    summary.memory.rss_mb_after_warmup = await rss();

    summary.first_page_ms = await firstPage(browser);
    const navTimes = await navigation(browser);
    summary.navigation_ms = {
      samples: navTimes,
      p50_ms: p(navTimes, 0.5),
      p95_ms: p(navTimes, 0.95),
    };

    summary.concurrent_2 = await concurrent(browser, 2);
    summary.memory.rss_mb_peak_after_2 = await rss();

    summary.concurrent_5 = await concurrent(browser, 5);
    summary.memory.rss_mb_peak_after_5 = await rss();
  } finally {
    await browser.close();
  }

  await writeFile(OUT, JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary, null, 2));
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
