// Compare paired React/Streamlit screenshots and produce a per-page
// pixel-difference percentage plus an overall summary.
//
// Usage:
//   node parity_evidence/diff_screenshots.mjs
//
// Reads:
//   parity_evidence/screenshots/react/{viewport}-{page}.png
//   parity_evidence/screenshots/streamlit/{viewport}-{page}.png
//
// Writes:
//   parity_evidence/diff/summary.json
//   parity_evidence/diff/{viewport}-{page}.png   (pixel-diff image)
//
// Implementation: uses the `sharp` package for image decoding and a
// plain per-pixel L1 distance in RGBA space, then averages over the
// pixel count. Pixel-diff images are produced by writing the
// per-pixel absolute difference as an RGBA PNG (alpha = where
// pixels differ significantly).

import { readdir, mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import sharp from 'sharp';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REACT_DIR = join(__dirname, 'screenshots', 'react');
const SL_DIR = join(__dirname, 'screenshots', 'streamlit');
const OUT = join(__dirname, 'diff');

// \u00A713 tolerance: <=2% diff at desktop, <=3% at tablet.
const TOLERANCE = { desktop: 0.02, tablet: 0.03 };
// Per-channel distance threshold for marking a pixel as "different".
const PIXEL_THRESHOLD = 24;

async function diffPair(reactPath, slPath, outPath) {
  const [a, b] = await Promise.all([sharp(reactPath).raw().toBuffer({ resolveWithObject: true }),
                                   sharp(slPath).raw().toBuffer({ resolveWithObject: true })]);
  if (a.info.width !== b.info.width || a.info.height !== b.info.height) {
    return { error: `size mismatch: ${a.info.width}x${a.info.height} vs ${b.info.width}x${b.info.height}` };
  }
  const { width, height, channels } = a.info;
  const total = width * height;
  const out = Buffer.alloc(total * 4);
  let diffPixels = 0;
  let totalDistance = 0;
  for (let i = 0, p = 0; i < a.data.length; i += channels, p += 4) {
    const dr = Math.abs(a.data[i] - b.data[i]);
    const dg = Math.abs(a.data[i + 1] - b.data[i + 1]);
    const db = Math.abs(a.data[i + 2] - b.data[i + 2]);
    const dist = (dr + dg + db) / 3;
    totalDistance += dist;
    if (dist > PIXEL_THRESHOLD) {
      diffPixels += 1;
      out[p] = 255;
      out[p + 1] = 0;
      out[p + 2] = 0;
      out[p + 3] = 255;
    } else {
      out[p] = a.data[i];
      out[p + 1] = a.data[i + 1];
      out[p + 2] = a.data[i + 2];
      out[p + 3] = 96;
    }
  }
  await sharp(out, { raw: { width, height, channels: 4 } }).png().toFile(outPath);
  return {
    diff_pixels: diffPixels,
    total_pixels: total,
    diff_ratio: diffPixels / total,
    mean_distance: totalDistance / total,
  };
}

async function run() {
  await mkdir(OUT, { recursive: true });
  const [reactFiles, slFiles] = await Promise.all([readdir(REACT_DIR), readdir(SL_DIR)]);
  const reactSet = new Set(reactFiles);
  const summary = { generated_at: new Date().toISOString(), pages: [] };
  for (const f of slFiles) {
    if (!reactSet.has(f)) continue;
    const [viewport, ...rest] = f.split('-');
    const page = rest.join('-').replace(/\.png$/, '');
    const r = await diffPair(join(REACT_DIR, f), join(SL_DIR, f), join(OUT, f));
    if (r.error) {
      summary.pages.push({ viewport, page, error: r.error });
      continue;
    }
    const tolerance = TOLERANCE[viewport] ?? 0.02;
    summary.pages.push({
      viewport, page, ...r,
      within_tolerance: r.diff_ratio <= tolerance,
      tolerance,
    });
  }
  await writeFile(join(OUT, 'summary.json'), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary, null, 2));
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
