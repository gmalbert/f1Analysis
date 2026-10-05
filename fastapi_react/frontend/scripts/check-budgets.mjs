import {readFile, readdir, writeFile} from 'node:fs/promises';
import {gzipSync} from 'node:zlib';
import {resolve, relative, sep} from 'node:path';
import {fileURLToPath, URL} from 'node:url';
import console from 'node:console';
import process from 'node:process';

const defaultDist = () => fileURLToPath(new URL('../dist/', import.meta.url));
export const ENTRY_GZIP_BUDGET = 500000;

export async function checkBudgets(directory = defaultDist(), limit = ENTRY_GZIP_BUDGET) {
  const root = resolve(directory);
  const manifest = JSON.parse(await readFile(resolve(root, '.vite/manifest.json'), 'utf8'));
  const entries = Object.values(manifest).filter(item => item.isEntry);
  if (entries.length !== 1) throw new Error('Expected exactly one production entry.');
  const initial = new Set();
  function visit(item) {
    if (!item || typeof item.file !== 'string') throw new Error('Invalid production manifest.');
    if (initial.has(item.file)) return;
    initial.add(item.file);
    for (const key of item.imports || []) visit(manifest[key]);
  }
  visit(entries[0]);
  const rows = [];
  async function scan(folder) {
    for (const entry of await readdir(folder, {withFileTypes: true})) {
      const path = resolve(folder, entry.name);
      if (entry.isDirectory()) await scan(path);
      else if (entry.name.endsWith('.map')) throw new Error('Production source maps must not be published.');
      else if (entry.name.endsWith('.js')) {
        const body = await readFile(path);
        rows.push({name: relative(root, path).split(sep).join('/'), bytes: body.length,
          gzip_bytes: gzipSync(body, {level: 5}).length});
      }
    }
  }
  await scan(root);
  const initialRows = rows.filter(row => initial.has(row.name));
  if (initialRows.length !== initial.size) throw new Error('An initial JavaScript chunk is missing.');
  const initialBytes = initialRows.reduce((sum, row) => sum + row.gzip_bytes, 0);
  if (initialBytes > limit) throw new Error(`Initial JavaScript is ${initialBytes} gzip bytes; budget is ${limit}.`);
  return {budget_gzip_bytes: limit, initial_javascript_gzip_bytes: initialBytes,
    all_javascript_gzip_bytes: rows.reduce((sum, row) => sum + row.gzip_bytes, 0),
    compression_level: 5, source_maps: false, chunks: rows.sort((a, b) => a.name.localeCompare(b.name))};
}

if (import.meta.url.startsWith('file:') && process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const directory = process.argv[2] || defaultDist();
  const report = await checkBudgets(directory);
  await writeFile(resolve(directory, 'build-budget.json'), JSON.stringify(report, null, 2) + '\n');
  console.info(JSON.stringify(report, null, 2));
}
