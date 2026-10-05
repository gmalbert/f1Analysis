/* global fetch, AbortSignal */
import assert from 'node:assert/strict';
import process from 'node:process';
import console from 'node:console';

// Verify actual Nginx headers, not the Vite development proxy's policies.
const base = process.argv[2] || 'http://127.0.0.1:8080';
const get = (path, options) => fetch(base + path, {...options, signal: AbortSignal.timeout(15000)});
const checks = [];
async function expectCache(path, cache, status = 200) {
  const response = await get(path);
  assert.equal(response.status, status, path);
  assert.equal(response.headers.get('cache-control'), cache, path);
  checks.push({path, status: response.status, cache_control: cache});
  return response;
}
const html = await (await expectCache('/index.html', 'no-cache')).text();
await expectCache('/route-that-uses-the-spa-fallback', 'no-cache');
const entry = html.match(/src="([^"]+\.js)"/)?.[1];
assert.ok(entry, 'The built entry is absent from index.html.');
const main = await expectCache(entry, 'public, max-age=31536000, immutable');
assert.equal(main.headers.get('content-encoding'), 'gzip');
assert.match(main.headers.get('vary') || '', /Accept-Encoding/i);
await expectCache('/betting-oracle-logo-60.webp', 'public, max-age=3600');
await expectCache('/favicon.png', 'public, max-age=3600');
await expectCache(entry + '.map', 'no-store', 404);
assert.equal((await get('/.vite/manifest.json')).status, 404);
assert.equal((await get('/assets/missing.js')).status, 404);
for (const path of ['/api/health', '/api/missing.png', '/api/missing.map']) {
  const response = await get(path);
  assert.equal(response.headers.get('cache-control'), 'no-store');
  checks.push({path, status: response.status, cache_control: 'no-store'});
}
console.info(JSON.stringify({pass: true, base, checks}, null, 2));
