# Deployment, measurements, and validation

## O1 — Responsive footer assets

The existing footer PNG is roughly 1.13 MB although it displays at a small size. The supplied Sharp script creates resized losslessly encoded WebP files at 60- and 120-pixel heights, retaining the source PNG as fallback. The image uses width/height attributes, lazy loading, and asynchronous decoding.

The generated preview assets are **4,938 bytes at 1×** and **15,096 bytes at 2×**. This is a measured asset-size reduction, not a measured whole-page load-time improvement. Resizing intentionally reduces source resolution; “lossless” describes the encoding of the resized output. The visible branding source stays the same. [Sharp resize documentation](https://sharp.pixelplumbing.com/api-resize/).

Copy `optimize-assets.mjs` into `frontend/scripts/` and use the supplied package build scripts. It writes responsive assets into `public/` before Vite copies them to `dist/`. The proposed `App.jsx` references them with a PNG fallback.

## O2 — Build budget and production delivery

**Budget.** `check-budgets.mjs` actually fails if the main entry exceeds 500,000 gzip bytes. The current Vite `chunkSizeWarningLimit` is a warning, not a build failure. The supplied Vite config disables public sourcemaps; Nginx also denies `.map` requests.

**Recorded bundle.** The final preview's main entry is **699,541 bytes plain / 226,969 bytes using the budget script's gzip settings**. All JavaScript chunks together are **2,015,252 gzip bytes**. Plotly alone is about **1.48 MB gzip**, loaded as a separate chunk. These are output artifact sizes; actual browser bandwidth depends on which routes/charts are visited, HTTP compression, cache state, and source maps. Vite's printed gzip estimate differs slightly because its compression settings differ.

**Serving policy.** Enable gzip level five, one-year immutable caching for hashed `/assets/` files, revalidation for `index.html`, one-hour caching for unversioned images/fonts, no shared API caching, a 600-second API read timeout, and a 256 MiB aggregate ingress cap. The configuration proxies to `backend:8000`; adapt that hostname to the deployed service topology.

**Limits.** Nginx deployment was not executed here. The supplied port-80 configuration assumes the hosting platform or an upstream ingress terminates TLS; provide that before exposing administrator tokens. Keep a single API worker/instance if using the local job queue. Do not describe static gzip or response reuse as a proven production throughput gain without measurements. [Vite build documentation](https://vite.dev/guide/build.html), [Nginx gzip documentation](https://nginx.org/en/docs/http/ngx_http_gzip_module.html).

## Recorded results

| Check | Result | Evidence/meaning |
| --- | --- | --- |
| Integrated backend pytest | 66 passed, 87.90% coverage | Existing 80% minimum retained |
| Integrated frontend Vitest | 63 passed, 73.78% statement/line coverage | Existing coverage thresholds retained |
| Backend compilation, Ruff, mypy | Passed | Mypy checked 18 source files |
| Frontend ESLint and TypeScript | Passed | No warnings/errors under the existing configuration |
| Vite production build | Passed | 1,112 modules transformed; existing large lazy chunks produce a nonfatal size warning |
| Enforced main gzip budget | Passed | [validation-budgets.json](validation-budgets.json) |
| Browser flows/screenshots | Five flows passed; zero captured errors | [validation-browser.json](validation-browser.json) |
| Real API reuse/timing/auth/raw identity | Passed | [validation-api.json](validation-api.json) |
| Standalone module contracts | Four Python tests and Node contracts passed | Source in the verification chapter |

The [complete quality-check record](validation-quality.json) includes the test counts, coverage, lint/type results, and explicit `py_compile` verification of 22 integrated Python files.

The browser flows cover semantic table paging/search, driver comparison, saved-view/cache controls, section search/navigation, and context JSON download. They capture desktop at 1280×900 and mobile at 390×844. Error collection includes uncaught page exceptions, console errors, and HTTP status codes of 400 or higher in the visited flows.

The API probe verifies a genuine `X-F1-Cache: HIT`, timing headers, guarded administrator routes, and the raw table's canonical content checksum:

```text
0389ff31e162ebc06710cbbfc77c9ea8d3028ecce30dd502f4de5f044598415e
```

This matches the prior optimization baseline across all 4,629 rows and 561 columns. It is a data-content check; the stat-based source revision used by the proposed cache has a different purpose.

One backend test warning comes from the installed Starlette/httpx test-client deprecation. The production build retains warnings for the existing large lazy Plotly/Vega chunks. Neither is a browser console failure; the explicit main-entry budget passes. A full accessibility audit, actual expensive research run, production deployment, and multi-user load benchmark remain unperformed.

## Recreate the isolated preview

The package includes a generator that produces full integrated replacements and copies them to `fastapi_react/.runtime/enhancement-preview/`. It checks integration anchors and stops if the baseline has changed unexpectedly. It does not edit main application source files. Generated full replacements in `code/` are tied to that baseline.

From the repository root:

```powershell
.venv/Scripts/python.exe fastapi_react/enhancement_proposals/2026-10-01/prepare_preview.py
```

The generator copies application source, frontend configs/public assets/build scripts, and backend test configuration. It creates a node_modules junction to the main frontend's installed dependencies and refuses to replace an unexpected dependency path. Install the main frontend dependencies first. **Do not run `npm ci` in this staging copy**, because it shares the main checkout's dependency tree. Use the installed CLI entry points, or use a separate ordinary checkout with its own node_modules for dependency installation.

Build from the staged frontend:

```powershell
$env:VITE_F1_ENHANCEMENTS = '1'
node scripts/optimize-assets.mjs
node node_modules/vite/bin/vite.js build --config vite.config.js
node scripts/check-budgets.mjs
```

The optimizer/budget scripts use the current working directory. Run them in the staged frontend so they write/read its `public`/`dist`. The deployment appendix contains the exact scripts.

Start the staged API in a separate terminal. From its `backend` directory, set `F1_REPO_ROOT` to the absolute main repository path, `F1_ENHANCEMENTS=1`, and `F1_VIEW_RESPONSE_CACHE=1`. Run the main repository's virtualenv Uvicorn on an unused port, for example 9008. Never stop a preexisting listener just to claim this port.

Then, from the main repository root:

```powershell
$env:PROPOSAL_API_PORT = '9008'
node fastapi_react/enhancement_proposals/2026-10-01/checks/browser.mjs
.venv/Scripts/python.exe fastapi_react/enhancement_proposals/2026-10-01/checks/api.py
node fastapi_react/enhancement_proposals/2026-10-01/checks/frontend.mjs
.venv/Scripts/python.exe -m pytest fastapi_react/enhancement_proposals/2026-10-01/checks/test_backend.py --no-cov
```

The browser script launches temporary static preview servers itself and closes them afterward. The “current” screenshots read the main frontend's existing `dist` build; build that baseline separately if it is missing. The API probe accepts `PROPOSAL_API_PORT` as well. Use the ordinary frontend/backend validation commands in their implementation chapters to run the full integrated suites.

## Rollout

1. Review complete replacements against the recorded source snapshot and the current checkout. Install in a review branch with recoverable original files.
2. Run Python compilation, lint/type checks, both full unit suites, the Vite build, and the main-entry budget.
3. Start the integrated backend with enhancements enabled but response cache off. Verify full raw content, exports, uploads, and normal analysis flows.
4. Build the frontend with tools enabled. Check light/dark themes, keyboard access, mobile layout, years, numeric fonts, original CSV downloads, and the existing parity workflows.
5. Enable server/client reuse only after revision invalidation checks pass. Measure cold/warm results separately and watch retained memory.
6. Enable administrator jobs only for a trusted local session. Test a small real audit before a long computation and observe the separate process's memory/CPU.
7. Deploy the reviewed Nginx/static configuration, then verify real cache headers, compression, SPA fallback, request limits, and denied source maps.

## Rollback

Turn off `F1_ENHANCEMENTS` and `F1_VIEW_RESPONSE_CACHE`, restart the API, and rebuild the frontend with `VITE_F1_ENHANCEMENTS=0`. A user can immediately disable the readability/cache choices in the tools drawer. Running job shutdown waits for completion; plan restarts accordingly. Restore the original tracked files to remove lifecycle and asset implementation changes as well. Job IDs/results are process-local and will not survive the restart.

## Full deployment source

The files below are complete. The supplied package file changes scripts, not dependency versions. The Nginx file should replace the frontend serving configuration only after adapting the upstream host and reviewing the actual hosting setup.

## code/deployment/check-budgets.mjs

[Separate source file](code/deployment/check-budgets.mjs)

```javascript
/* global console */
import {readFile,readdir,stat} from 'node:fs/promises';
import {gzipSync} from 'node:zlib';
import {resolve} from 'node:path';

// Install at frontend/scripts/check-budgets.mjs; run after npm run build.
const assets = resolve('dist/assets');
const rows = [];
for(const name of await readdir(assets)) {
  if(!name.endsWith('.js'))continue;
  const path = resolve(assets,name),body = await readFile(path);
  rows.push({name,bytes:(await stat(path)).size,gzip_bytes:gzipSync(body,{level:5}).length});
}
const main = rows.find(row => /^index-.*\.js$/.test(row.name));
if(!main)throw new Error('The main build chunk is missing.');
if(main.gzip_bytes > 500000)throw new Error('Main JavaScript exceeds the 500 KB gzip budget.');
console.log(JSON.stringify({main,all_javascript_gzip_bytes:rows.reduce((sum,row) => sum+row.gzip_bytes,0),chunks:rows},null,2));
```

## code/deployment/logging.json

[Separate source file](code/deployment/logging.json)

```json
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "text": {"format": "%(levelname)s %(name)s %(message)s"},
    "json_record": {"format": "%(message)s"}
  },
  "handlers": {
    "console": {"class": "logging.StreamHandler", "formatter": "text", "stream": "ext://sys.stderr"},
    "requests": {"class": "logging.StreamHandler", "formatter": "json_record", "stream": "ext://sys.stderr"}
  },
  "loggers": {
    "f1.request": {"handlers": ["requests"], "level": "INFO", "propagate": false},
    "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": false},
    "uvicorn.access": {"handlers": ["console"], "level": "INFO", "propagate": false}
  },
  "root": {"handlers": ["console"], "level": "INFO"}
}
```

## code/deployment/nginx.conf

[Separate source file](code/deployment/nginx.conf)

```nginx
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    gzip on;
    gzip_vary on;
    gzip_comp_level 5;
    gzip_min_length 1000;
    gzip_types text/css application/javascript application/json image/svg+xml;

    location /api/ {
        proxy_pass http://backend:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600;
        proxy_request_buffering on;
        client_max_body_size 256m;
        # JSON presentations and uploads must not enter a shared HTTP cache.
        add_header Cache-Control "no-store" always;
    }
    location /assets/ {
        try_files $uri =404;
        add_header Cache-Control "public, max-age=31536000, immutable";
    }
    location ~* \.(woff2|png|webp|ico)$ {
        try_files $uri =404;
        add_header Cache-Control "public, max-age=3600";
    }
    location = /index.html {
        add_header Cache-Control "no-cache";
    }
    location ~ \.map$ {
        return 404;
    }
    location / {
        try_files $uri /index.html;
        add_header Cache-Control "no-cache";
    }
}
```

## code/deployment/optimize-assets.mjs

[Separate source file](code/deployment/optimize-assets.mjs)

```javascript
/* global console */
import sharp from 'sharp';
import {mkdir,stat} from 'node:fs/promises';
import {resolve} from 'node:path';

// Install at frontend/scripts/optimize-assets.mjs; run from frontend.
const publicDir = resolve('public');
await mkdir(publicDir,{recursive:true});
const source = resolve(publicDir,'betting-oracle-logo.png');
for(const height of [60,120]) {
  const target = resolve(publicDir,'betting-oracle-logo-'+height+'.webp');
  await sharp(source).resize({height,withoutEnlargement:true}).webp({lossless:true}).toFile(target);
  console.log(JSON.stringify({file:target,bytes:(await stat(target)).size}));
}
```

## code/deployment/package.json

[Separate source file](code/deployment/package.json)

```json
{
  "name": "f1-analysis-react",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "postinstall": "node scripts/patch-glide.mjs",
    "dev": "vite",
    "build": "vite build && node scripts/check-budgets.mjs",
    "preview": "vite preview",
    "lint": "eslint . --max-warnings=0",
    "typecheck": "tsc --noEmit",
    "test": "vitest run --coverage",
    "test:watch": "vitest",
    "audit": "npm audit --omit=dev",
    "audit:dev": "npm audit",
    "capture:react": "node ../parity_evidence/capture_react.mjs",
    "capture:streamlit": "node ../parity_evidence/capture_streamlit.mjs",
    "capture:diff": "node ../parity_evidence/diff_screenshots.mjs",
    "audit:a11y": "node ../parity_evidence/audit_accessibility.mjs",
    "benchmark": "node ../parity_evidence/benchmark.mjs",
    "prebuild": "node scripts/optimize-assets.mjs"
  },
  "dependencies": {
    "@glideapps/glide-data-grid": "^6.0.3",
    "lodash": "^4.18.1",
    "marked": "^4.3.0",
    "papaparse": "^5.4.1",
    "plotly.js-dist-min": "^4.1.1",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "react-markdown": "^10.1.0",
    "react-responsive-carousel": "^3.2.23",
    "recharts": "^2.15.0",
    "vega": "^6.4.0",
    "vega-embed": "^7.3.0",
    "vega-lite": "^6.4.3"
  },
  "devDependencies": {
    "@eslint/js": "^9.13.0",
    "@testing-library/dom": "^10.4.2",
    "@testing-library/jest-dom": "^6.6.3",
    "@testing-library/react": "^16.1.0",
    "@testing-library/user-event": "^14.5.2",
    "@types/papaparse": "^5.3.15",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^4.3.4",
    "@vitest/coverage-v8": "^2.1.8",
    "axe-core": "^4.13.0",
    "eslint": "^9.13.0",
    "eslint-plugin-jsx-a11y": "^6.10.2",
    "eslint-plugin-react": "^7.37.2",
    "eslint-plugin-react-hooks": "^5.0.0",
    "globals": "^15.11.0",
    "jsdom": "^25.0.1",
    "playwright": "^1.49.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "sharp": "^0.33.5",
    "typescript": "^5.7.2",
    "vite": "^6.0.0",
    "vitest": "^2.1.8"
  }
}
```

## code/deployment/vite.config.js

[Separate source file](code/deployment/vite.config.js)

```javascript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    sourcemap: false,
    // Per PARITY_CHECKLIST §14: production main chunk must be < 500 KB gzipped.
    // This setting warns; scripts/check-budgets.mjs enforces the gzip budget.
    chunkSizeWarningLimit: 500,
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.js'],
    css: false,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      include: ['src/**/*.{js,jsx}'],
      exclude: ['src/test/**', 'src/main.jsx', '**/*.test.{js,jsx}'],
      // Thresholds are intentionally below the §14 80% target: page-level
      // tests for App.jsx, the full Betting Research workflow, and
      // interactive Data Explorer filter combinations are tracked as
      // follow-up work in PARITY_REPORT.md. The infrastructure (vitest,
      // coverage, the api mock pattern, and 30+ component tests) is in
      // place; only the additional tests are deferred.
      thresholds: {
        lines: 60,
        functions: 40,
        branches: 60,
        statements: 60,
      },
    },
  },
});
```
