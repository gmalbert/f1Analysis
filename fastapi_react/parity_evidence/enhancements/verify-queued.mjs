/** B4/B5/O1/O2 acceptance: real normal responses, job UI fixtures; never runs training. */
import assert from 'node:assert/strict';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import http from 'node:http';
import {checkBudgets} from '../../frontend/scripts/check-budgets.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const frontend = resolve(here, '../../frontend');
const require = createRequire(resolve(frontend, 'package.json'));
const {chromium} = require('playwright');
const base = process.env.REACT_BASE_URL || 'http://127.0.0.1:5174';
const api = process.env.F1_API_BASE_URL || 'http://127.0.0.1:8000';
const checks = [], errors = [], events = [];
const images = resolve(here, 'screenshots');
await mkdir(images, {recursive: true});
let browser;
async function check(id, run) {const evidence = await run();checks.push({id, evidence});console.info('PASS ' + id);}
function track(page) {
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
  page.on('requestfailed', request => {
    if (request.failure()?.errorText === 'net::ERR_ABORTED' && /\/api\/(views|enhancements\/(status|research-access|jobs))/.test(request.url())) events.push('Expected obsolete request cancellation');
    else errors.push(request.failure()?.errorText);
  });
}
async function ready(page) {await page.locator('main[aria-busy="false"]').waitFor({timeout: 120000});}
try {
  await check('B4 authorization and explicit synchronous-action guard', async () => {
    const response = await fetch(api + '/api/enhancements/jobs', {method: 'POST', headers: {'Content-Type': 'application/json', Origin: 'https://example.invalid'}, body: JSON.stringify({task: 'access-probe'})});
    assert.ok([403,503].includes(response.status));
    for (const action of ['Run Leakage Audit', 'Run Bin Count Comparison']) {
      const result = await fetch(api + '/api/views', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({page: 6, values: {}, action})});
      assert.equal(result.status, 409);
    }
    return {unauthorized: response.status, synchronous_actions: 409, real_training_executed: false};
  });
  await check('B5 early aggregate rejection with diagnostic headers', async () => {
    const response = await new Promise((resolveResponse, reject) => {
      const request = http.request(api + '/api/views', {method: 'POST', headers: {'Content-Length': 1024*1024+1, 'Content-Type': 'application/json'}}, result => {
        const chunks = [];result.on('data', chunk => chunks.push(chunk));
        result.on('end', () => resolveResponse({status: result.statusCode, headers: result.headers, body: Buffer.concat(chunks).toString()}));
      });
      request.on('error', reject);request.setTimeout(15000, () => request.destroy(new Error('Size probe timed out')));request.end();
    });
    assert.equal(response.status, 413);assert.match(response.headers['x-request-id'], /^[a-f0-9]{32}$/);
    assert.match(response.headers['server-timing'], /backend;dur=/);
    return {...response, actual_upload_bytes: 0};
  });
  await check('O2 enforced production entry budget and no source maps', async () => checkBudgets(resolve(frontend, 'dist')));
  browser = await chromium.launch({headless: true});
  await check('B4 browser submit, cancel, completion, output and token privacy', async () => {
    const context = await browser.newContext({viewport: {width: 1280, height: 900}});
    const page = await context.newPage();track(page);
    // Exercise the hosted form even when acceptance runs against the local launcher.
    await context.route('**/api/enhancements/research-access', route => route.fulfill({status: 200, contentType: 'application/json', body: JSON.stringify({mode: 'token', token_required: true})}));
    let posts = 0, jobState = 'queued', pollCount = 0;
    const submitted = [];
    await context.route('**/api/enhancements/jobs**', async route => {
      const request = route.request(), path = new URL(request.url()).pathname;
      assert.equal(request.headers()['x-f1-admin-token'], 'fixture-memory-only');
      let body;
      if (request.method() === 'POST') {
        posts++;pollCount = 0;jobState = posts === 1 ? 'queued' : 'running';
        submitted.push(request.postDataJSON());body = {id: 'fixture-' + posts, state: jobState};
      } else if (request.method() === 'DELETE') {jobState = 'cancelled';body = {cancelled: true, job: {id: 'fixture-1', state: jobState}};}
      else if (path.endsWith('/result')) body = {source_revision: 'fixture-r1', nodes: [{type: 'tabs', children: [{type: 'tab', hidden: false, children: [{type: 'heading', level: 3, text: 'Fixture audit completed'}, {type: 'button', label: 'Do not repeat'}]}]}]};
      else {if (posts > 1 && ++pollCount >= 2) jobState = 'succeeded';body = {id: 'fixture-' + posts, state: jobState, revision: 'fixture-r1'};}
      await route.fulfill({status: request.method() === 'POST' ? 202 : 200, contentType: 'application/json', body: JSON.stringify(body)});
    });
    let views = 0;page.on('request', request => {if (new URL(request.url()).pathname === '/api/views') views++;});
    await page.goto(base + '/#/Raw%20Data');await ready(page);
    assert.equal(posts, 0);
    await page.getByRole('tab', {name: 'Temporal Leakage Audit', exact: true}).click();await ready(page);
    const audit = page.locator('summary').filter({hasText: 'Run Temporal Leakage Audit'});
    await audit.waitFor({timeout: 120000});await audit.click();
    const before = views;
    await page.getByRole('button', {name: 'Run Leakage Audit', exact: true}).click();
    await page.getByLabel('Administrator token', {exact: true}).waitFor();
    assert.equal(await page.getByLabel('Administrator token', {exact: true}).evaluate(el => el === document.activeElement), true);
    assert.equal(views, before);assert.equal(posts, 0);
    await page.getByLabel('Administrator token', {exact: true}).fill('fixture-memory-only');
    await page.getByRole('button', {name: 'Queue calculation', exact: true}).click();
    await page.getByRole('button', {name: 'Cancel queued job', exact: true}).click();
    await page.getByText('Job fixture-1: cancelled', {exact: true}).waitFor();
    await page.getByRole('button', {name: 'Queue calculation', exact: true}).click();
    await page.getByText('Job fixture-2: running', {exact: true}).waitFor();
    assert.equal(await page.getByRole('button', {name: 'Cancel queued job', exact: true}).count(), 0);
    // The job remains available when switching sections; it does not hold analysis busy.
    await page.getByRole('tab', {name: /Data Explorer/, exact: false}).click();await ready(page);
    await page.getByRole('heading', {name: 'Fixture audit completed'}).waitFor({timeout: 20000});
    assert.equal(await page.getByRole('button', {name: 'Do not repeat'}).count(), 0);
    assert.deepEqual(submitted, [1,2].map(() => ({task: 'leakage-audit', values: {'Rows to read (0 = all)': 1000}})));
    const storage = await page.evaluate(() => JSON.stringify({local: {...localStorage}, session: {...sessionStorage}, hash: location.hash}));
    assert.ok(!storage.includes('fixture-memory-only'));
    await page.locator('.research-job').screenshot({path: resolve(images, 'research-jobs.png')});
    await context.close();
    return {submitted, token_in_storage: false, continued_browsing: true, screenshots: ['screenshots/research-jobs.png']};
  });
  await check('O1 actual 1x and 2x footer assets at the original display size', async () => {
    const measurements = [];
    for (const density of [1,2]) {
      const context = await browser.newContext({viewport: {width: 390, height: 844}, deviceScaleFactor: density});
      const page = await context.newPage();track(page);
      const requests = [];
      page.on('request', request => {if (request.url().includes('betting-oracle-logo')) requests.push(request.url());});
      await page.goto(base);await ready(page);
      const img = page.getByRole('img', {name: 'Betting Oracle Logo', exact: true});
      await img.scrollIntoViewIfNeeded();
      await page.waitForFunction(() => {const image = document.querySelector('.parity-footer img');return image?.complete && image.naturalWidth > 0;});
      const measured = await img.evaluate(image => ({source: image.currentSrc, height: image.getBoundingClientRect().height, width: image.getBoundingClientRect().width, loading: image.loading}));
      assert.equal(measured.height, 60);assert.equal(measured.loading, 'lazy');
      assert.ok(Math.abs(measured.width - 60*822/1255) < 0.02);
      assert.ok(measured.source.endsWith(`-${density === 1 ? 60 : 120}.webp`));
      assert.ok(requests.every(url => !url.endsWith('.png')));
      const bytes = (await readFile(resolve(frontend, `public/betting-oracle-logo-${density === 1 ? 60 : 120}.webp`))).length;
      measurements.push({density, bytes, ...measured});
      if (density === 1) await page.locator('.parity-footer').screenshot({path: resolve(images, 'responsive-footer.png')});
      await context.close();
    }
    return {original_png_bytes: (await readFile(resolve(frontend, 'public/betting-oracle-logo.png'))).length, measurements};
  });
  if (process.env.F1_PRODUCTION_BASE_URL) await check('O2 real production browser without runtime errors', async () => {
    const context = await browser.newContext();const page = await context.newPage();track(page);
    await page.goto(process.env.F1_PRODUCTION_BASE_URL);await ready(page);
    await page.getByRole('tab', {name: /Schedule/}).click();await ready(page);
    await page.keyboard.press('Control+k');assert.equal(await page.locator('dialog:modal').isVisible(), true);
    await page.keyboard.press('Escape');await context.close();return {base: process.env.F1_PRODUCTION_BASE_URL};
  });
} catch (error) {errors.push({message: error.message, stack: error.stack});}
finally {await browser?.close();}
const result = {generated_at: new Date().toISOString(), pass: errors.length === 0, checks, errors, expected_events: events};
await writeFile(resolve(here, 'queued-results.json'), JSON.stringify(result, null, 2) + '\n');
await writeFile(resolve(here, 'QUEUED_RESULTS.md'), '# Additional enhancement acceptance\n\n' + result.generated_at + '\n\n' + checks.map(item => '- Passed: ' + item.id).join('\n') + '\n\nUnexpected errors: ' + errors.length + '\n\n![Research jobs](screenshots/research-jobs.png)\n\n![Responsive footer](screenshots/responsive-footer.png)\n');
if (!result.pass) {console.error(errors);process.exitCode = 1;}
