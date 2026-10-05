/** Real local authorization and browser workflow; job fixtures never run training. */
import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import http from 'node:http';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(resolve(here, '../../frontend/package.json'));
const {chromium} = require('playwright');
const base = process.env.REACT_BASE_URL || 'http://127.0.0.1:5174';
const api = process.env.F1_API_BASE_URL || 'http://127.0.0.1:8000';
const checks = [], errors = [], expectedEvents = [];
const images = resolve(here, 'screenshots');
await mkdir(images, {recursive: true});
let browser;
async function check(id, run) {checks.push({id, evidence: await run()});console.info('PASS ' + id);}
async function ready(page) {await page.locator('main[aria-busy="false"]').waitFor({timeout: 120000});}
try {
  await check('Live local access without credentials', async () => {
    const access = await fetch(base + '/api/enhancements/research-access');
    assert.equal(access.status, 200);
    assert.deepEqual(await access.json(), {mode: 'local', token_required: false});
    const diagnostics = await fetch(base + '/api/enhancements/metrics');
    assert.equal(diagnostics.status, 200);
    for (const headers of [{Origin: 'https://example.invalid'}, {Host: 'example.invalid:8000'}, {'X-Forwarded-For': '127.0.0.1'}, {'Sec-Fetch-Site': 'cross-site'}]) {
      // Node fetch normalizes Host; use raw HTTP to exercise the actual header.
      // An unsupported task also prevents computation if an access regression occurs.
      const payload = JSON.stringify({task: 'access-probe'});
      const status = await new Promise((resolveStatus, reject) => {
        const request = http.request(api + '/api/enhancements/jobs', {method: 'POST', headers: {'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload), ...headers}}, response => {
          response.resume();response.on('end', () => resolveStatus(response.statusCode));
        });
        request.on('error', reject);request.setTimeout(15000, () => request.destroy(new Error('Access probe timed out')));request.end(payload);
      });
      assert.equal(status, 403, 'Access probe: ' + JSON.stringify(headers));
    }
    const preflight = await fetch(api + '/api/enhancements/jobs', {method: 'OPTIONS', headers: {Origin: 'https://example.invalid', 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'content-type'}});
    assert.equal(preflight.status, 400);
    assert.equal(preflight.headers.get('access-control-allow-origin'), null);
    return {mode: 'local', diagnostics: 200, blocked_probes: 4, cross_site_preflight: 400, real_training_executed: false};
  });
  browser = await chromium.launch({headless: true});
  await check('Local form, original action, queue, cancel, results and continued browsing', async () => {
    const context = await browser.newContext({viewport: {width: 1280, height: 900}});
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
    page.on('requestfailed', request => {
      if (request.failure()?.errorText === 'net::ERR_ABORTED' && /\/api\/(views|enhancements\/(status|research-access|jobs))/.test(request.url())) expectedEvents.push('Obsolete request cancelled');
      else errors.push(request.failure()?.errorText);
    });
    let posts = 0, state = 'queued', polls = 0;
    const submitted = [];
    await context.route('**/api/enhancements/jobs**', async route => {
      const request = route.request(), path = new URL(request.url()).pathname;
      assert.equal(request.headers()['x-f1-admin-token'], undefined);
      let body;
      if (request.method() === 'POST') {
        posts++;polls = 0;state = posts === 1 ? 'queued' : 'running';
        submitted.push(request.postDataJSON());body = {id: 'local-fixture-' + posts, state};
      } else if (request.method() === 'DELETE') {
        state = 'cancelled';body = {cancelled: true, job: {id: 'local-fixture-1', state}};
      } else if (path.endsWith('/result')) {
        body = {source_revision: 'local-fixture-r1', nodes: [{type: 'heading', level: 3, text: 'Local fixture completed'}, {type: 'button', label: 'Do not repeat'}]};
      } else {
        if (posts > 1 && ++polls >= 2) state = 'succeeded';
        body = {id: 'local-fixture-' + posts, state};
      }
      await route.fulfill({status: request.method() === 'POST' ? 202 : 200, contentType: 'application/json', body: JSON.stringify(body)});
    });
    let views = 0;page.on('request', request => {if (new URL(request.url()).pathname === '/api/views') views++;});
    await page.goto(base + '/#/Raw%20Data');await ready(page);
    await page.getByRole('tab', {name: 'Temporal Leakage Audit', exact: true}).click();await ready(page);
    await page.locator('summary').filter({hasText: 'Run Temporal Leakage Audit'}).click();
    await page.getByText('Research tools are ready on this computer.', {exact: true}).waitFor({state: 'attached'});
    const before = views;
    await page.getByRole('button', {name: 'Run Leakage Audit', exact: true}).click();
    assert.equal(await page.getByLabel('Administrator token', {exact: true}).count(), 0);
    assert.equal(await page.getByRole('combobox', {name: /Research task/}).evaluate(el => el === document.activeElement), true);
    assert.equal(await page.getByRole('button', {name: 'Queue calculation', exact: true}).isEnabled(), true);
    assert.equal(views, before);assert.equal(posts, 0);
    await page.getByRole('button', {name: 'Queue calculation', exact: true}).click();
    await page.getByRole('button', {name: 'Cancel queued job', exact: true}).click();
    await page.getByText('Job local-fixture-1: cancelled', {exact: true}).waitFor();
    await page.getByRole('button', {name: 'Queue calculation', exact: true}).click();
    await page.getByText('Job local-fixture-2: running', {exact: true}).waitFor();
    await page.getByRole('tab', {name: /Data Explorer/}).click();await ready(page);
    await page.getByRole('heading', {name: 'Local fixture completed', exact: true}).waitFor({timeout: 20000});
    assert.equal(await page.getByRole('button', {name: 'Do not repeat'}).count(), 0);
    assert.deepEqual(submitted, [1,2].map(() => ({task: 'leakage-audit', values: {'Rows to read (0 = all)': 1000}})));
    await page.getByRole('combobox', {name: /Research task/}).selectOption('bin-comparison');
    assert.equal(await page.getByRole('button', {name: 'Queue calculation', exact: true}).isEnabled(), true);
    assert.equal(await page.getByRole('checkbox', {name: '2', exact: true}).isChecked(), true);
    await page.locator('.research-job').screenshot({path: resolve(images, 'trusted-local-research.png')});
    await context.close();
    return {submitted, token_field: false, credential_header: false, completed_snapshot: true, continued_browsing: true, bin_form_ready: true, screenshot: 'screenshots/trusted-local-research.png'};
  });
} catch (error) {errors.push({message: error.message, stack: error.stack});}
finally {await browser?.close();}
const result = {generated_at: new Date().toISOString(), pass: errors.length === 0, checks, errors, expected_events: expectedEvents};
await writeFile(resolve(here, 'local-access-results.json'), JSON.stringify(result, null, 2) + '\n');
await writeFile(resolve(here, 'LOCAL_ACCESS_RESULTS.md'), '# Trusted local research acceptance\n\n' + result.generated_at + '\n\n' + checks.map(item => '- Passed: ' + item.id).join('\n') + '\n\nUnexpected browser or assertion errors: ' + errors.length + '\n\nJob lifecycle uses browser fixtures; server access checks and normal views use the running API. No real training was executed. Hosted authentication is covered by backend and hosted-form tests.\n\n![Local research form](screenshots/trusted-local-research.png)\n');
if (!result.pass) {console.error(errors);process.exitCode = 1;}
