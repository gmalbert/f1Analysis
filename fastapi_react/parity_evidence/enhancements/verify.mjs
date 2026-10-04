/**
 * Exercise the installed application, not the proposal preview.
 * Run from any directory: node fastapi_react/parity_evidence/enhancements/verify.mjs
 * Overrides: REACT_BASE_URL, F1_API_BASE_URL, F1_ADMIN_TOKEN (optional), HEADLESS=0.
 * No data/model files are changed and no research/training action is executed.
 */
import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(resolve(here, '../../frontend/package.json'));
const {chromium} = require('playwright');
const Papa = require('papaparse');
const base = process.env.REACT_BASE_URL || 'http://127.0.0.1:5174';
const apiBase = process.env.F1_API_BASE_URL || 'http://127.0.0.1:8000';
const imageDir = resolve(here, 'screenshots');
await mkdir(imageDir, {recursive: true});

const checks = [], errors = [], expectedEvents = [], screenshots = [], observations = {};
let phase = 'Startup', intentionalFailure = false, intentionalCancellation = false;
let browser;
function isView(url) {return new URL(url).pathname === '/api/views';}
function tracked(page) {
  page.on('pageerror', error => errors.push({type: 'pageerror', phase, message: error.message, stack: error.stack}));
  page.on('console', message => {
    if (message.type() !== 'error') return;
    const entry = {type: 'console', phase, message: message.text()};
    if (intentionalFailure && /503|Failed to load resource/.test(message.text())) expectedEvents.push(entry);
    else errors.push(entry);
  });
  page.on('response', response => {
    if (response.status() < 400) return;
    const entry = {type: 'http', phase, url: response.url(), status: response.status()};
    if (intentionalFailure && isView(response.url()) && response.status() === 503) expectedEvents.push(entry);
    else errors.push(entry);
  });
  page.on('requestfailed', request => {
    const entry = {type: 'requestfailed', phase, url: request.url(), message: request.failure()?.errorText};
    const abortableEndpoint = isView(request.url()) || ['/api/enhancements/status', '/api/enhancements/research-access'].includes(new URL(request.url()).pathname);
    if (abortableEndpoint && /ERR_ABORTED|cancelled/i.test(entry.message || '')) expectedEvents.push({...entry, reason: 'Obsolete analysis/provenance fetch canceled by its AbortController'});
    else if (intentionalCancellation && /ERR_ABORTED|cancelled/i.test(entry.message || '')) expectedEvents.push(entry);
    else errors.push(entry);
  });
}
async function check(id, name, operation) {
  phase = name;
  const started = Date.now();
  try {
    const evidence = await operation();
    checks.push({id, name, result: 'pass', elapsed_ms: Date.now() - started, evidence: evidence ?? null});
    console.log('PASS ' + id + ' — ' + name);
  } catch (error) {
    checks.push({id, name, result: 'fail', elapsed_ms: Date.now() - started, error: error.message});
    errors.push({type: 'assertion', phase, message: error.message, stack: error.stack});
    console.error('FAIL ' + id + ' — ' + error.message);
    throw error;
  }
}
async function ready(page) {
  await page.locator('main[aria-busy="false"]').waitFor({timeout: 180000});
  await page.waitForTimeout(150);
  await page.locator('main[aria-busy="false"]').waitFor({timeout: 180000});
  await page.evaluate(async () => {
    await document.fonts.ready;
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  });
}
async function shot(page, name, locator) {
  const path = resolve(imageDir, name + '.png');
  if (locator) await locator.screenshot({path});
  else await page.screenshot({path, fullPage: false});
  screenshots.push({name, path: 'screenshots/' + name + '.png', viewport: page.viewportSize()});
}
function tools(page) {return page.getByRole('region', {name: 'Analysis tools', exact: true});}
async function openTools(page) {
  const summary = page.locator('summary').filter({hasText: /^Analysis tools$/});
  if (await summary.count() && !await tools(page).isVisible()) await summary.click();
  await tools(page).waitFor();
}
async function section(page, name) {
  await page.getByRole('tab', {name}).first().click();
  await ready(page);
}
function allTables(nodes) {
  return (nodes || []).flatMap(node => [...(node.type === 'table' ? [node] : []), ...allTables(node.children)]);
}
async function view(page) {
  return page.evaluate(async () => {
    const routeNames = ['Data Explorer', 'Analytics', 'Current Season', 'Next Race', 'Predictive Models', 'Raw Data', 'Betting Research'];
    const route = decodeURIComponent(location.hash.replace('#/', '').split('?')[0]);
    const currentPage = routeNames.indexOf(route) + 1 || 1;
    const values = JSON.parse(sessionStorage.getItem('f1analysis.view-values') || '{}');
    const response = await fetch('/api/views', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({page: currentPage, values})});
    if (!response.ok) throw new Error('Verification data request failed: ' + response.status);
    return response.json();
  });
}
function contrast(foreground, background) {
  const luminance = color => {
    const channels = color.match(/[\d.]+/g)?.slice(0, 3).map(Number);
    assert.equal(channels?.length, 3, 'Expected an RGB computed color: ' + color);
    const linear = channels.map(value => {const srgb = value / 255; return srgb <= .04045 ? srgb / 12.92 : ((srgb + .055) / 1.055) ** 2.4;});
    return linear[0] * .2126 + linear[1] * .7152 + linear[2] * .0722;
  };
  const first = luminance(foreground), second = luminance(background);
  return (Math.max(first, second) + .05) / (Math.min(first, second) + .05);
}
async function colors(page) {
  return page.evaluate(() => {
    const caption = document.querySelector('.view-caption');
    const active = document.querySelector('.parity-nav [aria-selected="true"]');
    const background = getComputedStyle(document.body).backgroundColor;
    const notice = document.querySelector('.view-notice.info');
    return {
      background, caption: caption ? {color: getComputedStyle(caption).color, opacity: getComputedStyle(caption).opacity} : null,
      active: {color: getComputedStyle(active).color},
      notice: notice ? {color: getComputedStyle(notice).color, background: getComputedStyle(notice).backgroundColor} : null,
      logo_width: document.querySelector('.parity-header > img').getBoundingClientRect().width,
      padding_top: getComputedStyle(document.querySelector('.main-shell')).paddingTop,
      navigation_position: getComputedStyle(document.querySelector('.parity-nav')).position,
      heading_size: getComputedStyle(document.querySelector('.shell-title, .shell-copy h1')).fontSize,
    };
  });
}
async function fetchApi(path, init) {
  const response = await fetch(apiBase + path, {...init, signal: AbortSignal.timeout(180000)});
  const text = await response.text();
  let body;
  try {body = JSON.parse(text);} catch {body = text;}
  return {response, body};
}
async function postApi(payload, headers = {}) {
  return fetchApi('/api/views', {method: 'POST', headers: {'Content-Type': 'application/json', ...headers}, body: JSON.stringify(payload)});
}
const safeFixture = {
  filter_results_main: true,
  filter_unicode_note: 'Écurie 🏎️ — 東京',
  '_tabs:Raw Data': 0,
  'Model probability': .83,
  f1bet_field_upload: {name: 'never-persist.csv', content: 'private,body\nsecret,value\n'},
  admin_token: 'never-share-token',
};

try {
  await check('B1-B3', 'Real API response reuse, revision, timing, and cache exclusions', async () => {
    const state = await fetchApi('/api/enhancements/status');
    assert.equal(state.response.status, 200, 'Backend enhancement routes must be enabled');
    assert.match(state.body.revision, /^[a-f\d]{64}$/);
    assert.ok(state.body.dataset?.name);
    assert.ok(Array.isArray(state.body.models));
    const values = {filter_results_main: false, _acceptance_run: 'audit-' + Date.now()};
    const first = await postApi({page: 1, values});
    const repeat = await postApi({page: 1, values});
    assert.equal(first.response.status, 200);
    assert.equal(first.response.headers.get('x-f1-cache'), 'MISS', 'Main backend response cache must be enabled');
    assert.equal(repeat.response.headers.get('x-f1-cache'), 'HIT');
    assert.deepEqual(repeat.body, first.body);
    assert.equal(repeat.response.headers.get('x-f1-revision'), state.body.revision);
    assert.match(repeat.response.headers.get('server-timing'), /^backend;dur=\d+(?:\.\d+)?$/);
    assert.match(first.response.headers.get('x-request-id'), /^[a-f\d]{32}$/);
    assert.notEqual(first.response.headers.get('x-request-id'), repeat.response.headers.get('x-request-id'));
    assert.equal(repeat.response.headers.get('cache-control'), 'no-store');
    const changed = await postApi({page: 1, values: {...values, _acceptance_run: values._acceptance_run + '-changed'}});
    assert.equal(changed.response.headers.get('x-f1-cache'), 'MISS');
    const noGzip = await postApi({page: 1, values}, {'Accept-Encoding': 'gzip;q=0,*;q=1'});
    assert.equal(noGzip.response.headers.get('content-encoding'), null, 'gzip;q=0 must survive the complete middleware stack');
    const privateView = await postApi({page: 1, values: {...values, test_upload: {name: 'fixture.csv', content: 'x\n1\n'}}});
    assert.equal(privateView.response.headers.get('x-f1-cache'), 'BYPASS');
    const raw = await postApi({page: 6, values: {show_raw_data_debug: false}});
    assert.equal(raw.response.headers.get('x-f1-cache'), 'BYPASS');
    const denied = await fetchApi('/api/enhancements/metrics', {headers: {Origin: 'https://example.invalid'}});
    assert.ok([403, 503].includes(denied.response.status));
    assert.match(denied.response.headers.get('server-timing'), /^backend;dur=/);
    observations.api = {revision: state.body.revision, dataset: state.body.dataset, model_count: state.body.models.length,
      initial_cache: first.response.headers.get('x-f1-cache'), repeated_cache: repeat.response.headers.get('x-f1-cache'),
      timing: repeat.response.headers.get('server-timing'), metrics_unauthorized_status: denied.response.status};
    const researchAccess = await fetchApi('/api/enhancements/research-access');
    if (researchAccess.body.mode === 'local' || process.env.F1_ADMIN_TOKEN) {
      const metrics = await fetchApi('/api/enhancements/metrics', {headers: process.env.F1_ADMIN_TOKEN ? {'X-F1-Admin-Token': process.env.F1_ADMIN_TOKEN} : {}});
      assert.equal(metrics.response.status, 200);
      assert.ok(metrics.body.requests.length <= 500);
      assert.ok(metrics.body.requests.some(record => record.request_id === repeat.response.headers.get('x-request-id')));
      for (const record of metrics.body.requests) assert.deepEqual(Object.keys(record).sort(), ['body_bytes', 'duration_ms', 'header_ms', 'method', 'request_id', 'route', 'status'].sort());
      observations.api.metrics_records_checked = metrics.body.requests.length;
    }
    return observations.api;
  });

  browser = await chromium.launch({headless: process.env.HEADLESS !== '0'});
  const context = await browser.newContext({viewport: {width: 1280, height: 900}, permissions: ['clipboard-read', 'clipboard-write']});
  await context.addInitScript(() => {
    if (!localStorage.getItem('f1analysis.enhancement-options')) localStorage.setItem('f1analysis.enhancement-options', JSON.stringify({design: true, cache: true}));
  });
  const page = await context.newPage();
  tracked(page);
  phase = 'Initial main application load';
  await page.goto(base); await ready(page);

  await check('D1-D2', 'Light contrast, focus, compact brand/header, and sticky navigation', async () => {
    await openTools(page);
    const readability = tools(page).getByRole('checkbox', {name: 'Improve readability', exact: true});
    if (!await readability.isChecked()) await readability.check();
    await page.waitForFunction(() => document.documentElement.dataset.enhancements === 'on');
    const style = await colors(page);
    assert.ok(style.caption, 'The real shell caption must be present');
    assert.equal(style.caption.opacity, '1');
    assert.ok(contrast(style.caption.color, style.background) >= 4.5, 'Caption contrast must meet 4.5:1');
    assert.ok(contrast(style.active.color, style.background) >= 4.5, 'Active tab text contrast must meet 4.5:1');
    if (style.notice) assert.ok(contrast(style.notice.color, style.notice.background) >= 4.5, 'Notice text contrast must meet 4.5:1');
    assert.ok(style.logo_width <= 280);
    assert.equal(style.padding_top, '40px');
    assert.equal(style.navigation_position, 'sticky');
    await page.getByRole('tab', {name: /Data Explorer/}).first().focus();
    await page.keyboard.press('Tab');
    const focus = await page.evaluate(() => ({tag: document.activeElement.tagName, width: getComputedStyle(document.activeElement).outlineWidth, style: getComputedStyle(document.activeElement).outlineStyle}));
    assert.ok(parseFloat(focus.width) >= 3 && focus.style !== 'none', 'Keyboard focus must have a visible 3px outline');
    await page.locator('summary').filter({hasText: /^Analysis tools$/}).click();
    await shot(page, 'desktop-light');
    observations.light = {...style, caption_contrast: contrast(style.caption.color, style.background), active_contrast: contrast(style.active.color, style.background), focus};
    return observations.light;
  });

  await check('D3-F4', 'Semantic all-field search, selectable columns, paging, years, and fonts', async () => {
    await page.getByRole('checkbox', {name: 'Filter Results', exact: true}).check(); await ready(page);
    const tableData = allTables((await view(page)).nodes)[0];
    assert.ok(tableData?.rows.length > 50, 'The real filtered history must have enough rows to page');
    observations.table = {rows: tableData.rows.length, columns: tableData.columns.length};
    const region = page.getByRole('region', {name: 'Table display', exact: true}).first();
    await region.scrollIntoViewIfNeeded();
    const toolbar = region.locator('.table-toolbar').first();
    const controls = await toolbar.getByRole('button').evaluateAll(buttons => buttons.map(button => ({name: button.getAttribute('aria-label'), width: button.getBoundingClientRect().width, height: button.getBoundingClientRect().height, opacity: getComputedStyle(button.parentElement).opacity, pointer_events: getComputedStyle(button.parentElement).pointerEvents})));
    assert.ok(controls.length >= 4);
    assert.ok(controls.every(control => control.width >= 44 && control.height >= 44 && control.opacity === '1' && control.pointer_events === 'auto'), 'Table tools must remain visible and at least 44x44');
    await region.getByRole('button', {name: 'Accessible table', exact: true}).click();
    const semantic = region.locator('.accessible-table').first();
    const table = semantic.getByRole('table').first();
    await table.waitFor();
    assert.equal(await table.locator('tbody tr').count(), 50);
    assert.ok(await table.locator('thead th[scope="col"]').count() >= 8);
    await semantic.getByRole('button', {name: 'Next rows', exact: true}).click();
    assert.match(await table.getByRole('caption').innerText(), /showing rows 51[–-]100/);
    await semantic.getByRole('button', {name: 'Previous rows', exact: true}).click();
    const visibleText = tableData.rows.flatMap(row => row.slice(0, 8).map(value => String(value ?? 'None').toLowerCase()));
    const candidate = [...new Set(tableData.rows.flatMap(row => row.slice(8).filter(value => typeof value === 'string' && value.length >= 3 && value.length < 50)))]
      .find(value => !visibleText.some(text => text.includes(value.toLowerCase())));
    assert.ok(candidate, 'The history must contain an example outside the initially displayed columns');
    const matches = tableData.rows.filter(row => row.some(value => String(value ?? 'None').toLowerCase().includes(candidate.toLowerCase()))).length;
    await semantic.getByRole('textbox', {name: 'Search all fields', exact: true}).fill(candidate);
    assert.ok((await table.getByRole('caption').innerText()).startsWith(matches.toLocaleString() + ' matching rows'));
    await semantic.getByRole('textbox', {name: 'Search all fields', exact: true}).fill('');
    await semantic.locator('summary').filter({hasText: /^Choose fields/}).click();
    assert.equal(await semantic.locator('.columns-list input[type="checkbox"]').count(), tableData.columns.length);
    const yearIndex = tableData.columns.findIndex(column => /year/i.test(column.key + ' ' + column.label));
    assert.ok(yearIndex >= 0, 'Year formatting requires a real year field');
    await semantic.locator('.columns-list input[type="checkbox"]').nth(yearIndex).check();
    const visible = await semantic.locator('.columns-list input[type="checkbox"]').evaluateAll(inputs => inputs.flatMap((input, index) => input.checked ? [index] : []));
    const yearCellIndex = visible.indexOf(yearIndex) + (tableData.hide_index ? 0 : 1);
    const yearText = await table.locator('tbody tr').first().locator('td,th').nth(yearCellIndex).innerText();
    assert.match(yearText, /^\d{4}$/, 'Years must not have thousands commas');
    const fonts = await table.evaluate(table => ({header: getComputedStyle(table.querySelector('thead th')).fontFamily, cell: getComputedStyle(table.querySelector('tbody td')).fontFamily}));
    assert.equal(fonts.header, fonts.cell, 'Numbers and words must use the same font');
    await semantic.locator('summary').filter({hasText: /^Choose fields/}).click();
    await shot(page, 'accessible-table', region);
    observations.table = {...observations.table, search_value: candidate, search_matches: matches, toolbar: controls, year: yearText, fonts};
    return observations.table;
  });

  await check('F5', 'Four-driver maximum and mathematically correct descriptive comparison', async () => {
    const region = page.getByRole('region', {name: 'Table display', exact: true}).first();
    const node = allTables((await view(page)).nodes)[0];
    const driverIndex = node.columns.findIndex(column => ['resultsDriverName', 'driverName', 'Driver'].includes(column.key));
    assert.ok(driverIndex >= 0);
    const drivers = [...new Set(node.rows.map(row => row[driverIndex]).filter(Boolean))].sort();
    assert.ok(drivers.length >= 5);
    await region.getByRole('button', {name: 'Compare drivers', exact: true}).click();
    const comparison = region.locator('.accessible-table').filter({has: page.getByRole('heading', {name: 'Compare race results', exact: true})});
    for (const driver of drivers.slice(0, 4)) await comparison.getByRole('checkbox', {name: driver, exact: true}).check();
    assert.equal(await comparison.locator('input[type="checkbox"]:checked').count(), 4);
    assert.ok(await comparison.getByRole('checkbox', {name: drivers[4], exact: true}).isDisabled());
    const rows = await comparison.getByRole('table').locator('tbody tr').all();
    const fields = ['resultsStartingGridPositionNumber', 'resultsFinalPositionNumber', 'positionsGained', 'DNF'].map(key => ({key, index: node.columns.findIndex(column => column.key === key)})).filter(field => field.index >= 0);
    assert.equal(rows.length, 4);
    for (let index = 0; index < 4; index++) {
      const sample = node.rows.filter(row => row[driverIndex] === drivers[index]);
      const expected = [drivers[index], String(sample.length), ...fields.map(field => {
        const values = sample.map(row => row[field.index]);
        if (field.key === 'DNF') {
          const known = values.filter(value => [true, false, 1, 0, 'true', 'false', '1', '0'].includes(typeof value === 'string' ? value.trim().toLowerCase() : value));
          return known.length ? (100 * known.filter(value => value === true || value === 1 || ['true', '1'].includes(String(value).trim().toLowerCase())).length / known.length).toFixed(1) + '%' : 'No data';
        }
        const known = values.filter(value => typeof value === 'number' && Number.isFinite(value));
        return known.length ? (known.reduce((sum, value) => sum + value, 0) / known.length).toFixed(2) : 'No data';
      })];
      assert.deepEqual(await rows[index].locator('th,td').allTextContents(), expected);
    }
    await comparison.getByRole('checkbox', {name: drivers[0], exact: true}).uncheck();
    assert.ok(await comparison.getByRole('checkbox', {name: drivers[4], exact: true}).isEnabled());
    await shot(page, 'driver-comparison', comparison);
    await region.getByRole('button', {name: 'Compare drivers', exact: true}).click();
    return {drivers: drivers.slice(0, 4), available_drivers: drivers.length, fields: fields.map(field => field.key)};
  });

  await check('D3-F4-regression', 'Original canvas grid search, column visibility, and complete CSV export', async () => {
    const region = page.getByRole('region', {name: 'Table display', exact: true}).first();
    await region.getByRole('button', {name: 'Interactive grid', exact: true}).click();
    const grid = region.locator('.canvas-table').first();
    await grid.locator('canvas').first().waitFor();
    await grid.getByRole('button', {name: 'Search table', exact: true}).click();
    await grid.locator('input').first().fill('Verstappen');
    await page.keyboard.press('Escape');
    await grid.getByRole('button', {name: 'Show or hide columns', exact: true}).click();
    const firstColumn = grid.locator('.column-picker input').first();
    await firstColumn.uncheck(); assert.ok(!await firstColumn.isChecked()); await firstColumn.check();
    await grid.getByRole('button', {name: 'Show or hide columns', exact: true}).click();
    const pendingDownload = page.waitForEvent('download');
    await grid.getByRole('button', {name: 'Download table as CSV', exact: true}).click();
    const download = await pendingDownload;
    assert.match(download.suggestedFilename(), /^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}_export\.csv$/);
    const parsed = Papa.parse(await readFile(await download.path(), 'utf8'), {skipEmptyLines: true});
    assert.equal(parsed.errors.length, 0);
    const node = allTables((await view(page)).nodes)[0];
    const headers = node.columns.map(column => column.key);
    if (!node.hide_index) headers.unshift(node.index_name || '');
    assert.deepEqual(parsed.data[0], headers);
    assert.equal(parsed.data.length - 1, node.rows.length, 'CSV retains every row rather than the HTML page only');
    return {download: download.suggestedFilename(), exported_rows: parsed.data.length - 1, exported_columns: parsed.data[0].length};
  });

  await check('F1', 'Named views survive reload and restore both page and filter settings', async () => {
    await page.getByRole('combobox', {name: 'Constructor', exact: true}).selectOption({label: 'Ferrari'}); await ready(page);
    await openTools(page);
    const name = 'Ferrari saved — 東京 🏎️';
    await tools(page).getByRole('textbox', {name: 'View name', exact: true}).fill(name);
    await tools(page).getByRole('button', {name: 'Save view', exact: true}).click();
    await page.getByText('View saved on this device.', {exact: true}).waitFor();
    await page.reload(); await ready(page); await openTools(page);
    await tools(page).getByRole('combobox', {name: 'Saved views', exact: true}).selectOption({label: name});
    await page.getByRole('checkbox', {name: 'Filter Results', exact: true}).uncheck(); await ready(page);
    await section(page, /Schedule/);
    await openTools(page);
    await tools(page).getByRole('combobox', {name: 'Saved views', exact: true}).selectOption({label: name});
    await tools(page).getByRole('button', {name: 'Load view', exact: true}).click(); await ready(page);
    assert.match(page.url(), /#\/Data%20Explorer$/);
    assert.ok(await page.getByRole('checkbox', {name: 'Filter Results', exact: true}).isChecked());
    assert.equal(await page.getByRole('combobox', {name: 'Constructor', exact: true}).inputValue(), JSON.stringify('Ferrari'));
    await page.reload(); await ready(page);
    assert.equal(await page.getByRole('combobox', {name: 'Constructor', exact: true}).inputValue(), JSON.stringify('Ferrari'));
    await page.getByRole('checkbox', {name: 'Filter Results', exact: true}).uncheck(); await ready(page); await openTools(page);
    await tools(page).getByRole('textbox', {name: 'View name', exact: true}).fill('No filters');
    await tools(page).getByRole('button', {name: 'Save view', exact: true}).click();
    await page.getByRole('checkbox', {name: 'Filter Results', exact: true}).check(); await ready(page);
    await tools(page).getByRole('combobox', {name: 'Saved views', exact: true}).selectOption({label: 'No filters'});
    await tools(page).getByRole('button', {name: 'Load view', exact: true}).click(); await ready(page);
    await page.reload(); await ready(page);
    assert.ok(!await page.getByRole('checkbox', {name: 'Filter Results', exact: true}).isChecked(), 'Legacy applied-filter storage must not override the restored false setting');
    await openTools(page);
    await tools(page).getByRole('combobox', {name: 'Saved views', exact: true}).selectOption({label: 'No filters'});
    await tools(page).getByRole('button', {name: 'Delete view', exact: true}).click();
    assert.equal(await tools(page).getByRole('combobox', {name: 'Saved views', exact: true}).getByRole('option', {name: 'No filters', exact: true}).count(), 0);
    await shot(page, 'saved-views');
    return {unicode_name: name, reload: true, restore_true_and_false: true, delete: true};
  });

  await check('F2', 'Unicode sharing restores settings in a fresh browser and excludes private values', async () => {
    await page.evaluate(values => {sessionStorage.setItem('f1analysis.view-values', JSON.stringify(values));sessionStorage.removeItem('f1analysis.filters');location.hash = '/Data%20Explorer';}, safeFixture);
    await page.reload(); await ready(page); await openTools(page);
    await tools(page).getByRole('button', {name: 'Copy view link', exact: true}).click();
    await page.getByText('View link copied. Uploads and betting inputs are excluded.', {exact: true}).waitFor();
    const url = await page.evaluate(() => navigator.clipboard.readText());
    const token = new URLSearchParams(new URL(url).hash.split('?')[1]).get('view');
    const shared = JSON.parse(Buffer.from(token, 'base64url').toString('utf8'));
    assert.equal(shared.values.filter_unicode_note, safeFixture.filter_unicode_note);
    assert.equal(shared.values.filter_results_main, true);
    assert.ok(!('f1bet_field_upload' in shared.values));
    assert.ok(!('Model probability' in shared.values));
    assert.ok(!('admin_token' in shared.values));
    const fresh = await browser.newContext({viewport: {width: 1280, height: 900}});
    const sharedPage = await fresh.newPage(); tracked(sharedPage);
    let loadedValues;
    sharedPage.on('request', request => {if (isView(request.url())) loadedValues = request.postDataJSON()?.values;});
    await sharedPage.goto(url); await ready(sharedPage);
    assert.equal(loadedValues.filter_unicode_note, safeFixture.filter_unicode_note);
    assert.ok(await sharedPage.getByRole('checkbox', {name: 'Filter Results', exact: true}).isChecked());
    await sharedPage.reload(); await ready(sharedPage);
    assert.equal(loadedValues.filter_unicode_note, safeFixture.filter_unicode_note);
    await fresh.close();
    return {utf8_value: shared.values.filter_unicode_note, excluded: ['upload body', 'betting probability', 'admin token'], fresh_origin_storage: true, reload: true};
  });

  await check('F3', 'Native command palette opens with keyboard, filters, navigates with Enter, and closes with Escape', async () => {
    if (await tools(page).isVisible()) await page.locator('summary').filter({hasText: /^Analysis tools$/}).click();
    assert.ok(!await tools(page).isVisible(), 'Exercise the global shortcut with Analysis tools collapsed');
    await page.keyboard.press('Control+k');
    const dialog = page.getByRole('dialog'); await dialog.waitFor();
    assert.ok(await dialog.evaluate(dialog => dialog.open && dialog.matches(':modal')), 'Use a native modal dialog');
    const search = dialog.getByRole('textbox', {name: 'Search sections', exact: true});
    assert.ok(await search.evaluate(input => input === document.activeElement));
    await search.fill('Models');
    assert.equal(await dialog.getByRole('button', {name: 'Predictive Models', exact: true}).count(), 1);
    await shot(page, 'command-palette');
    await search.press('Enter'); await ready(page);
    assert.match(page.url(), /#\/Predictive%20Models$/);
    assert.ok(!await dialog.isVisible());
    await page.keyboard.press('Control+k'); await dialog.waitFor();
    await page.keyboard.press('Escape');
    await dialog.waitFor({state: 'hidden'});
    await page.keyboard.press('Meta+k'); await dialog.waitFor();
    await dialog.getByRole('button', {name: 'Close', exact: true}).click();
    return {native_modal: true, collapsed_tools: true, control_k: true, meta_k: true, enter_navigation: true, escape: true};
  });

  await check('F6', 'Context export records safe settings, UTC time, revision and model provenance', async () => {
    await openTools(page);
    await page.waitForFunction(() => document.querySelector('.provenance summary')?.textContent?.includes('Data revision'));
    const pendingDownload = page.waitForEvent('download');
    await tools(page).getByRole('button', {name: 'Download analysis context', exact: true}).click();
    const download = await pendingDownload;
    const exported = JSON.parse(await readFile(await download.path(), 'utf8'));
    assert.equal(exported.schema, 'f1-analysis-context-v1');
    assert.equal(exported.page, 'Predictive Models');
    assert.match(exported.exported_at, /Z$/);
    assert.ok(Number.isFinite(Date.parse(exported.exported_at)));
    assert.match(exported.provenance.revision, /^[a-f\d]{64}$/);
    assert.equal(exported.analysis_revision, exported.provenance.revision, 'Provenance must describe the displayed analysis revision');
    assert.ok(exported.provenance.dataset.name);
    assert.ok(exported.provenance.models.length > 0, 'Recorded model manifests must be included');
    assert.ok(exported.provenance.models.some(model => model.model_name || model.estimator));
    assert.ok(!('f1bet_field_upload' in exported.values) && !('Model probability' in exported.values));
    await writeFile(resolve(here, 'exported-context.json'), JSON.stringify(exported, null, 2) + '\n');
    await page.evaluate(() => {window.__auditPrintCalls = 0;window.print = () => window.__auditPrintCalls++;});
    await tools(page).getByRole('button', {name: 'Print current view', exact: true}).click();
    assert.equal(await page.evaluate(() => window.__auditPrintCalls), 1);
    return {filename: download.suggestedFilename(), revision: exported.provenance.revision, model_count: exported.provenance.models.length, print_invoked: true};
  });

  await check('B1-client', 'Repeated navigation checks revision and reuses the bounded client response', async () => {
    await openTools(page);
    await tools(page).getByRole('checkbox', {name: 'Reuse recent views', exact: true}).check();
    const requests = [];
    const record = request => {if (new URL(request.url()).pathname.startsWith('/api/')) requests.push({path: new URL(request.url()).pathname, page: isView(request.url()) ? request.postDataJSON()?.page : null});};
    page.on('request', record);
    try {
      await section(page, /Schedule/);
      await section(page, /Next Race/);
      const scheduleRequests = requests.filter(request => request.path === '/api/views' && request.page === 3).length;
      const statusRequests = requests.filter(request => request.path === '/api/enhancements/status').length;
      await section(page, /Schedule/);
      assert.equal(requests.filter(request => request.path === '/api/views' && request.page === 3).length, scheduleRequests, 'A client hit should avoid another view POST');
      assert.ok(requests.filter(request => request.path === '/api/enhancements/status').length > statusRequests, 'A client hit must still check source revision');
      return {view_posts_avoided: 1, revision_checked: true, observed_requests: requests};
    } finally {page.off('request', record);}
  });

  await check('D1-D2-dark', 'Dark contrast and theme controls remain usable', async () => {
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    await page.getByRole('checkbox', {name: 'Use light theme', exact: true}).uncheck();
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    const style = await colors(page);
    assert.equal(style.caption.opacity, '1');
    assert.ok(contrast(style.caption.color, style.background) >= 4.5);
    assert.ok(contrast(style.active.color, style.background) >= 4.5);
    if (style.notice) assert.ok(contrast(style.notice.color, style.notice.background) >= 4.5, 'Dark info notices must retain readable foreground');
    observations.dark = {...style, caption_contrast: contrast(style.caption.color, style.background), active_contrast: contrast(style.active.color, style.background)};
    await shot(page, 'desktop-dark');
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    await page.getByRole('checkbox', {name: 'Use light theme', exact: true}).check();
    await page.getByRole('button', {name: 'Settings', exact: true}).click();
    return observations.dark;
  });

  await check('D1-D2-mobile', 'Compact mobile layout, visible navigation, and sidebar controls', async () => {
    await page.setViewportSize({width: 390, height: 844});
    if (await page.getByRole('button', {name: 'Close sidebar', exact: true}).isVisible()) await page.getByRole('button', {name: 'Close sidebar', exact: true}).click();
    await section(page, /Data Explorer/);
    if (await tools(page).isVisible()) await page.locator('summary').filter({hasText: /^Analysis tools$/}).click();
    const style = await colors(page);
    assert.ok(style.logo_width <= 210);
    assert.equal(style.padding_top, '56px');
    assert.ok(parseFloat(style.heading_size) <= 30);
    const overflow = await page.evaluate(() => ({scroll: document.documentElement.scrollWidth, viewport: innerWidth}));
    assert.ok(overflow.scroll <= overflow.viewport + 1, 'The root layout must not force horizontal page scrolling');
    const nav = page.getByRole('tablist', {name: 'Analysis sections', exact: true});
    await nav.getByRole('tab', {name: /Data Explorer/}).focus();
    await page.keyboard.press('End'); await ready(page);
    assert.ok(await nav.getByRole('tab', {name: /Betting Research/}).getAttribute('aria-selected') === 'true');
    await page.keyboard.press('Home'); await ready(page);
    if (await page.getByRole('button', {name: 'Open sidebar', exact: true}).count()) {
      await page.getByRole('button', {name: 'Open sidebar', exact: true}).click();
      await page.getByRole('complementary', {name: 'Data filters', exact: true}).waitFor();
      await page.getByRole('button', {name: 'Close sidebar', exact: true}).click();
    }
    await shot(page, 'mobile-light');
    return {...style, overflow, keyboard_navigation: true};
  });

  await context.close();
  const fixtures = await browser.newContext({viewport: {width: 1280, height: 900}});
  await fixtures.addInitScript(() => localStorage.setItem('f1analysis.enhancement-options', JSON.stringify({design: true, cache: false})));
  const fixturePage = await fixtures.newPage(); tracked(fixturePage);
  const delayed = new Map();
  async function pendingFixture(pageNumber) {
    const deadline = Date.now() + 10000;
    while (!delayed.has(pageNumber) && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 25));
    assert.ok(delayed.has(pageNumber), 'The controlled request must have reached the delayed fixture');
  }
  let scenario = 'loading', abortedRequestObserved = false;
  fixturePage.on('requestfailed', request => {if (isView(request.url()) && /ERR_ABORTED|cancelled/i.test(request.failure()?.errorText || '')) abortedRequestObserved = true;});
  function fixture(page, text) {return {page, shell: [{type: 'heading', level: 1, text: 'Acceptance fixture'}], nodes: [{type: 'heading', level: 2, text}, {type: 'checkbox', key: 'fixture_refresh', label: 'Fixture refresh', value: false}], tabs: ['📊 Data Explorer', '📈 Analytics & Visualizations', '🏎️ Schedule', '🏁 Next Race', '🤖 Predictive Models', '💾 Data & Debug', '📐 Betting Research'], sidebar: []};}
  await fixturePage.route('**/api/views', async route => {
    const payload = route.request().postDataJSON();
    if (scenario === 'failure') {await route.fulfill({status: 503, contentType: 'application/json', body: JSON.stringify({detail: 'Acceptance fixture: source unavailable. Retry this analysis.'})});return;}
    if (scenario === 'loading' || scenario === 'refresh' || scenario === 'stale' && payload.page === 3) {
      await new Promise(resolve => delayed.set(payload.page, resolve));
      delayed.delete(payload.page);
    }
    try {await route.fulfill({status: 200, contentType: 'application/json', body: JSON.stringify(fixture(payload.page, payload.page === 3 ? 'STALE schedule fixture' : 'CURRENT fixture page ' + payload.page))});}
    catch (error) {if (!/closed|handled|abort|cancel|Invalid Interception/i.test(error.message)) throw error;}
  });

  await check('D4-loading', 'Loading feedback is visible, polite, outside the busy region, and clears on completion', async () => {
    const navigation = fixturePage.goto(base);
    const feedback = fixturePage.locator('.load-feedback');
    await feedback.waitFor();
    assert.equal(await feedback.getAttribute('role'), 'status');
    assert.equal(await feedback.getAttribute('aria-live'), 'polite');
    assert.ok(!await feedback.evaluate(element => Boolean(element.closest('main[aria-busy="true"]'))));
    await fixturePage.waitForTimeout(2250);
    assert.match(await feedback.innerText(), /\d+s elapsed/);
    assert.ok(await feedback.locator('[aria-hidden="true"]').count());
    await shot(fixturePage, 'loading');
    scenario = 'normal';
    for (const release of delayed.values()) release();
    await navigation; await ready(fixturePage);
    await feedback.waitFor({state: 'hidden'});
    return {aria_live: 'polite', outside_busy_main: true, elapsed_time_hidden_from_live_announcements: true};
  });

  await check('D4-update', 'Updating a control retains the existing analysis while its replacement is loading', async () => {
    scenario = 'refresh';
    await fixturePage.getByRole('checkbox', {name: 'Fixture refresh', exact: true}).check();
    await fixturePage.locator('main[aria-busy="true"]').waitFor();
    await pendingFixture(1);
    await fixturePage.getByRole('heading', {name: 'CURRENT fixture page 1', exact: true}).waitFor();
    assert.match(await fixturePage.locator('.load-feedback').innerText(), /Updating analysis; existing results remain visible/);
    scenario = 'normal';
    for (const release of delayed.values()) release();
    await ready(fixturePage);
    assert.ok(await fixturePage.getByRole('checkbox', {name: 'Fixture refresh', exact: true}).isChecked());
    return {existing_results_retained: true, replacement_completed: true};
  });

  await check('D4-stale', 'Navigating away aborts a stale request and late content cannot replace current results', async () => {
    await section(fixturePage, /Data Explorer/);
    scenario = 'stale'; intentionalCancellation = true;
    await fixturePage.getByRole('tab', {name: /Schedule/}).click();
    await fixturePage.locator('main[aria-busy="true"]').waitFor();
    await pendingFixture(3);
    await fixturePage.getByRole('tab', {name: /Next Race/}).click();
    await ready(fixturePage);
    await fixturePage.getByRole('heading', {name: 'CURRENT fixture page 4', exact: true}).waitFor();
    await fixturePage.waitForTimeout(300);
    assert.ok(abortedRequestObserved, 'The browser should abort the unused upstream view request');
    for (const release of delayed.values()) release();
    await fixturePage.waitForTimeout(200);
    assert.equal(await fixturePage.getByRole('heading', {name: 'STALE schedule fixture', exact: true}).count(), 0);
    scenario = 'normal'; intentionalCancellation = false;
    return {browser_abort_observed: abortedRequestObserved, stale_result_rejected: true};
  });

  await check('D4-failure', 'A failure has understandable text and Retry recovers', async () => {
    scenario = 'failure'; intentionalFailure = true;
    await section(fixturePage, /Schedule/);
    const alert = fixturePage.getByRole('alert').filter({hasText: 'Acceptance fixture: source unavailable.'});
    await alert.waitFor();
    assert.match(await alert.innerText(), /Retry this analysis/);
    await shot(fixturePage, 'failure-retry');
    scenario = 'normal';
    await alert.getByRole('button', {name: 'Retry', exact: true}).click(); await ready(fixturePage);
    await fixturePage.getByRole('heading', {name: 'STALE schedule fixture', exact: true}).waitFor();
    assert.equal(await fixturePage.getByRole('alert').count(), 0);
    intentionalFailure = false;
    return {expected_http_error: 503, error_text_preserved: true, retry_recovered: true};
  });
  await fixtures.close();
} catch (error) {
  if (!errors.some(entry => entry.message === error.message)) errors.push({type: 'fatal', phase, message: error.message, stack: error.stack});
  process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  const report = {generated_at: new Date().toISOString(), frontend: base, backend: apiBase,
    scope: 'Installed main application. Real data in normal flows; explicit controlled fixtures only for loading, cancellation, and failure.',
    pass: !errors.length, checks, errors, expected_events: expectedEvents, screenshots, observations,
    fixture_unit_requirements: ['B1 client request deduplication with multiple aborting subscribers, TTL/entry/byte bounds, revision invalidation', 'B1 server TTL/entry/byte bounds and actions invalidate', 'B2 changed temporary artifact invalidates source loaders/presentation and response caches', 'B3 diagnostics bound of 500 records and no query/body/token retention', 'D4 Plotly asynchronous error and unmount/resize cleanup']};
  await writeFile(resolve(here, 'results.json'), JSON.stringify(report, null, 2) + '\n');
  await writeFile(resolve(here, 'RESULTS.md'), '# Main-app enhancement acceptance\n\nGenerated: ' + report.generated_at + '\n\n' + (report.pass ? 'All automated acceptance flows passed.' : 'Acceptance remains incomplete. See the failure evidence below.') + '\n\n| Requirement | Flow | Result |\n| --- | --- | --- |\n' + checks.map(check => '| ' + check.id + ' | ' + check.name + ' | ' + check.result + ' |').join('\n') + '\n\n' + (errors.length ? '## Failures\n\n```json\n' + JSON.stringify(errors, null, 2) + '\n```\n\n' : '') + '## Screenshots\n\n' + screenshots.map(image => '- [' + image.name + '](' + image.path + ')').join('\n') + '\n\nThe explicit loading, cancellation and failure fixtures are listed separately from unexpected browser/network errors in [results.json](results.json). Cache invalidation and bounded retention also require the fixture/unit checks listed there; production artifacts are never modified by this script.\n');
}
if (errors.length) process.exitCode = 1;
