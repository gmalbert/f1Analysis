/** Real tire comparison values and chart selection; no research calculations. */
import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(resolve(here, '../../frontend/package.json'));
const {chromium} = require('playwright');
const base = process.env.REACT_BASE_URL || 'http://127.0.0.1:5174';
const checks = [], errors = [], expectedEvents = [];
const selected = ['George Russell', 'Kimi Antonelli', 'Esteban Ocon'];
await mkdir(resolve(here, 'screenshots'), {recursive: true});
const browser = await chromium.launch({headless: true});
async function ready(page) {
  await page.locator('main[aria-busy="false"]').waitFor({timeout: 120000});
  await page.getByRole('region', {name: 'Race tire strategy comparison', exact: true}).waitFor();
}
async function check(id, run) {checks.push({id, evidence: await run()});console.info('PASS ' + id);}
try {
  const context = await browser.newContext({viewport: {width: 1280, height: 1000}, permissions: ['clipboard-read', 'clipboard-write']});
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
  page.on('requestfailed', request => {
    if (request.failure()?.errorText === 'net::ERR_ABORTED' && /\/api\/(views|enhancements\/status)/.test(request.url())) expectedEvents.push('Obsolete request cancelled');
    else errors.push(request.failure()?.errorText);
  });
  await page.goto(base + '/#/Analytics');await ready(page);
  await page.getByRole('combobox', {name: 'Year', exact: true}).selectOption({label: '2025'});await ready(page);
  await page.getByRole('combobox', {name: 'Grand Prix', exact: true}).selectOption({label: 'Canadian Grand Prix'});await ready(page);
  const source = await page.evaluate(async () => {
    const values = JSON.parse(sessionStorage.getItem('f1analysis.view-values') || '{}');
    const response = await fetch('/api/views', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({page: 2, values})});
    if (!response.ok) throw new Error('Source view is unavailable.');
    return response.json();
  });
  const raceTable = source.nodes.find(node => node.type === 'table' && node.columns.some(column => column.key === 'Start Compound'));
  const sourceChart = source.nodes.find(node => node.type === 'vega' && node.spec.encoding?.y?.field === 'Degradation (s/lap)');
  const originalRecords = sourceChart.spec.datasets[sourceChart.spec.data.name];
  const race = page.getByRole('region', {name: 'Race tire strategy comparison', exact: true});
  async function chartSpec() {
    await race.getByRole('button', {name: 'Copy Vega-Lite spec', exact: true}).click();
    return JSON.parse(await page.evaluate(() => navigator.clipboard.readText()));
  }
  function chartRecords(spec) {return spec.data.values || spec.datasets[spec.data.name];}
  await check('Real metrics and the selected-driver chart use the same three drivers', async () => {
    await race.getByRole('button', {name: 'Compare drivers', exact: true}).click();
    const comparison = race.getByRole('region', {name: 'Scrollable driver comparison', exact: true});
    assert.equal(await comparison.getByRole('columnheader', {name: 'Sample rows', exact: true}).count(), 0);
    assert.equal(await comparison.getByRole('columnheader', {name: 'Records included', exact: true}).count(), 0);
    for (const driver of selected) await race.getByRole('checkbox', {name: driver, exact: true}).check();
    await race.getByText('Comparing 3 selected drivers: ' + selected.join(', ') + '.', {exact: true}).waitFor();
    assert.deepEqual(await comparison.locator('tbody th').allTextContents(), selected);
    const driverIndex = raceTable.columns.findIndex(column => column.key === 'Driver');
    const degradationIndex = raceTable.columns.findIndex(column => column.key === 'Avg Deg (s/lap)');
    const compoundsIndex = raceTable.columns.findIndex(column => column.key === 'Start Compound');
    for (const driver of selected) {
      const row = raceTable.rows.find(row => row[driverIndex] === driver);
      const displayed = comparison.getByRole('rowheader', {name: driver, exact: true}).locator('..');
      const expected = new Intl.NumberFormat('en-US', {maximumFractionDigits: 4}).format(row[degradationIndex]);
      assert.ok((await displayed.locator('td').allTextContents()).includes(expected));
      assert.ok((await displayed.locator('td').allTextContents()).includes(row[compoundsIndex]));
    }
    const spec = await chartSpec(), records = chartRecords(spec);
    assert.deepEqual(records.map(row => row.driver).sort(), [...selected].sort());
    assert.deepEqual(spec.encoding.x.sort, selected);
    for (const record of records) assert.equal(record['Degradation (s/lap)'], originalRecords.find(row => row.driver === record.driver)['Degradation (s/lap)']);
    await race.getByText(/Showing only 3 selected drivers:/).waitFor();
    assert.ok((await race.innerText()).includes('Canadian Grand Prix 2025'));
    await race.locator('canvas').last().waitFor();
    await race.screenshot({path: resolve(here, 'screenshots/tire-driver-comparison.png')});
    return {race: 'Canadian Grand Prix', year: 2025, selected, chart_drivers: records.map(row => row.driver), metric: 'Tire degradation (s/lap)', values_match_live_source: true, screenshot: 'screenshots/tire-driver-comparison.png'};
  });
  await check('Clear or close comparison restores the complete field', async () => {
    for (const driver of selected) await race.getByRole('checkbox', {name: driver, exact: true}).uncheck();
    assert.equal(chartRecords(await chartSpec()).length, originalRecords.length);
    await race.getByText('Showing all drivers. Canadian Grand Prix 2025.', {exact: true}).waitFor();
    await race.getByRole('checkbox', {name: selected[0], exact: true}).check();
    assert.equal(chartRecords(await chartSpec()).length, 1);
    await race.getByRole('button', {name: 'Compare drivers', exact: true}).click();
    assert.equal(chartRecords(await chartSpec()).length, originalRecords.length);
    return {original_drivers: originalRecords.length, clearing_restores_all: true, closing_restores_all: true};
  });
  await check('Seasonal tire comparison uses the existing race counts and year scope', async () => {
    const history = page.locator('details').filter({has: page.locator('summary', {hasText: 'Historical Tire Management by Driver (all races in selected year)'})});
    await history.locator('summary').click();
    await history.getByRole('button', {name: 'Compare drivers', exact: true}).click();
    await history.getByRole('checkbox', {name: selected[0], exact: true}).check();
    const comparison = history.getByRole('region', {name: 'Scrollable driver comparison', exact: true});
    const summary = source.nodes.find(node => node.type === 'expander' && node.label.startsWith('Historical Tire Management')).children.find(node => node.type === 'table');
    const driverIndex = summary.columns.findIndex(column => column.key === 'Driver'), racesIndex = summary.columns.findIndex(column => column.key === 'Races');
    const recordedRaces = summary.rows.find(row => row[driverIndex] === selected[0])[racesIndex];
    assert.equal(await comparison.getByRole('columnheader', {name: 'Sample rows', exact: true}).count(), 0);
    assert.equal(await comparison.getByRole('columnheader', {name: 'Races', exact: true}).count(), 1);
    assert.ok((await comparison.locator('tbody td').allTextContents()).includes(String(recordedRaces)));
    assert.ok((await history.innerText()).includes('Season tire summaries for 2025'));
    return {driver: selected[0], recorded_races: recordedRaces, scope: '2025 season', fake_summary_row_count: false};
  });
  await page.setViewportSize({width: 390, height: 844});
  await race.getByRole('button', {name: 'Compare drivers', exact: true}).click();
  for (const driver of selected) await race.getByRole('checkbox', {name: driver, exact: true}).check();
  const dimensions = await page.evaluate(() => ({viewport: innerWidth, width: document.documentElement.scrollWidth}));
  assert.ok(dimensions.width <= dimensions.viewport + 1, 'Mobile comparison must not overflow the page.');
  await race.screenshot({path: resolve(here, 'screenshots/tire-driver-comparison-mobile.png')});
  checks.push({id: 'Mobile comparison stays within the viewport', evidence: dimensions});
  await context.close();
} catch (error) {errors.push({message: error.message, stack: error.stack});}
finally {await browser.close();}
const result = {generated_at: new Date().toISOString(), pass: errors.length === 0, checks, errors, expected_events: expectedEvents};
await writeFile(resolve(here, 'driver-comparison-results.json'), JSON.stringify(result, null, 2) + '\n');
await writeFile(resolve(here, 'DRIVER_COMPARISON_RESULTS.md'), '# Driver comparison acceptance\n\n' + result.generated_at + '\n\n' + checks.map(check => '- Passed: ' + check.id).join('\n') + '\n\nUnexpected errors: ' + errors.length + '\n\nUses the live 2025 Canadian Grand Prix tire tables. Selected chart values are checked against the original unrounded source; table values retain their displayed rounding.\n\n![Selected driver comparison](screenshots/tire-driver-comparison.png)\n\n![Mobile comparison](screenshots/tire-driver-comparison-mobile.png)\n');
if (!result.pass) {console.error(errors);process.exitCode = 1;}
