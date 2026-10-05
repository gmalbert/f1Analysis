import {mkdtemp, mkdir, writeFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {randomBytes} from 'node:crypto';
import {expect, it} from 'vitest';
import {checkBudgets} from './check-budgets.mjs';

it('enforces the initial static dependency budget and rejects published source maps', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'f1-build-budget-'));
  try {
    await mkdir(join(directory, '.vite'));
    await mkdir(join(directory, 'assets'));
    await writeFile(join(directory, '.vite/manifest.json'), JSON.stringify({
      'index.html': {isEntry: true, file: 'assets/main.js', imports: ['shared']},
      shared: {file: 'assets/shared.js'},
      lazy: {file: 'assets/lazy.js'},
    }));
    await writeFile(join(directory, 'assets/main.js'), 'x'.repeat(100));
    await writeFile(join(directory, 'assets/shared.js'), randomBytes(1000));
    await writeFile(join(directory, 'assets/lazy.js'), randomBytes(1000));
    const report = await checkBudgets(directory);
    expect(report.all_javascript_gzip_bytes).toBeGreaterThan(report.initial_javascript_gzip_bytes);
    await expect(checkBudgets(directory, 500)).rejects.toThrow('budget is 500');
    await writeFile(join(directory, 'assets/shared.js'), randomBytes(510000));
    await expect(checkBudgets(directory)).rejects.toThrow('budget is 500000');
    await writeFile(join(directory, 'assets/shared.js'), 'x');
    await writeFile(join(directory, 'assets/main.js.map'), '{}');
    await expect(checkBudgets(directory)).rejects.toThrow('source maps');
  } finally {await rm(directory, {recursive: true, force: true});}
});
