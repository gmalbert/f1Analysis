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
