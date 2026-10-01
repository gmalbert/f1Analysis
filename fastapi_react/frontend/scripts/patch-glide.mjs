// Glide 6.0.3 receives an out-of-bounds header hit when a resized/hidden grid
// changes beneath the pointer. Preserve its behavior while guarding that hit.
// Apply on every clean install so development and production use the same fix.
import {readFile,writeFile} from 'node:fs/promises';
import {URL} from 'node:url';
const old='header.hasMenu === true';
const fixed='header?.hasMenu === true';
for(const format of ['esm','cjs']) {
  const path=new URL(`../node_modules/@glideapps/glide-data-grid/dist/${format}/internal/data-grid/data-grid.js`,import.meta.url);
  const source=await readFile(path,'utf8');
  if(source.includes(fixed))continue;
  if(source.split(old).length!==2)throw new Error('Glide header patch needs review for the installed version.');
  await writeFile(path,source.replace(old,fixed));
}
