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
