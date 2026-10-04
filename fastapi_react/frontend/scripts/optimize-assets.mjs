import sharp from 'sharp';
import {mkdir, stat} from 'node:fs/promises';
import {fileURLToPath, URL} from 'node:url';
import console from 'node:console';

// Preserve the source mark and its transparency, with 1x/2x display-height variants.
const publicDir = new URL('../public/', import.meta.url);
await mkdir(publicDir, {recursive: true});
const source = new URL('betting-oracle-logo.png', publicDir);
for (const height of [60, 120]) {
  const target = new URL(`betting-oracle-logo-${height}.webp`, publicDir);
  await sharp(fileURLToPath(source)).resize({height, withoutEnlargement: true})
    .webp({lossless: true}).toFile(fileURLToPath(target));
  console.info(JSON.stringify({asset: fileURLToPath(target), bytes: (await stat(target)).size}));
}
