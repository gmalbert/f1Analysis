import {chromium} from 'playwright';
import {createServer,request} from 'node:http';
import {mkdir,readFile,stat,writeFile} from 'node:fs/promises';
import {dirname,extname,resolve,sep} from 'node:path';
import {fileURLToPath} from 'node:url';

const here = dirname(fileURLToPath(import.meta.url)), pack = resolve(here,'..');
const repo = resolve(pack,'../../..'), dist = resolve(repo,'fastapi_react/.runtime/enhancement-preview/frontend/dist');
const backendPort = Number(process.env.PROPOSAL_API_PORT || 9008);
const images = resolve(pack,'images');await mkdir(images,{recursive:true});
const mime = {'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png','.webp':'image/webp','.woff2':'font/woff2'};
function preview(directory) {return createServer(async(req,res) => {
  try {
    const url = new URL(req.url,'http://localhost');
    if(url.pathname.startsWith('/api/')) {
      const upstream = request({host:'127.0.0.1',port:backendPort,path:req.url,method:req.method,headers:{...req.headers,host:'127.0.0.1:'+backendPort}},
        response => {res.writeHead(response.statusCode,response.headers);response.pipe(res);});
      upstream.on('error',error => {res.writeHead(502);res.end(error.message);});req.pipe(upstream);return;
    }
    let path = resolve(directory,'.'+decodeURIComponent(url.pathname));
    if(path !== directory && !path.startsWith(directory+sep)){res.writeHead(403);res.end();return;}
    if(!(await stat(path).catch(() => null))?.isFile())path = resolve(directory,'index.html');
    const body = await readFile(path);
    res.writeHead(200,{'content-type':mime[extname(path)] || 'application/octet-stream'});res.end(body);
  }catch(error){res.writeHead(500);res.end(error.message);}
});}
const server = preview(dist), currentServer = preview(resolve(repo,'fastapi_react/frontend/dist'));
await new Promise(resolve => server.listen(0,'127.0.0.1',resolve));
await new Promise(resolve => currentServer.listen(0,'127.0.0.1',resolve));
const base = 'http://127.0.0.1:'+server.address().port;
const currentBase = 'http://127.0.0.1:'+currentServer.address().port;
const browser = await chromium.launch(), checks = [], errors = [];
function track(page) {
  page.on('pageerror',error => errors.push(error.message));
  page.on('console',message => {if(message.type() === 'error')errors.push(message.text());});
  page.on('response',response => {if(response.status() >= 400)errors.push(response.status()+' '+response.url());});
}
async function ready(page) {
  await page.waitForTimeout(150);
  await page.locator('main[aria-busy=false]').waitFor({timeout:120000});
  await page.evaluate(async() => {await document.fonts.ready;await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));});
}
async function screenshot(page,name) {await page.screenshot({path:resolve(images,name+'.png')});}
try {
  for(const viewport of [{name:'desktop',width:1280,height:900},{name:'mobile',width:390,height:844}]) {
    const current = await browser.newPage({viewport});track(current);
    await current.goto(currentBase);await ready(current);await screenshot(current,'current-'+viewport.name);await current.close();
    const page = await browser.newPage({viewport});track(page);
    await page.goto(base);await ready(page);
    await page.getByText('Analysis tools',{exact:true}).click();
    await page.getByRole('checkbox',{name:'Improve readability',exact:true}).check();
    await page.waitForFunction(() => getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() === '#b4232d');
    await page.getByText('Analysis tools',{exact:true}).click();await screenshot(page,'proposed-'+viewport.name);
    if(viewport.name === 'desktop') {
      await page.getByRole('checkbox',{name:'Filter Results',exact:true}).check();await ready(page);
      const region = page.getByRole('region',{name:'Table display',exact:true}).first();
      await region.scrollIntoViewIfNeeded();
      await region.getByRole('button',{name:'Accessible table',exact:true}).click();
      await region.getByRole('table').first().waitFor();
      await region.getByRole('button',{name:'Next rows',exact:true}).click();
      if(!await region.getByRole('caption').first().textContent().then(text => text.includes('51')))throw new Error('Accessible row paging failed.');
      await region.screenshot({path:resolve(images,'proposed-accessible-table.png')});
      checks.push('Accessible semantic table, all-field selector, row paging and formatting');
      await region.getByRole('button',{name:'Compare drivers',exact:true}).click();
      await region.getByRole('checkbox',{name:'Max Verstappen',exact:true}).check();
      await region.getByRole('checkbox',{name:'Lewis Hamilton',exact:true}).check();
      await region.getByRole('table').last().screenshot({path:resolve(images,'proposed-driver-comparison.png')});
      checks.push('Descriptive driver comparison on current filtered rows');
      await page.evaluate(() => window.scrollTo(0,0));
      await page.getByText('Analysis tools',{exact:true}).click();
      await page.getByRole('checkbox',{name:'Reuse recent views',exact:true}).check();await ready(page);
      await page.getByRole('textbox',{name:'View name',exact:true}).fill('Filtered history');
      await page.getByRole('button',{name:'Save view',exact:true}).click();
      await page.getByText('View saved on this device.',{exact:true}).waitFor();
      await screenshot(page,'proposed-analysis-tools');
      checks.push('Saved view and optional client cache controls');
      await page.getByRole('button',{name:'Find section (Ctrl/⌘ K)',exact:true}).click();
      await page.getByRole('dialog').getByRole('textbox').fill('Models');
      await screenshot(page,'proposed-command-palette');
      await page.getByRole('dialog').getByRole('button',{name:'Predictive Models',exact:true}).click();await ready(page);
      if(await page.getByRole('dialog').count() && await page.getByRole('dialog').isVisible())throw new Error('Command dialog did not close.');
      checks.push('Native modal search, keyboard-capable navigation and model loading');
      const evidence = page.waitForEvent('download');
      await page.getByRole('button',{name:'Download analysis context',exact:true}).click();
      const context = JSON.parse(await readFile(await(await evidence).path(),'utf8'));
      if(context.schema !== 'f1-analysis-context-v1' || !context.provenance.revision)throw new Error('Analysis context export is incomplete.');
      checks.push('Reproducible analysis context JSON with data/model provenance');
    }
    await page.close();
  }
  if(errors.length)throw new Error(JSON.stringify(errors));
}catch(error){errors.push(error.stack || error.message);process.exitCode=1;}
finally {
  await writeFile(resolve(pack,'validation-browser.json'),JSON.stringify({generated_at:new Date().toISOString(),checks,errors,preview_backend:'isolated port '+backendPort,baseline:'Existing production bundle in a temporary static preview',live_app_changed:false},null,2)+'\n');
  await browser.close();await new Promise(resolve => server.close(resolve));await new Promise(resolve => currentServer.close(resolve));
}
