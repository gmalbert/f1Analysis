import {chromium} from 'playwright';
import {createServer,request} from 'node:http';
import {readFile,writeFile,stat} from 'node:fs/promises';
import {dirname,resolve,sep,extname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {gzipSync,gunzipSync} from 'node:zlib';
import Papa from '../../frontend/node_modules/papaparse/papaparse.js';

const here=dirname(fileURLToPath(import.meta.url)),dist=resolve(here,'../../frontend/dist');
const mime={'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png','.woff2':'font/woff2'};
const cache=new Map();
let lastViewResponse;
const server=createServer(async(req,res)=>{
  try{
    const url=new URL(req.url,'http://localhost');
    if(url.pathname.startsWith('/api/')){
      const proxy=request({host:'127.0.0.1',port:8000,path:req.url,method:req.method,headers:{...req.headers,host:'127.0.0.1:8000'}},upstream=>{
        if(url.pathname==='/api/views'){
          const chunks=[];upstream.on('data',chunk=>chunks.push(chunk));
          upstream.on('end',()=>{lastViewResponse={body:Buffer.concat(chunks),encoding:upstream.headers['content-encoding']};});
        }
        res.writeHead(upstream.statusCode,upstream.headers);upstream.pipe(res);
      });
      proxy.on('error',e=>{res.writeHead(502);res.end(e.message);});req.pipe(proxy);return;
    }
    let path=resolve(dist,'.'+decodeURIComponent(url.pathname));
    if(path!==dist&&!path.startsWith(dist+sep)){res.writeHead(403);res.end();return;}
    if(path===dist||!(await stat(path).catch(()=>null))?.isFile())path=resolve(dist,'index.html');
    if(!cache.has(path)){
      const raw=await readFile(path),zipped=/\.(js|css|html)$/.test(path)&&/gzip/.test(req.headers['accept-encoding']||'');
      cache.set(path,{body:zipped?gzipSync(raw,{level:6}):raw,zipped});
    }
    const {body,zipped}=cache.get(path);res.writeHead(200,{'content-type':mime[extname(path)]||'application/octet-stream','content-length':body.length,...(zipped?{'content-encoding':'gzip'}:{})});res.end(body);
  }catch(e){res.writeHead(500);res.end(e.message);}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const base=`http://127.0.0.1:${server.address().port}`;
const browser=await chromium.launch(),samples=[],errors=[];
async function ready(page){
  await page.waitForTimeout(100);
  await page.locator('main[aria-busy=false]').waitFor({timeout:60000});
  await page.evaluate(async()=>{await document.fonts.ready;await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));});
}
function tables(nodes){return nodes.flatMap(n=>[...(n.type==='table'?[n]:[]),...tables(n.children||[])]);}
try{
  for(let run=0;run<=3;run++){
    const context=await browser.newContext({viewport:{width:1280,height:900}}),page=await context.newPage();
    const cdp=await context.newCDPSession(page);await cdp.send('Network.enable',{maxTotalBufferSize:128*1048576,maxResourceBufferSize:64*1048576});
    const requests=new Map();
    cdp.on('Network.requestWillBeSent',e=>requests.set(e.requestId,{url:e.request.url,start:e.timestamp}));
    cdp.on('Network.responseReceived',e=>{const r=requests.get(e.requestId);if(r)Object.assign(r,{ttfb_ms:(e.timestamp-r.start)*1000,status:e.response.status,headers:e.response.headers});});
    cdp.on('Network.loadingFinished',e=>{const r=requests.get(e.requestId);if(r)Object.assign(r,{duration_ms:(e.timestamp-r.start)*1000,encoded_bytes:e.encodedDataLength});});
    page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
    await page.goto(base,{waitUntil:'domcontentloaded'});await ready(page);
    await page.getByRole('checkbox',{name:'Filter Results',exact:true}).check();await ready(page);
    await page.getByRole('slider',{name:/Year minimum/}).focus();await page.keyboard.press('ArrowRight');await ready(page);
    await page.getByRole('tab',{name:/Data & Debug/}).first().click();await ready(page);
    requests.clear();const rawResponse=page.waitForResponse(r=>r.url().endsWith('/api/views')&&r.request().method()==='POST');
    const started=performance.now();await page.getByRole('checkbox',{name:'Show Raw Data',exact:true}).check();await ready(page);
    const readyMs=performance.now()-started,response=await rawResponse;
    if(response.status()!==200)throw new Error('Raw request failed');
    // Inspect the byte-preserving proxy copy: Chromium evicts large response
    // bodies from Playwright's inspector cache even after the page reads them.
    const payload=JSON.parse((lastViewResponse.encoding==='gzip'?gunzipSync(lastViewResponse.body):lastViewResponse.body).toString('utf8'));
    const table=tables(payload.nodes)[0];
    if(!table||table.rows.length!==4629)throw new Error('Raw dataset is incomplete');
    await page.locator('.canvas-table canvas').first().waitFor();
    const rawRequest=[...requests.values()].find(r=>r.url.endsWith('/api/views'));
    const sample={run,ready_ms:readyMs,ttfb_ms:rawRequest.ttfb_ms,transfer_bytes:rawRequest.encoded_bytes,rows:table.rows.length,columns:table.columns.length,encoding:rawRequest.headers['content-encoding']};
    if(run){samples.push(sample);console.log(JSON.stringify(sample));}
    if(run===1){
      const grid=page.locator('.canvas-table').first();await grid.scrollIntoViewIfNeeded();await grid.hover();
      await grid.getByRole('button',{name:'Search table',exact:true}).click();await grid.locator('input').first().fill('Verstappen');await page.keyboard.press('Escape');
      await grid.getByRole('button',{name:'Show or hide columns'}).click();await grid.locator('.column-picker input').first().uncheck();await grid.locator('.column-picker input').first().check();await grid.getByRole('button',{name:'Show or hide columns'}).click();
      const downloaded=page.waitForEvent('download');await grid.getByRole('button',{name:'Download table as CSV'}).click();
      const path=await(await downloaded).path();const csv=Papa.parse(await readFile(path,'utf8'),{skipEmptyLines:true});
      if(csv.errors.length)throw new Error('CSV parsing failed');
      if(csv.data.length!==table.rows.length+1||JSON.stringify(csv.data[0])!==JSON.stringify(table.columns.map(c=>c.key)))throw new Error('CSV rows or fields changed');
      sample.csv={rows:csv.data.length-1,columns:csv.data[0].length,headerMatches:true};
    }
    await context.close();
  }
  if(errors.length)throw new Error(JSON.stringify(errors));
}catch(e){errors.push(e.stack||e.message);throw e;}finally{
  await writeFile(resolve(here,'raw-optimization-browser.json'),JSON.stringify({generated_at:new Date().toISOString(),viewport:'1280x900',mode:'Local production preview, gzip static assets, same API, same year filter as original workflow',samples,errors},null,2)+'\n');
  await browser.close();await new Promise(r=>server.close(r));
}
