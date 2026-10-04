import {chromium} from 'playwright';
import {createServer, request} from 'node:http';
import {readFile, writeFile, readdir, stat} from 'node:fs/promises';
import {resolve, dirname, extname, sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {gzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';

const here=dirname(fileURLToPath(import.meta.url));
const root=resolve(here,'../../..'),dist=resolve(root,'fastapi_react/frontend/dist');
const output=resolve(here,'measurements.json');
const resumed=process.env.BENCH_RESUME==='1'?JSON.parse(await readFile(output,'utf8')):null;
const result=resumed||{generated_at:new Date().toISOString(),samples:[],memory_samples:[],errors:[],method:{runs:3,viewport:'1280x900',browser:'Chromium headless',cold:'Fresh browser context; warmed application servers',warm:'Reload in the same browser context',production:'Local static production build with gzip level 6 and unchanged FastAPI proxy',wire:'HTTP encoded transfer including response headers plus WebSocket payload bytes; WebSocket framing/TLS excluded',ready:'Finished application script/request, visible controls, local requests settled, fonts ready, two animation frames',collect_ms:500}};
result.method.readiness_floor_ms=100;
result.method.streamlit_websocket_compression='No extension negotiated; direct handshake verified';
result.method.embedded_urls='Inline data URLs retain their media type, character count and SHA-256 instead of repeated payloads. Timing and byte measurements are unchanged.';
const evidenceValue=(_key,value)=>typeof value==='string' && value.startsWith('data:') && !value.slice(value.indexOf(',')+1).startsWith('[payload omitted; ')
  ? value.slice(0,value.indexOf(',')+1)+`[payload omitted; ${value.length} URL characters; SHA-256 ${createHash('sha256').update(value).digest('hex')}]` : value;
const save=()=>writeFile(output,JSON.stringify(result,evidenceValue,2)+'\n');
const runFile=promisify(execFile);
let memoryBusy=false;
async function memory(phase){
  if(memoryBusy)return;
  memoryBusy=true;
  try {const {stdout}=await runFile(resolve(root,'.venv/Scripts/python.exe'),[resolve(here,'server_memory.py')]);result.memory_samples.push({time:new Date().toISOString(),phase,...JSON.parse(stdout)});} catch(e){result.errors.push({phase:'memory',message:e.message});} finally {memoryBusy=false;}
}

const types={'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml','.png':'image/png','.woff2':'font/woff2'};
const staticCache=new Map();
const production=createServer(async(req,res)=>{
  try {
    const url=new URL(req.url,'http://localhost');
    if(url.pathname.startsWith('/api/')){
      const proxy=request({host:'127.0.0.1',port:8000,path:req.url,method:req.method,headers:{...req.headers,host:'127.0.0.1:8000'}},upstream=>{res.writeHead(upstream.statusCode,upstream.headers);upstream.pipe(res);});
      proxy.on('error',e=>{res.writeHead(502);res.end(e.message);});req.pipe(proxy);return;
    }
    let path=resolve(dist,'.'+decodeURIComponent(url.pathname));
    if(path!==dist&&!path.startsWith(dist+sep)){res.writeHead(403);res.end();return;}
    if(path===dist||!(await stat(path).catch(()=>null))?.isFile())path=resolve(dist,'index.html');
    const zipped=/gzip/.test(req.headers['accept-encoding']||'')&&/\.(js|css|html|json|svg)$/.test(path);
    const key=path+zipped;
    if(!staticCache.has(key)){
      const raw=await readFile(path),body=zipped?gzipSync(raw,{level:6}):raw;
      staticCache.set(key,{body,etag:'"'+createHash('sha256').update(body).digest('hex')+'"'});
    }
    const {body,etag}=staticCache.get(key);
    const headers={'content-type':types[extname(path)]||'application/octet-stream',etag,'cache-control':path.includes(sep+'assets'+sep)?'public, max-age=31536000, immutable':'no-cache','vary':'Accept-Encoding',...(zipped?{'content-encoding':'gzip'}:{})};
    if(req.headers['if-none-match']===etag){res.writeHead(304,headers);res.end();return;}
    res.writeHead(200,{...headers,'content-length':body.length});res.end(body);
  }catch(e){res.writeHead(500);res.end(e.message);}
});
await new Promise(resolve=>production.listen(0,'127.0.0.1',resolve));
const apps=[{id:'streamlit',url:'http://127.0.0.1:8502',kind:'streamlit'},{id:'react_dev',url:'http://127.0.0.1:5174',kind:'react'},{id:'react_production',url:`http://127.0.0.1:${production.address().port}`,kind:'react'}];
result.applications=apps;const browser=await chromium.launch();result.browser_version=browser.version();
const pause=ms=>new Promise(r=>setTimeout(r,ms));

async function session(app){
  const context=await browser.newContext({viewport:{width:1280,height:900}}),page=await context.newPage();
  const cdp=await context.newCDPSession(page);await cdp.send('Network.enable');await cdp.send('Performance.enable');
  const records=[],ws=[],pending=new Set(),byId=new Map();
  cdp.on('Network.requestWillBeSent',event=>{const item={id:event.requestId,url:event.request.url,method:event.request.method,start:event.timestamp,type:event.type,encoded:0,decoded:0,from_cache:false};records.push(item);byId.set(event.requestId,item);if(item.url.startsWith(app.url))pending.add(item.id);});
  cdp.on('Network.responseReceived',event=>{const item=byId.get(event.requestId);if(item)Object.assign(item,{status:event.response.status,mime:event.response.mimeType,from_cache:event.response.fromDiskCache||event.response.fromPrefetchCache||false,headers:event.response.headers,response_ms:(event.timestamp-item.start)*1000});});
  cdp.on('Network.requestServedFromCache',event=>{const item=byId.get(event.requestId);if(item)item.from_cache=true;});
  cdp.on('Network.dataReceived',event=>{const item=byId.get(event.requestId);if(item)item.decoded+=event.dataLength;});
  cdp.on('Network.loadingFinished',event=>{pending.delete(event.requestId);const item=byId.get(event.requestId);if(item){item.encoded=event.encodedDataLength;item.duration_ms=(event.timestamp-item.start)*1000;}});
  cdp.on('Network.loadingFailed',event=>{pending.delete(event.requestId);const item=byId.get(event.requestId);if(item)item.failure=event.errorText;});
  cdp.on('Network.webSocketFrameReceived',event=>ws.push({direction:'received',bytes:event.response.opcode===2?Buffer.from(event.response.payloadData,'base64').length:Buffer.byteLength(event.response.payloadData),time:event.timestamp}));
  cdp.on('Network.webSocketFrameSent',event=>ws.push({direction:'sent',bytes:event.response.opcode===2?Buffer.from(event.response.payloadData,'base64').length:Buffer.byteLength(event.response.payloadData),time:event.timestamp}));
  page.on('pageerror',e=>result.errors.push({app:app.id,type:'pageerror',message:e.message}));
  page.on('console',m=>{if(m.type()==='error')result.errors.push({app:app.id,type:'console',message:m.text().slice(0,300)});});
  await page.addInitScript(()=>{window.benchmarkLongTasks=[];new PerformanceObserver(list=>window.benchmarkLongTasks.push(...list.getEntries().map(e=>({start:e.startTime,duration:e.duration})))).observe({type:'longtask',buffered:true});window.benchmarkLCP=[];new PerformanceObserver(list=>window.benchmarkLCP.push(...list.getEntries().map(e=>({start:e.startTime,size:e.size})))).observe({type:'largest-contentful-paint',buffered:true});});
  async function ready(){
    await pause(100);
    if(app.kind==='streamlit'){
      await page.getByRole('tab',{name:/Data Explorer/}).first().waitFor({timeout:180000});
      await page.waitForFunction(()=>document.querySelector('[data-testid="stApp"]')?.getAttribute('data-test-script-state')==='notRunning'&&!document.querySelector('[data-stale="true"]'),null,{timeout:180000});
    }else await page.locator('main[aria-busy=false]').waitFor({timeout:180000});
    for(let n=0;n<1800&&pending.size;n++)await pause(50);
    if(pending.size)throw new Error('Local requests did not finish: '+pending.size);
    await page.evaluate(async()=>{await document.fonts.ready;await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));});
  }
  async function measure(label,action){
    const existing=resumed&&result.samples.find(row=>row.app===app.id&&row.label===label);
    if(existing){await action();await ready();await pause(500);console.log(`Restored ${app.id} ${label}`);return existing;}
    const before=await cdp.send('Performance.getMetrics'),startRecords=records.length,startWs=ws.length;
    const start=performance.now();await action();await ready();const readyMs=performance.now()-start;
    await pause(500);
    const metricMap=Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map(m=>[m.name,m.value]));
    const previous=Object.fromEntries(before.metrics.map(m=>[m.name,m.value]));
    const timing=await page.evaluate(()=>{const nav=performance.getEntriesByType('navigation')[0];return {fcp_ms:performance.getEntriesByName('first-contentful-paint')[0]?.startTime,lcp_ms:window.benchmarkLCP.at(-1)?.start,ttfb_ms:nav?.responseStart,dom_content_ms:nav?.domContentLoadedEventEnd,long_tasks:window.benchmarkLongTasks,resources:performance.getEntriesByType('resource').map(r=>({url:r.name,transfer_bytes:r.transferSize,encoded_body_bytes:r.encodedBodySize,decoded_body_bytes:r.decodedBodySize}))};});
    const requests=records.slice(startRecords).map(({id,start,headers,...r})=>({...r,encoding:headers?.['content-encoding']||headers?.['Content-Encoding']||null,cache_control:headers?.['cache-control']||headers?.['Cache-Control']||null}));
    const frames=ws.slice(startWs),local=requests.filter(r=>r.url.startsWith(app.url));
    const received=frames.filter(f=>f.direction==='received').reduce((a,f)=>a+f.bytes,0),sent=frames.filter(f=>f.direction==='sent').reduce((a,f)=>a+f.bytes,0);
    await cdp.send('HeapProfiler.collectGarbage');const heap=Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map(m=>[m.name,m.value])).JSHeapUsedSize;
    const row={app:app.id,label,ready_ms:readyMs,http_bytes:local.reduce((a,r)=>a+r.encoded,0),ws_received_bytes:received,ws_sent_bytes:sent,download_bytes:local.reduce((a,r)=>a+r.encoded,0)+received,decoded_http_bytes:local.reduce((a,r)=>a+r.decoded,0),http_requests:local.length,api_requests:local.filter(r=>r.url.includes('/api/views')).length,cache_hits:local.filter(r=>r.from_cache).length,retained_js_heap_bytes:heap,dom_nodes:metricMap.Nodes,task_ms:Math.max(0,(metricMap.TaskDuration-(previous.TaskDuration||0))*1000),script_ms:Math.max(0,(metricMap.ScriptDuration-(previous.ScriptDuration||0))*1000),layout_ms:Math.max(0,(metricMap.LayoutDuration-(previous.LayoutDuration||0))*1000),timing,requests,external_requests:requests.filter(r=>!r.url.startsWith(app.url))};
    result.samples.push(row);await save();console.log(`${app.id} ${label}: ${readyMs.toFixed(0)} ms; ${(row.download_bytes/1048576).toFixed(2)} MiB; ${row.api_requests} API requests`);return row;
  }
  return {context,page,ready,measure,close:()=>context.close()};
}

async function inventory(path){const files=[];async function visit(dir){for(const item of await readdir(dir,{withFileTypes:true})){const name=resolve(dir,item.name);if(item.isDirectory())await visit(name);else{const data=await readFile(name);files.push({path:name.slice(path.length+1).replaceAll('\\','/'),bytes:data.length,gzip_bytes:gzipSync(data,{level:6}).length});}}}await visit(path);return files;}
let memoryTimer;
try{
  result.disk_inventory??={streamlit_static:await inventory(resolve(root,'.venv/Lib/site-packages/streamlit/static')),react_dist:await inventory(dist)};
  await memory('before');memoryTimer=setInterval(()=>memory('during'),3000);
  for(const app of apps){
    if(!result.samples.some(row=>row.app===app.id&&row.label==='warmup')){console.log(`Warmup ${app.id}`);const warmup=await session(app);await warmup.measure('warmup',()=>warmup.page.goto(app.url,{waitUntil:'domcontentloaded',timeout:60000}));await warmup.close();}
    for(let run=1;run<=3;run++){
      if(resumed&&result.samples.some(row=>row.app===app.id&&row.label===`cached_reload_${run}`))continue;
      const s=await session(app);await s.measure(`fresh_${run}`,()=>s.page.goto(app.url,{waitUntil:'domcontentloaded',timeout:60000}));
      await s.measure(`cached_reload_${run}`,()=>s.page.reload({waitUntil:'domcontentloaded',timeout:60000}));await s.close();
    }
    for(let run=1;run<=3;run++){
      if(resumed&&result.samples.some(row=>row.app===app.id&&row.label===`return_analytics_${run}`))continue;
      const s=await session(app);await s.measure(`workflow_home_${run}`,()=>s.page.goto(app.url,{waitUntil:'domcontentloaded',timeout:60000}));
      await s.measure(`enable_filters_${run}`,()=>app.kind==='streamlit'?s.page.getByText('Filter Results',{exact:true}).click():s.page.getByRole('checkbox',{name:'Filter Results',exact:true}).check());
      await s.measure(`filter_year_${run}`,async()=>{
        const slider=app.kind==='streamlit'?s.page.locator('[data-testid="stSlider"]').filter({has:s.page.getByText('Year',{exact:true})}).getByRole('slider').first():s.page.getByRole('slider',{name:/Year minimum/}).first();
        await slider.scrollIntoViewIfNeeded();await slider.focus();await s.page.keyboard.press('ArrowRight');
      });
      for(const name of ['Analytics & Visualizations','Schedule','Next Race','Predictive Models','Data & Debug','Betting Research']){
        await s.measure(`navigate_${name}_${run}`,()=>s.page.getByRole('tab',{name:new RegExp(name),exact:false}).first().click());
        if(name==='Data & Debug')await s.measure(`show_raw_data_${run}`,()=>app.kind==='streamlit'?s.page.getByText('Show Raw Data',{exact:true}).click():s.page.getByRole('checkbox',{name:'Show Raw Data',exact:true}).check());
      }
      await s.measure(`return_analytics_${run}`,()=>s.page.getByRole('tab',{name:/Analytics & Visualizations/}).first().click());
      await s.close();
    }
    await memory('after_'+app.id);
  }
  for(const app of apps){
    for(let run=1;run<=3;run++){
      const sessions=await Promise.all([session(app),session(app)]);
      const started=performance.now();
      await Promise.all(sessions.map((s,i)=>s.measure(`concurrent_2_${run}_${i+1}`,()=>s.page.goto(app.url,{waitUntil:'domcontentloaded',timeout:60000}))));
      result.concurrent_totals??=[];result.concurrent_totals.push({app:app.id,run,total_ms:performance.now()-started});
      await Promise.all(sessions.map(s=>s.close()));
    }
    await memory('after_concurrent_'+app.id);
  }
}catch(e){result.errors.push({type:'benchmark',message:e.stack});console.error(e.stack);process.exitCode=1;}
finally{clearInterval(memoryTimer);while(memoryBusy)await pause(100);await memory('after');result.completed_at=new Date().toISOString();await save();await browser.close();await new Promise(r=>production.close(r));}
