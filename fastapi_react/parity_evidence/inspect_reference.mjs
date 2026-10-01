import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
const out = new URL('./inspection/', import.meta.url);
await mkdir(out, {recursive:true});
const browser = await chromium.launch();
const page = await browser.newPage({viewport:{width:1280,height:800}});
await page.goto('http://127.0.0.1:8502');
await page.waitForTimeout(7000);
await writeFile(new URL('shell.json',out), JSON.stringify(await page.evaluate(()=>[...document.querySelectorAll('h1,h2,h3,[data-testid="stCaptionContainer"],[data-testid="stMainBlockContainer"],[data-testid="stTabs"], [data-testid="stCheckbox"], [data-testid="stImage"]')].map(e=>{const s=getComputedStyle(e),r=e.getBoundingClientRect();return {testid:e.getAttribute('data-testid'),text:e.innerText?.slice(0,120),rect:{x:r.x,y:r.y,width:r.width,height:r.height},style:{font:s.font,color:s.color,padding:s.padding,margin:s.margin,gap:s.gap}}})),null,2));
for(const [name,label] of [['analytics','Analytics & Visualizations'],['schedule','Schedule'],['next','Next Race'],['models','Predictive Models'],['raw','Data & Debug'],['betting','Betting Research']]) {
 await page.getByRole('tab',{name:new RegExp(label)}).first().click();
 await page.waitForTimeout(2000);
 const panel=page.getByRole('tabpanel').filter({visible:true}).first();
 await writeFile(new URL(name+'.txt',out),await panel.innerText());
 await page.screenshot({path:new URL(name+'.png',out).pathname.replace(/^\/C:/,'C:')});
 if(name==='models' || name==='betting' || name==='raw') {
  const tabs=await panel.getByRole('tab').all();
  for(let i=0;i<tabs.length;i++){const t=tabs[i];await t.click();await page.waitForTimeout(800);await writeFile(new URL(name+'-'+i+'.txt',out),await panel.innerText());}
 }
}
await page.getByRole('tab',{name:/Data Explorer/}).first().click();
await page.getByRole('checkbox',{name:'Filter Results',exact:true}).click({force:true});
await page.waitForTimeout(10000);
await writeFile(new URL('filtered.txt',out),await page.locator('body').innerText());
await writeFile(new URL('sidebar.json',out),JSON.stringify(await page.locator('[data-testid="stSidebar"]').evaluate(e=>{const r=e.getBoundingClientRect();return {rect:{x:r.x,y:r.y,width:r.width,height:r.height},html:e.innerHTML}}),null,2));
await page.screenshot({path:new URL('filtered.png',out).pathname.replace(/^\/C:/,'C:')});
await page.getByRole('tab',{name:/Analytics & Visualizations/}).first().click();
await writeFile(new URL('analytics-filtered.txt',out),await page.getByRole('tabpanel').filter({visible:true}).first().innerText());
await page.screenshot({path:new URL('analytics-filtered.png',out).pathname.replace(/^\/C:/,'C:')});
await browser.close();
