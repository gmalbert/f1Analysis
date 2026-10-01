import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
const here=dirname(fileURLToPath(import.meta.url));
const out=join(process.env.PARITY_SCREENSHOT_DIR || join(here,'visual'),'react');
const views=[{name:'desktop',width:1280,height:800},{name:'tablet',width:768,height:1024},{name:'mobile',width:390,height:844}];
const sections=['home','data-explorer','analytics','current-season','next-race','models','raw-data','betting-research'];
const routes=['Data Explorer','Data Explorer','Analytics','Current Season','Next Race','Predictive Models','Raw Data','Betting Research'];
const labels=[/Data Explorer/,/Data Explorer/,/Analytics & Visualizations/,/Schedule/,/Next Race/,/Predictive Models/,/Data & Debug/,/Betting Research/];
const base=process.env.REACT_BASE_URL || 'http://127.0.0.1:5174';
const errors=[];
async function settle(page){await page.locator('main[aria-busy="false"]').waitFor({timeout:120000}); await page.waitForTimeout(1000); await page.locator('main[aria-busy="false"]').waitFor({timeout:120000});}
const browser=await chromium.launch();
try {
  await mkdir(out,{recursive:true});
  for(const view of views){
    const context=await browser.newContext({viewport:{width:view.width,height:view.height}});
    const page=await context.newPage();
    page.on('pageerror',e=>errors.push({view:view.name,type:'pageerror',message:e.message}));
    page.on('console',e=>{if(e.type()==='error')errors.push({view:view.name,type:'console',message:e.text()});});
    page.on('response',r=>{if(r.status()>=400)errors.push({view:view.name,type:'http',url:r.url(),status:r.status()});});
    await page.goto(base,{waitUntil:'domcontentloaded'}); await settle(page);
    for(let i=0;i<sections.length;i++){
      if(sections[i]==='analytics'){await page.getByRole('tab',{name:/Data Explorer/}).first().click(); await settle(page); await page.getByRole('checkbox',{name:'Filter Results',exact:true}).check(); await settle(page); if(view.name==='mobile')await page.getByRole('button',{name:'Close sidebar'}).click();}
      await page.getByRole('tab',{name:labels[i]}).first().click(); await settle(page);
      if(sections[i]==='raw-data' && view.name!=='mobile'){await page.getByRole('checkbox',{name:'Show Raw Data',exact:true}).check(); await settle(page);}
      await page.evaluate(()=>window.scrollTo(0,0));
      await page.evaluate(()=>document.fonts.ready);
      await page.screenshot({path:join(out,`${view.name}-${sections[i]}.png`)});
      console.log(`${view.name}-${sections[i]}`);
    }
    await context.close();
  }
  await writeFile(join(out,'browser-errors.json'),JSON.stringify(errors,null,2));
  if(errors.length)throw Error(`${errors.length} browser errors; see browser-errors.json`);
} finally {await browser.close();}
