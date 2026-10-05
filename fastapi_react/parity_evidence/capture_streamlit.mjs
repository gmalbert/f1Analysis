// Capture the selected reference tab after every rerun has settled.
import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
const here = dirname(fileURLToPath(import.meta.url));
const out = join(process.env.PARITY_SCREENSHOT_DIR || join(here, 'visual'), 'streamlit');
const views = [{name:'desktop',width:1280,height:800},{name:'tablet',width:768,height:1024},{name:'mobile',width:390,height:844}];
const sections = ['home','data-explorer','analytics','current-season','next-race','models','raw-data','betting-research'];
const labels = [/Data Explorer/,/Data Explorer/,/Analytics & Visualizations/,/Schedule/,/Next Race/,/Predictive Models/,/Data & Debug/,/Betting Research/];
const base = process.env.STREAMLIT_BASE_URL || 'http://127.0.0.1:8502';
async function settle(page) {
  await page.waitForTimeout(250);
  await page.waitForFunction(() => document.querySelector('[data-testid="stApp"]')?.getAttribute('data-test-script-state')==='notRunning' && !document.querySelector('[data-stale="true"]'), null, {timeout:180000});
  await page.waitForTimeout(2000);
}
const browser = await chromium.launch();
try {
  await mkdir(out,{recursive:true});
  for (const view of views) {
    const context = await browser.newContext({viewport:{width:view.width,height:view.height}});
    const page = await context.newPage();
    // Serve the same public footer image locally when its host is unavailable.
    await page.route('**/gmalbert/betting-oracle/main/data_files/logo.png', route=>route.fulfill({path:join(here,'../frontend/public/betting-oracle-logo.png'),contentType:'image/png'}));
    await page.goto(base,{waitUntil:'domcontentloaded',timeout:60000});
    await page.getByRole('checkbox',{name:'Filter Results',exact:true}).waitFor({timeout:120000});
    await settle(page);
    for (let i=0;i<sections.length;i++) {
      if (sections[i]==='analytics') {
        await page.getByRole('tab',{name:/Data Explorer/}).first().click();
        const toggle=page.getByRole('checkbox',{name:'Filter Results',exact:true});
        if(!await toggle.isChecked()) await page.getByText('Filter Results',{exact:true}).click();
        await page.waitForFunction(()=>document.querySelector('input[aria-label="Filter Results"]')?.checked, null, {timeout:30000});
        await settle(page);
        if(view.name==='mobile') await page.locator('[data-testid="stSidebarCollapseButton"]').click();
      }
      const tab = page.getByRole('tab',{name:labels[i]}).first();
      await tab.click();
      await settle(page);
      if (await tab.getAttribute('aria-selected')!=='true') {await tab.click(); await settle(page);}
      if (await tab.getAttribute('aria-selected')!=='true') throw Error(`Wrong reference tab: ${sections[i]}`);
      if (sections[i]==='raw-data' && view.name!=='mobile') {
        const toggle=page.getByRole('checkbox',{name:'Show Raw Data',exact:true});
        if(!await toggle.isChecked()) await page.getByText('Show Raw Data',{exact:true}).click();
        await settle(page);
      }
      await page.evaluate(()=>{window.scrollTo(0,0);document.querySelectorAll('*').forEach(e=>{if(e.scrollTop)e.scrollTop=0;});});
      await page.waitForTimeout(300);
      await page.evaluate(()=>{window.scrollTo(0,0);document.querySelectorAll('*').forEach(e=>{if(e.scrollTop)e.scrollTop=0;});});
      await page.evaluate(()=>document.fonts.ready);
      await page.screenshot({path:join(out,`${view.name}-${sections[i]}.png`)});
      console.log(`${view.name}-${sections[i]}`);
    }
    await context.close();
  }
} finally {await browser.close();}
