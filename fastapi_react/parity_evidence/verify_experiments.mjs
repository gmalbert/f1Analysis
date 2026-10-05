import {chromium} from 'playwright';
import {writeFile} from 'node:fs/promises';
const browser=await chromium.launch();
const page=await browser.newPage({viewport:{width:1280,height:900}});
const errors=[],checks=[];
page.on('pageerror',e=>errors.push(e.message));
page.on('console',e=>{if(e.type()==='error')errors.push(e.text());});
page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`);});
async function ready(){await page.locator('main[aria-busy=false]').waitFor({timeout:180000});await page.waitForTimeout(500);await page.locator('main[aria-busy=false]').waitFor({timeout:180000});}
try{
 await page.goto('http://127.0.0.1:5174/#/Predictive%20Models');await ready();
 await page.getByRole('tab',{name:'🛠️ Debug & Experiments',exact:true}).click();await ready();
 await page.getByRole('button',{name:'Clear Select q values (number of bins)'}).click();await ready();
 const select=page.getByRole('combobox',{name:'Select q values (number of bins)'});
 await select.fill('2');await select.press('Enter');await ready();await select.press('Escape');
 await page.getByRole('button',{name:'Run Bin Count Comparison',exact:true}).click();await ready();
 await page.getByText('MAE for each bin count (q):',{exact:true}).waitFor();
 await page.locator('.canvas-table canvas').first().waitFor();
 checks.push('Explicit bin-count experiment returns results');
 // Source administrative audit accepts a row limit for a small verification.
 await page.getByRole('tab',{name:/Data & Debug/}).first().click();await ready();
 await page.getByRole('tab',{name:'Temporal Leakage Audit',exact:true}).click();await ready();
 await page.getByText('🔍 Run Temporal Leakage Audit (Admin)',{exact:true}).click();
 const input=page.getByRole('spinbutton',{name:'Rows to read (0 = all)'});
 await input.fill('100');await input.press('Enter');await ready();
 await page.getByRole('button',{name:'Run Leakage Audit',exact:true}).click();await ready();
 if(await page.getByRole('alert').count())throw new Error(await page.getByRole('alert').allTextContents());
 checks.push('Explicit temporal audit returns results');
}catch(e){errors.push(e.message);}
finally{await writeFile(new URL('./experiment-results.json',import.meta.url),JSON.stringify({checks,errors},null,2));await browser.close();}
if(errors.length)process.exitCode=1;
