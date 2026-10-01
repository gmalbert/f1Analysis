// Exercise actual React controls, charts, downloads and CSV workflows.
import {chromium} from 'playwright';
import {writeFile} from 'node:fs/promises';
const browser=await chromium.launch();
const errors=[],checks=[];
let phase='Initial load';
const context=await browser.newContext({viewport:{width:1280,height:900},permissions:['clipboard-read','clipboard-write']});
const page=await context.newPage();
page.on('pageerror',e=>errors.push({type:'pageerror',phase,message:e.message,stack:e.stack}));
page.on('console',e=>{if(e.type()==='error')errors.push({type:'console',message:e.text()});});
page.on('response',r=>{if(r.status()>=400)errors.push({type:'http',url:r.url(),status:r.status()});});
const base=process.env.REACT_BASE_URL || 'http://127.0.0.1:5174';
async function ready(){await page.locator('main[aria-busy=false]').waitFor({timeout:180000});await page.waitForTimeout(300);await page.locator('main[aria-busy=false]').waitFor({timeout:180000});}
async function tab(name){await page.getByRole('tab',{name}).first().click();await ready();}
async function check(name,fn){phase=name;await fn();checks.push({name,result:'pass'});console.log(name);}
try{
 await page.goto(base);await ready();
 await check('Enable filters, search, column visibility and CSV download',async()=>{
  await page.getByRole('checkbox',{name:'Filter Results',exact:true}).check();await ready();
  await page.locator('.canvas-table canvas').first().waitFor();
  const grid=page.locator('.canvas-table').first();await grid.scrollIntoViewIfNeeded();
  await grid.hover();
  await grid.getByRole('button',{name:'Search table',exact:true}).click();
  await grid.locator('input').first().fill('Verstappen');
  await page.keyboard.press('Escape');
  await grid.getByRole('button',{name:'Show or hide columns'}).click();
  await grid.locator('.column-picker input').first().uncheck();await grid.locator('.column-picker input').first().check();
  await grid.getByRole('button',{name:'Show or hide columns'}).click();
  const download=page.waitForEvent('download');await grid.getByRole('button',{name:'Download table as CSV'}).click();if(!/^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}_export.csv$/.test((await download).suggestedFilename()))throw Error('CSV filename changed');
 });
 await check('Boolean, category and date filters',async()=>{
  await page.getByRole('checkbox',{name:'DNF',exact:true}).check();await ready();
  await page.getByRole('combobox',{name:'Constructor',exact:true}).selectOption({label:'Ferrari'});await ready();
  await page.getByRole('checkbox',{name:'DNF',exact:true}).uncheck();await ready();
  await page.getByRole('combobox',{name:'Constructor',exact:true}).selectOption({label:' All'});await ready();
  const date=page.getByRole('slider',{name:/Race Date minimum/});await date.scrollIntoViewIfNeeded();await date.focus();await page.keyboard.press('ArrowRight');await ready();
 });
 await check('Analytics, tire selectors, chart data and PNG export',async()=>{
  await tab(/Analytics & Visualizations/);
  const chart=page.locator('.chart-shell').first();await chart.scrollIntoViewIfNeeded();await chart.locator('canvas').waitFor();
  await chart.hover();
  await chart.getByRole('button',{name:'Show data',exact:true}).click();await chart.locator('.canvas-table canvas').first().waitFor();
  await chart.getByRole('button',{name:'Show chart',exact:true}).click();
  const download=page.waitForEvent('download');await chart.getByRole('button',{name:'Download chart as PNG'}).click();await download;
  const year=page.getByRole('combobox',{name:'Year',exact:true});await year.selectOption({index:1});await ready();
  await page.getByRole('combobox',{name:'Grand Prix',exact:true}).selectOption({index:1});await ready();
 });
 await check('Schedule and Next Race panels',async()=>{await tab(/Schedule/);await page.locator('.canvas-table canvas').first().waitFor();await tab(/Next Race/);await page.getByRole('checkbox',{name:'Show Next Race',exact:true}).uncheck();await ready();await page.getByRole('checkbox',{name:'Show Next Race',exact:true}).check();await ready();});
 await check('Every model and all seven nested model panels',async()=>{
  await tab(/Predictive Models/);
  const select=page.getByRole('combobox',{name:'Select Model Type'});
  for(const model of ['XGBoost','LightGBM','CatBoost','Ensemble (XGBoost + LightGBM + CatBoost)','Position Group','Track-Weighted Ensemble']){await select.selectOption({label:model});await ready();}
  await select.selectOption({label:'XGBoost'});await ready();
  for(const name of [/Model Performance/,/Feature Analysis/,/Feature Selection/,/Position-Specific Analysis/,/Hyperparameters/,/Historical Validation/,/Debug & Experiments/])await tab(name);
  await page.getByRole('combobox',{name:'Select q values (number of bins)'}).focus();await page.keyboard.press('Escape');
 });
 await check('All Data & Debug panels and raw grid',async()=>{await tab(/💾 Data & Debug/);await page.getByRole('checkbox',{name:'Show Raw Data'}).check();await ready();await page.locator('.canvas-table canvas').first().waitFor();await tab('Temporal Leakage Audit');await tab('Hyperparameter Tuning');await tab('Raw Data');});
 await check('Calculator responds to odds and de-vig controls',async()=>{
  await tab(/Betting Research/);await page.getByRole('spinbutton',{name:'Model probability',exact:true}).fill('0.8');await page.getByRole('spinbutton',{name:'Model probability',exact:true}).press('Enter');await ready();
  await page.getByRole('combobox',{name:'De-vig method'}).selectOption({label:'power'});await ready();
 });
 await check('CSV upload, coherent simulation and downloads',async()=>{
  await tab('Field simulation');
  const download=page.waitForEvent('download');await page.getByRole('link',{name:'Download input template'}).click();await download;
  await page.getByLabel('Field CSV', {exact:true}).setInputFiles({name:'field.csv',mimeType:'text/csv',buffer:Buffer.from('driver_id,constructor_id,pace_score,dnf_probability,uncertainty,race_sensitivity\na,t,1,0.05,0.8,1\nb,t,2,0.06,0.8,1\nc,u,3,0.08,0.8,1\n')});await ready();
  await page.getByRole('slider',{name:'Simulations',exact:true}).focus();await page.keyboard.press('Home');await ready();
  await page.getByRole('button',{name:'Run coherent field simulation'}).click();await ready();await page.locator('.canvas-table canvas').first().waitFor();
 });
 await check('Paper replay with timestamped CSV',async()=>{
  await tab('Paper replay');
  const csv='event_id,selection_id,market,forecast_at,quote_at,event_start_at,probability,uncertainty,fair_market_probability,decimal_odds,outcome\ne,a,winner,2026-01-01T10:00:00Z,2026-01-01T10:01:00Z,2026-01-01T12:00:00Z,0.8,0.01,0.5,2,1\n';
  await page.getByLabel('Backtest ledger CSV',{exact:true}).setInputFiles({name:'ledger.csv',mimeType:'text/csv',buffer:Buffer.from(csv)});await ready();await page.getByRole('button',{name:'Run paper backtest'}).click();await ready();await page.getByRole('heading',{name:'Placed paper bets'}).waitFor();
 });
 await check('Calibration metrics, reliability table and chart',async()=>{
  await tab('Calibration');await page.getByLabel('Calibration CSV',{exact:true}).setInputFiles({name:'calibration.csv',mimeType:'text/csv',buffer:Buffer.from('probability,outcome\n0.1,0\n0.2,0\n0.3,1\n0.6,0\n0.7,1\n0.9,1\n')});await ready();await page.getByRole('heading',{name:'Adaptive reliability table'}).waitFor();await page.locator('.chart-shell canvas').first().waitFor();
 });
 await check('Theme selection, mobile navigation and sidebar',async()=>{await page.getByRole('button',{name:'Settings'}).click();await page.getByRole('checkbox',{name:'Use light theme'}).uncheck();await page.getByRole('checkbox',{name:'Use light theme'}).check();await page.getByRole('button',{name:'Settings'}).click();await page.setViewportSize({width:390,height:844});await page.getByRole('button',{name:'Close sidebar'}).click();await page.getByRole('button',{name:'Open sidebar'}).click();await page.getByRole('button',{name:'Close sidebar'}).click();await tab(/Data Explorer/);});
 if(await page.getByRole('alert').count())throw Error(await page.getByRole('alert').allTextContents());
}catch(error){errors.push({type:'verification',message:error.message});process.exitCode=1;}
finally{await writeFile(new URL('./interaction-results.json',import.meta.url),JSON.stringify({generated_at:new Date().toISOString(),checks,errors},null,2));await browser.close();}
if(errors.length)process.exitCode=1;
