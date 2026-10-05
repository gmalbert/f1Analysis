import {chromium} from 'playwright';
import {readFile,writeFile} from 'node:fs/promises';
import Papa from 'papaparse';
const browser=await chromium.launch();
const exports=[];
for(const port of [8502,5174]){
 const page=await browser.newPage({viewport:{width:1280,height:900}});
 // Exercise the reference's download fallback without opening an OS save dialog.
 await page.addInitScript(()=>{delete window.showSaveFilePicker;});
 await page.goto(`http://127.0.0.1:${port}`);
 if(port===8502)await page.waitForFunction(()=>document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state')==='notRunning',null,{timeout:180000});
 else await page.locator('main[aria-busy=false]').waitFor({timeout:180000});
 await page.getByText('Filter Results',{exact:true}).click();await page.waitForTimeout(1000);
 if(port===8502)await page.waitForFunction(()=>document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state')==='notRunning',null,{timeout:180000});
 else await page.locator('main[aria-busy=false]').waitFor({timeout:180000});
 if(process.env.DOWNLOAD_SECTION){await page.getByRole('tab',{name:new RegExp(process.env.DOWNLOAD_SECTION)}).first().click();await page.waitForTimeout(1000);if(port===5174)await page.locator('main[aria-busy=false]').waitFor({timeout:180000});}
 const grid=page.locator(port===8502?'[data-testid=stDataFrame]:visible':'.canvas-table').first();
 await grid.scrollIntoViewIfNeeded();await grid.hover();await page.waitForTimeout(500);
 console.log(port,await grid.locator('button').evaluateAll(els=>els.map(e=>({label:e.getAttribute('aria-label'),title:e.getAttribute('title'),text:e.innerText}))));
 const event=page.waitForEvent('download');await grid.getByRole('button',{name:port===8502?'Download as CSV':'Download table as CSV',exact:true}).click();
 const download=await event;
 const text=await readFile(await download.path(),'utf8');
 exports.push({port,filename:download.suggestedFilename(),csv:text,rows:Papa.parse(text,{skipEmptyLines:true}).data});
 console.log(port,exports.at(-1).rows[0].slice(0,8),exports.at(-1).rows[1].slice(0,8));
 await page.close();
}
await browser.close();
const reference=exports[0],react=exports[1];
const equal=JSON.stringify(reference.rows)===JSON.stringify(react.rows);
const mismatches=[];
for(let i=0;i<Math.max(reference.rows.length,react.rows.length)&&mismatches.length<10;i++)for(let j=0;j<Math.max(reference.rows[i]?.length||0,react.rows[i]?.length||0)&&mismatches.length<10;j++)if(reference.rows[i]?.[j]!==react.rows[i]?.[j])mismatches.push({row:i,column:j,key:reference.rows[0][j],reference:reference.rows[i]?.[j],react:react.rows[i]?.[j]});
const result={rows:react.rows.length-1,columns:react.rows[0].length,contents_match:equal,mismatches,reference_header:reference.rows[0].slice(0,8),react_header:react.rows[0].slice(0,8)};
await writeFile(new URL(process.env.DOWNLOAD_SECTION?'./download-next-race-parity.json':'./download-parity.json',import.meta.url),JSON.stringify(result,null,2));
if(!equal)process.exitCode=1;
