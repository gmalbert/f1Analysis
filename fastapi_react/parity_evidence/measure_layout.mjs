import {chromium} from 'playwright';
const browser=await chromium.launch();
for(const width of [1280,768,390]){
 for(const port of [8502,5174]){
  const page=await browser.newPage({viewport:{width,height:1024}});
  await page.goto(`http://127.0.0.1:${port}`); await page.waitForTimeout(5000);
  console.log(width,port,JSON.stringify(await page.evaluate(()=>[...document.querySelectorAll('h1,h2,[data-testid="stCaptionContainer"],.view-caption,[data-testid="stMainBlockContainer"],.main-shell')].filter(e=>e.getBoundingClientRect().height).slice(0,9).map(e=>{const s=getComputedStyle(e),r=e.getBoundingClientRect();return{text:e.innerText.slice(0,90),rect:[r.x,r.y,r.width,r.height],font:s.font,padding:s.padding,gap:s.gap}}))));
  await page.close();
 }
}
await browser.close();
