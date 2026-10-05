import {chromium} from 'playwright';
const b=await chromium.launch();
for(const port of [8502,5174]){
 const p=await b.newPage({viewport:{width:768,height:1024}});await p.goto(`http://127.0.0.1:${port}`);
 await p.getByRole('checkbox',{name:'Filter Results',exact:true}).waitFor();
 await p.getByText('Filter Results',{exact:true}).click();await p.waitForTimeout(1000);
 if(port===8502)await p.waitForFunction(()=>document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state')==='notRunning',null,{timeout:180000});
 else await p.locator('main[aria-busy=false]').waitFor({timeout:120000});
 await p.getByRole('tab',{name:/Analytics & Visualizations/}).first().click();await p.waitForTimeout(2000);
 console.log(port,JSON.stringify(await p.evaluate(()=>[...document.querySelectorAll('h2,h3,.stCaption p,.view-caption p,[data-testid=stWidgetLabel] p,.view-field,[data-testid=stSidebar] label,.filter-sidebar label,[data-testid=stVegaLiteChart],.view-chart')].filter(e=>e.getBoundingClientRect().height && e.getBoundingClientRect().top<1050).map(e=>{const s=getComputedStyle(e),r=e.getBoundingClientRect();return {text:e.innerText?.slice(0,80),tag:e.tagName,rect:[r.x,r.y,r.width,r.height],font:s.font,color:s.color,padding:s.padding,margin:s.margin,lineHeight:s.lineHeight,gap:s.gap}}))));
 await p.close();
}await b.close();
