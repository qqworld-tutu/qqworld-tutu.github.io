// Visual and interaction QA for the ten article figures.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'),assert=require('node:assert/strict');
const manifest=JSON.parse(fs.readFileSync('source/figures/manifest.json','utf8'));
const base=process.env.BLOG_PREVIEW_URL||'http://127.0.0.1:4341';
const out='output/playwright/figures';fs.mkdirSync(out,{recursive:true});
(async()=>{
 const b=await chromium.launch({channel:'chrome',headless:true});
 const p=await b.newPage({viewport:{width:1280,height:1000}}),errors=[],report=[];
 p.on('pageerror',e=>errors.push(e.message));
 p.on('response',r=>{if(r.url().startsWith(base)&&r.status()>=400)errors.push(`${r.status()} ${r.url()}`)});
 for(const f of manifest){
  await p.setViewportSize({width:1280,height:1000});await p.goto(base+f.url);await p.evaluate(()=>document.fonts.ready);
  for(const img of await p.locator('img').all())await img.evaluate(e=>e.decode());
  const kind=await p.locator('body').getAttribute('data-kind');
  if(kind==='qkv'){
   for(let i=1;i<=4;i++){await p.locator(`[data-i="${i}"]`).click();assert.equal(await p.locator('.q-head:not(.dim)').count(),6);assert.equal(await p.locator('.k-head:not(.dim)').first().getAttribute('data-head'),String(Math.ceil(i/2)));}
   await p.locator('[data-i="1"]').click();
  }
  if(kind==='heads'){
   await p.locator('[data-i="4"]').click();assert.deepEqual(await p.locator('.kv-node:not(.dim-path)').evaluateAll(es=>es.map(e=>e.dataset.group)),['4','2','1']);await p.locator('[data-i="all"]').click();
  }
  if(kind==='moe'){
   await p.locator('[data-moe="b"]').click();assert.deepEqual(await p.locator('.selected-id').allTextContents(),['1','4']);assert.match(await p.locator('.moe-result').textContent(),/0.548 f₁\(x\) \+ 0.452 f₄\(x\)/);await p.locator('[data-moe="a"]').click();
  }
  if(kind==='cache'){
   await p.locator('#cache-next').click();await p.locator('#cache-next').click();assert.equal(await p.locator('#cache-next').isDisabled(),true);assert.match(await p.locator('#sharing').textContent(),/T = 6/);assert.equal(await p.locator('.stage').first().locator('.matrix[data-rows="6"]').count(),2);await p.locator('#cache-reset').click();assert.match(await p.locator('#sharing').textContent(),/T = 4/);
  }
  if(kind==='gradient'){
   await p.locator('[data-path="direct"]').click();assert.match(await p.locator('.probability-path').getAttribute('class'),/dim-path/);assert(!/dim-path/.test(await p.locator('.direct-path').getAttribute('class')));await p.locator('[data-path="all"]').click();
  }
  if(kind==='future'){
   assert.equal(await p.locator('.term-cell').count(),14);await p.locator('[data-row-select="2"]').click();assert.equal(await p.locator('.term-cell:not(.dim-path)').count(),4);await p.locator('[data-row-select="all"]').click();
  }
  await p.screenshot({path:`${out}/${f.topic}-${f.stem}-desktop.png`,fullPage:true});
  for(const width of [812,390,360]){
   await p.setViewportSize({width,height:900});await p.goto(base+f.url+'?embed=1');await p.evaluate(()=>document.fonts.ready);
   const sizes=await p.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,main:document.querySelector('main').getBoundingClientRect().height}));
   assert(sizes.scroll<=width,JSON.stringify({f:f.stem,...sizes}));
   if(width!==360)await p.screenshot({path:`${out}/${f.topic}-${f.stem}-${width}.png`,fullPage:true});
   report.push({figure:f.stem,...sizes});
  }
  // Full SVG fallback must remain usable outside the article.
  const response=await p.goto(base+f.svg);assert.equal(response.status(),200);assert.equal(await p.locator('parsererror').count(),0);
 }
 assert.deepEqual(errors,[]);fs.writeFileSync(out+'/checks.json',JSON.stringify({report,errors},null,2));
 console.log(JSON.stringify({figures:manifest.length,viewports:report.length,errors},null,2));await b.close();
})().catch(e=>{console.error(e);process.exit(1)});
