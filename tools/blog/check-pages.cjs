// Actual browser checks for the two technical articles and their interactive examples.
// PLAYWRIGHT_MODULE can point to an existing installation; no test framework needed.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
const base=process.env.BLOG_PREVIEW_URL||'http://127.0.0.1:4341';
const out=path.resolve('output/playwright');
fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1280,height:900}});
 const errors=[],missing=[],report=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('response',r=>{if(r.url().startsWith(base)&&r.status()>=400)missing.push({url:r.url(),status:r.status()})});
 // Comments are an unrelated external service; a new article has no discussion yet.
 await page.route('https://giscus.app/**',route=>route.fulfill({status:200,body:''}));
 for(const slug of ['transformer-model-arithmetic','kl-divergence-in-rl']){
  await page.goto(base+'/llm/'+slug+'/',{waitUntil:'networkidle'});
  await page.locator('#post-body').waitFor();
  await page.evaluate(()=>document.fonts.ready);
  await page.screenshot({path:path.join(out,slug+'-desktop.png'),timeout:15000});
  await page.locator('.article-lab').scrollIntoViewIfNeeded();
  const lab=page.frameLocator('.article-lab');
  await lab.locator('main strong').first().waitFor();
  await page.locator('.article-lab').screenshot({path:path.join(out,slug+'-lab.png'),timeout:15000});
  assert.equal(await page.locator('.katex-error').count(),0);
  const figures=page.locator('.article-figure');
  assert.equal(await figures.count(),slug==='transformer-model-arithmetic'?6:4);
  for(const frame of await figures.all()){
   await frame.scrollIntoViewIfNeeded();
   const content=await frame.contentFrame();await content.locator('main').waitFor();
   await content.locator('body').evaluate(()=>document.fonts.ready);
   await page.waitForFunction(el=>{const main=el.contentDocument?.querySelector('main');return main&&Math.abs(el.clientHeight-main.getBoundingClientRect().height)<8},await frame.elementHandle());
  }
  const images=await figures.evaluateAll(es=>es.map(e=>({src:e.getAttribute('src'),ok:!!e.contentDocument?.querySelector('main')})));
  const math=await page.locator('#post-body .katex').count();
  await page.context().grantPermissions(['clipboard-read','clipboard-write']);
  await page.locator('.copy-code').first().click();
  assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),await page.locator('#post-body .technical-code-block').first().locator('td.code pre, :scope > pre').first().evaluate(pre=>(pre.querySelector('code')||pre).textContent));
  if(slug==='transformer-model-arithmetic'){
   assert.equal(await lab.locator('#total').textContent(),'8.0303 B');
   assert.equal(await lab.locator('#cache').textContent(),'1.000 GiB');
   await lab.locator('#kv').selectOption('32');
   assert.equal(await lab.locator('#cache').textContent(),'4.000 GiB');
   await lab.locator('#model').selectOption('mixtral');
   assert.equal(await lab.locator('#total').textContent(),'46.7028 B');
   assert.equal(await lab.locator('#active').textContent(),'12.8799 B');
   await lab.locator('#model').selectOption('llama3');
  }else{
   assert.equal(await lab.locator('#reverse').textContent(),'0.192745');
   assert((await lab.locator('#rows').innerText()).includes('1.875000'));
   await lab.locator('#p').fill('50');
   assert.equal(await lab.locator('#reverse').textContent(),'0.000000');
   await lab.locator('#p').fill('80');
   await page.locator('#post-body details summary').first().click();
   assert.equal(await page.locator('#post-body details').first().getAttribute('open'),'');
   await page.locator('#post-body details summary').first().click();
  }
  // Inline figures open a full-width HTML view in a new tab.
  const imageLink=page.frameLocator('.article-figure').first().locator('.standalone-link');
  assert.equal(await imageLink.getAttribute('target'),'_blank');
  const popupPromise=page.waitForEvent('popup');
  await imageLink.click();
  const popup=await popupPromise;await popup.waitForLoadState();
  assert(popup.url().includes('/figures/'));await popup.close();
  for(const width of [390,360]){
   await page.setViewportSize({width,height:844});
   await page.evaluate(()=>scrollTo(0,0));
   const size=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,wide:[...document.querySelectorAll('#post-body > *')].filter(e=>e.getBoundingClientRect().right>innerWidth+1).map(e=>({tag:e.tagName,class:e.className}))}));
   assert(size.scroll<=width+1,JSON.stringify(size));
   const frameSize=await lab.locator('body').evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));
   assert(frameSize.scroll<=frameSize.width+1,JSON.stringify(frameSize));
   for(const frame of await figures.all()){
    await frame.scrollIntoViewIfNeeded();const content=await frame.contentFrame();
    const fsize=await content.locator('body').evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));
    assert(fsize.scroll<=fsize.width,JSON.stringify(fsize));
    await page.waitForFunction(el=>{const main=el.contentDocument?.querySelector('main');return main&&Math.abs(el.clientHeight-main.getBoundingClientRect().height)<8},await frame.elementHandle());
   }
   report.push({slug,viewport:width,math,images:images.length,...size});
   await page.screenshot({path:path.join(out,slug+'-'+width+'.png'),timeout:15000});
  }
  await page.setViewportSize({width:1280,height:900});
 }
 await page.goto(base+'/llm/transformer-model-arithmetic/',{waitUntil:'networkidle'});
 for(const name of ['02-qkvo-tensor-flow','06-swiglu']){
  const frame=page.locator('.article-figure[src*="'+name+'"]');
  await frame.scrollIntoViewIfNeeded();
  await frame.screenshot({path:path.join(out,name+'-browser.png'),timeout:15000});
 }
 assert.deepEqual(errors,[]);assert.deepEqual(missing,[]);
 const nojs=await browser.newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});
 const staticPage=await nojs.newPage();
 await staticPage.goto(base+'/llm/kl-divergence-in-rl/',{waitUntil:'domcontentloaded'});
 await staticPage.locator('#post-body details summary').first().click();
 assert.equal(await staticPage.locator('#post-body details').first().getAttribute('open'),'');
 assert.equal(await staticPage.locator('.katex-error').count(),0);
 assert.equal(await staticPage.locator('.article-figure:visible').count(),0);
 assert.equal(await staticPage.locator('.figure-static:visible').count(),4);
 for(const img of await staticPage.locator('.figure-static img').all()){await img.scrollIntoViewIfNeeded();await img.evaluate(e=>e.decode())}
 await staticPage.locator('.figure-static').first().screenshot({path:path.join(out,'kl-nojs-fallback.png')});
 await nojs.close();
 const result={checks:report,errors,missing,interactive:'model presets, KV heads, binary gradients, equal distributions, details, figure tabs, clipboard, 10 figure iframe heights/widths, no-JS figures and details'};
 fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify(result,null,2));
 console.log(JSON.stringify(result,null,2));
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
