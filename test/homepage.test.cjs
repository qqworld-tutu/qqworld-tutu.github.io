const test = require('node:test');
const assert = require('node:assert/strict');
const render = require('../lib/homepage.cjs');
const options = {base:'https://blog.chenquan-tutu.top/',cssVersion:'css123',jsVersion:'js123',categoryRoutes:{新增分类:'https://blog.chenquan-tutu.top/categories/new/'},tagRoutes:{新标签:'https://blog.chenquan-tutu.top/tags/new/'}};
test('navigation works with Hexo-normalized URLs, with or without a trailing slash',()=>{
  for(const base of ['https://blog.chenquan-tutu.top','https://blog.chenquan-tutu.top/']){
    const html=render({...options,base,posts:[]});
    for(const path of ['categories/','tags/','archives/','friends/','about/']){
      assert(html.includes(`href="https://blog.chenquan-tutu.top/${path}"`));
    }
    for(const [,href] of html.matchAll(/href="(https:[^"]+)"/g)){
      assert(['blog.chenquan-tutu.top','www.chenquan-tutu.top','github.com'].includes(new URL(href).hostname),href);
    }
  }
});
test('arbitrary posts, categories, tags and summaries render without fixed slots',()=>{
  const posts=Array.from({length:24},(_,i)=>({title:`新增文章 ${i}`,date:'2026-09-13',categories:['新增分类'],tags:['新标签'],desc:'写在 Markdown 里的摘要。',url:`https://blog.chenquan-tutu.top/posts/${i}/`}));
  const html=render({...options,posts});assert.equal((html.match(/data-post="/g)||[]).length,24);assert(html.includes('新增分类'));assert(html.includes('24 篇'));assert(html.includes('css123'));assert(!html.includes('本地预览'));assert(!html.includes('jinrishici'));assert(html.includes('fetchpriority="high"'));
});
test('text and embedded search data cannot inject HTML',()=>{
  const title='</script><img src=x onerror=alert(1)>';
  const html=render({...options,posts:[{title,date:'2026-09-13',categories:[],tags:[],desc:'<b>hello</b>',url:'https://blog.chenquan-tutu.top/post/'}]});
  assert(!html.includes(title));assert(html.includes('&lt;b&gt;hello&lt;/b&gt;'));const data=html.match(/id="post-data">([\s\S]*?)<\/script>/)[1];assert.equal(JSON.parse(data)[0].title,title);
});
