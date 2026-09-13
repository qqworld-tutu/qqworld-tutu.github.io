'use strict';
const chrome = require('./chrome.cjs');
const {renderProfile} = require('./profile.cjs');
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
module.exports=function renderHomepage({posts,base,cssVersion,jsVersion,profileVersion,chromeVersion,chromeJsVersion,categoryRoutes,tagRoutes}){
// Hexo normalizes config.url without a trailing slash, even when YAML has one.
base=base.replace(/\/+$/,'')+'/';
const url=(kind,value)=>(kind==='categories'?categoryRoutes:tagRoutes)[value];
const counts=key=>{const values=new Map();posts.forEach(p=>p[key].forEach(v=>values.set(v,(values.get(v)||0)+1)));return [...values].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0]));};
const categories=counts('categories'),tags=counts('tags');
return `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="canonical" href="${esc(base)}"><meta name="description" content="QQ 的博客。文章、笔记和项目。"><title>QQ’s blog</title><link rel="icon" href="/images/avatar.jpg"><link rel="preload" as="image" href="/images/cover.jpg"><link rel="stylesheet" href="/css/homepage.css?v=${cssVersion}"><link rel="stylesheet" href="/css/profile.css?v=${profileVersion}"><link rel="stylesheet" href="/css/chrome.css?v=${chromeVersion}"><script src="/js/chrome.js?v=${chromeJsVersion}" defer></script><script src="/js/homepage.js?v=${jsVersion}" defer></script></head><body id="top"><a class="skip" href="#posts">跳到文章</a>
${chrome.header()}
<section class="hero" aria-label="博客封面"><img class="cover" src="/images/cover.jpg" width="3000" height="1725" fetchpriority="high" alt="订制插画：兔子们在草地上野餐"><div class="cover-name"><h1>QQ’s blog</h1></div><a class="scroll-down" href="#posts" aria-label="浏览文章"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="m5 9 7 7 7-7"/></svg></a></section>
<main class="reading"><div class="columns"><section id="posts" class="feed"><div class="feed-header"><h2>最新文章</h2><label class="category-filter"><span class="sr-only">筛选文章分类</span><select id="category-select"><option value="">全部分类</option>${categories.map(([c])=>`<option value="${esc(c)}">${esc(c)}</option>`).join('')}</select></label></div><div class="post-list">${posts.map((p,i)=>`<article class="post" data-post="${i}"><div class="post-meta"><time datetime="${p.date}">${p.date.replaceAll('-','.')}</time>${p.categories.map(c=>`<a href="${url('categories',c)}">${esc(c)}</a>`).join('')}</div><h3><a href="${p.url}">${esc(p.title)}</a></h3>${p.desc?`<p>${esc(p.desc)}</p>`:''}<div class="post-bottom"><div class="post-tags">${p.tags.map(t=>`<a href="${url('tags',t)}">#${esc(t)}</a>`).join('')}</div><a class="read" href="${p.url}">阅读全文</a></div></article>`).join('')}</div><p class="empty" hidden>这个分类下暂无文章。</p><nav class="pagination" aria-label="文章分页" hidden><button data-page="previous">上一页</button><span aria-live="polite"></span><button data-page="next">下一页</button></nav></section>
<aside class="sidebar"><section class="author">${renderProfile()}</section><section class="categories"><h2><a href="${base}categories/">分类</a></h2><ul>${categories.map(([c,count])=>`<li><a href="${url('categories',c)}"><span>${esc(c)}</span><small>${count}</small></a></li>`).join('')}</ul></section><section class="tags"><h2><a href="${base}tags/">标签</a></h2><div>${tags.map(([t])=>`<a href="${url('tags',t)}">${esc(t)}</a>`).join('')}</div></section><a class="archive-link" href="${base}archives/">全部归档 <span>${posts.length} 篇</span></a></aside></div>
${chrome.footer()}</main>

${chrome.search(posts)}</body></html>`;

};
