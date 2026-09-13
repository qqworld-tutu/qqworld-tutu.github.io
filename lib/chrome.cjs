'use strict';
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const menu = [['首页','/'],['分类','/categories/'],['标签','/tags/'],['归档','/archives/'],['友链','/friends/'],['关于','/about/'],['学术主页','https://www.chenquan-tutu.top/']];
function header(current='/') {
  // Hidden legacy hooks keep Volantis comment/TOC initialization working;
  // the visible navigation and its interactions belong to this shared shell.
  const pathname='/'+current.replace(/^\/+/, '');
  const nav=()=>menu.map(([label,url])=>`<a href="${url}"${url==='/' ? pathname==='/'?' aria-current="page"':'' : pathname.startsWith(url)?' aria-current="page"':''}>${label}</a>`).join('');
  return `<header id="l_header" class="qq-header"><a class="qq-brand" href="/"><img no-lazy src="/images/avatar.jpg" alt="" width="30" height="30"><span>QQ’s blog</span></a><nav class="qq-desktop-nav" aria-label="主导航">${nav()}</nav><button class="qq-search-open" type="button" aria-label="搜索文章"><svg viewBox="0 0 24 24" width="19" height="19" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/></svg></button><details class="qq-mobile-menu"><summary>菜单 <span aria-hidden="true">☰</span></summary><nav aria-label="手机导航">${nav()}</nav></details><div id="wrapper" hidden aria-hidden="true"><div class="nav-sub"><span class="title"></span><button id="s-comment" type="button" tabindex="-1">评论</button><button id="s-toc" type="button" tabindex="-1">目录</button></div><div class="nav-main"><div class="menu-phone list-v"></div></div></div></header>`;
}
function footer() { return '<footer class="qq-footer"><span>© 2025–2026 QQ</span><a href="#top" data-back-top>回到顶部</a><a href="https://www.chenquan-tutu.top/">学术主页</a></footer>'; }
function search(posts) {return `<dialog class="qq-search" aria-labelledby="qq-search-title"><div class="qq-search-heading"><h2 id="qq-search-title">搜索文章</h2><button type="button" class="qq-search-close" aria-label="关闭搜索">×</button></div><input id="qq-search-input" type="search" placeholder="输入关键词" aria-label="搜索标题、摘要或标签"><div class="qq-search-results" aria-live="polite"></div></dialog><script type="application/json" id="post-data">${JSON.stringify(posts).replace(/</g,'\\u003c')}</script>`;}
module.exports={header,footer,search,escape,menu};
