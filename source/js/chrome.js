(() => {
  const menu=document.querySelector('.qq-mobile-menu');
  menu.addEventListener('click',event=>{if(event.target.closest('a'))menu.open=false;});
  document.addEventListener('click',event=>{if(!menu.contains(event.target))menu.open=false;});
  const dialog=document.querySelector('.qq-search'),input=document.querySelector('#qq-search-input'),results=document.querySelector('.qq-search-results');
  const posts=JSON.parse(document.querySelector('#post-data').textContent);
  function search(){
    const query=input.value.trim().toLowerCase();results.replaceChildren();
    const matches=posts.filter(p=>[p.title,p.desc,...p.categories,...p.tags].join(' ').toLowerCase().includes(query));
    for(const post of matches){const a=document.createElement('a');a.href=post.url;a.textContent=post.title;const meta=document.createElement('span');meta.textContent=[post.date,...post.categories].join(' / ');a.append(meta);results.append(a);}
    if(!matches.length)results.textContent='没有找到相关文章。';
  }
  document.querySelector('.qq-search-open').addEventListener('click',()=>{menu.open=false;input.value='';search();dialog.showModal();input.focus();});
  document.querySelector('.qq-search-close').addEventListener('click',()=>dialog.close());
  input.addEventListener('input',search);
  dialog.addEventListener('keydown',event=>{if(event.key==='Escape'){event.preventDefault();dialog.close();}});
  dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
  document.querySelectorAll('[data-back-top]').forEach(a=>a.addEventListener('click',event=>{event.preventDefault();scrollTo({top:0,behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'instant':'smooth'});}));
  for(const img of document.querySelectorAll('.qq-friend img')){
    const fallback=()=>{const span=document.createElement('span');span.className='qq-friend-initial';span.setAttribute('aria-hidden','true');span.textContent=img.closest('a').querySelector('strong').textContent.slice(0,1);img.replaceWith(span);};
    img.addEventListener('error',fallback,{once:true});if(img.complete&&!img.naturalWidth)fallback();
  }
  for(const wechat of document.querySelectorAll('.qq-wechat')){
    const button=wechat.querySelector('button');
    button.addEventListener('click',()=>{wechat.classList.remove('dismissed');const open=wechat.classList.toggle('is-open');button.setAttribute('aria-expanded',String(open));if(!open)wechat.classList.add('dismissed');});
    wechat.addEventListener('pointerenter',()=>wechat.classList.remove('dismissed'));
    button.addEventListener('focus',()=>wechat.classList.remove('dismissed'));
    document.addEventListener('click',event=>{if(!wechat.contains(event.target)){wechat.classList.remove('is-open');button.setAttribute('aria-expanded','false');}});
  }
  document.addEventListener('keydown',event=>{if(event.key!=='Escape')return;if(menu.open){menu.open=false;menu.querySelector('summary').focus();}document.querySelectorAll('.qq-wechat').forEach(w=>{w.classList.remove('is-open');w.classList.add('dismissed');w.querySelector('button').setAttribute('aria-expanded','false');});});
})();
