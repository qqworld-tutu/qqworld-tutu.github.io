'use strict';
document.addEventListener('DOMContentLoaded',()=>{
 const body=document.getElementById('post-body');if(!body)return;
 for(const img of body.querySelectorAll('img[src^="/images/transformer-arithmetic/"],img[src^="/images/kl-divergence/"]')){
   img.setAttribute('no-lazy','');
   if(!img.closest('a')){const a=document.createElement('a');a.href=img.src;a.target='_blank';a.rel='noopener';a.className='technical-figure-link';a.setAttribute('aria-label','在新标签页查看原图：'+img.alt);img.replaceWith(a);a.append(img)}
 }
 for(const pre of body.querySelectorAll('figure.highlight td.code pre, pre')){
   const figure=pre.closest('figure.highlight');
   if(figure&&!pre.closest('td.code'))continue;
   if(!navigator.clipboard)continue;
   let block=figure;
   if(!block){block=document.createElement('div');pre.before(block);block.append(pre)}
   block.classList.add('technical-code-block');
   if(block.querySelector('.copy-code'))continue;
   const button=document.createElement('button');button.type='button';button.className='copy-code';button.textContent='复制代码';
   button.addEventListener('click',async()=>{try{await navigator.clipboard.writeText((pre.querySelector('code')||pre).textContent);button.textContent='已复制'}catch{button.textContent='请选中代码复制'}setTimeout(()=>button.textContent='复制代码',1800)});
   const toolbar=document.createElement('div');toolbar.className='code-toolbar';
   const label=document.createElement('span');label.className='code-language';
   const language=figure?[...figure.classList].find(name=>!['highlight','technical-code-block'].includes(name)):'';
   label.textContent=({python:'Python',bash:'Shell',text:'输出'})[language]||language||'代码';
   toolbar.append(label,button);block.prepend(toolbar);
 }
 const frames=[...body.querySelectorAll('.article-lab,.article-figure')];
 const observed=new WeakSet();
 function resize(frame){try{const main=frame.contentDocument?.querySelector('main');if(main)frame.style.height=Math.ceil(main.getBoundingClientRect().height+4)+'px'}catch{}}
 function watch(frame){
   try{const main=frame.contentDocument?.querySelector('main');if(!main)return;resize(frame);
     if(!observed.has(main)&&'ResizeObserver'in window){const observer=new ResizeObserver(()=>resize(frame));observer.observe(main);observed.add(main)}
   }catch{}
 }
 for(const frame of frames){frame.addEventListener('load',()=>watch(frame));watch(frame)}
 window.addEventListener('message',event=>{
   if(event.origin!==location.origin||event.data?.type!=='technical-figure-height')return;
   const frame=frames.find(f=>f.contentWindow===event.source);if(frame)watch(frame);
 });
});
