'use strict';
// No remote dependencies. Controls only select illustrations; no claims of
// running a model or estimating gradients from a simulation.
(()=>{
 const kind=document.body.dataset.kind;
 const sub=['₀','₁','₂','₃','₄','₅','₆'];
 const sharing=document.getElementById('sharing');
 function pressed(selector,chosen){document.querySelectorAll(selector).forEach(b=>b.setAttribute('aria-pressed',String(b===chosen)))}
 if(kind==='heads'){
  const select=i=>{
   if(i==='all'){document.querySelectorAll('.dim-path').forEach(el=>el.classList.remove('dim-path'));sharing.textContent='MHA 每个 Query 头使用一组 K/V，GQA 每两个头共享一组，MQA 四个头共享一组。';return}
   document.querySelectorAll('.route,.q-node').forEach(el=>el.classList.toggle('dim-path',Number(el.dataset.query)!==i));
   document.querySelectorAll('.kv-node').forEach(el=>el.classList.toggle('dim-path',Number(el.dataset.group)!==Math.floor((i-1)/(4/Number(el.dataset.groups)))+1));
   sharing.textContent=`Q${sub[i]} 在 MHA 中使用 K${sub[i]}/V${sub[i]}，在 GQA 中使用 K${sub[Math.ceil(i/2)]}/V${sub[Math.ceil(i/2)]}，在 MQA 中使用 K₁/V₁。`;
  };
  document.querySelectorAll('[data-i]').forEach(b=>b.addEventListener('click',()=>{pressed('[data-i]',b);select(b.dataset.i==='all'?'all':Number(b.dataset.i))}));select('all');
 }
 if(kind==='moe'){
  const scenarios={a:[.1,.5,.3,.1],b:[.46,.06,.1,.38]};
  function select(key){
   const p=scenarios[key],ids=[0,1,2,3].sort((a,b)=>p[b]-p[a]).slice(0,2),mass=ids.reduce((s,i)=>s+p[i],0),weights=ids.map(i=>p[i]/mass);
   document.querySelectorAll('.prob-bar').forEach((r,i)=>{r.setAttribute('y',130-p[i]*100);r.setAttribute('height',p[i]*100)});
   document.querySelectorAll('.prob-value').forEach((el,i)=>el.textContent=p[i].toFixed(2));
   document.querySelectorAll('.logit-value').forEach((el,i)=>el.textContent='logit '+Math.log(p[i]).toFixed(2));
   document.querySelectorAll('.selected-id').forEach((el,i)=>el.textContent=ids[i]+1);
   document.querySelector('.selected-values').textContent='('+ids.map(i=>p[i].toFixed(2)).join(', ')+')';
   document.querySelector('.selected-weights').textContent='('+weights.map(w=>w.toFixed(3)).join(', ')+')';
   document.querySelectorAll('.expert,.expert-route').forEach(el=>el.classList.toggle('dim-path',!ids.includes(Number(el.dataset.expert)-1)));
   const result=weights.map((w,i)=>`${w.toFixed(3)} f${sub[ids[i]+1]}(x)`).join(' + ');
   document.querySelector('.moe-result').textContent='y = '+result;
   sharing.textContent=`选择专家 ${ids.map(i=>i+1).join('、')}；归一化权重为 ${weights.map(w=>w.toFixed(3)).join('、')}。`;
   // Keep the textual alternative accurate when the selected experts change.
   document.querySelectorAll('svg')[1].setAttribute('aria-label',sharing.textContent);
  }
  document.querySelectorAll('[data-moe]').forEach(b=>b.addEventListener('click',()=>{pressed('[data-moe]',b);select(b.dataset.moe)}));select('a');
 }
 if(kind==='gradient'){
  document.querySelectorAll('[data-path]').forEach(b=>b.addEventListener('click',()=>{
   pressed('[data-path]',b);const p=b.dataset.path;
   document.querySelector('.probability-path').classList.toggle('dim-path',p==='direct');
   document.querySelector('.direct-path').classList.toggle('dim-path',p==='prob');
   sharing.textContent={all:'完整梯度包含两项。',prob:'概率路径：k 停止梯度，通过采样概率的变化产生贡献。',direct:'直接反传固定样本的 k，只计算公式路径；它不会自动补上概率路径。'}[p];
  }));
 }
 if(kind==='future'){
  document.querySelectorAll('[data-row-select]').forEach(b=>b.addEventListener('click',()=>{
   pressed('[data-row-select]',b);const t=b.dataset.rowSelect;
   document.querySelectorAll('.term-cell').forEach(el=>el.classList.toggle('dim-path',t!=='all'&&el.dataset.row!==t));
   sharing.textContent=t==='all'?'完整序列梯度保留上三角；固定前缀的局部梯度只保留对角项。':`对 g${sub[Number(t)]}：序列梯度使用 ${Array.from({length:5-Number(t)},(_,i)=>'u'+sub[Number(t)+i]).join(' + ')}；局部梯度只使用 u${sub[Number(t)]}。`;
  }));
 }
 if(kind==='cache'){
  let T=4;
  const C={q:'#4f8fa5',k:'#ca8b4f',v:'#8a74b5',a:'#bb6465'};
  const text=(x,y,s,cls='symbol')=>`<text x="${x}" y="${y}" text-anchor="middle" class="${cls}">${s}</text>`;
  const arrow=(x1,x2,y)=>`<path d="M${x1},${y}H${x2}" class="arrow" marker-end="url(#tip)"/>`;
  function mat(x,y,r,c,role,label,shape){
   let s=`<g class="matrix" data-rows="${r}" data-cols="${c}">`;
   for(let i=0;i<r;i++)for(let j=0;j<c;j++)s+=`<rect x="${x+j*16+1}" y="${y+i*16+1}" width="14" height="14" rx="1.5" fill="${C[role]}" fill-opacity="${[.42,.65,.86][(i*7+j*13+i*j)%3]}"/>`;
   return s+`<path class="bracket" d="M${x-3},${y}h-4v${r*16}h4 M${x+c*16+3},${y}h4v${r*16}h-4"/>`+text(x+c*8,y+r*16+29,label)+text(x+c*8,y+r*16+52,shape,'shape')+'</g>';
  }
  function render(){
   let first='';
   for(const[y,role,name]of [[20,'k','K'],[188,'v','V']])first+=mat(30,y,T-1,2,role,name+'旧',`${T-1} × 2`)+text(150,y+30,'追加','operation')+mat(239,y,1,2,role,name+'新','1 × 2')+arrow(312,384,y+23)+mat(437,y,T,2,role,name,`${T} × 2`)+text(680,y+29,'沿位置轴追加当前 K/V','annotation');
   const second=mat(25,20,1,2,'q','q','1 × 2')+text(94,43,'×','op')+mat(133,12,2,T,'k','Kᵀ',`2 × ${T}`)+arrow(265,309,28)+text(400,32,'缩放 → softmax','operation')+arrow(490,529,28)+mat(574,20,1,T,'a','a',`1 × ${T}`);
   const third=mat(25,45,1,T,'a','a',`1 × ${T}`)+text(164,67,'×','op')+mat(211,13,T,2,'v','V',`${T} × 2`)+text(296,67,'=','op')+mat(348,45,1,2,'q','H','1 × 2')+text(652,48,'各头输出拼接后乘 W_O','annotation')+text(652,83,'得到当前 token 的注意力输出','shape');
   document.querySelectorAll('.stage svg').forEach((el,i)=>{const defs=el.querySelector('defs').outerHTML;el.innerHTML=defs+[first,second,third][i]});
   sharing.textContent=`缓存长度 T = ${T}，包含 ${T-1} 个历史位置和 1 个当前位置。`;
   document.getElementById('cache-next').disabled=T===6;
  }
  document.getElementById('cache-next').addEventListener('click',()=>{if(T<6){T++;render()}});
  document.getElementById('cache-reset').addEventListener('click',()=>{T=4;render()});
 }
 // Parent sizes only same-origin figure iframes. Message height is backed by
 // document geometry and is also checked by the parent ResizeObserver.
 const main=document.querySelector('main');
 const report=()=>{if(parent!==window)parent.postMessage({type:'technical-figure-height',height:Math.ceil(main.getBoundingClientRect().height+2)},location.origin)};
 new ResizeObserver(report).observe(main);window.addEventListener('load',report);
})();
