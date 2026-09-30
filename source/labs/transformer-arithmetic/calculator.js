'use strict';
const configs={llama3:{v:128256,f:14336,e:1,k:1,kv:8,max:8192},llama2:{v:32000,f:11008,e:1,k:1,kv:32,max:4096},mixtral:{v:32000,f:14336,e:8,k:2,kv:8,max:32768}};
const el=id=>document.getElementById(id);
function update(){
  const c=configs[el('model').value],kv=Number(el('kv').value),b=Number(el('precision').value),T=Number(el('length').value),B=Number(el('batch').value);
  const d=4096,L=32,dh=128,attn=2*d*d+2*d*kv*dh,router=c.e>1?d*c.e:0,ends=2*c.v*d+d;
  const total=L*(attn+3*c.e*d*c.f+2*d+router)+ends;
  const active=L*(attn+3*c.k*d*c.f+2*d+router)+ends;
  const cache=2*B*L*T*kv*dh*b;
  el('config').textContent=`词表 ${c.v.toLocaleString('en-US')} · FFN 中间维度 ${c.f.toLocaleString('en-US')} · ${c.e} 个专家 / 选 ${c.k}`;
  el('length-value').textContent=T.toLocaleString('en-US');el('batch-value').textContent=B;
  el('total').textContent=(total/1e9).toFixed(4)+' B';el('active').textContent=(active/1e9).toFixed(4)+' B';
  el('weight').textContent=(total*b/2**30).toFixed(3)+' GiB';el('cache').textContent=(cache/2**30).toFixed(3)+' GiB';
  el('formula').textContent=`精确总参数：${total.toLocaleString('en-US')}。KV = 2 × ${B} × 32 × ${T} × ${kv} × 128 × ${b} 字节。`;
  el('context').textContent=T>c.max?`当前 T 超过所选原版配置的 ${c.max.toLocaleString('en-US')} token 上限；此处仅作缓存大小的数学外推。`:'观察：改变 batch 或长度只改变缓存，模型的权重个数不变。';
}
el('model').addEventListener('change',()=>{el('kv').value=configs[el('model').value].kv;update()});
for(const id of ['kv','precision','length','batch'])el(id).addEventListener('input',update);
update();
