'use strict';
const el=id=>document.getElementById(id),fmt=x=>(Math.abs(x)<1e-12?0:x).toFixed(6);
function update(){
 const p=Number(el('p').value)/100,q=Number(el('q').value)/100,P=[p,1-p],Q=[q,1-q];
 const u=P.map((v,i)=>Math.log(v/Q[i])),w=P.map((v,i)=>Q[i]/v),g=[1/p,-1/(1-p)],avg=a=>a.reduce((s,v,i)=>s+P[i]*v,0);
 const reverse=avg(u),forward=Q.reduce((s,v,i)=>s+v*Math.log(v/P[i]),0),dr=u[0]-u[1],df=-q/p+(1-q)/(1-p);
 el('p-value').textContent=p.toFixed(2);el('q-value').textContent=q.toFixed(2);el('p-bar').style.width=p*100+'%';el('q-bar').style.width=q*100+'%';
 el('reverse').textContent=fmt(reverse);el('forward').textContent=fmt(forward);
 const rows=[['精确 reverse KL',reverse,dr],['精确 forward KL',forward,df],['k₁ = u',reverse,avg(g)],['k₂ = u² / 2',avg(u.map(v=>v*v/2)),avg(u.map((v,i)=>v*g[i]))],['k₃ = exp(−u) − 1 + u',avg(u.map(v=>Math.expm1(-v)+v)),avg(w.map((v,i)=>(1-v)*g[i]))]];
 el('rows').replaceChildren(...rows.map(row=>{const tr=document.createElement('tr');row.forEach((v,i)=>{const td=document.createElement(i?'td':'th');td.textContent=i?fmt(v):v;if(!i)td.scope='row';tr.append(td)});return tr}));
 el('explanation').textContent=Math.abs(p-q)<1e-10?'P 与 Q 相同时，KL 和这些期望梯度都为零。把两个分布移开，才能观察差异。':`k₁ 与 k₃ 的数值期望都等于 reverse KL；直接反传时，k₁ 的期望梯度为 0，k₃ 则对应 forward KL。k₂ 数值有偏，梯度却等于 ${fmt(dr)}。`;
}
el('p').addEventListener('input',update);el('q').addEventListener('input',update);update();
