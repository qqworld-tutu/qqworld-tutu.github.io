"""One self-contained, editable HTML figure. Geometry: one axis unit = 16 px.

Ledger (batch B=1 omitted): X=(3,8), WQ=(8,8), WK/WV=(8,4),
Q=(3,8), K/V=(3,4); split Q into 4 and K/V into 2 heads of width 2.
Qi/Kg/Vg=(3,2), Kg.T=(2,3), Ai=(3,3), Hi=(3,2), concat H=(3,8),
WO=(8,8), O=(3,8). Q/K below denote tensors after RoPE.
All values are real except A (row probability); mask is causal support.
Tile shades are illustrative, not computed values. No numeric color legend.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C = {'q':'#4f8fa5','k':'#ca8b4f','v':'#8a74b5','a':'#bb6465','n':'#85898f'}

def text(x,y,s,cls='',anchor='middle'):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}">{s}</text>'

def brackets(x,y,r,c):
    return f'<path class="bracket" d="M{x-3},{y}h-4v{r*16}h4 M{x+c*16+3},{y}h4v{r*16}h-4"/>'

def matrix(x,y,r,c,role,label='',shape='',mask=False,headgroups=False,groupoffset=0,border=True):
    out = [f'<g class="matrix" data-rows="{r}" data-cols="{c}">']
    # Exact geometry is independent of whitespace between cells.
    for i in range(r):
        for j in range(c):
            if mask and j>i: continue
            group = j//2+1+groupoffset
            extra=f' class="head-cell {role}-head" data-head="{group}"' if headgroups else ''
            seed=((i+1)*73856093)^((j+1)*19349663)
            opacity=[.42,.65,.86][((seed^(seed>>13))*1274126177)%3]
            out.append(f'<rect x="{x+j*16+1}" y="{y+i*16+1}" width="14" height="14" rx="1.5" fill="{C[role]}" fill-opacity="{opacity}"{extra}/>')
    if border: out.append(brackets(x,y,r,c))
    if label: out.append(text(x+c*8,y+r*16+29,label,'symbol'))
    if shape: out.append(text(x+c*8,y+r*16+52,shape,'shape'))
    out.append('</g>')
    return ''.join(out)

def arrow(x1,x2,y):
    return f'<path class="arrow" d="M{x1},{y}H{x2}" marker-end="url(#tip)"/>'

def svg(content,height,label):
    return f'<svg viewBox="0 0 900 {height}" role="img" aria-label="{label}"><defs><marker id="tip" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M1 1L7 4L1 7" fill="none" stroke="#8c9291" stroke-width="1.2"/></marker></defs>{content}</svg>'

# 1. Fused projection: preserve widths 8:4:4 exactly.
s1=matrix(22,58,3,8,'n','X','3 × 8')+text(184,91,'×','op')
for x,c,role,label in [(222,8,'q','Q'),(350,4,'k','K'),(414,4,'v','V')]:
    s1+=matrix(x,18,8,c,role,border=False)+text(x+c*8,175,f'W<tspan baseline-shift="sub" font-size="12">{label}</tspan>','symbol '+role)
s1+=brackets(222,18,8,16)
s1+=text(350,203,'8 × (8 + 4 + 4)','shape')+arrow(509,554,82)
for x,c,role,label,shape in [(594,8,'q','Q','3 × 8'),(722,4,'k','K','3 × 4'),(786,4,'v','V','3 × 4')]:
    s1+=matrix(x,58,3,c,role,label,shape,headgroups=True,border=False)
    for j in range(2,c,2): s1+=f'<path d="M{x+j*16},55v54" class="head-divider"/>'
s1+=brackets(594,58,3,16)+text(722,203,'每 2 列拆为一个头','shape')

# 2. Q_i K_g^T contracts d_h; causal softmax has a square T x T face.
s2=matrix(22,44,3,2,'q','Qᵢ','3 × 2')+text(89,75,'×','op')
s2+=matrix(124,52,2,3,'k','Kᵍᵀ','2 × 3')+arrow(197,236,68)
s2+=text(374,52,'缩放 · 因果遮罩 · softmax','operation')
s2+=text(374,85,'沿 Key 位置归一化','shape')+arrow(506,548,68)
s2+=matrix(583,44,3,3,'a','Aᵢ','3 × 3',mask=True)
s2+=text(697,57,'行：Query 位置','annotation','start')
s2+=text(697,86,'列：Key 位置','annotation','start')
s2+=text(697,115,'未来位置的概率为 0','shape','start')

# 3. Contract Key positions; output matches Q_i geometry.
s3=matrix(22,22,3,3,'a','Aᵢ','3 × 3',mask=True)+text(105,55,'×','op')
s3+=matrix(140,22,3,2,'v','Vᵍ','3 × 2')+text(218,55,'=','op')
s3+=matrix(267,22,3,2,'q','Hᵢ','3 × 2')
s3+=text(397,38,'各 Query 头分别计算注意力概率。','annotation','start')
s3+=text(397,69,'共享 K/V 的头仍有各自的 A 与 H。','annotation','start')

# 4. Concatenation conserves head widths: 4 x 2 = 8.
s4=matrix(22,52,3,8,'q','[H₁ | H₂ | H₃ | H₄]','3 × 8')
for x in [54,86,118]: s4+=f'<path d="M{x},49v54" class="head-divider"/>'
s4+=text(199,84,'×','op')+matrix(248,12,8,8,'n','Wₒ','8 × 8')
s4+=text(423,84,'=','op')+matrix(472,52,3,8,'q','O','3 × 8')
s4+=text(663,65,'输出宽度为 d = 8','annotation','start')
s4+=text(663,94,'与残差流的宽度一致','shape','start')

stages=[
 ('01','投影与拆头','输入分别投影为 Q、K、V，再按每头宽度拆分。',svg(s1,224,'输入 X 乘拼接权重得到 Q、K、V，再各按每两列拆头。')),
 ('02','注意力概率','Q 与 K 的匹配分数经缩放、因果遮罩和 softmax 转为概率。',svg(s2,160,'Q 乘转置的 K，再缩放、加因果遮罩和 softmax，得到下三角注意力概率矩阵 A。')),
 ('03','汇总 Value','按注意力概率对各位置的 Value 加权求和，得到单个头的输出。',svg(s3,140,'注意力概率 A 乘 V，得到一个 Query 头的输出 H。')),
 ('04','拼接与输出','各头输出沿特征维拼接，再由输出投影映射回 d 维。',svg(s4,208,'四个 H 沿特征轴拼接后，乘输出权重 W O，得到和输入相同形状的输出。')),
]
rows='\n'.join(f'<section class="stage"><div class="stage-copy"><span class="step">{n}</span><h2>{title}</h2><p>{desc}</p></div><div class="diagram-scroll" tabindex="0" aria-label="{title}，窄屏可横向滚动">{drawing}</div></section>' for n,title,desc,drawing in stages)

html='''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Q、K、V 的投影与注意力计算 · 图示预览</title>
<style>
:root{--ink:#293b40;--muted:#68767a;--line:#e3e8e8;--q:#4f8fa5;--k:#ca8b4f;--v:#8a74b5;--a:#bb6465;--sans:"PingFang SC","Noto Sans CJK SC","Microsoft YaHei",sans-serif;--serif:"Songti SC","Noto Serif CJK SC",serif}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font:15px/1.75 var(--sans)}main{max-width:1220px;padding:48px 36px 36px;margin:auto}header{max-width:800px;margin:0 0 30px}h1{font:500 36px/1.3 var(--serif);letter-spacing:.02em;margin:0 0 13px}header p{color:var(--muted);margin:0;font-size:15px}.q{color:var(--q);fill:var(--q)}.k{color:var(--k);fill:var(--k)}.v{color:var(--v);fill:var(--v)}
.formula{font:20px/1.7 Georgia,"Times New Roman",var(--serif);display:flex;flex-wrap:wrap;gap:8px 26px;margin:21px 0 0}.formula sub{font-size:65%}.formula span{white-space:nowrap}
.figure{margin:0}.controls{display:flex;flex-wrap:wrap;gap:12px 24px;align-items:center;border-block:1px solid var(--line);padding:15px 0;margin:0 0 14px}.head-picker{display:flex;align-items:center;gap:7px}.head-picker>span{margin-right:6px;font-size:13px;color:var(--muted)}button{font:inherit;cursor:pointer;border:0;background:transparent;color:var(--muted);padding:6px 10px;border-radius:4px;min-width:39px;line-height:1.4}button[aria-pressed=true]{background:#eaf2f4;color:#325e6d}button:hover{background:#f0f5f6}button:focus-visible,.diagram-scroll:focus-visible{outline:2px solid var(--q);outline-offset:4px}.sharing{font-size:13px;color:var(--muted);margin:0}.sharing strong{font-weight:500;color:var(--ink)}
.stage{display:grid;grid-template-columns:170px minmax(0,1fr);gap:34px;align-items:center;padding:16px 0}.stage+.stage{border-top:1px solid var(--line)}.step{font:italic 17px Georgia,serif;color:#87969a}.stage h2{font-size:17px;font-weight:600;line-height:1.5;margin:4px 0 8px}.stage p{font-size:13px;line-height:1.85;color:var(--muted);margin:0}.diagram-scroll{min-width:0;overflow-x:auto;overscroll-behavior-x:contain;scrollbar-width:thin;scrollbar-color:#ccd8dc transparent}.diagram-scroll svg{display:block;width:100%;height:auto;min-width:760px;overflow:visible}svg text{fill:var(--ink);font-family:var(--sans);font-size:15px}.symbol{font-family:Georgia,"Times New Roman",serif;font-size:18px}.shape{fill:var(--muted);font-size:13px}.annotation{font-size:15px}.operation{font-size:16px}.op{font:26px Georgia,serif;fill:#748085}.bracket{fill:none;stroke:#859194;stroke-width:1}.arrow{fill:none;stroke:#8c9291;stroke-width:1.2}.head-divider{stroke:#fff;stroke-width:3}.head-cell{transition:opacity .18s}.head-cell.dim{opacity:.22}svg .q{fill:var(--q)}svg .k{fill:var(--k)}svg .v{fill:var(--v)}
figcaption{border-top:1px solid var(--line);padding-top:20px;margin-top:8px;font-size:12px;color:var(--muted)}.notes{background:#f6f8f8;padding:17px 20px;display:grid;grid-template-columns:40px 1fr;gap:7px 14px;margin:0}.notes dt{font-weight:600;color:#536569}.notes dd{margin:0}.mobile-hint{display:none;font-size:12px;color:var(--muted);margin:0 0 10px}footer{margin-top:17px;font-size:12px;color:var(--muted)}footer a{color:var(--muted);text-underline-offset:4px}noscript p{font-size:13px;color:var(--muted)}
@media(min-width:1000px){.diagram-scroll svg{min-width:0}}
@media(max-width:999px){main{padding:30px 24px}.stage{grid-template-columns:1fr;gap:10px;padding:22px 0}.stage-copy{display:grid;grid-template-columns:26px 150px 1fr;align-items:baseline;gap:10px}.stage h2,.stage p{margin:0}.mobile-hint{display:block}.diagram-scroll svg{min-width:820px}.controls{gap:10px}}
@media(max-width:560px){main{padding:26px 18px}h1{font-size:28px}header p{font-size:14px}.formula{font-size:17px;gap:4px 18px}.head-picker{gap:3px}.sharing{font-size:12px}.stage-copy{grid-template-columns:24px 1fr;gap:4px 10px}.stage-copy p{grid-column:2}.notes{padding:14px;gap:8px 10px}.diagram-scroll svg{min-width:800px}.stage h2{font-size:16px}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
@media print{main{padding:0}.controls,footer,.mobile-hint{display:none}.stage{break-inside:avoid;grid-template-columns:150px 1fr;gap:20px}.diagram-scroll{overflow:visible}.diagram-scroll svg{min-width:0}.head-cell.dim{opacity:1}}
</style></head><body><main>
<header><h1>Q、K、V 的投影与注意力计算</h1><p>以包含 4 个 Query 头和 2 个 KV 头的 GQA 为例，展示各步的张量形状。</p>
<div class="formula" aria-label="Q 等于 X W Q；K 等于 X W K；V 等于 X W V；H i 等于 softmax 括号 Q i K g 转置除以根号 d h 加 mask 括号乘 V g"><span><b class="q">Q</b> = XW<sub>Q</sub>　<b class="k">K</b> = XW<sub>K</sub>　<b class="v">V</b> = XW<sub>V</sub></span><span>H<sub>i</sub> = softmax(Q<sub>i</sub>K<sub>g</sub><sup>T</sup> / √d<sub>h</sub> + M)V<sub>g</sub></span></div></header>
<figure class="figure">
<div class="controls"><div class="head-picker" role="group" aria-label="选择 Query 头"><span>查看 Query 头</span><button type="button" data-i="1" aria-pressed="true">Q₁</button><button type="button" data-i="2" aria-pressed="false">Q₂</button><button type="button" data-i="3" aria-pressed="false">Q₃</button><button type="button" data-i="4" aria-pressed="false">Q₄</button></div><p class="sharing" aria-live="polite" id="sharing">Q₁、Q₂ 共享 <strong>K₁ / V₁</strong>，当前查看 Q₁。</p></div>
<p class="mobile-hint">图示可左右滑动，保留矩阵比例与文字大小。</p>
ROWS
<figcaption><dl class="notes"><dt>维度</dt><dd>B = 1（省略批次轴），T = 3 个位置，d = 8，h = 4，h<sub>kv</sub> = 2，d<sub>h</sub> = 2。每格对应一个元素。</dd><dt>对象</dt><dd>W 表示可学习权重，Q、K、V、H、O 表示中间张量。A 的每行是一组注意力概率，和为 1。色块深浅仅作示意。</dd><dt>约定</dt><dd>头编号从 1 开始，g = ⌈i / 2⌉。Q、K 在匹配前应用 RoPE；M 在可见位置取 0，在未来位置取 −∞。图中省略偏置与 dropout。</dd></dl></figcaption>
</figure><noscript><p>当前显示完整计算流程；启用 JavaScript 后可切换 Query 头。</p></noscript>
<footer><a href="/images/transformer-arithmetic/02-qkvo-tensor-flow.svg" target="_blank" rel="noopener">静态矢量图</a></footer>
</main><script>
const sub=['₀','₁','₂','₃','₄'];
function selectHead(i){
 const g=Math.ceil(i/2);
 document.querySelectorAll('[data-i]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.i)===i)));
 document.querySelectorAll('.head-cell').forEach(c=>c.classList.toggle('dim',Number(c.dataset.head)!==(c.classList.contains('q-head')?i:g)));
 document.querySelector('#sharing').innerHTML=`Q${sub[2*g-1]}、Q${sub[2*g]} 共享 <strong>K${sub[g]} / V${sub[g]}</strong>，当前查看 Q${sub[i]}。`;
 document.querySelectorAll('svg .symbol').forEach(t=>{
  if(!t.dataset.generic)t.dataset.generic=t.textContent;
  if(/[ᵢᵍ]/.test(t.dataset.generic))t.textContent=t.dataset.generic.replaceAll('Qᵢ','Q'+sub[i]).replaceAll('Kᵍᵀ','K'+sub[g]+'ᵀ').replaceAll('Vᵍ','V'+sub[g]).replaceAll('Aᵢ','A'+sub[i]).replaceAll('Hᵢ','H'+sub[i]);
 });
}
document.querySelectorAll('[data-i]').forEach(b=>b.addEventListener('click',()=>selectHead(Number(b.dataset.i))));
selectHead(1);
</script></body></html>'''.replace('ROWS',rows)
dest=ROOT/'source/labs/qkv-flow-preview/index.html'
dest.parent.mkdir(parents=True,exist_ok=True)
dest.write_text(html)
print(dest)
