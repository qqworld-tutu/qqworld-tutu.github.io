"""Build the ten article figures as HTML + inline SVG, with vector fallbacks.

Run build_editorial_charts.py first for the four scientific chart panels.
All diagram geometry uses explicit axis lengths; process nodes are not matrices.
The QKV prototype remains available independently for comparison.
"""
from pathlib import Path
from html import escape
import hashlib
import json
import re
import runpy
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
P = runpy.run_path(str(ROOT/'tools/blog/build-qkv-preview.py'))
mat, text, arrow, svg = (P[k] for k in ['matrix','text','arrow','svg'])
BASE = ROOT/'source/figures'
BASE.mkdir(parents=True, exist_ok=True)
FIGURES = []

style = re.search(r'<style>(.*?)</style>', P['html'], re.S).group(1)
style += '''
html{color-scheme:light;background:#fafbf8}html.embedded{background:transparent;overflow:hidden;scrollbar-gutter:auto}html.embedded body{overflow:hidden}html.embedded,html.embedded body{scrollbar-width:none}html.embedded::-webkit-scrollbar,html.embedded body::-webkit-scrollbar{display:none;width:0;height:0}body{background:transparent}main{padding-bottom:22px}header{max-width:100%}.stage-copy{align-self:center}.diagram-scroll svg{overflow:hidden}.formula{font-weight:400}.formula b{font-weight:500}.formula.long{white-space:normal;font-size:19px}.figure-footer{display:flex;gap:22px;flex-wrap:wrap;margin-top:18px;font-size:12px}.figure-footer a{color:var(--muted);text-underline-offset:4px}.operation-node{fill:#f5f8f8;stroke:#d5dfe0;stroke-width:1}.process{font:17px var(--sans)}.process-small{font:14px var(--sans);fill:var(--muted)}.branch{fill:none;stroke:#8c9291;stroke-width:1.2}.dim-path{opacity:.38}.route,.expert,.term-cell,.gradient-path{transition:opacity .15s}.selected-prob{fill:var(--k)}.index-slot{fill:none;stroke:#98a5a8;stroke-width:1}.chart-stage .diagram-scroll img{display:block;width:100%;height:auto;min-width:520px}.chart-stage .diagram-scroll{max-width:100%}.formula .probability{color:var(--q)}.formula .direct{color:var(--v)}.reset-button{font-size:13px;text-decoration:underline;text-underline-offset:3px}.small-label{font-size:12px;color:var(--muted)}button:disabled{opacity:.4;cursor:default}.controls p{flex:1;min-width:200px}.stage-copy p{word-break:normal}.embedded main{padding:16px 0 12px}.embedded h1{font-size:25px}.embedded header>p{font-size:14px}.embedded header{margin-bottom:20px}.embedded .formula{font-size:18px}.embedded .figure-footer{margin-top:14px}.embedded .standalone-link{display:inline}.standalone-link{display:none}.embedded figcaption{font-size:12px}.embedded .notes{padding:14px}.figure-gallery{max-width:1000px;margin:auto;padding:40px 24px}.figure-gallery h1{font-size:30px}.figure-gallery li{margin:12px 0}.figure-gallery a{color:var(--q)}
@media(max-width:999px){.diagram-scroll svg{min-width:760px}.stage-copy{grid-template-columns:34px 155px 1fr}.stage{gap:14px}.stage p{font-size:14px}.mobile-hint{display:none}}
@media(max-width:759px){.mobile-hint{display:block}.stage-copy{grid-template-columns:30px 1fr}.stage-copy p{grid-column:2}.embedded .formula{font-size:16px}.embedded .controls{gap:12px}.embedded .sharing{flex-basis:100%}.embedded h1{font-size:23px}.notes{grid-template-columns:32px 1fr}.formula.long{font-size:16px}.chart-stage .diagram-scroll img{min-width:540px}}
@media print{body{background:white}.figure-footer{display:none}}
'''
# Slightly larger matrix captions remain legible at the article's ~800 px width.
style += 'svg .shape{font-size:15px}svg .symbol{font-size:20px}svg .annotation{font-size:16px}svg .operation{font-size:17px}'
style += '.formula.long{display:block;max-width:100%;overflow-x:auto;overflow-y:hidden;padding:4px 0}.formula.long>span+span{margin-left:1.2em}'
(BASE/'figure.css').write_text(style)
CSS_REV = hashlib.sha256(style.encode()).hexdigest()[:12]
JS_REV = hashlib.sha256((BASE/'figure.js').read_bytes()).hexdigest()[:12]

def node(x,y,w,h,title,detail='',role=''):
    s=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" class="operation-node"/>'
    s+=text(x+w/2,y+h/2+(0 if detail else 6),title,'process '+role)
    if detail: s+=text(x+w/2,y+h/2+25,detail,'process-small')
    return s

def path(d,cls='branch',tip=True):
    return f'<path d="{d}" class="{cls}"'+(' marker-end="url(#tip)"' if tip else '')+'/>'

def row(n,title,desc,drawing,cls=''):
    return {'n':n,'title':title,'desc':desc,'svg':drawing,'class':cls}

def notes(items):
    return '<dl class="notes">'+''.join(f'<dt>{a}</dt><dd>{b}</dd>' for a,b in items)+'</dl>'

def wrap_text(s,limit):
    lines=[];current='';width=0
    for c in re.sub('<[^>]*>','',s):
        n=2 if unicodedata.east_asian_width(c) in 'WF' else 1
        if width+n>limit:lines.append(current);current='';width=0
        current+=c;width+=n
    if current:lines.append(current)
    return lines

def export_svg(topic,stem,title,subtitle,rows,note_items):
    """Same vector panels plus readable prose, for print and no-JS fallback."""
    out=[f'<style>{style}</style>', '<rect width="1220" height="100%" fill="#fafbf8"/>']
    out.append(text(34,48,escape(title),'','start').replace('<text ','<text style="font-size:28px;font-weight:500" ',1))
    for i,l in enumerate(wrap_text(subtitle,116)):out.append(text(34,81+i*23,escape(l),'shape','start'))
    y=120
    for r in rows:
        image=r['svg']
        if image.startswith('<img'):
            source=re.search(r'src="([^"]+)"',image).group(1)
            image=(ROOT/'source'/source.lstrip('/')).read_text()
            image=image[image.index('<svg'):]
        match=re.search(r'viewBox="([^"]+)"',image)
        box=list(map(float,match.group(1).split()))
        h=box[3]*900/box[2]
        image=re.sub(r'<svg\b[^>]*>',f'<svg x="286" y="{y}" width="900" height="{h}" viewBox="{match.group(1)}">',image,count=1)
        out.append(f'<path d="M34,{y-10}H1186" stroke="#e3e8e8"/>')
        out.append(text(34,y+30,escape(r['n']),'shape','start'))
        out.append(text(34,y+61,escape(r['title']),'operation','start'))
        for i,l in enumerate(wrap_text(r['desc'],25)):out.append(text(34,y+97+i*25,escape(l),'shape','start'))
        out.append(image)
        y+=max(h,165)+35
    for label,value in note_items:
        for i,l in enumerate(wrap_text(label+'：'+value,138)):
            out.append(text(34,y+25,escape(l),'shape','start'));y+=25
        y+=7
    # Marker ids within multiple SVG panels all have identical appearance.
    result=f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1220" height="{y+20}" viewBox="0 0 1220 {y+20}" role="img"><title>{escape(title)}</title>'+''.join(out)+'</svg>'
    dest=ROOT/f'source/images/{topic}/{stem}.svg'
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(result)

def emit(topic,stem,title,subtitle,rows,note_items,formula='',controls='',script='',kind=''):
    slug=f'{topic}/{stem}'
    content=''.join(f'<section class="stage {r["class"]}"><div class="stage-copy"><span class="step">{r["n"]}</span><h2>{r["title"]}</h2><p>{r["desc"]}</p></div><div class="diagram-scroll" tabindex="0" aria-label="{r["title"]}，窄屏可横向滚动">{r["svg"]}</div></section>' for r in rows)
    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title><link rel="stylesheet" href="/figures/figure.css?v={CSS_REV}"><script>if(new URLSearchParams(location.search).has('embed'))document.documentElement.classList.add('embedded');</script></head><body data-kind="{kind}"><main><header><h1>{title}</h1><p>{subtitle}</p>{('<div class="formula long">'+formula+'</div>') if formula else ''}</header><figure class="figure">{controls}<p class="mobile-hint">图示可左右滑动，保留矩阵比例与文字大小。</p>{content}<figcaption>{notes(note_items)}</figcaption></figure><nav class="figure-footer" aria-label="图示文件"><a class="standalone-link" href="/figures/{slug}/" target="_blank" rel="noopener">放大查看</a><a href="/images/{topic}/{stem}.svg" target="_blank" rel="noopener">静态矢量图</a></nav></main><script>{script}</script><script src="/figures/figure.js?v={JS_REV}"></script></body></html>'''
    dest=BASE/slug/'index.html';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(doc)
    export_svg(topic,stem,title,subtitle,rows,note_items)
    FIGURES.append({'topic':topic,'stem':stem,'title':title,'url':f'/figures/{slug}/','svg':f'/images/{topic}/{stem}.svg'})

def controls(buttons,message='',label='查看'):
    return '<div class="controls"><div class="head-picker" role="group" aria-label="'+label+'"><span>'+label+'</span>'+buttons+'</div><p class="sharing" id="sharing" aria-live="polite">'+message+'</p></div>'

def button(value,label,attr='data-i',first=False):
    return f'<button type="button" {attr}="{value}" aria-pressed="{str(first).lower()}">{label}</button>'

TOPIC='transformer-arithmetic'

# Decoder: thin process lanes. Nodes name operations, not tensor faces.
s=text(35,69,'token ID','symbol','start')+arrow(125,175,64)+node(197,29,227,70,'词嵌入查表','W_E：V × d','q')+arrow(455,518,64)+text(609,68,'X','symbol')+text(609,104,'B × T × d','shape')+text(745,69,'输入表示','annotation')
def residual(input_,operation,output,role):
    s=path('M75,100V40H651V81',tip=True)+text(362,19,'残差支路 '+input_,'shape')
    s+=text(56,107,input_,'symbol')+arrow(84,146,100)+node(168,68,146,64,'RMSNorm')+arrow(339,383,100)+node(405,68,158,64,operation,'',role)+arrow(588,627,100)
    s+=f'<circle cx="651" cy="100" r="17" fill="white" stroke="#89999d"/>'+text(651,107,'+','op')+arrow(691,745,100)+text(808,107,output,'symbol')+text(808,141,'B × T × d','shape')
    return svg(s,180,input_+' 经归一化和 '+operation+' 后加回残差，得到 '+output)
end=text(39,65,'Yᴸ','symbol')+arrow(81,132,59)+node(157,28,191,65,'最终 RMSNorm')+arrow(374,427,59)+node(452,18,214,84,'输出头 W_U','d × V','v')+arrow(692,743,59)+text(811,60,'logits','symbol')+text(811,94,'B × T × V','shape')
emit(TOPIC,'01-decoder-architecture','Transformer 的计算流程','以 Pre-RMSNorm 结构为例，展示词嵌入、残差更新与输出投影。',[
 row('01','词嵌入','根据 token ID 查表，得到对应的 d 维向量。',svg(s,150,'token ID 经词嵌入表得到输入 X。')),
 row('02a','Attention 子层','归一化后的表示参与注意力计算，结果与输入 X 相加。',residual('X','Attention','H','q')),
 row('02b','FFN 子层','H 经归一化与前馈变换后，通过残差相加得到 Y。',residual('H','FFN / MoE','Y','k')),
 row('03','词表投影','02a 与 02b 构成一个 Block。经过 L 层后，最终表示映射为词表分数。',svg(end,138,'最后一层输出经过最终归一化和词表输出投影得到 logits。')),
],[('维度','B 为批大小，T 为序列长度，d 为隐藏维度，V 为词表大小。各 Block 的输入输出形状均为 B × T × d。'),('对象','方框表示计算操作。token ID 为离散编号，残差流中的表示为实数向量，logits 为未归一化的词表分数。'),('参数','可学习参数来自 Norm 的缩放、线性投影和嵌入表。残差相加、mask 与本文采用的 RoPE 不引入参数。')])

# Approved QKV visual, now embedded using the shared responsive figure shell.
qrows=[row(*r[:3],r[3]) for r in P['stages']]
qrows[-1]['svg']=qrows[-1]['svg'].replace('class="symbol">[H₁', 'class="symbol" style="font-size:17px">[H₁')
qscript=re.search(r'<script>(.*?)</script>',P['html'],re.S).group(1)
emit(TOPIC,'02-qkvo-tensor-flow','Q、K、V 的投影与注意力计算','以包含 4 个 Query 头和 2 个 KV 头的 GQA 为例，展示各步的张量形状。',qrows,[
 ('维度','B = 1（省略批次轴），T = 3，d = 8，h = 4，h_kv = 2，d_h = 2。每格对应一个元素。'),
 ('对象','W 表示可学习权重，Q、K、V、H、O 表示中间张量。A 的每行是一组注意力概率，和为 1。色块深浅仅作示意。'),
 ('约定','头编号从 1 开始，g = ⌈i / 2⌉。Q、K 在匹配前应用 RoPE；M 在可见位置取 0，在未来位置取 −∞。图中省略偏置与 dropout。')],
 formula='<span><b class="q">Q</b> = XW<sub>Q</sub>　<b class="k">K</b> = XW<sub>K</sub>　<b class="v">V</b> = XW<sub>V</sub></span><span>H<sub>i</sub> = softmax(Q<sub>i</sub>K<sub>g</sub><sup>T</sup> / √d<sub>h</sub> + M)V<sub>g</sub></span>',
 controls=controls(''.join(button(i,'Q'+['','₁','₂','₃','₄'][i],first=i==1) for i in range(1,5)),'Q₁、Q₂ 共享 K₁ / V₁，当前查看 Q₁。','Query 头'),script=qscript,kind='qkv')

# Head identity diagram: equal nodes explicitly denote IDs, not tensor geometry.
sub=['₀','₁','₂','₃','₄']; headrows=[]
for label,n,desc in [('MHA',4,'每个 Query 头使用独立的一组 K/V。'),('GQA',2,'每两个 Query 头共享一组 K/V。'),('MQA',1,'所有 Query 头共享同一组 K/V。')]:
    coords=[125,335,545,755];s=''
    for i,x in enumerate(coords,1):
        group=(i-1)//(4//n)
        dest=sum(coords[group*(4//n):(group+1)*(4//n)])/(4//n)
        s+=f'<g class="route" data-query="{i}">'+path(f'M{x},53V88H{dest}V123')+'</g>'
    for i,x in enumerate(coords,1):
        s+=f'<g class="expert q-node" data-query="{i}">'+node(x-50,12,100,41,'Q'+sub[i],'','q')+'</g>'
    for g in range(n):
        x=sum(coords[g*(4//n):(g+1)*(4//n)])/(4//n)
        s+=f'<g class="expert kv-node" data-group="{g+1}" data-groups="{n}">'+node(x-62,124,124,48,'K'+sub[g+1]+' / V'+sub[g+1],'','k')+'</g>'
    headrows.append(row(str(n)+' 组',label,desc,svg(s,195,label+' 的四个 Query 头和 K/V 共享关系。')))
emit(TOPIC,'03-mha-gqa-mqa','MHA、GQA 与 MQA 的头共享关系','固定 4 个 Query 头，比较三种结构的 K/V 分组方式。选择一个头可高亮对应关系。',headrows,[('维度','固定 h = 4，每个头的宽度为 d_h。图中节点用于标识各头，尺寸不表示张量大小。'),('对象','Q_i 表示第 i 个 Query 头，连接线指向它使用的 K_g / V_g。'),('参数','共享 K/V 会减少其投影参数和缓存。各 Query 头仍分别计算注意力概率与输出。')],controls=controls(button('all','全部',first=True)+''.join(button(i,'Q'+sub[i]) for i in range(1,5)),'图中显示全部头的对应关系。','Query 头'),kind='heads')

# SwiGLU: S=2,d=4,d_ff=6, with a constant 16 px per axis unit.
s=mat(22,30,2,4,'q','X','2 × 4')+text(120,54,'×','op')+mat(162,14,4,6,'k','W₁','4 × 6')+arrow(288,334,46)+text(399,53,'SiLU','operation')+arrow(461,511,46)+mat(550,30,2,6,'k','G','2 × 6')+text(754,53,'门控支路','annotation')
s+=mat(22,187,2,4,'q','X','2 × 4')+text(120,212,'×','op')+mat(162,171,4,6,'v','W₃','4 × 6')+arrow(288,511,203)+mat(550,187,2,6,'v','U','2 × 6')+text(754,210,'数值支路','annotation')
s2=mat(22,23,2,6,'k','G','2 × 6')+text(158,48,'⊙','op')+mat(204,23,2,6,'v','U','2 × 6')+text(341,48,'=','op')+mat(387,23,2,6,'q','Z','2 × 6')+text(681,46,'逐元素相乘保持形状','annotation')
s3=mat(22,45,2,6,'q','Z','2 × 6')+text(158,69,'×','op')+mat(204,13,6,4,'n','W₂','6 × 4')+text(322,69,'=','op')+mat(374,45,2,4,'q','Y','2 × 4')+text(656,69,'输出宽度恢复为 d','annotation')
emit(TOPIC,'06-swiglu','SwiGLU 的门控与投影','输入经过两个独立投影，所得结果逐元素相乘，再映射回隐藏维度。',[
 row('01','并行投影','W₁ 和 W₃ 分别将输入映射到 d_ff 维，门控支路随后应用 SiLU。',svg(s,304,'同一个 X 分别乘 W1 和 W3，门控支路应用 SiLU 得到 G，另一支路得到 U。')),
 row('02','逐元素门控','G 与 U 逐元素相乘，得到相同形状的 Z。',svg(s2,123,'形状相同的 G 和 U 逐元素相乘得到 Z。')),
 row('03','输出投影','W₂ 将中间结果映射回 d 维，与残差流的宽度一致。',svg(s3,185,'Z 乘六行四列的 W2 得到两行四列的 Y。')),
],[('维度','取 T = 2，d = 4，d_ff = 6，并省略批次轴。相同维度对应相同边长。'),('对象','W₁、W₂、W₃ 为权重矩阵，G、U、Z、Y 为中间张量。颜色区分计算支路，深浅仅作示意；⊙ 表示逐元素相乘。'),('参数','三次投影共包含 3 d d_ff 个权重，本例为 72。SiLU 与逐元素相乘均不引入参数。')],formula='Y = [SiLU(XW₁) ⊙ (XW₃)]W₂')

# MoE probabilities, discrete selected IDs and weights are separate objects.
s=''
for i,x in enumerate([120,330,540,750],1):
    p=[.1,.5,.3,.1][i-1]
    s+=text(x,21,'专家 '+str(i),'annotation')
    s+=f'<path d="M{x-54},130h108" stroke="#d5dfe0"/><rect class="prob-bar" data-expert="{i}" x="{x-30}" y="{130-100*p}" width="60" height="{100*p}" rx="2" fill="#8a74b5" fill-opacity=".7"/>'
    s+=text(x,161,f'{p:.2f}','prob-value')+text(x,194,'logit '+f'{__import__("math").log(p):.2f}','logit-value')
s2=text(140,22,'选中编号（整数）','annotation')
for i,x in enumerate([83,157]):s2+=f'<rect x="{x}" y="43" width="52" height="45" rx="3" class="index-slot"/>'+text(x+26,73,str(i+2),'selected-id symbol')
s2+=arrow(241,289,67)+text(411,22,'选中概率','annotation')+text(411,75,'(0.50, 0.30)','selected-values symbol')+arrow(534,583,67)+text(730,22,'选中概率归一化','annotation')+text(730,75,'(0.625, 0.375)','selected-weights symbol')
s3=text(435,22,'输入 x','symbol')
for i,x in enumerate([120,330,540,750],1):
    active=i in [2,3]
    s3+=f'<g class="route expert-route{ " dim-path" if not active else ""}" data-expert="{i}">'+path(f'M435,34V50H{x}V75')+path(f'M{x},127V157H435V178')+'</g>'
for i,x in enumerate([120,330,540,750],1):
    s3+=f'<g class="expert{ " dim-path" if i not in [2,3] else ""}" data-expert="{i}">'+node(x-73,77,146,49,'专家 '+str(i),'','k')+'</g>'
s3+=text(435,214,'y = 0.625 f₂(x) + 0.375 f₃(x)','moe-result symbol')
emit(TOPIC,'04-moe-routing','MoE 的路由与加权汇总','Router 为每个 token 选择 4 个专家中的 2 个，并确定其输出的混合权重。',[
 row('01','路由概率','路由分数 xW_r 经 softmax 转为专家概率，图中以柱高表示。',svg(s,225,'四个专家的概率柱和对应路由分数。')),
 row('02','专家选择','取概率最大的两项，保留专家编号，并将对应概率重新归一化。',svg(s2,118,'选中专家编号二和三，对应概率零点五和零点三，归一化后为零点六二五和零点三七五。')),
 row('03','分发与汇总','输入 x 分发给选中的专家，各专家输出按归一化权重求和。',svg(s3,244,'同一个输入分发到选中的专家，再按归一化权重相加。')),
],[('维度','x 的形状为 1 × d，Router 权重为 d × E，取 E = 4。各专家输出均为 1 × d；图中节点表示专家身份。'),('对象','Top-k 返回按概率降序排列的专家编号，取值为 1 到 4 的整数。选中概率归一化后，两项混合权重之和为 1。'),('计算','示例采用 Mixtral 的选中概率归一化方式，路由分数取 log(p)。本 token 仅在选中的专家中进行前向计算。')],controls=controls(button('a','示例 A','data-moe',True)+button('b','示例 B','data-moe'),'选择专家 2、3，混合权重为 0.625、0.375。','路由结果'),kind='moe')

# Cache panels are regenerated locally when advancing one position.
def cache_panels(T):
    s=''
    for y,role,name in [(20,'k','K'),(188,'v','V')]:
        s+=mat(30,y,T-1,2,role,name+'旧',f'{T-1} × 2')+text(150,y+30,'追加','operation')+mat(239,y,1,2,role,name+'新','1 × 2')+arrow(312,384,y+23)+mat(437,y,T,2,role,name,f'{T} × 2')
        s+=text(680,y+29,'沿位置轴追加当前 K/V','annotation')
    s2=mat(25,20,1,2,'q','q','1 × 2')+text(94,43,'×','op')+mat(133,12,2,T,'k','Kᵀ',f'2 × {T}')+arrow(265,309,28)+text(400,32,'缩放 → softmax','operation')+arrow(490,529,28)+mat(574,20,1,T,'a','a',f'1 × {T}')
    s3=mat(25,45,1,T,'a','a',f'1 × {T}')+text(164,67,'×','op')+mat(211,13,T,2,'v','V',f'{T} × 2')+text(296,67,'=','op')+mat(348,45,1,2,'q','H','1 × 2')+text(652,48,'各头输出拼接后乘 W_O','annotation')+text(652,83,'得到当前 token 的注意力输出','shape')
    return [svg(s,365,'分别在 K 和 V 的位置轴末尾追加当前行。'),svg(s2,114,'当前 Query 乘全量缓存 K 的转置，得到对所有缓存位置的概率。'),svg(s3,173,'注意力概率乘缓存 V，得到当前 token 的头输出。')]
cp=cache_panels(4)
emit(TOPIC,'05-kv-cache','KV Cache 的追加与读取','示例从 3 个 prompt token 的 prefill 开始，展示后续 decode 中缓存的变化。',[
 row('01','更新缓存','取一层中的一组 KV 头，K 和 V 分别沿位置轴追加当前 token 的一行。',cp[0]),
 row('02','读取 Key 缓存','当前 Query 与缓存中的全部 Key 匹配，得到各位置的注意力概率。',cp[1]),
 row('03','汇总 Value 缓存','按注意力概率对缓存中的 Value 加权求和，K/V 继续保留供后续步骤使用。',cp[2]),
],[('维度','取 B = 1、d_h = 2，图中展示一组 KV 头。单层完整 K/V 的形状均为 B × n_kv × T × d_h。'),('对象','缓存包含应用 RoPE 后的 K，以及未经 RoPE 的 V。Q 仅用于当前步骤，色块深浅仅作数值示意。'),('容量','L 层缓存共包含 2 B L n_kv T d_h 个元素。图中展示位置的追加过程，实际实现可通过预分配等方式管理内存。')],controls='<div class="controls"><div class="head-picker"><button type="button" id="cache-next">生成一步</button><button type="button" id="cache-reset" class="reset-button">重置</button></div><p class="sharing" aria-live="polite" id="sharing">缓存长度 T = 4，包含 3 个历史位置和 1 个当前位置。</p></div>',kind='cache')

TOPIC='kl-divergence'

# Scientific panels use Matplotlib coordinates, independent of diagram rendering.
def chart(name,alt):return f'<img src="/images/kl-divergence/{name}.svg" alt="{alt}">'
emit(TOPIC,'01-directions','同一个参考分布，两种 KL 方向','把双峰参考分布 Q，用一个高斯 P 来拟合。改变 KL 的方向，就改变了平均误差的位置。',[
 row('P → Q','Reverse KL','从 P 采样，惩罚 P 把质量放在 Q 很小的区域；这个例子中最优解靠近一个峰。',chart('01-directions-reverse','双峰参考分布和反向 KL 拟合的单峰高斯。'),'chart-stage'),
 row('Q → P','Forward KL','从 Q 采样，P 必须照顾两个峰；单个高斯用更大的方差覆盖它们。',chart('01-directions-forward','双峰参考分布和前向 KL 拟合的宽高斯。'),'chart-stage'),
],[('分布','Q = ½ N(−3, 1) + ½ N(3, 1)；P 限制为一个高斯，横轴为样本取值，纵轴为概率密度。'),('拟合','Reverse KL 取右峰的对称最优解：μ ≈ 2.984，σ ≈ 1.023；Forward KL 为 μ = 0，σ = √10。'),('边界','“选峰 / 覆盖”是这个受限拟合例子的表现，不是任意分布族和优化过程都满足的定理。')],formula='<span>D<sub>reverse</sub> = E<sub>P</sub>[log(P / Q)]</span><span>D<sub>forward</sub> = E<sub>Q</sub>[log(Q / P)]</span>')

emit(TOPIC,'02-estimators','估计值的形状，与估计量的方差','k₁、k₂、k₃ 在原点附近很接近；离开局部区域，它们的数值和方差会显著分离。',[
 row('01','单样本数值','横轴 u = log(P / Q)。k₁ 可以为负；k₂ 非负但有偏；k₃ 非负且有指数尾部。',chart('02-estimators-shape','三个估计量随 log-ratio 变化的曲线。'),'chart-stage'),
 row('02','相对标准差','单位方差高斯中，均值差 δ 增大时，k₃ 的相对标准差也可能迅速升高。',chart('02-estimators-variance','三个估计量的标准差除以真实 KL，随高斯均值差变化。'),'chart-stage'),
],[('定义','u = log(P / Q)，k₁ = u，k₂ = u² / 2，k₃ = exp(−u) − 1 + u。'),('例子','样本取自 P = N(δ, 1)，Q = N(0, 1)，真实 KL = δ² / 2；第二张图显示标准差 / 真实 KL。'),('边界','k₂ 有偏；第二张图只比较标准差，不包含偏差，不是均方误差比较。')])

def gradient_lane(prob=True):
    title='Pθ(y)' if prob else 'kθ(y)'
    term='kθ(y) ∇θ log Pθ(y)' if prob else '∇θ kθ(y)'
    s=text(45,65,'θ','symbol')+arrow(85,146,59)+node(169,25,170,68,title,'','q' if prob else 'v')+arrow(367,424,59)+text(610,64,term,'symbol')
    return svg(s,127,'参数通过'+('采样概率' if prob else '固定样本公式')+'的变化贡献梯度。')
emit(TOPIC,'03-gradient-paths','一个期望的梯度，有两条路径','参数 θ 既影响回答出现的概率，也影响同一条回答上的公式。',[
 row('01','概率路径','哪些回答更常出现？这项由 policy gradient 计算，k 在这一项中停止梯度。',gradient_lane(True),'gradient-path probability-path'),
 row('02','公式路径','固定已经采到的回答，k 的值怎样变化？直接把 k 放进 loss 反传，计算的是这一项。',gradient_lane(False),'gradient-path direct-path'),
],[('对象','Pθ(y) 是采样概率，kθ(y) 是样本上的实数函数；参考分布视为固定。'),('机制','完整求导要对两项一起取期望。直接对固定样本反传，不会自动补上采样概率变化的贡献。'),('符号','图示为 E[k] 的梯度。最小化 KL 时，策略奖励中的相应惩罚写为 −β stop-gradient(k)。')],formula='∇<sub>θ</sub> E<sub>Pθ</sub>[k<sub>θ</sub>] = E<sub>Pθ</sub>[<span class="probability">k<sub>θ</sub> ∇<sub>θ</sub> log P<sub>θ</sub></span> + <span class="direct">∇<sub>θ</sub> k<sub>θ</sub></span>]',controls=controls(button('all','完整梯度','data-path',True)+button('prob','概率路径','data-path')+button('direct','直接反传','data-path'),'完整梯度包含两项。','查看路径'),kind='gradient')

# Upper triangle is exact causal reward-to-go support; diagonal is local only.
s=''
for left,full in [(100,True),(568,False)]:
    s+=text(left+108,26,'序列 KL：当前 + 未来' if full else '局部 token KL：只看当前','operation')
    for j in range(4):s+=text(left+j*54+27,69,'u'+sub[j+1],'symbol')
    for i in range(4):
        s+=text(left-25,112+i*54,'g'+sub[i+1],'symbol')
        for j in range(4):
            active=j>=i if full else i==j
            if active:
                s+=f'<g class="term-cell" data-row="{i+1}"><rect x="{left+j*54+2}" y="{83+i*54+2}" width="50" height="50" rx="2" fill="{"#4f8fa5" if full else "#8a74b5"}" fill-opacity=".82"/>'+text(left+j*54+27,114+i*54,'u'+sub[j+1]+'g'+sub[i+1],'term-label')+'</g>'
            else:s+=f'<rect x="{left+j*54+2}" y="{83+i*54+2}" width="50" height="50" rx="2" fill="none" stroke="#e3e8e8"/>'
    s+=text(left+108,334,'列：哪个位置的 log-ratio','shape')
s=s.replace('class="term-label"','class="term-label" style="fill:white;font-size:17px"')
emit(TOPIC,'04-future-kl','当前的选择，也会改变未来的 KL','行表示更新哪个位置的概率，列表示使用哪个位置的 log-ratio。选择一行看它包含哪些项。',[
 row('t × j','梯度项的结构','白格表示期望中不保留的项，着色格表示保留的 u_j g_t。',svg(s,366,'左侧为包含当前和未来项的四乘四上三角，右侧为只保留当前项的四乘四对角阵。')),
],[('定义','u_j = log[Pθ(y_j | h_j) / Q(y_j | h_j)]；g_t = ∇θ log Pθ(y_t | h_t)。'),('机制','过去项 j < t 的期望为 0；未来项 j > t 一般不能删，因为当前选择会改变后续前缀。'),('边界','图示为期望梯度的组成关系，不表示两种采样估计在每条轨迹上数值相同。')],formula='∇D<sub>seq</sub> = E[∑<sub>t</sub> (∑<sub>j≥t</sub> u<sub>j</sub>) g<sub>t</sub>]',controls=controls(button('all','全部','data-row-select',True)+''.join(button(i,'g'+sub[i],'data-row-select') for i in range(1,5)),'完整序列梯度保留上三角；固定前缀的局部梯度只保留对角项。','梯度位置'),kind='future')

(BASE/'manifest.json').write_text(json.dumps(FIGURES,ensure_ascii=False,indent=2))
gallery=''.join(f'<li><a href="{f["url"]}">{f["title"]}</a></li>' for f in FIGURES)
(BASE/'index.html').write_text(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>两篇文章的图示</title><link rel="stylesheet" href="/figures/figure.css?v={CSS_REV}"><body><main class="figure-gallery"><h1>两篇文章的图示</h1><p>6 张 Transformer 图与 4 张 KL 图。点击标题可单独查看。</p><ol>{gallery}</ol></main></body></html>')
print(f'Generated {len(FIGURES)} HTML figures and SVG fallbacks.')
