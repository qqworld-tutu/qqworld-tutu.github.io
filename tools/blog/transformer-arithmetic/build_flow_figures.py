"""Build editable draw.io diagrams and SVG previews from one geometry model.

Run with Python 3. Both exports use the same boxes, labels and connector points.
SVGs are generated here, not by the draw.io application. After editing a .drawio
file manually, export its replacement SVG from draw.io instead of rerunning this.
"""
from pathlib import Path
from html import escape
import json
import re
import xml.etree.ElementTree as ET

DEST = Path(__file__).resolve().parents[3] / 'source/images/transformer-arithmetic'
COLORS = {'blue': '#e3eff4', 'orange': '#f9e7d8', 'purple': '#eee8f5',
          'green': '#e4f0ea', 'gray': '#f2f3f5', 'white': '#ffffff'}


def math_subscripts(label, svg=False):
    safe = escape(label)
    if svg:
        return re.sub(r'\b([A-Za-z])_([A-Za-z]+)\b', r'\1<tspan baseline-shift="sub" font-size="75%">\2</tspan>', safe)
    return re.sub(r'\b([A-Za-z])_([A-Za-z]+)\b', r'\1<sub>\2</sub>', safe)


class Diagram:
    def __init__(self, stem, height, title, subtitle):
        self.stem, self.height = stem, height
        self.nodes, self.edges = [], []
        self.box('title', 20, 14, 760, 32, title, plain=True, size=24)
        self.box('subtitle', 20, 52, 760, 28, subtitle, plain=True, size=17)

    def box(self, name, x, y, w, h, label, color='gray', plain=False, size=18, dashed=False):
        self.nodes.append(dict(id=name, x=x, y=y, w=w, h=h, label=label,
                               fill=COLORS[color], plain=plain, size=size, dashed=dashed))

    def edge(self, *points, dashed=False):
        self.edges.append(dict(points=points, dashed=dashed))

    def save(self):
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="{self.height}" viewBox="0 0 800 {self.height}">',
               '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8" fill="none" stroke="#536473" stroke-width="1.5"/></marker></defs>',
               '<rect width="100%" height="100%" fill="white"/>']
        mx = ET.Element('mxfile', host='app.diagrams.net', type='device')
        page = ET.SubElement(mx, 'diagram', name=self.stem, id=self.stem)
        graph = ET.SubElement(page, 'mxGraphModel', page='1', pageWidth='800', pageHeight=str(self.height), math='0')
        root = ET.SubElement(graph, 'root')
        ET.SubElement(root, 'mxCell', id='0')
        ET.SubElement(root, 'mxCell', id='1', parent='0')
        for i, edge in enumerate(self.edges):
            pts = edge['points']
            dash = ' stroke-dasharray="5 5"' if edge['dashed'] else ''
            svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in pts)}" fill="none" stroke="#536473" stroke-width="1.7" marker-end="url(#arrow)"{dash}/>')
            cell = ET.SubElement(root, 'mxCell', id=f'edge{i}', parent='1', edge='1',
                                 style=f'endArrow=open;endFill=0;strokeColor=#536473;strokeWidth=1.7;rounded=0;dashed={int(edge["dashed"])};')
            # Attach endpoints to matching node borders for editing in draw.io.
            for point, attr, port in [(pts[0],'source','exit'),(pts[-1],'target','entry')]:
                x,y = point
                for n in self.nodes:
                    if n['plain'] or n['dashed']:
                        continue
                    inside = n['x'] <= x <= n['x']+n['w'] and n['y'] <= y <= n['y']+n['h']
                    border = x in (n['x'],n['x']+n['w']) or y in (n['y'],n['y']+n['h'])
                    if inside and border:
                        cell.set(attr,n['id'])
                        cell.set('style',cell.get('style')+f'{port}X={(x-n["x"])/n["w"]};{port}Y={(y-n["y"])/n["h"]};{port}Perimeter=0;')
                        break
            geo = ET.SubElement(cell, 'mxGeometry', relative='1', **{'as':'geometry'})
            ET.SubElement(geo, 'mxPoint', x=str(pts[0][0]), y=str(pts[0][1]), **{'as':'sourcePoint'})
            ET.SubElement(geo, 'mxPoint', x=str(pts[-1][0]), y=str(pts[-1][1]), **{'as':'targetPoint'})
            if len(pts) > 2:
                arr = ET.SubElement(geo, 'Array', **{'as':'points'})
                for x,y in pts[1:-1]:
                    ET.SubElement(arr, 'mxPoint', x=str(x), y=str(y))
        for node in self.nodes:
            n = node
            if not n['plain']:
                dash = ' stroke-dasharray="6 5"' if n['dashed'] else ''
                fill = 'none' if n['dashed'] else n['fill']
                svg.append(f'<rect x="{n["x"]}" y="{n["y"]}" width="{n["w"]}" height="{n["h"]}" rx="8" fill="{fill}" stroke="#b6c1c9"{dash}/>')
            lines = n['label'].split('\n')
            center = n['y'] + n['h']/2
            for j,line in enumerate(lines):
                baseline = center + (j-(len(lines)-1)/2)*25 + n['size']*.35
                svg.append(f'<text x="{n["x"]+n["w"]/2}" y="{baseline}" text-anchor="middle" font-family="PingFang SC,Noto Sans CJK SC,Arial,sans-serif" font-size="{n["size"]}" fill="#253747">{math_subscripts(line,svg=True)}</text>')
            style = f'rounded=1;whiteSpace=wrap;html=1;fontFamily=Helvetica;fontSize={n["size"]};fontColor=#253747;fillColor={n["fill"]};strokeColor=#b6c1c9;dashed={int(n["dashed"])};'
            if n['dashed']:
                style += 'fillColor=none;'
            if n['plain']:
                style += 'fillColor=none;strokeColor=none;'
            cell = ET.SubElement(root, 'mxCell', id=n['id'], value='<br>'.join(math_subscripts(x) for x in lines), parent='1', vertex='1', style=style)
            ET.SubElement(cell, 'mxGeometry', x=str(n['x']), y=str(n['y']), width=str(n['w']), height=str(n['h']), **{'as':'geometry'})
        svg.append('</svg>')
        DEST.mkdir(parents=True, exist_ok=True)
        (DEST / f'{self.stem}.svg').write_text('\n'.join(svg), encoding='utf8')
        ET.ElementTree(mx).write(DEST / f'{self.stem}.drawio', encoding='utf-8', xml_declaration=True)
        return {'stem':self.stem, 'nodes':self.nodes, 'edges':self.edges}


all_diagrams = []
d = Diagram('01-decoder-architecture', 850, 'Decoder-only：先看一层，再堆叠 L 层', '主线始终保留每个 token 的 d 维表示')
d.box('ids', 50, 108, 180, 54, 'token ID')
d.box('embed', 310, 108, 340, 54, '查词嵌入表 W_E [V, d]', 'blue')
d.edge((230,135),(310,135))
d.box('scope', 210, 195, 500, 460, '', 'white', dashed=True)
d.box('repeat', 480, 208, 210, 30, 'Transformer Block × L', plain=True, size=17)
d.box('x', 310, 240, 340, 42, 'X：本层输入', 'white')
d.edge((480,162),(480,240))
d.box('norm1',310,310,340,46,'RMSNorm 1','gray')
d.box('attn',310,384,340,50,'Attention：Q / K / V → O','blue')
d.box('add1',310,462,340,44,'相加：H = X + Attention 输出','green')
d.box('norm2',310,534,150,46,'RMSNorm 2','gray')
d.box('ffn',500,534,150,46,'FFN / MoE','orange')
d.box('add2',310,608,340,34,'相加：Y = H + FFN 输出','green',size=17)
d.edge((480,282),(480,310)); d.edge((480,356),(480,384)); d.edge((480,434),(480,462))
d.edge((310,261),(250,261),(250,484),(310,484))
d.box('res1',55,335,140,60,'第一条残差\n保留 X',plain=True)
d.edge((385,506),(385,534)); d.edge((460,557),(500,557)); d.edge((575,580),(575,608))
d.edge((335,506),(335,520),(275,520),(275,625),(310,625))
d.box('res2',55,535,140,60,'第二条残差\n保留 H',plain=True)
d.box('normfinal',310,695,340,46,'最终 RMSNorm','gray')
d.box('head',310,777,340,48,'输出头 W_U [d, V] → logits','purple')
d.edge((480,642),(480,695)); d.edge((480,741),(480,777))
d.box('next',45,687,210,68,'Y 作为下一层输入\n最后一层后走向输出头',plain=True,size=16)
all_diagrams.append(d.save())

d = Diagram('03-mha-gqa-mqa', 636, '头数的差别，只在 K/V 的共享方式', '固定 h = 4；每个 Query 头都处理全部 token')
for row,(label,groups) in enumerate([('MHA：4 组 K/V',[[0],[1],[2],[3]]),('GQA：2 组 K/V',[[0,1],[2,3]]),('MQA：1 组 K/V',[[0,1,2,3]])]):
    y=120+row*145
    d.box(f'l{row}',15,y,215,55,label,plain=True,size=18)
    for q in range(4):
        x=260+q*130
        d.box(f'q{row}{q}',x,y,100,42,f'Q{q+1}','blue')
    for j,group in enumerate(groups):
        center=sum(310+q*130 for q in group)/len(group)
        d.box(f'kv{row}{j}',center-50,y+79,100,42,f'K{j+1} / V{j+1}','orange')
        for q in group:
            qx=310+q*130
            d.edge((qx,y+42),(qx,y+60),(center,y+60),(center,y+79))
d.box('note',45,554,710,62,'箭头表示使用对应的 K/V，不是把多个 Query 头合并。\n方框表示头的身份，不表示矩阵大小；每个头宽度都为 d_h。',plain=True,size=17)
all_diagrams.append(d.save())

d = Diagram('04-moe-routing', 890, 'MoE：一个 token，只进入选中的专家', '示例 E = 4、k = 2；采用 Mixtral 式概率归一化')
d.box('x',30,108,150,65,'输入 x\n[1, d]','blue')
d.box('router',225,108,225,65,'Router：x W_r\nW_r [d, E]','purple')
d.box('prob',500,108,265,65,'softmax → 专家概率\n[0.10, 0.50, 0.30, 0.10]','purple',size=17)
d.edge((180,140),(225,140));d.edge((450,140),(500,140))
d.box('topk',500,220,265,54,'Top-k：取概率最大的 2 项','gray')
d.edge((632,173),(632,220))
d.box('ids',225,315,225,60,'选中编号：(2, 3)\n整数，取值 1 到 4','blue')
d.box('values',500,315,265,60,'选中概率\n(0.50, 0.30)','purple')
d.edge((580,274),(580,294),(337,294),(337,315))
d.edge((665,274),(665,315))
d.box('weights',500,440,265,70,'选中概率重新归一化\nα₂ = 0.625，α₃ = 0.375','purple')
d.edge((632,375),(632,440))
d.box('dispatch',30,440,290,70,'Dispatch：按编号 (2, 3)\n把同一个 x 送给专家 2、3','blue')
d.edge((105,173),(105,440));d.edge((337,375),(337,407),(175,407),(175,440))
for j in range(4):
    x=30+j*190
    active=j in (1,2)
    d.box(f'exp{j}',x,580,165,65,f'专家 {j+1}：SwiGLU\n'+(f'计算 f{j+1}(x)' if active else '本 token 不调用'), 'orange' if active else 'gray', size=17)
d.edge((175,510),(175,550),(302,550),(302,580))
d.edge((175,550),(492,550),(492,580))
d.box('gather',210,715,380,74,'Gather：按权重相加\ny = 0.625 f₂(x) + 0.375 f₃(x)','green')
d.edge((302,645),(302,715));d.edge((492,645),(492,715))
d.edge((705,510),(705,535),(788,535),(788,752),(590,752))
d.box('note',40,834,720,35,'编号决定送给谁；权重决定各专家输出占多少。未选中专家不参与本次计算。',plain=True,size=16)
all_diagrams.append(d.save())

d = Diagram('05-kv-cache', 950, 'KV Cache：每层各存一份 K 和 V', '先处理 prompt；之后每次只计算当前 token')
d.box('prefill',35,110,730,72,'Prefill：逐层处理 prompt；每层并行计算 S 个位置的 Q/K/V\n缓存各层的 K、V；末位置 logits 给出第一个生成 token','blue')
d.box('step',30,205,740,35,'下面放大某一层的一步 decode：缓存长度从 T − 1 变成 T',plain=True,size=19)
d.box('x',260,270,280,50,'当前 token 的归一化输入','blue')
d.box('q',35,365,210,64,'Q 投影 + RoPE\n当前 q [B, h, 1, d_h]','blue',size=17)
d.box('k',295,365,210,64,'K 投影 + RoPE\n新 k [B, n_kv, 1, d_h]','orange',size=17)
d.box('v',555,365,210,64,'V 投影\n新 v [B, n_kv, 1, d_h]','orange',size=17)
d.edge((400,320),(400,342),(140,342),(140,365));d.edge((400,342),(400,365));d.edge((400,342),(660,342),(660,365))
d.box('kc',295,482,210,82,'追加到旧 K 后\nK [B, n_kv, T, d_h]\n旧 T − 1 + 新 1','orange',size=17)
d.box('vc',555,482,210,82,'追加到旧 V 后\nV [B, n_kv, T, d_h]\n旧 T − 1 + 新 1','orange',size=17)
d.edge((400,429),(400,482));d.edge((660,429),(660,482))
d.box('scores',35,620,470,60,'q Kᵀ / √d_h → softmax\n每个 Query 头得到 [1, T] 的概率','purple')
d.edge((140,429),(140,620));d.edge((400,564),(400,620))
d.box('av',230,739,535,60,'概率 × V → 各头输出 → 拼接 → W_O\n本层 Attention 输出 [B, 1, d]','green')
d.edge((270,680),(270,739));d.edge((660,564),(660,739))
d.box('end',135,857,630,65,'继续：残差、FFN、后续层 → 最终 Norm 与输出头\n得到下一个 token 的 logits；每层缓存都保留到下一步','gray',size=17)
d.edge((495,799),(495,857))
all_diagrams.append(d.save())

(DEST / 'flow-geometry.json').write_text(json.dumps(all_diagrams,ensure_ascii=False,indent=2),encoding='utf8')
print('Generated 4 SVG previews, 4 editable draw.io files, and geometry ledger.')
