"""Reproducible figures and exact checks for the KL blog. Run with numpy/scipy/matplotlib."""
from pathlib import Path
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.optimize import minimize
from scipy.special import logsumexp, roots_hermitenorm

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "source/images/kl-divergence"
OUT.mkdir(parents=True, exist_ok=True)
font_path = os.environ.get("BLOG_CJK_FONT", "/System/Library/Fonts/STHeiti Light.ttc")
if Path(font_path).is_file():
    font_manager.fontManager.addfont(font_path)
    font = font_manager.FontProperties(fname=font_path).get_name()
else:
    font = "Noto Sans CJK SC"
plt.rcParams.update({"font.family": [font, "DejaVu Sans"], "font.size": 12,
 "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
 "axes.edgecolor": "#cbd5df", "axes.labelcolor": "#46546b", "text.color": "#17273d",
 "xtick.color": "#627084", "ytick.color": "#627084", "figure.facecolor": "#ffffff",
 "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff", "svg.fonttype": "path"})
BLUE, TEAL, ORANGE, GRAY = "#315cce", "#188478", "#c56b26", "#99a5b6"
def save(fig, name):
    fig.savefig(OUT / (name+".svg"), bbox_inches="tight", pad_inches=.2)
    fig.savefig(OUT / (name+".png"), dpi=180, bbox_inches="tight", pad_inches=.2)
    plt.close(fig)

def normal_log(x, mu, sd):
    return -.5*((x-mu)/sd)**2-np.log(sd)-.5*np.log(2*np.pi)
def qlog(x):
    return logsumexp(np.stack([normal_log(x,-3,1),normal_log(x,3,1)]),axis=0)-np.log(2)
nodes, weights = roots_hermitenorm(160)
weights /= np.sqrt(2*np.pi)
def reverse(z):
    mu, lsd=z
    sd=np.exp(lsd)
    x=mu+sd*nodes
    return float(weights@(normal_log(x,mu,sd)-qlog(x)))
solutions=[minimize(reverse,[mu,ls],bounds=[(-5,5),(-2,2)])
           for mu in [-3,0,3] for ls in [0,1]]
best=min(solutions,key=lambda s:s.fun)
mu, sd=abs(float(best.x[0])),float(np.exp(best.x[1]))
x=np.linspace(-9,9,2001)
q=np.exp(qlog(x))
fig, axes=plt.subplots(1,2,figsize=(11.6,4.2),layout="constrained")
for ax in axes:
    ax.fill_between(x,q,color="#e9edf3")
    ax.plot(x,q,color=GRAY,lw=2,label="参考分布 Q：两个峰")
    ax.set(xlim=(-9,9),ylim=(0,.42),xlabel="样本取值",ylabel="概率密度")
    ax.grid(axis="y",alpha=.12)
axes[0].plot(x,np.exp(normal_log(x,mu,sd)),color=BLUE,lw=2.6,label="当前分布 P：单个高斯")
axes[0].set_title("Reverse KL：倾向于选中一个峰",loc="left",pad=18,fontweight="bold")
axes[1].plot(x,np.exp(normal_log(x,0,np.sqrt(10))),color=TEAL,lw=2.6,label="当前分布 P：单个高斯")
axes[1].set_title("Forward KL：同时覆盖两个峰",loc="left",pad=18,fontweight="bold")
for ax in axes: ax.legend(loc="upper left",frameon=False,fontsize=10)
save(fig,"01-directions")

fig, axes=plt.subplots(1,2,figsize=(11.6,4.2),layout="constrained")
u=np.linspace(-2,2,500)
for yy,c,l in [(u,GRAY,r"$k_1=u$"),(.5*u*u,BLUE,r"$k_2=u^2/2$"),(np.expm1(-u)+u,TEAL,r"$k_3=e^{-u}-1+u$")]:
    axes[0].plot(u,yy,color=c,lw=2.4,label=l)
axes[0].axhline(0,color=GRAY,lw=.7)
axes[0].set(xlabel=r"$u=\log(P/Q)$",ylabel="单样本估计值",ylim=(-2.2,4.7))
axes[0].set_title("数值形状：负值、平方与指数尾部",loc="left",pad=18,fontweight="bold")
axes[0].legend(frameon=False,fontsize=11)
d=np.linspace(.1,2,400)
kl=.5*d*d
axes[1].plot(d,2/d,color=GRAY,lw=2.4,label=r"$k_1$")
axes[1].plot(d,np.sqrt(.5*d**4+.25*d**6)/kl,color=BLUE,lw=2.4,label=r"$k_2$（有偏）")
axes[1].plot(d,np.sqrt(np.expm1(d*d)-d*d)/kl,color=TEAL,lw=2.4,label=r"$k_3$")
axes[1].set(xlabel="单位方差高斯的均值差 δ",ylabel="单样本标准差 / 真实 KL",ylim=(0,8))
axes[1].set_title("高斯例子：k₃ 的方差也会变大",loc="left",pad=18,fontweight="bold")
axes[1].legend(frameon=False,fontsize=11)
for ax in axes: ax.grid(alpha=.14)
save(fig,"02-estimators")

fig,ax=plt.subplots(figsize=(11.6,5))
ax.set(xlim=(0,12),ylim=(0,5)); ax.axis("off")
def box(x,y,w,h,title,body,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=.13",facecolor=color,edgecolor="none"))
    ax.text(x+w/2,y+h*.7,title,ha="center",va="center",fontsize=14,fontweight="bold")
    ax.text(x+w/2,y+h*.32,body,ha="center",va="center",fontsize=12,linespacing=1.65)
box(.15,1.8,2.8,1.35,"更新参数 θ","分布与样本上的公式\n都会随之改变","#edf1f8")
box(4.05,2.9,3.3,1.45,"采样概率 Pθ(y) 改变","哪些回答更常出现？\n"+r"$k_\theta(y)\,\nabla\log P_\theta(y)$","#eaf0ff")
box(4.05,.55,3.3,1.45,"固定回答的 kθ(y) 改变","同一条回答怎样重新打分？\n"+r"$\nabla k_\theta(y)$","#e4f3ef")
box(8.3,2.9,3.5,1.45,"经由 policy gradient","把停止梯度的 k 放进奖励\n计算概率变化的贡献","#eaf0ff")
box(8.3,.55,3.5,1.45,"经由普通反向传播","把 k 直接放进 loss\n计算公式变化的贡献","#e4f3ef")
for a,b,c in [((2.95,2.7),(4,3.6),BLUE),((2.95,2.2),(4,1.25),TEAL),((7.4,3.6),(8.25,3.6),BLUE),((7.4,1.25),(8.25,1.25),TEAL)]:
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=18,lw=1.7,color=c))
ax.text(.15,4.75,"期望求导需要两项；普通反向传播只计算固定样本上的那一项",fontsize=15,fontweight="bold")
save(fig,"03-gradient-paths")

fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout="constrained")
for ax,mat,title,c in [(axes[0],np.triu(np.ones((4,4))),"序列 KL：当前 + 未来",BLUE),
                       (axes[1],np.eye(4),"局部 token KL：只看当前",TEAL)]:
    from matplotlib.colors import ListedColormap
    ax.imshow(mat,cmap=ListedColormap(["#f0f3f7",c]),vmin=0,vmax=1)
    ax.set_xticks(range(4),[r"$u_1$",r"$u_2$",r"$u_3$",r"$u_4$"])
    ax.set_yticks(range(4),[r"$g_1$",r"$g_2$",r"$g_3$",r"$g_4$"])
    ax.set_xlabel("哪个 token 的 log-ratio",labelpad=12)
    ax.set_ylabel("更新哪个 token 的概率",labelpad=10)
    ax.set_title(title,loc="left",pad=18,fontweight="bold")
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True)
    ax.grid(which="minor",color="white",linewidth=4)
    ax.tick_params(which="both",length=0)
    for i in range(4):
        for j in range(4):
            if mat[i,j]:ax.text(j,i,r"$u_{%d}\,g_{%d}$"%(j+1,i+1),ha="center",va="center",color="white",fontsize=14)
    for spine in ax.spines.values():spine.set_visible(False)
save(fig,"04-future-kl")

# Exact enumeration: no Monte Carlo noise in these checks.
p=np.array([[.5,.5],[.5,.5],[.01,.99]])
q=np.array([[.25,.75],[.5,.5],[.99,.01]])
u=np.log(p/q); cond=(p*u).sum(1)
mass=(p[0,:,None]*p[1:]).ravel()
mc=(u[0,:,None]+u[1:]).ravel()
rb=np.repeat(cond[0]+cond[1:],2)
mean=float(mass@mc)
report={"reverse_fit":{"mean":mu,"std":sd,"kl":float(best.fun)},
        "rb_counterexample":{"P":p.tolist(),"Q":q.tolist(),"mean_MC":mean,"mean_RB":float(mass@rb),
          "var_MC":float(mass@(mc-mean)**2),"var_RB":float(mass@(rb-mean)**2)}}
assert np.isclose(mass@mc,mass@rb)
assert report["rb_counterexample"]["var_RB"]>report["rb_counterexample"]["var_MC"]
p=np.array([.8,.2]);q=np.array([.5,.5]);u=np.log(p/q);w=q/p
score=np.array([1/p[0],-1/p[1]]) # derivative w.r.t. p[0]
exact=np.log(p[0]/q[0])-np.log(p[1]/q[1])
assert np.isclose(p@(u*score),exact)
assert np.isclose(p@score,0)
forward=-q@score
assert np.isclose(p@((1-w)*score),forward)
# Correct tail-gradient compensation for a fixed head S={0}.
tail=np.array([0,1])
head_grad=p[0]*(u[0]+1)*score[0]
tail_grad=p@(tail*(u+1)*score)
assert np.isclose(head_grad+tail_grad,exact)
report["two_outcome"]={"KL":float(p@u),"reverse_gradient":float(exact),"forward_gradient":float(forward)}
naive_topk=head_grad+p@(tail*u*score)
assert np.isclose(naive_topk-exact,1.0)
report["topk_gradient_check"]={"exact":float(exact),"masked_k4_plus_exact_head":float(naive_topk),"corrected":float(head_grad+tail_grad)}
report["checks"]="Exact KL gradients, head/tail gradient and RB counterexample verified."
(OUT/"checks.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
