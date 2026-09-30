"""Scientific chart panels for the HTML figures; same distributions and formulas.

Requires NumPy, SciPy and Matplotlib. Fits the reverse KL again, rather than
tracing the old image. Also writes the fit and analytic variance values for QA.
"""
from pathlib import Path
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import minimize
from scipy.special import roots_hermitenorm, logsumexp

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'source/images/kl-divergence'
OUT.mkdir(parents=True,exist_ok=True)
fontpath=os.environ.get('BLOG_CJK_FONT','/System/Library/Fonts/STHeiti Light.ttc')
font_manager.fontManager.addfont(fontpath)
font=font_manager.FontProperties(fname=fontpath).get_name()
plt.rcParams.update({'font.family':[font,'DejaVu Sans'],'font.size':14,
 'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#b7c2c5','axes.linewidth':.7,'axes.labelcolor':'#68767a',
 'text.color':'#293b40','xtick.color':'#68767a','ytick.color':'#68767a',
 'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
 'svg.fonttype':'path','svg.hashsalt':'technical-article-editorial-v1'})
Q,K,V,N='#4f8fa5','#ca8b4f','#8a74b5','#899397'

def normal_log(x,mu,sd):return -.5*((x-mu)/sd)**2-np.log(sd)-.5*np.log(2*np.pi)
def qlog(x):return logsumexp(np.array([normal_log(x,-3,1),normal_log(x,3,1)]),axis=0)-np.log(2)
nodes,weights=roots_hermitenorm(160);weights/=np.sqrt(2*np.pi)
def reverse(z):
    mu,logsd=z;sd=np.exp(logsd);x=mu+sd*nodes
    return weights@(normal_log(x,mu,sd)-qlog(x))
best=min((minimize(reverse,[mu,ls],bounds=[(-5,5),(-2,2)]) for mu in [-3,0,3] for ls in [0,1]),key=lambda r:r.fun)
mu,sd=abs(float(best.x[0])),float(np.exp(best.x[1]))
assert np.isclose(mu,2.98430726668006,atol=1e-3)

def canvas():
    fig,ax=plt.subplots(figsize=(9,3.6),layout='constrained')
    ax.tick_params(length=3,width=.6,labelsize=12)
    ax.grid(axis='y',color='#e3e8e8',linewidth=.65,zorder=0)
    return fig,ax

def save(fig,name):
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',pad_inches=.14,metadata={'Date':None})
    plt.close(fig)

x=np.linspace(-9,9,2001);q=np.exp(qlog(x))
for forward,name in [(False,'01-directions-reverse'),(True,'01-directions-forward')]:
    fig,ax=canvas()
    ax.fill_between(x,q,color='#f0f3f4',zorder=1)
    ax.plot(x,q,color=N,lw=1.7,ls=(0,(4,3)),label='参考 Q：双峰',zorder=2)
    ax.plot(x,np.exp(normal_log(x,0 if forward else mu,np.sqrt(10) if forward else sd)),color=V if forward else Q,lw=2.2,label='拟合 P：单个高斯',zorder=3)
    ax.set(xlim=(-9,9),ylim=(0,.43),xlabel='样本取值',ylabel='概率密度')
    ax.set_xticks([-9,-6,-3,0,3,6,9]);ax.set_yticks([0,.1,.2,.3,.4])
    ax.legend(frameon=False,loc='upper left',ncols=2,fontsize=12,handlelength=2.7,columnspacing=2)
    save(fig,name)

fig,ax=canvas();u=np.linspace(-2,2,600)
for y,color,label,ls in [(u,N,'k₁ = u','--'),(.5*u*u,Q,'k₂ = u² / 2','-'),(np.expm1(-u)+u,V,'k₃ = exp(−u) − 1 + u','-')]:
    ax.plot(u,y,color=color,lw=2.1,label=label,ls=ls)
ax.axhline(0,color='#98a5a8',lw=.8);ax.set(xlim=(-2,2),ylim=(-2.2,4.7),xlabel='u = log(P / Q)',ylabel='单样本估计值')
ax.set_xticks([-2,-1,0,1,2]);ax.legend(frameon=False,fontsize=12,loc='upper right')
save(fig,'02-estimators-shape')

fig,ax=canvas();d=np.linspace(.1,2,500);kl=.5*d*d
v1=2/d;v2=np.sqrt(.5*d**4+.25*d**6)/kl;v3=np.sqrt(np.expm1(d*d)-d*d)/kl
for yy,c,label,ls in [(v1,N,'k₁','--'),(v2,Q,'k₂（有偏）','-'),(v3,V,'k₃','-')]:ax.plot(d,yy,color=c,lw=2.1,label=label,ls=ls)
ax.set(xlim=(.1,2),ylim=(0,8),xlabel='均值差 δ',ylabel='单样本标准差 / 真实 KL')
ax.legend(frameon=False,loc='upper center',ncols=3,fontsize=12)
save(fig,'02-estimators-variance')
report={'reverse_fit':{'mean':mu,'std':sd,'KL':float(best.fun)},'relative_std_at_delta_2':{'k1':float(v1[-1]),'k2':float(v2[-1]),'k3':float(v3[-1])}}
(OUT/'editorial-chart-checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
