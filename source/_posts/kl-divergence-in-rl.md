---
title: 大模型强化学习中的 KL 散度与梯度估计
date: 2026-09-29 12:01:00
permalink: llm/kl-divergence-in-rl/
description: 从二元分布出发，讨论 KL 的采样估计、奖励修正与直接反传，再推导自回归序列中的梯度关系，并用可枚举的例子核对结果。
categories:
  - 大模型
tags:
  - 强化学习
  - KL散度
  - PPO
  - GRPO
technical_article: true
---

KL 散度是大模型强化学习中常见的正则项。我们希望模型提高任务奖励，同时保留参考模型的部分行为，因此通常会在奖励目标中加入对分布偏离的约束。不过，从目标函数到训练代码，还有几处需要区分的细节。同样使用当前模型与参考模型的对数概率，有的实现将它们作为奖励修正，有的则直接构造可微的损失，两者未必对应相同的参数更新。

其中一个容易忽略的原因是，KL 本身是关于概率分布的期望。模型参数变化时，样本出现的概率也随之变化，而对一批已经采到的样本直接反传，通常只计算固定样本上的函数变化。因此，估计量在数值上是否无偏，与它能否产生所需的梯度，是两个需要分别讨论的问题。进一步考虑自回归生成时，还要计入当前选择对后续前缀分布的影响。

本文从一个二元分布出发，先整理 KL 的方向和常用估计量，再推导奖励修正与直接反传各自对应的梯度，最后将这些结果推广到完整序列。这样既能看清几个常见公式之间的联系，也便于理解它们在 PPO、GRPO 等实现中的适用条件。

<!-- more -->

## KL 散度的定义

先固定一个提示词，并将回答空间简化为 A、B 两种结果。记当前模型的分布为 $P$，参考模型的分布为 $Q$，取下面这组概率。

| 回答 | $P$ | $Q$ | $\log(P/Q)$ |
|---|---:|---:|---:|
| A | 0.8 | 0.5 | 0.470 |
| B | 0.2 | 0.5 | −0.916 |

在这个例子中，当前模型生成 A 的概率高于参考模型。按照 $P$ 对两种回答的对数概率比取平均，就得到

$$
D_{\mathrm{KL}}(P\|Q)
=\sum_yP(y)\log\frac{P(y)}{Q(y)}
\approx0.8\times0.470+0.2\times(-0.916)
\approx0.193.
$$

这里需要区分样本上的对数概率比与它的期望。前者可以为负，例如回答 B 对应的值约为 $-0.916$；后者才是非负的 KL 散度。因而，有限样本的平均值偶尔为负，并不与 KL 的非负性矛盾，也不能仅凭这一现象判断实现有误。

从预测的角度看，若数据实际来自 $P$，却用 $Q$ 为其分配概率，那么相对于使用 $P$，平均增加的对数损失就是 $D_{\mathrm{KL}}(P\|Q)$。本文统一使用自然对数，因此 KL 的单位为 nat。

定义中还包含对分布支撑集的要求。如果某个回答在 $P$ 下具有正概率，而在 $Q$ 下概率为零，对应的 KL 就会发散。为便于后续推导，我们先假设两者在所讨论的样本空间上都为正，相关期望存在，并且求导可以与求和交换。涉及截断采样时，再单独检查这些条件。

## 两个方向的差别

KL 散度一般不对称，因此交换两个分布的顺序，也会改变优化目标。固定参考分布 $Q$、优化当前分布 $P$ 时，本文将两个方向分别记为

$$
\text{reverse KL}:D_{\mathrm{KL}}(P\|Q),
\qquad
\text{forward KL}:D_{\mathrm{KL}}(Q\|P).
$$

不同领域对 forward 和 reverse 的命名并不完全一致，下面的讨论均以这两个表达式为准。

两者的区别首先来自取期望的分布。Reverse KL 按 $P$ 加权，因此会较强地惩罚当前模型频繁生成、而参考模型概率很低的回答。Forward KL 则按 $Q$ 加权，对于参考模型中较常见、当前模型却很少生成的回答，会产生较大的惩罚。

<div class="technical-figure"><iframe class="article-figure" scrolling="no" src="/figures/kl-divergence/01-directions/?embed=1" title="固定双峰参考分布，用单个高斯拟合时，两个 KL 方向得到不同结果。" loading="lazy"></iframe><a class="figure-static" href="/images/kl-divergence/01-directions.svg" target="_blank" rel="noopener"><img src="/images/kl-divergence/01-directions.svg" alt="固定双峰参考分布，用单个高斯拟合时，两个 KL 方向得到不同结果。" no-lazy></a></div>

图中给出了一个连续分布的例子。参考分布 $Q$ 包含两个峰，而 $P$ 被限制为单个高斯。最小化 reverse KL 时，$P$ 可以集中在其中一个峰附近，从而减少在两个峰之间低密度区域的概率质量；最小化 forward KL 时，两个峰都会参与平均，所得高斯因而更宽。

这种差别与分布族的表达能力有关。如果 $P$ 所在的分布族能够完整表示 $Q$，两个目标都在 $P=Q$ 时达到最小值。因而，图中的结果应理解为受限分布族下的拟合行为，而不宜推广为任意 KL 优化都必须满足的规律。[Tuan Anh Le 的相关说明](https://www.tuananhle.co.uk/notes/reverse-forward-kl.html)也讨论了这一点。

## KL 的采样估计

前面的二元例子可以直接求和，但语言模型的完整回答空间通常无法枚举。即使只考虑下一个 token，精确计算也需要访问整个词表的概率。当训练过程只保留被采中 token 的概率时，就需要用采样来估计 KL。

假设样本来自当前分布，即 $y\sim P$。为简化表达，记

$$
u(y)=\log\frac{P(y)}{Q(y)},\qquad
w(y)=\frac{Q(y)}{P(y)}=e^{-u(y)}.
$$

给定一个样本 $y$，$u(y)$ 和 $w(y)$ 都可以由两个模型对该样本的概率计算得到。下面考虑如何利用它们估计 $\mathbb E_P[u]$。

### 三种常用估计量

最直接的估计量是 $k_1=u$，它的期望按定义就等于 KL。不过，无偏性并没有限制单个样本的波动。当两个分布接近时，真实 KL 已经很小，而 $u$ 仍可能出现正负变化，因而相对于 KL 本身，采样误差可能相当明显。

另一种常见选择是 $k_2=u^2/2$。平方保证了样本值非负，但它的期望一般不再等于 KL。对于足够接近的分布，$k_2$ 与 KL 具有相同的局部二阶近似，因此可以在引入一定偏差的同时减小波动。

如果还希望保留无偏性，可以在 $k_1$ 上加入一个期望为零的修正项。由概率归一化可得

$$
\mathbb E_P[w-1]=\sum_yQ(y)-1=0,
$$

于是，定义

$$
k_3=u+w-1=e^{-u}-1+u
$$

便不会改变估计量的期望。另外，由 $e^{-u}\ge1-u$ 可知，$k_3$ 的每个样本值也都非负。三种估计量的基本性质如下。

| 估计量 | 公式 | 数值无偏 | 单样本一定非负 |
|---|---|:---:|:---:|
| $k_1$ | $u$ | 是 | 否 |
| $k_2$ | $u^2/2$ | 否 | 是 |
| $k_3$ | $e^{-u}-1+u$ | 是 | 是 |

上述无偏性指的是对采样分布取期望后的结果，并不保证每一批有限样本都接近真值。其中，$k_3$ 的证明还使用了 $Q$ 的全部概率质量都位于 $P$ 的支撑集内这一条件。如果有些位置在 $Q$ 下概率为正，却无法从 $P$ 采到，那么 $\sum_{y:P(y)>0}Q(y)$ 可能小于 1，修正项的期望也就不再为零。

### 局部近似与方差

<div class="technical-figure"><iframe class="article-figure" scrolling="no" src="/figures/kl-divergence/02-estimators/?embed=1" title="三个估计量的数值曲线，以及单位方差高斯例子中的相对标准差。" loading="lazy"></iframe><a class="figure-static" href="/images/kl-divergence/02-estimators.svg" target="_blank" rel="noopener"><img src="/images/kl-divergence/02-estimators.svg" alt="三个估计量的数值曲线，以及单位方差高斯例子中的相对标准差。" no-lazy></a></div>

为了进一步比较 $k_2$ 与 $k_3$，可以将后者在 $u=0$ 附近展开，得到

$$
k_3=\frac12u^2-\frac16u^3+O(u^4),
$$

两者的差别从三阶项开始，因此在分布接近时往往具有相似表现。随着偏离增大，高阶项不能再忽略，尤其是 $k_3$ 中的指数项，可能使少数样本对估计结果产生较大影响。

以单位方差的高斯分布为例，取 $P=\mathcal N(\delta,1)$、$Q=\mathcal N(0,1)$，则真实 KL 为 $\delta^2/2$。各估计量的偏差和方差都可以解析计算，下表列出其中两组结果。

| $\delta$ | KL | $k_2$ 相对偏差 | $k_1$ 相对标准差 | $k_2$ 相对标准差 | $k_3$ 相对标准差 |
|---|---:|---:|---:|---:|---:|
| 0.1 | 0.005 | 0.25% | 20.00 | 1.418 | 1.417 |
| 1 | 0.5 | 25% | 2.00 | 1.732 | 1.695 |

表中的相对标准差等于标准差除以真实 KL，未计入估计量的偏差。若对 $n$ 个独立样本取平均，标准差会缩小为原来的 $1/\sqrt n$，但 $k_2$ 的偏差不会因此消失。有关这三种估计量的构造，可以参考 Schulman 的 [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)。

## 从数值估计到梯度估计

到这里，我们讨论的都是固定分布下的数值估计。训练时，当前分布还依赖于模型参数，需要写成 $P_\theta$；参考模型 $Q$ 则在当前更新中保持固定。考虑带有 reverse-KL 正则的奖励目标

$$
J(\theta)=\mathbb E_{y\sim P_\theta}[R(y)]
-\beta D_{\mathrm{KL}}(P_\theta\|Q),
$$

其中，$R(y)$ 是不直接依赖参数 $\theta$ 的奖励，$\beta\ge0$ 决定正则强度。我们先将整条回答看作一个样本，推导这一目标的梯度，稍后再讨论逐 token 实现。

### 对期望求导

先考虑一般情形。若样本上的函数 $k_\theta(y)$ 也依赖参数，其期望为

$$
F(\theta)=\sum_yP_\theta(y)k_\theta(y).
$$

这个求和式中的概率权重与函数值都可能随参数变化，因此应对二者使用乘积法则。令 $g(y)=\nabla_\theta\log P_\theta(y)$，并利用 $\nabla P=P\nabla\log P$，可以写成

$$
\boxed{
\nabla_\theta F
=\mathbb E_{P_\theta}[k_\theta g]
+\mathbb E_{P_\theta}[\nabla_\theta k_\theta].
}
$$

右侧第一项来自采样概率的变化，第二项来自固定样本上 $k_\theta$ 的变化。二者分别对应不同的求导路径，这也是期望的梯度与样本函数的梯度不能直接等同的原因。

仍以 A、B 为例。假设一批样本中有八条 A、两条 B，那么直接对这批样本的均值反传，相当于对下面的表达式求导

$$
0.8k_\theta(A)+0.2k_\theta(B).
$$

这里的 0.8 和 0.2 表示已经观测到的频率，在反向传播中作为常数处理。因此，这一步只会计算两个 $k_\theta$ 的变化，并不会包含模型改变后采样概率的变化。即使这批频率恰好等于当前模型的概率，二者在计算图中的含义也仍然不同。

<div class="technical-figure"><iframe class="article-figure" scrolling="no" src="/figures/kl-divergence/03-gradient-paths/?embed=1" title="期望的梯度包含采样概率变化与固定样本函数变化两部分。" loading="lazy"></iframe><a class="figure-static" href="/images/kl-divergence/03-gradient-paths.svg" target="_blank" rel="noopener"><img src="/images/kl-divergence/03-gradient-paths.svg" alt="期望的梯度包含采样概率变化与固定样本函数变化两部分。" no-lazy></a></div>

由此可见，固定样本后对估计量直接反传，其期望通常只对应上式的第二项。若能够显式枚举样本空间，并把可导的 $P_\theta$ 作为权重参与求和，自动微分才会同时计算这两项。

### Reverse KL 的梯度

对于 reverse KL，样本函数为 $u=\log P_\theta-\log Q$，于是 $\nabla u=g$。另一方面，概率归一化给出

$$
\mathbb E_{P_\theta}[g]
=\sum_y\nabla_\theta P_\theta(y)
=\nabla_\theta1=0.
$$

将 $k=u$ 代入前面的期望求导公式，第二项的期望恰好消失，因此有

$$
\nabla_\theta D_{\mathrm{KL}}(P_\theta\|Q)
=\mathbb E[ug]+\mathbb E[g]
=\boxed{\mathbb E[ug]}.
$$

这里为零的是 $g$ 在当前分布下的平均，而不是每个样本的梯度。剩下的 $\mathbb E[ug]$ 表明，reverse-KL 梯度需要用各样本的 log-ratio 对其对数概率梯度加权。下面几种实现的差异，都可以通过与这个表达式比较得到。

## 奖励修正与直接反传

### 将 $k_1$ 用于奖励修正

由于 $R$ 不直接依赖参数，原始奖励的策略梯度为 $\mathbb E[Rg]$。若将样本奖励改为

$$
\widetilde R=R-\beta\operatorname{sg}(u),
$$

其中 $\operatorname{sg}$ 表示停止梯度，那么以修正后的奖励作为策略梯度的权重，就得到

$$
\mathbb E[\widetilde Rg]
=\mathbb E[Rg]-\beta\mathbb E[ug]
=\nabla_\theta J.
$$

因此，在这些假设下，将 $k_1$ 作为奖励惩罚能够产生原目标所需的梯度。这里使用的是 reverse KL 的直接求导项在期望下为零这一性质，并非任意 KL 估计量都可以作同样的替换。

对应的整条回答 REINFORCE 实现如下。`logp` 和 `logq` 分别为回答内有效 token 的对数概率之和，形状均为 `[batch]`；`reward` 也是每条回答对应一个数。

```python
# 回答必须由当前 P 采样；logp 重新计算并保留梯度。
u = logp - logq.detach()
adjusted_reward = reward.detach() - beta * u.detach()
loss = -(adjusted_reward * logp).mean()
```

最小化这里的 `loss`，就对应于最大化 $J$。`adjusted_reward` 只提供权重，梯度通过可导的 `logp` 传回当前模型。这段代码用于说明基本的求导关系，尚未包含价值基线、PPO 裁剪或旧样本修正。

### 将估计量作为可微损失

另一种实现是直接将 $k_1$、$k_2$ 或 $k_3$ 放入损失函数，对固定样本上的表达式求导。由各自的定义可得

$$
\nabla k_1=g,\qquad
\nabla k_2=ug,\qquad
\nabla k_3=(1-e^{-u})g.
$$

结合 $\mathbb E_P[g]=0$，可知直接对 $k_1$ 反传时，期望梯度为零。$k_2$ 的直接梯度则恰好为 $ug$，与所需的 reverse-KL 梯度估计一致。因此，虽然 $k_2$ 不是 KL 数值的无偏估计，它仍可以在上述条件下作为产生相应梯度的替代损失。

对于 $k_3$，将直接梯度按 $P$ 取期望，有

$$
\mathbb E_P[(1-Q/P)g]
=-\mathbb E_Q[g]
=\nabla_\theta D_{\mathrm{KL}}(Q\|P_\theta).
$$

可见，它对应的是 forward KL 的梯度。若仅优化 KL，且模型能够表示参考分布，两个方向都在 $P=Q$ 时取得最小值；加入奖励项后，二者一般会产生不同的折中，不能据此认为梯度可以互换。

| 实现方式 | 样本提供的 KL 梯度 | 当前策略采样下的期望 |
|---|---|---|
| $k_1$ 放入奖励，停止梯度后做策略梯度 | $ug$ | reverse-KL 梯度 |
| 直接对 $k_1$ 反传 | $g$ | 0 |
| 直接对 $k_2$ 反传 | $ug$ | reverse-KL 梯度 |
| 直接对 $k_3$ 反传 | $(1-e^{-u})g$ | forward-KL 梯度 |

表中省略了系数 $\beta$，并统一采用最小化 KL 的符号。所有结果都假设样本来自当前策略、参考模型固定，而且 $y$ 表示同一个完整事件。若换成旧策略样本，或将整条回答拆为局部 token 损失，还需要重新处理相应的分布依赖。[Tang 与 Munos](https://arxiv.org/html/2506.09477v1)对这些梯度差异作了系统讨论。

类似地，将 $k_3$ 停止梯度后用于奖励修正，也不会得到与 $k_1$ 相同的结果。此时只保留 $\mathbb E[k_3g]$，而 $k_3$ 的直接求导项在期望下并不为零。根据完整的期望求导关系，可以得到

$$
\mathbb E[k_3g]
=\nabla D_{\mathrm{KL}}(P\|Q)
-\nabla D_{\mathrm{KL}}(Q\|P).
$$

### 二元分布中的验证

这些关系可以在开头的例子中直接核对。将 $p=P(A)$ 作为参数，令 $P(B)=1-p$，参考分布保持为 $(0.5,0.5)$。在 $p=0.8$ 处，两个方向的精确导数分别为

$$
\frac{dD_{\mathrm R}}{dp}=\log4\approx1.38629,
\qquad
\frac{dD_{\mathrm F}}{dp}=1.875.
$$

下面的交互示例直接枚举 A、B，并按其概率加权，因此不包含有限次采样带来的误差。调整两个分布后，可以分别比较估计量的期望和直接梯度的期望，观察前面推导的关系。

<iframe class="article-lab kl-lab" src="/labs/kl-gradients/" title="KL 数值和梯度的双结果交互实验" loading="lazy"></iframe>

也可以[单独打开交互示例](/labs/kl-gradients/)。这里的导数针对概率参数 $p$；如果改用对应的 logit 作为参数，还需按链式法则乘上 $p(1-p)$。

## 自回归序列中的 KL 梯度

前面的结果将整条回答视为一个事件，而语言模型通常按 token 计算概率和损失。要把两种写法联系起来，需要先展开序列概率。设回答为 $y_1,\ldots,y_T$，第 $t$ 步之前的前缀为 $h_t=(x,y_{<t})$，记

$$
u_t=\log\frac{P_\theta(y_t\mid h_t)}{Q(y_t\mid h_t)},
\qquad
g_t=\nabla_\theta\log P_\theta(y_t\mid h_t).
$$

由自回归分解，完整回答的 log-ratio 为 $U=\sum_tu_t$，对数概率梯度为 $\sum_tg_t$。将它们代入前面的 reverse-KL 梯度公式，得到

$$
\nabla D_{\mathrm{seq}}
=\mathbb E\left[\left(\sum_j u_j\right)
\left(\sum_tg_t\right)\right].
$$

展开乘积后，除了同一步的 $u_tg_t$，还包含不同位置之间的交叉项 $u_jg_t$。这些交叉项正是完整序列目标与逐 token 局部损失之间的主要差别。

### 交叉项与前缀依赖

先考虑 $j<t$ 的情形。给定前缀 $h_t$，过去的 $u_j$ 已经确定，而当前 token 的对数概率梯度满足 $\mathbb E[g_t\mid h_t]=0$，所以 $\mathbb E[u_jg_t]=0$。这部分项可以在期望中消去。

对于 $j>t$，同样的论证不再成立。当前 token 的选择会改变后续前缀，因此未来的 $u_j$ 一般不能视为与当前选择无关的常数。保留这些项后，序列梯度可以化为

$$
\boxed{
\nabla D_{\mathrm{seq}}
=\mathbb E\left[\sum_t
\left(\sum_{j=t}^T u_j\right)g_t\right].
}
$$

一个简单例子可以说明未来项的来源。假设第一步选择 A 后，后续条件 KL 为 0；选择 B 后，后续条件 KL 为 2。若 B 的概率为 $p$，未来 KL 的期望就是 $2p$，对 $p$ 的导数为 2。仅计算第一步自身的 KL，便不会包含这部分贡献。

<div class="technical-figure"><iframe class="article-figure" scrolling="no" src="/figures/kl-divergence/04-future-kl/?embed=1" title="完整序列梯度包含当前与未来的跨时间项；固定前缀的局部梯度只保留对角项。" loading="lazy"></iframe><a class="figure-static" href="/images/kl-divergence/04-future-kl.svg" target="_blank" rel="noopener"><img src="/images/kl-divergence/04-future-kl.svg" alt="完整序列梯度包含当前与未来的跨时间项；固定前缀的局部梯度只保留对角项。" no-lazy></a></div>

图中以 $g_t$ 为行、$u_j$ 为列。完整序列梯度保留当前及未来位置对应的上三角项，而固定前缀的局部梯度只保留同一步的对角项。这里比较的是期望梯度的组成关系，不能据此认为两种采样估计在每条轨迹上都相等。

### 序列损失与局部损失

同一差别也可以从 $k_2$ 的构造中看出。对整条回答的 log-ratio 求平方，与分别对每个 token 的 log-ratio 求平方再相加，会得到不同的导数

$$
\begin{aligned}
\nabla\frac12\left(\sum_tu_t\right)^2
&=\left(\sum_tu_t\right)\left(\sum_tg_t\right),\\
\nabla\sum_t\frac12u_t^2
&=\sum_tu_tg_t.
\end{aligned}
$$

第一行包含各位置之间的交叉项，第二行只保留同一步的乘积。因此，前面关于 $k_2$ 直接梯度的结论，需要连同样本事件的定义一起使用；将完整事件换成局部 token 后，所对应的目标也发生了变化。

即使在每个已采样前缀上精确计算全词表 KL，再对它直接反传，也只计算了固定前缀下条件分布的变化，没有计入这些前缀出现概率的变化。逐 token 的 $k_3$ 同样对应采样前缀上的局部 forward-KL 梯度，而不是完整序列的 forward-KL 梯度。

### 累计回报中的 KL 项

策略梯度中的 reward-to-go 可以自然保留上述未来项。将每步奖励修正为 $\widetilde r_t=r_t-\beta u_t$，再定义从当前步到序列结束的累计回报

$$
\widetilde G_t=\sum_{j=t}^T\widetilde r_j.
$$

以停止梯度的 $\widetilde G_t$ 对 $g_t$ 加权，KL 部分便给出 $-\beta\nabla D_{\mathrm{seq}}$。因此，代码中应先修改逐步奖励，再计算累计回报，使当前位置的更新能够包含后续 KL 的贡献。

```python
# logp、logq、token_reward、mask 均为 [batch, time]。
# mask 覆盖有效生成 token（含 EOS），padding 为 0。
# token_reward 不直接依赖当前参数；此处不使用折扣与自举。
u = (logp - logq.detach()).detach()
step_reward = (token_reward.detach() - beta * u) * mask
returns = step_reward.flip(-1).cumsum(-1).flip(-1)
loss = -(returns.detach() * logp * mask).sum(-1).mean()
```

代码先对每条回答的有效 token 求和，再在 batch 内取平均，与这里的序列目标一致。如果额外按每条回答的长度归一化，就会改变不同回答之间的权重。实现时还应保证参与运算的 log probability 有限，避免在 padding 位置直接计算 `0 * (-inf)`。

在此基础上，可以通过价值基线降低方差。若进一步使用 GAE、自举、PPO 裁剪或对同一批旧样本进行多轮更新，还需要分析这些步骤引入的近似；上面的等式只刻画了当前策略采样下的基本梯度关系。

## 当前策略、参考策略与行为策略

回到 PPO、GRPO 的代码中，还会同时遇到当前模型与旧模型的概率比。为避免将它与 KL 中的比率混淆，可以先区分下面三个模型的作用。

| 模型 | 用途 |
|---|---|
| $\pi_\theta$ | 当前正在更新的模型，对应前文的 $P_\theta$ |
| $\pi_{\mathrm{ref}}$ | KL 的参考模型，对应 $Q$ |
| $\pi_{\mathrm{old}}$ | 采集这批回答时保存的旧策略 |

PPO 中的 $\pi_\theta/\pi_{\mathrm{old}}$ 比较当前策略与采样时的旧策略，用于构造更新目标；$k_3$ 中的 $\pi_{\mathrm{ref}}/\pi_\theta$ 则比较参考策略与当前策略，用于构造正则项。虽然两者都是概率比，所涉及的分布与用途并不相同。

GRPO 使用同一提示词下的组内奖励构造相对优势，其原始形式保留了 PPO 的裁剪目标，并另外加入 KL 项。因而，分析其中的 $k_3$ 时，仍需结合它是否参与反传、样本来自哪个策略，以及损失采用何种归一化方式。[DeepSeekMath](https://arxiv.org/abs/2402.03300)给出了原始 GRPO 的定义。

尤其是一批回答被重复使用时，模型更新后，这些回答通常就不再来自当前 $P_\theta$。设实际生成样本的行为分布为 $\mu$，若希望恢复当前分布下的期望，可以使用重要性加权

$$
\mathbb E_{P_\theta}[f(y)]
=\mathbb E_\mu\left[\frac{P_\theta(y)}{\mu(y)}f(y)\right].
$$

这一等式要求 $\mu$ 覆盖目标分布所需的支撑集。生成时采用温度、top-p、top-k 或异步策略，都可能使实际行为分布与保存的旧模型概率不同。对于行为策略完全无法采到的区域，重要性加权也无法恢复其贡献。

完整序列的重要性比率是各步比率的乘积，因而可能具有很大的方差。只保留当前 token 的比率，无法同时修正前缀分布；对比率进行裁剪虽然有助于控制更新幅度，也会改变严格的重要性加权等式。

重要性权重是否停止梯度，则取决于它在推导中的作用。如果只是用它加权一个已经得到的梯度估计，应将权重视为常数；如果构造的是整个加权期望，再由自动微分求导，那么权重的导数本来就是乘积法则的一部分。因此，`.detach()` 的位置需要由目标和估计方式共同决定。

## 全词表求和与部分求和

除了修正采样分布，还可以通过增加精确计算来减少某些采样误差。固定前缀 $h$，如果能够取得两个模型的完整词表概率，就可以直接计算条件 KL

$$
D(h)=\sum_{v\in\mathcal V}P(v\mid h)
\log\frac{P(v\mid h)}{Q(v\mid h)}.
$$

只要概率权重 $P$ 及其 log-ratio 都参与求导，就能得到该前缀上的条件 KL 梯度，并消除当前 token 的抽样误差。不过，前缀本身仍是采样得到的，因此这一步没有消除前缀的随机性，也不会自动加入前缀分布变化所对应的梯度。

从方差的角度看，用条件期望替代某一步的采样 log-ratio，确实不会增加该步的方差。但如果将各步结果相加，替换还会改变步间协方差，因此不能直接据此断定整个序列的总方差也会下降。

<details>
<summary>一个关于序列总方差的两步反例</summary>

考虑一个每步只能取 0、1 的两步过程，其条件概率如下。

| 条件 | $P$ | $Q$ |
|---|---|---|
| 第一步 | $(0.5,0.5)$ | $(0.25,0.75)$ |
| 第一步为 0 后 | $(0.5,0.5)$ | $(0.5,0.5)$ |
| 第一步为 1 后 | $(0.01,0.99)$ | $(0.99,0.01)$ |

这个过程只有四条可能路径，可以直接枚举。采样 log-ratio 的总和 $U=u_1+u_2$ 与条件 KL 的总和 $C=D_1+D_2(h_2)$ 具有相同均值，约为 2.39545，而两者的方差分别为

$$
\operatorname{Var}(U)\approx3.31591,
\qquad\operatorname{Var}(C)\approx5.06974.
$$

在原来的采样和中，第一步的 log-ratio 与后续偏离存在负相关，使部分波动相互抵消。逐步替换为条件期望后，这种协方差关系也随之变化。由于各步使用了不同的条件信息，不能将整个替换视为对同一个随机变量作一次条件期望，因此不与条件期望的方差不增性质矛盾。

文末的枚举代码可以复现这一结果。这个有限状态例子说明的是单步结论的适用范围；条件 KL 在实际训练中的收益，还需要结合具体任务进行比较。

</details>

全词表求和需要读取或传输大量概率，因此也可以采用部分精确计算的折中方案。选取集合 $S_k$，对集合内的项求和，再通过样本估计其余部分，得到

$$
\widehat D=
\sum_{v\in S_k}P(v)\log\frac{P(v)}{Q(v)}
+\mathbf1_{\{y\notin S_k\}}\log\frac{P(y)}{Q(y)},
\quad y\sim P.
$$

固定前缀和集合 $S_k$ 后，对上式取期望即可恢复全词表 KL；如果省略尾部项，则一般会引入偏差。这里的 Top-k 用于选出需要精确计算的词表项，样本 $y$ 仍来自完整的 $P$，与生成时截断采样分布的 top-k 是不同的操作。[EMA-PG](https://arxiv.org/html/2602.04417v1)讨论了这一类估计方法。

<details>
<summary>部分求和中的尾部梯度</summary>

尾部项的数值无偏，并不意味着直接反传也会得到所需梯度。令 $u_v=\log[P(v)/Q(v)]$，$a_\theta=P/\operatorname{sg}(P)$，在固定集合 $S_k$ 时，可以构造

$$
\widehat D_{\mathrm{train}}
=\sum_{v\in S_k}P(v)u_v
+\mathbf1_{\{y\notin S_k\}}
\left[a_\theta(y)\operatorname{sg}(u_y+1)-1\right].
$$

由于 $a_\theta$ 的前向值为 1，尾部表达式的数值等于 $u_y$；反向传播时，梯度则为 $(u_y+1)g_y$。按 $P$ 取期望后，它恰好对应尾部精确求和的导数，因此与头部相加，便同时保留了数值与局部梯度的正确性。

其中的 $+1$ 来自尾部概率质量的变化。虽然在整个分布上有 $\sum_vP(v)g_v=0$，将求和限制到尾部后一般不再为零，所以不能沿用全空间的抵消关系将这一项删去。

例如，在 $P=(0.8,0.2),Q=(0.5,0.5)$ 中只精确计算 A。若尾部仅提供 $u_Bg_B$，得到的期望导数约为 2.38629；补上概率质量项后，才恢复精确值 1.38629。附带的验证脚本也检查了这一结果。

当集合由 top-k 产生时，上述求导将当前选中的索引视为固定，排名交换处需要另行处理。此外，这里得到的仍是固定前缀上的局部梯度，并未包含完整序列中的前缀分布导数。

</details>

## 数值验证

前面的结论可以先在小分布上检验，无需运行完整的模型训练。下面以二元分布为例，将精确 KL 的导数与固定采样频率后的各估计量导数进行比较。

```python
import torch

p = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
P = torch.stack((p, 1 - p))
Q = torch.tensor([0.5, 0.5], dtype=torch.float64)
u = P.log() - Q.log()

exact = (P * u).sum()            # 概率权重也参与求导
k1 = (P.detach() * u).sum()      # 模拟已采样后固定的频率
k2 = (P.detach() * u.square() / 2).sum()
k3 = (P.detach() * (torch.expm1(-u) + u)).sum()

for name, loss in [("exact", exact), ("k1", k1), ("k2", k2), ("k3", k3)]:
    grad, = torch.autograd.grad(loss, p, retain_graph=True)
    print(name, round(grad.item(), 6))
```

运行后得到 `exact 1.386294`、`k1 0.0`、`k2 1.386294`、`k3 1.875`。其中，`P.detach()` 保留当前分布的精确频率，但不对这些频率求导，相当于直接计算固定样本反传结果的期望。代码也提供了[可下载的完整版本](/downloads/kl-divergence/gradient_demo.py)。

另外，[完整验证脚本](/downloads/kl-divergence/check_kl.py)枚举了两步序列，比较完整梯度、局部梯度和 reward-to-go，并检查上述方差反例与 Top-k 尾部修正。该脚本只依赖 Python 标准库，可以用下面的命令运行；前面的自动微分示例则需要 PyTorch。

```bash
python3 check_kl.py
```

## 讨论

本文从 KL 的采样估计出发，区分了估计值、固定样本的直接梯度和完整期望的梯度。对于当前策略采样下的完整序列 reverse KL，将 $k_1$ 作为逐步奖励惩罚，并在计算 reward-to-go 时保留未来项，可以得到相应的序列梯度。若采用固定前缀上的局部 token 正则，则应按局部目标理解其更新。

这些推导说明了实现与目标之间的关系，但不能单独决定训练效果。估计方差、奖励质量、裁剪方式、数据分布与优化器都会影响最终结果。因此，在比较不同 KL 实现时，应先明确采样分布、参考模型、求导路径以及序列层级，再结合实验判断具体选择。

同一个 $k_1$ 或 $k_3$ 表达式，放在不同的计算图中，可能承担不同的作用。将这些条件与公式一起写清楚，才能把数值上的比较落实到训练目标上。

## 参考与复现

- 数值估计可参考 [Schulman, Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)。
- 梯度与序列的讨论见 [Tang & Munos, On a few pitfalls in KL divergence gradient estimation for RL](https://arxiv.org/html/2506.09477v1)。
- 关于条件 KL，参见 [Better Estimation of the KL Divergence Between Language Models](https://arxiv.org/html/2504.10637v2)。本文附带有限状态枚举，用于区分单步与序列总方差结论。
- 局部正则与 Top-k 可参考 [EMA Policy Gradient](https://arxiv.org/html/2602.04417v1)。本文的尾部梯度公式由头尾分解直接推导，并附独立数值检查。
- 不同实现的实验比较包括 [Rethinking KL Regularization](https://arxiv.org/html/2510.01555v1)、[A Comedy of Estimators](https://arxiv.org/html/2512.21852v1)、[On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](https://arxiv.org/abs/2505.17508)。
- 中文讨论可参考 [繁华落尽见真淳：大模型强化学习中 KL 散度的正确形式是 k1 in Reward](https://mp.weixin.qq.com/s/HRDmhG-ODpuozsOHM__f9Q)。本文将“正确”限定到明确的优化目标与采样条件。
- [验证代码、配图脚本与源文件](/downloads/kl-divergence/sources.zip)。图中的分布与曲线来自明确的数学例子，不代表实际模型训练结果。
