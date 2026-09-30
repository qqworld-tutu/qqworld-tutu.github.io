---
title: 大模型强化学习中的 KL：数值与梯度
date: 2026-09-29 12:01:00
permalink: llm/kl-divergence-in-rl/
description: 用一个两种回答的例子，理解 KL 的方向、k1/k2/k3 估计量，以及放进 reward 和 loss 后不同的梯度。再推导序列中的未来 KL，附交互实验和可运行验证代码。
categories:
  - 大模型
tags:
  - 强化学习
  - KL散度
  - PPO
  - GRPO
technical_article: true
---

在大模型强化学习里，奖励鼓励模型改变回答，KL 惩罚则限制它偏离参考模型。实现起来似乎只需要几行：计算当前模型和参考模型的 log probability，做一个差，再加到 loss 里。

问题恰好藏在“加到 loss 里”这一步。**一个公式可以正确估计 KL 的数值，直接反向传播却不产生我们需要的 KL 梯度。** 即使单步梯度正确，拆成逐 token 的实现后，还可能漏掉当前选择对后续回答的影响。

这篇文章沿着一个两种回答的小例子，把数值估计、梯度和序列依赖连起来。先弄清楚我们想优化什么，再判断代码实际做了什么。

<!-- more -->

## 从两种回答开始

固定一个提示词，假设模型只能回答 A 或 B。当前模型记作 $P$，参考模型记作 $Q$：

| 回答 | $P$ | $Q$ | $\log(P/Q)$ |
|---|---:|---:|---:|
| A | 0.8 | 0.5 | 0.470 |
| B | 0.2 | 0.5 | −0.916 |

当前模型更偏爱 A。按当前模型的概率，对最后一列取平均，得到

$$
D_{\mathrm{KL}}(P\|Q)
=\sum_yP(y)\log\frac{P(y)}{Q(y)}
\approx0.8\times0.470+0.2\times(-0.916)
\approx0.193.
$$

这里有两个容易混淆的量：单个回答上的对数概率比，以及整个分布上的平均。只采到 B 时，估计值会是负数；KL 本身则非负。训练日志里的采样 KL 偶尔为负，并不必然说明程序错了。

KL 也可以理解为额外的预测代价：数据来自 $P$，却用 $Q$ 给它分配概率，相对于直接用 $P$，平均多出的对数损失就是 KL。全文使用自然对数，单位为 nat。

当某个回答在 $P$ 下有正概率、在 $Q$ 下概率为零时，KL 为无穷大。后面的求导先假设两者在所讨论的样本空间上都为正，相关期望存在，且求导可以与求和交换。这个假设在讨论截断采样时需要重新检查。

## 调换方向，惩罚的行为也不同

固定参考模型 $Q$、优化当前模型 $P$，本文采用以下命名：

$$
\text{reverse KL}:D_{\mathrm{KL}}(P\|Q),
\qquad
\text{forward KL}:D_{\mathrm{KL}}(Q\|P).
$$

命名在不同领域可能不同，公式中的顺序才是可靠依据。

Reverse KL 按 $P$ 取平均。如果当前模型经常生成参考模型几乎不会生成的内容，这些回答会得到较大惩罚。Forward KL 按 $Q$ 取平均；参考模型常见、当前模型却几乎不生成的内容，会成为主要惩罚来源。

<div class="technical-figure"><iframe class="article-figure" src="/figures/kl-divergence/01-directions/?embed=1" title="固定双峰参考分布，用单个高斯拟合时，两个 KL 方向得到不同结果。" loading="lazy"></iframe><a class="figure-static" href="/images/kl-divergence/01-directions.svg" target="_blank" rel="noopener"><img src="/images/kl-divergence/01-directions.svg" alt="固定双峰参考分布，用单个高斯拟合时，两个 KL 方向得到不同结果。" no-lazy></a></div>

图中把 $P$ 限制为一个高斯，$Q$ 则有两个峰。Reverse KL 倾向于停在一个峰附近，避免落入 $Q$ 密度很低的中间区域；forward KL 更倾向于用一个宽分布覆盖两边。

这是有表达限制时的拟合倾向。如果 $P$ 可以完整表示 $Q$，两个方向的最优解都是 $P=Q$，并不存在必须“只选一个峰”的要求。[Tuan Anh Le 的说明](https://www.tuananhle.co.uk/notes/reverse-forward-kl.html)

## 只采一个回答，怎样估计 KL

真实语言模型不可能枚举所有完整回答。即使只看下一个 token，精确计算也要对整个词表求和。很多训练流程只保留被采中 token 的概率，因此需要样本估计。

设 $y\sim P$，并记

$$
u(y)=\log\frac{P(y)}{Q(y)},\qquad
w(y)=\frac{Q(y)}{P(y)}=e^{-u(y)}.
$$

$u$ 和 $w$ 都只是采到的回答上的一个数。目标是用这些数估计 $\mathbb E_P[u]$。

### 三种常用估计量

最直接的办法是 $k_1=u$。按定义，它的期望就是 KL，因此无偏。但两个分布接近时，真实 KL 很小，单个 $u$ 仍有正负波动，相对误差可能很大。

第二种是 $k_2=u^2/2$。它不会为负，但平均值通常不等于 KL。当两个分布接近时，它匹配 KL 的局部二阶近似，可以用偏差换取较小波动。

第三种在 $k_1$ 上加一个零均值项。因为

$$
\mathbb E_P[w-1]=\sum_yQ(y)-1=0,
$$

所以

$$
k_3=u+w-1=e^{-u}-1+u
$$

仍然无偏。由 $e^{-u}\ge1-u$，它还逐样本非负。

| 估计量 | 公式 | 数值无偏 | 单样本一定非负 |
|---|---|:---:|:---:|
| $k_1$ | $u$ | 是 | 否 |
| $k_2$ | $u^2/2$ | 否 | 是 |
| $k_3$ | $e^{-u}-1+u$ | 是 | 是 |

无偏意味着反复采样后的平均正确，不保证某一批样本接近真值。这里 $k_3$ 的无偏性还要求：$Q$ 有概率的位置，$P$ 也能采到。否则 $\sum_{y:P(y)>0}Q(y)$ 可能小于 1，上面的零均值证明就不成立。

### 非负、无偏，也不保证方差小

<div class="technical-figure"><iframe class="article-figure" src="/figures/kl-divergence/02-estimators/?embed=1" title="左：三个估计量随 log-ratio 的变化。右：单位方差高斯例子的单样本相对标准差。" loading="lazy"></iframe><a class="figure-static" href="/images/kl-divergence/02-estimators.svg" target="_blank" rel="noopener"><img src="/images/kl-divergence/02-estimators.svg" alt="左：三个估计量随 log-ratio 的变化。右：单位方差高斯例子的单样本相对标准差。" no-lazy></a></div>

在 $u=0$ 附近，

$$
k_3=\frac12u^2-\frac16u^3+O(u^4),
$$

所以 $k_2$ 与 $k_3$ 接近。偏离较大时，$k_3$ 中的指数项也可能带来很大波动。

取 $P=\mathcal N(0,1)$、$Q=\mathcal N(\delta,1)$，真实 KL 为 $\delta^2/2$。按解析公式计算：

| $\delta$ | KL | $k_2$ 相对偏差 | $k_1$ 相对标准差 | $k_2$ 相对标准差 | $k_3$ 相对标准差 |
|---|---:|---:|---:|---:|---:|
| 0.1 | 0.005 | 0.25% | 20.00 | 1.418 | 1.417 |
| 1 | 0.5 | 25% | 2.00 | 1.732 | 1.695 |

相对标准差是“标准差 ÷ 真实 KL”，不包含偏差。对 $n$ 个独立样本取平均，标准差会除以 $\sqrt n$，但 $k_2$ 的偏差不会随样本增多而消失。这三种估计量的经典介绍见 [Schulman：Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)。

## 为什么数值无偏，直接求导却不对

现在让当前模型依赖参数 $\theta$，写成 $P_\theta$，参考模型 $Q$ 在当前更新中保持固定。我们希望最大化

$$
J(\theta)=\mathbb E_{y\sim P_\theta}[R(y)]
-\beta D_{\mathrm{KL}}(P_\theta\|Q),
$$

其中 $R$ 是不直接依赖 $\theta$ 的奖励，$\beta\ge0$ 控制惩罚强度。先把整条回答当成一个样本，暂时不拆 token。

### 期望的求导，比样本公式的求导多一步

对任意依赖参数的 $k_\theta(y)$，真实期望为

$$
F(\theta)=\sum_yP_\theta(y)k_\theta(y).
$$

参数变化时，概率和 $k$ 都会变。由乘积法则，令 $g(y)=\nabla_\theta\log P_\theta(y)$，得到

$$
\boxed{
\nabla_\theta F
=\mathbb E_{P_\theta}[k_\theta g]
+\mathbb E_{P_\theta}[\nabla_\theta k_\theta].
}
$$

第一项回答“哪些回答变得更常出现”，第二项回答“同一条回答的数值如何变化”。等式使用了 $\nabla P=P\nabla\log P$。

回到 A、B。如果这次采到八条 A、两条 B，对样本均值反传，相当于求导

$$
0.8k_\theta(A)+0.2k_\theta(B).
$$

0.8 和 0.2 是已经发生的采样频率，不会随参数更新改变。自动微分只计算 $k$ 的变化，不会把已经采到的八条 A 变成九条。

<div class="technical-figure"><iframe class="article-figure" src="/figures/kl-divergence/03-gradient-paths/?embed=1" title="参数改变期望的两条路径：采样概率变化和固定回答上的公式变化。" loading="lazy"></iframe><a class="figure-static" href="/images/kl-divergence/03-gradient-paths.svg" target="_blank" rel="noopener"><img src="/images/kl-divergence/03-gradient-paths.svg" alt="参数改变期望的两条路径：采样概率变化和固定回答上的公式变化。" no-lazy></a></div>

因此，**对样本估计直接反传，只拿到了完整期望梯度中的第二项。** 如果显式枚举所有回答，把 $P_\theta$ 的加权也放进计算图，才能直接用自动微分拿到两项。

### Reverse KL 恰好只需要第一项

因为 $u=\log P_\theta-\log Q$，所以 $\nabla u=g$。另外，归一化概率满足

$$
\mathbb E_{P_\theta}[g]
=\sum_y\nabla_\theta P_\theta(y)
=\nabla_\theta1=0.
$$

把 $k=u$ 代入完整求导公式，得到

$$
\nabla_\theta D_{\mathrm{KL}}(P_\theta\|Q)
=\mathbb E[ug]+\mathbb E[g]
=\boxed{\mathbb E[ug]}.
$$

不同回答的 $g$ 按概率平均后抵消，不代表每个回答的梯度都是零。我们真正需要的是：让每条回答的 $g$ 再乘上自己的 log-ratio $u$。

## 放进 reward 和放进 loss，分别算了什么

### $k_1$ 放进 reward

原始奖励的策略梯度是 $\mathbb E[Rg]$。把奖励改为

$$
\widetilde R=R-\beta\operatorname{sg}(u),
$$

其中 $\operatorname{sg}$ 表示停止梯度，新的策略梯度就是

$$
\mathbb E[\widetilde Rg]
=\mathbb E[Rg]-\beta\mathbb E[ug]
=\nabla_\theta J.
$$

这个结果并不依赖“KL 看起来像奖励”，而是因为忽略的直接求导项恰好为零。

下面是最简单的整条回答 REINFORCE 写法。`logp` 和 `logq` 是回答内有效 token 的 log probability 之和，形状均为 `[batch]`，`reward` 也是 `[batch]`：

```python
# 回答必须由当前 P 采样；logp 重新计算并保留梯度。
u = logp - logq.detach()
adjusted_reward = reward.detach() - beta * u.detach()
loss = -(adjusted_reward * logp).mean()
```

最小化 `loss` 对应最大化 $J$。奖励停止梯度以后，`logp` 仍然可导，所以模型仍会更新。这里是解释梯度路径的最小实现，还没有加入 PPO 裁剪、价值函数或旧样本修正。

### 直接把估计量放进 loss

直接求导时，三个公式分别产生

$$
\nabla k_1=g,\qquad
\nabla k_2=ug,\qquad
\nabla k_3=(1-e^{-u})g.
$$

前两项已经能看出结果：$k_1$ 的期望梯度为零，$k_2$ 则恰好得到所需的 reverse-KL 梯度。$k_2$ 数值有偏，却可以作为产生正确梯度的替代损失。

对于 $k_3$，

$$
\mathbb E_P[(1-Q/P)g]
=-\mathbb E_Q[g]
=\nabla_\theta D_{\mathrm{KL}}(Q\|P_\theta).
$$

它给出的是另一个方向的梯度。两个方向只优化 KL 时有相同最优点，但加入奖励后，它们与奖励折中的方式会不同。

| 实现方式 | 样本提供的 KL 梯度 | 当前策略采样下的期望 |
|---|---|---|
| $k_1$ 放入奖励，停止梯度后做策略梯度 | $ug$ | reverse-KL 梯度 |
| 直接对 $k_1$ 反传 | $g$ | 0 |
| 直接对 $k_2$ 反传 | $ug$ | reverse-KL 梯度 |
| 直接对 $k_3$ 反传 | $(1-e^{-u})g$ | forward-KL 梯度 |

表格省略了 $\beta$，统一写成最小化 KL 的方向。结论针对同一个完整事件、当前策略采样、固定参考模型；不能原样套到旧策略样本或逐 token 的序列实现上。[Tang 与 Munos](https://arxiv.org/html/2506.09477v1)系统讨论了这些梯度差异。

同理，$k_3$ 只放 reward 也不等价于 $k_1$：它留下 $\mathbb E[k_3g]$，缺少的直接梯度并不为零。完整关系是

$$
\mathbb E[k_3g]
=\nabla D_{\mathrm{KL}}(P\|Q)
-\nabla D_{\mathrm{KL}}(Q\|P).
$$

### 动手检查 A、B 的例子

把 $p=P(A)$ 直接作为参数，$P(B)=1-p$，参考模型仍为 $(0.5,0.5)$。在 $p=0.8$ 时，两个方向的精确导数是

$$
\frac{dD_{\mathrm R}}{dp}=\log4\approx1.38629,
\qquad
\frac{dD_{\mathrm F}}{dp}=1.875.
$$

下面直接枚举 A、B，因此没有有限次随机采样的误差。移动滑块，比较“数值的期望”和“直接求导的期望”，就能看到两者为什么必须分开检查。

<iframe class="article-lab kl-lab" src="/labs/kl-gradients/" title="KL 数值和梯度的双结果交互实验" loading="lazy"></iframe>

[单独打开交互实验](/labs/kl-gradients/)。这里求的是对概率 $p$ 的导数；若改用 logit 参数，还要乘 $p(1-p)$。

## 从整条回答到 token，还差一个未来

前面把整条回答视为一个事件。真实回答由 $y_1,\ldots,y_T$ 组成，每一步的前缀是 $h_t=(x,y_{<t})$。定义

$$
u_t=\log\frac{P_\theta(y_t\mid h_t)}{Q(y_t\mid h_t)},
\qquad
g_t=\nabla_\theta\log P_\theta(y_t\mid h_t).
$$

整条回答的 log-ratio 为 $U=\sum_tu_t$，对数概率梯度为 $\sum_tg_t$，所以完整序列 KL 的梯度是

$$
\nabla D_{\mathrm{seq}}
=\mathbb E\left[\left(\sum_j u_j\right)
\left(\sum_tg_t\right)\right].
$$

展开之后出现交叉项：当前 token 的梯度，不仅乘当前 KL，也乘其他位置的 KL。

### 过去项可以删，未来项不能

如果 $j<t$，在选第 $t$ 个 token 之前，过去的 $u_j$ 已经确定。固定前缀，对当前 token 平均，有 $\mathbb E[g_t\mid h_t]=0$，所以 $\mathbb E[u_jg_t]=0$。

未来的 $u_j$ 不同：当前选什么 token，会改变后面的前缀，也就改变未来 KL。因此只能删除过去项，留下

$$
\boxed{
\nabla D_{\mathrm{seq}}
=\mathbb E\left[\sum_t
\left(\sum_{j=t}^T u_j\right)g_t\right].
}
$$

想象第一步选 A 后，后续条件 KL 为 0；选 B 后，后续条件 KL 为 2。如果 B 的概率是 $p$，未来 KL 的期望就是 $2p$，对 $p$ 的导数为 2。只看第一步自身的 KL，会漏掉这个影响。

<div class="technical-figure"><iframe class="article-figure" src="/figures/kl-divergence/04-future-kl/?embed=1" title="完整序列梯度包含当前与未来的跨时间项；固定前缀的局部梯度只保留对角项。" loading="lazy"></iframe><a class="figure-static" href="/images/kl-divergence/04-future-kl.svg" target="_blank" rel="noopener"><img src="/images/kl-divergence/04-future-kl.svg" alt="完整序列梯度包含当前与未来的跨时间项；固定前缀的局部梯度只保留对角项。" no-lazy></a></div>

图中行是 $g_t$，列是 $u_j$。左图保留当前与未来项，右图只保留同一步的 $u_tg_t$。这是期望梯度的组成关系，不表示两种采样估计在每条轨迹上数值相同。

### 逐 token 求和，不等于先求和再平方

这个差别也能从 $k_2$ 直接看出来：

$$
\begin{aligned}
\nabla\frac12\left(\sum_tu_t\right)^2
&=\left(\sum_tu_t\right)\left(\sum_tg_t\right),\\
\nabla\sum_t\frac12u_t^2
&=\sum_tu_tg_t.
\end{aligned}
$$

第一行保留序列交叉项，第二行只保留当前项。因而“$k_2$ 的直接梯度正确”必须说明计算层级。

即使在每个前缀上精确计算全词表 KL，再只对这个局部 KL 反传，也只考虑“这个前缀上的分布如何改变”，没有考虑“这个前缀以后多常出现”。逐 token 的 $k_3$ 则给出采样前缀上的局部 forward-KL 梯度，同样不能当成完整序列 forward KL。

### 用 reward-to-go 保留未来项

把每步奖励改为 $\widetilde r_t=r_t-\beta u_t$，再累加从当前到结束的奖励：

$$
\widetilde G_t=\sum_{j=t}^T\widetilde r_j.
$$

使用停止梯度的 $\widetilde G_t$ 乘 $g_t$，KL 部分正好是 $-\beta\nabla D_{\mathrm{seq}}$。实现顺序很关键：先修改每步奖励，再计算累计回报，不能只拿当前 KL 给当前梯度加权。

```python
# logp、logq、token_reward、mask 均为 [batch, time]。
# mask 覆盖有效生成 token（含 EOS），padding 为 0。
# token_reward 不直接依赖当前参数；此处不使用折扣与自举。
u = (logp - logq.detach()).detach()
step_reward = (token_reward.detach() - beta * u) * mask
returns = step_reward.flip(-1).cumsum(-1).flip(-1)
loss = -(returns.detach() * logp * mask).sum(-1).mean()
```

这里每条回答按 token 求和，再对回答取平均，对应序列目标。变成每条回答除以自身长度，会改变目标。代码假设参与计算的 log probability 有限；不要把 padding 处的 `-inf` 直接乘 0。

加入价值基线可以降低方差。进一步使用 GAE、自举、PPO 裁剪或多轮旧样本更新时，还会引入其他近似。上面的推导不是对所有 PPO 实现无偏性的保证。

## 读 PPO、GRPO 代码时，先认清三个模型

很多混淆来自两种概率比同时出现。需要区分：

| 模型 | 用途 |
|---|---|
| $\pi_\theta$ | 当前正在更新的模型，对应前文的 $P_\theta$ |
| $\pi_{\mathrm{ref}}$ | KL 的参考模型，对应 $Q$ |
| $\pi_{\mathrm{old}}$ | 采集这批回答时保存的旧策略 |

PPO 的比率 $\pi_\theta/\pi_{\mathrm{old}}$ 比较当前与旧策略，用于构造更新；$k_3$ 里的比率 $\pi_{\mathrm{ref}}/\pi_\theta$ 比较参考与当前策略，用于正则。两个分母和目的都不同。

GRPO 用同一提示词下的组内奖励构造相对优势，原始形式沿用 PPO 的裁剪目标并另加 KL 项。因此，看到 GRPO 中的 $k_3$，还要继续看它是否直接参与反传、数据来自哪个策略、按 token 还是按回答归一化。[DeepSeekMath 中的 GRPO 定义](https://arxiv.org/abs/2402.03300)

如果一批回答用于多次更新，第一次更新之后，它们通常就不再来自当前 $P_\theta$。设真实采样分布为 $\mu$，恢复当前分布的期望需要

$$
\mathbb E_{P_\theta}[f(y)]
=\mathbb E_\mu\left[\frac{P_\theta(y)}{\mu(y)}f(y)\right].
$$

前提是 $\mu$ 覆盖目标分布需要的样本。温度、top-p、top-k 或异步生成都可能让实际行为策略不同于保存的旧模型；完全采不到的区域，不能只靠加权补回来。

完整序列的概率比是逐步比率的乘积，可能产生很大方差。只使用当前 token 的比率无法同时修正前缀分布；裁剪虽能稳定更新，也不再严格保持上述无偏等式。

重要性权重是否停止梯度也取决于构造：若它只用于加权一个已经推导正确的梯度估计，应停止梯度；若要对整个加权期望求导，权重的导数就是乘积法则的一部分。不能从另一段代码随意搬一个 `.detach()`。

## 全词表与 Top-k：能减少哪一部分误差

固定前缀 $h$，如果拿到了两个模型的完整概率，可以直接计算

$$
D(h)=\sum_{v\in\mathcal V}P(v\mid h)
\log\frac{P(v\mid h)}{Q(v\mid h)}.
$$

这里每个 $P$ 及其 log-ratio 都在计算图中，因此能得到该前缀上的完整条件 KL 梯度，也消除了当前 token 的抽样误差。它没有消除前缀的抽样误差，更不会自动补上前面讨论的未来项。

逐步用条件期望代替采样 log-ratio，单步方差不会增加。但沿整条回答求和时，各步协方差也改变，不能直接推出序列总方差必然下降。

<details>
<summary>可复现的两步反例：单步降方差，不保证总和也降</summary>

每步有 0、1 两个取值，使用以下条件概率：

| 条件 | $P$ | $Q$ |
|---|---|---|
| 第一步 | $(0.5,0.5)$ | $(0.25,0.75)$ |
| 第一步为 0 后 | $(0.5,0.5)$ | $(0.5,0.5)$ |
| 第一步为 1 后 | $(0.01,0.99)$ | $(0.99,0.01)$ |

枚举四条路径。采样 log-ratio 总和 $U=u_1+u_2$ 与条件 KL 总和 $C=D_1+D_2(h_2)$ 的均值都约为 2.39545，但

$$
\operatorname{Var}(U)\approx3.31591,
\qquad\operatorname{Var}(C)\approx5.06974.
$$

原来第一步 log-ratio 与后续偏离存在负相关，部分波动相互抵消。逐步替换成条件期望后，这种抵消也发生了变化。它没有违反单步条件期望的方差不增定理，因为各步使用了不同的条件信息。

这是一个可枚举的数学例子，不是模型训练实验。文末代码会重算四条路径，用来说明适用边界；条件 KL 在实际任务中是否更有效，仍需实验判断。

</details>

全词表计算还涉及大量概率的保存、读取或传输。折中办法是选一个集合 $S_k$，精确计算其中的项，再用样本补尾部：

$$
\widehat D=
\sum_{v\in S_k}P(v)\log\frac{P(v)}{Q(v)}
+\mathbf1_{\{y\notin S_k\}}\log\frac{P(y)}{Q(y)},
\quad y\sim P.
$$

固定前缀和选中集合，取期望就恢复全词表 KL。只保留头部、丢掉尾部，一般有偏。这里的 Top-k 是选择哪些项精确计算，**不是把生成时的采样分布截断成 top-k**。[EMA-PG](https://arxiv.org/html/2602.04417v1)讨论了这一类方法。

<details>
<summary>实现细节：尾部数值无偏后，还要检查尾部梯度</summary>

令 $u_v=\log[P(v)/Q(v)]$，$a_\theta=P/\operatorname{sg}(P)$。固定集合 $S_k$，可以构造

$$
\widehat D_{\mathrm{train}}
=\sum_{v\in S_k}P(v)u_v
+\mathbf1_{\{y\notin S_k\}}
\left[a_\theta(y)\operatorname{sg}(u_y+1)-1\right].
$$

尾项前向等于 $u_y$，直接梯度等于 $(u_y+1)g_y$。它的期望对应尾部精确和的导数，所以头尾相加后，数值与局部梯度都正确。

$+1$ 不能随意删去。整个分布上有 $\sum_vP(v)g_v=0$，仅取尾部后一般不为零；尾部概率质量本身也会变化。

在 $P=(0.8,0.2),Q=(0.5,0.5)$ 中，只精确计算 A。若尾部仅提供 $u_Bg_B$，期望导数会是 2.38629；加入概率质量项后，恢复精确值 1.38629。附带脚本会检查这一点。

集合由 top-k 选择时，上述直接求导把选中索引视为固定；排名交换处需要单独考虑。这里说明的是固定前缀上的局部梯度，没有补入完整序列的前缀分布导数。

</details>

## 让代码对照精确答案

最小验证不需要训练大模型。对二元分布，可以先枚举精确 KL，再检查每种样本损失的导数按 $P$ 加权后是什么。

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

预期输出是 `exact 1.386294`、`k1 0.0`、`k2 1.386294`、`k3 1.875`。`P.detach()` 用精确频率模拟“固定样本后直接反传”的期望，不对抽样频率本身求导。也可以[下载这段完整示例](/downloads/kl-divergence/gradient_demo.py)直接运行。

[完整验证脚本](/downloads/kl-divergence/check_kl.py)还会枚举两步序列，比较完整梯度、局部梯度、reward-to-go，以及上面的方差和 Top-k 例子。它只依赖 Python 标准库；上面这段自动微分示例则需要 PyTorch。

```bash
python3 check_kl.py
```

## 最后回到训练目标

如果想优化完整序列的 reverse KL，当前策略采样下，把 $k_1$ 加入逐步奖励，再计算完整 reward-to-go，是一条清楚的实现路径。如果希望约束已采到前缀上的下一步分布，也可以有意采用局部 token 正则；它表达的是另一种更新目标。

梯度与目标一致，仍不保证某种方法在所有任务上最优。方差、奖励质量、裁剪、数据分布和优化器都会影响结果。比较训练结果时，应先说清楚目标与实现，再比较收益，而不是只按 $k_1$ 或 $k_3$ 的名字判断对错。

阅读一段 KL 代码，最有用的四个问题是：**样本从哪里来，参考模型是谁，哪些量参与求导，优化的是单步还是完整序列。** 这四件事弄清楚，几行看似相似的代码才有明确含义。

## 参考与复现

- 数值估计：[Schulman, Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)。
- 梯度与序列：[Tang & Munos, On a few pitfalls in KL divergence gradient estimation for RL](https://arxiv.org/html/2506.09477v1)。
- 条件 KL：[Better Estimation of the KL Divergence Between Language Models](https://arxiv.org/html/2504.10637v2)。本文附带有限状态枚举，用于区分单步与序列总方差结论。
- 局部正则与 Top-k：[EMA Policy Gradient](https://arxiv.org/html/2602.04417v1)。本文的尾部梯度公式由头尾分解直接推导，并附独立数值检查。
- 实现与实验比较：[Rethinking KL Regularization](https://arxiv.org/html/2510.01555v1)、[A Comedy of Estimators](https://arxiv.org/html/2512.21852v1)、[On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](https://arxiv.org/abs/2505.17508)。
- 中文阅读线索：[繁华落尽见真淳：大模型强化学习中 KL 散度的正确形式是 k1 in Reward](https://mp.weixin.qq.com/s/HRDmhG-ODpuozsOHM__f9Q)。本文将“正确”限定到明确的优化目标与采样条件。
- [验证代码、配图脚本与源文件](/downloads/kl-divergence/sources.zip)。图中的分布与曲线来自明确的数学例子，不代表实际模型训练结果。
