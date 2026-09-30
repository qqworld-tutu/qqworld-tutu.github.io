---
title: Transformer 的参数量、计算量与显存
date: 2026-09-29 12:00:00
permalink: llm/transformer-model-arithmetic/
description: 从一层 Transformer 的矩阵形状出发，算清 Llama、Mixtral 的参数量，再把同一套推导用于 KV Cache、推理计算量和训练显存。附交互计算器与可运行代码。
categories:
  - 大模型
tags:
  - Transformer
  - 模型架构
  - 推理
technical_article: true
---

参数量通常是我们认识一个大模型时最先看到的数字。不过，知道模型有多少参数，还不足以判断它需要多少显存、一次生成又要做多少计算。以一个 8B 模型为例，用 BF16 保存权重大约需要 16 GB，但运行时还要容纳 KV Cache、中间激活和临时工作区，因此 16 GB 显存未必能完成推理。Mixtral 8×7B 则提供了另一个例子，它总共有约 47B 参数，每个 token 参与计算的参数却只有约 13B。

这些差别都可以从模型实际执行的运算中得到解释。参数量取决于模型包含哪些权重，计算量还与这些权重被使用的次数有关，而显存占用则要进一步考虑运行过程中同时保留了哪些张量。三者有共同的结构基础，也各自依赖不同的条件。

本文从一层 Transformer 的矩阵形状出发，逐步推导这三个量的估算方法，并用 Llama 和 Mixtral 的配置核对结果。我们关心的不只是公式本身，也包括公式中各项的来源，以及哪些实现细节会使实际结果偏离估算。文中的“B 参数”表示十亿个参数；容量使用 GiB 时，按 $2^{30}$ 字节计算。

<!-- more -->

## 模型结构与记号

为了使讨论具体一些，我们以 Llama 一类 decoder-only 模型为例。它采用 Pre-RMSNorm 和 RoPE，线性层不带偏置，前馈部分使用 SwiGLU。后面的计数都建立在这些约定上，遇到不同结构时，再补上相应的参数项。

<div class="technical-figure"><iframe class="article-figure" scrolling="no" src="/figures/transformer-arithmetic/01-decoder-architecture/?embed=1" title="Decoder-only 模型从词嵌入、Transformer Block 到词表输出的计算流程。" loading="lazy"></iframe><a class="figure-static" href="/images/transformer-arithmetic/01-decoder-architecture.svg" target="_blank" rel="noopener"><img src="/images/transformer-arithmetic/01-decoder-architecture.svg" alt="Decoder-only 模型从词嵌入、Transformer Block 到词表输出的计算流程。" no-lazy></a></div>

输入 token 首先通过词嵌入表转换为向量，随后经过若干个 Transformer Block。在每个 Block 中，Attention 根据可见位置的信息更新表示，FFN 则对各个位置分别进行非线性变换，两部分的输出都通过残差相加回到主干。经过所有层之后，最终表示再由输出头映射为词表中各个 token 的分数。

这条计算路径中，只有一部分操作包含可学习参数。例如，线性投影需要保存权重矩阵，而残差相加、softmax、SiLU 和因果 mask 本身不引入参数。本文采用的 RoPE 也没有可学习的位置表。因此，在计算参数量时，我们可以先把注意力放在需要保存的权重上，稍后估算计算量和显存时，再考虑其余操作。

下面列出全文使用的主要记号。除特别说明外，同一个符号在各节中保持相同含义。

| 符号 | 含义 |
|---|---|
| $d$ | 每个 token 的隐藏维度，也是残差流的宽度 |
| $L$ | Transformer Block 的层数 |
| $V$ | 词表大小；注意与 Attention 中的 Value 张量区分 |
| $h$、$n_{kv}$ | Query 头数、Key/Value 头数 |
| $d_h$ | 每个头的维度 |
| $d_{ff}$ | FFN 的中间维度 |
| $B$ | 一个 batch 中的序列数 |
| $S$、$T$ | 一次 prefill 的输入长度、某一步的缓存长度 |

参数量的计算可以归结为对矩阵元素的计数。一个将 $m$ 维输入映射到 $n$ 维输出的线性层，需要保存一个 $m\times n$ 的权重矩阵，因而包含 $mn$ 个权重；若另外使用 bias，还需加上 $n$ 个参数。

本文将线性变换写成 $XW$，所以权重形状按“输入维度 × 输出维度”排列。PyTorch 的 `Linear.weight` 使用相反的存储次序，前向计算时再转置。这个差异会影响形状的写法，但不会改变元素总数。

## Attention 的参数量

先看 Attention。它的主要参数来自生成 Q、K、V 的三个投影，以及将各头输出映射回隐藏空间的输出投影。输入宽度为 $d$，每个头的宽度为 $d_h$；Q 包含 $h$ 个头，K 和 V 各包含 $n_{kv}$ 个头，因此四个矩阵的形状分别为

$$
\begin{aligned}
W_Q&\in\mathbb R^{d\times(hd_h)},\\
W_K,W_V&\in\mathbb R^{d\times(n_{kv}d_h)},\\
W_O&\in\mathbb R^{(hd_h)\times d}.
\end{aligned}
$$

这里将所有头的投影合并写在一个矩阵中。得到 Q、K、V 后，再沿特征维拆成各个头，分别完成注意力计算。最后将各头的结果沿特征维拼接，由 $W_O$ 映射回 $d$ 维。

<div class="technical-figure"><iframe class="article-figure" scrolling="no" src="/figures/transformer-arithmetic/02-qkvo-tensor-flow/?embed=1" title="GQA 中 Q、K、V 的投影、拆头与注意力计算，以及各步的张量形状。" loading="lazy"></iframe><a class="figure-static" href="/images/transformer-arithmetic/02-qkvo-tensor-flow.svg" target="_blank" rel="noopener"><img src="/images/transformer-arithmetic/02-qkvo-tensor-flow.svg" alt="GQA 中 Q、K、V 的投影、拆头与注意力计算，以及各步的张量形状。" no-lazy></a></div>

为了看清拆头前后的形状，可以取一个小例子。假设输入包含 3 个 token，$d=8$，共有 4 个 Query 头和 2 个 KV 头，每个头的宽度为 2。此时各步的矩阵乘法如下。

| 步骤 | 形状变化 |
|---|---|
| Q 投影 | $[3,8][8,8]\to[3,8]$，拆成 4 个 $[3,2]$ 的头 |
| K/V 投影，各一次 | $[3,8][8,4]\to[3,4]$，各拆成 2 个 $[3,2]$ 的头 |
| 一个 Query 头的匹配 | $[3,2][2,3]\to[3,3]$ 的分数矩阵 |
| 概率加权 Value | $[3,3][3,2]\to[3,2]$ 的头输出 |
| 拼接与输出投影 | 4 个头拼成 $[3,8]$，再乘 $[8,8]$ 的 $W_O$ |

拆头之后，每个头依然保留全部 3 个 token，发生变化的是特征维的组织方式。在这个例子中，每两个 Query 头共用一组 K/V，但由于 Query 不同，它们会分别得到自己的注意力概率，输出也一般不同。

按照矩阵形状相加，四个投影共有 $8\times8+2\times8\times4+8\times8=192$ 个权重。若把输入增加到 30 个 token，这些矩阵仍然是同一组矩阵，只是应用到了更多位置上。增加的是计算量和中间张量的大小，参数量仍为 192。

将同样的计数推广到一般维度，便得到

$$
P_{\mathrm{attn}}=2d(hd_h)+2d(n_{kv}d_h).
$$

常见配置满足 $hd_h=d$，于是

$$
\boxed{P_{\mathrm{attn}}=2d^2+2dn_{kv}d_h.}
$$

其中，第一项对应 Q 和输出投影，第二项对应 K、V 投影。这个化简依赖于 $hd_h=d$，而有些模型会单独指定 `head_dim`，使总头宽与隐藏维度不同。对于这些配置，应回到化简前的表达式逐项计算。

### 头数与参数共享

<div class="technical-figure"><iframe class="article-figure" scrolling="no" src="/figures/transformer-arithmetic/03-mha-gqa-mqa/?embed=1" title="MHA、GQA 和 MQA 中 Query 头与 KV 头的共享关系。" loading="lazy"></iframe><a class="figure-static" href="/images/transformer-arithmetic/03-mha-gqa-mqa.svg" target="_blank" rel="noopener"><img src="/images/transformer-arithmetic/03-mha-gqa-mqa.svg" alt="MHA、GQA 和 MQA 中 Query 头与 KV 头的共享关系。" no-lazy></a></div>

| 结构 | KV 头数 | 在 $hd_h=d$ 时的投影参数 |
|---|---:|---:|
| MHA | $n_{kv}=h$ | $4d^2$ |
| GQA | $1<n_{kv}<h$ | $2d^2+2dn_{kv}d_h$ |
| MQA | $n_{kv}=1$ | $2d^2+2dd_h$ |

[GQA](https://arxiv.org/abs/2305.13245)将若干 Query 头分为一组，让组内的头使用相同的 K/V。例如取 $h=32,n_{kv}=8$，就有每 4 个 Query 头共享一组 K/V。与相同隐藏宽度的 MHA 相比，K/V 投影的参数量降为四分之一，Q 和输出投影则保持不变，所以整个 Attention 的参数量只减少了其中一部分。

这也说明了为什么在前面的公式中不需要再额外乘一次头数。头数已经包含在投影的输出宽度里；只要总宽度 $hd_h$ 不变，将这些列划分为多少个头，就只是对同一批元素作不同的分组。

## FFN 与 SwiGLU

接下来考虑前馈部分。传统 FFN 由两次线性变换和中间的激活函数组成，可以写成

$$
\operatorname{FFN}(x)=\phi(xW_1)W_2.
$$

其中，两次变换的形状分别为 $d\times d_{ff}$ 和 $d_{ff}\times d$。第一步将表示映射到中间维度，第二步再映射回来，因而在不含 bias 时共有 $2dd_{ff}$ 个参数。

SwiGLU 在此基础上引入了一条门控支路，将前馈变换改为

$$
\operatorname{SwiGLU}(x)
=\bigl[\operatorname{SiLU}(xW_1)\odot(xW_3)\bigr]W_2.
$$

<div class="technical-figure"><iframe class="article-figure" scrolling="no" src="/figures/transformer-arithmetic/06-swiglu/?embed=1" title="SwiGLU 的两个投影结果经过门控相乘，再由输出投影映射回隐藏维度。" loading="lazy"></iframe><a class="figure-static" href="/images/transformer-arithmetic/06-swiglu.svg" target="_blank" rel="noopener"><img src="/images/transformer-arithmetic/06-swiglu.svg" alt="SwiGLU 的两个投影结果经过门控相乘，再由输出投影映射回隐藏维度。" no-lazy></a></div>

$W_1$ 与 $W_3$ 分别产生两个 $d_{ff}$ 维的中间结果，其中一个经过 SiLU 后，与另一个逐元素相乘。相乘后的宽度仍为 $d_{ff}$，最后再由 $W_2$ 映射回 $d$。这样一来，需要保存的权重矩阵由两个变成了三个，参数量为

$$
\boxed{P_{\mathrm{SwiGLU}}=3dd_{ff}.}
$$

和 Attention 投影一样，这三块矩阵在所有 token 之间共享。序列变长会增加它们的使用次数，但不会增加 FFN 的参数个数。

若固定中间维度，SwiGLU 的参数量确实比传统 FFN 多出一半。不过，实际设计通常会同时调整中间维度，以维持相近的参数和计算预算。例如，要与 $d_{ff}=4d$ 的传统 FFN 对齐，可以将 SwiGLU 的中间维度设为约 $8d/3$，再按实现要求取整。[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)中讨论了这类门控结构的比较。

## 完整模型的参数量

有了 Attention 和 FFN 的结果，还需要补上归一化层、词嵌入和输出头。RMSNorm 通常为每个特征保存一个缩放参数，所以每层的两个 RMSNorm 共贡献 $2d$ 个参数，所有 Block 之后的最终 RMSNorm 再贡献 $d$ 个。若使用同时包含缩放和偏移的 LayerNorm，则每个 Norm 应计为 $2d$。

词嵌入表的形状为 $V\times d$，输出头的形状为 $d\times V$，两者各有 $Vd$ 个参数。有些模型将它们绑定为同一组权重，这时只需要计数一次。这样的共享减少了需要保存的参数，但生成 logits 时的矩阵乘法仍然需要执行。

沿用前面的结构约定，并假设 $hd_h=d$，将一层中的 Attention、SwiGLU 和两个 RMSNorm 相加，就有

$$
P_{\mathrm{layer}}=2d^2+2dn_{kv}d_h+3dd_{ff}+2d.
$$

设 $c=1$ 表示两端共享权重，$c=2$ 表示不共享，则

$$
\boxed{N=LP_{\mathrm{layer}}+cVd+d.}
$$

这个表达式只涉及前面约定的模块，并没有包含 bias、可学习位置表或额外的 Norm。若模型采用了这些结构，只需在对应位置补上它们的参数量。例如，长度上限为 $S_{\max}$ 的可学习位置表会额外贡献 $S_{\max}d$ 个参数。

### Llama-2 7B

[发布配置](https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json)中，$d=4096,L=32,h=n_{kv}=32,d_h=128,d_{ff}=11008,V=32000$，词嵌入与输出头不共享。

| 参数项 | 计算 | 结果 |
|---|---|---:|
| 每层 Attention | $4\times4096^2$ | 67,108,864 |
| 每层 SwiGLU | $3\times4096\times11008$ | 135,266,304 |
| 每层两个 RMSNorm | $2\times4096$ | 8,192 |
| 每层合计 | 上述三项相加 | 202,383,360 |
| 32 层 | $32\times202,383,360$ | 6,476,267,520 |
| 两端与最终 Norm | $2\times32000\times4096+4096$ | 262,148,096 |
| **总参数** | 所有项相加 | **6,738,415,616** |

计算结果约为 6.74B，与名称中的“7B”属于同一个近似规模。表中还可以看到，FFN 约占每层参数的三分之二。虽然讨论 Transformer 时往往更关注 Attention，但在这个配置下，前馈部分才是参数量的主要来源。

### Llama-3 8B

再看 Llama-3 8B。根据[官方配置](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json)，它保留了相同的层数和隐藏维度，但将 KV 头数降到 8，同时把 FFN 中间维度增至 14336、词表大小增至 128256。它也不共享词嵌入与输出头，因此可以直接用同一个公式比较两者。

| 参数项 | Llama-2 7B | Llama-3 8B |
|---|---:|---:|
| 每层 Q/O | 33,554,432 | 33,554,432 |
| 每层 K/V | 33,554,432 | 8,388,608 |
| 每层 SwiGLU | 135,266,304 | 176,160,768 |
| 每层两个 RMSNorm | 8,192 | 8,192 |
| 32 层合计 | 6,476,267,520 | 6,979,584,000 |
| 两端与最终 Norm | 262,148,096 | 1,050,677,248 |
| **总参数** | **6,738,415,616** | **8,030,261,248** |

尽管 GQA 减少了 K/V 投影的参数，更宽的 FFN 和更大的词表带来了更大的增量，最终总参数量上升到约 8.03B。可见，层数和隐藏维度只能描述模型规模的一部分，前馈维度、词表以及参数共享方式也会明显影响结果。

## MoE 的总参数与激活参数

前面讨论的 Dense 模型会在每个 token 上使用完整的 FFN。Mixtral 一类稀疏 MoE 则保留 Attention，将 FFN 扩展为多个专家，再由 Router 为每个 token 选择其中的一部分。于是，模型需要保存的参数和一个 token 实际使用的参数开始出现差别。

<div class="technical-figure"><iframe class="article-figure" scrolling="no" src="/figures/transformer-arithmetic/04-moe-routing/?embed=1" title="MoE 的路由概率、专家选择、输入分发与输出加权汇总。" loading="lazy"></iframe><a class="figure-static" href="/images/transformer-arithmetic/04-moe-routing.svg" target="_blank" rel="noopener"><img src="/images/transformer-arithmetic/04-moe-routing.svg" alt="MoE 的路由概率、专家选择、输入分发与输出加权汇总。" no-lazy></a></div>

设每层包含 $E$ 个 SwiGLU 专家，每个 token 选择其中的 $k$ 个。每个专家有 $3dd_{ff}$ 个参数，而 Router 使用一个 $d\times E$ 的矩阵为所有专家打分。由此得到 FFN 子层的总参数量与激活参数量

$$
\begin{aligned}
P_{\mathrm{MoE,total}}&=E\cdot3dd_{ff}+dE,\\
P_{\mathrm{MoE,active}}&=k\cdot3dd_{ff}+dE.
\end{aligned}
$$

两个式子的差别只在专家数量上。由于 Router 每次都要计算全部专家的分数，它的 $dE$ 个权重在两种计数中都需要保留。选中的专家分别处理同一个 token，输出再按路由权重相加，得到 $d$ 维结果；不同 token 可以选择不同的专家组合。

[Mixtral 8×7B](https://arxiv.org/abs/2401.04088)在每层设置 8 个专家，每个 token 选择其中 2 个。其[官方配置](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json)为 $d=4096,L=32,n_{kv}=8,d_h=128,d_{ff}=14336,V=32000$。将这些数值代入，并加上 Attention、Norm、词嵌入和输出头，可得

$$
N_{\mathrm{total}}=46,702,792,704,
$$

$$
N_{\mathrm{active}}=12,879,925,248.
$$

这就对应于通常所说的约 47B 总参数和约 13B 激活参数。“8×7B”中的乘号容易让人联想到 56B，但这里扩展的是 FFN，其他模块并没有复制八份。另外，上述激活参数是按参与计算的模块统计的，其中也计入了完整的 embedding 模块，没有逐个追踪一次查表实际访问的元素。

部署时，只要所有专家都常驻设备，就仍需为全部权重分配存储。激活参数较少，主要意味着每个 token 执行的专家矩阵乘法较少，因此它更适合用来估算计算量。实际耗时还受到路由、跨卡传输和负载均衡的影响，不能简单地认为会按 $k/E$ 的比例缩减。

上述模型在各层采用相同的专家结构，所以可以按层计数后统一求和。对于只在部分层使用 MoE，或另外设置了共享专家的模型，则需要按各层的实际结构分别计算，再对 $L$ 层求和。

## KV Cache 的大小

参数量确定之后，就可以继续考虑推理过程中的中间状态。自回归生成通常分为两个阶段，首先是一次处理完整 prompt 的 prefill，随后是逐步生成的 decode。每次 decode 将一个新 token 送入模型，用它对应的输出预测再下一个 token。

在因果注意力中，新增一个位置并不会改变此前各位置的表示。因此，已经算出的 Key 和 Value 可以保存下来，供后续位置的 Query 读取，从而省去重复计算。历史 Query 则不会再参与后续位置的注意力计算，所以没有同样的缓存需求。

<div class="technical-figure"><iframe class="article-figure" scrolling="no" src="/figures/transformer-arithmetic/05-kv-cache/?embed=1" title="Prefill 建立缓存，decode 为每层追加当前 token 的 K/V 并读取历史缓存。" loading="lazy"></iframe><a class="figure-static" href="/images/transformer-arithmetic/05-kv-cache.svg" target="_blank" rel="noopener"><img src="/images/transformer-arithmetic/05-kv-cache.svg" alt="Prefill 建立缓存，decode 为每层追加当前 token 的 K/V 并读取历史缓存。" no-lazy></a></div>

缓存的大小可以直接由形状读出。在一层中，K 和 V 的形状均为 $[B,n_{kv},T,d_h]$。各层使用不同的权重和输入表示，因此需要分别保存自己的缓存。若每个元素占 $b$ 字节，将两份缓存以及全部 $L$ 层相加，得到

$$
\boxed{M_{\mathrm{KV}}=2BLTn_{kv}d_hb.}
$$

式子中的系数 2 来自 K、V 两份张量，存储精度则由 $b$ 单独表示，例如 BF16 对应 $b=2$。这里的 $T$ 包含 prompt 和已经送入模型的生成 token；当一步 decode 将当前 K/V 追加到缓存后，当前 Query 也可以关注这个新位置。

Llama-3 8B 每个 token、每条序列的 BF16 KV 为

$$
2\times32\times8\times128\times2
=131,072\text{ bytes}=128\text{ KiB}.
$$

因此，一条序列缓存 8192 个 token 时需要 1 GiB。作为比较，Llama-2 7B 有 32 个 KV 头，每个 token 需要 512 KiB，缓存 4096 个 token 就要占用 2 GiB。虽然它的模型参数更少，但较多的 KV 头使缓存增长得更快。

下面的计算器可以用来比较这些变化。固定模型后，将序列长度或 batch 翻倍，参数量保持不变，KV Cache 则随之翻倍；若将 KV 头数从 8 改为 32，K/V 投影的参数量和缓存大小都会变化。

<iframe class="article-lab arithmetic-lab" src="/labs/transformer-arithmetic/" title="Transformer 参数与 KV Cache 计算器" loading="lazy"></iframe>

[单独打开计算器](/labs/transformer-arithmetic/)。计算器只统计所列结构的权重和 KV 数据，不含运行时开销。

以上公式给出的是 KV 数据本身的容量。实现中还可能有分页、对齐和量化元数据等额外开销。对于长度不同的一组序列，若缓存按实际长度紧凑存储，可以将 $BT$ 换为 $\sum_iT_i$；若按最大长度预分配，则可能占用更多空间。[PagedAttention](https://arxiv.org/abs/2309.06180)主要针对这类内存分配与共享问题。

还需要区分模型权重和缓存各自使用的精度，权重量化并不意味着 KV 也采用了相同的量化方式。类似地，滑动窗口限制的是可见范围，只有实现同时释放了窗口外的缓存，才会减少实际占用；单独使用窗口 mask 无法达到这一点。

## 计算量的估算

参数量的推导已经给出了各个矩阵的形状，估算计算量时，只需进一步考虑它们如何相乘。一个 $m\times k$ 矩阵与一个 $k\times n$ 矩阵相乘，约需执行 $mkn$ 次乘加。本文将一次乘法和一次加法分别计为一个 FLOP，因此这次矩阵乘法约需 $2mkn$ FLOPs。

由此可知，一个包含 $P$ 个权重的线性层，在 $BS$ 个 token 上各使用一次时，主要计算量约为 $2BSP$。不过，Attention 中还包含 $QK^\top$ 和注意力概率与 Value 的乘法。这些运算不引入新的权重，却会产生计算开销，所以需要在投影部分之外单独计入。

### Prefill 阶段

对于 Dense 模型，记一层中 Attention 投影与 FFN 的权重总数为 $P_{\mathrm{linear}}$。Prefill 会一次处理长度为 $S$ 的输入，因此线性部分的计算量为

$$
F_{\mathrm{linear}}\approx2BSP_{\mathrm{linear}}.
$$

对于 MoE，这一项应按每个 token 激活的专家统计，并另外考虑路由开销。再看注意力的核心运算。在因果条件下，第一个位置只与自身匹配，第二个位置可与前两个位置匹配，依此类推，总共包含 $S(S+1)/2$ 对有效位置。每对位置在 QK 点积和 Value 汇总中分别贡献约 $2d_h$ FLOPs，对 $h$ 个头求和，并利用 $hd_h=d$，便有

$$
F_{\mathrm{core}}\approx2BdS(S+1).
$$

所以一层的主要 prefill 计算量为

$$
\boxed{F_{\mathrm{prefill,layer}}\approx
2BSP_{\mathrm{linear}}+2BdS(S+1).}
$$

这里仅计入了因果下三角中的有效配对。若按完整的 $S\times S$ 矩阵计算，注意力核心项则约为 $4BS^2d$，两种写法的主导系数相差一倍。它们对应不同的计算口径；实际 kernel 采用分块实现时，也可能执行部分被 mask 掉的位置所对应的运算。

GQA 减少了 K/V 的投影宽度，但仍然保留 $h$ 份 Query 和各自的注意力概率。因此，上面的核心计算量并不会再按 $n_{kv}/h$ 缩减。输出头也在层内公式之外，若为全部位置计算 logits，需要 $2BSdV$ FLOPs；推理中若只计算末位置，则只需 $2BdV$。

### Decode 阶段

Decode 每步只处理 $B$ 个新 token，所以线性部分变为约 $2BP_{\mathrm{linear}}$ FLOPs。与此同时，新 Query 仍要与长度为 $T$ 的缓存交互，两次注意力乘法合计约需 $4BTd$ FLOPs。将两部分相加，得到

$$
\boxed{F_{\mathrm{decode,layer}}\approx
2BP_{\mathrm{linear}}+4BTd.}
$$

对整个模型，还需将层内结果乘以 $L$，再加上输出头的 $2BdV$。可以看到，一步 decode 的注意力计算量随缓存长度线性增长；连续生成多个 token 时，$T$ 逐步增加，因此总计算量应对各步分别累加。

累加时还需要考虑生成过程的起点。Prefill 的末位置 logits 已经用于采样第一个新 token，因此若总共输出 $G$ 个 token，随后只需执行 $G-1$ 次 decode。最后一个输出 token 若没有再送回模型，它对应的 K/V 也就尚未写入缓存。

### 训练计算量

训练还需要在前向之后执行反向传播。设每个 token 都使用的一组线性权重共有 $N_{\mathrm{matmul}}$ 个参数，其前向计算量约为 $2N_{\mathrm{matmul}}$ FLOPs。反向中还要计算输入梯度与权重梯度，两部分的总量通常约为前向的两倍。因此，对 $D$ 个 token 进行训练时，有

$$
C_{\mathrm{train}}\approx6N_{\mathrm{matmul}}D.
$$

常见的 $6ND$ 估算进一步用模型参数量近似上述矩阵权重数，[Chinchilla](https://arxiv.org/abs/2203.15556)中也采用了相应的训练计算估算。使用这个近似时，需要记住它的适用范围。Embedding 查表不会将完整的词表矩阵乘一遍，注意力核心运算却会在长序列下贡献额外开销；若启用激活重算，还会增加前向计算的次数。对于 MoE，主要矩阵运算则应按激活专家统计。

参数量能够用来近似计算量，是因为我们对权重的使用方式作了约定。参数共享、稀疏专家和序列长度一旦改变，就需要重新检查这个约定，而不能只将新的总参数量代入同一个系数。

## 显存的组成

### 推理显存

显存可以沿用前面对张量形状的分析，但需要统计的是某一时刻同时存在的状态。最直接的一项是模型权重，$N$ 个参数、每个占 $b$ 字节时，理想存储量为 $Nb$。例如，恰好 7B 个参数使用 BF16 保存时需要 14 GB，使用 INT4 时需要 3.5 GB，这里的 GB 均按十进制计。量化所需的 scale、zero point，以及仍保留高精度的层，会使实际容量有所增加。

除了权重，推理还要保存 KV Cache，并为当前激活、临时工作区和运行时分配空间。因此可以近似写成

$$
M_{\mathrm{infer}}\approx
M_{\mathrm{weights}}+M_{\mathrm{KV}}+M_{\mathrm{workspace}}+M_{\mathrm{runtime}}.
$$

以 Llama-3 8B 为例，其 BF16 权重约占 14.96 GiB，再加上一条 8192-token 序列的 BF16 KV，就已接近 16 GiB。由于这里尚未计入临时计算空间，仅按权重容量选择设备，很容易低估实际需求。

### 训练显存

训练时，模型状态中还包括梯度和优化器状态。以一种使用 BF16 和 Adam 的混合精度方案为例，每个参数对应的存储如下。

| 状态 | 每参数字节数 |
|---|---:|
| BF16 权重 | 2 |
| BF16 梯度 | 2 |
| FP32 主权重副本 | 4 |
| Adam 的一阶与二阶状态 | 8 |
| **合计** | **16** |

将这些状态相加，便得到常见的每参数 16 字节估算。这个数值对应的是表中的具体方案；若改用 FP32 梯度、不保存主权重副本，或对优化器状态进行量化，就需要重新求和。

模型状态之外，反向传播还需要使用前向的中间激活。它们的大小依赖于 batch、序列长度、FFN 宽度、Attention 实现和重算策略，因而不能仅由参数量决定。例如，[FlashAttention](https://arxiv.org/abs/2205.14135)通过避免保存完整的注意力矩阵来减少这部分开销，activation checkpointing 则以额外重算换取更少的激活存储。

如果进一步采用数据并行切分，单卡需要持有的模型状态还会减少。沿用上面的 16 字节方案，设数据并行度为 $p$，[ZeRO](https://arxiv.org/abs/1910.02054)在三个阶段中逐步切分优化器状态、梯度和参数，可得到每卡模型状态的近似容量

$$
\begin{aligned}
\mathrm{ZeRO\text{-}1}:&\quad4N+12N/p,\\
\mathrm{ZeRO\text{-}2}:&\quad2N+14N/p,\\
\mathrm{ZeRO\text{-}3}:&\quad16N/p.
\end{aligned}
$$

这些表达式没有计入激活、通信缓冲，以及计算时临时聚合的权重，因此也不等于显存峰值。并行切分改变的是模型状态在设备之间的分布，全模型的参数总量仍然相同。

### LoRA 的情形

LoRA 则从可训练参数的范围入手。它将一个 $d_{\mathrm{in}}\times d_{\mathrm{out}}$ 的权重更新表示为内维度为 $r$ 的两个矩阵之积，因此新增

$$
P_{\mathrm{LoRA}}=r(d_{\mathrm{in}}+d_{\mathrm{out}})
$$

个可训练参数。若在 Llama-2 7B 每层的 Q/V 投影上加 $r=8$ 的 LoRA，一共是

$$
32\times2\times8(4096+4096)=4,194,304.
$$

这约为 419 万个可训练参数，相比 6.74B 的基座小得多。不过，冻结的基座仍参与前向计算，也仍需保存权重，所以减少可训练参数并不等于按相同比例减少全部显存。若把 LoRA 用于 GQA 的 K/V 投影，还需注意它们的输出宽度为 $n_{kv}d_h$，应按实际矩形形状计数。相关低秩更新的定义可见 [LoRA 论文](https://arxiv.org/abs/2106.09685)。

## 一个完整的估算例子

下面把参数、缓存和计算量放到同一个例子中。仍取 Llama-3 8B，权重与 KV 均使用 BF16，模型保存在单个设备上，不作切分。为了区分结构带来的结果与实现带来的开销，以下会明确给出每一步的假设，数值也都属于估算，而非硬件实测。

### 可以容纳多少条序列

假设设备有 24 GiB 显存，先为临时激活、工作区和运行时预留 3 GiB。扣除模型权重后，可以分配给 KV Cache 的容量约为

$$
24-3-14.9575\approx6.0425\text{ GiB}.
$$

若每条序列都缓存 8192 个 token，其 KV 大小为 1 GiB，因此在这些假设下可容纳 6 条序列。实际部署时，预留的 3 GiB 是否足够，还要通过 prefill 峰值和内存分配情况来确认，同时也需要遵守模型的上下文长度上限。

### 一次生成的计算量

接下来考虑单条请求，输入 1024 个 token，并生成 128 个 token。令 $S_0=1024,G=128,R=G-1=127$，每层线性权重为 $P_{\mathrm{linear}}=218,103,808$。沿用只统计有效因果配对、prefill 只计算末位置 logits 的约定，将一次 prefill 与后续 decode 相加，有

$$
\begin{aligned}
F_{\mathrm{linear}}&=2BLP_{\mathrm{linear}}(S_0+R),\\
F_{\mathrm{core}}&=2BLdS_0(S_0+1)\\
&\quad+4BLd\left(RS_0+\frac{R(R+1)}2\right),\\
F_{\mathrm{head}}&=2BGdV.
\end{aligned}
$$

取单条序列对应的 batch 为 1，三项合计约 $16.55\times10^{12}$ FLOPs。生成结束时，最后一个输出 token 尚未送回模型，所以缓存长度为 $1024+127=1151$，对应 143.875 MiB。这与输出总长度相差一个位置，正是前面区分 prefill 与 decode 次数的结果。

### 从计算量到延迟

计算量确定之后，还不能直接得到运行时间，因为执行这些运算也需要从显存搬运权重和缓存。如果暂时只考虑算力与显存带宽，可以写出一个简单的下界

$$
t\gtrsim\max\left(
\frac{\text{FLOPs}}{\text{每秒算力}},
\frac{\text{搬运字节数}}{\text{显存带宽}}
\right).
$$

单条序列、缓存长度 8192 的一步 decode，主要计算量约 19.3 GFLOPs。在假设的 200 TFLOPs/s 算力下，计算下界约 0.097 ms；但若线性权重与 KV 各从显存读一遍，约需搬运 16.08 GB，在 1 TB/s 下就要约 16.1 ms。

在这个假设下，数据搬运所需的时间明显高于纯计算下界，小 batch decode 因而更容易受到显存带宽的限制。增大 batch 后，一次权重读取可以供多个 token 使用，有机会提高计算资源的利用率。实际延迟仍取决于缓存命中、kernel 调度和额外读写等因素，不能由上述下界直接代替。

## 代码核对

为了核对前面的算例，本文附上了核算脚本 [check_transformer_arithmetic.py](/downloads/transformer-arithmetic/check_transformer_arithmetic.py)，只依赖 Python 标准库。它无需下载模型，而是根据配置逐块统计矩阵元素，再与闭式表达式比较，并复算 KV Cache、LoRA 以及上面请求中的计算量。

将上面的脚本链接另存为 `check_transformer_arithmetic.py`，在文件所在目录打开终端，运行下面的命令。

```bash
python3 check_transformer_arithmetic.py
```

运行后，三个模型的参数统计结果如下。

```text
Llama-2 7B       total=6,738,415,616 active=6,738,415,616
Llama-3 8B       total=8,030,261,248 active=8,030,261,248
Mixtral 8x7B     total=46,702,792,704 active=12,879,925,248
```

如果已经构建了 PyTorch 模型对象，也可以直接遍历参数进行核对。

```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

这两个结果分别对应总参数量与当前可训练参数量。当共享权重绑定为同一个 `Parameter` 时，`model.parameters()` 不会重复遍历它；若直接对 `state_dict()` 求和，则可能把共享别名或非参数 buffer 一并计入。对于支持相应初始化方式的大模型，还可以使用 [meta device](https://docs.pytorch.org/docs/stable/meta.html)只建立形状，从而在不分配真实权重的情况下完成统计。

对于新的模型结构，同样可以从配置与实际模块出发，依次确认投影宽度、FFN 形式、参数共享方式，以及专家和位置编码等额外结构。本文中的参数公式都是由这些具体形状相加得到的，而计算量和显存的估算则在此基础上加入了使用次数与状态保存方式。当估算与实测存在差异时，也就可以回到这些假设中逐项查找原因。

## 参考与复现

- 架构：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[GQA](https://arxiv.org/abs/2305.13245)、[GLU Variants](https://arxiv.org/abs/2002.05202)、[RMSNorm](https://arxiv.org/abs/1910.07467)。
- 模型：[Llama 2](https://arxiv.org/abs/2307.09288)、[Llama 3 模型卡](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)、[Mixtral](https://arxiv.org/abs/2401.04088)。
- 系统：[Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)、[PagedAttention](https://arxiv.org/abs/2309.06180)、[FlashAttention](https://arxiv.org/abs/2205.14135)、[ZeRO](https://arxiv.org/abs/1910.02054)。
- [核算代码与可编辑配图源文件](/downloads/transformer-arithmetic/sources.zip)。所有算例都遵循正文列出的结构、精度与运算计数假设。
