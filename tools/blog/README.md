# 两篇技术博客的维护与复现

正文位于 `source/_posts/transformer-model-arithmetic.md` 和 `source/_posts/kl-divergence-in-rl.md`。
原始草稿未覆盖。当前仓库只准备发布内容，推送部署由用户另行决定。

## 数值核查

```sh
python3 source/downloads/transformer-arithmetic/check_transformer_arithmetic.py
python3 source/downloads/kl-divergence/check_kl.py
```

两个脚本只依赖 Python 标准库。前者核算矩阵参数、缓存、LoRA、生成 FLOPs；后者枚举概率分布并用有限差分检查序列梯度、reward-to-go、Top-k 尾部和方差反例。
正文中的 PyTorch 自动微分例子需要 PyTorch，已用 CPU float64 运行核对。

## 配图

当前 10 张图采用 HTML + 内嵌 SVG，科学曲线由 Matplotlib 生成。共享风格延续已确认的 Q/K/V 原型：白底、细线、明确的阶段、统一的张量颜色与形状。正文嵌入可自动调整高度的 iframe，并保留静态 SVG 用于打印、关闭 JavaScript 和单独查看。

- `source/figures/<topic>/<stem>/index.html`：各图的独立页面，可直接改文字。
- `source/figures/figure.css`：共享排版；由 HTML 生成器中的样式模板生成。
- `source/figures/figure.js`：头共享、MoE、KV Cache、梯度路径等交互。
- `tools/blog/build_html_figures.py`：图示内容与矢量备用图的生成源。若改成品 HTML 后再次运行生成器，成品修改会被覆盖；长期修改请同步到生成源。
- `tools/blog/build-qkv-preview.py`：最初确认的 Q/K/V 样稿与矩阵图元；生成器复用其几何定义。
- `tools/blog/build_editorial_charts.py`：重算 KL 拟合与方差曲线；需 NumPy、SciPy、Matplotlib。可通过 `BLOG_CJK_FONT` 指定中文字体。

```sh
python3 tools/blog/build_editorial_charts.py
python3 tools/blog/build_html_figures.py
python3 tools/blog/package_sources.py
```

Q/K/V 与 SwiGLU 的每个维度单位为 16 px；所有相同形状的矩阵面保持相同尺寸，转置交换高宽。流程节点和专家节点只表示操作或身份，不表示矩阵大小。概率、整数专家编号、连续混合权重分别显示。KV 缓存交互从 T=4 到 T=6，只示意追加位置，不运行模型。

原有 `tools/blog/transformer-arithmetic/` 与 `tools/blog/kl-divergence/` 内的 TikZ / draw.io / Matplotlib 脚本是旧版保留源，运行它们会覆盖同名 SVG，不再是当前图示的构建入口。打包脚本不会用旧图覆盖新图。

## 网页与交互

两篇均使用 `technical_article: true`，只在这些页面加载额外阅读样式、复制按钮、本地 KaTeX 样式与字体。KaTeX HTML 由已有 Markdown 渲染器在构建时生成。
两套交互在 `source/labs/`，用普通 HTML/CSS/JS 实现，不依赖网络库；文章 iframe 按内容高度自动调整。`scripts/technical-posts.js` 在主题处理前为技术配图设置原生懒加载，避免外部懒加载脚本未完成时图片仍被模糊。新版图示提供放大查看与静态矢量图入口。高级推导使用原生 details，即使关闭 JavaScript 仍可展开。

```sh
npm run build
node tools/preview.mjs
# 在另一终端执行，需已安装 Playwright 和 Chrome：
node tools/blog/check-pages.cjs
node tools/blog/check-figures.cjs
```

可通过 `PLAYWRIGHT_MODULE` 指向已有 Playwright 包。预览检查使用独立的无头 Chrome，截图和结果保存在 `output/playwright/`。它检查桌面及 390/360 px 页面宽度、公式解析、10 张图的内嵌高度、交互映射、SVG 备用图、模型预设、KV 头变化、KL 梯度、折叠展开和原图链接。
检查时隔离 giscus 外部请求：尚未有评论的新文章会返回“discussion not found”，这不是正文加载错误。

语言组织参考了 Jay Alammar 的逐层拆解与 Tuan Anh Le 的小分布例子。没有复制其图或连续段落。正文技术依据以各处链接的原始论文、模型配置与实现为准。
