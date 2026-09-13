# QQ's Blog

本站是独立的 Hexo 博客，地址为 https://blog.chenquan-tutu.top/ 。学术主页在独立仓库 [qqworld-tutu/homepage](https://github.com/qqworld-tutu/homepage)，地址为 https://www.chenquan-tutu.top/ 。

## 新建文章

```bash
npm run new -- "文章标题"
```

文章会创建在 `source/_posts/`。编辑完成后可在本地预览：

```bash
npm run server
```

## 一键发布

```bash
npm run publish -- "提交说明"
```

发布命令会依次完成：

1. 构建 Hexo 网站；
2. 提交当前仓库中的全部改动；
3. 推送到 GitHub；
4. 由 GitHub Actions 自动部署到 GitHub Pages。

首次在新电脑上使用时，先安装 Node.js 22 或以上版本，再运行一次 `npm ci`。

GitHub 仓库的 Pages 发布来源设置为 GitHub Actions，自定义域名为 `blog.chenquan-tutu.top`。推送 `master` 后自动构建并部署，无需运行 `hexo deploy`。

文章路径保持不变，giscus 仍按 `pathname` 匹配已有评论。迁移前的主域名文章链接由学术主页仓库中的兼容页面跳转到博客。

## 外观维护

原封面插画保存在 `source/images/cover.jpg`，不修改原图。首页使用确认的 A 版排版，由 `scripts/homepage.js` 接入 Hexo，`lib/homepage.cjs` 渲染；样式和交互在 `source/css/homepage.css`、`source/js/homepage.js`。文章、分类、标签、归档、友链和关于页面继续使用 Volantis，样式通过 `source/_volantis/headEnd.ejs` 加载 `source/css/editorial.css`，没有修改 `node_modules`。

首页从 Hexo 内容模型生成文章列表、分类和标签，超过 8 篇自动显示客户端分页；搜索涵盖全部已发布文章的标题、摘要、分类和标签。归档页面保留 Hexo 原生分页。导航、侧栏和其他页面色彩配置在 `_config.volantis.yml`。页面间使用正常跳转，避免不同布局之间的 PJAX 冲突。正文、数学公式、评论和原有文章地址保留。

个人资料与联系方式统一保存在 `source/_data/profile.json`。首页和内页侧栏共同使用它，修改院系、邮箱或链接后重新构建即可；共享样式在 `source/css/profile.css`。站名使用本机的 Avenir Next / Trebuchet MS 字体，不额外下载网络字体。

## 修改首页摘要

在文章 Markdown 顶部的 front matter 里填写 `description`，例如：

```yaml
title: 一篇新文章
date: 2026-09-13 16:00:00
description: 直接写这篇文章讲什么，一两句话即可。
categories:
  - 项目
tags:
  - Python
```

不用改页面代码。没有 `description` 时使用 `<!-- more -->` 前的摘要；两者都没有就不显示摘要，不自动拼凑介绍或截取公式。新增分类、标签会自动出现在首页，新增项目与笔记都按文章日期排序。日期请使用半角冒号。

## 本地验收

运行 `npm run build` 后执行 `node tools/preview.mjs`，访问 http://127.0.0.1:4341/。修改后重新构建并刷新。正式首页的 CSS/JS 带内容版本号；首屏插画直接高优先级加载，不依赖诗词或懒加载脚本。原图尚未转码，当前优化主要消除了约 10 秒的脚本等待。
