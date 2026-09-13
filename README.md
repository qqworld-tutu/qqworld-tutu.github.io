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
