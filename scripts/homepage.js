'use strict';
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const renderHomepage = require('../lib/homepage.cjs');
const {renderProfile} = require('../lib/profile.cjs');
const chrome = require('../lib/chrome.cjs');

// Keep the homepage and theme sidebar on the same markup and contact data.
hexo.extend.filter.register('before_generate', function () {
  this.theme.setView('_widget/blogger.ejs', fs.readFileSync(path.join(this.source_dir, '_volantis/blogger.ejs'), 'utf8'));
  for(const name of ['header','footer']) this.theme.setView(`_partial/${name}.ejs`, fs.readFileSync(path.join(this.source_dir, `_volantis/${name}.ejs`), 'utf8'));
  for(const name of ['category','tag','archive','friends','qq-about']) this.theme.setView(`${name}.ejs`, fs.readFileSync(path.join(this.source_dir, `_volantis/${name}.ejs`), 'utf8'));
});
hexo.extend.helper.register('qq_profile', renderProfile);
hexo.extend.helper.register('qq_header', function(){return chrome.header(this.path || this.page.path || '/');});
hexo.extend.helper.register('qq_footer', chrome.footer);
hexo.extend.helper.register('qq_search', function(){return chrome.search(this.site.posts.sort('date',-1).toArray().map(p=>({title:p.title,date:p.date.format('YYYY-MM-DD'),desc:this.strip_html(p.description||p.excerpt||''),categories:p.categories.toArray().map(c=>c.name),tags:p.tags.toArray().map(t=>t.name),url:p.permalink})));});

// Replace only Hexo's homepage generator. Volantis still renders posts and archives.
hexo.extend.filter.register('after_init', function () {
  hexo.extend.generator.register('index', function (locals) {
    const plain = value => this.extend.helper.get('strip_html').call(this, String(value || '')).replace(/\s+/g, ' ').trim();
    const posts = locals.posts.sort(this.config.index_generator.order_by || '-date').toArray().map(post => ({
      title: post.title,
      date: post.date.format('YYYY-MM-DD'),
      categories: post.categories.toArray().map(category => category.name),
      tags: post.tags.toArray().map(tag => tag.name),
      // Authors can set description or use <!-- more -->. Do not summarize with an LLM.
      desc: plain(post.description || post.excerpt).slice(0, 220),
      url: post.permalink
    }));
    const routes = model => Object.fromEntries(model.toArray().map(item => [item.name, new URL(item.path, this.config.url).href]));
    const revision = file => createHash('sha256').update(fs.readFileSync(path.join(this.source_dir, file))).digest('hex').slice(0, 12);
    return {path: 'index.html', data: renderHomepage({posts, base: this.config.url, categoryRoutes: routes(locals.categories), tagRoutes: routes(locals.tags), chromeVersion: revision('css/chrome.css'), chromeJsVersion: revision('js/chrome.js'), profileVersion: revision('css/profile.css'), cssVersion: revision('css/homepage.css'), jsVersion: revision('js/homepage.js')})};
  });
});

hexo.extend.helper.register('qq_asset', function(file) {
  const content = fs.readFileSync(path.join(hexo.source_dir, file));
  return this.url_for(file) + '?v=' + createHash('sha256').update(content).digest('hex').slice(0, 12);
});
