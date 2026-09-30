'use strict';
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');

// Mark technical figures before the theme's lazyload filter (default priority 10).
// Native loading does not depend on the theme's external lazyload script.
hexo.extend.filter.register('after_post_render',function(data){
  if(!data.technical_article)return data;
  data.content=data.content.replace(/<img\b[^>]*>/g,img=>{
    if(!/src="\/images\/(transformer-arithmetic|kl-divergence)\//.test(img))return img;
    return img.replace('<img','<img no-lazy loading="lazy" decoding="async"');
  });
  // Each figure URL changes with its document, including versioned CSS/JS links.
  data.content=data.content.replace(/src="(\/figures\/[^"?]+\/)\?embed=1(?:&amp;v=[a-f0-9]+|&v=[a-f0-9]+)?"/g,(_,url)=>{
    const document=fs.readFileSync(path.join(hexo.source_dir,url.slice(1),'index.html'));
    const revision=createHash('sha256').update(document).digest('hex').slice(0,12);
    return `src="${url}?embed=1&amp;v=${revision}"`;
  });
  return data;
},5);
