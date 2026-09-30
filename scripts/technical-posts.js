'use strict';

// Mark technical figures before the theme's lazyload filter (default priority 10).
// Native loading does not depend on the theme's external lazyload script.
hexo.extend.filter.register('after_post_render',function(data){
  if(!data.technical_article)return data;
  data.content=data.content.replace(/<img\b[^>]*>/g,img=>{
    if(!/src="\/images\/(transformer-arithmetic|kl-divergence)\//.test(img))return img;
    return img.replace('<img','<img no-lazy loading="lazy" decoding="async"');
  });
  return data;
},5);
