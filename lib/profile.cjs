'use strict';
const profile = require('../source/_data/profile.json');
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const paths = {
  github: '<path d="M9 19c-4.3 1.3-4.3-2-6-2m12 4v-3.4a3 3 0 0 0-.8-2.3c2.7-.3 5.5-1.3 5.5-6A4.7 4.7 0 0 0 18.4 6a4.3 4.3 0 0 0-.1-3.2s-1-.3-3.4 1.3a11.6 11.6 0 0 0-6 0C6.5 2.5 5.5 2.8 5.5 2.8A4.3 4.3 0 0 0 5.4 6a4.7 4.7 0 0 0-1.3 3.3c0 4.7 2.8 5.7 5.5 6A3 3 0 0 0 9 17.6V21"/>',
  mail: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 6 9 7 9-7"/>',
  academic: '<path d="m2 9 10-5 10 5-10 5-10-5Zm4 2v6c3 3 9 3 12 0v-6m4-2v8"/>'
};
function contacts() {
  const links=profile.contacts.map(item => `<a href="${esc(item.url)}" title="${esc(item.label === '邮箱' ? item.url.slice(7) : item.label)}"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[item.icon]}</svg><span>${esc(item.label)}</span></a>`).join('');
  return `<nav class="qq-contacts" aria-label="联系与个人主页">${links}<span class="qq-wechat"><button type="button" aria-label="查看微信号" aria-describedby="qq-wechat-id" aria-expanded="false"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M14 10c0-3-2.8-5-6-5S2 7 2 10c0 1.5.7 2.8 2 3.7L3.5 16 6 14.8c.7.2 1.3.2 2 .2m14 0c0-2.5-2.5-4.5-5.5-4.5S11 12.5 11 15s2.5 4.5 5.5 4.5c.5 0 1.1 0 1.6-.2l2.4 1.2-.5-2.1c1.3-.8 2-2 2-3.4Z"/><path d="M5.8 9h.4m3.6 0h.4m4.3 5.5h.4m3.4 0h.4" stroke-width="2" stroke-linecap="round"/></svg><span>微信</span></button><span class="qq-wechat-tip" id="qq-wechat-id" role="tooltip"><strong>WeChat ID:</strong> ${esc(profile.wechat)}</span></span></nav>`;
}
function renderProfile() {
  const affiliation = esc(profile.affiliation).replace(' &amp; ', '<span class="qq-degree"> &amp; ') + (profile.affiliation.includes(' & ') ? '</span>' : '');
  return `<div class="qq-profile"><a class="qq-avatar" href="${esc(profile.about)}" aria-label="关于 ${esc(profile.name)}"><img no-lazy src="${esc(profile.avatar)}" width="72" height="72" alt="${esc(profile.name)} 的头像"></a><h2>${esc(profile.name)}</h2><p class="qq-affiliation">${affiliation}</p>${contacts()}</div>`;
}
module.exports = {renderProfile, contacts, profile};
