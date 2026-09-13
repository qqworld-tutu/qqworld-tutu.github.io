'use strict';
const profile = require('../source/_data/profile.json');
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const paths = {
  github: '<path d="M9 19c-4.3 1.3-4.3-2-6-2m12 4v-3.4a3 3 0 0 0-.8-2.3c2.7-.3 5.5-1.3 5.5-6A4.7 4.7 0 0 0 18.4 6a4.3 4.3 0 0 0-.1-3.2s-1-.3-3.4 1.3a11.6 11.6 0 0 0-6 0C6.5 2.5 5.5 2.8 5.5 2.8A4.3 4.3 0 0 0 5.4 6a4.7 4.7 0 0 0-1.3 3.3c0 4.7 2.8 5.7 5.5 6A3 3 0 0 0 9 17.6V21"/>',
  mail: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 6 9 7 9-7"/>',
  academic: '<path d="m2 9 10-5 10 5-10 5-10-5Zm4 2v6c3 3 9 3 12 0v-6m4-2v8"/>'
};
function contacts() {
  return `<nav class="qq-contacts" aria-label="联系与个人主页">${profile.contacts.map(item => `<a href="${esc(item.url)}" title="${esc(item.label === '邮箱' ? item.url.slice(7) : item.label)}"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[item.icon]}</svg><span>${esc(item.label)}</span></a>`).join('')}</nav>`;
}
function renderProfile() {
  const affiliation = esc(profile.affiliation).replace(' &amp; ', '<span class="qq-degree"> &amp; ') + (profile.affiliation.includes(' & ') ? '</span>' : '');
  return `<div class="qq-profile"><a class="qq-avatar" href="${esc(profile.about)}" aria-label="关于 ${esc(profile.name)}"><img no-lazy src="${esc(profile.avatar)}" width="72" height="72" alt="${esc(profile.name)} 的头像"></a><h2>${esc(profile.name)}</h2><p class="qq-affiliation">${affiliation}</p>${contacts()}</div>`;
}
module.exports = {renderProfile, contacts, profile};
