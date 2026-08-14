// 低空遥感智能分析平台 — 共享前端逻辑
// 核心安全原则：所有动态文本一律用 textContent / el() 写入，绝不 innerHTML 拼接，
// 因此日志或文件名里出现 < > & </pre> 都不会破坏页面（这是旧版崩坏的根因）。

// ---- DOM 小工具 ----
function $(sel, root = document) { return root.querySelector(sel); }
function $all(sel, root = document) { return Array.from(root.querySelectorAll(sel)); }

// 创建元素：el('div', {class:'x'}, ['文本', el('span',{},['子')])])
function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;          // 安全文本
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) e.setAttribute(k, v);
  }
  // 递归展平 children，支持任意层级嵌套数组
  function _append(c) {
    if (c === null || c === undefined) return;
    if (Array.isArray(c)) { c.forEach(_append); return; }
    e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  _append(children);
  return e;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

// ---- 轻量 toast 提示 ----
function toast(msg, kind = 'ok') {
  let box = $('#toastBox');
  if (!box) {
    box = el('div', { id: 'toastBox' });
    box.style.cssText = 'position:fixed;bottom:24px;left:50%;transform:translateX(-50%);z-index:2000;display:flex;flex-direction:column;gap:8px;align-items:center';
    document.body.appendChild(box);
  }
  const colors = { ok: 'var(--green)', err: 'var(--red)', info: 'var(--blue)' };
  const t = el('div', { text: msg });
  t.style.cssText = `background:var(--panel-2);border:1px solid ${colors[kind] || colors.ok};color:var(--text);padding:9px 16px;border-radius:8px;font-size:13px;box-shadow:0 4px 16px rgba(0,0,0,.4);opacity:0;transition:opacity .15s`;
  box.appendChild(t);
  requestAnimationFrame(() => { t.style.opacity = '1'; });
  setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 200); }, 1400);
}

// ---- API ----
async function apiGet(url) {
  const r = await fetch(url);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || data.error || ('HTTP ' + r.status));
  return data;
}
async function apiPost(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok && data.ok === undefined) data.ok = false;
  if (!r.ok && !data.error) data.error = data.detail || ('HTTP ' + r.status);
  return data;
}

// ---- 顶栏高亮 ----
function markNav() {
  const path = location.pathname;
  $all('.nav a').forEach(a => {
    const href = a.getAttribute('href');
    if (href === path || (href === '/' && path === '/')) a.classList.add('active');
  });
}

// 将仍使用旧 HTML 结构的业务页面挂载到统一应用外壳。
function mountStandardShell() {
  const oldTop = document.querySelector('body > .top');
  const wrap = document.querySelector('body > .wrap');
  if (!oldTop || !wrap || document.querySelector('.legacy-shell')) return;

  const path = location.pathname;
  const pages = [
    ['/', '⌂', '任务控制台'],
    ['/review', '✓', '结果审核'],
    ['/report', '▤', '分析报告'],
    ['/test', '⌁', '模型验证'],
    ['/files', '□', '数据管理'],
  ];
  const titles = { '/review': '结果审核', '/report': '分析报告', '/test': '模型验证', '/files': '数据管理' };
  document.title = `${titles[path] || '任务控制台'} · 低空遥感智能分析平台`;
  const nav = el('nav', { class: 'modern-nav', 'aria-label': '主导航' });
  pages.forEach(([href, glyph, label]) => {
    nav.appendChild(el('a', { href, class: href === path ? 'active' : '' }, [
      el('span', { class: 'modern-nav-glyph', text: glyph }), el('span', { text: label }),
    ]));
  });

  const aside = el('aside', { class: 'modern-sidebar' }, [
    el('div', { class: 'modern-logo' }, [
      el('span', { class: 'modern-logo-mark', text: '◈' }),
      el('span', {}, ['低空遥感视觉', el('small', { text: '智能分析平台' })]),
    ]),
    el('div', { class: 'modern-nav-label', text: '业务工作区' }), nav,
    el('div', { class: 'modern-nav-sep' }),
    el('div', { class: 'modern-project' }, [
      el('div', { class: 'modern-project-label', text: '当前项目' }),
      el('div', { class: 'modern-project-name', text: 'lowalt' }),
      el('div', { class: 'modern-project-state' }, [el('i'), '控制台在线']),
    ]),
  ]);
  const pagebar = el('header', { class: 'legacy-pagebar' }, [
    el('div', { class: 'legacy-pagebar-title', text: titles[path] || '项目空间' }),
    el('div', { class: 'legacy-pagebar-state' }, [el('i'), '控制台运行正常']),
  ]);
  const shell = el('div', { class: 'legacy-shell' }, [aside, el('div', { class: 'legacy-main' }, [pagebar, wrap])]);
  document.body.classList.add('legacy-modern');
  document.body.classList.add('page-' + path.replace(/^\//, '').replace(/[^a-z0-9_-]/gi, '-') || 'home');
  document.body.appendChild(shell);
}

// ---- 任务：启动 + 增量轮询 ----
// 在装有 #jobbar / #jobtext / #joblog 的页面上调用 attachJob()。
let _jobReceived = 0;     // 已收到的日志行数（用于增量）
let _jobPolling = null;
let _activeJobId = null;

async function startJob(action, params = {}) {
  if (action.startsWith('pipeline_full') && !confirm('全链路任务可能运行较长时间，确认开始？')) return;
  const trigger = document.activeElement instanceof HTMLButtonElement ? document.activeElement : null;
  if (trigger) trigger.disabled = true;
  try {
    const d = await apiPost('/api/job/start', { action, params });
    if (!d.ok) { toast(d.error || '启动失败', 'err'); return; }
    trackJob(d.job_id);
  } catch (error) {
    toast('启动失败：' + error, 'err');
  } finally {
    if (trigger) trigger.disabled = false;
  }
}

async function cancelActiveJob() {
  if (!_activeJobId) { toast('当前没有可取消的任务', 'info'); return; }
  const d = await apiPost('/api/job/cancel/' + encodeURIComponent(_activeJobId), {});
  if (!d.ok) { toast(d.error || '取消失败', 'err'); return; }
  toast(d.already_finished ? '任务已经结束' : '任务已取消', 'info');
}

function _setBar(progress, status) {
  const bar = $('#jobbar'), txt = $('#jobtext');
  if (bar) {
    bar.style.width = (progress || 0) + '%';
    bar.classList.toggle('done', status === 'done');
    bar.classList.toggle('error', status === 'error');
  }
  if (txt) {
    const map = { pending: '等待中', running: '运行中', done: '已完成', error: '出错', cancelled: '已取消' };
    txt.textContent = (map[status] || status || '') + '  ' + (progress || 0) + '%';
  }
}

function _appendLog(lines) {
  const box = $('#joblog');
  if (!box) return;
  const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 30;
  for (const line of lines) {
    const isErr = line.includes('错误:') || line.toLowerCase().includes('error') || line.includes('失败');
    const div = document.createElement('div');
    div.className = isErr ? 'err' : '';
    div.textContent = line;
    box.appendChild(div);
  }
  if (atBottom) box.scrollTop = box.scrollHeight;
}

function trackJob(jobId) {
  _activeJobId = jobId;
  _jobReceived = 0;
  const box = $('#joblog');
  if (box) clear(box);
  if (_jobPolling) clearTimeout(_jobPolling);

  async function poll() {
    let d;
    try {
      d = await apiGet(`/api/job/status/${jobId}?since=${_jobReceived}`);
    } catch (e) {
      _jobPolling = setTimeout(poll, 1200);
      return;
    }
    if (!d.ok) return;
    if (d.log && d.log.length) { _appendLog(d.log); _jobReceived = d.log_total; }
    _setBar(d.progress, d.status);
    if (d.status === 'running' || d.status === 'pending') {
      _jobPolling = setTimeout(poll, 700);
    } else if (typeof onJobDone === 'function') {
      _activeJobId = null;
      onJobDone(d);   // 页面可定义钩子，比如刷新概览
    } else {
      _activeJobId = null;
    }
  }
  poll();
}

// 页面加载时若已有任务，自动接上最近一个
async function resumeLatestJob() {
  try {
    const d = await apiGet('/api/job/latest');
    if (d.job_id) trackJob(d.job_id);
  } catch (e) {}
}

document.addEventListener('DOMContentLoaded', () => {
  markNav();
  mountStandardShell();
});

// ---- 全屏放大 lightbox ----
// 用法：openLightbox([{src,label}, ...], startIndex)
// 支持滚轮缩放、拖拽平移、+/- 按钮、双图切换、Esc 关闭。
// 现在内容支持两种：单图(items.src) 或 分层(items.layers) — 后者用真实图层叠加，无重编码、无损放大。
const _LB = { items: [], idx: 0, scale: 1, tx: 0, ty: 0, dragging: false, sx: 0, sy: 0 };

function _lbEnsureDom() {
  if ($('#lbMask')) return;
  // content 是被变换的整个内容节点（图或图层组），改 transform 即可整体缩放/平移
  const content = el('div', { class: 'lb-content', id: 'lbContent' });
  const stage = el('div', { class: 'lb-stage', id: 'lbStage' }, [content]);
  const bar = el('div', { class: 'lb-bar' }, [
    el('button', { id: 'lbPrev', onclick: () => _lbSwitch(-1), text: '‹ 上一张' }),
    el('span', { class: 'lb-label', id: 'lbLabel', text: '' }),
    el('button', { onclick: () => _lbZoom(1 / 1.3), text: '−' }),
    el('button', { onclick: () => _lbZoom(1.3), text: '+' }),
    el('button', { onclick: _lbReset, text: '复位' }),
    el('button', { id: 'lbNext', onclick: () => _lbSwitch(1), text: '下一张 ›' }),
  ]);
  const close = el('button', { class: 'lb-close', onclick: closeLightbox, text: '×' });
  const hint = el('div', { class: 'lb-hint', text: '滚轮缩放 · 拖拽平移 · 双击复位 · Esc 关闭' });
  const mask = el('div', { class: 'lb-mask', id: 'lbMask' }, [stage, bar, close, hint]);
  document.body.appendChild(mask);

  stage.addEventListener('wheel', (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    _lbZoom(factor, e.clientX, e.clientY);
  }, { passive: false });
  stage.addEventListener('mousedown', (e) => {
    _LB.dragging = true; _LB.sx = e.clientX - _LB.tx; _LB.sy = e.clientY - _LB.ty;
    stage.classList.add('drag');
  });
  window.addEventListener('mousemove', (e) => {
    if (!_LB.dragging) return;
    _LB.tx = e.clientX - _LB.sx; _LB.ty = e.clientY - _LB.sy; _lbApply();
  });
  window.addEventListener('mouseup', () => { _LB.dragging = false; stage.classList.remove('drag'); });
  stage.addEventListener('dblclick', _lbReset);
  mask.addEventListener('mousedown', (e) => { if (e.target === mask) closeLightbox(); });
}

function openLightbox(items, startIndex = 0) {
  _lbEnsureDom();
  _LB.items = items; _LB.idx = startIndex;
  _lbLoad();
  $('#lbMask').classList.add('open');
}

// 分层放大：layers = { baseSrc, maskSrc?, geom?, show?, label }
// 不做任何 toDataURL/JPEG 重编码——base/mask 都是浏览器原生 img(PNG/JPEG 原文件),
// 框/轮廓用 canvas 矢量画，放大到任意倍数都清晰。
function openLayeredLightbox(layers, label) {
  _lbEnsureDom();
  _LB.items = [{ layers, label }];
  _LB.idx = 0;
  _lbLoad();
  $('#lbMask').classList.add('open');
}

function closeLightbox() {
  const m = $('#lbMask'); if (m) m.classList.remove('open');
}

function _lbLoad() {
  const it = _LB.items[_LB.idx]; if (!it) return;
  const content = $('#lbContent'); if (!content) return;
  clear(content);
  $('#lbLabel').textContent = it.label || '';
  const multi = _LB.items.length > 1;
  $('#lbPrev').style.display = multi ? '' : 'none';
  $('#lbNext').style.display = multi ? '' : 'none';

  if (it.layers) {
    // 分层渲染：base img + mask img + canvas(矢量)，全部按 natural 尺寸放置
    const base = el('img', { class: 'lb-layer-base', src: it.layers.baseSrc });
    content.appendChild(base);
    if (it.layers.show && it.layers.show.mask && it.layers.maskSrc) {
      content.appendChild(el('img', { class: 'lb-layer-mask', src: it.layers.maskSrc }));
    }
    const cv = el('canvas', { class: 'lb-layer-canvas' });
    content.appendChild(cv);
    base.addEventListener('load', () => {
      // 内容节点尺寸 = 原图自然尺寸（transform 控制缩放，所以不需要 width:100%）
      const W = base.naturalWidth, H = base.naturalHeight;
      content.style.width = W + 'px'; content.style.height = H + 'px';
      cv.width = W; cv.height = H;
      _lbDrawBoxes(cv, it.layers);
      _lbFitInitial();
    });
  } else {
    // 单图
    const img = el('img', { class: 'lb-img', src: it.src });
    content.appendChild(img);
    img.addEventListener('load', () => {
      content.style.width = img.naturalWidth + 'px';
      content.style.height = img.naturalHeight + 'px';
      _lbFitInitial();
    });
  }
}

function _lbDrawBoxes(cv, L) {
  const g = L.geom; if (!g) return;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  const show = L.show || {};
  if (show.aabb && g.bbox && g.bbox.length === 4) {
    const [x1, y1, x2, y2] = g.bbox;
    ctx.strokeStyle = 'rgba(255,70,70,0.95)'; ctx.lineWidth = 4;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
  if (show.obb && g.obb) {
    ctx.strokeStyle = 'rgba(60,220,60,0.95)'; ctx.lineWidth = 4;
    ctx.beginPath(); g.obb.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    ctx.closePath(); ctx.stroke();
  }
  if (show.contour && g.contour) {
    ctx.strokeStyle = 'rgba(0,200,255,0.95)'; ctx.lineWidth = 3;
    ctx.beginPath(); g.contour.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    ctx.closePath(); ctx.stroke();
  }
}

// 打开时按 stage 视口缩放到合适大小、居中
function _lbFitInitial() {
  const stage = $('#lbStage'); const content = $('#lbContent');
  if (!stage || !content) return;
  const sw = stage.clientWidth, sh = stage.clientHeight;
  const cw = parseFloat(content.style.width), ch = parseFloat(content.style.height);
  if (!cw || !ch) return;
  // 初始让图能整张装下：取最小缩放比例（但不超过 1，保持像素级清晰，必要时再放大）
  const fit = Math.min(sw / cw, sh / ch, 1);
  _LB.scale = fit; _LB.tx = 0; _LB.ty = 0;
  _lbApply();
}

function _lbSwitch(d) {
  if (_LB.items.length < 2) return;
  _LB.idx = (_LB.idx + d + _LB.items.length) % _LB.items.length;
  _lbLoad();
}

function _lbZoom(factor, cx, cy) {
  const stage = $('#lbStage'); if (!stage) return;
  const rect = stage.getBoundingClientRect();
  const ox = (cx === undefined ? rect.left + rect.width / 2 : cx) - rect.left - rect.width / 2;
  const oy = (cy === undefined ? rect.top + rect.height / 2 : cy) - rect.top - rect.height / 2;
  const old = _LB.scale;
  _LB.scale = Math.min(40, Math.max(0.05, _LB.scale * factor));
  const r = _LB.scale / old;
  _LB.tx = ox - (ox - _LB.tx) * r;
  _LB.ty = oy - (oy - _LB.ty) * r;
  _lbApply();
}

function _lbReset() { _lbFitInitial(); }

function _lbApply() {
  const content = $('#lbContent'); if (!content) return;
  content.style.transform = `translate(-50%,-50%) translate(${_LB.tx}px,${_LB.ty}px) scale(${_LB.scale})`;
}

document.addEventListener('keydown', (e) => {
  const m = $('#lbMask');
  if (!m || !m.classList.contains('open')) return;
  if (e.key === 'Escape') closeLightbox();
  else if (e.key === 'ArrowLeft') { e.stopPropagation(); _lbSwitch(-1); }
  else if (e.key === 'ArrowRight') { e.stopPropagation(); _lbSwitch(1); }
}, true);
