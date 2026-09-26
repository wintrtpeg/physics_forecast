/* pforecast 로컬 앱.
   프레임워크도 CDN 도 쓰지 않는다 — 사내 잠긴 PC 에서 외부 요청 하나 없이 떠야 한다.
   차트는 SVG 로 직접 그린다 (크로스헤어 + 툴팁 포함). */

'use strict';

// ── 유틸 ────────────────────────────────────────────────────────────────
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

function h(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    if (kid instanceof Node) { n.append(kid); continue; }
    if (typeof kid === 'object') {
      // {html:...} 를 자식 자리에 넘긴 실수를 조용히 삼키지 않는다
      throw new TypeError('h(): 객체는 자식이 될 수 없습니다. 속성 자리에 넘기세요: ' +
                          JSON.stringify(kid).slice(0, 80));
    }
    n.append(document.createTextNode(String(kid)));
  }
  return n;
}
const svgEl = (tag, attrs = {}) => {
  const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) n.setAttribute(k, v);
  return n;
};

// 파일에서 온 글자(컬럼 이름, 상태 문자열)가 섞이므로 먼저 이스케이프하고 **굵게**만 살린다
function mdBold(t) {
  const esc = String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return esc.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
}

function fmt(v, d) {
  if (v === null || v === undefined || !isFinite(v)) return '—';
  const a = Math.abs(v);
  if (d !== undefined) return v.toFixed(d);
  if (a === 0) return '0';
  if (a >= 100000 || a < 0.001) return v.toExponential(2);
  if (a >= 1000) return v.toLocaleString('ko-KR', { maximumFractionDigits: 0 });
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

let toastTimer;
function toast(msg, kind) {
  const t = $('#toast');
  t.textContent = msg;
  t.className = 'toast show' + (kind ? ' ' + kind : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = 'toast'), 3600);
}

async function api(path, body) {
  const res = await fetch(path, {
    method: body ? 'POST' : 'GET',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({ error: '응답을 읽지 못했습니다' }));
  if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function niceTicks(lo, hi, n) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / (n || 5);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(v);
  return out.length >= 2 ? out : [lo, hi];
}

const COLORS = () => {
  const cs = getComputedStyle(document.documentElement);
  return [cs.getPropertyValue('--s1').trim(), cs.getPropertyValue('--s2').trim(),
          cs.getPropertyValue('--s3').trim()];
};

// ── 차트 ────────────────────────────────────────────────────────────────
function lineChart(host, opt) {
  const draw = () => {
    host.innerHTML = '';
    const W = Math.max(host.clientWidth || 640, 320);
    const H = opt.height || 230;
    const m = { t: 12, r: 14, b: 26, l: 52 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const series = opt.series.filter(s => s.values.some(v => v !== null && isFinite(v)));
    if (!series.length) { host.append(h('div', { class: 'empty' }, '표시할 값이 없습니다')); return; }

    let lo = Infinity, hi = -Infinity;
    for (const s of series) for (const v of s.values)
      if (v !== null && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (opt.limit !== undefined && opt.limit !== null && isFinite(opt.limit)) {
      lo = Math.min(lo, opt.limit); hi = Math.max(hi, opt.limit);
    }
    if (opt.marks && opt.marks.length && isFinite(lo)) {
      // 뺀 점까지 축에 넣되, 999.9 같은 단선값이 축을 망치지 않게 범위의 2배까지만 늘린다
      const span = Math.max(hi - lo, 1e-9);
      for (const mk of opt.marks) if (mk.v !== null && isFinite(mk.v)) {
        lo = Math.min(lo, Math.max(mk.v, lo - span)); hi = Math.max(hi, Math.min(mk.v, hi + span));
      }
    }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (hi === lo) { hi = lo + Math.abs(lo || 1) * 0.1; }
    const pad = (hi - lo) * 0.09; lo -= pad; hi += pad;
    const n = opt.x.length;
    const X = i => m.l + (n <= 1 ? iw / 2 : (i / (n - 1)) * iw);
    const Y = v => m.t + ih - ((v - lo) / (hi - lo)) * ih;

    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
    const g = svgEl('g', { class: 'grid' });
    for (const v of niceTicks(lo, hi, 5)) {
      const y = Y(v);
      if (y < m.t - 1 || y > m.t + ih + 1) continue;
      g.append(svgEl('line', { x1: m.l, x2: W - m.r, y1: y, y2: y }));
      const tx = svgEl('text', { x: m.l - 7, y: y + 3.5, 'text-anchor': 'end' });
      tx.textContent = fmt(v); g.append(tx);
    }
    svg.append(svgEl('g', { class: 'axis' }).appendChild(g).parentNode);

    // x 축 라벨 (양끝 + 중앙)
    const ax = svgEl('g', { class: 'axis' });
    [[0, 'start'], [Math.floor((n - 1) / 2), 'middle'], [n - 1, 'end']].forEach(([i, anc]) => {
      if (i < 0 || i >= n) return;
      const tx = svgEl('text', { x: X(i), y: H - 8, 'text-anchor': anc });
      tx.textContent = opt.xfmt ? opt.xfmt(opt.x[i]) : String(opt.x[i]);
      ax.append(tx);
    });
    svg.append(ax);

    if (opt.limit !== undefined && opt.limit !== null && isFinite(opt.limit)) {
      svg.append(svgEl('line', { class: 'limit', x1: m.l, x2: W - m.r, y1: Y(opt.limit), y2: Y(opt.limit) }));
      const lt = svgEl('text', { x: W - m.r, y: Y(opt.limit) - 5, 'text-anchor': 'end' });
      lt.setAttribute('fill', getComputedStyle(document.documentElement).getPropertyValue('--bad').trim());
      lt.setAttribute('font-size', '10.5');
      lt.textContent = `${opt.limitLabel || '관리기준'} ${fmt(opt.limit)}`;
      svg.append(lt);
    }

    const cols = COLORS();
    series.forEach((s, si) => {
      const color = s.color || cols[si % cols.length];
      let d = '', pen = false;
      s.values.forEach((v, i) => {
        if (v === null || !isFinite(v)) { pen = false; return; }
        d += (pen ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(v).toFixed(1) + ' ';
        pen = true;
      });
      svg.append(svgEl('path', { class: 'mark', d, stroke: color, 'stroke-width': s.width || 2 }));
    });

    // 결측 처리한 점: 원래 값 그대로 빨간 점으로 (무엇을 뺐는지 보이게)
    if (opt.marks && opt.marks.length) {
      const mg = svgEl('g', { class: 'excluded' });
      for (const mk of opt.marks) {
        if (mk.v === null || !isFinite(mk.v)) continue;
        const y = Math.max(m.t, Math.min(m.t + ih, Y(mk.v)));
        mg.append(svgEl('circle', { cx: X(mk.i).toFixed(1), cy: y.toFixed(1), r: 2.6 }));
      }
      svg.append(mg);
    }

    const cross = svgEl('line', { class: 'crosshair', y1: m.t, y2: m.t + ih, opacity: 0 });
    svg.append(cross);
    const dots = series.map((s, si) => {
      const c = svgEl('circle', { r: 4, fill: s.color || cols[si % cols.length],
                                  stroke: 'var(--panel)', 'stroke-width': 2, opacity: 0 });
      svg.append(c); return c;
    });
    const hot = svgEl('rect', { class: 'hot', x: m.l, y: m.t, width: iw, height: ih });
    svg.append(hot);

    const tip = h('div', { class: 'tip' });
    host.append(svg, tip);
    if (series.length >= 2) {
      host.append(h('div', { class: 'legend' }, series.map((s, si) =>
        h('span', {}, h('i', { style: `background:${s.color || cols[si % cols.length]}` }), s.name))));
    }
    if (opt.caption) {
      host.append(h('div', { class: 'sub', style: 'margin:7px 2px 0' }, opt.caption));
    }

    hot.addEventListener('mousemove', ev => {
      const r = svg.getBoundingClientRect();
      const px = (ev.clientX - r.left) * (W / r.width);
      const i = Math.max(0, Math.min(n - 1, Math.round(((px - m.l) / iw) * (n - 1))));
      cross.setAttribute('x1', X(i)); cross.setAttribute('x2', X(i)); cross.setAttribute('opacity', 1);
      const rows = [];
      series.forEach((s, si) => {
        const v = s.values[i];
        if (v === null || !isFinite(v)) { dots[si].setAttribute('opacity', 0); return; }
        dots[si].setAttribute('cx', X(i)); dots[si].setAttribute('cy', Y(v));
        dots[si].setAttribute('opacity', 1);
        rows.push(h('div', { class: 'r' },
          h('span', { class: 'sw', style: `background:${s.color || cols[si % cols.length]}` }),
          `${s.name} ${fmt(v)}${opt.unit ? ' ' + opt.unit : ''}`));
      });
      tip.innerHTML = '';
      tip.append(h('div', { class: 'tt' }, opt.xfmt ? opt.xfmt(opt.x[i]) : String(opt.x[i])), ...rows);
      tip.style.opacity = 1;
      const tw = tip.offsetWidth, left = (X(i) / W) * r.width;
      tip.style.left = Math.max(4, Math.min(r.width - tw - 4, left + 12)) + 'px';
      tip.style.top = '8px';
    });
    hot.addEventListener('mouseleave', () => {
      tip.style.opacity = 0; cross.setAttribute('opacity', 0);
      dots.forEach(d => d.setAttribute('opacity', 0));
    });
  };
  draw();
  if (host._ro) host._ro.disconnect();
  host._ro = new ResizeObserver(() => draw());
  host._ro.observe(host);
}

function barChart(host, opt) {
  const draw = () => {
    host.innerHTML = '';
    const items = opt.items.filter(i => isFinite(i.value));
    if (!items.length) { host.append(h('div', { class: 'empty' }, '표시할 값이 없습니다')); return; }
    const W = Math.max(host.clientWidth || 600, 300);
    const rowH = 30, m = { t: 6, r: 14, b: 22, l: Math.min(230, W * 0.42) };
    const H = m.t + items.length * rowH + m.b;
    const iw = W - m.l - m.r;
    const raw = Math.max(...items.map(i => i.value), opt.threshold || 0) * 1.1 || 1;
    const tk = niceTicks(0, raw, 4);
    const max = Math.max(tk[tk.length - 1], raw);
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
    const cols = COLORS();
    const tip = h('div', { class: 'tip' });

    for (const v of [0, ...tk]) {
      if (v > max) continue;
      const x = m.l + (v / max) * iw;
      svg.append(svgEl('line', { x1: x, x2: x, y1: m.t, y2: H - m.b,
                                 stroke: 'var(--line)', 'stroke-width': 1 }));
      const t = svgEl('text', { x, y: H - 7, 'text-anchor': 'middle', fill: 'var(--ink-3)',
                                'font-size': 10.5 });
      t.textContent = opt.valueFmt ? opt.valueFmt(v) : fmt(v); svg.append(t);
    }
    items.forEach((it, i) => {
      const y = m.t + i * rowH, bh = 15;
      const w = Math.max(2, (it.value / max) * iw);
      const color = it.color || cols[0];
      const lab = svgEl('text', { x: m.l - 9, y: y + bh / 2 + 4.5, 'text-anchor': 'end',
                                  fill: 'var(--ink-2)', 'font-size': 11.5 });
      lab.textContent = it.label.length > 34 ? it.label.slice(0, 32) + '…' : it.label;
      svg.append(lab);
      svg.append(svgEl('rect', { x: m.l, y: y + 3, width: w, height: bh, rx: 3, fill: color }));
      const val = svgEl('text', { x: m.l + w + 7, y: y + bh / 2 + 4.5, fill: 'var(--ink-2)',
                                  'font-size': 11 });
      val.textContent = opt.valueFmt ? opt.valueFmt(it.value) : fmt(it.value);
      svg.append(val);
      const hot = svgEl('rect', { x: m.l, y, width: iw, height: rowH, fill: 'transparent' });
      hot.addEventListener('mousemove', ev => {
        const r = svg.getBoundingClientRect();
        tip.innerHTML = '';
        tip.append(h('div', { class: 'tt' }, it.label),
          h('div', { class: 'r' }, (opt.valueFmt ? opt.valueFmt(it.value) : fmt(it.value))),
          it.note ? h('div', { class: 'r' }, it.note) : null);
        tip.style.opacity = 1;
        tip.style.left = Math.min(r.width - tip.offsetWidth - 6,
                                  (ev.clientX - r.left) + 12) + 'px';
        tip.style.top = ((y + rowH) / H) * r.height + 'px';
      });
      hot.addEventListener('mouseleave', () => (tip.style.opacity = 0));
      svg.append(hot);
    });
    if (opt.threshold) {
      const x = m.l + (opt.threshold / max) * iw;
      svg.append(svgEl('line', { class: 'limit', x1: x, x2: x, y1: m.t, y2: H - m.b }));
    }
    host.append(svg, tip);
  };
  draw();
  if (host._ro) host._ro.disconnect();
  host._ro = new ResizeObserver(() => draw());
  host._ro.observe(host);
}

// ── 상태 ────────────────────────────────────────────────────────────────
const S = {
  csv: null, profile: null, units: {}, target: null, timeCol: null,
  controllable: [], analysis: null, jobId: null,
  modelPath: null, meta: null, drivers: {}, kpis: [], limits: {},
};

function go(screen) {
  $$('#nav button').forEach(b => b.setAttribute('aria-current', String(b.dataset.screen === screen)));
  $$('.screen').forEach(s => s.classList.toggle('active', s.id === 'screen-' + screen));
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
function enableTab(screen, on) { $(`#nav button[data-screen="${screen}"]`).disabled = !on; }

// ── 1. 데이터 ───────────────────────────────────────────────────────────
async function loadWorkspace() {
  const ws = await api('/api/workspace');
  const list = $('#datasets'); list.innerHTML = '';
  if (!ws.datasets.length) list.append(h('div', { class: 'empty' }, 'CSV 파일이 없습니다'));
  ws.datasets.forEach(d => list.append(h('button', {
    class: 'item', 'data-path': d.path,
    onclick: () => pickDataset(d.path),
  }, h('span', { class: 't' }, d.name), h('span', { class: 'd' }, `${d.path} · ${d.size_kb} KB`))));

  const ml = $('#models'); ml.innerHTML = '';
  if (!ws.models.length) ml.append(h('div', { class: 'empty' }, '모델 파일이 없습니다'));
  ws.models.forEach(m => ml.append(h('button', {
    class: 'item', 'data-path': m.path, onclick: () => pickModel(m.path),
  }, h('span', { class: 't' }, m.name), h('span', { class: 'd' }, `${m.path} · ${m.kind}`))));
}

async function pickDataset(path) {
  $$('#datasets .item').forEach(b => b.setAttribute('aria-selected', String(b.dataset.path === path)));
  S.csv = path;
  $('#profile-pane').innerHTML = '';
  $('#profile-pane').append(h('div', { class: 'card' },
    h('div', { class: 'sub' }, '프로파일링 중…'), h('div', { class: 'progress' }, h('i'))));
  try {
    S.profile = await api('/api/profile', { csv: path });
    S.timeCol = S.profile.time_column;
    S.units = {}; S.target = null; S.controllable = [];
    S.profile.columns.forEach(c => (S.units[c.name] = c.unit));
    renderProfile();
  } catch (e) {
    $('#profile-pane').innerHTML = '';
    $('#profile-pane').append(h('div', { class: 'card' }, h('div', { class: 'note bad' }, e.message)));
  }
}

function renderProfile() {
  const p = S.profile, pane = $('#profile-pane');
  const known = p.columns.filter(c => S.units[c.name]).length;
  const review = p.columns.filter(c => c.needs_review).length;
  pane.innerHTML = '';
  pane.append(h('div', { class: 'tiles' },
    tile('행 수', p.rows.toLocaleString('ko-KR'), '', `샘플링 ${fmt(p.interval_s, 0)}초`),
    tile('기간', (p.t_start || '').slice(0, 10), '', `~ ${(p.t_end || '').slice(0, 10)}`),
    tile('준정상 구간', fmt(p.steady_fraction * 100, 0), '%', '전 컬럼이 동시에 안정된 비율'),
    tile('단위 지정', `${known}/${p.columns.length}`, '',
         review ? `${review}개 확인 필요` : '전부 확정', review ? 'warn' : 'good'),
  ));
  const ing = (p.ingest && p.ingest.lines) || [], qual = (p.quality && p.quality.lines) || [];
  if (ing.length > 1 || (p.quality && p.quality.excluded)) {
    const list = lines => h('ul', { class: 'log' }, lines.map(t => h('li', { html: mdBold(t) })));
    pane.append(h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h3', {}, '데이터 정리 내역'),
        h('span', { class: 'sub' }, '조용히 고치지 않습니다. 무엇을 고치고 뺐는지 전부 적습니다.')),
      h('div', { class: 'cols2' },
        h('div', {}, h('div', { class: 'k' }, '파일에서 고친 것'), list(ing)),
        h('div', {}, h('div', { class: 'k' }, '값에서 뺀 것 · 정보'), list(qual)))));
  }
  if (p.warnings.length) {
    pane.append(h('div', { class: 'note warn' },
      h('b', {}, '데이터 경고'), h('br'), p.warnings.slice(0, 5).join(' · ')));
  }
  pane.append(h('div', { class: 'card', id: 'colpreview', hidden: true }));

  const rows = p.columns.map(c => {
    const unitInput = h('input', {
      class: 'unit' + (c.needs_review ? ' review' : ''), type: 'text', value: S.units[c.name] || '',
      placeholder: '단위', title: c.reason,
      oninput: e => { S.units[c.name] = e.target.value.trim(); e.target.classList.remove('review'); },
    });
    const targetRadio = h('input', {
      type: 'radio', name: 'target', checked: S.target === c.name, disabled: !c.usable,
      onchange: () => { S.target = c.name; renderTargetBar(); $$('#coltable tr').forEach(tr =>
        tr.classList.toggle('picked', tr.dataset.name === c.name)); },
    });
    const ctrl = h('input', {
      type: 'checkbox', checked: S.controllable.includes(c.name), disabled: !c.usable,
      onchange: e => {
        S.controllable = e.target.checked ? [...S.controllable, c.name]
                                          : S.controllable.filter(x => x !== c.name);
        renderTargetBar();
      },
    });
    const badges = (c.issues || []).map(i => h('span', {
      class: 'badge ' + (i.n ? 'bad' : 'warn'), title: i.message },
      i.n ? `${i.label} ${i.n.toLocaleString('ko-KR')}` : i.label));
    const st = Object.entries(c.status_strings || {});
    if (st.length) badges.push(h('span', { class: 'badge', title: st.map(([k, v]) => `${k} ${v}`).join(', ') },
      `문자 ${st.reduce((a, [, v]) => a + v, 0).toLocaleString('ko-KR')}`));
    return h('tr', { 'data-name': c.name, class: S.target === c.name ? 'picked' : '' },
      h('td', { class: 'name clickable', title: '클릭하면 시계열을 봅니다', onclick: () => showColumn(c) },
        c.name, c.desc ? h('div', { class: 'desc' }, c.desc) : null),
      h('td', {}, unitInput),
      h('td', { class: 'num' }, fmt(c.min)),
      h('td', { class: 'num' }, fmt(c.max)),
      h('td', { class: 'num' }, c.missing_pct ? fmt(c.missing_pct, 1) + '%' : '—'),
      h('td', {}, c.status === '사용'
        ? h('span', { class: 'badge good' }, '사용')
        : h('span', { class: 'badge' }, c.status)),
      h('td', { class: 'issues' }, badges.length ? badges : '—'),
      h('td', {}, targetRadio), h('td', {}, ctrl));
  });

  pane.append(h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h3', {}, '컬럼'),
      h('span', { class: 'sub' }, '단위를 확인하고 타깃을 고르세요. 노란 칸은 추론 신뢰도가 낮습니다. 이름을 누르면 시계열이 보입니다.')),
    h('div', { class: 'tablewrap' },
      h('table', { id: 'coltable' },
        h('thead', {}, h('tr', {},
          h('th', {}, '컬럼'), h('th', {}, '단위'), h('th', {}, '최소'), h('th', {}, '최대'),
          h('th', {}, '결측'), h('th', {}, '상태'), h('th', {}, '이상'),
          h('th', {}, '타깃'), h('th', {}, '제어'))),
        h('tbody', {}, rows))),
    h('div', { id: 'targetbar' })));
  renderTargetBar();
}

async function showColumn(c) {
  const card = $('#colpreview');
  card.hidden = false; card.innerHTML = '';
  card.append(h('div', { class: 'card-head' }, h('h3', {}, c.name),
    h('span', { class: 'sub' }, [c.desc, S.units[c.name]].filter(Boolean).join(' · '))));
  const host = h('div', { class: 'chart' });
  card.append(host);
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  try {
    const r = await api('/api/preview', { csv: S.csv, columns: [c.name], limit: 900 });
    const marks = (r.excluded && r.excluded[c.name]) || [];
    lineChart(host, {
      series: [{ name: c.name, values: r.series[c.name] }], x: r.t, unit: S.units[c.name] || '',
      xfmt: t => String(t).slice(0, 16), height: 220, marks,
      caption: marks.length ? `빨간 점 ${marks.length.toLocaleString('ko-KR')}개는 결측 처리한 원래 값입니다 (선에서 빠짐).`
                            : '결측 처리한 점이 없습니다.',
    });
    (r.issues[c.name] || []).forEach(i => card.append(h('div', {
      class: 'note ' + (i.n_points ? 'bad' : 'warn') }, h('b', {}, i.label + ' '), i.message.replace(/\*\*/g, ''))));
  } catch (e) { host.append(h('div', { class: 'note bad' }, e.message)); }
}

function renderTargetBar() {
  const bar = $('#targetbar'); bar.innerHTML = '';
  const ok = !!S.target;
  bar.append(h('div', { class: 'actions' },
    h('button', { class: 'btn', disabled: !ok, onclick: runAnalysis },
      ok ? `‘${S.target}’ 분석 실행` : '타깃을 선택하세요'),
    h('span', { class: 'sub' }, S.controllable.length
      ? `제어 가능: ${S.controllable.join(', ')}` : '제어 가능한 변수를 체크하면 개선안도 계산합니다')));
}

const tile = (k, v, unit, note, status) => h('div', { class: 'tile' + (status ? ' ' + status : '') },
  h('div', { class: 'k' }, k),
  h('div', { class: 'v' }, v, unit ? h('small', {}, unit) : null),
  note ? h('div', { class: 'n' }, note) : null);

// ── 2. 분석 ─────────────────────────────────────────────────────────────
async function runAnalysis() {
  enableTab('analyze', true); go('analyze');
  const body = $('#analyze-body'); body.innerHTML = '';
  body.append(h('div', { class: 'card' },
    h('div', { class: 'sub' }, '분석 중입니다. 데이터 크기에 따라 10~60초 걸립니다…'),
    h('div', { class: 'progress' }, h('i')),
    h('div', { class: 'sub', id: 'job-msg', style: 'margin-top:10px' }, '시작하는 중')));
  try {
    const { job } = await api('/api/analyze', {
      csv: S.csv, target: S.target, target_unit: S.units[S.target] || '',
      timestamp: S.timeCol || 'timestamp', units: S.units,
      controllable: S.controllable, train_fraction: 0.6,
    });
    S.jobId = job.id;
    poll(job.id);
  } catch (e) {
    body.innerHTML = ''; body.append(h('div', { class: 'card' }, h('div', { class: 'note bad' }, e.message)));
  }
}

async function poll(id) {
  try {
    const j = await api('/api/job?id=' + id);
    const m = $('#job-msg'); if (m) m.textContent = j.message || j.status;
    if (j.status === 'running') return setTimeout(() => poll(id), 900);
    if (j.status === 'error') {
      $('#analyze-body').innerHTML = '';
      $('#analyze-body').append(h('div', { class: 'card' }, h('div', { class: 'note bad' }, j.error)));
      return;
    }
    S.analysis = j.result; renderAnalysis();
  } catch (e) { toast(e.message, 'bad'); }
}

function renderAnalysis() {
  const a = S.analysis, s = a.surrogate, body = $('#analyze-body');
  body.innerHTML = '';

  body.append(h('div', { class: 'tiles' },
    tile('설명력 R²', s ? fmt(s.r2_cv, 3) : '—', '', '시간블록 교차검증',
         s && s.r2_cv > 0.7 ? 'good' : s && s.r2_cv > 0.3 ? 'warn' : 'bad'),
    tile('예측오차 RMSE', s ? fmt(s.rmse_cv) : '—', s ? ' ' + (s.unit || '') : '',
         s ? `평균예측 대비 ${fmt(s.skill * 100, 0)}% 개선` : ''),
    tile('자동 발견 물리 관계', String(a.balances.filter(b => b.confidence === '확실').length), '개',
         '보존식·이중화 계측 (확실)',
         a.balances.some(b => b.confidence === '확실') ? 'good' : ''),
    tile('검증구간 외삽', fmt(a.holdout_outside * 100, 1), '%', '학습 범위 밖 = 근거 없음',
         a.holdout_outside > 0.2 ? 'bad' : a.holdout_outside > 0.05 ? 'warn' : 'good'),
  ));

  body.append(h('h2', {}, '먼저 읽을 것'));
  a.headline.forEach(t => body.append(h('div', {
    class: 'note', html: mdBold(t),
  })));

  body.append(h('h2', {}, '분석 사다리'));
  body.append(h('div', { class: 'ladder' }, a.ladder.map(r => h('div', {
    class: 'rung ' + (r.status === '완료' ? 'done' : r.status === '부분' ? 'part' : 'block'),
  },
    h('div', { class: 'lvl' }, r.level),
    h('div', {},
      h('div', { class: 'nm' }, r.name),
      r.finding ? h('div', { class: 'fi' }, r.finding) : null,
      r.blocker ? h('div', { class: 'bl' }, '막힌 이유: ' + r.blocker) : null,
      r.how_to_unblock ? h('div', { class: 'bl' }, '풀려면: ' + r.how_to_unblock) : null),
    h('span', { class: 'badge ' + (r.status === '완료' ? 'good' : r.status === '부분' ? 'warn' : '') },
      r.status)))));

  if (a.balances.length) {
    body.append(h('h2', {}, '데이터에서 자동으로 찾은 물리'));
    a.balances.forEach(b => body.append(h('div', {
      class: 'note ' + (b.confidence === '확실' ? 'good' : 'warn'),
    },
      h('span', { class: 'badge ' + (b.confidence === '확실' ? 'good' : 'warn') }, b.confidence),
      ' ', h('span', { class: 'mono' }, b.formula),
      h('div', { class: 'sub', style: 'margin:5px 0 0' },
        `상대잔차 ${fmt(b.residual_pct, 2)}% · R² ${fmt(b.r2, 3)} · 차원 ${b.dimension}` +
        (b.is_conservation ? ' · 보존식 형태' : '')))));
  }
  if (a.pi_groups.length) {
    body.append(h('h3', {}, '무차원군 (Buckingham Π)'));
    body.append(h('div', { class: 'note' }, a.pi_groups.map(g =>
      h('div', { class: 'mono' }, (g.has_target ? '★ ' : '   ') + g.formula))));
  }

  if (s) {
    body.append(h('h2', {}, '영향인자 — 묶음 기준으로 먼저 보세요'));
    body.append(h('div', { class: 'note warn', html:
      '순열 중요도는 <b>서로 상관된 변수들 사이에서 기여를 임의로 나눠 갖습니다.</b> ' +
      '상관 0.9 이상인 변수를 묶어 함께 섞은 결과가 아래입니다. 묶음 안의 순위는 ' +
      '데이터가 결정해 주지 못합니다.' }));
    const chart = h('div', { class: 'chart' });
    body.append(h('div', { class: 'card' }, chart));
    barChart(chart, {
      items: s.clusters.slice(0, 8).map(c => ({
        label: c.label, value: c.pct,
        note: c.is_group ? `${c.members.length}개 변수 · 내부상관 ${fmt(c.internal_corr, 4)}` : '단독',
        color: c.is_group ? COLORS()[1] : COLORS()[0],
      })),
      valueFmt: v => fmt(v, 1) + '%',
    });
    body.append(h('div', { class: 'legend' },
      h('span', {}, h('i', { style: `background:${COLORS()[1]}` }), '구분 불가 묶음'),
      h('span', {}, h('i', { style: `background:${COLORS()[0]}` }), '단독 (개별 해석 가능)')));

    if (s.pred) {
      body.append(h('h2', {}, '예측 추종 (시간블록 교차검증)'));
      const c2 = h('div', { class: 'chart' });
      body.append(h('div', { class: 'card' }, c2));
      lineChart(c2, {
        x: s.pred.t, unit: s.unit, height: 250,
        xfmt: t => String(t).slice(0, 16),
        series: [{ name: '실측', values: s.pred.meas }, { name: '예측', values: s.pred.y }],
      });
    }
  }

  if (a.improvement) {
    const im = a.improvement;
    body.append(h('h2', {}, '개선안'));
    body.append(h('div', { class: 'card' },
      h('div', { class: 'tiles' },
        tile('기준 예측', fmt(im.baseline), s ? ' ' + s.unit : ''),
        tile('개선 후', fmt(im.predicted), s ? ' ' + s.unit : '',
             `${fmt(im.change_pct, 1)}%`, im.change_pct < 0 ? 'good' : 'warn'),
        tile('채택 가능', im.feasible ? '예' : '아니오', '',
             im.feasible ? '학습 범위 안' : '포락선 밖', im.feasible ? 'good' : 'bad')),
      h('div', { class: 'sub', style: 'margin-top:11px' },
        Object.entries(im.settings).map(([k, v]) => `${k} → ${fmt(v)}`).join(' · '))));
  }

  body.append(h('div', { class: 'actions' },
    h('a', { class: 'btn ghost', href: '/' + a.report, target: '_blank' }, '전체 리포트 열기'),
    h('button', { class: 'btn ghost', onclick: () => go('model') }, '물리 모델로 →')));
}

// ── 3. 물리모델 ─────────────────────────────────────────────────────────
async function pickModel(path) {
  $$('#models .item').forEach(b => b.setAttribute('aria-selected', String(b.dataset.path === path)));
  const pane = $('#model-pane'); pane.innerHTML = '';
  pane.append(h('div', { class: 'card' }, h('div', { class: 'sub' }, '조립하고 구조를 검사하는 중…'),
    h('div', { class: 'progress' }, h('i'))));
  try {
    S.modelPath = path;
    S.meta = await api('/api/model/meta', { model: path });
    // 모델이 바뀌면 시나리오 상태를 전부 비운다. 남겨두면 이전 계통의 KPI 와
    // 관리기준이 엉뚱하게 따라붙는다.
    S.kpis = []; S.limits = {}; S.sweepKey = null; S.sweepOut = null;
    S.drivers = {}; S.meta.drivers.forEach(d => (S.drivers[d.key] = d.value));
    renderModelMeta();
    enableTab('scenario', true);
    renderScenario();
  } catch (e) {
    pane.innerHTML = '';
    pane.append(h('div', { class: 'card' }, h('div', { class: 'note bad' }, e.message)));
  }
}

function renderModelMeta() {
  const m = S.meta, pane = $('#model-pane'); pane.innerHTML = '';
  pane.append(h('div', { class: 'tiles' },
    tile('구조', m.ok ? '정상' : '문제 있음', '',
         `방정식 ${m.n_eqs} / 미지수 ${m.n_vars}`, m.ok ? 'good' : 'bad'),
    tile('컴포넌트', String(m.n_components), '개', `연결 ${m.n_connections}개`),
    tile('BLT 블록', String(m.blocks), '개', `최대 ${m.largest_block}`),
    tile('야코비안 non-zero', m.jacobian_nnz.toLocaleString('ko-KR'), '',
         `dense 라면 ${(m.largest_block ** 2).toLocaleString('ko-KR')}`),
  ));
  pane.append(h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h3', {}, '구조 진단'),
      h('span', { class: 'sub' }, '풀기 전에 방정식 수가 맞는지부터 본다')),
    h('pre', { class: 'mono' }, m.describe)));
  pane.append(h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h3', {}, '컴포넌트')),
    h('div', { class: 'tablewrap' }, h('table', {},
      h('thead', {}, h('tr', {}, h('th', {}, '이름'), h('th', {}, '타입'))),
      h('tbody', {}, m.components.map(c =>
        h('tr', {}, h('td', { class: 'name' }, c.name), h('td', {}, c.type))))))));
  pane.append(h('div', { class: 'actions' },
    h('button', { class: 'btn', onclick: () => go('scenario') }, '시나리오 실행 →'),
    h('button', { class: 'btn ghost', onclick: async () => {
      S.meta = await api('/api/model/reload', { model: S.modelPath });
      renderModelMeta(); toast('모델을 다시 불러왔습니다');
    } }, '다시 불러오기')));
}

// ── 4. 시나리오 ─────────────────────────────────────────────────────────
function defaultKpis(outputs, limits) {
  const picked = [];
  // 관리기준이 걸린 출력이 있으면 그게 곧 그 계통의 KPI 다.
  for (const name of Object.keys(limits || {})) {
    if (outputs.some(o => o.name === name) && !picked.includes(name)) picked.push(name);
    if (picked.length >= 4) break;
  }
  const pri = ['C_dry', 'ppm_dry', 'COP', 'kW_per_RT', 'W_total', 'power', 'Q_n',
               'eta', 'approach', 'dp_mmAq'];
  for (const p of pri) {
    const hit = outputs.find(o => o.name.endsWith('.' + p) && !picked.includes(o.name));
    if (hit) picked.push(hit.name);
    if (picked.length >= 4) break;
  }
  while (picked.length < 4 && picked.length < outputs.length) {
    const o = outputs[picked.length]; if (!picked.includes(o.name)) picked.push(o.name);
  }
  return picked;
}

function renderScenario() {
  const m = S.meta;
  if (!m) return;
  if (!S.kpis.length) S.kpis = defaultKpis(m.outputs, m.limits);
  if (!S.sweepKey && m.drivers.length) S.sweepKey = m.drivers[0].key;
  if (!S.sweepOut) S.sweepOut = S.kpis[0] || (m.outputs[0] || {}).name;
  Object.entries(m.limits || {}).forEach(([k, v]) => {
    if (S.limits[k] === undefined && v.max !== undefined) S.limits[k] = v.max;
  });

  const dv = $('#drivers'); dv.innerHTML = '';
  m.drivers.forEach(d => {
    const val = S.drivers[d.key];
    const out = h('span', { class: 'v' }, fmt(val), h('small', {}, d.unit === '1' ? '' : d.unit));
    const range = h('input', {
      type: 'range', min: d.lo, max: d.hi, step: (d.hi - d.lo) / 200, value: val,
      oninput: e => {
        S.drivers[d.key] = parseFloat(e.target.value);
        out.firstChild.textContent = fmt(S.drivers[d.key]);
        scheduleSolve();
      },
    });
    dv.append(h('div', { class: 'driver' },
      h('div', { class: 'lbl' },
        h('span', { class: 'n' }, d.label || d.key,
          d.count > 1 ? h('em', {}, ` ×${d.count}`) : null,
          d.mode === 'scale' ? h('em', {}, ' 설계 대비') : null),
        out),
      range,
      h('div', { class: 'ticks' }, h('span', {}, fmt(d.lo)), h('span', {}, fmt(d.hi)))));
  });
  solveNow();
}

function driverOverrides() {
  const o = {};
  (S.meta ? S.meta.drivers : []).forEach(d => {
    o[d.key] = { value: S.drivers[d.key], unit: d.unit, mode: d.mode || 'set' };
  });
  return o;
}

let solveTimer = null, solving = false, pending = false;
function scheduleSolve() {
  clearTimeout(solveTimer);
  solveTimer = setTimeout(solveNow, 90);
}
async function solveNow() {
  if (!S.modelPath) return;
  if (solving) { pending = true; return; }
  solving = true;
  const t0 = performance.now();
  try {
    const r = await api('/api/model/solve', { model: S.modelPath, overrides: driverOverrides() });
    $('#solve-stat').textContent = r.converged
      ? `${(performance.now() - t0).toFixed(0)} ms · 뉴턴 ${r.newton}회`
      : '수렴 실패';
    renderScenarioOut(r);
  } catch (e) { toast(e.message, 'bad'); }
  finally {
    solving = false;
    if (pending) { pending = false; scheduleSolve(); }
  }
}

function renderScenarioOut(r) {
  const pane = $('#scenario-pane'); pane.innerHTML = '';
  const m = S.meta;
  const unitOf = n => (m.outputs.find(o => o.name === n) || {}).unit || '';

  if (!r.converged) {
    pane.append(h('div', { class: 'card' }, h('div', { class: 'note bad' },
      h('b', {}, '수렴 실패'), h('pre', { class: 'mono' }, r.message))));
    return;
  }

  pane.append(h('div', { class: 'tiles' }, S.kpis.map(name => {
    const v = r.outputs[name], spec = (m.limits || {})[name] || {};
    const lim = S.limits[name] !== undefined ? S.limits[name] : spec.max;
    const lo = spec.min;
    const bad = (lim !== undefined && v !== null && v > lim) ||
                (lo !== undefined && v !== null && v < lo);
    const note = name.split('.')[0] +
      (lim !== undefined ? ` · 상한 ${fmt(lim)}` : lo !== undefined ? ` · 하한 ${fmt(lo)}` : '');
    return tile(name.split('.').slice(-1)[0], fmt(v), ' ' + unitOf(name), note,
      bad ? 'bad' : (lim !== undefined || lo !== undefined) ? 'good' : '');
  })));

  // 스윕
  const driverSel = h('select', {},
    m.drivers.map(d => h('option', { value: d.key, selected: d.key === S.sweepKey },
                         d.label || d.key)));
  const outSel = h('select', { onchange: e => {
    S.sweepOut = e.target.value;
    limInput.value = S.limits[S.sweepOut] ?? '';
  } }, m.outputs.map(o => h('option', { value: o.name, selected: o.name === S.sweepOut }, o.name)));
  const limInput = h('input', { type: 'number', placeholder: '관리기준 (선택)',
    value: S.limits[S.sweepOut] ?? '', style: 'max-width:150px' });
  const chart = h('div', { class: 'chart' });
  const runBtn = h('button', { class: 'btn sm', onclick: async () => {
    S.sweepKey = driverSel.value; S.sweepOut = outSel.value;
    const lim = parseFloat(limInput.value);
    if (isFinite(lim)) S.limits[S.sweepOut] = lim; else delete S.limits[S.sweepOut];
    const d = m.drivers.find(x => x.key === S.sweepKey);
    const vals = Array.from({ length: 13 }, (_, i) => d.lo + ((d.hi - d.lo) * i) / 12);
    chart.innerHTML = '<div class="empty">계산 중…</div>';
    try {
      const ov = driverOverrides();
      const res = await api('/api/model/sweep',
        { model: S.modelPath, key: S.sweepKey, values: vals, overrides: ov });
      const ys = res.rows.map(r2 => r2.outputs[S.sweepOut] ?? null);
      const fin = ys.filter(v => v !== null && isFinite(v));
      let caption = '';
      if (fin.length > 1) {
        const lo2 = Math.min(...fin), hi2 = Math.max(...fin);
        const rel = (Math.abs(hi2 - lo2) / Math.max(Math.abs((hi2 + lo2) / 2), 1e-12)) * 100;
        const lim = S.limits[S.sweepOut];
        caption = `${d.label || d.key} ${fmt(d.lo)} → ${fmt(d.hi)} ${d.unit === '1' ? '' : d.unit}` +
          ` 구간에서 ${S.sweepOut} 는 ${fmt(lo2)} ~ ${fmt(hi2)} ${unitOf(S.sweepOut)}` +
          ` (변화폭 ${fmt(rel, 1)}%)`;
        if (rel < 2) caption += ' — 사실상 영향이 없습니다. 세로축 눈금이 좁아 기울기가 커 보일 뿐입니다.';
        else if (lim !== undefined && hi2 > lim) {
          const idx = ys.findIndex(v => v !== null && v > lim);
          if (idx > 0) caption += ` · ${fmt(res.rows[idx].x)} 부근에서 관리기준 ${fmt(lim)} 초과`;
        }
      }
      lineChart(chart, {
        x: res.rows.map(r2 => fmt(r2.x)), unit: unitOf(S.sweepOut), height: 260,
        limit: S.limits[S.sweepOut], limitLabel: '관리기준', caption,
        series: [{ name: S.sweepOut, values: ys }],
      });
    } catch (e) { chart.innerHTML = ''; chart.append(h('div', { class: 'note bad' }, e.message)); }
  } }, '스윕 실행');

  pane.append(h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h3', {}, '민감도 스윕'),
      h('span', { class: 'sub' }, '손잡이 하나를 끝에서 끝까지 훑는다')),
    h('div', { class: 'actions', style: 'margin:0 0 12px' },
      h('span', { class: 'sub' }, '변수'), driverSel,
      h('span', { class: 'sub' }, '출력'), outSel, limInput, runBtn),
    chart));

  const rows = Object.entries(r.outputs).sort(([a], [b]) => a.localeCompare(b));
  pane.append(h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h3', {}, '전체 출력'),
      h('span', { class: 'sub' }, `${rows.length}개`)),
    h('div', { class: 'tablewrap' }, h('table', {},
      h('thead', {}, h('tr', {}, h('th', {}, '이름'), h('th', {}, '값'), h('th', {}, '단위'),
        h('th', {}, 'KPI'))),
      h('tbody', {}, rows.map(([k, v]) => h('tr', {},
        h('td', { class: 'name' }, k),
        h('td', { class: 'num' }, fmt(v)),
        h('td', {}, unitOf(k)),
        h('td', {}, h('input', {
          type: 'checkbox', checked: S.kpis.includes(k),
          onchange: e => {
            S.kpis = e.target.checked ? [...S.kpis, k].slice(-6) : S.kpis.filter(x => x !== k);
            solveNow();
          },
        })))))))));
}
function renderScenarioOutAfterSweep() { /* 차트는 그대로 두고 타일만 최신 상태 유지 */ }

// ── 부팅 ────────────────────────────────────────────────────────────────
$('#theme').addEventListener('click', () => {
  const cur = document.documentElement.getAttribute('data-theme');
  document.documentElement.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
  localStorage.setItem('pf-theme', document.documentElement.getAttribute('data-theme'));
  if (S.analysis) renderAnalysis();
});
(() => {
  const saved = localStorage.getItem('pf-theme');
  const dark = saved ? saved === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
})();
$$('#nav button').forEach(b => b.addEventListener('click', () => go(b.dataset.screen)));
$('#reset-drivers').addEventListener('click', () => {
  if (!S.meta) return;
  S.meta.drivers.forEach(d => (S.drivers[d.key] = d.value));
  renderScenario();
});
loadWorkspace().catch(e => toast(e.message, 'bad'));
