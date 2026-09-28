/* 모델 만들기 — 코드 없이 계통을 조립한다.
   화면이 다루는 것은 system.yaml 과 같은 모양의 dict (MB.model). 고칠 때마다 서버가 검사한다
   (불러오기 → 연결 → 구조 → 설계점 풀이 → 선언). 토폴로지는 사람이 정한다. */

'use strict';

const MB = {
  palette: null, model: null, path: null, dirty: false,
  check: null, checkSeq: 0, filter: '', open: {}, saved: null,
};

const KIND_COLOR = { gas: '--s1', thermal: '--s2', liquid: '--s3', power: '--ink-3' };
const KIND_LABEL = { gas: '가스', thermal: '열', liquid: '액체', power: '전력' };

function mbBlank(name) {
  return { name: name || '새_모델', description: '', components: {}, connections: [],
           drivers: [], observables: {}, calibrate: [], limits: {} };
}

async function mbEnter() {
  if (!MB.palette) {
    try { MB.palette = (await api('/api/builder/palette')).palette; }
    catch (e) { toast(e.message, 'bad'); MB.palette = []; }
  }
  if (!MB.model) MB.model = mbBlank();
  mbRender();
  mbCheck(true);
}

function mbMark() { MB.dirty = true; mbRenderBar(); mbCheck(); }

// ── 검사 (입력 멈추면 0.4초 뒤) ─────────────────────────────────────────
let mbTimer;
function mbCheck(now) {
  clearTimeout(mbTimer);
  mbTimer = setTimeout(async () => {
    const seq = ++MB.checkSeq;
    const pane = $('#mb-check'); if (pane) pane.classList.add('busy');
    try {
      const r = await api('/api/builder/check', { model: MB.model, path: MB.path });
      if (seq !== MB.checkSeq) return;
      MB.check = r;
    } catch (e) {
      MB.check = { ok: false, stage: 'error', errors: [e.message], warnings: [] };
    }
    mbRenderCheck(); mbRenderDiagram(); mbRenderComponents(); mbRenderConnections(); mbRenderDecl(); mbRenderYaml();
  }, now ? 0 : 400);
}

// ── 화면 ────────────────────────────────────────────────────────────────
function mbRender() {
  const body = $('#builder-body'); body.innerHTML = '';
  body.append(h('div', { class: 'mbbar', id: 'mb-bar' }));
  body.append(h('div', { class: 'mbgrid' },
    h('aside', { class: 'mbleft' }, h('div', { class: 'card', id: 'mb-palette' })),
    h('div', { class: 'mbmain' },
      h('div', { class: 'card', id: 'mb-diagram' }),
      h('div', { id: 'mb-components' }),
      h('div', { class: 'card', id: 'mb-connections' }),
      h('div', { class: 'card', id: 'mb-decl' }),
      h('details', { class: 'card fold', id: 'mb-yaml' })),
    h('aside', { class: 'mbright' }, h('div', { class: 'card', id: 'mb-check' }))));
  mbRenderBar(); mbRenderPalette(); mbRenderDiagram(); mbRenderComponents(); mbRenderConnections(); mbRenderDecl();
  mbRenderYaml(); mbRenderCheck();
}

async function mbRenderBar() {
  const bar = $('#mb-bar'); if (!bar) return;
  bar.innerHTML = '';
  let models = [];
  try { models = (await api('/api/workspace')).models.filter(m => m.kind === 'yaml'); } catch (_) { /* 목록 없이도 쓴다 */ }
  const open = h('select', { onchange: async e => {
    const v = e.target.value; e.target.value = '';
    if (!v) return;
    if (MB.dirty && !window.__mbConfirm) toast('저장하지 않은 변경은 버려집니다');
    await mbOpen(v);
  } }, h('option', { value: '' }, '열기 · 템플릿에서 시작…'),
    models.map(m => h('option', { value: m.path }, `${m.name}  (${m.path})`)));
  const name = h('input', { type: 'text', value: MB.model.name || '', placeholder: '모델 이름', class: 'mbname',
    oninput: e => { MB.model.name = e.target.value; MB.dirty = true; mbStatus(); } });
  bar.append(
    h('div', { class: 'mbtitle' }, name,
      h('span', { class: 'mono sub', id: 'mb-path' }, MB.path ? MB.path : '새 파일 (저장하면 models/ 에)'),
      h('span', { class: 'badge', id: 'mb-status' })),
    h('div', { class: 'grow' }),
    h('button', { class: 'btn ghost sm', onclick: () => { MB.model = mbBlank(); MB.path = null; MB.dirty = false; mbRender(); mbCheck(true); } }, '새 모델'),
    open,
    h('button', { class: 'btn sm', onclick: () => mbSave(false) }, MB.path ? '저장' : 'models/ 에 저장'),
    h('button', { class: 'btn ghost sm', onclick: () => mbSave(true) }, '다른 이름으로'));
  mbStatus();
}

function mbStatus() {
  const s = $('#mb-status'); if (!s) return;
  s.className = 'badge ' + (MB.dirty ? 'warn' : 'good');
  s.textContent = MB.dirty ? '저장 안 됨' : (MB.path ? '저장됨' : '빈 모델');
}

async function mbOpen(path) {
  try {
    const r = await api('/api/builder/load', { path });
    MB.model = Object.assign(mbBlank(), r.model);
    // 예제는 덮어쓰지 않는다 — 템플릿으로 열고 models/ 에 새로 저장
    MB.path = path.startsWith('examples/') ? null : r.path;
    if (path.startsWith('examples/')) {
      MB.model.name = (MB.model.name || 'model') + '_사본';
      toast('예제를 템플릿으로 열었습니다 — 저장하면 models/ 에 새 파일로 저장됩니다');
    }
    MB.dirty = path.startsWith('examples/'); MB.open = {};
    mbRender(); mbCheck(true);
  } catch (e) { toast(e.message, 'bad'); }
}

async function mbSave(asNew) {
  let path = MB.path;
  if (asNew || !path) {
    const base = (MB.model.name || 'model').replace(/[^0-9A-Za-z가-힣_\-]+/g, '_');
    path = `models/${base}.yaml`;
    if (asNew) {
      const inp = $('#mb-saveas');
      if (!inp) {
        const row = h('div', { class: 'mbsaveas' }, h('span', { class: 'sub', style: 'margin:0' }, '저장 경로'),
          h('input', { type: 'text', id: 'mb-saveas', value: path, class: 'mono' }),
          h('button', { class: 'btn sm', onclick: () => { const v = $('#mb-saveas').value.trim(); row.remove(); mbSaveTo(v); } }, '저장'),
          h('button', { class: 'btn ghost sm', onclick: () => row.remove() }, '취소'));
        $('#mb-bar').after(row);
        return;
      }
    }
  }
  await mbSaveTo(path);
}

async function mbSaveTo(path) {
  if (MB.check && !MB.check.ok) toast('검사를 통과하지 못한 모델입니다 — 그래도 저장합니다 (나중에 고칠 수 있게)');
  try {
    const r = await api('/api/builder/save', { model: MB.model, path });
    MB.path = r.path; MB.dirty = false; MB.saved = r.path;
    if (typeof EZ !== 'undefined') EZ.modelsList = null;          // 간편 예측의 모델 목록 새로 읽기
    toast(`저장했습니다: ${r.path}`);
    mbRenderBar(); mbRenderCheck();
  } catch (e) { toast(e.message, 'bad'); }
}

// ── 팔레트 ─────────────────────────────────────────────────────────────
function mbRenderPalette() {
  const card = $('#mb-palette'); card.innerHTML = '';
  const list = h('div', { class: 'pal' });
  const draw = () => {
    list.innerHTML = '';
    const q = MB.filter.trim().toLowerCase();
    const groups = {};
    (MB.palette || []).filter(p => !q || (p.title + p.description + (p.ref || p.type)).toLowerCase().includes(q))
      .forEach(p => (groups[p.category] = groups[p.category] || []).push(p));
    Object.entries(groups).forEach(([cat, items]) => {
      list.append(h('div', { class: 'k' }, cat));
      items.forEach(p => list.append(h('button', { class: 'palitem', title: p.description, onclick: () => mbAdd(p) },
        h('b', {}, p.title),
        (() => { const d = String(p.description || '').split('.')[0].replace(p.title, '').replace(/^[\s:·—-]+/, '');
                 return d ? h('span', {}, d) : null; })(),
        h('span', { class: 'ports' }, p.ports.map(pt => h('i', { class: 'pk', style: `background:var(${KIND_COLOR[pt.kind] || '--ink-3'})`,
          title: `${pt.name} (${KIND_LABEL[pt.kind] || pt.kind} ${pt.role === 'in' ? '입구' : '출구'})` }))))));
    });
  };
  card.append(h('div', { class: 'card-head' }, h('h3', {}, '컴포넌트'), h('span', { class: 'sub' }, '눌러서 추가')),
    h('input', { type: 'text', placeholder: '찾기', value: MB.filter, oninput: e => { MB.filter = e.target.value; draw(); } }),
    list,
    h('div', { class: 'sub', style: 'margin:10px 0 0' }, '작업 폴더의 components/*.yaml 도 여기에 뜹니다. 새 장비는 "사용자 정의"로 식을 직접 적거나 YAML 파일로 만드세요.'));
  draw();
}

function mbUniqueName(base) {
  const b = base.toUpperCase().replace(/[^0-9A-Z_]+/g, '_').slice(0, 10) || 'C';
  let i = 1, n = b;
  while (MB.model.components[n]) n = `${b}_${++i}`;
  return n;
}

function mbAdd(p) {
  const base = p.ref ? p.ref.split(':').pop().split('/').pop().replace('.yaml', '') : (p.blank ? 'CUSTOM' : p.type);
  const name = mbUniqueName(base);
  let cfg;
  if (p.blank) cfg = JSON.parse(JSON.stringify(p.template));
  else if (p.ref) cfg = { type: 'equation', spec: p.ref };
  else cfg = Object.assign({ type: p.type }, p.options || {});
  MB.model.components[name] = cfg;
  MB.open = { [name]: true };
  toast(`${name} 추가 — 포트를 연결하세요`);
  mbMark(); mbRenderComponents();
  setTimeout(() => { const el = $(`#mbc-${CSS.escape(name)}`); if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' }); }, 50);
}

// ── 흐름도 ─────────────────────────────────────────────────────────────
function mbPorts() {
  // 검사 결과가 있으면 실제 포트, 없으면 팔레트 기준
  const out = {};
  ((MB.check && MB.check.components) || []).forEach(c => (out[c.name] = c.ports));
  Object.entries(MB.model.components).forEach(([n, cfg]) => {
    if (out[n]) return;
    const p = (MB.palette || []).find(x => (cfg.spec ? x.ref === cfg.spec : x.type === cfg.type && !x.ref && !x.blank));
    out[n] = cfg.ports ? Object.entries(cfg.ports).map(([k, v]) => ({ name: k, kind: v.kind || 'gas', role: v.role || 'in' }))
                       : (p ? p.ports : []);
  });
  return out;
}

function mbRenderDiagram() {
  const card = $('#mb-diagram'); if (!card) return;
  card.innerHTML = '';
  const names = Object.keys(MB.model.components);
  card.append(h('div', { class: 'card-head' }, h('h3', {}, '흐름도'),
    h('span', { class: 'sub' }, '출구 → 입구 방향. 선 색은 포트 종류'),
    h('div', { class: 'grow' }),
    h('div', { class: 'legend', style: 'margin:0' }, Object.entries(KIND_LABEL).map(([k, v]) =>
      h('span', {}, h('i', { style: `background:var(${KIND_COLOR[k]})` }), v)))));
  if (!names.length) { card.append(h('div', { class: 'empty' }, '왼쪽 목록에서 컴포넌트를 추가하세요.')); return; }
  const ports = mbPorts();
  const role = {}; Object.entries(ports).forEach(([n, ps]) => ps.forEach(p => (role[`${n}.${p.name}`] = p)));
  const edges = MB.model.connections.map(([a, b]) => {
    const ra = role[a], rb = role[b];
    let [s, t] = [a, b];
    if (ra && ra.role === 'in' && rb && rb.role === 'out') [s, t] = [b, a];
    return { s: s.split('.')[0], t: t.split('.')[0], kind: (ra || rb || {}).kind || 'gas', label: `${s} → ${t}` };
  }).filter(e => names.includes(e.s) && names.includes(e.t));
  // 층: 들어오는 선이 없는 것부터 (순환은 방문 순서로 끊는다)
  const layer = {}; names.forEach(n => (layer[n] = 0));
  for (let it = 0; it < names.length; it++) {
    let changed = false;
    edges.forEach(e => { if (layer[e.t] < layer[e.s] + 1 && layer[e.s] + 1 < names.length) { layer[e.t] = layer[e.s] + 1; changed = true; } });
    if (!changed) break;
  }
  const cols = {}; names.forEach(n => (cols[layer[n]] = cols[layer[n]] || []).push(n));
  const nCol = Math.max(...Object.keys(cols).map(Number)) + 1;
  const maxRow = Math.max(...Object.values(cols).map(c => c.length));
  const W = Math.max(card.clientWidth - 32, 480), bw = Math.min(150, (W - 20) / nCol - 26), bh = 44;
  const gapX = (W - nCol * bw) / Math.max(nCol, 1), H = Math.max(maxRow * (bh + 22) + 20, 90);
  const pos = {};
  Object.entries(cols).forEach(([c, ns]) => ns.forEach((n, i) => {
    const colH = ns.length * (bh + 22) - 22;
    pos[n] = { x: gapX / 2 + Number(c) * (bw + gapX), y: (H - colH) / 2 + i * (bh + 22) };
  }));
  const P = getComputedStyle(document.documentElement);
  const col = k => P.getPropertyValue(KIND_COLOR[k] || '--ink-3').trim();
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: 'mbsvg' });
  const defs = svgEl('defs'); svg.append(defs);
  Object.keys(KIND_COLOR).forEach(k => {
    const m = svgEl('marker', { id: 'arr-' + k, viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto' });
    m.append(svgEl('path', { d: 'M0,0 L8,4 L0,8 z', fill: col(k) })); defs.append(m);
  });
  const dup = {};
  edges.forEach(e => {
    const a = pos[e.s], b = pos[e.t];
    const key = e.s + '>' + e.t; dup[key] = (dup[key] || 0) + 1;
    const off = (dup[key] - 1) * 8;
    const x1 = a.x + bw, y1 = a.y + bh / 2 + off, x2 = b.x, y2 = b.y + bh / 2 + off;
    const back = x2 <= x1;
    const d = back ? `M${a.x + bw / 2},${a.y + bh} C${a.x + bw / 2},${a.y + bh + 40} ${b.x + bw / 2},${b.y + bh + 40} ${b.x + bw / 2},${b.y + bh}`
                   : `M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2 - 2},${y2}`;
    const path = svgEl('path', { d, fill: 'none', stroke: col(e.kind), 'stroke-width': 1.8, 'marker-end': `url(#arr-${e.kind})`,
      'stroke-dasharray': e.kind === 'power' ? '4 3' : null });
    const t = svgEl('title'); t.textContent = e.label; path.append(t);
    svg.append(path);
  });
  const bad = new Set(((MB.check && MB.check.dangling) || []).map(p => p.split('.')[0]));
  names.forEach(n => {
    const p = pos[n], g = svgEl('g', { class: 'mbnode', transform: `translate(${p.x},${p.y})` });
    g.append(svgEl('rect', { width: bw, height: bh, rx: 7, class: bad.has(n) ? 'dangling' : '' }));
    const t1 = svgEl('text', { x: bw / 2, y: 18, 'text-anchor': 'middle', class: 'n1' }); t1.textContent = n;
    const title = ((MB.check && MB.check.components) || []).find(c => c.name === n);
    const t2 = svgEl('text', { x: bw / 2, y: 34, 'text-anchor': 'middle', class: 'n2' });
    t2.textContent = title ? (title.title.length > 14 ? title.title.slice(0, 13) + '…' : title.title) : '';
    g.append(t1, t2);
    g.addEventListener('click', () => { MB.open[n] = true; mbRenderComponents();
      const el = $(`#mbc-${CSS.escape(n)}`); if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' }); });
    svg.append(g);
  });
  card.append(h('div', { class: 'mbdiag' }, svg));
  if (bad.size) card.append(h('div', { class: 'sub', style: 'margin:6px 0 0' }, '빨간 테두리: 연결 안 된 포트가 있는 컴포넌트'));
}

// ── 컴포넌트 카드 ───────────────────────────────────────────────────────
function mbInfo(name) { return ((MB.check && MB.check.components) || []).find(c => c.name === name); }

function mbRenderComponents() {
  const host = $('#mb-components'); if (!host) return;
  host.innerHTML = '';
  Object.entries(MB.model.components).forEach(([name, cfg]) => host.append(mbComponentCard(name, cfg)));
}

function mbRename(oldName, newName) {
  newName = newName.trim().replace(/[^0-9A-Za-z_가-힣]+/g, '_');
  if (!newName || newName === oldName) return;
  if (MB.model.components[newName]) { toast('같은 이름이 있습니다', 'bad'); return; }
  const comps = {};
  Object.entries(MB.model.components).forEach(([k, v]) => (comps[k === oldName ? newName : k] = v));
  MB.model.components = comps;
  const ren = r => (r.split('.')[0] === oldName ? newName + r.slice(oldName.length) : r);
  MB.model.connections = MB.model.connections.map(([a, b]) => [ren(a), ren(b)]);
  MB.open[newName] = MB.open[oldName];
  mbMark(); mbRenderComponents();
}

function mbSetParam(name, cfg, pname, value) {
  const v = value === '' ? null : Number(value);
  if (v !== null && !isFinite(v)) return;
  if (cfg.type === 'equation' && cfg.spec) {
    cfg.values = cfg.values || {};
    if (v === null) delete cfg.values[pname]; else cfg.values[pname] = v;
  } else if (cfg.type === 'equation') {
    cfg.params[pname] = Object.assign({}, cfg.params[pname], { value: v });
  } else if (v === null) delete cfg[pname]; else cfg[pname] = v;
  mbMark();
}

function mbComponentCard(name, cfg) {
  const info = mbInfo(name);
  const custom = cfg.type === 'equation' && !cfg.spec;
  const open = !!MB.open[name];
  const conn = {}; MB.model.connections.forEach(([a, b]) => { conn[a] = b; conn[b] = a; });
  const d = h('details', { class: 'card mbcomp', id: 'mbc-' + name, open },
    h('summary', {},
      h('b', { class: 'mono' }, name), h('span', { class: 'sub', style: 'margin:0' }, info ? info.title : (cfg.spec || cfg.type)),
      info && info.ports.some(p => !conn[`${name}.${p.name}`]) ? h('span', { class: 'badge bad' }, '미연결 포트') : null,
      h('div', { class: 'grow' }),
      h('button', { class: 'icon-x', title: '삭제', onclick: e => { e.preventDefault(); mbDelete(name); } }, '✕')));
  d.addEventListener('toggle', () => (MB.open[name] = d.open));
  if (!open) return d;
  const body = h('div', { class: 'mbbody' });
  body.append(h('div', { class: 'row', style: 'gap:8px;flex-wrap:wrap' },
    h('label', { class: 'row' }, '이름', h('input', { type: 'text', value: name, class: 'mono', style: 'width:160px',
      onchange: e => mbRename(name, e.target.value) })),
    cfg.type === 'Mixer' ? h('label', { class: 'row' }, '입구 수', h('input', { type: 'number', min: 1, max: 20, value: cfg.n_inlets || 2, style: 'width:70px',
      onchange: e => { cfg.n_inlets = Math.max(1, Math.round(+e.target.value || 2)); mbMark(); } })) : null,
    info && info.description ? h('span', { class: 'sub', style: 'margin:0;flex:1 1 300px' }, info.description) : null));
  // 구성방정식 후보
  if (info && info.closures.length) {
    body.append(h('div', { class: 'k' }, '구성방정식 (후보에서 고르기)'));
    info.closures.forEach(sl => body.append(h('label', { class: 'row', style: 'margin:4px 0' }, sl.title,
      h('select', { onchange: e => { cfg.closures = cfg.closures || {}; if (e.target.value === sl.default) delete cfg.closures[sl.key];
        else cfg.closures[sl.key] = e.target.value; if (!Object.keys(cfg.closures).length) delete cfg.closures; mbMark(); } },
        sl.options.map(o => h('option', { value: o.id, selected: o.id === sl.current }, `${o.title} · ${o.role_label}`))))));
  }
  // 포트
  if (info) {
    body.append(h('div', { class: 'k' }, '포트'));
    body.append(h('div', { class: 'mbports' }, info.ports.map(p => {
      const ref = `${name}.${p.name}`;
      return h('span', { class: 'chip static' + (conn[ref] ? '' : ' warnchip') },
        h('i', { class: 'pk', style: `background:var(${KIND_COLOR[p.kind] || '--ink-3'})` }),
        `${p.name} (${KIND_LABEL[p.kind] || p.kind} ${p.role === 'in' ? '입구' : '출구'}) `,
        conn[ref] ? h('span', { class: 'mono' }, '↔ ' + conn[ref]) : '연결 안 됨');
    })));
  }
  // 파라미터
  const params = info ? info.params : [];
  if (params.length && !custom) {
    body.append(h('div', { class: 'k' }, '파라미터 (선언한 단위)'));
    body.append(h('div', { class: 'tablewrap' }, h('table', { class: 'mbparams' },
      h('thead', {}, h('tr', {}, h('th', {}, '이름'), h('th', {}, '값'), h('th', {}, '단위'), h('th', {}, '설명'))),
      h('tbody', {}, params.map(p => h('tr', {},
        h('td', { class: 'name' }, p.name, p.tunable ? h('span', { class: 'badge', title: '데이터로 보정할 수 있는 파라미터' }, '보정') : null),
        h('td', {}, h('input', { type: 'number', step: 'any', value: p.value, style: 'width:110px',
          onchange: e => mbSetParam(name, cfg, p.name, e.target.value) })),
        h('td', { class: 'mono' }, p.unit), h('td', { class: 'desc' }, p.desc)))))));
  }
  if (custom) body.append(mbCustomEditor(name, cfg));
  // 식 (사람이 읽는 형태)
  if (info && info.laws.length) body.append(h('details', { class: 'fold', style: 'margin-top:10px' },
    h('summary', {}, h('b', {}, `식 ${info.laws.length}개`)),
    h('div', { class: 'laws' }, info.laws.map(l => h('div', { class: 'law' }, h('span', { class: 'kind' }, l.kind),
      h('div', {}, h('div', { class: 'lt' }, l.title), h('div', { class: 'fm', html: typeof fmla === 'function' ? fmla(l.formula) : l.formula })))))));
  d.append(body);
  return d;
}

function mbDelete(name) {
  delete MB.model.components[name];
  MB.model.connections = MB.model.connections.filter(([a, b]) => a.split('.')[0] !== name && b.split('.')[0] !== name);
  mbMark(); mbRenderComponents();
}

// 사용자 정의 컴포넌트: 포트·파라미터·미지수·방정식·출력 표 편집
function mbCustomEditor(name, cfg) {
  const wrap = h('div', { class: 'mbcustom' });
  const table = (title, key, cols, blankRow, hint) => {
    const obj = cfg[key] = cfg[key] || {};
    const rows = Object.entries(obj);
    const tb = h('tbody');
    rows.forEach(([k, v]) => {
      const tr = h('tr', {}, h('td', {}, h('input', { type: 'text', value: k, class: 'mono', style: 'width:110px',
        onchange: e => { const nk = e.target.value.trim(); if (!nk || nk === k) return;
          const o = {}; Object.entries(obj).forEach(([a, b]) => (o[a === k ? nk : a] = b)); cfg[key] = o; mbMark(); mbRenderComponents(); } })));
      cols.forEach(c => {
        const val = typeof v === 'object' && v !== null ? v[c.f] : v;
        let inp;
        if (c.options) inp = h('select', { onchange: e => { obj[k][c.f] = e.target.value; mbMark(); } },
          c.options.map(o => h('option', { value: o, selected: o === val }, o)));
        else if (c.bool) inp = h('input', { type: 'checkbox', checked: !!val, onchange: e => { obj[k][c.f] = e.target.checked; mbMark(); } });
        else inp = h('input', { type: c.num ? 'number' : 'text', step: 'any', value: val === undefined || val === null ? '' : val,
          class: c.mono ? 'mono' : '', style: `width:${c.w || 90}px`,
          onchange: e => { const x = e.target.value; if (typeof v !== 'object' || v === null) obj[k] = x;
            else obj[k][c.f] = c.num ? (x === '' ? undefined : Number(x)) : x; mbMark(); } });
        tr.append(h('td', {}, inp));
      });
      tr.append(h('td', {}, h('button', { class: 'icon-x', onclick: () => { delete obj[k]; mbMark(); mbRenderComponents(); } }, '✕')));
      tb.append(tr);
    });
    return h('div', { class: 'mbtab' }, h('div', { class: 'k' }, title, hint ? h('span', { class: 'sub', style: 'margin:0 0 0 6px;font-weight:400' }, hint) : null),
      h('div', { class: 'tablewrap' }, h('table', { class: 'mbparams' },
        h('thead', {}, h('tr', {}, h('th', {}, '이름'), cols.map(c => h('th', {}, c.label)), h('th', {}, ''))), tb)),
      h('button', { class: 'btn ghost sm', style: 'margin-top:6px', onclick: () => {
        let i = 1, nk = blankRow.name; while (obj[nk]) nk = blankRow.name + (++i);
        obj[nk] = JSON.parse(JSON.stringify(blankRow.value)); mbMark(); mbRenderComponents(); } }, '+ 추가'));
  };
  wrap.append(h('label', { class: 'row', style: 'margin:6px 0' }, '설명',
    h('input', { type: 'text', value: cfg.description || '', style: 'flex:1', onchange: e => { cfg.description = e.target.value; mbMark(); } })));
  wrap.append(table('포트', 'ports', [
    { f: 'kind', label: '종류', options: ['gas', 'thermal', 'liquid', 'power'] },
    { f: 'role', label: '방향', options: ['in', 'out'] }], { name: 'p', value: { kind: 'thermal', role: 'in' } },
    'gas: 압력·유량·온도·조성 / thermal: 온도·열류 / liquid: 압력·유량·온도 / power: 전력'));
  wrap.append(table('파라미터', 'params', [
    { f: 'value', label: '값', num: true }, { f: 'unit', label: '단위', mono: true, w: 80 },
    { f: 'desc', label: '설명', w: 160 }, { f: 'tunable', label: '보정', bool: true },
    { f: 'lo', label: '하한', num: true, w: 70 }, { f: 'hi', label: '상한', num: true, w: 70 }],
    { name: 'k', value: { value: 1.0, unit: '1', desc: '' } }, '차원이 있으면 단위를 꼭 적습니다'));
  wrap.append(table('미지수', 'vars', [
    { f: 'unit', label: '단위', mono: true, w: 80 }, { f: 'start', label: '초기값', num: true },
    { f: 'lo', label: '하한', num: true, w: 70 }, { f: 'hi', label: '상한', num: true, w: 70 }, { f: 'desc', label: '설명', w: 140 }],
    { name: 'x', value: { unit: '1', start: 1.0 } }, '포트 변수 말고 내부에서 푸는 값'));
  wrap.append(table('방정식', 'equations', [{ f: null, label: '식 (좌변 = 우변)', mono: true, w: 360 }],
    { name: 'eq', value: '0 = 0' }, '개수 = 포트 수 × 3 (thermal·power 는 포트 변수 수) + 미지수 수. 상수에 단위: 273.15[K]'));
  wrap.append(table('출력', 'outputs', [{ f: 'expr', label: '식', mono: true, w: 220 }, { f: 'unit', label: '표시 단위', mono: true, w: 80 }],
    { name: 'y', value: { expr: '0', unit: '1' } }, '화면·계측값 연결에 쓸 값'));
  return wrap;
}

// ── 연결 ───────────────────────────────────────────────────────────────
function mbRenderConnections() {
  const card = $('#mb-connections'); if (!card) return;
  card.innerHTML = '';
  const ports = mbPorts();
  const all = []; Object.entries(ports).forEach(([n, ps]) => ps.forEach(p => all.push({ ref: `${n}.${p.name}`, ...p })));
  const used = new Set(MB.model.connections.flat());
  const free = all.filter(p => !used.has(p.ref));
  card.append(h('div', { class: 'card-head' }, h('h3', {}, `연결 ${MB.model.connections.length}개`),
    h('span', { class: 'sub' }, '출구 포트 → 입구 포트. 연결점에서는 압력(온도)이 같고 유량(열류) 합이 0')));
  card.append(h('div', { class: 'mbconns' }, MB.model.connections.map(([a, b], i) => h('div', { class: 'mbconn' },
    h('span', { class: 'mono' }, a), h('span', { class: 'arr' }, '→'), h('span', { class: 'mono' }, b),
    h('button', { class: 'icon-x', onclick: () => { MB.model.connections.splice(i, 1); mbMark(); mbRenderConnections(); mbRenderDiagram(); } }, '✕')))));
  const from = h('select', {}, h('option', { value: '' }, '출구 포트'),
    free.filter(p => p.role === 'out').map(p => h('option', { value: p.ref }, `${p.ref} (${KIND_LABEL[p.kind] || p.kind})`)));
  const to = h('select', {}, h('option', { value: '' }, '입구 포트'));
  from.addEventListener('change', () => {
    const src = all.find(p => p.ref === from.value);
    to.innerHTML = '';
    to.append(h('option', { value: '' }, '입구 포트'));
    free.filter(p => p.role === 'in' && src && p.kind === src.kind && p.ref.split('.')[0] !== src.ref.split('.')[0])
      .forEach(p => to.append(h('option', { value: p.ref }, p.ref)));
  });
  card.append(h('div', { class: 'mbconnadd' }, from, h('span', { class: 'arr' }, '→'), to,
    h('button', { class: 'btn sm', onclick: () => {
      if (!from.value || !to.value) { toast('두 포트를 고르세요', 'bad'); return; }
      MB.model.connections.push([from.value, to.value]); mbMark(); mbRenderConnections(); mbRenderDiagram(); } }, '연결 추가')));
  if (free.length) card.append(h('div', { class: 'sub', style: 'margin:8px 0 0' },
    '아직 연결 안 된 포트: ' + free.map(p => p.ref).join(', ')));
}

// ── 선언: 운전 손잡이·계측값·보정 대상·관리기준 ───────────────────────────
function mbRenderDecl() {
  const card = $('#mb-decl'); if (!card) return;
  card.innerHTML = '';
  const decl = (MB.check && MB.check.declarations) || {};
  const m = MB.model;
  m.drivers = m.drivers || []; m.observables = m.observables || {}; m.calibrate = m.calibrate || []; m.limits = m.limits || {};
  card.append(h('div', { class: 'card-head' }, h('h3', {}, '이 모델을 어떻게 쓰나'),
    h('span', { class: 'sub' }, '간편 예측·시나리오 화면이 이 선언을 씁니다. 구조 검사를 통과하면 고를 수 있는 목록이 채워집니다')));
  const keys = decl.driver_keys || [], outs = decl.outputs || [], tun = decl.tunable || [];
  // 운전 손잡이
  card.append(h('div', { class: 'k' }, '운전 손잡이 — 미리 아는 값 (생산계획·설정값·예보)'));
  card.append(h('div', { class: 'mbdecl' }, m.drivers.map((d, i) => h('div', { class: 'row', style: 'gap:6px;flex-wrap:wrap;margin:3px 0' },
    h('select', { onchange: e => { d.key = e.target.value; mbMark(); } }, [h('option', { value: d.key }, d.key || '파라미터')].concat(
      keys.filter(k => k !== d.key).map(k => h('option', { value: k }, k)))),
    h('input', { type: 'text', value: d.label || '', placeholder: '이름', style: 'width:120px', onchange: e => { d.label = e.target.value; mbMark(); } }),
    h('input', { type: 'text', value: d.unit || '', placeholder: '단위', class: 'mono', style: 'width:70px', onchange: e => { d.unit = e.target.value; mbMark(); } }),
    h('input', { type: 'number', value: d.lo ?? '', placeholder: '최소', style: 'width:80px', onchange: e => { d.lo = +e.target.value; mbMark(); } }),
    h('input', { type: 'number', value: d.hi ?? '', placeholder: '최대', style: 'width:80px', onchange: e => { d.hi = +e.target.value; mbMark(); } }),
    h('select', { onchange: e => { if (e.target.value === 'scale') d.mode = 'scale'; else delete d.mode; mbMark(); } },
      h('option', { value: 'set', selected: d.mode !== 'scale' }, '값으로'), h('option', { value: 'scale', selected: d.mode === 'scale' }, '배율로 (대수 등)')),
    h('button', { class: 'icon-x', onclick: () => { m.drivers.splice(i, 1); mbMark(); mbRenderDecl(); } }, '✕')))),
    h('button', { class: 'btn ghost sm', disabled: !keys.length, onclick: () => { m.drivers.push({ key: keys[0], label: keys[0] }); mbMark(); mbRenderDecl(); } }, '+ 손잡이'));
  // 계측값
  card.append(h('div', { class: 'k' }, '계측값 — 현장에서 재는 값 (보정·검증에 씀)'));
  card.append(h('div', { class: 'mbdecl' }, Object.entries(m.observables).map(([k, v]) => h('div', { class: 'row', style: 'gap:6px;margin:3px 0' },
    h('span', { class: 'mono', style: 'min-width:180px' }, k),
    h('input', { type: 'text', value: v || '', placeholder: '설명 (예: 굴뚝 NOx [mg/Sm3])', style: 'flex:1', onchange: e => { m.observables[k] = e.target.value; mbMark(); } }),
    h('button', { class: 'icon-x', onclick: () => { delete m.observables[k]; mbMark(); mbRenderDecl(); } }, '✕')))),
    h('select', { disabled: !outs.length, onchange: e => { if (e.target.value) { m.observables[e.target.value] = ''; mbMark(); mbRenderDecl(); } } },
      h('option', { value: '' }, '+ 계측값 추가 (모델 출력에서)'), outs.filter(o => !(o in m.observables)).map(o => h('option', { value: o }, o))));
  // 보정 대상
  card.append(h('div', { class: 'k' }, '기본 보정 파라미터 — 식별 가능한 것만 적게'));
  card.append(h('div', { class: 'mbtun' }, tun.length ? tun.map(p => h('label', { class: 'row chip static' },
    h('input', { type: 'checkbox', checked: m.calibrate.includes(p), onchange: e => {
      m.calibrate = e.target.checked ? [...m.calibrate, p] : m.calibrate.filter(x => x !== p); mbMark(); } }), h('span', { class: 'mono' }, p)))
    : h('span', { class: 'sub' }, '구조 검사를 통과하면 보정 가능한 파라미터가 나옵니다.')));
  // 관리기준
  card.append(h('div', { class: 'k' }, '관리기준'));
  card.append(h('div', { class: 'mbdecl' }, Object.entries(m.limits).map(([k, v]) => h('div', { class: 'row', style: 'gap:6px;margin:3px 0' },
    h('span', { class: 'mono', style: 'min-width:180px' }, k),
    h('select', { onchange: e => { const val = v.max ?? v.min; delete v.max; delete v.min; v[e.target.value] = val; mbMark(); } },
      h('option', { value: 'max', selected: v.max !== undefined }, '상한'), h('option', { value: 'min', selected: v.min !== undefined }, '하한')),
    h('input', { type: 'number', step: 'any', value: v.max ?? v.min ?? '', style: 'width:90px', onchange: e => { v[v.min !== undefined ? 'min' : 'max'] = +e.target.value; mbMark(); } }),
    h('input', { type: 'text', value: v.label || '', placeholder: '이름', style: 'width:140px', onchange: e => { v.label = e.target.value; mbMark(); } }),
    h('button', { class: 'icon-x', onclick: () => { delete m.limits[k]; mbMark(); mbRenderDecl(); } }, '✕')))),
    h('select', { disabled: !outs.length, onchange: e => { if (e.target.value) { m.limits[e.target.value] = { max: 0, label: '' }; mbMark(); mbRenderDecl(); } } },
      h('option', { value: '' }, '+ 관리기준 추가'), outs.filter(o => !(o in m.limits)).map(o => h('option', { value: o }, o))));
}

// ── YAML ───────────────────────────────────────────────────────────────
function mbRenderYaml() {
  const card = $('#mb-yaml'); if (!card) return;
  const wasOpen = card.open;
  card.innerHTML = '';
  const ta = h('textarea', { class: 'mono mbyaml', spellcheck: 'false' });
  ta.value = (MB.check && MB.check.yaml) || '';
  card.append(h('summary', {}, h('b', {}, 'YAML 보기 · 직접 편집'),
    h('span', { class: 'sub' }, ' 저장되는 파일 그대로입니다. 고친 뒤 "적용"을 누르면 화면에 반영합니다')),
    ta, h('div', { class: 'actions' }, h('button', { class: 'btn sm', onclick: async () => {
      try {
        const r = await api('/api/builder/check', { yaml: ta.value, path: MB.path });
        if (r.stage === 'yaml') { toast(r.errors[0], 'bad'); return; }
        MB.model = Object.assign(mbBlank(), r.model); MB.check = r; MB.dirty = true;
        mbRender(); toast('YAML 을 적용했습니다');
      } catch (e) { toast(e.message, 'bad'); }
    } }, 'YAML 적용')));
  card.open = wasOpen;
}

// ── 검사 결과 ───────────────────────────────────────────────────────────
function mbRenderCheck() {
  const card = $('#mb-check'); if (!card) return;
  card.classList.remove('busy');
  card.innerHTML = '';
  const r = MB.check;
  const STAGE = { load: '불러오기', connect: '연결', structure: '구조', solve: '설계점 풀이', yaml: 'YAML', error: '서버' };
  card.append(h('div', { class: 'card-head' }, h('h3', {}, '검사'), h('span', { class: 'sub' }, '고칠 때마다 자동으로')));
  if (!r) { card.append(h('div', { class: 'sub' }, '검사 중…')); return; }
  if (!Object.keys(MB.model.components).length) {
    // 빈 모델은 오류가 아니라 시작점이다
    card.append(h('div', { class: 'note' }, h('b', {}, '시작하기 '), '① 왼쪽에서 컴포넌트를 추가하고 ② 출구 → 입구로 연결한 뒤 ',
      '③ 경계(유입 조건·대기)를 채우면 방정식과 미지수 개수가 맞는지, 설계점에서 풀리는지 여기서 바로 알려줍니다. ',
      '비슷한 계통이 있으면 위 “열기 · 템플릿에서 시작”으로 복사해서 고치는 게 빠릅니다.'));
    return;
  }
  card.append(h('div', { class: 'mbstate ' + (r.ok ? 'ok' : 'no') },
    h('span', { class: 'ic' }, r.ok ? '✓' : '✕'),
    h('div', {}, h('b', {}, r.ok ? '풀립니다' : `${STAGE[r.stage] || r.stage} 단계에서 막힘`),
      h('div', { class: 'sub', style: 'margin:0' }, r.ok ? '구조 정상 · 설계점에서 수렴' : '아래 내용을 고치세요'))));
  const c = r.counts || {};
  if (c.equations !== undefined) card.append(h('div', { class: 'mbcounts' },
    tile('방정식', String(c.equations), '', `미지수 ${c.unknowns}`, c.equations === c.unknowns ? 'good' : 'bad'),
    tile('컴포넌트', String(c.components), '', `연결 ${c.connections}`),
    tile('BLT 블록', String(c.blocks), '', `최대 ${c.largest_block}`)));
  (r.errors || []).forEach(e => card.append(h('div', { class: 'note bad', style: 'white-space:pre-wrap' }, e)));
  (r.warnings || []).forEach(w => card.append(h('div', { class: 'note warn' }, w)));
  if (r.solve && r.solve.success) {
    card.append(h('details', { class: 'fold', style: 'margin-top:10px' }, h('summary', {}, h('b', {}, '설계점 결과')),
      Object.entries(r.solve.outputs).map(([comp, vals]) => h('div', { class: 'mbout' }, h('div', { class: 'k' }, comp),
        vals.map(v => h('div', { class: 'row', style: 'justify-content:space-between' }, h('span', { class: 'mono' }, v.name),
          h('span', { class: 'num' }, `${fmt(v.value)} ${v.unit === '1' ? '' : v.unit}`)))))));
  }
  if (r.ok && MB.path && !MB.dirty) card.append(h('div', { class: 'actions' },
    h('button', { class: 'btn sm', onclick: () => {
      // 모델이 바뀌면 컬럼 연결·보정 파라미터·구성방정식 선택은 전부 새로 한다
      ezUseModel(MB.path);
      if (EZ.profile && EZ.target && EZ.train[0]) { unlock('ez-model'); ezGo('ez-model'); }
      else if (EZ.profile) { toast('변수와 기간을 고르면 4단계에서 이 모델이 선택되어 있습니다'); ezGo('ez-vars'); }
      else { toast('데이터(CSV)를 올리면 4단계에서 이 모델이 선택되어 있습니다'); ezGo('ez-data'); }
    } }, '이 모델로 예측하기 →')));
}

// ── 부팅 ───────────────────────────────────────────────────────────────
$$('#nav button[data-screen="builder"]').forEach(b => b.addEventListener('click', mbEnter));
