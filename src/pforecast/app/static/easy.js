/* 간편 예측 — 업로드 → 변수 → 기간 → 모델·실행 → 결과.
   app.js 의 유틸($, h, api, fmt, tile, toast, go, lineChart, barChart)을 그대로 쓴다.
   누수는 화면에서 먼저 막고(예측 시작은 학습 끝 + 간격 뒤로만), 서버가 한 번 더 검사한다. */

'use strict';

const EZ = {
  csv: null, profile: null, presets: [], preset: null, spark: {},
  target: null, features: [], units: {}, filter: '',
  timeline: null, train: [null, null], test: [null, null], embargo: 1, dragMode: 'train',
  check: null, checkSeq: 0,
  mode: 'physics', model: null, catalog: null, fmap: {}, tmap: null, extra: {},
  params: [], band: false, limit: '',
  closures: {}, advice: null, adviceKey: null, adviceRun: null,
  job: null, result: null, hidden: {},
};

// ── 날짜 ────────────────────────────────────────────────────────────────
const DAY = 86400000;
const pad2 = n => String(n).padStart(2, '0');
const toMs = s => Date.parse(String(s).trim().replace(' ', 'T'));
const dayOf = ms => { const d = new Date(ms); return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`; };
const dayMs = s => toMs(s + 'T00:00:00');
const addDays = (s, n) => dayOf(dayMs(s) + n * DAY);
const firstDay = () => dayOf(toMs(EZ.profile.t_start));
const lastDay = () => dayOf(toMs(EZ.profile.t_end));
const minTestStart = () => (EZ.train[1] ? addDays(EZ.train[1], 1 + EZ.embargo) : null);
const shortT = t => String(t).slice(0, 16).replace('T', ' ');

const PAL = () => {
  const cs = getComputedStyle(document.documentElement);
  const v = k => cs.getPropertyValue(k).trim();
  return { y: v('--ink-2'), physics: v('--s1'), poly: v('--s2'), boost: v('--s3'),
           train: v('--s1'), test: v('--s2'), bad: v('--bad'), line: v('--line'), ink3: v('--ink-3') };
};
const colorOf = (kind, P) => ({ physics: P.physics, poly: P.poly, boost: P.boost })[kind] || P.y;

function unlock(upto) {
  const order = ['ez-data', 'ez-vars', 'ez-period', 'ez-model', 'ez-result'];
  order.forEach((s, i) => enableTab(s, i <= order.indexOf(upto)));
}

// ── 1. 데이터 ───────────────────────────────────────────────────────────
async function ezLoadDatasets() {
  const ws = await api('/api/workspace');
  const box = $('#ez-datasets'); box.innerHTML = '';
  if (!ws.datasets.length) { box.append(h('span', { class: 'sub' }, '아직 없습니다. 위에 CSV 를 올리세요.')); return; }
  // 올린 파일을 앞에, 이름이 같은 파일이 많으니 바로 위 폴더까지 보여준다
  const ds = [...ws.datasets].sort((x, y) => (y.path.startsWith('uploads/') - x.path.startsWith('uploads/')) || x.path.localeCompare(y.path));
  ds.forEach(d => box.append(h('button', {
    class: 'chip' + (EZ.csv === d.path ? ' on' : ''), title: d.path,
    onclick: () => ezSelectCsv(d.path),
  }, d.path.split('/').slice(-2).join('/'), h('small', {}, ` ${Math.round(d.size_kb).toLocaleString('ko-KR')} KB`))));
}

function ezSetupDrop() {
  const dz = $('#dropzone'), input = $('#file');
  const pick = f => { if (f) ezUpload(f); };
  input.addEventListener('change', () => pick(input.files[0]));
  dz.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  ['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('over'); }));
  dz.addEventListener('drop', e => pick(e.dataTransfer.files[0]));
}

function ezUpload(file) {
  if (!/\.(csv|txt)$/i.test(file.name)) { toast('CSV 파일만 올릴 수 있습니다', 'bad'); return; }
  const bar = $('#dropzone .dz-progress'), msg = bar.querySelector('span');
  bar.hidden = false; msg.textContent = `${file.name} 올리는 중…`;
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/upload?name=' + encodeURIComponent(file.name));
  xhr.upload.onprogress = e => { if (e.lengthComputable) msg.textContent = `${file.name} ${Math.round(e.loaded / e.total * 100)}%`; };
  xhr.onload = async () => {
    bar.hidden = true;
    let r = {};
    try { r = JSON.parse(xhr.responseText); } catch (_) { r = { error: '응답을 읽지 못했습니다' }; }
    if (xhr.status !== 200 || r.error) { toast(r.error || `업로드 실패 (${xhr.status})`, 'bad'); return; }
    toast(`올렸습니다: ${r.path}`);
    await ezLoadDatasets().catch(() => {});
    ezSelectCsv(r.path);
  };
  xhr.onerror = () => { bar.hidden = true; toast('업로드 실패 — 서버가 켜져 있는지 확인하세요', 'bad'); };
  xhr.send(file);
}

async function ezSelectCsv(path) {
  EZ.csv = path;
  $$('#ez-datasets .chip').forEach(c => c.classList.toggle('on', c.title === path));
  const box = $('#ez-data-summary'); box.innerHTML = '';
  box.append(h('div', { class: 'card' }, h('div', { class: 'sub' }, '파일을 읽고 정리하는 중… (10MB 에 몇 초)'),
    h('div', { class: 'progress' }, h('i'))));
  try {
    const [prof, pre] = await Promise.all([
      api('/api/profile', { csv: path }),
      api('/api/easy/presets', { csv: path }).catch(() => ({ presets: [] })),
    ]);
    Object.assign(EZ, { profile: prof, presets: pre.presets || [], preset: (pre.presets || [])[0] || null,
      target: null, features: [], units: {}, train: [null, null], test: [null, null], check: null,
      timeline: null, fmap: {}, tmap: null, extra: {}, params: [], model: null, catalog: null,
      result: null, spark: {}, limit: '', hidden: {}, filter: '', closures: {}, advice: null, adviceKey: null });
    prof.columns.forEach(c => (EZ.units[c.name] = c.unit || ''));
    // 같은 태그를 쓰는 설정이 있으면 타깃·입력·단위를 미리 채운다
    const p = EZ.preset;
    if (p) {
      if (prof.columns.some(x => x.name === p.target_column)) EZ.target = p.target_column;
      EZ.features = p.inputs.map(i => i.column).filter(c => prof.columns.some(x => x.name === c));
      p.inputs.concat(p.observations).forEach(m => { if (m.unit) EZ.units[m.column] = m.unit; });
    }
    unlock('ez-vars');
    ezRenderDataSummary();
    ezLoadSparks();
  } catch (e) {
    box.innerHTML = ''; box.append(h('div', { class: 'note bad' }, e.message));
  }
}

function ezRenderDataSummary() {
  const p = EZ.profile, box = $('#ez-data-summary'); box.innerHTML = '';
  const usable = p.columns.filter(c => c.usable).length;
  box.append(h('div', { class: 'tiles', style: 'margin-top:14px' },
    tile('행 수', p.rows.toLocaleString('ko-KR'), '', `간격 ${ezInterval(p.interval_s)}`),
    tile('기간', (p.t_start || '').slice(0, 10), '', `~ ${(p.t_end || '').slice(0, 10)}`),
    tile('컬럼', String(p.columns.length), '개', `쓸 수 있는 것 ${usable}개`),
    tile('정리한 값', (p.quality.excluded || 0).toLocaleString('ko-KR'), '점',
         '교정 창·고착·스파이크 → 결측', p.quality.excluded ? 'warn' : 'good')));
  if (EZ.preset) {
    const pr = EZ.preset;
    box.append(h('div', { class: 'note good', html: mdBold(
      `이 파일의 태그가 저장된 설정 **‘${pr.name}’** 과 ${Math.round(pr.coverage * 100)}% 일치합니다. ` +
      '예측할 값·입력·물리모델 연결을 미리 채워 둡니다 (다음 단계에서 바꿀 수 있습니다).') }));
  }
  const ing = p.ingest.lines || [], qual = p.quality.lines || [];
  box.append(h('details', { class: 'card fold' },
    h('summary', {}, h('b', {}, '데이터 정리 내역'), h('span', { class: 'sub' }, ` 파일에서 고친 것 ${ing.length}건 · 값에서 뺀 것 ${qual.length}건 — 조용히 고치지 않습니다`)),
    h('div', { class: 'cols2', style: 'margin-top:12px' },
      h('div', {}, h('div', { class: 'k' }, '파일에서 고친 것'), h('ul', { class: 'log' }, ing.map(t => h('li', { html: mdBold(t) })))),
      h('div', {}, h('div', { class: 'k' }, '값에서 뺀 것 · 정보'), h('ul', { class: 'log' }, qual.map(t => h('li', { html: mdBold(t) })))))));
  box.append(h('div', { class: 'actions' },
    h('button', { class: 'btn lg', onclick: () => ezGo('ez-vars') }, '다음: 변수 고르기 →')));
}
const ezInterval = s => (!s ? '—' : s >= 86400 ? `${fmt(s / 86400, 1)}일` : s >= 3600 ? `${fmt(s / 3600, 1)}시간` : s >= 60 ? `${fmt(s / 60, 0)}분` : `${fmt(s, 0)}초`);

async function ezLoadSparks() {
  const cols = EZ.profile.columns.filter(c => c.usable || c.status === '상수').map(c => c.name);
  try {
    const r = await api('/api/preview', { csv: EZ.csv, columns: cols, limit: 140 });
    EZ.spark = r.series;
    if ($('#screen-ez-vars').classList.contains('active')) ezRenderVars();
  } catch (_) { /* 미리보기는 없어도 된다 */ }
}

function sparkline(vals) {
  const W = 110, H = 26;
  const v = (vals || []).filter(x => x !== null && isFinite(x));
  if (v.length < 2) return h('span', { class: 'sub' }, '—');
  const lo = Math.min(...v), hi = Math.max(...v), n = vals.length;
  let d = '', pen = false;
  vals.forEach((x, i) => {
    if (x === null || !isFinite(x)) { pen = false; return; }
    const px = (i / (n - 1)) * (W - 2) + 1, py = H - 2 - ((x - lo) / ((hi - lo) || 1)) * (H - 4);
    d += (pen ? 'L' : 'M') + px.toFixed(1) + ' ' + py.toFixed(1) + ' '; pen = true;
  });
  const s = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: 'spark' });
  s.append(svgEl('path', { d, fill: 'none', stroke: 'currentColor', 'stroke-width': 1.3 }));
  return s;
}

// ── 2. 변수 ─────────────────────────────────────────────────────────────
function ezObservedCols() {
  // 저장된 설정에서 '관측(결과값)'으로 쓰인 컬럼 — 입력으로 고르면 누수 경고
  return new Set(EZ.preset ? EZ.preset.observations.map(o => o.column) : []);
}

function ezRenderVars() {
  const body = $('#ez-vars-body'); body.innerHTML = '';
  const p = EZ.profile, obsCols = ezObservedCols();
  const summary = h('div', { class: 'pickbar' });
  const renderSummary = () => {
    summary.innerHTML = '';
    summary.append(
      h('div', {}, h('span', { class: 'k' }, '예측할 값 (y)'),
        EZ.target ? h('b', {}, EZ.target) : h('span', { class: 'sub' }, '아래 표에서 ◉ 를 고르세요')),
      h('div', {}, h('span', { class: 'k' }, '입력 (x)'),
        EZ.features.length ? h('b', {}, `${EZ.features.length}개`) : h('span', { class: 'sub' }, '☑ 를 고르세요'),
        EZ.features.length ? h('span', { class: 'sub' }, ' ' + EZ.features.slice(0, 4).join(', ') + (EZ.features.length > 4 ? ' …' : '')) : null),
      h('button', { class: 'btn lg', disabled: !(EZ.target && EZ.features.length),
        onclick: () => { unlock('ez-period'); ezGo('ez-period'); } }, '다음: 기간 →'));
  };
  renderSummary();
  body.append(summary);

  const search = h('input', { type: 'text', placeholder: '컬럼 이름·설명으로 찾기', value: EZ.filter,
    oninput: e => { EZ.filter = e.target.value; draw(); } });
  const bulk = h('div', { class: 'actions', style: 'margin:0' },
    h('button', { class: 'btn ghost sm', onclick: () => { EZ.features = []; unlock('ez-vars'); draw(); renderSummary(); } }, '입력 모두 해제'));
  const tbody = h('tbody');
  const preview = h('div', { class: 'card', hidden: true });
  const draw = () => {
    tbody.innerHTML = '';
    const q = EZ.filter.trim().toLowerCase();
    p.columns.filter(c => !q || c.name.toLowerCase().includes(q) || (c.desc || '').toLowerCase().includes(q))
      .forEach(c => {
        const isY = EZ.target === c.name, isX = EZ.features.includes(c.name);
        const warnObs = isX && obsCols.has(c.name);
        const yRadio = h('input', { type: 'radio', name: 'ez-y', checked: isY, disabled: !c.usable,
          title: '예측할 값', onchange: () => {
            EZ.target = c.name; EZ.features = EZ.features.filter(x => x !== c.name);
            EZ.check = null; EZ.timeline = null; EZ.limit = ''; EZ.tmap = ezPresetTarget();
            unlock('ez-vars'); draw(); renderSummary();
          } });
        const xBox = h('input', { type: 'checkbox', checked: isX, disabled: isY || c.status.startsWith('중복'),
          title: '예측 시점에 미리 아는 값', onchange: e => {
            EZ.features = e.target.checked ? [...EZ.features, c.name] : EZ.features.filter(x => x !== c.name);
            EZ.check = null; unlock('ez-vars'); draw(); renderSummary();
          } });
        const unit = h('input', { class: 'unit', type: 'text', value: EZ.units[c.name] || '', placeholder: '단위',
          title: c.reason || '', oninput: e => (EZ.units[c.name] = e.target.value.trim()) });
        const badges = [];
        if (c.status !== '사용') badges.push(h('span', { class: 'badge' }, c.status));
        if (warnObs) badges.push(h('span', { class: 'badge bad', title: '저장된 설정에서 결과값(관측)으로 쓰인 컬럼입니다' }, '결과값 — 누수 주의'));
        (c.issues || []).slice(0, 2).forEach(i => badges.push(h('span', { class: 'badge warn', title: i.message }, i.label)));
        tbody.append(h('tr', { class: (isY ? 'is-y' : '') + (isX ? ' is-x' : '') },
          h('td', { class: 'pick' }, yRadio), h('td', { class: 'pick' }, xBox),
          h('td', { class: 'name clickable', onclick: () => ezPreview(c, preview) }, c.name,
            c.desc ? h('div', { class: 'desc' }, c.desc) : null),
          h('td', { class: 'sparkcell' }, sparkline(EZ.spark[c.name])),
          h('td', {}, unit),
          h('td', { class: 'num' }, `${fmt(c.min)} ~ ${fmt(c.max)}`),
          h('td', { class: 'num' }, c.missing_pct ? fmt(c.missing_pct, 1) + '%' : '—'),
          h('td', { class: 'issues' }, badges.length ? badges : '')));
      });
  };
  draw();
  body.append(h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h3', {}, `컬럼 ${p.columns.length}개`), h('div', { class: 'grow' }), search, bulk),
    h('div', { class: 'tablewrap tall' }, h('table', { class: 'vars' },
      h('thead', {}, h('tr', {}, h('th', { title: '예측할 값' }, 'y'), h('th', { title: '입력 — 미리 아는 값' }, 'x'),
        h('th', {}, '컬럼'), h('th', {}, '추이'), h('th', {}, '단위'), h('th', {}, '범위'), h('th', {}, '결측'), h('th', {}, ''))),
      tbody))), preview);
}

async function ezPreview(c, card) {
  card.hidden = false; card.innerHTML = '';
  card.append(h('div', { class: 'card-head' }, h('h3', {}, c.name),
    h('span', { class: 'sub' }, [c.desc, EZ.units[c.name]].filter(Boolean).join(' · '))));
  const host = h('div', { class: 'chart' }); card.append(host);
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  try {
    const r = await api('/api/preview', { csv: EZ.csv, columns: [c.name], limit: 900 });
    const marks = (r.excluded && r.excluded[c.name]) || [];
    lineChart(host, { series: [{ name: c.name, values: r.series[c.name] }], x: r.t, unit: EZ.units[c.name] || '',
      xfmt: shortT, height: 200, marks,
      caption: marks.length ? `빨간 점 ${marks.length.toLocaleString('ko-KR')}개는 정리하면서 결측으로 바꾼 원래 값입니다.` : '' });
  } catch (e) { host.append(h('div', { class: 'note bad' }, e.message)); }
}

// ── 3. 기간 ─────────────────────────────────────────────────────────────
async function ezRenderPeriod() {
  const body = $('#ez-period-body'); body.innerHTML = '';
  body.append(h('div', { class: 'card' }, h('div', { class: 'sub' }, '차트를 불러오는 중…'), h('div', { class: 'progress' }, h('i'))));
  try {
    if (!EZ.timeline) EZ.timeline = await api('/api/preview', { csv: EZ.csv, columns: [EZ.target], limit: 1400 });
  } catch (e) { body.innerHTML = ''; body.append(h('div', { class: 'note bad' }, e.message)); return; }
  if (!EZ.train[0]) ezPresetSplit('frac');
  ezDrawPeriod();
}

function ezPresetSplit(kind, days) {
  const a = firstDay(), z = lastDay();
  if (kind === 'frac') {
    const span = dayMs(z) - dayMs(a);
    const b = dayOf(dayMs(a) + span * 0.6);
    EZ.train = [a, b]; EZ.test = [addDays(b, 1 + EZ.embargo), z];
  } else {
    const c = addDays(z, -(days - 1));
    EZ.test = [c, z]; EZ.train = [a, addDays(c, -(1 + EZ.embargo))];
  }
  ezFixOrder();
}

function ezFixOrder(changed) {
  // 누수를 화면에서 먼저 막는다: 예측 시작 ≥ 학습 끝 + 1일 + 간격
  const clamp = s => (s < firstDay() ? firstDay() : s > lastDay() ? lastDay() : s);
  EZ.train = EZ.train.map(clamp); EZ.test = EZ.test.map(clamp);
  if (EZ.train[1] < EZ.train[0]) EZ.train[1] = EZ.train[0];
  const minC = minTestStart();
  if (EZ.test[0] < minC) {
    if (changed === 'test') toast(`예측은 학습 끝 다음 날부터 간격 ${EZ.embargo}일을 띄운 ${minC} 이후만 됩니다`);
    EZ.test[0] = minC;
  }
  if (EZ.test[1] < EZ.test[0]) EZ.test[1] = lastDay() >= EZ.test[0] ? lastDay() : EZ.test[0];
}

function ezDrawPeriod() {
  const body = $('#ez-period-body'); body.innerHTML = '';
  const seg = h('div', { class: 'seg' },
    ...[['train', '학습 기간 끌어서 지정'], ['test', '예측 기간 끌어서 지정']].map(([k, lab]) => h('button', {
      class: EZ.dragMode === k ? 'on ' + k : k, onclick: () => { EZ.dragMode = k; ezDrawPeriod(); } }, lab)));
  const presets = h('div', { class: 'actions', style: 'margin:0' },
    h('span', { class: 'sub' }, '빠른 선택'),
    h('button', { class: 'btn ghost sm', onclick: () => { ezPresetSplit('frac'); ezDrawPeriod(); } }, '앞 60% 학습 → 나머지 예측'),
    h('button', { class: 'btn ghost sm', onclick: () => { ezPresetSplit('last', 30); ezDrawPeriod(); } }, '마지막 30일 예측'),
    h('button', { class: 'btn ghost sm', onclick: () => { ezPresetSplit('last', 7); ezDrawPeriod(); } }, '마지막 7일 예측'));
  const chart = h('div', { class: 'chart period' });
  const date = (val, on) => h('input', { type: 'date', value: val, min: firstDay(), max: lastDay(), onchange: e => on(e.target.value) });
  const inputs = h('div', { class: 'periods' },
    h('div', { class: 'pbox train' }, h('div', { class: 'k' }, '■ 학습 기간'),
      h('div', { class: 'row' }, date(EZ.train[0], v => { EZ.train[0] = v; ezFixOrder('train'); ezDrawPeriod(); }),
        h('span', {}, '~'), date(EZ.train[1], v => { EZ.train[1] = v; ezFixOrder('train'); ezDrawPeriod(); }))),
    h('div', { class: 'pbox gap' }, h('div', { class: 'k' }, '간격'),
      h('div', { class: 'row' }, h('input', { type: 'number', min: 1, max: 60, step: 1, value: EZ.embargo, style: 'width:64px',
        onchange: e => { EZ.embargo = Math.max(1, Math.round(+e.target.value || 1)); ezFixOrder('test'); ezDrawPeriod(); } }), h('span', {}, '일'))),
    h('div', { class: 'pbox test' }, h('div', { class: 'k' }, '■ 예측 기간'),
      h('div', { class: 'row' }, date(EZ.test[0], v => { EZ.test[0] = v; ezFixOrder('test'); ezDrawPeriod(); }),
        h('span', {}, '~'), date(EZ.test[1], v => { EZ.test[1] = v; ezFixOrder('test'); ezDrawPeriod(); }))));
  const checks = h('div', { id: 'ez-checks' }, h('div', { class: 'sub' }, '검사 중…'));
  const next = h('button', { class: 'btn lg', id: 'ez-period-next', disabled: true,
    onclick: () => { unlock('ez-model'); ezGo('ez-model'); } }, '다음: 모델 →');
  body.append(h('div', { class: 'card' },
    h('div', { class: 'card-head' }, seg, h('div', { class: 'grow' }), presets),
    chart, inputs), checks, h('div', { class: 'actions' }, next));
  periodChart(chart, {
    t: EZ.timeline.t, y: EZ.timeline.series[EZ.target], unit: EZ.units[EZ.target] || '',
    train: EZ.train, test: EZ.test, mode: EZ.dragMode,
    onSelect: (d0, d1) => {
      if (EZ.dragMode === 'train') { EZ.train = [d0, d1]; ezFixOrder('train'); }
      else { EZ.test = [d0, d1]; ezFixOrder('test'); }
      ezDrawPeriod();
    },
  });
  ezRunCheck();
}

function ezBody(extra) {
  const fm = {};
  EZ.features.forEach(c => {
    const m = EZ.fmap[c] || {};
    fm[c] = { targets: m.targets || [], unit: m.unit || EZ.units[c] || '1', scale: m.scale || 1 };
  });
  const b = {
    csv: EZ.csv, target: EZ.target, features: EZ.features, train: EZ.train, test: EZ.test,
    embargo_days: EZ.embargo, time_column: EZ.profile.time_column, target_unit: EZ.units[EZ.target] || '',
  };
  if (extra && EZ.mode === 'physics' && EZ.model) {
    Object.assign(b, {
      model: EZ.model, feature_map: fm,
      target_map: EZ.tmap && EZ.tmap.targets && EZ.tmap.targets.length
        ? { targets: EZ.tmap.targets, unit: EZ.tmap.unit || EZ.units[EZ.target] || '1', sigma: EZ.tmap.sigma } : null,
      extra_obs: Object.entries(EZ.extra)
        .filter(([col, m]) => col !== EZ.target && !EZ.features.includes(col) && m.targets && m.targets.length)
        .map(([col, m]) => ({ column: col, targets: m.targets, unit: m.unit || EZ.units[col] || '1', sigma: m.sigma })),
      params: EZ.params, band: EZ.band,
      // 기본값과 다른 선택만 보낸다 (기본값은 모델 그대로)
      closures: Object.fromEntries(Object.entries(EZ.closures).filter(([k, v]) => {
        const sl = ((EZ.catalog && EZ.catalog.closures) || []).find(c => c.key === k);
        return v && (!sl || v !== sl.default);
      })),
    });
  }
  if (EZ.limit !== '' && isFinite(+EZ.limit)) b.limit = +EZ.limit;
  return b;
}

let checkTimer;
function ezRunCheck() {
  clearTimeout(checkTimer);
  checkTimer = setTimeout(async () => {
    const seq = ++EZ.checkSeq;
    try {
      const r = await api('/api/easy/check', ezBody(false));
      if (seq !== EZ.checkSeq) return;
      EZ.check = r; ezRenderChecks();
    } catch (e) { const c = $('#ez-checks'); if (c) { c.innerHTML = ''; c.append(h('div', { class: 'note bad' }, e.message)); } }
  }, 220);
}

function ezRenderChecks() {
  const r = EZ.check, box = $('#ez-checks'); if (!box) return;
  box.innerHTML = '';
  const item = (ok, title, detail) => h('div', { class: 'check ' + (ok === true ? 'ok' : ok === false ? 'no' : 'warn') },
    h('span', { class: 'ic' }, ok === true ? '✓' : ok === false ? '✕' : '!'),
    h('div', {}, h('div', { class: 't', html: mdBold(title) }), detail ? h('div', { class: 'd', html: mdBold(detail) }) : null));
  const list = h('div', { class: 'checks' });
  (r.errors || []).forEach(e => list.append(item(false, e)));
  if (r.n_train !== undefined) {
    list.append(item(r.is_future, r.is_future ? '예측 기간이 학습 기간보다 **뒤**입니다' : '예측 기간이 학습과 시간상 섞였습니다',
      '학습 데이터에 미래 시점이 섞이면 이미 답을 본 셈입니다 (누수).'));
    list.append(item(r.embargo_days >= 1 - 1e-9, `학습 끝과 예측 시작 사이 간격 **${fmt(r.embargo_days, 1)}일**`,
      '경계 부근은 거의 같은 상태라 예측이 쉬워집니다. 1일 이상 띄웁니다.'));
    list.append(item(true, `학습 **${r.n_train.toLocaleString('ko-KR')}행** · 예측 **${r.n_test.toLocaleString('ko-KR')}행**`,
      '값 정제(교정 창·스파이크 제거)는 학습·간격·예측 구간을 따로 합니다.'));
    const ex = r.extrapolation;
    if (ex === ex && ex !== null) {
      list.append(item(ex >= 0.5 ? true : null,
        `예측 기간의 **${fmt(ex * 100, 0)}%** 가 학습 때 없던 운전조건 (외삽)`,
        ex >= 0.5 ? '물리모델과 ML 의 차이가 드러나는 조건입니다.'
                  : '학습 범위 안에서는 ML 도 잘 맞힙니다. 운전조건이 달라진 기간을 예측 기간으로 고르면 차이가 드러납니다.'));
    }
  }
  (r.warnings || []).forEach(w => list.append(item(null, w)));
  box.append(h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', {}, '누수·외삽 검사'),
    h('span', { class: 'sub' }, '기간이나 입력을 바꿀 때마다 다시 봅니다')), list));
  if (r.ranges && r.ranges.length) box.append(ezRangeCard(r.ranges));
  const btn = $('#ez-period-next'); if (btn) btn.disabled = !r.ok;
}

function ezRangeCard(ranges) {
  const rows = ranges.map(g => {
    const lo = Math.min(g.train_lo, g.test_lo), hi = Math.max(g.train_hi, g.test_hi), span = (hi - lo) || 1;
    const pos = (a, b) => `left:${((a - lo) / span) * 100}%;width:${Math.max(((b - a) / span) * 100, 0.8)}%`;
    return h('div', { class: 'rrow' },
      h('div', { class: 'rname', title: g.column }, g.column),
      h('div', { class: 'rbars' },
        h('div', { class: 'rbar train', style: pos(g.train_lo, g.train_hi), title: `학습 ${fmt(g.train_lo)} ~ ${fmt(g.train_hi)}` }),
        h('div', { class: 'rbar test', style: pos(g.test_lo, g.test_hi), title: `예측 ${fmt(g.test_lo)} ~ ${fmt(g.test_hi)}` })),
      h('div', { class: 'rpct ' + (g.outside_pct >= 50 ? 'hi' : '') }, g.outside_pct > 0 ? `${fmt(g.outside_pct, 0)}% 범위 밖` : '범위 안'));
  });
  return h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h3', {}, '입력별 범위 — 학습 vs 예측'),
      h('span', { class: 'sub' }, '주황 막대가 파란 막대 밖으로 나간 만큼이 외삽입니다')),
    h('div', { class: 'ranges' }, rows),
    h('div', { class: 'legend' }, h('span', {}, h('i', { style: `background:${PAL().train}` }), '학습 범위'),
      h('span', {}, h('i', { style: `background:${PAL().test}` }), '예측 기간 범위')));
}

function periodChart(host, opt) {
  const draw = () => {
    host.innerHTML = '';
    const P = PAL();
    const W = Math.max(host.clientWidth || 700, 320), H = 250, m = { t: 26, r: 14, b: 26, l: 52 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const ts = opt.t.map(toMs), y = opt.y || [];
    const t0 = ts[0], t1 = ts[ts.length - 1] + 1;
    const X = t => m.l + ((t - t0) / (t1 - t0)) * iw;
    const fin = y.filter(v => v !== null && isFinite(v));
    let lo = Math.min(...fin), hi = Math.max(...fin);
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
    const Y = v => m.t + ih - ((v - lo) / (hi - lo)) * ih;
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
    const band = (a, b, fill, label, cls) => {
      const x0 = Math.max(m.l, X(a)), x1 = Math.min(W - m.r, X(b));
      if (x1 <= x0) return;
      svg.append(svgEl('rect', { x: x0, y: m.t, width: x1 - x0, height: ih, fill, opacity: 0.13, class: cls }));
      const tx = svgEl('text', { x: (x0 + x1) / 2, y: m.t - 9, 'text-anchor': 'middle', 'font-size': 11.5,
        'font-weight': 600, fill }); tx.textContent = label; svg.append(tx);
    };
    const trA = dayMs(opt.train[0]), trB = dayMs(opt.train[1]) + DAY, teA = dayMs(opt.test[0]), teB = dayMs(opt.test[1]) + DAY;
    band(trA, trB, P.train, '학습', 'b-train');
    if (teA > trB) {
      const x0 = X(trB), x1 = X(teA);
      if (x1 > x0) svg.append(svgEl('rect', { x: x0, y: m.t, width: x1 - x0, height: ih, fill: 'url(#hatch)', opacity: 0.8 }));
    }
    band(teA, teB, P.test, '예측', 'b-test');
    const defs = svgEl('defs');
    const pat = svgEl('pattern', { id: 'hatch', width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' });
    pat.append(svgEl('line', { x1: 0, y1: 0, x2: 0, y2: 6, stroke: P.ink3, 'stroke-width': 1.2, opacity: 0.5 }));
    defs.append(pat); svg.prepend(defs);
    const g = svgEl('g', { class: 'grid axis' });
    for (const v of niceTicks(lo, hi, 4)) {
      g.append(svgEl('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
      const tx = svgEl('text', { x: m.l - 7, y: Y(v) + 3.5, 'text-anchor': 'end' }); tx.textContent = fmt(v); g.append(tx);
    }
    svg.append(g);
    const ax = svgEl('g', { class: 'axis' });
    for (let k = 0; k <= 4; k++) {
      const t = t0 + ((t1 - t0) * k) / 4;
      const tx = svgEl('text', { x: X(t), y: H - 8, 'text-anchor': k === 0 ? 'start' : k === 4 ? 'end' : 'middle' });
      tx.textContent = dayOf(t).slice(2); ax.append(tx);
    }
    svg.append(ax);
    let d = '', pen = false;
    y.forEach((v, i) => {
      if (v === null || !isFinite(v)) { pen = false; return; }
      d += (pen ? 'L' : 'M') + X(ts[i]).toFixed(1) + ' ' + Y(v).toFixed(1) + ' '; pen = true;
    });
    svg.append(svgEl('path', { class: 'mark', d, stroke: P.y, 'stroke-width': 1.3 }));
    const sel = svgEl('rect', { y: m.t, height: ih, fill: opt.mode === 'train' ? P.train : P.test, opacity: 0, class: 'sel' });
    svg.append(sel);
    const hot = svgEl('rect', { class: 'hot drag', x: m.l, y: m.t, width: iw, height: ih });
    svg.append(hot);
    const tip = h('div', { class: 'tip' });
    host.append(svg, tip);
    host.append(h('div', { class: 'sub', style: 'margin:6px 2px 0' },
      `차트 위를 끌면 ${opt.mode === 'train' ? '학습' : '예측'} 기간이 됩니다 · 빗금은 학습과 예측 사이 간격 · y: ${EZ.target}${opt.unit ? ' [' + opt.unit + ']' : ''}`));
    const tAt = ev => { const r = svg.getBoundingClientRect(); const px = (ev.clientX - r.left) * (W / r.width);
      return t0 + (Math.max(m.l, Math.min(W - m.r, px)) - m.l) / iw * (t1 - t0); };
    let start = null;
    hot.addEventListener('pointerdown', ev => { start = tAt(ev); hot.setPointerCapture(ev.pointerId); });
    hot.addEventListener('pointermove', ev => {
      const t = tAt(ev), r = svg.getBoundingClientRect();
      tip.innerHTML = ''; tip.append(h('div', { class: 'tt' }, dayOf(t))); tip.style.opacity = 1;
      tip.style.left = Math.min(r.width - 110, (X(t) / W) * r.width + 10) + 'px'; tip.style.top = '30px';
      if (start === null) return;
      const a = Math.min(start, t), b = Math.max(start, t);
      sel.setAttribute('x', X(a)); sel.setAttribute('width', Math.max(1, X(b) - X(a))); sel.setAttribute('opacity', 0.28);
    });
    hot.addEventListener('pointerleave', () => (tip.style.opacity = 0));
    hot.addEventListener('pointerup', ev => {
      if (start === null) return;
      const t = tAt(ev), a = Math.min(start, t), b = Math.max(start, t);
      start = null;
      if (b - a < DAY / 2) { sel.setAttribute('opacity', 0); return; }
      opt.onSelect(dayOf(a), dayOf(b));
    });
  };
  draw();
  if (host._ro) host._ro.disconnect();
  host._ro = new ResizeObserver(() => draw()); host._ro.observe(host);
}

// ── 4. 모델·실행 ────────────────────────────────────────────────────────
async function ezRenderModel() {
  const body = $('#ez-model-body'); body.innerHTML = '';
  if (!EZ.modelsList) { try { EZ.modelsList = (await api('/api/workspace')).models; } catch (_) { EZ.modelsList = []; } }
  if (EZ.model === null && EZ.preset && EZ.preset.model) EZ.model = EZ.preset.model;
  if (EZ.model === null && !EZ.modelsList.length) EZ.mode = 'ml';

  const card = (mode, title, desc, badge) => h('button', { class: 'choice' + (EZ.mode === mode ? ' on' : ''),
    onclick: () => { EZ.mode = mode; ezRenderModel(); } },
    h('div', { class: 't' }, title, badge ? h('span', { class: 'badge good' }, badge) : null), h('div', { class: 'd' }, desc));
  body.append(h('div', { class: 'choices' },
    card('physics', '물리모델 + ML 비교', '지배방정식 모델을 보정하고, 같은 조건의 ML 과 나란히 비교합니다. 학습 범위 밖(증설·새 운전조건)에서 차이가 납니다.', '권장'),
    card('ml', 'ML 만', '물리모델 없이 다항식·그래디언트 부스팅만 비교합니다. 계통 모델이 아직 없을 때.')));

  const pane = h('div', { id: 'ez-model-pane' });
  body.append(pane);
  if (EZ.mode === 'physics') await ezRenderPhysics(pane);
  else pane.append(h('div', { class: 'note' }, h('b', {}, 'ML 만 비교합니다. '),
    '물리모델은 계통 구조(무엇이 무엇에 연결되는가)를 알아야 만들 수 있고, 그 구조는 데이터에서 유도할 수 없습니다. ' +
    '모델이 생기면 ‘물리모델 + ML 비교’를 고르세요.'));

  const opts = h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h3', {}, '선택 사항')),
    h('div', { class: 'optrow' },
      h('label', { class: 'row' }, `${EZ.target} 관리기준`,
        h('input', { type: 'number', value: EZ.limit, placeholder: '없음', style: 'width:110px',
          oninput: e => (EZ.limit = e.target.value) }),
        h('span', { class: 'sub', style: 'margin:0' }, '넣으면 초과 시간 적중·오경보를 셉니다')),
      EZ.mode === 'physics' ? h('label', { class: 'row' },
        h('input', { type: 'checkbox', checked: EZ.band, onchange: e => (EZ.band = e.target.checked) }),
        '상태 변동 폭도 계산 (학습 기간을 토막 내 다시 보정 — 2~4분 더 걸림)') : null));
  body.append(opts);
  const run = h('button', { class: 'btn lg', id: 'ez-run', onclick: ezRun }, '▶ 실행');
  body.append(h('div', { class: 'actions' }, run,
    h('span', { class: 'sub', style: 'margin:0' },
      EZ.mode === 'physics' ? '물리 보정 1~3분 + 예측 행 수에 비례 (5분 데이터 한 달에 약 1분)' : '수 초')));
  body.append(h('div', { id: 'ez-progress' }));
  ezDrawProgress();
}

async function ezRenderPhysics(pane) {
  pane.innerHTML = '';
  const models = EZ.modelsList || [];
  const sel = h('select', { onchange: async e => {
    // 다른 계통의 변수 이름이 따라붙지 않게 연결을 전부 비운다
    Object.assign(EZ, { model: e.target.value, catalog: null, params: [], fmap: {}, tmap: null, extra: {},
                        closures: {}, advice: null, adviceKey: null });
    await ezRenderPhysics(pane);
  } },
    h('option', { value: '' }, '— 모델을 고르세요 —'),
    models.map(m => h('option', { value: m.path, selected: m.path === EZ.model }, `${m.name}  (${m.path})`)));
  pane.append(h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h3', {}, '물리모델'),
      h('span', { class: 'sub' }, EZ.preset && EZ.preset.model === EZ.model ? `저장된 설정 ‘${EZ.preset.name}’ 의 모델` : '작업 폴더의 모델 파일')),
    sel));
  if (!EZ.model) return;
  if (!EZ.catalog || EZ.catalog._model !== EZ.model) {
    pane.append(h('div', { class: 'card' }, h('div', { class: 'sub' }, '모델을 조립하는 중…'), h('div', { class: 'progress' }, h('i'))));
    ezApplyPreset();
    const hint = EZ.tmap && EZ.tmap.targets && EZ.tmap.targets[0];
    try {
      EZ.catalog = await api('/api/easy/catalog', { model: EZ.model, target: hint || null });
      EZ.catalog._model = EZ.model;
    } catch (e) { pane.lastChild.remove(); pane.append(h('div', { class: 'note bad' }, e.message)); return; }
    pane.lastChild.remove();
  }
  const cat = EZ.catalog;
  const inputOpts = cur => {
    const groups = {};
    cat.inputs.forEach(o => (groups[o.group] = groups[o.group] || []).push(o));
    const curVal = (cur || []).join(',');
    const known = cat.inputs.some(o => o.value === curVal);
    return [h('option', { value: '' }, '— 물리모델에 안 씀 (ML 만) —'),
      !known && curVal ? h('option', { value: curVal, selected: true }, curVal) : null,
      ...Object.entries(groups).map(([g, os]) => h('optgroup', { label: g },
        os.map(o => h('option', { value: o.value, selected: o.value === curVal }, `${o.label}${o.unit ? ` [${o.unit}]` : ''}`))))];
  };
  const obsOpts = cur => [h('option', { value: '' }, '— 연결 안 함 —'),
    ...cat.observables.map(o => h('option', { value: o.value, selected: (cur || [])[0] === o.value },
      o.unit && !o.label.includes('[') ? `${o.label} [${o.unit}]` : o.label))];
  const unitIn = (col, m) => h('input', { class: 'unit', type: 'text', value: (m && m.unit) || EZ.units[col] || '',
    oninput: e => { m.unit = e.target.value.trim(); ezCheckMapping(); } });

  EZ.tmap = EZ.tmap || { targets: [], unit: EZ.units[EZ.target] || '' };
  const rows = [h('tr', { class: 'is-y' },
    h('td', {}, h('span', { class: 'badge' }, 'y')), h('td', { class: 'name' }, EZ.target),
    h('td', {}, h('select', { onchange: e => { EZ.tmap.targets = e.target.value ? [e.target.value] : []; ezCheckMapping(); } }, obsOpts(EZ.tmap.targets))),
    h('td', {}, unitIn(EZ.target, EZ.tmap)), h('td', {}, ''))];
  EZ.features.forEach(c => {
    const m = EZ.fmap[c] = EZ.fmap[c] || { targets: [], unit: EZ.units[c] || '', scale: 1 };
    rows.push(h('tr', {},
      h('td', {}, h('span', { class: 'badge' }, 'x')), h('td', { class: 'name' }, c),
      h('td', {}, h('select', { onchange: e => { m.targets = e.target.value ? e.target.value.split(',') : []; ezCheckMapping(); } }, inputOpts(m.targets))),
      h('td', {}, unitIn(c, m)),
      h('td', {}, h('input', { type: 'number', step: 'any', value: m.scale, style: 'width:90px', title: '컬럼 값에 곱할 배율 (예: 60 Hz → 회전수비 1.0 이면 0.016667)',
        oninput: e => { m.scale = +e.target.value || 1; ezCheckMapping(); } }))));
  });
  const others = EZ.profile.columns.filter(c => c.usable && c.name !== EZ.target && !EZ.features.includes(c.name));
  const extraRows = others.map(c => {
    const m = EZ.extra[c.name] = EZ.extra[c.name] || { targets: [], unit: EZ.units[c.name] || '' };
    return h('tr', {}, h('td', { class: 'name' }, c.name, c.desc ? h('div', { class: 'desc' }, c.desc) : null),
      h('td', {}, h('select', { onchange: e => { m.targets = e.target.value ? [e.target.value] : []; ezCheckMapping(); } }, obsOpts(m.targets))),
      h('td', {}, unitIn(c.name, m)));
  });
  const nExtra = Object.values(EZ.extra).filter(m => m.targets && m.targets.length).length;
  pane.append(h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h3', {}, '컬럼 ↔ 모델 연결'),
      h('span', { class: 'sub' }, '데이터의 어떤 컬럼이 모델의 무엇인지 알려주세요. 단위가 다르면 실행 전에 알려줍니다.')),
    h('div', { class: 'tablewrap' }, h('table', { class: 'map' },
      h('thead', {}, h('tr', {}, h('th', {}, ''), h('th', {}, 'CSV 컬럼'), h('th', {}, '모델 변수'), h('th', {}, '컬럼 단위'), h('th', {}, '배율'))),
      h('tbody', {}, rows))),
    h('div', { id: 'ez-map-issues' }),
    others.length ? h('details', { class: 'fold', style: 'margin-top:12px', open: nExtra > 0 },
      h('summary', {}, h('b', {}, `보조 관측 (선택) — 연결 ${nExtra}개`),
        h('span', { class: 'sub' }, ' 유량·차압·온도처럼 모델이 계산하는 다른 계측값을 연결하면 보정이 정확해집니다. 입력으로는 쓰지 않습니다.')),
      h('div', { class: 'tablewrap', style: 'margin-top:8px' }, h('table', { class: 'map' },
        h('thead', {}, h('tr', {}, h('th', {}, 'CSV 컬럼'), h('th', {}, '모델 출력'), h('th', {}, '단위'))),
        h('tbody', {}, extraRows)))) : null));

  if (!EZ.params.length) EZ.params = [...(cat.suggested || [])];
  const prow = cat.params.map(p => h('label', { class: 'prm' + (EZ.params.includes(p.value) ? ' on' : '') },
    h('input', { type: 'checkbox', checked: EZ.params.includes(p.value), onchange: e => {
      EZ.params = e.target.checked ? [...EZ.params, p.value] : EZ.params.filter(x => x !== p.value);
      e.target.parentNode.classList.toggle('on', e.target.checked);
    } }),
    h('span', { class: 'mono' }, p.value),
    h('span', { class: 'sub' }, `${p.desc || ''}${p.sensitivity_pct ? ` · 민감도 ${fmt(p.sensitivity_pct, 0)}%` : ''}`)));
  pane.append(h('details', { class: 'card fold' },
    h('summary', {}, h('b', {}, `보정할 물리 파라미터 — ${EZ.params.length}개 선택`),
      h('span', { class: 'sub' }, cat.declared_params.length ? ' 모델이 권하는 것을 골라 두었습니다' : ' 타깃에 민감한 것을 골라 두었습니다')),
    h('div', { class: 'note', style: 'margin-top:10px' }, '많이 고를수록 좋은 게 아닙니다. 데이터로 서로 구분되지 않는 파라미터를 같이 풀면 값이 흔들립니다 — 결과 화면의 식별성 경고를 보세요.'),
    h('div', { class: 'prms' }, prow)));
  if ((cat.closures || []).length) pane.append(ezClosureCard());
  ezCheckMapping();
}

// ── 구성방정식 후보: 코드가 추리고 근거를 붙이면, 고르는 건 사용자 ─────────────
const ST_CLS = { '추천': 'good', '근소하게 나음': 'good', '불안정': 'warn', '현재 식': '', '비슷함': '', '나쁨': 'bad', '제외': 'bad', '구별 불가': 'warn', '비교 안 함': '' };

function ezAdviceKey() {
  const b = ezBody(true);
  return JSON.stringify([b.csv, b.target, b.features, b.train, b.test, b.embargo_days, b.model, b.feature_map,
                         b.target_map, b.extra_obs, b.params]);
}

function ezClosureCard() {
  const card = h('div', { class: 'card', id: 'ez-closures' });
  ezDrawClosures(card);
  return card;
}

function ezDrawClosures(card) {
  card = card || $('#ez-closures'); if (!card) return;
  card.innerHTML = '';
  const cat = EZ.catalog.closures || [];
  const adv = EZ.advice;
  const stale = adv && EZ.adviceKey !== ezAdviceKey();
  const bySlot = {}; (adv ? adv.slots : []).forEach(s => (bySlot[s.key] = s));
  const run = EZ.adviceRun;
  card.append(h('div', { class: 'card-head' }, h('h3', {}, '구성방정식 — 후보에서 고르기'),
    h('span', { class: 'sub' }, '보존법칙은 고정하고, 장비 특성을 나타내는 식만 후보로 바꿔 봅니다')));
  card.append(h('div', { class: 'actions', style: 'margin:0 0 10px' },
    h('button', { class: 'btn', id: 'ez-advise', disabled: !!run, onclick: ezAdvise },
      adv ? '다시 비교하기' : '데이터로 후보 비교하기'),
    h('span', { class: 'sub', style: 'margin:0' },
      '학습 기간만 씁니다 (앞부분으로 보정 → 1일 간격 → 끝부분으로 채점). 예측 기간은 보지 않습니다. 후보 수만큼 보정하므로 수 분 걸립니다.')));
  if (run) card.append(h('div', { class: 'card inset' }, h('div', { class: 'card-head' }, h('h3', {}, '후보 비교 중'),
    h('span', { class: 'sub' }, `${Math.round((Date.now() - run.t0) / 1000)}초`)), h('div', { class: 'progress' }, h('i')),
    h('div', { class: 'sub', style: 'margin:8px 0 0' }, run.msg)));
  if (EZ.adviceError) card.append(h('div', { class: 'note bad' }, h('b', {}, '비교하지 못했습니다. '), EZ.adviceError));
  if (adv) {
    const sp = adv.split;
    card.append(h('div', { class: 'note' + (stale ? ' warn' : '') },
      stale ? h('b', {}, '기간·연결이 바뀌었습니다 — 결과가 오래되었습니다. 다시 비교하세요. ') : null,
      `보정 ${sp.inner[0].slice(0, 10)} ~ ${sp.inner[1].slice(0, 10)} (${sp.n_rows}행) · 간격 ${sp.embargo_days}일 · `,
      `검증 ${sp.validate[0].slice(0, 10)} ~ ${sp.validate[1].slice(0, 10)} (${sp.n_validate_hours}시간, 외삽 행 ${fmt(sp.validate_outside_pct, 0)}%) · `,
      `보정 ${adv.n_calibrations}회 · ${fmt(adv.seconds / 60, 1)}분`));
    (adv.notes || []).forEach(n => card.append(h('div', { class: 'note warn' }, n)));
  }
  // 슬롯: 비교 결과가 있으면 영향 큰 순, '영향 작음'은 접어 둔다
  const order = adv ? adv.slots.map(s => s.key) : cat.map(c => c.key);
  const rows = order.map(k => cat.find(c => c.key === k)).filter(Boolean);
  const main = rows.filter(c => !bySlot[c.key] || bySlot[c.key].status !== '영향 작음');
  const minor = rows.filter(c => bySlot[c.key] && bySlot[c.key].status === '영향 작음');
  main.forEach(c => card.append(ezSlotView(c, bySlot[c.key])));
  if (minor.length) card.append(h('details', { class: 'fold', style: 'margin-top:10px' },
    h('summary', {}, h('b', {}, `예측값에 영향이 작아 비교하지 않은 식 ${minor.length}개`),
      h('span', { class: 'sub' }, ' 파라미터를 10% 바꿔도 예측값이 1% 미만으로 변합니다. 직접 바꿀 수는 있습니다.')),
    minor.map(c => ezSlotView(c, bySlot[c.key]))));
}

function ezSlotView(c, a) {
  const cur = EZ.closures[c.key] || c.current;
  const chips = [];
  if (a) {
    chips.push(h('span', { class: 'chip static' }, `예측값 영향 ${fmt(a.relevance_pct, 1)}%`));
    a.excitation.forEach(e => chips.push(h('span', { class: 'chip static' + (e.ok ? '' : ' warnchip') },
      `${e.var} 변화 폭 ${fmt(e.span_pct, 0)}%`)));
    a.residual_corr.filter(r => Math.abs(r.r) >= 0.2).forEach(r => chips.push(h('span', { class: 'chip static' },
      `현재 식 잔차 ↔ ${r.var} 상관 ${r.r >= 0 ? '+' : ''}${fmt(r.r, 2)}`)));
  }
  const cands = c.candidates.map(o => {
    const e = a ? a.candidates.find(x => x.id === o.id) : null;
    const st = e && e.status;
    const ev = [];
    if (e && e.evaluated && e.val_rmse !== undefined) {
      ev.push(`검증 RMSE ${fmt(e.val_rmse)} (${EZ.advice.split.scored_on})`
        + (e.val_halves ? ` · 앞/뒤 절반 ${fmt(e.val_halves[0])}/${fmt(e.val_halves[1])}` : ''));
      const v = e.vs_current || e.vs_recommended;
      if (v) ev.push(`${e.vs_current ? '현재 식' : '추천'} 대비 ${v.diff >= 0 ? '+' : ''}${fmt(v.diff)} ± ${fmt(v.se)} → ${v.verdict}`);
      const own = e.id === a.current ? e.fit : (e.new_params || []);
      if (own.length && e.ident) ev.push(`${e.id === a.current ? '보정 파라미터' : '새 파라미터 ' + own.map(p => p.split('.').pop()).join(', ')} `
        + `상대표준오차 ${fmt(e.ident.max_rel_se_pct, 0)}% · 최악 상관 ${fmt(e.ident.worst_corr, 2)}`);
    }
    return h('label', { class: 'cand' + (cur === o.id ? ' on' : '') },
      h('input', { type: 'radio', name: 'cl-' + c.key, checked: cur === o.id,
        onchange: () => { EZ.closures[c.key] = o.id; ezDrawClosures(); } }),
      h('div', { class: 'cbody' },
        h('div', { class: 'ct' }, h('b', {}, o.title), h('span', { class: 'badge' }, o.role_label),
          st ? h('span', { class: 'badge ' + (ST_CLS[st] || '') }, st) : null,
          o.id === c.default ? h('span', { class: 'sub', style: 'margin:0' }, '기본값') : null),
        h('div', { class: 'fm', html: fmla(o.formula, new Set((e ? e.fit : o.fit).map(p => p.split('.').pop()))) }),
        ev.length ? h('div', { class: 'cev' }, ev.join(' · ')) : null,
        e && e.reasons.length ? h('ul', { class: 'log' }, e.reasons.map(r => h('li', {}, r))) : null,
        h('div', { class: 'cnote' }, o.note)));
  });
  return h('div', { class: 'slot' },
    h('div', { class: 'shead' }, h('b', {}, `${c.component} — ${c.title}`), h('span', { class: 'mono sub' }, c.key),
      a && a.recommended ? h('span', { class: 'badge good', style: 'margin-left:auto' },
        `추천: ${(c.candidates.find(o => o.id === a.recommended) || {}).title || a.recommended}`) : null),
    chips.length ? h('div', { class: 'xchips' }, chips) : null,
    a && a.message ? h('div', { class: 'smsg' }, a.message) : null,
    h('div', { class: 'cands' }, cands));
}

async function ezAdvise() {
  if (EZ.adviceRun) return;
  EZ.adviceError = null;
  EZ.adviceRun = { t0: Date.now(), msg: '시작하는 중…' };
  ezDrawClosures();
  const tick = setInterval(() => ezDrawClosures(), 1000);
  const key = ezAdviceKey();
  try {
    const { job } = await api('/api/easy/closures', ezBody(true));
    for (;;) {
      await new Promise(r => setTimeout(r, 1500));
      const j = await api('/api/job?id=' + job.id);
      EZ.adviceRun.msg = j.message || j.status;
      if (j.status === 'running') continue;
      if (j.status === 'error') throw new Error(j.error);
      EZ.advice = j.result; EZ.adviceKey = key; break;
    }
    // 추천을 골라 둔다 — 바꾸는 것은 사용자
    EZ.advice.slots.forEach(sl => { EZ.closures[sl.key] = sl.recommended; });
    const changed = EZ.advice.slots.filter(sl => sl.recommended !== sl.default);
    toast(changed.length ? `추천대로 ${changed.length}개 식을 바꿔 두었습니다 — 직접 바꿀 수 있습니다`
                         : '모든 슬롯에서 현재 식 유지가 추천됐습니다');
  } catch (e) {
    EZ.adviceError = e.message;
  } finally {
    clearInterval(tick); EZ.adviceRun = null; ezDrawClosures();
  }
}

function ezPresetTarget() {
  const p = EZ.preset;
  const o = p && p.model === EZ.model && p.observations.find(x => x.column === EZ.target);
  return o ? { targets: o.targets, unit: o.unit, sigma: o.sigma } : null;
}

// 저장된 설정의 연결을 비어 있는 칸에만 채운다 (사용자가 바꾼 것은 건드리지 않는다)
function ezApplyPreset() {
  const p = EZ.preset;
  if (!p || p.model !== EZ.model) return;
  p.inputs.forEach(i => {
    if (EZ.features.includes(i.column) && !(EZ.fmap[i.column] && EZ.fmap[i.column].targets.length))
      EZ.fmap[i.column] = { targets: i.targets, unit: i.unit, scale: i.scale || 1 };
  });
  if (!(EZ.tmap && EZ.tmap.targets.length)) EZ.tmap = ezPresetTarget();
  p.observations.forEach(o => {
    if (o.column !== EZ.target && !EZ.features.includes(o.column) && !(EZ.extra[o.column] && EZ.extra[o.column].targets.length))
      EZ.extra[o.column] = { targets: o.targets, unit: o.unit, sigma: o.sigma };
  });
  if (!EZ.params.length && p.params && p.params.length) EZ.params = [...p.params];
}

let mapTimer;
function ezCheckMapping() {
  clearTimeout(mapTimer);
  mapTimer = setTimeout(async () => {
    const box = $('#ez-map-issues'); if (!box) return;
    try {
      const r = await api('/api/easy/check', ezBody(true));
      box.innerHTML = '';
      const issues = r.mapping_issues || [];
      if (!EZ.tmap || !EZ.tmap.targets.length)
        box.append(h('div', { class: 'note warn' }, `y(${EZ.target})를 모델 변수에 연결해야 물리모델을 돌릴 수 있습니다.`));
      if (!EZ.features.some(c => EZ.fmap[c] && EZ.fmap[c].targets.length))
        box.append(h('div', { class: 'note warn' }, '입력(x)을 하나 이상 모델 변수에 연결하세요. 연결하지 않은 입력은 ML 만 씁니다.'));
      issues.forEach(i => box.append(h('div', { class: 'note bad' }, h('b', {}, `${i.column} → ${i.target}: `), i.message)));
      if (!issues.length && EZ.tmap && EZ.tmap.targets.length)
        box.append(h('div', { class: 'note good' }, '연결과 단위가 맞습니다.'));
    } catch (e) { box.innerHTML = ''; box.append(h('div', { class: 'note bad' }, e.message)); }
  }, 250);
}

function ezDrawProgress() {
  const prog = $('#ez-progress'), run = $('#ez-run');
  if (run) run.disabled = !!EZ.running;
  if (!prog) return;
  prog.innerHTML = '';
  const R = EZ.running;
  if (R) {
    prog.append(h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h3', {}, '실행 중'), h('span', { class: 'sub' }, `${Math.round((Date.now() - R.t0) / 1000)}초`)),
      h('div', { class: 'progress' }, h('i')), h('div', { class: 'sub', style: 'margin:8px 0 0' }, R.msg)));
  } else if (EZ.runError) {
    prog.append(h('div', { class: 'note bad' }, h('b', {}, '실행하지 못했습니다. '), EZ.runError));
  }
}

async function ezRun() {
  if (EZ.running) return;
  if (EZ.mode === 'physics') {
    const why = !EZ.model ? '물리모델을 고르세요.'
      : !(EZ.tmap && EZ.tmap.targets && EZ.tmap.targets.length) ? `y(${EZ.target})를 모델 변수에 연결하세요.`
      : !EZ.features.some(c => EZ.fmap[c] && EZ.fmap[c].targets.length) ? '입력(x)을 하나 이상 모델 변수에 연결하세요.'
      : !EZ.params.length ? '보정할 물리 파라미터를 하나 이상 고르세요.' : null;
    if (why) { toast(why, 'bad'); return; }
  }
  EZ.runError = null;
  EZ.running = { t0: Date.now(), msg: '시작하는 중…' };
  ezDrawProgress();
  const pc = $('#ez-progress'); if (pc) pc.scrollIntoView({ behavior: 'smooth', block: 'center' });
  const tick = setInterval(ezDrawProgress, 1000);
  try {
    const { job } = await api('/api/easy/run', ezBody(true));
    EZ.job = job.id;
    for (;;) {
      await new Promise(r => setTimeout(r, 1000));
      const j = await api('/api/job?id=' + job.id);
      EZ.running.msg = j.message || j.status;
      if (j.status === 'running') continue;
      if (j.status === 'error') throw new Error(j.error);
      EZ.result = j.result; break;
    }
    EZ.running = null; clearInterval(tick); ezDrawProgress();
    unlock('ez-result'); ezGo('ez-result');
  } catch (e) {
    EZ.running = null; clearInterval(tick);
    EZ.runError = e.message; ezDrawProgress();
  }
}

// ── 5. 결과 ─────────────────────────────────────────────────────────────
function ezMetricKey(res) { return res.split.is_extrapolation ? 'outside' : 'test'; }

function ezRenderResult() {
  const r = EZ.result, body = $('#ez-result-body'); body.innerHTML = '';
  const P = PAL(), u = r.target.unit ? ' ' + r.target.unit : '', key = ezMetricKey(r);
  $('#ez-result-sub').textContent = `${r.target.column} · 학습 ${r.split.train_start.slice(0, 10)} ~ ${r.split.train_end.slice(0, 10)} → 예측 ${r.split.test_start.slice(0, 10)} ~ ${r.split.test_end.slice(0, 10)} · ${fmt(r.seconds, 0)}초`;
  r.headline.forEach(t => body.append(h('div', { class: 'note' + (t.includes('아닙니다') || t.includes('수렴하지') ? ' bad' : ''), html: mdBold(t) })));
  (r.notes || []).forEach(n => body.append(h('div', { class: 'note warn', html: mdBold(n) })));

  const best = Math.min(...r.models.map(m => ((m.metrics[key] || m.metrics.test) || {}).rmse ?? Infinity));
  body.append(h('div', { class: 'tiles' },
    r.models.map(m => {
      const mm = m.metrics[key] || m.metrics.test;
      const isBest = mm && mm.rmse === best;
      return h('div', { class: 'tile model' + (isBest ? ' best' : '') },
        h('div', { class: 'k' }, h('i', { class: 'sw', style: `background:${colorOf(m.kind, P)}` }), m.name, isBest ? h('span', { class: 'badge good' }, '가장 정확') : null),
        h('div', { class: 'v' }, fmt(mm && mm.rmse), h('small', {}, `RMSE${u}`)),
        h('div', { class: 'n' }, `편향 ${fmt(mm && mm.bias)}${u} · R² ${fmt(mm && mm.r2, 3)}${m.metrics.predicted_pct < 99.5 ? ` · 예측한 행 ${fmt(m.metrics.predicted_pct, 1)}%` : ''}`));
    }),
    tile('예측 기간의 외삽', fmt((r.split.extrapolation || 0) * 100, 0), '%', `학습 때 없던 운전조건 · 간격 ${fmt(r.split.embargo_days, 1)}일`,
         r.split.is_extrapolation ? 'good' : 'warn')));
  body.append(h('div', { class: 'sub', style: 'margin:6px 2px 0' },
    key === 'outside' ? 'RMSE 는 예측 기간 중 학습 운전영역 밖 행 기준입니다 (물리모델과 ML 의 차이가 나는 곳).' : 'RMSE 는 예측 기간 전체 기준입니다.'));

  // 시계열
  const chart = h('div', { class: 'chart' });
  const legend = h('div', { class: 'legend toggles' });
  body.append(h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', {}, '실측 vs 예측'),
    h('span', { class: 'sub' }, '파란 배경 = 학습(표본), 주황 배경 = 예측 기간 · 범례를 누르면 켜고 끕니다')), chart, legend));
  const drawTs = () => {
    const series = [{ key: 'y', name: '실측', values: r.series.y, color: P.y, width: 1.2 }];
    r.models.forEach(m => series.push({ key: m.name, name: m.name, values: r.series.models[m.name], color: colorOf(m.kind, P), width: m.kind === 'physics' ? 2 : 1.6 }));
    tsChart(chart, { t: r.series.t, region: r.series.region, unit: r.target.unit, limit: r.limit,
      series: series.filter(s => !EZ.hidden[s.key]) });
    legend.innerHTML = '';
    series.forEach(s => legend.append(h('button', { class: EZ.hidden[s.key] ? 'off' : '',
      onclick: () => { EZ.hidden[s.key] = !EZ.hidden[s.key]; drawTs(); } }, h('i', { style: `background:${s.color}` }), s.name)));
  };
  drawTs();

  // 산점도 + 막대
  const sc = h('div', { class: 'chart' }), bars = h('div', { class: 'chart' });
  body.append(h('div', { class: 'cols2' },
    h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', {}, '예측 vs 실측 (예측 기간)'),
      h('span', { class: 'sub' }, '대각선에 가까울수록 정확')), sc),
    h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', {}, 'RMSE — 외삽 행 vs 학습 범위 안 행'),
      h('span', { class: 'sub' }, '차이는 외삽에서 드러납니다')), bars)));
  const test = r.series.region.map(x => x === 'test');
  scatterChart(sc, { unit: r.target.unit, series: r.models.map(m => ({
    name: m.name, color: colorOf(m.kind, P),
    x: r.series.y.filter((_, i) => test[i]), y: r.series.models[m.name].filter((_, i) => test[i]) })) });
  const items = [];
  r.models.forEach(m => {
    if (m.metrics.outside) items.push({ label: `${m.name} · 외삽 행`, value: m.metrics.outside.rmse, color: colorOf(m.kind, P), note: `${m.metrics.outside.n.toLocaleString('ko-KR')}행` });
    if (m.metrics.inside) items.push({ label: `${m.name} · 범위 안`, value: m.metrics.inside.rmse, color: colorOf(m.kind, P) + '88', note: `${m.metrics.inside.n.toLocaleString('ko-KR')}행` });
    if (!m.metrics.outside && !m.metrics.inside) items.push({ label: m.name, value: m.metrics.test.rmse, color: colorOf(m.kind, P) });
  });
  barChart(bars, { items, valueFmt: v => fmt(v) + u });

  // 표
  const rowsT = [];
  r.models.forEach(m => [['train', '학습(표본)'], ['test', '예측 기간 전체'], ['outside', '예측 · 외삽 행'], ['inside', '예측 · 범위 안 행']]
    .forEach(([k, lab]) => {
      const x = m.metrics[k]; if (!x || !x.n) return;
      rowsT.push(h('tr', {}, h('td', {}, h('i', { class: 'sw', style: `background:${colorOf(m.kind, P)}` }), ' ' + m.name), h('td', {}, lab),
        h('td', { class: 'num' }, x.n.toLocaleString('ko-KR')), h('td', { class: 'num' }, fmt(x.rmse)), h('td', { class: 'num' }, fmt(x.mae)),
        h('td', { class: 'num' }, fmt(x.bias)), h('td', { class: 'num' }, fmt(x.r2, 3)),
        h('td', { class: 'num' }, k === 'test' && m.metrics.exceed ? `${m.metrics.exceed.hit}/${m.metrics.exceed.hours} · 오경보 ${m.metrics.exceed.false_alarm}` : '')));
    }));
  body.append(h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', {}, '모델 비교 표'),
    h('span', { class: 'sub' }, `단위: ${r.target.unit || '—'}${r.limit !== null && r.limit !== undefined ? ` · 관리기준 ${fmt(r.limit)} 초과 시간 적중/전체` : ''}`)),
    h('div', { class: 'tablewrap' }, h('table', { class: 'cmp' },
      h('thead', {}, h('tr', {}, h('th', {}, '모델'), h('th', {}, '구간'), h('th', {}, '행'), h('th', {}, 'RMSE'), h('th', {}, 'MAE'),
        h('th', {}, '편향'), h('th', {}, 'R²'), h('th', {}, r.limit !== null && r.limit !== undefined ? '초과 적중' : ''))),
      h('tbody', {}, rowsT)))));

  if (r.explain && (r.explain.physics || (r.explain.ml || []).length)) {
    // 설명이 깨져도 결과(차트·표)는 보여야 한다
    try { body.append(ezExplainCard(r)); }
    catch (e) { body.append(h('div', { class: 'note bad' }, '모델 설명을 그리지 못했습니다: ' + e.message)); }
  }

  if (r.calibration) {
    const c = r.calibration;
    body.append(h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', {}, '보정된 물리 파라미터'),
      h('span', { class: 'sub' }, `${c.rows}행으로 보정 (1시간 평균, 드문 운전상태 포함)`)),
      h('div', { class: 'tablewrap' }, h('table', {},
        h('thead', {}, h('tr', {}, h('th', {}, '파라미터'), h('th', {}, '단위'), h('th', {}, '설계값'), h('th', {}, '보정값'), h('th', {}, '변화'), h('th', {}, '상대표준오차'))),
        h('tbody', {}, c.table.map(p => h('tr', {}, h('td', { class: 'name' }, p.parameter), h('td', {}, p.unit),
          h('td', { class: 'num' }, fmt(p.initial)), h('td', { class: 'num' }, fmt(p.fitted)),
          h('td', { class: 'num' }, (p.change_pct >= 0 ? '+' : '') + fmt(p.change_pct, 1) + '%'),
          h('td', { class: 'num' + (p.rel_stderr_pct > 50 ? ' badcell' : '') }, fmt(p.rel_stderr_pct, 1) + '%')))))),
      c.warnings.length ? h('div', { class: 'note warn' }, h('b', {}, '식별성 경고 — 이 값들은 데이터가 결정하지 못했습니다. '),
        h('ul', { class: 'log' }, c.warnings.map(w => h('li', {}, w)))) :
        h('div', { class: 'note good' }, '식별성 경고 없음 — 보정한 파라미터가 데이터로 결정됐습니다.'),
      h('div', { class: 'sub', style: 'margin:8px 0 0' }, '보정값은 모델 오차를 흡수한 유효값입니다. 설비 진단(막힘 %)에 그대로 쓰기 전에 모델을 검증하세요.')));
  }
  if (r.band) {
    const b = r.band;
    body.append(h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', {}, '상태 변동 폭'),
      h('span', { class: 'sub' }, '학습 기간의 토막별 설비 상태로 예측 기간을 각각 풀어 본 평균')),
      h('div', { class: 'tiles' }, tile('예측 평균 범위', `${fmt(b.lo)} ~ ${fmt(b.hi)}`, u, `실측 평균 ${fmt(b.measured_mean)}${u}`,
        b.measured_mean >= b.lo && b.measured_mean <= b.hi ? 'good' : 'warn')),
      h('ul', { class: 'log', style: 'margin-top:10px' }, b.drift.map(d => h('li', {}, `${d.name}: ${d.values.map(v => fmt(v)).join(' → ')} ${d.unit} [${d.trend}]`)))));
  }
  if (r.data_log && r.data_log.length) body.append(h('details', { class: 'card fold' },
    h('summary', {}, h('b', {}, '데이터 정리 내역'), h('span', { class: 'sub' }, ' (학습·간격·예측 구간을 따로 정제)')),
    h('ul', { class: 'log', style: 'margin-top:10px' }, r.data_log.map(t => h('li', { html: mdBold(t) })))));
  body.append(h('div', { class: 'actions' },
    r.files.predictions ? h('a', { class: 'btn', href: '/' + r.files.predictions, download: r.files.predictions.split('/').pop() }, '⭳ 예측 결과 CSV') : null,
    r.files.config ? h('a', { class: 'btn ghost', href: '/' + r.files.config, target: '_blank', title: 'pf calibrate / pf improve 로 다시 돌릴 수 있는 설정' }, '설정 파일(YAML)') : null,
    h('button', { class: 'btn ghost', onclick: () => ezGo('ez-period') }, '← 기간 바꿔 다시'),
    h('button', { class: 'btn ghost', onclick: () => ezGo('ez-model') }, '← 모델 바꿔 다시')));
}

// ── 모델 설명 ───────────────────────────────────────────────────────────
// 식 표기: _{..} 아래첨자, ^{..} 위첨자, `이름` 은 파라미터 (보정한 것은 강조)
function fmla(text, fitted) {
  let s = String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  s = s.replace(/`([A-Za-z_][A-Za-z0-9_]*)`/g, (_, p) =>
    `<code class="fp${fitted && fitted.has(p) ? ' fit' : ''}" title="${fitted && fitted.has(p) ? '보정한 파라미터' : '파라미터'}">${p}</code>`);
  for (let i = 0; i < 3; i++) s = s.replace(/_\{([^{}]*)\}/g, '<sub>$1</sub>').replace(/\^\{([^{}]*)\}/g, '<sup>$1</sup>');
  return s;
}

function ezExplainCard(r) {
  const P = PAL(), ex = r.explain, tabs = [];
  if (ex.physics) tabs.push({ key: 'physics', label: '물리모델', color: P.physics });
  (ex.ml || []).forEach(m => tabs.push({ key: m.name, label: m.name, color: colorOf(m.kind, P) }));
  if (!EZ.explainTab || !tabs.some(t => t.key === EZ.explainTab)) EZ.explainTab = tabs[0].key;
  const pane = h('div', { class: 'xpane' });
  const seg = h('div', { class: 'seg', role: 'tablist' });
  const show = () => {
    seg.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.key === EZ.explainTab));
    pane.innerHTML = '';
    if (EZ.explainTab === 'physics') pane.append(...ezExplainPhysics(ex.physics, r));
    else pane.append(...ezExplainML(ex.ml.find(m => m.name === EZ.explainTab)));
  };
  tabs.forEach(t => seg.append(h('button', { 'data-key': t.key, role: 'tab', onclick: () => { EZ.explainTab = t.key; show(); } },
    h('i', { class: 'sw', style: `background:${t.color}` }), ' ', t.label)));
  show();
  return h('div', { class: 'card', id: 'ez-explain' },
    h('div', { class: 'card-head' }, h('h3', {}, '모델 설명 — 무엇으로 어떻게 예측했나'),
      h('span', { class: 'sub' }, '같은 학습 데이터·같은 입력으로 만든 세 모델의 식과 설정')),
    seg, pane);
}

function ezExplainPhysics(x, r) {
  const out = [];
  const nfit = x.groups.reduce((a, g) => a + g.fitted.length, 0);
  out.push(h('div', { class: 'xchips' },
    h('span', { class: 'chip static' }, h('b', {}, `방정식 ${x.counts.equations}개`), ` = 미지수 ${x.counts.unknowns}개`),
    h('span', { class: 'chip static' }, `컴포넌트 ${x.counts.components}개`),
    h('span', { class: 'chip static' }, `보정한 파라미터 ${nfit}개`)));
  out.push(h('p', { class: 'xtext' }, x.solve));

  // 흐름
  out.push(h('div', { class: 'xsec' }, h('div', { class: 'k' }, '계통 흐름'),
    h('div', { class: 'flows' }, x.segments.map(seg => h('div', { class: 'flow' },
      seg.flatMap((n, i) => [i ? h('span', { class: 'arr' }, '→') : null, h('span', { class: 'node' }, n)]))))));

  // 예측값과 입력
  const t = x.target;
  const inRows = x.inputs.map(i => {
    const laws = [...new Set(i.uses.flatMap(u => u.laws))];
    const tg = i.uses.map(u => u.target);
    return h('tr', {}, h('td', { class: 'name' }, i.column),
      h('td', { class: 'mono' }, tg.length > 2 ? `${tg[0]} 외 ${tg.length - 1}개` : tg.join(', '),
        i.scale !== 1 ? h('span', { class: 'sub' }, ` × ${+i.scale.toPrecision(4)}`) : null),
      h('td', { class: 'wrap' }, laws.map(l => h('span', { class: 'lawchip' }, l))));
  });
  out.push(h('div', { class: 'xsec' }, h('div', { class: 'k' }, '데이터가 식에 들어가는 곳'),
    h('div', { class: 'note', style: 'margin:0 0 8px' }, h('b', {}, `예측값 ${t.column}`), ` = 모델의 ${t.variable}`,
      t.desc ? ` — ${t.desc}` : '', t.component ? ` (${t.component})` : ''),
    h('div', { class: 'tablewrap' }, h('table', { class: 'xin' },
      h('thead', {}, h('tr', {}, h('th', {}, '입력 컬럼'), h('th', {}, '모델 파라미터'), h('th', {}, '들어가는 식'))),
      h('tbody', {}, inRows)))));

  // 컴포넌트별 식
  const groups = x.groups.map(g => {
    const fitted = new Set(g.fitted.map(f => f.param));
    return h('details', { class: 'xgroup', open: true },
      h('summary', {}, h('b', {}, g.title), h('span', { class: 'mono sub' }, ` ${g.instances.join(', ')}`),
        g.fitted.length ? h('span', { class: 'badge good' }, `보정 ${g.fitted.length}`) : null),
      h('div', { class: 'laws' }, g.laws.map(l => h('div', { class: 'law' },
        h('span', { class: 'kind k-' + ({ '보존법칙': 'c', '상태·정의': 's', '구성방정식': 'x', '경계조건': 'b' }[l.kind] || 'd') }, l.kind),
        h('div', {}, h('div', { class: 'lt' }, l.title), h('div', { class: 'fm', html: fmla(l.formula, fitted) }))))),
      g.fitted.length ? h('div', { class: 'fitline' }, '보정: ', g.fitted.flatMap((f, i) => [i ? ' · ' : '',
        h('code', { class: 'fp fit' }, `${f.instance}.${f.param}`),
        ` ${fmt(f.initial)} → ${fmt(f.fitted)}${f.unit && f.unit !== '1' ? ' ' + f.unit : ''} (${f.change_pct >= 0 ? '+' : ''}${fmt(f.change_pct, 1)}%)`])) : null);
  });
  out.push(h('div', { class: 'xsec' }, h('div', { class: 'k' }, '컴포넌트별 지배방정식'),
    h('div', { class: 'kinds' }, Object.entries(x.kinds).filter(([k]) => x.groups.some(g => g.laws.some(l => l.kind === k)))
      .map(([k, v]) => h('div', {}, h('span', { class: 'kind k-' + ({ '보존법칙': 'c', '상태·정의': 's', '구성방정식': 'x', '경계조건': 'b' }[k] || 'd') }, k), ' ', v))),
    h('p', { class: 'xtext' }, x.connection),
    groups));
  out.push(h('div', { class: 'xsec' }, h('div', { class: 'k' }, '보정 방법'), h('p', { class: 'xtext' }, x.calibrate)));
  return out;
}

function ezExplainML(m) {
  return [
    h('div', { class: 'xchips' }, h('span', { class: 'chip static' }, h('b', {}, m.algorithm))),
    h('div', { class: 'xsec' }, h('div', { class: 'k' }, '식'), h('div', { class: 'fm big', html: fmla(m.formula) }),
      h('p', { class: 'xtext' }, m.fit)),
    h('div', { class: 'xsec' }, h('div', { class: 'k' }, '설정 (학습된 모델에서 읽은 값)'),
      h('div', { class: 'tablewrap' }, h('table', { class: 'xset' },
        h('tbody', {}, m.settings.map(([k, v]) => h('tr', {}, h('td', {}, k), h('td', {}, v))))))),
    h('div', { class: 'xsec' }, h('div', { class: 'k' }, `입력 ${m.inputs.length}개 (물리모델과 같은 컬럼)`),
      h('div', { class: 'wrap' }, m.inputs.map(c => h('span', { class: 'lawchip mono' + (m.frozen.includes(c) ? ' frozen' : '') },
        c, m.frozen.includes(c) ? ' — 학습 동안 고정' : '')))),
    h('div', { class: 'note warn' }, h('b', {}, '학습 범위 밖에서: '), m.extrapolation,
      m.frozen.length ? ` 학습 동안 값이 한 번도 바뀌지 않은 입력(${m.frozen.join(', ')})은 이 모델이 효과를 배울 수 없습니다.` : ''),
  ];
}

function tsChart(host, opt) {
  const draw = () => {
    host.innerHTML = '';
    const P = PAL();
    const W = Math.max(host.clientWidth || 700, 320), H = opt.height || 300, m = { t: 14, r: 14, b: 26, l: 52 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const ts = opt.t.map(toMs), n = ts.length;
    const t0 = ts[0], t1 = ts[n - 1] || t0 + 1;
    const X = t => m.l + ((t - t0) / ((t1 - t0) || 1)) * iw;
    let lo = Infinity, hi = -Infinity;
    opt.series.forEach(s => s.values.forEach(v => { if (v !== null && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
    if (opt.limit !== null && opt.limit !== undefined && isFinite(opt.limit)) { lo = Math.min(lo, opt.limit); hi = Math.max(hi, opt.limit); }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    const pad = (hi - lo) * 0.07 || 1; lo -= pad; hi += pad;
    const Y = v => m.t + ih - ((v - lo) / (hi - lo)) * ih;
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
    // 배경 구간
    let i = 0;
    while (i < n) {
      let j = i; while (j + 1 < n && opt.region[j + 1] === opt.region[i]) j++;
      const x0 = X(ts[i]), x1 = X(ts[j]);
      svg.append(svgEl('rect', { x: x0, y: m.t, width: Math.max(1, x1 - x0), height: ih,
        fill: opt.region[i] === 'train' ? P.train : P.test, opacity: 0.07 }));
      i = j + 1;
    }
    const g = svgEl('g', { class: 'grid axis' });
    for (const v of niceTicks(lo, hi, 5)) {
      g.append(svgEl('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
      const tx = svgEl('text', { x: m.l - 7, y: Y(v) + 3.5, 'text-anchor': 'end' }); tx.textContent = fmt(v); g.append(tx);
    }
    svg.append(g);
    const ax = svgEl('g', { class: 'axis' });
    for (let k = 0; k <= 4; k++) {
      const t = t0 + ((t1 - t0) * k) / 4;
      const tx = svgEl('text', { x: X(t), y: H - 8, 'text-anchor': k === 0 ? 'start' : k === 4 ? 'end' : 'middle' });
      tx.textContent = dayOf(t).slice(2); ax.append(tx);
    }
    svg.append(ax);
    if (opt.limit !== null && opt.limit !== undefined && isFinite(opt.limit)) {
      svg.append(svgEl('line', { class: 'limit', x1: m.l, x2: W - m.r, y1: Y(opt.limit), y2: Y(opt.limit) }));
      const lt = svgEl('text', { x: W - m.r, y: Y(opt.limit) - 5, 'text-anchor': 'end', fill: P.bad, 'font-size': 11 });
      lt.textContent = `관리기준 ${fmt(opt.limit)}`; svg.append(lt);
    }
    // 학습 표본과 예측 사이(간격)는 선을 잇지 않는다
    opt.series.forEach(s => {
      let d = '', pen = false;
      s.values.forEach((v, k) => {
        const brk = k > 0 && opt.region[k] !== opt.region[k - 1];
        if (v === null || !isFinite(v) || brk) pen = false;
        if (v === null || !isFinite(v)) return;
        d += (pen ? 'L' : 'M') + X(ts[k]).toFixed(1) + ' ' + Y(v).toFixed(1) + ' '; pen = true;
      });
      svg.append(svgEl('path', { class: 'mark', d, stroke: s.color, 'stroke-width': s.width || 1.6, opacity: s.key === 'y' ? 0.75 : 0.95 }));
    });
    const cross = svgEl('line', { class: 'crosshair', y1: m.t, y2: m.t + ih, opacity: 0 });
    svg.append(cross);
    const hot = svgEl('rect', { class: 'hot', x: m.l, y: m.t, width: iw, height: ih });
    svg.append(hot);
    const tip = h('div', { class: 'tip' });
    host.append(svg, tip);
    hot.addEventListener('mousemove', ev => {
      const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * (W / r.width);
      const t = t0 + ((px - m.l) / iw) * (t1 - t0);
      let k = 0, best = Infinity;
      for (let q = 0; q < n; q++) { const dd = Math.abs(ts[q] - t); if (dd < best) { best = dd; k = q; } }
      cross.setAttribute('x1', X(ts[k])); cross.setAttribute('x2', X(ts[k])); cross.setAttribute('opacity', 1);
      tip.innerHTML = '';
      tip.append(h('div', { class: 'tt' }, `${shortT(opt.t[k])} · ${opt.region[k] === 'train' ? '학습' : '예측'}`),
        ...opt.series.map(s => h('div', { class: 'r' }, h('span', { class: 'sw', style: `background:${s.color}` }),
          `${s.name} ${fmt(s.values[k])}${opt.unit ? ' ' + opt.unit : ''}`)));
      tip.style.opacity = 1;
      tip.style.left = Math.max(4, Math.min(r.width - tip.offsetWidth - 4, (X(ts[k]) / W) * r.width + 12)) + 'px';
      tip.style.top = '8px';
    });
    hot.addEventListener('mouseleave', () => { tip.style.opacity = 0; cross.setAttribute('opacity', 0); });
  };
  draw();
  if (host._ro) host._ro.disconnect();
  host._ro = new ResizeObserver(() => draw()); host._ro.observe(host);
}

function scatterChart(host, opt) {
  const draw = () => {
    host.innerHTML = '';
    const P = PAL();
    const W = Math.max(host.clientWidth || 420, 280), H = Math.min(340, W * 0.8), m = { t: 10, r: 12, b: 34, l: 52 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    let lo = Infinity, hi = -Infinity;
    opt.series.forEach(s => s.x.forEach((xv, i) => { const yv = s.y[i];
      if (xv !== null && yv !== null && isFinite(xv) && isFinite(yv)) { lo = Math.min(lo, xv, yv); hi = Math.max(hi, xv, yv); } }));
    if (!isFinite(lo)) { host.append(h('div', { class: 'empty' }, '표시할 값이 없습니다')); return; }
    const pad = (hi - lo) * 0.05 || 1; lo -= pad; hi += pad;
    const X = v => m.l + ((v - lo) / (hi - lo)) * iw, Y = v => m.t + ih - ((v - lo) / (hi - lo)) * ih;
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
    const g = svgEl('g', { class: 'grid axis' });
    for (const v of niceTicks(lo, hi, 5)) {
      g.append(svgEl('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
      const ty = svgEl('text', { x: m.l - 7, y: Y(v) + 3.5, 'text-anchor': 'end' }); ty.textContent = fmt(v); g.append(ty);
      const tx = svgEl('text', { x: X(v), y: H - 18, 'text-anchor': 'middle' }); tx.textContent = fmt(v); g.append(tx);
    }
    svg.append(g);
    svg.append(svgEl('line', { x1: X(lo), y1: Y(lo), x2: X(hi), y2: Y(hi), stroke: P.ink3, 'stroke-dasharray': '4 4', 'stroke-width': 1 }));
    const lab = svgEl('text', { x: m.l + iw / 2, y: H - 3, 'text-anchor': 'middle', fill: P.ink3, 'font-size': 10.5 });
    lab.textContent = `실측${opt.unit ? ' [' + opt.unit + ']' : ''} → 세로: 예측`; svg.append(lab);
    opt.series.forEach(s => {
      const gg = svgEl('g', { fill: s.color, opacity: 0.45 });
      const step = Math.max(1, Math.floor(s.x.length / 700));
      for (let i = 0; i < s.x.length; i += step) {
        const xv = s.x[i], yv = s.y[i];
        if (xv === null || yv === null || !isFinite(xv) || !isFinite(yv)) continue;
        gg.append(svgEl('circle', { cx: X(xv).toFixed(1), cy: Y(yv).toFixed(1), r: 1.8 }));
      }
      svg.append(gg);
    });
    host.append(svg, h('div', { class: 'legend' }, opt.series.map(s => h('span', {}, h('i', { style: `background:${s.color}` }), s.name))));
  };
  draw();
  if (host._ro) host._ro.disconnect();
  host._ro = new ResizeObserver(() => draw()); host._ro.observe(host);
}

// ── 화면 진입 ───────────────────────────────────────────────────────────
function ezGo(screen) {
  if (screen === 'ez-vars') ezRenderVars();
  else if (screen === 'ez-period') ezRenderPeriod();
  else if (screen === 'ez-model') ezRenderModel();
  else if (screen === 'ez-result' && EZ.result) ezRenderResult();
  go(screen);
}

// ── 부팅 ────────────────────────────────────────────────────────────────
$$('#nav button[data-screen^="ez-"]').forEach(b => b.addEventListener('click', () => ezGo(b.dataset.screen)));
ezSetupDrop();
ezLoadDatasets().catch(e => toast(e.message, 'bad'));
