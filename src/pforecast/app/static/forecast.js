/* 6. 미래 예측 — 계획(앞으로의 입력 x)을 넣고 예측한다. 정답이 없으므로 채점 대신
   설비 상태 변동 폭 · 학습 범위 밖 표시 · 관리기준 초과 시간을 보여준다.
   변수·모델·연결은 1~4단계에서 정한 것을 그대로 쓴다 (ezBody). */

'use strict';

const FC_MODES = [['keep', '그대로'], ['scale', '× 배율'], ['add', '+ 더하기'], ['set', '= 고정값']];

function fcState() {
  if (!EZ.fc) EZ.fc = { history: 'all', src: 'make', plan: null, check: null, preview: null, plans: null,
                        gen: { base_days: 7, horizon_days: 14, step: '1h', adjust: {} },
                        band: true, result: null, running: null, error: null };
  return EZ.fc;
}

function fcBody(extra) {
  const F = fcState();
  const b = ezBody(true);
  delete b.train; delete b.test;
  b.history = F.history === 'train' && EZ.train[0] ? EZ.train : null;
  b.band = F.band;
  return Object.assign(b, extra || {});
}

async function fcRender() {
  const F = fcState(), body = $('#ez-forecast-body');
  body.innerHTML = '';
  if (!EZ.profile || !EZ.target || !EZ.features.length) {
    body.append(h('div', { class: 'card' }, h('div', { class: 'empty' }, '1~4단계에서 데이터·변수·모델을 먼저 정하세요.')));
    return;
  }
  const physics = EZ.mode === 'physics' && EZ.model;
  if (!EZ.result) body.append(h('div', { class: 'note warn' }, h('b', {}, '검증을 먼저 해 보길 권합니다. '),
    '5단계(검증 결과)에서 학습 때 없던 조건에서도 맞는지 확인한 모델이어야 미래 예측을 믿을 수 있습니다.'));
  body.append(h('div', { class: 'pickbar' },
    h('div', {}, h('span', { class: 'k' }, '예측할 값'), h('b', {}, EZ.target)),
    h('div', {}, h('span', { class: 'k' }, '입력'), h('b', {}, `${EZ.features.length}개`)),
    h('div', {}, h('span', { class: 'k' }, '모델'), h('b', {}, physics ? EZ.model.split('/').slice(-2).join('/') + ' + ML' : 'ML 만')),
    h('button', { class: 'btn ghost sm', onclick: () => ezGo('ez-model') }, '바꾸기')));

  // ① 학습에 쓸 과거
  const histCard = h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h3', {}, '① 학습에 쓸 과거')),
    h('div', { class: 'choices' },
      fcChoice(F.history === 'all', '데이터 전체 (권장)', `${(EZ.profile.t_start || '').slice(0, 10)} ~ ${(EZ.profile.t_end || '').slice(0, 10)}. 운전 조건이 다양할수록 물리 파라미터가 잘 갈립니다.`,
        () => { F.history = 'all'; fcRender(); }),
      fcChoice(F.history === 'train', '3단계 학습 기간만', EZ.train[0] ? `${EZ.train[0]} ~ ${EZ.train[1]}. 검증한 모델과 똑같이 맞춥니다.` : '3단계에서 기간을 먼저 고르세요.',
        () => { if (EZ.train[0]) { F.history = 'train'; fcRender(); } })),
    h('div', { class: 'sub', style: 'margin:0' }, '최근 몇 주만으로 다시 맞추지 마세요 — 운전 다양성이 없어 파라미터 귀속이 바뀌고, 계측 드리프트까지 흡수해 예측이 오히려 나빠집니다.'));
  body.append(histCard);

  // ② 계획
  const planCard = h('div', { class: 'card', id: 'fc-plan' });
  body.append(planCard);
  fcRenderPlan(planCard);

  // ③ 실행
  const run = h('button', { class: 'btn lg', id: 'fc-run', disabled: !(F.check && F.check.ok) || !!F.running, onclick: fcRun }, '▶ 미래 예측 실행');
  body.append(h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h3', {}, '③ 실행')),
    h('div', { class: 'optrow' },
      h('label', { class: 'row' }, `${EZ.target} 관리기준`,
        h('input', { type: 'number', value: EZ.limit, placeholder: '없음', style: 'width:110px', oninput: e => (EZ.limit = e.target.value) }),
        h('span', { class: 'sub', style: 'margin:0' }, '넣으면 초과 예상 시간을 셉니다')),
      physics ? h('label', { class: 'row' },
        h('input', { type: 'checkbox', checked: F.band, onchange: e => (F.band = e.target.checked) }),
        '설비 상태 변동 폭 (과거를 네 토막으로 다시 보정 — 2~4분 더)') : null),
    h('div', { class: 'actions' }, run,
      h('span', { class: 'sub', style: 'margin:0' }, F.plan && !F.check ? '저장한 계획을 다시 검사하는 중…'
        : !(F.check && F.check.ok) ? '계획을 먼저 정하세요' : physics ? '물리 보정 1~3분 + 상태 폭' : '수 초')),
    h('div', { id: 'fc-progress' })));
  fcDrawProgress();

  const res = h('div', { id: 'fc-result' });
  body.append(res);
  if (F.result) fcRenderResult();
}

function fcChoice(on, title, desc, onclick) {
  return h('button', { class: 'choice' + (on ? ' on' : ''), onclick }, h('div', { class: 't' }, title), h('div', { class: 'd' }, desc));
}

// ── ② 계획 ──────────────────────────────────────────────────────────────
function fcRenderPlan(card) {
  const F = fcState();
  card.innerHTML = '';
  card.append(h('div', { class: 'card-head' }, h('h3', {}, '② 계획 — 앞으로의 입력 값'),
    h('span', { class: 'sub' }, '학습 CSV 와 같은 컬럼 이름·단위. y 는 없어도 됩니다')));
  const seg = h('div', { class: 'seg', role: 'tablist', style: 'margin-bottom:12px' },
    [['make', '최근 패턴으로 만들기'], ['file', '계획 CSV 올리기']].map(([k, label]) =>
      h('button', { class: F.src === k ? 'on' : '', onclick: () => { F.src = k; fcRenderPlan(card); } }, label)));
  card.append(seg);
  if (F.src === 'make') card.append(fcMaker());
  else card.append(fcFilePicker());
  const chk = h('div', { id: 'fc-check' });
  card.append(chk);
  fcDrawCheck();
}

function fcMaker() {
  const F = fcState(), G = F.gen;
  const ranges = {};
  ((F.check && F.check.ranges) || []).forEach(r => (ranges[r.column] = r));
  const num = (key, w, attrs) => h('input', Object.assign({ type: 'number', value: G[key], style: `width:${w}px`,
    oninput: e => (G[key] = +e.target.value) }, attrs || {}));
  const rows = EZ.features.map(c => {
    const a = G.adjust[c] = G.adjust[c] || { mode: 'keep', value: '' };
    const val = h('input', { type: 'number', step: 'any', value: a.value, style: 'width:100px', disabled: a.mode === 'keep',
      placeholder: a.mode === 'scale' ? '예: 1.1' : a.mode === 'add' ? '예: 2' : a.mode === 'set' ? '값' : '',
      oninput: e => (a.value = e.target.value) });
    const r = ranges[c];
    return h('tr', {},
      h('td', { class: 'name' }, c),
      h('td', {}, h('select', { onchange: e => { a.mode = e.target.value; val.disabled = a.mode === 'keep'; if (a.mode === 'keep') { a.value = ''; val.value = ''; } } },
        FC_MODES.map(([k, l]) => h('option', { value: k, selected: a.mode === k }, l)))),
      h('td', {}, val),
      h('td', { class: 'mono sub' }, r ? `${fmt(r.train_lo)} ~ ${fmt(r.train_hi)}${r.frozen ? ' (고정)' : ''}` : ''));
  });
  return h('div', {},
    h('div', { class: 'fcgen' },
      h('label', { class: 'row' }, '최근', num('base_days', 70, { min: 1 }), '일 패턴을'),
      h('label', { class: 'row' }, num('horizon_days', 70, { min: 1 }), '일 동안 반복,'),
      h('label', { class: 'row' }, '간격', h('select', { onchange: e => (G.step = e.target.value) },
        [['5min', '5분'], ['15min', '15분'], ['1h', '1시간'], ['1D', '1일']].map(([k, l]) => h('option', { value: k, selected: G.step === k }, l))))),
    h('div', { class: 'sub', style: 'margin:6px 0 10px' }, '바꿀 입력만 조정하세요 (예: 가동률 × 1.1, 장비 대수 = 증설 후 대수). 값은 CSV 와 같은 단위입니다. 만든 계획은 plans/ 에 CSV 로 저장되어 엑셀에서 고친 뒤 다시 올릴 수 있습니다.'),
    h('div', { class: 'tablewrap' }, h('table', { class: 'map' },
      h('thead', {}, h('tr', {}, h('th', {}, '입력 컬럼'), h('th', {}, '조정'), h('th', {}, '값'), h('th', {}, '과거 범위'))),
      h('tbody', {}, rows))),
    h('div', { class: 'actions' }, h('button', { class: 'btn', onclick: fcMake }, '계획 만들기')));
}

function fcFilePicker() {
  const F = fcState();
  const input = h('input', { type: 'file', accept: '.csv,.txt,text/csv', hidden: true, onchange: e => e.target.files[0] && fcUpload(e.target.files[0]) });
  const list = h('div', { class: 'chips', id: 'fc-plans' }, h('span', { class: 'sub' }, '불러오는 중…'));
  (async () => {
    try { F.plans = (await api('/api/forecast/plans')).plans; } catch (_) { F.plans = []; }
    list.innerHTML = '';
    if (!F.plans.length) list.append(h('span', { class: 'sub' }, '아직 없습니다.'));
    F.plans.forEach(p => list.append(h('button', { class: 'chip' + (F.plan === p.path ? ' on' : ''), title: p.path, onclick: () => fcPick(p.path) },
      p.path.split('/').slice(-2).join('/'))));
  })();
  return h('div', {},
    h('label', { class: 'dropzone small', tabindex: 0 }, input,
      h('div', { class: 'dz-title' }, '계획 CSV 를 고르세요 ', h('u', {}, '(클릭)')),
      h('div', { class: 'dz-sub' }, '첫 컬럼은 시각, 나머지는 입력 컬럼. 양식이 필요하면 ‘최근 패턴으로 만들기’로 하나 만들어 받으세요.')),
    h('div', { class: 'k', style: 'margin-top:12px' }, '작업 폴더의 계획 (plans/ · uploads/)'), list);
}

async function fcUpload(file) {
  const r = await fetch('/api/upload?name=' + encodeURIComponent(file.name), { method: 'POST', body: file });
  const j = await r.json().catch(() => ({ error: '응답을 읽지 못했습니다' }));
  if (!r.ok || j.error) { toast(j.error || '업로드 실패', 'bad'); return; }
  toast(`올렸습니다: ${j.path}`);
  fcPick(j.path);
}

async function fcPick(path) {
  const F = fcState();
  F.plan = path; F.check = { busy: true };
  fcDrawCheck();
  try {
    const r = await api('/api/forecast/check', fcBody({ plan: path }));
    F.check = r.check; F.preview = r.preview; F.planLog = r.log;
  } catch (e) { F.check = { ok: false, errors: [e.message], warnings: [] }; }
  fcRender();
}

async function fcMake() {
  const F = fcState(), G = F.gen;
  const adjust = Object.entries(G.adjust).filter(([, a]) => a.mode !== 'keep' && a.value !== '' && isFinite(+a.value))
    .map(([column, a]) => ({ column, mode: a.mode, value: +a.value }));
  F.check = { busy: true }; fcDrawCheck();
  try {
    const r = await api('/api/forecast/make', fcBody({ base_days: G.base_days, horizon_days: G.horizon_days, step: G.step, adjust,
      plan_name: `${(EZ.project && EZ.project.name) || EZ.csv.split('/').pop().replace(/\.[^.]+$/, '')}_계획` }));
    F.plan = r.path; F.check = r.check; F.preview = r.preview; F.planLog = [];
    toast(`계획을 만들었습니다: ${r.path}`);
  } catch (e) { F.check = { ok: false, errors: [e.message], warnings: [] }; }
  fcRender();
}

function fcDrawCheck() {
  const F = fcState(), box = $('#fc-check');
  if (!box) return;
  box.innerHTML = '';
  const c = F.check;
  if (!c) return;
  if (c.busy) { box.append(h('div', { class: 'sub' }, '계획을 읽고 검사하는 중…'), h('div', { class: 'progress' }, h('i'))); return; }
  box.append(h('div', { class: 'k', style: 'margin-top:14px' }, '선택한 계획'),
    F.plan ? h('div', { class: 'row', style: 'gap:10px;flex-wrap:wrap' }, h('span', { class: 'mono' }, F.plan),
      h('a', { class: 'btn ghost sm', href: '/' + F.plan, download: F.plan.split('/').pop() }, '⭳ CSV 받기 (엑셀에서 고치기)')) : null);
  if (c.n_rows) box.append(h('div', { class: 'tiles', style: 'margin-top:10px' },
    tile('기간', (c.start || '').slice(0, 10), '', `~ ${(c.end || '').slice(0, 10)}`),
    tile('행', c.n_rows.toLocaleString('ko-KR'), '', c.step_minutes ? `간격 ${ezInterval(c.step_minutes * 60)}` : ''),
    tile('학습 범위 밖', fmt(c.outside_pct, 0), '%', '과거에 없던 운전조건 비율', c.outside_pct > 0 ? 'warn' : 'good')));
  (c.errors || []).forEach(e => box.append(h('div', { class: 'note bad' }, e)));
  (c.warnings || []).forEach(w => box.append(h('div', { class: 'note warn' }, w)));
  if (c.ranges && c.ranges.length) box.append(h('details', { class: 'fold', style: 'margin-top:10px', open: c.outside_pct > 0 },
    h('summary', {}, h('b', {}, '입력별 범위'), h('span', { class: 'sub' }, ' 과거(학습) 범위 vs 계획 범위')),
    h('div', { class: 'tablewrap' }, h('table', { class: 'map' },
      h('thead', {}, h('tr', {}, h('th', {}, '입력'), h('th', {}, '과거'), h('th', {}, '계획'), h('th', {}, '밖 %'), h('th', {}, '계획 모양'))),
      h('tbody', {}, c.ranges.map(r => h('tr', {},
        h('td', { class: 'name' }, r.column, r.frozen ? h('span', { class: 'badge warn' }, '과거 고정') : null),
        h('td', { class: 'mono' }, `${fmt(r.train_lo)} ~ ${fmt(r.train_hi)}`),
        h('td', { class: 'mono' }, r.test_lo === null ? '—' : `${fmt(r.test_lo)} ~ ${fmt(r.test_hi)}`),
        h('td', { class: 'num' + (r.outside_pct > 0 ? ' warntext' : '') }, fmt(r.outside_pct, 0)),
        h('td', {}, F.preview && F.preview.columns[r.column] ? sparkline(F.preview.columns[r.column]) : ''))))))));
  if (F.planLog && F.planLog.length) box.append(h('details', { class: 'fold', style: 'margin-top:6px' },
    h('summary', {}, h('b', {}, '계획 파일 정리 내역')), h('ul', { class: 'log' }, F.planLog.map(t => h('li', { html: mdBold(t) })))));
}

// ── ③ 실행 ──────────────────────────────────────────────────────────────
function fcDrawProgress() {
  const F = fcState(), box = $('#fc-progress'), run = $('#fc-run');
  if (run) run.disabled = !!F.running || !(F.check && F.check.ok);
  if (!box) return;
  box.innerHTML = '';
  if (F.running) box.append(h('div', { class: 'card', style: 'margin-top:10px' },
    h('div', { class: 'card-head' }, h('h3', {}, '예측 중'), h('span', { class: 'sub' }, `${Math.round((Date.now() - F.running.t0) / 1000)}초`)),
    h('div', { class: 'progress' }, h('i')), h('div', { class: 'sub', style: 'margin:8px 0 0' }, F.running.msg)));
  else if (F.error) box.append(h('div', { class: 'note bad', style: 'margin-top:10px' }, h('b', {}, '예측하지 못했습니다. '), F.error));
}

async function fcRun() {
  const F = fcState();
  if (F.running || !F.plan) return;
  F.error = null; F.running = { t0: Date.now(), msg: '시작하는 중…' };
  fcDrawProgress();
  const tick = setInterval(fcDrawProgress, 1000);
  try {
    const { job } = await api('/api/forecast/run', fcBody({ plan: F.plan }));
    for (;;) {
      await new Promise(r => setTimeout(r, 1000));
      const j = await api('/api/job?id=' + job.id);
      F.running.msg = j.message || j.status;
      if (j.status === 'running') continue;
      if (j.status === 'error') throw new Error(j.error);
      F.result = j.result; break;
    }
    F.running = null; clearInterval(tick); fcDrawProgress();
    fcRenderResult();
    $('#fc-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
    if (typeof projAutoSave === 'function') projAutoSave();
  } catch (e) {
    F.running = null; clearInterval(tick); F.error = e.message; fcDrawProgress();
  }
}

// ── 결과 ────────────────────────────────────────────────────────────────
function fcRenderResult() {
  const F = fcState(), r = F.result, box = $('#fc-result');
  if (!box || !r) return;
  box.innerHTML = '';
  const P = PAL(), u = r.target.unit || '';
  const phys = r.summary.find(s => s.kind === 'physics');
  const main = phys || r.summary[0];
  const tiles = [
    tile('예측 기간', (r.plan.start || '').slice(0, 10), '', `~ ${(r.plan.end || '').slice(0, 10)} · ${r.plan.n_rows.toLocaleString('ko-KR')}행`),
    tile(`${main.name} 평균`, fmt(main.mean), u, `최대 ${fmt(main.max)} · 95% ${fmt(main.p95)}`),
  ];
  if (r.band) tiles.push(tile('설비 상태 변동 폭', `${fmt(r.band.mean_lo)}~${fmt(r.band.mean_hi)}`, u, '평균의 범위 (과거 상태를 그대로 가져왔을 때)'));
  if (r.limit !== null && r.limit !== undefined) {
    const ex = main.exceed_hours;
    tiles.push(tile('관리기준 초과 예상', String(ex), '시간', `${main.name} · 1시간 평균 > ${fmt(r.limit)}`, ex > 0 ? 'bad' : 'good'));
    if (r.band) tiles.push(tile('초과 가능성', String(r.band.exceed_possible_hours), '시간',
      `상태 폭 상한 기준 · 확실(하한도 초과) ${r.band.exceed_certain_hours}시간`, r.band.exceed_possible_hours > 0 ? 'warn' : 'good'));
  }
  tiles.push(tile('학습 범위 밖', fmt(r.outside_pct, 0), '%', '과거에 없던 운전조건 — ML 은 믿기 어려움', r.outside_pct > 0 ? 'warn' : 'good'));
  box.append(h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h3', {}, '미래 예측 결과'),
      h('span', { class: 'sub' }, `학습 ${r.history.start.slice(0, 10)} ~ ${r.history.end.slice(0, 10)} (${r.history.n_rows.toLocaleString('ko-KR')}행) · ${fmt(r.seconds, 0)}초`)),
    h('div', { class: 'tiles' }, tiles)));

  // 차트: 최근 실측 + 예측 + 상태 폭 + 학습 범위 밖 음영
  F.hidden = F.hidden || {};
  const names = Object.keys(r.series.models);
  const colors = {};
  r.summary.forEach(s => (colors[s.name] = colorOf(s.kind, P)));
  const legend = h('div', { class: 'legend' },
    h('span', {}, h('i', { class: 'sw', style: `background:${P.y}` }), `실측 (최근, 1시간 평균)`),
    ...names.map(n => h('label', { class: 'lg' + (F.hidden[n] ? ' off' : '') },
      h('input', { type: 'checkbox', checked: !F.hidden[n], onchange: e => { F.hidden[n] = !e.target.checked; fcRenderResult(); } }),
      h('i', { class: 'sw', style: `background:${colors[n]}` }), n)),
    r.series.band_lo ? h('span', {}, h('i', { class: 'sw band', style: `background:${P.physics}` }), '물리모델 상태 폭') : null,
    r.outside_pct > 0 ? h('span', {}, h('i', { class: 'sw', style: `background:${P.bad};opacity:.25` }), '학습 범위 밖') : null);
  const host = h('div', { class: 'chart' });
  box.append(h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h3', {}, `${r.target.column}${u ? ` [${u}]` : ''}`),
      h('span', { class: 'sub' }, '세로선 왼쪽이 과거 실측, 오른쪽이 계획에 대한 예측')),
    legend, host));
  fcChart(host, r, colors);

  if (r.notes.length) box.append(h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', {}, '읽을 때 주의')),
    ...r.notes.map(n => h('div', { class: 'note warn', html: mdBold(n) }))));

  box.append(h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', {}, '모델별 요약')),
    h('div', { class: 'tablewrap' }, h('table', { class: 'map' },
      h('thead', {}, h('tr', {}, h('th', {}, '모델'), h('th', {}, `평균 ${u}`), h('th', {}, `95% 분위`), h('th', {}, `최대`),
        r.limit !== null && r.limit !== undefined ? h('th', {}, '초과 시간') : null)),
      h('tbody', {}, r.summary.map(s => h('tr', {},
        h('td', { class: 'name' }, h('i', { class: 'sw', style: `background:${colors[s.name]}` }), ' ', s.name),
        h('td', { class: 'num' }, fmt(s.mean)), h('td', { class: 'num' }, fmt(s.p95)), h('td', { class: 'num' }, fmt(s.max)),
        r.limit !== null && r.limit !== undefined ? h('td', { class: 'num' }, `${s.exceed_hours} / ${s.hours}`) : null)))))));

  if (r.drift && r.drift.length) box.append(h('details', { class: 'card fold' },
    h('summary', {}, h('b', {}, '설비 상태 변동 — 과거 토막별 파라미터'), h('span', { class: 'sub' }, ' 상태 폭의 근거. 예측에 쓰지 않고 진단으로만 봅니다')),
    h('div', { class: 'tablewrap', style: 'margin-top:8px' }, h('table', { class: 'map' },
      h('thead', {}, h('tr', {}, h('th', {}, '파라미터'), ...r.drift[0].values.map((_, i) => h('th', {}, `토막 ${i + 1}`)), h('th', {}, '추세'))),
      h('tbody', {}, r.drift.map(d => h('tr', {}, h('td', { class: 'mono' }, d.name),
        ...d.values.map(v => h('td', { class: 'num' }, fmt(v))), h('td', {}, d.trend))))))));

  if (r.explain && (r.explain.physics || (r.explain.ml || []).length)) box.append(ezExplainCard(r));
  box.append(h('div', { class: 'actions' },
    h('a', { class: 'btn', href: '/' + r.files.forecast, download: r.files.forecast.split('/').pop() }, '⭳ 예측 결과 CSV'),
    h('button', { class: 'btn ghost', onclick: () => { F.src = 'make'; fcRender(); $('#fc-plan').scrollIntoView({ behavior: 'smooth' }); } }, '← 계획 바꿔 다시')));
}

function fcChart(host, r, colors) {
  const F = fcState();
  const draw = () => {
    host.innerHTML = '';
    const P = PAL();
    const W = Math.max(host.clientWidth || 700, 320), H = 320, m = { t: 14, r: 14, b: 26, l: 52 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const th = r.history_tail.t.map(toMs), tf = r.series.t.map(toMs);
    const all = th.concat(tf);
    const t0 = Math.min(...all), t1 = Math.max(...all);
    const X = t => m.l + ((t - t0) / ((t1 - t0) || 1)) * iw;
    let lo = Infinity, hi = -Infinity;
    const scan = arr => (arr || []).forEach(v => { if (v !== null && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });
    scan(r.history_tail.y);
    Object.entries(r.series.models).forEach(([n, v]) => { if (!F.hidden[n]) scan(v); });
    scan(r.series.band_lo); scan(r.series.band_hi);
    if (r.limit !== null && r.limit !== undefined) { lo = Math.min(lo, r.limit); hi = Math.max(hi, r.limit); }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    const pad = (hi - lo) * 0.07 || 1; lo -= pad; hi += pad;
    const Y = v => m.t + ih - ((v - lo) / (hi - lo)) * ih;
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
    // 학습 범위 밖 음영
    const out = r.series.outside;
    let i = 0;
    while (i < tf.length) {
      if (!out[i]) { i++; continue; }
      let j = i; while (j + 1 < tf.length && out[j + 1]) j++;
      const x0 = X(tf[i]), x1 = X(tf[Math.min(j + 1, tf.length - 1)]);
      svg.append(svgEl('rect', { x: x0, y: m.t, width: Math.max(1, x1 - x0), height: ih, fill: P.bad, opacity: 0.08 }));
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
    // 지금 (과거 | 미래 경계)
    if (tf.length) {
      const xn = X(tf[0]);
      svg.append(svgEl('line', { x1: xn, x2: xn, y1: m.t, y2: m.t + ih, stroke: P.ink3, 'stroke-dasharray': '4 3' }));
      const tn = svgEl('text', { x: xn + 4, y: m.t + 11, fill: P.ink3, 'font-size': 11 }); tn.textContent = '예측 →'; svg.append(tn);
    }
    if (r.limit !== null && r.limit !== undefined) {
      svg.append(svgEl('line', { class: 'limit', x1: m.l, x2: W - m.r, y1: Y(r.limit), y2: Y(r.limit) }));
      const lt = svgEl('text', { x: W - m.r, y: Y(r.limit) - 5, 'text-anchor': 'end', fill: P.bad, 'font-size': 11 });
      lt.textContent = `관리기준 ${fmt(r.limit)}`; svg.append(lt);
    }
    // 상태 폭 (구간이 끊기면 따로 그린다)
    if (r.series.band_lo && !F.hidden['물리모델']) {
      const lo_ = r.series.band_lo, hi_ = r.series.band_hi;
      let k = 0;
      while (k < tf.length) {
        if (lo_[k] === null || hi_[k] === null) { k++; continue; }
        let j = k; while (j + 1 < tf.length && lo_[j + 1] !== null && hi_[j + 1] !== null) j++;
        let d = '';
        for (let q = k; q <= j; q++) d += (q === k ? 'M' : 'L') + X(tf[q]).toFixed(1) + ' ' + Y(hi_[q]).toFixed(1) + ' ';
        for (let q = j; q >= k; q--) d += 'L' + X(tf[q]).toFixed(1) + ' ' + Y(lo_[q]).toFixed(1) + ' ';
        svg.append(svgEl('path', { d: d + 'Z', fill: P.physics, opacity: 0.16, stroke: 'none' }));
        k = j + 1;
      }
    }
    const line = (ts, vals, color, width, dash) => {
      let d = '', pen = false;
      vals.forEach((v, k) => {
        if (v === null || !isFinite(v)) { pen = false; return; }
        d += (pen ? 'L' : 'M') + X(ts[k]).toFixed(1) + ' ' + Y(v).toFixed(1) + ' '; pen = true;
      });
      const attrs = { class: 'mark', d, stroke: color, 'stroke-width': width, fill: 'none' };
      if (dash) attrs['stroke-dasharray'] = dash;
      svg.append(svgEl('path', attrs));
    };
    line(th, r.history_tail.y, P.y, 1.4);
    Object.entries(r.series.models).forEach(([n, v]) => { if (!F.hidden[n]) line(tf, v, colors[n], n === '물리모델' ? 2 : 1.5); });

    const cross = svgEl('line', { class: 'crosshair', y1: m.t, y2: m.t + ih, opacity: 0 });
    svg.append(cross);
    const hot = svgEl('rect', { class: 'hot', x: m.l, y: m.t, width: iw, height: ih });
    svg.append(hot);
    const tip = h('div', { class: 'tip' });
    host.append(svg, tip);
    const nearest = (ts, t) => { let k = -1, best = Infinity; ts.forEach((x, q) => { const dd = Math.abs(x - t); if (dd < best) { best = dd; k = q; } }); return [k, best]; };
    hot.addEventListener('mousemove', ev => {
      const rc = svg.getBoundingClientRect(), px = (ev.clientX - rc.left) * (W / rc.width);
      const t = t0 + ((px - m.l) / iw) * (t1 - t0);
      const [kh, dh] = nearest(th, t), [kf, df] = nearest(tf, t);
      const fut = df <= dh;
      const tx = fut ? tf[kf] : th[kh];
      cross.setAttribute('x1', X(tx)); cross.setAttribute('x2', X(tx)); cross.setAttribute('opacity', 1);
      tip.innerHTML = '';
      const U = r.target.unit ? ' ' + r.target.unit : '';
      if (fut) {
        tip.append(h('div', { class: 'tt' }, `${shortT(r.series.t[kf])} · 예측${r.series.outside[kf] ? ' · 학습 범위 밖' : ''}`),
          ...Object.entries(r.series.models).filter(([n]) => !F.hidden[n]).map(([n, v]) =>
            h('div', { class: 'r' }, h('span', { class: 'sw', style: `background:${colors[n]}` }), `${n} ${fmt(v[kf])}${U}`)),
          r.series.band_lo ? h('div', { class: 'r' }, `상태 폭 ${fmt(r.series.band_lo[kf])} ~ ${fmt(r.series.band_hi[kf])}${U}`) : null);
      } else {
        tip.append(h('div', { class: 'tt' }, `${shortT(r.history_tail.t[kh])} · 과거 실측`),
          h('div', { class: 'r' }, h('span', { class: 'sw', style: `background:${P.y}` }), `실측 ${fmt(r.history_tail.y[kh])}${U}`));
      }
      tip.style.opacity = 1;
      tip.style.left = Math.max(4, Math.min(rc.width - tip.offsetWidth - 4, (X(tx) / W) * rc.width + 12)) + 'px';
      tip.style.top = '8px';
    });
    hot.addEventListener('mouseleave', () => { tip.style.opacity = 0; cross.setAttribute('opacity', 0); });
  };
  draw();
  if (host._ro) host._ro.disconnect();
  host._ro = new ResizeObserver(() => draw()); host._ro.observe(host);
}
