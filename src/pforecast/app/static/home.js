/* 시작 화면과 프로젝트(설정 저장·열기).
   계통이 무엇이든 흐름은 같다: 데이터 → 변수 → 기간 → 모델·연결 → 검증 → 미래 예측.
   프로젝트는 그 흐름에서 고른 것을 이름 붙여 projects/ 에 저장한다 (결과 포함). */

'use strict';

// 저장하는 화면 상태 — 이것만 있으면 같은 화면을 다시 만들 수 있다
const PROJ_KEYS = ['csv', 'target', 'features', 'units', 'train', 'test', 'embargo', 'mode', 'model',
                   'fmap', 'tmap', 'extra', 'params', 'band', 'limit', 'closures'];

async function homeRender() {
  const cards = $('#home-cards');
  cards.innerHTML = '';
  const card = (icon, title, desc, action, onclick, primary) => h('button', { class: 'homecard' + (primary ? ' primary' : ''), onclick },
    h('div', { class: 'hc-icon', 'aria-hidden': 'true' }, icon), h('div', { class: 'hc-t' }, title),
    h('div', { class: 'hc-d' }, desc), h('div', { class: 'hc-a' }, action));
  cards.append(
    card('⇪', '데이터로 예측 모델 만들기', 'CSV 하나로 시작합니다. 계통 모델이 있으면 연결해 물리모델 + ML 을 같은 조건으로 검증하고, 없으면 ML 만으로도 비교합니다.',
      '1단계부터 →', () => ezGo('ez-data'), true),
    card('⊞', '계통 모델 만들기·고치기', '장비(컴포넌트)를 골라 출구 → 입구로 연결합니다. 코드 없이, 고칠 때마다 방정식 개수·연결·설계점 풀이를 검사합니다.',
      '모델 만들기 →', () => { go('builder'); if (typeof mbEnter === 'function') mbEnter(); }),
    card('◎', '운전 조건 바꿔 보기', '모델만으로 가동률·설정값을 움직여 결과를 바로 풉니다. 과거 데이터에 없던 조건도 지배방정식으로 답합니다.',
      '모델 구조 · 시나리오 →', () => go('model')));
  api('/api/workspace').then(ws => { $('#home-version').textContent = `pforecast 버전 ${ws.version || '?'} · 작업 폴더 ${ws.root}`; })
    .catch(() => {});
  const box = $('#home-projects');
  let items = [];
  try { items = (await api('/api/project/list')).projects; } catch (e) { box.innerHTML = ''; box.append(h('div', { class: 'note bad' }, e.message)); return; }
  box.innerHTML = '';
  if (!items.length) {
    box.append(h('div', { class: 'empty' }, '아직 없습니다. 검증이나 미래 예측을 실행하면 자동으로 저장됩니다.'));
    return;
  }
  box.append(h('div', { class: 'tablewrap' }, h('table', { class: 'map projlist' },
    h('thead', {}, h('tr', {}, h('th', {}, '이름'), h('th', {}, '데이터'), h('th', {}, '예측할 값'), h('th', {}, '모델'), h('th', {}, '저장'), h('th', {}, ''))),
    h('tbody', {}, items.map(p => h('tr', {},
      h('td', { class: 'name' }, p.name, p.error ? h('div', { class: 'desc bad' }, p.error) : null),
      h('td', { class: 'mono' }, p.csv ? p.csv.split('/').pop() : '—'),
      h('td', { class: 'mono' }, p.target || '—'),
      h('td', { class: 'mono' }, p.model ? p.model.split('/').slice(-2).join('/') : 'ML 만'),
      h('td', { class: 'sub', style: 'margin:0' }, (p.saved_at || '').replace('T', ' ').slice(0, 16)),
      h('td', { class: 'row', style: 'gap:6px;justify-content:flex-end' },
        h('button', { class: 'btn ghost sm', disabled: !!p.error, onclick: () => projOpen(p.path) }, '열기'),
        h('button', { class: 'btn sm', disabled: !!p.error, onclick: () => projOpen(p.path, 'ez-forecast') }, '미래 예측'))))))));
}

function projSnapshot() {
  const ui = {};
  PROJ_KEYS.forEach(k => (ui[k] = EZ[k]));
  const cur = $$('.screen.active')[0];
  ui.step = cur ? cur.id.replace('screen-', '') : 'ez-data';
  if (EZ.fc) {
    const { history, src, plan, gen, band } = EZ.fc;
    ui.fc = { history, src, plan, gen, band };
  }
  let run = {};
  try { if (EZ.csv && EZ.target && EZ.profile) run = ezBody(true); } catch (_) { run = {}; }
  return { run, ui };
}

function projDefaultName() {
  const stem = (EZ.csv || 'project').split('/').pop().replace(/\.[^.]+$/, '');
  return EZ.target ? `${stem}_${EZ.target}` : stem;
}

async function projSave(name, quiet) {
  if (!EZ.csv) { toast('데이터를 먼저 고르세요', 'bad'); return; }
  const nm = name || (EZ.project && EZ.project.name) || projDefaultName();
  const { run, ui } = projSnapshot();
  try {
    const r = await api('/api/project/save', { name: nm, run, ui, results: EZ.result || null,
      forecast_results: (EZ.fc && EZ.fc.result) || null });
    EZ.project = { name: nm, path: r.path, savedAt: new Date() };
    projChip();
    if (!quiet) toast(`프로젝트를 저장했습니다: ${r.path}`);
  } catch (e) { toast(e.message, 'bad'); }
}

// 실행이 끝날 때마다 저장한다 — 다시 열면 결과까지 바로 보인다
function projAutoSave() { return projSave(null, true); }

function projChip() {
  const chip = $('#proj-chip');
  if (!chip) return;
  chip.innerHTML = '';
  chip.hidden = !EZ.csv;
  if (!EZ.csv) return;
  const P = EZ.project;
  const t = P && P.savedAt ? `${pad2(P.savedAt.getHours())}:${pad2(P.savedAt.getMinutes())} 저장` : '저장 안 됨';
  chip.append(
    h('span', { class: 'pc-k' }, '프로젝트'),
    h('button', { class: 'pc-name', title: '이름 바꿔 저장', onclick: () => {
      const v = prompt('프로젝트 이름', (P && P.name) || projDefaultName());
      if (v && v.trim()) { EZ.project = null; projSave(v.trim()); }
    } }, (P && P.name) || projDefaultName()),
    h('span', { class: 'pc-t' }, t),
    h('button', { class: 'btn ghost sm', onclick: () => projSave() }, '저장'));
}

async function projOpen(path, screen) {
  let doc;
  try { doc = await api('/api/project/load', { path }); } catch (e) { toast(e.message, 'bad'); return; }
  const ui = doc.ui || {};
  if ((doc.missing || []).length) toast('없는 파일: ' + doc.missing.join(', ') + ' — 작업 폴더를 확인하세요', 'bad');
  if (!ui.csv || (doc.missing || []).includes(ui.csv)) { ezGo('ez-data'); return; }
  await ezSelectCsv(ui.csv);
  if (!EZ.profile) return;
  PROJ_KEYS.forEach(k => { if (ui[k] !== undefined && ui[k] !== null) EZ[k] = ui[k]; });
  if ((doc.missing || []).includes(EZ.model)) ezUseModel(null);
  EZ.catalog = null; EZ.modelsList = null;
  EZ.result = doc.results || null;
  EZ.fc = null;
  const F = fcState();
  Object.assign(F, ui.fc || {}, { result: doc.forecast_results || null, check: null });
  EZ.project = { name: doc.name, path, savedAt: doc.saved_at ? new Date(doc.saved_at) : null };
  unlock(EZ.result ? 'ez-result' : EZ.train && EZ.train[0] ? 'ez-model' : EZ.target ? 'ez-period' : 'ez-vars');
  if (F.plan) fcPickQuiet(F.plan);
  const target = screen || (EZ.fc.result ? 'ez-forecast' : EZ.result ? 'ez-result' : ui.step && ui.step.startsWith('ez-') ? ui.step : 'ez-vars');
  ezGo(target);
  toast(`열었습니다: ${doc.name}`);
}

// 저장한 계획을 다시 검사해 실행 버튼을 연다 (화면은 검사가 끝나면 다시 그린다)
async function fcPickQuiet(path) {
  const F = fcState();
  try {
    const r = await api('/api/forecast/check', fcBody({ plan: path }));
    F.check = r.check; F.preview = r.preview; F.planLog = r.log;
  } catch (e) { F.check = { ok: false, errors: [e.message], warnings: [] }; }
  if ($('#screen-ez-forecast').classList.contains('active')) fcRender();
}

// ── 부팅 ────────────────────────────────────────────────────────────────
$$('#nav button[data-screen="home"]').forEach(b => b.addEventListener('click', homeRender));
homeRender().catch(e => toast(e.message, 'bad'));
