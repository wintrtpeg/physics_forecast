"""워크플로 결과 -> 리포트 조립."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.units import from_si
from .html import Report, parity_fig, sweep_fig, timeseries_fig
from .style import SERIES_LIGHT


def _unit_of(res, target: str) -> str:
    return next((e.unit for e in res.tagmap.observations if e.primary == target), "1")


def _conv(values, unit: str) -> np.ndarray:
    return np.array([from_si(float(v), unit) for v in np.asarray(values, dtype=float)])


def calibration_report(cfg, res, out_path: str | Path) -> Path:
    """보정 + 외삽 검증 리포트."""
    tgt = cfg.baseline_target or res.obs_cols[0]
    unit = _unit_of(res, tgt)
    rep = Report(f"{cfg.name} — 물리기반 NOx 예측 모델 검증",
                 f"모델 {res.model.system.name} · 관측 {len(res.obs_cols)}종")

    m_train = res.train[tgt]
    m_test = res.test[tgt]
    p_train = res.physics_train[tgt]
    p_test = res.physics_test[tgt]

    def rmse(a, b):
        a, b = _conv(a, unit), _conv(b, unit)
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.any() else float("nan")

    phys_test_rmse = rmse(p_test, m_test)
    tiles = [
        {"label": "학습 구간 RMSE (물리)", "value": f"{rmse(p_train, m_train):.2f}", "unit": unit,
         "note": f"{len(res.train):,}행"},
        {"label": "외삽 구간 RMSE (물리)", "value": f"{phys_test_rmse:.2f}", "unit": unit,
         "note": f"{len(res.test):,}행 · 학습에 없던 영역", "status": "good"},
    ]
    if res.baseline is not None:
        bl_rmse = rmse(res.baseline_test, m_test)
        ratio = bl_rmse / max(phys_test_rmse, 1e-9)
        tiles.append({"label": "외삽 구간 RMSE (ML)", "value": f"{bl_rmse:.2f}", "unit": unit,
                      "note": f"물리모델 대비 {ratio:.1f}배",
                      "status": "bad" if ratio > 1.5 else "warn"})
    rep.tiles(tiles)

    rep.h2("1. 무엇을 검증했는가")
    rep.bullets([
        f"동일한 <b>저부하 구간 데이터만</b> 써서 물리모델을 보정하고 ML 기준모델을 학습시켰다. "
        f"(조건 <code>{cfg.train_query}</code>)",
        f"그 다음 <b>학습에 전혀 쓰지 않은 고부하 구간</b>에서 두 모델을 비교했다. "
        f"(조건 <code>{cfg.test_query}</code>)",
        "물리모델이 맞춘 것은 회귀계수가 아니라 덕트 저항·후드 흡입압·제거효율 상수 같은 "
        "<b>물리 파라미터</b>다. 운전조건이 바뀌어도 이 값들은 변하지 않는다.",
    ])

    if res.calibration is not None:
        rep.h2("2. 보정된 물리 파라미터")
        t = res.calibration.table()
        t.columns = ["파라미터", "단위", "설계값", "보정값", "변화[%]", "표준오차", "상대표준오차[%]"]
        rep.table(t)
        warns = res.calibration.identifiability_warnings()
        if warns:
            rep.note(
                "<b>식별성 경고</b> — 아래 파라미터는 이 데이터만으로는 개별 값이 결정되지 "
                "않습니다. 예측에는 문제가 없지만(식별 가능한 조합은 맞았으므로), "
                "개별 값을 물리적 해석에 쓰면 안 됩니다."
                + "<ul>" + "".join(f"<li>{w}</li>" for w in warns) + "</ul>",
                kind="warn")
        else:
            rep.note("모든 보정 파라미터가 데이터로부터 충분히 결정되었습니다.", kind="")

    rep.h2("3. 외삽 성능 비교")
    if res.comparison is not None and len(res.comparison):
        rep.table(res.comparison, "같은 학습 데이터, 같은 검증 데이터로 공정 비교")
    rep.html(_extrapolation_figure(cfg, res, tgt, unit))

    rep.h2("4. 시계열 추종")
    full = pd.concat([res.train, res.test]).sort_index()
    pred_full = pd.concat([p_train, p_test]).sort_index()
    step = max(1, len(full) // 1500)
    idx = full.index[::step]
    series = {"실측 (TMS)": _conv(full[tgt].to_numpy()[::step], unit),
              "물리모델 예측": _conv(pred_full.to_numpy()[::step], unit)}
    spans = []
    if len(res.train):
        spans.append((res.train.index.min(), res.train.index.max(), "보정 구간", SERIES_LIGHT[0]))
    rep.figure(timeseries_fig(idx, series, f"{tgt} [{unit}]",
                              "실측 대 물리모델 예측", spans=spans),
               "음영은 보정에 사용한 시점 분포. 그 밖은 전부 외삽이다.")
    rep.figure(parity_fig({"학습(보정) 구간": (_conv(m_train, unit), _conv(p_train, unit)),
                           "검증(외삽) 구간": (_conv(m_test, unit), _conv(p_test, unit))},
                          f"{tgt} [{unit}]", "예측-실측 대응도"),
               "1:1 선에서 벗어난 정도가 오차. 회색 띠는 ±10%.")

    rep.h2("5. 다른 관측값 동시 적합도")
    rows = []
    for c in res.obs_cols:
        u = _unit_of(res, c)
        rows.append({
            "관측": c, "단위": u,
            "학습 RMSE": rmse(res.physics_train[c], res.train[c]),
            "검증 RMSE": rmse(res.physics_test[c], res.test[c]),
            "계측 불확도(sigma)": from_si(res.tagmap.sigmas().get(c, np.nan), u),
        })
    rep.table(pd.DataFrame(rows),
              "하나의 물리모델이 NOx·유량·온도·차압을 동시에 설명한다. "
              "회귀모형이라면 관측값마다 별도 모델이 필요하다.")

    _structural_questions_section(cfg, res, rep, tgt, unit)
    return rep.render(out_path)


#: ML 기준모델이 구조적으로 답할 수 없는 질문들.
#: (라벨, 파라미터 패턴 -> 값 지시, 그 조건이 학습 데이터에 존재했는지)
_STRUCTURAL_CASES = [
    ("장비 30% 증설", {"scale": {"SRC_*.n_tools": 1.30}}, "대수는 학습 데이터에서 상수였다"),
    ("장비 30% 증설 + 풀가동", {"scale": {"SRC_*.n_tools": 1.30},
                                 "set": {"SRC_*.util": 1.0}}, "대수는 학습 데이터에서 상수였다"),
    ("송풍기 85% 감속", {"set": {"FAN.n_ratio": 0.85}}, "학습 구간은 0.96~1.00"),
    ("스크러버 순환수 반감", {"scale": {"SCR.L": 0.5}}, "순환수는 태그조차 없었다"),
    ("충전재 교체 (효율 회복)", {"set": {"SCR.ntu_a": 14.0}}, "설비 상태는 특징이 아니다"),
]


def _structural_questions_section(cfg, res, rep: Report, tgt: str, unit: str) -> None:
    """설비/운전 구성이 바뀌는 질문에 두 모델이 각각 어떻게 답하는지."""
    from ..core.solvers import solve_steady
    from ..scenario.spec import apply_settings

    model = res.model
    p0 = model.p0()
    # 기준점: 검증 구간의 평균 운전조건
    base_inputs = res.test[res.inputs_cols].mean()
    from ..calib import build_param_rows
    base_p = build_param_rows(model, base_inputs.to_frame().T,
                              base_p=p0, expansion=res.tagmap.expansion())[0]
    r0 = solve_steady(model, base_p)
    if not r0.success:
        return
    base_val = from_si(r0.x[model.var_index(tgt)], unit) if tgt in [
        v.name for v in model.variables] else float("nan")

    rows = []
    warm = r0.x
    for label, directives, why in _STRUCTURAL_CASES:
        try:
            p = apply_settings(model, base_p, directives.get("set"), directives.get("scale"))
        except KeyError:
            continue
        r = solve_steady(model, p, x0=warm)
        if not r.success:
            continue
        warm = r.x
        val = from_si(r.x[model.var_index(tgt)], unit)
        rows.append({
            "설비/운전 변경": label,
            f"물리모델 [{unit}]": val,
            "기준 대비 [%]": 100.0 * (val / max(base_val, 1e-9) - 1.0),
            "ML 기준모델": "예측 불가",
            "이유": why,
        })
    if not rows:
        return
    rep.h2("6. ML 기준모델이 구조적으로 답할 수 없는 질문")
    rep.text(
        f"아래는 과거 운전 데이터에 존재하지 않는 조건이다. 기준점은 검증 구간의 "
        f"평균 운전조건이며 이때 물리모델 예측은 {base_val:.1f} {unit} 이다.")
    rep.table(pd.DataFrame(rows), float_fmt="{:,.1f}")
    rep.note(
        "<b>여기가 핵심이다.</b> 앞 절의 외삽 오차 차이(RMSE 기준 수십 퍼센트)는 "
        "정도의 문제지만, 이 표는 <b>종류의 문제</b>다. 증설·설비 교체·운전방식 변경은 "
        "과거 데이터에 변동이 없었으므로 어떤 회귀모형도 계수를 가질 수 없다. "
        "반면 물리모델은 대수·회전수·순환수가 지배방정식에 들어 있으므로 "
        "학습 여부와 무관하게 답을 낸다.",
        kind="")


def _extrapolation_figure(cfg, res, tgt: str, unit: str) -> str:
    """가동율 구간별 오차 곡선 (물리 vs ML)."""
    drive = cfg.baseline_features[0] if cfg.baseline_features else res.inputs_cols[0]
    full = pd.concat([res.train, res.test]).sort_index()
    pred_phys = pd.concat([res.physics_train[tgt], res.physics_test[tgt]]).sort_index()
    meas = _conv(full[tgt], unit)
    phys = _conv(pred_phys, unit)
    x = full[drive].to_numpy()
    base = None
    if res.baseline is not None:
        feats = [f for f in cfg.baseline_features if f in full.columns]
        base = _conv(res.baseline.predict(full[feats].to_numpy()), unit)

    bins = np.linspace(np.nanmin(x), np.nanmax(x), 13)
    centers, mm, pp, bb = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (x >= lo) & (x < hi)
        if sel.sum() < 5:
            continue
        centers.append(0.5 * (lo + hi))
        mm.append(np.nanmean(meas[sel]))
        pp.append(np.nanmean(phys[sel]))
        if base is not None:
            bb.append(np.nanmean(base[sel]))
    series = {"실측 (구간평균)": np.array(mm), "물리모델": np.array(pp)}
    if bb:
        series["ML 기준모델"] = np.array(bb)
    tr = res.train[drive]
    fig = sweep_fig(np.array(centers), series, f"{drive} [-]", f"{tgt} [{unit}]",
                    train_range=(float(tr.min()), float(tr.max())),
                    title="운전구간별 평균 — 보정 구간 밖에서 갈라진다")
    from .html import fig_to_img
    return (f'<figure>{fig_to_img(fig, "외삽 비교")}'
            f'<figcaption>학습 구간 안에서는 두 모델이 비슷하다. 밖으로 나가면 '
            f'물리모델만 실측을 따라간다.</figcaption></figure>')


def scenario_report(result, out_path: str | Path, title: str | None = None) -> Path:
    """what-if 시나리오 리포트."""
    spec = result.spec
    rep = Report(title or f"{spec.name} — 시나리오 분석", spec.description or "물리기반 what-if")

    if len(result.violations):
        v = result.violations
        rep.tiles([{"label": "관리기준 초과", "value": f"{len(v)}", "unit": "건",
                    "note": ", ".join(sorted(set(v['target']))), "status": "bad"}])
        rep.note("<b>관리기준을 넘는 케이스가 있습니다.</b> 아래 표에서 초과 폭을 확인하세요.",
                 kind="bad")
        rep.table(v.rename(columns={"where": "구분", "label": "케이스", "target": "항목",
                                    "value": "예측값", "limit": "기준", "type": "유형",
                                    "margin_%": "초과율[%]"}))
    else:
        rep.tiles([{"label": "관리기준 초과", "value": "0", "unit": "건",
                    "note": "모든 케이스 기준 이내", "status": "good"}])

    if spec.description:
        rep.h2("검토 배경")
        rep.text(spec.description)

    rep.h2("케이스별 결과")
    rep.table(result.cases)

    for name, df in result.sweeps.items():
        rep.h2(f"스윕: {name}")
        xcol = df.columns[0]
        ycols = [c for c in df.columns if c not in (xcol, "converged")][:3]
        lim = None
        for t, l in (spec.limits or {}).items():
            if t in ycols:
                lim = l if not isinstance(l, dict) else l.get("max")
                break
        if ycols:
            fig = sweep_fig(df[xcol].to_numpy(),
                            {c: df[c].to_numpy() for c in ycols[:1]},
                            xcol, ycols[0], limit=lim, title=name)
            rep.figure(fig, f"{xcol} 변화에 따른 {ycols[0]}")
        rep.table(df)
    return rep.render(out_path)
