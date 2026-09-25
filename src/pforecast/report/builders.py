"""워크플로 결과 -> 리포트 조립."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.units import from_si
from .html import (Report, grouped_bar_fig, parity_fig, sweep_fig,
                   timeseries_fig)
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


def selection_report(result, out_path: str | Path) -> Path:
    """구성방정식 후보 비교 리포트."""
    import numpy as np

    cfg = result.config
    df = result.table()
    ok = [r for r in result.results if r.converged and np.isfinite(r.test_rmse)]
    unit = ok[0].unit if ok else "1"
    rep = Report(f"{cfg.name}", f"교체 슬롯 {cfg.slot} · 후보 {len(result.results)}개 · "
                                f"대상 {cfg.target}")

    if ok:
        best = min(ok, key=lambda r: r.test_rmse)
        worst = max(ok, key=lambda r: r.test_rmse)
        rep.tiles([
            {"label": "외삽 최소 오차 후보", "value": best.id,
             "note": f"{best.test_rmse:.2f} {unit} · 파라미터 {best.n_params}개",
             "status": "good"},
            {"label": "최악 후보 대비", "value": f"{worst.test_rmse / best.test_rmse:.2f}",
             "unit": "배", "note": f"{worst.id} = {worst.test_rmse:.2f} {unit}",
             "status": "bad" if worst.test_rmse > best.test_rmse * 1.3 else "warn"},
            {"label": "학습 구간 오차 차이", "value":
                f"{max(r.train_rmse for r in ok) / min(r.train_rmse for r in ok):.2f}",
             "unit": "배", "note": "학습만 보면 후보가 거의 구분되지 않는다"},
        ])

    rep.h2("1. 무엇을 후보로 두었는가")
    rep.note(
        "<b>보존법칙은 후보가 아니다.</b> 질량·화학종·에너지·운동량 보존과 상태방정식은 "
        "네 후보 모두 글자 하나까지 같다. 갈아끼운 것은 <code>closure_eta</code> "
        "한 줄 — 제거효율이 액가스비에 어떻게 의존하는가 하는 <b>구성방정식</b>뿐이다.<br><br>"
        "보존법칙을 후보로 돌리면 외삽 보증이 사라진다. 그러면 물리모델을 쓸 이유가 없다.")
    rep.table(pd.DataFrame([{"후보": r.id, "구성방정식": r.description,
                             "보정 파라미터 수": r.n_params} for r in result.results]))

    rep.h2("2. 비교 결과")
    rep.table(df)
    if ok:
        # "차이가 없다"가 결론일 때는 절대값 막대 4개가 아무것도 말해 주지 않는다.
        # 최선 후보 대비 **차이**를 그리고 판정선을 함께 긋는다.
        ranked = sorted(ok, key=lambda r: r.test_rmse)
        base = ranked[0].test_rmse
        sigma = result.sigma
        thr = result.practical_fraction * sigma if np.isfinite(sigma) else None
        rep.figure(grouped_bar_fig(
            [f"{r.id}" for r in ranked[1:]] or [ranked[0].id],
            {"1위 대비 외삽 RMSE 증가":
                np.array([r.test_rmse - base for r in ranked[1:]] or [0.0])},
            f"1위({ranked[0].id}) 대비 RMSE 차이 [{unit}]",
            "차이의 크기를 계측 불확도와 견준다",
            value_fmt="{:.3f}", threshold=thr,
            threshold_label=f"실무적 구분 한계 (계측 불확도의 "
                            f"{result.practical_fraction*100:.0f}%)" if thr else ""),
            f"막대가 판정선 왼쪽에 있으면 그 후보는 1위와 실무적으로 같습니다. "
            f"계측 불확도 σ = {sigma:.3g} {unit}.")
        rep.figure(grouped_bar_fig(
            [r.id for r in ranked],
            {"학습 구간 RMSE": np.array([r.train_rmse for r in ranked]),
             "외삽 구간 RMSE": np.array([r.test_rmse for r in ranked])},
            f"RMSE [{unit}]", "절대값으로 보면 네 후보가 겹친다"),
            "같은 데이터를 절대값으로 그린 것. 눈으로는 구분되지 않는다 — 그게 결론이다.")

    rep.h2("3. 판정")
    rep.bullets(result.verdict(), markdown=True)
    rep.note(
        "<b>1등만 보고 고르지 마세요.</b> 외삽 오차가 비슷하면 파라미터가 적은 쪽이 낫습니다. "
        "식별성이 무너진 후보는 예측이 맞더라도 파라미터를 물리적으로 해석할 수 없습니다. "
        "파라미터가 허용 경계에 붙었다면 대개 모델 형태 자체가 데이터와 맞지 않는다는 "
        "신호입니다.", kind="warn")

    rep.h2("4. 운전구간별 예측 비교")
    rep.html(_selection_curve(result, ok, unit))

    rep.h2("5. 한계")
    rep.bullets([
        "AICc 는 <b>학습 구간 적합도</b> 기반이라 외삽 능력을 직접 재지 못합니다. "
        "파라미터 수가 값을 하는지 보는 보조 지표로만 쓰세요.",
        "외삽 검증은 <b>운전영역으로 자른 분할</b>입니다. 무작위 k-fold 로 하면 학습과 검증이 "
        "같은 분포가 되어 외삽 능력을 전혀 못 잽니다.",
        "후보 집합 밖에 정답이 있으면 이 비교는 <b>가장 덜 틀린 것</b>을 고를 뿐입니다. "
        "모든 후보의 외삽 오차가 크면 형태를 더 찾아야 한다는 뜻입니다.",
    ])
    return rep.render(out_path)


def _selection_curve(result, ok, unit: str) -> str:
    """가동율 구간별 평균: 실측 vs 최선/최악 후보."""
    import numpy as np
    from .html import fig_to_img

    cfg = result.config
    if len(ok) < 1:
        return ""
    drive = next((c for c in result.train.columns if c.endswith(".util")), None)
    if drive is None:
        return ""
    full = pd.concat([result.train, result.test]).sort_index()
    x = full[drive].to_numpy()
    meas = _conv(full[cfg.target], unit)
    best = min(ok, key=lambda r: r.test_rmse)
    worst = max(ok, key=lambda r: r.test_rmse)
    picks = [("실측 (구간평균)", meas)]
    for r in ([best] if best is worst else [best, worst]):
        pred = pd.concat([r.train_pred, r.test_pred]).sort_index()
        picks.append((f"{r.id}", _conv(pred, unit)))

    bins = np.linspace(np.nanmin(x), np.nanmax(x), 13)
    centers, cols = [], [[] for _ in picks]
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (x >= lo) & (x < hi)
        if sel.sum() < 5:
            continue
        centers.append(0.5 * (lo + hi))
        for j, (_, arr) in enumerate(picks):
            cols[j].append(float(np.nanmean(arr[sel])))
    series = {name: np.array(vals) for (name, _), vals in zip(picks, cols)}
    tr = result.train[drive]
    fig = sweep_fig(np.array(centers), series, f"{drive} [-]", f"{cfg.target} [{unit}]",
                    train_range=(float(tr.min()), float(tr.max())),
                    title="후보별 예측 — 보정 구간 밖에서 갈라진다")
    return (f'<figure>{fig_to_img(fig, "후보 비교 곡선")}<figcaption>'
            f'음영 구간이 보정에 쓴 영역이다. 그 안에서는 후보가 겹치고, 밖으로 나가면 '
            f'벌어진다.</figcaption></figure>')


def analysis_report(result, out_path: str | Path) -> Path:
    """데이터 주도 분석(XAI) 대시보드.

    화면 구성의 원칙: **읽는 사람이 결과를 오독하지 못하게 한다.** 그래서
    숫자보다 먼저 '이 숫자를 어떻게 읽어야 하는가'를 띄우고, 상관으로 얽힌
    영향도는 묶음 기준을 먼저 보여준 뒤 개별 순위를 경고와 함께 붙인다.
    """
    import numpy as np

    from ..analyze.surrogate import partial_dependence

    cfg = result.config
    s = result.surrogate
    rep = Report(f"{cfg.name}", f"타깃 {cfg.target} · 데이터 {result.profile.n_rows:,}행")

    # --- 요약 타일 ---
    tiles = []
    if s is not None:
        skill = s.skill
        tiles.append({"label": "설명력 (교차검증 R²)", "value": f"{s.r2_cv:.3f}",
                      "note": f"{s.model_name} · 시간블록 5-fold",
                      "status": "good" if s.r2_cv > 0.7 else "warn" if s.r2_cv > 0.3 else "bad"})
        tiles.append({"label": "예측오차 RMSE", "value": f"{s.rmse_cv:.3g}",
                      "unit": f" {s.unit}",
                      "note": f"평균예측 대비 {skill*100:.0f}% 개선" if np.isfinite(skill) else ""})
    sure = [b for b in result.balances if b.confidence == "확실"]
    tiles.append({"label": "자동 발견 보존식", "value": f"{len(sure)}", "unit": "개",
                  "note": "데이터가 스스로 만족하는 물리 제약",
                  "status": "good" if sure else ""})
    if np.isfinite(result.holdout_outside):
        tiles.append({"label": "검증구간 외삽 비율",
                      "value": f"{result.holdout_outside*100:.1f}", "unit": "%",
                      "note": "학습 포락선 밖 = 대리모델 근거 없음",
                      "status": "bad" if result.holdout_outside > 0.2 else
                                "warn" if result.holdout_outside > 0.05 else "good"})
    rep.tiles(tiles)

    rep.h2("먼저 읽을 것")
    rep.bullets(result.headline(), markdown=True)

    # --- 사다리 ---
    rep.h2("1. 분석 사다리 — 어디까지 갔고 어디서 막혔는가")
    rep.note(
        "데이터만으로 되는 것(L0~L3)과 사람이 구조를 줘야 하는 것(L4~L5)은 다릅니다. "
        "<b>컬럼 이름과 숫자만으로는 '무엇이 무엇에 연결되는가'를 유도할 수 없습니다.</b> "
        "그래서 이 도구는 자동으로 갈 수 있는 데까지 가고, 그다음에 무엇이 더 필요한지 "
        "구체적으로 알려주는 방식으로 설계했습니다.")
    rep.table(result.ladder_table())

    # --- 발견된 물리 ---
    rep.h2("2. 데이터에서 자동으로 찾은 물리")
    if result.balances:
        rep.h3("보존식 후보")
        rep.table(pd.DataFrame([{
            "신뢰도": b.confidence, "관계식": b.formula(), "차원": b.dimension,
            "상대잔차[%]": b.residual_rel * 100, "R²": b.r2,
            "보존식 형태": "예" if b.is_conservation else "아니오",
        } for b in result.balances]))
        rep.note(
            "같은 차원을 가진 컬럼들 사이에서 <b>계수가 작은 정수인 선형 관계</b>만 "
            "찾습니다. 그래야 나온 식이 차원적으로 옳고 사람이 읽을 수 있습니다. "
            "차원을 무시하고 항 사전을 훑는 방식(SINDy 류)은 현장 노이즈에서 무너지고 "
            "보존법칙을 위반하는 식을 내놓습니다.<br><br>"
            "<b>신뢰도 '우연 의심'</b>은 잔차는 작지만 R²가 낮은 경우입니다 — 크기가 "
            "비슷한 두 신호가 우연히 맞아떨어진 것일 수 있습니다.")
    else:
        rep.text("같은 차원의 컬럼 묶음에서 성립하는 선형 관계를 찾지 못했습니다.")
    nontrivial = [g for g in result.pis if len(g.exponents) > 1]
    if nontrivial:
        rep.h3("무차원군 (Buckingham Π)")
        rep.table(pd.DataFrame([{
            "무차원군": g.formula(), "타깃 포함": "예" if g.contains_target else "",
        } for g in nontrivial]))
        rep.note(
            "무차원군으로 회귀하면 모델이 <b>상사법칙을 정확히 만족</b>합니다. 순수 회귀와 "
            "달리 스케일 방향으로는 외삽이 성립하므로, 물리모델과 ML 사이의 중간 사다리로 "
            "쓸 수 있습니다.")

    if s is None:
        return rep.render(out_path)

    # --- 예측 ---
    rep.h2("3. 예측 성능")
    if s.pred is not None:
        step = max(1, len(s.pred) // 1500)
        idx = s.pred.index[::step]
        meas = result.df.loc[s.pred.index, cfg.target].to_numpy()[::step]
        rep.figure(timeseries_fig(
            idx, {"실측": meas, "예측 (교차검증)": s.pred.to_numpy()[::step]},
            f"{cfg.target} [{s.unit}]", "시간블록 교차검증 예측"),
            "각 구간은 그 구간을 빼고 학습한 모델로 예측한 값입니다. "
            "무작위 k-fold 를 쓰면 앞뒤 시점이 서로 새어 들어가 성능이 과대평가됩니다.")
        ok = np.isfinite(s.pred.to_numpy()) & np.isfinite(
            result.df.loc[s.pred.index, cfg.target].to_numpy())
        rep.figure(parity_fig(
            {"교차검증": (result.df.loc[s.pred.index, cfg.target].to_numpy()[ok],
                          s.pred.to_numpy()[ok])},
            f"{cfg.target} [{s.unit}]", "예측-실측 대응도"),
            "회색 띠는 ±10%.")
    rep.table(pd.DataFrame([{
        "지표": "교차검증 RMSE", "값": s.rmse_cv,
    }, {"지표": "교차검증 MAE", "값": s.mae_cv},
        {"지표": "교차검증 R²", "값": s.r2_cv},
        {"지표": "학습 RMSE (참고)", "값": s.rmse_train},
        {"지표": "평균예측 RMSE (기준선)", "값": s.baseline_rmse},
        {"지표": "홀드아웃 RMSE", "값": result.holdout_rmse}]),
        f"단위 {s.unit}. 학습 RMSE 가 교차검증보다 훨씬 작으면 과적합입니다.")

    # --- 영향인자 ---
    rep.h2("4. 영향인자 분석 (XAI)")
    rep.h3("4-1. 인자 묶음 기준 — 이쪽을 먼저 보세요")
    rep.note(
        "순열 중요도는 <b>서로 상관된 변수들 사이에서 기여를 임의로 나눠 갖습니다.</b> "
        "그래서 상관 0.9 이상인 변수를 묶어 <b>함께 섞은</b> 결과를 먼저 보여줍니다. "
        "묶음 안의 순위는 데이터가 결정해 주지 못합니다.")
    rep.table(s.cluster_table())
    if s.clusters:
        rep.figure(grouped_bar_fig(
            [c.label for c in s.clusters[:8]],
            {"묶음 중요도": np.array([c.importance_pct for c in s.clusters[:8]])},
            "중요도 [%]", "인자 묶음별 기여", value_fmt="{:.1f}%"),
            "묶음 단위로 잰 값. 구성 변수는 위 표를 보세요.")

    rep.h3("4-2. 개별 순열 중요도 — 얽힘 표시를 반드시 같이 보세요")
    rep.table(s.importance_table())
    ent = [i for i in s.importances if i.entangled]
    if ent:
        rep.note(
            f"<b>{len(ent)}개 변수가 다른 변수와 상관 0.8 을 넘습니다.</b> 이 변수들의 "
            "개별 순위는 모델이 어느 쪽을 먼저 썼는지에 따라 달라질 뿐, 인과의 크기가 "
            "아닙니다. 실제로 이 예제에서는 NOx 발생량이 가장 적은 장비군이 1위로 "
            "올라왔습니다.", kind="warn")

    # --- 부분의존도 ---
    pdp_targets = []
    for c in s.clusters[:4]:
        member = max(c.members, key=lambda m: next(
            (i.importance for i in s.importances if i.feature == m), 0.0))
        pdp_targets.append((member, c.is_group))
    if pdp_targets and s.model is not None:
        rep.h3("4-3. 부분의존도 (변수를 바꾸면 예측이 어떻게 변하는가)")
        X = result.df[s.features].dropna()
        Xn = X.to_numpy(dtype=float)
        for name, is_group in pdp_targets[:3]:
            j = s.features.index(name)
            lo, hi = float(np.nanpercentile(Xn[:, j], 2)), float(np.nanpercentile(Xn[:, j], 98))
            if not np.isfinite(lo + hi) or hi <= lo:
                continue
            grid = np.linspace(lo, hi, 25)
            pd_vals = partial_dependence(s.model, Xn, j, grid)
            title = f"{name}" + (" (묶음 대표 — 개별 해석 불가)" if is_group else "")
            rep.figure(sweep_fig(grid, {"예측 평균": pd_vals}, name,
                                 f"{cfg.target} [{s.unit}]", title=title,
                                 figsize=(6.6, 3.2)),
                       ("이 변수는 다른 변수와 강하게 얽혀 있어, 곡선의 기울기를 "
                        "'이 변수만 바꿨을 때의 효과'로 읽으면 안 됩니다."
                        if is_group else
                        "다른 변수는 실제 분포에서 표본추출해 평균낸 곡선입니다."))

    # --- 외삽 ---
    rep.h2("5. 외삽 경고")
    if s.envelope is not None:
        env_rows = []
        for f in s.features:
            env_rows.append({"피처": f, "학습 최소": s.envelope.lo.get(f, np.nan),
                             "학습 최대": s.envelope.hi.get(f, np.nan)})
        rep.table(pd.DataFrame(env_rows), "이 범위를 벗어난 입력에 대한 대리모델 예측은 "
                                          "근거가 없습니다.")
    rep.note(
        f"검증 구간의 <b>{result.holdout_outside*100:.1f}%</b> 가 학습 포락선 밖입니다. "
        "외삽이 필요한 질문(증설, 설비 교체, 가동율 상향)은 대리모델로 답할 수 없습니다 — "
        "지배방정식 모델(L4)로 넘어가야 합니다.",
        kind="bad" if result.holdout_outside > 0.2 else "warn")

    # --- 개선안 ---
    if result.improvement is not None:
        imp = result.improvement
        rep.h2("6. 개선안 (제어 가능한 변수만)")
        rep.table(pd.DataFrame([{
            "제어변수": k, "제안값": v,
            "학습 범위": f"{s.envelope.lo.get(k, np.nan):.4g} ~ {s.envelope.hi.get(k, np.nan):.4g}",
        } for k, v in imp.settings.items()]))
        rep.table(pd.DataFrame([{
            "기준 예측": imp.baseline_value, "개선 후 예측": imp.predicted_value,
            "변화[%]": imp.change_pct, "포락선 이탈": imp.extrapolation,
            "채택 가능": "예" if imp.feasible else "아니오 (범위 밖)",
        }]))
        ctrl_share = sum(c.importance_pct for c in s.clusters
                         if set(c.members) & set(cfg.controllable))
        rep.note(
            f"제어변수가 설명하는 비중은 전체의 <b>{ctrl_share:.1f}%</b> 입니다. "
            + ("이 정도로 약한 신호에서 나온 개선 방향은 부호가 뒤집히기도 합니다. "
               "<b>물리 모델로 반드시 교차검증하세요.</b>" if ctrl_share < 5 else
               "영향력이 충분하므로 제안을 검토할 만합니다."),
            kind="bad" if ctrl_share < 5 else "")

    # --- 한계 ---
    rep.h2("7. 이 분석의 한계")
    rep.bullets([
        "<b>대리모델은 학습 포락선 안에서만 유효합니다.</b> 증설·설비교체·운전방식 변경처럼 "
        "과거에 없던 조건은 원리적으로 답할 수 없습니다.",
        "<b>영향인자는 상관이지 인과가 아닙니다.</b> 특히 한 덩어리로 묶인 변수들 사이에서는 "
        "순위 자체가 의미 없습니다.",
        "<b>보존식 탐지는 같은 차원의 컬럼이 2개 이상 있을 때만</b> 동작합니다. 지류별 "
        "유량계처럼 합이 맞아야 하는 계측을 함께 넣을수록 많이 찾습니다.",
        "<b>계통 토폴로지는 데이터에서 유도할 수 없습니다.</b> L4 로 가려면 컴포넌트 목록, "
        "연결 관계, 설계 제원 세 가지를 사람이 줘야 합니다.",
    ])
    return rep.render(out_path)
