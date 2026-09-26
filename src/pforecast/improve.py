"""개선 피드백 루프 — ``pf improve calibration.yaml``.

물리모델을 좋게 만드는 일은 한 번에 끝나지 않는다. 보정하고, 모델이 어디서 틀리는지
보고, 설정이나 모델을 고치고, 다시 잰다. 이 명령은 그 한 바퀴를 **학습 구간만으로**
돌려서 다음에 무엇을 고칠지 근거와 함께 알려준다.

1. 보정 설계 후보(행 선택 방식, 추가 파라미터)를 **학습 구간 안의 전진 폴드**로 비교한다.
   학습 끝 두 달을 한 달씩 검증으로 쓰고, 그 앞만으로 보정한다. 검증 구간(test)의
   결과값은 한 번도 읽지 않는다 — 여기서 고른 설정의 성능은 나중에 test 가 판정한다.
2. 잔차 피드백: 보정하지 않은 파라미터 중 풀면 잔차가 줄어드는 것 (score test).
3. 파라미터 시변성: 학습 구간을 토막 내 다시 맞췄을 때 움직이는 파라미터 = 설비 상태 변화.
4. 검증된 적 없는 외삽 방향: 예측 구간의 **입력**(생산계획·설정값처럼 미리 아는 값)이
   학습에서 한 번도 안 변했거나 드물게만 변한 방향.

내부 폴드가 말해주지 못하는 것도 있다. 학습에서 한 번도 변하지 않은 방향(증설 전
장비 대수)은 어떤 폴드에서도 검증되지 않는다. 실제로 '최근 3주로 다시 맞추기'는 내부
폴드에서 오차를 절반으로 줄였지만 증설 이후 예측에서는 세 배로 키웠다. 그래서 4번을
따로 보고하고, 내부 폴드 점수만으로 설계를 바꾸지 않는다 (복잡한 쪽은 2% 이상 나아야).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calib import CalibrationSpec, build_param_rows, calibrate, simulate
from .calib.design import (Direction, ForecastSensitivity, ParamDrift, ParamProposal, _restore,
                           forecast_sensitivity, parameter_drift, score_test, select_rows,
                           state_band, untested_directions)
from .core.units import from_si
from .data.split import check_split
from .scenario import load_model
from .workflow import WorkflowConfig, load_dataset

#: 복잡한 설계가 채택되려면 내부 검증 오차가 이만큼(상대) 줄어야 한다
MIN_GAIN = 0.02


@dataclass
class Candidate:
    name: str
    sampling: str
    params: list[str]
    note: str = ""


@dataclass
class ImproveResult:
    target: str
    unit: str
    folds: list[dict] = field(default_factory=list)
    scores: pd.DataFrame | None = None           # 후보 x 폴드
    candidates: list[Candidate] = field(default_factory=list)
    best: str = ""
    current: str = ""
    proposals: list[ParamProposal] = field(default_factory=list)
    drift: list[ParamDrift] = field(default_factory=list)
    directions: list[Direction] = field(default_factory=list)
    identifiability: list[str] = field(default_factory=list)
    #: 파라미터 불확도가 예측 평균에 주는 영향 — 학습 조건 / 예측 조건
    sens_train: ForecastSensitivity | None = None
    sens_future: ForecastSensitivity | None = None
    #: 학습 기간 토막별 설비 상태로 각각 푼 예측 구간 평균 [(토막, 값)]
    band: list[tuple[str, float]] = field(default_factory=list)
    seconds: float = 0.0

    def mean_scores(self) -> pd.Series:
        if self.scores is None or not len(self.scores):
            return pd.Series(dtype=float)
        return self.scores.groupby("후보")["점수"].mean()

    def recommendations(self) -> list[str]:
        out = []
        ms = self.mean_scores()
        by = {c.name: c for c in self.candidates}
        if self.best and self.best != self.current and len(ms):
            b, c = by[self.best], by[self.current]
            diff = []
            if b.sampling != c.sampling:
                diff.append(f"`calibrate.sampling: {b.sampling}`")
            extra = [p for p in b.params if p not in c.params]
            if extra:
                diff.append("`calibrate.params` 에 " + ", ".join(f"`{p}`" for p in extra) + " 추가")
            out.append(f"**설정 변경 권고:** {' / '.join(diff)} — 내부 검증 {self.target} 오차 "
                       f"{ms[self.current]:.3g} → {ms[self.best]:.3g} {self.unit} "
                       f"({len(self.folds)}개 폴드 평균).")
        elif len(ms):
            out.append(f"현재 보정 설정이 후보 중 가장 낫거나, 더 나은 후보가 {MIN_GAIN * 100:.0f}% 이상 "
                       "좋아지지 않습니다. 설정은 그대로 두세요.")
        st, sf = self.sens_train, self.sens_future
        if st is not None and sf is not None and st.sigma > 0 and sf.contributions:
            ratio = sf.sigma / st.sigma
            out.append(f"**예측 불확도:** 파라미터 불확도만으로 예측 평균이 ±{sf.sigma:.2g} {self.unit} "
                       f"(1σ) 흔들립니다 — 학습 조건(±{st.sigma:.2g})의 {ratio:.1f}배. 주범: "
                       f"{sf.top()}. 이 파라미터들을 따로 움직이는 운전 데이터(예: 장비가 대기 "
                       "상태인 구간, 계통을 바꾼 구간)가 늘수록 외삽 예측이 좁아집니다. "
                       "(자기상관 때문에 절대 크기는 과소평가일 수 있습니다 — 비율과 순위를 보세요.)")
        if len(self.band) >= 2:
            vals = [v for _, v in self.band]
            out.append(f"**상태 변동 폭:** 학습 기간의 토막별 설비 상태로 예측 구간을 각각 풀면 평균이 "
                       f"{min(vals):.4g}~{max(vals):.4g} {self.unit} (폭 {max(vals) - min(vals):.2g}) 입니다. "
                       "파라미터 공분산보다 이쪽이 실제 예측 오차에 가깝습니다 — 설비 상태는 앞으로도 "
                       "변하기 때문입니다.")
        low = [f for f in self.folds if f["extrapolation"] < 0.5]
        if low:
            out.append("**내부 폴드가 외삽이 아닙니다** (검증 구간 중 학습 운전영역 밖: "
                       + ", ".join(f"{f['extrapolation'] * 100:.0f}%" for f in low)
                       + "). 학습 기간이 짧거나 그동안 운전 조건이 거의 안 변했습니다. 위 설계 비교는 "
                       "외삽 능력이 아니라 학습 범위 안의 성능을 잰 것이라, 설정을 바꾸는 근거로 약합니다.")
        tested = {c.name[2:] for c in self.candidates if c.name.startswith("+ ")}
        ms = self.mean_scores()
        for p in self.proposals:
            if p.name in tested and p.gain_pct >= 5.0 and self.best != f"+ {p.name}":
                sc = ms.get(f"+ {p.name}", float("nan"))
                out.append(f"잔차는 `{p.name}` 를 가리키지만(기대 감소 {p.gain_pct:.0f}%, 독립성 "
                           f"{p.independence:.2f}) 풀어 봐도 {self.target} 예측은 나아지지 않았습니다 "
                           f"({sc:.3g} vs 현재 {ms.get(self.current, float('nan')):.3g}). 목표가 아닌 관측"
                           "(온도·유량·차압)만 맞추는 파라미터로 보입니다 — 그 관측의 계측이나 모델을 따로 보세요.")
        weak = [p for p in self.proposals if not p.usable and p.gain_pct >= 5.0]
        for p in weak[:2]:
            out.append(f"잔차는 `{p.name}` 도 가리키지만(기대 감소 {p.gain_pct:.0f}%) 이미 보정한 "
                       f"파라미터와 구분되지 않습니다(독립성 {p.independence:.2f}). 분리하려면 이 "
                       "파라미터만 따로 움직이는 운전 데이터가 필요합니다.")
        for d in self.drift:
            if d.significant:
                seg = " → ".join(f"{v:.4g}" for v in d.values)
                out.append(f"`{d.name}` 가 학습 기간 동안 {seg} {d.unit} ({d.trend}, "
                           f"{d.change_pct:+.0f}%) — 설비 상태가 변하고 있습니다. 예측은 학습 전체의 "
                           "평균 상태를 가정합니다. 정비·교체 이력과 맞춰 보세요.")
        traded = sorted({tuple(sorted([d.name] + d.traded_with[:1])) for d in self.drift if d.traded_with})
        for a, b in traded:
            out.append(f"`{a}` 와 `{b}` 는 토막마다 서로 반대로 움직입니다 — 상태 변화가 아니라 "
                       "데이터가 둘을 구분하지 못하는 것입니다 (둘의 운전 입력이 같이 움직인다).")
        untested = [d for d in self.directions if d.untested]
        if untested:
            out.append("**검증된 적 없는 외삽 방향:** " + "; ".join(d.short() for d in untested)
                       + ". 이 방향의 반응은 모델 구조와 설계값을 그대로 믿는 것입니다. 검증하려면 "
                       "이 입력을 단계적으로 바꾼 운전 데이터(한 단계당 하루 이상)를 확보하세요.")
        ranged = [d for d in self.directions if not d.untested]
        if ranged:
            out.append("학습 범위 밖 (변한 적은 있는 방향): " + "; ".join(d.short() for d in ranged)
                       + ". 기울기는 데이터로 맞췄지만 범위 밖에서 구성방정식의 모양이 맞는지가 관건입니다.")
        return out

    def lines(self) -> list[str]:
        out = [f"내부 전진 폴드 {len(self.folds)}개 (학습 구간 안에서만):"]
        for f in self.folds:
            out.append(f"  - {f['train_end']} 까지 학습 → {f['val_start']}~{f['val_end']} 검증 "
                       f"({f['n_val']:,}행, 학습 운전영역 밖 {f['extrapolation'] * 100:.0f}%)")
        if self.scores is not None and len(self.scores):
            piv = self.scores.pivot(index="후보", columns="폴드", values="점수")
            piv["평균"] = piv.mean(axis=1)
            piv = piv.sort_values("평균")
            out.append(f"\n{self.target} 내부 검증 RMSE [{self.unit}] "
                       "(외삽 행이 30개 이상인 폴드는 외삽 행만, 아니면 전체 행):")
            out.append(piv.to_string(float_format=lambda v: f"{v:.3f}"))
            out.append(f"→ 선택: {self.best}" + (" (현재 설정)" if self.best == self.current else ""))
        if self.identifiability:
            out.append("\n선택한 설계의 식별성:")
            out += ["  ! " + w for w in self.identifiability]
        for lab, sn in (("학습 조건", self.sens_train), ("예측 조건", self.sens_future)):
            if sn is not None:
                out.append(f"예측 평균의 파라미터 불확도 ({lab}): {sn.mean:.4g} ± {sn.sigma:.2g} {self.unit}"
                           f" — 기여 {sn.top(4)}")
        if self.proposals:
            out.append("\n잔차 피드백 — 보정하지 않은 파라미터를 풀면 (상위 6개):")
            for p in self.proposals[:6]:
                tag = "" if p.usable else "  ← 기존 파라미터와 구분 안 됨"
                out.append(f"  {p.name:26s} 기대 감소 {p.gain_pct:5.1f}%  독립성 {p.independence:.2f}"
                           f"  (현재 {p.value:.4g} {p.unit}){tag}")
        if self.band:
            out.append("토막별 설비 상태로 푼 예측 구간 평균: "
                       + ", ".join(f"{lab} {v:.4g}" for lab, v in self.band) + f" {self.unit}")
        if self.drift:
            out.append("\n파라미터 시변성 (학습 구간을 토막 내 다시 맞춤):")
            for d in self.drift:
                seg = "  ".join(f"{lab} {v:.4g}" for lab, v, _ in d.segments)
                out.append(f"  {d.name:22s} {seg}  [{d.trend}]")
        rec = self.recommendations()
        if rec:
            out.append("\n다음에 할 일:")
            out += ["  - " + r.replace("**", "").replace("`", "") for r in rec]
        return out

    def to_markdown(self) -> str:
        body = "\n".join(self.lines())
        return f"# 개선 피드백 — {self.target}\n\n```\n{body}\n```\n"


def choose(mean_scores: pd.Series, current: str, min_gain: float = MIN_GAIN) -> str:
    """현재 설정 대비 ``min_gain`` 이상 나은 후보가 있을 때만 바꾼다."""
    if not len(mean_scores) or current not in mean_scores or not np.isfinite(mean_scores[current]):
        return current
    best = mean_scores.idxmin()
    return best if mean_scores[best] < mean_scores[current] * (1.0 - min_gain) else current


def _inner_folds(index: pd.DatetimeIndex, n_folds: int, fold_days: float,
                 embargo: str = "1D") -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    end = index.max()
    # 학습 기간이 짧으면(예제는 20일) 폴드를 줄인다. 첫 폴드도 절반 가까이는 학습에 남아야
    # 한다 — 30일 폴드 두 개를 20일에 넣으면 폴드가 하나도 안 생긴다.
    span = (end - index.min()) / pd.Timedelta(days=1)
    fold_days = min(fold_days, span / (n_folds + 2))
    out = []
    for k in range(n_folds, 0, -1):
        va_end = end - pd.Timedelta(days=fold_days * (k - 1))
        va_start = va_end - pd.Timedelta(days=fold_days)
        tr_end = va_start - pd.Timedelta(embargo)
        if (index < tr_end).sum() > 0:
            out.append((tr_end, va_start, va_end))
    return out


def _fit(cfg: WorkflowConfig, model, saved, train, ic, oc, cand: Candidate, tm, expansion):
    _restore(model, saved)
    rows = select_rows(train, cfg.max_rows, ic, oc, method=cand.sampling)
    spec = CalibrationSpec(params=cand.params, observations=oc, sigmas=tm.sigmas(),
                           max_rows=max(len(rows), 1), prior_weight=cfg.prior_weight,
                           loss=cfg.loss, f_scale=cfg.f_scale)
    return calibrate(model, rows[ic], rows[oc], spec, expansion=expansion, verbose=False), rows, spec


def run_improve(cfg: WorkflowConfig, n_folds: int = 2, fold_days: float = 30.0,
                sub: int = 12, verbose: bool = True, drift: bool = True) -> ImproveResult:
    t0 = time.perf_counter()
    log = print if verbose else (lambda *a, **k: None)
    system = load_model(cfg.model)
    model = system.compile()
    model.build()
    tm, df, train, test, ic, oc = load_dataset(cfg)
    oc = [c for c in (cfg.observations or oc) if c in df.columns]
    expansion = tm.expansion()
    target = cfg.baseline_target or oc[0]
    unit = next((e.unit for e in tm.observations if e.primary == target), "1")
    saved = {p.name: p.value for p in model.tunable_params()}
    res = ImproveResult(target=target, unit=unit)

    alt = "stride" if cfg.sampling == "space_filling" else "space_filling"
    cands = [Candidate("현재 설정", cfg.sampling, list(cfg.params)),
             Candidate(f"행 선택 {alt}", alt, list(cfg.params))]

    # 잔차 피드백: 현재 설정으로 학습 전체를 맞춘 뒤, 보정하지 않은 파라미터를 하나씩 풀어 본다
    log("잔차 피드백 계산 (현재 설정으로 학습 전체 보정) ...", flush=True)
    cal, rows, spec = _fit(cfg, model, saved, train, ic, oc, cands[0], tm, expansion)
    res.proposals = score_test(model, rows, ic, oc, tm.sigmas(), expansion, selected=list(cfg.params))
    for p in [p for p in res.proposals if p.usable and p.gain_pct >= 2.0][:2]:
        cands.append(Candidate(f"+ {p.name}", cfg.sampling, list(cfg.params) + [p.name],
                               note=f"잔차 피드백 기대 감소 {p.gain_pct:.1f}%"))
    res.candidates = cands
    res.current = cands[0].name

    rows_out = []
    for tr_end, va_start, va_end in _inner_folds(train.index, n_folds, fold_days):
        tr = train[train.index < tr_end]
        va = train[(train.index >= va_start) & (train.index <= va_end)]
        chk = check_split(tr, va, ic)
        vs = va.iloc[::sub]
        out_mask = chk.outside[::sub] if chk.outside is not None else np.zeros(len(vs), bool)
        label = f"{va_start:%m/%d}~{va_end:%m/%d}"
        res.folds.append({"train_end": f"{tr_end:%Y-%m-%d}", "val_start": f"{va_start:%Y-%m-%d}",
                          "val_end": f"{va_end:%Y-%m-%d}", "n_val": len(va),
                          "extrapolation": chk.extrapolation if np.isfinite(chk.extrapolation) else 0.0})
        for cand in cands:
            t1 = time.perf_counter()
            try:
                _fit(cfg, model, saved, tr, ic, oc, cand, tm, expansion)
            except Exception as exc:  # noqa: BLE001
                log(f"  [{label}] {cand.name}: 보정 실패 — {exc}")
                continue
            P = build_param_rows(model, vs[ic], base_p=model.p0(), expansion=expansion)
            pred = simulate(model, P, [target], index=vs.index).values[target].to_numpy()
            p = np.array([from_si(v, unit) for v in pred])
            y = np.array([from_si(v, unit) for v in vs[target].to_numpy()])
            ok = np.isfinite(p) & np.isfinite(y)
            use = ok & out_mask if (ok & out_mask).sum() >= 30 else ok
            score = float(np.sqrt(np.mean((p[use] - y[use]) ** 2))) if use.any() else np.nan
            rows_out.append({"후보": cand.name, "폴드": label, "점수": score,
                             "전체 RMSE": float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else np.nan,
                             "외삽 행": int((ok & out_mask).sum())})
            log(f"  [{label}] {cand.name:28s} {target} RMSE {score:.3f} {unit}  "
                f"({time.perf_counter() - t1:.0f}s)", flush=True)
    res.scores = pd.DataFrame(rows_out)

    # 선택: 현재 설정 대비 MIN_GAIN 이상 나아야 바꾼다
    res.best = choose(res.mean_scores(), res.current)

    # 선택한 설계로 학습 전체를 다시 맞춰 식별성·시변성을 본다
    chosen = next(c for c in cands if c.name == res.best)
    cal, rows, spec = _fit(cfg, model, saved, train, ic, oc, chosen, tm, expansion)
    res.identifiability = cal.identifiability_warnings()
    try:
        res.sens_train = forecast_sensitivity(model, cal, train[ic], target, unit, expansion)
        if len(test):
            res.sens_future = forecast_sensitivity(model, cal, test[ic], target, unit, expansion)
    except Exception as exc:  # noqa: BLE001 — 진단 실패가 나머지 피드백을 막으면 안 된다
        log(f"  예측 민감도 계산 실패: {exc}")
    if drift:
        log("파라미터 시변성 계산 ...", flush=True)
        res.drift = parameter_drift(model, train, ic, oc, spec, expansion)
        if len(test):
            res.band = state_band(model, res.drift, test[ic], target, unit, expansion)
    _restore(model, saved)

    # 예측 구간의 '입력'만 본다 (결과값은 읽지 않는다)
    if len(test):
        units_in = {e.primary: e.unit for e in tm.inputs}
        disp = lambda f: pd.DataFrame({c: [from_si(v, units_in.get(c, "1")) for v in f[c].to_numpy()]  # noqa: E731
                                       for c in ic}, index=f.index)
        res.directions = untested_directions(disp(train), disp(test), units_in)
    res.seconds = time.perf_counter() - t0
    return res
