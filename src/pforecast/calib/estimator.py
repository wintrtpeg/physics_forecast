"""파라미터 추정(캘리브레이션)과 식별성 진단.

ML 회귀와의 결정적 차이: 여기서 맞추는 것은 **물리 파라미터**다. 덕트 저항계수,
후드 설계 흡입압, 제거효율 상수, 대당 배출계수 같은 것들이다. 이들은 운전조건이
바뀌어도 변하지 않으므로, 한 번 맞춰 두면 학습 구간 밖에서도 모델이 그대로 선다.

그래서 두 가지를 반드시 같이 본다.
* **잔차**: 얼마나 잘 맞는가
* **식별성**: 그 값이 데이터로부터 실제로 결정된 것인가, 아니면 다른 파라미터와
  상쇄되어 아무 값이나 가능한가. 후자라면 잔차가 아무리 작아도 외삽은 위험하다.
  상관계수가 0.95 를 넘는 쌍은 사실상 하나의 자유도이므로 경고한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.solvers import solve_steady
from ..core.units import from_si
from .runner import build_param_rows, resolve_targets


@dataclass
class CalibrationSpec:
    """무엇을 맞출지에 대한 선언."""

    params: list[str]                       # 보정 대상 파라미터 (모델 이름)
    observations: list[str]                 # 비교할 관측 대상
    sigmas: dict[str, float] = field(default_factory=dict)   # SI 단위 계측 불확도
    prior_weight: float = 0.05              # 초기 설계값으로 끌어당기는 정칙화 강도
    max_rows: int = 250
    bounds_scale: tuple[float, float] = (0.2, 5.0)   # 초기값 대비 허용 배율
    #: 잔차 손실. "linear"(최소제곱) | "soft_l1" | "huber". 정제를 거쳐도 남는 이상치에
    #: 한두 점이 파라미터를 끌고 가는 것을 막는다. f_scale 은 sigma 단위.
    loss: str = "linear"
    f_scale: float = 3.0


@dataclass
class CalibrationResult:
    names: list[str]
    units: list[str]
    initial: np.ndarray                     # SI
    fitted: np.ndarray                      # SI
    stderr: np.ndarray                      # SI
    correlation: np.ndarray
    cost: float
    n_rows: int
    n_eval: int
    #: 최적화가 실제로 쓴 경계 (SI). 모델 경계와 초기값 대비 배율 경계의 교집합이다.
    lo_used: np.ndarray | None = None
    hi_used: np.ndarray | None = None
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    message: str = ""
    #: 설계값에서도 수렴하지 않아 보정에서 뺀 행 수 (입력이 모델 적용 범위 밖)
    n_excluded: int = 0
    #: 물리 범위 밖이라 경계로 자른 입력 {컬럼: 행 수}
    clipped: dict[str, int] = field(default_factory=dict)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "parameter": self.names,
            "unit": self.units,
            "initial": [from_si(v, u) for v, u in zip(self.initial, self.units)],
            "fitted": [from_si(v, u) for v, u in zip(self.fitted, self.units)],
            "change_%": 100.0 * (self.fitted / np.where(self.initial == 0, 1, self.initial) - 1.0),
            "stderr": [from_si(v, u) for v, u in zip(self.stderr, self.units)],
            "rel_stderr_%": 100.0 * self.stderr / np.maximum(np.abs(self.fitted), 1e-30),
        })

    def at_bound(self, rtol: float = 0.02) -> list[str]:
        """최적화 경계에 붙은 파라미터 이름.

        모델이 선언한 넓은 물리 경계가 아니라 **최적화가 실제로 쓴 경계** 기준이다.
        (물리 경계는 lo=0 처럼 넓어서, 작은 양수 파라미터가 늘 '경계에 붙은' 것으로
        오진된다.)
        """
        if self.lo_used is None or self.hi_used is None:
            return []
        hits = []
        for i, name in enumerate(self.names):
            lo, hi, v = self.lo_used[i], self.hi_used[i], self.fitted[i]
            span = hi - lo
            if not np.isfinite(span) or span <= 0:
                continue
            if abs(v - lo) <= rtol * span or abs(hi - v) <= rtol * span:
                hits.append(name)
        return hits

    def subset(self, names: list[str]) -> tuple[float, float]:
        """일부 파라미터만 본 (최악 상관, 최대 상대표준오차[%])."""
        idx = [i for i, n in enumerate(self.names) if n in set(names)]
        if not idx:
            return 0.0, 0.0
        rel = 100.0 * self.stderr[idx] / np.maximum(np.abs(self.fitted[idx]), 1e-30)
        rel = rel[np.isfinite(rel)]
        worst = 0.0
        for a in range(len(idx)):
            for b in range(len(self.names)):
                if idx[a] == b:
                    continue
                c = self.correlation[idx[a], b]
                if np.isfinite(c) and abs(c) > abs(worst):
                    worst = float(c)
        return worst, (float(np.max(rel)) if len(rel) else 0.0)

    def identifiability_warnings(self, threshold: float = 0.95) -> list[str]:
        out = []
        n = len(self.names)
        for i in range(n):
            for j in range(i + 1, n):
                c = self.correlation[i, j]
                if abs(c) > threshold:
                    out.append(
                        f"{self.names[i]} 와 {self.names[j]} 의 상관계수 {c:+.3f} "
                        "-> 데이터가 둘을 구분하지 못합니다. 하나를 고정하거나 "
                        "두 값을 분리할 수 있는 운전 구간 데이터를 확보하세요."
                    )
        for i, (nm, se, val) in enumerate(zip(self.names, self.stderr, self.fitted)):
            rel = se / max(abs(val), 1e-30)
            if rel > 0.5:
                out.append(f"{nm} 의 상대표준오차가 {rel*100:.0f}% 입니다 -> 사실상 결정되지 않았습니다.")
        return out


def _metrics(pred: np.ndarray, meas: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(pred) & np.isfinite(meas)
    if ok.sum() < 2:
        return {"n": int(ok.sum()), "rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan, "mape": np.nan}
    p, m = pred[ok], meas[ok]
    err = p - m
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((m - m.mean()) ** 2))
    denom = np.where(np.abs(m) < 1e-30, np.nan, m)
    return {
        "n": int(ok.sum()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "mape": float(np.nanmean(np.abs(err / denom)) * 100.0),
    }


def calibrate(
    model,
    inputs: pd.DataFrame,
    observations: pd.DataFrame,
    spec: CalibrationSpec,
    expansion: dict[str, list[str]] | None = None,
    verbose: bool = True,
) -> CalibrationResult:
    """경계가 있는 비선형 최소제곱으로 물리 파라미터를 추정한다."""
    from scipy.optimize import least_squares

    model.build()
    df = inputs.join(observations, how="inner")
    # 입력은 전부 있어야 하고(모델을 못 푼다), 관측은 하나라도 있으면 쓴다. 관측 하나가
    # 비었다고 행을 통째로 버리면, 유량계 하나가 고착된 이틀 동안 NOx 까지 잃는다.
    obs_cols = [c for c in spec.observations if c in df.columns]
    if not obs_cols:
        raise ValueError(f"관측 컬럼이 없습니다: {spec.observations}")
    df = df[df[list(inputs.columns)].notna().all(axis=1) & df[obs_cols].notna().any(axis=1)]
    if len(df) == 0:
        raise ValueError("입력과 관측이 겹치는 시점이 없습니다. 태그맵/시각 정렬을 확인하세요.")
    if len(df) > spec.max_rows:
        step = max(1, len(df) // spec.max_rows)
        df = df.iloc[::step]
    inp = df[[c for c in inputs.columns if c in df.columns]]
    obs = df[obs_cols]

    p_idx = np.array([model.par_index(n) for n in spec.params])
    p_base = model.p0()
    theta0 = p_base[p_idx].copy()
    scale = np.where(np.abs(theta0) > 1e-30, np.abs(theta0), 1.0)

    lo_model = np.array([model.parameters[i].lo for i in p_idx])
    hi_model = np.array([model.parameters[i].hi for i in p_idx])
    lo = np.maximum(theta0 / scale * spec.bounds_scale[0], lo_model / scale)
    hi = np.minimum(theta0 / scale * spec.bounds_scale[1], hi_model / scale)
    # 음수 파라미터(팬 곡선 2차항 등)는 배율 경계가 뒤집히므로 정렬한다
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    y0 = np.clip(theta0 / scale, lo + 1e-12, hi - 1e-12)

    var_targets, out_targets = resolve_targets(model, list(obs.columns))
    sig = np.array([max(spec.sigmas.get(c, 1.0), 1e-30) for c in obs.columns])
    clipped: dict[str, int] = {}
    base_rows = build_param_rows(model, inp, base_p=p_base, expansion=expansion,
                                 clip_report=clipped)

    # 설계값에서도 풀리지 않는 행은 모델 적용 범위 밖이다. 벌점을 매겨 최적화에 넣으면
    # 파라미터가 그 행을 '풀리게' 만드는 쪽으로 끌려간다 — 빼고, 몇 행인지 보고한다.
    x = model.x0()
    keep = np.ones(len(base_rows), dtype=bool)
    for i, p in enumerate(base_rows):
        r = solve_steady(model, p, x0=x)
        if r.success:
            x = r.x
        else:
            keep[i] = False
    n_excluded = int((~keep).sum())
    if keep.sum() == 0:
        raise ValueError("설계값에서 수렴하는 행이 하나도 없습니다. 입력 단위/태그맵을 확인하세요.")
    base_rows = base_rows[keep]
    df, inp, obs = df[keep], inp[keep], obs[keep]
    meas = obs.to_numpy(dtype=float)
    has = np.isfinite(meas)
    n_rows = len(base_rows)
    warm = {"x": model.x0()}
    n_eval = [0]

    def residual(y: np.ndarray) -> np.ndarray:
        n_eval[0] += 1
        rows = base_rows.copy()
        rows[:, p_idx] = y * scale
        pred = np.full((n_rows, len(obs.columns)), np.nan)
        x = warm["x"].copy()
        good = 0
        for i, p in enumerate(rows):
            r = solve_steady(model, p, x0=x)
            if not r.success:
                continue
            x = r.x
            good += 1
            for k, (name, vi) in enumerate(var_targets):
                pred[i, obs.columns.get_loc(name)] = r.x[vi]
            if out_targets:
                ov = model.output_values_si(r.x, p)
                for name in out_targets:
                    pred[i, obs.columns.get_loc(name)] = ov[name]
        if good > n_rows * 0.5:
            warm["x"] = x
        res = (pred - np.where(has, meas, 0.0)) / sig
        # 수렴 실패 시점은 큰 페널티로 처리 (파라미터가 비물리 영역으로 가는 것을 막는다).
        # 측정이 비어 있는 칸은 0 — 정보가 없을 뿐 벌점 대상이 아니다.
        res = np.where(np.isfinite(res), res, 1e3)
        res = np.where(has, res, 0.0)
        flat = res.ravel() / np.sqrt(n_rows)
        prior = spec.prior_weight * (y - theta0 / scale)
        return np.concatenate([flat, prior])

    kw = {}
    if spec.loss and spec.loss != "linear":
        # f_scale 은 잔차 척도(sigma 단위)다. 잔차를 sqrt(n_rows) 로 나눠 넣으므로 맞춰 준다.
        kw = {"loss": spec.loss, "f_scale": spec.f_scale / np.sqrt(n_rows)}
    sol = least_squares(residual, y0, bounds=(lo, hi), xtol=1e-10, ftol=1e-10,
                        diff_step=1e-4, verbose=2 if verbose else 0, **kw)
    fitted = sol.x * scale

    # 공분산 = (J^T J)^-1 * s^2  (스케일 좌표계에서 구해 SI 로 환산)
    #
    # 정칙화 항은 제외하고 **데이터 항만으로** 계산한다. 정칙화는 야코비안에
    # 단위행렬을 더하는 것과 같아서, 그대로 두면 상관계수가 인위적으로 낮아지고
    # "식별된 것처럼" 보인다. 식별성은 데이터가 말해주는 것이지 사전분포가
    # 말해주는 것이 아니다.
    n_prior = len(spec.params) if spec.prior_weight > 0 else 0
    J = sol.jac[:sol.jac.shape[0] - n_prior] if n_prior else sol.jac
    resid_data = sol.fun[:len(sol.fun) - n_prior] if n_prior else sol.fun
    # 비어 있던 측정 칸은 자유도에 넣지 않는다
    dof = max(int(has.sum()) - J.shape[1], 1)
    s2 = float(resid_data @ resid_data) / dof
    try:
        cov = np.linalg.pinv(J.T @ J) * s2
        stderr = np.sqrt(np.clip(np.diag(cov), 0.0, None)) * scale
        d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
        corr = cov / np.outer(d, d)
    except np.linalg.LinAlgError:
        stderr = np.full_like(fitted, np.nan)
        corr = np.full((len(fitted), len(fitted)), np.nan)

    # 최종 파라미터로 예측 다시 계산해 지표 산출
    rows = base_rows.copy()
    rows[:, p_idx] = fitted
    from .runner import simulate
    sim = simulate(model, rows, list(obs.columns), index=df.index)
    metrics = {c: _metrics(sim.values[c].to_numpy(), obs[c].to_numpy()) for c in obs.columns}

    for i, name in enumerate(spec.params):
        model.parameters[p_idx[i]].value = float(fitted[i])
        cname, pname = name.rsplit(".", 1)
        model.system.components[cname].set_param(pname, float(fitted[i]))

    return CalibrationResult(
        names=list(spec.params),
        units=[model.parameters[i].unit for i in p_idx],
        initial=theta0, fitted=fitted, stderr=stderr, correlation=corr,
        cost=float(sol.cost), n_rows=n_rows, n_eval=n_eval[0], metrics=metrics,
        message=str(sol.message), lo_used=lo * scale, hi_used=hi * scale,
        n_excluded=n_excluded, clipped=clipped,
    )
