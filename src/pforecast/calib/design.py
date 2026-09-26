"""보정 설계와 개선 피드백 — 어떤 행으로 맞추고, 다음에 무엇을 고칠 것인가.

보정 결과가 좋아지는 길은 두 가지뿐이다. **더 알려주는 행**으로 맞추거나, **모델이
놓친 것**을 찾아 고치거나. 이 모듈은 둘 다 데이터로 판단한다.

행 선택 (``select_rows``)
    예전에는 학습 구간을 시간 간격으로 150행 뽑았다. 하루 한 번꼴이라 순환펌프 정지
    8시간, 팹 PM 사흘 같은 **드문 운전상태가 통째로 빠진다.** 그런데 외삽에 필요한
    정보는 바로 그 행들에 있다 — 펌프가 서면 제거효율이 0 이 되어 발생량이 스크러버
    효율과 분리된다. 입력공간에서 서로 가장 먼 행(최원점)을 30%, 시간 균등을 70%
    뽑는다. 입력이 1시간 단위로 갱신되면(MES 가동율) 5분 행 12개는 같은 입력에 잡음만
    다른 복제본이므로 **1시간 평균 행**으로 맞춘다 (입력이 한 시간 안에서 계단을
    넘으면 평균이 가상의 상태가 되므로 뺀다).

    실측 (현장형 더미 데이터, ``docs/improvement_log.md``):
    * 1~4월 학습(최원점 50%): NTU 계수 상대표준오차 194% → 13%, 배출계수 세 쌍의 상관 경고 사라짐
    * 데이터 5벌, 1~5월 학습 → 증설 후 예측, 참값 대비 NOx RMSE 평균 3.44 → 2.96,
      최악 5.70 → 3.76. 중앙값은 3.04 → 3.07 로 같다 — 전형적인 경우보다 **나쁜 경우를
      줄인다.** 최원점 비율 50% 는 평균 3.07(4벌), 1시간 평균만 하고 최원점을 안 뽑으면
      3.94 로 오히려 나빠진다. 효과는 평균이 아니라 드문 상태에서 온다.

피드백
    * ``score_test`` — 보정하지 않은 파라미터 중 무엇을 풀면 잔차가 줄어드는가
      (라그랑주 승수 검정). 이미 푼 파라미터 방향은 빼고 보므로, 기존 파라미터와
      공선인 후보(독립성 0 근처)는 '풀어도 구분 안 됨'으로 따로 표시한다.
    * ``parameter_drift`` — 학습 구간을 몇 토막으로 나눠 토막마다 다시 맞춘다. 값이
      시간에 따라 움직이는 파라미터는 **설비 상태가 변하고 있다**는 뜻이다 (충전재
      오염, 덕트 퇴적, 비계측 레시피 변경). 예측에는 쓰지 않는다 — 아래 참고.
    * ``untested_directions`` — 예측 구간의 운전 입력이 학습에서 한 번도 변하지 않았거나
      드물게만 변한 방향. 그 방향의 반응은 데이터로 검증된 적이 없고 **모델 구조와
      설계값을 그대로 믿는 것**이다. 어떤 교차검증도 이것을 잡지 못한다.

하지 않는 것 — 최근 구간으로 다시 맞춰 예측하기
    학습 끝 3주로 파라미터를 다시 맞추면 1~2개월 앞 내부 검증에서는 오차가 크게 준다
    (5.46 → 2.95). 그런데 증설 이후 석 달을 맞히는 실제 외삽에서는 **세 배로 나빠졌다**
    (2.07 → 6.07). 좁은 창에는 운전 다양성이 없어 NOx 상승을 아무 파라미터(대당 대기
    배출)에나 붙이고, 월말에 최대가 되는 분석계 드리프트까지 공정 변화로 흡수한다.
    그래서 파라미터 변동은 진단으로만 보고한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.units import from_si


# ---------------------------------------------------------------------------
# 행 선택
# ---------------------------------------------------------------------------

def average_rows(frame: pd.DataFrame, inputs: list[str], obs: list[str],
                 period: str = "1h", min_frac: float = 0.8) -> pd.DataFrame:
    """``period`` 평균 행. 입력이 그 안에서 계단을 넘은 구간은 뺀다.

    '계단'의 기준은 컬럼 범위의 5% 와 평소 구간 내 흔들림의 3배 중 큰 쪽이다. 범위만
    보면 거의 일정한 입력(순환수)은 잡음만으로도 전부 '계단'이 된다.
    """
    if not isinstance(frame.index, pd.DatetimeIndex) or len(frame) < 2:
        return frame
    step = frame.index.to_series().diff().median()
    per = pd.Timedelta(period)
    if not pd.notna(step) or step >= per:
        return frame
    need = max(1, int(round(min_frac * per / step)))
    cols = [c for c in inputs + obs if c in frame.columns]
    g = frame[cols].resample(period)
    mean = g.mean()
    count = frame[inputs].notna().all(axis=1).resample(period).sum()
    rng = frame[inputs].resample(period).max() - frame[inputs].resample(period).min()
    span = frame[inputs].quantile(0.98) - frame[inputs].quantile(0.02)
    tol = np.maximum(0.05 * span, 3.0 * rng.median()).replace(0, 1e-12)
    steady = (rng <= tol).all(axis=1)
    out = mean[(count >= need) & steady]
    return out[out[inputs].notna().all(axis=1)]


def _normalized(frame: pd.DataFrame, inputs: list[str]) -> np.ndarray:
    X = frame[inputs].to_numpy(dtype=float)
    lo, hi = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return (X - lo) / span


def space_filling(frame: pd.DataFrame, n: int, inputs: list[str],
                  frac: float = 0.3) -> pd.DataFrame:
    """``n`` 행: (1-frac) 는 시간 균등, frac 는 입력공간 최원점(greedy maximin)."""
    if len(frame) <= n:
        return frame
    X = _normalized(frame, inputs)
    ok = np.flatnonzero(np.all(np.isfinite(X), axis=1))
    if len(ok) <= n:
        return frame.iloc[ok]
    k_far = int(round(n * frac))
    k_time = n - k_far
    chosen = list(ok[:: max(1, len(ok) // max(k_time, 1))][:k_time])
    Xo = X[ok]
    d = np.full(len(ok), np.inf)
    for c in chosen:
        d = np.minimum(d, ((Xo - X[c]) ** 2).sum(axis=1))
    for _ in range(k_far):
        j = int(np.argmax(d))
        if not np.isfinite(d[j]) or d[j] <= 0:
            break
        chosen.append(int(ok[j]))
        d = np.minimum(d, ((Xo - Xo[j]) ** 2).sum(axis=1))
    return frame.iloc[np.unique(chosen)]


def select_rows(frame: pd.DataFrame, n: int, inputs: list[str], obs: list[str],
                method: str = "space_filling", period: str = "1h",
                frac: float = 0.3) -> pd.DataFrame:
    """보정에 쓸 행. ``method``: ``space_filling`` (기본) | ``stride`` (예전 방식)."""
    have = frame[frame[inputs].notna().all(axis=1) & frame[obs].notna().any(axis=1)]
    if method == "stride":
        return have.iloc[:: max(1, len(have) // max(n, 1))] if len(have) > n else have
    if method != "space_filling":
        raise ValueError(f"알 수 없는 행 선택 방식: {method!r} (space_filling | stride)")
    avg = average_rows(have, inputs, obs, period)
    avg = avg[avg[obs].notna().any(axis=1)]
    if len(avg) < max(n // 2, 10):          # 평균 행이 너무 적으면(짧은 데이터) 원래 행으로
        avg = have
    return space_filling(avg, n, inputs, frac)


# ---------------------------------------------------------------------------
# 잔차 피드백: 무엇을 더 풀까 (score test)
# ---------------------------------------------------------------------------

@dataclass
class ParamProposal:
    name: str
    gain_pct: float          # 풀었을 때 기대되는 가중 잔차제곱합 감소 [%]
    independence: float     # 이미 푼 파라미터들과 구분되는 정도 (0 = 완전 공선, 1 = 독립)
    value: float             # 현재 값 (표시 단위)
    unit: str

    @property
    def usable(self) -> bool:
        return self.independence >= 0.3


def _residuals(model, rows: pd.DataFrame, inputs, obs, sigmas, expansion,
               override: dict[int, float] | None = None) -> np.ndarray:
    from .runner import build_param_rows, simulate
    P = build_param_rows(model, rows[inputs], base_p=model.p0(), expansion=expansion)
    if override:
        for j, v in override.items():
            P[:, j] = v
    sim = simulate(model, P, obs, index=rows.index).values
    meas = rows[obs].to_numpy(dtype=float)
    sig = np.array([max(sigmas.get(c, 1.0), 1e-30) for c in obs])
    r = (sim[obs].to_numpy(dtype=float) - meas) / sig
    return np.where(np.isfinite(meas) & np.isfinite(r), r, 0.0).ravel()


def score_test(model, rows: pd.DataFrame, inputs: list[str], obs: list[str],
               sigmas: dict[str, float], expansion=None, selected: list[str] | None = None,
               candidates: list[str] | None = None, rel_step: float = 1e-3) -> list[ParamProposal]:
    """보정하지 않은 파라미터를 하나씩 풀었을 때 잔차가 얼마나 줄지 (1차 근사).

    기울기는 **상대 변화**당 잔차 변화로 잰다 (단위가 다른 파라미터를 같은 저울에).
    이미 보정한 파라미터(``selected``) 방향을 빼고 남는 부분만 본다 — 남는 게 거의
    없으면(독립성 < 0.3) 풀어도 데이터가 기존 파라미터와 구분하지 못한다.
    """
    selected = list(selected or [])
    if candidates is None:
        candidates = [p.name for p in model.tunable_params() if p.name not in selected]
    r0 = _residuals(model, rows, inputs, obs, sigmas, expansion)
    ssr = float(r0 @ r0)
    p0 = model.p0()
    cols: dict[str, np.ndarray] = {}
    for name in selected + list(candidates):
        j = model.par_index(name)
        h = rel_step * max(abs(p0[j]), 1e-12)
        r1 = _residuals(model, rows, inputs, obs, sigmas, expansion, {j: p0[j] + h})
        cols[name] = (r1 - r0) / h * max(abs(p0[j]), 1e-12)
    JS = np.column_stack([cols[s] for s in selected]) if selected else None
    out = []
    for c in candidates:
        g = cols[c]
        gp = g - JS @ np.linalg.lstsq(JS, g, rcond=None)[0] if JS is not None else g
        nrm = float(gp @ gp)
        base = float(g @ g)
        if base <= 0:
            continue                        # 관측에 전혀 영향이 없다
        free = float(np.sqrt(nrm / base))
        # 거의 완전 공선(독립성 < 0.05)이면 남는 방향이 수치 잡음뿐이라 '기대 감소'가 뜻이 없다
        gain = float((r0 @ gp) ** 2 / nrm) if nrm > 1e-300 and free >= 0.05 else 0.0
        info = model.parameters[model.par_index(c)]
        out.append(ParamProposal(c, 100.0 * gain / max(ssr, 1e-300), free,
                                 from_si(float(p0[model.par_index(c)]), info.unit), info.unit))
    out.sort(key=lambda p: (not p.usable, -p.gain_pct))
    return out


# ---------------------------------------------------------------------------
# 파라미터가 시간에 따라 움직이는가 (설비 상태 변화)
# ---------------------------------------------------------------------------

@dataclass
class ParamDrift:
    name: str
    unit: str
    overall: float
    segments: list[tuple[str, float, float]] = field(default_factory=list)  # (구간, 값, 표준오차)
    #: 토막별 값이 이 파라미터와 반대로 움직이는 파라미터 (서로 값을 주고받음 — 구분 안 됨)
    traded_with: list[str] = field(default_factory=list)

    @property
    def values(self) -> np.ndarray:
        return np.array([v for _, v, _ in self.segments])

    @property
    def change_pct(self) -> float:
        v = self.values
        if len(v) < 2 or self.overall == 0:
            return 0.0
        return 100.0 * (v[-1] - v[0]) / abs(self.overall)

    @property
    def rel_se(self) -> float:
        """토막별 상대표준오차의 중앙값."""
        se = np.array([s for _, _, s in self.segments])
        v = np.abs(self.values)
        rel = se / np.maximum(v, 1e-30)
        return float(np.nanmedian(rel)) if len(rel) else float("inf")

    @property
    def identified(self) -> bool:
        """토막 안에서 값이 결정되는가 (상대표준오차 중앙값 15% 미만)."""
        return self.rel_se < 0.15

    @property
    def significant(self) -> bool:
        v = self.values
        se = np.array([s for _, _, s in self.segments])
        if len(v) < 2 or not self.identified or self.traded_with:
            return False
        spread = float(v.max() - v.min())
        noise = float(np.nanmedian(se)) if np.isfinite(se).any() else 0.0
        return spread > 3.0 * noise and spread > 0.03 * abs(self.overall)

    @property
    def trend(self) -> str:
        if self.traded_with:
            return "상쇄"
        if not self.identified:
            return "미결정"
        if not self.significant:
            return "안정"
        d = np.diff(self.values)
        if len(d) >= 2 and (d > 0).all():
            return "계속 증가"
        if len(d) >= 2 and (d < 0).all():
            return "계속 감소"
        return "변동"


def parameter_drift(model, frame: pd.DataFrame, inputs: list[str], obs: list[str], spec,
                    expansion=None, n_segments: int = 4, rows_per_segment: int = 60,
                    prior_weight: float = 0.3) -> list[ParamDrift]:
    """학습 구간을 시간순 ``n_segments`` 토막으로 나눠 토막마다 다시 맞춘다.

    각 토막 보정은 전체 보정값 쪽으로 당기는 정칙화(``prior_weight``)를 건다 — 한 토막의
    좁은 운전범위에서 공선인 파라미터끼리 값을 주고받는 것을 막기 위해서다. 모델의
    파라미터 값은 끝나면 원래대로 돌려놓는다.
    """
    from dataclasses import replace

    from .estimator import calibrate

    idx = [model.par_index(n) for n in spec.params]
    saved = {n: model.parameters[j].value for n, j in zip(spec.params, idx)}
    units = [model.parameters[j].unit for j in idx]
    drift = [ParamDrift(n, u, from_si(saved[n], u)) for n, u in zip(spec.params, units)]
    t = frame.index
    edges = pd.date_range(t.min(), t.max(), periods=n_segments + 1)
    try:
        for k in range(n_segments):
            seg = frame[(t >= edges[k]) & ((t < edges[k + 1]) if k < n_segments - 1 else (t <= edges[k + 1]))]
            rows = select_rows(seg, rows_per_segment, inputs, obs)
            if len(rows) < max(len(spec.params) * 3, 12):
                continue
            _restore(model, saved)
            s = replace(spec, prior_weight=prior_weight, max_rows=10 ** 9)
            try:
                res = calibrate(model, rows[inputs], rows[obs], s, expansion=expansion, verbose=False)
            except Exception:  # noqa: BLE001 — 한 토막이 안 풀려도 나머지는 보고한다
                continue
            label = f"{edges[k]:%m/%d}~{edges[k + 1]:%m/%d}"
            for i, dr in enumerate(drift):
                dr.segments.append((label, from_si(float(res.fitted[i]), units[i]),
                                    from_si(float(res.stderr[i]), units[i])))
    finally:
        _restore(model, saved)
    # 토막별 값이 서로 반대로 움직이는 쌍은 '상태 변화'가 아니라 식별성 문제다
    # (한 토막의 좁은 운전범위에서 둘을 구분 못 해 값을 주고받는다)
    # 토막이 넷뿐이라 상관은 우연히도 높게 나온다. 토막 안에서 잘 결정되는 파라미터(상대
    # 표준오차 5% 미만 — 차압 계측이 있는 저항계수)는 주고받기로 보지 않는다.
    for a in drift:
        if a.rel_se < 0.05:
            continue
        for b in drift:
            if a is b or len(a.segments) < 3 or len(a.segments) != len(b.segments):
                continue
            va, vb = a.values, b.values
            if np.std(va) > 0 and np.std(vb) > 0 and np.corrcoef(va, vb)[0, 1] < -0.8:
                a.traded_with.append(b.name)
    return drift


def _restore(model, values: dict[str, float]) -> None:
    for name, v in values.items():
        model.parameters[model.par_index(name)].value = float(v)
        cname, pname = name.rsplit(".", 1)
        model.system.components[cname].set_param(pname, float(v))


# ---------------------------------------------------------------------------
# 학습에서 검증된 적 없는 외삽 방향
# ---------------------------------------------------------------------------

@dataclass
class Direction:
    column: str
    unit: str
    train: tuple[float, float]          # 학습 2~98 백분위 ('평소 범위')
    train_full: tuple[float, float]     # 학습 최소~최대
    future: tuple[float, float]
    frac_outside: float                 # 예측 행 중 평소 범위 밖
    kind: str                           # "고정" | "평소 고정" | "범위 밖"

    @property
    def untested(self) -> bool:
        """이 방향의 반응을 데이터가 한 번도(또는 거의) 본 적이 없다."""
        return self.kind in ("고정", "평소 고정")

    def short(self) -> str:
        f = lambda v: f"{v:.4g}"  # noqa: E731
        u = f" {self.unit}" if self.unit and self.unit != "1" else ""
        fut = f"{f(self.future[0])}~{f(self.future[1])}{u}" if self.future[0] != self.future[1] \
            else f"{f(self.future[0])}{u}"
        if self.kind == "고정":
            return f"{self.column}: 학습 내내 {f(self.train[0])}{u} → 예측 {fut}"
        if self.kind == "평소 고정":
            return (f"{self.column}: 학습 평소 {f(self.train[0])}~{f(self.train[1])}{u} "
                    f"(드물게 {f(self.train_full[0])}~{f(self.train_full[1])}) → 예측 {fut}")
        return (f"{self.column}: 학습 {f(self.train_full[0])}~{f(self.train_full[1])}{u} → 예측 {fut} "
                f"({self.frac_outside * 100:.0f}% 범위 밖)")

    def message(self) -> str:
        if self.untested:
            return (f"**{self.short()}** — 이 방향의 반응은 데이터로 검증된 적이 없습니다. "
                    "모델 구조(구성방정식)와 설계값을 그대로 믿는 외삽입니다.")
        return (f"{self.short()} — 학습에서 변한 적이 있는 방향이라 기울기는 데이터로 맞췄지만, "
                "범위 밖에서도 구성방정식의 모양이 맞는지는 검증되지 않았습니다.")


def untested_directions(train_inputs: pd.DataFrame, future_inputs: pd.DataFrame,
                        units: dict[str, str] | None = None,
                        min_outside: float = 0.2) -> list[Direction]:
    """예측 구간 입력이 학습의 '평소 범위'를 크게 벗어나는 컬럼.

    예측 구간의 **입력**만 본다 (생산계획·설정값처럼 미리 아는 값). 결과값은 보지 않는다.
    '평소 고정'은 학습 대부분에서 거의 일정하고 드문 순간(펌프 정지, PM)에만 변한 입력이다.
    그 사이의 반응 곡선은 데이터가 본 적이 없다.
    """
    units = units or {}
    out = []
    for c in train_inputs.columns:
        if c not in future_inputs.columns:
            continue
        tr = pd.to_numeric(train_inputs[c], errors="coerce").dropna()
        fu = pd.to_numeric(future_inputs[c], errors="coerce").dropna()
        if len(tr) < 10 or len(fu) == 0:
            continue
        p2, p98 = float(tr.quantile(0.02)), float(tr.quantile(0.98))
        lo, hi = float(tr.min()), float(tr.max())
        scale = max(abs(float(tr.median())), hi - lo, 1e-12)
        span = max(p98 - p2, 1e-9 * scale)
        outside = float(((fu < p2 - 0.05 * span) | (fu > p98 + 0.05 * span)).mean())
        if outside < min_outside:
            continue
        if hi - lo <= 1e-9 * scale:
            kind = "고정"
        elif p98 - p2 < 0.10 * scale:
            # 설정값 운전(순환수 33 m³/h ± 계측 잡음 1)은 '평소 범위'가 척도의 몇 % 에 그친다
            kind = "평소 고정"
        else:
            kind = "범위 밖"
        out.append(Direction(c, units.get(c, ""), (p2, p98), (lo, hi),
                             (float(fu.min()), float(fu.max())), outside, kind))
    order = {"고정": 0, "평소 고정": 1, "범위 밖": 2}
    out.sort(key=lambda d: (order[d.kind], -d.frac_outside))
    return out


# ---------------------------------------------------------------------------
# 예측이 어느 파라미터의 불확도에 민감한가
# ---------------------------------------------------------------------------

@dataclass
class ForecastSensitivity:
    target: str
    unit: str
    mean: float                     # 예측 평균 (표시 단위)
    sigma: float                    # 파라미터 불확도로 인한 예측 평균의 1σ (표시 단위)
    contributions: list[tuple[str, float]] = field(default_factory=list)   # (파라미터, 몫 %)

    def top(self, k: int = 3) -> str:
        return ", ".join(f"{n} {s:.0f}%" for n, s in self.contributions[:k])


def forecast_sensitivity(model, cal, inputs: pd.DataFrame, target: str, unit: str,
                         expansion=None, n_rows: int = 150,
                         rel_step: float = 1e-3) -> ForecastSensitivity:
    """보정 공분산(데이터 항만)을 ``inputs`` 조건의 예측 평균으로 전파한다 (1차 근사).

    같은 파라미터 불확도라도 어느 조건에서 예측하느냐에 따라 결과 불확도가 다르다.
    학습 조건과 예측 조건에서 각각 재면, 외삽이 불확도를 몇 배로 키우는지와 그 주범이
    나온다 — 다음에 어떤 데이터를 모아야 하는지가 바로 그것이다.

    시계열 잔차의 자기상관 때문에 공분산 자체는 과소평가일 수 있다. 크기보다 **비율과
    순위**를 보라.
    """
    from .runner import build_param_rows, simulate

    rows = inputs.dropna()
    if len(rows) > n_rows:
        rows = rows.iloc[:: max(1, len(rows) // n_rows)]
    P0 = build_param_rows(model, rows, base_p=model.p0(), expansion=expansion)
    y0 = simulate(model, P0, [target], index=rows.index).values[target].to_numpy()
    ok = np.isfinite(y0)
    g = np.zeros(len(cal.names))
    for k, name in enumerate(cal.names):
        j = model.par_index(name)
        h = rel_step * max(abs(float(cal.fitted[k])), 1e-30)
        P = P0.copy()
        P[:, j] = P[:, j] + h
        y = simulate(model, P, [target], index=rows.index).values[target].to_numpy()
        m = ok & np.isfinite(y)
        g[k] = float(np.mean((y[m] - y0[m]) / h)) if m.any() else 0.0
    se = np.nan_to_num(np.asarray(cal.stderr, dtype=float))
    corr = np.nan_to_num(np.asarray(cal.correlation, dtype=float))
    cov = corr * np.outer(se, se)
    var = float(max(g @ cov @ g, 0.0))
    indiv = (g * se) ** 2
    tot = float(indiv.sum())
    contrib = sorted(((n, 100.0 * v / tot) for n, v in zip(cal.names, indiv)), key=lambda t: -t[1]) \
        if tot > 0 else []
    conv = lambda v: from_si(v, unit)  # noqa: E731
    mean = float(np.mean([conv(v) for v in y0[ok]])) if ok.any() else float("nan")
    sigma = abs(conv(float(np.mean(y0[ok])) + np.sqrt(var)) - conv(float(np.mean(y0[ok])))) \
        if ok.any() else float("nan")
    return ForecastSensitivity(target, unit, mean, sigma, contrib)


def state_band(model, drift: list[ParamDrift], inputs: pd.DataFrame, target: str, unit: str,
               expansion=None, n_rows: int = 150) -> list[tuple[str, float]]:
    """학습 기간의 토막별 설비 상태(``parameter_drift`` 의 토막 보정값)로 각각 예측한 평균.

    파라미터 공분산은 '같은 상태를 얼마나 정확히 쟀나'만 말한다. 실제 예측 오차는 대개
    **상태가 변해서** 생긴다 (오염, 비계측 레시피 변경, 정비). 학습 기간에 실제로 있었던
    상태들로 미래를 각각 풀어 보면 그 폭이 나온다. 미래에 새로운 종류의 변화가 오면 이
    폭도 넘는다.
    """
    from .runner import build_param_rows, simulate

    if not drift or not drift[0].segments:
        return []
    labels = [lab for lab, _, _ in drift[0].segments]
    rows = inputs.dropna()
    if len(rows) > n_rows:
        rows = rows.iloc[:: max(1, len(rows) // n_rows)]
    P0 = build_param_rows(model, rows, base_p=model.p0(), expansion=expansion)
    out = []
    for k, lab in enumerate(labels):
        P = P0.copy()
        for d in drift:
            if len(d.segments) != len(labels):
                continue
            P[:, model.par_index(d.name)] = _to_si(d.segments[k][1], d.unit)
        y = simulate(model, P, [target], index=rows.index).values[target].to_numpy()
        y = y[np.isfinite(y)]
        if len(y):
            out.append((lab, float(np.mean([from_si(v, unit) for v in y]))))
    return out


def _to_si(v: float, unit: str) -> float:
    from ..core.units import to_si
    return to_si(v, unit)
