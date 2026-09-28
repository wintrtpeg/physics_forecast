"""물리 모델 잔차로 '설명되지 않는 변화'를 찾는다.

원시 신호의 계단은 공정 변화(증설, 설정 변경)와 계측 문제(센서 교체, 영점 이동)를
구분하지 못한다 — 둘 다 똑같이 계단이다. 물리 모델은 **입력 변화가 만드는 출력 변화를
설명한다.** 그러니 잔차(측정 − 모델)에 남은 계단은 모델이 모르는 변화다: 센서 교체,
레시피 변경, 충전재 오염과 세정 같은 것들.

이건 ML 로는 하기 어려운 일이다. ML 은 변화를 '배워 버리거나'(학습 구간 안) '설명 못
하거나'(밖) 둘 중 하나이고, 어느 쪽인지 스스로 말하지 않는다.

방법
----
* 관측별 잔차를 **하루 중앙값**으로 모은다. 5분 잔차는 강하게 자기상관되어 있고
  (교정 창·펄스·드리프트), 하루 중앙값이면 그 대부분이 가라앉는다.
* 이진 분할로 평균이 바뀌는 날을 찾는다. 계단 크기를 일간 잡음(1차 차분의 MAD)으로
  나눈 z 로 판정하고, 양쪽 구간이 최소 길이 이상이어야 한다.
* 같은 날 다른 관측도 움직였는지, 그날 입력이 바뀌었는지를 붙여서 해석한다.
  - 이 관측만 움직였다 → 그 계측기(교체·영점) 또는 이 관측에만 닿는 공정 변화
  - 입력이 바뀐 날이다 → 모델이 그 변화의 효과를 다르게 예측한다 (구성방정식 점검)
  - 여러 관측이 같이 → 공정 자체의 변화 (모델 밖 요인)

해석은 **후보를 좁혀 주는 것**이지 판결이 아니다. 같은 날 정비 이력을 확인하라고 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ChangePoint:
    observation: str
    date: pd.Timestamp
    before: float            # 계단 앞 구간의 잔차 평균 (표시 단위)
    after: float
    z: float                 # 계단 크기 / 일간 잡음
    unit: str = ""
    also: list[str] = field(default_factory=list)
    inputs_changed: list[str] = field(default_factory=list)

    @property
    def shift(self) -> float:
        return self.after - self.before

    @property
    def kind(self) -> str:
        if self.inputs_changed:
            return "입력 변경일"
        if self.also:
            return "공정 변화"
        return "단독 변화"

    def interpretation(self) -> str:
        s = f"{self.shift:+.3g}{self.unit}"
        if self.inputs_changed:
            return (f"같은 날 {', '.join(self.inputs_changed)} 가 바뀌었습니다. 모델이 그 변화의 "
                    f"효과를 실제와 {s} 다르게 예측합니다 — 해당 구성방정식이나 파라미터를 "
                    "점검하세요.")
        if self.also:
            return (f"{', '.join(self.also)} 도 같은 날 움직였습니다 — 공정 자체가 바뀐 것으로 "
                    "보입니다 (오염/세정, 설비 교체 등 모델 밖 요인).")
        return (f"이 관측만 {s} 움직였고 입력은 그대로입니다 — 계측기 교체·영점 이동이거나, "
                "이 관측에만 닿는 공정 변화(예: 배출원 레시피)입니다. 그날 정비 이력을 먼저 "
                "확인하세요.")

    def to_dict(self) -> dict:
        return {"observation": self.observation, "date": str(self.date.date()),
                "before": self.before, "after": self.after, "shift": self.shift,
                "z": self.z, "unit": self.unit, "also": self.also,
                "inputs_changed": self.inputs_changed, "kind": self.kind,
                "interpretation": self.interpretation()}


def _robust_sigma(y: np.ndarray) -> float:
    d = np.diff(y)
    if len(d) < 3:
        return float("nan")
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def binary_segmentation(y: np.ndarray, sigma: float, min_seg: int = 5, z_min: float = 5.0,
                        max_cp: int = 8) -> list[tuple[int, float]]:
    """평균이 바뀌는 위치를 이진 분할로 찾는다. [(위치, z)] (위치 오름차순)."""
    y = np.asarray(y, dtype=float)
    if not np.isfinite(sigma) or sigma <= 0 or len(y) < 2 * min_seg:
        return []
    segs = [(0, len(y))]
    found: list[tuple[int, float]] = []
    while len(found) < max_cp:
        best = None
        for a, b in segs:
            n = b - a
            if n < 2 * min_seg:
                continue
            cs = np.cumsum(y[a:b])
            k = np.arange(min_seg, n - min_seg + 1)
            m1 = cs[k - 1] / k
            m2 = (cs[-1] - cs[k - 1]) / (n - k)
            z = np.abs(m2 - m1) / (sigma * np.sqrt(1.0 / k + 1.0 / (n - k)))
            j = int(np.argmax(z))
            if best is None or z[j] > best[0]:
                best = (float(z[j]), a + int(k[j]), (a, b))
        if best is None or best[0] < z_min:
            break
        z, cp, seg = best
        found.append((cp, z))
        segs.remove(seg)
        segs += [(seg[0], cp), (cp, seg[1])]
    return sorted(found)


def _daily(series: pd.Series, min_count: int) -> pd.Series:
    g = series.dropna().groupby(series.dropna().index.normalize())
    med = g.median()
    return med[g.count() >= min_count]


def residual_changepoints(residuals: pd.DataFrame, inputs: pd.DataFrame | None = None,
                          units: dict[str, str] | None = None, *, min_seg_days: int = 5,
                          z_min: float = 5.0, max_cp: int = 6, min_count: int = 48,
                          tol_days: int = 1,
                          min_shift: dict[str, float] | None = None) -> list[ChangePoint]:
    """관측별 잔차(측정 − 모델, 표시 단위)에서 계단 변화를 찾아 해석을 붙인다.

    ``min_shift`` 는 관측별 '의미 있는 계단'의 하한(표시 단위)이다. 통계적 유의성만
    보면 하루 중앙값이 워낙 조용해서 0.5 mmAq 계단도 z=50 이 나온다 — 오염이 서서히
    진행되는 램프를 계단 여러 개로 쪼갠 것이다. 계측 불확도의 절반 정도를 쓴다.
    """
    units = units or {}
    min_shift = min_shift or {}
    daily = {c: _daily(residuals[c], min_count) for c in residuals.columns}
    cps: list[ChangePoint] = []
    for c, d in daily.items():
        if len(d) < 2 * min_seg_days:
            continue
        y = d.to_numpy()
        sigma = _robust_sigma(y)
        for pos, z in binary_segmentation(y, sigma, min_seg_days, z_min, max_cp):
            # 계단 앞뒤 값은 이웃 변화점 사이 구간 평균이 아니라 가까운 두 주의 중앙값으로
            # 낸다 (드리프트가 있으면 긴 구간 평균은 계단 크기를 흐린다)
            before = float(np.median(y[max(0, pos - 14):pos]))
            after = float(np.median(y[pos:pos + 14]))
            if abs(after - before) < min_shift.get(c, 0.0):
                continue
            u = units.get(c, "")
            cps.append(ChangePoint(c, d.index[pos], before, after, z,
                                   unit=f" {u}" if u else ""))

    # 같은 날 움직인 다른 관측
    for cp in cps:
        cp.also = sorted({o.observation for o in cps if o is not cp and o.observation !=
                          cp.observation and abs((o.date - cp.date).days) <= tol_days})

    # 같은 날 바뀐 입력 (하루 중앙값이 앞뒤 사흘과 뚜렷이 다르면)
    if inputs is not None and len(cps):
        dail = {c: inputs[c].dropna().groupby(inputs[c].dropna().index.normalize()).median()
                for c in inputs.columns}
        for cp in cps:
            changed = []
            for c, d in dail.items():
                if len(d) < 8:
                    continue
                w0 = d[(d.index >= cp.date - pd.Timedelta(days=4))
                       & (d.index < cp.date - pd.Timedelta(days=tol_days - 1))]
                w1 = d[(d.index > cp.date + pd.Timedelta(days=tol_days - 1))
                       & (d.index <= cp.date + pd.Timedelta(days=4))]
                if len(w0) < 2 or len(w1) < 2:
                    continue
                s = _robust_sigma(d.to_numpy())
                step = abs(float(w1.median()) - float(w0.median()))
                span = float(d.quantile(0.95) - d.quantile(0.05))
                # 설정값(계단 신호)은 잡음이 0 이라 z 가 무한대가 된다. 범위 대비 크기로도 본다.
                if step > max(5.0 * (s if np.isfinite(s) else 0.0), 0.05 * span, 1e-12):
                    changed.append(c)
            cp.inputs_changed = changed
    cps.sort(key=lambda c: (c.date, -c.z))
    return cps
