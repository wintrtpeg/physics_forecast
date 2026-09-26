"""데이터셋 프로파일링: 분석 전에 데이터가 무엇인지 먼저 본다.

여기서 나오는 것들이 뒤 단계의 판단 근거가 된다.

* **운전 포락선** — 어디까지가 학습 범위인가. 외삽 경고의 기준.
* **준정상 구간** — 물리 모델이 준정상 가정을 쓰므로, 급변 구간은 빼야 한다.
* **상수/중복 컬럼** — 분석에 넣으면 중요도가 엉킨다.
* **결측/통신두절** — 현장 데이터에는 반드시 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    unit: str = ""
    n_missing: int = 0
    missing_pct: float = 0.0
    n_unique: int = 0
    vmin: float = np.nan
    vmax: float = np.nan
    mean: float = np.nan
    std: float = np.nan
    cv: float = np.nan           # 변동계수 - 변동이 없으면 아무것도 식별 못 한다
    span_ratio: float = np.nan   # max/min. 외삽 판정과 식별성에 직결
    is_constant: bool = False
    duplicate_of: str | None = None

    @property
    def usable(self) -> bool:
        return not self.is_constant and self.duplicate_of is None and self.missing_pct < 50.0


@dataclass
class DatasetProfile:
    n_rows: int
    t_start: object = None
    t_end: object = None
    interval_s: float = np.nan
    gap_count: int = 0
    columns: dict[str, ColumnProfile] = field(default_factory=dict)
    steady_fraction: float = np.nan
    warnings: list[str] = field(default_factory=list)

    def usable_columns(self) -> list[str]:
        return [n for n, c in self.columns.items() if c.usable]

    def table(self) -> pd.DataFrame:
        rows = []
        for c in self.columns.values():
            rows.append({
                "컬럼": c.name, "단위": c.unit, "결측[%]": c.missing_pct,
                "최소": c.vmin, "최대": c.vmax, "평균": c.mean,
                "변동계수": c.cv, "최대/최소": c.span_ratio,
                "상태": ("상수" if c.is_constant else
                         f"중복({c.duplicate_of})" if c.duplicate_of else "사용"),
            })
        return pd.DataFrame(rows)


def steady_state_mask(df: pd.DataFrame, columns: list[str], window: int = 6,
                      z: float = 2.0) -> np.ndarray:
    """변화율이 작은(준정상) 시점을 표시한다.

    각 컬럼의 1차 차분을 자기 표준편차로 정규화하고, 창 안에서 그 크기가 임계 이하인
    구간만 정상으로 본다. 기동/정지/급변 구간을 빼는 용도다.
    """
    if not columns:
        return np.ones(len(df), dtype=bool)
    ok = np.ones(len(df), dtype=bool)
    for c in columns:
        x = pd.to_numeric(df[c], errors="coerce")
        d = x.diff().abs()
        # 창 안에서 **가장 큰** 변화를 본다. 평균을 쓰면 계단 하나가 희석되어 묻힌다.
        s = d.rolling(window, min_periods=1, center=True).max()
        # 스케일은 '평소 변화량'. 표준편차를 쓰면 계단 자체가 스케일을 키워 자기를 숨긴다.
        scale = float(np.nanpercentile(d.to_numpy(), 75)) if d.notna().any() else 0.0
        sv = s.to_numpy(dtype=float)
        ok &= (sv <= z * scale) | ~np.isfinite(sv)
    return ok


def profile_dataset(df: pd.DataFrame, units: dict[str, str] | None = None,
                    duplicate_corr: float = 0.9999) -> DatasetProfile:
    units = units or {}
    num = df.select_dtypes(include=[np.number])
    prof = DatasetProfile(n_rows=len(df))

    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        prof.t_start, prof.t_end = df.index.min(), df.index.max()
        deltas = np.diff(df.index.to_numpy().astype("datetime64[s]").astype("int64"))
        if len(deltas):
            prof.interval_s = float(np.median(deltas))
            prof.gap_count = int(np.sum(deltas > 3 * prof.interval_s))
        # 규칙 격자(빠진 시각을 빈 행으로 채운 것)에서는 '전부 빈 행'이 이어진 곳이 공백이다
        empty = num.isna().all(axis=1).to_numpy().astype(np.int8)
        if empty.any():
            d = np.diff(np.r_[0, empty, 0])
            lengths = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
            prof.gap_count += int(np.sum(lengths >= 3))

    for name in num.columns:
        x = num[name]
        finite = x.dropna()
        c = ColumnProfile(
            name=name, unit=units.get(name, ""),
            n_missing=int(x.isna().sum()),
            missing_pct=100.0 * float(x.isna().mean()),
            n_unique=int(finite.nunique()),
        )
        if len(finite):
            c.vmin, c.vmax = float(finite.min()), float(finite.max())
            c.mean, c.std = float(finite.mean()), float(finite.std())
            c.cv = c.std / abs(c.mean) if abs(c.mean) > 1e-30 else np.nan
            if c.vmin > 0:
                c.span_ratio = c.vmax / c.vmin
            c.is_constant = bool(c.std < 1e-12 or c.n_unique <= 1)
        prof.columns[name] = c

    # 사실상 같은 컬럼 찾기 (중복을 넣으면 중요도가 반씩 갈린다)
    usable = [n for n, c in prof.columns.items() if not c.is_constant]
    if len(usable) > 1:
        corr = num[usable].corr().abs()
        for i, a in enumerate(usable):
            for b in usable[i + 1:]:
                if prof.columns[b].duplicate_of:
                    continue
                v = corr.loc[a, b]
                if np.isfinite(v) and v >= duplicate_corr:
                    prof.columns[b].duplicate_of = a

    live = [n for n, c in prof.columns.items() if c.usable]
    if live:
        prof.steady_fraction = float(steady_state_mask(num, live).mean())

    for n, c in prof.columns.items():
        if c.is_constant:
            prof.warnings.append(f"{n} 은 값이 변하지 않습니다 — 어떤 영향도 식별할 수 없습니다.")
        elif c.duplicate_of:
            prof.warnings.append(f"{n} 은 {c.duplicate_of} 와 사실상 같습니다 (상관 ≥ 0.9999).")
        elif c.missing_pct > 20:
            prof.warnings.append(f"{n} 의 결측이 {c.missing_pct:.0f}% 입니다.")
        elif np.isfinite(c.cv) and c.cv < 0.01:
            prof.warnings.append(
                f"{n} 의 변동계수가 {c.cv*100:.2f}% 입니다 — 거의 일정해서 영향을 "
                "분리하기 어렵습니다.")
    if prof.gap_count:
        prof.warnings.append(f"시간 간격이 벌어진 구간이 {prof.gap_count}곳 있습니다 (통신두절 추정).")
    return prof


@dataclass
class Envelope:
    """학습에 쓴 운전 범위. 예측 시점이 여기를 벗어나면 외삽이다."""

    lo: dict[str, float]
    hi: dict[str, float]

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str], q: float = 0.0) -> "Envelope":
        lo, hi = {}, {}
        for c in columns:
            x = pd.to_numeric(df[c], errors="coerce").dropna()
            if not len(x):
                continue
            lo[c] = float(x.quantile(q)) if q > 0 else float(x.min())
            hi[c] = float(x.quantile(1 - q)) if q > 0 else float(x.max())
        return cls(lo=lo, hi=hi)

    def distance(self, df: pd.DataFrame) -> np.ndarray:
        """포락선 밖으로 나간 정도 (구간 폭 대비 배수). 0 이면 범위 안."""
        out = np.zeros(len(df))
        for c, lo in self.lo.items():
            if c not in df.columns:
                continue
            hi = self.hi[c]
            span = max(hi - lo, 1e-30)
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            below = np.clip(lo - x, 0, None) / span
            above = np.clip(x - hi, 0, None) / span
            out = np.maximum(out, np.nan_to_num(below + above))
        return out

    def outside_fraction(self, df: pd.DataFrame) -> float:
        return float((self.distance(df) > 0).mean()) if len(df) else 0.0
