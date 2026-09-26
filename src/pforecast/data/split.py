"""학습/검증 분할 검사 — 미래 예측인가, 외삽인가, 새는 곳은 없는가.

물리모델과 ML 의 차이는 **학습 데이터에 없던 조건**에서만 드러난다. 그래서 비교 검증은
두 조건을 동시에 만족해야 한다.

1. **미래**: 검증 시점이 전부 학습 마지막 시점보다 뒤이고, 그 사이에 간격(엠바고)이
   있다. 가동율 값으로 자르면 학습과 검증이 시간상 뒤섞인다 — 같은 날의 계측 편향,
   같은 주의 오염 상태를 양쪽이 공유하므로 '미래를 맞혔다'고 말할 수 없다.
   (실제로 예제의 값 기준 분할은 검증 88% 가 학습보다 과거였다.)
2. **외삽**: 검증 시점의 운전 입력이 학습 포락선 밖이다. 시간만 미래이고 운전영역이
   같으면 ML 도 잘 맞힌다 — 그건 물리모델이 필요한 질문이 아니다.

이 모듈은 판정만 한다. 분할을 바꾸라고 강제하지 않지만, 어기면 보고서 맨 위에 적는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: 검증 행이 이 비율 이상 학습 포락선 밖이어야 '외삽 검증'이라 부른다
MIN_EXTRAPOLATION = 0.5


@dataclass
class SplitCheck:
    n_train: int
    n_test: int
    train_end: object = None
    test_start: object = None
    n_test_before_train_end: int = 0
    embargo_days: float = float("nan")
    #: 검증 행 중 학습 포락선 밖 비율 (운전 입력 기준)
    extrapolation: float = float("nan")
    per_column: dict[str, float] = field(default_factory=dict)
    outside: np.ndarray | None = None         # 검증 행별 외삽 여부
    ranges: dict[str, tuple] = field(default_factory=dict)   # 컬럼 -> (학습 lo, hi, 검증 lo, hi)

    @property
    def is_future(self) -> bool:
        return self.n_test_before_train_end == 0 and self.n_test > 0

    @property
    def is_extrapolation(self) -> bool:
        return np.isfinite(self.extrapolation) and self.extrapolation >= MIN_EXTRAPOLATION

    def warnings(self) -> list[str]:
        out = []
        if not self.is_future:
            out.append(
                f"**미래 예측 검증이 아닙니다.** 검증 {self.n_test:,}행 중 "
                f"{self.n_test_before_train_end:,}행이 학습 마지막 시점({self.train_end})보다 "
                "과거입니다. 학습과 검증이 시간상 섞여 있으면 같은 시기의 계측 편향·설비 상태를 "
                "양쪽이 공유합니다. 시간순으로 자르세요 (train: 'index < ...', test: 'index >= ...').")
        elif np.isfinite(self.embargo_days) and self.embargo_days < 1.0:
            out.append(
                f"학습 끝과 검증 시작 사이 간격이 {self.embargo_days * 24:.1f}시간입니다. "
                "하루 이상 띄우는 것을 권합니다 (경계 부근은 거의 같은 상태라 검증이 쉬워진다).")
        if np.isfinite(self.extrapolation) and not self.is_extrapolation:
            out.append(
                f"**외삽 검증이 아닙니다.** 검증 행의 {self.extrapolation * 100:.0f}% 만 학습 "
                "운전영역 밖입니다. 학습 범위 안에서는 ML 도 잘 맞히므로 물리모델과의 차이가 "
                "드러나지 않습니다. 운전 조건이 학습 때와 달라진 뒤의 기간을 검증으로 잡으세요.")
        return out

    def lines(self) -> list[str]:
        out = [f"학습 {self.n_train:,}행 (~{self.train_end}) → 검증 {self.n_test:,}행 "
               f"({self.test_start}~), 간격 {self.embargo_days:.1f}일"
               if np.isfinite(self.embargo_days) else
               f"학습 {self.n_train:,}행, 검증 {self.n_test:,}행"]
        if np.isfinite(self.extrapolation):
            top = sorted(self.per_column.items(), key=lambda kv: -kv[1])[:4]
            out.append(f"검증 행의 {self.extrapolation * 100:.0f}% 가 학습 운전영역 밖 — "
                       + ", ".join(f"{c} {v * 100:.0f}%" for c, v in top if v > 0))
        return out

    def to_dict(self) -> dict:
        return {"n_train": self.n_train, "n_test": self.n_test,
                "train_end": str(self.train_end), "test_start": str(self.test_start),
                "n_test_before_train_end": self.n_test_before_train_end,
                "embargo_days": self.embargo_days, "is_future": self.is_future,
                "extrapolation": self.extrapolation, "is_extrapolation": self.is_extrapolation,
                "per_column": self.per_column, "warnings": self.warnings(), "lines": self.lines()}


def check_split(train: pd.DataFrame, test: pd.DataFrame,
                inputs: list[str] | None = None) -> SplitCheck:
    """분할이 미래·외삽 검증인지 판정한다. ``inputs`` 는 운전 입력(포락선을 잴 컬럼)."""
    chk = SplitCheck(n_train=len(train), n_test=len(test))
    if isinstance(train.index, pd.DatetimeIndex) and isinstance(test.index, pd.DatetimeIndex) \
            and len(train) and len(test):
        chk.train_end, chk.test_start = train.index.max(), test.index.min()
        chk.n_test_before_train_end = int((test.index <= chk.train_end).sum())
        chk.embargo_days = float((chk.test_start - chk.train_end) / pd.Timedelta(days=1))
    cols = [c for c in (inputs or []) if c in train.columns and c in test.columns]
    if cols and len(test):
        outside = np.zeros(len(test), dtype=bool)
        for c in cols:
            tr = pd.to_numeric(train[c], errors="coerce").dropna()
            te = pd.to_numeric(test[c], errors="coerce").to_numpy(dtype=float)
            if not len(tr):
                continue
            lo, hi = float(tr.min()), float(tr.max())
            tol = 1e-9 * max(abs(lo), abs(hi), 1.0)
            out_c = np.isfinite(te) & ((te < lo - tol) | (te > hi + tol))
            chk.per_column[c] = float(out_c.mean())
            chk.ranges[c] = (lo, hi, float(np.nanmin(te)) if np.isfinite(te).any() else np.nan,
                             float(np.nanmax(te)) if np.isfinite(te).any() else np.nan)
            outside |= out_c
        chk.outside = outside
        chk.extrapolation = float(outside.mean())
    return chk


def future_split(index: pd.DatetimeIndex, train_fraction: float = 0.6,
                 embargo: pd.Timedelta | str = "1D") -> tuple[np.ndarray, np.ndarray]:
    """시간순 분할 마스크: 앞 ``train_fraction`` 학습, 엠바고 뒤부터 검증."""
    embargo = pd.Timedelta(embargo)
    n = len(index)
    cut = index[min(max(int(n * train_fraction), 1), n - 1)]
    return np.asarray(index < cut), np.asarray(index >= cut + embargo)
