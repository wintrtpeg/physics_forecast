"""비교용 데이터 기반 기준 모델.

물리 모델의 가치를 주장하려면 같은 데이터로 학습한 통계 모델과 나란히 놓고
**외삽 구간에서** 비교해야 한다. 여기 있는 모델은 의도적으로 단순하다
(다항 릿지 회귀). 요점은 모델의 정교함이 아니라, 학습 분포 밖에서는 어떤
데이터 기반 모델이든 근거 없는 외삽을 한다는 구조적 한계이기 때문이다.

GPU 나 무거운 프레임워크가 필요 없도록 numpy 최소제곱만 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _design(X: np.ndarray, degree: int) -> np.ndarray:
    """1, x_i, x_i*x_j (degree>=2), x_i^3 (degree>=3) 형태의 설계행렬."""
    n, k = X.shape
    cols = [np.ones(n)]
    cols.extend(X[:, i] for i in range(k))
    if degree >= 2:
        for i in range(k):
            for j in range(i, k):
                cols.append(X[:, i] * X[:, j])
    if degree >= 3:
        for i in range(k):
            cols.append(X[:, i] ** 3)
    return np.column_stack(cols)


@dataclass
class PolyRidgeBaseline:
    """다항 릿지 회귀 + 학습영역(convex hull 근사) 이탈 판정."""

    degree: int = 2
    alpha: float = 1e-6
    coef_: np.ndarray | None = None
    x_mean_: np.ndarray | None = None
    x_std_: np.ndarray | None = None
    train_lo_: np.ndarray | None = None
    train_hi_: np.ndarray | None = None
    feature_names_: list[str] = field(default_factory=list)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X, y = X[ok], y[ok]
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = np.where(X.std(axis=0) > 1e-12, X.std(axis=0), 1.0)
        Z = _design((X - self.x_mean_) / self.x_std_, self.degree)
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.coef_ = np.linalg.solve(A, Z.T @ y)
        self.train_lo_ = X.min(axis=0)
        self.train_hi_ = X.max(axis=0)
        self.feature_names_ = list(feature_names or [f"x{i}" for i in range(X.shape[1])])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Z = _design((X - self.x_mean_) / self.x_std_, self.degree)
        return Z @ self.coef_

    def extrapolation_flags(self, X: np.ndarray) -> np.ndarray:
        """학습 구간 밖으로 나간 행을 표시한다."""
        X = np.asarray(X, dtype=float)
        return np.any((X < self.train_lo_) | (X > self.train_hi_), axis=1)

    def extrapolation_distance(self, X: np.ndarray) -> np.ndarray:
        """학습 구간 밖으로 얼마나 멀리 나갔는지 (구간 폭 대비 배수)."""
        X = np.asarray(X, dtype=float)
        span = np.maximum(self.train_hi_ - self.train_lo_, 1e-12)
        below = np.clip(self.train_lo_ - X, 0, None) / span
        above = np.clip(X - self.train_hi_, 0, None) / span
        return np.max(below + above, axis=1)
