"""데이터 기반 대리모델과 XAI.

물리 구조를 모르는 상태에서 먼저 답을 내야 할 때 쓰는 사다리의 아래쪽 칸이다.
다만 **여기서 나온 답이 어디까지 유효한지**를 반드시 같이 낸다.

정직하게 붙여 둔 장치들
* **시간 블록 교차검증** — 무작위 k-fold 를 쓰면 5분 데이터는 앞뒤가 거의 같아서
  성능이 터무니없이 좋게 나온다. 시간순으로 잘라야 한다.
* **지연(lag) 탐색** — 물리 계통에는 수송 지연이 있다. 상호상관으로 찾아 반영한다.
* **외삽 거리** — 예측 시점이 학습 포락선에서 얼마나 벗어났는지 행마다 표시한다.
* **상관 얽힘 표시** — 순열 중요도는 서로 상관된 변수에서 신뢰할 수 없다. 각 변수의
  최대 상관 상대를 같이 보여줘서 오독을 막는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .profile import Envelope


def _fit_model(X: np.ndarray, y: np.ndarray, seed: int = 0):
    """설치된 것 중 가장 나은 회귀기를 쓴다. sklearn 이 없으면 다항 릿지로 떨어진다."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, max_depth=None, min_samples_leaf=20,
            l2_regularization=1.0, random_state=seed)
        m.fit(X, y)
        return m, "HistGradientBoosting"
    except ImportError:
        from ..calib.baseline import PolyRidgeBaseline
        m = PolyRidgeBaseline(degree=2, alpha=1e-6).fit(X, y)
        return m, "PolyRidge(2차)"


@dataclass
class LagInfo:
    feature: str
    best_lag: int          # 양수 = 피처가 타깃보다 앞선다 (수송 지연)
    corr_at_lag: float
    corr_at_zero: float


@dataclass
class FeatureCluster:
    """상관이 높아 서로 구분되지 않는 피처 묶음."""

    members: list[str]
    importance: float = 0.0
    importance_pct: float = 0.0
    max_internal_corr: float = 0.0

    @property
    def label(self) -> str:
        if len(self.members) == 1:
            return self.members[0]
        return f"{self.members[0]} 외 {len(self.members)-1}개"

    @property
    def is_group(self) -> bool:
        return len(self.members) > 1


@dataclass
class Importance:
    feature: str
    importance: float          # 순열 후 RMSE 증가분
    importance_pct: float
    max_corr_with: str = ""
    max_corr: float = 0.0

    @property
    def entangled(self) -> bool:
        return abs(self.max_corr) > 0.8


@dataclass
class SurrogateResult:
    model_name: str
    features: list[str]
    target: str
    unit: str = ""
    rmse_cv: float = np.nan
    mae_cv: float = np.nan
    r2_cv: float = np.nan
    rmse_train: float = np.nan
    baseline_rmse: float = np.nan     # 평균으로 예측했을 때
    importances: list[Importance] = field(default_factory=list)
    clusters: list[FeatureCluster] = field(default_factory=list)
    lags: list[LagInfo] = field(default_factory=list)
    envelope: Envelope | None = None
    pred: pd.Series | None = None
    fold_scores: list[float] = field(default_factory=list)
    model: object = None

    @property
    def skill(self) -> float:
        """평균 예측 대비 개선율. 0 이하면 모델이 아무 값도 못 한다."""
        if not np.isfinite(self.baseline_rmse) or self.baseline_rmse <= 0:
            return np.nan
        return 1.0 - self.rmse_cv / self.baseline_rmse

    def cluster_table(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "인자 묶음": c.label, "구성": ", ".join(c.members),
            "묶음 중요도[%]": c.importance_pct,
            "내부 최대상관": c.max_internal_corr if c.is_group else np.nan,
            "개별 분해": "불가 (서로 구분 안 됨)" if c.is_group else "가능",
        } for c in self.clusters])

    def importance_table(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "피처": i.feature, "중요도[%]": i.importance_pct,
            f"RMSE 증가": i.importance,
            "최대상관 상대": i.max_corr_with, "상관": i.max_corr,
            "해석주의": "다른 변수와 얽힘" if i.entangled else "",
        } for i in self.importances])


def find_lags(df: pd.DataFrame, target: str, features: list[str],
              max_lag: int = 12) -> list[LagInfo]:
    """피처를 앞뒤로 밀어 보며 상관이 가장 커지는 지연을 찾는다."""
    y = pd.to_numeric(df[target], errors="coerce")
    out = []
    for f in features:
        x = pd.to_numeric(df[f], errors="coerce")
        best, best_c = 0, 0.0
        c0 = float(x.corr(y)) if x.std() > 0 else 0.0
        for lag in range(-max_lag, max_lag + 1):
            c = x.shift(lag).corr(y)
            if np.isfinite(c) and abs(c) > abs(best_c):
                best, best_c = lag, float(c)
        out.append(LagInfo(feature=f, best_lag=best, corr_at_lag=best_c,
                           corr_at_zero=c0 if np.isfinite(c0) else 0.0))
    return out


def time_blocked_folds(n: int, k: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """시간순 블록 교차검증. 각 블록을 한 번씩 검증에 쓰고 나머지로 학습한다."""
    idx = np.arange(n)
    bounds = np.linspace(0, n, k + 1).astype(int)
    folds = []
    for i in range(k):
        te = idx[bounds[i]:bounds[i + 1]]
        tr = np.concatenate([idx[:bounds[i]], idx[bounds[i + 1]:]])
        if len(te) and len(tr):
            folds.append((tr, te))
    return folds


def permutation_importance(model, X: np.ndarray, y: np.ndarray, features: list[str],
                           n_repeats: int = 5, seed: int = 0,
                           groups: list[list[int]] | None = None) -> list[float]:
    """피처(또는 피처 묶음)를 섞었을 때 RMSE 가 얼마나 나빠지는가.

    ``groups`` 를 주면 묶음 안의 열을 **같은 순열로 함께** 섞는다. 묶음 내부의
    상관 구조는 유지한 채 묶음 전체와 타깃의 연결만 끊으므로, 서로 구분되지 않는
    변수들의 기여를 하나로 합쳐 정직하게 잴 수 있다.
    """
    rng = np.random.default_rng(seed)
    base = float(np.sqrt(np.mean((model.predict(X) - y) ** 2)))
    idx_groups = groups if groups is not None else [[j] for j in range(len(features))]
    out = []
    for cols in idx_groups:
        scores = []
        for _ in range(n_repeats):
            Xp = X.copy()
            order = rng.permutation(len(Xp))       # 묶음은 같은 순열로 함께 섞는다
            for j in cols:
                Xp[:, j] = Xp[order, j]
            scores.append(float(np.sqrt(np.mean((model.predict(Xp) - y) ** 2))))
        out.append(max(float(np.mean(scores)) - base, 0.0))
    return out


def correlation_clusters(df: pd.DataFrame, features: list[str],
                         threshold: float = 0.9) -> list[list[str]]:
    """|상관| 이 임계 이상인 피처를 연결요소로 묶는다.

    개별 순열 중요도는 상관된 변수들 사이에서 **기여를 임의로 나눠 갖는다**.
    실제로 이 예제에서 NOx 발생량이 가장 적은 장비군이 1위로 올라온 적이 있다
    (네 가동율이 서로 0.999 상관). 묶어서 재야 정직하다.
    """
    corr = df[features].corr().abs()
    parent = {f: f for f in features}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(features):
        for b in features[i + 1:]:
            v = corr.loc[a, b]
            if np.isfinite(v) and v >= threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    buckets: dict[str, list[str]] = {}
    for f in features:
        buckets.setdefault(find(f), []).append(f)
    return [sorted(v, key=features.index) for v in buckets.values()]


def partial_dependence(model, X: np.ndarray, j: int, grid: np.ndarray,
                       sample: int = 400, seed: int = 0) -> np.ndarray:
    """1차원 부분의존도. 다른 변수는 실제 분포에서 표본추출해 평균낸다."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    Xs = X[idx]
    out = []
    for v in grid:
        Xc = Xs.copy()
        Xc[:, j] = v
        out.append(float(np.mean(model.predict(Xc))))
    return np.array(out)


def fit_surrogate(df: pd.DataFrame, target: str, features: list[str], unit: str = "",
                  n_folds: int = 5, apply_lags: bool = True, max_lag: int = 12,
                  seed: int = 0, cluster_threshold: float = 0.9) -> SurrogateResult:
    """대리모델 학습 + 시간블록 교차검증 + 순열 중요도."""
    lags = find_lags(df, target, features, max_lag) if apply_lags else []
    work = df[[target] + features].copy()
    used = list(features)
    if apply_lags:
        for li in lags:
            if li.best_lag != 0 and abs(li.corr_at_lag) > abs(li.corr_at_zero) * 1.02:
                work[li.feature] = work[li.feature].shift(li.best_lag)
    work = work.dropna()
    if len(work) < 50:
        raise ValueError(f"유효한 행이 {len(work)}개뿐입니다. 결측/지연 설정을 확인하세요.")

    X = work[used].to_numpy(dtype=float)
    y = work[target].to_numpy(dtype=float)

    preds = np.full(len(y), np.nan)
    fold_scores = []
    for tr, te in time_blocked_folds(len(y), n_folds):
        m, _ = _fit_model(X[tr], y[tr], seed)
        p = m.predict(X[te])
        preds[te] = p
        fold_scores.append(float(np.sqrt(np.mean((p - y[te]) ** 2))))

    model, model_name = _fit_model(X, y, seed)
    ok = np.isfinite(preds)
    rmse_cv = float(np.sqrt(np.mean((preds[ok] - y[ok]) ** 2)))
    ss_tot = float(np.sum((y[ok] - y[ok].mean()) ** 2))
    res = SurrogateResult(
        model_name=model_name, features=used, target=target, unit=unit,
        rmse_cv=rmse_cv, mae_cv=float(np.mean(np.abs(preds[ok] - y[ok]))),
        r2_cv=1.0 - float(np.sum((preds[ok] - y[ok]) ** 2)) / ss_tot if ss_tot > 0 else np.nan,
        rmse_train=float(np.sqrt(np.mean((model.predict(X) - y) ** 2))),
        baseline_rmse=float(np.std(y)),
        envelope=Envelope.fit(work, used),
        pred=pd.Series(preds, index=work.index, name=f"{target}_pred"),
        fold_scores=fold_scores, model=model, lags=lags,
    )

    raw = permutation_importance(model, X, y, used, seed=seed)
    total = sum(raw) or 1.0
    corr = work[used].corr().abs()
    for j, f in enumerate(used):
        others = corr[f].drop(index=f)
        partner = others.idxmax() if len(others) else ""
        res.importances.append(Importance(
            feature=f, importance=raw[j], importance_pct=100.0 * raw[j] / total,
            max_corr_with=str(partner) if partner else "",
            max_corr=float(others.max()) if len(others) else 0.0))
    res.importances.sort(key=lambda i: -i.importance)

    # 상관 묶음 단위로 다시 잰다. 개별 순위보다 이쪽이 정직하다.
    groups = correlation_clusters(work, used, threshold=cluster_threshold)
    gidx = [[used.index(m) for m in g] for g in groups]
    graw = permutation_importance(model, X, y, used, seed=seed, groups=gidx)
    gtotal = sum(graw) or 1.0
    for g, imp in zip(groups, graw):
        internal = 0.0
        if len(g) > 1:
            sub = np.array(corr.loc[g, g], dtype=float)
            np.fill_diagonal(sub, 0.0)
            internal = float(np.max(sub))
        res.clusters.append(FeatureCluster(
            members=g, importance=imp, importance_pct=100.0 * imp / gtotal,
            max_internal_corr=internal))
    res.clusters.sort(key=lambda c: -c.importance)
    return res


@dataclass
class Improvement:
    """제어 가능한 변수를 움직여 목표를 개선하는 안."""

    settings: dict[str, float]
    baseline_value: float
    predicted_value: float
    change_pct: float
    extrapolation: float          # 0 이면 학습 범위 안
    feasible: bool

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{"제어변수": k, "제안값": v} for k, v in self.settings.items()])


def search_improvement(res: SurrogateResult, df: pd.DataFrame, controllable: list[str],
                       direction: str = "minimize", n_grid: int = 9,
                       reference: pd.Series | None = None,
                       max_extrapolation: float = 0.0) -> Improvement | None:
    """제어 가능한 변수만 격자 탐색한다.

    **학습 포락선 밖은 제안하지 않는다.** 대리모델은 그 밖에서 아무 근거가 없다.
    포락선 밖 답이 필요하면 물리 모델로 가야 한다.
    """
    from itertools import product

    ctrl = [c for c in controllable if c in res.features]
    if not ctrl or res.model is None or res.envelope is None:
        return None
    base = reference if reference is not None else df[res.features].median()
    x0 = np.array([float(base[f]) for f in res.features], dtype=float)
    base_val = float(res.model.predict(x0.reshape(1, -1))[0])

    grids = []
    for c in ctrl:
        lo, hi = res.envelope.lo.get(c), res.envelope.hi.get(c)
        if lo is None or hi is None or not np.isfinite(lo + hi) or hi <= lo:
            return None
        grids.append(np.linspace(lo, hi, n_grid))
    pos = [res.features.index(c) for c in ctrl]

    best, best_val = None, np.inf if direction == "minimize" else -np.inf
    for combo in product(*grids):
        x = x0.copy()
        for p, v in zip(pos, combo):
            x[p] = v
        val = float(res.model.predict(x.reshape(1, -1))[0])
        better = val < best_val if direction == "minimize" else val > best_val
        if better:
            best_val, best = val, combo
    if best is None:
        return None
    settings = {c: float(v) for c, v in zip(ctrl, best)}
    row = pd.DataFrame([{**{f: x0[i] for i, f in enumerate(res.features)}, **settings}])
    dist = float(res.envelope.distance(row)[0])
    return Improvement(
        settings=settings, baseline_value=base_val, predicted_value=best_val,
        change_pct=100.0 * (best_val / base_val - 1.0) if abs(base_val) > 1e-30 else np.nan,
        extrapolation=dist, feasible=dist <= max_extrapolation)
