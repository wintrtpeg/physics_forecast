"""차원 해석: 데이터에서 **자동으로** 물리 구조를 찾아낸다.

사용자가 각 컬럼의 단위만 알려주면(태그맵을 쓰면 이미 알려준 것이다) 다음 두 가지가
사람 손 없이 나온다.

1. **보존식 후보 탐지** — 같은 차원을 가진 컬럼들 사이에 ``Σ aᵢxᵢ ≈ 0`` 인 선형 관계가
   있는지 찾는다. 유량 컬럼들에서 ``m1 + m2 + m3 − m_total ≈ 0`` 이 나오면 그건
   질량 보존이다. 데이터가 스스로 말해 준 지배방정식이다.

2. **무차원군(Buckingham Π)** — 차원 행렬의 영공간에서 무차원 조합을 만든다.
   Π 공간에서 회귀하면 모델이 **상사법칙을 정확히 만족**하므로, 순수 회귀와 달리
   스케일 방향으로는 외삽이 성립한다. 물리모델과 ML 사이의 중간 사다리다.

SINDy 류의 희소회귀와 다른 점: 후보를 **같은 차원끼리의 선형결합**으로 제한한다.
그래서 나오는 식이 항상 차원적으로 옳고, 사람이 읽고 "아 이건 질량수지네" 하고
알아볼 수 있다. 차원을 무시하고 항 사전을 훑는 방법은 현장 노이즈에서 무너지고
해석 불가능한 식을 내놓는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np

from ..core.units import DIMLESS, Dim, dim_of, dim_str


# --- 유리수 선형대수 (지수는 정확히 맞아야 한다) ---------------------------

def rref(matrix: Sequence[Sequence[int | Fraction]]) -> tuple[list[list[Fraction]], list[int]]:
    """유리수 기약행사다리꼴. 부동소수점을 쓰면 지수가 0.9999 로 나온다."""
    M = [[Fraction(v) for v in row] for row in matrix]
    rows, cols = len(M), (len(M[0]) if M else 0)
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = Fraction(1) / M[r][c]
        M[r] = [v * inv for v in M[r]]
        for i in range(rows):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return M, pivots


def null_space_int(matrix: Sequence[Sequence[int]]) -> list[list[int]]:
    """정수 행렬의 영공간 기저를 **정수 벡터**로 돌려준다."""
    R, pivots = rref(matrix)
    n = len(matrix[0]) if matrix else 0
    free = [c for c in range(n) if c not in pivots]
    basis: list[list[int]] = []
    for f in free:
        vec = [Fraction(0)] * n
        vec[f] = Fraction(1)
        for r, p in enumerate(pivots):
            vec[p] = -R[r][f]
        den = 1
        for v in vec:
            den = den * v.denominator // _gcd(den, v.denominator)
        ints = [int(v * den) for v in vec]
        g = 0
        for v in ints:
            g = _gcd(g, abs(v))
        if g > 1:
            ints = [v // g for v in ints]
        # 부호 규약: 0 이 아닌 첫 성분을 양수로
        first = next((v for v in ints if v != 0), 1)
        if first < 0:
            ints = [-v for v in ints]
        basis.append(ints)
    return basis


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a or 1


# --- 무차원군 ---------------------------------------------------------------

@dataclass
class PiGroup:
    """무차원 조합 하나. ``exponents`` 는 변수별 지수."""

    exponents: dict[str, int]
    contains_target: bool = False

    def formula(self) -> str:
        num, den = [], []
        for name, e in self.exponents.items():
            if e == 0:
                continue
            token = name if abs(e) == 1 else f"{name}^{abs(e)}"
            (num if e > 0 else den).append(token)
        s = " · ".join(num) if num else "1"
        if den:
            s += " / (" + " · ".join(den) + ")"
        return s

    def is_trivial_ratio(self, units: dict[str, str]) -> bool:
        """같은 차원 변수들끼리의 비율인가.

        무차원이긴 하지만 물리적으로 알려주는 게 없다 (유량A/유량B 같은 것).
        진짜 쓸모 있는 Π 는 **서로 다른 차원을 섞어** 무차원이 된 조합이다.
        """
        dims = {dim_of(units.get(n, "1")) for n in self.exponents}
        dims.discard(DIMLESS)
        return len(dims) <= 1

    def evaluate(self, df, eps: float = 1e-30) -> np.ndarray:
        out = np.ones(len(df))
        for name, e in self.exponents.items():
            if e == 0:
                continue
            col = np.asarray(df[name], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = out * np.power(np.where(np.abs(col) < eps, np.nan, col), float(e))
        return out


def dimension_matrix(units: dict[str, str]) -> tuple[list[str], list[list[int]], list[int]]:
    """단위 dict -> (변수 이름, 7xN 차원 행렬, 실제로 쓰인 차원 인덱스)."""
    names = list(units)
    dims = [dim_of(units[n]) for n in names]
    used = [i for i in range(7) if any(d[i] != 0 for d in dims)]
    D = [[dims[j][i] for j in range(len(names))] for i in used]
    return names, D, used


def pi_groups(units: dict[str, str], target: str | None = None) -> list[PiGroup]:
    """Buckingham Π. 대상 변수를 마지막에 두어 자유변수가 되도록 한다.

    그러면 대상이 들어간 Π 가 정확히 하나 생기고, 나머지를 설명변수로 쓸 수 있다.
    """
    names = [n for n in units if n != target]
    if target is not None and target in units:
        names = names + [target]
    ordered = {n: units[n] for n in names}
    cols, D, _ = dimension_matrix(ordered)
    if not D:                       # 전부 무차원
        return [PiGroup({n: 1}, contains_target=(n == target)) for n in cols]
    basis = null_space_int(D)
    groups = []
    for vec in basis:
        exps = {cols[i]: vec[i] for i in range(len(cols)) if vec[i] != 0}
        if not exps:
            continue
        groups.append(PiGroup(exps, contains_target=(target in exps)))
    return groups


# --- 보존식 후보 탐지 -------------------------------------------------------

@dataclass
class Balance:
    """같은 차원 컬럼들 사이에서 발견된 선형 관계."""

    columns: list[str]
    coefficients: list[float]
    dimension: str
    residual_rel: float          # 잔차 RMS / 항 크기 RMS
    r2: float                    # 가장 큰 항을 나머지로 설명한 비율
    intercept: float = 0.0
    n_rows: int = 0

    def formula(self, unit: str = "") -> str:
        parts = []
        for c, name in zip(self.coefficients, self.columns):
            if abs(c) < 1e-12:
                continue
            sign = "+" if c > 0 else "−"
            mag = "" if abs(abs(c) - 1.0) < 1e-9 else f"{abs(c):g}·"
            parts.append(f"{sign} {mag}{name}")
        s = " ".join(parts).lstrip("+ ").strip()
        rhs = f"{-self.intercept:.4g}" if abs(self.intercept) > 1e-12 else "0"
        return f"{s} = {rhs}" + (f"  [{unit}]" if unit else "")

    @property
    def is_conservation(self) -> bool:
        """계수가 전부 ±1 에 가깝고 절편이 0 이면 전형적인 보존식이다."""
        return (abs(self.intercept) < 1e-9
                and all(abs(abs(c) - 1.0) < 1e-6 for c in self.coefficients if abs(c) > 1e-12))

    @property
    def confidence(self) -> str:
        """얼마나 믿을 만한가.

        R2 는 '주항을 나머지로 얼마나 설명하는가' 다. 진짜 보존식이면 높다.
        비슷한 크기의 두 신호가 우연히 맞아떨어진 경우는 잔차는 작아도 R2 가 낮다.
        """
        if self.residual_rel < 0.01 and self.r2 > 0.9:
            return "확실"
        if self.r2 > 0.5:
            return "가능"
        return "우연 의심"


def _rationalize(v: np.ndarray, max_den: int = 4) -> np.ndarray:
    """계수를 작은 유리수로 스냅. 열거가 불가능한 큰 묶음에서만 쓰는 보조 경로."""
    big = np.max(np.abs(v))
    if big <= 0:
        return v
    w = v / big
    snapped = np.array([float(Fraction(float(x)).limit_denominator(max_den)) for x in w])
    dens = [Fraction(float(x)).limit_denominator(max_den).denominator for x in w]
    lcm = 1
    for d in dens:
        lcm = lcm * d // _gcd(lcm, d)
    out = np.round(snapped * lcm)
    first = next((x for x in out if abs(x) > 0), 1.0)
    return out if first > 0 else -out


def _score(X: np.ndarray, coef: np.ndarray) -> tuple[float, float]:
    """(상대잔차, 주항 설명력 R2)."""
    resid = X @ coef
    term_scale = float(np.sqrt(np.mean((np.abs(X) @ np.abs(coef)) ** 2)))
    if term_scale <= 0:
        return np.inf, np.nan
    rel = float(np.sqrt(np.mean(resid ** 2)) / term_scale)
    lead = int(np.argmax(np.abs(coef) * np.abs(X).mean(axis=0)))
    y = X[:, lead] * coef[lead]
    pred = -(X @ coef - y)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot > 0 else np.nan
    return rel, float(r2)


def find_balances(df, units: dict[str, str], rel_tol: float = 0.03,
                  min_columns: int = 2, max_enumerate: int = 10,
                  min_r2: float = -1e9, coeff_set: tuple[int, ...] = (-1, 0, 1)) -> list["Balance"]:
    """같은 차원 컬럼 묶음마다 ``Σ aᵢxᵢ ≈ 0`` 인 관계를 찾는다.

    계수를 **작은 정수로 열거**한다. SVD 의 최소특이벡터를 쓰면 안 되는 이유:
    현장 데이터는 거의 언제나 강하게 공선적이라(전부 가동율이 끌고 간다) 영공간
    근처에 방향이 여러 개 있고, 그중 가장 작은 것은 노이즈가 고른다. 실제로
    질량 보존(계수 전부 ±1)을 SVD 로 찾게 했더니 (4,4,3,−3,−3) 이 나왔다.

    보존식의 계수는 현실에서 거의 항상 ±1 이므로 열거가 옳고, 결과가 사람이 읽을 수
    있는 형태로 나온다.
    """
    from itertools import product

    by_dim: dict[Dim, list[str]] = {}
    for name, unit in units.items():
        if name not in df.columns:
            continue
        d = dim_of(unit)
        if d == DIMLESS:
            continue
        by_dim.setdefault(d, []).append(name)

    found: list[Balance] = []
    for dim, cols in by_dim.items():
        if len(cols) < max(2, min_columns):
            continue
        sub = df[cols].dropna()
        if len(sub) < 20:
            continue
        X = sub.to_numpy(dtype=float)
        k = len(cols)
        candidates: list[tuple[float, int, np.ndarray, float]] = []

        if k <= max_enumerate:
            for combo in product(coeff_set, repeat=k):
                nz = [c for c in combo if c != 0]
                if len(nz) < 2 or nz[0] < 0:        # 부호 대칭 제거
                    continue
                coef = np.array(combo, dtype=float)
                rel, r2 = _score(X, coef)
                if rel <= rel_tol and (np.isnan(r2) or r2 >= min_r2):
                    candidates.append((rel, len(nz), coef, r2))
        else:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            coef = _rationalize(Vt[-1])
            rel, r2 = _score(X, coef)
            if rel <= rel_tol and (np.isnan(r2) or r2 >= min_r2):
                candidates.append((rel, int(np.count_nonzero(coef)), coef, r2))

        # 잔차가 작고 항이 적은 순. 이미 채택한 관계의 확장판은 버린다.
        candidates.sort(key=lambda t: (t[0], t[1]))
        accepted_supports: list[set[int]] = []
        for rel, nnz, coef, r2 in candidates:
            support = {i for i in range(k) if coef[i] != 0}
            if any(prev <= support for prev in accepted_supports):
                continue
            accepted_supports.append(support)
            keep = sorted(support)
            found.append(Balance(
                columns=[cols[i] for i in keep],
                coefficients=[float(coef[i]) for i in keep],
                dimension=dim_str(dim), residual_rel=rel, r2=r2,
                intercept=0.0, n_rows=len(sub)))
    found.sort(key=lambda b: (b.residual_rel, len(b.columns)))
    return found
