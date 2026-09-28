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
    #: 주항 = gain × (나머지 합) 으로 맞췄을 때의 이득. 1 이 아니면 계측기 교정 오차다.
    gain: float = 1.0
    residual_rel_gain: float = float("nan")
    r2_gain: float = float("nan")
    lead: str = ""
    #: 두 컬럼이 같은 양을 잴 때(A − B = 0) 평균 차이. degC 처럼 영점이 임의인 단위는
    #: 이득(비율)이 의미가 없고 이 값이 진단이다.
    offset: float = 0.0
    offset_scale: bool = False
    unit: str = ""

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
        # 계측기 하나가 몇 % 높게 읽으면 보존식인데도 R2 가 망가진다. 이득을 보정한
        # 뒤에 잔차가 계측 잡음 수준으로 떨어지면 보존식이고, 그 이득이 곧 진단이다.
        if (abs(self.gain - 1.0) <= 0.1 and self.residual_rel_gain < 0.01
                and self.r2_gain > 0.8):
            return "확실"
        if self.r2 > 0.5 or (np.isfinite(self.r2_gain) and self.r2_gain > 0.8):
            return "가능"
        return "우연 의심"

    @property
    def is_redundant_pair(self) -> bool:
        """같은 양을 재는 두 계측 (A − B = 0)."""
        return len(self.columns) == 2 and sorted(self.coefficients) == [-1.0, 1.0]

    def gain_note(self) -> str:
        """보존식이 계측기 이득(또는 영점) 오차를 드러내면 그 문장."""
        if self.confidence != "확실":
            return ""
        if self.offset_scale or self.is_redundant_pair:
            if self.is_redundant_pair and abs(self.offset) > 0:
                a, b = self.columns
                hi, lo = (a, b) if self.offset > 0 else (b, a)
                return (f"{hi} 가 {lo} 보다 평균 {abs(self.offset):.3g}{self.unit} 높게 읽습니다 "
                        "— 같은 양을 재는 이중화 계측으로 보이며, 차이는 영점 오차입니다.")
            return ""
        if not self.lead or abs(self.gain - 1.0) < 0.01:
            return ""
        why = ("계측기 교정 오차로 보입니다" if abs(self.gain - 1.0) <= 0.05 else
               "교정 오차치고는 커서, 계측되지 않는 항이 빠졌을 수도 있습니다")
        return (f"{self.lead} 가 나머지 항의 합보다 {(self.gain - 1.0) * 100:+.1f}% "
                f"높게 읽습니다 — {why}.").replace("+-", "-").replace("높게 읽습니다", "높게 읽습니다" if self.gain > 1 else "낮게 읽습니다")


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
    return _score_full(X, coef)[:2]


def _score_full(X: np.ndarray, coef: np.ndarray):
    """(상대잔차, R2, 이득, 이득 보정 상대잔차, 이득 보정 R2, 주항 위치)."""
    resid = X @ coef
    term_scale = float(np.sqrt(np.mean((np.abs(X) @ np.abs(coef)) ** 2)))
    if term_scale <= 0:
        return np.inf, np.nan, np.nan, np.inf, np.nan, -1
    rel = float(np.sqrt(np.mean(resid ** 2)) / term_scale)
    lead = int(np.argmax(np.abs(coef) * np.abs(X).mean(axis=0)))
    y = X[:, lead] * coef[lead]
    pred = -(X @ coef - y)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot > 0 else np.nan
    pp = float(pred @ pred)
    gain = float(pred @ y) / pp if pp > 0 else np.nan
    rg = y - gain * pred if np.isfinite(gain) else y
    rel_g = float(np.sqrt(np.mean(rg ** 2)) / term_scale)
    r2_g = 1.0 - float(np.sum(rg ** 2)) / ss_tot if ss_tot > 0 else np.nan
    return rel, float(r2), gain, rel_g, float(r2_g), lead


#: 영점이 임의인 단위. 비율(이득)이 의미가 없다.
_OFFSET_UNITS = {"degC", "C", "oC", "℃", "degF", "F"}


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
        # 전 컬럼이 동시에 살아 있는 행만 쓰면, 고장 난 계측기 하나가 나머지 관계까지
        # 못 찾게 만든다. 조합마다 **그 조합에 든 컬럼**이 살아 있는 행을 쓴다.
        Xall = df[cols].to_numpy(dtype=float)
        live = np.isfinite(Xall)
        k = len(cols)
        candidates: list[tuple] = []

        def consider(coef: np.ndarray):
            support = coef != 0
            rows = live[:, support].all(axis=1)
            if rows.sum() < 20:
                return
            X = np.where(np.isfinite(Xall[rows]), Xall[rows], 0.0)
            rel, r2, gain, rel_g, r2_g, lead = _score_full(X, coef)
            ok = rel <= rel_tol and (np.isnan(r2) or r2 >= min_r2)
            # 이득만 다른 보존식 (계측기 교정 오차): 보정 후 잔차가 잡음 수준이면 받는다
            ok_gain = (np.isfinite(gain) and abs(gain - 1.0) <= 0.15
                       and rel_g <= rel_tol / 3 and np.isfinite(r2_g) and r2_g >= 0.9)
            if ok or ok_gain:
                # 이득이 1 에서 멀수록 벌점. 빠진 항이 나머지에 비례하면(계측 안 되는 지류)
                # 이득만 다른 '가짜 보존식'이 생긴다 — 이득이 1 에 가까운 쪽이 물리적으로 옳다.
                key = min(rel, rel_g + 0.1 * abs(gain - 1.0)) if ok_gain else rel
                candidates.append((key, int(support.sum()),
                                   coef, r2, gain, rel_g, r2_g, lead, int(rows.sum()), rel))

        if k <= max_enumerate:
            for combo in product(coeff_set, repeat=k):
                nz = [c for c in combo if c != 0]
                if len(nz) < 2 or nz[0] < 0:        # 부호 대칭 제거
                    continue
                consider(np.array(combo, dtype=float))
        else:
            sub = df[cols].dropna()
            if len(sub) >= 20:
                _, _, Vt = np.linalg.svd(sub.to_numpy(dtype=float), full_matrices=False)
                consider(_rationalize(Vt[-1]))

        # 잔차가 작고 항이 적은 순. 이미 채택한 관계의 확장판은 버린다.
        candidates.sort(key=lambda t: (t[0], t[1]))
        accepted_supports: list[set[int]] = []
        for _, nnz, coef, r2, gain, rel_g, r2_g, lead, n_rows, rel in candidates:
            support = {i for i in range(k) if coef[i] != 0}
            # 채택한 관계의 확장판도, 부분집합도 버린다 (부분집합이 맞으려면 빠진 항이
            # 나머지에 비례해야 하는데, 그건 새 물리가 아니라 우연한 공선성이다)
            if any(prev <= support or support <= prev for prev in accepted_supports):
                continue
            accepted_supports.append(support)
            keep = sorted(support)
            offset = 0.0
            if len(keep) == 2:
                rows = live[:, keep].all(axis=1)
                d = Xall[rows][:, keep] @ coef[keep]
                offset = float(np.mean(d)) * (1.0 if coef[keep[0]] > 0 else -1.0)
            u0 = units.get(cols[keep[0]], "")
            found.append(Balance(
                columns=[cols[i] for i in keep],
                coefficients=[float(coef[i]) for i in keep],
                dimension=dim_str(dim), residual_rel=rel, r2=r2,
                intercept=0.0, n_rows=n_rows, gain=float(gain) if np.isfinite(gain) else 1.0,
                residual_rel_gain=rel_g, r2_gain=r2_g,
                lead=cols[lead] if lead >= 0 else "",
                offset=offset, offset_scale=u0 in _OFFSET_UNITS,
                unit=" ℃" if u0 in ("degC", "C", "℃") else (f" {u0}" if u0 else "")))
    found.sort(key=lambda b: (b.residual_rel, len(b.columns)))
    return found
