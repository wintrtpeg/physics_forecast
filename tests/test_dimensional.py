"""차원 해석: 무차원군과 보존식 자동 탐지."""

import numpy as np
import pandas as pd
import pytest

from pforecast.analyze.dimensional import (Balance, find_balances, null_space_int,
                                           pi_groups, rref)


def test_rref_is_exact_over_rationals():
    R, piv = rref([[2, 4, 6], [1, 3, 5]])
    assert piv == [0, 1]
    assert [[float(v) for v in row] for row in R] == [[1, 0, -1], [0, 1, 2]]


def test_null_space_returns_integer_vectors():
    basis = null_space_int([[1, 1, -1]])
    assert len(basis) == 2
    for v in basis:
        assert all(isinstance(x, int) for x in v)
        assert v[0] + v[1] - v[2] == 0


def test_buckingham_recovers_reynolds_and_euler():
    """관내 압력손실의 교과서 결과가 나와야 한다."""
    units = {"rho": "kg/m3", "v": "m/s", "L": "m", "mu": "Pa*s", "dp": "Pa"}
    groups = pi_groups(units, target="dp")
    assert len(groups) == 2                      # 변수 5개 - 차원 3개
    formulas = {frozenset(g.exponents.items()) for g in groups}
    reynolds = frozenset({"rho": 1, "v": 1, "L": 1, "mu": -1}.items())
    euler = frozenset({"rho": 1, "v": 2, "dp": -1}.items())
    assert reynolds in formulas
    assert euler in formulas
    # 타깃은 정확히 한 무차원군에만 들어가야 한다
    assert sum(g.contains_target for g in groups) == 1


def test_dimensionless_columns_are_their_own_group():
    groups = pi_groups({"util": "1", "rh": "%"})
    assert len(groups) == 2
    assert all(len(g.exponents) == 1 for g in groups)


def test_pi_group_evaluates_on_data():
    df = pd.DataFrame({"a": [2.0, 4.0], "b": [1.0, 2.0]})
    g = pi_groups({"a": "m", "b": "m"})[0]
    vals = g.evaluate(df)
    assert np.allclose(vals, vals[0])            # a/b 가 일정


# --- 보존식 탐지 ------------------------------------------------------------

def _flow_frame(n=500, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(1.0, 2.0, n)
    b = rng.uniform(0.5, 1.5, n)
    c = rng.uniform(0.2, 0.6, n)
    total = a + b + c
    f = lambda x: x + rng.normal(0, noise, n)    # noqa: E731
    return pd.DataFrame({"m_a": f(a), "m_b": f(b), "m_c": f(c), "m_total": f(total)})


UNITS = {"m_a": "kg/s", "m_b": "kg/s", "m_c": "kg/s", "m_total": "kg/s"}


def test_exact_mass_balance_is_found():
    bal = find_balances(_flow_frame(), UNITS)
    assert bal, "정확히 성립하는 질량수지를 찾지 못했다"
    best = bal[0]
    assert set(best.columns) == {"m_a", "m_b", "m_c", "m_total"}
    assert best.is_conservation
    assert best.confidence == "확실"
    assert best.residual_rel < 1e-6
    coef = dict(zip(best.columns, best.coefficients))
    assert coef["m_a"] == coef["m_b"] == coef["m_c"] == -coef["m_total"]


def test_balance_survives_measurement_noise():
    bal = find_balances(_flow_frame(noise=0.01), UNITS, rel_tol=0.03)
    assert bal and bal[0].is_conservation
    assert bal[0].residual_rel < 0.03


def test_no_balance_is_reported_when_none_exists():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"m_a": rng.uniform(1, 2, 400), "m_b": rng.uniform(1, 2, 400),
                       "m_c": rng.uniform(1, 2, 400)})
    assert find_balances(df, {"m_a": "kg/s", "m_b": "kg/s", "m_c": "kg/s"}) == []


def test_columns_of_different_dimensions_are_never_mixed():
    df = _flow_frame()
    df["T"] = np.linspace(20, 30, len(df))
    df["p"] = df["m_a"] * 1000.0
    bal = find_balances(df, {**UNITS, "T": "degC", "p": "Pa"})
    for b in bal:
        assert len({b.dimension}) == 1
        assert "T" not in b.columns or all(c == "T" for c in b.columns)


def test_coincidental_equality_is_labelled_suspicious():
    """크기가 비슷한 두 신호가 우연히 맞아떨어진 경우를 구분해야 한다."""
    rng = np.random.default_rng(2)
    base = 1.0 + 0.002 * rng.normal(0, 1, 600)
    df = pd.DataFrame({"x": base, "y": base + 0.002 * rng.normal(0, 1, 600)})
    bal = find_balances(df, {"x": "kg/s", "y": "kg/s"}, rel_tol=0.05)
    assert bal
    assert bal[0].confidence == "우연 의심"


def test_integer_enumeration_beats_svd_on_collinear_data():
    """전부 한 요인이 끌고 가는(공선적인) 데이터에서도 계수가 ±1 로 나와야 한다."""
    rng = np.random.default_rng(5)
    drive = rng.uniform(0.5, 1.0, 800)
    a, b, c = 1.4 * drive, 1.2 * drive, 0.4 * drive
    df = pd.DataFrame({"m_a": a + rng.normal(0, 0.012, 800),
                       "m_b": b + rng.normal(0, 0.012, 800),
                       "m_c": c + rng.normal(0, 0.012, 800),
                       "m_total": (a + b + c) + rng.normal(0, 0.012, 800)})
    best = find_balances(df, UNITS, rel_tol=0.03)[0]
    assert set(best.coefficients) <= {1.0, -1.0}
    assert best.is_conservation


def test_balance_formula_is_readable():
    b = Balance(columns=["a", "b", "c"], coefficients=[1.0, 1.0, -1.0],
                dimension="kg/s", residual_rel=0.001, r2=0.99)
    assert b.formula() == "a + b − c = 0"
