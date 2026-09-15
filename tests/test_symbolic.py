"""심볼릭 코어: 미분 정확도와 차원 검사."""

import numpy as np
import pytest

from pforecast.core import symbolic as S
from pforecast.core.units import MASS_FLOW, PRESSURE, TEMP, DimensionError, dim_str


def _numeric_grad(f, x, i, h=None):
    h = h or 1e-6 * max(1.0, abs(x[i]))
    xp, xm = x.copy(), x.copy()
    xp[i] += h
    xm[i] -= h
    return (f(xp)[0] - f(xm)[0]) / (2 * h)


@pytest.mark.parametrize("build", [
    lambda a, b: a * b + S.exp(a / b),
    lambda a, b: S.sqrt(a * a + b * b) * S.log(a),
    lambda a, b: a ** S.const(2.5) / (b + S.const(1.0)),
    lambda a, b: S.tanh(a * b) + S.sin(a) * S.cos(b),
    lambda a, b: S.signed_pow(a - b, 2.0),
    lambda a, b: S.smooth_max(a, b) - S.smooth_min(a, b),
])
def test_analytic_derivatives_match_finite_difference(build):
    a, b = S.sym("a"), S.sym("b")
    e = build(a, b)
    slots = {a.uid: "x[0]", b.uid: "x[1]"}
    f = S.compile_function([e], slots, arg_names=("x",))
    j = S.compile_function([S.diff(e, a), S.diff(e, b)], slots, arg_names=("x",))
    x = np.array([1.7, 0.9])
    grad = j(x)
    for i in range(2):
        assert grad[i] == pytest.approx(_numeric_grad(f, x, i), rel=1e-5, abs=1e-7)


def test_hash_consing_shares_subexpressions():
    a = S.sym("a")
    assert (a * a) is (a * a)
    assert S.exp(a + S.const(1.0)) is S.exp(S.const(1.0) + a * S.const(1.0))


def test_smart_constructors_fold_identities():
    a = S.sym("a")
    assert a + S.const(0.0) is a
    assert a * S.const(1.0) is a
    assert (a * S.const(0.0)) is S.const(0.0)
    assert (a / a) is S.const(1.0)


def test_dimension_mismatch_is_caught():
    p = S.sym("p", PRESSURE)
    m = S.sym("m", MASS_FLOW)
    with pytest.raises(DimensionError):
        S.dim_of_expr(p + m)
    # 곱/나눗셈은 통과하고 결과 차원이 맞아야 한다
    assert dim_str(S.dim_of_expr(p * m)) == dim_str(tuple(
        x + y for x, y in zip(PRESSURE, MASS_FLOW)))


def test_transcendental_requires_dimensionless_argument():
    T = S.sym("T", TEMP)
    with pytest.raises(DimensionError):
        S.dim_of_expr(S.exp(T))
    assert S.dim_of_expr(S.exp(T / S.const(1.0, TEMP))) == (0,) * 7


def test_substitute_replaces_symbols():
    a, b = S.sym("a"), S.sym("b")
    e = S.substitute(a * b + a, {a.uid: S.const(2.0)})
    slots = {b.uid: "x[0]"}
    f = S.compile_function([e], slots, arg_names=("x",))
    assert f(np.array([3.0]))[0] == pytest.approx(8.0)


def test_compiled_code_reuses_common_subexpressions():
    a = S.sym("a")
    big = S.exp(a * a)
    e = big + big * S.const(2.0) + big * big
    f = S.compile_function([e], {a.uid: "x[0]"}, arg_names=("x",))
    assert f.__pf_source__.count("np.exp") == 1
