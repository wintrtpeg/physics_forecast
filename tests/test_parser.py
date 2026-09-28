"""방정식 텍스트 파서."""

import numpy as np
import pytest

from pforecast.core import symbolic as S
from pforecast.core.parser import (EquationSyntaxError, FunctionTable, parse_equation,
                                   parse_expression, tokenize)
from pforecast.core.units import MASS_FLOW, PRESSURE, DimensionError, dim_str


@pytest.fixture
def env():
    return {"p": S.sym("p", PRESSURE), "q": S.sym("q", PRESSURE),
            "m": S.sym("m", MASS_FLOW), "K": S.sym("K", (-1, -1, 0, 0, 0, 0, 0)),
            "x": S.sym("x"), "y": S.sym("y")}


@pytest.fixture
def resolve(env):
    return lambda n: env.get(n)


def _eval(expr, values: dict):
    syms = S.collect_syms([expr])
    slots = {s.uid: f"x[{i}]" for i, s in enumerate(syms)}
    f = S.compile_function([expr], slots, arg_names=("x",))
    return float(f(np.array([values[s.name] for s in syms]))[0])


@pytest.mark.parametrize("text,vals,expect", [
    ("x + y", {"x": 2, "y": 3}, 5),
    ("x - y", {"x": 2, "y": 3}, -1),
    ("x * y + 1", {"x": 2, "y": 3}, 7),
    ("x / y", {"x": 6, "y": 3}, 2),
    ("-x", {"x": 2, "y": 0}, -2),
    ("x ^ 3", {"x": 2, "y": 0}, 8),
    ("2 ^ 3 ^ 2", {"x": 0, "y": 0}, 512),          # ^ 는 우결합
    ("1 - 2 - 3", {"x": 0, "y": 0}, -4),           # - 는 좌결합
    ("(x + y) * 2", {"x": 1, "y": 2}, 6),
    ("exp(0) + sqrt(y)", {"x": 0, "y": 9}, 4),
    ("max(x, y)", {"x": 1.0, "y": 5.0}, 5.0),
    ("sq(x)", {"x": 4}, 16),
])
def test_expression_evaluation(text, vals, expect, resolve):
    e = parse_expression(text, resolve)
    assert _eval(e, vals) == pytest.approx(expect, rel=1e-6)


def test_equation_becomes_residual(resolve, env):
    e = parse_equation("p = K * m", resolve)
    # p - K*m 이어야 한다
    assert _eval(e, {"p": 10.0, "K": 2.0, "m": 3.0}) == pytest.approx(4.0)


def test_equation_without_equals_is_already_a_residual(resolve):
    e = parse_equation("x + y", resolve)
    assert _eval(e, {"x": 1.0, "y": 2.0}) == pytest.approx(3.0)


def test_parsed_equation_is_dimensionally_checked(resolve):
    e = parse_equation("p = K * signed_pow(m, 2)", resolve)
    assert dim_str(S.dim_of_expr(e)) == dim_str(PRESSURE)
    with pytest.raises(DimensionError):
        S.dim_of_expr(parse_equation("p = m", resolve))


def test_unit_annotated_literal(resolve):
    e = parse_expression("150[mmAq]", resolve)
    assert _eval(e, {}) == pytest.approx(150 * 9.80665)
    assert S.dim_of_expr(e) == PRESSURE
    # 단위 없는 리터럴은 무차원 -> 압력에 더해도 통과한다 (주변 차원에 맞춰짐)
    assert S.dim_of_expr(parse_expression("1.5", resolve)) is None


def test_bad_unit_is_reported(resolve):
    with pytest.raises(EquationSyntaxError, match="단위"):
        parse_expression("3[furlongs]", resolve)


def test_parsed_graph_matches_handwritten(resolve, env):
    """파싱 결과가 손으로 짠 식 그래프와 같은 객체여야 한다 (해시콘싱)."""
    parsed = parse_expression("K * signed_pow(m, 2)", resolve)
    manual = env["K"] * S.signed_pow(env["m"], 2.0)
    assert parsed is manual


def test_derivatives_of_parsed_expressions(resolve, env):
    e = parse_expression("K * m ^ 2 + exp(m)", resolve)
    d = S.diff(e, env["m"])
    vals = {"K": 3.0, "m": 1.5}
    numeric = (_eval(e, {**vals, "m": 1.5 + 1e-6}) - _eval(e, {**vals, "m": 1.5 - 1e-6})) / 2e-6
    assert _eval(d, vals) == pytest.approx(numeric, rel=1e-5)


@pytest.mark.parametrize("bad,match", [
    ("x = = y", "식이 와야"),
    ("x +", "식이 와야"),
    ("nope + 1", "찾을 수 없"),
    ("frobnicate(x)", "알 수 없는 함수"),
    ("(x + y", "'\\)' 가 필요"),
    ("x = y = 1", "등호는 하나만"),
    ("x 3", "남는 내용"),
])
def test_syntax_errors_are_specific(bad, match, resolve):
    with pytest.raises(EquationSyntaxError, match=match):
        parse_equation(bad, resolve)


def test_error_message_points_at_the_position(resolve):
    with pytest.raises(EquationSyntaxError) as ei:
        parse_equation("x + nope * 2", resolve, context="MY.eq")
    msg = str(ei.value)
    assert "MY.eq" in msg
    lines = msg.splitlines()
    assert lines[-1].strip() == "^"
    assert lines[-2].index("nope") == lines[-1].index("^")


def test_comment_is_stripped(resolve):
    assert _eval(parse_expression("x + 1  # 주석", resolve), {"x": 2}) == pytest.approx(3)


def test_function_table_can_be_extended(resolve):
    ft = FunctionTable().extend({"double": lambda e: e * S.const(2.0)})
    assert _eval(parse_expression("double(x)", resolve, ft), {"x": 4}) == pytest.approx(8)


def test_der_requires_a_declared_variable(resolve):
    seen = {}

    def resolve_der(name):
        seen["name"] = name
        return S.sym(f"der({name})", kind="der")

    e = parse_equation("der(x) = -x", resolve, resolve_der=resolve_der)
    assert seen["name"] == "x"
    assert any(s.kind == "der" for s in S.collect_syms([e]))
    with pytest.raises(EquationSyntaxError, match="쓸 수 없습니다"):
        parse_equation("der(x) = 0", resolve)


def test_tokenizer_handles_dotted_names():
    toks = tokenize("a.mdot + b.w_NOx")
    assert [t.value for t in toks if t.kind == "name"] == ["a.mdot", "b.w_NOx"]
