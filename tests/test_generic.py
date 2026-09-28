"""YAML 선언형 컴포넌트.

두 가지를 확인한다.
1. 손으로 짠 컴포넌트와 **수치적으로 같은** 결과를 내는가
2. 파이썬을 전혀 건드리지 않고 **새 도메인**(액체 배관)을 모델링할 수 있는가
"""

import numpy as np
import pytest

from pforecast import solve_steady
from pforecast.core.system import ModelError, System
from pforecast.lib import EquationComponent, load_component_spec
from pforecast.scenario import load_python_model

BASE = "examples/nox_stack/closures/scrubber_base.yaml"


def _nox_with(scrubber_spec) -> tuple:
    system = load_python_model("examples/nox_stack/model.py")
    system.replace("SCR", EquationComponent("SCR", scrubber_spec))
    model = system.compile()
    model.build()
    r = solve_steady(model)
    return model, r


def test_declared_scrubber_matches_the_python_one():
    ref = load_python_model("examples/nox_stack/model.py").compile()
    r_ref = solve_steady(ref)
    model, r = _nox_with(BASE)
    assert r_ref.success and r.success
    assert model.n_eqs == ref.n_eqs and model.n_vars == ref.n_vars
    a = ref.output_values(r_ref.x, ref.p0())
    b = model.output_values(r.x, model.p0())
    for k in set(a) & set(b):
        assert b[k] == pytest.approx(a[k], rel=1e-7, abs=1e-9), k


def test_equations_are_named_in_diagnostics():
    model, _ = _nox_with(BASE)
    labels = [e.label for e in model.equations if e.source == "SCR"]
    assert "SCR.mass" in labels
    assert "SCR.closure_eta" in labels
    assert not any("eq[" in l for l in labels)


def test_extends_replaces_a_single_equation():
    spec = load_component_spec("examples/nox_stack/closures/scrubber_constant.yaml")
    base = load_component_spec(BASE)
    assert spec["equations"]["closure_eta"] == "eta = eta0"
    # 나머지 방정식은 원본 그대로
    for k, v in base["equations"].items():
        if k != "closure_eta":
            assert spec["equations"][k] == v
    # 파라미터는 합쳐진다
    assert "eta0" in spec["params"] and "K" in spec["params"]


def test_unused_parameters_drop_out_of_the_model():
    """상수 효율 후보는 ntu_a/ntu_b 를 쓰지 않으므로 모델에 나타나면 안 된다."""
    model, r = _nox_with("examples/nox_stack/closures/scrubber_constant.yaml")
    assert r.success
    names = {p.name for p in model.parameters}
    assert "SCR.eta0" in names
    assert "SCR.ntu_a" not in names
    assert "SCR.ntu_b" not in names


@pytest.mark.parametrize("path,expect_eta", [
    ("examples/nox_stack/closures/scrubber_constant.yaml", 0.20),
    ("examples/nox_stack/closures/scrubber_langmuir.yaml", None),
    ("examples/nox_stack/closures/scrubber_ntu_temp.yaml", None),
])
def test_every_candidate_compiles_and_solves(path, expect_eta):
    model, r = _nox_with(path)
    assert r.success, r.message
    eta = r.x[model.var_index("SCR.eta")]
    assert 0.0 < eta < 1.0
    if expect_eta is not None:
        assert eta == pytest.approx(expect_eta, rel=1e-6)


def test_replace_rejects_incompatible_ports():
    system = load_python_model("examples/nox_stack/model.py")
    bad = EquationComponent("SCR", {"ports": {"inlet": {"kind": "gas", "role": "in"}},
                                    "equations": {}})
    with pytest.raises(ModelError, match="포트"):
        system.replace("SCR", bad)


def test_unknown_port_kind_is_reported():
    with pytest.raises(ValueError, match="포트 종류"):
        EquationComponent("X", {"ports": {"a": {"kind": "plasma", "role": "in"}}})


# --- 완전히 다른 도메인: 액체 배관 (파이썬 코드 0줄) ---------------------------

PIPE = {
    "ports": {"a": {"kind": "liquid", "role": "in"}, "b": {"kind": "liquid", "role": "out"}},
    "params": {"K": {"value": 3000.0, "unit": "Pa*s2/kg2", "tunable": True}},
    "vars": {"dp": {"unit": "Pa", "start": 1.0e5}},
    "equations": {
        "mass": "a.mdot + b.mdot = 0",
        "friction": "dp = K * signed_pow(a.mdot, 2)",
        "momentum": "a.p - b.p = dp",
        "energy": "b.T = a.T",
    },
    "outputs": {"dp_bar": {"expr": "dp", "unit": "bar"}},
}

SUPPLY = {
    "ports": {"a": {"kind": "liquid", "role": "out"}},
    "params": {"p0": {"value": 4.0, "unit": "bar"}, "T0": {"value": 7.0, "unit": "degC"}},
    "equations": {"pressure": "a.p = p0", "temperature": "a.T = T0"},
}

RETURN = {
    "ports": {"a": {"kind": "liquid", "role": "in"}},
    "params": {"p1": {"value": 1.0, "unit": "bar"}},
    "equations": {"pressure": "a.p = p1"},
}


def _liquid_loop() -> System:
    s = System("liquid")
    s.add(EquationComponent("SUP", SUPPLY))
    s.add(EquationComponent("PIPE", PIPE))
    s.add(EquationComponent("RET", RETURN))
    s.connect("SUP.a", "PIPE.a")
    s.connect("PIPE.b", "RET.a")
    return s


def test_new_domain_from_yaml_only():
    """배관 계통을 파이썬 한 줄 없이 만들고 해석해와 대조한다."""
    model = _liquid_loop().compile()
    assert model.report.ok, model.describe()
    r = solve_steady(model)
    assert r.success, r.message
    mdot = r.x[model.var_index("PIPE.a.mdot")]
    # dp = K*mdot^2  ->  mdot = sqrt((p0-p1)/K)
    expect = np.sqrt((4.0e5 - 1.0e5) / 3000.0)
    assert mdot == pytest.approx(expect, rel=1e-6)
    assert r.x[model.var_index("RET.a.T")] == pytest.approx(280.15, rel=1e-9)


def test_new_domain_structure_is_square():
    model = _liquid_loop().compile()
    # 포트 4개 x 3변수 + 내부변수 1 = 13
    assert model.n_vars == 13
    assert model.n_eqs == 13


def test_liquid_and_gas_ports_cannot_be_connected():
    s = System("mixed")
    s.add(EquationComponent("SUP", SUPPLY))
    s.add(EquationComponent("G", {"ports": {"a": {"kind": "gas", "role": "in"}},
                                  "equations": {}}))
    s.connect("SUP.a", "G.a")
    with pytest.raises(ModelError, match="포트 종류가 다릅니다"):
        s.compile()


def test_declared_model_reports_bad_equations_with_context():
    s = System("broken")
    s.add(EquationComponent("SUP", SUPPLY))
    s.add(EquationComponent("P", {**PIPE, "equations": {**PIPE["equations"],
                                                        "friction": "dp = K * nonexistent"}}))
    s.add(EquationComponent("RET", RETURN))
    s.connect("SUP.a", "P.a")
    s.connect("P.b", "RET.a")
    with pytest.raises(ModelError) as ei:
        s.compile()
    assert "P.friction" in str(ei.value)
    assert "nonexistent" in str(ei.value)
