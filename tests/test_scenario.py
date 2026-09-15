"""시나리오 정의·실행과 모델 로더."""

import numpy as np
import pytest
import yaml

from pforecast import solve_steady
from pforecast.core.units import to_si
from pforecast.params import apply_params, load_params, save_params
from pforecast.scenario import (Case, ScenarioSpec, Sweep, apply_settings,
                                load_python_model, load_yaml_model, run_scenario)

EXAMPLE = "examples/nox_stack"


@pytest.fixture(scope="module")
def model():
    m = load_python_model(f"{EXAMPLE}/model.py").compile()
    m.build()
    return m


def test_yaml_and_python_models_agree():
    a = load_yaml_model(f"{EXAMPLE}/system.yaml").compile()
    b = load_python_model(f"{EXAMPLE}/model.py").compile()
    ra, rb = solve_steady(a), solve_steady(b)
    assert ra.success and rb.success
    oa = a.output_values(ra.x, a.p0())
    ob = b.output_values(rb.x, b.p0())
    assert set(oa) == set(ob)
    for k in oa:
        assert oa[k] == pytest.approx(ob[k], rel=1e-10, abs=1e-12)


def test_glob_pattern_sets_every_matching_parameter(model):
    p = apply_settings(model, model.p0(), {"SRC_*.util": 0.95})
    for name in ("SRC_DRY.util", "SRC_CVD.util", "SRC_WET.util", "SRC_IMP.util"):
        assert p[model.par_index(name)] == pytest.approx(0.95)


def test_unknown_pattern_raises(model):
    with pytest.raises(KeyError):
        apply_settings(model, model.p0(), {"NOPE_*.util": 1.0})
    with pytest.raises(KeyError):
        apply_settings(model, model.p0(), {"SRC_DRY.nonexistent": 1.0})


def test_settings_respect_declared_units(model):
    p = apply_settings(model, model.p0(), {"STK.T_amb": {"value": 35.0, "unit": "degC"}})
    assert p[model.par_index("STK.T_amb")] == pytest.approx(308.15)
    # 단위를 생략하면 파라미터가 선언한 단위로 해석한다
    p2 = apply_settings(model, model.p0(), {"SCR.L": 0.9})
    assert p2[model.par_index("SCR.L")] == pytest.approx(to_si(0.9, "m3/min"))


def test_scale_is_multiplicative(model):
    base = model.p0()
    p = apply_settings(model, base, None, {"SRC_*.n_tools": 1.3})
    i = model.par_index("SRC_DRY.n_tools")
    assert p[i] == pytest.approx(base[i] * 1.3)


def test_run_scenario_detects_limit_violations(model):
    spec = ScenarioSpec(
        name="t", outputs=["STK.C_dry"], limits={"STK.C_dry": {"max": 70.0}},
        cases=[Case(name="저부하", set={"SRC_*.util": 0.3}),
               Case(name="고부하", set={"SRC_*.util": 1.2})],
    )
    res = run_scenario(model, spec)
    assert res.cases["converged"].all()
    assert len(res.violations) == 1
    assert res.violations.iloc[0]["label"] == "고부하"
    assert res.violations.iloc[0]["value"] > 70.0


def test_sweep_produces_monotonic_concentration(model):
    spec = ScenarioSpec(name="t", outputs=["STK.C_dry"],
                        sweeps=[Sweep(name="util", over={"SRC_*.util": [0.4, 0.6, 0.8, 1.0]})])
    res = run_scenario(model, spec)
    c = res.sweeps["util"]["STK.C_dry"].to_numpy()
    assert np.all(np.diff(c) > 0)


def test_scenario_yaml_loads(tmp_path):
    spec = ScenarioSpec.load(f"{EXAMPLE}/scenarios.yaml")
    assert spec.name
    assert len(spec.cases) >= 5
    assert len(spec.sweeps) >= 2
    assert "STK.C_dry" in spec.limits


def test_params_roundtrip(tmp_path, model):
    path = tmp_path / "p.yaml"
    model.system.components["SCR"].set_param("K", 77.0)
    model.parameters[model.par_index("SCR.K")].value = 77.0
    save_params(model, path, ["SCR.K", "DCT_MAIN.K"])
    loaded = load_params(path)
    assert loaded["SCR.K"] == (77.0, "1/m4")

    other = load_python_model(f"{EXAMPLE}/model.py").compile()
    applied = apply_params(other, path)
    assert set(applied) == {"SCR.K", "DCT_MAIN.K"}
    assert other.parameters[other.par_index("SCR.K")].value == pytest.approx(77.0)
    assert other.system.components["SCR"].param_value("K") == pytest.approx(77.0)


def test_apply_params_reports_unknown_names(tmp_path, model):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"parameters": {"NOPE.x": {"value": 1.0, "unit": "1"}}}),
                    encoding="utf-8")
    with pytest.raises(KeyError):
        apply_params(model, path, strict=True)
    assert apply_params(model, path, strict=False) == []
