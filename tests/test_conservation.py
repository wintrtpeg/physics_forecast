"""물리 보존칙 검증.

풀린 해가 질량/화학종/에너지 보존을 만족하는지, 단위 환산이 맞는지 직접 확인한다.
방정식을 그대로 다시 쓰는 대신 **독립적으로 계산한 총량**과 대조한다.
"""

import numpy as np
import pytest

from pforecast import solve_steady
from pforecast.core.units import from_si
from pforecast.lib import Duct, Fan, Mixer, Stack, ToolGroupSource, WetScrubber
from pforecast.lib.gas import FLUE_GAS, MOLAR_MASS
from pforecast.core.system import System


def simple_line(**src_kw) -> System:
    """원 -> 덕트 -> 스크러버 -> 팬 -> 굴뚝 최소 계통."""
    s = System("mini")
    s.add_all(
        ToolGroupSource("SRC", n_tools=20, util=0.8, q_tool=200.0, ef_idle=0.5,
                        ef_process=5.0, T_exh=50.0, rh_exh=0.4, **src_kw),
        Duct("DCT", K=30.0, UA=200.0, T_amb=25.0),
        WetScrubber("SCR", eta_max=0.5, ntu_a=12.0, L=0.4, K=60.0),
        Fan("FAN", c0=4000.0, c2=-60.0),
        Stack("STK", H=20.0, K=10.0),
    )
    s.connect("SRC.outlet", "DCT.a")
    s.connect("DCT.b", "SCR.a")
    s.connect("SCR.b", "FAN.a")
    s.connect("FAN.b", "STK.a")
    return s


@pytest.fixture(scope="module")
def solved():
    model = simple_line().compile()
    model.build()
    r = solve_steady(model)
    assert r.success, r.message
    return model, r.x, model.p0()


def test_structure_is_square(solved):
    model, _, _ = solved
    assert model.report.ok
    assert model.n_eqs == model.n_vars


def test_total_mass_is_conserved(solved):
    model, x, p = solved
    v = lambda n: x[model.var_index(n)]  # noqa: E731
    m_src = -v("SRC.outlet.mdot")
    m_evap = v("SCR.m_evap")
    m_stack = v("STK.a.mdot")
    assert m_stack == pytest.approx(m_src + m_evap, rel=1e-9)


def test_nox_mass_balance_across_scrubber(solved):
    model, x, p = solved
    v = lambda n: x[model.var_index(n)]  # noqa: E731
    nox_in = v("SCR.a.mdot") * v("SCR.a.w_NOx")
    nox_out = -v("SCR.b.mdot") * v("SCR.b.w_NOx")
    eta = v("SCR.eta")
    assert nox_out == pytest.approx(nox_in * (1 - eta), rel=1e-9)
    assert 0.0 < eta < 1.0


def test_nox_emitted_equals_generated_minus_removed(solved):
    model, x, p = solved
    v = lambda n: x[model.var_index(n)]  # noqa: E731
    out = model.output_values_si(x, p)
    generated = -v("SRC.outlet.mdot") * v("SRC.outlet.w_NOx")
    emitted = v("STK.a.mdot") * v("STK.a.w_NOx")
    assert emitted == pytest.approx(generated * (1 - v("SCR.eta")), rel=1e-8)
    # 리포트로 나가는 g/h 값도 같은 양이어야 한다
    assert from_si(emitted, "g/h") == pytest.approx(out["STK.nox_mass_rate"] * 3.6e6, rel=1e-8)


def test_species_fractions_stay_physical(solved):
    model, x, _ = solved
    for vinfo in model.variables:
        if ".w_" in vinfo.name:
            assert 0.0 <= x[vinfo.index] <= 1.0, vinfo.name


def test_dry_basis_conversion_is_consistent(solved):
    model, x, p = solved
    v = lambda n: x[model.var_index(n)]  # noqa: E731
    y_h2o = v("STK.y_H2O")
    assert v("STK.C_dry") == pytest.approx(v("STK.C_wet") / (1 - y_h2o), rel=1e-9)
    assert 0.0 < y_h2o < 0.3


def test_o2_correction_is_off_by_default(solved):
    model, x, _ = solved
    assert x[model.var_index("STK.C_corr")] == pytest.approx(
        x[model.var_index("STK.C_dry")], rel=1e-9)


def test_o2_correction_applies_when_enabled():
    s = simple_line()
    s.components["STK"].set_param("o2_corr", 1.0)
    s.components["STK"].set_param("O2_ref", 0.04)
    model = s.compile()
    r = solve_steady(model)
    assert r.success, r.message
    v = lambda n: r.x[model.var_index(n)]  # noqa: E731
    expect = v("STK.C_dry") * (0.2095 - 0.04) / (0.2095 - v("STK.y_O2_dry"))
    assert v("STK.C_corr") == pytest.approx(expect, rel=1e-8)


def test_pressure_network_is_consistent(solved):
    """룸에서 굴뚝 입구까지 압력 강하를 더하면 정확히 닫혀야 한다."""
    model, x, p = solved
    v = lambda n: x[model.var_index(n)]  # noqa: E731
    p_room = p[model.par_index("SRC.p_room")]
    hood_dp = p_room - v("SRC.outlet.p")
    loop = (hood_dp + v("DCT.dp") + v("SCR.dp") - v("FAN.dp")
            - (p_room - v("STK.a.p")))
    assert loop == pytest.approx(0.0, abs=1e-6)
    assert hood_dp > 0 and v("FAN.dp") > 0


def test_stack_draft_reduces_inlet_pressure():
    """더운 배기가 굴뚝을 오르며 만드는 통풍력이 흡입 쪽으로 작용해야 한다."""
    s = simple_line()
    model_tall = s.compile()
    r_tall = solve_steady(model_tall)
    s2 = simple_line()
    s2.components["STK"].set_param("H", 0.0)
    model_flat = s2.compile()
    r_flat = solve_steady(model_flat)
    assert r_tall.success and r_flat.success
    p_tall = r_tall.x[model_tall.var_index("STK.a.p")]
    p_flat = r_flat.x[model_flat.var_index("STK.a.p")]
    assert p_tall < p_flat, "굴뚝이 높을수록 입구 압력이 낮아야 한다"


def test_higher_utilization_raises_concentration():
    model = simple_line().compile()
    model.build()
    p = model.p0()
    i = model.par_index("SRC.util")
    prev = None
    x = None
    for u in (0.4, 0.6, 0.8, 1.0):
        pp = p.copy()
        pp[i] = u
        r = solve_steady(model, pp, x0=x)
        assert r.success, f"util={u}: {r.message}"
        x = r.x
        c = r.x[model.var_index("STK.C_dry")]
        if prev is not None:
            assert c > prev, "가동율이 오르면 농도도 올라야 한다"
        prev = c


def test_flow_saturates_with_utilization():
    """가동율이 6배가 되어도 유량은 팬 곡선 때문에 훨씬 덜 는다."""
    model = simple_line().compile()
    model.build()
    p = model.p0()
    i = model.par_index("SRC.util")
    flows = []
    x = None
    for u in (0.2, 1.2):
        pp = p.copy()
        pp[i] = u
        r = solve_steady(model, pp, x0=x)
        assert r.success
        x = r.x
        flows.append(r.x[model.var_index("STK.Q_n")])
    assert flows[1] > flows[0]
    # 댐퍼 개도는 (0.35+0.65*1.2)/(0.35+0.65*0.2) = 2.35 배가 되지만,
    # 저항이 유량의 제곱으로 늘고 팬 곡선을 타고 내려가므로 유량은 그보다 덜 는다.
    open_ratio = (0.35 + 0.65 * 1.2) / (0.35 + 0.65 * 0.2)
    assert flows[1] / flows[0] < open_ratio, "압력구동 계통에서 유량은 포화해야 한다"


def test_scrubber_efficiency_falls_as_gas_flow_rises():
    model = simple_line().compile()
    model.build()
    p = model.p0()
    i = model.par_index("SRC.util")
    etas = []
    x = None
    for u in (0.3, 1.0):
        pp = p.copy()
        pp[i] = u
        r = solve_steady(model, pp, x0=x)
        assert r.success
        x = r.x
        etas.append(r.x[model.var_index("SCR.eta")])
    assert etas[1] < etas[0], "L/G 저하로 제거효율이 떨어져야 한다"
