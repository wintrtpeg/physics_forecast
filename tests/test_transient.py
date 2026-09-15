"""과도 해석: der() 선언과 음함수 오일러 이산화.

NOx 예제는 준정상(quasi-steady)으로 충분하지만, 덕트 열용량이나 스크러버 순환수
탱크를 넣으려면 상태변수가 필요하다. 엔진이 그걸 지원하는지 해석해를 아는 문제로
확인한다.
"""

import numpy as np
import pytest

from pforecast.core import symbolic as S
from pforecast.core.component import Component, ParamSpec, Scope, VarSpec
from pforecast.core.solvers import solve_steady, solve_transient
from pforecast.core.system import System


class ThermalMass(Component):
    """1차 지연: tau * dT/dt = T_inf - T. 해석해는 지수 이완."""

    PARAMS = {
        "tau": ParamSpec(120.0, "s", "시정수"),
        "T_inf": ParamSpec(20.0, "degC", "최종 도달 온도"),
    }
    VARS = {"T": VarSpec("K", start=353.15, desc="본체 온도")}
    PORTS: dict = {}

    def equations(self, s: Scope):
        return [s.tau * s.der("T") - (s.T_inf - s.T)]


def _system() -> System:
    sysm = System("thermal")
    sysm.add(ThermalMass("TM", tau=120.0, T_inf=20.0))
    return sysm


def test_steady_mode_drops_the_derivative():
    model = _system().compile(mode="steady")
    r = solve_steady(model)
    assert r.success
    assert r.x[model.var_index("TM.T")] == pytest.approx(293.15, rel=1e-9)


def test_transient_tracks_the_analytic_solution():
    model = _system().compile(mode="transient")
    model.build()
    assert model.mode == "transient"
    assert model.state_map == {"TM.T": "_prev.TM.T"}

    T0, T_inf, tau = 353.15, 293.15, 120.0
    t = np.arange(0.0, 601.0, 1.0)
    x0 = np.array([T0])
    _, X, results = solve_transient(model, t, lambda _t: model.p0(), x0=x0)
    assert all(r.success for r in results)

    exact = T_inf + (T0 - T_inf) * np.exp(-t / tau)
    # 음함수 오일러는 1차 정확도. dt=1s, tau=120s 이면 수 % 이내.
    assert np.max(np.abs(X[:, 0] - exact)) < 0.4
    assert X[-1, 0] == pytest.approx(exact[-1], abs=0.2)


def test_implicit_euler_is_unconditionally_stable():
    """시정수보다 훨씬 큰 스텝에서도 발산하지 않고 정상상태로 수렴해야 한다."""
    model = _system().compile(mode="transient")
    t = np.arange(0.0, 6001.0, 600.0)          # dt = 5*tau
    _, X, results = solve_transient(model, t, lambda _t: model.p0(),
                                    x0=np.array([353.15]))
    assert all(r.success for r in results)
    assert np.all(np.diff(X[:, 0]) < 0), "단조 감소해야 한다 (진동하면 불안정)"
    assert X[-1, 0] == pytest.approx(293.15, abs=1e-3)


def test_first_step_starts_from_the_given_state():
    """첫 스텝은 dt=무한대(정상상태)가 아니라 주어진 초기값에서 출발해야 한다."""
    model = _system().compile(mode="transient")
    t = np.array([0.0, 1.0, 2.0])
    _, X, _ = solve_transient(model, t, lambda _t: model.p0(), x0=np.array([353.15]))
    # 첫 행은 초기값 근처(정상상태 293.15 가 아님)
    assert X[0, 0] == pytest.approx(353.15, abs=1e-6)
    assert X[1, 0] < X[0, 0]


def test_der_symbol_has_the_right_dimension():
    comp = ThermalMass("TM")
    sc = comp.build_scope()
    d = sc.der("T")
    assert d.kind == "der"
    assert d.dim == (0, 0, -1, 1, 0, 0, 0)     # K/s
    # 방정식의 차원 동차성이 성립해야 한다
    assert S.dim_of_expr(comp.equations(sc)[0]) == (0, 0, 0, 1, 0, 0, 0)
