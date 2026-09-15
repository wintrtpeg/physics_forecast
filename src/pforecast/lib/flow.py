"""유동/열 계통 범용 컴포넌트: 덕트, 팬, 합류부.

모두 지배방정식(질량·운동량·에너지·화학종 보존)으로만 기술한다. 상관식이 들어가는
곳은 마찰계수와 팬 성능곡선뿐이고, 이들은 캘리브레이션 대상 파라미터로 노출한다.
"""

from __future__ import annotations

from ..core import symbolic as S
from ..core.component import Component, ParamSpec, PortSpec, Scope, VarSpec
from .gas import FLUE_GAS, G_ACCEL, GasMedium, density, enthalpy, gas_port, mixture_cp

# 자주 쓰는 차원 리터럴
_D_K = (0, 0, 0, 1, 0, 0, 0)
_D_CP = (2, 0, -2, -1, 0, 0, 0)
_MIN_MDOT = 1e-4  # kg/s, 0 유량에서 에너지식이 특이해지는 것을 막는 하한


class GasComponent(Component):
    """가스 매질을 갖는 컴포넌트의 공통 기반."""

    def __init__(self, name: str, medium: GasMedium = FLUE_GAS, **params):
        self.medium = medium
        super().__init__(name, **params)

    def _species_transport(self, a, b) -> list[S.Expr]:
        """반응/제거가 없는 경우의 화학종 전달."""
        return [b.w(sp) - a.w(sp) for sp in self.medium.species]

    # --- 초기값 전파 ------------------------------------------------------
    def _mix_inlets(self, inlets: dict[str, dict[str, float]]) -> dict[str, float]:
        """입구들을 질량가중 평균해 하나의 상태로 합친다."""
        m = sum(max(st.get("mdot", 0.0), 0.0) for st in inlets.values())
        if m <= 0.0:
            m = 1e-3
        T = sum(max(st.get("mdot", 0.0), 0.0) * st.get("T", 313.15) for st in inlets.values()) / m
        state = {"mdot": m, "T": T}
        for sp in self.medium.species:
            key = f"w_{sp}"
            state[key] = sum(
                max(st.get("mdot", 0.0), 0.0) * st.get(key, 0.0) for st in inlets.values()
            ) / m
        return state

    def initial_guess(self, inlets):
        if not inlets:
            return {}
        state = self._mix_inlets(inlets)
        outs = [pn for pn, ps in self.port_specs().items() if ps.role == "out"]
        if not outs:
            return {"vars": self._var_guess(state)}
        share = state["mdot"] / len(outs)
        ports = {}
        for pn in outs:
            st = dict(state)
            st["mdot"] = share
            ports[pn] = st
        return {"ports": ports, "vars": self._var_guess(state)}

    def _var_guess(self, state: dict[str, float]) -> dict[str, float]:
        """내부 변수 초기값. 하위 클래스가 필요한 만큼 채운다."""
        return {}


class Duct(GasComponent):
    """덕트 구간: 마찰 압력손실 + 주위로의 열손실.

    dp = K * mdot*|mdot| / rho  (K [1/m^4] 는 f*L/(2*D*A^2) 를 하나로 묶은 값)
    열손실은 대수평균 대신 산술평균 온도차를 쓴 선형식이라 뉴턴법에 안정적이다.
    """

    PARAMS = {
        "K": ParamSpec(0.5, "1/m4", "덕트 저항계수 f*L/(2*D*A^2)", lo=0.0, hi=1e5, tunable=True),
        "UA": ParamSpec(0.0, "W/K", "덕트 외벽 총괄 열전달", lo=0.0, hi=1e6, tunable=True),
        "T_amb": ParamSpec(25.0, "degC", "주위 온도"),
    }
    VARS = {
        "dp": VarSpec("Pa", start=50.0, desc="압력손실"),
        "rho": VarSpec("kg/m3", start=1.15, lo=0.05, hi=10.0, desc="밀도"),
    }

    def port_specs(self) -> dict[str, PortSpec]:
        return {"a": gas_port(self.medium, "in"), "b": gas_port(self.medium, "out")}

    def equations(self, s: Scope) -> list[S.Expr]:
        a, b = s.port("a"), s.port("b")
        cp = mixture_cp(a, self.medium)
        m_eff = S.smooth_abs(a.mdot, _MIN_MDOT)
        C = m_eff * cp
        half_UA = S.const(0.5) * s.UA
        return [
            a.mdot + b.mdot,
            s.rho - density(a, self.medium),
            s.dp - s.K * S.signed_pow(a.mdot, 2.0) / s.rho,
            a.p - b.p - s.dp,
            b.T * (C + half_UA) - ((C - half_UA) * a.T + s.UA * s.T_amb),
            *self._species_transport(a, b),
        ]

    def outputs(self, s: Scope) -> dict[str, tuple[S.Expr, str]]:
        a = s.port("a")
        return {
            "Q": (a.mdot / s.rho, "CMM"),
            "dp_mmAq": (s.dp, "mmAq"),
        }

    def _var_guess(self, state):
        from .gas import density_num
        w = {sp: state.get(f"w_{sp}", 0.0) for sp in self.medium.species}
        rho = density_num(101325.0, state["T"], w, self.medium)
        return {"rho": rho, "dp": self.param_value("K") * state["mdot"] ** 2 / rho}


class Fan(GasComponent):
    """원심 송풍기: 2차 성능곡선 + 상사법칙 + 단열 온도상승.

    dp = (n^2) * (c0 + c1*(Q/n) + c2*(Q/n)^2) * (rho/rho_ref)
    n 은 정격 대비 회전수비(인버터 주파수비). 가동율이 오르면 유량이 늘고
    성능곡선을 타고 내려가면서 압력이 떨어지는 관계가 자동으로 잡힌다.
    """

    PARAMS = {
        "c0": ParamSpec(3500.0, "Pa", "무부하 정압", tunable=True),
        "c1": ParamSpec(0.0, "Pa*s/m3", "1차항", tunable=True),
        "c2": ParamSpec(-6.0, "Pa*s2/m6", "2차항", tunable=True),
        "n_ratio": ParamSpec(1.0, "1", "회전수비 (인버터)", lo=0.2, hi=1.2),
        "eta": ParamSpec(0.70, "1", "전효율", lo=0.1, hi=0.95, tunable=True),
        "rho_ref": ParamSpec(1.2, "kg/m3", "성능곡선 기준 밀도"),
    }
    VARS = {
        "dp": VarSpec("Pa", start=2000.0, desc="정압 상승"),
        "Q": VarSpec("m3/s", start=10.0, lo=1e-4, hi=1e4, desc="흡입 체적유량"),
        "rho": VarSpec("kg/m3", start=1.15, lo=0.05, hi=10.0),
    }

    def port_specs(self) -> dict[str, PortSpec]:
        return {"a": gas_port(self.medium, "in"), "b": gas_port(self.medium, "out")}

    def equations(self, s: Scope) -> list[S.Expr]:
        a, b = s.port("a"), s.port("b")
        cp = mixture_cp(a, self.medium)
        m_eff = S.smooth_abs(a.mdot, _MIN_MDOT)
        n = s.n_ratio
        q_red = s.Q / n
        curve = s.c0 + s.c1 * q_red + s.c2 * q_red * q_red
        return [
            a.mdot + b.mdot,
            s.rho - density(a, self.medium),
            s.Q * s.rho - a.mdot,
            s.dp - n * n * curve * s.rho / s.rho_ref,
            b.p - a.p - s.dp,
            m_eff * cp * (b.T - a.T) * s.eta - s.dp * s.Q,
            *self._species_transport(a, b),
        ]

    def outputs(self, s: Scope) -> dict[str, tuple[S.Expr, str]]:
        return {
            "Q": (s.Q, "CMM"),
            "dp_mmAq": (s.dp, "mmAq"),
            "power_shaft": (s.dp * s.Q / s.eta, "kW"),
        }

    def _var_guess(self, state):
        from .gas import density_num
        w = {sp: state.get(f"w_{sp}", 0.0) for sp in self.medium.species}
        rho = density_num(101325.0, state["T"], w, self.medium)
        Q = state["mdot"] / rho
        n = self.param_value("n_ratio")
        qr = Q / max(n, 1e-6)
        dp = n * n * (self.param_value("c0") + self.param_value("c1") * qr
                      + self.param_value("c2") * qr * qr) * rho / self.param_value("rho_ref")
        return {"rho": rho, "Q": Q, "dp": dp}


class Mixer(GasComponent):
    """N개 지류를 합류시키는 헤더. 질량/에너지/화학종 보존 + 등압 합류."""

    def __init__(self, name: str, n_inlets: int = 2, medium: GasMedium = FLUE_GAS, **params):
        self.n_inlets = int(n_inlets)
        if self.n_inlets < 1:
            raise ValueError("n_inlets 는 1 이상이어야 합니다")
        super().__init__(name, medium=medium, **params)

    PARAMS: dict[str, ParamSpec] = {}

    def port_specs(self) -> dict[str, PortSpec]:
        specs = {f"in{i+1}": gas_port(self.medium, "in") for i in range(self.n_inlets)}
        specs["out"] = gas_port(self.medium, "out")
        return specs

    def inlets(self, s: Scope):
        return [s.port(f"in{i+1}") for i in range(self.n_inlets)]

    def equations(self, s: Scope) -> list[S.Expr]:
        ins = self.inlets(s)
        out = s.port("out")
        eqs: list[S.Expr] = []
        # 등압 합류 (헤더 내 손실은 상류 덕트가 갖는다)
        for p in ins:
            eqs.append(p.p - out.p)
        # 질량 보존
        m_sum = S.const(0.0)
        for p in ins:
            m_sum = m_sum + p.mdot
        eqs.append(m_sum + out.mdot)
        # 에너지 보존
        h_sum = S.const(0.0)
        for p in ins:
            h_sum = h_sum + p.mdot * enthalpy(p, self.medium)
        eqs.append(h_sum + out.mdot * enthalpy(out, self.medium))
        # 화학종 보존
        for sp in self.medium.species:
            w_sum = S.const(0.0)
            for p in ins:
                w_sum = w_sum + p.mdot * p.w(sp)
            eqs.append(w_sum + out.mdot * out.w(sp))
        return eqs

    def outputs(self, s: Scope) -> dict[str, tuple[S.Expr, str]]:
        out = s.port("out")
        return {"mdot_total": (-out.mdot, "kg/s"), "T_mix": (out.T, "degC")}
