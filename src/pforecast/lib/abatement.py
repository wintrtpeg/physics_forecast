"""배기 처리 계통 컴포넌트: 장비 배기원, 습식 스크러버, 굴뚝.

NOx 농도 예측에서 물리모델이 ML 대비 결정적으로 유리한 지점이 두 군데 있다.

1. **희석**: 굴뚝 농도는 NOx 발생량을 총 배기유량으로 나눈 값이다. 가동율이 오르면
   분자와 분모가 같이 커지므로 농도는 완만하게 변한다. 이 구조를 모르는 회귀모형은
   외삽 구간에서 엉뚱한 기울기를 낸다.
2. **L/G 저하**: 순환수량이 고정된 상태에서 가스량만 늘면 액가스비가 떨어지고
   제거효율이 비선형적으로 무너진다. 학습 데이터에 고부하가 없으면 ML 은 이 변곡을
   결코 알 수 없다. 반면 물질전달식은 그대로 외삽된다.
"""

from __future__ import annotations

from ..core import symbolic as S
from ..core.component import ParamSpec, PortSpec, Scope, VarSpec
from .flow import GasComponent, _MIN_MDOT
from .gas import (
    CP_WATER_LIQ, G_ACCEL, MOLAR_MASS, P_NORMAL, R_UNIVERSAL, T_NORMAL, T_REF,
    FLUE_GAS, GasMedium, density, enthalpy, gas_port, humidity_from_rh,
    mixture_molar_mass, normal_volume_flow,
)

_D_K = (0, 0, 0, 1, 0, 0, 0)
_D_MOLAR = (0, 1, 0, 0, -1, 0, 0)
_D_RHO = (-3, 1, 0, 0, 0, 0, 0)


class ToolGroupSource(GasComponent):
    """동일 공정 장비군의 배기 발생원 (경계 컴포넌트).

    **압력 구동 모델이다.** 유량을 상수로 고정하지 않는다. 각 장비의 후드/댐퍼를
    저항으로 보고, 룸 압력과 헤더 압력의 차이로 유량이 결정된다.

        p_room - p_header = dp_design * (rho_ref/rho) * (mdot / (n_eff * m_ref))^2
        n_eff = n_tools * (idle_frac + (1 - idle_frac) * util)

    왜 이렇게 하는가. 유량을 고정하면 팬 성능곡선도 덕트 저항도 결과에 영향을 주지
    않는다. 그러면 "가동율을 120% 로 올리면 배기 용량이 모자라 희석이 덜 되고 농도가
    급등한다" 는, 현장에서 가장 알고 싶은 비선형성을 모델이 아예 표현하지 못한다.
    압력 구동으로 두면 장비가 늘수록 병렬 저항이 줄어 유량이 늘지만, 팬 곡선을 타고
    내려가며 포화된다. 이 포화가 외삽 구간의 농도 급등을 만든다.

    NOx 발생량은 ``n_tools * (ef_idle + util * ef_process)`` 다. 대기 중에도 퍼지
    가스 때문에 기저 발생량이 있고 저부하에서는 이 항이 지배적이다. 이 구조가 있어야
    저부하 데이터로 보정해도 고부하 외삽이 무너지지 않는다.
    """

    PARAMS = {
        "n_tools": ParamSpec(10.0, "1", "설치 대수", lo=0.0, hi=1e4),
        "util": ParamSpec(0.8, "1", "가동율 (0~1)", lo=0.0, hi=1.5),
        "q_tool": ParamSpec(200.0, "Nm3/h", "설계 흡입압에서의 대당 배기 풍량", lo=0.0, hi=1e5, tunable=True),
        "dp_design": ParamSpec(150.0, "mmAq", "후드 설계 흡입압", lo=1.0, hi=1e4, tunable=True),
        "idle_frac": ParamSpec(0.35, "1", "대기 장비의 댐퍼 개도 비율", lo=0.0, hi=1.0),
        "ef_idle": ParamSpec(0.5, "mg/s", "대당 대기 시 NOx 발생량", lo=0.0, hi=1e4, tunable=True),
        "ef_process": ParamSpec(5.0, "mg/s", "대당 가동 시 추가 NOx 발생량", lo=0.0, hi=1e4, tunable=True),
        "T_exh": ParamSpec(45.0, "degC", "배기 온도"),
        "rh_exh": ParamSpec(0.45, "1", "배기 상대습도", lo=0.0, hi=1.0),
        "w_O2": ParamSpec(0.2314, "1", "배기 산소 질량분율 (건조 기준)", lo=0.0, hi=0.3),
        "p_room": ParamSpec(101325.0, "Pa", "클린룸 압력"),
        "rho_ref": ParamSpec(1.2, "kg/m3", "설계 기준 밀도"),
    }
    VARS = {
        "rho": VarSpec("kg/m3", start=1.12, lo=0.05, hi=10.0),
    }

    def port_specs(self) -> dict[str, PortSpec]:
        return {"outlet": gas_port(self.medium, "out")}

    def equations(self, s: Scope) -> list[S.Expr]:
        o = s.port("outlet")
        m = -o.mdot                                  # 유출 질량유량 (양수)
        M = mixture_molar_mass(o, self.medium)
        rho_n = (S.const(P_NORMAL, (-1, 1, -2, 0, 0, 0, 0)) * M
                 / (S.const(R_UNIVERSAL, (2, 1, -2, -1, -1, 0, 0)) * S.const(T_NORMAL, _D_K)))
        n_eff = s.n_tools * (s.idle_frac + (S.const(1.0) - s.idle_frac) * s.util)
        m_ref = n_eff * s.q_tool * rho_n             # 설계 흡입압에서의 군 전체 유량
        nox_rate = s.n_tools * (s.ef_idle + s.util * s.ef_process)
        return [
            s.rho - density(o, self.medium),
            # 병렬 후드 저항. n_eff=0 이면 m=0 으로 자연히 닫힌다.
            (s.p_room - o.p) * m_ref * m_ref * s.rho - s.dp_design * s.rho_ref * m * m,
            o.T - s.T_exh,
            m * o.w("NOx") - nox_rate,
            o.w("H2O") - humidity_from_rh(s.rh_exh, s.T_exh, o.p, self.medium),
            # 공기 조성은 건조 기준 물성이다. 습분은 그 위에 얹히므로 (1-w_H2O) 를 곱한다.
            o.w("O2") - s.w_O2 * (S.const(1.0) - o.w("H2O")),
        ]

    def outputs(self, s: Scope) -> dict[str, tuple[S.Expr, str]]:
        o = s.port("outlet")
        n_eff = s.n_tools * (s.idle_frac + (S.const(1.0) - s.idle_frac) * s.util)
        return {
            "mdot": (-o.mdot, "kg/s"),
            "Q": (-o.mdot / s.rho, "CMM"),
            "nox_rate": (s.n_tools * (s.ef_idle + s.util * s.ef_process), "g/h"),
            "n_active": (s.n_tools * s.util, "1"),
            "n_eff": (n_eff, "1"),
            "suction_mmAq": (o.p - s.p_room, "mmAq"),
            "q_per_tool": (-o.mdot / s.rho / n_eff, "CMM"),
        }

    def initial_guess(self, inlets):
        from .gas import density_num, humidity_from_rh_num, normal_density_num
        pv = self.param_value
        util, n = pv("util"), pv("n_tools")
        T = pv("T_exh")
        w_h2o = humidity_from_rh_num(pv("rh_exh"), T, 101325.0, self.medium)
        w = {"O2": pv("w_O2") * (1.0 - w_h2o), "H2O": w_h2o, "NOx": 0.0}
        rho_n = normal_density_num(w, self.medium)
        n_eff = n * (pv("idle_frac") + (1.0 - pv("idle_frac")) * util)
        mdot = max(n_eff * pv("q_tool") * rho_n, 1e-6)
        w["NOx"] = n * (pv("ef_idle") + util * pv("ef_process")) / mdot
        state = {"mdot": mdot, "T": T, **{f"w_{k}": v for k, v in w.items()}}
        rho = density_num(101325.0, T, w, self.medium)
        return {"ports": {"outlet": state}, "vars": {"rho": rho}}


class WetScrubber(GasComponent):
    """습식 스크러버 (충전탑 기준).

    제거효율은 물질전달 단위수(NTU)에서 온다.

        L/G  = Q_liquid / Q_gas
        NTU  = a * (L/G)^b
        eta  = eta_max * (1 - exp(-NTU))

    가스가 증발수를 실어 나가면서 단열 가습·냉각이 일어난다. 출구는 포화에 가깝다고
    보고 상대습도 ``rh_out`` 으로 닫는다. 이 에너지식이 있어야 굴뚝의 건조 기준
    환산이 맞는다 (습분이 틀리면 mg/Sm3 가 통째로 틀어진다).
    """

    PARAMS = {
        "eta_max": ParamSpec(0.55, "1", "도달 가능 최대 제거효율", lo=0.0, hi=1.0, tunable=True),
        "ntu_a": ParamSpec(0.9, "1", "NTU 계수", lo=1e-4, hi=100.0, tunable=True),
        "ntu_b": ParamSpec(0.55, "1", "NTU L/G 지수", lo=0.05, hi=2.0, tunable=True),
        "L": ParamSpec(0.9, "m3/min", "순환수 유량", lo=1e-6, hi=100.0),
        "K": ParamSpec(3.0, "1/m4", "충전층 저항계수", lo=0.0, hi=1e6, tunable=True),
        "rh_out": ParamSpec(0.97, "1", "출구 상대습도", lo=0.5, hi=1.0, tunable=True),
        "T_water": ParamSpec(22.0, "degC", "순환수 온도"),
    }
    VARS = {
        "eta": VarSpec("1", start=0.4, lo=0.0, hi=0.9999, desc="NOx 제거효율"),
        "m_evap": VarSpec("kg/s", start=0.02, lo=-1.0, hi=10.0, desc="증발 수분량"),
        "dp": VarSpec("Pa", start=300.0, desc="충전층 압력손실"),
        "rho": VarSpec("kg/m3", start=1.13, lo=0.05, hi=10.0),
        "LG": VarSpec("1", start=1.5e-3, lo=1e-9, hi=1.0, desc="액가스비 (m3/m3)"),
    }

    def port_specs(self) -> dict[str, PortSpec]:
        return {"a": gas_port(self.medium, "in"), "b": gas_port(self.medium, "out")}

    def equations(self, s: Scope) -> list[S.Expr]:
        a, b = s.port("a"), s.port("b")
        m_out = -b.mdot
        ntu = s.ntu_a * s.LG ** s.ntu_b
        cp_w = S.const(CP_WATER_LIQ, (2, 0, -2, -1, 0, 0, 0))
        return [
            a.mdot + b.mdot + s.m_evap,
            s.rho - density(a, self.medium),
            s.LG * (a.mdot / s.rho) - s.L,
            s.eta - s.eta_max * (S.const(1.0) - S.exp(-ntu)),
            m_out * b.w("NOx") - a.mdot * a.w("NOx") * (S.const(1.0) - s.eta),
            m_out * b.w("H2O") - (a.mdot * a.w("H2O") + s.m_evap),
            m_out * b.w("O2") - a.mdot * a.w("O2"),
            b.w("H2O") - humidity_from_rh(s.rh_out, b.T, b.p, self.medium),
            (a.mdot * enthalpy(a, self.medium) + b.mdot * enthalpy(b, self.medium)
             + s.m_evap * cp_w * (s.T_water - S.const(T_REF, _D_K))),
            s.dp - s.K * S.signed_pow(a.mdot, 2.0) / s.rho,
            a.p - b.p - s.dp,
        ]

    def outputs(self, s: Scope) -> dict[str, tuple[S.Expr, str]]:
        a = s.port("a")
        return {
            "eta": (s.eta, "%"),
            "LG_L_per_m3": (s.LG * S.const(1000.0), "1"),
            "Q_gas": (a.mdot / s.rho, "CMM"),
            "dp_mmAq": (s.dp, "mmAq"),
            "T_out": (s.port("b").T, "degC"),
            "m_evap": (s.m_evap, "kg/s"),
        }

    def initial_guess(self, inlets):
        import math
        from .gas import CP_MASS, density_num, humidity_from_rh_num
        if not inlets:
            return {}
        state = self._mix_inlets(inlets)
        pv = self.param_value
        w_in = {sp: state.get(f"w_{sp}", 0.0) for sp in self.medium.species}
        rho = density_num(101325.0, state["T"], w_in, self.medium)
        Q = state["mdot"] / rho
        LG = max(pv("L") / max(Q, 1e-9), 1e-9)
        eta = pv("eta_max") * (1.0 - math.exp(-pv("ntu_a") * LG ** pv("ntu_b")))
        # 단열 가습: 출구 온도를 습구온도 근처로 잡는다(초기값이므로 근사로 충분)
        T_out = 0.45 * state["T"] + 0.55 * pv("T_water")
        w_h2o_out = humidity_from_rh_num(pv("rh_out"), T_out, 101325.0, self.medium)
        m_evap = max(state["mdot"] * (w_h2o_out - w_in.get("H2O", 0.0)) / max(1.0 - w_h2o_out, 0.1), 0.0)
        m_out = state["mdot"] + m_evap
        out = {"mdot": m_out, "T": T_out}
        for sp in self.medium.species:
            f = (1.0 - eta) if sp == "NOx" else 1.0
            out[f"w_{sp}"] = w_h2o_out if sp == "H2O" else state.get(f"w_{sp}", 0.0) * f * state["mdot"] / m_out
        return {
            "ports": {"b": out},
            "vars": {"eta": eta, "LG": LG, "m_evap": max(m_evap, 1e-6), "rho": rho,
                     "dp": pv("K") * state["mdot"] ** 2 / rho},
        }


class Stack(GasComponent):
    """옥상 굴뚝 (대기 경계).

    규제 농도는 '건조 기준, 표준산소 보정, 표준상태' 값이다. 세 보정을 전부
    방정식으로 들고 있어야 현장 TMS 값과 직접 비교할 수 있다.

        C_dry  = C_wet / (1 - y_H2O)
        C_corr = C_dry * (21 - O2_ref) / (21 - O2_meas)

    통풍력(draft)도 포함한다. 여름/겨울 외기 밀도차가 굴뚝 흡입압에 수 mmAq 를
    만들고, 이는 팬 동작점을 통해 유량에 영향을 준다.
    """

    PARAMS = {
        "H": ParamSpec(25.0, "m", "굴뚝 유효 높이", lo=0.0, hi=300.0),
        "K": ParamSpec(1.2, "1/m4", "굴뚝 저항계수", lo=0.0, hi=1e6, tunable=True),
        "p_amb": ParamSpec(101325.0, "Pa", "대기압"),
        "T_amb": ParamSpec(20.0, "degC", "외기온도"),
        "O2_ref": ParamSpec(0.0, "1", "표준산소농도 (몰분율). 연소시설 4%/15% 등", lo=0.0, hi=0.20),
        "o2_corr": ParamSpec(0.0, "1", "표준산소 보정 적용 여부 (0=미적용, 1=적용)", lo=0.0, hi=1.0),
    }
    VARS = {
        "rho": VarSpec("kg/m3", start=1.10, lo=0.05, hi=10.0),
        "Q_n": VarSpec("m3/s", start=9.0, lo=1e-6, hi=1e5, desc="표준상태 체적유량"),
        "C_wet": VarSpec("mg/Nm3", start=5e-6, lo=0.0, hi=1.0, desc="습식 기준 NOx 농도"),
        "C_dry": VarSpec("mg/Nm3", start=6e-6, lo=0.0, hi=1.0, desc="건조 기준 NOx 농도"),
        "C_corr": VarSpec("mg/Nm3", start=6e-6, lo=0.0, hi=1.0, desc="표준산소 보정 NOx 농도"),
        "y_H2O": VarSpec("1", start=0.05, lo=0.0, hi=0.95, desc="수증기 몰분율"),
        "y_O2_dry": VarSpec("1", start=0.208, lo=0.0, hi=0.30, desc="건조 기준 산소 몰분율"),
        "dp": VarSpec("Pa", start=100.0),
    }

    def port_specs(self) -> dict[str, PortSpec]:
        return {"a": gas_port(self.medium, "in")}

    def equations(self, s: Scope) -> list[S.Expr]:
        a = s.port("a")
        # 'a' 는 입구 포트이므로 through 부호 규약상 유입이 (+) 다.
        m_out = a.mdot
        M = mixture_molar_mass(a, self.medium)
        M_h2o = S.const(MOLAR_MASS["H2O"], _D_MOLAR)
        M_o2 = S.const(MOLAR_MASS["O2"], _D_MOLAR)
        rho_amb = (s.p_amb * S.const(MOLAR_MASS["air"], _D_MOLAR)
                   / (S.const(R_UNIVERSAL, (2, 1, -2, -1, -1, 0, 0)) * s.T_amb))
        draft = (rho_amb - s.rho) * S.const(G_ACCEL, (1, 0, -2, 0, 0, 0, 0)) * s.H
        o2_std = S.const(0.2095)
        return [
            s.rho - density(a, self.medium),
            s.Q_n - normal_volume_flow(m_out, a, self.medium),
            s.C_wet * s.Q_n - m_out * a.w("NOx"),
            s.y_H2O * M_h2o - a.w("H2O") * M,
            s.C_dry * (S.const(1.0) - s.y_H2O) - s.C_wet,
            s.y_O2_dry * (S.const(1.0) - s.y_H2O) * M_o2 - a.w("O2") * M,
            # o2_corr=0 이면 양변의 o2_std 가 약분되어 C_corr = C_dry 가 된다.
            # 팹 배기는 사실상 공기라 O2 보정을 적용하면 분모가 0 에 붙어 발산한다.
            (s.C_corr * (o2_std - s.o2_corr * s.y_O2_dry)
             - s.C_dry * (o2_std - s.o2_corr * s.O2_ref)),
            s.dp - s.K * S.signed_pow(a.mdot, 2.0) / s.rho,
            a.p - s.p_amb - s.dp + draft,
        ]

    def outputs(self, s: Scope) -> dict[str, tuple[S.Expr, str]]:
        a = s.port("a")
        M = mixture_molar_mass(a, self.medium)
        M_nox = S.const(MOLAR_MASS["NOx"], _D_MOLAR)
        y_nox = a.w("NOx") * M / M_nox
        return {
            "C_corr": (s.C_corr, "mg/Nm3"),
            "C_dry": (s.C_dry, "mg/Nm3"),
            "C_wet": (s.C_wet, "mg/Nm3"),
            "ppm_wet": (y_nox, "ppm"),
            "ppm_dry": (y_nox / (S.const(1.0) - s.y_H2O), "ppm"),
            "Q_n": (s.Q_n, "Nm3/h"),
            "T_stack": (a.T, "degC"),
            "y_H2O": (s.y_H2O, "%"),
            "O2_dry": (s.y_O2_dry, "%"),
            "nox_mass_rate": (a.mdot * a.w("NOx"), "g/h"),
            "p_in_mmAq": (a.p - s.p_amb, "mmAq"),
        }

    def initial_guess(self, inlets):
        from .gas import MOLAR_MASS, density_num, molar_mass_num, normal_density_num
        if not inlets:
            return {}
        state = self._mix_inlets(inlets)
        pv = self.param_value
        w = {sp: state.get(f"w_{sp}", 0.0) for sp in self.medium.species}
        rho = density_num(pv("p_amb"), state["T"], w, self.medium)
        M = molar_mass_num(w, self.medium)
        Q_n = state["mdot"] / normal_density_num(w, self.medium)
        C_wet = state["mdot"] * w.get("NOx", 0.0) / max(Q_n, 1e-9)
        y_h2o = min(w.get("H2O", 0.0) * M / MOLAR_MASS["H2O"], 0.9)
        C_dry = C_wet / max(1.0 - y_h2o, 0.1)
        y_o2 = w.get("O2", 0.0) * M / MOLAR_MASS["O2"] / max(1.0 - y_h2o, 0.1)
        oc = pv("o2_corr")
        C_corr = C_dry * (0.2095 - oc * pv("O2_ref")) / max(0.2095 - oc * y_o2, 1e-3)
        return {"vars": {
            "rho": rho, "Q_n": Q_n, "C_wet": C_wet, "C_dry": C_dry,
            "C_corr": min(C_corr, 0.99), "y_H2O": y_h2o, "y_O2_dry": y_o2,
            "dp": pv("K") * state["mdot"] ** 2 / rho,
        }}
