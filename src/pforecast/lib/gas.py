"""가스 매질과 열물성.

CoolProp 없이 동작한다. 배기 계통은 상압 근처의 희박 혼합기라 이상기체 + 정압비열
상수 가정으로 충분한 정확도가 나온다(NOx 농도 계산에서 지배적인 오차원은 물성이
아니라 유량과 제거효율이다).

추적 화학종은 시스템 단위로 정한다. NOx 스택 모델은 ('NOx', 'H2O', 'O2') 를 쓴다.
* NOx : 규제 대상. NO2 환산 기준(46.0055 g/mol)으로 다룬다.
* H2O : 건조 기준(dry basis) 환산에 필요.
* O2  : 표준산소 보정에 필요.
나머지(N2 등)는 'carrier' 로 묶어 질량분율 합으로 암묵 처리한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core import symbolic as S
from ..core.component import PortSpec

R_UNIVERSAL = 8.314462618      # J/(mol*K)
G_ACCEL = 9.80665              # m/s^2
T_NORMAL = 273.15              # K, 표준상태(0 degC)
P_NORMAL = 101325.0            # Pa
T_REF = 273.15                 # 엔탈피 기준 온도
H_FG_0 = 2_501_000.0           # J/kg, 0 degC 에서 물의 증발잠열
CP_WATER_LIQ = 4186.0          # J/(kg*K)

#: 몰질량 [kg/mol]
MOLAR_MASS = {
    "NOx": 0.0460055,   # NO2 환산
    "NO": 0.0300061,
    "NO2": 0.0460055,
    "H2O": 0.0180153,
    "O2": 0.0319988,
    "N2": 0.0280134,
    "CO2": 0.0440095,
    "SO2": 0.0640638,
    "NH3": 0.0170305,
    "air": 0.0289647,
}

#: 정압비열 [J/(kg*K)], 300~500K 대표값
CP_MASS = {
    "NOx": 805.0,
    "NO": 995.0,
    "NO2": 805.0,
    "H2O": 1864.0,   # 수증기
    "O2": 918.0,
    "N2": 1040.0,
    "CO2": 844.0,
    "SO2": 622.0,
    "NH3": 2175.0,
    "air": 1006.0,
}


@dataclass(frozen=True)
class GasMedium:
    """추적 화학종 목록 + 캐리어 가스 물성."""

    species: tuple[str, ...] = ("NOx", "H2O", "O2")
    carrier: str = "N2"
    name: str = "flue_gas"

    def molar_mass(self, sp: str) -> float:
        return MOLAR_MASS[sp]

    def cp(self, sp: str) -> float:
        return CP_MASS[sp]

    @property
    def carrier_M(self) -> float:
        return MOLAR_MASS[self.carrier]

    @property
    def carrier_cp(self) -> float:
        return CP_MASS[self.carrier]


FLUE_GAS = GasMedium()


def gas_port(medium: GasMedium, role: str, p_start: float = 101325.0,
             T_start: float = 313.15, mdot_start: float = 10.0) -> PortSpec:
    """배기 가스 포트 사양을 만든다.

    across  : p     [Pa]
    through : mdot  [kg/s]  (컴포넌트로 들어가는 방향이 +)
    stream  : T [K], w_<species> [kg/kg]
    """
    stream = [("T", "K")] + [(f"w_{sp}", "1") for sp in medium.species]
    starts = {"p": p_start, "mdot": mdot_start, "T": T_start}
    bounds = {"T": (150.0, 1500.0), "p": (1.0e4, 5.0e5)}
    for sp in medium.species:
        starts[f"w_{sp}"] = 0.01 if sp != "NOx" else 1e-5
        bounds[f"w_{sp}"] = (0.0, 1.0)
    return PortSpec(
        across=(("p", "Pa"),),
        through=(("mdot", "kg/s"),),
        stream=tuple(stream),
        role=role,
        kind=f"gas:{medium.name}:{','.join(medium.species)}",
        starts=starts,
        bounds=bounds,
    )


# --- 심볼릭 물성식 --------------------------------------------------------

def mixture_molar_mass(port, medium: GasMedium) -> S.Expr:
    """1/M = sum(w_i / M_i). 캐리어는 잔여 질량분율로 처리."""
    inv = S.const(0.0)
    w_sum = S.const(0.0)
    for sp in medium.species:
        w = port.w(sp)
        inv = inv + w / S.const(medium.molar_mass(sp), (0, 1, 0, 0, -1, 0, 0))
        w_sum = w_sum + w
    inv = inv + (S.const(1.0) - w_sum) / S.const(medium.carrier_M, (0, 1, 0, 0, -1, 0, 0))
    return S.const(1.0) / inv


def mixture_cp(port, medium: GasMedium) -> S.Expr:
    """cp_mix = sum(w_i * cp_i)."""
    out = S.const(0.0)
    w_sum = S.const(0.0)
    for sp in medium.species:
        w = port.w(sp)
        out = out + w * S.const(medium.cp(sp), (2, 0, -2, -1, 0, 0, 0))
        w_sum = w_sum + w
    return out + (S.const(1.0) - w_sum) * S.const(medium.carrier_cp, (2, 0, -2, -1, 0, 0, 0))


def density(port, medium: GasMedium) -> S.Expr:
    """이상기체 rho = p*M/(R*T)."""
    M = mixture_molar_mass(port, medium)
    return port.p * M / (S.const(R_UNIVERSAL, (2, 1, -2, -1, -1, 0, 0)) * port.T)


def enthalpy(port, medium: GasMedium) -> S.Expr:
    """기준 0 degC 의 비엔탈피. 수증기 잠열을 포함해야 증발 냉각이 닫힌다."""
    cp = mixture_cp(port, medium)
    h = cp * (port.T - S.const(T_REF, (0, 0, 0, 1, 0, 0, 0)))
    if "H2O" in medium.species:
        h = h + port.w("H2O") * S.const(H_FG_0, (2, 0, -2, 0, 0, 0, 0))
    return h


def p_sat_water(T: S.Expr) -> S.Expr:
    """Magnus 식. 매끄럽고 미분 가능해서 뉴턴법에 안전하다. [Pa]"""
    Tc = T - S.const(273.15, (0, 0, 0, 1, 0, 0, 0))
    Tc_d = Tc / S.const(1.0, (0, 0, 0, 1, 0, 0, 0))   # 무차원화
    return S.const(610.94, (-1, 1, -2, 0, 0, 0, 0)) * S.exp(
        S.const(17.625) * Tc_d / (Tc_d + S.const(243.04))
    )


def humidity_from_rh(rh: S.Expr, T: S.Expr, p: S.Expr, medium: GasMedium) -> S.Expr:
    """상대습도로부터 수증기 질량분율(습공기 전체 질량 기준 근사)."""
    p_v = rh * p_sat_water(T)
    M_w = S.const(MOLAR_MASS["H2O"], (0, 1, 0, 0, -1, 0, 0))
    M_d = S.const(medium.carrier_M, (0, 1, 0, 0, -1, 0, 0))
    y = p_v / p                                   # 몰분율
    return y * M_w / (y * M_w + (S.const(1.0) - y) * M_d)


def normal_volume_flow(mdot: S.Expr, port, medium: GasMedium) -> S.Expr:
    """표준상태(0 degC, 1 atm) 기준 체적유량 [Nm3/s]."""
    M = mixture_molar_mass(port, medium)
    rho_n = S.const(P_NORMAL, (-1, 1, -2, 0, 0, 0, 0)) * M / (
        S.const(R_UNIVERSAL, (2, 1, -2, -1, -1, 0, 0)) * S.const(T_NORMAL, (0, 0, 0, 1, 0, 0, 0))
    )
    return mdot / rho_n


# --- 수치 버전 (초기값 추정용) --------------------------------------------

def molar_mass_num(w: dict[str, float], medium: GasMedium) -> float:
    inv = 0.0
    w_sum = 0.0
    for sp in medium.species:
        v = w.get(sp, 0.0)
        inv += v / MOLAR_MASS[sp]
        w_sum += v
    inv += (1.0 - w_sum) / medium.carrier_M
    return 1.0 / inv


def cp_num(w: dict[str, float], medium: GasMedium) -> float:
    out = 0.0
    w_sum = 0.0
    for sp in medium.species:
        v = w.get(sp, 0.0)
        out += v * CP_MASS[sp]
        w_sum += v
    return out + (1.0 - w_sum) * medium.carrier_cp


def density_num(p: float, T: float, w: dict[str, float], medium: GasMedium) -> float:
    return p * molar_mass_num(w, medium) / (R_UNIVERSAL * T)


def p_sat_water_num(T: float) -> float:
    import math
    Tc = T - 273.15
    return 610.94 * math.exp(17.625 * Tc / (Tc + 243.04))


def humidity_from_rh_num(rh: float, T: float, p: float, medium: GasMedium) -> float:
    y = rh * p_sat_water_num(T) / p
    M_w, M_d = MOLAR_MASS["H2O"], medium.carrier_M
    return y * M_w / (y * M_w + (1.0 - y) * M_d)


def normal_density_num(w: dict[str, float], medium: GasMedium) -> float:
    return P_NORMAL * molar_mass_num(w, medium) / (R_UNIVERSAL * T_NORMAL)
