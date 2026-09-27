"""갈아끼울 수 있는 구성방정식(closure) — 슬롯과 후보.

**후보가 되는 것은 구성방정식뿐이다.** 보존법칙과 상태식은 슬롯이 아니다. 컴포넌트는
``CLOSURES`` 로 슬롯을 선언하고, ``equations()`` 안에서 ``closure_expr(슬롯, ...)`` 로
지금 고른 식을 꺼내 쓴다. 고르는 일은 사람이 한다 — 코드는 데이터로 후보를 추려
근거를 보여줄 뿐이다 (``closure_advice.py``).

슬롯마다 두 역할을 표시한다.

* ``baseline`` — 가장 단순한 형태(하한선). 비교에 반드시 들어간다. 이것보다 낫지 않으면
  복잡도가 값을 하는지 알 수 없다.
* ``standard`` — 이론 표준형. 기본값이다. 데이터가 후보를 가르지 못하면 이것을 유지한다.

후보 파라미터의 설계값은 **0 이 아니어야 한다.** 보정기가 설계값의 배율(0.2~5배)로
경계를 잡기 때문에 0 이면 움직일 수 없다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from ..core import symbolic as S
from ..core.component import ParamSpec, Scope
from .explain import CLOSURE, Law

BASELINE = "baseline"
STANDARD = "standard"
EXTENDED = "extended"
ALTERNATIVE = "alternative"

ROLE_LABEL = {BASELINE: "하한선 (가장 단순)", STANDARD: "이론 표준형", EXTENDED: "확장형 (파라미터 추가)",
              ALTERNATIVE: "대안 형태"}


@dataclass(frozen=True)
class Closure:
    """슬롯에 끼울 수 있는 식 하나."""

    id: str
    title: str
    law: Law
    params: dict[str, ParamSpec]
    #: 심볼릭 식. ``(scope, ctx) -> Expr``. ctx 는 컴포넌트가 넘기는 지역 식(L/G, 온도 등)
    expr: Callable[[Scope, dict], S.Expr]
    #: 같은 식의 수치 버전 (초기값 추정용). ``(param_value, ctx) -> float``
    value: Callable[[Callable[[str], float], dict], float]
    #: 후보 비교 때 보정할 파라미터 (식별 가능한 최소 집합)
    fit: tuple[str, ...] = ()
    role: str = ALTERNATIVE
    #: 이 형태가 다른 형태와 갈리는 변수 (컴포넌트 기준 이름: 변수·파라미터·포트변수 a.T)
    needs: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class ClosureSlot:
    key: str
    title: str
    options: tuple[Closure, ...]
    default: str

    def option(self, cid: str) -> Closure:
        for o in self.options:
            if o.id == cid:
                return o
        raise KeyError(f"슬롯 {self.key!r} 에 후보 {cid!r} 가 없습니다. 가능: {[o.id for o in self.options]}")

    @property
    def baseline(self) -> Closure:
        return next((o for o in self.options if o.role == BASELINE), self.option(self.default))


@dataclass(frozen=True)
class SlotLaw:
    """``LAWS`` 안에서 슬롯 자리를 표시한다. 화면에는 지금 고른 식이 들어간다."""

    key: str


class ClosureMixin:
    """구성방정식 슬롯을 가진 컴포넌트. ``GasComponent`` 앞에 섞는다."""

    CLOSURES: tuple[ClosureSlot, ...] = ()

    def __init__(self, name: str, *args, closures: dict[str, str] | None = None, **params):
        self._closure_choice = {sl.key: sl.default for sl in self.CLOSURES}
        for key, cid in (closures or {}).items():
            self.slot(key).option(cid)
            self._closure_choice[key] = cid
        # 고르지 않은 후보의 파라미터 값도 받아 둔다 (설정에서 후보만 바꿔도 오류가 나지 않게,
        # 나중에 그 후보로 돌아가면 이 값으로 시작한다)
        current = self.param_specs()
        spare = {}
        for sl in self.CLOSURES:
            for o in sl.options:
                for pname, spec in o.params.items():
                    if pname in params and pname not in current:
                        spare[pname] = spec.si(params.pop(pname))
        super().__init__(name, *args, **params)
        self._param_values.update(spare)

    # --- 조회 ---
    def slot(self, key: str) -> ClosureSlot:
        for sl in self.CLOSURES:
            if sl.key == key:
                return sl
        raise KeyError(f"{type(self).__name__} 에 슬롯 {key!r} 가 없습니다. "
                       f"가능: {[s.key for s in self.CLOSURES]}")

    def closure(self, key: str) -> Closure:
        return self.slot(key).option(self._closure_choice[key])

    @property
    def closure_choices(self) -> dict[str, str]:
        return dict(self._closure_choice)

    def param_specs(self):
        specs = dict(self.PARAMS)
        for sl in self.CLOSURES:
            specs.update(self.closure(sl.key).params)
        return specs

    # --- 식 ---
    def closure_expr(self, key: str, s: Scope, **ctx) -> S.Expr:
        return self.closure(key).expr(s, ctx)

    def closure_value(self, key: str, **ctx) -> float:
        return self.closure(key).value(self.param_value, ctx)

    def laws(self) -> list[Law]:
        return [self.closure(x.key).law if isinstance(x, SlotLaw) else x for x in self.LAWS]

    # --- 고르기 ---
    def set_closure(self, key: str, cid: str) -> None:
        """다른 후보로 바꾼다. 새 파라미터는 선언한 설계값으로 시작한다."""
        opt = self.slot(key).option(cid)
        self._closure_choice[key] = cid
        for pname, spec in opt.params.items():
            self._param_values.setdefault(pname, spec.si())


def apply_closures(system, choices: dict[str, str] | None) -> list[str]:
    """``{"SCR.eta": "langmuir"}`` 를 시스템에 적용한다. 컴파일 전에 불러야 한다."""
    applied = []
    for full, cid in (choices or {}).items():
        inst, key = full.split(".", 1)
        comp = system.components.get(inst)
        if comp is None or not hasattr(comp, "set_closure"):
            raise KeyError(f"구성방정식 슬롯 {full!r} 를 찾을 수 없습니다")
        comp.set_closure(key, cid)
        applied.append(f"{full} = {cid}")
    return applied


def closure_slots(system) -> list[tuple[str, ClosureSlot]]:
    """시스템의 (인스턴스 이름, 슬롯) 목록. 계통에 있는 컴포넌트 종류가 슬롯을 정한다."""
    out = []
    for name, comp in system.components.items():
        for sl in getattr(type(comp), "CLOSURES", ()):
            out.append((name, sl))
    return out


# ---------------------------------------------------------------------------
# 후보 목록
# ---------------------------------------------------------------------------

_T_REF = 313.15   # K, 온도항 기준

_P_ETA_MAX = ParamSpec(0.55, "1", "도달 가능 최대 제거효율", lo=0.0, hi=1.0, tunable=True)
_P_NTU_A = ParamSpec(0.9, "1", "NTU 계수", lo=1e-4, hi=100.0, tunable=True)
_P_NTU_B = ParamSpec(0.55, "1", "NTU L/G 지수", lo=0.05, hi=2.0, tunable=True)


def _ntu(s, ctx):
    return s.ntu_a * ctx["LG"] ** s.ntu_b


SCRUBBER_ETA = ClosureSlot("eta", "NOx 제거효율", default="ntu_power", options=(
    Closure(
        "constant", "제거효율 상수",
        Law(CLOSURE, "NOx 제거효율 — 상수", "η = `eta0`"),
        {"eta0": ParamSpec(0.20, "1", "고정 제거효율", lo=0.0, hi=0.95, tunable=True)},
        lambda s, c: s.eta0,
        lambda pv, c: pv("eta0"),
        fit=("eta0",), role=BASELINE,
        note="L/G 와 무관. 순환수가 바뀌어도 효율이 그대로라고 본다."),
    Closure(
        "ntu_power", "물질전달 단위수(NTU) 거듭제곱형",
        Law(CLOSURE, "NOx 제거효율 (물질전달 단위수)",
            "η = `eta_max` · (1 − e^{−NTU}),   NTU = `ntu_a` · (L/G)^{`ntu_b`}"),
        {"eta_max": _P_ETA_MAX, "ntu_a": _P_NTU_A, "ntu_b": _P_NTU_B},
        lambda s, c: s.eta_max * (S.const(1.0) - S.exp(-_ntu(s, c))),
        lambda pv, c: pv("eta_max") * (1.0 - math.exp(-pv("ntu_a") * c["LG"] ** pv("ntu_b"))),
        fit=("ntu_a",), role=STANDARD, needs=("LG",),
        note="충전탑 물질전달 이론의 표준형. 순환수가 줄면 효율이 지수적으로 떨어진다."),
    Closure(
        "ntu_power_b", "NTU 거듭제곱형 (L/G 지수도 보정)",
        Law(CLOSURE, "NOx 제거효율 (물질전달 단위수, 지수 보정)",
            "η = `eta_max` · (1 − e^{−NTU}),   NTU = `ntu_a` · (L/G)^{`ntu_b`}"),
        {"eta_max": _P_ETA_MAX, "ntu_a": _P_NTU_A, "ntu_b": _P_NTU_B},
        lambda s, c: s.eta_max * (S.const(1.0) - S.exp(-_ntu(s, c))),
        lambda pv, c: pv("eta_max") * (1.0 - math.exp(-pv("ntu_a") * c["LG"] ** pv("ntu_b"))),
        fit=("ntu_a", "ntu_b"), role=EXTENDED, needs=("LG",),
        note="표준형과 같은 식에서 L/G 지수까지 데이터로 정한다. L/G 가 넓게 변해야 지수가 정해진다."),
    Closure(
        "langmuir", "포화형 (Langmuir)",
        Law(CLOSURE, "NOx 제거효율 — 포화형", "η = `eta_max` · (L/G) / (`k_LG` + L/G)"),
        {"eta_max": _P_ETA_MAX,
         "k_LG": ParamSpec(3.5e-3, "1", "반포화 액가스비", lo=1e-6, hi=1.0, tunable=True)},
        lambda s, c: s.eta_max * c["LG"] / (s.k_LG + c["LG"]),
        lambda pv, c: pv("eta_max") * c["LG"] / (pv("k_LG") + c["LG"]),
        fit=("k_LG",), role=ALTERNATIVE, needs=("LG",),
        note="L/G 가 커지면 포화. 낮은 L/G 에서 거의 선형."),
    Closure(
        "ntu_power_T", "NTU 거듭제곱형 + 가스 온도항",
        Law(CLOSURE, "NOx 제거효율 (NTU + 온도 보정)",
            "η = `eta_max` · (1 − e^{−NTU}),   NTU = `ntu_a` · (L/G)^{`ntu_b`} · (T_{in} / 313.15 K)^{`ntu_c`}"),
        {"eta_max": _P_ETA_MAX, "ntu_a": _P_NTU_A, "ntu_b": _P_NTU_B,
         "ntu_c": ParamSpec(1.0, "1", "NTU 온도 지수", lo=-6.0, hi=6.0, tunable=True)},
        lambda s, c: s.eta_max * (S.const(1.0) - S.exp(
            -_ntu(s, c) * (c["T"] / S.const(_T_REF, (0, 0, 0, 1, 0, 0, 0))) ** s.ntu_c)),
        lambda pv, c: pv("eta_max") * (1.0 - math.exp(
            -pv("ntu_a") * c["LG"] ** pv("ntu_b") * (c["T"] / _T_REF) ** pv("ntu_c"))),
        fit=("ntu_a", "ntu_c"), role=EXTENDED, needs=("LG", "a.T"),
        note="가스 온도가 물질전달에 영향을 준다고 본다. 파라미터가 하나 더 많다."),
))


def pressure_slot(key: str, title: str, k_spec: ParamSpec) -> ClosureSlot:
    """마찰·충전층 압력손실. ctx: m (질량유량), rho. 컴포넌트마다 K 의 설계값·설명이 다르다."""
    return ClosureSlot(key, title, default="quadratic", options=(
        Closure(
            "quadratic", "난류 2승 법칙",
            Law(CLOSURE, f"{title} (난류 2승)", "Δp = `K` · ṁ|ṁ| / ρ"),
            {"K": k_spec},
            lambda s, c: s.K * S.signed_pow(c["m"], 2.0) / c["rho"],
            lambda pv, c: pv("K") * c["m"] * abs(c["m"]) / c["rho"],
            fit=("K",), role=STANDARD,
            note="완전 난류에서 압력손실은 유량의 제곱에 비례한다."),
        Closure(
            "power", "거듭제곱 법칙 (지수 보정)",
            Law(CLOSURE, f"{title} (거듭제곱)", "Δp = `K` · (ṁ|ṁ| / ρ) · (|ṁ| / `m_ref`)^{`n_dp` − 2}"),
            {"K": k_spec,
             "n_dp": ParamSpec(2.0, "1", "유량 지수 (완전 난류 2, 천이 영역 1.75 안팎)", lo=1.0, hi=2.5,
                               tunable=True),
             "m_ref": ParamSpec(10.0, "kg/s", "지수 기준 질량유량", lo=1e-6, hi=1e5)},
            lambda s, c: (s.K * S.signed_pow(c["m"], 2.0) / c["rho"]
                          * (S.smooth_abs(c["m"]) / s.m_ref) ** (s.n_dp - S.const(2.0))),
            lambda pv, c: (pv("K") * c["m"] * abs(c["m"]) / c["rho"]
                           * (abs(c["m"]) / pv("m_ref")) ** (pv("n_dp") - 2.0)),
            fit=("K", "n_dp"), role=EXTENDED, needs=("a.mdot",),
            note="천이 영역이면 지수가 2 보다 작다. 유량이 넓게 변해야 지수를 정할 수 있다."),
    ))


EMISSION = ClosureSlot("emission", "NOx 발생량", default="idle_util", options=(
    Closure(
        "util_only", "가동률 비례",
        Law(CLOSURE, "NOx 발생량 — 가동률 비례", "ṁ · w_{NOx} = `n_tools` · `util` · `ef_process`"),
        {"ef_process": ParamSpec(5.0, "mg/s", "대당 가동 시 NOx 발생량", lo=0.0, hi=1e4, tunable=True)},
        lambda s, c: s.n_tools * s.util * s.ef_process,
        lambda pv, c: pv("n_tools") * pv("util") * pv("ef_process"),
        fit=("ef_process",), role=BASELINE,
        note="대기 중 장비의 발생량(퍼지 가스)을 0 으로 본다."),
    Closure(
        "idle_util", "대기 + 가동 (배출계수 2개)",
        Law(CLOSURE, "NOx 발생량 (배출계수)", "ṁ · w_{NOx} = `n_tools` · (`ef_idle` + `util` · `ef_process`)"),
        {"ef_idle": ParamSpec(0.5, "mg/s", "대당 대기 시 NOx 발생량", lo=0.0, hi=1e4, tunable=True),
         "ef_process": ParamSpec(5.0, "mg/s", "대당 가동 시 추가 NOx 발생량", lo=0.0, hi=1e4, tunable=True)},
        lambda s, c: s.n_tools * (s.ef_idle + s.util * s.ef_process),
        lambda pv, c: pv("n_tools") * (pv("ef_idle") + pv("util") * pv("ef_process")),
        fit=("ef_idle", "ef_process"), role=STANDARD, needs=("util",),
        note="대기 중에도 퍼지 가스로 기저 발생량이 있다. 저부하에서 이 항이 지배적이다."),
    Closure(
        "idle_util_power", "대기 + 가동 (비선형 부하)",
        Law(CLOSURE, "NOx 발생량 (비선형 부하)",
            "ṁ · w_{NOx} = `n_tools` · (`ef_idle` + `ef_process` · `util`^{`gamma_u`})"),
        {"ef_idle": ParamSpec(0.5, "mg/s", "대당 대기 시 NOx 발생량", lo=0.0, hi=1e4, tunable=True),
         "ef_process": ParamSpec(5.0, "mg/s", "대당 가동 시 추가 NOx 발생량", lo=0.0, hi=1e4, tunable=True),
         "gamma_u": ParamSpec(1.0, "1", "가동률 지수 (1 이면 선형)", lo=0.3, hi=3.0, tunable=True)},
        lambda s, c: s.n_tools * (s.ef_idle + s.ef_process * (s.util + S.const(1e-6)) ** s.gamma_u),
        lambda pv, c: pv("n_tools") * (pv("ef_idle") + pv("ef_process") * (pv("util") + 1e-6) ** pv("gamma_u")),
        fit=("ef_idle", "ef_process", "gamma_u"), role=EXTENDED, needs=("util",),
        note="가동률이 높을수록 대당 발생량이 비선형으로 변한다고 본다."),
))
