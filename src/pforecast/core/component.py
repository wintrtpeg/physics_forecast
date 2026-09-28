"""컴포넌트 / 포트 추상화 (도메인 비의존).

설계 핵심은 **acausal(무인과) 모델링**이다. 컴포넌트는 "입력을 받아 출력을 계산"하지
않고, 자기가 만족해야 하는 방정식만 선언한다. 누가 미지수인지는 시스템을 연결한
뒤에 구조 해석이 결정한다. 덕분에 같은 스크러버 모델을 그대로 두고
"유량을 주고 압력을 구할지 / 압력을 주고 유량을 구할지"를 바꿔 쓸 수 있다.

포트는 세 종류의 변수를 갖는다.
* across  : 연결 시 **같아진다** (압력, 전위)
* through : 연결 시 **합이 0** 이다 (질량유량, 전류). 부호는 '컴포넌트로 들어가는 방향'이 +.
* stream  : 유동을 따라 실려 가는 양 (온도, 조성). 연결 시 같아진다.

stream 변수를 Modelica 의 stream connector 처럼 완전 일반화하지 않고 1:1 연결로
제한한 것은 의도적이다. 유틸리티 배관망은 합류/분기 지점이 명확하므로
Mixer/Splitter 를 명시적 컴포넌트로 두는 편이 진단 메시지가 훨씬 친절해진다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from . import symbolic as S
from .units import DIMLESS, Dim, dim_of, to_si


@dataclass(frozen=True)
class VarSpec:
    """컴포넌트 내부 미지수."""

    unit: str = "1"
    start: float = 1.0
    lo: float = -float("inf")
    hi: float = float("inf")
    desc: str = ""

    @property
    def dim(self) -> Dim:
        return dim_of(self.unit)


@dataclass(frozen=True)
class ParamSpec:
    """설계값/물성/보정 대상 계수. 값은 선언 단위로 주고 내부는 SI 로 환산된다."""

    default: float
    unit: str = "1"
    desc: str = ""
    lo: float = -float("inf")
    hi: float = float("inf")
    #: 캘리브레이션 후보로 노출할지 여부
    tunable: bool = False

    @property
    def dim(self) -> Dim:
        return dim_of(self.unit)

    def si(self, value: float | None = None) -> float:
        return to_si(self.default if value is None else value, self.unit)


@dataclass(frozen=True)
class PortSpec:
    across: tuple[tuple[str, str], ...] = ()
    through: tuple[tuple[str, str], ...] = ()
    stream: tuple[tuple[str, str], ...] = ()
    #: "in" 이면 유동이 들어오는 쪽, "out" 이면 나가는 쪽. 연결 방향 검증에 쓴다.
    role: str = "in"
    kind: str = "generic"
    #: 포트 변수의 초기 추정값. 뉴턴법 출발점이 현실적이어야 수렴한다.
    starts: dict[str, float] = field(default_factory=dict)
    #: 물리적으로 허용되는 범위. 질량분율 [0,1], 절대온도 >0 같은 것.
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def all_vars(self) -> Iterable[tuple[str, str]]:
        yield from self.across
        yield from self.through
        yield from self.stream

    def start_of(self, name: str) -> float:
        return self.starts.get(name, 1.0)

    def bound_of(self, name: str) -> tuple[float, float]:
        return self.bounds.get(name, (-float("inf"), float("inf")))


class PortView:
    """방정식 작성 시 포트 변수를 ``a.p``, ``a.w("NOx")`` 로 꺼내 쓰기 위한 뷰."""

    def __init__(self, owner: str, pname: str, spec: PortSpec, syms: dict[str, S.Sym]):
        self._owner = owner
        self._name = pname
        self.spec = spec
        self._syms = syms

    def __getattr__(self, item: str) -> S.Sym:
        try:
            return self._syms[item]
        except KeyError:
            raise AttributeError(
                f"포트 {self._owner}.{self._name} 에 변수 {item!r} 가 없습니다. "
                f"가능한 변수: {sorted(self._syms)}"
            ) from None

    def w(self, species: str) -> S.Sym:
        return self.__getattr__(f"w_{species}")

    def has(self, item: str) -> bool:
        return item in self._syms

    @property
    def full_name(self) -> str:
        return f"{self._owner}.{self._name}"


class Scope:
    """``equations()`` 안에서 쓰는 이름공간. 변수/파라미터/포트를 심볼로 노출한다."""

    def __init__(self, comp: "Component"):
        self._comp = comp
        self._vars: dict[str, S.Sym] = {}
        self._pars: dict[str, S.Sym] = {}
        self._ports: dict[str, PortView] = {}

    def __getattr__(self, item: str) -> S.Sym:
        if item in self._vars:
            return self._vars[item]
        if item in self._pars:
            return self._pars[item]
        raise AttributeError(
            f"{self._comp.name}: 변수/파라미터 {item!r} 가 선언되지 않았습니다. "
            f"변수={sorted(self._vars)} 파라미터={sorted(self._pars)}"
        )

    def port(self, name: str) -> PortView:
        try:
            return self._ports[name]
        except KeyError:
            raise KeyError(
                f"{self._comp.name}: 포트 {name!r} 가 없습니다. 가능한 포트: {sorted(self._ports)}"
            ) from None

    def der(self, name: str) -> S.Sym:
        """상태변수의 시간미분 심볼. 과도 해석에서 적분기가 이산화한다."""
        v = self._vars[name]
        return self._comp._der_sym(name, v)

    @property
    def ports(self) -> dict[str, PortView]:
        return dict(self._ports)


class Component:
    """모든 물리 컴포넌트의 기반 클래스.

    하위 클래스는 ``PARAMS``/``VARS``/``PORTS`` 를 선언하고 ``equations`` 를 구현한다.
    잔차 형식으로 쓴다: 반환한 식이 모두 0 이 되도록 푼다.
    """

    PARAMS: dict[str, ParamSpec] = {}
    VARS: dict[str, VarSpec] = {}
    PORTS: dict[str, PortSpec] = {}

    def __init__(self, name: str, **params: Any):
        self.name = name
        self._param_values: dict[str, float] = {}
        specs = self.param_specs()
        unknown = set(params) - set(specs)
        if unknown:
            raise KeyError(
                f"{type(self).__name__}({name}): 알 수 없는 파라미터 {sorted(unknown)}. "
                f"가능: {sorted(specs)}"
            )
        for k, spec in specs.items():
            self._param_values[k] = spec.si(params.get(k))
        self._der_syms: dict[str, S.Sym] = {}
        self._states: set[str] = set()

    # --- 하위 클래스가 필요하면 덮어쓰는 훅 -------------------------------
    def param_specs(self) -> dict[str, ParamSpec]:
        return dict(self.PARAMS)

    def var_specs(self) -> dict[str, VarSpec]:
        return dict(self.VARS)

    def port_specs(self) -> dict[str, PortSpec]:
        return dict(self.PORTS)

    def equations(self, s: Scope) -> list[S.Expr]:
        raise NotImplementedError

    def outputs(self, s: Scope) -> dict[str, tuple[S.Expr, str]]:
        """리포트용 파생량. ``{이름: (식, 표시단위)}``."""
        return {}

    def equation_labels(self) -> list[str] | None:
        """방정식에 붙일 이름. ``None`` 이면 ``eq[0]``, ``eq[1]`` ... 로 번호를 쓴다.

        YAML 선언형 컴포넌트는 방정식마다 이름이 있으므로, 수렴 실패나 구조 오류
        메시지에 ``SCR.mass_balance`` 처럼 뜬다. 번호보다 훨씬 빨리 원인을 찾는다.
        """
        return None

    def initial_guess(self, inlets: dict[str, dict[str, float]]) -> dict[str, Any]:
        """초기값 전파 훅.

        ``inlets`` 는 상류에서 넘어온 입구 포트 상태 ``{포트명: {변수: SI값}}``.
        반환 형식은 ``{"ports": {포트명: {변수: 값}}, "vars": {내부변수: 값}}``.

        물리 모델은 초기값이 나쁘면 뉴턴법이 발산한다. 각 컴포넌트가 자기 출구
        상태를 대충이라도 계산해 하류로 넘겨주면, 사용자가 손으로 초기값을
        찍지 않아도 계통 전체가 합리적인 출발점에서 시작한다.
        """
        return {}

    # --- 내부 ------------------------------------------------------------
    def _der_sym(self, vname: str, v: S.Sym) -> S.Sym:
        if vname not in self._der_syms:
            spec = self.var_specs()[vname]
            d = S.sym(f"der({self.name}.{vname})", tuple(x - y for x, y in zip(spec.dim, (0, 0, 1, 0, 0, 0, 0))), kind="der")
            self._der_syms[vname] = d
            self._states.add(vname)
        return self._der_syms[vname]

    def build_scope(self) -> Scope:
        sc = Scope(self)
        for vname, spec in self.var_specs().items():
            sc._vars[vname] = S.sym(f"{self.name}.{vname}", spec.dim, kind="var",
                                    meta={"unit": spec.unit, "desc": spec.desc})
        for pname, spec in self.param_specs().items():
            sc._pars[pname] = S.sym(f"{self.name}.{pname}", spec.dim, kind="par",
                                    meta={"unit": spec.unit, "desc": spec.desc})
        for pname, pspec in self.port_specs().items():
            syms: dict[str, S.Sym] = {}
            for vname, unit in pspec.all_vars():
                syms[vname] = S.sym(f"{self.name}.{pname}.{vname}", dim_of(unit), kind="var",
                                    meta={"unit": unit, "port": pname})
            sc._ports[pname] = PortView(self.name, pname, pspec, syms)
        return sc

    def param_value(self, key: str) -> float:
        return self._param_values[key]

    def set_param(self, key: str, value_si: float) -> None:
        if key not in self._param_values:
            raise KeyError(f"{self.name}: 파라미터 {key!r} 없음")
        self._param_values[key] = float(value_si)

    def __repr__(self) -> str:  # noqa: D105
        return f"{type(self).__name__}({self.name!r})"


@dataclass
class Connection:
    a: str  # "COMP.PORT"
    b: str


def port_ref(comp: str, port: str) -> str:
    return f"{comp}.{port}"
