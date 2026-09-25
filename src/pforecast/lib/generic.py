"""YAML 선언형 컴포넌트.

파이썬을 건드리지 않고 지배방정식을 그대로 적어 컴포넌트를 만든다. 손으로 짠
컴포넌트와 **완전히 같은 식 그래프**가 나오므로 해석적 미분·차원 검사·구조 해석이
모두 그대로 적용된다.

    type: equation
    ports:
      a: {kind: gas, role: in}
      b: {kind: gas, role: out}
    params:
      K: {value: 30, unit: "1/m4", tunable: true}
    vars:
      dp: {unit: Pa, start: 100}
      rho: {unit: kg/m3, start: 1.15, lo: 0.05, hi: 10}
    equations:
      mass:     "a.mdot + b.mdot = 0"
      state:    "rho = density(a)"
      friction: "dp = K * signed_pow(a.mdot, 2) / rho"
      momentum: "a.p - b.p = dp"

``extends`` 로 다른 정의를 상속받아 **방정식 하나만 갈아끼울 수** 있다. 구성방정식
후보를 비교할 때 쓰는 핵심 기능이다 (``pf select`` 참고).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml

from ..core import symbolic as S
from ..core.component import Component, ParamSpec, PortSpec, Scope, VarSpec
from ..core.parser import FunctionTable, parse_equation, parse_expression
from .flow import GasComponent
from .gas import (FLUE_GAS, GasMedium, density, enthalpy, humidity_from_rh,
                  mixture_cp, mixture_molar_mass, normal_volume_flow, p_sat_water)

# --- 포트 종류 레지스트리 --------------------------------------------------

def _gas_port(role: str, medium: GasMedium, **kw) -> PortSpec:
    from .gas import gas_port
    return gas_port(medium, role, **kw)


def _thermal_port(role: str, medium: GasMedium, **kw) -> PortSpec:
    """열 포트: 온도(across) / 열류(through)."""
    return PortSpec(
        across=(("T", "K"),), through=(("Q", "W"),), role=role, kind="thermal",
        starts={"T": kw.get("T_start", 300.0), "Q": kw.get("Q_start", 0.0)},
        bounds={"T": (1.0, 5000.0)},
    )


def _liquid_port(role: str, medium: GasMedium, **kw) -> PortSpec:
    """단일 성분 액체 포트: 압력/유량/온도."""
    return PortSpec(
        across=(("p", "Pa"),), through=(("mdot", "kg/s"),), stream=(("T", "K"),),
        role=role, kind="liquid",
        starts={"p": kw.get("p_start", 3.0e5), "mdot": kw.get("mdot_start", 10.0),
                "T": kw.get("T_start", 293.15)},
        bounds={"T": (250.0, 500.0), "p": (1.0e3, 5.0e7)},
    )


PORT_KINDS: dict[str, Callable[..., PortSpec]] = {
    "gas": _gas_port,
    "thermal": _thermal_port,
    "liquid": _liquid_port,
}

MEDIA: dict[str, GasMedium] = {"flue_gas": FLUE_GAS}


# --- 도메인 함수 (파서에 얹는다) --------------------------------------------

def _require_port(x, fn: str):
    if not hasattr(x, "spec"):
        raise TypeError(f"{fn}() 의 인자는 포트 이름이어야 합니다 (예: {fn}(a))")
    return x


def domain_functions(medium: GasMedium) -> dict[str, Callable[..., Any]]:
    """가스 매질에 묶인 물성 함수들."""
    return {
        "density": lambda port: density(_require_port(port, "density"), medium),
        "enthalpy": lambda port: enthalpy(_require_port(port, "enthalpy"), medium),
        "cp": lambda port: mixture_cp(_require_port(port, "cp"), medium),
        "molar_mass": lambda port: mixture_molar_mass(_require_port(port, "molar_mass"), medium),
        "Qn": lambda mdot, port: normal_volume_flow(mdot, _require_port(port, "Qn"), medium),
        "p_sat": p_sat_water,
        "humidity": lambda rh, T, p: humidity_from_rh(rh, T, p, medium),
    }


# --- 사양 로딩 / 상속 -------------------------------------------------------

def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_component_spec(spec: str | Path | dict, base_dir: str | Path | None = None) -> dict:
    """``extends`` 를 따라가며 컴포넌트 사양을 합친다.

    자식이 같은 키를 다시 쓰면 덮어쓴다. 방정식이 dict 이므로 **이름으로 하나만
    교체**할 수 있다. 구성방정식 후보 비교의 토대다.
    """
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        if base_dir and not path.is_absolute():
            path = Path(base_dir) / path
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return load_component_spec(data, path.parent)
    data = dict(spec)
    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        return data
    parent = load_component_spec(parent_ref, base_dir)
    merged = _deep_merge(parent, data)
    merged.setdefault("_lineage", [])
    merged["_lineage"] = list(parent.get("_lineage", [])) + [str(parent_ref)]
    return merged


class EquationComponent(GasComponent):
    """선언된 방정식으로 동작하는 범용 컴포넌트."""

    def __init__(self, name: str, spec: str | Path | dict | None = None,
                 medium: GasMedium | None = None, base_dir: str | Path | None = None,
                 **overrides):
        data = load_component_spec(spec, base_dir) if spec is not None else {}
        data = _deep_merge(data, {k: v for k, v in overrides.items() if k != "type"})
        data.pop("type", None)
        self._lineage = data.pop("_lineage", [])
        self.description = data.pop("description", "")

        med = medium or MEDIA.get(str(data.pop("medium", "flue_gas")), FLUE_GAS)

        self._port_specs: dict[str, PortSpec] = {}
        for pname, pcfg in (data.get("ports") or {}).items():
            pcfg = dict(pcfg or {})
            kind = pcfg.pop("kind", "gas")
            role = pcfg.pop("role", "in")
            if kind not in PORT_KINDS:
                raise ValueError(
                    f"{name}: 알 수 없는 포트 종류 {kind!r}. 가능: {sorted(PORT_KINDS)}")
            self._port_specs[pname] = PORT_KINDS[kind](role, med, **pcfg)

        self._var_specs: dict[str, VarSpec] = {}
        for vname, vcfg in (data.get("vars") or {}).items():
            vcfg = dict(vcfg or {})
            self._var_specs[vname] = VarSpec(
                unit=str(vcfg.get("unit", "1")), start=float(vcfg.get("start", 1.0)),
                lo=float(vcfg.get("lo", -float("inf"))), hi=float(vcfg.get("hi", float("inf"))),
                desc=str(vcfg.get("desc", "")))

        self._param_specs: dict[str, ParamSpec] = {}
        for pname, pcfg in (data.get("params") or {}).items():
            if not isinstance(pcfg, dict):
                pcfg = {"value": pcfg}
            self._param_specs[pname] = ParamSpec(
                default=float(pcfg.get("value", 0.0)), unit=str(pcfg.get("unit", "1")),
                desc=str(pcfg.get("desc", "")), lo=float(pcfg.get("lo", -float("inf"))),
                hi=float(pcfg.get("hi", float("inf"))), tunable=bool(pcfg.get("tunable", False)))

        eqs = data.get("equations") or {}
        if isinstance(eqs, list):     # 이름 없이 리스트로 줘도 받아준다
            eqs = {f"eq{i}": e for i, e in enumerate(eqs)}
        self._equation_src: dict[str, str] = {str(k): str(v) for k, v in eqs.items()}
        self._output_src: dict[str, tuple[str, str]] = {}
        for oname, ocfg in (data.get("outputs") or {}).items():
            if isinstance(ocfg, str):
                ocfg = {"expr": ocfg}
            self._output_src[str(oname)] = (str(ocfg["expr"]), str(ocfg.get("unit", "1")))

        # 파라미터 값 오버라이드는 Component.__init__ 이 받는다
        param_values = {k: v for k, v in (data.get("values") or {}).items()}
        super().__init__(name, medium=med, **param_values)
        self._functions = FunctionTable().extend(domain_functions(med))

    # --- Component 인터페이스 ---
    def param_specs(self) -> dict[str, ParamSpec]:
        return dict(self._param_specs)

    def var_specs(self) -> dict[str, VarSpec]:
        return dict(self._var_specs)

    def port_specs(self) -> dict[str, PortSpec]:
        return dict(self._port_specs)

    def equation_labels(self) -> list[str]:
        return list(self._equation_src)

    def _resolver(self, s: Scope):
        def resolve(name: str):
            if "." in name:
                pname, vname = name.split(".", 1)
                return getattr(s.port(pname), vname)
            if name in s._ports:
                return s._ports[name]          # 함수 인자용 포트 객체
            return getattr(s, name)
        return resolve

    def equations(self, s: Scope) -> list[S.Expr]:
        resolve = self._resolver(s)
        out = []
        for label, text in self._equation_src.items():
            out.append(parse_equation(text, resolve, resolve_der=s.der,
                                      functions=self._functions,
                                      context=f"{self.name}.{label}"))
        return out

    def outputs(self, s: Scope) -> dict[str, tuple[S.Expr, str]]:
        resolve = self._resolver(s)
        return {
            name: (parse_expression(text, resolve, self._functions,
                                    context=f"{self.name}.outputs.{name}"), unit)
            for name, (text, unit) in self._output_src.items()
        }

    def __repr__(self) -> str:  # noqa: D105
        return f"EquationComponent({self.name!r}, {len(self._equation_src)} eqs)"
