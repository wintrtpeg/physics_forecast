"""시나리오 정의와 실행.

what-if 질문을 YAML 한 장으로 적고, 케이스별/스윕별로 정상상태를 풀어
결과 표를 만든다. 관리기준(limits)을 함께 선언하면 초과 여부까지 판정한다.

파라미터 이름에는 glob 패턴을 쓸 수 있다. ``SRC_*.util: 0.95`` 한 줄이면
모든 장비군의 가동율이 한꺼번에 바뀐다.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..core.solvers import solve_steady
from ..core.units import from_si, to_si


def expand_pattern(model, pattern: str) -> list[str]:
    """``SRC_*.util`` 같은 패턴을 실제 파라미터 이름 목록으로 편다."""
    if any(ch in pattern for ch in "*?["):
        hits = [p.name for p in model.parameters if fnmatch.fnmatch(p.name, pattern)]
        if not hits:
            raise KeyError(f"패턴 {pattern!r} 에 맞는 파라미터가 없습니다")
        return hits
    model.par_index(pattern)   # 없으면 KeyError
    return [pattern]


def apply_settings(model, p: np.ndarray, settings: dict[str, Any] | None,
                   scales: dict[str, Any] | None = None) -> np.ndarray:
    """``set`` 과 ``scale`` 지시를 파라미터 배열에 적용한다."""
    p = p.copy()
    for pattern, spec in (settings or {}).items():
        if isinstance(spec, dict):
            value, unit = spec.get("value"), spec.get("unit", None)
        else:
            value, unit = spec, None
        for name in expand_pattern(model, pattern):
            i = model.par_index(name)
            u = unit if unit is not None else model.parameters[i].unit
            p[i] = to_si(float(value), u)
    for pattern, factor in (scales or {}).items():
        for name in expand_pattern(model, pattern):
            i = model.par_index(name)
            p[i] = p[i] * float(factor)
    return p


@dataclass
class Case:
    name: str
    set: dict[str, Any] = field(default_factory=dict)
    scale: dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass
class Sweep:
    name: str
    over: dict[str, list]                     # {패턴: [값들]}
    set: dict[str, Any] = field(default_factory=dict)
    scale: dict[str, Any] = field(default_factory=dict)
    unit: str | None = None


@dataclass
class ScenarioSpec:
    name: str = "scenario"
    model: Any = None
    base_set: dict[str, Any] = field(default_factory=dict)
    base_scale: dict[str, Any] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    limits: dict[str, Any] = field(default_factory=dict)
    cases: list[Case] = field(default_factory=list)
    sweeps: list[Sweep] = field(default_factory=list)
    params_file: str | None = None            # 보정된 파라미터 YAML
    description: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioSpec":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        base = d.get("base") or {}
        return cls(
            name=d.get("name", Path(path).stem),
            model=d.get("model"),
            base_set=base.get("set") or {},
            base_scale=base.get("scale") or {},
            outputs=d.get("outputs") or [],
            limits=d.get("limits") or {},
            cases=[Case(name=c.get("name", f"case{i}"), set=c.get("set") or {},
                        scale=c.get("scale") or {}, note=c.get("note", ""))
                   for i, c in enumerate(d.get("cases") or [])],
            sweeps=[Sweep(name=s.get("name", f"sweep{i}"), over=s.get("over") or {},
                          set=s.get("set") or {}, scale=s.get("scale") or {},
                          unit=s.get("unit"))
                    for i, s in enumerate(d.get("sweeps") or [])],
            params_file=d.get("params"),
            description=d.get("description", ""),
        )


@dataclass
class ScenarioResult:
    cases: pd.DataFrame
    sweeps: dict[str, pd.DataFrame]
    violations: pd.DataFrame
    spec: ScenarioSpec


def _collect(model, x, p, outputs: list[str]) -> dict[str, float]:
    ov = model.output_values(x, p)
    row: dict[str, float] = {}
    for name in outputs:
        if name in ov:
            row[name] = ov[name]
            continue
        try:
            vi = model.var_index(name)
        except KeyError:
            row[name] = float("nan")
            continue
        row[name] = from_si(float(x[vi]), model.variables[vi].unit)
    return row


def run_scenario(model, spec: ScenarioSpec, x0=None) -> ScenarioResult:
    model.build()
    outputs = spec.outputs or sorted(model.outputs)
    p_base = apply_settings(model, model.p0(), spec.base_set, spec.base_scale)

    r0 = solve_steady(model, p_base, x0=x0)
    warm = r0.x if r0.success else None

    rows = []
    cases = spec.cases or [Case(name="base")]
    for c in cases:
        p = apply_settings(model, p_base, c.set, c.scale)
        r = solve_steady(model, p, x0=warm)
        row = {"case": c.name, "converged": r.success}
        if r.success:
            warm = r.x
            row.update(_collect(model, r.x, p, outputs))
        else:
            row["message"] = r.message.splitlines()[0]
        if c.note:
            row["note"] = c.note
        rows.append(row)
    case_df = pd.DataFrame(rows)

    sweep_out: dict[str, pd.DataFrame] = {}
    for sw in spec.sweeps:
        if len(sw.over) != 1:
            raise ValueError(f"sweep '{sw.name}' 의 over 는 항목 1개여야 합니다")
        pattern, values = next(iter(sw.over.items()))
        p_start = apply_settings(model, p_base, sw.set, sw.scale)
        srows = []
        w = warm
        for v in values:
            p = apply_settings(model, p_start, {pattern: ({"value": v, "unit": sw.unit}
                                                          if sw.unit else v)})
            r = solve_steady(model, p, x0=w)
            row = {pattern: v, "converged": r.success}
            if r.success:
                w = r.x
                row.update(_collect(model, r.x, p, outputs))
            srows.append(row)
        sweep_out[sw.name] = pd.DataFrame(srows)

    # 관리기준 판정
    vio = []
    for target, lim in (spec.limits or {}).items():
        lim = lim if isinstance(lim, dict) else {"max": lim}
        for df_name, df in [("cases", case_df)] + list(sweep_out.items()):
            if target not in df.columns:
                continue
            for _, r in df.iterrows():
                v = r[target]
                if not np.isfinite(v):
                    continue
                if "max" in lim and v > lim["max"]:
                    vio.append({"where": df_name, "label": r.get("case", r.iloc[0]),
                                "target": target, "value": v, "limit": lim["max"],
                                "type": "상한 초과",
                                "margin_%": 100.0 * (v / lim["max"] - 1.0)})
                if "min" in lim and v < lim["min"]:
                    vio.append({"where": df_name, "label": r.get("case", r.iloc[0]),
                                "target": target, "value": v, "limit": lim["min"],
                                "type": "하한 미달",
                                "margin_%": 100.0 * (v / lim["min"] - 1.0)})
    return ScenarioResult(cases=case_df, sweeps=sweep_out,
                          violations=pd.DataFrame(vio), spec=spec)
