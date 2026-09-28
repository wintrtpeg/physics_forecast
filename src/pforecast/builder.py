"""모델 만들기 — 코드 없이 계통을 조립하고 바로 검사한다.

화면이 다루는 것은 ``system.yaml`` 과 **같은 모양의 dict** 다. 파일로 저장하면 CLI·간편 예측·
시나리오가 그대로 읽는다. 파이썬 모델(``model.py``)은 여기서 열지 않는다 — 반복문·계산이 필요한
계통은 파이썬으로, 나머지는 여기서.

검사는 싼 것부터 한다: 불러오기(타입·파라미터 이름) → 연결(포트 종류·방향·미연결) → 구조(방정식
= 미지수, 과결정·부족결정) → 설계점 풀이 → 선언(운전 손잡이·계측값·보정 대상·관리기준) 확인.
어느 단계에서 막혔는지와 어느 컴포넌트·포트 탓인지를 사람이 읽는 말로 돌려준다.

토폴로지(무엇이 무엇에 연결되는가)는 **사람이 정한다.** 데이터에서 유도하지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .core.units import from_si

#: 파이썬 컴포넌트의 팔레트 분류
_CATEGORY = {"ToolGroupSource": "배기·가스 처리", "WetScrubber": "배기·가스 처리", "Stack": "배기·가스 처리",
             "Duct": "가스 유동", "Fan": "가스 유동", "Mixer": "가스 유동"}
#: 생성자 옵션 (파라미터가 아닌 것)
_OPTIONS = {"Mixer": {"n_inlets": 2}}

BLANK_EQUATION = {
    "type": "equation",
    "description": "사용자 정의 컴포넌트. 포트·파라미터·미지수·방정식을 직접 적는다.",
    "ports": {"a": {"kind": "thermal", "role": "in"}, "b": {"kind": "thermal", "role": "out"}},
    "params": {"UA": {"value": 1000.0, "unit": "W/K", "desc": "열전달 계수", "tunable": True}},
    "vars": {},
    "equations": {"heat": "a.Q + b.Q = 0", "temperature": "a.T - b.T = a.Q / UA"},
}


def _component_info(comp, ctype: str, ref: str | None = None) -> dict:
    from .lib.closures import ROLE_LABEL
    from .lib.explain import laws_of, title_of

    doc = (type(comp).__doc__ or "").strip().split("\n\n")[0].replace("\n", " ")
    params = []
    for name, spec in comp.param_specs().items():
        try:
            value = from_si(comp.param_value(name), spec.unit)
        except KeyError:
            value = spec.default
        params.append({"name": name, "value": float(value), "default": float(spec.default), "unit": spec.unit,
                       "desc": spec.desc, "tunable": bool(spec.tunable),
                       "lo": spec.lo if spec.lo != -float("inf") else None,
                       "hi": spec.hi if spec.hi != float("inf") else None})
    ports = [{"name": n, "kind": ps.kind, "role": ps.role} for n, ps in comp.port_specs().items()]
    closures = []
    for sl in getattr(type(comp), "CLOSURES", ()):
        closures.append({"key": sl.key, "title": sl.title, "current": comp.closure_choices[sl.key],
                         "default": sl.default,
                         "options": [{"id": o.id, "title": o.title, "role_label": ROLE_LABEL.get(o.role, ""),
                                      "formula": o.law.formula} for o in sl.options]})
    return {"type": ctype, "ref": ref, "title": title_of(comp),
            "category": getattr(comp, "category", "") or _CATEGORY.get(ctype, "기타"),
            "description": getattr(comp, "description", "") or doc,
            "params": params, "ports": ports, "closures": closures,
            "options": dict(_OPTIONS.get(ctype, {})),
            "laws": [{"kind": law.kind, "title": law.title, "formula": law.formula} for law in laws_of(comp)]}


def palette(root: str | Path | None = None) -> list[dict]:
    """고를 수 있는 컴포넌트: 파이썬 라이브러리, 패키지 YAML 라이브러리, 작업 폴더 ``components/``."""
    from .lib import REGISTRY, EquationComponent
    from .lib.generic import library_components

    out = []
    for name, cls in REGISTRY.items():
        if cls is EquationComponent:
            continue
        comp = cls("X", **_OPTIONS.get(name, {}))
        out.append(_component_info(comp, name))
    for lc in library_components():
        comp = EquationComponent("X", lc["ref"])
        out.append(_component_info(comp, "equation", lc["ref"]))
    if root is not None:
        folder = Path(root) / "components"
        for p in sorted(folder.glob("*.yaml")) if folder.is_dir() else []:
            try:
                comp = EquationComponent("X", str(p))
            except Exception:  # noqa: BLE001 — 깨진 파일은 팔레트에서 뺀다
                continue
            info = _component_info(comp, "equation", f"components/{p.name}")
            info["category"] = info["category"] or "작업 폴더"
            out.append(info)
    blank = EquationComponent("X", None, **{k: v for k, v in BLANK_EQUATION.items() if k != "type"})
    info = _component_info(blank, "equation")
    info.update({"title": "사용자 정의 (식 직접 입력)", "category": "사용자 정의", "blank": True,
                 "template": BLANK_EQUATION})
    out.append(info)
    return out


def _dangling(system) -> list[str]:
    used = {p for c in system.connections for p in c}
    return [f"{n}.{p}" for n, c in system.components.items() for p in c.port_specs()
            if f"{n}.{p}" not in used]


def check_model(data: dict, base_dir: str | Path | None = None, solve: bool = True) -> dict:
    """모델 dict 를 단계별로 검사한다. 화면이 입력할 때마다 부른다 (보통 1초 안)."""
    from .core.solvers import solve_steady
    from .core.system import ModelError
    from .easy import _segments
    from .scenario.loader import system_from_dict

    res: dict = {"ok": False, "stage": "load", "errors": [], "warnings": [], "components": [],
                 "dangling": [], "counts": {}, "solve": None, "segments": [], "declarations": {}}
    if not (data.get("components") or {}):
        res["errors"].append("컴포넌트가 없습니다. 왼쪽 목록에서 추가하세요.")
        return res
    try:
        system = system_from_dict(data, base_dir=base_dir)
    except (ModelError, KeyError, ValueError, TypeError, FileNotFoundError) as exc:
        res["errors"].append(str(exc))
        return res

    for name, comp in system.components.items():
        cfg = (data.get("components") or {}).get(name) or {}
        info = _component_info(comp, cfg.get("type", type(comp).__name__), cfg.get("spec"))
        info["name"] = name
        res["components"].append(info)
    res["segments"] = _segments(system)

    # 연결: 포트 이름·종류·방향을 먼저 사람이 읽는 말로
    res["stage"] = "connect"
    ports = {f"{n}.{p}": ps for n, c in system.components.items() for p, ps in c.port_specs().items()}
    seen: dict[str, str] = {}
    for a, b in system.connections:
        for ref in (a, b):
            if ref not in ports:
                res["errors"].append(f"없는 포트입니다: {ref}")
            elif ref in seen:
                res["errors"].append(f"포트 {ref} 가 두 번 연결되었습니다 ({seen[ref]} 와도). 합류·분기는 "
                                     "헤더(Mixer) 같은 컴포넌트로 하세요.")
        if a in ports and b in ports:
            if ports[a].kind != ports[b].kind:
                res["errors"].append(f"포트 종류가 다릅니다: {a} ({ports[a].kind}) ↔ {b} ({ports[b].kind})")
            elif ports[a].role == ports[b].role:
                res["errors"].append(f"둘 다 {'출구' if ports[a].role == 'out' else '입구'} 포트입니다: {a} ↔ {b}. "
                                     "출구 → 입구로 연결하세요.")
        seen[a], seen[b] = b, a
    res["dangling"] = _dangling(system)
    if res["dangling"]:
        res["errors"].append("연결되지 않은 포트: " + ", ".join(res["dangling"])
                             + " — 모든 포트를 연결하거나, 계통 끝은 경계 컴포넌트(배기원·굴뚝·외기 등)로 닫으세요.")
    if res["errors"]:
        return res

    res["stage"] = "structure"
    try:
        model = system.compile()
    except (ModelError, KeyError, ValueError, TypeError) as exc:
        res["errors"].append(str(exc))
        return res
    r = model.report
    res["counts"] = {"components": len(system.components), "connections": len(system.connections),
                     "equations": r.n_eqs, "unknowns": r.n_vars, "params": len(model.parameters),
                     "blocks": len(r.blocks), "largest_block": r.largest_block}
    if not r.ok:
        if r.n_eqs != r.n_vars:
            res["errors"].append(f"방정식 {r.n_eqs}개, 미지수 {r.n_vars}개 — 개수가 같아야 풀 수 있습니다.")
        if r.underdetermined_vars:
            names = [model.variables[i].name for i in r.underdetermined_vars[:10]]
            res["errors"].append("이 미지수를 정할 식이 부족합니다: " + ", ".join(names)
                                 + (" …" if len(r.underdetermined_vars) > 10 else ""))
        if r.overdetermined_eqs:
            names = [model.equations[i].label for i in r.overdetermined_eqs[:10]]
            res["errors"].append("이 방정식들이 중복되거나 서로 모순입니다: " + ", ".join(names)
                                 + (" …" if len(r.overdetermined_eqs) > 10 else ""))
        return res

    res["declarations"] = _check_declarations(data, model)
    res["warnings"] += res["declarations"].get("problems", [])
    if not solve:
        res["ok"] = True
        return res
    res["stage"] = "solve"
    model.build()
    sol = solve_steady(model, model.p0(), model.x0())
    out = {}
    if sol.success:
        vals = model.output_values(sol.x, model.p0())
        for k in sorted(vals):
            comp, name = k.split(".", 1)
            out.setdefault(comp, []).append({"name": name, "value": float(vals[k]), "unit": model.outputs[k][1]})
    res["solve"] = {"success": bool(sol.success), "message": sol.message, "outputs": out}
    if not sol.success:
        res["errors"].append("설계값에서 풀리지 않습니다. 파라미터 값(단위 포함)과 경계 컴포넌트를 확인하세요.\n"
                             + sol.message)
        return res
    res["ok"] = True
    return res


def _check_declarations(data: dict, model) -> dict:
    """운전 손잡이·계측값·보정 대상·관리기준이 모델에 실제로 있는지."""
    from .calib.runner import resolve_targets

    problems = []
    shorts: dict[str, list[str]] = {}
    for p in model.parameters:
        shorts.setdefault(p.name.rsplit(".", 1)[-1], []).append(p.name)
    for d in data.get("drivers") or []:
        if d.get("key") not in shorts:
            problems.append(f"운전 손잡이 {d.get('key')!r} 에 해당하는 파라미터가 없습니다.")
    for name in (data.get("observables") or {}):
        try:
            resolve_targets(model, [name])
        except (KeyError, ValueError, IndexError):
            problems.append(f"계측값 {name!r} 는 모델의 변수·출력이 아닙니다.")
    pnames = {p.name: p for p in model.parameters}
    for name in data.get("calibrate") or []:
        if name not in pnames:
            problems.append(f"보정 대상 {name!r} 파라미터가 없습니다.")
        elif not pnames[name].tunable:
            problems.append(f"보정 대상 {name!r} 는 보정 가능한 파라미터(tunable)가 아닙니다.")
    for name in (data.get("limits") or {}):
        if name not in model.outputs:
            problems.append(f"관리기준 {name!r} 는 모델 출력이 아닙니다.")
    # 화면이 고를 수 있게: 손잡이 후보(보정 대상 아님), 보정 후보, 출력
    return {"problems": problems,
            "driver_keys": sorted(k for k, v in shorts.items()
                                  if not all(model.parameters[model.par_index(n)].tunable for n in v)),
            "tunable": sorted(n for n, p in pnames.items() if p.tunable),
            "outputs": sorted(model.outputs)}


def load_model_dict(path: str | Path) -> dict:
    p = Path(path)
    if p.suffix not in (".yaml", ".yml"):
        raise ValueError("모델 만들기는 YAML 모델만 엽니다. 파이썬 모델(model.py)은 코드에서 고치세요.")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def dump_model(data: dict) -> str:
    """사람이 읽기 좋은 순서로 YAML 을 만든다."""
    order = ["name", "description", "drivers", "limits", "observables", "calibrate", "components", "connections"]
    d = {k: data[k] for k in order if data.get(k) not in (None, [], {}, "")}
    d.update({k: v for k, v in data.items() if k not in d and k not in order and v not in (None, [], {}, "")})
    return yaml.safe_dump(d, allow_unicode=True, sort_keys=False, width=110)


def save_model(data: dict, path: str | Path) -> Path:
    p = Path(path)
    if p.suffix not in (".yaml", ".yml"):
        p = p.with_suffix(".yaml")
    p.parent.mkdir(parents=True, exist_ok=True)
    head = ("# pforecast 모델 — 앱의 '모델 만들기'에서 저장했습니다. 직접 고쳐도 됩니다.\n"
            "#   pf check " + p.name + "   # 구조 검사 + 설계점 풀이\n\n")
    p.write_text(head + dump_model(data), encoding="utf-8")
    return p
