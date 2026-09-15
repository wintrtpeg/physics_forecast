"""보정된 파라미터의 저장/적용.

보정 결과는 코드가 아니라 **데이터**다. 모델 코드는 설계값을 들고 있고, 현장에
맞춘 값은 YAML 로 따로 둔다. 라인별/호기별로 파일만 갈아끼우면 같은 모델이
여러 계통에 붙는다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml

from .core.units import from_si, to_si


def save_params(model, path: str | Path, names: Iterable[str] | None = None,
                meta: dict | None = None) -> Path:
    """파라미터를 사람이 읽는 단위로 저장한다."""
    names = list(names) if names is not None else [p.name for p in model.tunable_params()]
    entries = {}
    for n in names:
        i = model.par_index(n)
        info = model.parameters[i]
        entries[n] = {"value": round(float(from_si(info.value, info.unit)), 10),
                      "unit": info.unit}
        if info.desc:
            entries[n]["desc"] = info.desc
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "model": model.system.name,
        **({"meta": meta} if meta else {}),
        "parameters": entries,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def load_params(path: str | Path) -> dict[str, tuple[float, str]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out = {}
    for name, spec in (data.get("parameters") or {}).items():
        if isinstance(spec, dict):
            out[name] = (float(spec["value"]), spec.get("unit", "1"))
        else:
            out[name] = (float(spec), "1")
    return out


def apply_params(model, path: str | Path, strict: bool = True) -> list[str]:
    """저장된 파라미터를 컴파일된 모델과 원본 시스템 양쪽에 반영한다."""
    applied = []
    for name, (value, unit) in load_params(path).items():
        try:
            i = model.par_index(name)
        except KeyError:
            if strict:
                raise
            print(f"[경고] 파라미터 {name!r} 가 모델에 없어 건너뜁니다")
            continue
        si = to_si(value, unit)
        model.parameters[i].value = si
        cname, pname = name.rsplit(".", 1)
        if cname in model.system.components:
            model.system.components[cname].set_param(pname, si)
        applied.append(name)
    return applied
