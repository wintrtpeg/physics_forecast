"""모델 로더: 파이썬 빌더 또는 YAML 선언에서 ``System`` 을 만든다.

파이썬 쪽이 표현력이 높고(반복문, 계산), YAML 쪽은 사람이 읽고 고치기 쉽고
에이전트가 생성하기도 쉽다. 둘 다 지원한다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml

from ..core.system import ModelError, System


def load_python_model(path: str | Path, builder: str = "build", **kwargs) -> System:
    """``model.py`` 의 ``build()`` 를 호출해 시스템을 만든다."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {path}")
    spec = importlib.util.spec_from_file_location(f"_pf_model_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ModelError(f"모델 모듈을 불러올 수 없습니다: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        if sys.path and sys.path[0] == str(path.parent):
            sys.path.pop(0)
    fn = getattr(mod, builder, None)
    if fn is None:
        raise ModelError(f"{path} 에 {builder}() 가 없습니다")
    system = fn(**kwargs)
    system.source_module = mod  # type: ignore[attr-defined]
    return system


def load_yaml_model(path: str | Path) -> System:
    """YAML 선언에서 시스템을 만든다.

    형식::

        name: nox_stack
        components:
          SRC_DRY:  {type: ToolGroupSource, n_tools: 24, util: 0.8, q_tool: 220}
          DCT_DRY:  {type: Duct, K: 120}
          HDR:      {type: Mixer, n_inlets: 4}
        connections:
          - [SRC_DRY.outlet, DCT_DRY.a]
          - [DCT_DRY.b, HDR.in1]

    파라미터 값은 컴포넌트가 선언한 단위로 해석된다.
    """
    from ..lib import REGISTRY

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    system = System(data.get("name", Path(path).stem))
    comps: dict[str, Any] = data.get("components") or {}
    for cname, cfg in comps.items():
        cfg = dict(cfg or {})
        ctype = cfg.pop("type", None)
        if ctype is None:
            raise ModelError(f"컴포넌트 {cname} 에 type 이 없습니다")
        cls = REGISTRY.get(ctype)
        if cls is None:
            raise ModelError(f"알 수 없는 컴포넌트 타입 {ctype!r}. 가능: {sorted(REGISTRY)}")
        system.add(cls(cname, **cfg))
    for conn in data.get("connections") or []:
        if isinstance(conn, (list, tuple)) and len(conn) == 2:
            system.connect(conn[0], conn[1])
        elif isinstance(conn, dict):
            system.connect(conn["from"], conn["to"])
        else:
            raise ModelError(f"연결 선언 형식이 잘못되었습니다: {conn!r}")
    return system


def load_model(spec: dict | str | Path, **kwargs) -> System:
    """시나리오 파일의 ``model:`` 항목을 해석한다."""
    if isinstance(spec, (str, Path)):
        p = Path(spec)
        return load_yaml_model(p) if p.suffix in (".yaml", ".yml") else load_python_model(p, **kwargs)
    if "yaml" in spec:
        return load_yaml_model(spec["yaml"])
    if "python" in spec:
        return load_python_model(spec["python"], spec.get("builder", "build"),
                                 **(spec.get("args") or {}))
    raise ModelError("model 항목에는 python: 또는 yaml: 이 필요합니다")
