"""프로젝트 파일 — 한 번 맞춰 둔 설정(데이터·변수·기간·모델·연결·계획)을 이름 붙여 저장한다.

계통이 바뀌어도 코드가 바뀌지 않게, 프로젝트는 **화면이 서버에 보내는 설정 그대로**를
적는다 (``easy.config_from_body`` 가 읽는 형식). 그래서 앱·CLI 가 같은 파일을 읽는다.

    projects/<이름>.yaml          설정 (사람이 읽고 고칠 수 있음)
    projects/<이름>.result.json   마지막 결과 (다시 열 때 바로 보여주려고)

``projects/`` 는 태그 이름·설계값이 들어가므로 커밋하지 않는다 (.gitignore).
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import yaml

FORMAT = 1
DIR = "projects"


def safe_name(name: str) -> str:
    from .fileio import safe_stem
    return safe_stem(re.sub(r"[^0-9A-Za-z가-힣_\-]+", "_", str(name)).strip("_") or "project", "project")


def project_path(root: str | Path, name: str) -> Path:
    return Path(root) / DIR / f"{safe_name(name)}.yaml"


def save_project(root: str | Path, name: str, run: dict, ui: dict | None = None,
                 forecast: dict | None = None, results: dict | None = None,
                 forecast_results: dict | None = None) -> Path:
    """설정을 저장한다. ``results`` 를 주면 옆에 결과 JSON 도 쓴다 (주지 않으면 기존 결과를 둔다)."""
    path = project_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"pforecast_project": FORMAT, "name": str(name),
           "saved_at": datetime.now().isoformat(timespec="seconds"),
           "run": run or {}, "ui": ui or {}}
    if forecast:
        doc["forecast"] = forecast
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    res = _result_path(path)
    old = _read_json(res)
    if results is not None or forecast_results is not None:
        new = {"results": results if results is not None else old.get("results"),
               "forecast": forecast_results if forecast_results is not None else old.get("forecast")}
        res.write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")
    return path


def _result_path(path: Path) -> Path:
    return path.with_name(path.stem + ".result.json")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_project(path: str | Path) -> dict:
    p = Path(path)
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict) or "pforecast_project" not in doc:
        raise ValueError(f"프로젝트 파일이 아닙니다: {p.name}")
    if int(doc["pforecast_project"]) > FORMAT:
        raise ValueError(f"더 새 버전의 프로젝트 파일입니다 (형식 {doc['pforecast_project']}). 프로그램을 업데이트하세요.")
    saved = _read_json(_result_path(p))
    doc["results"] = saved.get("results")
    doc["forecast_results"] = saved.get("forecast")
    return doc


def list_projects(root: str | Path) -> list[dict]:
    """최근 저장 순. 깨진 파일은 건너뛰지 않고 오류와 함께 보여준다."""
    folder = Path(root) / DIR
    out = []
    for p in sorted(folder.glob("*.yaml")) if folder.is_dir() else []:
        try:
            doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            run = doc.get("run") or {}
            out.append({"name": doc.get("name") or p.stem, "path": p, "saved_at": doc.get("saved_at", ""),
                        "csv": run.get("csv"), "target": run.get("target"), "model": run.get("model"),
                        "step": (doc.get("ui") or {}).get("step"),
                        "has_results": _result_path(p).exists()})
        except Exception as exc:  # noqa: BLE001 — 목록은 다 보여주고 문제만 표시한다
            out.append({"name": p.stem, "path": p, "saved_at": "", "error": str(exc)})
    out.sort(key=lambda d: d.get("saved_at") or "", reverse=True)
    return out
