"""로컬 웹 앱 서버 — 표준 라이브러리만 쓴다.

왜 FastAPI/Streamlit 이 아닌가. 사내 잠긴 PC 에서 ``python -m pforecast.app`` 한 줄로
떠야 한다. 의존성이 하나라도 더 있으면 그게 막히는 순간 도구 전체가 멈춘다.
단일 사용자 로컬 도구에 웹 프레임워크는 과하다.

구조
----
* 짧은 작업(모델 풀이, 프로파일링)은 요청 안에서 바로 처리한다. 정상상태 1회가
  4ms 라 시나리오 슬라이더를 실시간으로 끌 수 있다.
* 긴 작업(분석, 보정)은 스레드로 돌리고 진행상황을 폴링한다.
* 컴파일된 모델은 캐시하고 직전 해를 warm start 로 재사용한다.
"""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

STATIC_DIR = Path(__file__).parent / "static"

#: 업로드 상한. 5분 데이터 1년치 × 수십 태그가 수십 MB 수준이다.
MAX_UPLOAD = 300 * 1024 * 1024


# --- 작업 큐 ----------------------------------------------------------------

@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"          # running | done | error
    message: str = ""
    result: Any = None
    error: str = ""
    started: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def payload(self) -> dict:
        return {"id": self.id, "kind": self.kind, "status": self.status,
                "message": self.message, "result": self.result, "error": self.error,
                "started": self.started}


class JobRegistry:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, fn: Callable[[Job], Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._jobs[job.id] = job

        def run():
            try:
                job.result = fn(job)
                job.status = "done"
                job.message = job.message or "완료"
            except Exception as exc:               # 사용자에게 원인을 그대로 보여준다
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = "실패"
                traceback.print_exc()

        threading.Thread(target=run, daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


# --- 워크스페이스 ------------------------------------------------------------

class Workspace:
    """CSV 와 모델 파일을 찾아 목록으로 만든다."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.models: dict[str, Any] = {}          # 경로 -> CompiledModel
        self._locks: dict[str, threading.Lock] = {}
        self._warm: dict[str, np.ndarray] = {}
        self._tables: dict[str, tuple] = {}       # 경로 -> (mtime, size, Table, QualityReport)
        self._tlock = threading.Lock()

    def table(self, rel: str, time_column: str | None = None):
        """CSV 를 읽어 정리한 결과를 캐시한다. 10MB 에 2~3초라 미리보기마다 다시 읽을 수 없다."""
        from ..data.ingest import read_table
        from ..data.quality import assess
        path = self.resolve(rel)
        st = path.stat()
        key = f"{rel}|{time_column or ''}"
        with self._tlock:
            hit = self._tables.get(key)
            if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
                return hit[2], hit[3]
        tab = read_table(path, time_column=time_column)
        q = assess(tab.df)
        with self._tlock:
            self._tables[key] = (st.st_mtime, st.st_size, tab, q)
        return tab, q

    def _rel(self, p: Path) -> str:
        try:
            return str(p.resolve().relative_to(self.root))
        except ValueError:
            return str(p.resolve())

    #: 작업 폴더를 훑을 때 들어가지 않는 폴더. 저장소 안에 만든 가상환경(.venv, venv)은
    #: 파일이 수만 개라 목록이 느려지고 남의 YAML 이 모델로 잡힌다.
    SKIP_DIRS = {".git", "__pycache__", "out", "node_modules", "venv", "env", "site-packages",
                 "build", "dist", "wheels", "offline"}

    def _walk(self, pattern: str):
        import os
        import fnmatch
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames
                                 if d not in self.SKIP_DIRS and not d.startswith((".", "_")))
            for f in sorted(filenames):
                if fnmatch.fnmatch(f, pattern):
                    yield Path(dirpath) / f

    #: 계획 CSV 는 학습 데이터가 아니다 — 데이터 목록에서 빼고 계획 목록에만 보인다
    PLAN_DIRS = {"plans"}

    def datasets(self) -> list[dict]:
        out = []
        for p in self._walk("*.csv"):
            if self.PLAN_DIRS & set(p.relative_to(self.root).parts[:-1]):
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            out.append({"path": self._rel(p), "name": p.name, "size_kb": round(size / 1024, 1)})
        return out

    def model_files(self) -> list[dict]:
        out = []
        for p in self._walk("*.yaml"):
            try:
                text = p.read_text(encoding="utf-8")
            except OSError:
                continue
            if "\ncomponents:" in text or text.startswith("components:"):
                out.append({"path": self._rel(p), "name": p.stem, "kind": "yaml"})
        for p in self._walk("model.py"):
            out.append({"path": self._rel(p), "name": p.parent.name, "kind": "python"})
        return out

    def resolve(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        # 문자열 접두사로 보면 '작업폴더_other' 같은 이웃 폴더가 통과한다
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"작업 폴더 밖 경로입니다: {rel}")
        return p

    # --- 모델 캐시 ---
    def model(self, rel: str):
        from ..scenario import load_model
        if rel not in self.models:
            path = self.resolve(rel)
            spec = ({"yaml": str(path)} if path.suffix in (".yaml", ".yml")
                    else {"python": str(path)})
            m = load_model(spec).compile()
            m.build()
            self.models[rel] = m
            self._locks[rel] = threading.Lock()
        return self.models[rel]

    def lock(self, rel: str) -> threading.Lock:
        self._locks.setdefault(rel, threading.Lock())
        return self._locks[rel]

    def warm(self, rel: str):
        return self._warm.get(rel)

    def set_warm(self, rel: str, x) -> None:
        self._warm[rel] = x

    def forget(self, rel: str) -> None:
        self.models.pop(rel, None)
        self._warm.pop(rel, None)


# --- 모델 메타데이터 ---------------------------------------------------------

#: 슬라이더로 내보낼 만한 파라미터인지. 보정 대상(tunable)은 운전 손잡이가 아니다.
_DRIVER_RANGES = {
    "util": (0.0, 1.3), "n_tools": (0.5, 2.0), "n_ratio": (0.7, 1.15),
}


#: 자동 추출에서 빼는 파라미터. 물리 상수·기준값이라 운전 중 바뀌지 않는다.
_NOT_DRIVERS = {
    "p_amb", "p_room", "rho_ref", "o2_corr", "O2_ref", "w_O2", "T_ref",
    "idle_frac", "q_idle_frac",
}


def declared_drivers(model) -> list[dict]:
    """모델이 직접 선언한 운전 손잡이 (``DRIVERS``/``drivers:``)."""
    sysm = model.system
    mod = getattr(sysm, "source_module", None)
    if mod is not None and hasattr(mod, "DRIVERS"):
        return list(mod.DRIVERS)
    return list(getattr(sysm, "drivers", []) or [])


def declared_limits(model) -> dict:
    sysm = model.system
    mod = getattr(sysm, "source_module", None)
    if mod is not None and hasattr(mod, "LIMITS"):
        return dict(mod.LIMITS)
    return dict(getattr(sysm, "limits", {}) or {})


def model_meta(model, rel: str) -> dict:
    """슬라이더로 쓸 파라미터를 뽑는다.

    모델이 ``DRIVERS`` 를 선언했으면 그것을 쓰고, 없으면 파라미터 이름의 **마지막
    조각으로 묶어** 자동 추출한다 (``SRC_DRY.util``, ``SRC_CVD.util`` -> 손잡이 하나).
    팹 계통은 같은 종류 설비가 여러 대 붙는 구조라 이 규칙이 잘 맞는다. 다만 자동
    추출은 대기압·기준밀도 같은 상수까지 끌어오므로 선언하는 편이 훨씬 낫다.
    """
    from ..core.units import from_si

    groups: dict[str, list] = {}
    for p in model.parameters:
        if p.name.startswith("_"):
            continue
        key = p.name.rsplit(".", 1)[-1]
        groups.setdefault(key, []).append(p)

    # 선언이 있으면 그대로 따른다 (보정 대상이어도 손잡이로 쓸 수 있다 - 예: 냉각탑
    # 접근온도차는 보정으로 현재 상태를 맞추고, 시나리오에서는 열화를 가정해 본다).
    # 선언이 없을 때만 자동 추출하며, 이때는 보정 대상을 뺀다.
    declared = {d["key"]: d for d in declared_drivers(model)}
    keys = list(declared) if declared else sorted(
        k for k, members in groups.items()
        if k not in _NOT_DRIVERS and not any(m.tunable for m in members))

    drivers = []
    for key in keys:
        members = groups.get(key, [])
        if not members:
            continue
        spec = declared.get(key, {})
        mode = spec.get("mode", "set")
        unit = spec.get("unit", members[0].unit)
        if mode == "scale":
            cur, unit = 1.0, "x"
            lo, hi = float(spec.get("lo", 0.5)), float(spec.get("hi", 2.0))
        else:
            vals = [from_si(m.value, unit) for m in members]
            cur = float(np.median(vals))
            if "lo" in spec and "hi" in spec:
                lo, hi = float(spec["lo"]), float(spec["hi"])
            else:
                lo_hi = _DRIVER_RANGES.get(key)
                if lo_hi:
                    lo, hi = lo_hi
                    if unit == "%":
                        lo, hi = lo * 100, hi * 100
                else:
                    span = abs(cur) if abs(cur) > 1e-12 else 1.0
                    lo, hi = cur - 0.6 * span, cur + 0.6 * span
                spec_lo = min(from_si(m.lo, unit) for m in members) \
                    if np.isfinite(members[0].lo) else -np.inf
                spec_hi = max(from_si(m.hi, unit) for m in members) \
                    if np.isfinite(members[0].hi) else np.inf
                lo = float(max(lo, spec_lo)) if np.isfinite(spec_lo) else float(lo)
                hi = float(min(hi, spec_hi)) if np.isfinite(spec_hi) else float(hi)
            if hi <= lo:
                hi = lo + max(abs(lo) * 0.5, 1.0)
        drivers.append({
            "key": key, "label": spec.get("label", key), "unit": unit, "mode": mode,
            "value": cur, "lo": float(lo), "hi": float(hi),
            "members": [m.name for m in members], "count": len(members),
            "desc": spec.get("desc") or members[0].desc,
        })

    outputs = [{"name": n, "unit": u} for n, (_, u) in model.outputs.items()]
    r = model.report
    return {
        "path": rel, "name": model.system.name,
        "n_eqs": model.n_eqs, "n_vars": model.n_vars,
        "n_components": len(model.system.components),
        "n_connections": len(model.system.connections),
        "ok": r.ok, "blocks": len(r.blocks), "largest_block": r.largest_block,
        "jacobian_nnz": model.jacobian_nnz(),
        "describe": model.describe(),
        "drivers": drivers, "outputs": outputs,
        "limits": {k: (v if isinstance(v, dict) else {"max": v})
                   for k, v in declared_limits(model).items()},
        "declared_drivers": bool(declared),
        "components": [{"name": n, "type": type(c).__name__}
                       for n, c in model.system.components.items()],
    }


def solve_with(model, rel: str, ws: Workspace, overrides: dict) -> dict:
    """드라이버 값을 적용해 정상상태를 푼다. 직전 해를 warm start 로 쓴다."""
    from ..core.solvers import solve_steady
    from ..core.units import to_si

    p = model.p0()
    applied = []
    for key, spec in (overrides or {}).items():
        value = spec.get("value") if isinstance(spec, dict) else spec
        mode = spec.get("mode", "set") if isinstance(spec, dict) else "set"
        # 손잡이가 선언한 단위로 환산한다. 파라미터 단위로 환산하면
        # "가동율 100%" 가 util=100 이 되어 버린다 (실제로 겪은 버그).
        unit = spec.get("unit") if isinstance(spec, dict) else None
        if value is None:
            continue
        for info in model.parameters:
            if info.tunable or info.name.rsplit(".", 1)[-1] != key:
                continue
            i = info.index
            if mode == "scale":
                p[i] = p[i] * float(value)
            else:
                p[i] = to_si(float(value), unit or info.unit)
            applied.append(info.name)

    with ws.lock(rel):
        r = solve_steady(model, p, x0=ws.warm(rel))
        if r.success:
            ws.set_warm(rel, r.x)
        out = model.output_values(r.x, p) if r.success else {}
    return {
        "converged": bool(r.success),
        "message": r.message,
        "applied": applied,
        "outputs": {k: (float(v) if np.isfinite(v) else None) for k, v in out.items()},
        "newton": r.n_newton,
    }


# --- HTTP ------------------------------------------------------------------

class Api:
    """요청 처리기. 라우팅 테이블을 들고 있다."""

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.jobs = JobRegistry()

    # -- 엔드포인트 --
    def workspace(self, _body) -> dict:
        return {"root": str(self.ws.root), "datasets": self.ws.datasets(),
                "models": self.ws.model_files()}

    def profile(self, body) -> dict:
        from ..analyze.profile import profile_dataset
        from ..analyze.units_guess import guess_units
        from ..data.quality import apply

        tab, q = self.ws.table(body["csv"], body.get("timestamp"))
        num = apply(tab.df, q)
        prof = profile_dataset(num)
        guesses = guess_units(num)
        rep = tab.report
        by_col = q.by_column()
        cols = []
        for name, c in prof.columns.items():
            g = guesses[name]
            unit, conf, reason, review = g.unit, g.confidence, g.reason, g.needs_review
            if rep.units.get(name):
                # 파일에 단위 행이 있으면 그게 1순위다 (사람이 적어 둔 것)
                unit, conf, review = rep.units[name], 0.95, False
                reason = f"파일의 단위 행: '{rep.units_raw.get(name, unit)}'"
            cols.append({
                "name": name, "unit": unit, "confidence": round(conf, 2),
                "reason": reason, "alternatives": g.alternatives,
                "needs_review": review, "desc": rep.descriptions.get(name, ""),
                "missing_pct": round(c.missing_pct, 2),
                "min": _num(c.vmin), "max": _num(c.vmax), "mean": _num(c.mean),
                "cv": _num(c.cv), "status": ("상수" if c.is_constant else
                                             f"중복({c.duplicate_of})" if c.duplicate_of
                                             else "사용"),
                "usable": c.usable,
                "status_strings": rep.status_counts.get(name, {}),
                "issues": [{"label": i.label, "kind": i.kind, "n": i.n_points,
                            "message": i.message} for i in by_col.get(name, [])],
                "excluded": q.excluded(name),
            })
        return {
            "rows": prof.n_rows, "interval_s": _num(prof.interval_s),
            "t_start": str(prof.t_start) if prof.t_start is not None else None,
            "t_end": str(prof.t_end) if prof.t_end is not None else None,
            "steady_fraction": _num(prof.steady_fraction),
            "gap_count": prof.gap_count, "time_column": rep.time_column,
            "warnings": prof.warnings, "columns": cols,
            "ingest": {"lines": rep.lines(), "encoding": rep.encoding,
                       "header_rows": rep.header_rows,
                       "status_total": rep.status_total(), "gaps": rep.gaps[:5],
                       "dropped_columns": rep.dropped_columns,
                       "text_columns": list(rep.text_columns)},
            "quality": {"lines": q.lines(), "excluded": q.excluded(),
                        "n_issues": len(q.issues)},
        }

    def preview(self, body) -> dict:
        """컬럼 몇 개의 시계열 (차트용, 다운샘플). 결측 처리한 점은 따로 표시한다."""
        from ..data.quality import apply
        tab, q = self.ws.table(body["csv"], body.get("timestamp"))
        df = tab.df
        cols = [c for c in body.get("columns", []) if c in df.columns]
        limit = int(body.get("limit", 800))
        step = max(1, len(df) // limit)
        # 선은 정제한 값, 뺀 점은 원래 값 그대로 점으로 찍는다 (무엇을 왜 뺐는지 보이게)
        sub = apply(df[cols], q).iloc[::step]
        marks = {}
        for c in cols:
            m = q.masks.get(c)
            if m is None:
                continue
            idx = np.flatnonzero(m)[:3000]
            vals = df[c].to_numpy()[idx]
            marks[c] = [{"i": round(float(i) / step, 3), "v": _num(v)} for i, v in zip(idx, vals)]
        return {
            "t": [str(i) for i in sub.index],
            "series": {c: [_num(v) for v in sub[c]] for c in cols},
            "excluded": marks,
            "issues": {c: [i.to_dict() for i in q.by_column().get(c, [])] for c in cols},
        }

    def analyze(self, body) -> dict:
        from ..analyze import AnalysisConfig, run_analysis
        from ..report import analysis_report

        cfg = AnalysisConfig(
            name=body.get("name", "분석"),
            csv=str(self.ws.resolve(body["csv"])),
            timestamp=body.get("timestamp") or "timestamp",
            target=body["target"], target_unit=body.get("target_unit", ""),
            units=body.get("units") or {}, exclude=body.get("exclude") or [],
            controllable=body.get("controllable") or [],
            objective=body.get("objective", "minimize"),
            train_fraction=float(body.get("train_fraction", 0.6)),
            steady_only=bool(body.get("steady_only", False)),
            drivers=body.get("drivers") or [],
        )
        out_dir = self.ws.root / "out"
        out_dir.mkdir(exist_ok=True)
        report_path = out_dir / f"analysis_{cfg.target[:30].replace('/', '_')}.html"

        def work(job: Job):
            job.message = "프로파일링 및 대리모델 학습 중 ..."
            res = run_analysis(cfg, verbose=False)
            job.message = "리포트 생성 중 ..."
            analysis_report(res, report_path)
            return _analysis_payload(res, self.ws._rel(report_path))

        return {"job": self.jobs.start("analyze", work).payload()}

    def job(self, body) -> dict:
        j = self.jobs.get(body["id"])
        if j is None:
            raise KeyError(f"작업 {body['id']} 를 찾을 수 없습니다")
        return j.payload()

    def model_meta(self, body) -> dict:
        rel = body["model"]
        return model_meta(self.ws.model(rel), rel)

    def model_solve(self, body) -> dict:
        rel = body["model"]
        return solve_with(self.ws.model(rel), rel, self.ws, body.get("overrides") or {})

    def model_sweep(self, body) -> dict:
        """드라이버 하나를 훑으며 정상상태를 연속으로 푼다."""
        rel = body["model"]
        model = self.ws.model(rel)
        key, values = body["key"], body["values"]
        base = dict(body.get("overrides") or {})
        prev = base.get(key) if isinstance(base.get(key), dict) else {}
        rows = []
        for v in values:
            ov = dict(base)
            ov[key] = {"value": v, "mode": prev.get("mode", "set"), "unit": prev.get("unit")}
            r = solve_with(model, rel, self.ws, ov)
            rows.append({"x": v, "converged": r["converged"], "outputs": r["outputs"]})
        return {"key": key, "rows": rows}

    def reload_model(self, body) -> dict:
        self.ws.forget(body["model"])
        return self.model_meta(body)

    # -- 모델 만들기 --
    def builder_palette(self, _body) -> dict:
        from ..builder import palette
        return {"palette": palette(self.ws.root)}

    def builder_load(self, body) -> dict:
        from ..builder import load_model_dict
        path = self.ws.resolve(body["path"])
        return {"model": load_model_dict(path), "path": self.ws._rel(path)}

    def builder_check(self, body) -> dict:
        """화면의 모델(dict) 또는 YAML 글을 검사한다. 파일은 쓰지 않는다."""
        import yaml as _yaml

        from ..builder import check_model, dump_model
        if "yaml" in body:
            try:
                model = _yaml.safe_load(body["yaml"]) or {}
            except _yaml.YAMLError as exc:
                return {"ok": False, "stage": "yaml", "errors": [f"YAML 문법 오류: {exc}"], "warnings": []}
            if not isinstance(model, dict):
                return {"ok": False, "stage": "yaml", "errors": ["YAML 최상위는 name/components/connections "
                                                                  "를 가진 사전이어야 합니다."], "warnings": []}
        else:
            model = body.get("model") or {}
        base = self.ws.resolve(body["path"]).parent if body.get("path") else self.ws.root / "models"
        res = check_model(model, base_dir=base, solve=bool(body.get("solve", True)))
        return {**res, "model": model, "yaml": dump_model(model)}

    def builder_save(self, body) -> dict:
        from ..builder import save_model
        rel = body.get("path") or f"models/{_safe_name(body['model'].get('name') or 'model')}.yaml"
        path = self.ws.resolve(rel)
        if path.suffix not in (".yaml", ".yml"):
            raise ValueError("모델은 .yaml 로 저장합니다.")
        saved = save_model(body["model"], path)
        self.ws.forget(self.ws._rel(saved))
        return {"path": self.ws._rel(saved)}

    # -- 프로젝트 (설정 저장·열기) --
    def project_list(self, _body) -> dict:
        from ..project import list_projects
        items = list_projects(self.ws.root)
        for it in items:
            it["path"] = self.ws._rel(it["path"])
        return {"projects": items}

    def project_save(self, body) -> dict:
        from ..project import save_project
        path = save_project(self.ws.root, body.get("name") or "project", body.get("run") or {},
                            ui=body.get("ui"), forecast=body.get("forecast"),
                            results=body.get("results"), forecast_results=body.get("forecast_results"))
        return {"path": self.ws._rel(path)}

    def project_load(self, body) -> dict:
        from ..project import load_project
        doc = load_project(self.ws.resolve(body["path"]))
        # 옮겨 온 프로젝트는 데이터·모델 경로가 없을 수 있다 — 열되 알려준다
        run = doc.get("run") or {}
        missing = [run[k] for k in ("csv", "model") if run.get(k) and not (self.ws.root / run[k]).exists()]
        return {**doc, "path": body["path"], "missing": missing}

    # -- 미래 예측 (계획 → 예측) --
    def forecast_plans(self, _body) -> dict:
        out = []
        for folder in ("plans", "uploads"):
            d = self.ws.root / folder
            for p in sorted(d.glob("*.csv")) if d.is_dir() else []:
                out.append({"path": self.ws._rel(p), "name": p.name, "folder": folder,
                            "size_kb": round(p.stat().st_size / 1024, 1)})
        return {"plans": out}

    def _history(self, body):
        from ..forecast import _history_table
        return _history_table(self.ws.resolve(body["csv"]), body.get("time_column"), list(body.get("features") or []))

    def forecast_make(self, body) -> dict:
        """최근 N일 패턴을 반복하고 조정한 계획을 plans/ 에 저장한다 (엑셀에서 고쳐 다시 올릴 수 있음)."""
        from ..forecast import check_plan, make_plan, save_plan
        feats = list(body.get("features") or [])
        plan = make_plan(self.ws.resolve(body["csv"]), body.get("time_column"), feats,
                         base_days=float(body.get("base_days", 7)), horizon_days=float(body.get("horizon_days", 14)),
                         step=str(body.get("step") or "1h"), adjust=body.get("adjust") or [])
        name = _safe_name(body.get("plan_name") or f"{Path(body['csv']).stem}_계획_{plan.index.min():%Y%m%d}")
        path = save_plan(plan, self.ws.root / "plans" / f"{name}.csv")
        chk = check_plan(plan, feats, self._history(body))
        return {"path": self.ws._rel(path), "check": chk, "preview": _plan_preview(plan, feats)}

    def forecast_check(self, body) -> dict:
        from ..forecast import check_plan, read_plan
        plan, log = read_plan(self.ws.resolve(body["plan"]))
        feats = list(body.get("features") or [])
        chk = check_plan(plan, feats, self._history(body))
        return {"path": body["plan"], "check": chk, "log": log, "preview": _plan_preview(plan, feats)}

    def forecast_run(self, body) -> dict:
        from ..easy import config_from_body, mapping_issues
        from ..forecast import forecast_easy
        cfg = config_from_body(body, self.ws.resolve)
        if cfg.model:
            issues = mapping_issues(self.ws.model(body["model"]), cfg)
            if issues:
                raise ValueError("컬럼 연결을 고치세요: " + " / ".join(
                    f"{i['column']} → {i['target']}: {i['message']}" for i in issues))
        plan = self.ws.resolve(body["plan"])
        hist = body.get("history")
        hist = tuple(hist) if hist and all(hist) else None
        root, band = self.ws.root, bool(body.get("band", True))

        def work(job: Job):
            return forecast_easy(cfg, plan, root, history=hist, band=band,
                                 progress=lambda m: setattr(job, "message", m))

        return {"job": self.jobs.start("forecast", work).payload()}

    # -- 간편 예측 (업로드 → 변수 → 기간 → 모델 → 결과) --
    def upload(self, name: str, data: bytes) -> dict:
        """CSV 를 작업 폴더의 ``uploads/`` 에 저장한다. 이 PC 밖으로는 아무것도 나가지 않는다."""
        base = Path(str(name).replace("\\", "/")).name
        stem, suffix = Path(base).stem, Path(base).suffix.lower()
        if suffix not in (".csv", ".txt"):
            raise ValueError("CSV 파일만 올릴 수 있습니다 (.csv). 엑셀은 'CSV 로 저장' 후 올리세요.")
        if not data:
            raise ValueError("빈 파일입니다.")
        safe = re.sub(r"[^0-9A-Za-z가-힣._()\- ]+", "_", stem).strip(" ._") or "upload"
        folder = self.ws.root / "uploads"
        folder.mkdir(exist_ok=True)
        path, i = folder / f"{safe}.csv", 1
        while path.exists() and path.read_bytes() != data:
            path, i = folder / f"{safe}_{i}.csv", i + 1
        path.write_bytes(data)
        return {"path": self.ws._rel(path), "name": path.name,
                "size_kb": round(len(data) / 1024, 1)}

    def easy_presets(self, body) -> dict:
        from ..easy import find_presets
        tab, _ = self.ws.table(body["csv"], body.get("timestamp"))
        pre = find_presets(self.ws.root, list(tab.df.columns))
        for p in pre:
            p["config"] = self.ws._rel(Path(p["config"]))
            if p.get("model"):
                p["model"] = self.ws._rel(Path(p["model"]))
        return {"presets": pre}

    def easy_catalog(self, body) -> dict:
        from ..easy import model_catalog, param_sensitivity
        rel = body["model"]
        model = self.ws.model(rel)
        with self.ws.lock(rel):
            cat = model_catalog(model)
            sens = {}
            if body.get("target"):
                try:
                    sens = param_sensitivity(model, body["target"])
                except Exception:  # noqa: BLE001 — 민감도는 추천용일 뿐이다
                    sens = {}
        for p in cat["params"]:
            p["sensitivity_pct"] = _num(sens.get(p["value"]))
        suggested = cat["declared_params"] or [
            k for k, v in sorted(sens.items(), key=lambda kv: -kv[1]) if v > 1.0][:4]
        # 구성방정식 슬롯과 후보 (계통에 있는 컴포넌트 종류가 정한다)
        from ..easy import closure_catalog
        from ..scenario import load_model
        path = self.ws.resolve(rel)
        system = load_model({"yaml": str(path)} if path.suffix in (".yaml", ".yml") else {"python": str(path)})
        return {**cat, "suggested": suggested, "closures": closure_catalog(system)}

    def easy_check(self, body) -> dict:
        from ..easy import check, config_from_body, mapping_issues
        cfg = config_from_body(body, self.ws.resolve)
        tab, _ = self.ws.table(body["csv"], body.get("time_column"))
        out = check(tab.df, cfg)
        if body.get("model") and (cfg.target_map is not None or cfg.feature_map):
            out["mapping_issues"] = mapping_issues(self.ws.model(body["model"]), cfg)
        return out

    def easy_closures(self, body) -> dict:
        """구성방정식 후보 비교 (학습 기간만). 보정을 후보 수만큼 하므로 작업으로 돌린다."""
        from ..easy import advise_easy, config_from_body, mapping_issues, period_errors
        cfg = config_from_body(body, self.ws.resolve)
        errs = period_errors(cfg.train, cfg.test, cfg.embargo_days)
        if errs:
            raise ValueError(" / ".join(e.replace("**", "") for e in errs))
        if cfg.model:
            issues = mapping_issues(self.ws.model(body["model"]), cfg)
            if issues:
                raise ValueError("컬럼 연결을 고치세요: " + " / ".join(
                    f"{i['column']} → {i['target']}: {i['message']}" for i in issues))
        root = self.ws.root
        slots = body.get("slots") or None

        def work(job: Job):
            return advise_easy(cfg, root, progress=lambda m: setattr(job, "message", m), slots=slots)

        return {"job": self.jobs.start("closures", work).payload()}

    def easy_run(self, body) -> dict:
        from ..easy import config_from_body, mapping_issues, period_errors, run_easy
        cfg = config_from_body(body, self.ws.resolve)
        errs = period_errors(cfg.train, cfg.test, cfg.embargo_days)
        if errs:
            raise ValueError(" / ".join(e.replace("**", "") for e in errs))
        if cfg.model:
            issues = mapping_issues(self.ws.model(body["model"]), cfg)
            if issues:
                raise ValueError("컬럼 연결을 고치세요: " + " / ".join(
                    f"{i['column']} → {i['target']}: {i['message']}" for i in issues))
        root = self.ws.root

        def work(job: Job):
            return run_easy(cfg, root, progress=lambda m: setattr(job, "message", m))

        return {"job": self.jobs.start("easy", work).payload()}


def _plan_preview(plan, feats: list[str], n: int = 400) -> dict:
    """계획 미리보기 (입력마다 시계열, 최대 n 점)."""
    cols = [c for c in feats if c in plan.columns]
    k = max(1, len(plan) // n)
    sub = plan[cols].iloc[::k]
    return {"t": [str(t) for t in sub.index],
            "columns": {c: [_num(v) for v in sub[c].to_numpy()] for c in cols}}


def _safe_name(name: str) -> str:
    """파일 이름으로 쓸 수 있게 (한글·영문·숫자·_- 만)."""
    return re.sub(r"[^0-9A-Za-z가-힣_\-]+", "_", str(name)).strip("_") or "model"


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _detect_time_column(df) -> str | None:
    for c in df.columns[:3]:
        lc = str(c).lower()
        if "time" in lc or "date" in lc or lc in ("ts", "t"):
            return str(c)
    return None


def _analysis_payload(res, report_rel: str) -> dict:
    s = res.surrogate
    return {
        "report": report_rel,
        "headline": res.headline(),
        "ladder": [{"level": r.level, "name": r.name, "status": r.status,
                    "finding": r.finding, "blocker": r.blocker,
                    "how_to_unblock": r.how_to_unblock} for r in res.rungs],
        "balances": [{"formula": b.formula(), "confidence": b.confidence,
                      "residual_pct": round(b.residual_rel * 100, 3),
                      "r2": _num(b.r2), "dimension": b.dimension,
                      "is_conservation": b.is_conservation} for b in res.balances],
        "pi_groups": [{"formula": g.formula(), "has_target": g.contains_target}
                      for g in res.pis if len(g.exponents) > 1
                      and not g.is_trivial_ratio(res.config.units)],
        "surrogate": None if s is None else {
            "model": s.model_name, "r2_cv": _num(s.r2_cv), "rmse_cv": _num(s.rmse_cv),
            "rmse_train": _num(s.rmse_train), "baseline_rmse": _num(s.baseline_rmse),
            "unit": s.unit, "skill": _num(s.skill),
            "clusters": [{"label": c.label, "members": c.members,
                          "pct": _num(c.importance_pct),
                          "internal_corr": _num(c.max_internal_corr),
                          "is_group": c.is_group} for c in s.clusters],
            "importances": [{"feature": i.feature, "pct": _num(i.importance_pct),
                             "entangled": i.entangled, "corr": _num(i.max_corr),
                             "partner": i.max_corr_with} for i in s.importances],
            "pred": {
                "t": [str(i) for i in s.pred.index[::max(1, len(s.pred) // 600)]],
                "y": [_num(v) for v in s.pred.to_numpy()[::max(1, len(s.pred) // 600)]],
                "meas": [_num(v) for v in
                         res.df.loc[s.pred.index, res.config.target]
                         .to_numpy()[::max(1, len(s.pred) // 600)]],
            } if s.pred is not None else None,
        },
        "data_log": {"ingest": res.ingest.lines() if res.ingest is not None else [],
                     "quality": res.quality.lines() if res.quality is not None else []},
        "untrainable": [{"feature": f, "train_value": _num(v), "test_min": _num(lo),
                         "test_max": _num(hi)} for f, v, lo, hi in res.untrainable],
        "split": res.split.to_dict() if res.split is not None else None,
        "forecast": ({k: (_num(v) if isinstance(v, (int, float, np.floating)) else v)
                      for k, v in res.forecast.items() if k != "pred"}
                     if res.forecast else {}),
        "holdout_outside": _num(res.holdout_outside),
        "holdout_rmse": _num(res.holdout_rmse),
        "improvement": None if res.improvement is None else {
            "settings": res.improvement.settings,
            "baseline": _num(res.improvement.baseline_value),
            "predicted": _num(res.improvement.predicted_value),
            "change_pct": _num(res.improvement.change_pct),
            "feasible": res.improvement.feasible,
        },
    }


_ROUTES: dict[str, str] = {
    "/api/workspace": "workspace",
    "/api/profile": "profile",
    "/api/preview": "preview",
    "/api/analyze": "analyze",
    "/api/job": "job",
    "/api/model/meta": "model_meta",
    "/api/model/solve": "model_solve",
    "/api/model/sweep": "model_sweep",
    "/api/model/reload": "reload_model",
    "/api/easy/presets": "easy_presets",
    "/api/easy/catalog": "easy_catalog",
    "/api/easy/check": "easy_check",
    "/api/easy/run": "easy_run",
    "/api/easy/closures": "easy_closures",
    "/api/builder/palette": "builder_palette",
    "/api/builder/load": "builder_load",
    "/api/builder/check": "builder_check",
    "/api/builder/save": "builder_save",
    "/api/project/list": "project_list",
    "/api/project/save": "project_save",
    "/api/project/load": "project_load",
    "/api/forecast/plans": "forecast_plans",
    "/api/forecast/make": "forecast_make",
    "/api/forecast/check": "forecast_check",
    "/api/forecast/run": "forecast_run",
}


def make_handler(api: Api):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "pforecast"

        def log_message(self, fmt, *args):       # 요청 로그로 콘솔을 어지럽히지 않는다
            if "/api/" in (args[0] if args else ""):
                return

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict) -> None:
            self._send(code, json.dumps(payload, ensure_ascii=False,
                                        default=str).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):                         # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if path.startswith("/api/"):
                query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                return self._handle(path, query)
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path.startswith("/out/"):
                return self._workspace_file(unquote(path).lstrip("/"))   # 한글 파일 이름
            self._json(404, {"error": f"없는 경로: {path}"})

        def do_POST(self):                        # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            parsed = urlparse(self.path)
            if parsed.path == "/api/upload":
                if length > MAX_UPLOAD:
                    # 본문을 읽지 않으면 연결이 꼬이므로 버리면서 읽는다
                    left = length
                    while left > 0:
                        left -= len(self.rfile.read(min(left, 1 << 20)) or b"x" * left)
                    return self._json(413, {"error": f"파일이 {MAX_UPLOAD // 2**20} MB 를 넘습니다."})
                data = self.rfile.read(length) if length else b""
                query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                try:
                    return self._json(200, api.upload(query.get("name", "upload.csv"), data))
                except ValueError as exc:
                    return self._json(400, {"error": str(exc)})
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError as exc:
                return self._json(400, {"error": f"JSON 파싱 실패: {exc}"})
            self._handle(urlparse(self.path).path, body)

        def _handle(self, path: str, body: dict) -> None:
            name = _ROUTES.get(path)
            if name is None:
                return self._json(404, {"error": f"없는 API: {path}"})
            try:
                self._json(200, api.__getattribute__(name)(body))
            except KeyError as exc:
                self._json(400, {"error": f"입력이 부족합니다: {exc}"})
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            except FileNotFoundError as exc:
                self._json(404, {"error": str(exc)})
            except Exception as exc:
                traceback.print_exc()
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def _static(self, rel: str) -> None:
            p = (STATIC_DIR / rel).resolve()
            if not str(p).startswith(str(STATIC_DIR.resolve())) or not p.exists():
                return self._json(404, {"error": f"없는 파일: {rel}"})
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            self._send(200, p.read_bytes(), ctype)

        def _workspace_file(self, rel: str) -> None:
            try:
                p = api.ws.resolve(rel)
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            if not p.exists():
                return self._json(404, {"error": f"없는 파일: {rel}"})
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            if ctype.startswith("text/"):
                ctype += "; charset=utf-8"
            self._send(200, p.read_bytes(), ctype)

    return Handler


def serve(root: str | Path = ".", host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True) -> None:
    ws = Workspace(Path(root))
    api = Api(ws)
    httpd = ThreadingHTTPServer((host, port), make_handler(api))
    url = f"http://{host}:{port}/"
    print(f"pforecast 앱 실행 중 → {url}")
    print(f"  작업 폴더: {ws.root}")
    print(f"  데이터 {len(ws.datasets())}개, 모델 {len(ws.model_files())}개")
    print("  종료: Ctrl+C")
    if open_browser:
        try:
            import webbrowser
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n종료합니다.")
    finally:
        httpd.server_close()
