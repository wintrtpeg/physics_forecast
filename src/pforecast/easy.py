"""간편 예측 — CSV 하나로 타깃·입력·기간을 고르고 물리모델과 ML 을 같은 조건에서 비교한다.

앱의 첫 화면이 이것을 부른다. 규칙은 CLI(`pf calibrate`)와 같다.

* 학습·예측 기간은 **시간순**이다. 예측 시작은 학습 끝 + 간격(기본 1일) 이후여야 하고,
  어기면 실행하지 않는다 (``period_errors``).
* 값 정제(교정 창·고착·스파이크)는 학습·간격·예측 구간을 **따로** 한다.
* 입력(x)은 예측 시점에 **미리 아는 값**이어야 한다. 물리모델의 관측값(결과)과 같은
  컬럼을 x 로 고르면 경고한다 — 같은 시각의 결과값으로 맞히는 것은 예측이 아니다.
* ML 은 다항식과 그래디언트 부스팅 둘 다, 같은 학습 행·같은 입력으로 학습한다.
* 채점은 학습 운전영역 밖 행(외삽)과 안쪽 행을 따로 낸다.

물리모델이 없으면(또는 사용자가 'ML 만'을 고르면) ML 만 돌리고, 그 이유를 적는다.
토폴로지는 데이터에서 유도할 수 없으므로 물리모델은 사용자가 고른 모델과 컬럼 연결로만
만든다.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .core.units import from_si
from .data.split import check_split
from .data.tagmap import TagEntry, TagMap

#: 예측 결과 차트에 보내는 최대 점 수 (브라우저가 가볍게 그리는 선)
MAX_POINTS = 2400


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

@dataclass
class ColumnMap:
    """CSV 컬럼 하나를 모델의 무엇에 연결하는가."""

    column: str
    targets: list[str] = field(default_factory=list)   # 비면 물리모델과 연결하지 않음
    unit: str = "1"
    scale: float = 1.0
    sigma: float | None = None

    @classmethod
    def from_dict(cls, column: str, d: dict | None) -> "ColumnMap":
        d = d or {}
        t = d.get("targets", d.get("target")) or []
        if isinstance(t, str):
            t = [x.strip() for x in t.split(",") if x.strip()]
        sigma = d.get("sigma")
        return cls(column, list(t), str(d.get("unit") or "1"), float(d.get("scale") or 1.0),
                   float(sigma) if sigma not in (None, "") else None)


@dataclass
class EasyConfig:
    csv: str
    target: str                                   # y 컬럼
    features: list[str]                           # x 컬럼 (미리 아는 값)
    train: tuple[str, str]                        # 날짜, 양끝 포함
    test: tuple[str, str]
    embargo_days: float = 1.0
    time_column: str | None = None
    target_unit: str = ""
    model: str | None = None                      # 모델 파일 경로. None 이면 ML 만
    target_map: ColumnMap | None = None
    feature_map: dict[str, ColumnMap] = field(default_factory=dict)
    extra_obs: list[ColumnMap] = field(default_factory=list)   # 보정에 같이 쓸 관측
    params: list[str] = field(default_factory=list)
    max_rows: int = 150
    band: bool = False                            # 상태 변동 폭 (느림)
    limit: float | None = None                    # y 관리기준 (초과 시간 채점)
    name: str = "easy"


# ---------------------------------------------------------------------------
# 기간
# ---------------------------------------------------------------------------

def _bound(s: str, end: bool) -> pd.Timestamp:
    """'2025-05-31' 은 끝이면 그날 24시까지로 본다 (양끝 포함)."""
    t = pd.Timestamp(s)
    if end and re.fullmatch(r"\s*\d{4}-\d{2}-\d{2}\s*", str(s)):
        t = t + pd.Timedelta(days=1)
    return t


def period_bounds(train, test) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    return _bound(train[0], False), _bound(train[1], True), _bound(test[0], False), _bound(test[1], True)


def period_masks(index: pd.DatetimeIndex, train, test) -> tuple[np.ndarray, np.ndarray]:
    a, b, c, d = period_bounds(train, test)
    return (np.asarray((index >= a) & (index < b)), np.asarray((index >= c) & (index < d)))


def period_errors(train, test, embargo_days: float = 1.0) -> list[str]:
    """실행을 막아야 하는 기간 설정. 누수는 경고가 아니라 거절한다."""
    try:
        a, b, c, d = period_bounds(train, test)
    except (ValueError, TypeError) as exc:
        return [f"날짜를 읽지 못했습니다: {exc}"]
    out = []
    if b <= a:
        out.append("학습 기간의 끝이 시작보다 앞입니다.")
    if d <= c:
        out.append("예측 기간의 끝이 시작보다 앞입니다.")
    gap = (c - b) / pd.Timedelta(days=1)
    if c < b:
        out.append("예측 기간이 학습 기간과 겹치거나 앞에 있습니다. 예측은 학습이 끝난 **뒤**여야 "
                   "합니다 — 겹치면 미래를 이미 본 셈입니다 (누수).")
    elif gap < embargo_days - 1e-9:
        out.append(f"학습 끝과 예측 시작 사이 간격이 {gap:.2f}일입니다. 최소 {embargo_days:g}일은 "
                   "띄워야 합니다 — 경계 부근은 거의 같은 상태라 예측이 쉬워집니다.")
    return out


# ---------------------------------------------------------------------------
# 모델 선언
# ---------------------------------------------------------------------------

def _decl(model, attr: str, yaml_key: str, default):
    sysm = model.system
    mod = getattr(sysm, "source_module", None)
    if mod is not None and hasattr(mod, attr):
        return getattr(mod, attr)
    return getattr(sysm, yaml_key, None) or default


def declared_calibration(model) -> list[str]:
    return [p for p in _decl(model, "CALIBRATE", "calibrate", []) if _has_param(model, p)]


def declared_observables(model) -> dict[str, str]:
    obs = _decl(model, "OBSERVABLES", "observables", {})
    return dict(obs) if isinstance(obs, dict) else {o: "" for o in obs}


def _has_param(model, name: str) -> bool:
    try:
        model.par_index(name)
        return True
    except KeyError:
        return False


def model_catalog(model) -> dict:
    """컬럼 연결 화면에 필요한 목록: 입력으로 쓸 파라미터, 관측으로 쓸 값, 보정 후보."""
    from .calib.runner import resolve_targets

    inputs = []
    seen = set()
    for d in _decl(model, "DRIVERS", "drivers", []):
        key = d.get("key")
        members = [p.name for p in model.parameters
                   if not p.tunable and p.name.rsplit(".", 1)[-1] == key]
        if not members:
            continue
        if len(members) > 1 and d.get("mode", "set") != "scale":
            inputs.append({"value": ",".join(members), "label": f"{d.get('label', key)} — 전체 {len(members)}개",
                           "unit": d.get("unit", ""), "group": "운전 손잡이"})
        for m in members:
            info = model.parameters[model.par_index(m)]
            inputs.append({"value": m, "label": f"{m} ({d.get('label', key)})",
                           "unit": d.get("unit", info.unit), "group": "운전 손잡이"})
            seen.add(m)
    for p in model.parameters:
        if p.tunable or p.name in seen or p.name.startswith("_"):
            continue
        inputs.append({"value": p.name, "label": f"{p.name} — {p.desc}" if p.desc else p.name,
                       "unit": p.unit, "group": "그 밖의 파라미터"})

    obs_decl = declared_observables(model)
    observables = [{"value": n, "label": f"{n} — {d}" if d else n} for n, d in obs_decl.items()]
    for n, (_, u) in model.outputs.items():
        if n not in obs_decl:
            observables.append({"value": n, "label": f"{n} [{u}]"})
    units = {}
    for o in observables:
        try:
            vt, ot = resolve_targets(model, [o["value"]])
            units[o["value"]] = (model.variables[vt[0][1]].unit if vt else model.outputs[ot[0]][1])
        except (KeyError, IndexError):
            units[o["value"]] = ""
    for o in observables:
        o["unit"] = units.get(o["value"], "")

    params = [{"value": p.name, "unit": p.unit, "desc": p.desc,
               "value_now": from_si(p.value, p.unit)} for p in model.tunable_params()]
    return {"inputs": inputs, "observables": observables, "params": params,
            "declared_params": declared_calibration(model)}


def param_sensitivity(model, target: str, rel: float = 0.01) -> dict[str, float]:
    """설계점에서 보정 후보 파라미터를 1% 바꿀 때 타깃이 몇 % 바뀌는가."""
    from .calib.runner import resolve_targets
    from .core.solvers import solve_steady

    vt, ot = resolve_targets(model, [target])

    def value(p, x0):
        r = solve_steady(model, p, x0=x0)
        if not r.success:
            return np.nan, x0
        if vt:
            return float(r.x[vt[0][1]]), r.x
        return float(model.output_values_si(r.x, p)[ot[0]]), r.x

    p0 = model.p0()
    y0, x = value(p0, model.x0())
    out = {}
    if not np.isfinite(y0) or y0 == 0:
        return out
    for info in model.tunable_params():
        p = p0.copy()
        p[info.index] = p[info.index] * (1 + rel) if p[info.index] != 0 else rel
        y1, _ = value(p, x)
        if np.isfinite(y1):
            out[info.name] = abs((y1 / y0 - 1.0) / rel) * 100.0
    return out


def suggest_params(model, target: str, n: int = 4) -> list[str]:
    """선언이 있으면 그것, 없으면 타깃에 가장 민감한 파라미터 n 개."""
    dec = declared_calibration(model)
    if dec:
        return dec
    sens = param_sensitivity(model, target)
    return [k for k, v in sorted(sens.items(), key=lambda kv: -kv[1]) if v > 1.0][:n]


def mapping_issues(model, cfg: EasyConfig) -> list[dict]:
    """컬럼 단위와 연결한 모델 변수의 차원이 맞는가. 안 맞으면 실행 전에 알린다."""
    from .calib.runner import resolve_targets
    from .core.units import dim_of, dim_str

    def model_unit(name: str) -> str | None:
        try:
            return model.parameters[model.par_index(name)].unit
        except KeyError:
            pass
        try:
            vt, ot = resolve_targets(model, [name])
            return model.variables[vt[0][1]].unit if vt else model.outputs[ot[0]][1]
        except (KeyError, IndexError):
            return None

    rows = [(c, m) for c, m in cfg.feature_map.items() if m.targets and c in cfg.features]
    if cfg.target_map is not None and cfg.target_map.targets:
        rows.append((cfg.target, cfg.target_map))
    rows += [(e.column, e) for e in cfg.extra_obs if e.targets]
    out = []
    for col, m in rows:
        for t in m.targets[:1]:
            mu = model_unit(t)
            if mu is None:
                out.append({"column": col, "target": t, "message": f"모델에 '{t}' 가 없습니다."})
                continue
            try:
                dc, dm = dim_of(m.unit or "1"), dim_of(mu or "1")
            except Exception as exc:  # noqa: BLE001 — 모르는 단위 기호
                out.append({"column": col, "target": t,
                            "message": f"단위 '{m.unit}' 를 해석하지 못했습니다 ({exc}). 예: %, degC, m3/h, mg/Nm3"})
                continue
            if dc != dm:
                out.append({"column": col, "target": t,
                            "message": f"단위 차원이 다릅니다: 컬럼 {m.unit} [{dim_str(dc)}] vs 모델 {mu} "
                                       f"[{dim_str(dm)}]. 단위를 고치거나, 비율로 넣으려면 단위를 1 로 두고 "
                                       "배율을 지정하세요 (예: 60 Hz = 1.0 → 배율 0.016667)."})
    return out


# ---------------------------------------------------------------------------
# 저장된 설정 찾기 (같은 태그를 쓰는 CSV 면 연결을 자동으로 채운다)
# ---------------------------------------------------------------------------

def find_presets(root: Path, columns: list[str]) -> list[dict]:
    """작업 폴더의 보정 설정(YAML) 중 이 CSV 의 컬럼과 태그가 맞는 것."""
    import yaml

    from .workflow import WorkflowConfig

    import os

    cols = set(columns)
    out = []
    skip = {".git", "__pycache__", "node_modules", ".venv", "venv", "site-packages", ".tox"}
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".")]
        files += [Path(dirpath) / f for f in filenames if f.endswith((".yaml", ".yml"))]
        if len(files) > 2000:            # 홈 폴더 전체를 작업 폴더로 띄운 경우
            break
    for p in sorted(files):
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 — 깨진 YAML 은 건너뛴다
            continue
        if not isinstance(d, dict) or "model" not in d or not (d.get("data") or {}).get("tagmap"):
            continue
        try:
            cfg = WorkflowConfig.load(p)
            tm = TagMap.load(cfg.tagmap)
        except Exception:  # noqa: BLE001
            continue
        tags = tm.all_tags
        hit = [t for t in tags if t in cols]
        if not tags or len(hit) / len(tags) < 0.6:
            continue
        model = cfg.model if isinstance(cfg.model, str) else (cfg.model or {}).get("python") or \
            (cfg.model or {}).get("yaml")
        # 설정이 말하는 타깃(기준모델 타깃)의 컬럼과 입력 컬럼 — 화면의 기본 선택
        tgt = next((e.tag for e in tm.observations if e.primary == cfg.baseline_target), None) \
            or next((e.tag for e in tm.observations if e.tag in cols), None)
        out.append({
            "config": str(p), "name": cfg.name, "model": model, "coverage": len(hit) / len(tags),
            "target_column": tgt,
            "params": list(cfg.params),
            "inputs": [{"column": e.tag, "targets": e.targets, "unit": e.unit, "scale": e.scale}
                       for e in tm.inputs if e.tag in cols],
            "observations": [{"column": e.tag, "targets": e.targets, "unit": e.unit,
                              "sigma": e.sigma} for e in tm.observations if e.tag in cols],
            "time_column": tm.timestamp_column,
        })
    out.sort(key=lambda x: -x["coverage"])
    return out


# ---------------------------------------------------------------------------
# 기간 검사 (화면에서 기간을 바꿀 때마다)
# ---------------------------------------------------------------------------

def check(df: pd.DataFrame, cfg: EasyConfig) -> dict:
    """원본 표(정리 전)로 빠르게 본다. 실행할 때는 정제 후 값으로 다시 검사한다."""
    errs = period_errors(cfg.train, cfg.test, cfg.embargo_days)
    if errs:
        return {"ok": False, "errors": errs, "warnings": []}
    tr, te = period_masks(df.index, cfg.train, cfg.test)
    feats = [c for c in cfg.features if c in df.columns]
    have_y = df[cfg.target].notna().to_numpy() if cfg.target in df.columns else np.zeros(len(df), bool)
    chk = check_split(df[tr], df[te], feats)
    warns = [w for w in chk.warnings()]
    n_tr, n_te = int((tr & have_y).sum()), int((te & have_y).sum())
    if n_tr < 50:
        errs.append(f"학습 기간에 타깃 값이 있는 행이 {n_tr}개뿐입니다. 기간을 늘리세요.")
    if n_te < 10:
        errs.append(f"예측 기간에 타깃 값이 있는 행이 {n_te}개뿐입니다. 채점할 수 없습니다.")
    if not feats:
        errs.append("입력(x)을 하나 이상 고르세요.")
    ranges = [{"column": c, "train_lo": lo, "train_hi": hi, "test_lo": tlo, "test_hi": thi,
               "outside_pct": chk.per_column.get(c, 0.0) * 100}
              for c, (lo, hi, tlo, thi) in chk.ranges.items()]
    return {"ok": not errs, "errors": errs, "warnings": warns,
            "n_train": n_tr, "n_test": n_te, "is_future": chk.is_future,
            "embargo_days": chk.embargo_days, "extrapolation": chk.extrapolation,
            "ranges": ranges, "lines": chk.lines()}


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

_SAFE = re.compile(r"[^0-9A-Za-z가-힣_.-]+")


def _slug(s: str) -> str:
    return _SAFE.sub("_", s).strip("_")[:60] or "easy"


def _tagmap(cfg: EasyConfig, physics: bool) -> TagMap:
    """사용자가 고른 연결로 태그맵을 만든다. ML 전용 컬럼은 모델과 무관한 이름으로 싣는다."""
    tm = TagMap(timestamp_column=cfg.time_column or "timestamp", resample=None)
    for c in cfg.features:
        m = cfg.feature_map.get(c)
        if physics and m is not None and m.targets:
            tm.inputs.append(TagEntry(tag=c, target=list(m.targets), unit=m.unit, scale=m.scale))
        else:
            # 물리모델과 연결하지 않은 x: ML 만 쓴다. 관측 쪽에 실어 두면 모델 입력으로
            # 해석되지 않고, 같은 정제를 거친다.
            tm.observations.append(TagEntry(tag=c, target=f"x::{c}", unit="1"))
    if physics and cfg.target_map is not None and cfg.target_map.targets:
        tm.observations.append(TagEntry(tag=cfg.target, target=list(cfg.target_map.targets)[:1],
                                        unit=cfg.target_map.unit, sigma=cfg.target_map.sigma))
    else:
        tm.observations.append(TagEntry(tag=cfg.target, target=f"y::{cfg.target}", unit="1"))
    if physics:
        for e in cfg.extra_obs:
            if e.targets and e.column not in (cfg.target, *cfg.features):
                tm.observations.append(TagEntry(tag=e.column, target=list(e.targets)[:1],
                                                unit=e.unit, sigma=e.sigma))
    return tm


def _metrics(pred, meas) -> dict:
    from .calib.estimator import _metrics as m
    r = m(np.asarray(pred, float), np.asarray(meas, float))
    return {k: (float(v) if v is not None and np.isfinite(v) else None) for k, v in r.items()}


def _ml_models(Xtr, ytr):
    """(이름, 종류, 예측함수, 설명) 목록. 설명의 설정값은 학습된 객체에서 읽는다."""
    from .calib import PolyRidgeBaseline
    models = []
    ok = np.all(np.isfinite(Xtr), axis=1) & np.isfinite(ytr)
    k, n = Xtr.shape[1], int(ok.sum())
    poly = PolyRidgeBaseline(degree=2)
    poly.fit(Xtr[ok], ytr[ok], [f"x{i}" for i in range(k)])
    models.append(("ML 다항(2차)", "poly", poly.predict, {
        "algorithm": "다항 릿지 회귀 (2차, 교차항 포함)",
        "formula": "ŷ = β_{0} + Σ_{i} β_{i} z_{i} + Σ_{i≤j} β_{ij} z_{i} z_{j},   "
                   "z_{i} = (x_{i} − 평균_{i}) / 표준편차_{i}",
        "fit": "정규방정식 (ZᵀZ + αI) β = Zᵀy 를 한 번에 풉니다. 평균·표준편차는 학습 행에서만 구합니다.",
        "settings": [["차수", "2"], ["항 수", f"{len(poly.coef_)} (= 1 + {k} + {k * (k + 1) // 2})"],
                     ["릿지 α", f"{poly.alpha:g}"], ["학습 행", f"{n:,}"]],
        "extrapolation": "입력이 학습 범위를 넘으면 2차 곡면을 그대로 연장합니다. 학습 구간의 "
                         "곡률이 계속된다고 가정하므로 멀리 갈수록 과대·과소 예측이 커집니다.",
    }))
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, min_samples_leaf=20,
                                           l2_regularization=1.0, random_state=0)
        gb.fit(Xtr[ok], ytr[ok])
        early = bool(getattr(gb, "do_early_stopping_", False))
        models.append(("ML 부스팅", "boost", gb.predict, {
            "algorithm": "그래디언트 부스팅 회귀 트리 (scikit-learn HistGradientBoostingRegressor)",
            "formula": "ŷ = F_{0} + Σ_{m=1}^{M} ν · f_{m}(x),   f_{m} = 앞 단계 잔차(제곱오차 기울기)에 맞춘 회귀 트리",
            "fit": "트리를 하나씩 더하며 남은 오차를 줄입니다. 각 입력은 학습 행의 분위수로 최대 "
                   f"{gb.max_bins}개 구간으로 나눠 분기점을 찾습니다.",
            "settings": [["트리 수 M", f"{gb.n_iter_} (최대 {gb.max_iter})"],
                         ["학습률 ν", f"{gb.learning_rate:g}"],
                         ["트리당 잎 수", f"최대 {gb.max_leaf_nodes}"],
                         ["잎당 최소 행", f"{gb.min_samples_leaf}"],
                         ["L2 정칙화", f"{gb.l2_regularization:g}"],
                         ["조기 종료", (f"학습 행 안의 무작위 {gb.validation_fraction:.0%} 로 판단, "
                                    f"{gb.n_iter_no_change}회 개선 없으면 멈춤 (예측 기간은 쓰지 않음)")
                          if early else "사용 안 함"],
                         ["학습 행", f"{n:,}"]],
            "extrapolation": "트리는 학습 때 본 값 사이에서만 분기합니다. 입력이 학습 범위를 넘으면 "
                             "가장 바깥 구간의 값을 그대로 내므로 예측이 평평해집니다.",
        }))
    except ImportError:
        pass
    return models


def _predict_nan(fn, X):
    out = np.full(len(X), np.nan)
    ok = np.all(np.isfinite(X), axis=1)
    if ok.any():
        out[ok] = fn(X[ok])
    return out


def _segments(system) -> list[list[str]]:
    """연결을 따라 계통 흐름을 끊지 않고 이어지는 구간들로 나눈다 (합류·분기에서 끊음)."""
    succ: dict[str, list[str]] = {}
    pred: dict[str, list[str]] = {}
    for a, b in system.connections:
        ca, pa = a.split(".", 1)
        cb = b.split(".", 1)[0]
        role = system.components[ca].port_specs()[pa].role
        src, dst = (ca, cb) if role == "out" else (cb, ca)
        succ.setdefault(src, []).append(dst)
        pred.setdefault(dst, []).append(src)
    names = list(system.components)
    starts = [n for n in names if len(pred.get(n, [])) != 1
              or len(succ.get(pred[n][0], [])) != 1]
    starts.sort(key=lambda n: len(pred.get(n, [])) > 0)      # 발생원(들어오는 것 없음)부터
    segs = []
    for s0 in starts:
        seg, cur = [s0], s0
        while len(succ.get(cur, [])) == 1:
            nxt = succ[cur][0]
            seg.append(nxt)
            if len(pred.get(nxt, [])) != 1 or nxt in starts:
                break
            cur = nxt
        segs.append(seg)
    return segs


def explain_physics(system, model, cfg: EasyConfig, fitted_table: list[dict],
                    rows: int, n_obs: int) -> dict:
    """결과 화면용 물리모델 설명: 흐름, 컴포넌트별 지배방정식, 입력·예측값이 들어가는 식."""
    from .lib.explain import KIND_NOTE, laws_of, title_of

    fitted = {r["parameter"]: r for r in fitted_table}
    segments = _segments(system)
    order = list(dict.fromkeys([n for seg in segments for n in seg] + list(system.components)))
    groups: dict[str, dict] = {}
    where: dict[str, str] = {}                      # 인스턴스 -> 그룹 키 (흐름 순서)
    for name in order:
        comp = system.components[name]
        key = type(comp).__name__ if getattr(type(comp), "LAWS", None) else f"decl:{name}"
        g = groups.get(key)
        if g is None:
            g = groups[key] = {"title": title_of(comp), "instances": [], "fitted": [],
                               "laws": [{"kind": law.kind, "title": law.title, "formula": law.formula,
                                         "params": list(law.params)} for law in laws_of(comp)]}
        g["instances"].append(name)
        where[name] = key
    for full, r in fitted.items():
        inst, short = full.split(".", 1)
        if inst in where:
            groups[where[inst]]["fitted"].append({**r, "instance": inst, "param": short})

    def laws_with(full: str) -> list[str]:
        inst, short = full.split(".", 1)
        g = groups.get(where.get(inst, ""))
        return [law["title"] for law in g["laws"] if short in law["params"]] if g else []

    inputs = []
    for col in cfg.features:
        m = cfg.feature_map.get(col)
        if not (m and m.targets):
            continue
        uses = []
        for full in m.targets:
            inst = full.split(".", 1)[0]
            uses.append({"target": full, "component": groups[where[inst]]["title"] if inst in where else "",
                         "laws": laws_with(full)})
        inputs.append({"column": col, "scale": m.scale, "unit": m.unit, "uses": uses})

    y_var = cfg.target_map.targets[0] if cfg.target_map and cfg.target_map.targets else None
    obs_desc = declared_observables(model)
    return {
        "counts": {"equations": len(model.equations), "unknowns": int(model.n_vars),
                   "components": len(system.components)},
        "segments": segments,
        "groups": list(groups.values()),
        "inputs": inputs,
        "target": {"column": cfg.target, "variable": y_var,
                   "desc": obs_desc.get(y_var, "") if y_var else "",
                   "component": groups[where[y_var.split(".")[0]]]["title"]
                   if y_var and y_var.split(".")[0] in where else ""},
        "kinds": KIND_NOTE,
        "connection": "연결점에서는 압력이 같고, 질량유량의 합이 0 이며, 온도·조성이 그대로 전달됩니다.",
        "solve": (f"예측 행마다(5분 간격이면 5분마다) 방정식 {len(model.equations)}개를 연립해 정상상태를 "
                  "풉니다. 구조 해석(BLT)으로 작은 블록으로 나눈 뒤 블록별 뉴턴법을 쓰고, 직전 행의 해에서 "
                  "출발합니다. 과거 데이터를 보고 답을 고르는 것이 아니라 그 시점의 입력으로 식을 풉니다."),
        "calibrate": (f"보정은 학습 기간에서 고른 {rows}개 행(1시간 평균, 드문 운전상태 포함)과 관측 "
                      f"{n_obs}개로 합니다. 목적함수는 (예측 − 실측)/계측 불확도의 제곱합이고, 설계값 쪽으로 "
                      "약하게 당기는 항(가중 0.05)을 더합니다. 파라미터는 설계값의 0.2~5배 안에서만 움직입니다. "
                      "보정하는 것은 구성방정식의 물리 계수뿐이고, 보존법칙은 건드리지 않습니다."),
    }


def run_easy(cfg: EasyConfig, root: str | Path = ".",
             progress: Callable[[str], None] | None = None) -> dict:
    from .calib import CalibrationSpec, calibrate
    from .calib.design import select_rows
    from .scenario import load_model
    from .workflow import WorkflowConfig, _predict, load_dataset

    t0 = time.perf_counter()
    say = progress or (lambda m: None)
    errs = period_errors(cfg.train, cfg.test, cfg.embargo_days)
    if errs:
        raise ValueError(" / ".join(e.replace("**", "") for e in errs))
    physics = bool(cfg.model) and cfg.target_map is not None and bool(cfg.target_map.targets) \
        and any(m.targets for m in cfg.feature_map.values())
    root = Path(root)
    out_dir = root / "out" / "easy"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _slug(cfg.name or Path(cfg.csv).stem)

    # 1) 데이터 — 기간을 먼저 정하고 정제는 구간마다 따로 (load_dataset 이 한다)
    say("데이터 정리 중 (학습·간격·예측 구간을 따로 정제)")
    tm = _tagmap(cfg, physics)
    tm_path = out_dir / f"{stem}.tagmap.yaml"
    tm.dump(tm_path)
    a, b, c, d = period_bounds(cfg.train, cfg.test)
    wcfg = WorkflowConfig(
        name=cfg.name, csv=str(cfg.csv), tagmap=str(tm_path),
        train_query=f"index >= '{a}' and index < '{b}'",
        test_query=f"index >= '{c}' and index < '{d}'",
        params=list(cfg.params), max_rows=cfg.max_rows,
        model=({"yaml": cfg.model} if str(cfg.model).endswith((".yaml", ".yml")) else
               {"python": cfg.model}) if physics else None,
    )
    tmap, df, train, test, ic, oc, report = load_dataset(wcfg, return_report=True)
    y_col = next(e.primary for e in tmap.observations if e.tag == cfg.target)
    y_unit = next(e.unit for e in tmap.observations if e.tag == cfg.target)
    show_unit = cfg.target_unit if not physics else y_unit
    x_cols, x_tag = [], {}
    for col in cfg.features:
        e = next((e for e in tmap.inputs + tmap.observations if e.tag == col), None)
        if e is not None and e.primary in df.columns:
            x_cols.append(e.primary)
            x_tag[e.primary] = col
    model_inputs = [c2 for c2 in ic if c2 in x_cols]
    model_obs = [e.primary for e in tmap.observations
                 if not e.primary.startswith(("x::", "y::"))]
    train = train[train[y_col].notna()]
    test = test[test[y_col].notna()]
    if len(train) < 50 or len(test) < 10:
        raise ValueError(f"정리 후 남은 행이 부족합니다 (학습 {len(train)}, 예측 {len(test)}). "
                         "기간을 넓히거나 입력 컬럼의 결측을 확인하세요.")
    split = check_split(train, test, x_cols)

    conv = (lambda v: from_si(v, y_unit)) if physics else (lambda v: v)
    to_disp = lambda arr: np.array([conv(v) for v in np.asarray(arr, float)])  # noqa: E731
    y_te = to_disp(test[y_col])
    step_tr = max(1, len(train) // 1500)
    tr_sub = train.iloc[::step_tr]
    y_trs = to_disp(tr_sub[y_col])
    outside = split.outside if split.outside is not None else np.zeros(len(test), bool)

    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}    # 이름 -> (학습 표본, 예측 전체)
    kinds: dict[str, str] = {}
    cal_payload = None
    physics_explain = None
    physics_failed = 0
    band = None
    notes: list[str] = []

    # 2) 물리모델
    if physics:
        say("물리모델 조립 중")
        system = load_model(wcfg.model)
        model = system.compile()
        model.build()
        params = [p for p in (cfg.params or suggest_params(model, cfg.target_map.targets[0]))
                  if _has_param(model, p)]
        if not params:
            raise ValueError("보정할 파라미터가 없습니다. 모델에 보정 대상을 고르세요.")
        rows = select_rows(train, cfg.max_rows, model_inputs, model_obs)
        spec = CalibrationSpec(params=params, observations=model_obs, sigmas=tmap.sigmas(),
                               max_rows=max(len(rows), 1), prior_weight=0.05)
        say(f"물리 파라미터 {len(params)}개 보정 중 ({len(rows)}행, 드문 운전상태 포함) — 1~3분")
        cal = calibrate(model, rows[model_inputs], rows[model_obs], spec,
                        expansion=tmap.expansion(), verbose=False)
        say(f"물리모델로 예측 중 (예측 {len(test):,}행)")
        p_te = _predict(model, test, model_inputs, [y_col], tmap.expansion()).values[y_col].to_numpy()
        p_tr = _predict(model, tr_sub, model_inputs, [y_col], tmap.expansion()).values[y_col].to_numpy()
        physics_failed = int(np.isnan(p_te).sum())
        preds["물리모델"] = (to_disp(p_tr), to_disp(p_te))
        kinds["물리모델"] = "physics"
        t = cal.table()
        cal_payload = {
            "rows": cal.n_rows, "excluded": cal.n_excluded,
            "table": [{"parameter": r.parameter, "unit": r.unit, "initial": _f(r.initial),
                       "fitted": _f(r.fitted), "change_pct": _f(r["change_%"]),
                       "rel_stderr_pct": _f(r["rel_stderr_%"])} for _, r in t.iterrows()],
            "warnings": cal.identifiability_warnings(),
        }
        physics_explain = explain_physics(system, model, cfg, cal_payload["table"], rows=cal.n_rows,
                                          n_obs=len(model_obs))
        if cfg.band:
            say("상태 변동 폭 계산 중 (학습 기간을 네 토막으로 다시 보정)")
            from .calib.design import parameter_drift, state_band
            drift = parameter_drift(model, train, model_inputs, model_obs, spec, tmap.expansion())
            vals = [v for _, v in state_band(model, drift, test[model_inputs], y_col, y_unit,
                                             tmap.expansion())]
            if vals:
                band = {"lo": min(vals), "hi": max(vals), "values": vals,
                        "measured_mean": float(np.nanmean(y_te)),
                        "drift": [{"name": dd.name, "trend": dd.trend, "unit": dd.unit,
                                   "values": [float(v) for v in dd.values]} for dd in drift]}
        unmapped = [c2 for c2 in cfg.features if not (cfg.feature_map.get(c2) and cfg.feature_map[c2].targets)]
        if unmapped:
            notes.append("물리모델과 연결하지 않은 입력은 ML 만 씁니다: " + ", ".join(unmapped))
    else:
        notes.append("물리모델 없이 ML 만 비교했습니다. 물리모델을 쓰려면 모델을 고르고 타깃·입력을 "
                     "모델 변수에 연결하세요 — 계통 구조(토폴로지)는 데이터에서 유도할 수 없습니다.")

    # 3) ML — 같은 학습 행, 같은 입력
    # 학습 동안 한 번도 변하지 않은 입력(증설 전 장비 대수 등)은 ML 이 원리적으로 못 본다
    frozen = []
    for xc in x_cols:
        tr_v, te_v = train[xc].dropna(), test[xc].dropna()
        if len(tr_v) and len(te_v) and tr_v.nunique() == 1 and \
                not np.allclose(te_v.to_numpy(float), float(tr_v.iloc[0])):
            frozen.append(x_tag[xc])
    if frozen:
        notes.append("학습 기간에 한 번도 변하지 않았는데 예측 기간에 바뀐 입력: " + ", ".join(frozen)
                     + ". ML 은 이 변화의 효과를 배울 방법이 없어 무시합니다"
                     + (" — 물리모델은 지배방정식으로 반영합니다." if physics else "."))

    say("ML(다항·부스팅) 학습 중")
    Xtr = train[x_cols].to_numpy(dtype=float)
    ml_explain = []
    for name, kind, fn, info in _ml_models(Xtr, train[y_col].to_numpy(dtype=float)):
        preds[name] = (to_disp(_predict_nan(fn, tr_sub[x_cols].to_numpy(dtype=float))),
                       to_disp(_predict_nan(fn, test[x_cols].to_numpy(dtype=float))))
        kinds[name] = kind
        ml_explain.append({"name": name, "kind": kind, **info,
                           "inputs": [x_tag[c] for c in x_cols],
                           "frozen": [c for c in frozen]})

    # 4) 채점
    say("채점·정리 중")
    models = []
    for name, (ptr, pte) in preds.items():
        m = {"train": _metrics(ptr, y_trs), "test": _metrics(pte, y_te),
             "outside": _metrics(np.where(outside, pte, np.nan), y_te) if outside.any() else None,
             "inside": _metrics(np.where(~outside, pte, np.nan), y_te) if (~outside).any() else None,
             "predicted_pct": float(np.isfinite(pte).mean() * 100)}
        if cfg.limit is not None:
            hr = pd.DataFrame({"p": pte, "t": y_te}, index=test.index).resample("1h").mean()
            hr = hr[hr["t"].notna()]
            ex, over = hr["t"] > cfg.limit, (hr["p"] > cfg.limit).fillna(False)
            m["exceed"] = {"hours": int(ex.sum()), "hit": int((ex & over).sum()),
                           "false_alarm": int((~ex & over).sum())}
        models.append({"name": name, "kind": kinds[name], "metrics": m})

    # 5) 차트용 시계열 (학습 표본 + 예측 전체를 합쳐 다운샘플)
    t_all = list(tr_sub.index) + list(test.index)
    reg = ["train"] * len(tr_sub) + ["test"] * len(test)
    y_all = np.r_[y_trs, y_te]
    out_all = np.r_[np.zeros(len(tr_sub), bool), outside]
    k = max(1, len(t_all) // MAX_POINTS)
    sel = np.arange(0, len(t_all), k)
    series = {"t": [str(t_all[i]) for i in sel], "region": [reg[i] for i in sel],
              "y": [_f(y_all[i]) for i in sel], "outside": [bool(out_all[i]) for i in sel],
              "models": {name: [_f(v) for v in np.r_[ptr, pte][sel]]
                         for name, (ptr, pte) in preds.items()}}

    # 6) 파일 — 예측 결과 CSV, 다시 돌릴 수 있는 설정
    pred_df = pd.DataFrame({"실측": y_te, "학습범위밖": outside}, index=test.index)
    for name, (_, pte) in preds.items():
        pred_df[name] = pte
    pred_df.index.name = "timestamp"
    pred_path = out_dir / f"{stem}.predictions.csv"
    pred_df.to_csv(pred_path, encoding="utf-8-sig")
    cfg_path = None
    if physics:
        cfg_path = out_dir / f"{stem}.calibration.yaml"
        _write_calibration_yaml(cfg_path, cfg, wcfg, tm_path, params, x_cols, y_col)

    headline = _headline(models, split, show_unit, physics, physics_failed)
    rel = lambda p: str(Path(p).resolve().relative_to(root.resolve())) if p else None  # noqa: E731
    return {
        "mode": "physics" if physics else "ml",
        "target": {"column": cfg.target, "unit": show_unit},
        "split": {**split.to_dict(), "outside": None,
                  "train_start": str(train.index.min()), "train_end": str(train.index.max()),
                  "test_start": str(test.index.min()), "test_end": str(test.index.max())},
        "models": models, "series": series, "calibration": cal_payload,
        "explain": {"physics": physics_explain, "ml": ml_explain},
        "physics_failed": physics_failed, "band": band, "notes": notes,
        "data_log": report.lines() if report is not None else [],
        "headline": headline, "limit": cfg.limit,
        "files": {"predictions": rel(pred_path), "tagmap": rel(tm_path), "config": rel(cfg_path)},
        "seconds": time.perf_counter() - t0,
    }


def _f(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _headline(models, split, unit, physics, failed) -> list[str]:
    u = f" {unit}" if unit else ""
    out = []
    key = "outside" if split.is_extrapolation else "test"
    scored = [(m["name"], (m["metrics"].get(key) or m["metrics"]["test"] or {}).get("rmse"))
              for m in models]
    scored = [(n, r) for n, r in scored if r is not None]
    where = "학습 운전영역 밖 행" if key == "outside" else "예측 기간 전체"
    if scored:
        best = min(scored, key=lambda t: t[1])
        out.append(f"**{where}에서 가장 정확한 모델: {best[0]}** (RMSE {best[1]:.3g}{u}). "
                   + " · ".join(f"{n} {r:.3g}" for n, r in scored))
    if not split.is_future:
        out.append("**미래 예측이 아닙니다** — 예측 기간이 학습과 시간상 섞였습니다.")
    if split.extrapolation == split.extrapolation:     # nan 이 아니면
        pct = split.extrapolation * 100
        if split.is_extrapolation:
            out.append(f"예측 기간의 {pct:.0f}% 가 학습 때 없던 운전조건입니다 — "
                       + ("물리모델과 ML 의 차이가 드러나는 조건입니다." if physics else
                          "ML 이 학습 범위 밖으로 얼마나 버티는지 보는 조건입니다."))
        else:
            out.append(f"예측 기간의 {pct:.0f}% 만 학습 범위 밖입니다. 학습 범위 안에서는 ML 도 잘 "
                       "맞히므로 차이가 작게 나올 수 있습니다. 운전조건이 달라진 기간을 고르면 차이가 "
                       "드러납니다.")
    if physics and failed:
        out.append(f"**물리모델이 {failed:,}행에서 수렴하지 못했습니다.** 표는 푼 행만으로 계산됩니다.")
    return out


def _write_calibration_yaml(path, cfg: EasyConfig, wcfg, tm_path, params, x_cols, y_col) -> None:
    """같은 설정을 CLI(pf calibrate / pf improve)로 다시 돌릴 수 있게 남긴다."""
    import yaml
    model_path = cfg.model
    d = {
        "name": cfg.name,
        "model": {"yaml": model_path} if str(model_path).endswith((".yaml", ".yml"))
        else {"python": model_path, "builder": "build"},
        "data": {"csv": str(Path(cfg.csv).resolve()), "tagmap": str(Path(tm_path).resolve())},
        "split": {"train": wcfg.train_query, "test": wcfg.test_query},
        "calibrate": {"params": list(params), "max_rows": cfg.max_rows, "sampling": "space_filling"},
        "baseline": {"features": [c for c in x_cols if not c.startswith("x::")], "target": y_col,
                     "degree": 2},
    }
    Path(path).write_text("# 앱의 간편 예측에서 만든 설정 — pf calibrate / pf improve 로 다시 돌릴 수 있다\n"
                          + yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")


def config_from_body(body: dict, resolve: Callable[[str], Path]) -> EasyConfig:
    """앱 요청 본문 -> 설정. 경로는 작업 폴더 안으로 제한한다 (``resolve``)."""
    fm = {c: ColumnMap.from_dict(c, (body.get("feature_map") or {}).get(c))
          for c in body.get("features") or []}
    tmap = body.get("target_map")
    return EasyConfig(
        csv=str(resolve(body["csv"])), target=body["target"],
        features=list(body.get("features") or []),
        train=tuple(body["train"]), test=tuple(body["test"]),
        embargo_days=float(body.get("embargo_days", 1.0)),
        time_column=body.get("time_column"), target_unit=body.get("target_unit") or "",
        model=str(resolve(body["model"])) if body.get("model") else None,
        target_map=ColumnMap.from_dict(body["target"], tmap) if tmap else None,
        feature_map=fm,
        extra_obs=[ColumnMap.from_dict(e["column"], e) for e in body.get("extra_obs") or []],
        params=list(body.get("params") or []),
        max_rows=int(body.get("max_rows", 150)), band=bool(body.get("band", False)),
        limit=float(body["limit"]) if body.get("limit") not in (None, "") else None,
        name=body.get("name") or Path(body["csv"]).stem,
    )

