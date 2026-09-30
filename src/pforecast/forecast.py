"""미래 예측 — 실측이 없는 계획(생산계획·설정값·예보)을 넣고 예측한다.

검증(``easy.run_easy``)은 과거 기간을 되짚어 채점한다. 여기는 **정답이 없는 미래**다. 그래서
채점 대신 세 가지를 같이 낸다.

* **설비 상태 변동 폭** — 학습 기간을 네 토막으로 다시 보정해 토막마다 계획을 풀어 본 폭.
  예측 구간은 파라미터 공분산이 아니라 이 폭으로 말한다 (CLAUDE.md, ``state_paths``).
* **학습 범위 밖 표시** — 계획 행 중 학습 때 없던 운전조건. ML 은 여기서 믿기 어렵다
  (부스팅은 평평해지고 다항식은 곡면을 연장한다). 학습 동안 고정이던 입력이 계획에서
  바뀌면 따로 알린다.
* **관리기준 초과 시간** — 모델별, 그리고 상태 변동 폭 기준 '가능성'과 '확실'.

학습 데이터는 기본으로 **전체 기간**을 쓴다. 좁은 최근 구간만으로 다시 맞추면 운전 다양성이
없어 귀속이 바뀌고 예측이 틀어진다 (docs/improvement_log.md).

계획 파일은 학습 CSV 와 **같은 컬럼 이름·단위**여야 한다. 입력(x)만 있으면 되고 y 는 없어도
된다. 변환은 학습 때와 같은 태그맵(``load_frames``)으로 한다.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .core.units import from_si
from .data.split import check_split


# ---------------------------------------------------------------------------
# 계획 만들기·읽기·검사
# ---------------------------------------------------------------------------

def _history_table(csv: str | Path, time_column: str | None, features: list[str]) -> pd.DataFrame:
    """학습 CSV 의 입력 컬럼 (파일 단위, 값 이상은 결측으로 정리)."""
    from .data.ingest import read_table
    from .data.quality import apply, assess

    tab = read_table(csv, time_column=time_column)
    cols = [c for c in features if c in tab.df.columns]
    df = tab.df[cols]
    return apply(df, assess(df, cols))


def make_plan(csv: str | Path, time_column: str | None, features: list[str], base_days: float = 7.0,
              horizon_days: float = 14.0, step: str = "1h", adjust: list[dict] | None = None) -> pd.DataFrame:
    """최근 ``base_days`` 의 입력 패턴을 ``horizon_days`` 동안 반복하고 조정을 적용한 계획.

    ``adjust``: ``[{"column": 이름, "mode": "scale"|"add"|"set", "value": 수}]``. 값은 파일 단위.
    빈 시간(결측)은 채우지 않는다 — 그 시점 예측이 비어 있을 뿐, 값을 지어내지 않는다.
    """
    hist = _history_table(csv, time_column, features)
    if hist.empty:
        raise ValueError("학습 CSV 에서 입력 컬럼을 찾지 못했습니다.")
    end = hist.index.max()
    base = hist[hist.index > end - pd.Timedelta(days=base_days)].resample(step).mean()
    if len(base) == 0:
        raise ValueError("최근 구간에 데이터가 없습니다.")
    dt = pd.Timedelta(step)
    n = int(round(pd.Timedelta(days=horizon_days) / dt))
    idx = pd.date_range(end.floor(step) + dt, periods=n, freq=dt)
    reps = np.resize(np.arange(len(base)), n)
    plan = pd.DataFrame(base.to_numpy()[reps], index=idx, columns=base.columns)
    for a in adjust or []:
        col, mode, v = a.get("column"), a.get("mode", "keep"), a.get("value")
        if col not in plan.columns or mode in ("keep", None) or v in (None, ""):
            continue
        v = float(v)
        if mode == "scale":
            plan[col] = plan[col] * v
        elif mode == "add":
            plan[col] = plan[col] + v
        elif mode == "set":
            plan[col] = v
        else:
            raise ValueError(f"알 수 없는 조정 방식: {mode!r} (scale | add | set)")
    plan.index.name = time_column or "timestamp"
    return plan


def save_plan(plan: pd.DataFrame, path: str | Path) -> tuple[Path, str | None]:
    """엑셀에서 열어 고칠 수 있게 UTF-8(BOM) CSV 로. 같은 이름 파일이 엑셀에 열려 있으면(윈도우 잠금)
    새 이름으로 쓰고 알림을 돌려준다 — (실제 경로, 알림 또는 None)."""
    from .fileio import write_locked_ok
    return write_locked_ok(path, lambda p: plan.to_csv(p, encoding="utf-8-sig", float_format="%.6g"))


def read_plan(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """계획 CSV. 현장 CSV 와 같은 수집기로 읽는다 (CP949·여러 줄 헤더·날짜 표기)."""
    from .data.ingest import read_table

    tab = read_table(path)
    return tab.df, tab.report.lines()


def check_plan(plan: pd.DataFrame, features: list[str], history: pd.DataFrame) -> dict:
    """계획이 쓸 만한지: 빠진 컬럼, 결측, 과거 시점, 학습 범위 밖 (파일 단위로)."""
    missing = [c for c in features if c not in plan.columns]
    have = [c for c in features if c in plan.columns]
    errors, warnings = [], []
    if missing:
        errors.append("계획에 없는 입력 컬럼: " + ", ".join(missing)
                      + " — 학습 CSV 와 같은 컬럼 이름으로 넣으세요.")
    if len(plan) == 0:
        errors.append("계획에 행이 없습니다.")
        return {"ok": False, "errors": errors, "warnings": warnings}
    step = plan.index.to_series().diff().median()
    nan_rows = int(plan[have].isna().any(axis=1).sum()) if have else 0
    end = history.index.max()
    past = int((plan.index <= end).sum())
    if past:
        warnings.append(f"계획 {past:,}행이 학습 데이터 끝({str(end)[:16]}) 이전 시점입니다. 과거 검증은 "
                        "5단계(검증 결과)에서 하세요 — 여기 결과는 채점하지 않습니다.")
    if nan_rows:
        warnings.append(f"입력이 빈 계획 행 {nan_rows:,}개는 예측하지 않습니다 (값을 지어내지 않음).")
    ranges, outside_any = [], np.zeros(len(plan), bool)
    for c in have:
        h = history[c].dropna() if c in history else pd.Series(dtype=float)
        v = plan[c].to_numpy(dtype=float)
        if len(h) == 0:
            continue
        lo, hi = float(h.min()), float(h.max())
        out = (v < lo - 1e-9 * abs(lo)) | (v > hi + 1e-9 * abs(hi))
        outside_any |= out & np.isfinite(v)
        frozen = h.nunique() == 1
        ranges.append({"column": c, "train_lo": lo, "train_hi": hi,
                       "test_lo": float(np.nanmin(v)) if np.isfinite(v).any() else None,
                       "test_hi": float(np.nanmax(v)) if np.isfinite(v).any() else None,
                       "outside_pct": float(100 * out.mean()), "frozen": bool(frozen)})
        if frozen and out.any():
            warnings.append(f"{c} 는 학습 동안 {lo:g} 로 고정이었는데 계획에서 바뀝니다. ML 은 이 변화의 효과를 "
                            "배울 방법이 없습니다 — 물리모델 예측을 보세요.")
    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "n_rows": int(len(plan)), "start": str(plan.index.min()), "end": str(plan.index.max()),
            "step_minutes": float(step / pd.Timedelta(minutes=1)) if pd.notna(step) else None,
            "nan_rows": nan_rows, "past_rows": past, "history_end": str(end),
            "outside_pct": float(100 * outside_any.mean()), "ranges": ranges}


# ---------------------------------------------------------------------------
# 예측
# ---------------------------------------------------------------------------

def forecast_easy(cfg, plan: str | Path | pd.DataFrame, root: str | Path = ".",
                  history: tuple[str, str] | None = None, band: bool = True,
                  progress: Callable[[str], None] | None = None) -> dict:
    """``cfg`` (간편 예측 설정)의 모델·연결로 과거 전체를 학습하고 계획을 예측한다.

    ``history``: 학습에 쓸 과거 (날짜, 양끝 포함). ``None`` 이면 데이터 전체.
    """
    from .easy import (_bound, _cal_payload, _f, _fit_physics, _ml_models, _predict_nan, _slug,
                       _tagmap, explain_physics)
    from .workflow import WorkflowConfig, _predict, load_dataset

    t0 = time.perf_counter()
    say = progress or (lambda m: None)
    root = Path(root)
    physics = bool(cfg.model) and cfg.target_map is not None and bool(cfg.target_map.targets) \
        and any(m.targets for m in cfg.feature_map.values())
    out_dir = root / "out" / "forecast"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _slug(cfg.name or Path(cfg.csv).stem)

    # 1) 계획
    if isinstance(plan, pd.DataFrame):
        plan_raw, plan_log, plan_src = plan, [], "만든 계획"
    else:
        plan_raw, plan_log = read_plan(plan)
        plan_src = str(plan)

    # 2) 학습 데이터 (과거 전체가 기본)
    say("과거 데이터 정리 중")
    tm = _tagmap(cfg, physics)
    tm_path = out_dir / f"{stem}.tagmap.yaml"
    from .fileio import write_locked_ok as _wl
    tm_path, _ = _wl(tm_path, tm.dump)
    q = ""
    if history:
        q = f"index >= '{_bound(history[0], False)}' and index < '{_bound(history[1], True)}'"
    wcfg = WorkflowConfig(name=cfg.name, csv=str(cfg.csv), tagmap=str(tm_path), train_query=q,
                          test_query="", params=list(cfg.params), max_rows=cfg.max_rows,
                          model=({"yaml": cfg.model} if str(cfg.model).endswith((".yaml", ".yml")) else
                                 {"python": cfg.model}) if physics else None)
    # 과거 전체를 한 덩어리로 정제한다 (예측 기간은 이 파일에 없다)
    tmap, df, train, _all, ic, oc, report = load_dataset(wcfg, return_report=True)
    y_col = next(e.primary for e in tmap.observations if e.tag == cfg.target)
    y_unit = next(e.unit for e in tmap.observations if e.tag == cfg.target)
    show_unit = cfg.target_unit if not physics else y_unit
    x_cols, x_tag = [], {}
    for col in cfg.features:
        e = next((e for e in tmap.inputs + tmap.observations if e.tag == col), None)
        if e is not None and e.primary in df.columns:
            x_cols.append(e.primary)
            x_tag[e.primary] = col
    model_inputs = [c for c in ic if c in x_cols]
    model_obs = [e.primary for e in tmap.observations if not e.primary.startswith(("x::", "y::"))]
    train = train[train[y_col].notna()]
    if len(train) < 50:
        raise ValueError(f"학습에 쓸 행이 {len(train)}개뿐입니다. 과거 기간을 넓히세요.")
    conv = (lambda v: from_si(v, y_unit)) if physics else (lambda v: v)
    to_disp = lambda arr: np.array([conv(v) if np.isfinite(v) else np.nan for v in np.asarray(arr, float)])  # noqa: E731

    # 3) 계획 → 모델 좌표 (학습과 같은 태그맵 변환)
    # 계획에는 관측(y)이 없다 — 입력 컬럼의 태그맵 항목만 적용한다 (단위·배율을 학습과 똑같이)
    from .data.sources import _apply_entries
    missing = [x_tag[c] for c in x_cols if x_tag[c] not in plan_raw.columns]
    entries = [e for e in tmap.inputs + tmap.observations if e.primary in x_cols and e.tag in plan_raw.columns]
    pf = _apply_entries(plan_raw, entries, strict=True)
    if missing:
        raise ValueError("계획에 없는 입력 컬럼: " + ", ".join(missing))
    pf = pf[pf[x_cols].notna().all(axis=1)]
    if len(pf) == 0:
        raise ValueError("입력이 다 있는 계획 행이 없습니다.")

    notes: list[str] = []
    preds: dict[str, np.ndarray] = {}
    kinds: dict[str, str] = {}
    lo = hi = None
    cal_payload = physics_explain = None
    drift_payload = None

    # 4) 물리모델
    if physics:
        system, model, cal, spec, params = _fit_physics(cfg, wcfg, tmap, train, model_inputs, model_obs,
                                                        say, notes)
        say(f"물리모델로 계획 예측 중 ({len(pf):,}행)")
        p = _predict(model, pf, model_inputs, [y_col], tmap.expansion()).values[y_col].to_numpy()
        preds["물리모델"] = to_disp(p)
        kinds["물리모델"] = "physics"
        cal_payload = _cal_payload(cal)
        physics_explain = explain_physics(system, model, cfg, cal_payload["table"], rows=cal.n_rows,
                                          n_obs=len(model_obs))
        if band:
            say("설비 상태 변동 폭 계산 중 (과거를 네 토막으로 다시 보정)")
            from .calib.design import parameter_drift, state_paths
            drift = parameter_drift(model, train, model_inputs, model_obs, spec, tmap.expansion())
            paths = state_paths(model, drift, pf[model_inputs], y_col, y_unit, tmap.expansion())
            if paths:
                stack = np.vstack([v for _, v in paths] + [preds["물리모델"]])
                with np.errstate(all="ignore"):
                    lo, hi = np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)
                drift_payload = [{"name": d.name, "unit": d.unit, "trend": d.trend,
                                  "values": [float(v) for v in d.values]} for d in drift]
        failed = int(np.isnan(preds["물리모델"]).sum())
        if failed:
            notes.append(f"물리모델이 계획 {failed:,}행에서 수렴하지 못했습니다 — 모델 적용 범위 밖일 수 있습니다.")

    # 5) ML — 같은 과거 행, 같은 입력
    say("ML(다항·부스팅) 학습·예측 중")
    ml_explain = []
    for name, kind, fn, info in _ml_models(train[x_cols].to_numpy(dtype=float), train[y_col].to_numpy(dtype=float)):
        preds[name] = to_disp(_predict_nan(fn, pf[x_cols].to_numpy(dtype=float)))
        kinds[name] = kind
        ml_explain.append({"name": name, "kind": kind, **info, "inputs": [x_tag[c] for c in x_cols], "frozen": []})

    # 6) 학습 범위 밖 (학습 때 없던 운전조건)
    chk = check_split(train, pf, x_cols)
    outside = chk.outside if chk.outside is not None else np.zeros(len(pf), bool)
    frozen = [x_tag[c] for c in x_cols if train[c].nunique() == 1
              and not np.allclose(pf[c].to_numpy(float), float(train[c].iloc[0]))]
    for m in ml_explain:
        m["frozen"] = frozen
    if outside.mean() > 0:
        notes.append(f"계획의 {outside.mean() * 100:.0f}% 가 학습 때 없던 운전조건입니다. 이 구간에서 ML 은 믿기 "
                     "어렵습니다 — 부스팅은 학습 때 본 값 근처에서 평평해지고, 다항식은 곡면을 그대로 연장합니다.")
    if frozen:
        notes.append("학습 동안 한 번도 변하지 않았는데 계획에서 바뀌는 입력: " + ", ".join(frozen)
                     + ". ML 은 이 변화를 반영하지 못합니다" + (" — 물리모델은 지배방정식으로 반영합니다." if physics else "."))

    # 7) 관리기준 초과 시간 (1시간 평균)
    idx = pf.index
    summary = []
    for name, v in preds.items():
        s = pd.Series(v, index=idx)
        row = {"name": name, "kind": kinds[name], "mean": _f(np.nanmean(v)), "max": _f(np.nanmax(v)),
               "p95": _f(np.nanpercentile(v[np.isfinite(v)], 95)) if np.isfinite(v).any() else None}
        if cfg.limit is not None:
            hr = s.resample("1h").mean().dropna()
            row["exceed_hours"] = int((hr > cfg.limit).sum())
            row["hours"] = int(len(hr))
        summary.append(row)
    band_summary = None
    if lo is not None:
        band_summary = {"mean_lo": _f(np.nanmean(lo)), "mean_hi": _f(np.nanmean(hi))}
        if cfg.limit is not None:
            hl = pd.DataFrame({"lo": lo, "hi": hi}, index=idx).resample("1h").mean().dropna()
            band_summary["exceed_possible_hours"] = int((hl["hi"] > cfg.limit).sum())
            band_summary["exceed_certain_hours"] = int((hl["lo"] > cfg.limit).sum())

    # 8) 차트용: 최근 과거 실측(1시간 평균) + 계획 예측
    tail_days = 14
    tail = train[train.index > train.index.max() - pd.Timedelta(days=tail_days)]
    tail_h = tail[[y_col]].resample("1h").mean().dropna()
    k = max(1, len(pf) // 3000)
    series = {"t": [str(t) for t in idx[::k]], "models": {n: [_f(x) for x in v[::k]] for n, v in preds.items()},
              "band_lo": [_f(x) for x in lo[::k]] if lo is not None else None,
              "band_hi": [_f(x) for x in hi[::k]] if hi is not None else None,
              "outside": [bool(x) for x in outside[::k]]}
    history_tail = {"t": [str(t) for t in tail_h.index], "y": [_f(x) for x in to_disp(tail_h[y_col])]}

    # 9) 파일
    out = pd.DataFrame({f"계획:{x_tag[c]}": plan_raw[x_tag[c]].reindex(idx).to_numpy()
                        if x_tag[c] in plan_raw.columns else np.nan for c in x_cols}, index=idx)
    for n, v in preds.items():
        out[n] = v
    if lo is not None:
        out["물리모델_상태폭_하한"], out["물리모델_상태폭_상한"] = lo, hi
    out["학습범위밖"] = outside
    out.index.name = "timestamp"
    from .fileio import rel_posix, write_locked_ok
    path, locked = write_locked_ok(out_dir / f"{stem}.forecast.csv", lambda p_: out.to_csv(p_, encoding="utf-8-sig"))
    if locked:
        notes.append(locked)
    rel = lambda p_: rel_posix(p_, root)  # noqa: E731 — 화면은 '/' 경로만 안다
    return {
        "mode": "physics" if physics else "ml",
        "target": {"column": cfg.target, "unit": show_unit},
        "history": {"start": str(train.index.min()), "end": str(train.index.max()), "n_rows": int(len(train))},
        "plan": {"source": plan_src, "start": str(idx.min()), "end": str(idx.max()), "n_rows": int(len(pf)),
                 "log": plan_log},
        "series": series, "history_tail": history_tail, "summary": summary, "band": band_summary,
        "drift": drift_payload, "outside_pct": float(outside.mean() * 100),
        "per_column": [{"column": x_tag[c], "outside_pct": float(chk.per_column.get(c, 0.0) * 100)} for c in x_cols],
        "notes": notes, "calibration": cal_payload, "limit": cfg.limit,
        "explain": {"physics": physics_explain, "ml": ml_explain},
        "data_log": report.lines() if report is not None else [],
        "files": {"forecast": rel(path)}, "seconds": time.perf_counter() - t0,
    }
