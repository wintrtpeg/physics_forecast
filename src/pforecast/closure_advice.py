"""구성방정식 후보 추천 — 데이터와 계통 특성으로 후보를 추리고 근거를 붙인다.

**고르는 것은 사람이다.** 이 모듈은 순위와 근거를 낼 뿐이고, 결과는 화면에서 사용자가
슬롯마다 최종 선택한다. LLM 없이 규칙과 보정·검증만 쓴다.

누수 규칙
---------
* **예측 기간은 받지 않는다.** 학습 기간만 받아서 다시 시간순으로 나눈다:
  앞부분(보정) → 간격(1일 이상) → 끝부분(검증). 후보는 검증 구간으로 고르고, 최종 성능은
  사용자가 정한 예측 기간으로 따로 잰다 (test 로 고르고 test 로 보고하면 선택 누수다).
* 값 정제는 이미 분할별로 끝난 학습 행을 받는다.

순서 (싼 것부터)
----------------
1. **계통 특성** — 계통에 있는 컴포넌트 종류가 슬롯을 정한다 (``lib/closures.py``).
2. **영향** — 슬롯의 현재 식 파라미터를 10% 바꿨을 때 예측값이 1% 도 안 변하면 비교하지
   않는다 ("영향 작음").
3. **가진(excitation)** — 후보를 가르는 변수(L/G, 가동률, 유량 ...)가 학습 데이터에서
   거의 변하지 않았으면 그 후보는 "구별 불가"다. 데이터가 흔들지 않은 방향은 어떤
   검증으로도 가릴 수 없다.
4. **관측 민감도** — 후보 파라미터를 10% 바꿔도 관측이 계측 불확도의 0.1배도 안 움직이면
   보정으로 정할 수 없다 ("구별 불가: 관측 둔감").
5. **잔차 진단** — 현재 식 잔차가 후보를 가르는 변수와 상관되어 있으면 근거로 적는다.
6. **보정 + 검증** — 남은 후보만 같은 행으로 보정하고 검증 구간에서 채점한다.

추천 규칙
---------
* 기본은 **현재 식 유지**다. 다른 후보는 검증 구간 오차가 2% 이상 낮고, 1차 자기상관으로
  깎은 유효 표본수 기준 2 표준오차 이상 차이가 날 때만 추천한다.
* 후보에만 있는 파라미터가 식별되지 않거나(상대표준오차 > 50%, |상관| > 0.95) 경계에
  붙으면 제외한다.
* 하한선(가장 단순한 형태)보다 현재 식이 낫다는 근거가 없으면 그렇다고 적는다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .calib import CalibrationSpec, build_param_rows, calibrate, simulate
from .calib.design import average_rows, select_rows
from .core.units import from_si
from .data.split import check_split
from .lib.closures import ROLE_LABEL, apply_closures, closure_slots
from .lib.explain import title_of
from .scenario import load_model
from .selection import _aicc, paired_abs_error


@dataclass
class AdviceConfig:
    max_rows: int = 60              # 보정 행 (1시간 평균, 드문 운전상태 포함)
    val_frac: float = 0.25          # 학습 기간 끝의 이 비율을 검증 구간으로
    embargo_days: float = 1.0
    min_relevance_pct: float = 1.0  # 예측값 변화가 이보다 작으면 "영향 작음"
    min_span_pct: float = 5.0       # 가르는 변수의 변화 폭이 이보다 작으면 "구별 불가"
    min_obs_sensitivity: float = 0.1   # σ 단위
    min_gain_pct: float = 2.0
    sample_rows: int = 16           # 선별 계산용 행 수
    slots: list[str] | None = None  # 특정 슬롯만 ("SCR.eta" ...)
    prior_weight: float = 0.05


def _var_values(model, P: np.ndarray, X: np.ndarray, inst: str, need: str) -> np.ndarray | None:
    """후보를 가르는 변수 값 (변수·포트변수는 풀이 결과에서, 파라미터는 입력 행에서)."""
    full = f"{inst}.{need}"
    try:
        return X[:, model.var_index(full)]
    except (KeyError, ValueError):
        pass
    try:
        return P[:, model.par_index(full)]
    except (KeyError, ValueError):
        return None


def _span_pct(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return 0.0
    lo, hi = np.percentile(v, [5, 95])
    mid = float(np.median(np.abs(v)))
    return float(100.0 * (hi - lo) / mid) if mid > 0 else (100.0 if hi > lo else 0.0)


def _solve_rows(model, P: np.ndarray) -> np.ndarray:
    from .calib.runner import solve_with_fallback
    X = np.full((len(P), model.n_vars), np.nan)
    x = model.x0()
    for i, p in enumerate(P):
        r = solve_with_fallback(model, p, x)
        if r.success:
            x = r.x
            X[i] = r.x
    return X


def advise_closures(model_spec, tmap, train: pd.DataFrame, inputs: list[str], obs: list[str],
                    target: str, common: list[str], current: dict[str, str] | None = None,
                    cfg: AdviceConfig | None = None,
                    progress: Callable[[str], None] | None = None) -> dict:
    """학습 기간 ``train`` 만으로 슬롯별 후보를 비교한다. 결과는 화면용 dict."""
    cfg = cfg or AdviceConfig()
    say = progress or (lambda m: None)
    t_start = time.perf_counter()
    current = dict(current or {})
    expansion = tmap.expansion()
    sigmas = tmap.sigmas(train)            # 불확도를 안 적은 관측은 학습 값 크기의 2%
    unit = next((e.unit for e in tmap.observations if e.primary == target), "1")
    disp = lambda v: np.array([from_si(float(x), unit) for x in np.asarray(v, float)])  # noqa: E731

    # 1) 학습 기간을 다시 시간순으로: 보정 | 간격 | 검증
    train = train[train[inputs].notna().all(axis=1)]
    t0, t1 = train.index.min(), train.index.max()
    v0 = t1 - (t1 - t0) * cfg.val_frac
    inner = train[train.index < v0 - pd.Timedelta(days=cfg.embargo_days)]
    val = train[train.index >= v0]
    inner_h = average_rows(inner, inputs, obs)
    val_h = average_rows(val, inputs, obs)
    val_h = val_h[val_h[target].notna()] if target in val_h else val_h.iloc[:0]
    if len(inner_h) < 20 or len(val_h) < 20:
        raise ValueError(f"학습 기간이 짧아 후보를 비교할 수 없습니다 (보정 {len(inner_h)}시간, "
                         f"검증 {len(val_h)}시간). 학습 기간을 늘리세요.")
    chk = check_split(inner_h, val_h, inputs)
    outside = chk.outside if chk.outside is not None else np.zeros(len(val_h), bool)
    use_out = outside.mean() >= 0.3 and outside.sum() >= 20
    rows = select_rows(inner, cfg.max_rows, inputs, obs)
    step = max(1, len(rows) // cfg.sample_rows)
    sample = rows.iloc[::step]
    notes = []
    if not use_out:
        notes.append(f"검증 구간의 {outside.mean() * 100:.0f}% 만 보정 구간의 운전영역 밖입니다. 순위는 외삽 "
                     "능력보다 보간 능력을 잽니다 — 참고로만 보세요.")

    models: dict[tuple, object] = {}
    sys0 = load_model(model_spec)
    defaults = {f"{inst}.{sl.key}": sl.default for inst, sl in closure_slots(sys0)}

    def norm(choices: dict[str, str]) -> tuple:
        """기본값과 같은 선택은 뺀다 — 같은 모델을 두 번 보정하지 않게."""
        return tuple(sorted((k, v) for k, v in choices.items() if defaults.get(k) != v))

    def build(choices: dict[str, str]):
        key = norm(choices)
        if key not in models:
            system = load_model(model_spec)
            apply_closures(system, choices)
            m = system.compile()
            m.build()
            models[key] = m
        return models[key]

    def rows_P(model, frame):
        return build_param_rows(model, frame[inputs], base_p=model.p0(), expansion=expansion)

    cals: dict[tuple, dict] = {}

    def evaluate(choices: dict[str, str], params: list[str], label: str) -> dict:
        key = (norm(choices), tuple(sorted(params)))
        if key in cals:
            return cals[key]
        t_c = time.perf_counter()
        m = build(choices)
        params = [p for p in params if _has(m, p)]
        spec = CalibrationSpec(params=params, observations=obs, sigmas=sigmas,
                               max_rows=max(len(rows), 1), prior_weight=cfg.prior_weight)
        try:
            cal = calibrate(m, rows[inputs], rows[obs], spec, expansion=expansion, verbose=False)
        except Exception as exc:  # noqa: BLE001 — 후보 하나가 못 풀려도 나머지는 본다
            cals[key] = {"error": f"보정 실패: {exc}", "seconds": time.perf_counter() - t_c}
            return cals[key]
        sim_v = simulate(m, rows_P(m, val_h), [target], index=val_h.index).values[target].to_numpy()
        sim_c = simulate(m, rows_P(m, rows), obs, index=rows.index).values
        pv, mv = disp(sim_v), disp(val_h[target])
        err = pv - mv
        sel = outside if use_out else np.ones(len(err), bool)
        ok = np.isfinite(err)
        # 검증 구간 앞·뒤 절반 (시간순) — 한쪽에서만 나은 후보는 기간에 따라 뒤집힐 수 있다
        idx = np.flatnonzero(sel & ok)
        halves = [float(np.sqrt(np.mean(err[h] ** 2))) if len(h) else np.nan
                  for h in (idx[: len(idx) // 2], idx[len(idx) // 2:])]
        res = {
            "pred": pv[sel], "meas": mv[sel], "cal": cal, "params": params, "halves": halves,
            "val_rmse": float(np.sqrt(np.nanmean(err[sel & ok] ** 2))) if (sel & ok).any() else np.nan,
            "val_rmse_all": float(np.sqrt(np.nanmean(err[ok] ** 2))) if ok.any() else np.nan,
            "val_bias": float(np.nanmean(err[sel & ok])) if (sel & ok).any() else np.nan,
            "val_failed": int((~ok).sum()),
            "aicc": _aicc(sim_c[obs], rows[obs], sigmas, len(params)),
            "seconds": time.perf_counter() - t_c,
        }
        cals[key] = res
        return res

    # 2) 기준: 현재 식 (설계값) — 선별 계산용
    base = build(current)
    P_s = rows_P(base, sample)
    X_s = _solve_rows(base, P_s)
    P_r = rows_P(base, rows)
    X_r = _solve_rows(base, P_r)
    y_idx = base.var_index(target) if _is_var(base, target) else None

    def target_on(model, P, X=None):
        if y_idx is not None and model is base and X is not None:
            return X[:, y_idx]
        return simulate(model, P, [target]).values[target].to_numpy()

    y_s = target_on(base, P_s, X_s)

    slots = [(inst, sl) for inst, sl in closure_slots(sys0)
             if not cfg.slots or f"{inst}.{sl.key}" in cfg.slots]
    out_slots = []

    # 3) 영향 (싼 선별)
    say("후보 선별 중 (영향·변화 폭·관측 민감도)")
    relevance = {}
    for inst, sl in slots:
        cur = sl.option(current.get(f"{inst}.{sl.key}", sl.default))
        rel = 0.0
        for p in cur.fit:
            j = _par_idx(base, f"{inst}.{p}")
            if j is None:
                continue
            P2 = P_s.copy()
            P2[:, j] *= 1.1
            y2 = target_on(base, P2)
            d = np.abs(y2 - y_s)
            ok = np.isfinite(d) & np.isfinite(y_s)
            if ok.any():
                rel = max(rel, float(100.0 * np.mean(d[ok]) / max(np.mean(np.abs(y_s[ok])), 1e-30)))
        relevance[(inst, sl.key)] = rel

    # 4) 기준 보정 (공통 파라미터, 현재 식) — 잔차 진단과 '현재 식' 후보에 같이 쓴다
    say(f"현재 식으로 기준 보정 중 ({len(rows)}행)")
    ref = evaluate(current, list(common), "현재 식")
    resid = None
    if "cal" in ref:
        m = build(current)
        sim_r = simulate(m, rows_P(m, rows), [target], index=rows.index).values[target].to_numpy()
        resid = sim_r - rows[target].to_numpy(dtype=float)

    # 평가할 후보 목록 먼저 정하고 (진행률), 그다음 보정
    plan = []
    for inst, sl in slots:
        skey = f"{inst}.{sl.key}"
        cur_id = current.get(skey, sl.default)
        comp_title = title_of(sys0.components[inst])
        entry = {"key": skey, "instance": inst, "slot": sl.key, "title": sl.title, "component": comp_title,
                 "current": cur_id, "default": sl.default, "recommended": cur_id,
                 "relevance_pct": relevance[(inst, sl.key)], "excitation": [], "residual_corr": [],
                 "status": "비교함", "message": "", "candidates": []}
        needs = sorted({n for o in sl.options for n in o.needs})
        spans = {}
        for n in needs:
            v = _var_values(base, P_r, X_r, inst, n)
            if v is None:
                continue
            spans[n] = _span_pct(v)
            vv = v[np.isfinite(v)]
            entry["excitation"].append({"var": f"{inst}.{n}", "lo": float(np.min(vv)) if len(vv) else None,
                                        "hi": float(np.max(vv)) if len(vv) else None,
                                        "span_pct": spans[n], "ok": spans[n] >= cfg.min_span_pct})
            if resid is not None:
                ok = np.isfinite(resid) & np.isfinite(v)
                if ok.sum() > 10 and np.std(v[ok]) > 0 and np.std(resid[ok]) > 0:
                    entry["residual_corr"].append({"var": f"{inst}.{n}",
                                                   "r": float(np.corrcoef(resid[ok], v[ok])[0, 1])})
        slot_params = {f"{inst}.{p}" for o in sl.options for p in o.params}
        cur_opt = sl.option(cur_id)
        for o in sl.options:
            if o.id == cur_id:
                # 현재 식은 사용자가 정한 보정 그대로 (기준 보정을 그대로 쓴다)
                fit = [p.split(".", 1)[1] for p in common if p in slot_params]
                new = []
            else:
                # 현재 식도 보정하던 파라미터는 사용자 선택을 따르고, 이 후보가 새로 푸는 것만 더한다
                fit = [p for p in o.fit if p not in cur_opt.fit or f"{inst}.{p}" in common]
                new = [p for p in fit if p not in cur_opt.fit]
            c = {"id": o.id, "title": o.title, "role": o.role, "role_label": ROLE_LABEL.get(o.role, ""),
                 "formula": o.law.formula, "fit": [f"{inst}.{p}" for p in fit],
                 "new_params": [f"{inst}.{p}" for p in new], "note": o.note,
                 "needs": [f"{inst}.{n}" for n in o.needs], "status": "", "reasons": [], "evaluated": False}
            entry["candidates"].append(c)
            if entry["relevance_pct"] < cfg.min_relevance_pct:
                continue
            flat = [n for n in o.needs if n in spans and spans[n] < cfg.min_span_pct]
            if flat and o.id != cur_id:
                c["status"] = "구별 불가"
                c["reasons"].append("학습 데이터에서 " + ", ".join(f"{inst}.{n} 변화 폭 {spans[n]:.1f}%" for n in flat)
                                    + " — 이 형태를 다른 형태와 가를 수 없습니다")
                continue
            params = (list(common) if o.id == cur_id else
                      [p for p in common if p not in slot_params] + c["fit"])
            plan.append((entry, c, {**current, skey: o.id}, list(dict.fromkeys(params))))
        if entry["relevance_pct"] < cfg.min_relevance_pct:
            entry["status"] = "영향 작음"
            entry["message"] = (f"이 식의 파라미터를 10% 바꿔도 예측값이 {entry['relevance_pct']:.2f}% 만 변합니다. "
                                "비교하지 않고 현재 식을 유지합니다.")
        out_slots.append(entry)

    # 5) 관측 민감도 (보정 전에 거른다)
    todo = []
    for entry, c, choices, params in plan:
        if c["id"] == entry["current"]:
            todo.append((entry, c, choices, params))
            continue
        m = build(choices)
        own = c["new_params"]
        sens = _obs_sensitivity(m, rows_P(m, sample), own, obs, sigmas)
        if own and sens < cfg.min_obs_sensitivity:
            c["status"] = "구별 불가"
            c["reasons"].append(f"{', '.join(own)} 를 10% 바꿔도 관측이 계측 불확도의 {sens:.2f}배만 움직입니다 "
                                "— 보정으로 정할 수 없습니다")
            continue
        todo.append((entry, c, choices, params))

    # 6) 보정 + 검증
    for k, (entry, c, choices, params) in enumerate(todo):
        say(f"후보 비교 {k + 1}/{len(todo)}: {entry['key']} = {c['id']} 보정 중")
        r = evaluate(choices, params, c["id"])
        c["evaluated"] = True
        c["_r"] = r
        if "error" in r:
            c["status"] = "제외"
            c["reasons"].append(r["error"])
            continue
        cal = r["cal"]
        # 식별성은 그 후보에만 있는 파라미터로 본다 (공통 파라미터의 식별성은 설정의 성질)
        own = [p for p in (c["new_params"] if c["id"] != entry["current"] else c["fit"]) if p in cal.names]
        worst, max_rel = cal.subset(own) if own else (0.0, 0.0)
        bound = [p for p in cal.at_bound() if p in own]
        c.update({"val_rmse": r["val_rmse"], "val_rmse_all": r["val_rmse_all"], "val_bias": r["val_bias"],
                  "val_halves": r["halves"],
                  "val_failed": r["val_failed"], "aicc": r["aicc"], "seconds": r["seconds"],
                  "ident": {"worst_corr": worst, "max_rel_se_pct": max_rel}, "at_bound": bound,
                  "fitted": [{"param": n, "initial": float(from_si(i0, u)), "fitted": float(from_si(f, u)),
                              "unit": u} for n, u, i0, f in zip(cal.names, cal.units, cal.initial, cal.fitted)
                             if n in own]})
        if c["id"] == entry["current"]:
            if own and (max_rel > 50 or abs(worst) > 0.95):
                c["reasons"].append(f"현재 보정하는 {', '.join(own)} 가 이 데이터로 식별되지 않습니다 "
                                    f"(상대표준오차 {max_rel:.0f}%, 최악 상관 {worst:+.2f}) — 설계값에 고정하는 것을 "
                                    "검토하세요")
        else:
            if bound:
                c["status"] = "제외"
                c["reasons"].append(f"{', '.join(bound)} 가 보정 경계에 붙었습니다 — 이 형태가 데이터와 맞지 않습니다")
            elif max_rel > 50 or abs(worst) > 0.95:
                c["status"] = "제외"
                c["reasons"].append(f"후보 파라미터가 식별되지 않습니다 (상대표준오차 {max_rel:.0f}%, "
                                    f"최악 상관 {worst:+.2f})")

    # 7) 슬롯별 판정
    for entry in out_slots:
        cands = entry["candidates"]
        cur = next(c for c in cands if c["id"] == entry["current"])
        ev = [c for c in cands if c.get("evaluated") and "val_rmse" in c and np.isfinite(c["val_rmse"])]
        if entry["status"] == "영향 작음":
            for c in cands:
                c["status"] = c["status"] or ("현재 식" if c is cur else "비교 안 함")
            continue
        viable = [c for c in ev if c["status"] != "제외"]
        if cur not in viable:
            viable_ref = min(viable, key=lambda c: c["val_rmse"]) if viable else None
        else:
            viable_ref = cur
        if viable_ref is None:
            entry["status"] = "구별 불가"
            entry["message"] = "비교할 수 있는 후보가 없습니다. 현재 식을 유지합니다."
            for c in cands:
                c["status"] = c["status"] or "비교 안 함"
            continue
        # 모든 평가 후보를 현재 식(또는 기준)과 짝 비교
        for c in viable:
            if c is viable_ref:
                continue
            cmp = paired_abs_error(viable_ref["_r"]["meas"], viable_ref["_r"]["pred"], c["_r"]["pred"])
            c["vs_current"] = _cmp_dict(cmp)
        rec = viable_ref
        better = [c for c in viable if c is not viable_ref and c.get("vs_current")
                  and c["vs_current"]["diff"] <= -2 * c["vs_current"]["se"]
                  and c["val_rmse"] <= viable_ref["val_rmse"] * (1 - cfg.min_gain_pct / 100)]
        # 검증 구간 앞·뒤 절반 모두에서 나아야 바꾼다
        unstable = [c for c in better if not all(a <= b for a, b in zip(c["val_halves"], viable_ref["val_halves"]))]
        for c in unstable:
            c["reasons"].append("검증 구간 앞·뒤 절반 중 한쪽에서는 현재 식보다 나빴습니다 "
                                f"({c['val_halves'][0]:.3g}/{c['val_halves'][1]:.3g} vs "
                                f"{viable_ref['val_halves'][0]:.3g}/{viable_ref['val_halves'][1]:.3g}) — "
                                "기간에 따라 뒤집힐 수 있어 바꾸자고 하지 않습니다")
        better = [c for c in better if c not in unstable]
        if better:
            rec = min(better, key=lambda c: c["val_rmse"])
        entry["recommended"] = rec["id"]
        # 제외했지만 검증 오차는 더 낮았던 후보 — 숨기지 않고 적는다 (고르는 건 사용자)
        for c in ev:
            if c["status"] == "제외" and c["val_rmse"] < viable_ref["val_rmse"]:
                gain = 100 * (1 - c["val_rmse"] / viable_ref["val_rmse"])
                c["reasons"].append(f"검증 오차는 {gain:.0f}% 낮았지만 위 이유로 믿기 어렵습니다 — 식별되지 않는 "
                                    "파라미터가 우연히 검증 구간에 맞았을 수 있습니다")
        for c in viable:
            if c is rec:
                c["status"] = "추천"
                continue
            cmp = paired_abs_error(rec["_r"]["meas"], rec["_r"]["pred"], c["_r"]["pred"])
            c["vs_recommended"] = _cmp_dict(cmp)
            c["status"] = ("불안정" if c in unstable else
                           "나쁨" if cmp and cmp[0] >= 2 * cmp[1] else
                           "근소하게 나음" if cmp and cmp[0] <= -2 * cmp[1] else "비슷함")
        for c in cands:
            c["status"] = c["status"] or "비교 안 함"
        entry["message"] = _slot_message(entry, rec, cur, viable, cfg)
        if len(ev) <= 1:
            entry["status"] = "구별 불가"

    for entry in out_slots:
        for c in entry["candidates"]:
            c.pop("_r", None)
    out_slots.sort(key=lambda e: -e["relevance_pct"])
    return {
        "split": {"inner": [str(inner.index.min()), str(inner.index.max())],
                  "validate": [str(val.index.min()), str(val.index.max())],
                  "embargo_days": cfg.embargo_days, "n_rows": len(rows), "n_validate_hours": len(val_h),
                  "validate_outside_pct": float(outside.mean() * 100), "scored_on": "외삽 행" if use_out else "전체"},
        "unit": unit, "target": target, "common": list(common), "current": current,
        "slots": out_slots, "notes": notes, "n_calibrations": len(cals),
        "seconds": time.perf_counter() - t_start,
    }


# ---------------------------------------------------------------------------

def _has(model, name: str) -> bool:
    try:
        model.par_index(name)
        return True
    except (KeyError, ValueError):
        return False


def _par_idx(model, name: str):
    try:
        return model.par_index(name)
    except (KeyError, ValueError):
        return None


def _is_var(model, name: str) -> bool:
    try:
        model.var_index(name)
        return True
    except (KeyError, ValueError):
        return False


def _obs_sensitivity(model, P: np.ndarray, params: list[str], obs: list[str],
                     sigmas: dict[str, float]) -> float:
    """파라미터를 10% 바꿨을 때 관측이 움직이는 크기 (계측 불확도 배수, 관측·행 평균의 최대).

    불확도가 없는 관측은 값 크기의 2% 를 불확도로 본다.
    """
    if not params:
        return float("inf")
    y0 = simulate(model, P, obs).values
    best = 0.0
    for p in params:
        j = _par_idx(model, p)
        if j is None:
            continue
        P2 = P.copy()
        P2[:, j] *= 1.1
        y1 = simulate(model, P2, obs).values
        for o in obs:
            base = y0[o].to_numpy()
            # 계측 불확도를 모르면 값 크기의 2% 로 본다 (SI 1.0 을 쓰면 mg/m3 관측이 늘 '둔감'이 된다)
            s = sigmas.get(o) or 0.02 * float(np.nanmedian(np.abs(base)) or 1.0)
            d = np.abs(y1[o].to_numpy() - base) / max(s, 1e-30)
            if np.isfinite(d).any():
                best = max(best, float(np.nanmean(d)))
    return best


def _cmp_dict(cmp) -> dict | None:
    if cmp is None:
        return None
    diff, se, n_eff = cmp
    return {"diff": diff, "se": se, "n_eff": n_eff,
            "verdict": "유의미하게 좋음" if diff <= -2 * se else "유의미하게 나쁨" if diff >= 2 * se
            else "구분 안 됨"}


def _slot_message(entry, rec, cur, viable, cfg) -> str:
    u = ""
    parts = []
    if rec is cur:
        parts.append(f"현재 식 '{cur['title']}' 을 유지합니다 (검증 RMSE {cur['val_rmse']:.3g}{u}).")
        alts = [c for c in viable if c is not cur]
        if alts:
            parts.append(f"다른 후보 중 {cfg.min_gain_pct:g}% 이상 낮으면서 통계적으로 구분되는 것이 없었습니다.")
    else:
        gain = 100 * (1 - rec["val_rmse"] / cur["val_rmse"]) if cur.get("val_rmse") else float("nan")
        parts.append(f"'{rec['title']}' 이 현재 식보다 검증 오차가 {gain:.1f}% 낮고 차이가 유의합니다.")
    shaky = [c for c in viable if c.get("status") == "불안정"]
    if shaky:
        parts.append(" · ".join(f"'{c['title']}' 은 검증 구간 전체로는 {100 * (1 - c['val_rmse'] / rec['val_rmse']):.1f}% "
                                "낮지만 앞·뒤 절반 중 한쪽에서 나빴습니다" for c in shaky)
                     + " — 기간에 따라 뒤집힐 수 있습니다. 직접 판단하세요.")
    near = [c for c in viable if c.get("status") == "근소하게 나음"]
    if near:
        parts.append(" · ".join(f"'{c['title']}' 은 유의하게 낫지만 개선이 {cfg.min_gain_pct:g}% 미만"
                                f"({100 * (1 - c['val_rmse'] / rec['val_rmse']):.2f}%)" for c in near)
                     + " — 바꿀지는 직접 판단하세요.")
    excl_base = next((c for c in entry["candidates"] if c["role"] == "baseline" and c["status"] == "제외"), None)
    if excl_base is not None:
        parts.append(f"하한선 '{excl_base['title']}' 은 제외됐습니다: {'; '.join(excl_base['reasons'])}.")
    base = next((c for c in viable if c["role"] == "baseline"), None)
    if base is not None and base is not rec and base.get("vs_recommended"):
        v = base["vs_recommended"]
        if v["verdict"] == "구분 안 됨":
            parts.append("하한선(가장 단순한 식)과 구분되지 않습니다 — 이 데이터로는 더 복잡한 식이 값을 한다는 "
                         "근거가 약합니다. 가르는 변수가 더 크게 변한 데이터가 생기면 다시 비교하세요.")
    if cur.get("reasons") and cur.get("evaluated"):
        parts.extend(cur["reasons"])
    flat = [e for e in entry["excitation"] if not e["ok"]]
    if flat:
        parts.append("변화가 작았던 변수: " + ", ".join(f"{e['var']} ({e['span_pct']:.1f}%)" for e in flat) + ".")
    return " ".join(parts)
