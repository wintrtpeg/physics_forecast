"""현장형 더미 데이터로 툴 전체를 채점한다 (정답지 대비).

``make_field_data.py`` 가 숨겨 둔 문제를 툴이 얼마나 찾아내고, 그 데이터로 만든
물리모델이 학습 범위 밖(증설 + 여름 + 가동율 상승)을 얼마나 맞히는지 잰다.

    python examples/nox_field/evaluate.py            # 약 25분 (보정 2회 + 7만 점 예측 2회)
    python examples/nox_field/evaluate.py --quick    # 정제 없는 비교(ablation)를 건너뜀

결과: ``out/scorecard.md`` (사람용), ``out/scorecard.json`` (숫자).
문서에 쓰는 수치는 전부 이 스크립트가 실제로 낸 값이다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "out"

from pforecast.analyze import AnalysisConfig, run_analysis  # noqa: E402
from pforecast.core.units import from_si  # noqa: E402
from pforecast.data.ingest import read_table  # noqa: E402
from pforecast.data.quality import assess  # noqa: E402
from pforecast.workflow import WorkflowConfig, run_workflow  # noqa: E402

TARGET = "STK.C_dry"
NOX_TAG = "F2_UT_STK01_NOX_DRY"
LIMIT = 100.0


def _rmse(a, b) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.any() else float("nan")


def _bias(a, b) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(a[ok] - b[ok])) if ok.any() else float("nan")


def score_ingest(tab, anomalies) -> dict:
    r = tab.report
    fmt = next(e for e in anomalies["events"] if e["type"] == "format")
    return {
        "encoding": r.encoding, "header_rows": r.header_rows,
        "unit_row_parsed": sum(1 for u in r.units.values() if u),
        "time_formats": r.time_formats, "repeated_header_rows": r.n_repeated_header,
        "duplicate_rows": r.n_duplicate_rows, "expected_duplicate_rows": fmt["overlap_rows"],
        "out_of_order": r.n_out_of_order, "status_cells": r.status_total(),
        "thousands_cells": sum(r.thousands.values()), "grid_rows": len(tab.df),
        "missing_grid": r.n_missing_grid, "longest_gap": r.gaps[0] if r.gaps else None,
    }


def score_quality(tab, q, truth) -> dict:
    out = {}
    bad_types = ["cal_zero", "cal_span", "cal_recover", "spike", "frozen"]
    tot_fp = tot_clean = 0
    for tag in tab.df.columns:
        lab = truth[f"label:{tag}"].fillna("").astype(str).to_numpy()
        m = q.masks.get(tag, np.zeros(len(lab), bool))
        present = tab.df[tag].notna().to_numpy() | m
        bad = np.isin(lab, bad_types) & present
        clean = ~np.isin(lab, bad_types) & present
        tot_fp += int((m & clean).sum())
        tot_clean += int(clean.sum())
        if bad.sum() == 0:
            continue
        per = {t: {"injected": int(((lab == t) & present).sum()),
                   "caught": int((m & (lab == t) & present).sum())} for t in bad_types
               if ((lab == t) & present).any()}
        out[tag] = {"injected": int(bad.sum()), "flagged": int(m.sum()),
                    "true_pos": int((m & bad).sum()), "false_pos": int((m & ~bad & present).sum()),
                    "by_type": per}
    out["_total"] = {"false_pos": tot_fp, "clean_points": tot_clean}
    out["_zero_hold_kept"] = any(i.kind == "zero_hold" for i in q.issues)
    return out


def score_analysis(cfg_path, csv=None) -> dict:
    cfg = AnalysisConfig.load(cfg_path)
    if csv is not None:
        cfg.csv = str(csv)
    t0 = time.perf_counter()
    res = run_analysis(cfg, verbose=False)
    s = res.surrogate
    bal = [{"formula": b.formula(), "confidence": b.confidence, "gain": b.gain,
            "offset": b.offset, "note": b.gain_note()} for b in res.balances]
    return {
        "seconds": time.perf_counter() - t0,
        "r2_cv": s.r2_cv if s else None, "rmse_cv": s.rmse_cv if s else None,
        "holdout_outside": res.holdout_outside,
        "untrainable": [f for f, *_ in res.untrainable],
        "top_cluster": s.clusters[0].members if s and s.clusters else [],
        "top_cluster_pct": s.clusters[0].importance_pct if s and s.clusters else None,
        "balances": bal,
        "split": res.split.to_dict() if res.split is not None else None,
        "forecast": {k: v for k, v in res.forecast.items() if k != "pred"},
    }


def score_workflow(cfg: WorkflowConfig, truth: pd.DataFrame, label: str) -> tuple[dict, object]:
    t0 = time.perf_counter()
    res = run_workflow(cfg, verbose=False)
    secs = time.perf_counter() - t0
    conv = lambda a: np.array([from_si(v, "mg/Nm3") for v in np.asarray(a, float)])  # noqa: E731
    te = res.test.index
    true_c = truth.loc[te, "true:STK.C_dry"].to_numpy()
    meas = conv(res.test[TARGET])
    phys = conv(res.physics_test[TARGET])
    models = [("물리모델", phys)]
    if res.baseline_test is not None:
        models.append(("ML 다항(2차)", conv(res.baseline_test)))
    for name, (_, p_te) in res.extra_baselines.items():
        models.append((name, conv(p_te)))
    # 채점은 학습 운전영역 밖(외삽) 행만. 이 설계에서는 검증 행 전부가 해당된다.
    out = res.split.outside if res.split.outside is not None else np.ones(len(te), bool)
    rows = {}
    for name, pred in models:
        pred = np.where(out, pred, np.nan)
        # 초과 판정은 1시간 평균 기준 (5분 값은 잡음이 크다)
        hr = pd.DataFrame({"p": pred, "t": true_c}, index=te).resample("1h").mean().dropna()
        exceed = hr["t"] > LIMIT
        rows[name] = {
            "rmse_vs_true": _rmse(pred, true_c), "bias_vs_true": _bias(pred, true_c),
            "rmse_vs_measured": _rmse(pred, meas),
            "exceed_hours_true": int(exceed.sum()),
            "exceed_hit": int((exceed & (hr["p"] > LIMIT)).sum()),
            "false_alarm_hours": int((~exceed & (hr["p"] > LIMIT)).sum()),
        }
    cal = res.calibration
    params = {}
    if cal is not None:
        for n, u, v, se in zip(cal.names, cal.units, cal.fitted, cal.stderr):
            params[n] = {"fitted": from_si(v, u), "rel_stderr_pct": 100 * se / max(abs(v), 1e-30),
                         "unit": u}
    cps = [c.to_dict() for c in res.changepoints]
    train_true = truth.loc[res.train.index, "true:STK.C_dry"].to_numpy()
    return {
        "label": label, "seconds": secs, "n_train": len(res.train), "n_test": len(te),
        "noise_floor_rmse": _rmse(meas, true_c),
        "physics_train_rmse_vs_true": _rmse(conv(res.physics_train[TARGET]), train_true),
        "models": rows, "params": params,
        "identifiability": cal.identifiability_warnings() if cal is not None else [],
        "excluded_rows": cal.n_excluded if cal is not None else 0,
        "clipped": dict(res.clipped), "changepoints": cps,
        "test_true_range": [float(np.nanmin(true_c)), float(np.nanmax(true_c))],
        "split": res.split.to_dict(),
    }, res


def score_changepoints(cps: list[dict], anomalies: dict, tol_days: int = 2) -> dict:
    """잔차 변화점 채점: 숨긴 사건을 잡았는가 + 잡은 것마다 정답이 있는가."""
    ev = anomalies["EV"]
    truth = [
        ("레시피 변경 (DRY 배출 +15%, 비계측)", ev["recipe_change"], {"STK.C_dry"}, True),
        ("정압 전송기 교체 (영점 오프셋)", ev["fan_sp_offset"], {"FAN.dp_mmAq"}, True),
        ("충전재 세정 (차압·효율 복원)", ev["packing_clean"][1],
         {"SCR.dp_mmAq", "STK.C_dry", "STK.Q_n", "FAN.dp_mmAq"}, True),
        ("NOx 분석계 교체 (오프셋)", ev["analyzer_swap"], {"STK.C_dry"}, True),
        ("순환수 증량 (모델 NTU 지수 0.55 vs 실제 0.62)", ev["water_increase"], {"STK.C_dry"}, False),
        ("증설 (팬 곡선 마모가 드러남)", ev["expansion"], None, False),
    ]
    truth += [("분석계 월간 교정 (드리프트 복귀)", f"2025-{m:02d}-01 10:00", {"STK.C_dry"}, False)
              for m in range(2, 9)]
    for when, hz in anomalies.get("fan_hz", [])[1:]:
        truth.append((f"송풍기 설정 {hz:g} Hz (팬 곡선 마모)", when, None, False))

    def match(c):
        d = pd.Timestamp(c["date"])
        for desc, when, obs, _ in truth:
            if abs((d - pd.Timestamp(when).normalize()).days) <= tol_days and \
                    (obs is None or c["observation"] in obs):
                return desc
        return None

    detected = [{**{k: c[k] for k in ("date", "observation", "shift", "kind")},
                 "truth": match(c)} for c in cps]
    main = []
    for desc, when, obs, is_main in truth:
        if not is_main:
            continue
        t = pd.Timestamp(when).normalize()
        hit = [c for c in cps if c["observation"] in obs
               and abs((pd.Timestamp(c["date"]) - t).days) <= tol_days]
        main.append({"desc": desc, "date": str(t.date()), "found": bool(hit),
                     "observations": sorted({c["observation"] for c in hit}),
                     "kind": hit[0]["kind"] if hit else None})
    return {"main": main, "detected": detected,
            "unmatched": [d for d in detected if d["truth"] is None]}


def write_markdown(sc: dict, path: Path) -> None:
    ds = sc.get("dataset") or {}
    kind = ("시험용 변형 — 개발 중 한 번도 보지 않은 데이터 (사건 날짜·교정 시각·결함 크기도 다름)"
            if ds.get("holdout") else "개발용 데이터 (탐지 임계값을 이 데이터를 보며 조정했다)")
    L = ["# 현장형 더미 데이터 채점표", "",
         f"데이터: {kind}, 시드 {ds.get('seed')}", "",
         f"생성: {sc['generated']} · 소요 {sc['total_seconds']/60:.1f}분", ""]
    ing = sc["ingest"]
    L += ["## 1. 파일 읽기", "",
          f"* 인코딩 {ing['encoding']}, 헤더 {ing['header_rows']}행, 단위 행 {ing['unit_row_parsed']}개 해석",
          "* 시각 표기: " + ", ".join(f"{k} {v:,}" for k, v in ing['time_formats'].items()),
          f"* 반복 헤더 {ing['repeated_header_rows']}행 · 중복 {ing['duplicate_rows']}행 "
          f"(주입 {ing['expected_duplicate_rows']}) · 역순 {ing['out_of_order']}곳",
          "* 상태 문자열: " + ", ".join(f"{k} {v:,}" for k, v in ing['status_cells'].items()),
          f"* 천 단위 콤마 {ing['thousands_cells']:,}셀, 격자 {ing['grid_rows']:,}행 "
          f"(빈 시각 {ing['missing_grid']}개)", ""]
    L += ["## 2. 값 이상 탐지 (정답 라벨 대비)", "", "| 태그 | 주입 | 탐지 | 놓침 | 오탐 |",
          "|---|---:|---:|---:|---:|"]
    for tag, v in sc["quality"].items():
        if tag.startswith("_"):
            continue
        L.append(f"| {tag} | {v['injected']:,} | {v['true_pos']:,} | "
                 f"{v['injected']-v['true_pos']:,} | {v['false_pos']:,} |")
    tq = sc["quality"]["_total"]
    L += ["", f"정상 점 {tq['clean_points']:,}개 중 오탐 {tq['false_pos']}개. 펌프 정지(값 0 유지)는 "
          f"{'지우지 않음' if sc['quality']['_zero_hold_kept'] else '지움(오류)'}.", ""]
    an = sc["analysis"]
    fc = an.get("forecast") or {}
    L += ["## 3. 데이터 주도 분석 (L0~L3)", "",
          f"* 현재값 추정(같은 시각 측정값 사용) 전진 교차검증 R² {an['r2_cv']:.3f}",
          (f"* 미래 예측(운전 입력만, 시간순 분할): 검증 RMSE {fc['rmse']:.2f}, "
           f"외삽 행 {fc['n_outside']:,}개에서 {fc['rmse_outside']:.2f}, "
           f"내삽 행 {fc['n_inside']:,}개에서 {fc['rmse_inside']:.2f} mg/Sm³"
           if fc.get("rmse") is not None else "* 미래 예측: 운전 입력 미지정"),
          f"* 학습 구간에서 변하지 않아 대리모델이 배울 수 없는 변수: {', '.join(an['untrainable']) or '없음'}"]
    for b in an["balances"]:
        L.append(f"* [{b['confidence']}] {b['formula']} — {b['note'] or '이득 1.00'}")
    L.append("")
    for wf in sc["workflows"]:
        sp = wf["split"]
        L += [f"## 4. 물리모델 vs ML — 미래 · 외삽 검증 ({wf['label']})", "",
              f"* 미래 예측인가: {'예' if sp['is_future'] else '**아니오**'} "
              f"(검증 행 중 학습보다 과거 {sp['n_test_before_train_end']}개, "
              f"간격 {sp['embargo_days']:.1f}일)",
              f"* 외삽인가: 검증 행의 **{sp['extrapolation'] * 100:.0f}%** 가 학습 운전영역 밖 "
              "(채점은 그 행들만)",
              f"* 계측 잡음 하한(측정 vs 참값 RMSE) {wf['noise_floor_rmse']:.2f} mg/Sm³", "",
              "| 모델 | 참값 대비 RMSE | 편향 | 측정값 대비 RMSE | 초과 시간 적중 | 오경보 |",
              "|---|---:|---:|---:|---:|---:|"]
        for name, m in wf["models"].items():
            L.append(f"| {name} | {m['rmse_vs_true']:.2f} | {m['bias_vs_true']:+.2f} | "
                     f"{m['rmse_vs_measured']:.2f} | {m['exceed_hit']}/{m['exceed_hours_true']} | "
                     f"{m['false_alarm_hours']} |")
        L += ["", f"보정에서 뺀 행 {wf['excluded_rows']}개, 범위 밖이라 자른 입력 {wf['clipped'] or '없음'}.", ""]
        cs = wf.get("changepoint_score")
        if cs:
            L += ["잔차 변화점 — 숨긴 사건:", "", "| 사건 | 날짜 | 잡힘 | 움직인 관측 | 해석 유형 |",
                  "|---|---|---|---|---|"]
            for c in cs["main"]:
                L.append(f"| {c['desc']} | {c['date']} | {'예' if c['found'] else '아니오'} | "
                         f"{', '.join(c['observations']) or '—'} | {c['kind'] or '—'} |")
            L += ["", "탐지한 변화점 전부:", "", "| 날짜 | 관측 | 계단 | 유형 | 정답 |",
                  "|---|---|---:|---|---|"]
            for d in cs["detected"]:
                L.append(f"| {d['date']} | {d['observation']} | {d['shift']:+.3g} | {d['kind']} | "
                         f"{d['truth'] or '정답 없음'} |")
            L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="정제 없는 비교를 건너뜀")
    ap.add_argument("--data", default=str(DATA),
                    help="채점할 데이터 폴더 (field_raw.csv + _answer/). 시험용 변형은 data/holdout")
    ap.add_argument("--name", default="scorecard", help="결과 파일 이름")
    args = ap.parse_args()
    data = Path(args.data)
    if not (data / "field_raw.csv").exists():
        sys.exit("먼저 make_field_data.py 를 실행하세요")
    OUT.mkdir(exist_ok=True)
    t_all = time.perf_counter()
    anomalies = json.loads((data / "_answer" / "anomalies.json").read_text(encoding="utf-8"))

    print("[1/4] 파일 읽기 + 값 이상 탐지", flush=True)
    tab = read_table(data / "field_raw.csv")
    q = assess(tab.df)
    truth = pd.read_csv(data / "_answer" / "field_truth.csv", index_col=0, parse_dates=True,
                        low_memory=False).reindex(tab.df.index)
    sc = {"generated": pd.Timestamp.now().isoformat(timespec="seconds"),
          "ingest": score_ingest(tab, anomalies), "quality": score_quality(tab, q, truth)}

    sc["dataset"] = {"path": str(data), "seed": anomalies.get("seed"),
                     "holdout": anomalies.get("holdout", False), "var": anomalies.get("var")}
    print("[2/4] 데이터 주도 분석", flush=True)
    sc["analysis"] = score_analysis(HERE / "analysis.yaml", data / "field_raw.csv")

    print("[3/4] 보정 + 외삽 검증 (정제 적용)", flush=True)
    sc["workflows"] = []
    cfg = WorkflowConfig.load(HERE / "calibration.yaml")
    cfg.csv = str(data / "field_raw.csv")
    cfg.report_out = None
    cfg.params_out = None
    wf, _ = score_workflow(cfg, truth, "정제 적용")
    wf["changepoint_score"] = score_changepoints(wf["changepoints"], anomalies)
    sc["workflows"].append(wf)

    if not args.quick:
        print("[4/4] 보정 + 외삽 검증 (값 정제 없이 — 비교용)", flush=True)
        cfg2 = WorkflowConfig.load(HERE / "calibration.yaml")
        cfg2.csv = str(data / "field_raw.csv")
        cfg2.report_out = cfg2.params_out = None
        cfg2.clean = False
        cfg2.diagnose = False
        wf2, _ = score_workflow(cfg2, truth, "값 정제 없음")
        sc["workflows"].append(wf2)

    sc["total_seconds"] = time.perf_counter() - t_all
    (OUT / f"{args.name}.json").write_text(
        json.dumps(sc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_markdown(sc, OUT / f"{args.name}.md")
    print((OUT / f"{args.name}.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
