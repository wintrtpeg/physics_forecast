"""pforecast 명령줄 인터페이스.

    pf check   examples/nox_stack/model.py      # 구조 해석 + 설계점 풀이
    pf solve   examples/nox_stack/model.py      # 설계점 상세 결과
    pf calibrate examples/nox_stack/calibration.yaml
    pf run     examples/nox_stack/scenarios.yaml
    pf demo                                     # NOx 예제 전 과정
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def _load(model_arg: str, builder: str = "build"):
    from .scenario import load_model
    p = Path(model_arg)
    spec = {"yaml": str(p)} if p.suffix in (".yaml", ".yml") else {"python": str(p), "builder": builder}
    return load_model(spec)


def cmd_check(args) -> int:
    from .core.solvers import solve_steady
    t0 = time.perf_counter()
    system = _load(args.model, args.builder)
    model = system.compile(check_dims=not args.no_dim_check)
    print(model.describe())
    print(f"  야코비안 non-zero {model.jacobian_nnz()}개 "
          f"(조립 {time.perf_counter()-t0:.2f}s)")
    if not model.report.ok:
        print("\n구조가 정방이 아닙니다. 위 진단을 보고 방정식/연결을 고치세요.")
        return 1
    t0 = time.perf_counter()
    r = solve_steady(model)
    print(f"\n설계점 풀이: {'성공' if r.success else '실패'} ({time.perf_counter()-t0:.3f}s)")
    print("  " + r.message.replace("\n", "\n  "))
    return 0 if r.success else 1


def cmd_solve(args) -> int:
    from .core.solvers import solve_steady
    from .params import apply_params
    system = _load(args.model, args.builder)
    model = system.compile()
    model.build()
    if args.params:
        applied = apply_params(model, args.params, strict=False)
        print(f"보정 파라미터 {len(applied)}개 적용: {args.params}")
    r = solve_steady(model)
    if not r.success:
        print("풀이 실패:\n  " + r.message.replace("\n", "\n  "))
        return 1
    print(r.message)
    out = model.output_values(r.x, model.p0())
    groups: dict[str, list[tuple[str, float]]] = {}
    for k in sorted(out):
        groups.setdefault(k.split(".")[0], []).append((k.split(".", 1)[1], out[k]))
    for comp in sorted(groups):
        print(f"\n[{comp}]")
        for name, val in groups[comp]:
            unit = model.outputs[f"{comp}.{name}"][1]
            print(f"  {name:20s} {val:14.4f}  {unit}")
    if args.csv:
        import pandas as pd
        pd.DataFrame([out]).to_csv(args.csv, index=False)
        print(f"\n저장: {args.csv}")
    return 0


def cmd_calibrate(args) -> int:
    from .report import calibration_report
    from .workflow import WorkflowConfig, run_workflow
    cfg = WorkflowConfig.load(args.config)
    if args.max_rows:
        cfg.max_rows = args.max_rows
    t0 = time.perf_counter()
    res = run_workflow(cfg)
    print(f"완료 ({time.perf_counter()-t0:.1f}s)")
    if res.calibration is not None:
        print("\n" + res.calibration.table().to_string(
            index=False, float_format=lambda v: f"{v:.5g}"))
        warns = res.calibration.identifiability_warnings()
        if warns:
            print("\n[식별성 경고]")
            for w in warns:
                print("  ! " + w)
    if res.comparison is not None and len(res.comparison):
        print("\n[외삽 성능 비교]")
        print(res.comparison.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if res.changepoints:
        print("\n[잔차 변화점 — 모델이 설명하지 못한 변화]")
        for c in res.changepoints:
            print(f"  {c.date.date()} {c.observation:14s} {c.shift:+9.3g}{c.unit}  "
                  f"(z={c.z:.0f}, {c.kind})")
            print(f"      {c.interpretation()}")
    if res.clipped:
        print("\n[범위 밖 입력을 경계로 자름] " + ", ".join(f"{k} {v:,}행" for k, v in res.clipped.items()))
    out = args.out or cfg.report_out
    if out:
        p = calibration_report(cfg, res, out)
        print(f"\n리포트: {p}")
    if cfg.params_out:
        print(f"보정 파라미터: {cfg.params_out}")
    return 0


def cmd_improve(args) -> int:
    from pathlib import Path

    from .improve import run_improve
    from .workflow import WorkflowConfig
    cfg = WorkflowConfig.load(args.config)
    if args.max_rows:
        cfg.max_rows = args.max_rows
    res = run_improve(cfg, n_folds=args.folds, fold_days=args.fold_days,
                      drift=not args.no_drift)
    print(f"\n완료 ({res.seconds:.0f}s)\n")
    for line in res.lines():
        print(line)
    out = Path(args.out) if args.out else (Path(cfg.report_out).with_name("improve.md")
                                          if cfg.report_out else None)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(res.to_markdown(), encoding="utf-8")
        print(f"\n저장: {out}")
    return 0


def cmd_closures(args) -> int:
    """구성방정식 후보 추천 (학습 구간만). 고르는 것은 사람 — 결과를 보고 설정의 model.closures 에 적는다."""
    from .closure_advice import AdviceConfig, advise_closures
    from .scenario import load_model
    from .workflow import WorkflowConfig, load_dataset

    cfg = WorkflowConfig.load(args.config)
    tm, df, train, _test, ic, oc = load_dataset(cfg)
    del _test                                   # 시험 구간은 후보 선택에 쓰지 않는다
    obs = [c for c in (cfg.observations or oc) if c in df.columns]
    target = args.target or cfg.baseline_target
    model = load_model(cfg.model).compile()
    common = [p for p in cfg.params if p in {q.name for q in model.parameters}]
    res = advise_closures(cfg.model, tm, train, ic, obs, target, common,
                          current=(cfg.model or {}).get("closures") if isinstance(cfg.model, dict) else None,
                          cfg=AdviceConfig(max_rows=args.max_rows, slots=args.slot or None),
                          progress=lambda m: print("  · " + m, flush=True))
    sp = res["split"]
    print(f"\n보정 {sp['inner'][0][:10]}~{sp['inner'][1][:10]} ({sp['n_rows']}행) | 간격 {sp['embargo_days']}일 | "
          f"검증 {sp['validate'][0][:10]}~{sp['validate'][1][:10]} ({sp['n_validate_hours']}시간, "
          f"외삽 행 {sp['validate_outside_pct']:.0f}%) | 보정 {res['n_calibrations']}회 {res['seconds']:.0f}s")
    for n in res["notes"]:
        print("  !! " + n)
    for sl in res["slots"]:
        print(f"\n[{sl['key']}] {sl['component']} — {sl['title']}  ({sl['status']}, 예측값 영향 "
              f"{sl['relevance_pct']:.1f}%)  추천: {sl['recommended']}")
        for e in sl["excitation"]:
            print(f"    변화 폭 {e['var']}: {e['span_pct']:.1f}%")
        for r in sl["residual_corr"]:
            print(f"    현재 식 잔차 ↔ {r['var']}: r={r['r']:+.2f}")
        for c in sl["candidates"]:
            v = f"검증 RMSE {c['val_rmse']:.3f} {res['unit']}" if c.get("val_rmse") is not None and \
                c.get("evaluated") and c.get("val_rmse") == c.get("val_rmse") else ""
            print(f"    - {c['id']:16s} {c['status']:6s} {v}  {'; '.join(c['reasons'])}")
        if sl["message"]:
            print("    → " + sl["message"])
    print("\n고른 식은 설정 파일에 적습니다:\n  model:\n    closures: {" + ", ".join(
        f"{s['key']}: {s['recommended']}" for s in res["slots"] if s["recommended"] != s["default"]) + "}")
    return 0


def cmd_select(args) -> int:
    import pickle
    from .report import selection_report
    from .selection import SelectionConfig, run_selection
    cfg = SelectionConfig.load(args.config)
    if args.max_rows:
        cfg.max_rows = args.max_rows
    if args.load:
        # 보정을 다시 하지 않고 리포트만 다시 그린다 (후보 비교는 수 분이 걸린다)
        with open(args.load, "rb") as fh:
            res = pickle.load(fh)
        print(f"저장된 결과 사용: {args.load}")
    else:
        t0 = time.perf_counter()
        res = run_selection(cfg)
        print(f"완료 ({time.perf_counter()-t0:.0f}s)\n")
        if args.save:
            with open(args.save, "wb") as fh:
                pickle.dump(res, fh)
            print(f"결과 저장: {args.save}")
    import pandas as pd
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(res.table().to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n[판정]")
    for v in res.verdict():
        print("  - " + v.replace("**", ""))
    out = args.out or cfg.report_out
    if out:
        print(f"\n리포트: {selection_report(res, out)}")
    return 0


def _project(args):
    """이름 또는 경로로 프로젝트를 찾아 (문서, 작업 폴더, 설정) 을 돌려준다."""
    from .easy import config_from_body
    from .project import load_project, project_path
    root = Path(args.root).resolve()
    p = Path(args.project)
    path = p if p.suffix in (".yaml", ".yml") and p.exists() else project_path(root, args.project)
    doc = load_project(path)
    run = doc.get("run") or {}
    if not run.get("csv") or not run.get("target"):
        raise ValueError(f"{path.name}: 데이터·예측할 값이 정해지지 않은 프로젝트입니다 (앱에서 1~4단계를 먼저).")
    cfg = config_from_body(run, lambda r: (root / r).resolve())
    return doc, root, cfg, path


def cmd_project(args) -> int:
    from .project import list_projects
    if args.action == "list":
        items = list_projects(args.root)
        if not items:
            print("저장된 프로젝트가 없습니다 (앱에서 실행하면 projects/ 에 자동 저장).")
        for it in items:
            print(f"  {it['name']:30s} {str(it.get('saved_at', ''))[:16]:16s} {it.get('csv') or ''} → {it.get('target') or ''}"
                  + (f"  [오류: {it['error']}]" if it.get("error") else ""))
        return 0
    from .easy import run_easy
    doc, root, cfg, _ = _project(args)
    if not cfg.train[0] or not cfg.test[0]:
        raise ValueError("학습·검증 기간이 없는 프로젝트입니다 (앱 3단계).")
    print(f"[{doc['name']}] 검증: 학습 {cfg.train[0]}~{cfg.train[1]} / 예측 {cfg.test[0]}~{cfg.test[1]}")
    res = run_easy(cfg, root, progress=lambda m: print("  · " + m, flush=True))
    key = "outside" if res["split"]["is_extrapolation"] else "test"
    for m in res["models"]:
        mt = m["metrics"].get(key) or m["metrics"]["test"]
        print(f"  {m['name']:14s} RMSE {mt['rmse']:.4g}  MAE {mt['mae']:.4g}  ({'학습 범위 밖 행' if key == 'outside' else '전체'})")
    for h in res["headline"]:
        print("  - " + h.replace("**", ""))
    print(f"\n결과: {res['files']['predictions']}")
    return 0


def _parse_adjust(items: list[str] | None) -> list[dict]:
    """``컬럼=scale:1.1`` / ``컬럼=add:2`` / ``컬럼=set:12``."""
    out = []
    for it in items or []:
        col, _, rhs = it.partition("=")
        mode, _, val = rhs.partition(":")
        if not col or mode not in ("scale", "add", "set") or not val:
            raise ValueError(f"--adjust 형식: 컬럼=scale:1.1 | 컬럼=add:2 | 컬럼=set:12 (받은 값: {it!r})")
        out.append({"column": col, "mode": mode, "value": float(val)})
    return out


def cmd_forecast(args) -> int:
    """저장한 프로젝트의 모델·연결로 과거를 학습하고 계획(미래 입력)을 예측한다."""
    from .forecast import check_plan, forecast_easy, make_plan, read_plan, save_plan, _history_table
    doc, root, cfg, _ = _project(args)
    if args.plan:
        plan_path = Path(args.plan)
    else:
        plan = make_plan(cfg.csv, cfg.time_column, cfg.features, base_days=args.base_days,
                         horizon_days=args.days, step=args.step, adjust=_parse_adjust(args.adjust))
        from .fileio import safe_stem
        plan_path, locked = save_plan(plan, root / "plans" / f"{safe_stem(doc['name'])}_계획_{plan.index.min():%Y%m%d}.csv")
        if locked:
            print("  !  " + locked)
        print(f"계획을 만들었습니다: {plan_path}  (엑셀에서 고쳐 --plan 으로 다시 줄 수 있음)")
    plan_df, _log = read_plan(plan_path)
    chk = check_plan(plan_df, cfg.features, _history_table(cfg.csv, cfg.time_column, cfg.features))
    for e in chk["errors"]:
        print("  !! " + e)
    for w in chk["warnings"]:
        print("  !  " + w)
    if not chk["ok"]:
        return 2
    hist = tuple(cfg.train) if args.history == "train" and cfg.train[0] else None
    res = forecast_easy(cfg, plan_path, root, history=hist, band=not args.no_band,
                        progress=lambda m: print("  · " + m, flush=True))
    u = res["target"]["unit"]
    print(f"\n[{doc['name']}] 학습 {res['history']['start'][:10]}~{res['history']['end'][:10]} → "
          f"예측 {res['plan']['start'][:16]}~{res['plan']['end'][:16]} ({res['plan']['n_rows']}행, 학습 범위 밖 {res['outside_pct']:.0f}%)")
    for s in res["summary"]:
        ex = f"  관리기준 초과 {s['exceed_hours']}/{s['hours']}시간" if "exceed_hours" in s else ""
        print(f"  {s['name']:14s} 평균 {s['mean']:.4g} {u}  95% {s['p95']:.4g}  최대 {s['max']:.4g}{ex}")
    if res["band"]:
        b = res["band"]
        print(f"  설비 상태 변동 폭 (평균): {b['mean_lo']:.4g} ~ {b['mean_hi']:.4g} {u}"
              + (f"  초과 가능 {b['exceed_possible_hours']}시간 / 확실 {b['exceed_certain_hours']}시간"
                 if "exceed_possible_hours" in b else ""))
    for n in res["notes"]:
        print("  - " + n)
    print(f"\n결과: {res['files']['forecast']}")
    return 0


def cmd_serve(args) -> int:
    from .app import serve
    serve(args.root, args.host, args.port, not args.no_browser)
    return 0


def cmd_analyze(args) -> int:
    from .analyze import AnalysisConfig, run_analysis
    from .report import analysis_report
    cfg = AnalysisConfig.load(args.config)
    if args.target:
        cfg.target = args.target
    if args.steady_only:
        cfg.steady_only = True
    t0 = time.perf_counter()
    res = run_analysis(cfg)
    print(f"완료 ({time.perf_counter()-t0:.1f}s)\n")
    import pandas as pd
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 60)
    print("[분석 사다리]")
    print(res.ladder_table()[["단계", "내용", "상태", "결과"]].to_string(index=False))
    print("\n[먼저 읽을 것]")
    for h in res.headline():
        print("  - " + h.replace("**", ""))
    if res.balances:
        print("\n[자동 발견 물리 관계]")
        for b in res.balances:
            print(f"  [{b.confidence}] {b.formula()}  (잔차 {b.residual_rel*100:.2f}%)")
    if res.surrogate is not None:
        print("\n[인자 묶음 중요도]")
        print(res.surrogate.cluster_table().to_string(index=False,
                                                      float_format=lambda v: f"{v:.4g}"))
    out = args.out or cfg.report_out
    if out:
        print(f"\n대시보드: {analysis_report(res, out)}")
    return 0


def cmd_equations(args) -> int:
    """모델의 방정식을 이름과 함께 나열한다. 선언형 컴포넌트 디버깅용."""
    system = _load(args.model, args.builder)
    model = system.compile(check_dims=not args.no_dim_check)
    from .core import symbolic as S
    from .core.units import dim_str
    src = args.component
    shown = 0
    for info in model.equations:
        if src and not info.source.startswith(src):
            continue
        shown += 1
        line = f"{info.label:34s} {S.expr_to_str(info.expr)}"
        if args.dims:
            try:
                line += f"   [{dim_str(S.dim_of_expr(info.expr))}]"
            except Exception as exc:
                line += f"   [차원오류: {exc}]"
        print(line if len(line) < 200 or args.full else line[:197] + "...")
    print(f"\n{shown}개 방정식 / 미지수 {model.n_vars}개")
    return 0


def cmd_run(args) -> int:
    from .params import apply_params
    from .report import scenario_report
    from .scenario import ScenarioSpec, load_model, run_scenario
    spec = ScenarioSpec.load(args.scenario)
    root = Path(args.scenario).parent

    def rel(v):
        if not v:
            return v
        p = Path(v)
        return p if p.is_absolute() or p.exists() else root / p

    mspec = spec.model
    if isinstance(mspec, dict):
        mspec = {k: (str(rel(v)) if k in ("python", "yaml") else v) for k, v in mspec.items()}
    model = load_model(mspec).compile()
    model.build()
    pf = args.params or spec.params_file
    if pf and Path(rel(pf)).exists():
        apply_params(model, rel(pf), strict=False)
        print(f"보정 파라미터 적용: {rel(pf)}")
    res = run_scenario(model, spec)
    import pandas as pd
    pd.set_option("display.width", 200, "display.max_columns", 40)
    print(f"\n[{spec.name}] 케이스 {len(res.cases)}개")
    print(res.cases.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    for name, df in res.sweeps.items():
        print(f"\n[스윕: {name}]")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if len(res.violations):
        print(f"\n!! 관리기준 초과 {len(res.violations)}건")
        print(res.violations.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    else:
        print("\n관리기준 초과 없음")
    if args.out:
        print(f"\n리포트: {scenario_report(res, args.out)}")
    if args.csv:
        res.cases.to_csv(args.csv, index=False)
        print(f"저장: {args.csv}")
    return 1 if len(res.violations) else 0


def _example_root() -> Path | None:
    """예제 디렉터리를 찾는다. 편집 설치/PYTHONPATH/현재 작업 디렉터리 모두 지원."""
    candidates = [
        Path(__file__).resolve().parents[2] / "examples" / "nox_stack",
        Path.cwd() / "examples" / "nox_stack",
        Path.cwd(),
    ]
    for c in candidates:
        if (c / "model.py").exists() and (c / "scenarios.yaml").exists():
            return c
    return None


def cmd_demo(args) -> int:
    root = _example_root()
    if root is None:
        print("예제 디렉터리를 찾을 수 없습니다. 저장소 최상위에서 실행하세요.")
        return 1
    print(f"예제 위치: {root}\n")
    steps = [
        ("1) 모델 구조 검사", ["check", str(root / "model.py")]),
        ("2) 설계점 풀이", ["solve", str(root / "model.py")]),
        ("3) 시나리오 실행", ["run", str(root / "scenarios.yaml"),
                              "-o", str(root / "out" / "scenario.html")]),
    ]
    for label, argv in steps:
        print("=" * 72)
        print(label)
        print("=" * 72)
        main(argv)
        print()
    data = root / "data" / "plant_5min.csv"
    if data.exists():
        print("=" * 72)
        print("4) 보정 + 외삽 검증")
        print("=" * 72)
        main(["calibrate", str(root / "calibration.yaml")])
    else:
        print(f"[건너뜀] 가상 데이터가 없습니다. 먼저 실행하세요:\n"
              f"  python {root / 'make_synthetic_data.py'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pf", description="pforecast - 물리 지배방정식 기반 모델링/예측 툴")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="모델 구조 해석 (방정식/미지수/BLT 진단)")
    c.add_argument("model")
    c.add_argument("--builder", default="build")
    c.add_argument("--no-dim-check", action="store_true", help="차원 검사 생략")
    c.set_defaults(func=cmd_check)

    c = sub.add_parser("solve", help="설계점 정상상태 풀이")
    c.add_argument("model")
    c.add_argument("--builder", default="build")
    c.add_argument("--params", help="보정 파라미터 YAML")
    c.add_argument("--csv", help="결과 CSV 저장 경로")
    c.set_defaults(func=cmd_solve)

    c = sub.add_parser("calibrate", help="실측으로 물리 파라미터 보정 + 외삽 검증")
    c.add_argument("config")
    c.add_argument("-o", "--out", help="HTML 리포트 경로")
    c.add_argument("--max-rows", type=int, help="보정에 쓸 최대 행 수")
    c.set_defaults(func=cmd_calibrate)

    c = sub.add_parser("closures", help="구성방정식 후보 추천 (학습 구간만, 고르는 것은 사람)")
    c.add_argument("config", help="pf calibrate 설정 YAML")
    c.add_argument("--target", help="채점할 관측 (기본: baseline.target)")
    c.add_argument("--slot", action="append", help="이 슬롯만 비교 (예: SCR.eta). 여러 번 줄 수 있음")
    c.add_argument("--max-rows", type=int, default=60, help="보정 행 수 (기본 60)")
    c.set_defaults(func=cmd_closures)

    c = sub.add_parser("improve", help="개선 피드백: 학습 구간 안에서 보정 설계 비교 + 다음에 고칠 것")
    c.add_argument("config")
    c.add_argument("-o", "--out", help="결과 마크다운 경로")
    c.add_argument("--max-rows", type=int, help="보정에 쓸 최대 행 수")
    c.add_argument("--folds", type=int, default=2, help="내부 전진 폴드 수 (기본 2)")
    c.add_argument("--fold-days", type=float, default=30.0, help="폴드 하나의 검증 일수")
    c.add_argument("--no-drift", action="store_true", help="파라미터 시변성 계산 생략")
    c.set_defaults(func=cmd_improve)

    c = sub.add_parser("run", help="what-if 시나리오 실행")
    c.add_argument("scenario")
    c.add_argument("-o", "--out", help="HTML 리포트 경로")
    c.add_argument("--csv", help="케이스 결과 CSV 경로")
    c.add_argument("--params", help="보정 파라미터 YAML")
    c.set_defaults(func=cmd_run)

    c = sub.add_parser("project", help="앱에서 저장한 프로젝트 목록 / 같은 설정으로 다시 검증")
    c.add_argument("action", choices=["list", "run"])
    c.add_argument("project", nargs="?", help="프로젝트 이름 또는 projects/*.yaml 경로 (run)")
    c.add_argument("--root", default=".", help="작업 폴더")
    c.set_defaults(func=cmd_project)

    c = sub.add_parser("forecast", help="프로젝트의 모델로 미래 예측 (계획 CSV 또는 최근 패턴으로 만든 계획)")
    c.add_argument("project", help="프로젝트 이름 또는 projects/*.yaml 경로")
    c.add_argument("--plan", help="계획 CSV (학습 CSV 와 같은 입력 컬럼). 없으면 최근 패턴으로 만든다")
    c.add_argument("--days", type=float, default=14, help="만들 계획의 길이 (일)")
    c.add_argument("--base-days", type=float, default=7, help="반복할 최근 패턴 길이 (일)")
    c.add_argument("--step", default="1h", help="계획 간격 (5min, 1h, 1D ...)")
    c.add_argument("--adjust", action="append", help="입력 조정: 컬럼=scale:1.1 | 컬럼=add:2 | 컬럼=set:12")
    c.add_argument("--history", choices=["all", "train"], default="all", help="학습에 쓸 과거 (기본: 전체)")
    c.add_argument("--no-band", action="store_true", help="설비 상태 변동 폭 생략 (빠름)")
    c.add_argument("--root", default=".", help="작업 폴더")
    c.set_defaults(func=cmd_forecast)

    c = sub.add_parser("serve", help="로컬 웹 앱 실행 (데이터 → 분석 → 물리모델 → 시나리오)")
    c.add_argument("--root", default=".", help="작업 폴더 (CSV/모델을 찾을 위치)")
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--port", type=int, default=8765)
    c.add_argument("--no-browser", action="store_true")
    c.set_defaults(func=cmd_serve)

    c = sub.add_parser("analyze", help="타깃 컬럼을 지정해 데이터 주도 분석 + XAI 대시보드")
    c.add_argument("config")
    c.add_argument("-o", "--out", help="HTML 대시보드 경로")
    c.add_argument("--target", help="타깃 컬럼 (설정 파일을 덮어씀)")
    c.add_argument("--steady-only", action="store_true", help="준정상 구간만 사용")
    c.set_defaults(func=cmd_analyze)

    c = sub.add_parser("select", help="구성방정식 후보를 비교해 고른다")
    c.add_argument("config")
    c.add_argument("-o", "--out", help="HTML 리포트 경로")
    c.add_argument("--max-rows", type=int, help="후보별 보정에 쓸 최대 행 수")
    c.add_argument("--save", help="비교 결과를 저장할 경로 (리포트 재생성용)")
    c.add_argument("--load", help="저장된 결과로 리포트만 다시 생성")
    c.set_defaults(func=cmd_select)

    c = sub.add_parser("equations", help="조립된 방정식을 이름과 함께 출력")
    c.add_argument("model")
    c.add_argument("--builder", default="build")
    c.add_argument("--component", help="이 컴포넌트의 방정식만")
    c.add_argument("--dims", action="store_true", help="각 식의 차원도 표시")
    c.add_argument("--full", action="store_true", help="긴 식도 자르지 않음")
    c.add_argument("--no-dim-check", action="store_true")
    c.set_defaults(func=cmd_equations)

    c = sub.add_parser("demo", help="NOx 예제를 처음부터 끝까지 실행")
    c.set_defaults(func=cmd_demo)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"파일을 찾을 수 없습니다: {exc}")
        return 2
    except Exception as exc:  # 사용자에게는 스택 대신 원인을 보여준다
        print(f"오류: {type(exc).__name__}: {exc}")
        if "--traceback" in (argv or sys.argv):
            raise
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
