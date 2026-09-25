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
    out = args.out or cfg.report_out
    if out:
        p = calibration_report(cfg, res, out)
        print(f"\n리포트: {p}")
    if cfg.params_out:
        print(f"보정 파라미터: {cfg.params_out}")
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

    c = sub.add_parser("run", help="what-if 시나리오 실행")
    c.add_argument("scenario")
    c.add_argument("-o", "--out", help="HTML 리포트 경로")
    c.add_argument("--csv", help="케이스 결과 CSV 경로")
    c.add_argument("--params", help="보정 파라미터 YAML")
    c.set_defaults(func=cmd_run)

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
