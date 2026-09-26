"""엔드투엔드 워크플로: 데이터 -> 보정 -> 검증 -> 리포트.

CLI 와 노트북이 공유하는 상위 조립 코드. 여기서 '물리모델 vs 데이터모델' 비교를
같은 학습/검증 분할로 공정하게 수행한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .calib import CalibrationSpec, PolyRidgeBaseline, build_param_rows, calibrate, simulate
from .calib.diagnostics import residual_changepoints
from .calib.estimator import _metrics
from .core.units import from_si
from .data import TagMap, read_csv_report
from .data.split import check_split
from .params import apply_params, save_params
from .scenario import load_model


@dataclass
class WorkflowConfig:
    name: str = "calibration"
    model: Any = None
    csv: str = ""
    tagmap: str = ""
    train_query: str = ""
    test_query: str = ""
    params: list[str] = field(default_factory=list)
    max_rows: int = 120
    prior_weight: float = 0.05
    observations: list[str] = field(default_factory=list)
    baseline_features: list[str] = field(default_factory=list)
    baseline_target: str = ""
    baseline_degree: int = 2
    params_in: str | None = None
    params_out: str | None = None
    report_out: str | None = None
    clean: bool = True                 # 값 이상(교정 창·고착·스파이크)을 결측으로
    loss: str = "linear"               # 보정 손실: linear | soft_l1 | huber
    f_scale: float = 3.0
    diagnose: bool = True              # 잔차 변화점 진단

    @classmethod
    def load(cls, path: str | Path) -> "WorkflowConfig":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data = d.get("data") or {}
        split = d.get("split") or {}
        cal = d.get("calibrate") or {}
        base = d.get("baseline") or {}
        root = Path(path).parent

        def rel(v):
            if not v:
                return v
            p = Path(v)
            return str(p if p.is_absolute() or p.exists() else root / p)

        model = d.get("model")
        if isinstance(model, dict):
            model = {k: (rel(v) if k in ("python", "yaml") else v) for k, v in model.items()}
        return cls(
            name=d.get("name", Path(path).stem),
            model=model,
            csv=rel(data.get("csv", "")),
            tagmap=rel(data.get("tagmap", "")),
            train_query=split.get("train", ""),
            test_query=split.get("test", ""),
            params=cal.get("params") or [],
            max_rows=int(cal.get("max_rows", 120)),
            prior_weight=float(cal.get("prior_weight", 0.05)),
            observations=cal.get("observations") or [],
            baseline_features=base.get("features") or [],
            baseline_target=base.get("target", ""),
            baseline_degree=int(base.get("degree", 2)),
            params_in=rel(d.get("params_in")),
            params_out=rel(d.get("params_out")),
            report_out=rel(d.get("report")),
            clean=bool(data.get("clean", True)),
            loss=str(cal.get("loss", "linear")),
            f_scale=float(cal.get("f_scale", 3.0)),
            diagnose=bool(d.get("diagnose", True)),
        )


@dataclass
class WorkflowResult:
    model: Any
    tagmap: TagMap
    train: pd.DataFrame
    test: pd.DataFrame
    inputs_cols: list[str]
    obs_cols: list[str]
    calibration: Any = None
    physics_train: pd.DataFrame | None = None
    physics_test: pd.DataFrame | None = None
    baseline: PolyRidgeBaseline | None = None
    baseline_test: np.ndarray | None = None
    baseline_train: np.ndarray | None = None
    comparison: pd.DataFrame | None = None
    data_report: object = None                 # DataReport (파일 정리 + 값 정제 내역)
    changepoints: list = field(default_factory=list)   # 잔차 변화점
    clipped: dict = field(default_factory=dict)        # 물리 범위 밖이라 자른 입력
    split: object = None                       # SplitCheck — 미래인가, 외삽인가
    #: 추가 ML 기준모델 {이름: (학습 예측, 검증 예측)} — 같은 입력·같은 학습 구간
    extra_baselines: dict = field(default_factory=dict)


def _split_masks(cfg: WorkflowConfig, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    tr = (frame.index.isin(frame.query(cfg.train_query).index) if cfg.train_query
          else np.ones(len(frame), dtype=bool))
    te = (frame.index.isin(frame.query(cfg.test_query).index) if cfg.test_query
          else np.ones(len(frame), dtype=bool))
    te = te & ~tr
    return {"학습": tr, "검증": te, "기타": ~(tr | te)}


def load_dataset(cfg: WorkflowConfig, return_report: bool = False):
    tm = TagMap.load(cfg.tagmap)
    # 분할을 정제 전에 정하고, 값 이상 진단은 분할마다 따로 한다 (누수 방지)
    inputs, obs, report = read_csv_report(cfg.csv, tm, clean=cfg.clean,
                                          split=lambda f: _split_masks(cfg, f))
    df = inputs.join(obs, how="inner")
    # 입력은 전부 있어야 모델을 풀 수 있다. 관측은 하나라도 있으면 남긴다.
    df = df[df[list(inputs.columns)].notna().all(axis=1) & df[list(obs.columns)].notna().any(axis=1)]
    train = df.query(cfg.train_query) if cfg.train_query else df
    test = df.query(cfg.test_query) if cfg.test_query else df
    out = (tm, df, train, test, list(inputs.columns), list(obs.columns))
    return out + (report,) if return_report else out


def _predict(model, df, inputs_cols, obs_cols, expansion, clip_report=None):
    rows = build_param_rows(model, df[inputs_cols], base_p=model.p0(), expansion=expansion,
                            clip_report=clip_report)
    return simulate(model, rows, obs_cols, index=df.index)


def run_workflow(cfg: WorkflowConfig, verbose: bool = True) -> WorkflowResult:
    """보정 -> 외삽 검증 -> ML 기준모델 비교까지 한 번에."""
    system = load_model(cfg.model)
    model = system.compile()
    model.build()
    if cfg.params_in and Path(cfg.params_in).exists():
        apply_params(model, cfg.params_in)

    tm, df, train, test, inputs_cols, obs_cols, report = load_dataset(cfg, return_report=True)
    obs_cols = [c for c in (cfg.observations or obs_cols) if c in df.columns]
    expansion = tm.expansion()

    split = check_split(train, test, inputs_cols)
    if verbose:
        print(f"데이터 {len(df)}행 | 학습 {len(train)}행 | 검증 {len(test)}행", flush=True)
        for line in split.lines():
            print("  ▶ " + line)
        for w in split.warnings():
            print("  !! " + w.replace("**", ""))
        for line in report.lines():
            print("  · " + line.replace("**", ""))

    res = WorkflowResult(model=model, tagmap=tm, train=train, test=test,
                         inputs_cols=inputs_cols, obs_cols=obs_cols, data_report=report,
                         split=split)

    if cfg.params:
        spec = CalibrationSpec(params=cfg.params, observations=obs_cols,
                               sigmas=tm.sigmas(), max_rows=cfg.max_rows,
                               prior_weight=cfg.prior_weight, loss=cfg.loss,
                               f_scale=cfg.f_scale)
        if verbose:
            print(f"보정 중: {len(cfg.params)}개 파라미터, 최대 {cfg.max_rows}행 ...")
        res.calibration = calibrate(model, train[inputs_cols], train[obs_cols], spec,
                                    expansion=expansion, verbose=False)
        if cfg.params_out:
            save_params(model, cfg.params_out, cfg.params,
                        meta={"config": cfg.name, "train_rows": len(train),
                              "train_query": cfg.train_query})

    if verbose:
        print(f"물리모델 예측 중: 학습 {len(train)}행 + 검증 {len(test)}행 ...", flush=True)
    res.physics_train = _predict(model, train, inputs_cols, obs_cols, expansion,
                                 res.clipped).values
    res.physics_test = _predict(model, test, inputs_cols, obs_cols, expansion,
                                res.clipped).values

    # 잔차에 남은 계단 = 모델이 모르는 변화 (센서 교체, 레시피, 오염/세정 ...)
    if cfg.diagnose:
        both = pd.concat([res.physics_train, res.physics_test])
        both = both[~both.index.duplicated()].sort_index()
        meas = df.loc[both.index, obs_cols]
        units = {e.primary: e.unit for e in tm.observations}
        resid = pd.DataFrame({c: [from_si(m, units.get(c, "1")) - from_si(q, units.get(c, "1"))
                                  for m, q in zip(meas[c].to_numpy(), both[c].to_numpy())]
                              for c in obs_cols}, index=both.index)
        ins = df.loc[both.index, inputs_cols]
        # 계측 불확도의 절반보다 작은 계단은 실무적으로 의미가 없다
        floor = {e.primary: 0.5 * e.sigma for e in tm.observations if e.sigma}
        res.changepoints = residual_changepoints(resid, ins, units, min_shift=floor)

    # 같은 학습 데이터로 학습한 데이터 기반 기준모델
    if cfg.baseline_target and cfg.baseline_features:
        feats = [f for f in cfg.baseline_features if f in df.columns]
        bl = PolyRidgeBaseline(degree=cfg.baseline_degree)
        bl.fit(train[feats].to_numpy(), train[cfg.baseline_target].to_numpy(), feats)
        res.baseline = bl
        res.baseline_train = bl.predict(train[feats].to_numpy())
        res.baseline_test = bl.predict(test[feats].to_numpy())
        # 현업에서 가장 흔한 쪽(트리 앙상블)도 같이 둔다. 학습 범위 밖에서 예측이 평평해진다.
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            ok = np.all(np.isfinite(train[feats].to_numpy()), axis=1) & \
                np.isfinite(train[cfg.baseline_target].to_numpy())
            gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                               min_samples_leaf=20, l2_regularization=1.0,
                                               random_state=0)
            gb.fit(train[feats].to_numpy()[ok], train[cfg.baseline_target].to_numpy()[ok])
            res.extra_baselines["ML 부스팅"] = (gb.predict(train[feats].to_numpy()),
                                               gb.predict(test[feats].to_numpy()))
        except ImportError:
            pass

    res.comparison = _comparison_table(cfg, res)
    return res


def _comparison_table(cfg: WorkflowConfig, res: WorkflowResult) -> pd.DataFrame:
    tgt = cfg.baseline_target or (res.obs_cols[0] if res.obs_cols else None)
    if tgt is None:
        return pd.DataFrame()
    unit = next((e.unit for e in res.tagmap.observations if e.primary == tgt), "1")
    conv = lambda a: np.array([from_si(v, unit) for v in np.asarray(a, dtype=float)])  # noqa: E731
    rows = []

    def add(model_name, split, pred, meas):
        m = _metrics(conv(pred), conv(meas))
        rows.append({"모델": model_name, "구간": split, "n": m["n"],
                     f"RMSE [{unit}]": m["rmse"], f"MAE [{unit}]": m["mae"],
                     f"편향 [{unit}]": m["bias"], "MAPE [%]": m["mape"], "R2": m["r2"]})

    out = res.split.outside if res.split is not None and res.split.outside is not None \
        else np.zeros(len(res.test), dtype=bool)
    preds = [("물리모델", res.physics_train[tgt].to_numpy(), res.physics_test[tgt].to_numpy())]
    if res.baseline is not None:
        preds.append(("ML 다항(2차)", res.baseline_train, res.baseline_test))
    for name, (p_tr, p_te) in res.extra_baselines.items():
        preds.append((name, p_tr, p_te))
    meas_te = res.test[tgt].to_numpy()
    for name, p_tr, p_te in preds:
        add(name, "학습(보정)", p_tr, res.train[tgt].to_numpy())
        add(name, "검증 전체", p_te, meas_te)
        # 핵심 비교: 학습 운전영역 밖(외삽) 행만. 안쪽 행은 ML 도 근거가 있다.
        if out.any():
            add(name, "검증 · 외삽 행", np.where(out, p_te, np.nan), meas_te)
        if (~out).any() and out.any():
            add(name, "검증 · 내삽 행", np.where(~out, p_te, np.nan), meas_te)
    return pd.DataFrame(rows)
