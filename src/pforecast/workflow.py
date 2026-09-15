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
from .calib.estimator import _metrics
from .core.units import from_si
from .data import TagMap, read_csv
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


def load_dataset(cfg: WorkflowConfig):
    tm = TagMap.load(cfg.tagmap)
    inputs, obs = read_csv(cfg.csv, tm)
    df = inputs.join(obs, how="inner").dropna()
    train = df.query(cfg.train_query) if cfg.train_query else df
    test = df.query(cfg.test_query) if cfg.test_query else df
    return tm, df, train, test, list(inputs.columns), list(obs.columns)


def _predict(model, df, inputs_cols, obs_cols, expansion):
    rows = build_param_rows(model, df[inputs_cols], base_p=model.p0(), expansion=expansion)
    return simulate(model, rows, obs_cols, index=df.index)


def run_workflow(cfg: WorkflowConfig, verbose: bool = True) -> WorkflowResult:
    """보정 -> 외삽 검증 -> ML 기준모델 비교까지 한 번에."""
    system = load_model(cfg.model)
    model = system.compile()
    model.build()
    if cfg.params_in and Path(cfg.params_in).exists():
        apply_params(model, cfg.params_in)

    tm, df, train, test, inputs_cols, obs_cols = load_dataset(cfg)
    obs_cols = [c for c in (cfg.observations or obs_cols) if c in df.columns]
    expansion = tm.expansion()

    if verbose:
        print(f"데이터 {len(df)}행 | 학습 {len(train)}행 | 검증 {len(test)}행")

    res = WorkflowResult(model=model, tagmap=tm, train=train, test=test,
                         inputs_cols=inputs_cols, obs_cols=obs_cols)

    if cfg.params:
        spec = CalibrationSpec(params=cfg.params, observations=obs_cols,
                               sigmas=tm.sigmas(), max_rows=cfg.max_rows,
                               prior_weight=cfg.prior_weight)
        if verbose:
            print(f"보정 중: {len(cfg.params)}개 파라미터, 최대 {cfg.max_rows}행 ...")
        res.calibration = calibrate(model, train[inputs_cols], train[obs_cols], spec,
                                    expansion=expansion, verbose=False)
        if cfg.params_out:
            save_params(model, cfg.params_out, cfg.params,
                        meta={"config": cfg.name, "train_rows": len(train),
                              "train_query": cfg.train_query})

    res.physics_train = _predict(model, train, inputs_cols, obs_cols, expansion).values
    res.physics_test = _predict(model, test, inputs_cols, obs_cols, expansion).values

    # 같은 학습 데이터로 학습한 데이터 기반 기준모델
    if cfg.baseline_target and cfg.baseline_features:
        feats = [f for f in cfg.baseline_features if f in df.columns]
        bl = PolyRidgeBaseline(degree=cfg.baseline_degree)
        bl.fit(train[feats].to_numpy(), train[cfg.baseline_target].to_numpy(), feats)
        res.baseline = bl
        res.baseline_train = bl.predict(train[feats].to_numpy())
        res.baseline_test = bl.predict(test[feats].to_numpy())

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

    add("물리모델", "학습(보정)", res.physics_train[tgt], res.train[tgt])
    add("물리모델", "검증(외삽)", res.physics_test[tgt], res.test[tgt])
    if res.baseline is not None:
        add("ML 기준모델", "학습(보정)", res.baseline_train, res.train[tgt])
        add("ML 기준모델", "검증(외삽)", res.baseline_test, res.test[tgt])
    return pd.DataFrame(rows)
