"""분석 사다리: 데이터만으로 되는 것부터 물리 구조가 필요한 것까지.

사용자가 타깃 컬럼 하나만 지정해도 **어디까지는 자동으로 되고 어디부터는 안 되는지**
가 분명해야 한다. 이 모듈은 단계마다 무엇을 얻었고 무엇이 막혔는지를 기록한다.

    L0  데이터 프로파일링        항상 (자동)
    L1  대리모델 + XAI           타깃만 있으면 (자동)   <- 학습 범위 안에서만 유효
    L2  차원 해석 (무차원군)     각 컬럼의 단위가 필요
    L3  보존식 탐지              각 컬럼의 단위가 필요   <- 데이터가 말해 주는 지배방정식
    L4  구조 물리 모델           계통 토폴로지가 필요    <- 사람이 줘야 한다
    L5  하이브리드               L4 + 잔차 보정

**L4 를 자동화할 수 없다는 점이 이 설계의 핵심 제약이다.** 컬럼 이름과 숫자만으로는
"이건 스크러버이고 팬이 뒤에 달려 있다"를 알 수 없다. 대신 L0~L3 이 그 앞까지를
전부 자동으로 끌어주고, 무엇을 더 주면 L4 로 갈 수 있는지 구체적으로 알려준다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..core.units import dim_of, from_si, to_si
from .dimensional import Balance, PiGroup, find_balances, pi_groups
from .profile import DatasetProfile, Envelope, profile_dataset, steady_state_mask
from .surrogate import Improvement, SurrogateResult, fit_surrogate, search_improvement


@dataclass
class Rung:
    """사다리 한 칸의 결과."""

    level: str
    name: str
    status: str            # "완료" | "부분" | "막힘" | "생략"
    finding: str = ""
    blocker: str = ""
    how_to_unblock: str = ""


@dataclass
class AnalysisConfig:
    name: str = "analysis"
    csv: str = ""
    timestamp: str = "timestamp"
    resample: str | None = None
    target: str = ""
    target_unit: str = ""
    units: dict[str, str] = field(default_factory=dict)
    features: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    controllable: list[str] = field(default_factory=list)
    objective: str = "minimize"
    train_fraction: float = 0.6
    max_lag: int = 12
    steady_only: bool = False
    balance_tol: float = 0.03
    report_out: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "AnalysisConfig":
        root = Path(path).parent
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data = d.get("data") or {}
        tgt = d.get("target") or {}
        if isinstance(tgt, str):
            tgt = {"column": tgt}

        def rel(v):
            if not v:
                return v
            p = Path(v)
            return str(p if p.is_absolute() or p.exists() else root / p)

        units = dict(d.get("units") or {})
        tm_path = rel(d.get("units_from_tagmap"))
        if tm_path and Path(tm_path).exists():
            from ..data import TagMap
            tm = TagMap.load(tm_path)
            for e in list(tm.inputs) + list(tm.observations):
                units.setdefault(e.tag, e.unit)
        split = d.get("split") or {}
        return cls(
            name=d.get("name", Path(path).stem), csv=rel(data.get("csv", "")),
            timestamp=data.get("timestamp", "timestamp"), resample=data.get("resample"),
            target=tgt.get("column", ""), target_unit=tgt.get("unit", units.get(tgt.get("column", ""), "")),
            units=units, features=d.get("features") or [], exclude=d.get("exclude") or [],
            controllable=d.get("controllable") or [],
            objective=d.get("objective", "minimize"),
            train_fraction=float(split.get("train_fraction", 0.6)),
            max_lag=int(d.get("max_lag", 12)),
            steady_only=bool(d.get("steady_only", False)),
            balance_tol=float(d.get("balance_tol", 0.03)),
            report_out=rel(d.get("report")),
        )


@dataclass
class AnalysisResult:
    config: AnalysisConfig
    df: pd.DataFrame
    profile: DatasetProfile
    rungs: list[Rung] = field(default_factory=list)
    balances: list[Balance] = field(default_factory=list)
    pis: list[PiGroup] = field(default_factory=list)
    surrogate: SurrogateResult | None = None
    holdout_outside: float = np.nan
    holdout_rmse: float = np.nan
    improvement: Improvement | None = None
    features: list[str] = field(default_factory=list)

    def ladder_table(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "단계": r.level, "내용": r.name, "상태": r.status,
            "결과": r.finding, "막힌 이유": r.blocker, "풀려면": r.how_to_unblock,
        } for r in self.rungs])

    def headline(self) -> list[str]:
        """표를 보기 전에 읽어야 할 문장들. 오독을 막는 것이 목적이다."""
        out: list[str] = []
        s = self.surrogate
        if s is None:
            return ["대리모델을 만들지 못했습니다. 사다리 표의 막힌 이유를 확인하세요."]

        if np.isfinite(s.skill) and s.skill <= 0.05:
            out.append(
                f"**{self.config.target} 을 설명하는 신호가 거의 없습니다.** 교차검증 RMSE 가 "
                f"그냥 평균으로 예측한 것보다 {max(s.skill,0)*100:.0f}% 나을 뿐입니다. "
                "피처가 부족하거나 타깃이 노이즈에 가깝습니다.")
        else:
            out.append(
                f"대리모델이 {self.config.target} 변동의 **{s.r2_cv*100:.0f}%** 를 설명합니다 "
                f"(시간블록 교차검증 R², RMSE {s.rmse_cv:.4g} {s.unit}).")

        # 영향인자는 반드시 '묶음' 기준으로 말한다. 개별 순위는 상관에 따라 임의로 갈린다.
        groups = [c for c in s.clusters if c.is_group]
        singles = [c for c in s.clusters if not c.is_group]
        if s.clusters:
            top = s.clusters[0]
            if top.is_group:
                out.append(
                    f"**영향의 {top.importance_pct:.0f}% 가 서로 구분되지 않는 한 덩어리에 "
                    f"몰려 있습니다** ({len(top.members)}개 변수, 내부 상관 최대 "
                    f"{top.max_internal_corr:.4f}). 이 안에서 누가 원인인지는 **이 데이터로 "
                    "알 수 없습니다** — 개별 순열 중요도 순위를 인과로 읽으면 안 됩니다.")
                out.append(
                    "이 지점이 물리 모델이 필요한 이유입니다. 지배방정식에는 각 인자가 "
                    "**구조적으로 다른 자리**에 들어가므로(예: 장비군별 배출계수는 각자의 "
                    "질량수지에 들어간다) 상관이 높아도 분리가 가능합니다. 그래도 분리가 "
                    "안 되면 캘리브레이션의 식별성 진단이 그렇다고 말해 줍니다.")
            else:
                names = ", ".join(f"**{c.label}**({c.importance_pct:.0f}%)"
                                  for c in s.clusters[:3])
                out.append(f"영향이 큰 순서: {names}. 서로 얽히지 않아 개별 해석이 가능합니다.")
        if groups and singles:
            sep = ", ".join(f"{c.label}({c.importance_pct:.1f}%)" for c in singles[:3])
            out.append(f"반면 {sep} 는 독립적이라 개별 효과를 믿을 수 있습니다.")

        conserved = [b for b in self.balances if b.is_conservation and b.confidence == "확실"]
        if conserved:
            out.append(
                f"데이터에서 **보존식 {len(conserved)}개**를 찾았습니다: "
                + "; ".join(b.formula() for b in conserved[:2])
                + ". 이건 회귀가 아니라 물리 제약이라 외삽 구간에서도 성립합니다.")

        if np.isfinite(self.holdout_outside) and self.holdout_outside > 0.05:
            out.append(
                f"검증 구간의 **{self.holdout_outside*100:.0f}% 가 학습 포락선 밖**입니다. "
                "그 구간의 대리모델 예측에는 근거가 없습니다.")

        imp = self.improvement
        if imp is not None:
            ctrl_share = sum(c.importance_pct for c in s.clusters
                             if set(c.members) & set(self.config.controllable))
            if ctrl_share < 5.0:
                out.append(
                    f"개선 탐색이 답을 내긴 했지만, 제어변수의 영향력이 전체의 "
                    f"{ctrl_share:.1f}% 뿐입니다. **이 제안은 믿지 마세요** — 약한 신호에서 "
                    "나온 방향은 부호가 뒤집히기도 합니다. 물리 모델로 교차검증해야 합니다.")
            elif not imp.feasible:
                out.append("개선안이 학습 포락선을 벗어납니다. 대리모델로는 답할 수 없습니다.")
        return out


def _load_frame(cfg: AnalysisConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv)
    if cfg.timestamp in df.columns:
        df[cfg.timestamp] = pd.to_datetime(df[cfg.timestamp], errors="coerce")
        df = df.dropna(subset=[cfg.timestamp]).set_index(cfg.timestamp).sort_index()
    num = df.select_dtypes(include=[np.number])
    if cfg.resample and isinstance(num.index, pd.DatetimeIndex):
        num = num.resample(cfg.resample).mean()
    return num


def run_analysis(cfg: AnalysisConfig, verbose: bool = True) -> AnalysisResult:
    df = _load_frame(cfg)
    if cfg.target not in df.columns:
        raise KeyError(f"타깃 컬럼 {cfg.target!r} 이 없습니다. 가능: {list(df.columns)[:15]}")

    prof = profile_dataset(df, cfg.units)
    res = AnalysisResult(config=cfg, df=df, profile=prof)

    # ---- L0 프로파일링 ----------------------------------------------------
    res.rungs.append(Rung(
        "L0", "데이터 프로파일링", "완료",
        finding=(f"{prof.n_rows:,}행, 샘플링 {prof.interval_s:.0f}초, "
                 f"사용 가능 컬럼 {len(prof.usable_columns())}/{len(prof.columns)}개, "
                 f"준정상 구간 {prof.steady_fraction*100:.0f}%"),
    ))

    features = cfg.features or [c for c in prof.usable_columns() if c != cfg.target]
    features = [f for f in features if f not in set(cfg.exclude) and f != cfg.target]
    res.features = features
    if not features:
        res.rungs.append(Rung("L1", "대리모델 + XAI", "막힘",
                              blocker="쓸 수 있는 피처가 없습니다",
                              how_to_unblock="상수/중복 컬럼을 빼고 다시 확인하세요"))
        return res

    work = df
    if cfg.steady_only:
        mask = steady_state_mask(df, [cfg.target] + features)
        work = df[mask]
        if verbose:
            print(f"준정상 구간만 사용: {len(work)}/{len(df)}행")

    # ---- L1 대리모델 + XAI -----------------------------------------------
    n = len(work)
    cut = int(n * cfg.train_fraction)
    train, test = work.iloc[:cut], work.iloc[cut:]
    if verbose:
        print(f"L1 대리모델: 피처 {len(features)}개, 학습 {len(train)}행 / 검증 {len(test)}행")
    try:
        sur = fit_surrogate(train, cfg.target, features, unit=cfg.target_unit,
                            max_lag=cfg.max_lag)
        res.surrogate = sur
        if len(test) > 10:
            res.holdout_outside = sur.envelope.outside_fraction(test)
            Xte = test[features].to_numpy(dtype=float)
            yte = test[cfg.target].to_numpy(dtype=float)
            ok = np.all(np.isfinite(Xte), axis=1) & np.isfinite(yte)
            if ok.sum() > 5:
                pred = sur.model.predict(Xte[ok])
                res.holdout_rmse = float(np.sqrt(np.mean((pred - yte[ok]) ** 2)))
        res.rungs.append(Rung(
            "L1", "대리모델 + XAI", "완료",
            # 1위 인자는 반드시 **묶음** 기준으로 적는다. 개별 순위는 상관에 따라 갈린다.
            finding=(f"{sur.model_name}, 교차검증 R²={sur.r2_cv:.3f}, "
                     f"RMSE={sur.rmse_cv:.4g} {sur.unit}, "
                     f"1위 인자묶음 {sur.clusters[0].label}"
                     f"({sur.clusters[0].importance_pct:.0f}%)"
                     if sur.clusters else ""),
            blocker="학습 포락선 밖에서는 근거가 없습니다",
            how_to_unblock="외삽이 필요하면 L4 구조 모델로 가야 합니다"))
    except Exception as exc:
        res.rungs.append(Rung("L1", "대리모델 + XAI", "막힘", blocker=str(exc)))

    # ---- L2 차원 해석 -----------------------------------------------------
    known = {c: u for c, u in cfg.units.items() if c in df.columns and u}
    missing = [c for c in [cfg.target] + features if c not in known]
    if len(known) < 2:
        res.rungs.append(Rung(
            "L2", "차원 해석 (무차원군)", "막힘",
            blocker=f"단위를 아는 컬럼이 {len(known)}개뿐입니다",
            how_to_unblock="설정의 units: 또는 units_from_tagmap: 으로 컬럼 단위를 주세요"))
    else:
        try:
            res.pis = pi_groups(known, target=cfg.target if cfg.target in known else None)
        except Exception:
            res.pis = []
        nontrivial = [g for g in res.pis
                      if len(g.exponents) > 1 and not g.is_trivial_ratio(known)]
        status = "완료" if nontrivial else "부분"
        res.rungs.append(Rung(
            "L2", "차원 해석 (무차원군)", status,
            finding=(f"무차원군 {len(res.pis)}개 중 물리적으로 의미 있는 것 "
                     f"{len(nontrivial)}개 (나머지는 같은 차원끼리의 단순 비율)"
                     if res.pis else "무차원군 없음"),
            blocker=("변수 수가 차원 수보다 충분히 많지 않습니다"
                     if not nontrivial else ""),
            how_to_unblock=("같은 현상을 지배하는 변수를 더 넣으면 무차원군이 생깁니다"
                            if not nontrivial else "")))

    # ---- L3 보존식 탐지 ---------------------------------------------------
    if len(known) < 2:
        res.rungs.append(Rung("L3", "보존식 탐지", "막힘",
                              blocker="단위 정보가 없습니다",
                              how_to_unblock="units: 를 채우세요"))
    else:
        res.balances = find_balances(df, known, rel_tol=cfg.balance_tol)
        sure = [b for b in res.balances if b.confidence == "확실"]
        res.rungs.append(Rung(
            "L3", "보존식 탐지", "완료" if sure else "부분",
            finding=(f"{len(res.balances)}개 발견 (확실 {len(sure)}개): "
                     + "; ".join(b.formula() for b in sure[:2]) if res.balances
                     else "같은 차원 컬럼 묶음에서 성립하는 선형 관계 없음"),
            blocker="" if sure else "같은 차원의 컬럼이 2개 이상 있어야 찾을 수 있습니다",
            how_to_unblock=("" if sure else
                            "지류별 유량계처럼 합이 맞아야 하는 계측을 함께 넣으세요")))

    # ---- L4 구조 물리 모델 (자동 불가) ------------------------------------
    res.rungs.append(Rung(
        "L4", "구조 물리 모델", "막힘",
        blocker="계통 토폴로지(무엇이 무엇에 연결되는가)는 데이터에서 유도할 수 없습니다",
        how_to_unblock=("계통도와 설비 제원을 주면 `pf check` 로 넘어갑니다. "
                        "컴포넌트 목록·연결 관계·설계 제원 세 가지가 필요합니다")))
    res.rungs.append(Rung(
        "L5", "하이브리드 (물리 + 잔차 ML)", "막힘",
        blocker="L4 가 있어야 시작할 수 있습니다",
        how_to_unblock="L4 완료 후 잔차를 이 대리모델로 학습시키고 포락선 밖에서 감쇠시킵니다"))

    # ---- 개선 탐색 --------------------------------------------------------
    if res.surrogate is not None and cfg.controllable:
        res.improvement = search_improvement(
            res.surrogate, train, cfg.controllable, direction=cfg.objective)
    return res
