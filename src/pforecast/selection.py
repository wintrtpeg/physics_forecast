"""구성방정식(closure) 후보 비교 — "어느 물리식이 맞는가"를 데이터로 고른다.

**무엇을 후보로 두는가가 이 기능의 전부다.**

* 후보가 **아닌 것**: 질량·운동량·에너지·화학종 보존, 상태방정식. 이건 공리다.
  이걸 후보로 돌리는 순간 외삽 보증이 사라지고, 그러면 물리모델을 쓸 이유가 없다.
* 후보로 **두는 것**: 구성방정식(closure). 마찰 상관식, 물질전달 상관식, 성능곡선
  형태, 열전달 상관식. 이것들은 원래 경험식이고 형태가 여럿이다.

판정 기준은 넷을 **같이** 본다. 하나로 줄이면 반드시 속는다.

1. **외삽 구간 오차** (1차 기준). 학습에 쓰지 않은 운전영역에서의 RMSE.
   학습 구간 적합도는 파라미터를 늘리면 언제나 좋아지므로 기준이 될 수 없다.
2. **AICc**. 학습 적합도에 파라미터 수 벌점. 복잡도가 값을 하는지 본다.
3. **식별성**. 파라미터가 실제로 결정되었는가 (상관계수·상대표준오차).
4. **물리적 타당성**. 보정값이 경계에 붙었는가 (붙었으면 모델이 틀렸다는 신호다).

1등만 보고 고르지 말 것. 외삽 오차가 비슷한데 파라미터가 적은 쪽이 대개 낫다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .calib import CalibrationSpec, build_param_rows, calibrate, simulate
from .core.units import from_si
from .data import TagMap, read_csv
from .lib import EquationComponent
from .scenario import load_model


@dataclass
class Candidate:
    """비교할 구성방정식 후보 하나."""

    id: str
    component: str | dict | None = None      # 슬롯에 끼울 컴포넌트 사양
    calibrate: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class SelectionConfig:
    name: str = "closure_selection"
    model: Any = None
    csv: str = ""
    tagmap: str = ""
    train_query: str = ""
    test_query: str = ""
    target: str = ""
    slot: str = ""
    common_calibrate: list[str] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    max_rows: int = 100
    prior_weight: float = 0.05
    report_out: str | None = None
    base_dir: str | None = None
    #: 후보들이 서로 갈리는 지점의 변수 (예: 스크러버 액가스비 ``SCR.LG``).
    #: 후보가 구분되지 않을 때 "이 변수를 더 흔들어야 한다"고 알려주는 데 쓴다.
    discriminator: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "SelectionConfig":
        root = Path(path).parent
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data, split, cal = d.get("data") or {}, d.get("split") or {}, d.get("calibrate") or {}

        def rel(v):
            if not v:
                return v
            p = Path(v)
            return str(p if p.is_absolute() or p.exists() else root / p)

        model = d.get("model")
        if isinstance(model, dict):
            model = {k: (rel(v) if k in ("python", "yaml") else v) for k, v in model.items()}
        return cls(
            name=d.get("name", Path(path).stem), model=model,
            csv=rel(data.get("csv", "")), tagmap=rel(data.get("tagmap", "")),
            train_query=split.get("train", ""), test_query=split.get("test", ""),
            target=d.get("target", ""), slot=d.get("slot", ""),
            common_calibrate=cal.get("common") or [],
            observations=cal.get("observations") or [],
            max_rows=int(cal.get("max_rows", 100)),
            prior_weight=float(cal.get("prior_weight", 0.05)),
            discriminator=d.get("discriminator", ""),
            candidates=[Candidate(id=c.get("id", f"cand{i}"), component=c.get("component"),
                                  calibrate=c.get("calibrate") or [],
                                  description=c.get("description", ""))
                        for i, c in enumerate(d.get("candidates") or [])],
            report_out=rel(d.get("report")), base_dir=str(root),
        )


@dataclass
class CandidateResult:
    id: str
    description: str
    n_params: int
    train_rmse: float
    test_rmse: float
    test_bias: float
    test_r2: float
    aicc: float
    worst_corr: float
    max_rel_stderr: float
    at_bound: int
    converged: bool
    unit: str
    calibration: Any = None
    train_pred: pd.Series | None = None
    test_pred: pd.Series | None = None
    message: str = ""
    #: 이 후보에만 있는 구성방정식 파라미터 (공통 파라미터는 뺀다)
    closure_params: list[str] = field(default_factory=list)
    at_bound_names: list[str] = field(default_factory=list)
    discriminator_span: tuple[float, float] | None = None


@dataclass
class SelectionResult:
    config: SelectionConfig
    results: list[CandidateResult]
    train: pd.DataFrame
    test: pd.DataFrame
    tagmap: TagMap
    #: 실무적 동률로 볼 기준 — 계측 불확도의 이 배수 이내면 "차이 없음" 으로 본다.
    practical_fraction: float = 0.2

    @property
    def sigma(self) -> float:
        """대상 관측의 계측 불확도 (표시 단위)."""
        unit = next((e.unit for e in self.tagmap.observations
                     if e.primary == self.config.target), "1")
        si = self.tagmap.sigmas().get(self.config.target)
        return float(from_si(si, unit)) if si is not None else float("nan")

    def table(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        best_aicc = min((r.aicc for r in self.results if np.isfinite(r.aicc)), default=np.nan)
        u = self.results[0].unit
        rows = []
        for r in self.results:
            rows.append({
                "후보": r.id,
                "설명": r.description,
                "파라미터": r.n_params,
                f"학습 RMSE [{u}]": r.train_rmse,
                f"외삽 RMSE [{u}]": r.test_rmse,
                f"외삽 편향 [{u}]": r.test_bias,
                "외삽 R2": r.test_r2,
                "ΔAICc": r.aicc - best_aicc if np.isfinite(r.aicc) else np.nan,
                "최악 상관": r.worst_corr,
                "최대 상대표준오차[%]": r.max_rel_stderr,
                "경계에 붙은 파라미터": r.at_bound,
            })
        df = pd.DataFrame(rows).sort_values(f"외삽 RMSE [{u}]").reset_index(drop=True)
        return df

    def paired_compare(self, a: "CandidateResult", b: "CandidateResult"):
        """외삽 구간에서 두 후보의 절대오차를 **같은 시점끼리** 비교한다.

        5분 시계열 잔차는 강하게 자기상관되어 있다. 그냥 n=6000 으로 표준오차를
        내면 무엇이든 "유의미하게 다르다"고 나온다. 1차 자기상관으로 유효 표본수를
        깎아야 정직한 판정이 된다.
        """
        import numpy as np

        tgt = self.config.target
        unit = a.unit
        if a.test_pred is None or b.test_pred is None or tgt not in self.test:
            return None
        conv = lambda v: np.array([from_si(float(x), unit) for x in np.asarray(v, float)])  # noqa: E731
        meas = conv(self.test[tgt])
        ea = np.abs(conv(a.test_pred) - meas)
        eb = np.abs(conv(b.test_pred) - meas)
        d = eb - ea
        d = d[np.isfinite(d)]
        n = len(d)
        if n < 30:
            return None
        sd = float(np.std(d, ddof=1))
        if sd <= 0:
            return None
        c = np.corrcoef(d[:-1], d[1:])[0, 1] if n > 2 else 0.0
        rho = float(np.clip(c if np.isfinite(c) else 0.0, 0.0, 0.999))
        n_eff = max(n * (1.0 - rho) / (1.0 + rho), 2.0)
        se = sd / np.sqrt(n_eff)
        return float(np.mean(d)), float(se), float(n_eff)

    def pairwise_table(self) -> pd.DataFrame:
        """최선 후보 대비 각 후보의 외삽 오차 차이와 유의성."""
        import numpy as np

        ok = [r for r in self.results if r.converged and np.isfinite(r.test_rmse)]
        if len(ok) < 2:
            return pd.DataFrame()
        ok.sort(key=lambda r: r.test_rmse)
        best = ok[0]
        rows = []
        for r in ok[1:]:
            cmp = self.paired_compare(best, r)
            if cmp is None:
                continue
            diff, se, n_eff = cmp
            rows.append({
                "후보": r.id,
                f"평균 절대오차 차이 [{r.unit}]": diff,
                "표준오차": se,
                "유효 표본수": n_eff,
                "차이/표준오차": diff / se if se > 0 else np.nan,
                "판정": "유의미하게 나쁨" if diff >= 2.0 * se else "구분 안 됨",
            })
        return pd.DataFrame(rows)

    def verdict(self) -> list[str]:
        """표만 보고 놓치기 쉬운 것을 문장으로 짚어 준다."""
        import numpy as np

        out: list[str] = []
        ok = [r for r in self.results if r.converged and np.isfinite(r.test_rmse)]
        if not ok:
            return ["수렴한 후보가 없습니다. 초기값이나 파라미터 경계를 확인하세요."]
        ok.sort(key=lambda r: r.test_rmse)
        best = ok[0]
        u = best.unit
        out.append(f"외삽 오차가 가장 작은 후보는 **{best.id}** ({best.test_rmse:.3g} {u}) 입니다.")

        # ① 실무적으로 구분되는가. 이게 1등 발표보다 훨씬 중요하다.
        #    n 이 수천이면 통계적 유의성은 거의 항상 나온다 - 기준이 될 수 없다.
        sig = self.sigma
        thr = self.practical_fraction * sig if np.isfinite(sig) else np.nan
        spread = max(r.test_rmse for r in ok) - best.test_rmse
        practically_tied = np.isfinite(thr) and spread < thr

        # ② 통계적으로 구분되는가 (자기상관 보정 후). 보조 지표다.
        tied = []
        for r in ok[1:]:
            cmp = self.paired_compare(best, r)
            if cmp is None:
                continue
            diff, se, n_eff = cmp
            if diff < 2.0 * se:
                tied.append((r, diff, se))

        if practically_tied and ok[1:]:
            out.append(
                f"**그러나 이 데이터로는 후보를 고를 수 없습니다.** 최선과 최악의 외삽 "
                f"오차 차이가 {spread:.3g} {u} 로, 계측 불확도 {sig:.3g} {u} 의 "
                f"{spread / sig * 100:.0f}% 에 불과합니다. 하한선 모델까지 포함해 "
                f"전부 사실상 같은 성능입니다.")
            if not tied:
                out.append(
                    "통계적으로는 차이가 '유의미'하게 나옵니다. 하지만 표본이 수천 개면 "
                    "무의미한 차이도 유의미해집니다 — **유의성이 아니라 크기를 보세요.**")
            min_n = min(r.n_params for r in ok)
            simplest = [r for r in ok if r.n_params == min_n]
            if len(simplest) == 1:
                r = simplest[0]
                out.append(
                    f"구분이 안 될 때는 가장 단순한 **{r.id}** (파라미터 {r.n_params}개, "
                    f"외삽 {r.test_rmse:.3g} {u}) 를 쓰는 것이 맞습니다. 근거 없는 복잡도는 "
                    "유지보수 비용만 늘립니다.")
            else:
                names = ", ".join(f"**{r.id}**" for r in simplest)
                out.append(
                    f"파라미터 수도 {min_n}개로 같습니다 ({names}). **데이터로는 우열을 "
                    "가릴 수 없으니 물리적 근거로 고르세요** — 이 데이터는 어느 형태를 "
                    "지지하지도 반박하지도 않습니다. 물질전달 이론에 뿌리가 있는 형태를 "
                    "택하고, 나중에 구분 가능한 데이터가 생기면 다시 판정하세요.")
        if practically_tied and ok[1:]:
            spans = [r.discriminator_span for r in ok if r.discriminator_span]
            if self.config.discriminator and spans:
                lo = min(s[0] for s in spans)
                hi = max(s[1] for s in spans)
                out.append(
                    f"후보들이 갈리려면 `{self.config.discriminator}` 가 충분히 변해야 "
                    f"하는데, 이 데이터에서는 {lo:.4g} ~ {hi:.4g} ({hi/max(lo,1e-30):.2f}배) "
                    f"밖에 움직이지 않았습니다. **의도적으로 이 값을 바꿔 운전한 구간이 "
                    f"있어야** 형태를 고를 수 있습니다.")
            elif self.config.discriminator:
                out.append(
                    f"후보를 구분하려면 `{self.config.discriminator}` 를 의도적으로 바꿔 "
                    f"운전한 데이터가 필요합니다.")
        elif tied:
            names = ", ".join(f"**{r.id}**" for r, _, _ in tied)
            out.append(f"{names} 와는 통계적으로도 구분되지 않습니다 (차이가 표준오차 2배 이내).")

        # 오컴의 면도날
        simpler = [r for r in ok[1:]
                   if r.n_params < best.n_params and r.test_rmse <= best.test_rmse * 1.05]
        if simpler:
            s = min(simpler, key=lambda r: r.n_params)
            out.append(
                f"**{s.id}** 는 파라미터가 {s.n_params}개로 더 적은데 외삽 오차가 "
                f"{s.test_rmse:.3g} {u} 로 5% 이내입니다. 더 단순한 쪽을 택하는 것이 낫습니다.")

        # 과적합 신호
        for r in ok:
            if r.train_rmse < best.train_rmse * 1.02 and r.test_rmse > best.test_rmse * 1.3:
                out.append(
                    f"**{r.id}** 는 학습 구간에서는 {best.id} 와 비슷한데 외삽 오차가 "
                    f"{r.test_rmse / best.test_rmse:.1f}배입니다 - 학습 적합도만 보면 속습니다.")

        # 구성방정식 파라미터의 식별성 (공통 파라미터는 제외하고 본다)
        for r in ok:
            if not r.closure_params:
                continue
            if r.max_rel_stderr > 200 or abs(r.worst_corr) > 0.98:
                out.append(
                    f"**{r.id}** 의 구성방정식 파라미터 {r.closure_params} 가 결정되지 "
                    f"않았습니다 (상대표준오차 {r.max_rel_stderr:.0f}%, 최악 상관 "
                    f"{r.worst_corr:+.3f}). 이 형태를 쓰더라도 계수를 물리적으로 "
                    "해석하면 안 됩니다.")
            if r.at_bound_names:
                out.append(
                    f"**{r.id}** 는 {r.at_bound_names} 가 최적화 경계에 붙었습니다 - "
                    "보통 모델 형태가 데이터와 맞지 않는다는 신호입니다.")
        return out


def _build_model(cfg: SelectionConfig, cand: Candidate):
    system = load_model(cfg.model)
    if cand.component is not None:
        if not cfg.slot:
            raise ValueError("candidates 에 component 를 주려면 slot 을 지정해야 합니다")
        system.replace(cfg.slot, EquationComponent(cfg.slot, cand.component,
                                                   base_dir=cfg.base_dir))
    model = system.compile()
    model.build()
    return model


def _rmse(pred, meas, unit) -> tuple[float, float, float]:
    p = np.array([from_si(float(v), unit) for v in np.asarray(pred, float)])
    m = np.array([from_si(float(v), unit) for v in np.asarray(meas, float)])
    ok = np.isfinite(p) & np.isfinite(m)
    if ok.sum() < 2:
        return np.nan, np.nan, np.nan
    err = p[ok] - m[ok]
    ss_tot = float(np.sum((m[ok] - m[ok].mean()) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else np.nan
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(err)), r2


def _aicc(pred_df: pd.DataFrame, meas_df: pd.DataFrame, sigmas: dict[str, float],
          k: int) -> float:
    """분산 미지 형태의 AICc. 관측 여러 종을 sigma 로 정규화해 합친다."""
    res = []
    for c in pred_df.columns:
        s = max(sigmas.get(c, 1.0), 1e-30)
        r = (pred_df[c].to_numpy(float) - meas_df[c].to_numpy(float)) / s
        res.append(r[np.isfinite(r)])
    if not res:
        return np.nan
    r = np.concatenate(res)
    n = len(r)
    rss = float(r @ r)
    if n <= k + 2 or rss <= 0:
        return np.nan
    return n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def evaluate_candidate(cfg: SelectionConfig, cand: Candidate, tm: TagMap,
                       train: pd.DataFrame, test: pd.DataFrame,
                       inputs_cols: list[str], obs_cols: list[str],
                       verbose: bool = True) -> CandidateResult:
    unit = next((e.unit for e in tm.observations if e.primary == cfg.target), "1")
    params = list(dict.fromkeys(cfg.common_calibrate + cand.calibrate))
    if verbose:
        print(f"  [{cand.id}] 파라미터 {len(params)}개 보정 중 ...", flush=True)
    try:
        model = _build_model(cfg, cand)
    except Exception as exc:
        return CandidateResult(cand.id, cand.description, len(params), *[np.nan] * 7,
                               0, False, unit, message=f"모델 조립 실패: {exc}")
    spec = CalibrationSpec(params=params, observations=obs_cols, sigmas=tm.sigmas(),
                           max_rows=cfg.max_rows, prior_weight=cfg.prior_weight)
    try:
        cal = calibrate(model, train[inputs_cols], train[obs_cols], spec,
                        expansion=tm.expansion(), verbose=False)
    except Exception as exc:
        return CandidateResult(cand.id, cand.description, len(params), *[np.nan] * 7,
                               0, False, unit, message=f"보정 실패: {exc}")

    exp = tm.expansion()
    pred_tr = simulate(model, build_param_rows(model, train[inputs_cols], model.p0(), exp),
                       obs_cols, index=train.index).values
    pred_te = simulate(model, build_param_rows(model, test[inputs_cols], model.p0(), exp),
                       obs_cols, index=test.index).values

    tr_rmse, _, _ = _rmse(pred_tr[cfg.target], train[cfg.target], unit)
    te_rmse, te_bias, te_r2 = _rmse(pred_te[cfg.target], test[cfg.target], unit)

    k = len(params)
    corr = cal.correlation
    off = corr[~np.eye(len(params), dtype=bool)] if len(params) > 1 else np.array([0.0])
    off = off[np.isfinite(off)]
    worst = float(off[np.argmax(np.abs(off))]) if len(off) else 0.0
    rel = 100.0 * cal.stderr / np.maximum(np.abs(cal.fitted), 1e-30)
    rel = rel[np.isfinite(rel)]

    bound_names = cal.at_bound()
    # 식별성은 **이 후보에만 있는 파라미터** 기준으로 본다. 공통 파라미터의
    # 식별성은 설정의 성질이지 후보의 성질이 아니므로 후보 비교에 도움이 안 된다.
    closure_params = [p for p in cand.calibrate if p not in set(cfg.common_calibrate)]
    if closure_params:
        worst, rel_max = cal.subset(closure_params)
    else:
        rel_max = float(np.max(rel)) if len(rel) else np.nan

    span = None
    if cfg.discriminator:
        try:
            vi = model.var_index(cfg.discriminator)
            rows_te = build_param_rows(model, test[inputs_cols], model.p0(), exp)
            sim_d = simulate(model, rows_te[:: max(1, len(rows_te) // 200)],
                             obs_cols, index=None, keep_states=True)
            st = getattr(sim_d, "states", None)
            if st is not None:
                col = st[:, vi]
                col = col[np.isfinite(col) & (col != 0)]
                if len(col):
                    span = (float(np.min(col)), float(np.max(col)))
        except (KeyError, AttributeError):
            span = None

    return CandidateResult(
        id=cand.id, description=cand.description or "", n_params=k,
        train_rmse=tr_rmse, test_rmse=te_rmse, test_bias=te_bias, test_r2=te_r2,
        aicc=_aicc(pred_tr, train[obs_cols], tm.sigmas(), k),
        worst_corr=worst, max_rel_stderr=rel_max,
        at_bound=len(bound_names), converged=True, unit=unit, calibration=cal,
        train_pred=pred_tr[cfg.target], test_pred=pred_te[cfg.target],
        closure_params=closure_params, at_bound_names=bound_names,
        discriminator_span=span,
    )


def run_selection(cfg: SelectionConfig, verbose: bool = True) -> SelectionResult:
    tm = TagMap.load(cfg.tagmap)
    inputs, obs = read_csv(cfg.csv, tm)
    df = inputs.join(obs, how="inner").dropna()
    train = df.query(cfg.train_query) if cfg.train_query else df
    test = df.query(cfg.test_query) if cfg.test_query else df
    obs_cols = [c for c in (cfg.observations or list(obs.columns)) if c in df.columns]
    if cfg.target not in obs_cols:
        raise ValueError(f"target {cfg.target!r} 이 관측 목록에 없습니다: {obs_cols}")
    if verbose:
        print(f"데이터 {len(df)}행 | 학습 {len(train)}행 | 외삽검증 {len(test)}행")
        print(f"후보 {len(cfg.candidates)}개 비교")
    results = [evaluate_candidate(cfg, c, tm, train, test, list(inputs.columns), obs_cols,
                                  verbose)
               for c in cfg.candidates]
    return SelectionResult(config=cfg, results=results, train=train, test=test, tagmap=tm)
