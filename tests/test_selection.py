"""구성방정식 후보 비교 로직.

실제 보정은 수십 초가 걸리므로 여기서는 설정 로딩과 **판정 규칙**을 검증한다.
전체 파이프라인은 `pf select` 로 확인한다.
"""

import numpy as np
import pandas as pd
import pytest

from pforecast.data import TagMap
from pforecast.data.tagmap import TagEntry
from pforecast.selection import (Candidate, CandidateResult, SelectionConfig,
                                 SelectionResult, _aicc)


def _res(id, n_params, train, test, aicc=100.0, corr=0.1, rel=10.0, bound=(),
         closure=("k",)):
    return CandidateResult(id=id, description="", n_params=n_params, train_rmse=train,
                           test_rmse=test, test_bias=0.0, test_r2=0.9, aicc=aicc,
                           worst_corr=corr, max_rel_stderr=rel, at_bound=len(bound),
                           converged=True, unit="mg/Nm3",
                           closure_params=list(closure), at_bound_names=list(bound))


def _sel(results):
    return SelectionResult(config=SelectionConfig(target="y"), results=results,
                           train=pd.DataFrame(), test=pd.DataFrame(), tagmap=TagMap())


def test_table_sorts_by_extrapolation_error():
    s = _sel([_res("a", 5, 2.0, 5.0), _res("b", 6, 1.9, 2.5), _res("c", 4, 2.1, 3.0)])
    df = s.table()
    assert list(df["후보"]) == ["b", "c", "a"]
    assert "외삽 RMSE [mg/Nm3]" in df.columns


def test_delta_aicc_is_relative_to_the_best():
    s = _sel([_res("a", 5, 2.0, 3.0, aicc=120.0), _res("b", 6, 1.9, 2.5, aicc=100.0)])
    df = s.table().set_index("후보")
    assert df.loc["b", "ΔAICc"] == pytest.approx(0.0)
    assert df.loc["a", "ΔAICc"] == pytest.approx(20.0)


def test_verdict_names_the_best_candidate():
    s = _sel([_res("a", 5, 2.0, 5.0), _res("b", 6, 1.9, 2.5)])
    assert "**b**" in s.verdict()[0]


def test_verdict_prefers_the_simpler_model_when_close():
    """외삽 오차가 5% 이내면 파라미터가 적은 쪽을 권해야 한다."""
    s = _sel([_res("complex", 9, 1.8, 2.50), _res("simple", 4, 1.9, 2.55)])
    joined = " ".join(s.verdict())
    assert "simple" in joined and "단순한" in joined


def test_verdict_flags_overfitting():
    """학습은 비슷한데 외삽이 훨씬 나쁜 후보를 짚어야 한다."""
    s = _sel([_res("good", 5, 2.00, 2.0), _res("overfit", 8, 2.01, 4.0)])
    joined = " ".join(s.verdict())
    assert "overfit" in joined and "속습니다" in joined


def test_verdict_flags_unidentifiable_parameters():
    s = _sel([_res("ok", 5, 2.0, 2.0), _res("bad", 6, 2.1, 2.2, rel=500.0)])
    assert any("bad" in v and "결정되지 않" in v for v in s.verdict())
    s2 = _sel([_res("ok", 5, 2.0, 2.0), _res("collin", 6, 2.1, 2.2, corr=-0.999)])
    assert any("collin" in v for v in s2.verdict())


def test_identifiability_is_judged_on_closure_parameters_only():
    """공통 파라미터의 식별성은 후보의 성질이 아니므로 경고하지 않는다."""
    s = _sel([_res("ok", 5, 2.0, 2.0, closure=()),
              _res("no_closure", 6, 2.1, 2.2, rel=9999.0, closure=())])
    assert not any("결정되지 않" in v for v in s.verdict())


def test_verdict_flags_parameters_pinned_at_bounds():
    s = _sel([_res("ok", 5, 2.0, 2.0), _res("pinned", 6, 2.1, 2.2, bound=("SCR.K", "a"))])
    assert any("pinned" in v and "경계" in v for v in s.verdict())


def test_verdict_handles_total_failure():
    s = _sel([CandidateResult("x", "", 3, *[np.nan] * 7, 0, False, "1")])
    assert "수렴한 후보가 없습니다" in s.verdict()[0]


def _ar1(n, rho, scale, rng):
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = rho * e[i - 1] + rng.normal(0, scale)
    return e


def _timeseries_sel(spread: float, n: int = 400, rho: float = 0.9):
    """자기상관이 있는 예측오차를 가진 두 후보. spread 만큼 b 가 계통적으로 나쁘다."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    meas = 60.0 + _ar1(n, rho, 1.0, rng)
    err_a = _ar1(n, rho, 1.0, rng)
    err_b = _ar1(n, rho, 1.0, rng) + spread
    a = _res("a", 5, 2.0, float(np.sqrt(np.mean(err_a ** 2))))
    b = _res("b", 5, 2.0, float(np.sqrt(np.mean(err_b ** 2))))
    # SI 로 저장 (mg/Nm3 -> kg/m3)
    a.test_pred = pd.Series((meas + err_a) * 1e-6, index=idx)
    b.test_pred = pd.Series((meas + err_b) * 1e-6, index=idx)
    cfg = SelectionConfig(target="y", discriminator="SCR.LG")
    tm = TagMap(observations=[TagEntry("TAG", "y", "mg/Nm3", sigma=3.0)])
    return SelectionResult(config=cfg, results=[a, b],
                           train=pd.DataFrame(),
                           test=pd.DataFrame({"y": meas * 1e-6}, index=idx),
                           tagmap=tm)


def test_paired_compare_accounts_for_autocorrelation():
    s = _timeseries_sel(spread=3.0)
    diff, se, n_eff = s.paired_compare(s.results[0], s.results[1])
    assert n_eff < 400, "자기상관이 있으면 유효 표본수가 줄어야 한다"
    assert diff > 0, "b 가 더 나쁘므로 오차 차이가 양수여야 한다"


def test_sigma_comes_from_the_tagmap_in_display_units():
    s = _timeseries_sel(spread=0.01)
    assert s.sigma == pytest.approx(3.0)


def test_verdict_says_candidates_are_indistinguishable_when_they_are():
    """차이가 계측 불확도보다 훨씬 작으면 1등을 발표하면 안 된다."""
    s = _timeseries_sel(spread=0.01)
    joined = " ".join(s.verdict())
    assert "고를 수 없습니다" in joined
    assert "SCR.LG" in joined, "무엇을 흔들어야 갈리는지 알려줘야 한다"


def test_verdict_recommends_the_simpler_model_when_it_is_unique():
    s = _timeseries_sel(spread=0.01)
    s.results[1].n_params = 9          # b 를 더 복잡하게
    joined = " ".join(s.verdict())
    assert "가장 단순한" in joined


def test_verdict_hands_the_choice_back_when_complexity_ties():
    """파라미터 수까지 같으면 데이터가 아니라 물리로 고르라고 해야 한다."""
    s = _timeseries_sel(spread=0.01)   # 둘 다 5개
    joined = " ".join(s.verdict())
    assert "물리적 근거로 고르세요" in joined
    assert "가장 단순한" not in joined


def test_verdict_warns_that_significance_is_not_size():
    """표본이 크면 무의미한 차이도 '유의미'해진다는 점을 짚어야 한다."""
    s = _timeseries_sel(spread=0.01, n=3000)
    joined = " ".join(s.verdict())
    if "통계적으로는" in joined:
        assert "크기를 보세요" in joined


def test_verdict_does_not_cry_wolf_on_a_real_difference():
    s = _timeseries_sel(spread=8.0)           # 계측 불확도보다 훨씬 큰 차이
    joined = " ".join(s.verdict())
    assert "고를 수 없습니다" not in joined


def test_practical_threshold_is_configurable():
    s = _timeseries_sel(spread=8.0)
    s.practical_fraction = 100.0              # 사실상 무한대 -> 전부 동률
    assert "고를 수 없습니다" in " ".join(s.verdict())


def test_aicc_penalises_extra_parameters():
    n = 200
    rng = np.random.default_rng(0)
    meas = pd.DataFrame({"y": rng.normal(0, 1, n)})
    pred = pd.DataFrame({"y": meas["y"] + rng.normal(0, 0.1, n)})
    a = _aicc(pred, meas, {"y": 1.0}, k=3)
    b = _aicc(pred, meas, {"y": 1.0}, k=8)
    assert b > a, "같은 적합도라면 파라미터가 많은 쪽이 벌점을 더 받아야 한다"


def test_aicc_returns_nan_when_underdetermined():
    meas = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    assert np.isnan(_aicc(meas, meas, {"y": 1.0}, k=10))


def test_example_selection_config_loads():
    cfg = SelectionConfig.load("examples/nox_stack/selection.yaml")
    assert cfg.slot == "SCR"
    assert cfg.target == "STK.C_dry"
    assert len(cfg.candidates) == 4
    ids = {c.id for c in cfg.candidates}
    assert {"ntu_power", "constant", "langmuir", "ntu_power_T"} <= ids
    # 하한선 모델이 반드시 포함되어야 한다
    assert any(c.id == "constant" for c in cfg.candidates)
    assert all(c.component for c in cfg.candidates)


def test_candidate_defaults():
    c = Candidate(id="x")
    assert c.component is None and c.calibrate == []
