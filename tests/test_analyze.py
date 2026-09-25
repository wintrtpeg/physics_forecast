"""프로파일링 · 대리모델 · XAI · 분석 사다리."""

import numpy as np
import pandas as pd
import pytest

from pforecast.analyze import (AnalysisConfig, Envelope, fit_surrogate, profile_dataset,
                               run_analysis, steady_state_mask)
from pforecast.analyze.surrogate import (correlation_clusters, find_lags,
                                         permutation_importance, search_improvement,
                                         time_blocked_folds)


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 1200
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    drive = np.clip(0.5 + 0.3 * np.sin(np.arange(n) / 90) + rng.normal(0, 0.02, n), 0.05, 1)
    return pd.DataFrame({
        "drive_a": drive,
        "drive_b": drive * 1.02 + rng.normal(0, 0.02, n),    # drive_a 와 강하게 상관(동일하진 않다)
        "drive_dup": drive * 3.0 + rng.normal(0, 1e-7, n),   # drive_a 와 사실상 동일
        "indep": rng.uniform(0, 1, n),                       # 독립 잡음
        "fixed": np.full(n, 3.0),                            # 상수
        "y": 10 * drive + 2.0 * rng.uniform(0, 1, n) + rng.normal(0, 0.2, n),
    }, index=idx)


# --- 프로파일링 -------------------------------------------------------------

def test_profile_flags_constant_and_duplicate_columns(frame):
    p = profile_dataset(frame)
    assert p.columns["fixed"].is_constant
    assert not p.columns["fixed"].usable
    assert p.columns["drive_dup"].duplicate_of == "drive_a"
    assert not p.columns["drive_dup"].usable
    assert any("사실상 같습니다" in w for w in p.warnings)
    assert any("변하지 않습니다" in w for w in p.warnings)


def test_profile_detects_sampling_interval(frame):
    p = profile_dataset(frame)
    assert p.interval_s == pytest.approx(300.0)
    assert p.n_rows == len(frame)


def test_steady_state_mask_excludes_step_changes():
    n = 300
    x = np.concatenate([np.ones(100), np.full(100, 5.0), np.ones(100)])
    df = pd.DataFrame({"x": x})
    mask = steady_state_mask(df, ["x"], window=6)
    assert mask.mean() > 0.8          # 대부분은 정상
    assert not mask[100]              # 계단이 생긴 지점은 제외
    assert mask[50] and mask[250]     # 평탄 구간은 살아 있어야 한다


# --- 포락선 ----------------------------------------------------------------

def test_envelope_distance_is_zero_inside_and_positive_outside():
    df = pd.DataFrame({"a": [0.0, 1.0], "b": [10.0, 20.0]})
    env = Envelope.fit(df, ["a", "b"])
    inside = pd.DataFrame({"a": [0.5], "b": [15.0]})
    outside = pd.DataFrame({"a": [1.5], "b": [15.0]})
    assert env.distance(inside)[0] == 0.0
    assert env.distance(outside)[0] == pytest.approx(0.5)
    assert env.outside_fraction(outside) == 1.0


# --- 교차검증 / 지연 --------------------------------------------------------

def test_time_blocked_folds_never_shuffle():
    folds = time_blocked_folds(100, 5)
    assert len(folds) == 5
    for tr, te in folds:
        assert len(np.intersect1d(tr, te)) == 0
        assert np.all(np.diff(te) == 1), "검증 블록은 연속된 시간이어야 한다"


def test_lag_detection_finds_a_known_shift():
    n = 600
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({"x": x, "y": np.roll(x, 4)})
    lags = find_lags(df.iloc[10:], "y", ["x"], max_lag=10)
    assert lags[0].best_lag == 4


# --- 상관 묶음 / 순열 중요도 -------------------------------------------------

def test_correlation_clusters_group_near_duplicates(frame):
    groups = correlation_clusters(frame, ["drive_a", "drive_b", "indep"], threshold=0.9)
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]
    pair = next(g for g in groups if len(g) == 2)
    assert set(pair) == {"drive_a", "drive_b"}


def test_grouped_permutation_beats_individual_on_collinear_features(frame):
    """개별 순열은 상관된 변수끼리 기여를 나눠 갖지만, 묶어서 재면 합쳐진다."""
    res = fit_surrogate(frame, "y", ["drive_a", "drive_b", "indep"], apply_lags=False)
    cluster = next(c for c in res.clusters if c.is_group)
    assert set(cluster.members) == {"drive_a", "drive_b"}
    # 묶음 중요도는 개별 중요도 어느 쪽보다도 커야 한다
    ind = {i.feature: i.importance for i in res.importances}
    assert cluster.importance > max(ind["drive_a"], ind["drive_b"])
    assert cluster.importance_pct > 50


def test_entanglement_is_flagged(frame):
    res = fit_surrogate(frame, "y", ["drive_a", "drive_b", "indep"], apply_lags=False)
    ent = {i.feature for i in res.importances if i.entangled}
    assert {"drive_a", "drive_b"} <= ent
    assert "indep" not in ent


def test_permutation_importance_is_zero_for_an_ignored_feature():
    rng = np.random.default_rng(7)
    n = 400
    df = pd.DataFrame({"a": rng.uniform(0, 1, n), "junk": rng.uniform(0, 1, n)})
    df["y"] = 5 * df["a"]
    res = fit_surrogate(df, "y", ["a", "junk"], apply_lags=False)
    imp = {i.feature: i.importance for i in res.importances}
    assert imp["a"] > 10 * max(imp["junk"], 1e-9)


def test_surrogate_skill_is_near_zero_for_pure_noise():
    rng = np.random.default_rng(11)
    n = 500
    df = pd.DataFrame({"x": rng.uniform(0, 1, n), "y": rng.normal(0, 1, n)})
    res = fit_surrogate(df, "y", ["x"], apply_lags=False)
    assert res.skill < 0.2, "신호가 없으면 평균예측 대비 개선이 거의 없어야 한다"


# --- 개선 탐색 --------------------------------------------------------------

def test_improvement_never_leaves_the_training_envelope(frame):
    res = fit_surrogate(frame, "y", ["drive_a", "indep"], apply_lags=False)
    imp = search_improvement(res, frame, ["indep"], direction="minimize")
    assert imp is not None
    lo, hi = res.envelope.lo["indep"], res.envelope.hi["indep"]
    assert lo <= imp.settings["indep"] <= hi
    assert imp.extrapolation == 0.0 and imp.feasible


def test_improvement_returns_none_for_unknown_controllable(frame):
    res = fit_surrogate(frame, "y", ["drive_a"], apply_lags=False)
    assert search_improvement(res, frame, ["nonexistent"]) is None


# --- 사다리 ----------------------------------------------------------------

def test_example_analysis_config_loads():
    cfg = AnalysisConfig.load("examples/nox_stack/analysis.yaml")
    assert cfg.target == "F2_UT_STK01_NOX_DRY"
    assert cfg.units["F2_UT_SCR01_HDR_FLOW"] == "kg/s"
    assert cfg.train_fraction == pytest.approx(0.6)


def test_ladder_always_marks_structural_model_as_blocked(tmp_path, frame):
    csv = tmp_path / "d.csv"
    frame.to_csv(csv, index_label="timestamp")
    cfg = AnalysisConfig(csv=str(csv), target="y",
                         units={"y": "kg/s", "drive_a": "1", "indep": "1"})
    res = run_analysis(cfg, verbose=False)
    levels = {r.level: r for r in res.rungs}
    assert levels["L0"].status == "완료"
    assert levels["L1"].status == "완료"
    # 토폴로지는 데이터에서 유도할 수 없다 - 언제나 막힘이어야 한다
    assert levels["L4"].status == "막힘"
    assert "토폴로지" in levels["L4"].blocker
    assert levels["L4"].how_to_unblock


def test_ladder_blocks_dimensional_rungs_without_units(tmp_path, frame):
    csv = tmp_path / "d.csv"
    frame.to_csv(csv, index_label="timestamp")
    res = run_analysis(AnalysisConfig(csv=str(csv), target="y"), verbose=False)
    levels = {r.level: r for r in res.rungs}
    assert levels["L2"].status == "막힘" and "단위" in levels["L2"].blocker
    assert levels["L3"].status == "막힘"


def test_headline_warns_about_the_dominant_entangled_cluster(tmp_path, frame):
    csv = tmp_path / "d.csv"
    frame.to_csv(csv, index_label="timestamp")
    res = run_analysis(AnalysisConfig(csv=str(csv), target="y", exclude=["fixed"]),
                       verbose=False)
    joined = " ".join(res.headline())
    assert "구분되지 않는" in joined
    assert "인과로 읽으면 안 됩니다" in joined
