"""누수 방지 — 검증 구간의 어떤 정보도 학습 쪽 처리에 들어가면 안 된다.

규칙을 말로만 두지 않고 테스트로 박아 둔다. 검증 구간 데이터를 마음대로 바꿔도
학습 쪽 결과가 **한 비트도** 바뀌지 않아야 한다.
"""

import numpy as np
import pandas as pd
import pytest

from pforecast.analyze import AnalysisConfig, run_analysis
from pforecast.data.quality import assess_parts
from pforecast.data.split import check_split, future_split


def _series_frame(seed=0, days=30):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=days * 288, freq="5min")
    n = len(idx)
    t = np.arange(n)
    df = pd.DataFrame({
        "flow": 12000 + 300 * np.sin(2 * np.pi * t / 700) + rng.normal(0, 120, n),
        "nox": 60 + 5 * np.sin(2 * np.pi * t / 288) + rng.normal(0, 1.5, n),
    }, index=idx)
    for d in range(days):                      # 매일 03:00 교정 창
        i0 = d * 288 + 36
        df.iloc[i0:i0 + 2, 1] = 0.3
        df.iloc[i0 + 2:i0 + 4, 1] = 160.0
    return df


def test_train_cleaning_is_blind_to_the_validation_period():
    df = _series_frame()
    tr, te = future_split(df.index, 0.6, "1D")
    base = assess_parts(df, {"학습": tr, "검증": te})
    # 검증 구간을 완전히 망가뜨린다: 잡음 10배, 스파이크 난무, 고착
    wild = df.copy()
    rng = np.random.default_rng(9)
    wild.loc[te, "flow"] = wild.loc[te, "flow"] + rng.normal(0, 1200, te.sum())
    wild.loc[te, "nox"] = 999.9
    after = assess_parts(wild, {"학습": tr, "검증": te})
    for c in df.columns:
        assert np.array_equal(base.masks.get(c, np.zeros(len(df), bool))[tr],
                              after.masks.get(c, np.zeros(len(df), bool))[tr]), c


def test_value_based_split_is_flagged_as_not_future():
    idx = pd.date_range("2025-01-01", periods=1000, freq="5min")
    u = 0.5 + 0.2 * np.sin(np.arange(1000) / 50)
    df = pd.DataFrame({"u": u}, index=idx)
    chk = check_split(df[df.u <= 0.55], df[df.u >= 0.62], ["u"])
    assert not chk.is_future
    assert chk.is_extrapolation              # 값으로는 외삽이지만
    assert any("미래 예측 검증이 아닙니다" in w for w in chk.warnings())


def test_chronological_split_inside_the_envelope_is_flagged_as_not_extrapolation():
    idx = pd.date_range("2025-01-01", periods=2000, freq="5min")
    u = 0.5 + 0.2 * np.sin(np.arange(2000) / 40)      # 같은 범위를 계속 오간다
    df = pd.DataFrame({"u": u}, index=idx)
    tr, te = future_split(df.index, 0.6, "1h")
    chk = check_split(df[tr], df[te], ["u"])
    assert chk.is_future and chk.embargo_days >= 1 / 24 - 1e-9
    assert not chk.is_extrapolation
    assert any("외삽 검증이 아닙니다" in w for w in chk.warnings())


def test_future_extrapolation_split_passes():
    idx = pd.date_range("2025-01-01", periods=2000, freq="5min")
    u = np.linspace(0.3, 0.9, 2000)                    # 계속 올라가는 가동율
    df = pd.DataFrame({"u": u}, index=idx)
    tr, te = future_split(df.index, 0.6, "1D")
    chk = check_split(df[tr], df[te], ["u"])
    assert chk.is_future and chk.is_extrapolation
    assert chk.warnings() == []
    assert chk.extrapolation == pytest.approx(1.0)


@pytest.fixture
def ramp_csv(tmp_path):
    rng = np.random.default_rng(1)
    idx = pd.date_range("2025-01-01", periods=40 * 288, freq="5min")
    n = len(idx)
    util = np.clip(np.linspace(0.3, 0.95, n) + rng.normal(0, 0.02, n), 0, 1)
    flow = 10000 + 3000 * util + rng.normal(0, 50, n)        # 결과 변수 (같은 시각 측정)
    nox = 40 + 60 * util ** 1.5 + rng.normal(0, 1.0, n)
    df = pd.DataFrame({"timestamp": idx, "F_UTIL": util * 100, "STK_FLOW": flow,
                       "STK_NOX": nox})
    p = tmp_path / "ramp.csv"
    df.to_csv(p, index=False)
    return p


def test_analysis_splits_chronologically_and_forecasts_with_drivers_only(ramp_csv):
    cfg = AnalysisConfig(csv=str(ramp_csv), target="STK_NOX", target_unit="mg/Nm3",
                         units={"F_UTIL": "%", "STK_FLOW": "Nm3/h", "STK_NOX": "mg/Nm3"},
                         drivers=["F_UTIL"], train_fraction=0.6)
    res = run_analysis(cfg, verbose=False)
    assert res.split.is_future and res.split.embargo_days >= 1.0
    assert res.forecast["drivers"] == ["F_UTIL"]          # 결과 변수(유량)는 미래에 없다
    assert res.split.is_extrapolation
    assert res.forecast["n_outside"] > 0
    lines = " ".join(res.headline())
    assert "미래 예측(운전 입력 1개만)" in lines
