"""값 수준 품질 진단: 현장 계측이 흔히 만드는 이상."""

import numpy as np
import pandas as pd

from pforecast.data.quality import apply, assess


def _frame(days=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=days * 288, freq="5min")
    n = len(idx)
    t = np.arange(n)
    base = 60 + 5 * np.sin(2 * np.pi * t / 288)
    df = pd.DataFrame({
        "nox": base + rng.normal(0, 1.5, n),
        "flow": 12000 + 300 * np.sin(2 * np.pi * t / 700) + rng.normal(0, 120, n),
        "water": 33 + rng.normal(0, 0.4, n),
        "hz": np.where(t < n // 2, 58.0, 60.0),
        "util": pd.Series(50 + 10 * np.sin(2 * np.pi * t / 2000), index=idx)
                  .resample("1h").mean().reindex(idx, method="ffill").round(1).to_numpy(),
        "spare": np.full(n, 4.0),
        "temp": 40 + rng.normal(0, 0.3, n),
    }, index=idx)
    return df


def test_daily_calibration_window_is_recognised_as_such():
    df = _frame()
    for d in range(20):
        i0 = d * 288 + 36                    # 매일 03:00
        df.iloc[i0:i0 + 2, 0] = 0.3
        df.iloc[i0 + 2:i0 + 4, 0] = 160.0
    rep = assess(df)
    kinds = {(i.column, i.kind) for i in rep.issues}
    assert ("nox", "calibration") in kinds
    assert ("nox", "spike") not in kinds       # 교정 창을 스파이크로 흘리지 않는다
    assert rep.excluded("nox") == 20 * 4
    cal = next(i for i in rep.issues if i.kind == "calibration")
    assert "03:00" in cal.message


def test_frozen_sensor_is_masked_but_zero_hold_is_kept():
    df = _frame()
    df.iloc[1000:1400, 1] = df.iloc[999, 1]    # 마지막 값 유지 (999 부터 401점이 같은 값)
    df.iloc[2000:2100, 2] = 0.0                 # 펌프 정지
    rep = assess(df)
    frozen = [i for i in rep.issues if i.kind == "frozen"]
    assert [i.column for i in frozen] == ["flow"]
    assert frozen[0].n_points == 400      # 첫 점(진짜 값)만 남긴다
    assert any(i.kind == "zero_hold" and i.column == "water" for i in rep.issues)
    assert rep.excluded("water") == 0
    out = apply(df, rep)
    assert out["water"].iloc[2050] == 0.0
    assert np.isnan(out["flow"].iloc[1200])


def test_spikes_but_not_sustained_steps():
    df = _frame()
    df.iloc[3000, 6] = 999.9
    df.iloc[3001, 6] = 999.9
    df.iloc[4000:, 6] += 8.0                    # 지속되는 계단은 스파이크가 아니다
    rep = assess(df)
    sp = [i for i in rep.issues if i.kind == "spike" and i.column == "temp"]
    assert len(sp) == 1 and sp[0].n_points == 2


def test_step_signals_are_classified():
    rep = assess(_frame())
    labels = {i.column: i.label for i in rep.issues if i.kind == "step_signal"}
    assert labels["hz"] == "설정값"
    assert labels["util"] == "1시간 갱신"
    assert labels["spare"] == "상수"


def test_dead_sensor_reported_once_even_across_comm_outages():
    df = _frame()
    df.iloc[2000:, 1] = np.nan
    df.iloc[3000:3010, :] = np.nan              # 통신 두절 (전 태그)
    df.iloc[4000:4003, :] = np.nan
    rep = assess(df)
    dead = [i for i in rep.issues if i.kind == "long_missing" and i.column == "flow"]
    assert len(dead) == 1 and dead[0].label == "센서 정지"
