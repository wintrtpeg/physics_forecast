"""보정 행 선택 · 잔차 피드백(score test) · 검증된 적 없는 외삽 방향."""

import numpy as np
import pandas as pd
import pytest

from pforecast.calib import build_param_rows, simulate
from pforecast.calib.design import (average_rows, score_test, select_rows, space_filling,
                                    untested_directions)
from pforecast.scenario import load_python_model


@pytest.fixture
def plant():
    """5분 데이터: 가동율은 1시간 계단(MES), 순환수는 거의 일정하고 4시간만 펌프 정지."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=30 * 288, freq="5min")
    n = len(idx)
    util = np.repeat(0.5 + 0.2 * rng.random(n // 12 + 1), 12)[:n]
    water = 33 + rng.normal(0, 0.2, n)
    water[5050:5098] = 0.0                       # 4시간 정지 (시간 간격 12시간 사이)
    y = 50 + 30 * util - 0.3 * water + rng.normal(0, 2, n)
    return pd.DataFrame({"util": util, "water": water, "nox": y}, index=idx)


def test_average_rows_makes_hourly_means_and_drops_hours_with_a_step(plant):
    h = average_rows(plant, ["util", "water"], ["nox"])
    assert h.index.freq is None or pd.Timedelta(h.index.freq) == pd.Timedelta("1h")
    assert len(h) < len(plant) / 11
    # 펌프가 서고(5050) 다시 도는(5098) 시각이 걸린 시간대는 평균이 가상의 상태라 뺀다
    start, stop = plant.index[5050].floor("1h"), plant.index[5098].floor("1h")
    if plant.index[5050] != start:
        assert start not in h.index
    assert stop not in h.index or plant.index[5098] == stop
    # 이미 1시간 데이터면 그대로
    assert len(average_rows(h, ["util", "water"], ["nox"])) == len(h)


def test_space_filling_keeps_rare_operating_states_that_stride_misses(plant):
    rows = select_rows(plant, 60, ["util", "water"], ["nox"], method="space_filling")
    old = select_rows(plant, 60, ["util", "water"], ["nox"], method="stride")
    assert len(rows) <= 60 and len(old) >= 55
    assert (rows["water"] < 1).any(), "펌프 정지 행이 보정에 들어가야 한다"
    assert not (old["water"] < 1).any(), "시간 간격 추출은 4시간짜리 정지를 놓친다 (옛 방식)"


def test_space_filling_returns_everything_when_small():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert len(space_filling(df, 10, ["a"])) == 3


def test_untested_directions_separate_never_varied_from_out_of_range():
    idx = pd.date_range("2025-01-01", periods=1000, freq="1h")
    train = pd.DataFrame({"n_tools": 24.0, "water": 33.0 + 0.1 * np.sin(np.arange(1000)),
                          "util": np.linspace(0.4, 0.7, 1000),
                          "temp": np.linspace(0.0, 20.0, 1000)}, index=idx)
    train.iloc[500:508, 1] = 0.0                          # 드문 펌프 정지
    fut = pd.DataFrame({"n_tools": 30.0, "water": 42.0, "util": np.linspace(0.75, 0.9, 100),
                        "temp": np.linspace(5.0, 15.0, 100)})
    dirs = {d.column: d for d in untested_directions(train, fut, {"water": "m3/h"})}
    assert dirs["n_tools"].kind == "고정" and dirs["n_tools"].untested
    # 평소 33 으로 일정하고 펌프 정지(0) 때만 변했다 — 0~33 사이 반응은 본 적이 없다
    assert dirs["water"].kind == "평소 고정" and dirs["water"].untested
    # 가동율은 학습에서 변했던 방향이다 — 범위 밖이지만 기울기는 데이터로 맞췄다
    assert dirs["util"].kind == "범위 밖" and not dirs["util"].untested
    assert "temp" not in dirs                               # 학습 범위 안
    assert "검증된 적이 없습니다" in dirs["n_tools"].message()
    assert [d.column for d in dirs.values()][:2] == ["n_tools", "water"]


@pytest.fixture(scope="module")
def nox():
    m = load_python_model("examples/nox_stack/model.py").compile()
    m.build()
    return m


def test_score_test_points_at_the_parameter_that_is_actually_wrong(nox):
    """충전층 저항이 실제로 30% 큰데 보정 대상이 배출계수뿐이면, 잔차가 SCR.K 를 가리켜야 한다."""
    rng = np.random.default_rng(1)
    n = 24
    inp = pd.DataFrame({"SRC_DRY.util": rng.uniform(0.4, 0.9, n),
                        "FAN.n_ratio": rng.uniform(0.9, 1.0, n)},
                       index=pd.date_range("2025-01-01", periods=n, freq="1h"))
    j = nox.par_index("SCR.K")
    base = nox.p0()[j]
    try:
        nox.parameters[j].value = base * 1.3
        rows = build_param_rows(nox, inp)
        truth = simulate(nox, rows, ["STK.C_dry", "SCR.dp_mmAq"], index=inp.index).values
    finally:
        nox.parameters[j].value = base
    data = inp.join(truth)
    props = score_test(nox, data, list(inp.columns), ["STK.C_dry", "SCR.dp_mmAq"],
                       {"STK.C_dry": 3e-6, "SCR.dp_mmAq": 30.0},
                       selected=["SRC_DRY.ef_process"],
                       candidates=["SCR.K", "SRC_WET.ef_idle", "FAN.eta"])
    assert props[0].name == "SCR.K" and props[0].usable
    assert props[0].gain_pct > 50


def test_forecast_sensitivity_grows_outside_the_training_conditions(nox):
    """같은 파라미터 불확도라도 학습 범위 밖 조건에서 예측하면 흔들림이 커진다."""
    from pforecast.calib import CalibrationResult
    from pforecast.calib.design import forecast_sensitivity
    names = ["SRC_DRY.ef_process", "SRC_DRY.ef_idle"]
    idx = [nox.par_index(n) for n in names]
    val = nox.p0()[idx]
    cal = CalibrationResult(names=names, units=["mg/s", "mg/s"], initial=val, fitted=val,
                            stderr=0.1 * val, correlation=np.eye(2), cost=0.0, n_rows=10, n_eval=1)
    t = pd.date_range("2025-01-01", periods=12, freq="1h")
    lo = pd.DataFrame({"SRC_DRY.util": np.full(12, 0.4)}, index=t)
    hi = pd.DataFrame({"SRC_DRY.util": np.full(12, 0.4), "SRC_DRY.n_tools": np.full(12, 36.0)}, index=t)
    a = forecast_sensitivity(nox, cal, lo, "STK.C_dry", "mg/Nm3")
    b = forecast_sensitivity(nox, cal, hi, "STK.C_dry", "mg/Nm3")
    assert b.sigma > a.sigma > 0
    assert {n for n, _ in b.contributions} == set(names)
    assert abs(sum(s for _, s in b.contributions) - 100.0) < 1e-6


def test_improve_folds_stay_inside_training_and_move_forward():
    from pforecast.improve import _inner_folds
    idx = pd.date_range("2025-01-01", "2025-05-31 23:00", freq="1h")
    folds = _inner_folds(idx, 2, 30.0)
    assert len(folds) == 2
    for tr_end, va_start, va_end in folds:
        assert tr_end < va_start <= va_end <= idx.max()
        assert va_start - tr_end == pd.Timedelta("1D")          # 엠바고
    assert folds[0][2] <= folds[1][1]                            # 두 번째 폴드가 뒤


def test_improve_changes_the_design_only_for_a_clear_gain():
    from pforecast.improve import choose
    ms = pd.Series({"현재 설정": 5.00, "행 선택 stride": 4.95, "+ FAN.c2": 5.20})
    assert choose(ms, "현재 설정") == "현재 설정"                   # 1% 는 우연일 수 있다
    ms["행 선택 stride"] = 4.80
    assert choose(ms, "현재 설정") == "행 선택 stride"
    assert choose(pd.Series(dtype=float), "현재 설정") == "현재 설정"


def test_improve_folds_shrink_for_a_short_training_period():
    """예제처럼 학습이 20일뿐이어도 폴드가 생겨야 한다 (30일 폴드 두 개는 안 들어간다)."""
    from pforecast.improve import ImproveResult, _inner_folds
    idx = pd.date_range("2025-03-01", "2025-03-19 23:55", freq="5min")
    folds = _inner_folds(idx, 2, 30.0)
    assert len(folds) == 2
    first_train = (idx < folds[0][0]).sum() / len(idx)
    assert first_train > 0.4
    assert ImproveResult(target="y", unit="1").mean_scores().empty    # 폴드가 없어도 죽지 않는다
