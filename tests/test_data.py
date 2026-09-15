"""태그맵과 데이터 어댑터."""

import numpy as np
import pandas as pd
import pytest

from pforecast.data import TagMap, load_frames, pivot_long, stratified_sample
from pforecast.data.tagmap import TagEntry


def test_unit_conversion_to_si():
    e = TagEntry("T", "SRC.util", "%")
    assert e.to_si_series(pd.Series([80.0, 50.0])).tolist() == [0.8, 0.5]
    e2 = TagEntry("T", "STK.T_stack", "degC")
    assert e2.to_si_series(pd.Series([25.0])).tolist() == [298.15]


def test_scale_and_offset_apply_before_unit():
    e = TagEntry("HZ", "FAN.n_ratio", "1", scale=1 / 60.0)
    assert e.to_si_series(pd.Series([60.0, 54.0])).tolist() == pytest.approx([1.0, 0.9])


def test_multiple_targets_from_one_tag():
    e = TagEntry("AMB", ["STK.T_amb", "DCT_MAIN.T_amb"], "degC")
    assert e.targets == ["STK.T_amb", "DCT_MAIN.T_amb"]
    assert e.primary == "STK.T_amb"
    e2 = TagEntry("AMB", "A.x, B.x", "1")
    assert e2.targets == ["A.x", "B.x"]


def test_sigma_converts_to_si():
    tm = TagMap(observations=[TagEntry("N", "STK.C_dry", "mg/Nm3", sigma=3.0)])
    assert tm.sigmas()["STK.C_dry"] == pytest.approx(3e-6)
    # degC 의 sigma 는 차이값이므로 오프셋을 적용하지 않는다
    tm2 = TagMap(observations=[TagEntry("T", "STK.T_stack", "degC", sigma=0.5)])
    assert tm2.sigmas()["STK.T_stack"] == pytest.approx(0.5)


def test_load_frames_resamples_to_five_minutes():
    idx = pd.date_range("2025-01-01", periods=20, freq="1min")
    raw = pd.DataFrame({"timestamp": idx, "U": np.arange(20.0), "N": np.arange(20.0) * 2})
    tm = TagMap(inputs=[TagEntry("U", "SRC.util", "1")],
                observations=[TagEntry("N", "STK.C_dry", "mg/Nm3")], resample="5min")
    inp, obs = load_frames(raw, tm)
    assert len(inp) == 4
    assert inp["SRC.util"].iloc[0] == pytest.approx(2.0)   # 0..4 평균


def test_missing_input_tag_is_an_error():
    raw = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=3, freq="5min"),
                        "A": [1.0, 2.0, 3.0]})
    tm = TagMap(inputs=[TagEntry("MISSING", "SRC.util", "1")])
    with pytest.raises(KeyError, match="CSV 에 없는 태그"):
        load_frames(raw, tm)


def test_pivot_long_to_wide():
    df = pd.DataFrame({"t": ["a", "a", "b", "b"], "tag": ["X", "Y", "X", "Y"],
                       "v": [1.0, 2.0, 3.0, 4.0]})
    wide = pivot_long(df, "t", "tag", "v")
    assert list(wide.columns) == ["t", "X", "Y"]
    assert wide["X"].tolist() == [1.0, 3.0]


def test_stratified_sample_covers_the_operating_range():
    rng = np.random.default_rng(0)
    # 90% 는 정상운전, 10% 만 저부하. 단순 랜덤이면 저부하가 거의 안 뽑힌다.
    u = np.concatenate([rng.normal(0.8, 0.01, 900), rng.uniform(0.2, 0.5, 100)])
    df = pd.DataFrame({"u": u, "y": u * 2})
    picked = stratified_sample(df, "u", n=100, bins=10, seed=1)
    assert len(picked) <= 100
    low = (picked["u"] < 0.6).mean()
    assert low > 0.25, "층화 추출이면 저부하 구간 비중이 원본(10%)보다 커야 한다"


def test_example_tagmap_loads_and_expands():
    tm = TagMap.load("examples/nox_stack/tagmap.yaml")
    exp = tm.expansion()
    assert len(exp["STK.T_amb"]) == 6      # 외기온도 태그 하나가 6개 파라미터로
    assert "STK.C_dry" in tm.observation_targets()
