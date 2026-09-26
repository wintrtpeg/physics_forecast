"""태그 이름 기반 단위 추론."""

import numpy as np
import pandas as pd
import pytest

from pforecast.analyze.units_guess import guess_unit, guess_units, summary


@pytest.mark.parametrize("name,expect", [
    ("F2_UT_STK01_TEMP", "degC"),
    ("F2_UT_AMB_TEMP", "degC"),
    ("F2_UT_STK01_NOX_DRY", "mg/Nm3"),
    ("SCR01_SO2_CONC", "mg/Nm3"),
    ("F2_FAB_DRY_UTIL", "%"),
    ("CHILLER_LOAD_PCT", "%"),
    ("F2_UT_SCR01_FAN_HZ", "Hz"),
    ("PUMP_01_RPM", "rpm"),
    ("CH1_POWER_KW", "kW"),
    ("PLANT_KWH_TOTAL", "kWh"),
    ("CW_PH", "1"),
])
def test_name_rules(name, expect):
    assert guess_unit(name).unit == expect


def test_flow_unit_comes_from_magnitude():
    small = guess_unit("BR_DRY_FLOW", np.full(50, 1.4))
    mid = guess_unit("HDR_FLOW", np.full(50, 260.0))
    big = guess_unit("STK_FLOW", np.full(50, 13500.0))
    assert small.unit == "kg/s"
    assert mid.unit == "CMM"
    assert big.unit == "Nm3/h"
    assert all(g.needs_review for g in (small, mid, big)), "유량은 사람이 확인해야 한다"


def test_pressure_unit_comes_from_magnitude():
    assert guess_unit("FAN_SP", np.full(50, 330.0)).unit == "mmAq"
    assert guess_unit("HDR_PRESS", np.full(50, 6.5)).unit == "kPa"


def test_ratio_detects_zero_to_one_scale():
    frac = guess_unit("VALVE_OP", np.full(40, 0.65))
    pct = guess_unit("VALVE_OP", np.full(40, 65.0))
    assert frac.unit == "1"
    assert pct.unit == "%"


def test_unknown_name_returns_empty_with_reason():
    g = guess_unit("XZ_9931", np.arange(10.0))
    assert g.unit == ""
    assert g.confidence == 0.0
    assert g.needs_review
    assert "단서" in g.reason


def test_guess_units_over_a_frame():
    df = pd.DataFrame({"A_TEMP": [20.0, 21.0], "B_UTIL": [80.0, 85.0], "ZZZ": [1.0, 2.0]})
    g = guess_units(df)
    assert g["A_TEMP"].unit == "degC"
    assert g["B_UTIL"].unit == "%"
    assert g["ZZZ"].unit == ""
    s = summary(g)
    assert s["확실"] == 2 and s["모름"] == 1


def test_every_column_of_the_example_gets_a_unit():
    df = pd.read_csv("examples/nox_stack/data/plant_5min.csv",
                     parse_dates=["timestamp"]).set_index("timestamp")
    g = guess_units(df)
    assert all(v.unit for v in g.values()), "예제 태그는 전부 추론되어야 한다"
