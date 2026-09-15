"""단위 파서와 환산."""

import pytest

from pforecast.core.units import (DIMLESS, PRESSURE, VOL_FLOW, dim_of, dim_str,
                                  from_si, parse_unit, to_si)


@pytest.mark.parametrize("text,factor", [
    ("Pa", 1.0), ("kPa", 1e3), ("mmAq", 9.80665), ("bar", 1e5),
    ("CMM", 1 / 60), ("CMH", 1 / 3600), ("mg/Nm3", 1e-6),
    ("kg/s", 1.0), ("g/h", 1e-3 / 3600), ("%", 0.01), ("ppm", 1e-6),
])
def test_unit_factors(text, factor):
    assert parse_unit(text)[0] == pytest.approx(factor, rel=1e-12)


@pytest.mark.parametrize("text,expect", [
    ("Pa", PRESSURE), ("N/m2", PRESSURE), ("kg/(m*s^2)", PRESSURE),
    ("CMM", VOL_FLOW), ("m3/s", VOL_FLOW), ("m3/min", VOL_FLOW),
    ("%", DIMLESS), ("1", DIMLESS),
])
def test_unit_dimensions(text, expect):
    assert dim_of(text) == expect, dim_str(dim_of(text))


def test_digit_suffix_exponents():
    assert parse_unit("1/m4")[1] == tuple(-4 if i == 0 else 0 for i in range(7))
    assert parse_unit("Pa*s2/m6")[1] == parse_unit("Pa*s^2/m^6")[1]


def test_temperature_offsets():
    assert to_si(25.0, "degC") == pytest.approx(298.15)
    assert from_si(298.15, "degC") == pytest.approx(25.0)
    assert to_si(32.0, "degF") == pytest.approx(273.15)


def test_roundtrip():
    for unit, value in [("mmAq", 150.0), ("CMM", 262.0), ("mg/Nm3", 64.4), ("degC", 45.0)]:
        assert from_si(to_si(value, unit), unit) == pytest.approx(value)


def test_unknown_unit_raises():
    with pytest.raises(ValueError, match="알 수 없는 단위"):
        parse_unit("furlongs")
