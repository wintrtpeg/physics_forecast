"""물리 잔차 진단 · 계측기 이득/영점 진단 · 입력 경계 처리 · 결측 허용 보정."""

import numpy as np
import pandas as pd
import pytest

from pforecast.analyze.dimensional import find_balances
from pforecast.calib import CalibrationSpec, build_param_rows, calibrate, simulate
from pforecast.calib.diagnostics import binary_segmentation, residual_changepoints
from pforecast.scenario import load_python_model


def test_binary_segmentation_finds_a_step():
    rng = np.random.default_rng(0)
    y = np.r_[rng.normal(0, 1, 60), rng.normal(6, 1, 60)]
    cps = binary_segmentation(y, sigma=1.0, min_seg=5, z_min=5.0)
    assert len(cps) == 1 and abs(cps[0][0] - 60) <= 2


def test_changepoint_interpretation_uses_other_observations_and_inputs():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2025-01-01", periods=60 * 288, freq="5min")
    n = len(idx)
    day = np.asarray((idx - idx[0]) / pd.Timedelta(days=1))
    resid = pd.DataFrame({
        "a": rng.normal(0, 2, n) + np.where(day >= 20, 5.0, 0.0),     # a 만 단독 계단
        "b": rng.normal(0, 2, n) + np.where(day >= 40, -4.0, 0.0),    # 입력 변경일
        "c": rng.normal(0, 2, n),
    }, index=idx)
    inputs = pd.DataFrame({"hz": np.where(day >= 40, 61.0, 58.0)}, index=idx)
    cps = residual_changepoints(resid, inputs, {"a": "mg"})
    got = {(c.observation, str(c.date.date())): c for c in cps}
    a = got[("a", "2025-01-21")]
    assert a.kind == "단독 변화" and a.shift == pytest.approx(5.0, abs=0.8)
    b = got[("b", "2025-02-10")]
    assert b.kind == "입력 변경일" and b.inputs_changed == ["hz"]
    assert not any(c.observation == "c" for c in cps)


def test_meter_gain_error_is_diagnosed_from_a_mass_balance():
    rng = np.random.default_rng(2)
    n = 3000
    a = 1.2 + 0.2 * rng.random(n)
    b = 0.9 + 0.2 * rng.random(n)
    c = 0.3 + 0.1 * rng.random(n)
    df = pd.DataFrame({"A": a + rng.normal(0, 0.005, n), "B": b + rng.normal(0, 0.005, n),
                       "C": c + rng.normal(0, 0.005, n),
                       "H": 1.03 * (a + b + c) + rng.normal(0, 0.005, n)})
    df.loc[1500:, "C"] = np.nan                    # 도중에 고장 난 지류 계측기
    bal = find_balances(df, {k: "kg/s" for k in df.columns})
    full = [x for x in bal if set(x.columns) == {"A", "B", "C", "H"}]
    assert full and full[0].confidence == "확실"
    assert full[0].gain == pytest.approx(1.03, abs=0.003)
    assert "+3.0%" in full[0].gain_note()


def test_redundant_temperature_sensors_report_offset_not_gain():
    rng = np.random.default_rng(3)
    t = 40 + 5 * np.sin(np.linspace(0, 20, 4000))
    df = pd.DataFrame({"T1": t + rng.normal(0, 0.1, 4000),
                       "T2": t + 0.45 + rng.normal(0, 0.1, 4000)})
    bal = find_balances(df, {"T1": "degC", "T2": "degC"})
    assert bal and bal[0].is_redundant_pair
    assert bal[0].offset == pytest.approx(-0.45, abs=0.02)     # mean(T1 - T2)
    note = bal[0].gain_note()
    assert note.startswith("T2 가 T1 보다") and "℃" in note and "%" not in note


@pytest.fixture(scope="module")
def nox():
    m = load_python_model("examples/nox_stack/model.py").compile()
    m.build()
    return m


def test_inputs_outside_physical_bounds_are_clipped(nox):
    inp = pd.DataFrame({"SCR.L": [0.55 / 60, 0.0]})
    rep = {}
    rows = build_param_rows(nox, inp, clip_report=rep)
    assert rep == {"SCR.L": 1}
    assert rows[1, nox.par_index("SCR.L")] > 0
    sim = simulate(nox, rows, ["STK.C_dry", "SCR.eta"])
    assert sim.ok.all()
    # 펌프 정지 = 제거효율 0 -> 굴뚝 농도가 오른다
    assert sim.values["SCR.eta"].iloc[1] < 1e-3
    assert sim.values["STK.C_dry"].iloc[1] > sim.values["STK.C_dry"].iloc[0]


def test_calibration_uses_rows_with_partial_observations(nox):
    j = nox.par_index("SRC_DRY.ef_process")
    true = nox.parameters[j].value * 1.3
    util = np.linspace(0.3, 0.9, 16)
    inp = pd.DataFrame({f"SRC_{g}.util": util for g in ("DRY", "CVD", "WET", "IMP")})
    rows = build_param_rows(nox, inp)
    rows[:, j] = true
    sim = simulate(nox, rows, ["STK.C_dry", "STK.Q_n"])
    obs = sim.values.copy()
    obs.loc[::2, "STK.Q_n"] = np.nan              # 유량계가 절반 비어 있어도
    spec = CalibrationSpec(params=["SRC_DRY.ef_process"], observations=["STK.C_dry", "STK.Q_n"],
                           sigmas={"STK.C_dry": 3e-6, "STK.Q_n": 0.05}, max_rows=50,
                           prior_weight=0.0)
    before = nox.parameters[j].value
    try:
        res = calibrate(nox, inp, obs, spec, verbose=False)
        assert res.n_rows == 16                    # 행을 버리지 않았다
        assert res.fitted[0] == pytest.approx(true, rel=1e-3)
    finally:
        nox.parameters[j].value = before
        nox.system.components["SRC_DRY"].set_param("ef_process", before)


def test_simulation_recovers_after_a_pump_stop(nox):
    """펌프 정지(액가스비 ≈ 0) 시점의 해에서 출발하면 펌프가 다시 돈 시점을 못 푼다.

    예전에는 실패하면 출발점이 갱신되지 않아 그 뒤 몇 주가 연쇄적으로 빠졌다 (시드 5150,
    검증 구간 65%). 이제는 기본 출발점에서 한 번 더 푼다.
    """
    from pforecast.calib.runner import simulate as sim
    from pforecast.core.units import to_si
    vals = {"SRC_DRY.ef_process": to_si(12.453, "mg/s"), "SRC_DRY.ef_idle": to_si(1.105, "mg/s"),
            "SRC_CVD.ef_process": to_si(7.852, "mg/s"), "SCR.ntu_a": 27.637,
            "DCT_MAIN.K": 28.168, "SCR.K": 70.93}
    saved = {k: nox.parameters[nox.par_index(k)].value for k in vals}
    try:
        for k, v in vals.items():
            nox.parameters[nox.par_index(k)].value = v
        n = 3
        inp = pd.DataFrame({"SRC_DRY.util": [0.8, 0.49, 0.49], "SRC_CVD.util": [0.78] * n,
                            "SRC_WET.util": [0.8] * n, "SRC_IMP.util": [0.83] * n,
                            "SRC_DRY.n_tools": [30.0] * n, "STK.T_amb": [303.0] * n,
                            "FAN.n_ratio": [61 / 60] * n, "SCR.L": [42 / 3600, 0.0, 42 / 3600]})
        res = sim(nox, build_param_rows(nox, inp), ["STK.C_dry"])
        assert res.ok.all()
    finally:
        for k, v in saved.items():
            nox.parameters[nox.par_index(k)].value = v
