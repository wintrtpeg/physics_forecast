"""가상 현장 데이터 생성기.

실데이터를 아직 못 받았어도 캘리브레이션 파이프라인 전체를 검증할 수 있어야 한다.
여기서는 '진짜 공장'을 물리 모델로 돌리되, 파라미터를 설계값에서 일부러 틀어놓고
(현장은 설계대로 지어지지 않는다) 계측 노이즈와 센서 바이어스를 얹는다.

캘리브레이션의 목표는 이 숨겨둔 '진짜 파라미터'를 관측만으로 되찾는 것이다.
그리고 결정적으로, **저부하 구간 데이터만 주고 고부하를 맞히는지** 본다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from model import TOOL_GROUPS, build
from pforecast.calib.runner import build_param_rows, simulate

#: 현장의 '진짜' 파라미터 (설계값 대비 어긋나 있음). 캘리브레이션이 되찾아야 할 값.
TRUE_PARAMS = {
    "SRC_DRY.ef_process": 8.60e-6,   # 설계 7.0 mg/s -> 실제 8.6 (레시피가 바뀌었다)
    "SRC_DRY.ef_idle": 0.80e-6,      # 설계 0.55 -> 실제 0.80
    "SRC_CVD.ef_process": 4.10e-6,   # 설계 5.0 -> 실제 4.1
    "SCR.eta_max": 0.47,             # 설계 0.55 -> 충전재 열화로 0.47
    "SCR.ntu_a": 11.0,               # 설계 14.0 -> 11.0
    "DCT_MAIN.K": 23.0,              # 설계 17.0 -> 덕트 오염으로 저항 증가
    "SCR.K": 68.0,                   # 설계 51.0 -> 충전층 막힘
}

#: 계측 특성 (표시 단위 기준)
SENSOR = {
    "STK.C_dry":    dict(unit="mg/Nm3", noise=2.5, bias=1.5, lod=1.0),
    "STK.Q_n":      dict(unit="Nm3/h",  noise=180.0, bias=-90.0),
    "STK.T_stack":  dict(unit="degC",   noise=0.35, bias=0.2),
    "FAN.dp_mmAq":  dict(unit="mmAq",   noise=4.0,  bias=0.0),
    "SCR.dp_mmAq":  dict(unit="mmAq",   noise=2.5,  bias=3.0),
}

TAGS = {
    "STK.C_dry":   "F2_UT_STK01_NOX_DRY",
    "STK.Q_n":     "F2_UT_STK01_FLOW",
    "STK.T_stack": "F2_UT_STK01_TEMP",
    "FAN.dp_mmAq": "F2_UT_SCR01_FAN_SP",
    "SCR.dp_mmAq": "F2_UT_SCR01_DP",
}

#: 지류별 질량유량계. 현장에 늘 있지는 않지만 있으면 가치가 크다 -
#: 이 컬럼들 사이에 성립하는 질량 보존을 툴이 **자동으로 찾아낸다**.
FLOW_TAGS = {
    "SRC_DRY.mdot": "F2_UT_SCR01_BR_DRY_FLOW",
    "SRC_CVD.mdot": "F2_UT_SCR01_BR_CVD_FLOW",
    "SRC_WET.mdot": "F2_UT_SCR01_BR_WET_FLOW",
    "SRC_IMP.mdot": "F2_UT_SCR01_BR_IMP_FLOW",
    "HDR.mdot_total": "F2_UT_SCR01_HDR_FLOW",
}
FLOW_SENSOR = dict(unit="kg/s", noise=0.012, bias=0.0)


def utilization_profile(n: int, rng: np.random.Generator) -> np.ndarray:
    """현실적인 가동율 시계열: 완만한 추세 + 주간 주기 + 단기 변동 + 가끔 감산.

    마지막 열흘은 **신제품 램프업**으로 가동율이 한 단계 올라간다. 앞 20일로 학습하고
    뒤 열흘을 맞히면, 그게 곧 '학습 범위 밖의 미래'를 예측하는 검증이 된다.
    (예전에는 한 달 내내 완만히 오르게 만들고 가동율 값으로 학습/검증을 갈랐는데,
    그러면 검증 행의 88% 가 학습보다 과거였다 — 미래 예측 검증이 아니었다.)
    """
    t = np.arange(n)
    day = t / 288.0
    trend = np.where(day < 20.0, 0.42 + 0.12 * day / 20.0,        # 기존 제품: 서서히 증가
                     0.75 + 0.10 * np.clip((day - 21.0) / 9.0, 0, 1))  # 램프업 후
    ramp = (day >= 20.0) & (day < 21.0)                           # 하루 동안 전환
    trend = np.where(ramp, 0.54 + (0.75 - 0.54) * (day - 20.0), trend)
    weekly = 0.035 * np.sin(2 * np.pi * t / (7 * 288))
    daily = 0.025 * np.sin(2 * np.pi * t / 288 + 0.7)
    # 단기 변동: 시간상수 6시간, 표준편차 0.02 의 AR(1). (예전의 누적합 잡음은 한 달에
    # ±0.3 까지 떠돌아 추세를 덮었다 — 가동율이 학습 구간에서 이미 0.95 에 닿았다.)
    phi = np.exp(-1.0 / 72.0)
    e = rng.normal(0, 0.02 * np.sqrt(1 - phi ** 2), n)
    noise = np.empty(n)
    noise[0] = rng.normal(0, 0.02)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + e[i]
    u = trend + weekly + daily + noise
    # 감산/PM 구간 (램프업 이전 기간에만 — 이후에 넣으면 학습 범위 안으로 돌아온다)
    early = int(20 * 288)
    for _ in range(max(1, n // 2200)):
        s = rng.integers(0, max(min(n, early) - 300, 1))
        w = int(rng.integers(60, 280))
        u[s:s + w] *= rng.uniform(0.45, 0.75)
    return np.clip(u, 0.05, 0.95)


def main() -> None:
    ap = argparse.ArgumentParser(description="NOx 계통 가상 운전 데이터 생성")
    ap.add_argument("--days", type=float, default=30.0, help="생성 기간 (일)")
    ap.add_argument("--start", default="2025-03-01", help="시작 일시")
    ap.add_argument("--seed", type=int, default=20250301)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "plant_5min.csv"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = int(args.days * 24 * 12)          # 5분 간격
    idx = pd.date_range(args.start, periods=n, freq="5min")

    model = build().compile()
    model.build()
    for name, val in TRUE_PARAMS.items():
        model.parameters[model.par_index(name)].value = val

    util = utilization_profile(n, rng)
    group_offset = {g[0]: o for g, o in zip(TOOL_GROUPS, [0.0, -0.04, 0.06, -0.02])}
    t_amb = (13.0 + 9.0 * np.sin(2 * np.pi * np.arange(n) / (365 * 288) - 1.2)
             + 5.5 * np.sin(2 * np.pi * np.arange(n) / 288 - 2.0)
             + rng.normal(0, 0.8, n))
    fan_ratio = np.clip(0.98 + 0.02 * np.sin(2 * np.pi * np.arange(n) / (3 * 288)), 0.9, 1.05)

    inputs = pd.DataFrame(index=idx)
    for g in TOOL_GROUPS:
        inputs[f"SRC_{g[0]}.util"] = np.clip(util + group_offset[g[0]], 0.02, 1.0)
    inputs["STK.T_amb"] = t_amb + 273.15
    inputs["FAN.n_ratio"] = fan_ratio

    expansion = {"STK.T_amb": ["STK.T_amb", "DCT_MAIN.T_amb"]
                 + [f"DCT_{g[0]}.T_amb" for g in TOOL_GROUPS]}
    rows = build_param_rows(model, inputs, base_p=model.p0(), expansion=expansion)
    targets = list(SENSOR) + list(FLOW_TAGS)
    print(f"{n} 스텝 시뮬레이션 중 ...")
    sim = simulate(model, rows, targets, index=idx)
    print(f"  수렴률 {sim.success_rate*100:.2f}%")

    from pforecast.core.units import from_si
    out = pd.DataFrame(index=idx)
    out.index.name = "timestamp"
    for name, tag in FLOW_TAGS.items():
        true_disp = np.array([from_si(v, FLOW_SENSOR["unit"]) for v in sim.values[name]])
        out[tag] = np.round(true_disp + rng.normal(0, FLOW_SENSOR["noise"], n), 4)
    for name, spec in SENSOR.items():
        unit = spec["unit"]
        true_disp = np.array([from_si(v, unit) for v in sim.values[name]])
        meas = true_disp + spec["bias"] + rng.normal(0, spec["noise"], n)
        if "lod" in spec:
            meas = np.maximum(meas, spec["lod"])
        out[TAGS[name]] = np.round(meas, 3)

    # 운전 입력 태그 (현장에서 같이 수집되는 값)
    for g in TOOL_GROUPS:
        out[f"F2_FAB_{g[0]}_UTIL"] = np.round(
            inputs[f"SRC_{g[0]}.util"].to_numpy() * 100.0 + rng.normal(0, 0.4, n), 2)
    out["F2_UT_AMB_TEMP"] = np.round(t_amb + rng.normal(0, 0.2, n), 2)
    out["F2_UT_SCR01_FAN_HZ"] = np.round(fan_ratio * 60.0, 2)

    # 통신 두절 구간 (현장 데이터에는 반드시 있다)
    for _ in range(max(1, n // 3000)):
        s = rng.integers(0, max(n - 40, 1))
        out.iloc[s:s + int(rng.integers(3, 30))] = np.nan

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)
    print(f"저장: {path}  ({len(out)} 행, {len(out.columns)} 태그)")
    print(f"  기간 {idx[0]} ~ {idx[-1]}")
    print(f"  가동율 범위 {util.min()*100:.1f}% ~ {util.max()*100:.1f}%")
    print(f"  NOx 범위 {out[TAGS['STK.C_dry']].min():.1f} ~ {out[TAGS['STK.C_dry']].max():.1f} mg/Sm3")


if __name__ == "__main__":
    main()
