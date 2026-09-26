"""현장형(비이상적) 가상 데이터 — 실데이터 없이 툴을 괴롭혀 본다.

``nox_stack/make_synthetic_data.py`` 는 '깨끗한 공장'이다. 현장 데이터는 그렇지 않다.
이 생성기는 실제로 겪는 문제를 세 층으로 얹는다. 각 문제가 언제 어디에 들어갔는지는
정답 파일(``anomalies.json``, ``field_truth.csv``)에 따로 적어 두고, 툴이 그것을 얼마나
찾아내는지 ``evaluate.py`` 가 채점한다.

1. 공정 — 물리는 맞지만 모델이 모르는 것
   * 레시피 변경으로 DRY 배출계수 +15% (04-07, 비계측)
   * 충전재 오염으로 스크러버 효율·차압이 서서히 나빠지다가 세정 PM 에 복원 (06-21)
   * 메인 덕트 분진 퇴적 (저항 서서히 증가)
   * CVD 챔버 클리닝(NF3) 때마다 NOx 펄스 (하루 3회 안팎, 비계측)
   * 팬 마모(성능곡선 처짐), NTU 지수가 설계와 다름 — 보정 대상이 아닌 모델 오차
   * 6월 DRY 장비 6대 증설, 여름 고온, 가동율 상승 — **학습 구간에 없던 운전영역**
2. 계측
   * NOx 분석계 매일 03:00 자동 점검(영점 -> 스팬 가스), 월 1회 수동 교정, 드리프트
   * 분석계 응답 지연, 교체 후 오프셋
   * 유량계 고착, 온도 센서 단선 스파이크, 정압 전송기 교체 오프셋
   * 지류 유량계 고장, 헤더 유량계 게인 오차
   * MES 가동율은 1시간 단위 갱신 (계단형)
3. 파일
   * CP949 인코딩, 3행 헤더(태그/설명/단위), 첫 컬럼 이름 '일시'
   * 엑셀 저장분은 '2025-01-01 오후 3:05' 형식 + 천 단위 콤마, 시스템 추출분은 ISO
   * 월별 파일을 이어 붙이며 생긴 중복 구간, 중간 헤더, 순서 뒤바뀜
   * 'Bad', 'I/O Timeout', 'Comm Fail', 'Maint', '#N/A' 같은 상태 문자열
   * 끝의 빈 컬럼, 빈 줄, 통째로 빠진 하루

사용::

    python examples/nox_field/make_field_data.py            # 약 7분 (정상상태 7만 회)
    python examples/nox_field/make_field_data.py --reuse    # 물리 시뮬레이션 재사용

정답지(``data/_answer/``)는 채점용이다. 툴은 ``field_raw.csv`` 만 본다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nox_stack"))

from model import TOOL_GROUPS, build  # noqa: E402

from pforecast.calib.runner import build_param_rows, simulate  # noqa: E402
from pforecast.core.units import from_si, to_si  # noqa: E402

START, END = "2025-01-01 00:00", "2025-08-31 23:55"
GROUPS = [g[0] for g in TOOL_GROUPS]

#: 사건 달력. 정답지이자 문서다.
EV = {
    "seollal": ("2025-01-28", "2025-01-31"),
    "comm_fail": ("2025-02-14 13:00", "2025-02-14 19:00"),
    "server_down": ("2025-03-09 00:00", "2025-03-10 00:00"),
    "nox_maint": ("2025-03-18 09:00", "2025-03-20 17:00"),
    "imp_meter_dead": "2025-04-01 00:00",
    "recipe_change": "2025-04-07 06:00",
    "pump_maint": ("2025-04-15 06:00", "2025-04-15 14:00"),
    "fab_pm": ("2025-05-03 00:00", "2025-05-06 00:00"),
    "mes_na": ("2025-05-12 00:00", "2025-05-12 12:00"),
    "flow_frozen": ("2025-05-20 08:00", "2025-05-22 16:00"),
    "fan_sp_offset": "2025-05-28 10:00",
    "expansion": "2025-06-10 00:00",
    "packing_clean": ("2025-06-21 08:00", "2025-06-21 18:00"),
    "water_increase": "2025-07-01 10:00",
    "analyzer_swap": "2025-07-15 10:00",
    "heat_wave": ("2025-07-24", "2025-08-07"),
}

#: 송풍기 인버터 설정 이력 (운전원이 바꾼 값 그대로)
FAN_HZ = [
    ("2025-01-01 00:00", 58.0), ("2025-02-20 10:00", 58.5), ("2025-04-01 09:00", 59.0),
    ("2025-05-03 00:00", 45.0), ("2025-05-06 06:00", 59.0), ("2025-06-10 09:00", 61.0),
    ("2025-07-20 14:00", 62.0), ("2025-08-25 10:00", 61.0),
]

#: 설계값(모델 기본값)과 다른 현장의 '진짜' 상수. 시변 파라미터는 아래에서 만든다.
TRUE_CONST = {
    "SRC_DRY.ef_idle": 0.75,      # mg/s   (설계 0.55)
    "SRC_CVD.ef_process": 4.3,    # mg/s   (설계 5.0)
    "SCR.ntu_a": 11.5,            # -      (설계 14.0)
    "SCR.ntu_b": 0.62,            # -      (설계 0.55)  <- 보정 대상 아님: 모델 오차
    "FAN.c2": -71.0,              # 팬 마모 (설계 -68) <- 보정 대상 아님: 모델 오차
}

SPAN_GAS = 160.0          # 분석계 스팬 가스 (레인지 200 의 80%)
DRIFT_PER_DAY = 0.12      # mg/Sm3/일, 월 1회 수동 교정으로 0 복귀
CAL_SKIP = 0.10           # 자동 점검을 건너뛰는 날의 비율

#: 계측 결함의 크기와 시각. 기본값이 개발에 쓴 데이터다. ``--holdout`` 은 이 값들과
#: 사건 날짜를 시드로 흔들어, 툴을 만들며 한 번도 보지 않은 데이터를 만든다
#: (탐지 임계값을 개발 데이터에 맞춰 놓았을 수 있으므로 — 그것도 누수다).
VAR = {
    "cal_hour": 3.0,            # 자동 점검 시각
    "analyzer_offset": -3.0,    # 분석계 교체 후 오프셋 [mg/Sm3]
    "fan_sp_offset": 12.0,      # 정압 전송기 교체 오프셋 [mmAq]
    "flow_gain": 1.02,          # 굴뚝 유량계 게인
    "hdr_gain": 1.03,           # 헤더 유량계 게인
    "temp2_offset": 0.45,       # 예비 온도계 영점 [degC]
}


# --------------------------------------------------------------------------
# 1. 공정 (진짜 운전 조건과 숨은 파라미터)
# --------------------------------------------------------------------------

def _t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _between(idx: pd.DatetimeIndex, a: str, b: str) -> np.ndarray:
    return np.asarray((idx >= _t(a)) & (idx < _t(b)))


def _ar1(n: int, tau_steps: float, sigma: float, rng) -> np.ndarray:
    phi = np.exp(-1.0 / tau_steps)
    e = rng.normal(0.0, sigma * np.sqrt(1.0 - phi ** 2), n)
    e[0] = rng.normal(0.0, sigma)
    return lfilter([1.0], [1.0, -phi], e)


def _smooth_step(idx, center: str, width_h: float) -> np.ndarray:
    """0 -> 1 로 부드럽게 바뀌는 계단 (램프업/다운)."""
    x = np.asarray((idx - _t(center)) / pd.Timedelta(hours=width_h), dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x * 6.0, -60.0, 60.0)))


def process_truth(idx: pd.DatetimeIndex, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """진짜 운전 조건(표시 단위)과 숨은 파라미터(표시 단위)."""
    n = len(idx)
    days = np.asarray((idx - idx[0]) / pd.Timedelta(days=1), dtype=float)
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    doy = idx.dayofyear.to_numpy().astype(float)

    # ---- 가동율 -----------------------------------------------------------
    keys = pd.DatetimeIndex([_t(s) for s in ("2025-01-01", "2025-05-31", "2025-06-09",
                                             "2025-07-15", "2025-08-31 23:55")])
    kd = np.asarray((keys - idx[0]) / pd.Timedelta(days=1), dtype=float)
    base = np.interp(days, kd, [0.44, 0.66, 0.68, 0.84, 0.90])
    base = base + 0.02 * np.sin(2 * np.pi * (hour - 8.0) / 24.0)
    base = base - 0.03 * (idx.dayofweek.to_numpy() >= 5)
    base = np.where(_between(idx, *EV["seollal"]), base * 0.75, base)
    # 팹 정기 PM: 6시간에 걸쳐 내려가고 12시간에 걸쳐 올라온다
    pm0, pm1 = EV["fab_pm"]
    down = _smooth_step(idx, pm0, 3.0)
    up = _smooth_step(idx, pm1, 6.0)
    pm_w = down * (1.0 - up)
    base = base * (1.0 - pm_w) + 0.03 * pm_w

    offsets = {"DRY": 0.0, "CVD": -0.04, "WET": 0.06, "IMP": -0.02}
    util = {}
    for g in GROUPS:
        indiv = _ar1(n, tau_steps=2 * 288, sigma=0.035, rng=rng)
        util[g] = base + offsets[g] + indiv * (1.0 - pm_w)
    # 개별 장비군 감산 3건
    for g, s, hrs in [("CVD", "2025-02-25 08:00", 18), ("WET", "2025-04-23 20:00", 10),
                      ("DRY", "2025-07-08 07:00", 14)]:
        e = _t(s) + pd.Timedelta(hours=hrs)
        m = np.asarray((idx >= _t(s)) & (idx < e))
        util[g] = np.where(m, util[g] * 0.6, util[g])
    # 증설: 신규 6대는 열흘간 인증(가동 5%) 후 월말까지 기존 장비 수준으로 올라온다
    n_dry = np.where(idx >= _t(EV["expansion"]), 30.0, 24.0)
    ramp = np.clip((days - (_t("2025-06-20") - idx[0]).days) / 10.0, 0.0, 1.0)
    new_u = 0.05 + (util["DRY"] - 0.05) * ramp
    util["DRY"] = np.where(n_dry > 24, (24 * util["DRY"] + 6 * new_u) / 30.0, util["DRY"])
    for g in GROUPS:
        util[g] = np.clip(util[g], 0.02, 1.0)

    # ---- 외기 --------------------------------------------------------------
    clim = 12.3 - 14.8 * np.cos(2 * np.pi * (doy - 18.0) / 365.0)
    diurnal = 4.5 * np.sin(2 * np.pi * (hour - 9.0) / 24.0)
    synoptic = _ar1(n, tau_steps=3 * 288, sigma=2.6, rng=rng)
    hw0, hw1 = EV["heat_wave"]
    heat = 3.5 * _smooth_step(idx, hw0, 24.0) * (1.0 - _smooth_step(idx, hw1, 24.0))
    t_amb = clim + diurnal + synoptic + heat
    rh = np.clip(64.0 + 12.0 * np.sin(2 * np.pi * (doy - 110.0) / 365.0)
                 - 1.8 * diurnal + _ar1(n, 288, 8.0, rng), 18.0, 100.0)

    # ---- 운전 설정 ----------------------------------------------------------
    hz = np.zeros(n)
    for s, v in FAN_HZ:
        hz[np.asarray(idx >= _t(s))] = v
    water = np.where(idx >= _t(EV["water_increase"]), 42.0, 33.0)
    water = water + _ar1(n, 36, 0.25, rng)
    pump_off = _between(idx, *EV["pump_maint"]) | _between(idx, *EV["packing_clean"])
    water = np.where(pump_off, 0.0, water)

    # ---- 숨은 파라미터 -------------------------------------------------------
    ef_dry = np.where(idx >= _t(EV["recipe_change"]), 9.4, 8.2)
    # CVD 챔버 클리닝 펄스: 하루 평균 3회, 15~25분, 가동 배출 +50%
    pulse = np.zeros(n)
    for d0 in pd.date_range(idx[0].normalize(), idx[-1].normalize(), freq="D"):
        for _ in range(rng.poisson(3.0)):
            s = d0 + pd.Timedelta(minutes=int(rng.integers(0, 24 * 60)))
            i0 = idx.searchsorted(s)
            pulse[i0:i0 + int(rng.integers(3, 6))] = 1.0
    ef_cvd = TRUE_CONST["SRC_CVD.ef_process"] * (1.0 + 0.5 * pulse)
    clean = _t(EV["packing_clean"][1])
    since_clean = np.where(idx >= clean, np.asarray((idx - clean) / pd.Timedelta(days=1)),
                           days)
    eta_max = np.where(idx >= clean, 0.51, 0.50) - 0.045 * since_clean / 171.0
    scr_k = np.where(idx >= clean, 55.0 + 13.0 * since_clean / 171.0,
                     62.0 + 13.0 * days / 171.0)
    duct_k = 22.0 + 3.0 * days / days[-1]

    drivers = pd.DataFrame({
        **{f"util_{g}": util[g] for g in GROUPS},
        "n_tools_DRY": n_dry, "T_amb": t_amb, "RH_amb": rh,
        "fan_hz": hz, "water_m3h": water,
    }, index=idx)
    hidden = pd.DataFrame({
        "ef_dry_process": ef_dry, "cvd_pulse": pulse, "ef_cvd_process": ef_cvd,
        "eta_max": eta_max, "scr_K": scr_k, "duct_K": duct_k,
    }, index=idx)
    return drivers, hidden


def simulate_truth(drivers: pd.DataFrame, hidden: pd.DataFrame) -> pd.DataFrame:
    """진짜 공장을 물리 모델로 돌린다 (숨은 파라미터가 시점마다 다르다)."""
    model = build().compile()
    model.build()
    for name, val in TRUE_CONST.items():
        unit = model.parameters[model.par_index(name)].unit
        model.parameters[model.par_index(name)].value = to_si(val, unit)

    si = pd.DataFrame(index=drivers.index)
    for g in GROUPS:
        si[f"SRC_{g}.util"] = drivers[f"util_{g}"]
    si["SRC_DRY.n_tools"] = drivers["n_tools_DRY"]
    si["STK.T_amb"] = drivers["T_amb"] + 273.15
    si["FAN.n_ratio"] = drivers["fan_hz"] / 60.0
    # 펌프 정지(0) 를 그대로 넣으면 L/G -> 0 에서 NTU 기울기가 발산한다.
    # 진짜 공장은 당연히 돌아가므로 여기서는 물리적 하한(1e-6 m3/min)으로 둔다.
    si["SCR.L"] = np.maximum(drivers["water_m3h"] / 60.0, 1e-6) / 60.0
    si["SRC_DRY.ef_process"] = hidden["ef_dry_process"] * 1e-6
    si["SRC_CVD.ef_process"] = hidden["ef_cvd_process"] * 1e-6
    si["SCR.eta_max"] = hidden["eta_max"]
    si["SCR.K"] = hidden["scr_K"]
    si["DCT_MAIN.K"] = hidden["duct_K"]
    expansion = {"STK.T_amb": ["STK.T_amb", "DCT_MAIN.T_amb"] + [f"DCT_{g}.T_amb" for g in GROUPS]}
    rows = build_param_rows(model, si, base_p=model.p0(), expansion=expansion)

    targets = ["STK.C_dry", "STK.Q_n", "STK.T_stack", "FAN.dp_mmAq", "SCR.dp_mmAq",
               "SCR.eta", "HDR.mdot_total"] + [f"SRC_{g}.mdot" for g in GROUPS]
    t0 = time.perf_counter()
    print(f"진짜 공장 시뮬레이션: 정상상태 {len(rows):,}회 ...", flush=True)
    sim = simulate(model, rows, targets, index=drivers.index)
    print(f"  {time.perf_counter()-t0:.0f}s, 수렴률 {sim.success_rate*100:.3f}%", flush=True)
    if sim.success_rate < 0.999:
        raise RuntimeError("진짜 공장 시뮬레이션이 수렴하지 않은 시점이 있습니다")
    units = {"STK.C_dry": "mg/Nm3", "STK.Q_n": "Nm3/h", "STK.T_stack": "degC",
             "FAN.dp_mmAq": "mmAq", "SCR.dp_mmAq": "mmAq", "SCR.eta": "%",
             "HDR.mdot_total": "kg/s", **{f"SRC_{g}.mdot": "kg/s" for g in GROUPS}}
    out = pd.DataFrame(index=drivers.index)
    for k, u in units.items():
        out[k] = [from_si(v, u) for v in sim.values[k].to_numpy()]
    return out


# --------------------------------------------------------------------------
# 2. 계측
# --------------------------------------------------------------------------

#: CSV 컬럼 (현장 추출 순서 그대로 - 논리적으로 정렬되어 있지 않다)
COLUMNS = [
    # tag, 설명, 단위
    ("F2_UT_STK01_NOX_DRY", "굴뚝 NOx(건조)", "mg/Sm³"),
    ("F2_UT_STK01_FLOW", "굴뚝 배출유량", "Sm³/h"),
    ("F2_UT_STK01_TEMP", "굴뚝 온도", "℃"),
    ("F2_UT_STK01_TEMP_2", "굴뚝 온도(예비)", "℃"),
    ("F2_UT_SCR01_FAN_SP", "유인송풍기 정압", "mmAq"),
    ("F2_UT_SCR01_FAN_HZ", "유인송풍기 인버터", "Hz"),
    ("F2_UT_SCR01_DP", "스크러버 차압", "mmAq"),
    ("F2_UT_SCR01_CIRC_FLOW", "스크러버 순환수량", "㎥/h"),
    ("F2_UT_SCR01_PH", "순환수 pH", "pH"),
    ("F2_UT_SCR01_LEVEL", "순환조 레벨", "%"),
    ("F2_UT_SCR01_PUMP_A_RUN", "순환펌프 A 운전", ""),
    ("F2_UT_SCR01_SPARE_AI", "예비 AI", "mA"),
    ("F2_UT_SCR01_BR_DRY_FLOW", "DRY 지류 유량", "kg/s"),
    ("F2_UT_SCR01_BR_CVD_FLOW", "CVD 지류 유량", "kg/s"),
    ("F2_UT_SCR01_BR_WET_FLOW", "WET 지류 유량", "kg/s"),
    ("F2_UT_SCR01_BR_IMP_FLOW", "IMP 지류 유량", "kg/s"),
    ("F2_UT_SCR01_HDR_FLOW", "헤더 합류 유량", "kg/s"),
    ("F2_FAB_DRY_UTIL", "건식식각 가동률", "%"),
    ("F2_FAB_CVD_UTIL", "증착 가동률", "%"),
    ("F2_FAB_WET_UTIL", "습식세정 가동률", "%"),
    ("F2_FAB_IMP_UTIL", "이온주입 가동률", "%"),
    ("F2_FAB_DRY_EQP_CNT", "건식식각 설치대수", "대"),
    ("F2_UT_AMB_TEMP", "외기온도", "℃"),
    ("F2_UT_AMB_HUMID", "외기습도", "%RH"),
]
TAGS = [c[0] for c in COLUMNS]
DECIMALS = {"F2_UT_STK01_NOX_DRY": 1, "F2_UT_STK01_FLOW": 1, "F2_UT_STK01_TEMP": 2,
            "F2_UT_STK01_TEMP_2": 2, "F2_UT_SCR01_FAN_SP": 1, "F2_UT_SCR01_FAN_HZ": 1,
            "F2_UT_SCR01_DP": 1, "F2_UT_SCR01_CIRC_FLOW": 1, "F2_UT_SCR01_PH": 2,
            "F2_UT_SCR01_LEVEL": 1, "F2_UT_SCR01_PUMP_A_RUN": 0, "F2_UT_SCR01_SPARE_AI": 2,
            "F2_FAB_DRY_EQP_CNT": 0, "F2_UT_AMB_TEMP": 1, "F2_UT_AMB_HUMID": 1}


class Sheet:
    """계측값 + 셀 단위 정답 라벨 + 상태 문자열."""

    def __init__(self, idx: pd.DatetimeIndex):
        self.idx = idx
        self.n = len(idx)
        self.val = {t: np.full(self.n, np.nan) for t in TAGS}
        self.label = {t: np.full(self.n, "", dtype=object) for t in TAGS}
        self.status = {t: np.full(self.n, None, dtype=object) for t in TAGS}
        self.events: list[dict] = []

    def mask(self, a: str, b: str | None = None) -> np.ndarray:
        if b is None:
            return np.asarray(self.idx >= _t(a))
        return _between(self.idx, a, b)

    def mark(self, tag: str, m: np.ndarray, label: str, status: str | None = None):
        self.label[tag][m] = label
        if status is not None:
            self.status[tag][m] = status

    def event(self, kind: str, typ: str, tags, start, end=None, desc: str = "", **extra):
        self.events.append({"kind": kind, "type": typ, "tags": list(tags),
                            "start": str(start), "end": str(end) if end else None,
                            "desc": desc, **extra})


def measure(drivers: pd.DataFrame, true: pd.DataFrame, rng) -> Sheet:
    idx = drivers.index
    sh = Sheet(idx)
    n = len(idx)
    V = sh.val

    # ---- NOx 분석계 --------------------------------------------------------
    c = true["STK.C_dry"].to_numpy()
    # 샘플링 라인 지연 5분 + 1차 응답
    lagged = np.r_[c[0], c[:-1]]
    resp = lfilter([0.6], [1.0, -0.4], lagged, zi=[lagged[0] * 0.4])[0]
    drift = np.zeros(n)
    swap = _t(EV["analyzer_swap"])
    last_reset = idx[0]
    for i, t in enumerate(idx):
        if (t.day == 1 and t.hour == 10 and t.minute == 0) or t == swap:
            last_reset = t
        drift[i] = DRIFT_PER_DAY * (t - last_reset) / pd.Timedelta(days=1)
    offset = np.where(idx >= swap, VAR["analyzer_offset"], 0.0)
    V["F2_UT_STK01_NOX_DRY"] = resp + drift + offset + rng.normal(0, 1.8, n)
    sh.event("sensor", "drift", ["F2_UT_STK01_NOX_DRY"], idx[0], idx[-1],
             f"분석계 드리프트 +{DRIFT_PER_DAY} mg/Sm3/일, 매월 1일 10:00 수동 교정으로 복귀")
    sh.event("sensor", "level_shift", ["F2_UT_STK01_NOX_DRY"], swap, None,
             f"NOx 분석계 교체 — 새 분석계 오프셋 {VAR['analyzer_offset']:+.2f} mg/Sm3",
             magnitude=VAR["analyzer_offset"])

    tag = "F2_UT_STK01_NOX_DRY"
    maint_a, maint_b = EV["nox_maint"]
    maint = sh.mask(maint_a, maint_b)
    days = pd.date_range(idx[0].normalize(), idx[-1].normalize(), freq="D")
    n_cal = 0
    cal_at = pd.Timedelta(hours=VAR["cal_hour"])
    for d in days:
        if rng.random() < CAL_SKIP or maint[min(idx.searchsorted(d + cal_at), n - 1)]:
            continue
        n_cal += 1
        i0 = idx.searchsorted(d + cal_at)
        if i0 + 5 > n or idx[i0] != d + cal_at:
            continue
        V[tag][i0:i0 + 2] = rng.normal(0.4, 0.3, 2)
        sh.label[tag][i0:i0 + 2] = "cal_zero"
        V[tag][i0 + 2:i0 + 4] = rng.normal(SPAN_GAS, 0.8, 2)
        sh.label[tag][i0 + 2:i0 + 4] = "cal_span"
        V[tag][i0 + 4] = 0.5 * SPAN_GAS + 0.5 * V[tag][i0 + 4]
        sh.label[tag][i0 + 4] = "cal_recover"
    c0 = pd.Timestamp(0) + cal_at
    sh.event("sensor", "calibration", [tag], f"매일 {c0:%H:%M}",
             f"{c0 + pd.Timedelta(minutes=25):%H:%M}",
             f"자동 점검: 영점가스 10분 -> 스팬가스({SPAN_GAS:g}) 10분 -> 회복 5분. "
             f"{n_cal}일 수행, 약 {CAL_SKIP*100:.0f}% 건너뜀", n_days=n_cal)
    sh.mark(tag, maint, "maint", "Maint")
    sh.event("sensor", "maintenance", [tag], maint_a, maint_b, "분석계 정비 — 'Maint' 기록")
    for d in pd.date_range("2025-02-01", "2025-08-01", freq="MS"):
        m = sh.mask(str(d + pd.Timedelta(hours=10)), str(d + pd.Timedelta(minutes=635)))
        sh.mark(tag, m, "maint", "Maint")
    sh.event("sensor", "maintenance", [tag], "매월 1일 10:00", "10:35", "월간 수동 교정 — 'Maint'")
    # 시료 라인 응축 스파이크
    cand = np.where(sh.label[tag] == "")[0]
    for i in rng.choice(cand, 12, replace=False):
        V[tag][i] += rng.uniform(40, 80)
        sh.label[tag][i] = "spike"
    sh.event("sensor", "spike", [tag], idx[0], idx[-1], "시료 라인 응축 스파이크 12건 (+40~80)",
             count=12)

    # ---- 굴뚝 유량계 --------------------------------------------------------
    tag = "F2_UT_STK01_FLOW"
    q = true["STK.Q_n"].to_numpy()
    V[tag] = q * VAR["flow_gain"] + rng.normal(0, 0.012, n) * q
    cand = np.where(sh.label[tag] == "")[0]
    for i in rng.choice(cand, 15, replace=False):
        V[tag][i] *= rng.choice([0.7, 1.3])
        sh.label[tag][i] = "spike"
    sh.event("sensor", "spike", [tag], idx[0], idx[-1], "초음파 유량계 액적 스파이크 15건 (x0.7/x1.3)",
             count=15)
    fa, fb = EV["flow_frozen"]
    fz = sh.mask(fa, fb)
    V[tag][fz] = V[tag][np.argmax(fz) - 1]
    sh.mark(tag, fz, "frozen")
    sh.event("sensor", "frozen", [tag], fa, fb, "유량계 통신 모듈 이상 — 마지막 값 유지")
    sh.event("sensor", "gain", [tag], idx[0], idx[-1],
             f"유량계 게인 {(VAR['flow_gain'] - 1) * 100:+.1f}%", magnitude=VAR["flow_gain"] - 1)

    # ---- 굴뚝 온도 (2중화) ---------------------------------------------------
    ts = true["STK.T_stack"].to_numpy()
    V["F2_UT_STK01_TEMP"] = ts + rng.normal(0, 0.3, n)
    V["F2_UT_STK01_TEMP_2"] = ts + VAR["temp2_offset"] + rng.normal(0, 0.3, n)
    tag = "F2_UT_STK01_TEMP"
    for s, k, v in [("2025-02-03 14:10", 2, 999.9), ("2025-06-02 03:40", 1, -40.0),
                    ("2025-08-11 22:15", 3, 999.9)]:
        i0 = idx.searchsorted(_t(s))
        V[tag][i0:i0 + k] = v
        sh.label[tag][i0:i0 + k] = "spike"
        sh.event("sensor", "spike", [tag], s, None, f"열전대 단선 — {v:g} 가 {k}점", count=k)

    # ---- 송풍기 / 스크러버 ------------------------------------------------------
    tag = "F2_UT_SCR01_FAN_SP"
    V[tag] = true["FAN.dp_mmAq"].to_numpy() + rng.normal(0, 3.0, n)
    V[tag] = V[tag] + np.where(idx >= _t(EV["fan_sp_offset"]), VAR["fan_sp_offset"], 0.0)
    sh.event("sensor", "level_shift", [tag], EV["fan_sp_offset"], None,
             f"정압 전송기 교체 — 영점 미조정으로 {VAR['fan_sp_offset']:+.1f} mmAq",
             magnitude=VAR["fan_sp_offset"])
    V["F2_UT_SCR01_FAN_HZ"] = drivers["fan_hz"].to_numpy().copy()
    V["F2_UT_SCR01_DP"] = true["SCR.dp_mmAq"].to_numpy() + 2.0 + rng.normal(0, 2.0, n)
    V["F2_UT_SCR01_CIRC_FLOW"] = np.where(
        drivers["water_m3h"].to_numpy() > 0,
        drivers["water_m3h"].to_numpy() + rng.normal(0, 0.4, n), 0.0)
    for key in ("pump_maint", "packing_clean"):
        a, b = EV[key]
        sh.event("process", "pump_off", ["F2_UT_SCR01_CIRC_FLOW"], a, b,
                 "순환펌프 정지 — 제거효율이 0 이 되어 NOx 가 실제로 오른다 (이상치 아님)")

    # 방해 컬럼: pH, 레벨, 펌프 A/B 교대, 죽은 예비 입력
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    V["F2_UT_SCR01_PH"] = np.clip(7.8 + 0.25 * np.sin(2 * np.pi * hour / 24.0)
                                  + _ar1(n, 288 * 4, 0.2, rng), 6.5, 9.0)
    V["F2_UT_SCR01_LEVEL"] = np.clip(62 + 6 * np.sin(2 * np.pi * hour / 8.0)
                                     + _ar1(n, 60, 3.0, rng), 20, 95)
    week = ((idx - idx[0]) / pd.Timedelta(days=7)).astype(int)
    V["F2_UT_SCR01_PUMP_A_RUN"] = (np.asarray(week) % 2 == 0).astype(float)
    V["F2_UT_SCR01_SPARE_AI"] = np.full(n, 4.0)

    # ---- 지류 유량계 ----------------------------------------------------------
    for g in GROUPS:
        V[f"F2_UT_SCR01_BR_{g}_FLOW"] = (true[f"SRC_{g}.mdot"].to_numpy()
                                         + rng.normal(0, 0.012, n))
    V["F2_UT_SCR01_HDR_FLOW"] = (true["HDR.mdot_total"].to_numpy() * VAR["hdr_gain"]
                                 + rng.normal(0, 0.02, n))
    sh.event("sensor", "gain", ["F2_UT_SCR01_HDR_FLOW"], idx[0], idx[-1],
             f"헤더 유량계 게인 {(VAR['hdr_gain'] - 1) * 100:+.1f}%",
             magnitude=VAR["hdr_gain"] - 1)
    tag = "F2_UT_SCR01_BR_IMP_FLOW"
    dead = sh.mask(EV["imp_meter_dead"])
    sh.mark(tag, dead, "status", "Bad")
    sh.event("sensor", "dead", [tag], EV["imp_meter_dead"], idx[-1], "IMP 지류 유량계 고장 — 'Bad'")

    # ---- MES / 기상 ----------------------------------------------------------
    for g in GROUPS:
        s = pd.Series(drivers[f"util_{g}"].to_numpy() * 100.0, index=idx)
        # MES 는 한 시간에 한 번 집계한다. 같은 시간 안의 5분 행은 전부 같은 값이다.
        V[f"F2_FAB_{g}_UTIL"] = s.resample("1h").mean().reindex(idx, method="ffill").to_numpy()
    sh.event("input", "hourly", [f"F2_FAB_{g}_UTIL" for g in GROUPS], idx[0], idx[-1],
             "MES 가동율은 1시간 평균값을 5분 행에 채운 것 (계단형)")
    V["F2_FAB_DRY_EQP_CNT"] = drivers["n_tools_DRY"].to_numpy().copy()
    V["F2_UT_AMB_TEMP"] = drivers["T_amb"].to_numpy() + rng.normal(0, 0.2, n)
    V["F2_UT_AMB_HUMID"] = drivers["RH_amb"].to_numpy() + rng.normal(0, 1.0, n)
    na_a, na_b = EV["mes_na"]
    for g in GROUPS:
        sh.mark(f"F2_FAB_{g}_UTIL", sh.mask(na_a, na_b), "status", "#N/A")
    sh.event("file", "status", [f"F2_FAB_{g}_UTIL" for g in GROUPS], na_a, na_b,
             "엑셀 VLOOKUP 으로 MES 를 붙이다 생긴 '#N/A'")

    # ---- 통신 두절과 산발적 결함 -----------------------------------------------
    ca, cb = EV["comm_fail"]
    cm = sh.mask(ca, cb)
    for t in TAGS:
        sh.mark(t, cm, "status", "Comm Fail")
    sh.event("file", "status", TAGS, ca, cb, "DCS 통신 두절 — 전 태그 'Comm Fail'")
    n_io = 0
    for t in TAGS:
        free = np.where(sh.label[t] == "")[0]
        pick = rng.choice(free, int(0.0005 * n), replace=False)
        sh.mark(t, np.isin(np.arange(n), pick), "status", "I/O Timeout")
        n_io += len(pick)
        free = np.where(sh.label[t] == "")[0]
        blank = rng.choice(free, int(0.002 * n), replace=False)
        sh.mark(t, np.isin(np.arange(n), blank), "blank")
    sh.event("file", "status", TAGS, idx[0], idx[-1], f"산발적 'I/O Timeout' {n_io}셀", count=n_io)
    sh.event("file", "blank", TAGS, idx[0], idx[-1], "산발적 빈 셀 (태그당 0.2%)")

    for t in TAGS:
        d = DECIMALS.get(t, 3 if "_BR_" in t or "HDR" in t else 1)
        V[t] = np.round(V[t], d)
    return sh


# --------------------------------------------------------------------------
# 3. 파일 (월별 추출본을 사람이 이어 붙인 결과물)
# --------------------------------------------------------------------------

def _ko_ampm(t: pd.Timestamp) -> str:
    """한국어 엑셀이 CSV 로 저장할 때 쓰는 날짜 표기."""
    h12 = t.hour % 12 or 12
    ampm = "오전" if t.hour < 12 else "오후"
    return f"{t.year}-{t.month:02d}-{t.day:02d} {ampm} {h12}:{t.minute:02d}"


def _fmt(v: float, decimals: int, excel: bool, comma: bool) -> str:
    if not np.isfinite(v):
        return ""
    if comma:
        return f"{v:,.{decimals}f}"
    s = f"{v:.{decimals}f}"
    if excel and "." in s:                      # 엑셀은 끝의 0 을 지운다
        s = s.rstrip("0").rstrip(".")
    return s


def write_file(sh: Sheet, path: Path, rng) -> dict:
    idx = sh.idx
    header = [["일시"] + TAGS, [""] + [c[1] for c in COLUMNS], [""] + [c[2] for c in COLUMNS]]

    def rows_for(a: str, b: str, excel: bool) -> list[list[str]]:
        m = np.where(sh.mask(a, b))[0]
        out = []
        for i in m:
            t = idx[i]
            row = [_ko_ampm(t) if excel else t.strftime("%Y-%m-%d %H:%M:%S")]
            for tag in TAGS:
                st = sh.status[tag][i]
                if st is not None:
                    row.append(st)
                elif sh.label[tag][i] == "blank":
                    row.append("")
                else:
                    d = DECIMALS.get(tag, 3 if "_BR_" in tag or "HDR" in tag else 1)
                    row.append(_fmt(sh.val[tag][i], d, excel,
                                    comma=excel and tag == "F2_UT_STK01_FLOW"))
            out.append(row)
        return out

    chunks = {
        "01": rows_for("2025-01-01", "2025-02-01", excel=True),
        "02": rows_for("2025-02-01", "2025-03-01", excel=True),
        # 3월 추출본이 4/1 06:00 까지 딸려 나와 4월 추출본과 72행 겹친다
        "03": rows_for("2025-03-01", "2025-04-01 06:00", excel=True),
        "04": rows_for("2025-04-01", "2025-05-01", excel=False),
        "05": rows_for("2025-05-01", "2025-06-01", excel=False),
        "06": rows_for("2025-06-01", "2025-07-01", excel=False),
        "07": rows_for("2025-07-01", "2025-08-01", excel=False),
        "08": rows_for("2025-08-01", "2025-09-01", excel=False),
    }
    # 서버 장애로 하루가 통째로 없다 + 산발적으로 빠진 행
    sa, sb = EV["server_down"]
    day = f"{_t(sa).year}-{_t(sa).month:02d}-{_t(sa).day:02d}"
    chunks["03"] = [r for r in chunks["03"] if not r[0].startswith(day)]
    dropped = 0
    for k, rows in chunks.items():
        keep = rng.random(len(rows)) > 0.001
        dropped += int((~keep).sum())
        chunks[k] = [r for r, ok in zip(rows, keep) if ok]

    order = ["01", "02", "03", "04", "05", "06", "08", "07"]      # 7월이 8월 뒤에 붙었다
    lines: list[list[str]] = list(header)
    for k in order:
        if k in ("04", "07"):                                     # 복사하다 딸려온 헤더
            lines.extend(header)
        lines.extend(chunks[k])
    lines = [r + [""] for r in lines]                              # 엑셀이 남긴 끝 콤마
    lines.extend([[""] * (len(TAGS) + 2)] * 2)                     # 빈 줄 두 개

    buf = io.StringIO()
    csv.writer(buf, lineterminator="\r\n").writerows(lines)
    path.write_bytes(buf.getvalue().encode("cp949"))
    return {"rows_written": len(lines), "rows_dropped_random": dropped,
            "overlap_rows": 72, "order": order}


# --------------------------------------------------------------------------

def randomize(seed: int) -> None:
    """시험용 변형: 사건 날짜·교정 시각·결함 크기를 시드로 흔든다.

    증설일(6/10)과 학습/검증 경계는 그대로 둔다 — 비교 설계는 같고 결함만 다르다.
    """
    r = np.random.default_rng(seed + 7919)
    windows = {                       # 사건별 허용 구간 (날짜를 옮겨도 이야기가 성립하게)
        "comm_fail": ("2025-01-10", "2025-03-20"), "server_down": ("2025-02-01", "2025-03-25"),
        "nox_maint": ("2025-02-10", "2025-04-20"), "recipe_change": ("2025-03-10", "2025-05-10"),
        "pump_maint": ("2025-03-20", "2025-04-30"), "mes_na": ("2025-04-15", "2025-05-25"),
        "flow_frozen": ("2025-04-20", "2025-05-25"), "fan_sp_offset": ("2025-04-15", "2025-07-31"),
        "packing_clean": ("2025-06-15", "2025-07-25"), "water_increase": ("2025-06-20", "2025-07-20"),
        "analyzer_swap": ("2025-07-01", "2025-08-15"),
    }
    for key, (lo, hi) in windows.items():
        v = EV[key]
        first = v[0] if isinstance(v, tuple) else v
        span = (_t(hi) - _t(lo)).days
        new0 = _t(lo) + pd.Timedelta(days=int(r.integers(0, span))) + (
            _t(first) - _t(first).normalize())
        shift = new0 - _t(first)
        EV[key] = (tuple(str(_t(x) + shift) for x in v) if isinstance(v, tuple)
                   else str(_t(v) + shift))
    VAR["cal_hour"] = float(r.choice([1.0, 1.5, 2.0, 4.0, 4.5, 5.0]))
    VAR["analyzer_offset"] = float(r.choice([-1, 1]) * r.uniform(2.0, 4.5))
    VAR["fan_sp_offset"] = float(r.choice([-1, 1]) * r.uniform(8.0, 16.0))
    VAR["flow_gain"] = float(1.0 + r.uniform(-0.03, 0.03))
    VAR["hdr_gain"] = float(1.0 + r.choice([-1, 1]) * r.uniform(0.02, 0.05))
    VAR["temp2_offset"] = float(r.choice([-1, 1]) * r.uniform(0.3, 0.8))


def main() -> None:
    ap = argparse.ArgumentParser(description="현장형 비이상 데이터 생성")
    ap.add_argument("--seed", type=int, default=20250101)
    ap.add_argument("--out", default=str(HERE / "data"))
    ap.add_argument("--reuse", action="store_true", help="저장된 물리 시뮬레이션을 재사용")
    ap.add_argument("--holdout", action="store_true",
                    help="사건 날짜·교정 시각·결함 크기까지 시드로 흔든 시험용 변형")
    args = ap.parse_args()
    if args.holdout:
        randomize(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # 정답지·캐시는 '_' 폴더에 둔다. 앱의 데이터 목록에 뜨지 않는다 (현장 사용자는 정답이 없다).
    ans = out / "_answer"
    ans.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    idx = pd.date_range(START, END, freq="5min")
    drivers, hidden = process_truth(idx, rng)

    cache = ans / "sim_cache.csv"
    if args.reuse and cache.exists():
        true = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"시뮬레이션 재사용: {cache}")
    else:
        true = simulate_truth(drivers, hidden)
        true.to_csv(cache)

    sh = measure(drivers, true, rng)
    info = write_file(sh, out / "field_raw.csv", rng)

    truth = pd.concat([drivers, hidden, true.add_prefix("true:")], axis=1)
    for t in TAGS:
        truth[f"label:{t}"] = sh.label[t]
        truth[f"meas:{t}"] = sh.val[t]
    truth.index.name = "timestamp"
    truth.to_csv(ans / "field_truth.csv")
    sh.event("file", "format", ["*"], idx[0], idx[-1],
             "CP949, 3행 헤더(태그/설명/단위), 1~3월은 '오전/오후' 표기와 천 단위 콤마, "
             "4월·7월 앞에 중간 헤더, 3월/4월 72행 중복, 8월이 7월보다 앞, "
             "3/9 하루 누락, 끝 콤마와 빈 줄", **info)
    (ans / "anomalies.json").write_text(
        json.dumps({"events": sh.events, "EV": EV, "fan_hz": FAN_HZ, "var": VAR,
                    "true_const": TRUE_CONST, "seed": args.seed, "holdout": args.holdout},
                   ensure_ascii=False, indent=1),
        encoding="utf-8")

    c = true["STK.C_dry"]
    print(f"저장: {out/'field_raw.csv'} ({info['rows_written']:,}줄, CP949)")
    print(f"  정답: _answer/field_truth.csv, _answer/anomalies.json ({len(sh.events)}건)")
    print(f"  진짜 NOx: 1~5월 {c[:'2025-05-31'].min():.1f}~{c[:'2025-05-31'].max():.1f}, "
          f"6/10~8월 {c['2025-06-10':].min():.1f}~{c['2025-06-10':].max():.1f} mg/Sm3")


if __name__ == "__main__":
    main()
