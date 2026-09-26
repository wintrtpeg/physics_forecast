"""배기 스크러버 -> 옥상 굴뚝 NOx 농도 모델.

계통 구성
---------
    장비군 4개 ──┐ (지류 덕트)
                 ├─► 헤더(Mixer) ─► 메인 덕트 ─► 습식 스크러버 ─► 유인송풍기 ─► 굴뚝
               ──┘

모든 유량은 **압력 구동**이다. 룸 압력과 굴뚝 대기압 사이의 압력장을 팬이 만들고,
각 장비군의 후드 저항과 덕트 저항이 그 압력을 나눠 갖는다. 그래서 장비가 늘거나
가동율이 오르면 유량이 늘긴 하지만 팬 성능곡선을 타고 내려가며 포화된다.

주 구동변수
-----------
* ``SRC_*.util``   : 장비군별 가동율        <- 공장 가동율 시나리오
* ``SRC_*.n_tools``: 설치 대수              <- 증설 시나리오
* ``STK.T_amb``    : 외기 온도              <- 기상 시나리오
* ``FAN.n_ratio``  : 송풍기 인버터 회전수비  <- 운전 시나리오
* ``SCR.L``        : 스크러버 순환수량       <- 운전 시나리오

예측 대상
---------
* ``STK.C_dry`` : 굴뚝 NOx 농도 [mg/Sm3, 건조 기준]  <- TMS 비교 대상
* ``STK.ppm_dry``: 동일 값의 ppm 표기
"""

from __future__ import annotations

from pforecast import System
from pforecast.lib import Duct, Fan, Mixer, Stack, ToolGroupSource, WetScrubber

#: 장비군 설계 제원
#:   name, 대수, 대당풍량[Nm3/h], 배기온도[degC], 상대습도, 대기NOx[mg/s], 가동NOx[mg/s], 지류덕트K[1/m4]
TOOL_GROUPS = [
    ("DRY", 24, 220.0, 52.0, 0.40, 0.55, 7.00, 120.0),   # 건식식각 - NOx 주 발생원
    ("CVD", 18, 260.0, 61.0, 0.35, 0.85, 5.00, 150.0),   # 증착 - 퍼지 기저 발생량이 큼
    ("WET", 30, 160.0, 38.0, 0.85, 0.10, 0.80, 150.0),   # 습식세정 - NOx 적고 습분 많음
    ("IMP", 8, 180.0, 47.0, 0.40, 0.30, 3.40, 1600.0),   # 이온주입 - 지류가 가늘다
]

DESIGN_UTIL = 0.80


def build(n_groups: int = 4, util: float = DESIGN_UTIL) -> System:
    """NOx 계통 시스템을 조립해 반환한다."""
    sysm = System("nox_stack")
    groups = TOOL_GROUPS[:n_groups]

    sysm.add(Mixer("HDR", n_inlets=len(groups)))

    for i, (tag, n, q, t_exh, rh, ef_i, ef_p, k_duct) in enumerate(groups):
        sysm.add_all(
            ToolGroupSource(
                f"SRC_{tag}", n_tools=n, util=util, q_tool=q, dp_design=150.0,
                idle_frac=0.35, ef_idle=ef_i, ef_process=ef_p, T_exh=t_exh, rh_exh=rh,
            ),
            Duct(f"DCT_{tag}", K=k_duct, UA=250.0, T_amb=25.0),
        )
        sysm.connect(f"SRC_{tag}.outlet", f"DCT_{tag}.a")
        sysm.connect(f"DCT_{tag}.b", f"HDR.in{i + 1}")

    sysm.add_all(
        Duct("DCT_MAIN", K=17.0, UA=1800.0, T_amb=25.0),
        WetScrubber("SCR", eta_max=0.55, ntu_a=14.0, ntu_b=0.55, L=0.55, K=51.0,
                    rh_out=0.97, T_water=22.0),
        Fan("FAN", c0=4900.0, c1=0.0, c2=-68.0, n_ratio=1.0, eta=0.70),
        Stack("STK", H=25.0, K=8.4, T_amb=20.0, o2_corr=0.0),
    )
    sysm.connect("HDR.out", "DCT_MAIN.a")
    sysm.connect("DCT_MAIN.b", "SCR.a")
    sysm.connect("SCR.b", "FAN.a")
    sysm.connect("FAN.b", "STK.a")
    return sysm


#: 시나리오/보정에서 자주 건드리는 파라미터의 짧은 별칭
ALIASES = {
    "util": [f"SRC_{g[0]}.util" for g in TOOL_GROUPS],
    "n_tools": [f"SRC_{g[0]}.n_tools" for g in TOOL_GROUPS],
    "T_amb": ["STK.T_amb", "DCT_MAIN.T_amb"] + [f"DCT_{g[0]}.T_amb" for g in TOOL_GROUPS],
    "fan_speed": ["FAN.n_ratio"],
    "scrubber_water": ["SCR.L"],
}

#: 앱 시나리오 화면에 슬라이더로 뜨는 운전 손잡이.
#: 선언하지 않으면 파라미터 이름 끝조각으로 자동 추출하지만, 그러면 대기압·기준밀도
#: 같은 상수까지 손잡이로 나온다. 무엇이 '운전 중 실제로 변하는 값'인지는 사람이 안다.
DRIVERS = [
    {"key": "util",     "label": "가동율",         "unit": "%",      "lo": 20,  "hi": 130},
    {"key": "n_tools",  "label": "장비 대수",       "mode": "scale",  "lo": 0.5, "hi": 2.0},
    {"key": "T_amb",    "label": "외기온도",        "unit": "degC",   "lo": -10, "hi": 40},
    {"key": "n_ratio",  "label": "송풍기 회전수비",  "unit": "1",      "lo": 0.7, "hi": 1.15},
    {"key": "L",        "label": "스크러버 순환수",  "unit": "m3/min", "lo": 0.1, "hi": 1.5},
]

#: 관리기준. 시나리오 화면에서 초과 여부를 바로 표시한다.
LIMITS = {
    "STK.C_dry": {"max": 100.0, "label": "사내 관리기준"},
    "SRC_DRY.q_per_tool": {"min": 3.0, "label": "후드 포집 한계"},
}

#: 현장에서 계측되는 값 (캘리브레이션/검증 대상)
OBSERVABLES = {
    "STK.C_dry": "굴뚝 NOx 농도 [mg/Sm3, 건조]",
    "STK.Q_n": "굴뚝 배기 풍량 [Nm3/h]",
    "STK.T_stack": "굴뚝 온도 [degC]",
    "FAN.dp_mmAq": "송풍기 정압 [mmAq]",
    "SCR.dp_mmAq": "스크러버 차압 [mmAq]",
}
