"""태그 이름에서 단위를 추론한다.

범용화의 첫 관문이다. 새 계통의 CSV 를 받으면 컬럼이 수십~수백 개인데, 단위를 전부
손으로 적게 하면 아무도 안 쓴다. 그런데 단위가 없으면 차원 해석도 보존식 탐지도
못 한다 (그 둘이 이 도구의 물리적 근거다).

그래서 **이름 규칙 + 값 범위**로 후보를 제시하고 사람이 확인하는 방식을 쓴다.
자동으로 정하지 않는다 — 단위를 잘못 잡으면 모델이 조용히 틀린 답을 낸다.

이름 규칙은 국내 팹 태그 관례를 따른다 (``F2_UT_SCR01_FAN_SP`` 처럼 계통·호기·
측정항목이 언더바로 이어지는 형태).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

#: (토큰 정규식, 단위, 신뢰도, 설명). 위에서부터 먼저 맞는 것을 쓴다.
_RULES: list[tuple[str, str, float, str]] = [
    (r"(NOX|SOX|NO2|SO2|HCL|NH3|VOC|THC|TOC|DUST|PM10|PM25)", "mg/Nm3", 0.75, "오염물질 농도"),
    (r"(^|_)(O2|CO2|CO)(_|$)", "%", 0.6, "가스 농도"),
    (r"(UTIL|UTILIZ|RATIO|RATE_?PCT|PERCENT|_PCT|_OP(_|$)|OPEN|VALVE|LOAD)", "%", 0.7, "비율/개도"),
    (r"(RH|HUMID|HUM(_|$))", "%", 0.7, "상대습도"),
    (r"(^|_)PH(_|$)", "1", 0.8, "pH (무차원)"),
    (r"(TEMP|TMP|_T(_|$)|_TE(_|$))", "degC", 0.85, "온도"),
    (r"(_DP(_|$)|DIFF_?PRESS|DELTAP)", "mmAq", 0.6, "차압"),
    (r"(_SP(_|$)|STATIC_?PRESS)", "mmAq", 0.6, "정압"),
    (r"(PRESS|PRS|_P(_|$)|BAR(_|$))", "kPa", 0.5, "압력"),
    (r"(FREQ|_HZ(_|$))", "Hz", 0.85, "인버터 주파수"),
    (r"(RPM|SPEED)", "rpm", 0.7, "회전수"),
    (r"(KWH|ENERGY|ELEC_?USE)", "kWh", 0.8, "전력량"),
    (r"(POWER|PWR|_KW(_|$)|MOTOR)", "kW", 0.75, "전력"),
    (r"(_A(_|$)|CURRENT|AMP)", "A", 0.5, "전류"),
    (r"(VOLT|_V(_|$))", "V", 0.5, "전압"),
    (r"(LEVEL|LVL)", "%", 0.5, "레벨"),
    (r"(MDOT|MASS_?FLOW)", "kg/s", 0.75, "질량유량"),
    (r"(FLOW|FLW|_Q(_|$)|CMH|CMM|NM3)", "", 0.0, "유량 (값 범위로 판단)"),
]

#: 유량은 이름만으로 단위를 못 정한다. 값의 크기로 고른다.
_FLOW_BY_MAGNITUDE: list[tuple[float, float, str, str]] = [
    (0.0, 50.0, "kg/s", "질량유량 또는 CMM"),
    (50.0, 3000.0, "CMM", "체적유량 (분당)"),
    (3000.0, 1e9, "Nm3/h", "체적유량 (시간당, 표준상태)"),
]

_PRESSURE_BY_MAGNITUDE: list[tuple[float, float, str, str]] = [
    (0.0, 30.0, "kPa", "게이지 압력"),
    (30.0, 3000.0, "mmAq", "수주 압력"),
    (3000.0, 5e4, "Pa", "파스칼"),
    (5e4, 5e5, "Pa", "절대압 (대기압 근처)"),
]


@dataclass
class UnitGuess:
    column: str
    unit: str
    confidence: float          # 0~1
    reason: str
    alternatives: list[str]

    @property
    def needs_review(self) -> bool:
        return self.confidence < 0.7 or not self.unit


def _magnitude_pick(values: np.ndarray, table) -> tuple[str, str]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return "", "값이 없음"
    scale = float(np.nanmedian(np.abs(finite)))
    for lo, hi, unit, why in table:
        if lo <= scale < hi:
            return unit, f"{why} (대푯값 {scale:.4g})"
    return "", f"대푯값 {scale:.4g} 로는 판단 불가"


def guess_unit(column: str, values: np.ndarray | None = None) -> UnitGuess:
    """컬럼 하나의 단위를 추론한다."""
    name = column.upper()
    vals = np.asarray(values, dtype=float) if values is not None else np.array([])

    for pattern, unit, conf, why in _RULES:
        if not re.search(pattern, name):
            continue
        if unit == "":                                   # 유량: 값 범위로 결정
            u, reason = _magnitude_pick(vals, _FLOW_BY_MAGNITUDE)
            alts = [t[2] for t in _FLOW_BY_MAGNITUDE if t[2] != u]
            return UnitGuess(column, u, 0.55 if u else 0.0,
                             f"이름에 유량 표기 — {reason}", alts)
        if unit in ("kPa", "mmAq") and len(vals):        # 압력: 값 범위로 보정
            u, reason = _magnitude_pick(vals, _PRESSURE_BY_MAGNITUDE)
            if u:
                return UnitGuess(column, u, 0.6, f"{why} — {reason}",
                                 [t[2] for t in _PRESSURE_BY_MAGNITUDE if t[2] != u])
        if unit == "%" and len(vals):                    # 비율: 0~1 인지 0~100 인지
            finite = vals[np.isfinite(vals)]
            if len(finite) and float(np.nanmax(finite)) <= 1.5:
                return UnitGuess(column, "1", 0.6, f"{why} — 값이 0~1 범위",
                                 ["%"])
        return UnitGuess(column, unit, conf, why, [])
    return UnitGuess(column, "", 0.0, "이름에서 단서를 찾지 못함", [])


def guess_units(df, columns: list[str] | None = None) -> dict[str, UnitGuess]:
    """DataFrame 전체에 대해 추론. 값 범위까지 같이 본다."""
    cols = columns if columns is not None else list(df.columns)
    out = {}
    for c in cols:
        try:
            vals = df[c].to_numpy(dtype=float)
        except (KeyError, TypeError, ValueError):
            vals = np.array([])
        out[c] = guess_unit(c, vals)
    return out


def summary(guesses: dict[str, UnitGuess]) -> dict[str, int]:
    return {
        "확실": sum(1 for g in guesses.values() if g.confidence >= 0.7),
        "확인필요": sum(1 for g in guesses.values() if 0 < g.confidence < 0.7),
        "모름": sum(1 for g in guesses.values() if not g.unit),
    }
