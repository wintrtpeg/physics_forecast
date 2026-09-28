"""단위/차원 시스템.

목적은 두 가지다.
1. 파라미터를 현장에서 쓰는 단위(CMM, mmAq, mg/Nm3 ...)로 선언하고 내부는 전부 SI로 돌린다.
2. 방정식의 차원 동차성을 검사해서 "kg/s 에 Pa 를 더하는" 류의 실수를 풀기 전에 잡는다.

외부 의존성(pint 등) 없이 동작한다. 사내 오프라인 PC를 가정.
"""

from __future__ import annotations

import re
from typing import Iterable

# (m, kg, s, K, mol, A, cd)
BASE_SYMBOLS = ("m", "kg", "s", "K", "mol", "A", "cd")
Dim = tuple  # tuple[int, ...] 길이 7

DIMLESS: Dim = (0, 0, 0, 0, 0, 0, 0)


def dim_mul(a: Dim, b: Dim) -> Dim:
    return tuple(x + y for x, y in zip(a, b))


def dim_div(a: Dim, b: Dim) -> Dim:
    return tuple(x - y for x, y in zip(a, b))


def dim_pow(a: Dim, n: float) -> Dim:
    out = []
    for x in a:
        v = x * n
        if abs(v - round(v)) > 1e-9:
            raise DimensionError(f"차원의 분수 거듭제곱은 지원하지 않습니다: {dim_str(a)}^{n}")
        out.append(int(round(v)))
    return tuple(out)


def dim_str(d: Dim) -> str:
    if d == DIMLESS:
        return "1"
    num, den = [], []
    for sym, e in zip(BASE_SYMBOLS, d):
        if e == 0:
            continue
        (num if e > 0 else den).append(sym if abs(e) == 1 else f"{sym}^{abs(e)}")
    s = "*".join(num) if num else "1"
    if den:
        s += "/" + "*".join(den)
    return s


class DimensionError(ValueError):
    """차원이 맞지 않을 때."""


def _d(**kw) -> Dim:
    return tuple(kw.get(s, 0) for s in BASE_SYMBOLS)


L = _d(m=1)
M = _d(kg=1)
T = _d(s=1)
TEMP = _d(K=1)
N = _d(mol=1)

AREA = _d(m=2)
VOLUME = _d(m=3)
VELOCITY = _d(m=1, s=-1)
ACCEL = _d(m=1, s=-2)
FORCE = _d(kg=1, m=1, s=-2)
PRESSURE = _d(kg=1, m=-1, s=-2)
ENERGY = _d(kg=1, m=2, s=-2)
POWER = _d(kg=1, m=2, s=-3)
MASS_FLOW = _d(kg=1, s=-1)
VOL_FLOW = _d(m=3, s=-1)
DENSITY = _d(kg=1, m=-3)
SPEC_ENTHALPY = _d(m=2, s=-2)
SPEC_HEAT = _d(m=2, s=-2, K=-1)
CONCENTRATION = _d(kg=1, m=-3)
TEMP_DIFF = TEMP

#: 심볼 -> (SI 환산 계수, 차원)
_UNITS: dict[str, tuple[float, Dim]] = {
    # 기본
    "m": (1.0, L), "kg": (1.0, M), "s": (1.0, T), "K": (1.0, TEMP),
    "mol": (1.0, N), "A": (1.0, _d(A=1)), "cd": (1.0, _d(cd=1)),
    # 무차원
    "1": (1.0, DIMLESS), "-": (1.0, DIMLESS), "frac": (1.0, DIMLESS),
    "%": (0.01, DIMLESS), "ppm": (1e-6, DIMLESS), "ppb": (1e-9, DIMLESS),
    "rad": (1.0, DIMLESS),
    # 질량
    "g": (1e-3, M), "mg": (1e-6, M), "ug": (1e-9, M), "ton": (1e3, M),
    # 길이
    "km": (1e3, L), "cm": (1e-2, L), "mm": (1e-3, L), "um": (1e-6, L),
    "inch": (0.0254, L), "ft": (0.3048, L),
    # 시간
    "min": (60.0, T), "h": (3600.0, T), "hr": (3600.0, T),
    "day": (86400.0, T), "yr": (31_536_000.0, T),
    # 부피 (Nm3 = 표준상태 기준 부피. 차원은 m3 와 같고, 표준상태 환산은 모델이 담당한다)
    "L": (1e-3, VOLUME), "mL": (1e-6, VOLUME), "m3": (1.0, VOLUME), "Nm3": (1.0, VOLUME), "Sm3": (1.0, VOLUME),
    "CMM": (1.0 / 60.0, VOL_FLOW), "CMH": (1.0 / 3600.0, VOL_FLOW), "LPM": (1e-3 / 60.0, VOL_FLOW),
    # 힘/압력 (mmAq: 현장 표준 표기. 4degC 물기둥 기준)
    "N": (1.0, FORCE), "Pa": (1.0, PRESSURE), "kPa": (1e3, PRESSURE),
    "MPa": (1e6, PRESSURE), "bar": (1e5, PRESSURE), "mbar": (100.0, PRESSURE),
    "atm": (101325.0, PRESSURE), "mmAq": (9.80665, PRESSURE), "mmH2O": (9.80665, PRESSURE),
    "mmHg": (133.322, PRESSURE), "psi": (6894.757, PRESSURE), "inH2O": (249.0889, PRESSURE),
    # 에너지/동력
    "J": (1.0, ENERGY), "kJ": (1e3, ENERGY), "MJ": (1e6, ENERGY),
    "kWh": (3.6e6, ENERGY), "cal": (4.184, ENERGY), "kcal": (4184.0, ENERGY),
    "W": (1.0, POWER), "kW": (1e3, POWER), "MW": (1e6, POWER), "RT": (3516.85, POWER),
    # 회전수
    "rpm": (1.0 / 60.0, _d(s=-1)), "Hz": (1.0, _d(s=-1)),
    # 전기 / 기타 현장 표기
    "V": (1.0, _d(m=2, kg=1, s=-3, A=-1)), "kV": (1e3, _d(m=2, kg=1, s=-3, A=-1)),
    "mA": (1e-3, _d(A=1)), "kgf": (9.80665, FORCE),
    "kgfcm2": (98066.5, PRESSURE), "MPag": (1e6, PRESSURE), "kPag": (1e3, PRESSURE),
}

_TOKEN = re.compile(r"\s*([A-Za-z_%][A-Za-z_0-9]*|1|-)\s*(?:\^?\s*(-?\d+))?\s*")


def parse_unit(expr: str) -> tuple[float, Dim]:
    """'kg/(m*s^2)', 'mg/Nm3', 'CMM' 같은 문자열을 (SI 계수, 차원)으로 변환."""
    expr = (expr or "1").strip()
    if not expr:
        return 1.0, DIMLESS
    factor = 1.0
    dim = DIMLESS
    # 괄호는 단순 치환으로 처리 (중첩 괄호는 쓰지 않는다는 전제)
    sign = 1
    i = 0
    depth_sign = 1
    while i < len(expr):
        ch = expr[i]
        if ch == "*":
            sign = depth_sign
            i += 1
            continue
        if ch == "/":
            sign = -depth_sign
            i += 1
            continue
        if ch == "(":
            depth_sign = sign
            i += 1
            continue
        if ch == ")":
            depth_sign = 1
            sign = 1
            i += 1
            continue
        if ch.isspace():
            i += 1
            continue
        m = _TOKEN.match(expr, i)
        if not m:
            raise ValueError(f"단위 문자열을 해석할 수 없습니다: {expr!r} (위치 {i})")
        sym, exp_txt = m.group(1), m.group(2)
        exp = int(exp_txt) if exp_txt else 1
        if sym in _UNITS:
            f, d = _UNITS[sym]
        else:
            # 'm4', 's2', 'kg2' 처럼 지수를 붙여 쓴 현장 표기를 허용한다
            m2 = re.fullmatch(r"([A-Za-z_%]+)(\d+)", sym)
            if m2 and m2.group(1) in _UNITS:
                bf, bd = _UNITS[m2.group(1)]
                n = int(m2.group(2))
                f, d = bf**n, dim_pow(bd, n)
            else:
                raise ValueError(f"알 수 없는 단위 심볼: {sym!r} (전체: {expr!r})")
        e = exp * sign
        factor *= f**e
        dim = dim_mul(dim, dim_pow(d, e))
        i = m.end()
        sign = depth_sign
    return factor, dim


def to_si(value: float, unit: str) -> float:
    """현장 단위 값을 SI 값으로. degC 는 오프셋이 있어 따로 처리한다."""
    u = (unit or "").strip()
    if u in ("degC", "C", "oC", "℃"):
        return value + 273.15
    if u in ("degF", "F"):
        return (value - 32.0) * 5.0 / 9.0 + 273.15
    f, _ = parse_unit(u)
    return value * f


def from_si(value: float, unit: str) -> float:
    """SI 값을 현장 단위로 되돌린다."""
    u = (unit or "").strip()
    if u in ("degC", "C", "oC", "℃"):
        return value - 273.15
    if u in ("degF", "F"):
        return (value - 273.15) * 9.0 / 5.0 + 32.0
    f, _ = parse_unit(u)
    return value / f


def dim_of(unit: str) -> Dim:
    u = (unit or "").strip()
    if u in ("degC", "C", "oC", "℃", "degF", "F"):
        return TEMP
    return parse_unit(u)[1]


#: 현장 파일에 적히는 단위 표기 -> 이 도구의 표기. 한글 엑셀·DCS 가 쓰는 전각 기호를 포함한다.
_CHAR_MAP = {
    "㎥": "m3", "㎡": "m2", "㎜": "mm", "㎝": "cm", "㎞": "km", "㎎": "mg", "㎏": "kg",
    "㎍": "ug", "μ": "u", "µ": "u", "㎪": "kPa", "㎫": "MPa", "㎩": "Pa", "㎾": "kW",
    "㎿": "MW", "㎐": "Hz", "ℓ": "L", "㎖": "mL", "㎃": "mA", "㎸": "kV", "³": "3", "²": "2", "₂": "2", "₃": "3",
    "·": "*", "×": "*", "／": "/", "％": "%", "（": "(", "）": ")", "°": "deg", "º": "deg",
}
_WORD_MAP = {
    "℃": "degC", "degc": "degC", "deg.c": "degC", "degC": "degC", "C": "degC", "℉": "degF",
    "degF": "degF", "%RH": "%", "RH%": "%", "%rh": "%", "RH": "%", "대": "1", "EA": "1",
    "ea": "1", "개": "1", "대수": "1", "count": "1", "cnt": "1", "pH": "1", "PH": "1",
    "mmWC": "mmAq", "mmwc": "mmAq", "mmaq": "mmAq", "mmAQ": "mmAq", "mmH2O": "mmAq",
    "kg/cm2": "kgfcm2", "kgf/cm2": "kgfcm2", "kg/cm2g": "kgfcm2", "kgf/cm2g": "kgfcm2",
    "kg/cm2G": "kgfcm2", "ppmv": "ppm", "t/h": "ton/h", "T/H": "ton/h", "RPM": "rpm",
    "HZ": "Hz", "KW": "kW", "Kw": "kW", "KPA": "kPa", "kpa": "kPa", "MPA": "MPa",
    "USRT": "RT", "rt": "RT", "Nm3/hr": "Nm3/h", "Sm3/hr": "Sm3/h", "m3/hr": "m3/h",
    "CMH": "CMH", "CMM": "CMM", "cmh": "CMH", "cmm": "CMM", "LPM": "LPM", "lpm": "LPM",
    "-": "1", "N/A": "", "n/a": "", "": "",
}


def normalize_unit(raw: str | None) -> str | None:
    """파일에 적힌 단위 표기를 이 도구가 아는 단위 문자열로 바꾼다.

    ``'mg/Sm³'`` -> ``'mg/Sm3'``, ``'㎥/h'`` -> ``'m3/h'``, ``'℃'`` -> ``'degC'``,
    ``'대'`` -> ``'1'``. 해석할 수 없으면 ``None`` (단위가 아닌 글자일 가능성이 크다).
    빈 칸은 ``None`` 이다 — '무차원'과 '모름'은 다르다.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if s.startswith(("[", "(")) and s.endswith(("]", ")")):
        s = s[1:-1].strip()
    if not s:
        return None
    if s in _WORD_MAP:
        return _WORD_MAP[s] or None
    for a, b in _CHAR_MAP.items():
        s = s.replace(a, b)
    s = s.replace(" ", "")
    if s in ("degC", "degF"):
        return s
    if s in _WORD_MAP:
        return _WORD_MAP[s] or None
    if s.lower().endswith("/hr"):
        s = s[:-3] + "/h"
    try:
        parse_unit(s)
        return s
    except ValueError:
        pass
    # 대소문자만 다른 표기 (KW, kpa ...)
    lower = {k.lower(): k for k in _UNITS}
    parts = re.split(r"([*/()])", s)
    fixed = []
    for p in parts:
        if p in "*/()" or not p:
            fixed.append(p)
            continue
        m = re.fullmatch(r"([A-Za-z_%]+)(\d*)", p)
        if m and m.group(1) not in _UNITS and m.group(1).lower() in lower:
            p = lower[m.group(1).lower()] + m.group(2)
        fixed.append(p)
    cand = "".join(fixed)
    try:
        parse_unit(cand)
        return cand
    except ValueError:
        return None


def register_unit(symbol: str, si_factor: float, dim: Dim) -> None:
    """사내 관용 단위를 추가 등록할 때 사용."""
    _UNITS[symbol] = (si_factor, dim)


def known_units() -> Iterable[str]:
    return sorted(_UNITS)
