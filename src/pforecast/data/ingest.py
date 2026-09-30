"""현장 CSV 를 있는 그대로 받아들이는 수집기.

``pd.read_csv(path)`` 한 줄은 현장 파일에서 거의 항상 깨진다. 실제로 겪는 것들:

* **인코딩** — 한글 윈도우 엑셀로 저장하면 CP949 다. UTF-8 로 읽으면 첫 줄에서 죽는다.
* **여러 줄 헤더** — 태그 / 설명 / 단위 행이 위에 붙어 있다. 그러면 모든 컬럼이 문자열이
  되어 ``select_dtypes(number)`` 가 **전부 조용히 버린다**.
* **엑셀 날짜 표기** — ``2025-01-01 오후 3:05``. pandas 는 이걸 못 읽고 NaT 로 만든 뒤
  그 행을 버린다. 석 달 치가 경고 없이 사라진다.
* **상태 문자열** — ``Bad``, ``I/O Timeout``, ``Comm Fail``, ``#N/A``. 한 셀만 있어도
  컬럼 전체가 문자열이 된다.
* **천 단위 콤마** — ``"11,234.5"``.
* **이어 붙인 월별 파일** — 중간에 반복된 헤더, 겹친 구간(중복 시각), 뒤바뀐 순서.
* **끝의 빈 컬럼, 빈 줄, 통째로 빠진 시간대.**

이 모듈은 그것들을 **고치고, 고친 내역을 전부 보고한다.** 조용히 고치면 사용자는 자기
데이터에 무슨 일이 있었는지 모른다. 보고서는 사람이 읽을 문장(``lines()``)으로 낸다.

결과는 규칙적인 시간 격자(빠진 시각은 NaN 행)에 놓인 숫자 DataFrame 이다. 격자가
규칙적이어야 지연(lag)·이동창·연속 구간 판정이 '행 수 = 시간'으로 맞는다.
"""

from __future__ import annotations

import csv
import dataclasses
import functools
import glob
import hashlib
import io
import json
import os
import re
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from operator import itemgetter
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..core.units import normalize_unit

#: 시각 컬럼 이름으로 흔히 쓰는 말
_TIME_WORDS = ("일시", "시간", "시각", "날짜", "일자", "time", "date", "timestamp", "datetime",
               "ts", "tstamp")

#: 상태 문자열 중 숫자로 바꿔도 되는 것 (컬럼 대부분이 이 토큰일 때만 적용)
_BOOL_TOKENS = {
    "ON": 1.0, "OFF": 0.0, "RUN": 1.0, "STOP": 0.0, "TRUE": 1.0, "FALSE": 0.0,
    "OPEN": 1.0, "CLOSE": 0.0, "CLOSED": 0.0, "운전": 1.0, "정지": 0.0,
    "열림": 1.0, "닫힘": 0.0, "가동": 1.0,
}

_RE_KO_AMPM = re.compile(
    r"^\s*(\d{4})\s*[-./년]\s*(\d{1,2})\s*[-./월]\s*(\d{1,2})\s*일?\s*"
    r"(오전|오후|AM|PM|am|pm)\s*(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?\s*$")
_RE_AMPM_AFTER = re.compile(
    r"^\s*(\d{4})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{1,2})\s+"
    r"(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?\s*(오전|오후|AM|PM|am|pm)\s*$")
_RE_KO_WORDS = re.compile(
    r"^\s*(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일\s*(\d{1,2})\s*[:시]\s*(\d{1,2})"
    r"\s*분?\s*(?:(\d{1,2})\s*초?)?\s*$")
#: 흔한 기록 주기 [초]
_NICE_INTERVALS = (1, 2, 5, 10, 15, 20, 30, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600,
                   7200, 10800, 21600, 43200, 86400)
_RE_DATE_SHAPE = r"\d{4}\s*[-./]\s*\d{1,2}\s*[-./]\s*\d{1,2}|\d{1,2}\s*[-./]\s*\d{1,2}\s*[-./]\s*\d{4}"
_RE_THOUSANDS = r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
#: 시각 뒤에 붙은 시간대 표기: 2025-01-01T00:05:00+09:00, ...Z, ... +0900, ... UTC
_RE_TZ_SUFFIX = re.compile(r"^(.*\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)\s*(Z|z|UTC|GMT|[+-]\d{2}:?\d{2})$")


def _offset_minutes(tok: str) -> int:
    if tok.upper() in ("Z", "UTC", "GMT"):
        return 0
    sign = -1 if tok[0] == "-" else 1
    d = tok[1:].replace(":", "")
    return sign * (int(d[:2]) * 60 + int(d[2:]))


def _fmt_offset(m: int) -> str:
    return "UTC(Z)" if m == 0 else f"{'+' if m > 0 else '-'}{abs(m) // 60:02d}:{abs(m) % 60:02d}"

_RE_NUM_SUFFIX = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([^\d\s.,+-].*?)\s*$")


# --------------------------------------------------------------------------
# 시각 해석
# --------------------------------------------------------------------------

def _from_parts(parts: pd.DataFrame, ampm_col: int | None, order: tuple[int, ...]) -> pd.Series:
    y, mo, d, h, mi, se = (pd.to_numeric(parts[i], errors="coerce") if i is not None else 0
                           for i in order)
    if ampm_col is not None:
        tok = parts[ampm_col].str.upper()
        pm = tok.isin(["오후", "PM"])
        h = (h % 12) + np.where(pm, 12, 0)
    if isinstance(se, int):
        se = pd.Series(0, index=parts.index)
    frame = pd.DataFrame({"year": y, "month": mo, "day": d, "hour": h, "minute": mi,
                          "second": se.fillna(0)})
    return pd.to_datetime(frame, errors="coerce")


def parse_times(values: pd.Series, counts=None) -> tuple[pd.Series, dict[str, int]]:
    """여러 표기가 섞인 시각 문자열을 해석한다. (결과, 표기별 행 수)를 돌려준다.

    ``counts`` 는 값마다 몇 행에 나왔는지 (서로 다른 값만 넘겨 해석할 때). 표기별 행 수는 이걸로 센다.
    """
    s = values.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    stats: dict[str, int] = {}
    w = pd.Series(1 if counts is None else np.asarray(counts), index=s.index, dtype="int64")

    def n(mask) -> int:
        return int(w[mask[mask].index].sum())
    # 0) 시간대 표기(+09:00, Z)는 떼고 **적힌 시각 그대로** 쓴다. 현장 데이터·생산계획은 현지
    #    시각으로 맞춰야 하므로 시간대를 붙인 채 두면 안 된다 (pandas 는 붙은 채로 섞으면 거부한다).
    #    여러 시간대가 섞였으면(서머타임, Z 와 +09:00 혼용) 가장 많은 시간대 기준으로 옮긴다.
    shift = None
    tz = s.str.extract(_RE_TZ_SUFFIX)
    has_tz = tz[0].notna()
    if has_tz.any():
        offs = tz.loc[has_tz, 1].map(_offset_minutes)
        common = int(offs.mode().iloc[0])
        s = s.copy()
        s.loc[has_tz] = tz.loc[has_tz, 0]
        moved = offs != common
        if moved.any():
            shift = pd.Series(0, index=s.index, dtype="int64")
            shift.loc[moved[moved].index] = (common - offs[moved]).astype("int64")
        kinds = ", ".join(_fmt_offset(int(m)) for m in sorted(offs.unique()))
        stats[f"시간대 표기({kinds}) 뗌 → {_fmt_offset(common)} 기준 시각"] = n(has_tz)
        if moved.any():
            stats[f"다른 시간대 → {_fmt_offset(common)} 로 옮김"] = n(moved)
    empty = s.isna() | (s == "")
    todo = ~empty

    def take(mask, parsed, label):
        parsed = pd.to_datetime(parsed, errors="coerce")
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):     # 방어: 시간대가 남았으면 적힌 시각으로
            parsed = parsed.dt.tz_localize(None)
        # 일반 파서는 '50.4' 같은 숫자도 날짜로 우긴다 (서기 161년). 상식적인 연도만 받는다.
        ok = parsed.notna() & parsed.dt.year.between(1980, 2100)
        if ok.any():
            out.loc[ok[ok].index] = parsed[ok].astype("datetime64[ns]")
            stats[label] = stats.get(label, 0) + n(ok)
        return mask & ~out.notna()

    # 1) 한국어 엑셀: 2025-01-01 오후 3:05
    sub = s[todo]
    if len(sub) and sub.str.contains("오전|오후|AM|PM|am|pm", regex=True).any():
        g = sub.str.extract(_RE_KO_AMPM)
        hit = g[0].notna()
        if hit.any():
            todo = take(todo, _from_parts(g[hit], 3, (0, 1, 2, 4, 5, 6)), "오전/오후 표기")
        sub = s[todo]
        g = sub.str.extract(_RE_AMPM_AFTER)
        hit = g[0].notna()
        if hit.any():
            todo = take(todo, _from_parts(g[hit], 6, (0, 1, 2, 3, 4, 5)), "AM/PM 뒤 표기")
    # 2) 2025년 1월 1일 13시 05분
    sub = s[todo]
    if len(sub) and sub.str.contains("년", regex=False).any():
        g = sub.str.extract(_RE_KO_WORDS)
        hit = g[0].notna()
        if hit.any():
            todo = take(todo, _from_parts(g[hit], None, (0, 1, 2, 3, 4, 5)), "년월일 표기")
    # 3) ISO 계열 (가장 흔하고 가장 빠르다). 날짜 모양인 문자열만 넘긴다.
    datey = s.str.contains(_RE_DATE_SHAPE, regex=True).fillna(False)
    sub = s[todo & datey]
    if len(sub):
        try:
            iso = pd.to_datetime(sub, errors="coerce", format="ISO8601")
        except (TypeError, ValueError):                 # pandas < 2.0
            iso = pd.to_datetime(sub, errors="coerce")
        todo = take(todo, iso, "ISO")
    # 4) 그 밖의 표기 (2025/01/01 0:05, 2025.01.01 13:05 ...)
    sub = s[todo & datey]
    if len(sub):
        try:
            mixed = pd.to_datetime(sub.str.replace(".", "-", regex=False), errors="coerce",
                                   format="mixed")
        except (TypeError, ValueError):
            mixed = pd.to_datetime(sub.str.replace(".", "-", regex=False), errors="coerce")
        todo = take(todo, mixed, "기타 표기")
    # 5) 엑셀 일련번호 (서식 없이 저장된 날짜: 45658.5 = 2025-01-01 12:00)
    sub = s[todo]
    if len(sub):
        num = pd.to_numeric(sub, errors="coerce")
        serial = num.between(20000, 80000)
        if serial.any():
            parsed = pd.to_datetime(num[serial], unit="D", origin="1899-12-30")
            todo = take(todo, parsed.dt.round("s"), "엑셀 일련번호")
    if shift is not None:
        out = out + pd.to_timedelta(shift, unit="min")
    stats["해석 불가"] = n(todo)
    return out, stats


def _time_score(values: list[str]) -> float:
    if not values:
        return 0.0
    parsed, _ = parse_times(pd.Series(values, dtype="string"))
    nonempty = sum(1 for v in values if v.strip())
    return float(parsed.notna().sum()) / max(nonempty, 1)


# --------------------------------------------------------------------------
# 보고서
# --------------------------------------------------------------------------

@dataclass
class IngestReport:
    path: str = ""
    encoding: str = ""
    delimiter: str = ","
    title_rows: int = 0
    header_rows: int = 1
    has_unit_row: bool = False
    has_desc_row: bool = False
    time_column: str | None = None
    time_column_note: str = ""
    time_formats: dict[str, int] = field(default_factory=dict)
    n_lines: int = 0
    n_data_rows: int = 0
    n_repeated_header: int = 0
    n_blank_rows: int = 0
    n_bad_time: int = 0
    bad_time_examples: list[str] = field(default_factory=list)
    n_out_of_order: int = 0
    n_duplicate_rows: int = 0
    n_conflicting_duplicates: int = 0
    n_snapped: int = 0
    n_missing_grid: int = 0
    interval_s: float = float("nan")
    gaps: list[dict] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    text_columns: dict[str, list[str]] = field(default_factory=dict)
    renamed_columns: dict[str, str] = field(default_factory=dict)
    status_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    thousands: dict[str, int] = field(default_factory=dict)
    suffix_units: dict[str, str] = field(default_factory=dict)
    bool_columns: list[str] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)          # 정규화된 단위
    units_raw: dict[str, str] = field(default_factory=dict)      # 파일에 적힌 그대로
    descriptions: dict[str, str] = field(default_factory=dict)
    long: dict = field(default_factory=dict)                     # 긴 형식을 펼쳤으면 그 내역
    chunk_stats: dict = field(default_factory=dict, repr=False)  # 큰 파일 조각을 합칠 때만 쓴다
    t_start: str | None = None
    t_end: str | None = None

    # ---- 요약 --------------------------------------------------------------
    def status_total(self) -> dict[str, int]:
        tot: dict[str, int] = {}
        for counts in self.status_counts.values():
            for k, v in counts.items():
                tot[k] = tot.get(k, 0) + v
        return dict(sorted(tot.items(), key=lambda kv: -kv[1]))

    def lines(self) -> list[str]:
        """사람이 읽을 정리 내역. 고친 것이 없으면 짧다."""
        out = []
        enc = {"cp949": "CP949 (한글 윈도우 엑셀 저장본)", "utf-8": "UTF-8",
               "utf-16": "UTF-16 (엑셀 '유니코드 텍스트')"}.get(self.encoding, self.encoding)
        out.append(f"인코딩 {enc}, 구분자 {'탭' if self.delimiter == chr(9) else repr(self.delimiter)}.")
        if self.header_rows > 1 or self.title_rows:
            kinds = ["태그"] + (["설명"] if self.has_desc_row else []) + (["단위"] if self.has_unit_row else [])
            msg = f"헤더 {self.header_rows}행({' / '.join(kinds)})을 인식했습니다"
            if self.title_rows:
                msg += f", 맨 위 제목 {self.title_rows}행은 건너뜀"
            if self.has_unit_row:
                ok = sum(1 for u in self.units.values() if u)
                msg += f". 단위 행에서 {ok}/{len(self.units_raw)}개 컬럼의 단위를 읽었습니다"
            out.append(msg + ".")
        if self.time_column:
            fm = ", ".join(f"{k} {v:,}행" for k, v in self.time_formats.items()
                           if v and k != "해석 불가")
            out.append(f"시각 컬럼 '{self.time_column}' — {fm}.{(' ' + self.time_column_note) if self.time_column_note else ''}")
        out.extend(self._long_lines())
        if self.n_repeated_header:
            out.append(f"파일 중간의 반복 헤더 {self.n_repeated_header}행을 뺐습니다 "
                       "(여러 파일을 이어 붙인 흔적).")
        if self.n_bad_time:
            ex = ", ".join(repr(e) for e in self.bad_time_examples[:3])
            out.append(f"시각을 읽을 수 없는 행 {self.n_bad_time}개를 뺐습니다 (예: {ex}).")
        if self.n_out_of_order:
            out.append(f"{'같은 이름 안에서 ' if self.long else ''}시간이 거꾸로 가는 곳이 "
                       f"{self.n_out_of_order}곳 있어 정렬했습니다.")
        if self.n_duplicate_rows and not self.long:
            msg = f"같은 시각이 두 번 이상 나온 행 {self.n_duplicate_rows:,}개를 합쳤습니다"
            if self.n_conflicting_duplicates:
                msg += f" (그중 {self.n_conflicting_duplicates:,}개는 값이 달라 평균)"
            out.append(msg + " — 추출 구간이 겹쳤을 가능성이 큽니다.")
        tot = self.status_total()
        if tot:
            n = sum(tot.values())
            top = " · ".join(f"{k} {v:,}" for k, v in list(tot.items())[:6])
            out.append(f"상태 문자열 {n:,}셀을 결측으로 처리했습니다: {top}.")
        if self.thousands:
            cols = ", ".join(self.thousands)
            out.append(f"천 단위 콤마가 붙은 값 {sum(self.thousands.values()):,}셀을 숫자로 "
                       f"바꿨습니다 ({cols}).")
        if self.suffix_units:
            out.append("값에 붙어 있던 단위 기호를 떼어 냈습니다: "
                       + ", ".join(f"{c}({u})" for c, u in self.suffix_units.items()) + ".")
        if self.bool_columns:
            out.append("ON/OFF 형 컬럼을 1/0 으로 바꿨습니다: " + ", ".join(self.bool_columns) + ".")
        if self.dropped_columns:
            out.append(f"값이 하나도 없는 컬럼 {len(self.dropped_columns)}개를 뺐습니다"
                       f" ({', '.join(self.dropped_columns[:4])}).")
        if self.text_columns:
            out.append("숫자가 아닌 컬럼은 분석에서 뺐습니다: " + ", ".join(self.text_columns) + ".")
        if self.n_snapped:
            out.append(f"격자에서 어긋난 시각 {self.n_snapped:,}개를 가장 가까운 격자에 맞췄습니다.")
        if self.n_missing_grid:
            msg = (f"{_fmt_interval(self.interval_s)} 격자 기준으로 빠진 시각 "
                   f"{self.n_missing_grid:,}개를 빈 행으로 채웠습니다")
            if self.gaps:
                g = self.gaps[0]
                msg += f". 가장 긴 공백: {g['start']} ~ {g['end']} ({g['hours']:.1f}시간)"
            out.append(msg + ".")
        return out

    def _long_lines(self) -> list[str]:
        L = self.long
        if not L:
            return []
        vals = ", ".join(f"'{v}'" for v in L["values"])
        cols = (f"컬럼 {L['n_columns']}개" if L["n_columns"] == L["n_names"]
                else f"이름 × 값 컬럼 = 컬럼 {L['n_columns']}개")
        out = [f"긴 형식(한 줄에 시각·이름·값 하나씩)을 컬럼별로 펼쳤습니다: '{L['key']}' 의 이름 "
               f"{L['n_names']}개 × {vals} → {cols}, {L['n_rows']:,}줄 → 시각 {L['n_times']:,}개."]
        got = [f"단위는 '{L['unit']}'" if L.get("unit") else "",
               f"설명은 '{L['desc']}'" if L.get("desc") else ""]
        if any(got):
            out.append(" · ".join(g for g in got if g) + " 컬럼에서 이름마다 읽었습니다.")
        if L.get("mixed_units"):
            ex = ", ".join(f"{k}({' / '.join(v)})" for k, v in L["mixed_units"].items())
            out.append(f"단위가 섞인 이름 {L['n_mixed_units']}개: {ex} — 가장 많이 나온 단위를 적었고 "
                       "값은 바꾸지 않았습니다. 확인하세요.")
        if L.get("status_numeric"):
            out.append(f"품질 컬럼 '{L['status']}' 은 숫자 코드라 해석하지 않았습니다 (값은 그대로 씀).")
        elif L.get("n_bad_status"):
            out.append(f"품질 컬럼 '{L['status']}' 이 불량(Bad 등)인 값 {L['n_bad_status']:,}개는 결측으로 "
                       "처리했습니다 (아래 상태 문자열 집계에 포함).")
        if L.get("n_blank_key"):
            out.append(f"이름이 빈 줄 {L['n_blank_key']:,}개를 뺐습니다.")
        if L.get("n_dup"):
            msg = f"같은 시각·같은 이름이 두 번 이상 나온 값 {L['n_dup']:,}개를 합쳤습니다"
            if L.get("n_dup_conflict"):
                msg += f" (그중 {L['n_dup_conflict']:,}개는 값이 달라 평균)"
            out.append(msg + " — 추출 구간이 겹쳤을 가능성이 큽니다.")
        if L.get("ignored"):
            out.append("쓰지 않은 컬럼: " + ", ".join(f"'{c}'" for c in L["ignored"]) + ".")
        return out

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "chunk_stats"}
        d["interval_s"] = None if not np.isfinite(self.interval_s) else self.interval_s
        d["lines"] = self.lines()
        d["status_total"] = self.status_total()
        return d


def _fmt_interval(sec: float) -> str:
    if not np.isfinite(sec):
        return "?"
    if sec >= 3600 and sec % 3600 == 0:
        return f"{int(sec // 3600)}시간"
    if sec >= 60 and sec % 60 == 0:
        return f"{int(sec // 60)}분"
    return f"{sec:g}초"


@dataclass
class Table:
    """수집 결과: 규칙적인 시간 격자 위의 숫자 표 + 정리 내역."""

    df: pd.DataFrame
    report: IngestReport

    @property
    def units(self) -> dict[str, str]:
        return {k: v for k, v in self.report.units.items() if v and k in self.df.columns}


# --------------------------------------------------------------------------
# 읽기
# --------------------------------------------------------------------------

def _decode(raw: bytes, encoding: str | None) -> tuple[str, str]:
    if encoding:
        return raw.decode(encoding), encoding.lower()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16"), "utf-16"
    try:
        return raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp949"), "cp949"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1"


def _sniff_delimiter(text: str) -> str:
    sample = [ln for ln in text.splitlines()[:40] if ln.strip()]
    best, best_score = ",", -1.0
    for d in (",", "\t", ";", "|"):
        counts = [len(r) for r in csv.reader(sample, delimiter=d)]
        if not counts:
            continue
        mode = max(set(counts), key=counts.count)
        if mode < 2:
            continue
        score = mode * counts.count(mode) / len(counts)
        if score > best_score:
            best, best_score = d, score
    return best


def _needs_strip(text: str, delimiter: str) -> bool:
    """칸 앞뒤에 공백이 있을 수 있는가. 구분자·줄 끝 옆에 공백이 하나도 없으면 칸마다 다듬을 필요가
    없다 — 큰 파일에서 칸마다 strip 하는 것이 가장 비싼 일이었다 (칸 수천만 개)."""
    for ws in " \t":
        if ws == delimiter:
            continue
        if (text.startswith(ws) or ws + delimiter in text or delimiter + ws in text
                or ws + "\n" in text or ws + "\r" in text or "\n" + ws in text):
            return True
    return False


def _parse_rows(text: str, delimiter: str) -> list[list[str]]:
    """문자열 그대로의 2차원 표. pandas C 토크나이저가 빠르고, 행마다 칸 수가 다르면
    (여러 출처를 이어 붙인 파일) 표준 csv 모듈로 떨어진다."""
    strip = _needs_strip(text, delimiter)
    try:
        raw = pd.read_csv(io.StringIO(text), sep=delimiter, header=None, dtype=str,
                          na_filter=False, skip_blank_lines=False, engine="c")
    except (pd.errors.ParserError, ValueError):
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        return [[c.strip() for c in r] for r in rows] if strip else rows
    if raw.isna().to_numpy().any():
        raw = raw.fillna("")
    if strip:
        raw = raw.apply(lambda s: s.str.strip())
    return raw.to_numpy(dtype=object).tolist()


def _is_num(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------
# 긴 형식 (한 줄에 시각·이름·값 하나씩) → 넓은 형식
# --------------------------------------------------------------------------
# Historian·DB 덤프는 대개 이 모양이다. 그대로 읽으면 이름 컬럼은 글자라 빠지고 값 컬럼 하나만
# 남는데, 같은 시각의 여러 태그 값이 '중복 시각'으로 평균되어 **한 컬럼으로 뭉개진다**.

#: 컬럼 이름으로 역할 짐작 — (들어 있으면, 정확히 같으면). 소문자·공백/_/- 제거 후 비교.
#: 앞에서부터 본다 ('상태값' 은 값이 아니라 상태, '태그설명' 은 이름이 아니라 설명).
_LONG_ROLES = {
    "status": (("status", "quality", "상태", "품질", "flag"), ("q", "qual")),
    "unit": (("unit", "단위", "uom", "engunit"), ("eu",)),
    "desc": (("desc", "설명", "comment", "비고", "remark"), ()),
    "value": (("value", "값", "reading", "average", "평균", "측정치"),
              ("val", "pv", "avg", "mean", "data", "v", "y")),
    "tag": (("tag", "태그", "항목", "point", "signal", "신호", "variable", "변수", "sensor", "센서",
             "계측", "parameter", "파라미터", "metric", "channel", "채널", "series"),
            ("name", "이름", "명칭", "item", "id", "tagid", "code", "코드", "attribute", "속성",
             "key", "키")),
}
#: 품질 컬럼에서 '이 값은 믿지 말라'는 표시
_RE_BAD_STATUS = r"(?i)bad|fail|err|time ?out|comm|i/o|no ?data|shutdown|offline|invalid|불량|오류|이상|통신|실패"


def _col_role(name: str) -> str | None:
    n = re.sub(r"[\s_\-]+", "", name.lower())
    for role, (subs, exact) in _LONG_ROLES.items():
        if n in exact or any(s in n for s in subs):
            return role
    return None


def _mode_per_key(key: np.ndarray, vals: np.ndarray) -> tuple[dict[str, str], dict[str, list[str]]]:
    """이름마다 가장 많이 나온 값, 그리고 둘 이상 섞인 이름."""
    s = pd.DataFrame({"k": key, "v": vals})
    s = s[s["v"] != ""]
    if s.empty:
        return {}, {}
    cnt = s.groupby(["k", "v"], sort=False).size().sort_values(ascending=False, kind="stable")
    top: dict[str, str] = {}
    mixed: dict[str, list[str]] = {}
    for (k, v), _ in cnt.items():
        if k in top:
            mixed.setdefault(k, [top[k]]).append(v)
        else:
            top[k] = v
    return top, mixed


def _join_keys(prof: dict, keys: list[int]) -> np.ndarray:
    if len(keys) == 1:
        return prof[keys[0]]["arr"]
    return np.array([".".join(x) if all(x) else "" for x in zip(*(prof[c]["arr"] for c in keys))],
                    dtype=object)


def _long_roles(names: list[str], spec: dict) -> dict:
    """판정 결과를 컬럼 이름으로 — 큰 파일은 표본에서 한 번 정하고 조각마다 그대로 쓴다."""
    def nm(j):
        return None if j is None else names[j]
    return {"keys": [names[j] for j in spec["keys"]], "values": [names[j] for j in spec["values"]],
            "unit": nm(spec["unit"]), "desc": nm(spec["desc"]), "status": nm(spec["status"]),
            "ignored": [names[j] for j in spec["ignored"]]}


def _spec_from_roles(names: list[str], body: list[list[str]], roles: dict) -> dict:
    idx = {n: j for j, n in enumerate(names)}
    need = roles["keys"] + roles["values"] + [roles[r] for r in ("unit", "desc", "status") if roles[r]]
    missing = [n for n in need if n not in idx]
    if missing:
        raise ValueError(f"긴 형식 컬럼이 이 부분에 없습니다: {missing}")
    stride = max(1, len(body) // 4000)
    prof = {}
    for n in need:
        col = list(map(itemgetter(idx[n]), body))
        samp = [c for c in col[::stride] if c]
        prof[idx[n]] = {"arr": np.array(col, dtype=object),
                        "num": sum(_is_num(c) for c in samp) / len(samp) if samp else 0.0}
    keys = [idx[n] for n in roles["keys"]]
    return {"keys": keys, "key": _join_keys(prof, keys), "values": [idx[n] for n in roles["values"]],
            **{r: (idx[roles[r]] if roles[r] else None) for r in ("unit", "desc", "status")},
            "ignored": [idx[n] for n in roles["ignored"] if n in idx], "prof": prof}


def _detect_long(names: list[str], body: list[list[str]], tj: int,
                 t_ns: np.ndarray, need_cover: bool = True) -> dict | None:
    """긴 형식이면 역할(이름·값·단위·설명·품질 컬럼)을, 아니면 None.

    넓은 파일에 글자 컬럼(운전 모드 등)이 하나 있는 경우와 헷갈리면 안 된다. 그래서
    ① 숫자 컬럼이 4개 이하, ② (시각, 이름) 쌍이 거의 유일, ③ 같은 시각이 여러 줄에 나오거나
    이름 컬럼답게 불리고 이름마다 전체 기간을 덮을 때만 긴 형식으로 본다. 파일 여기저기서 뜬 표본
    (큰 파일)은 기간을 덮는지 알 수 없어 ``need_cover=False`` 로 이름만 본다.
    """
    n = len(body)
    if n < 6 or len(names) > 12:
        return None
    stride = max(1, n // 4000)
    prof: dict[int, dict] = {}
    for j, name in enumerate(names):
        if j == tj:
            continue
        col = list(map(itemgetter(j), body))
        samp = [c for c in col[::stride] if c]
        if not samp:
            continue
        arr = np.array(col, dtype=object)
        prof[j] = {"role": _col_role(name), "arr": arr, "ne": float((arr != "").mean()),
                   "num": sum(_is_num(c) for c in samp) / len(samp),
                   "bool": sum(c.upper() in _BOOL_TOKENS for c in samp) / len(samp)}
    numeric = [j for j, p in prof.items() if p["num"] >= 0.5 and p["role"] not in ("status", "tag")]
    if not numeric or len(numeric) > 4:
        return None

    cands = []
    for j, p in prof.items():
        if p["ne"] < 0.95 or p["bool"] >= 0.8 or p["role"] not in (None, "tag"):
            continue
        if p["num"] > 0.2 and p["role"] != "tag":
            continue
        k = len(pd.unique(p["arr"]))
        if 2 <= k <= min(5000, n // 3):
            cands.append((p["role"] == "tag", k, j))
    if not cands:
        return None
    cands.sort(reverse=True)

    def unique_frac(cols: list[int]) -> float:
        d = pd.DataFrame({"t": t_ns, **{f"k{c}": prof[c]["arr"] for c in cols}})
        return 1.0 - float(d.duplicated().mean())

    keys = [cands[0][2]]
    u = unique_frac(keys)
    if u < 0.95:                              # 이름이 두 컬럼에 나뉘어 있을 수 있다 (설비 + 항목)
        best = max(((unique_frac(sorted(keys + [c[2]])), c[2]) for c in cands[1:]), default=None)
        if best and best[0] - u >= 0.05:
            u, keys = best[0], sorted(keys + [best[1]])
    if u < 0.8:
        return None
    key = _join_keys(prof, keys)

    t_dup = 1.0 - len(np.unique(t_ns)) / n
    g = pd.Series(t_ns).groupby(key)
    span = float(t_ns.max() - t_ns.min())
    cover = float(((g.max() - g.min()) / span).median()) if span > 0 else 0.0
    if g.size().median() < 3:
        return None
    named = any(prof[c]["role"] == "tag" for c in keys)
    if not (t_dup >= 0.3 or (named and (cover >= 0.5 or not need_cover))):
        return None

    rest = [j for j in prof if j not in keys]
    values = [j for j in rest if prof[j]["role"] == "value"]
    if not values:
        values = [j for j in rest if prof[j]["role"] is None and prof[j]["num"] >= 0.3]
    if not values:
        return None
    spec = {"keys": keys, "key": key, "values": values, "unit": None, "desc": None,
            "status": None, "ignored": []}
    for j in rest:
        if j in values:
            continue
        role = prof[j]["role"]
        if role in ("unit", "desc", "status") and spec[role] is None:
            spec[role] = j
            continue
        if role is None and prof[j]["num"] < 0.3:
            # 이름 없는 글자 컬럼: 단위 모양이면 단위, 이름마다 한 값이면 설명
            vals = pd.unique(prof[j]["arr"][::stride])
            vals = [v for v in vals if v]
            if spec["unit"] is None and vals and \
                    sum(normalize_unit(v) is not None for v in vals) >= 0.8 * len(vals):
                spec["unit"] = j
                continue
            if spec["desc"] is None:
                per = pd.Series(prof[j]["arr"]).groupby(key).nunique()
                if (per <= 1).mean() >= 0.95:
                    spec["desc"] = j
                    continue
        spec["ignored"].append(j)
    spec["prof"] = prof
    return spec


def _pivot_long(names: list[str], t_ns: np.ndarray, spec: dict, rep: IngestReport) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """긴 형식을 (문자열 그대로) 넓게 편다. 숫자 변환·상태 문자열 처리는 넓힌 뒤 컬럼마다
    기존 규칙 그대로 한다 — 그래야 'Bad' 가 어느 태그에서 났는지 보고할 수 있다."""
    prof, key, vals = spec["prof"], spec["key"], spec["values"]
    info: dict = {"n_rows": int(len(key)), "key": ", ".join(names[j] for j in spec["keys"]),
                  "values": [names[j] for j in vals]}
    long = pd.DataFrame({"t": t_ns, "k": key, **{f"v{j}": prof[j]["arr"] for j in vals}})

    if spec["status"] is not None:
        j = spec["status"]
        st = pd.Series(prof[j]["arr"], dtype=object)
        info["status"] = names[j]
        if prof[j]["num"] >= 0.9:
            info["status_numeric"] = True             # 품질 코드 체계는 계통마다 다르다 — 짐작하지 않는다
        else:
            bad = st.str.contains(_RE_BAD_STATUS, regex=True).fillna(False).to_numpy(dtype=bool)
            n_bad = 0
            for j2 in vals:
                c = f"v{j2}"
                hit = bad & (long[c].to_numpy() != "")
                long.loc[hit, c] = st[hit].to_numpy()  # 값 대신 상태 글자 → 아래에서 결측 + 태그별 집계
                n_bad = max(n_bad, int(hit.sum()))
            info["n_bad_status"] = n_bad

    blank = long["k"] == ""
    info["n_blank_key"] = int(blank.sum())
    long = long[~blank.to_numpy()]
    # 이름별로 시간이 거꾸로 가는 곳 (이름 순으로 정렬된 파일은 이름이 바뀔 때마다 시각이 되돌아간다)
    rep.n_out_of_order = int((long.groupby("k", sort=False)["t"].diff() < 0).sum())
    occ = long.groupby(["t", "k"], sort=False).cumcount()
    dup = occ.to_numpy() > 0
    info["n_dup"] = int(dup.sum())
    if dup.any():
        first = long[~dup].set_index(["t", "k"])
        again = long[dup].set_index(["t", "k"])
        cols = [f"v{j}" for j in vals]
        differ = (again[cols] != first.loc[again.index, cols].to_numpy()).any(axis=1)
        info["n_dup_conflict"] = int(differ.sum())

    order = list(pd.unique(long["k"]))
    long = long.assign(occ=occ.to_numpy()).set_index(["t", "occ", "k"])
    wide = long[[f"v{j}" for j in vals]].unstack("k")
    one = len(vals) == 1
    colnames, pick = [], []
    for k in order:
        for j in vals:
            colnames.append(k if one else f"{k}.{names[j]}")
            pick.append((f"v{j}", k))
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(pick))
    times = pd.Series(wide.index.get_level_values("t").to_numpy().astype("datetime64[ns]"))
    cells = np.array(wide.to_numpy(dtype=object), dtype=object)   # to_numpy 는 읽기 전용 뷰일 수 있다
    cells[pd.isna(cells)] = ""                          # 컬럼마다 fillna 하면 태그 수백 개에서 느리다
    frame = pd.DataFrame(cells, columns=colnames)         # 새 번호 — 이름이 't' 인 태그와 부딪히지 않게

    for role in ("unit", "desc"):
        j = spec[role]
        if j is None:
            continue
        info[role] = names[j]
        top, mixed = _mode_per_key(long.index.get_level_values("k").to_numpy(),
                                   prof[j]["arr"][~blank.to_numpy()])
        for k, v in top.items():
            for c in ([k] if one else [f"{k}.{names[j2]}" for j2 in vals]):
                if role == "unit":
                    rep.units_raw[c] = v
                    rep.units[c] = normalize_unit(v) or ""
                else:
                    rep.descriptions[c] = v
        if role == "unit" and mixed:
            info["mixed_units"] = {k: v for k, v in list(mixed.items())[:5]}
            info["n_mixed_units"] = len(mixed)
    rep.has_unit_row = bool(rep.units_raw)
    rep.has_desc_row = bool(rep.descriptions)
    info["ignored"] = [names[j] for j in spec["ignored"]]
    info["n_names"] = len(order)
    info["n_columns"] = len(colnames)
    info["n_times"] = int(times.nunique())
    rep.long = info
    return frame, times, colnames


#: 이보다 큰 파일은 조각으로 나눠 읽는다. 통째로 읽으면 칸마다 파이썬 문자열이 생겨 메모리가 파일
#: 크기의 10배 넘게 든다 (5GB 파일이면 수십 GB).
BIG_FILE = 64 * 2 ** 20
#: 조각 하나의 크기 [바이트]. 조각 하나를 읽는 동안 메모리는 대략 이것의 10~20배.
CHUNK_BYTES = 32 * 2 ** 20
#: 큰 파일의 정리 결과를 두는 곳 (CSV 옆). 앱은 '.' 으로 시작하는 폴더를 목록에서 뺀다.
CACHE_DIR = ".pforecast_cache"


def read_table(path: str | Path, time_column: str | None = None, *,
               encoding: str | None = None, regularize: bool = True,
               progress: Callable[[str], None] | None = None, chunk_bytes: int | None = None,
               cache: bool = True, workers: int | None = None) -> Table:
    """현장 CSV 를 읽어 정리한다. 무엇을 고쳤는지는 ``Table.report`` 에 남는다.

    ``BIG_FILE`` 보다 큰 파일은 조각으로 나눠 읽어 메모리가 파일 크기를 따라가지 않게 하고, 결과를
    CSV 옆 ``.pforecast_cache/`` 에 저장해 두었다가 다음부터는 바로 읽는다 (CSV 가 바뀌거나 이 모듈이
    바뀌면 다시 읽는다). ``progress(메시지)`` 로 진행 상황을 알린다. ``chunk_bytes`` 를 주면 그보다 큰
    파일을 그 크기로 나눠 읽는다 (시험용). ``workers`` 는 조각을 나눠 읽을 프로세스 수 (기본: 파일이 크면 코어 수 - 1,
    최대 4).
    """
    p = Path(path)
    size = p.stat().st_size
    if size <= (chunk_bytes or BIG_FILE):
        text, enc = _decode(p.read_bytes(), encoding)
        tab = read_text(text, time_column=time_column, regularize=regularize)
        tab.report.encoding = enc
    else:
        cpath = _cache_path(p, time_column, encoding, regularize) if cache else None
        tab = _cache_load(cpath) if cpath is not None else None
        if tab is None:
            tab = _read_chunks(p, time_column, encoding, regularize, chunk_bytes or CHUNK_BYTES,
                               progress or (lambda m: None), workers)
            if cpath is not None:
                if progress:
                    progress("정리 결과를 저장하는 중 (다음부터는 바로 열립니다)")
                _cache_save(cpath, tab)
    tab.report.path = str(path)
    return tab


# --------------------------------------------------------------------------
# 큰 파일: 조각으로 나눠 읽기
# --------------------------------------------------------------------------

def _sample_blocks(p: Path, size: int, head: int = 2 << 20, n: int = 24,
                   block: int = 256 << 10) -> list[bytes]:
    """맨 앞 조각 + 파일 여기저기서 뜬 조각들. 앞 조각 말고는 줄 경계로 자른다."""
    out = []
    with open(p, "rb") as f:
        raw = f.read(head)
        out.append(raw[:raw.rfind(b"\n") + 1] or raw)
        if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):      # UTF-16 은 줄 경계를 바이트로 못 자른다
            return out
        # 조각끼리 겹치면 같은 줄이 두 번 들어가 (시각, 이름) 쌍이 중복으로 보인다
        start = len(out[0])
        blk = min(block, (size - start) // n)
        if blk < 4096:
            return out
        for i in range(n):
            f.seek(start + int((size - start) * i / n))
            b = f.read(blk)
            s, e = b.find(b"\n") + 1, b.rfind(b"\n") + 1
            if 0 < s < e:
                out.append(b[s:e])
    return out


def _sniff_encoding(blocks: list[bytes], encoding: str | None) -> str:
    """앞부분만 보면 안 된다 — 앞은 영문 태그뿐이고 뒤에 한글(CP949)이 나오는 파일이 있다."""
    if encoding:
        return encoding.lower()
    if blocks[0][:2] in (b"\xff\xfe", b"\xfe\xff"):
        return "utf-16"
    for enc in ("utf-8-sig", "cp949"):
        try:
            for i, b in enumerate(blocks):
                b.decode(enc if i == 0 else enc.replace("-sig", ""))
            return enc.replace("-sig", "")
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _read_chunks(p: Path, time_column: str | None, encoding: str | None, regularize: bool,
                 step: int, say: Callable[[str], None], workers: int | None = None) -> Table:
    size = p.stat().st_size
    say(f"파일 구조를 보는 중 ({_fmt_size(size)})")
    blocks = _sample_blocks(p, size)
    enc = _sniff_encoding(blocks, encoding)
    py_enc = "utf-8-sig" if enc == "utf-8" else enc
    head_text = blocks[0].decode(py_enc, errors="replace")

    # ---- 헤더·시각 컬럼은 앞부분으로, 긴 형식 판정은 파일 곳곳의 표본으로 한 번만 -----------
    lay = IngestReport()
    lay.delimiter = _sniff_delimiter(head_text)
    names, _, _, tcol, _, n_head = _split_header(_parse_rows(head_text, lay.delimiter), time_column, lay)
    lines = head_text.split("\n")
    hb = "\n".join(lines[lay.title_rows:n_head]) + "\n"       # 조각마다 앞에 붙일 헤더 (단위·설명 행 포함)
    hb_lines = n_head - lay.title_rows
    roles: dict | bool = False
    if tcol is not None and lay.header_rows == 1:
        sample = hb + "\n".join(lines[n_head:]) + "".join(
            b.decode(enc, errors="replace") for b in blocks[1:])
        srep = IngestReport()
        names_s, _, body, _, tj, _ = _split_header(_parse_rows(sample, lay.delimiter), tcol, srep)
        body = [r for r in body if any(r)]
        if body:
            t, _ = parse_times(pd.Series(list(map(itemgetter(tj), body)), dtype=object))
            ok = t.notna().to_numpy()
            body = [r for r, k in zip(body, ok) if k]
            if body:
                t_ns = t[ok].to_numpy().astype("datetime64[ns]").astype("int64")
                spec = _detect_long(names_s, body, tj, t_ns, need_cover=False)
                roles = _long_roles(names_s, spec) if spec else False

    # ---- 조각마다 기존 규칙 그대로 (헤더를 붙여 read_text) — 코어가 여럿이면 나눠서 -------------
    args = (tcol, lay.delimiter, roles)

    def chunks():
        with open(p, "r", encoding=py_enc, errors="replace", newline="") as f:
            for _ in range(n_head):
                f.readline()
            while True:
                lines_ = f.readlines(step)
                if not lines_:
                    return
                yield hb + "".join(lines_), len(lines_), f.buffer.tell()

    n_workers = workers if workers is not None else _auto_workers(size, step)
    try:
        acc = _collect(chunks(), args, n_workers, hb_lines, size, say)
    except (OSError, BrokenProcessPool) as exc:       # 작업 프로세스를 못 띄우는 PC — 한 코어로 다시
        if n_workers <= 1:
            raise
        say(f"여러 코어로 읽지 못해 한 코어로 다시 읽습니다 ({type(exc).__name__})")
        acc = _collect(chunks(), args, 1, hb_lines, size, say)
    reps, frames, order, counts, idx, n_boundary = (acc[k] for k in (
        "reps", "frames", "order", "counts", "idx", "n_boundary"))
    if idx is None:
        raise ValueError("시각을 읽을 수 있는 데이터 행이 없습니다")

    # ---- 합치기: 조각 경계에서 갈린 시각은 한 행으로, 진짜 중복은 평균 + 보고 --------------------
    # 최종 표를 한 번에 잡아 두고 컬럼마다 채운다. 컬럼마다 시각 색인을 따로 만들면 5GB 파일에서
    # 메모리가 두세 배로 불었다 (6.3 GB).
    say("조각을 합치는 중")
    rep = _merge_reports(reps, lay, n_head, hb_lines)
    long_mode = bool(rep.long)
    if not long_mode:
        rep.n_out_of_order += n_boundary
    drop = set()
    for c, (n_num, n_ne) in counts.items():        # 통째로 읽을 때와 같은 기준 (read_text 의 숫자 변환)
        if n_num < max(3, 0.05 * n_ne):
            drop.add(c)
            st = rep.status_counts.pop(c, {})
            rep.text_columns[c] = [k for k, _ in sorted(st.items(), key=lambda kv: -kv[1])[:3]]
    seen = set(order)
    cols = [c for c in (order if long_mode else names) if c in seen and c not in drop]
    positions = [idx.get_indexer(d.index) for d in frames]
    out = np.full((len(idx), len(cols)), np.nan)
    dup_rows = conf_rows = 0
    for j, c in enumerate(cols):
        P = [pos for pos, d in zip(positions, frames) if c in d.columns]
        V = [d[c].to_numpy(dtype=float) for d in frames if c in d.columns]
        if not P:
            continue
        P, V = np.concatenate(P), np.concatenate(V)
        ok = ~np.isnan(V)
        P, V = P[ok], V[ok]
        cnt = np.bincount(P, minlength=len(idx))
        if len(cnt) and cnt.max() <= 1:
            out[P, j] = V
            continue
        has = cnt > 0
        out[has, j] = np.bincount(P, weights=V, minlength=len(idx))[has] / cnt[has]
        extra = int((cnt[cnt > 1] - 1).sum())
        o = np.lexsort((V, P))
        Ps, Vs = P[o], V[o]
        starts = np.r_[0, np.flatnonzero(np.diff(Ps)) + 1]
        ends = np.r_[starts[1:], len(Ps)]
        multi = (ends - starts) > 1
        scale = float(np.median(np.abs(V))) or 1.0
        conf = int(np.sum(Vs[ends[multi] - 1] - Vs[starts[multi]] > 1e-9 * scale + 1e-12))
        if long_mode:
            rep.long["n_dup"] = rep.long.get("n_dup", 0) + extra
            rep.long["n_dup_conflict"] = rep.long.get("n_dup_conflict", 0) + conf
        else:
            dup_rows, conf_rows = max(dup_rows, extra), max(conf_rows, conf)
    frames.clear()
    rep.n_duplicate_rows += dup_rows
    rep.n_conflicting_duplicates += conf_rows
    df = pd.DataFrame(out, index=idx, columns=cols)
    rep.dropped_columns = [c for c in rep.dropped_columns if c not in df.columns]
    rep.text_columns = {c: v for c, v in rep.text_columns.items() if c not in df.columns}
    if long_mode:
        rep.long["n_columns"] = df.shape[1]
        rep.long["n_names"] = df.shape[1] // max(1, len(rep.long["values"]))
        rep.long["n_times"] = len(idx)
    rep.encoding = enc
    say("시간 격자를 맞추는 중")
    return Table(df=_regularize(df, rep, regularize, tcol), report=rep)


def _auto_workers(size: int, step: int) -> int:
    """조각이 넉넉할 때만 여러 코어로. 작업 프로세스 하나가 조각 하나에 메모리를 수백 MB 쓴다."""
    if size < 8 * step:
        return 1
    return max(1, min(4, (os.cpu_count() or 2) - 1))


def _parse_chunk(text: str, tcol: str | None, delimiter: str, roles) -> Table | None:
    """조각 하나 (작업 프로세스에서도 돈다). 빈 줄·반복 헤더뿐인 조각은 None."""
    try:
        return read_text(text, time_column=tcol, regularize=False, delimiter=delimiter,
                         long=roles, chunk=True)
    except ValueError as exc:
        if "데이터 행이 없습니다" not in str(exc):
            raise
        return None


def _collect(chunks, args: tuple, n_workers: int, hb_lines: int, size: int,
             say: Callable[[str], None]) -> dict:
    """조각 결과를 **파일 순서대로** 모은다 (경계의 시간 역행·긴 형식 컬럼 순서가 순서에 달렸다)."""
    acc = {"reps": [], "frames": [], "order": [], "counts": {}, "idx": None, "n_boundary": 0, "last": None}

    def take(part: Table | None, n_lines: int, pos: int) -> None:
        say(f"읽는 중 {min(99, int(pos / size * 100))}% ({_fmt_size(pos)} / {_fmt_size(size)})")
        if part is None:
            blank = IngestReport()
            blank.n_lines = hb_lines + n_lines
            blank.n_blank_rows = n_lines
            acc["reps"].append(blank)
            return
        acc["reps"].append(part.report)
        cs = part.report.chunk_stats
        for c, (a, b) in cs.get("counts", {}).items():
            cnt = acc["counts"].setdefault(c, [0, 0])
            cnt[0] += a
            cnt[1] += b
        if "edges" in cs:                       # 조각 경계에서 시간이 거꾸로 가는가 (파일 순서 기준)
            if acc["last"] is not None and cs["edges"][0] < acc["last"]:
                acc["n_boundary"] += 1
            acc["last"] = cs["edges"][1]
        d = part.df
        if len(d.index):
            acc["idx"] = d.index if acc["idx"] is None else acc["idx"].union(d.index)
            acc["frames"].append(d)
        known = set(acc["order"])
        acc["order"] += [c for c in d.columns if c not in known]

    if n_workers <= 1:
        for text, n_lines, pos in chunks:
            take(_parse_chunk(text, *args), n_lines, pos)
        return acc
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        pending: deque = deque()
        for text, n_lines, pos in chunks:
            pending.append((ex.submit(_parse_chunk, text, *args), n_lines, pos))
            while len(pending) >= 2 * n_workers:       # 읽어 둔 조각이 쌓여 메모리가 불지 않게
                fut, nl, ps = pending.popleft()
                take(fut.result(), nl, ps)
        while pending:
            fut, nl, ps = pending.popleft()
            take(fut.result(), nl, ps)
    return acc


def _merge_reports(reps: list[IngestReport], lay: IngestReport, n_head: int, hb_lines: int) -> IngestReport:
    """조각별 정리 내역을 파일 하나의 내역으로."""
    r = IngestReport()
    first = next((x for x in reps if x.time_column is not None), reps[0])
    r.delimiter, r.title_rows, r.header_rows = lay.delimiter, lay.title_rows, lay.header_rows
    r.time_column, r.time_column_note = lay.time_column, lay.time_column_note
    r.has_unit_row, r.has_desc_row = first.has_unit_row, first.has_desc_row
    r.renamed_columns = dict(first.renamed_columns)
    r.n_lines = n_head + sum(x.n_lines - hb_lines for x in reps)
    for a in ("n_data_rows", "n_repeated_header", "n_blank_rows", "n_bad_time", "n_out_of_order",
              "n_duplicate_rows", "n_conflicting_duplicates"):
        setattr(r, a, sum(getattr(x, a) for x in reps))
    for x in reps:
        for k, v in x.time_formats.items():
            r.time_formats[k] = r.time_formats.get(k, 0) + v
        r.bad_time_examples.extend(x.bad_time_examples[:5 - len(r.bad_time_examples)])
        for c, d in x.status_counts.items():
            tgt = r.status_counts.setdefault(c, {})
            for k, v in d.items():
                tgt[k] = tgt.get(k, 0) + v
        for c, v in x.thousands.items():
            r.thousands[c] = r.thousands.get(c, 0) + v
        for c, u in x.suffix_units.items():
            r.suffix_units.setdefault(c, u)
        r.bool_columns += [c for c in x.bool_columns if c not in r.bool_columns]
        for name in ("units", "units_raw", "descriptions"):
            dst = getattr(r, name)
            for k, v in getattr(x, name).items():
                if not dst.get(k):
                    dst[k] = v
        r.dropped_columns += [c for c in x.dropped_columns if c not in r.dropped_columns]
        for c, v in x.text_columns.items():
            r.text_columns.setdefault(c, v)
    longs = [x.long for x in reps if x.long]
    if longs:
        L = dict(longs[0])
        for k in ("n_rows", "n_blank_key", "n_bad_status", "n_dup", "n_dup_conflict"):
            if any(k in x for x in longs):
                L[k] = sum(x.get(k, 0) for x in longs)
        mixed: dict[str, list[str]] = {}
        for x in longs:
            for k, v in x.get("mixed_units", {}).items():
                mixed.setdefault(k, v)
        L.pop("mixed_units", None)
        L.pop("n_mixed_units", None)
        if mixed:
            L["mixed_units"] = dict(list(mixed.items())[:5])
            L["n_mixed_units"] = len(mixed)
        r.long = L
    return r


def _fmt_size(n: float) -> str:
    return f"{n / 2 ** 30:.1f} GB" if n >= 2 ** 30 else f"{n / 2 ** 20:.0f} MB"


# ---- 정리 결과 캐시 (큰 파일만) ---------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _code_version() -> str:
    """이 모듈이나 단위 표가 바뀌면 캐시를 버린다 — 고친 정리 규칙이 옛 결과에 가려지지 않게."""
    h = hashlib.sha256()
    for f in (Path(__file__), Path(__file__).parents[1] / "core" / "units.py"):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def _cache_path(p: Path, time_column, encoding, regularize) -> Path:
    st = p.stat()
    key = f"{st.st_size}|{st.st_mtime_ns}|{time_column}|{encoding}|{regularize}|{_code_version()}"
    return p.parent / CACHE_DIR / f"{p.name}.{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


def _cache_load(c: Path) -> Table | None:
    if not c.exists():
        return None
    try:
        with np.load(c, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            values, index = z["values"], z["index"]
        freq = pd.Timedelta(seconds=meta["freq_s"]) if meta.get("freq_s") else None
        idx = pd.DatetimeIndex(index.astype("datetime64[ns]"), name=meta["index_name"], freq=freq)
        df = pd.DataFrame(values, index=idx, columns=meta["columns"])
        return Table(df=df, report=IngestReport(**meta["report"]))
    except (OSError, ValueError, KeyError, TypeError):      # 깨진 캐시는 무시하고 다시 읽는다
        return None


def _cache_save(c: Path, tab: Table) -> None:
    df = tab.df
    if not isinstance(df.index, pd.DatetimeIndex):
        return
    try:
        c.parent.mkdir(exist_ok=True)
        stem = c.name.rsplit(".", 2)[0]
        for old in c.parent.glob(glob.escape(stem) + ".*.npz"):
            old.unlink(missing_ok=True)
        meta = {"columns": [str(x) for x in df.columns], "index_name": df.index.name,
                "freq_s": tab.report.interval_s if df.index.freq is not None else None,
                "report": dataclasses.asdict(tab.report)}
        tmp = c.with_name(c.name + ".part")
        with open(tmp, "wb") as f:
            np.savez(f, values=df.to_numpy(dtype=float),
                     index=df.index.to_numpy().astype("datetime64[ns]").astype("int64"),
                     meta=np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8))
        os.replace(tmp, c)
    except (OSError, TypeError, ValueError):                # 읽기 전용 폴더 등 — 캐시는 없어도 된다
        return


def read_text(text: str, time_column: str | None = None, *, regularize: bool = True,
              delimiter: str | None = None, long: bool | dict | None = None,
              chunk: bool = False) -> Table:
    """``long`` — None: 긴 형식이면 알아서 펼친다, False: 펼치지 않는다, dict: 이 역할(`_long_roles`)대로
    펼친다 (큰 파일의 조각은 한 조각에 이름이 하나뿐일 수 있어 판정을 파일 표본으로 미리 한다).
    ``chunk`` — 큰 파일의 한 조각: 글자 컬럼 판정은 합친 뒤 파일 전체로 하도록 컬럼을 남기고 개수만 센다."""
    rep = IngestReport()
    rep.delimiter = delimiter or _sniff_delimiter(text)
    rows = _parse_rows(text, rep.delimiter)
    rep.n_lines = len(rows)
    names, header_set, body, tcol, tj, _ = _split_header(rows, time_column, rep)
    return _body_to_table(names, header_set, body, tcol, tj, rep, regularize, long, chunk)


def _split_header(rows: list[list[str]], time_column: str | None, rep: IngestReport):
    """(이름, 헤더·메타 행들, 데이터 행, 시각 컬럼, 그 위치, 데이터 앞 줄 수)."""
    # ---- 헤더: 비어 있지 않은 칸이 둘 이상이고 대부분 숫자가 아닌 첫 행 ------------
    h = 0
    while h < len(rows):
        cells = [c for c in rows[h] if c]
        if len(cells) >= 2 and sum(_is_num(c) for c in cells) <= len(cells) * 0.3:
            break
        h += 1
    if h >= len(rows):
        raise ValueError("헤더 행을 찾지 못했습니다 (숫자가 아닌 이름이 둘 이상 있는 행)")
    rep.title_rows = h
    header = rows[h]
    width = max(map(len, rows[h:]))
    header = header + [""] * (width - len(header))
    names, seen = [], {}
    for j, c in enumerate(header):
        name = c or f"col{j}"
        if name in seen:
            seen[name] += 1
            new = f"{name}__{seen[name]}"
            rep.renamed_columns[new] = name
            name = new
        else:
            seen[name] = 1
        names.append(name)

    body = rows[h + 1:]
    if min(map(len, body), default=width) < width:          # 칸 수가 다른 행 (csv 모듈로 읽은 경우)
        body = [r + [""] * (width - len(r)) for r in body]

    # ---- 시각 컬럼 -------------------------------------------------------------
    sample = [r for r in body[:600] if any(r)]
    tcol = None
    if time_column and time_column in names:
        tcol = time_column
    else:
        cands = [j for j, n in enumerate(names)
                 if j < 3 or any(w in n.lower() for w in _TIME_WORDS)]
        best = (0.0, -1)
        for j in cands:
            sc = _time_score([r[j] for r in sample])
            bonus = 0.05 if any(w in names[j].lower() for w in _TIME_WORDS) else 0.0
            if sc >= 0.5 and sc + bonus > best[0]:
                best = (sc + bonus, j)
        if best[1] >= 0:
            tcol = names[best[1]]
            if time_column:
                rep.time_column_note = (f"설정의 시각 컬럼 '{time_column}' 이 파일에 없어 "
                                        f"'{tcol}' 을 썼습니다.")
    rep.time_column = tcol
    tj = names.index(tcol) if tcol else None

    # ---- 설명/단위 행: 헤더 바로 아래, 시각 칸이 시각이 아니고 대부분 문자인 행 -------
    meta: list[list[str]] = []
    k = 0
    while k < min(len(body), 5):
        r = body[k]
        tcell = r[tj] if tj is not None else ""
        cells = [c for j, c in enumerate(r) if c and j != tj]
        if not cells:
            k += 1
            meta.append(r)
            continue
        if tj is not None and _time_score([tcell]) > 0:
            break
        if sum(_is_num(c) for c in cells) > len(cells) * 0.3:
            break
        meta.append(r)
        k += 1
    for r in meta:
        cells = [(j, c) for j, c in enumerate(r) if c and j != tj]
        if not cells:
            continue
        parsed = [(j, c, normalize_unit(c)) for j, c in cells]
        n_unit = sum(1 for _, _, u in parsed if u is not None)
        if not rep.has_unit_row and n_unit >= 0.5 * len(cells):
            rep.has_unit_row = True
            for j, c, u in parsed:
                rep.units_raw[names[j]] = c
                rep.units[names[j]] = u or ""
        elif not rep.has_desc_row:
            rep.has_desc_row = True
            for j, c in cells:
                rep.descriptions[names[j]] = c
    rep.header_rows = 1 + sum(1 for r in meta if any(r))
    return names, [header] + meta, body[len(meta):], tcol, tj, h + 1 + len(meta)


def _body_to_table(names, header_set, body, tcol, tj, rep: IngestReport, regularize: bool,
                   long: bool | dict | None, chunk: bool = False) -> Table:
    # ---- 데이터 행 분류 ----------------------------------------------------------
    # 시각을 먼저 읽는다. 반복 헤더·잡음 행은 시각이 안 읽히는 소수의 행 안에만 있으므로
    # 그 행들만 따로 들여다보면 된다 (7만 행 전체를 문자열 비교하면 수 초가 걸린다).
    if not body:
        raise ValueError("데이터 행이 없습니다")
    nonblank = [r for r in body if any(r)]
    rep.n_blank_rows = len(body) - len(nonblank)
    body = nonblank

    def looks_like_header(r: list[str]) -> bool:
        for hr in header_set:
            ref = [(j, c) for j, c in enumerate(hr) if c]
            if ref and sum(1 for j, c in ref if j < len(r) and r[j] == c) >= 0.8 * len(ref):
                return True
        return False

    if tcol is not None:
        tvals = pd.Series(list(map(itemgetter(tj), body)), dtype=object)
        codes, uniq = pd.factorize(tvals)
        if len(uniq) < 0.8 * len(tvals):
            # 긴 형식은 같은 시각이 태그 수만큼 반복된다 — 서로 다른 시각만 해석한다
            pu, rep.time_formats = parse_times(pd.Series(uniq, dtype=object),
                                               counts=np.bincount(codes, minlength=len(uniq)))
            times = pd.Series(pu.to_numpy()[codes])
        else:
            times, rep.time_formats = parse_times(tvals)
        bad = times.isna().to_numpy()
        keep = ~bad
        for i in np.flatnonzero(bad):
            r = body[i]
            if looks_like_header(r):
                rep.n_repeated_header += 1
            elif not any(c for j, c in enumerate(r) if j != tj):
                rep.n_blank_rows += 1
            else:
                rep.n_bad_time += 1
                if len(rep.bad_time_examples) < 5:
                    rep.bad_time_examples.append(r[tj])
        rep.time_formats["해석 불가"] = rep.n_bad_time
        body = [r for r, k in zip(body, keep) if k]
        times = times[keep].reset_index(drop=True)
        # 파일 순서 그대로에서 시간이 거꾸로 가는 곳
        t = times.to_numpy()
        if len(t) > 1:
            rep.n_out_of_order = int(np.sum(t[1:] < t[:-1]))
        if chunk and len(t):
            rep.chunk_stats["edges"] = (t[0], t[-1])
        data_cols = [c for c in names if c != tcol]
    else:
        hdr = [looks_like_header(r) for r in body]
        rep.n_repeated_header = int(sum(hdr))
        body = [r for r, h_ in zip(body, hdr) if not h_]
        times = None
        data_cols = list(names)
    rep.n_data_rows = len(body)
    if not body:
        raise ValueError("시각을 읽을 수 있는 데이터 행이 없습니다")
    spec = None
    if tcol is not None and rep.header_rows == 1 and long is not False:   # 단위·설명 행이 있으면 넓은 형식
        t_ns = times.to_numpy().astype("datetime64[ns]").astype("int64")
        spec = (_spec_from_roles(names, body, long) if isinstance(long, dict)
                else _detect_long(names, body, tj, t_ns))
    if spec is not None:
        frame, times, data_cols = _pivot_long(names, t_ns, spec, rep)
    else:
        frame = pd.DataFrame(body, columns=names, dtype=object)

    # ---- 숫자 변환 ------------------------------------------------------------
    numeric: dict[str, pd.Series] = {}
    # 한 번에 숫자로 바꿔 본다 — 컬럼 수백 개를 하나씩 바꾸면 컬럼마다 드는 고정비가 더 크다.
    # 숫자가 아닌 칸이 남은 컬럼만 아래에서 하나씩 들여다본다 (콤마·ON/OFF·단위 기호·상태 문자열).
    A = frame[data_cols].to_numpy(dtype=object) if data_cols else np.empty((len(frame), 0), dtype=object)
    NE = A != ""
    V = np.asarray(pd.to_numeric(np.where(NE, A, None).ravel(), errors="coerce"),
                   dtype=float).reshape(A.shape)
    for jc, c in enumerate(data_cols):
        arr = A[:, jc]
        nonempty_np = NE[:, jc]
        n_nonempty = int(nonempty_np.sum())
        if n_nonempty == 0:
            rep.dropped_columns.append(c)
            continue
        vj = V[:, jc]
        if not (np.isnan(vj) & nonempty_np).any() and n_nonempty >= 3:
            if chunk:
                rep.chunk_stats.setdefault("counts", {})[c] = (n_nonempty, n_nonempty)
            numeric[c] = pd.Series(vj, index=frame.index)
            continue
        nonempty = pd.Series(nonempty_np, index=frame.index)
        v = pd.Series(vj.copy(), index=frame.index, dtype=float)
        left = v.isna() & nonempty
        if left.any():
            t = pd.Series(arr[left.to_numpy()], index=left[left].index, dtype=object)
            comma = t.str.fullmatch(_RE_THOUSANDS).fillna(False).astype(bool)
            if comma.any():
                idx = comma[comma].index
                v.loc[idx] = pd.to_numeric(t[idx].str.replace(",", "", regex=False),
                                           errors="coerce").astype(float)
                rep.thousands[c] = int(comma.sum())
            left = v.isna() & nonempty
        if left.any():
            t = pd.Series(arr[left.to_numpy()], index=left[left].index, dtype=object)
            up = t.str.upper()
            is_bool = up.isin(list(_BOOL_TOKENS))
            if is_bool.sum() >= 0.8 * n_nonempty:
                v.loc[is_bool[is_bool].index] = up[is_bool].map(_BOOL_TOKENS).astype(float)
                rep.bool_columns.append(c)
            else:
                g = t.str.extract(_RE_NUM_SUFFIX)
                has = g[0].notna()
                if has.sum() >= 0.5 * n_nonempty and g.loc[has, 1].nunique() == 1:
                    v.loc[has[has].index] = pd.to_numeric(g.loc[has, 0], errors="coerce")
                    suffix = str(g.loc[has, 1].iloc[0])
                    rep.suffix_units[c] = suffix
                    if c not in rep.units or not rep.units[c]:
                        rep.units[c] = normalize_unit(suffix) or ""
                        rep.units_raw.setdefault(c, suffix)
            left = v.isna() & nonempty
        n_num = n_nonempty - int(left.sum())
        if chunk:
            rep.chunk_stats.setdefault("counts", {})[c] = (n_num, n_nonempty)
        elif n_num < max(3, 0.05 * n_nonempty):
            rep.text_columns[c] = [str(x) for x in
                                   pd.Series(arr[nonempty_np]).value_counts().head(3).index]
            continue
        if left.any():
            rep.status_counts[c] = {str(k): int(n) for k, n in
                                    pd.Series(arr[left.to_numpy()]).value_counts().head(12).items()}
        numeric[c] = v

    df = pd.DataFrame(numeric, index=frame.index)
    if times is None:
        return Table(df=df.reset_index(drop=True), report=rep)
    df.index = pd.DatetimeIndex(times.to_numpy(), name=tcol)
    df = df.sort_index(kind="stable")
    return Table(df=_regularize(_dedupe(df, rep), rep, regularize, tcol), report=rep)


def _dedupe(df: pd.DataFrame, rep: IngestReport) -> pd.DataFrame:
    """같은 시각 행을 평균으로 합친다 (정렬된 표)."""
    if df.index.has_duplicates:
        dup = df.index.duplicated(keep=False)
        g = df[dup].groupby(level=0)
        spread = (g.max() - g.min()).abs()
        scale = df.abs().median().replace(0, 1.0)
        conflict = (spread > 1e-9 * scale + 1e-12).any(axis=1)
        n_before = len(df)
        df = df.groupby(level=0).mean()
        rep.n_duplicate_rows = n_before - len(df)
        rep.n_conflicting_duplicates = int(conflict.sum())
    return df


def _regularize(df: pd.DataFrame, rep: IngestReport, regularize: bool, tcol: str | None) -> pd.DataFrame:
    """기록 주기를 정하고 규칙 격자에 놓는다 (빠진 시각은 빈 행)."""
    if len(df) > 2:
        d = np.diff(df.index.to_numpy().astype("datetime64[ms]").astype("int64")) / 1000.0
        d = d[d > 0]
        if rep.long and df.shape[1] > 1:
            # 이름마다 기록 시각이 다를 수 있다 (예외 기반 저장). 합친 시각의 간격은 실제 주기보다
            # 훨씬 짧게 나오므로 이름별 주기의 중앙값을 쓴다.
            per = []
            for c in df.columns:
                ix = df.index[df[c].notna().to_numpy()]
                dc = np.diff(ix.to_numpy().astype("datetime64[ms]").astype("int64")) / 1000.0
                dc = dc[dc > 0]
                if len(dc):
                    per.append(float(np.median(dc)))
            if per:
                d = np.array(per)
        if len(d):
            step = float(np.median(d))
            # 현장 기록 주기는 정해진 값 중 하나다. 시각이 1~2초씩 어긋나면 중앙값이
            # 302초처럼 나오는데, 그대로 쓰면 격자 전체가 틀어진다.
            nice = min(_NICE_INTERVALS, key=lambda v: abs(v - step) / v)
            if abs(nice - step) / nice < 0.05:
                step = float(nice)
            elif step >= 1:
                step = float(round(step))
            rep.interval_s = step
    if regularize and np.isfinite(rep.interval_s) and rep.interval_s > 0 and len(df) > 2:
        step = pd.Timedelta(seconds=rep.interval_s)
        origin = df.index[0].normalize()
        off = ((df.index - origin) % step) / step
        off = np.minimum(off, 1 - off)
        if (off > 1e-6).mean() > 0.01:
            snapped = origin + ((df.index - origin) / step).round().astype(int) * step
            rep.n_snapped = int((off > 1e-6).sum())
            df.index = pd.DatetimeIndex(snapped, name=tcol)
            if df.index.has_duplicates:
                df = df.groupby(level=0).mean()
        full = pd.date_range(df.index[0], df.index[-1], freq=step, name=tcol)
        rep.n_missing_grid = len(full) - len(df.index.intersection(full))
        df = df.reindex(full)
        rep.gaps = _gaps(df, rep.interval_s)
    rep.t_start = str(df.index[0]) if len(df) else None
    rep.t_end = str(df.index[-1]) if len(df) else None
    return df


def _gaps(df: pd.DataFrame, interval_s: float, min_steps: int = 3, top: int = 10) -> list[dict]:
    """모든 컬럼이 비어 있는 연속 구간 (행 자체가 없었거나 전 태그가 상태 문자열)."""
    empty = df.isna().all(axis=1).to_numpy()
    out = []
    i, n = 0, len(empty)
    while i < n:
        if not empty[i]:
            i += 1
            continue
        j = i
        while j < n and empty[j]:
            j += 1
        if j - i >= min_steps:
            out.append({"start": str(df.index[i]), "end": str(df.index[j - 1]),
                        "steps": int(j - i), "hours": (j - i) * interval_s / 3600.0})
        i = j
    out.sort(key=lambda g: -g["steps"])
    return out[:top]
