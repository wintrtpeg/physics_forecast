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
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

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


def parse_times(values: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """여러 표기가 섞인 시각 문자열을 해석한다. (결과, 표기별 행 수)를 돌려준다."""
    s = values.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    stats: dict[str, int] = {}
    empty = s.isna() | (s == "")
    todo = ~empty

    def take(mask, parsed, label):
        parsed = pd.to_datetime(parsed, errors="coerce")
        # 일반 파서는 '50.4' 같은 숫자도 날짜로 우긴다 (서기 161년). 상식적인 연도만 받는다.
        ok = parsed.notna() & parsed.dt.year.between(1980, 2100)
        if ok.any():
            out.loc[ok[ok].index] = parsed[ok].astype("datetime64[ns]")
            stats[label] = stats.get(label, 0) + int(ok.sum())
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
    stats["해석 불가"] = int(todo.sum())
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
        if self.n_repeated_header:
            out.append(f"파일 중간의 반복 헤더 {self.n_repeated_header}행을 뺐습니다 "
                       "(여러 파일을 이어 붙인 흔적).")
        if self.n_bad_time:
            ex = ", ".join(repr(e) for e in self.bad_time_examples[:3])
            out.append(f"시각을 읽을 수 없는 행 {self.n_bad_time}개를 뺐습니다 (예: {ex}).")
        if self.n_out_of_order:
            out.append(f"시간이 거꾸로 가는 곳이 {self.n_out_of_order}곳 있어 정렬했습니다.")
        if self.n_duplicate_rows:
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

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
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


def _parse_rows(text: str, delimiter: str) -> list[list[str]]:
    """문자열 그대로의 2차원 표. pandas C 토크나이저가 빠르고, 행마다 칸 수가 다르면
    (여러 출처를 이어 붙인 파일) 표준 csv 모듈로 떨어진다."""
    try:
        raw = pd.read_csv(io.StringIO(text), sep=delimiter, header=None, dtype=str,
                          na_filter=False, skip_blank_lines=False, engine="c")
        rows = raw.to_numpy(dtype=object).tolist()
    except (pd.errors.ParserError, ValueError):
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    return [[c.strip() if isinstance(c, str) else "" for c in r] for r in rows]


def _is_num(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def read_table(path: str | Path, time_column: str | None = None, *,
               encoding: str | None = None, regularize: bool = True) -> Table:
    """현장 CSV 를 읽어 정리한다. 무엇을 고쳤는지는 ``Table.report`` 에 남는다."""
    raw = Path(path).read_bytes()
    text, enc = _decode(raw, encoding)
    tab = read_text(text, time_column=time_column, regularize=regularize)
    tab.report.path = str(path)
    tab.report.encoding = enc
    return tab


def read_text(text: str, time_column: str | None = None, *, regularize: bool = True,
              delimiter: str | None = None) -> Table:
    rep = IngestReport()
    rep.delimiter = delimiter or _sniff_delimiter(text)
    rows = _parse_rows(text, rep.delimiter)
    rep.n_lines = len(rows)

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
    width = max(len(r) for r in rows[h:])
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

    body = [r + [""] * (width - len(r)) for r in rows[h + 1:]]

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
    body = body[len(meta):]
    header_set = [header] + meta

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
        times, rep.time_formats = parse_times(pd.Series([r[tj] for r in body], dtype=object))
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
    frame = pd.DataFrame(body, columns=names, dtype=object)

    # ---- 숫자 변환 ------------------------------------------------------------
    numeric: dict[str, pd.Series] = {}
    for c in data_cols:
        arr = frame[c].to_numpy(dtype=object)
        nonempty_np = arr != ""
        n_nonempty = int(nonempty_np.sum())
        if n_nonempty == 0:
            rep.dropped_columns.append(c)
            continue
        nonempty = pd.Series(nonempty_np, index=frame.index)
        v = pd.Series(pd.to_numeric(np.where(nonempty_np, arr, None), errors="coerce"),
                      index=frame.index, dtype=float)
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
        if n_num < max(3, 0.05 * n_nonempty):
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

    # ---- 중복 시각 --------------------------------------------------------------
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

    # ---- 격자 ------------------------------------------------------------------
    if len(df) > 2:
        d = np.diff(df.index.to_numpy().astype("datetime64[ms]").astype("int64")) / 1000.0
        d = d[d > 0]
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
    return Table(df=df, report=rep)


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
