"""데이터 소스 어댑터: CSV 와 SQL.

보안상 CSV 반출이 기본 경로이고, 필요할 때만 SQLAlchemy 로 직접 뽑는다.
어느 쪽이든 최종 산출물은 동일하다: 5분 평균으로 정렬된, 모델 변수명과 SI 단위를
쓰는 ``DataFrame`` 두 장 (inputs / observations).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .ingest import IngestReport, read_table
from .quality import QualityReport, apply, assess
from .tagmap import TagEntry, TagMap


@dataclass
class DataReport:
    """파일을 읽으며 고친 것(ingest)과 값에서 뺀 것(quality)."""

    ingest: IngestReport | None = None
    quality: QualityReport | None = None

    def lines(self) -> list[str]:
        out = list(self.ingest.lines()) if self.ingest else []
        if self.quality is not None:
            out += self.quality.lines()
        return out


def _apply_entries(df: pd.DataFrame, entries: Iterable[TagEntry], strict: bool) -> pd.DataFrame:
    out = {}
    missing = []
    for e in entries:
        if e.tag not in df.columns:
            missing.append(e.tag)
            continue
        out[e.primary] = e.to_si_series(pd.to_numeric(df[e.tag], errors="coerce"))
    if missing:
        msg = "CSV 에 없는 태그: " + ", ".join(missing)
        if strict:
            raise KeyError(msg + f"\n  파일의 컬럼: {list(df.columns)[:20]}")
        print(f"[경고] {msg}")
    return pd.DataFrame(out, index=df.index)


def load_frames(
    raw: pd.DataFrame,
    tagmap: TagMap,
    strict: bool = True,
    resample: str | None = "__default__",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """원본 와이드 테이블을 (inputs, observations) 로 변환한다."""
    df = raw.copy()
    tcol = tagmap.timestamp_column
    if not isinstance(df.index, pd.DatetimeIndex) and tcol in df.columns:
        df[tcol] = pd.to_datetime(df[tcol], format=tagmap.timestamp_format, errors="coerce")
        df = df.dropna(subset=[tcol]).set_index(tcol).sort_index()
    rule = tagmap.resample if resample == "__default__" else resample
    if rule and isinstance(df.index, pd.DatetimeIndex):
        num = df.select_dtypes(include=[np.number])
        df = num.resample(rule).mean()
    inputs = _apply_entries(df, tagmap.inputs, strict)
    obs = _apply_entries(df, tagmap.observations, strict=False)
    return inputs, obs


def read_csv(path: str | Path, tagmap: TagMap, **kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """5분 평균 CSV 를 읽어 모델 좌표계로 변환한다."""
    inputs, obs, _ = read_csv_report(path, tagmap, **kw)
    return inputs, obs


def read_csv_report(path: str | Path, tagmap: TagMap, clean: bool = True,
                    **kw) -> tuple[pd.DataFrame, pd.DataFrame, DataReport]:
    """현장 CSV 를 읽고(인코딩·헤더·날짜 표기·상태 문자열 ...) 값 이상을 걸러
    모델 좌표계로 옮긴다. 무엇을 고치고 뺐는지 ``DataReport`` 로 같이 돌려준다.

    ``clean=False`` 면 값 이상(교정 창, 고착, 스파이크)을 그대로 둔다. 파일 형식은
    어느 쪽이든 고친다 — 그건 선택의 문제가 아니다.
    """
    tab = read_table(path, time_column=tagmap.timestamp_column)
    df = tab.df
    quality = None
    if clean:
        cols = [t for t in tagmap.all_tags if t in df.columns]
        quality = assess(df, cols)
        df = apply(df, quality)
    inputs, obs = load_frames(df, tagmap, **kw)
    return inputs, obs, DataReport(ingest=tab.report, quality=quality)


def read_sql(query: str, url: str, tagmap: TagMap, params: dict | None = None, **kw):
    """사내 DB 직접 조회. SQLAlchemy 는 선택 의존성이다.

    쿼리는 태그가 컬럼으로 펼쳐진 와이드 형태를 반환해야 한다. 롱 포맷이면
    ``pivot_long`` 을 거쳐 넘긴다.
    """
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "SQL 조회에는 SQLAlchemy 가 필요합니다: pip install 'pforecast[sql]'"
        ) from exc
    engine = create_engine(url)
    with engine.connect() as conn:
        raw = pd.read_sql(text(query), conn, params=params or {})
    return load_frames(raw, tagmap, **kw)


def pivot_long(df: pd.DataFrame, time_col: str, tag_col: str, value_col: str) -> pd.DataFrame:
    """(시각, 태그, 값) 롱 포맷을 와이드로 편다. Historian 덤프가 대개 이 형태다."""
    wide = df.pivot_table(index=time_col, columns=tag_col, values=value_col, aggfunc="mean")
    wide.columns.name = None
    return wide.reset_index()


def align(inputs: pd.DataFrame, obs: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """입력과 관측을 같은 시각으로 맞춘다."""
    merged = inputs.join(obs, how="inner", rsuffix="__obs")
    return merged.dropna() if dropna else merged


def stratified_sample(df: pd.DataFrame, by: str, n: int = 200, bins: int = 20,
                      seed: int = 0) -> pd.DataFrame:
    """운전영역이 고르게 들어가도록 층화 추출.

    캘리브레이션에 수만 행을 다 쓸 필요는 없다. 중요한 것은 운전 구간의 **폭**이지
    행 수가 아니다. 오히려 정상운전 구간만 잔뜩 뽑으면 파라미터가 식별되지 않는다.
    """
    if by not in df.columns or len(df) <= n:
        return df
    rng = np.random.default_rng(seed)
    cats = pd.cut(df[by], bins=bins, duplicates="drop")
    per = max(1, n // max(cats.nunique(), 1))
    picks = []
    for _, grp in df.groupby(cats, observed=True):
        take = min(len(grp), per)
        picks.append(grp.iloc[rng.choice(len(grp), take, replace=False)])
    out = pd.concat(picks).sort_index()
    if len(out) > n:
        out = out.iloc[rng.choice(len(out), n, replace=False)].sort_index()
    return out
