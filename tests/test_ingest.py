"""현장 CSV 수집기: 한글 엑셀·DCS 추출본이 실제로 가진 문제들."""

import pandas as pd
import pytest

from pforecast.core.units import normalize_unit
from pforecast.data.ingest import parse_times, read_table, read_text


def _messy_csv() -> str:
    head = ("일시,NOX,FLOW,PUMP,\n"
            ",굴뚝 NOx,배출유량,펌프,\n"
            ",mg/Sm³,Sm³/h,,\n")
    rows = [
        '2025-01-01 오전 12:00,50.1,"11,006.5",ON,',
        '2025-01-01 오전 12:05,Bad,"11,018.6",ON,',
        '2025-01-01 오후 12:10,51.0,"11,040.3",OFF,',     # 오후 12:10 = 12:10
        '2025-01-01 오후 1:15,I/O Timeout,"11,180.3",ON,',
    ]
    iso = [
        "2025-01-01 13:15:00,52.0,11180.3,ON,",          # 위 행과 같은 시각 (겹친 추출)
        "2025-01-01 13:20:00,53.0,11200.0,OFF,",
        "일시,NOX,FLOW,PUMP,",                             # 이어 붙이며 딸려온 헤더
        ",굴뚝 NOx,배출유량,펌프,",
        ",mg/Sm³,Sm³/h,,",
        "2025-01-01 13:10:00,49.0,11000.0,ON,",           # 순서 뒤바뀜
        ",,,,",
    ]
    return head + "\n".join(rows + iso) + "\n\n"


def test_messy_korean_excel_export_is_read_and_every_fix_is_reported(tmp_path):
    p = tmp_path / "raw.csv"
    p.write_bytes(_messy_csv().encode("cp949"))
    tab = read_table(p, time_column="timestamp", regularize=False)
    r = tab.report
    assert r.encoding == "cp949"
    assert r.time_column == "일시" and "없어" in r.time_column_note
    assert r.header_rows == 3 and r.has_unit_row and r.has_desc_row
    assert r.units["NOX"] == "mg/Sm3" and r.units["FLOW"] == "Sm3/h"
    assert r.descriptions["NOX"] == "굴뚝 NOx"
    assert r.time_formats["오전/오후 표기"] == 4 and r.time_formats["ISO"] == 3
    assert r.n_repeated_header == 3
    assert r.n_out_of_order >= 1
    assert r.n_duplicate_rows == 1
    assert r.thousands["FLOW"] == 4
    assert r.status_counts["NOX"] == {"Bad": 1, "I/O Timeout": 1}
    assert "PUMP" in r.bool_columns
    assert r.dropped_columns == ["col4"]            # 끝 콤마가 만든 빈 컬럼
    df = tab.df
    assert df.index.is_monotonic_increasing and not df.index.has_duplicates
    assert df.loc["2025-01-01 12:10", "NOX"] == 51.0
    assert df.loc["2025-01-01 12:10", "PUMP"] == 0.0
    assert df.loc["2025-01-01 00:00", "FLOW"] == pytest.approx(11006.5)
    # 겹친 시각은 평균 (13:15 는 NaN 과 52.0 -> 52.0)
    assert df.loc["2025-01-01 13:15", "NOX"] == 52.0
    lines = " ".join(r.lines())
    for word in ("CP949", "반복 헤더", "상태 문자열", "천 단위"):
        assert word in lines


def test_regular_grid_fills_missing_times_and_snaps_jitter():
    txt = "time,a\n" + "\n".join(
        f"2025-01-01 00:{m:02d}:{s:02d},{v}" for m, s, v in
        [(0, 0, 1), (4, 59, 2), (10, 1, 3), (25, 0, 4)])
    tab = read_text(txt)
    df = tab.df
    assert tab.report.interval_s == pytest.approx(300.0, rel=0.02)
    assert list(df.index.minute) == [0, 5, 10, 15, 20, 25]
    assert df["a"].isna().sum() == 2
    assert tab.report.n_snapped == 2
    assert tab.report.n_missing_grid == 2


@pytest.mark.parametrize("text,expect", [
    ("2025-03-01 오후 3:05", "2025-03-01 15:05"),
    ("2025-03-01 오전 12:05", "2025-03-01 00:05"),
    ("2025-03-01 3:05:10 PM", "2025-03-01 15:05:10"),
    ("2025년 3월 1일 13시 05분", "2025-03-01 13:05"),
    ("2025/03/01 13:05", "2025-03-01 13:05"),
    ("2025.03.01 13:05", "2025-03-01 13:05"),
    ("45717.5", "2025-03-01 12:00"),                 # 엑셀 일련번호
])
def test_parse_times_handles_field_formats(text, expect):
    out, _ = parse_times(pd.Series([text]))
    assert out.iloc[0] == pd.Timestamp(expect)


def test_numbers_are_not_mistaken_for_dates():
    out, stats = parse_times(pd.Series(["50.4", "12", "Bad"]))
    assert out.isna().all()


@pytest.mark.parametrize("raw,unit", [
    ("mg/Sm³", "mg/Sm3"), ("㎥/h", "m3/h"), ("℃", "degC"), ("(℃)", "degC"),
    ("대", "1"), ("%RH", "%"), ("KW", "kW"), ("kgf/cm2", "kgfcm2"), ("Nm3/hr", "Nm3/h"),
    ("mmH₂O", "mmAq"), ("", None), ("굴뚝 NOx", None),
])
def test_normalize_unit(raw, unit):
    assert normalize_unit(raw) == unit


def test_text_column_is_reported_not_silently_dropped():
    txt = "ts,a,mode\n" + "\n".join(
        f"2025-01-01 00:{m:02d}:00,{m},{'AUTO' if m % 2 else 'MAN'}" for m in range(0, 50, 5))
    tab = read_text(txt)
    assert "mode" not in tab.df.columns
    assert "mode" in tab.report.text_columns
    assert any("숫자가 아닌 컬럼" in line for line in tab.report.lines())
