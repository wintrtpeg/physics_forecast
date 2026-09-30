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


@pytest.mark.parametrize("values, expect, moved", [
    # 한 가지 시간대: 떼고 적힌 시각 그대로
    (["2025-01-01T00:00:00+09:00", "2025-01-01T00:05:00+09:00"], ["2025-01-01 00:00", "2025-01-01 00:05"], 0),
    (["2025-01-01T00:00:00Z", "2025-01-01T00:05:00.000Z"], ["2025-01-01 00:00", "2025-01-01 00:05"], 0),
    (["2025-01-01 00:00:00+0900", "2025-01-01 00:05:00 +09:00"], ["2025-01-01 00:00", "2025-01-01 00:05"], 0),
    # 섞이면 가장 많은 시간대 기준으로 옮긴다 (서머타임 전환, Z 와 +09:00 혼용)
    (["2025-03-30T01:55:00+01:00", "2025-03-30T03:00:00+02:00", "2025-03-30T03:05:00+02:00"],
     ["2025-03-30 02:55", "2025-03-30 03:00", "2025-03-30 03:05"], 1),
    (["2024-12-31T15:00:00Z", "2025-01-01T00:05:00+09:00", "2025-01-01T00:10:00+09:00"],
     ["2025-01-01 00:00", "2025-01-01 00:05", "2025-01-01 00:10"], 1),
])
def test_timezone_suffixes_become_local_wall_clock(values, expect, moved):
    """시간대가 붙은 시각(+09:00, Z)은 예전엔 TypeError 로 업로드가 통째로 실패했다."""
    out, stats = parse_times(pd.Series(values))
    assert [str(t)[:16] for t in out] == expect
    assert out.dt.tz is None
    assert any(k.startswith("시간대 표기") for k in stats)
    assert sum(v for k, v in stats.items() if k.startswith("다른 시간대")) == moved


def test_timezone_csv_reads_and_reports(tmp_path):
    idx = pd.date_range("2025-01-01", periods=12, freq="5min")
    text = "time,x\n" + "\n".join(f"{t:%Y-%m-%dT%H:%M:%S}+09:00,{i}" for i, t in enumerate(idx))
    p = tmp_path / "tz.csv"
    p.write_text(text, encoding="utf-8")
    tab = read_table(p)
    assert tab.df.index.tz is None and tab.df.index[0] == pd.Timestamp("2025-01-01 00:00")
    assert len(tab.df) == 12 and any("+09:00" in line for line in tab.report.lines())


# ---- 긴 형식 (한 줄에 시각·이름·값 하나씩) --------------------------------------------
# 예전엔 이름 컬럼이 글자라 빠지고, 같은 시각의 여러 태그 값이 '중복 시각'으로 평균되어
# 컬럼 하나로 뭉개졌다.

def _wide_example() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=48, freq="5min")
    return pd.DataFrame({"A_FLOW": range(48), "B_TEMP": [20.0 + 0.1 * i for i in range(48)],
                         "C_UTIL": [0.5 + 0.01 * i for i in range(48)]}, index=idx)


@pytest.mark.parametrize("order", ["tag", "time"])
def test_long_format_is_spread_to_the_same_columns_as_wide(order):
    w = _wide_example()
    long = w.rename_axis("time").reset_index().melt(id_vars="time", var_name="TagName", value_name="Value")
    if order == "time":
        long = long.sort_values("time", kind="stable")
    tab = read_text(long.to_csv(index=False))
    assert list(tab.df.columns) == list(w.columns)
    pd.testing.assert_frame_equal(tab.df, w, check_names=False, check_freq=False, check_dtype=False,
                                  check_index_type=False)
    assert tab.report.long["n_names"] == 3 and tab.report.n_out_of_order == 0
    assert any("긴 형식" in line for line in tab.report.lines())


def test_long_korean_export_with_unit_desc_quality_columns(tmp_path):
    rows = ["일시,태그명,태그설명,측정값,단위,품질"]
    tags = [("CH1_KW", "냉동기1 전력", "kW"), ("CH1_TCHW", "냉수 공급온도", "℃")]
    for i in range(24):
        t = pd.Timestamp("2025-01-01 12:00") + pd.Timedelta(minutes=5 * i)
        ts = f"{t:%Y-%m-%d} 오후 {((t.hour - 1) % 12) + 1}:{t.minute:02d}"
        for k, (tag, desc, u) in enumerate(tags):
            q = "Bad" if tag == "CH1_KW" and i in (3, 4) else "Good"
            rows.append(f'{ts},{tag},{desc},"{1000 + i + 10 * k:,.1f}",{u},{q}')
    rows.append(rows[3])                                    # 겹친 추출
    rows.append(rows[1].replace('"1,000.0"', '"999.0"'))     # 같은 시각·이름인데 값이 다름
    rows.append(f"{ts},,없음,1,kW,Good")                     # 이름이 빈 줄
    p = tmp_path / "long.csv"
    p.write_bytes(("\n".join(rows) + "\n").encode("cp949"))
    tab = read_table(p)
    r = tab.report
    assert r.encoding == "cp949" and list(tab.df.columns) == ["CH1_KW", "CH1_TCHW"]
    assert tab.units == {"CH1_KW": "kW", "CH1_TCHW": "degC"}
    assert r.descriptions == {"CH1_KW": "냉동기1 전력", "CH1_TCHW": "냉수 공급온도"}
    assert len(tab.df) == 24 and tab.df.index[0] == pd.Timestamp("2025-01-01 12:00")
    assert tab.df["CH1_KW"].isna().sum() == 2 and r.status_counts["CH1_KW"] == {"Bad": 2}
    assert tab.df["CH1_TCHW"].iloc[5] == 1015.0                      # 천 단위 콤마
    assert tab.df["CH1_KW"].iloc[0] == pytest.approx((1000.0 + 999.0) / 2)
    L = r.long
    assert (L["n_dup"], L["n_dup_conflict"], L["n_blank_key"], L["n_bad_status"]) == (2, 1, 1, 2)
    text = " ".join(r.lines())
    assert "품질 컬럼 '품질'" in text and "이름이 빈 줄 1개" in text and "값이 달라 평균" in text


def test_long_with_name_split_over_two_columns():
    rows = ["time,equipment,item,value"]
    for i in range(12):
        for e in ("CH1", "CH2"):
            for it in ("kW", "flow"):
                rows.append(f"2025-01-01T00:{i:02d}:00,{e},{it},{i + 100 * (e == 'CH2') + 10 * (it == 'flow')}")
    tab = read_text("\n".join(rows))
    assert list(tab.df.columns) == ["CH1.kW", "CH1.flow", "CH2.kW", "CH2.flow"]
    assert tab.df.iloc[3].tolist() == [3, 13, 103, 113]


def test_long_with_several_value_columns_per_equipment():
    rows = ["일시,설비,온도,유량"]
    for i in range(12):
        for e in ("P1", "P2"):
            rows.append(f"2025-01-01 00:{i:02d},{e},{20 + i + (e == 'P2')},{5 + i}")
    tab = read_text("\n".join(rows))
    assert list(tab.df.columns) == ["P1.온도", "P1.유량", "P2.온도", "P2.유량"]
    assert tab.df["P2.온도"].iloc[0] == 21


def test_long_with_per_tag_timestamps_uses_each_tags_own_interval():
    """예외 기반 저장: 이름마다 기록 시각이 조금씩 다르다. 합친 시각 간격(17초)으로 격자를 만들면
    거의 빈 표가 된다 — 이름별 주기(5분)로 잡아야 한다."""
    rows = ["Timestamp,Tag,Value"]
    for i in range(30):
        for k, tag in enumerate(("A", "B", "C")):
            t = pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=5 * i, seconds=17 * k + 3)
            rows.append(f"{t},{tag},{10 * k + i}")
    tab = read_text("\n".join(rows))
    assert tab.report.interval_s == 300 and len(tab.df) == 30
    assert tab.df.notna().all().all() and tab.df["C"].iloc[4] == 24


def test_wide_file_with_a_text_column_is_not_taken_for_long():
    rows = ["ts,a,b,mode"]
    for i in range(60):
        t = pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=5 * i)
        rows.append(f"{t},{i},{2 * i},{'AUTO' if (i // 3) % 2 else 'MAN'}")
    tab = read_text("\n".join(rows))
    assert not tab.report.long and list(tab.df.columns) == ["a", "b"]
    assert "mode" in tab.report.text_columns
