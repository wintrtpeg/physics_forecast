"""현장 CSV 수집기: 한글 엑셀·DCS 추출본이 실제로 가진 문제들."""

import pandas as pd
import pytest

from pforecast.core.units import normalize_unit
from pforecast.data.ingest import parse_times, read_layout, read_table, read_text, write_layout


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


# ---- 큰 파일: 조각으로 나눠 읽기 ---------------------------------------------------------
# 통째로 읽으면 칸마다 파이썬 문자열이 생겨 5GB 파일에 메모리가 수십 GB 든다. 조각으로 읽은 결과는
# 통째로 읽은 결과와 **표도 정리 내역도** 같아야 한다 (작은 조각 크기로 강제해서 비교).

def _messy_big_csv(n: int = 400) -> str:
    """현장 엑셀 저장본 모양 (3행 헤더·오후 표기·상태 문자열·천 단위 콤마·순서 뒤바뀜·겹친 구간·빈 줄)."""
    head = "일시,NOX,FLOW,PUMP,DEAD\n,굴뚝 NOx,배출유량,펌프,죽은 입력\n,mg/Sm³,Sm³/h,,\n"
    t0 = pd.Timestamp("2025-01-01")
    rows = []
    for i in range(n):
        t = t0 + pd.Timedelta(minutes=5 * i)
        ts = f"{t:%Y-%m-%d} {'오후' if t.hour >= 12 else '오전'} {((t.hour - 1) % 12) + 1}:{t.minute:02d}"
        nox = "Bad" if i % 37 == 5 else f"{50 + (i % 11) * 0.5:.1f}"
        rows.append(f'{ts},{nox},"{11000 + i:,.1f}",{"ON" if i % 7 else "OFF"},Bad')
    rows[100:110] = rows[150:160] + rows[100:110]        # 순서 뒤바뀜 + 겹친 구간
    rows.insert(200, "")
    rows.insert(250, "일시,NOX,FLOW,PUMP,DEAD")          # 이어 붙이며 딸려온 헤더
    return head + "\n".join(rows) + "\n"


def _assert_same_read(p, skip=(), **kw):
    a = read_table(p)
    b = read_table(p, cache=False, **kw)
    pd.testing.assert_frame_equal(a.df, b.df, check_freq=False)
    assert a.report.lines() == b.report.lines()
    for k in ("n_data_rows", "n_repeated_header", "n_blank_rows", "n_out_of_order", "n_duplicate_rows",
              "n_conflicting_duplicates", "interval_s", "units", "descriptions", "status_counts",
              "text_columns", "bool_columns", "thousands"):
        if k not in skip:
            assert getattr(a.report, k) == getattr(b.report, k), k
    return a, b


def test_chunked_read_equals_whole_read_on_a_messy_field_export(tmp_path):
    p = tmp_path / "field.csv"
    p.write_bytes(_messy_big_csv().encode("cp949"))
    a, b = _assert_same_read(p, chunk_bytes=2000)
    assert a.report.header_rows == 3 and a.report.n_out_of_order >= 1 and a.report.n_duplicate_rows == 10
    assert "DEAD" in a.report.text_columns and "DEAD" not in b.df.columns


@pytest.mark.parametrize("order", ["tag", "time"])
def test_chunked_read_of_long_format_equals_whole_read(tmp_path, order):
    """이름 순으로 정렬된 긴 파일은 한 조각에 이름이 하나뿐이다 — 긴 형식 판정을 파일 곳곳의 표본으로
    미리 해야 한다. 시각 순이면 한 시각의 태그들이 조각 경계에서 갈린다 — 한 행으로 합쳐야 한다."""
    w = _wide_example()
    long = w.rename_axis("time").reset_index().melt(id_vars="time", var_name="Tag", value_name="Value")
    if order == "time":
        long = long.sort_values("time", kind="stable")
    long["Unit"] = long["Tag"].map({"A_FLOW": "m3/h", "B_TEMP": "℃", "C_UTIL": "%"})
    p = tmp_path / "long.csv"
    p.write_bytes(long.to_csv(index=False, lineterminator="\n").encode("utf-8"))   # 줄 끝은 아래 시험에서
    a, b = _assert_same_read(p, chunk_bytes=700)
    assert list(b.df.columns) == list(w.columns) and b.report.long["n_names"] == 3
    assert b.units == {"A_FLOW": "m3/h", "B_TEMP": "degC", "C_UTIL": "%"}


@pytest.mark.parametrize("nl", ["\n", "\r\n", "\r\r\n", "\r"])
def test_chunked_read_with_any_line_endings(tmp_path, nl):
    """'\\r\\r\\n' 은 윈도우에서 줄 끝이 두 번 바뀐 파일이다 (to_csv 가 \\r\\n 을 주고 텍스트 모드 쓰기가
    또 바꿈). 헤더를 split('\\n') 으로 세면 pandas 와 줄 수가 어긋나 첫 데이터 줄이 조각마다 복사됐다."""
    w = _wide_example()
    long = w.rename_axis("time").reset_index().melt(id_vars="time", var_name="Tag", value_name="Value")
    p = tmp_path / "long.csv"
    p.write_bytes(long.to_csv(index=False, lineterminator=nl).encode("utf-8"))
    # '\r\r\n' 의 빈 줄 수만은 다를 수 있다 (pandas 가 글 끝의 빈 줄을 조각마다 하나씩 빼고 센다 — 화면에 안 나감)
    a, b = _assert_same_read(p, skip=("n_blank_rows",), chunk_bytes=700)
    assert len(b.df) == 48 and b.report.long["n_rows"] == 144 and not b.report.long.get("n_dup")


def test_chunked_read_with_worker_processes(tmp_path):
    p = tmp_path / "field.csv"
    p.write_bytes(_messy_big_csv(1500).encode("cp949"))
    _assert_same_read(p, chunk_bytes=8000, workers=2)


def test_big_file_result_is_cached_next_to_the_csv(tmp_path):
    p = tmp_path / "field.csv"
    p.write_bytes(_messy_big_csv().encode("cp949"))
    a = read_table(p, chunk_bytes=2000)
    caches = list((tmp_path / ".pforecast_cache").glob("field.csv.*.npz"))
    assert len(caches) == 1
    b = read_table(p, chunk_bytes=2000)                        # 캐시에서
    pd.testing.assert_frame_equal(a.df, b.df)
    assert a.report.lines() == b.report.lines()
    p.write_bytes(_messy_big_csv(300).encode("cp949"))         # CSV 가 바뀌면 다시 읽고 옛 캐시는 지운다
    c = read_table(p, chunk_bytes=2000)
    assert len(c.df) < len(a.df)
    assert len(list((tmp_path / ".pforecast_cache").glob("field.csv.*.npz"))) == 1


def test_wide_file_with_a_text_column_is_not_taken_for_long():
    rows = ["ts,a,b,mode"]
    for i in range(60):
        t = pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=5 * i)
        rows.append(f"{t},{i},{2 * i},{'AUTO' if (i // 3) % 2 else 'MAN'}")
    tab = read_text("\n".join(rows))
    assert not tab.report.long and list(tab.df.columns) == ["a", "b"]
    assert "mode" in tab.report.text_columns
    assert not tab.report.long_hint                  # 넓은 파일에 '긴 형식일 수 있다' 고 묻지 않는다


# ---- 현장 긴 형식 변형 — 예전 판정이 놓쳐 '쓸 수 있는 컬럼 1개' 가 됐던 것들 ---------------------------
TAGS = ["CH1_KW", "CH1_TCHW", "CT_FAN_HZ", "AHU_SA_T"]


def _base_long(n=30):
    idx = pd.date_range("2025-03-01", periods=n, freq="5min")
    rows = [(t, tag, round(100.0 * k + i, 3)) for i, t in enumerate(idx) for k, tag in enumerate(TAGS)]
    return pd.DataFrame(rows, columns=["t", "tag", "v"])


def _variant(kind: str) -> str:
    b = _base_long()
    ts = b.t.dt.strftime("%Y-%m-%d %H:%M:%S")
    if kind == "historian_many_columns":        # 부가 컬럼이 많은 Historian 덤프 (품질 코드·태그 번호 …)
        df = pd.DataFrame({"DateTime": ts, "TagName": b.tag, "Value": b.v, "vValue": b.v.astype(str),
                           "Quality": 0, "QualityDetail": 192, "OPCQuality": 192,
                           "wwTagKey": b.tag.map({t: 1000 + i for i, t in enumerate(TAGS)}),
                           "wwRowCount": 100, "wwResolution": 300000, "wwRetrievalMode": "Cyclic",
                           "wwTimeZone": "Korea Standard Time", "wwVersion": "Latest", "wwCycleCount": 30})
    elif kind == "date_and_time_columns":       # 일자·시간이 두 컬럼
        df = pd.DataFrame({"일자": b.t.dt.strftime("%Y-%m-%d"), "시간": b.t.dt.strftime("%H:%M:%S"),
                           "태그명": b.tag, "값": b.v})
    elif kind == "date_and_ampm_time":
        df = pd.DataFrame({"날짜": b.t.dt.strftime("%Y-%m-%d"),
                           "시각": b.t.map(lambda t: f"{'오후' if t.hour >= 12 else '오전'} {(t.hour - 1) % 12 + 1}:{t.minute:02d}"),
                           "Tag": b.tag, "Value": b.v})
    elif kind == "mostly_on_off":               # 디지털 태그가 대부분 — 값 컬럼 대부분이 글자
        v = [("ON" if (i // 2) % 2 else "OFF") if tag != TAGS[-1] else str(x)
             for i, (tag, x) in enumerate(zip(b.tag, b.v))]
        df = pd.DataFrame({"Time": ts, "Tag": b.tag, "Value": v})
    elif kind == "value_with_unit":
        df = pd.DataFrame({"Time": ts, "Tag": b.tag, "Value": b.v.map(lambda x: f"{x} kW")})
    elif kind == "overlapping_exports":         # 같은 구간을 두 번 받아 이어 붙임 (60%)
        o = pd.concat([b, b.iloc[: int(len(b) * 0.6)]]).sort_values("t", kind="stable")
        df = pd.DataFrame({"Time": o.t.dt.strftime("%Y-%m-%d %H:%M:%S"), "TagName": o.tag, "Value": o.v})
    elif kind == "row_number_first":
        df = pd.DataFrame({"No": range(1, len(b) + 1), "TagName": b.tag, "DateTime": ts, "Value": b.v})
    else:
        raise KeyError(kind)
    return df.to_csv(index=False, lineterminator="\n")


@pytest.mark.parametrize("kind", ["historian_many_columns", "date_and_time_columns", "date_and_ampm_time",
                                  "mostly_on_off", "value_with_unit", "overlapping_exports", "row_number_first"])
def test_field_long_variants_are_spread(kind, tmp_path):
    p = tmp_path / "long.csv"
    p.write_bytes(_variant(kind).encode("utf-8"))
    tab = read_table(p)
    assert list(tab.df.columns) == TAGS, tab.report.lines()
    assert len(tab.df) == 30 and tab.df.index[0] == pd.Timestamp("2025-03-01 00:00")
    assert tab.df.notna().all().all()
    if kind != "mostly_on_off":
        assert tab.df["AHU_SA_T"].iloc[7] == pytest.approx(307.0)
    b = read_table(p, chunk_bytes=1500, cache=False)          # 큰 파일 경로도 같아야 한다
    pd.testing.assert_frame_equal(tab.df, b.df, check_freq=False)
    assert tab.report.lines() == b.report.lines()


def test_date_and_time_columns_are_combined_for_wide_files_too():
    rows = ["일자,시간,a,b"] + [f"2025-03-01,{h:02d}:{m:02d},{h},{m}" for h in range(3) for m in (0, 30)]
    tab = read_text("\n".join(rows))
    assert tab.report.time_column2 == "시간" and list(tab.df.columns) == ["a", "b"]
    assert tab.df.index[3] == pd.Timestamp("2025-03-01 01:30") and len(tab.df) == 6
    assert any("두 컬럼을 합쳐" in ln for ln in tab.report.lines())


def test_mixed_sampling_rates_use_a_grid_most_names_fill():
    """1분 태그와 1시간 태그가 섞인 덤프 — 중앙값(1분) 격자면 1시간 태그가 98% 비어 '쓸 수 없는 컬럼' 이 된다.
    대부분이 채워지는 격자로 맞추고 촘촘한 태그는 칸마다 평균한다 (빈칸을 지어내 채우지 않는다)."""
    t0 = pd.Timestamp("2025-03-01")
    rows = [(t0 + pd.Timedelta(minutes=i), tag, float(i)) for tag in ("F1", "F2") for i in range(240)]
    rows += [(t0 + pd.Timedelta(hours=i), tag, 10.0 * i) for tag in ("S1", "S2", "S3") for i in range(4)]
    df = pd.DataFrame(rows, columns=["Time", "Tag", "Value"]).sort_values("Time", kind="stable")
    tab = read_text(df.to_csv(index=False, lineterminator="\n"))
    assert tab.report.interval_s == 3600 and tab.df.notna().mean().min() > 0.7
    assert 59 <= tab.df["F1"].iloc[1] <= 61                               # 01:00 앞뒤 30분의 평균
    assert any("기록 주기가 다릅니다" in ln for ln in tab.report.lines())


def test_historian_extra_columns_are_reported_not_spread():
    tab = read_text(_variant("historian_many_columns"))
    L = tab.report.long
    assert L["key"] == "TagName" and L["values"] == ["Value"]        # 태그 번호(wwTagKey)가 아니라 이름
    assert {"vValue", "wwTagKey", "QualityDetail"} <= set(L["ignored"])


def test_layout_file_spreads_what_auto_detection_refuses(tmp_path):
    """이름이 숫자 번호이고 컬럼 이름도 평범하면 자동으로는 못 알아본다 — '파일 형식' 에서 고르면 펼친다."""
    b = _base_long()
    p = tmp_path / "x.csv"
    p.write_text(pd.DataFrame({"time": b.t, "ch": b.tag.map({t: i + 1 for i, t in enumerate(TAGS)}),
                               "reading": b.v}).to_csv(index=False, lineterminator="\n"), encoding="utf-8")
    assert not read_table(p).report.long
    write_layout(p, {"format": "long", "keys": ["ch"], "values": ["reading"]})
    assert read_layout(p)["keys"] == ["ch"]
    tab = read_table(p)
    assert list(tab.df.columns) == ["1", "2", "3", "4"] and tab.report.long["manual"]
    assert any("지정한 대로" in ln for ln in tab.report.lines())
    big = read_table(p, chunk_bytes=900, cache=False)                   # 큰 파일 경로도 지정을 따른다
    pd.testing.assert_frame_equal(tab.df, big.df, check_freq=False)
    write_layout(p, None)
    assert not read_table(p).report.long


def test_layout_can_force_wide_and_hint_offers_the_choice(tmp_path):
    """자동으로 못 펼쳤지만 긴 형식일지 모르는 파일은 이유와 함께 고르라고 알린다. 넓게 못 박으면 묻지 않는다."""
    idx = pd.date_range("2025-03-01", periods=2, freq="5min")              # 이름마다 두 줄뿐 → 자동은 거절
    rows = [(t, f"T{k}", float(k)) for t in idx for k in range(30)]
    p = tmp_path / "x.csv"
    p.write_text(pd.DataFrame(rows, columns=["Time", "Tag", "Value"]).to_csv(index=False, lineterminator="\n"),
                 encoding="utf-8")
    r = read_table(p).report
    assert not r.long and r.long_hint["keys"] == ["Tag"] and r.long_why
    assert any("긴 형식일 수 있습니다" in ln for ln in r.lines())
    write_layout(p, {"format": "wide"})
    r2 = read_table(p).report
    assert not r2.long and not r2.long_hint and r2.layout == {"format": "wide"}
    with pytest.raises(ValueError):
        write_layout(p, {"format": "long", "keys": ["Tag"], "values": []})
