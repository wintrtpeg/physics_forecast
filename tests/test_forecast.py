"""미래 예측 (계획 → 예측), 프로젝트 파일, 모델 만들기 API.

물리 보정이 들어간 미래 예측은 수 분이 걸려 여기서는 ML 경로와 계획 처리만 본다.
물리 경로의 수치는 ``docs/forecast.md`` 에 실제로 돌린 값으로 적었다.
"""

import json
import shutil
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pforecast.app.server import Api, Workspace, make_handler
from pforecast.easy import config_from_body
from pforecast.forecast import check_plan, forecast_easy, make_plan, read_plan, save_plan
from pforecast.project import list_projects, load_project, save_project

EXAMPLE = Path("examples/nox_stack/data/plant_5min.csv").resolve()
FEATS = ["F2_FAB_DRY_UTIL", "F2_FAB_CVD_UTIL", "F2_FAB_WET_UTIL", "F2_FAB_IMP_UTIL", "F2_UT_AMB_TEMP"]
BODY = {"csv": "plant.csv", "target": "F2_UT_STK01_NOX_DRY", "features": FEATS,
        "target_unit": "mg/Nm3", "time_column": "timestamp"}


@pytest.fixture()
def ws(tmp_path):
    shutil.copy(EXAMPLE, tmp_path / "plant.csv")
    return tmp_path


def _cfg(ws, **kw):
    return config_from_body({**BODY, **kw}, lambda r: (ws / r).resolve())


# ── 계획 ────────────────────────────────────────────────────────────────

def test_make_plan_repeats_recent_pattern_and_applies_adjustments(ws):
    base = make_plan(ws / "plant.csv", "timestamp", FEATS, base_days=2, horizon_days=5, step="1h")
    plan = make_plan(ws / "plant.csv", "timestamp", FEATS, base_days=2, horizon_days=5, step="1h",
                     adjust=[{"column": "F2_FAB_DRY_UTIL", "mode": "scale", "value": 1.1},
                             {"column": "F2_UT_AMB_TEMP", "mode": "add", "value": 3},
                             {"column": "F2_FAB_IMP_UTIL", "mode": "set", "value": 80}])
    assert len(plan) == 5 * 24
    # 데이터 끝 바로 다음 시각부터, 1시간 간격
    assert plan.index[0] == pd.Timestamp("2025-03-31 00:00")
    assert (plan.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()
    # 최근 2일(48시간) 패턴이 반복된다
    np.testing.assert_allclose(base["F2_FAB_CVD_UTIL"].iloc[:48].to_numpy(), base["F2_FAB_CVD_UTIL"].iloc[48:96].to_numpy())
    np.testing.assert_allclose(plan["F2_FAB_DRY_UTIL"], base["F2_FAB_DRY_UTIL"] * 1.1)
    np.testing.assert_allclose(plan["F2_UT_AMB_TEMP"], base["F2_UT_AMB_TEMP"] + 3)
    assert (plan["F2_FAB_IMP_UTIL"] == 80).all()
    with pytest.raises(ValueError, match="조정 방식"):
        make_plan(ws / "plant.csv", "timestamp", FEATS, adjust=[{"column": "F2_FAB_DRY_UTIL", "mode": "mul", "value": 2}])


def test_plan_roundtrips_through_the_field_csv_reader(ws):
    """엑셀에서 고쳐 다시 올린 계획도 현장 CSV 와 같은 수집기로 읽는다."""
    plan = make_plan(ws / "plant.csv", "timestamp", FEATS, base_days=1, horizon_days=2, step="1h")
    p, note = save_plan(plan, ws / "plans" / "p.csv")
    assert note is None
    back, _log = read_plan(p)
    assert list(back.columns) == FEATS
    np.testing.assert_allclose(back.to_numpy(), plan.to_numpy(), rtol=1e-5)


def test_check_plan_reports_missing_columns_outside_rows_and_frozen_inputs():
    idx = pd.date_range("2025-01-01", periods=100, freq="1h")
    hist = pd.DataFrame({"a": np.linspace(0, 10, 100), "n": 5.0}, index=idx)
    fut = pd.date_range("2025-01-06", periods=10, freq="1h")
    plan = pd.DataFrame({"a": np.r_[np.full(5, 5.0), np.full(5, 20.0)], "n": 6.0}, index=fut)
    chk = check_plan(plan, ["a", "n"], hist)
    assert chk["ok"] and chk["n_rows"] == 10
    assert chk["outside_pct"] == pytest.approx(100.0)            # n 이 전부 밖
    by = {r["column"]: r for r in chk["ranges"]}
    assert by["a"]["outside_pct"] == pytest.approx(50.0)
    assert by["n"]["frozen"] and any("고정" in w for w in chk["warnings"])
    bad = check_plan(plan[["a"]], ["a", "n"], hist)
    assert not bad["ok"] and "n" in bad["errors"][0]
    past = check_plan(pd.DataFrame({"a": [1.0], "n": [5.0]}, index=[idx[3]]), ["a", "n"], hist)
    assert past["past_rows"] == 1 and any("이전 시점" in w for w in past["warnings"])


# ── 예측 (ML 경로) ──────────────────────────────────────────────────────

def test_forecast_needs_no_measured_target_and_flags_extrapolation(ws):
    cfg = _cfg(ws, limit=60)
    plan = make_plan(ws / "plant.csv", "timestamp", FEATS, base_days=3, horizon_days=4, step="1h",
                     adjust=[{"column": "F2_FAB_DRY_UTIL", "mode": "scale", "value": 1.5}])
    assert BODY["target"] not in plan.columns
    res = forecast_easy(cfg, plan, root=ws)
    assert res["mode"] == "ml" and res["plan"]["n_rows"] == len(plan)
    assert {s["name"] for s in res["summary"]} == {"ML 다항(2차)", "ML 부스팅"}
    assert all(np.isfinite(s["mean"]) and "exceed_hours" in s for s in res["summary"])
    assert res["outside_pct"] > 50 and any("학습 때 없던 운전조건" in n for n in res["notes"])
    out = pd.read_csv(ws / res["files"]["forecast"], index_col=0, encoding="utf-8-sig")
    assert len(out) == len(plan) and "학습범위밖" in out.columns
    # 최근 실측 꼬리는 과거에만 있다
    assert max(res["history_tail"]["t"]) < min(res["series"]["t"])


def test_forecast_does_not_see_data_after_the_history_window(ws):
    """과거를 기간으로 제한하면 그 뒤의 값을 망가뜨려도 예측이 한 비트도 안 바뀐다."""
    cfg = _cfg(ws)
    plan = make_plan(ws / "plant.csv", "timestamp", FEATS, base_days=2, horizon_days=2, step="1h")
    a = forecast_easy(cfg, plan, root=ws, history=("2025-03-01", "2025-03-20"))
    df = pd.read_csv(ws / "plant.csv")
    late = pd.to_datetime(df["timestamp"]) >= "2025-03-21"
    df.loc[late, BODY["target"]] = 999.0
    df.loc[late, "F2_FAB_DRY_UTIL"] *= 3
    df.to_csv(ws / "plant.csv", index=False)
    b = forecast_easy(cfg, plan, root=ws, history=("2025-03-01", "2025-03-20"))
    assert a["history"]["end"] < "2025-03-21"
    assert a["series"]["models"] == b["series"]["models"]


def test_forecast_rejects_a_plan_without_the_inputs(ws):
    plan = make_plan(ws / "plant.csv", "timestamp", FEATS, base_days=1, horizon_days=1)
    with pytest.raises(ValueError, match="계획에 없는 입력"):
        forecast_easy(_cfg(ws), plan.drop(columns=["F2_UT_AMB_TEMP"]), root=ws)


# ── 프로젝트 ────────────────────────────────────────────────────────────

def test_project_save_load_list_keeps_settings_and_results(tmp_path):
    run = {**BODY, "train": ["2025-03-01", "2025-03-20"], "test": ["2025-03-22", "2025-03-30"]}
    p = save_project(tmp_path, "NOx 굴뚝/1호기", run, ui={"step": "ez-result", "limit": "60"},
                     results={"models": [1, 2]})
    assert p.parent.name == "projects" and "/" not in p.name
    doc = load_project(p)
    assert doc["run"] == run and doc["ui"]["limit"] == "60" and doc["results"] == {"models": [1, 2]}
    # 결과 없이 다시 저장하면 이전 결과를 지우지 않는다
    save_project(tmp_path, "NOx 굴뚝/1호기", run, forecast_results={"summary": []})
    doc = load_project(p)
    assert doc["results"] == {"models": [1, 2]} and doc["forecast_results"] == {"summary": []}
    (tmp_path / "projects" / "broken.yaml").write_text("pforecast_project: [", encoding="utf-8")
    items = list_projects(tmp_path)
    assert items[0]["name"] == "NOx 굴뚝/1호기" and items[0]["has_results"]
    assert any(it.get("error") for it in items)
    (tmp_path / "x.yaml").write_text("name: a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="프로젝트 파일이 아닙니다"):
        load_project(tmp_path / "x.yaml")


# ── API ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def server(ws):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Api(Workspace(ws))))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", ws
    httpd.shutdown()
    httpd.server_close()


def call(base, path, body=None, status=200):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            assert r.status == status
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        assert e.code == status, e.read()
        return json.loads(e.read())


def test_forecast_api_make_check_run_and_plans_stay_out_of_the_data_list(server):
    base, ws = server
    r = call(base, "/api/forecast/make", {**BODY, "base_days": 2, "horizon_days": 3, "step": "1h",
                                          "adjust": [{"column": "F2_FAB_DRY_UTIL", "mode": "scale", "value": 1.2}]})
    assert r["path"].startswith("plans/") and r["check"]["ok"] and r["check"]["n_rows"] == 72
    assert set(r["preview"]["columns"]) == set(FEATS)
    # 계획 CSV 는 학습 데이터 목록에 섞이지 않는다
    assert all(not d["path"].startswith("plans/") for d in call(base, "/api/workspace")["datasets"])
    assert any(p["path"] == r["path"] for p in call(base, "/api/forecast/plans")["plans"])
    chk = call(base, "/api/forecast/check", {**BODY, "plan": r["path"]})
    assert chk["check"]["ok"]
    job = call(base, "/api/forecast/run", {**BODY, "plan": r["path"], "band": False})["job"]
    t0 = time.time()
    while time.time() - t0 < 120:
        j = call(base, f"/api/job?id={job['id']}")
        if j["status"] != "running":
            break
        time.sleep(0.3)
    assert j["status"] == "done", j.get("error")
    assert j["result"]["plan"]["n_rows"] == 72
    call(base, "/api/forecast/check", {**BODY, "plan": "../outside.csv"}, status=400)


def test_project_api_roundtrip_and_missing_files(server):
    base, ws = server
    run = {**BODY, "model": "gone/model.py"}
    saved = call(base, "/api/project/save", {"name": "p1", "run": run, "ui": {"csv": "plant.csv"},
                                             "results": {"x": 1}})
    assert saved["path"] == "projects/p1.yaml"
    assert call(base, "/api/project/list")["projects"][0]["name"] == "p1"
    doc = call(base, "/api/project/load", {"path": saved["path"]})
    assert doc["results"] == {"x": 1} and doc["missing"] == ["gone/model.py"]


def test_builder_api_checks_each_stage_and_saves_inside_the_workspace(server):
    base, ws = server
    pal = call(base, "/api/builder/palette")["palette"]
    assert any(p.get("ref") == "lib:chiller_carnot" for p in pal)
    src = Path("examples/chiller_plant/system.yaml").read_text(encoding="utf-8")
    ok = call(base, "/api/builder/check", {"yaml": src})
    assert ok["ok"] and ok["counts"]["equations"] == ok["counts"]["unknowns"]
    broken = dict(ok["model"], connections=ok["model"]["connections"][1:])
    r = call(base, "/api/builder/check", {"model": broken})
    assert not r["ok"] and r["stage"] == "connect" and r["dangling"]
    r = call(base, "/api/builder/check", {"yaml": "components: ["})
    assert r["stage"] == "yaml"
    saved = call(base, "/api/builder/save", {"model": ok["model"]})
    assert saved["path"].startswith("models/") and (ws / saved["path"]).exists()
    assert any(m["path"] == saved["path"] for m in call(base, "/api/workspace")["models"])
    call(base, "/api/builder/save", {"model": ok["model"], "path": "../evil.yaml"}, status=400)
    call(base, "/api/builder/save", {"model": ok["model"], "path": "models/x.py"}, status=400)
    call(base, "/api/builder/load", {"path": "../system.yaml"}, status=400)
