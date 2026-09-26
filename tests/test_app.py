"""로컬 웹 앱 API.

서버는 표준 라이브러리만 쓴다. 여기서는 실제로 띄우고 HTTP 로 두드린다.
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from pforecast.app.server import Api, Workspace, make_handler, model_meta, solve_with

ROOT = Path(".").resolve()


@pytest.fixture(scope="module")
def server():
    ws = Workspace(ROOT)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Api(ws)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base
    httpd.shutdown()
    httpd.server_close()


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def get(base, path):
    return json.loads(urllib.request.urlopen(base + path).read())


def test_static_files_are_served(server):
    for path in ("/", "/static/app.js", "/static/style.css"):
        with urllib.request.urlopen(server + path) as r:
            assert r.status == 200
            assert len(r.read()) > 500


def test_workspace_lists_datasets_and_models(server):
    ws = get(server, "/api/workspace")
    assert any(d["name"] == "plant_5min.csv" for d in ws["datasets"])
    paths = {m["path"] for m in ws["models"]}
    assert "examples/nox_stack/model.py" in paths
    assert "examples/chiller_plant/system.yaml" in paths


def test_profile_returns_units_and_stats(server):
    p = post(server, "/api/profile", {"csv": "examples/nox_stack/data/plant_5min.csv"})
    assert p["rows"] == 8640
    assert p["time_column"] == "timestamp"
    assert p["interval_s"] == 300
    by = {c["name"]: c for c in p["columns"]}
    assert by["F2_UT_STK01_TEMP"]["unit"] == "degC"
    assert by["F2_UT_STK01_NOX_DRY"]["unit"] == "mg/Nm3"


def test_model_meta_uses_declared_drivers(server):
    m = post(server, "/api/model/meta", {"model": "examples/nox_stack/model.py"})
    assert m["ok"] and m["n_eqs"] == m["n_vars"]
    assert m["declared_drivers"]
    keys = [d["key"] for d in m["drivers"]]
    assert keys == ["util", "n_tools", "T_amb", "n_ratio", "L"]
    assert {d["key"]: d["label"] for d in m["drivers"]}["util"] == "가동율"
    assert m["limits"]["STK.C_dry"]["max"] == 100.0


def test_yaml_model_also_declares_drivers(server):
    m = post(server, "/api/model/meta", {"model": "examples/nox_stack/system.yaml"})
    assert [d["key"] for d in m["drivers"]] == ["util", "n_tools", "T_amb", "n_ratio", "L"]


def test_solve_respects_the_declared_driver_unit(server):
    """가동율을 100 으로 주면 util=1.0 이어야 한다 (util=100 이 아니라)."""
    r = post(server, "/api/model/solve", {
        "model": "examples/nox_stack/model.py",
        "overrides": {"util": {"value": 100, "unit": "%", "mode": "set"}}})
    assert r["converged"]
    assert 70 < r["outputs"]["STK.C_dry"] < 80


def test_scale_mode_multiplies(server):
    base = post(server, "/api/model/solve", {"model": "examples/nox_stack/model.py",
                                             "overrides": {}})
    more = post(server, "/api/model/solve", {
        "model": "examples/nox_stack/model.py",
        "overrides": {"n_tools": {"value": 1.3, "mode": "scale"}}})
    assert more["outputs"]["STK.C_dry"] > base["outputs"]["STK.C_dry"]
    assert len(more["applied"]) == 4


def test_sweep_returns_a_monotonic_curve(server):
    r = post(server, "/api/model/sweep", {
        "model": "examples/nox_stack/model.py", "key": "util",
        "values": [40, 60, 80, 100], "overrides": {"util": {"unit": "%", "mode": "set"}}})
    ys = [row["outputs"]["STK.C_dry"] for row in r["rows"]]
    assert all(row["converged"] for row in r["rows"])
    assert ys == sorted(ys)


def test_chiller_plant_model_loads_and_solves(server):
    m = post(server, "/api/model/meta", {"model": "examples/chiller_plant/system.yaml"})
    assert m["ok"] and m["n_eqs"] == m["n_vars"] == 25
    assert [d["key"] for d in m["drivers"]] == ["Q_load", "T_wb", "T_chws", "app_nom"]
    r = post(server, "/api/model/solve", {"model": "examples/chiller_plant/system.yaml",
                                          "overrides": {}})
    assert r["converged"]
    assert 5.0 < r["outputs"]["CHILLER.COP"] < 6.0


def test_bad_paths_are_rejected(server):
    with pytest.raises(urllib.error.HTTPError):
        post(server, "/api/profile", {"csv": "../../etc/passwd"})
    with pytest.raises(urllib.error.HTTPError):
        get(server, "/api/nope")


def test_missing_input_gives_a_clear_error(server):
    try:
        post(server, "/api/profile", {})
        pytest.fail("에러가 나야 한다")
    except urllib.error.HTTPError as e:
        assert e.code == 400
        assert "부족" in json.loads(e.read())["error"]
