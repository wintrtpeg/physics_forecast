"""간편 예측 (업로드 → 변수 → 기간 → 실행) API.

누수를 막는 규칙은 화면이 먼저 막지만 서버가 한 번 더 거절해야 한다. 화면을 거치지 않고
API 를 부르는 경우도 있기 때문이다. 물리 보정이 들어간 실행은 수 분이 걸려 여기서는 ML 만
돌린다 (물리 경로는 브라우저 점검과 ``docs/app.md`` 의 수치로 확인한다).
"""

import json
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pforecast.app.server import Api, Workspace, make_handler
from pforecast.easy import period_errors

EXAMPLE = Path("examples/nox_stack/data/plant_5min.csv").resolve()


@pytest.fixture()
def ws_server(tmp_path):
    """임시 작업 폴더에 띄운다 — 업로드·결과 파일이 저장소에 남지 않게."""
    shutil.copy(EXAMPLE, tmp_path / "plant.csv")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Api(Workspace(tmp_path))))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", tmp_path
    httpd.shutdown()
    httpd.server_close()


def call(base, path, body=None, raw=None, status=200):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            assert r.status == status
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        assert e.code == status, e.read()
        return json.loads(e.read())


def run_job(base, body, timeout=120):
    job = call(base, "/api/easy/run", body)["job"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = call(base, f"/api/job?id={job['id']}")
        if j["status"] != "running":
            assert j["status"] == "done", j.get("error")
            return j["result"]
        time.sleep(0.3)
    raise TimeoutError


BODY = {"csv": "plant.csv", "target": "F2_UT_STK01_NOX_DRY",
        "features": ["F2_FAB_DRY_UTIL", "F2_FAB_CVD_UTIL", "F2_FAB_WET_UTIL", "F2_FAB_IMP_UTIL",
                     "F2_UT_AMB_TEMP"],
        "train": ["2025-03-01", "2025-03-20"], "test": ["2025-03-22", "2025-03-30"],
        "embargo_days": 1, "target_unit": "mg/Nm3"}


# ── 기간 규칙 ─────────────────────────────────────────────────────────────

def test_period_rules_are_inclusive_dates_with_an_embargo():
    # 학습 끝 3/20 은 그날 24시까지, 예측 3/22 00시 시작 → 간격 정확히 1일
    assert period_errors(["2025-03-01", "2025-03-20"], ["2025-03-22", "2025-03-30"], 1) == []
    # 다음 날 바로 시작하면 간격 0일
    errs = period_errors(["2025-03-01", "2025-03-20"], ["2025-03-21", "2025-03-30"], 1)
    assert len(errs) == 1 and "간격" in errs[0]
    # 예측이 학습과 겹치면 누수
    errs = period_errors(["2025-03-01", "2025-03-20"], ["2025-03-10", "2025-03-30"], 1)
    assert any("누수" in e for e in errs)
    # 더 긴 간격을 요구하면 그만큼 막는다
    assert period_errors(["2025-03-01", "2025-03-20"], ["2025-03-22", "2025-03-30"], 3)


# ── 업로드 ───────────────────────────────────────────────────────────────

def test_upload_stays_inside_the_workspace_and_keeps_korean_names(ws_server):
    base, root = ws_server
    data = EXAMPLE.read_bytes()[:20000]
    name = urllib.parse.quote("../../밖으로 나가기 테스트.csv")
    r = call(base, f"/api/upload?name={name}", raw=data)
    assert r["path"] == "uploads/밖으로 나가기 테스트.csv"
    assert (root / r["path"]).read_bytes() == data
    # 같은 내용은 같은 파일, 다른 내용은 새 이름 (덮어쓰지 않는다)
    assert call(base, f"/api/upload?name={name}", raw=data)["path"] == r["path"]
    r2 = call(base, f"/api/upload?name={name}", raw=data[:10000])
    assert r2["path"] == "uploads/밖으로 나가기 테스트_1.csv"
    assert (root / r["path"]).read_bytes() == data
    # CSV 가 아니면 거절
    bad = call(base, "/api/upload?name=report.xlsx", raw=b"PK\x03\x04", status=400)
    assert "CSV" in bad["error"]
    assert "빈 파일" in call(base, "/api/upload?name=a.csv", raw=b"", status=400)["error"]
    assert not list(root.parent.glob("밖으로*"))


def test_workspace_files_reject_traversal_but_serve_korean_names(ws_server):
    base, root = ws_server
    (root / "out").mkdir()
    (root / "out" / "내_결과.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with urllib.request.urlopen(base + "/out/" + urllib.parse.quote("내_결과.csv")) as r:
        assert r.read().startswith(b"a,b")
    for bad in ("/out/%2e%2e/%2e%2e/etc/passwd", "/out/../../etc/passwd"):
        req = urllib.request.Request(base + bad)
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code in (400, 404)


# ── 기간 검사 ─────────────────────────────────────────────────────────────

def test_check_reports_future_and_extrapolation(ws_server):
    base, _ = ws_server
    r = call(base, "/api/easy/check", BODY)
    assert r["ok"] and r["is_future"]
    assert r["embargo_days"] >= 1
    assert r["n_train"] > 1000 and r["n_test"] > 500
    assert 0 <= r["extrapolation"] <= 1
    assert {g["column"] for g in r["ranges"]} == set(BODY["features"])


def test_leaky_periods_are_refused_by_check_and_by_run(ws_server):
    base, _ = ws_server
    leaky = {**BODY, "test": ["2025-03-15", "2025-03-30"]}
    r = call(base, "/api/easy/check", leaky)
    assert not r["ok"] and any("누수" in e for e in r["errors"])
    # 화면을 거치지 않고 실행을 불러도 거절한다
    err = call(base, "/api/easy/run", leaky, status=400)
    assert "누수" in err["error"]
    no_gap = {**BODY, "test": ["2025-03-21", "2025-03-30"]}
    assert "간격" in call(base, "/api/easy/run", no_gap, status=400)["error"]


# ── 실행 (ML 만) ─────────────────────────────────────────────────────────

def test_ml_only_run_scores_the_future_and_writes_predictions(ws_server):
    base, root = ws_server
    res = run_job(base, {**BODY, "limit": 80})
    assert res["mode"] == "ml"
    assert res["split"]["is_future"] and res["split"]["embargo_days"] >= 1
    names = [m["name"] for m in res["models"]]
    assert "ML 부스팅" in names          # 트리 앙상블은 반드시 들어간다
    for m in res["models"]:
        assert np.isfinite(m["metrics"]["test"]["rmse"])
        assert "exceed" in m["metrics"]
    # 학습 표본과 예측 구간이 시간순으로 이어진다
    reg = res["series"]["region"]
    assert reg[0] == "train" and reg[-1] == "test"
    assert reg.index("test") > len(reg) - reg[::-1].index("train") - 1
    pred = pd.read_csv(root / res["files"]["predictions"], index_col=0, encoding="utf-8-sig")
    assert pd.Timestamp(pred.index.min()) >= pd.Timestamp("2025-03-22")
    assert "실측" in pred.columns and "ML 부스팅" in pred.columns


def test_an_input_frozen_in_training_is_reported(tmp_path):
    """증설 전 장비 대수처럼 학습 동안 고정된 입력은 ML 이 못 본다 — 반드시 알린다."""
    from pforecast.easy import EasyConfig, run_easy

    t = pd.date_range("2025-01-01", periods=24 * 12 * 40, freq="5min")
    rng = np.random.default_rng(0)
    load = 50 + 20 * np.sin(np.arange(len(t)) / 300) + rng.normal(0, 1, len(t))
    n = np.where(t < pd.Timestamp("2025-02-01"), 3, 4)
    y = 0.8 * load * n / 3 + rng.normal(0, 0.5, len(t))
    pd.DataFrame({"timestamp": t, "LOAD": load, "N_UNITS": n, "Y": y}).to_csv(tmp_path / "d.csv", index=False)
    cfg = EasyConfig(csv=str(tmp_path / "d.csv"), target="Y", features=["LOAD", "N_UNITS"],
                     train=["2025-01-01", "2025-01-25"], test=["2025-02-01", "2025-02-09"])
    res = run_easy(cfg, tmp_path)
    assert any("N_UNITS" in n and "변하지 않" in n for n in res["notes"])
