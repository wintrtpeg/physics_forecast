"""윈도우에서만 터지는 것들 — 리눅스에서도 재현할 수 있는 만큼은 여기서 막는다.

실제 윈도우 점검은 `.github/workflows/windows.yml` (run_pforecast.bat 을 그대로 실행하고 API·화면을
끝까지 돌린다).
"""

import json
import shutil
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

from pforecast.app.server import Api, Workspace, make_handler
from pforecast.fileio import rel_posix, safe_stem, write_locked_ok

ROOT = Path(__file__).resolve().parents[1]


def test_reserved_windows_names_are_renamed():
    assert safe_stem("con") == "con_" and safe_stem("NUL.backup") == "NUL.backup_"
    assert safe_stem("COM3") == "COM3_" and safe_stem("console") == "console"
    assert safe_stem('a:b/c*d?"e') == "a_b_c_d_e" and safe_stem("...") == "file"


def test_locked_file_is_written_under_a_new_name_and_reported(tmp_path):
    """엑셀이 연 CSV 는 잠겨 있다 (PermissionError). 작업을 실패시키지 않고 새 이름 + 알림."""
    target = tmp_path / "out" / "x.csv"

    def writer(p: Path):
        if p == target:
            raise PermissionError(13, "Permission denied")
        p.write_text("ok", encoding="utf-8")

    path, note = write_locked_ok(target, writer)
    assert path != target and path.parent == target.parent and path.read_text(encoding="utf-8") == "ok"
    assert path.name.startswith("x_") and "엑셀" in note
    path2, note2 = write_locked_ok(tmp_path / "y.csv", lambda p: p.write_text("ok", encoding="utf-8"))
    assert path2.name == "y.csv" and note2 is None


def test_paths_sent_to_the_screen_always_use_forward_slashes(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    assert rel_posix(tmp_path / "a" / "b" / "c.csv", tmp_path) == "a/b/c.csv"


def test_ml_run_survives_a_locked_predictions_file(tmp_path, monkeypatch):
    from pforecast.easy import config_from_body, run_easy
    shutil.copy(ROOT / "examples/nox_stack/data/plant_5min.csv", tmp_path / "plant.csv")
    body = {"csv": "plant.csv", "target": "F2_UT_STK01_NOX_DRY", "features": ["F2_FAB_DRY_UTIL", "F2_UT_AMB_TEMP"],
            "train": ["2025-03-01", "2025-03-20"], "test": ["2025-03-22", "2025-03-30"], "time_column": "timestamp"}
    real = pd.DataFrame.to_csv

    def locked(self, path=None, *a, **k):
        if str(path).endswith("plant.predictions.csv"):
            raise PermissionError(13, "Permission denied")
        return real(self, path, *a, **k)

    monkeypatch.setattr(pd.DataFrame, "to_csv", locked)
    res = run_easy(config_from_body(body, lambda r: (tmp_path / r).resolve()), tmp_path)
    assert res["files"]["predictions"].startswith("out/easy/plant.predictions_")
    assert (tmp_path / res["files"]["predictions"]).exists()
    assert any("엑셀" in n for n in res["notes"])


@pytest.fixture()
def server(tmp_path):
    shutil.copytree(ROOT / "examples/chiller_plant", tmp_path / "examples/chiller_plant")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Api(Workspace(tmp_path))))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", tmp_path
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def _post(url, body, raw=None):
    req = urllib.request.Request(url, data=raw if raw is not None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_static_files_have_fixed_content_types(server):
    """윈도우 mimetypes 는 레지스트리를 읽는다 — .js 가 text/plain 인 PC 에서도 화면이 떠야 한다."""
    base, _ = server
    assert _get(base + "/static/app.js")[1].startswith("text/javascript")
    assert _get(base + "/static/style.css")[1].startswith("text/css")
    assert _get(base + "/")[1].startswith("text/html")


def test_examples_are_never_overwritten_and_uploads_are_not_rewritten(server):
    base, ws = server
    code, r = _post(base + "/api/builder/load", {"path": "examples/chiller_plant/system.yaml"})
    assert code == 200 and r["path"] == "examples/chiller_plant/system.yaml"      # '/' 경로
    code, r2 = _post(base + "/api/builder/save", {"model": r["model"], "path": "examples/chiller_plant/system.yaml"})
    assert code == 400 and "예제" in r2["error"]
    data = "t,x\n2025-01-01 00:00,1\n".encode()
    code, up = _post(base + "/api/upload?name=con.csv", None, raw=data)
    assert code == 200 and up["path"] == "uploads/con_.csv"
    f = ws / "uploads" / "con_.csv"
    before = f.stat().st_mtime_ns
    code, up2 = _post(base + "/api/upload?name=con.csv", None, raw=data)   # 같은 내용 → 다시 쓰지 않음
    assert up2["path"] == up["path"] and f.stat().st_mtime_ns == before


def test_upload_is_streamed_to_disk_and_limit_is_enforced(server, monkeypatch):
    """수 GB 업로드: 본문을 메모리에 모으지 않고 조각으로 임시 파일에 쓴 뒤 이름을 바꾼다."""
    from pforecast.app import server as srv
    base, ws = server
    rows = "\n".join(f"2025-01-01 00:{i % 60:02d}:00,{i}" for i in range(400_000))
    data = ("t,x\n" + rows + "\n").encode()                    # 약 9 MB — 조각(8 MB) 여러 개
    code, up = _post(base + "/api/upload?name=big.csv", None, raw=data)
    assert code == 200 and up["path"] == "uploads/big.csv"
    assert (ws / "uploads/big.csv").read_bytes() == data
    assert not list((ws / "uploads").glob(".*.part"))        # 임시 파일이 남지 않는다
    code, up2 = _post(base + "/api/upload?name=big.csv", None, raw=data[:-2] + b"8\n")   # 크기는 같고 내용만 다름
    assert code == 200 and up2["path"] == "uploads/big_1.csv"
    monkeypatch.setattr(srv, "MAX_UPLOAD", 1000)
    code, r = _post(base + "/api/upload?name=big.csv", None, raw=data)
    assert code == 413 and "넘습니다" in r["error"]
    code, _ = _post(base + "/api/upload?name=ok.csv", None, raw=b"t,x\n2025-01-01,1\n")
    assert code == 200                                        # 거절한 뒤에도 연결이 살아 있다


def test_table_prepare_runs_as_a_job_then_is_ready(server):
    base, ws = server
    (ws / "uploads").mkdir(exist_ok=True)
    shutil.copy(ROOT / "examples/nox_stack/data/plant_5min.csv", ws / "uploads/p.csv")
    code, r = _post(base + "/api/table/prepare", {"csv": "uploads/p.csv"})
    assert code == 200 and "job" in r
    for _ in range(100):
        j = json.loads(_get(base + f"/api/job?id={r['job']['id']}")[2])
        if j["status"] != "running":
            break
        threading.Event().wait(0.1)
    assert j["status"] == "done" and j["result"]["rows"] == 8640
    code, r2 = _post(base + "/api/table/prepare", {"csv": "uploads/p.csv"})
    assert r2 == {"ready": True}


def test_file_format_can_be_set_from_the_screen(server):
    """'파일 형식' 카드: 자동이 못 알아본 긴 형식을 골라 펼친다 → 저장(.layout.json) → 다시 읽으면 펼쳐져 있다."""
    base, ws = server
    (ws / "uploads").mkdir(exist_ok=True)
    rows = ["time,ch,reading"] + [f"2025-03-01 00:{5 * i:02d}:00,{k},{100 * k + i}" for i in range(10) for k in (1, 2, 3)]
    (ws / "uploads/x.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    code, prof = _post(base + "/api/profile", {"csv": "uploads/x.csv"})
    assert code == 200 and [c["name"] for c in prof["columns"]] == ["ch", "reading"]
    assert prof["ingest"]["columns_raw"] == ["time", "ch", "reading"] and prof["ingest"]["long"] is None
    code, r = _post(base + "/api/table/layout",
                    {"csv": "uploads/x.csv", "layout": {"format": "long", "keys": ["ch"], "values": ["reading"]}})
    assert code == 200 and (ws / "uploads/x.csv.layout.json").exists()
    code, prof = _post(base + "/api/profile", {"csv": "uploads/x.csv"})       # 캐시가 아니라 다시 읽는다
    assert [c["name"] for c in prof["columns"]] == ["1", "2", "3"] and prof["rows"] == 10
    assert prof["ingest"]["long"]["manual"] and prof["ingest"]["long"]["roles"]["keys"] == ["ch"]
    code, r = _post(base + "/api/table/layout",
                    {"csv": "uploads/x.csv", "layout": {"format": "long", "keys": ["없는컬럼"], "values": ["reading"]}})
    assert code == 400 and "없는컬럼" in r["error"]
    ds = json.loads(_get(base + "/api/workspace")[2])
    assert [d["path"] for d in ds["datasets"]].count("uploads/x.csv") == 1 and ds["version"]
    code, r = _post(base + "/api/table/layout", {"csv": "uploads/x.csv", "layout": None})
    assert code == 200 and not (ws / "uploads/x.csv.layout.json").exists()
    code, prof = _post(base + "/api/profile", {"csv": "uploads/x.csv"})
    assert [c["name"] for c in prof["columns"]] == ["ch", "reading"]


def test_launcher_bat_is_plain_ascii_crlf_and_delegates_to_python():
    """cmd.exe 는 chcp 65001 뒤 한글(UTF-8) 배치 파일을 외부 프로그램이 끝난 다음부터 잘못 읽고
    조용히 멈춘다 (윈도우 점검에서 '가상환경을 만드는 중...' 뒤 아무 말 없이 끝났다). 그래서 배치 파일은
    ASCII 만, 일과 한글 안내는 scripts/launch.py 가 한다."""
    raw = (ROOT / "run_pforecast.bat").read_bytes()
    assert all(b < 128 for b in raw), "배치 파일에 ASCII 가 아닌 글자"
    assert b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n"), "배치 파일은 CRLF 만"
    text = raw.decode("ascii")
    assert not any(ln.strip().lower().startswith("chcp") for ln in text.splitlines())   # 주석 속 낱말은 괜찮다
    assert 'set "PYTHONUTF8=1"' in text and "scripts\\launch.py" in text
    assert "if not defined PF_NO_PAUSE pause" in text


def test_launcher_script_runs_on_old_pythons_long_enough_to_explain():
    """3.10 미만이면 문법 오류 대신 안내가 나와야 한다 — 버전 검사가 다른 import 보다 먼저."""
    src = (ROOT / "scripts/launch.py").read_text(encoding="utf-8")
    head = src.split("import hashlib")[0]
    assert "sys.version_info < (3, 10)" in head
