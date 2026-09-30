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


def test_launcher_bat_is_crlf_utf8_and_sets_utf8_mode():
    raw = (ROOT / "run_pforecast.bat").read_bytes()
    assert b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n"), "배치 파일은 CRLF 만"
    text = raw.decode("utf-8")
    lines = text.splitlines()
    assert lines[0] == "@echo off" and lines[1] == "chcp 65001 >nul"
    assert 'set "PYTHONUTF8=1"' in text
    assert 'set "PAUSE=rem"' not in text              # rem 은 한 줄 블록의 뒷부분까지 주석으로 만든다
    assert 'WORK=%~dp0."' in text                     # 끝의 \ 가 따옴표를 삼키지 않게
