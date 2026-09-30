"""떠 있는 앱을 API 로 끝까지 돌려 본다 — 윈도우 점검(`.github/workflows/windows.yml`)이 쓴다.

    python scripts/windows_smoke.py [--port 8765] [--root .]

사용자가 실제로 겪은 경로를 그대로 탄다: 한글 엑셀 저장본(CP949, 태그/설명/단위 3행 헤더)에 시각이
``+09:00`` 으로 적힌 CSV 업로드 → 프로파일 → 4단계 자동 연결 → 물리+ML 실행 → 미래 예측 →
프로젝트 저장·열기 → 모델 만들기. 하나라도 틀리면 0 이 아닌 값으로 끝난다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: 예제 태그의 설명·단위 (현장 CSV 의 2·3행처럼)
DESC = {
    "F2_UT_STK01_NOX_DRY": ("굴뚝 NOx(건조)", "mg/Sm3"), "F2_UT_STK01_FLOW": ("굴뚝 배출유량", "Sm3/h"),
    "F2_UT_STK01_TEMP": ("굴뚝 온도", "degC"), "F2_UT_SCR01_FAN_SP": ("유인송풍기 정압", "mmAq"),
    "F2_UT_SCR01_DP": ("스크러버 차압", "mmAq"), "F2_FAB_DRY_UTIL": ("건식식각 가동률", "%"),
    "F2_FAB_CVD_UTIL": ("증착 가동률", "%"), "F2_FAB_WET_UTIL": ("습식세정 가동률", "%"),
    "F2_FAB_IMP_UTIL": ("이온주입 가동률", "%"), "F2_UT_AMB_TEMP": ("외기온도", "degC"),
    "F2_UT_SCR01_FAN_HZ": ("유인송풍기 인버터", "Hz"),
}
Y = "F2_UT_STK01_NOX_DRY"
X = ["F2_FAB_DRY_UTIL", "F2_FAB_CVD_UTIL", "F2_FAB_WET_UTIL", "F2_FAB_IMP_UTIL", "F2_UT_AMB_TEMP", "F2_UT_SCR01_FAN_HZ"]
FAILS: list[str] = []


def ok(cond: bool, what: str) -> None:
    print(("  OK   " if cond else "  FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def site_csv() -> bytes:
    """예제 5분 데이터를 현장 엑셀 저장본 모양으로: CP949, 3행 헤더, 시각 +09:00."""
    lines = (ROOT / "examples/nox_stack/data/plant_5min.csv").read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    cols = head[1:]
    out = ["일시," + ",".join(cols),
           "," + ",".join(DESC.get(c, ("", ""))[0] for c in cols),
           "," + ",".join(DESC.get(c, ("", ""))[1] for c in cols)]
    for ln in lines[1:]:
        t, rest = ln.split(",", 1)
        out.append(t.replace(" ", "T") + "+09:00," + rest)
    return ("\r\n".join(out) + "\r\n").encode("cp949")


def site_long_csv() -> bytes:
    """같은 데이터를 Historian 덤프 모양(긴 형식)으로: 한 줄에 일시·태그명·설명·값·단위, CP949."""
    lines = (ROOT / "examples/nox_stack/data/plant_5min.csv").read_text(encoding="utf-8").splitlines()
    cols = lines[0].split(",")[1:]
    out = ["일시,태그명,태그설명,측정값,단위"]
    for ln in lines[1:]:
        t, *vals = ln.split(",")
        for col, v in zip(cols, vals):
            d, u = DESC.get(col, ("", ""))
            out.append(f"{t},{col},{d},{v},{u}")
    return ("\r\n".join(out) + "\r\n").encode("cp949")


class Client:
    def __init__(self, base: str):
        self.base = base

    def call(self, path: str, body=None, raw: bytes | None = None, expect: int = 200):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                code, payload = r.status, r.read()
        except urllib.error.HTTPError as e:
            code, payload = e.code, e.read()
        if code != expect:
            raise RuntimeError(f"{path}: HTTP {code} (기대 {expect}) {payload[:300]!r}")
        return json.loads(payload) if payload[:1] in (b"{", b"[") else payload

    def job(self, path: str, body: dict, timeout: float = 1500):
        j = self.call(path, body)["job"]
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = self.call(f"/api/job?id={j['id']}")
            if s["status"] != "running":
                if s["status"] != "done":
                    raise RuntimeError(f"{path}: {s.get('error')}")
                return s["result"]
            time.sleep(2)
        raise TimeoutError(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", default=".", help="앱의 작업 폴더 (결과 파일을 확인하려고)")
    a = ap.parse_args()
    c = Client(f"http://127.0.0.1:{a.port}")
    root = Path(a.root).resolve()
    t0 = time.time()

    print("[화면 파일]")
    for p, typ in (("/", "text/html"), ("/static/app.js", "text/javascript"), ("/static/style.css", "text/css")):
        with urllib.request.urlopen(c.base + p) as r:
            ok(r.headers.get("Content-Type", "").startswith(typ), f"{p} → {r.headers.get('Content-Type')}")

    print("[1. 업로드 — CP949, 3행 헤더, 시각 +09:00]")
    up = c.call("/api/upload?name=" + urllib.parse.quote("현장 데이터(시간대).csv"), raw=site_csv())
    ok(up["path"].startswith("uploads/") and "\\" not in up["path"], f"저장 경로 {up['path']}")
    prof = c.call("/api/profile", {"csv": up["path"]})
    ok(prof["rows"] == 8640, f"행 수 {prof['rows']}")
    ok(any("+09:00" in ln for ln in prof["ingest"]["lines"]), "정리 내역에 시간대 처리가 적힘")
    ok(str(prof["t_start"]).startswith("2025-03-01 00:00"), f"시작 시각 {prof['t_start']} (적힌 시각 그대로)")
    units = {col["name"]: col["unit"] for col in prof["columns"]}
    ok(units.get("F2_UT_SCR01_FAN_HZ") == "Hz", f"단위 행을 읽음 (FAN_HZ={units.get('F2_UT_SCR01_FAN_HZ')})")

    print("[1-2. 업로드 — 긴 형식 (한 줄에 일시·태그·값), CP949]")
    upl = c.call("/api/upload?name=" + urllib.parse.quote("긴형식 덤프.csv"), raw=site_long_csv())
    pl = c.call("/api/profile", {"csv": upl["path"]})
    ok(pl["rows"] == 8640 and [col["name"] for col in pl["columns"]] == [col["name"] for col in prof["columns"]],
       f"태그별 컬럼으로 펼침 ({pl['rows']}행 × {len(pl['columns'])}컬럼)")
    ok(any("긴 형식" in ln for ln in pl["ingest"]["lines"]), "정리 내역에 긴 형식 처리가 적힘")
    lunits = {col["name"]: col["unit"] for col in pl["columns"]}
    ok(all(lunits.get(k) == units.get(k) for k in DESC), "단위 컬럼에서 태그별 단위를 읽음")
    lcat = c.call("/api/easy/catalog", {"model": next(m["path"] for m in c.call("/api/workspace")["models"]
                                                      if m["path"].endswith("nox_stack/model.py")),
                                        "csv": upl["path"], "y": Y, "features": X, "units": lunits})
    ok(lcat["suggest"]["best"]["target"] == "STK.C_dry", "긴 형식에서도 자동 연결 (y)")

    print("[4. 모델·연결 자동 추천]")
    ws = c.call("/api/workspace")
    model = next(m["path"] for m in ws["models"] if m["path"].endswith("nox_stack/model.py"))
    ok("\\" not in model, f"모델 경로 {model}")
    cat = c.call("/api/easy/catalog", {"model": model, "csv": up["path"], "time_column": prof.get("time_column"),
                                       "y": Y, "features": X, "units": units})
    best = cat["suggest"]["best"]
    ok(best["target"] == "STK.C_dry", f"y → {best['target']}")
    want = {"F2_FAB_DRY_UTIL": "SRC_DRY.util", "F2_FAB_CVD_UTIL": "SRC_CVD.util", "F2_FAB_WET_UTIL": "SRC_WET.util",
            "F2_FAB_IMP_UTIL": "SRC_IMP.util", "F2_UT_SCR01_FAN_HZ": "FAN.n_ratio"}
    for col, v in want.items():
        got = (best["inputs"].get(col) or {}).get("value")
        ok(got == v, f"{col} → {got}")
    ok(len((best["inputs"].get("F2_UT_AMB_TEMP") or {}).get("value", "").split(",")) == 6, "외기온도 → 전체 6곳")
    fm = {col: {"targets": b["value"].split(","), "unit": b.get("unit") or units.get(col) or "1",
                "scale": b.get("scale") or 1} for col, b in best["inputs"].items()}
    body = {"csv": up["path"], "target": Y, "features": X, "time_column": prof.get("time_column"),
            "train": ["2025-03-01", "2025-03-20"], "test": ["2025-03-22", "2025-03-30"], "embargo_days": 1,
            "target_unit": units.get(Y, ""), "model": model, "feature_map": fm,
            "target_map": {"targets": [best["target"]], "unit": units.get(Y) or "1"},
            "extra_obs": [{"column": col, "targets": [v], "unit": units.get(col) or "1"}
                          for col, v in best["extra"].items()],
            "params": cat["suggested"], "limit": 100, "name": "윈도우 점검"}
    chk = c.call("/api/easy/check", body)
    ok(not chk.get("mapping_issues"), f"단위 검사 {chk.get('mapping_issues')}")

    print("[5. 물리 + ML 실행]")
    t1 = time.time()
    res = c.job("/api/easy/run", body)
    rmse = {m["name"]: (m["metrics"].get("outside") or m["metrics"]["test"])["rmse"] for m in res["models"]}
    print(f"       {', '.join(f'{k} {v:.3g}' for k, v in rmse.items())}  ({time.time() - t1:.0f}s)")
    ok(set(rmse) == {"물리모델", "ML 다항(2차)", "ML 부스팅"}, "모델 세 개")
    ok(rmse["물리모델"] < rmse["ML 부스팅"], "학습 범위 밖에서 물리모델이 부스팅보다 정확")
    pred = res["files"]["predictions"]
    ok(pred.startswith("out/easy/") and "\\" not in pred, f"결과 경로 {pred}")
    ok(len(c.call("/" + urllib.parse.quote(pred))) > 1000, "결과 CSV 내려받기")

    print("[6. 미래 예측]")
    mk = c.call("/api/forecast/make", {**body, "base_days": 3, "horizon_days": 3, "step": "1h",
                                       "adjust": [{"column": "F2_FAB_DRY_UTIL", "mode": "scale", "value": 1.1}]})
    ok(mk["check"]["ok"] and mk["path"].startswith("plans/"), f"계획 {mk['path']} ({mk['check'].get('n_rows')}행)")
    fres = c.job("/api/forecast/run", {**body, "plan": mk["path"], "band": False})
    ok(fres["plan"]["n_rows"] == 72 and fres["summary"][0]["mean"] is not None, "계획 72시간 예측")
    ok(len(c.call("/" + urllib.parse.quote(fres["files"]["forecast"]))) > 500, "예측 CSV 내려받기")

    print("[프로젝트]")
    sv = c.call("/api/project/save", {"name": "윈도우 점검: 1호기", "run": body, "results": {"x": 1}})
    ok(sv["path"].startswith("projects/") and ":" not in sv["path"], f"저장 {sv['path']}")
    ld = c.call("/api/project/load", {"path": sv["path"]})
    ok(ld["run"]["target"] == Y and ld["results"] == {"x": 1}, "다시 열기")

    print("[모델 만들기]")
    ex = next(m["path"] for m in ws["models"] if m["path"].endswith("chiller_plant/system.yaml"))
    mdl = c.call("/api/builder/load", {"path": ex})["model"]
    chk2 = c.call("/api/builder/check", {"model": mdl})
    ok(chk2["ok"] and chk2["counts"]["equations"] == chk2["counts"]["unknowns"], "칠러 모델 검사 통과")
    c.call("/api/builder/save", {"model": mdl, "path": ex}, expect=400)
    ok(True, "예제 덮어쓰기 거절")
    sv2 = c.call("/api/builder/save", {"model": {**mdl, "name": "윈도우_점검"}})
    ok(sv2["path"] == "models/윈도우_점검.yaml" and (root / sv2["path"]).exists(), f"사본 저장 {sv2['path']}")

    print(f"\n{'실패 ' + str(len(FAILS)) + '건' if FAILS else '전부 통과'} ({time.time() - t0:.0f}s)")
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — 점검 스크립트: 원인을 보이고 실패로 끝낸다
        print(f"\n중단: {type(exc).__name__}: {exc}")
        sys.exit(2)
