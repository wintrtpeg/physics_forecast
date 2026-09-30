"""떠 있는 앱을 브라우저로 끝까지 눌러 본다 — 윈도우 점검(`.github/workflows/windows.yml`)이 쓴다.

    python scripts/windows_ui_smoke.py --channel msedge          # 윈도우: 설치된 Edge
    python scripts/windows_ui_smoke.py --executable /opt/pw-browsers/chromium

시작 → CSV 업로드(CP949·시각 +09:00) → 변수 → 기간 → 모델(자동 연결 확인) → 실행 → 검증 결과 →
미래 예측 → 시작(최근 프로젝트) → 모델 만들기. 콘솔 오류가 하나라도 있으면 실패. 화면은 ui_shots/ 에.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from windows_smoke import X, Y, site_csv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--channel", help="msedge | chrome (설치된 브라우저)")
    ap.add_argument("--executable", help="브라우저 실행 파일 경로")
    ap.add_argument("--shots", default="ui_shots")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright

    shots = Path(a.shots)
    shots.mkdir(exist_ok=True)
    csv = Path(tempfile.mkdtemp()) / "현장 데이터(화면).csv"
    csv.write_bytes(site_csv())
    errors: list[str] = []
    step = "시작"
    with sync_playwright() as p:
        kw = {"channel": a.channel} if a.channel else {"executable_path": a.executable} if a.executable else {}
        b = p.chromium.launch(**kw)
        page = b.new_page(viewport={"width": 1366, "height": 900})
        page.on("pageerror", lambda e: errors.append(f"[{step}] {e}"))
        page.on("console", lambda m: errors.append(f"[{step}] {m.text}") if m.type == "error" else None)
        try:
            page.goto(f"http://127.0.0.1:{a.port}/")
            page.wait_for_selector(".homecard")
            assert page.locator(".homecard").count() == 3
            page.screenshot(path=str(shots / "01_home.png"))

            step = "1 데이터"
            page.click(".homecard.primary")
            page.set_input_files("#file", str(csv))
            page.wait_for_selector("text=다음: 변수 고르기", timeout=120000)
            page.screenshot(path=str(shots / "02_data.png"), full_page=True)
            page.click("text=다음: 변수 고르기")

            step = "2 변수"
            # 저장된 설정이 미리 채운 선택이 있을 수 있다 — 누르지 말고 원하는 상태로 맞춘다
            # 누를 때마다 표가 다시 그려지므로 매번 이름으로 새로 찾는다
            setter = """([col, kind, want]) => {
                const r = [...document.querySelectorAll('#ez-vars-body tr')].find(r => {
                    const n = r.querySelector('td.name'); return n && n.firstChild && n.firstChild.textContent === col; });
                const b = r && r.querySelector(`input[type=${kind}]`);
                if (b && !b.disabled && b.checked !== want) b.click(); }"""
            page.evaluate(setter, [Y, "radio", True])
            names = page.eval_on_selector_all("#ez-vars-body tr td.name", "ns => ns.map(n => n.firstChild.textContent)")
            for col in names:
                page.evaluate(setter, [col, "checkbox", col in X])
            page.click("#screen-ez-vars button:has-text('다음: 기간')")

            step = "3 기간"
            page.click("#screen-ez-period button:has-text('다음: 모델'):not([disabled])", timeout=120000)

            step = "4 모델"
            sel = page.locator("#ez-model-pane select").first
            if not sel.input_value():               # 모델 파일이 여럿이면 NOx 예제를 고른다
                sel.select_option(value=[o for o in sel.locator("option").evaluate_all(
                    "os => os.map(o => o.value)") if o.endswith("nox_stack/model.py")][0])
            page.wait_for_selector("#ez-map", timeout=120000)
            page.wait_for_selector("#ez-map-issues .note", timeout=60000)
            stats = page.eval_on_selector_all("#ez-map td.mstat", "ts => ts.map(t => t.innerText)")
            print("  연결 확인칸:", [s.split("\n")[0] for s in stats])
            assert all(s.startswith("✓") for s in stats), stats
            page.screenshot(path=str(shots / "04_model.png"), full_page=True)
            page.click("#ez-run")
            page.wait_for_selector("#screen-ez-result.active", timeout=1500000)

            step = "5 검증 결과"
            page.wait_for_timeout(800)
            print("  결과:", page.inner_text("#ez-result-body .tiles").replace("\n", " ")[:200])
            page.screenshot(path=str(shots / "05_result.png"), full_page=True)
            page.click("#screen-ez-result button:has-text('다음: 미래 예측')")

            step = "6 미래 예측"
            page.wait_for_selector(".fcgen")
            page.fill(".fcgen input >> nth=0", "3")
            page.fill(".fcgen input >> nth=1", "3")
            page.click("#fc-plan button:has-text('계획 만들기')")
            page.wait_for_selector("#fc-check .tiles", timeout=120000)
            band = page.locator("#screen-ez-forecast input[type=checkbox]")
            if band.count() and band.first.is_checked():
                band.first.uncheck()
            page.click("#fc-run")
            page.wait_for_selector("#fc-result .tiles", timeout=1500000)
            page.wait_for_timeout(800)
            page.screenshot(path=str(shots / "06_forecast.png"), full_page=True)

            step = "시작(최근 프로젝트)"
            page.click("#nav button[data-screen='home']")
            page.wait_for_selector("#home-projects table", timeout=30000)

            step = "모델 만들기"
            page.click("#nav button[data-screen='builder']")
            page.wait_for_selector(".palitem", timeout=60000)
            opts = page.locator("#mb-bar select option").evaluate_all("os => os.map(o => o.value)")
            ex = [o for o in opts if o.endswith("chiller_plant/system.yaml")][0]
            page.select_option("#mb-bar select", ex)
            page.wait_for_selector(".mbstate.ok", timeout=60000)
            assert page.inner_text("#mb-bar button.btn.sm:not(.ghost)") == "models/ 에 저장", "예제는 사본으로"
            page.screenshot(path=str(shots / "07_builder.png"), full_page=True)
        except Exception as exc:  # noqa: BLE001
            page.screenshot(path=str(shots / "zz_failed.png"), full_page=True)
            print(f"\n중단 [{step}]: {type(exc).__name__}: {exc}")
            b.close()
            return 2
        b.close()
    if errors:
        print("\n콘솔 오류:")
        for e in errors:
            print("  - " + e)
        return 1
    print("\n화면 점검 전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
