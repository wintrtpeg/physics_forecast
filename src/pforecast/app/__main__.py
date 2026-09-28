"""``python -m pforecast.app`` 진입점."""

from __future__ import annotations

import argparse

from .server import serve


def main() -> None:
    ap = argparse.ArgumentParser(prog="pforecast.app", description="pforecast 로컬 웹 앱")
    ap.add_argument("--root", default=".", help="작업 폴더 (CSV/모델을 찾을 위치)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    serve(a.root, a.host, a.port, not a.no_browser)


if __name__ == "__main__":
    main()
