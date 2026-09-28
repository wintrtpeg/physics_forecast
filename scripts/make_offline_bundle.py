"""사내 PC(인터넷 차단)에 설치할 묶음을 만든다 — 인터넷 되는 PC 에서 한 번 실행.

    python scripts/make_offline_bundle.py                 # 윈도우 64비트 + 파이썬 3.11 용
    python scripts/make_offline_bundle.py --python 3.12

결과: ``dist/pforecast_offline_win_py311.zip``
  pforecast/            소스 (git 이 추적하는 파일만 — 현장 데이터·작업물은 들어가지 않는다)
  pforecast/wheels/     의존 패키지 (numpy, scipy, pandas, PyYAML, matplotlib, scikit-learn ...)

사내 PC 에서: 압축을 풀고 ``run_pforecast.bat`` 더블클릭. wheels 폴더가 있으면 인터넷 없이 설치한다.
파이썬 자체(설치 파일)는 python.org 또는 사내 소프트웨어 센터에서 따로 받는다.
GPU 패키지(torch 등)는 넣지 않는다 — 필요 없고 사내 PC 에 설치되지 않는다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRAS = ("plot", "ml")                  # 앱이 쓰는 것. sql 은 사내 DB 직접 조회용이라 뺀다
BUILD = ["setuptools>=68", "wheel", "pip"]


def requirements() -> list[str]:
    try:
        import tomllib
    except ImportError:                   # 파이썬 3.10
        sys.exit("이 스크립트는 파이썬 3.11 이상에서 실행하세요 (묶음 대상 버전은 --python 으로 따로 정합니다).")
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    reqs = list(meta["dependencies"])
    for e in EXTRAS:
        reqs += meta.get("optional-dependencies", {}).get(e, [])
    return reqs + BUILD


def pin_to_here(reqs: list[str]) -> list[str]:
    """이 PC 에 깔린 버전으로 고정한다 — 테스트를 통과한 조합을 그대로 옮기려고."""
    import re
    from importlib.metadata import PackageNotFoundError, version
    out = []
    for r in reqs:
        name = re.split(r"[<>=!~ ;\[]", r, 1)[0]
        try:
            out.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            out.append(r)
    return out


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
        return [ROOT / p for p in out.decode("utf-8").split("\0") if p]
    except (OSError, subprocess.CalledProcessError):
        sys.exit("git 저장소 안에서 실행하세요 (추적 파일만 묶어야 현장 데이터가 섞이지 않습니다).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", default="3.11", help="사내 PC 의 파이썬 버전 (기본 3.11)")
    ap.add_argument("--platform", default="win_amd64", help="기본 win_amd64 (윈도우 64비트)")
    ap.add_argument("--out", help="zip 경로 (기본 dist/pforecast_offline_<플랫폼>_py<버전>.zip)")
    ap.add_argument("--latest", action="store_true",
                    help="이 PC 에 깔린 버전 대신 최신 버전을 받는다 (기본은 이 PC 버전으로 고정)")
    args = ap.parse_args()

    tag = f"{args.platform.split('_')[0]}_py{args.python.replace('.', '')}"
    wheels = ROOT / "wheels"
    wheels.mkdir(exist_ok=True)
    reqs = requirements() if args.latest else pin_to_here(requirements())
    print("의존 패키지 받는 중:", ", ".join(reqs))
    cmd = [sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--platform", args.platform,
           "--python-version", args.python, "--implementation", "cp", "-d", str(wheels), *reqs]
    if subprocess.run(cmd).returncode != 0:
        print("pip download 실패 — 인터넷 연결(또는 HTTPS_PROXY)을 확인하세요.")
        return 1
    files = sorted(wheels.glob("*.whl"))
    size = sum(f.stat().st_size for f in files) / 1e6
    print(f"  wheels/ {len(files)}개, {size:.0f} MB")

    out = Path(args.out) if args.out else ROOT / "dist" / f"pforecast_offline_{tag}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    src = [p for p in tracked_files() if p.is_file()]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src:
            z.write(p, Path("pforecast") / p.relative_to(ROOT))
        for p in files:
            z.write(p, Path("pforecast") / "wheels" / p.name)
    print(f"\n묶음: {out}  ({out.stat().st_size / 1e6:.0f} MB, 소스 {len(src)}개 + wheels {len(files)}개)")
    print("사내 PC: 압축 풀기 → run_pforecast.bat 더블클릭 (파이썬 3.11 은 따로 설치)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
