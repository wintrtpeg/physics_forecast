"""pforecast 실행기 — ``run_pforecast.bat`` 이 부른다.

배치 파일은 영문(ASCII)만 두고 파이썬을 찾는 일만 한다. ``cmd.exe`` 는 ``chcp 65001`` 뒤 한글(UTF-8)이
든 배치 파일을, 외부 프로그램(파이썬·pip)이 끝난 다음 줄부터 잘못 읽고 **아무 말 없이 멈춘다**
(윈도우 점검에서 실제로 난 일). 그래서 가상환경·설치·실행과 한글 안내는 여기서 한다.

    python scripts/launch.py [작업폴더]

1. 이 폴더에 가상환경(.venv)이 없거나 깨졌으면 만든다.
2. 처음이거나 ``pyproject.toml`` 이 바뀌었으면 설치한다. ``wheels/`` 에 wheel 이 있으면 인터넷 없이.
3. 앱을 띄운다 (브라우저가 열린다). 창을 닫거나 Ctrl+C 로 끝낸다.

환경 변수: ``PF_PORT`` (기본 8765), ``PF_NO_BROWSER=1`` (브라우저 안 엶).
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 10):
    print(f"[오류] 파이썬 3.10 이상이 필요합니다 (지금 {sys.version.split()[0]}). python.org 에서 3.11 을 설치하세요.")
    sys.exit(1)

import hashlib  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
VPY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
STAMP = VENV / "pforecast.installed"


def build_version() -> str:
    """src/pforecast/_build.py 의 커밋 (zip 으로 받았으면 채워져 있다). 어느 판을 돌리는지 창에 보인다."""
    import re
    try:
        src = (ROOT / "src" / "pforecast" / "_build.py").read_text(encoding="utf-8")
    except OSError:
        return "?"
    m = re.search(r'COMMIT = "([^"$]+)"', src), re.search(r'DATE = "([^"$]+)"', src)
    if m[0]:
        return f"{m[0].group(1)} ({m[1].group(1) if m[1] else ''})"
    return "(git 작업 사본)"


def say(msg: str = "") -> None:
    print(msg, flush=True)


def fail(msg: str) -> int:
    say()
    say("[오류] " + msg)
    return 1


def venv_ok() -> bool:
    if not VPY.exists():
        return False
    try:                                   # 만든 파이썬을 지웠거나 옮기면 .venv 가 깨진다
        return subprocess.run([str(VPY), "-c", "import sys"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def wanted_stamp() -> str:
    return hashlib.sha256((ROOT / "pyproject.toml").read_bytes()).hexdigest()


def install() -> int:
    wheels = ROOT / "wheels"
    if wheels.is_dir() and any(wheels.glob("*.whl")):
        say("오프라인 설치 중 (wheels 폴더) ...")
        base = [str(VPY), "-m", "pip", "install", "--no-index", "--find-links", str(wheels)]
    else:
        say("인터넷(또는 사내 미러)에서 설치 중 ... 처음 한 번만 몇 분 걸립니다")
        base = [str(VPY), "-m", "pip", "install"]
    subprocess.run(base + ["--quiet", "--upgrade", "pip"])   # 실패해도 계속 (묶음에 pip 가 없을 수 있다)
    r = subprocess.run(base + ["-e", f"{ROOT}[plot,ml]"]).returncode
    if r != 0:
        STAMP.unlink(missing_ok=True)
        return fail("설치하지 못했습니다.\n"
                    "  - 사내망이면: 인터넷 되는 PC 에서 'python scripts\\make_offline_bundle.py' 로 묶음을 만들어\n"
                    "    wheels 폴더째 이 폴더에 복사한 뒤 다시 실행하세요 (묶음의 파이썬 버전이 이 PC 와 같아야 합니다).\n"
                    "  - 프록시가 필요하면: 명령창에서 set HTTPS_PROXY=http://프록시:포트 후 다시 실행하세요.")
    STAMP.write_text(wanted_stamp(), encoding="utf-8")
    return 0


def main(argv: list[str]) -> int:
    work = Path(argv[1].strip().strip('"')).resolve() if len(argv) > 1 and argv[1].strip() else ROOT
    if not work.is_dir():
        return fail(f"작업 폴더가 없습니다: {work}")
    port = os.environ.get("PF_PORT", "8765")
    say(f"pforecast {build_version()} — 파이썬 {sys.version.split()[0]} ({sys.executable})")

    if not venv_ok():
        say("가상환경을 만드는 중 (.venv) ...")
        r = subprocess.run([sys.executable, "-m", "venv", "--clear", str(VENV)]).returncode
        if r != 0 or not VPY.exists():
            return fail("가상환경을 만들지 못했습니다. 폴더에 쓰기 권한이 있는지, 백신이 막지 않는지 확인하세요.")
    try:
        done = STAMP.read_text(encoding="utf-8").strip() == wanted_stamp()
    except OSError:
        done = False
    if not done and install() != 0:
        return 1

    say()
    say(f"pforecast 를 띄웁니다: http://127.0.0.1:{port}")
    say(f"작업 폴더: {work}")
    say("끝내려면 이 창을 닫거나 Ctrl+C 를 누르세요.")
    say()
    cmd = [str(VPY), "-m", "pforecast.cli", "serve", "--root", str(work), "--port", port]
    if os.environ.get("PF_NO_BROWSER"):
        cmd.append("--no-browser")
    try:
        r = subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        return 0
    if r not in (0, None) and r != -2 and r != 3221225786:   # Ctrl+C 종료 코드는 오류가 아니다
        return fail(f"앱이 비정상 종료했습니다 (코드 {r}). 위 메시지를 확인하세요.\n"
                    f"  포트 {port} 를 다른 프로그램이 쓰고 있으면: set PF_PORT=8766 후 다시 실행하세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
