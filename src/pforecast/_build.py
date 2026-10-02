"""배포 묶음의 버전. git archive (GitHub 의 'Download ZIP' 도 같다) 가 아래 두 칸을 커밋으로 채운다
(.gitattributes 의 export-subst). git 으로 받은 작업 사본에서는 git 에 물어본다.

PC 에서 '고쳤다는데 그대로' 일 때 어느 판을 돌리는지 화면(시작 화면 아래)과 실행 창에서 확인한다.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

COMMIT = "$Format:%h$"
DATE = "$Format:%cs$"


def version() -> str:
    if not COMMIT.startswith("$"):
        return f"{COMMIT} ({DATE})"
    root = Path(__file__).resolve().parents[2]
    try:
        r = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%h (%cs)"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "버전 정보 없음"
