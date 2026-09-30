"""결과 파일 쓰기 — 윈도우에서 엑셀이 연 파일은 잠겨 있다.

엑셀은 CSV 를 열면 다른 프로그램이 쓰지 못하게 잠근다 (``PermissionError``). 사용자는 결과·계획
CSV 를 엑셀로 열어 두고 앱에서 다시 실행하는 일이 흔하다. 그때 작업을 통째로 실패시키지 않고
새 이름으로 저장한 뒤 **그 사실을 알린다** (조용히 다른 파일에 쓰지 않는다).

윈도우 예약 이름(CON, NUL, COM1 …)은 파일로 만들 수 없으므로 이름을 바꾼다.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Callable

_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_stem(name: str, default: str = "file") -> str:
    """파일 이름으로 쓸 수 있게: 한글·영문·숫자·_-. 만, 윈도우 예약 이름 피하기."""
    s = re.sub(r"[^0-9A-Za-z가-힣_\-.()\s]+", "_", str(name)).strip(" ._") or default
    if s.split(".")[0].upper() in _RESERVED:
        s = s + "_"
    return s


def write_locked_ok(path: str | Path, writer: Callable[[Path], None]) -> tuple[Path, str | None]:
    """``writer(path)`` 로 쓴다. 잠겨 있으면 ``이름_시분초`` 로 쓰고 (실제 경로, 알림)을 돌려준다."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer(p)
        return p, None
    except PermissionError:
        alt = p.with_name(f"{p.stem}_{datetime.now():%H%M%S}{p.suffix}")
        writer(alt)
        return alt, (f"‘{p.name}’ 이 다른 프로그램(엑셀 등)에서 열려 있어 ‘{alt.name}’ 으로 저장했습니다. "
                     "그 파일을 닫으면 다음부터는 원래 이름으로 저장됩니다.")


def rel_posix(p: str | Path, root: str | Path) -> str:
    """작업 폴더 기준 경로를 항상 ``/`` 로 (윈도우의 ``\\`` 를 화면이 가정하지 않게)."""
    p, root = Path(p).resolve(), Path(root).resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()
