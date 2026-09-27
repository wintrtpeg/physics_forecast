"""컴포넌트 방정식을 사람이 읽는 형태로 설명하기.

결과를 받는 사람은 ``equations()`` 의 잔차식(174개)을 읽지 않는다. 그래서 컴포넌트마다
**사람이 읽는 식**을 ``LAWS`` 로 따로 적어 둔다. 규칙은 두 가지다.

* ``LAWS`` 는 ``equations()`` 와 **같은 내용**이어야 한다. 식을 고치면 여기도 고친다.
  (자동으로 맞춰 볼 수 없으니 ``tests/test_explain.py`` 가 파라미터 이름과 보정 대상
  포함 여부만 검사한다.)
* 식의 종류를 적는다. 보존법칙·상태식은 데이터와 무관하게 성립하므로 외삽의 근거가
  되고, 보정 파라미터는 구성방정식에만 들어간다. 결과 화면이 이 구분을 보여준다.

식 표기: ``_{..}`` 아래첨자, ``^{..}`` 위첨자, 백틱 한 쌍으로 감싼 이름은 파라미터
(예: 백틱 ef_process 백틱). 화면이 파라미터를 코드체로 표시하고, 보정한 것은 강조한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CONSERVATION = "보존법칙"
STATE = "상태·정의"
CLOSURE = "구성방정식"
BOUNDARY = "경계조건"
DECLARED = "선언식"

KIND_NOTE = {
    CONSERVATION: "질량·에너지·화학종 보존. 데이터와 무관하게 성립하므로 학습 범위 밖에서도 맞습니다.",
    STATE: "물성과 정의(이상기체, 농도 환산 등). 조건이 바뀌어도 그대로 성립합니다.",
    CLOSURE: "장비 특성을 나타내는 식. 보정 파라미터는 여기에만 들어갑니다.",
    BOUNDARY: "설비 경계에서 주어지는 값(배기 온도, 대기압 등).",
    DECLARED: "YAML 로 선언한 식을 그대로 옮겼습니다.",
}

_PARAM = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")


@dataclass(frozen=True)
class Law:
    kind: str
    title: str
    formula: str

    @property
    def params(self) -> tuple[str, ...]:
        """식에 백틱으로 표시한 파라미터 이름."""
        return tuple(dict.fromkeys(_PARAM.findall(self.formula)))


def laws_of(comp) -> list[Law]:
    """컴포넌트의 사람이 읽는 식. ``LAWS`` 가 없으면 선언식(YAML)을 그대로 쓴다."""
    laws = getattr(type(comp), "LAWS", None)
    if laws:
        return list(laws)
    src = getattr(comp, "_equation_src", None)
    if src:
        names = sorted(comp.param_specs(), key=len, reverse=True)
        out = []
        for label, text in src.items():
            f = text
            for n in names:
                f = re.sub(rf"(?<![\w.`]){re.escape(n)}(?![\w`])", f"`{n}`", f)
            out.append(Law(DECLARED, label, f))
        return out
    return []


def title_of(comp) -> str:
    """컴포넌트 종류의 한 줄 이름 (클래스 docstring 첫 줄 또는 YAML description)."""
    text = getattr(comp, "description", "") or type(comp).__doc__ or type(comp).__name__
    first = text.strip().split("\n")[0]
    return first.split(". ")[0].split(":")[0].split(" (")[0].rstrip(". ")
