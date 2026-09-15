"""현장 태그 <-> 모델 변수 매핑.

Historian 태그 이름은 ``FAB2_UT_SCR01_DRY_UTIL`` 처럼 생겼고, 모델은
``SRC_DRY.util`` 이라고 부른다. 이 사이를 YAML 한 장으로 잇는다. 단위 환산과
계측 불확도(sigma)도 여기서 함께 선언한다.

모델 코드가 태그 이름을 모르게 하는 것이 요점이다. 라인이 바뀌거나 태그 체계가
바뀌어도 YAML 만 갈아끼우면 같은 물리 모델을 그대로 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..core.units import to_si


@dataclass
class TagEntry:
    tag: str
    #: 모델 변수/파라미터 이름. 하나의 태그가 여러 파라미터에 걸릴 수 있다
    #: (예: 외기온도 하나가 모든 덕트의 ``T_amb`` 로 들어간다).
    target: str | list[str]
    unit: str = "1"
    sigma: float | None = None      # 계측 표준편차 (target 단위)
    scale: float = 1.0              # 태그값에 먼저 곱할 계수
    offset: float = 0.0             # 태그값에 먼저 더할 값
    desc: str = ""

    def to_si_series(self, series):
        raw = series * self.scale + self.offset
        if self.unit in ("degC", "C", "oC", "℃"):
            return raw + 273.15
        return raw * to_si(1.0, self.unit)

    @property
    def targets(self) -> list[str]:
        if isinstance(self.target, str):
            return [t.strip() for t in self.target.split(",") if t.strip()]
        return list(self.target)

    @property
    def primary(self) -> str:
        return self.targets[0]

    @property
    def sigma_si(self) -> float | None:
        if self.sigma is None:
            return None
        # 오프셋 단위(degC)는 차이값이므로 계수만 적용한다
        if self.unit in ("degC", "C", "oC", "℃"):
            return float(self.sigma)
        return float(self.sigma) * to_si(1.0, self.unit)


@dataclass
class TagMap:
    timestamp_column: str = "timestamp"
    timestamp_format: str | None = None
    resample: str | None = "5min"
    inputs: list[TagEntry] = field(default_factory=list)
    observations: list[TagEntry] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "TagMap":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        ts = data.get("timestamp", {}) or {}
        return cls(
            timestamp_column=ts.get("column", "timestamp"),
            timestamp_format=ts.get("format"),
            resample=data.get("resample", "5min"),
            inputs=[TagEntry(**e) for e in (data.get("inputs") or [])],
            observations=[TagEntry(**e) for e in (data.get("observations") or [])],
            meta=data.get("meta", {}) or {},
        )

    def dump(self, path: str | Path) -> None:
        def ser(e: TagEntry) -> dict:
            d = {"tag": e.tag, "target": e.target, "unit": e.unit}
            if e.sigma is not None:
                d["sigma"] = e.sigma
            if e.scale != 1.0:
                d["scale"] = e.scale
            if e.offset != 0.0:
                d["offset"] = e.offset
            if e.desc:
                d["desc"] = e.desc
            return d

        payload = {
            "timestamp": {"column": self.timestamp_column, **({"format": self.timestamp_format} if self.timestamp_format else {})},
            "resample": self.resample,
            "inputs": [ser(e) for e in self.inputs],
            "observations": [ser(e) for e in self.observations],
        }
        if self.meta:
            payload["meta"] = self.meta
        Path(path).write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

    @property
    def all_tags(self) -> list[str]:
        return [e.tag for e in self.inputs] + [e.tag for e in self.observations]

    def input_targets(self) -> list[str]:
        return [e.primary for e in self.inputs]

    def observation_targets(self) -> list[str]:
        return [e.primary for e in self.observations]

    def expansion(self) -> dict[str, list[str]]:
        """대표 이름 -> 실제로 값을 써 넣을 파라미터 목록."""
        return {e.primary: e.targets for e in self.inputs}

    def sigmas(self) -> dict[str, float]:
        return {e.primary: (e.sigma_si if e.sigma_si is not None else 1.0) for e in self.observations}
