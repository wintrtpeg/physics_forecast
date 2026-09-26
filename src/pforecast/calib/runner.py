"""모델 실행 유틸: 파라미터 행렬 만들기 + 배치 시뮬레이션."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.solvers import solve_steady
from ..core.units import from_si


def build_param_rows(model, inputs: pd.DataFrame, base_p: np.ndarray | None = None,
                     expansion: dict[str, list[str]] | None = None, clip: bool = True,
                     clip_report: dict[str, int] | None = None) -> np.ndarray:
    """입력 DataFrame(SI) 을 시점별 파라미터 배열로 펼친다.

    ``clip`` 이면 파라미터가 선언한 물리 범위 밖의 입력을 경계로 자른다. 현장 값은
    범위를 넘는다 — 순환펌프가 서면 유량계는 정확히 0 을 찍는데, 물질전달식
    ``NTU = a·(L/G)^b`` 는 L=0 에서 기울기가 발산해 뉴턴법이 죽는다. 경계(1e-6)로
    자르면 '효율 0' 이라는 올바른 물리가 나온다. 몇 행을 잘랐는지는 ``clip_report`` 에 남는다.
    """
    p0 = model.p0() if base_p is None else np.asarray(base_p, dtype=float)
    rows = np.tile(p0, (len(inputs), 1))
    exp = expansion or {}
    for col in inputs.columns:
        targets = exp.get(col, [col])
        vals = inputs[col].to_numpy(dtype=float)
        for t in targets:
            try:
                j = model.par_index(t)
            except KeyError:
                raise KeyError(
                    f"입력 컬럼 {col!r} -> 파라미터 {t!r} 를 모델에서 찾을 수 없습니다. "
                    "태그맵의 target 을 확인하세요."
                ) from None
            v = vals
            if clip:
                info = model.parameters[j]
                lo, hi = info.lo, info.hi
                out = np.isfinite(v) & ((v < lo) | (v > hi))
                if out.any():
                    v = np.clip(v, lo, hi)
                    if clip_report is not None and t == targets[0]:
                        clip_report[col] = clip_report.get(col, 0) + int(out.sum())
            rows[:, j] = v
    return rows


def resolve_targets(model, names: list[str]) -> tuple[list[tuple[str, int]], list[str]]:
    """관측 대상 이름을 (미지수 인덱스) 또는 (출력식 이름) 으로 해석한다."""
    var_targets: list[tuple[str, int]] = []
    out_targets: list[str] = []
    for n in names:
        try:
            var_targets.append((n, model.var_index(n)))
        except KeyError:
            if n in model.outputs:
                out_targets.append(n)
            else:
                raise KeyError(
                    f"관측 대상 {n!r} 를 모델의 미지수/출력에서 찾을 수 없습니다."
                ) from None
    return var_targets, out_targets


@dataclass
class SimulationResult:
    values: pd.DataFrame          # SI 단위
    display: pd.DataFrame         # 표시 단위
    ok: np.ndarray                # 시점별 수렴 여부
    n_newton: int = 0

    @property
    def success_rate(self) -> float:
        return float(self.ok.mean()) if len(self.ok) else 0.0


def simulate(model, p_rows: np.ndarray, targets: list[str], x0: np.ndarray | None = None,
             index=None, keep_states: bool = False) -> SimulationResult:
    """시점별 정상상태를 연속으로 풀고 관심 변수만 뽑는다 (warm start)."""
    model.build()
    var_targets, out_targets = resolve_targets(model, targets)
    n = len(p_rows)
    data = {name: np.full(n, np.nan) for name in targets}
    ok = np.zeros(n, dtype=bool)
    states = np.zeros((n, model.n_vars)) if keep_states else None
    x = model.x0() if x0 is None else np.asarray(x0, dtype=float)
    total = 0
    for i, p in enumerate(p_rows):
        r = solve_steady(model, p, x0=x)
        total += r.n_newton
        if r.success:
            x = r.x
            ok[i] = True
            for name, vi in var_targets:
                data[name][i] = r.x[vi]
            if out_targets:
                ov = model.output_values_si(r.x, p)
                for name in out_targets:
                    data[name][i] = ov[name]
        if states is not None:
            states[i] = r.x
    values = pd.DataFrame(data, index=index)
    disp = {}
    for name in targets:
        try:
            unit = model.variables[model.var_index(name)].unit
        except KeyError:
            unit = model.outputs[name][1] if name in model.outputs else "1"
        disp[f"{name} [{unit}]"] = [from_si(v, unit) for v in values[name]]
    res = SimulationResult(values=values, display=pd.DataFrame(disp, index=index), ok=ok,
                           n_newton=total)
    if keep_states:
        res.states = states  # type: ignore[attr-defined]
    return res
