"""시스템 조립: 컴포넌트 + 연결 -> 컴파일된 방정식계.

``System`` 은 선언만 모으고, ``compile()`` 이 다음을 수행한다.
  1) 각 컴포넌트의 방정식 수집
  2) 연결 방정식 생성 (across 동일 / through 합=0 / stream 전달)
  3) 차원 동차성 검사
  4) 구조 해석(매칭·BLT)
  5) 블록별 잔차/야코비안 코드 생성

결과물 ``CompiledModel`` 은 순수 numpy 함수 묶음이라 pickle 없이도 빠르게 반복 호출된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from . import symbolic as S
from .component import Component, PortSpec, Scope
from .structural import StructuralReport, analyze
from .units import DimensionError, dim_str, from_si, to_si


class ModelError(ValueError):
    """모델 조립 단계에서 잡히는 오류."""


@dataclass
class BlockFns:
    """한 BLT 블록의 컴파일된 잔차/야코비안."""

    residual: Callable          # (x, p) -> 잔차 벡터
    jac_values: Callable        # (x, p) -> 희소 야코비안 값 (COO 순서)
    jac_rows: np.ndarray        # 블록 내 지역 행 인덱스
    jac_cols: np.ndarray        # 블록 내 지역 열 인덱스
    eqs: np.ndarray             # 전역 방정식 인덱스
    vars: np.ndarray            # 전역 미지수 인덱스

    #: 이 크기를 넘으면 희소 LU 로 푼다. 아래에서는 dense 가 더 빠르다.
    SPARSE_THRESHOLD: int = 50
    _csc: object = None
    _perm: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self.vars)

    @property
    def use_sparse(self) -> bool:
        return self.size > self.SPARSE_THRESHOLD and len(self.jac_rows) > 0

    def dense_jacobian(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        n = self.size
        J = np.zeros((n, n))
        if len(self.jac_rows):
            J[self.jac_rows, self.jac_cols] = self.jac_values(x, p)
        return J

    def sparse_jacobian(self, x: np.ndarray, p: np.ndarray):
        """희소 구조는 한 번만 만들고 값만 갈아 끼운다 (매번 재구성하면 그게 더 비싸다)."""
        from scipy.sparse import csc_matrix

        vals = self.jac_values(x, p)
        if self._csc is None:
            n = self.size
            nnz = len(self.jac_rows)
            probe = csc_matrix((np.arange(nnz, dtype=float), (self.jac_rows, self.jac_cols)),
                               shape=(n, n))
            self._perm = probe.data.astype(int)
            probe.data = vals[self._perm]
            self._csc = probe
        else:
            self._csc.data = vals[self._perm]
        return self._csc

    def row_abs_sums(self, x: np.ndarray, p: np.ndarray, col_scale: np.ndarray) -> np.ndarray:
        """행별 |J_ij| * c_j 합. 수렴 판정의 스케일로 쓴다."""
        out = np.zeros(self.size)
        if len(self.jac_rows):
            np.add.at(out, self.jac_rows, np.abs(self.jac_values(x, p)) * col_scale[self.jac_cols])
        return out


@dataclass
class EquationInfo:
    index: int
    expr: S.Expr
    source: str          # "SCR1" 또는 "connect(A.out, B.in)"
    label: str           # 사람이 읽는 설명


@dataclass
class VarInfo:
    index: int
    name: str
    unit: str
    start: float
    lo: float
    hi: float


@dataclass
class ParInfo:
    index: int
    name: str
    unit: str
    value: float          # SI
    lo: float
    hi: float
    tunable: bool
    desc: str = ""


class System:
    """컴포넌트와 연결을 모으는 컨테이너."""

    def __init__(self, name: str = "system"):
        self.name = name
        self.components: dict[str, Component] = {}
        self.connections: list[tuple[str, str]] = []
        self._extra_eqs: list[tuple[S.Expr, str, str]] = []

    def add(self, comp: Component) -> Component:
        if comp.name in self.components:
            raise ModelError(f"컴포넌트 이름 중복: {comp.name!r}")
        self.components[comp.name] = comp
        return comp

    def add_all(self, *comps: Component) -> None:
        for c in comps:
            self.add(c)

    def connect(self, a: str, b: str) -> None:
        """``connect("SRC1.outlet", "DUCT1.inlet")`` 형태로 포트를 잇는다."""
        self.connections.append((a, b))

    def add_equation(self, expr: S.Expr, label: str = "", source: str = "system") -> None:
        """시스템 레벨 구속조건(예: 두 팬의 회전수 동일)을 직접 추가."""
        self._extra_eqs.append((expr, source, label or S.expr_to_str(expr)))

    # --- 조립 -------------------------------------------------------------
    def _resolve_port(self, ref: str):
        if "." not in ref:
            raise ModelError(f"포트 참조 형식은 'COMP.PORT' 입니다: {ref!r}")
        cname, pname = ref.rsplit(".", 1)
        comp = self.components.get(cname)
        if comp is None:
            raise ModelError(f"연결 대상 컴포넌트를 찾을 수 없습니다: {cname!r} (참조 {ref!r})")
        if pname not in comp.port_specs():
            raise ModelError(
                f"{cname} 에 포트 {pname!r} 가 없습니다. 가능: {sorted(comp.port_specs())}"
            )
        return cname, pname, comp.port_specs()[pname]

    def topological_order(self) -> list[str]:
        """연결 방향(out -> in)에 따른 컴포넌트 위상 순서. 순환이 있으면 남은 순서대로."""
        succ: dict[str, list[str]] = {n: [] for n in self.components}
        indeg: dict[str, int] = {n: 0 for n in self.components}
        for a, b in self.connections:
            ca, pa, sa = self._resolve_port(a)
            cb, pb, sb = self._resolve_port(b)
            up, down = (ca, cb) if sa.role == "out" else (cb, ca)
            succ[up].append(down)
            indeg[down] += 1
        queue = [n for n, d in indeg.items() if d == 0]
        order: list[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for m in succ[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        order.extend(n for n in self.components if n not in order)
        return order

    def estimate_starts(self, p_amb: float = 101325.0) -> dict[str, float]:
        """상류에서 하류로 초기값을 전파해 뉴턴법 출발점을 만든다."""
        # 연결 맵: 출구 포트 -> 입구 포트
        downstream: dict[str, str] = {}
        for a, b in self.connections:
            ca, pa, sa = self._resolve_port(a)
            cb, pb, sb = self._resolve_port(b)
            if sa.role == "out":
                downstream[f"{ca}.{pa}"] = f"{cb}.{pb}"
            else:
                downstream[f"{cb}.{pb}"] = f"{ca}.{pa}"
        inbox: dict[str, dict[str, dict[str, float]]] = {n: {} for n in self.components}
        starts: dict[str, float] = {}
        for cname in self.topological_order():
            comp = self.components[cname]
            try:
                guess = comp.initial_guess(inbox[cname]) or {}
            except Exception:
                guess = {}
            for vname, val in (guess.get("vars") or {}).items():
                starts[f"{cname}.{vname}"] = float(val)
            for pname, state in (guess.get("ports") or {}).items():
                spec = comp.port_specs()[pname]
                sign = -1.0 if spec.role == "out" else 1.0
                for vn, val in state.items():
                    key = f"{cname}.{pname}.{vn}"
                    starts[key] = float(val) * (sign if vn == "mdot" else 1.0)
                nxt = downstream.get(f"{cname}.{pname}")
                if nxt:
                    ncomp, nport = nxt.rsplit(".", 1)
                    inbox[ncomp][nport] = dict(state)
                    nspec = self.components[ncomp].port_specs()[nport]
                    nsign = -1.0 if nspec.role == "out" else 1.0
                    for vn, val in state.items():
                        starts[f"{nxt}.{vn}"] = float(val) * (nsign if vn == "mdot" else 1.0)
        # 압력은 대기압 근처에서 출발시킨다 (스케일 1e5, 실제 편차는 1e3 수준)
        for cname, comp in self.components.items():
            for pname, spec in comp.port_specs().items():
                for vn, _ in spec.across:
                    starts.setdefault(f"{cname}.{pname}.{vn}", p_amb)
        return starts

    def compile(self, check_dims: bool = True, mode: str = "steady",
                propagate_starts: bool = True) -> "CompiledModel":
        scopes: dict[str, Scope] = {n: c.build_scope() for n, c in self.components.items()}
        eqs: list[EquationInfo] = []

        def push(expr: S.Expr, source: str, label: str) -> None:
            eqs.append(EquationInfo(len(eqs), expr, source, label))

        for cname, comp in self.components.items():
            sc = scopes[cname]
            try:
                comp_eqs = comp.equations(sc)
            except Exception as exc:  # 모델 작성 실수를 컴포넌트 이름과 함께 보여준다
                raise ModelError(f"{cname}({type(comp).__name__}) 방정식 생성 실패: {exc}") from exc
            for k, e in enumerate(comp_eqs):
                push(e, cname, f"{cname} eq[{k}]")

        # 연결 방정식
        used: dict[str, str] = {}
        for a, b in self.connections:
            (ca, pa, sa) = self._resolve_port(a)
            (cb, pb, sb) = self._resolve_port(b)
            for ref in (a, b):
                if ref in used:
                    raise ModelError(
                        f"포트 {ref!r} 가 이미 {used[ref]!r} 에 연결되어 있습니다. "
                        "합류/분기는 Mixer/Splitter 컴포넌트를 쓰세요."
                    )
            used[a], used[b] = b, a
            if sa.kind != sb.kind:
                raise ModelError(f"포트 종류가 다릅니다: {a}({sa.kind}) <-> {b}({sb.kind})")
            if sa.role == sb.role:
                raise ModelError(
                    f"같은 방향 포트끼리 연결했습니다: {a}(role={sa.role}) <-> {b}(role={sb.role})"
                )
            va, vb = scopes[ca].port(pa), scopes[cb].port(pb)
            tag = f"connect({a}, {b})"
            for vn, _ in sa.across:
                push(getattr(va, vn) - getattr(vb, vn), tag, f"{tag}: {vn} 동일")
            for vn, _ in sa.through:
                push(getattr(va, vn) + getattr(vb, vn), tag, f"{tag}: {vn} 합=0")
            for vn, _ in sa.stream:
                push(getattr(va, vn) - getattr(vb, vn), tag, f"{tag}: {vn} 전달")

        # 미연결 포트 검사
        dangling = []
        for cname, comp in self.components.items():
            for pname in comp.port_specs():
                if f"{cname}.{pname}" not in used:
                    dangling.append(f"{cname}.{pname}")
        if dangling:
            raise ModelError(
                "연결되지 않은 포트가 있습니다: " + ", ".join(sorted(dangling))
                + "\n  경계조건은 Source/Ambient 같은 경계 컴포넌트로 닫아야 합니다."
            )

        for expr, source, label in self._extra_eqs:
            push(expr, source, label)

        if not eqs:
            raise ModelError("방정식이 하나도 없습니다.")

        # der(x) 처리: 정상상태는 0, 과도해석은 음함수 오일러로 이산화
        der_syms = [s for s in S.collect_syms([e.expr for e in eqs]) if s.kind == "der"]
        state_map: dict[str, str] = {}
        if der_syms:
            if mode == "steady":
                mapping = {d.uid: S.const(0.0) for d in der_syms}
            elif mode == "transient":
                dt = S.sym("_dt", (0, 0, 1, 0, 0, 0, 0), kind="par",
                           meta={"unit": "s", "internal": True, "value": 1.0})
                mapping = {}
                for d in der_syms:
                    vname = d.name[4:-1]
                    vsym = S.sym(vname, tuple(a + b for a, b in zip(d.dim, (0, 0, 1, 0, 0, 0, 0))), kind="var")
                    prev = S.sym(f"_prev.{vname}", vsym.dim, kind="par",
                                 meta={"unit": vsym.meta.get("unit", "1"), "internal": True, "value": 0.0})
                    mapping[d.uid] = (vsym - prev) / dt
                    state_map[vname] = prev.name
            else:
                raise ModelError(f"알 수 없는 compile mode: {mode!r} (steady|transient)")
            memo: dict = {}
            for info in eqs:
                info.expr = S.substitute(info.expr, mapping, memo)

        # 차원 검사
        if check_dims:
            memo: dict = {}
            for info in eqs:
                try:
                    S.dim_of_expr(info.expr, memo)
                except DimensionError as exc:
                    raise ModelError(f"[{info.label}] 차원 오류\n  {exc}") from None

        estimates = self.estimate_starts() if propagate_starts else {}
        model = _build_compiled(self, eqs, scopes, estimates)
        model.mode = mode
        model.state_map = state_map
        return model


def _build_compiled(system: System, eqs: list[EquationInfo], scopes: dict[str, Scope],
                    estimates: dict[str, float] | None = None) -> "CompiledModel":
    syms = S.collect_syms([e.expr for e in eqs])
    var_syms = [s for s in syms if s.kind == "var"]
    par_syms = [s for s in syms if s.kind == "par"]
    der_syms = [s for s in syms if s.kind == "der"]
    var_syms.sort(key=lambda s: s.name)
    par_syms.sort(key=lambda s: s.name)
    der_syms.sort(key=lambda s: s.name)

    slots: dict[int, str] = {}
    for i, s in enumerate(var_syms):
        slots[s.uid] = f"x[{i}]"
    for j, s in enumerate(par_syms):
        slots[s.uid] = f"p[{j}]"
    for k, s in enumerate(der_syms):
        slots[s.uid] = f"d[{k}]"

    # 변수 메타데이터 (start/bounds 는 컴포넌트 선언에서 가져온다)
    var_meta: dict[str, tuple[str, float, float, float]] = {}
    for cname, comp in system.components.items():
        for vn, spec in comp.var_specs().items():
            var_meta[f"{cname}.{vn}"] = (spec.unit, spec.start, spec.lo, spec.hi)
        for pn, pspec in comp.port_specs().items():
            for vn, unit in pspec.all_vars():
                lo, hi = pspec.bound_of(vn)
                var_meta[f"{cname}.{pn}.{vn}"] = (unit, pspec.start_of(vn), lo, hi)

    estimates = estimates or {}
    variables = []
    for i, s in enumerate(var_syms):
        unit, start, lo, hi = var_meta.get(s.name, (s.meta.get("unit", "1"), 1.0, -np.inf, np.inf))
        est = estimates.get(s.name)
        if est is not None and np.isfinite(est):
            start = float(np.clip(est, lo, hi))
        variables.append(VarInfo(i, s.name, unit, start, lo, hi))

    parameters = []
    for j, s in enumerate(par_syms):
        if s.meta.get("internal"):
            parameters.append(
                ParInfo(j, s.name, s.meta.get("unit", "1"), float(s.meta.get("value", 0.0)),
                        -np.inf, np.inf, False, "적분기 내부 파라미터")
            )
            continue
        cname, pn = s.name.rsplit(".", 1)
        comp = system.components[cname]
        spec = comp.param_specs()[pn]
        parameters.append(
            ParInfo(j, s.name, spec.unit, comp.param_value(pn),
                    to_si(spec.lo, spec.unit) if np.isfinite(spec.lo) else -np.inf,
                    to_si(spec.hi, spec.unit) if np.isfinite(spec.hi) else np.inf,
                    spec.tunable, spec.desc)
        )

    var_index = {s.uid: i for i, s in enumerate(var_syms)}
    incidence: list[set[int]] = []
    for info in eqs:
        idx = {var_index[s.uid] for s in S.collect_syms([info.expr]) if s.kind == "var"}
        incidence.append(idx)

    report = analyze(incidence, len(var_syms))

    outputs: dict[str, tuple[S.Expr, str]] = {}
    for cname, comp in system.components.items():
        for oname, (expr, unit) in comp.outputs(scopes[cname]).items():
            outputs[f"{cname}.{oname}"] = (expr, unit)

    return CompiledModel(
        system=system, equations=eqs, variables=variables, parameters=parameters,
        var_syms=var_syms, par_syms=par_syms, der_syms=der_syms,
        slots=slots, incidence=incidence, report=report, outputs=outputs,
    )


@dataclass
class CompiledModel:
    system: System
    equations: list[EquationInfo]
    variables: list[VarInfo]
    parameters: list[ParInfo]
    var_syms: list[S.Sym]
    par_syms: list[S.Sym]
    der_syms: list[S.Sym]
    slots: dict[int, str]
    incidence: list[set[int]]
    report: StructuralReport
    outputs: dict[str, tuple[S.Expr, str]]
    _block_fns: list["BlockFns"] = field(default_factory=list)
    _residual_fn: Callable | None = None
    _output_fn: Callable | None = None
    mode: str = "steady"
    state_map: dict[str, str] = field(default_factory=dict)
    _vidx: dict[str, int] = field(default_factory=dict)
    _pidx: dict[str, int] = field(default_factory=dict)

    #: 일관 초기화에 쓰는 '거의 0' 스텝 배율.
    INIT_DT_FRACTION: float = 1e-8

    def set_discretization(self, p: np.ndarray, x_prev: np.ndarray, dt: float,
                           first_step: bool = False) -> None:
        """과도해석 한 스텝의 ``_dt`` 와 ``_prev.*`` 파라미터를 채운다.

        첫 스텝은 **일관 초기화(consistent initialization)** 다. dt 를 아주 작게 두면
        상태변수는 주어진 초기값에 고정되고 대수방정식만 풀린다. dt 를 무한대로 두면
        (정상상태 초기화) 사용자가 준 초기조건이 지워진다 - 의도가 아니다.
        """
        if self.mode != "transient":
            return
        p[self.par_index("_dt")] = (
            max(float(dt) * self.INIT_DT_FRACTION, 1e-12) if first_step else float(dt))
        for vname, pname in self.state_map.items():
            p[self.par_index(pname)] = float(x_prev[self.var_index(vname)])

    # --- 조회 헬퍼 ---------------------------------------------------------
    @property
    def n_vars(self) -> int:
        return len(self.variables)

    @property
    def n_eqs(self) -> int:
        return len(self.equations)

    def var_index(self, name: str) -> int:
        if not self._vidx:
            self._vidx = {v.name: v.index for v in self.variables}
        try:
            return self._vidx[name]
        except KeyError:
            raise KeyError(f"미지수 {name!r} 를 찾을 수 없습니다") from None

    def par_index(self, name: str) -> int:
        if not self._pidx:
            self._pidx = {q.name: q.index for q in self.parameters}
        try:
            return self._pidx[name]
        except KeyError:
            raise KeyError(f"파라미터 {name!r} 를 찾을 수 없습니다") from None

    def tunable_params(self) -> list[ParInfo]:
        return [p for p in self.parameters if p.tunable]

    def p0(self) -> np.ndarray:
        return np.array([p.value for p in self.parameters], dtype=float)

    def x0(self) -> np.ndarray:
        return np.array([v.start for v in self.variables], dtype=float)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([v.lo for v in self.variables], dtype=float)
        hi = np.array([v.hi for v in self.variables], dtype=float)
        return lo, hi

    # --- 코드 생성 (지연) ---------------------------------------------------
    def build(self) -> "CompiledModel":
        if self._residual_fn is None:
            self._residual_fn = S.compile_function(
                [e.expr for e in self.equations], self.slots, "_residual", ("x", "p")
            )
        if self._output_fn is None and self.outputs:
            self._output_fn = S.compile_function(
                [e for e, _ in self.outputs.values()], self.slots, "_outputs", ("x", "p")
            )
        if not self._block_fns and self.report.ok:
            memo: dict = {}
            for b in self.report.blocks:
                exprs = [self.equations[i].expr for i in b.eqs]
                fb = S.compile_function(exprs, self.slots, "_fb", ("x", "p"))
                # 희소 야코비안: 구조적으로 등장하는 (식, 미지수) 쌍만 미분한다.
                # 160x160 블록이라면 25600 개가 아니라 450 개만 생성된다.
                pos = {gv: k for k, gv in enumerate(b.vars)}
                rows: list[int] = []
                cols: list[int] = []
                jac_exprs: list[S.Expr] = []
                for i, ei in enumerate(b.eqs):
                    for gv in sorted(self.incidence[ei]):
                        k = pos.get(gv)
                        if k is None:
                            continue
                        d = S.diff(self.equations[ei].expr, self.var_syms[gv], memo)
                        if isinstance(d, S.Const) and d.value == 0.0:
                            continue
                        rows.append(i)
                        cols.append(k)
                        jac_exprs.append(d)
                jb = S.compile_function(jac_exprs, self.slots, "_jb", ("x", "p"))
                self._block_fns.append(
                    BlockFns(fb, jb, np.array(rows, dtype=int), np.array(cols, dtype=int),
                             np.array(b.eqs, dtype=int), np.array(b.vars, dtype=int))
                )
        return self

    def jacobian_nnz(self) -> int:
        self.build()
        return sum(len(bf.jac_rows) for bf in self._block_fns)

    def residual(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        if self._residual_fn is None:
            self.build()
        return self._residual_fn(x, p)  # type: ignore[misc]

    def output_values_si(self, x: np.ndarray, p: np.ndarray) -> dict[str, float]:
        """파생량을 SI 단위 그대로 반환. 관측값 비교는 항상 SI 로 한다."""
        if not self.outputs:
            return {}
        if self._output_fn is None:
            self.build()
        raw = self._output_fn(x, p)  # type: ignore[misc]
        return {name: float(val) for name, val in zip(self.outputs, raw)}

    def output_values(self, x: np.ndarray, p: np.ndarray) -> dict[str, float]:
        if not self.outputs:
            return {}
        if self._output_fn is None:
            self.build()
        raw = self._output_fn(x, p)  # type: ignore[misc]
        return {
            name: from_si(float(val), unit)
            for (name, (_, unit)), val in zip(self.outputs.items(), raw)
        }

    def var_dict(self, x: np.ndarray, si: bool = False) -> dict[str, float]:
        if si:
            return {v.name: float(x[v.index]) for v in self.variables}
        return {v.name: from_si(float(x[v.index]), v.unit) for v in self.variables}

    # --- 진단 리포트 --------------------------------------------------------
    def describe(self) -> str:
        r = self.report
        lines = [
            f"모델: {self.system.name}",
            f"  컴포넌트 {len(self.system.components)}개, 연결 {len(self.system.connections)}개",
            f"  방정식 {r.n_eqs}개, 미지수 {r.n_vars}개, 파라미터 {len(self.parameters)}개",
        ]
        if r.ok:
            sizes: dict[int, int] = {}
            for b in r.blocks:
                sizes[b.size] = sizes.get(b.size, 0) + 1
            shape = ", ".join(f"{n}x{k}" for k, n in sorted(sizes.items()))
            lines.append(f"  구조: 정상 (BLT {len(r.blocks)}블록, 크기분포 {shape}, 최대 {r.largest_block})")
        else:
            lines.append("  구조: 문제 있음")
            if r.n_eqs != r.n_vars:
                lines.append(f"    ! 방정식/미지수 개수 불일치 ({r.n_eqs} vs {r.n_vars})")
            if r.underdetermined_vars:
                lines.append(f"    ! 부족결정: 아래 {len(r.underdetermined_vars)}개 미지수를 결정할 식이 부족합니다")
                for vi in r.underdetermined_vars[:12]:
                    v = self.variables[vi]
                    lines.append(f"        - {v.name} [{v.unit}]")
                if len(r.underdetermined_vars) > 12:
                    lines.append(f"        ... 외 {len(r.underdetermined_vars)-12}개")
            if r.overdetermined_eqs:
                lines.append(f"    ! 과결정: 아래 {len(r.overdetermined_eqs)}개 방정식이 서로 중복/모순입니다")
                for ei in r.overdetermined_eqs[:12]:
                    lines.append(f"        - {self.equations[ei].label}")
                if len(r.overdetermined_eqs) > 12:
                    lines.append(f"        ... 외 {len(r.overdetermined_eqs)-12}개")
        return "\n".join(lines)
