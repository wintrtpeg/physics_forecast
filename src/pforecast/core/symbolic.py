"""최소 심볼릭 엔진: 식 그래프 -> 해석적 미분 -> numpy 코드 생성.

왜 직접 만드는가
----------------
사내 PC에서 torch/jax/casadi 설치를 장담할 수 없다. 그런데 물리 모델을 제대로
풀려면 정확한 야코비안이 필요하다(수치미분은 느리고 강성 문제에서 수렴이 깨진다).
그래서 numpy 만으로 동작하는 작은 심볼릭 코어를 둔다. 이 모듈이 툴 전체의 심장이다.

특징
----
* hash-consing: 구조가 같은 부분식은 같은 객체로 공유된다 -> 자동 CSE.
* 스마트 생성자: 0/1 항등원 정리를 생성 시점에 적용해 그래프가 부풀지 않는다.
* ``diff`` 는 해석적 미분. 희소 야코비안을 구조 정보와 함께 만들 수 있다.
* ``compile_function`` 은 파이썬 소스를 생성해 ``exec`` 한다. 호출 오버헤드가 거의 없다.
* 각 심볼은 차원을 가지며 ``dim_of_expr`` 로 방정식 동차성을 사전 검사한다.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Sequence

from .units import DIMLESS, Dim, DimensionError, dim_div, dim_mul, dim_pow, dim_str

__all__ = [
    "Expr", "Const", "Sym", "Op", "sym", "const",
    "exp", "log", "sqrt", "sin", "cos", "tanh", "smooth_abs", "signed_pow",
    "smooth_max", "smooth_min", "diff", "dim_of_expr", "compile_function",
    "collect_syms", "expr_to_str", "clear_cache", "substitute",
]

_CACHE: dict[tuple, "Expr"] = {}
_NEXT_ID = [0]


def clear_cache() -> None:
    """장시간 실행 세션에서 그래프 캐시를 비운다."""
    _CACHE.clear()


def _new_id() -> int:
    _NEXT_ID[0] += 1
    return _NEXT_ID[0]


class Expr:
    """식 그래프의 노드. 해시콘싱되므로 ``is`` 비교가 구조 비교와 같다."""

    __slots__ = ("uid", "_hash")

    def __hash__(self) -> int:  # noqa: D105
        return self._hash

    # --- 연산자 오버로딩 -------------------------------------------------
    def __add__(self, other): return _add(self, _wrap(other))
    def __radd__(self, other): return _add(_wrap(other), self)
    def __sub__(self, other): return _add(self, _neg(_wrap(other)))
    def __rsub__(self, other): return _add(_wrap(other), _neg(self))
    def __mul__(self, other): return _mul(self, _wrap(other))
    def __rmul__(self, other): return _mul(_wrap(other), self)
    def __truediv__(self, other): return _div(self, _wrap(other))
    def __rtruediv__(self, other): return _div(_wrap(other), self)
    def __pow__(self, other): return _pow(self, _wrap(other))
    def __rpow__(self, other): return _pow(_wrap(other), self)
    def __neg__(self): return _neg(self)
    def __pos__(self): return self

    def __repr__(self) -> str:  # noqa: D105
        return expr_to_str(self)


class Const(Expr):
    __slots__ = ("value", "dim")

    def __init__(self, value: float, dim: Dim | None = None):
        self.value = float(value)
        #: ``None`` 이면 "주변 차원에 맞춰지는" 순수 수치 리터럴로 본다.
        self.dim = dim
        self.uid = _new_id()
        self._hash = hash(("c", self.value, dim))


class Sym(Expr):
    """미지수/파라미터/입력을 가리키는 잎 노드."""

    __slots__ = ("name", "dim", "kind", "meta")

    def __init__(self, name: str, dim: Dim = DIMLESS, kind: str = "var", meta: dict | None = None):
        self.name = name
        self.dim = dim
        self.kind = kind  # "var" | "par" | "input" | "der"
        self.meta = meta or {}
        self.uid = _new_id()
        self._hash = hash(("s", name, kind))


class Op(Expr):
    __slots__ = ("op", "args")

    def __init__(self, op: str, args: tuple[Expr, ...]):
        self.op = op
        self.args = args
        self.uid = _new_id()
        self._hash = hash(("o", op, tuple(a.uid for a in args)))


def _wrap(x: Any) -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, (int, float)):
        return const(float(x))
    raise TypeError(f"식으로 변환할 수 없는 값: {x!r}")


def const(value: float, dim: Dim | None = None) -> Const:
    key = ("c", float(value), dim)
    e = _CACHE.get(key)
    if e is None:
        e = Const(value, dim)
        _CACHE[key] = e
    return e  # type: ignore[return-value]


ZERO = const(0.0)
ONE = const(1.0)


def sym(name: str, dim: Dim = DIMLESS, kind: str = "var", meta: dict | None = None) -> Sym:
    key = ("s", name, kind)
    e = _CACHE.get(key)
    if e is None:
        e = Sym(name, dim, kind, meta)
        _CACHE[key] = e
    return e  # type: ignore[return-value]


def _op(op: str, *args: Expr) -> Expr:
    key = ("o", op, tuple(a.uid for a in args))
    e = _CACHE.get(key)
    if e is None:
        e = Op(op, tuple(args))
        _CACHE[key] = e
    return e


def _is_const(e: Expr, v: float | None = None) -> bool:
    return isinstance(e, Const) and (v is None or e.value == v)


# --- 스마트 생성자 --------------------------------------------------------

def _dimensionless_const(e: Expr, v: float) -> bool:
    """값이 v 인 **무차원** 상수인가. 차원을 가진 상수는 약분하면 안 된다."""
    return (isinstance(e, Const) and e.value == v
            and (e.dim is None or e.dim == DIMLESS))


def _canonical(a: Expr, b: Expr) -> tuple[Expr, Expr]:
    """교환법칙이 성립하는 연산의 인자 순서를 고정한다 (상수 먼저, 그다음 uid).

    ``a + 1`` 과 ``1 + a`` 가 같은 노드가 되어야 해시콘싱이 제 일을 한다.
    """
    if isinstance(b, Const) and not isinstance(a, Const):
        return b, a
    if isinstance(a, Const) == isinstance(b, Const) and b.uid < a.uid:
        return b, a
    return a, b


def _add(a: Expr, b: Expr) -> Expr:
    if _is_const(a) and _is_const(b):
        return const(a.value + b.value, a.dim or b.dim)  # type: ignore[attr-defined]
    if _dimensionless_const(a, 0.0):
        return b
    if _dimensionless_const(b, 0.0):
        return a
    if a is b:
        return _mul(const(2.0), a)
    return _op("+", *_canonical(a, b))


def _neg(a: Expr) -> Expr:
    if _is_const(a):
        return const(-a.value, a.dim)  # type: ignore[attr-defined]
    if isinstance(a, Op) and a.op == "neg":
        return a.args[0]
    return _op("neg", a)


def _mul(a: Expr, b: Expr) -> Expr:
    if _is_const(a) and _is_const(b):
        d = None if (a.dim is None and b.dim is None) else dim_mul(a.dim or DIMLESS, b.dim or DIMLESS)  # type: ignore[attr-defined]
        return const(a.value * b.value, d)  # type: ignore[attr-defined]
    if _dimensionless_const(a, 0.0) or _dimensionless_const(b, 0.0):
        return ZERO
    if _dimensionless_const(a, 1.0):
        return b
    if _dimensionless_const(b, 1.0):
        return a
    return _op("*", *_canonical(a, b))


def _div(a: Expr, b: Expr) -> Expr:
    if _dimensionless_const(b, 1.0):
        return a
    if _is_const(a) and _is_const(b) and b.value != 0.0:  # type: ignore[attr-defined]
        d = None if (a.dim is None and b.dim is None) else dim_div(a.dim or DIMLESS, b.dim or DIMLESS)  # type: ignore[attr-defined]
        return const(a.value / b.value, d)  # type: ignore[attr-defined]
    if _dimensionless_const(a, 0.0):
        return ZERO
    if a is b:
        return ONE
    return _op("/", a, b)


def _pow(a: Expr, b: Expr) -> Expr:
    if _is_const(b, 1.0):
        return a
    if _is_const(b, 0.0):
        return ONE
    if _is_const(a) and _is_const(b):
        return const(a.value ** b.value)  # type: ignore[attr-defined]
    return _op("^", a, b)


def _unary(name: str) -> Callable[[Any], Expr]:
    def f(x: Any) -> Expr:
        e = _wrap(x)
        if _is_const(e):
            fn = getattr(math, name)
            return const(fn(e.value))  # type: ignore[attr-defined]
        return _op(name, e)
    f.__name__ = name
    return f


exp = _unary("exp")
log = _unary("log")
sqrt = _unary("sqrt")
sin = _unary("sin")
cos = _unary("cos")
tanh = _unary("tanh")


def smooth_abs(x: Any, eps: float = 1e-6) -> Expr:
    """|x| 의 매끄러운 근사. 뉴턴법이 원점에서 깨지지 않게 한다."""
    e = _wrap(x)
    return sqrt(e * e + const(eps * eps))


def signed_pow(x: Any, n: float, eps: float = 1e-6) -> Expr:
    """sign(x)*|x|^n. 덕트 저항 dp = K*mdot*|mdot| 같은 양방향 유동식에 쓴다."""
    e = _wrap(x)
    return e * smooth_abs(e, eps) ** const(n - 1.0)


def smooth_max(a: Any, b: Any, eps: float = 1e-3) -> Expr:
    ea, eb = _wrap(a), _wrap(b)
    return const(0.5) * (ea + eb + smooth_abs(ea - eb, eps))


def smooth_min(a: Any, b: Any, eps: float = 1e-3) -> Expr:
    ea, eb = _wrap(a), _wrap(b)
    return const(0.5) * (ea + eb - smooth_abs(ea - eb, eps))


# --- 미분 ----------------------------------------------------------------

def diff(e: Expr, x: Sym, _memo: dict | None = None) -> Expr:
    """``e`` 를 심볼 ``x`` 로 해석적 미분."""
    memo = _memo if _memo is not None else {}
    key = (e.uid, x.uid)
    hit = memo.get(key)
    if hit is not None:
        return hit
    r = _diff_impl(e, x, memo)
    memo[key] = r
    return r


def _diff_impl(e: Expr, x: Sym, memo: dict) -> Expr:
    if isinstance(e, Const):
        return ZERO
    if isinstance(e, Sym):
        return ONE if e is x else ZERO
    assert isinstance(e, Op)
    op, args = e.op, e.args
    d = lambda a: diff(a, x, memo)  # noqa: E731
    if op == "+":
        return _add(d(args[0]), d(args[1]))
    if op == "neg":
        return _neg(d(args[0]))
    if op == "*":
        a, b = args
        return _add(_mul(d(a), b), _mul(a, d(b)))
    if op == "/":
        a, b = args
        return _div(_add(d(a), _neg(_div(_mul(a, d(b)), b))), b)
    if op == "^":
        a, b = args
        if isinstance(b, Const):
            return _mul(_mul(b, _pow(a, const(b.value - 1.0))), d(a))
        # 일반형: a^b * (b' ln a + b a'/a)
        return _mul(e, _add(_mul(d(b), log(a)), _div(_mul(b, d(a)), a)))
    if op == "exp":
        return _mul(e, d(args[0]))
    if op == "log":
        return _div(d(args[0]), args[0])
    if op == "sqrt":
        return _div(d(args[0]), _mul(const(2.0), e))
    if op == "sin":
        return _mul(cos(args[0]), d(args[0]))
    if op == "cos":
        return _neg(_mul(sin(args[0]), d(args[0])))
    if op == "tanh":
        return _mul(_add(ONE, _neg(_mul(e, e))), d(args[0]))
    raise NotImplementedError(f"미분 규칙이 없는 연산: {op}")


# --- 차원 추론 ------------------------------------------------------------

_TRANSCENDENTAL = {"exp", "log", "sin", "cos", "tanh"}


def dim_of_expr(e: Expr, _memo: dict | None = None) -> Dim | None:
    """식의 차원을 추론한다. ``None`` 은 '수치 리터럴이라 아직 자유롭다'는 뜻."""
    memo = _memo if _memo is not None else {}
    hit = memo.get(e.uid, "miss")
    if hit != "miss":
        return hit  # type: ignore[return-value]
    r = _dim_impl(e, memo)
    memo[e.uid] = r
    return r


def _dim_impl(e: Expr, memo: dict) -> Dim | None:
    if isinstance(e, Const):
        return e.dim
    if isinstance(e, Sym):
        return e.dim
    assert isinstance(e, Op)
    op, args = e.op, e.args
    ds = [dim_of_expr(a, memo) for a in args]
    if op in ("+",):
        a, b = ds
        if a is None:
            return b
        if b is None:
            return a
        if a != b:
            raise DimensionError(
                f"덧셈의 차원이 다릅니다: [{dim_str(a)}] + [{dim_str(b)}]  <-  {expr_to_str(e)}"
            )
        return a
    if op == "neg":
        return ds[0]
    if op == "*":
        if ds[0] is None and ds[1] is None:
            return None
        return dim_mul(ds[0] or DIMLESS, ds[1] or DIMLESS)
    if op == "/":
        if ds[0] is None and ds[1] is None:
            return None
        return dim_div(ds[0] or DIMLESS, ds[1] or DIMLESS)
    if op == "^":
        base, ex = args
        if not isinstance(ex, Const):
            return DIMLESS  # 지수가 변수면 밑은 무차원이어야 한다
        if ds[0] is None:
            return None
        return dim_pow(ds[0], ex.value)
    if op == "sqrt":
        if ds[0] is None:
            return None
        return dim_pow(ds[0], 0.5)
    if op in _TRANSCENDENTAL:
        d0 = ds[0]
        if d0 is not None and d0 != DIMLESS:
            raise DimensionError(
                f"{op}() 의 인자는 무차원이어야 합니다: [{dim_str(d0)}]  <-  {expr_to_str(e)}"
            )
        return DIMLESS
    raise NotImplementedError(f"차원 규칙이 없는 연산: {op}")


def substitute(e: Expr, mapping: dict[int, Expr], _memo: dict | None = None) -> Expr:
    """``mapping`` 은 ``Sym.uid -> 대체식``. 과도해석에서 der(x) 를 이산화식으로 바꿀 때 쓴다."""
    memo = _memo if _memo is not None else {}
    hit = memo.get(e.uid)
    if hit is not None:
        return hit
    if isinstance(e, Const):
        r: Expr = e
    elif isinstance(e, Sym):
        r = mapping.get(e.uid, e)
    else:
        assert isinstance(e, Op)
        new_args = tuple(substitute(a, mapping, memo) for a in e.args)
        if all(a is b for a, b in zip(new_args, e.args)):
            r = e
        elif e.op in _BIN or e.op == "neg":
            if e.op == "+":
                r = _add(*new_args)
            elif e.op == "*":
                r = _mul(*new_args)
            elif e.op == "/":
                r = _div(*new_args)
            elif e.op == "^":
                r = _pow(*new_args)
            else:
                r = _neg(new_args[0])
        else:
            r = _op(e.op, *new_args)
    memo[e.uid] = r
    return r


# --- 순회 / 출력 ----------------------------------------------------------

def collect_syms(exprs: Iterable[Expr]) -> list[Sym]:
    """식들에 등장하는 심볼을 등장 순서대로 모은다."""
    seen: dict[int, Sym] = {}
    visited: set[int] = set()
    stack = list(exprs)
    order: list[Sym] = []
    while stack:
        e = stack.pop()
        if e.uid in visited:
            continue
        visited.add(e.uid)
        if isinstance(e, Sym):
            if e.uid not in seen:
                seen[e.uid] = e
                order.append(e)
        elif isinstance(e, Op):
            stack.extend(e.args)
    return order


_BIN = {"+": "+", "*": "*", "/": "/", "^": "**"}


def expr_to_str(e: Expr) -> str:
    if isinstance(e, Const):
        return f"{e.value:g}"
    if isinstance(e, Sym):
        return e.name
    assert isinstance(e, Op)
    if e.op in _BIN:
        a, b = (expr_to_str(x) for x in e.args)
        return f"({a} {_BIN[e.op]} {b})"
    if e.op == "neg":
        return f"(-{expr_to_str(e.args[0])})"
    inner = ", ".join(expr_to_str(a) for a in e.args)
    return f"{e.op}({inner})"


# --- 코드 생성 ------------------------------------------------------------

_NUMPY_FN = {
    "exp": "np.exp", "log": "np.log", "sqrt": "np.sqrt",
    "sin": "np.sin", "cos": "np.cos", "tanh": "np.tanh",
}


def _topo(exprs: Sequence[Expr]) -> list[Expr]:
    """중복 없는 후위 순회 순서."""
    out: list[Expr] = []
    state: dict[int, int] = {}
    for root in exprs:
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            st = state.get(node.uid, 0)
            if st == 2:
                continue
            if expanded:
                state[node.uid] = 2
                out.append(node)
                continue
            state[node.uid] = 1
            stack.append((node, True))
            if isinstance(node, Op):
                for a in node.args:
                    if state.get(a.uid, 0) != 2:
                        stack.append((a, False))
    return out


def compile_function(
    exprs: Sequence[Expr],
    slots: dict[int, str],
    name: str = "_pf_fn",
    arg_names: Sequence[str] = ("x", "p"),
) -> Callable[..., Any]:
    """식 리스트를 numpy 배열을 반환하는 파이썬 함수로 컴파일한다.

    ``slots`` 는 ``Sym.uid -> 'x[3]'`` 처럼 심볼이 어느 인자의 어느 위치인지 알려준다.
    공통 부분식은 해시콘싱 덕분에 자동으로 한 번만 계산된다.
    """
    lines = [f"def {name}({', '.join(arg_names)}):", "    _o = np.empty(%d)" % len(exprs)]
    tmp: dict[int, str] = {}
    for node in _topo(list(exprs)):
        if isinstance(node, Const):
            tmp[node.uid] = repr(node.value)
            continue
        if isinstance(node, Sym):
            slot = slots.get(node.uid)
            if slot is None:
                raise KeyError(f"심볼 {node.name!r} 에 대응하는 슬롯이 없습니다")
            tmp[node.uid] = slot
            continue
        assert isinstance(node, Op)
        a = [tmp[x.uid] for x in node.args]
        if node.op in _BIN:
            rhs = f"({a[0]} {_BIN[node.op]} {a[1]})"
        elif node.op == "neg":
            rhs = f"(-{a[0]})"
        else:
            rhs = f"{_NUMPY_FN[node.op]}({a[0]})"
        v = f"_t{len(tmp)}"
        lines.append(f"    {v} = {rhs}")
        tmp[node.uid] = v
    for i, e in enumerate(exprs):
        lines.append(f"    _o[{i}] = {tmp[e.uid]}")
    lines.append("    return _o")
    src = "\n".join(lines)
    ns: dict[str, Any] = {}
    import numpy as np  # 지역 import: 생성 코드의 네임스페이스에 주입
    exec(compile(src, f"<pforecast:{name}>", "exec"), {"np": np, "math": math}, ns)
    fn = ns[name]
    fn.__pf_source__ = src  # type: ignore[attr-defined]
    return fn
