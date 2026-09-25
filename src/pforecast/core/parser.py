"""방정식 텍스트 -> 식 그래프 파서.

이 모듈이 있어야 툴이 "NOx 전용 도구 + 라이브러리" 에서 **범용 방정식 기반 모델링
환경**이 된다. 파이썬을 건드리지 않고 YAML 에 지배방정식을 그대로 적을 수 있다.

    equations:
      mass:  "a.mdot + b.mdot = 0"
      momentum: "a.p - b.p = K * signed_pow(a.mdot, 2) / rho"
      energy:   "mdot_eff * cp * (b.T - a.T) = Q"

파싱 결과는 손으로 짠 컴포넌트와 **완전히 같은 식 그래프**다. 해석적 미분도,
차원 검사도, 구조 해석도 그대로 적용된다. 성능 손해도 없다 (파싱은 조립 때 한 번).

수치 리터럴에 단위를 달 수 있다: ``9.80665[m/s^2]``. 이렇게 적어야 차원 검사가
산다. 맨 숫자는 무차원으로 취급된다.

core 계층이므로 **도메인 함수는 모른다.** 유체/열 관련 함수(density, enthalpy ...)는
lib 이 ``FunctionTable`` 에 얹는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from . import symbolic as S
from .units import dim_of, to_si


class EquationSyntaxError(ValueError):
    """방정식 문법 오류. 어디가 문제인지 위치를 표시한다."""

    def __init__(self, message: str, text: str, pos: int, context: str = ""):
        caret = " " * pos + "^"
        where = f" [{context}]" if context else ""
        super().__init__(f"{message}{where}\n    {text}\n    {caret}")
        self.text = text
        self.pos = pos


_NUMBER = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*")
_UNIT = re.compile(r"\[([^\]]*)\]")
_OPS = set("+-*/^(),=")


@dataclass
class Token:
    kind: str          # "num" | "name" | "op" | "end"
    value: Any
    pos: int
    unit: str | None = None


def tokenize(text: str) -> list[Token]:
    out: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "#":                      # 줄 끝 주석
            break
        m = _NUMBER.match(text, i)
        if m and (ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit())):
            i = m.end()
            unit = None
            um = _UNIT.match(text, i)
            if um:
                unit = um.group(1).strip()
                i = um.end()
            out.append(Token("num", float(m.group()), m.start(), unit))
            continue
        m = _NAME.match(text, i)
        if m:
            out.append(Token("name", m.group(), m.start()))
            i = m.end()
            continue
        if ch in _OPS:
            out.append(Token("op", ch, i))
            i += 1
            continue
        raise EquationSyntaxError(f"해석할 수 없는 문자 {ch!r}", text, i)
    out.append(Token("end", None, n))
    return out


#: 식(Expr) 을 받는 순수 수학 함수. lib 이 여기에 도메인 함수를 추가한다.
MATH_FUNCTIONS: dict[str, Callable[..., S.Expr]] = {
    "exp": S.exp,
    "log": S.log,
    "ln": S.log,
    "sqrt": S.sqrt,
    "sin": S.sin,
    "cos": S.cos,
    "tanh": S.tanh,
    "abs": lambda x, eps=1e-6: S.smooth_abs(x, float(_as_number(eps, "abs"))),
    "signed_pow": lambda x, n: S.signed_pow(x, float(_as_number(n, "signed_pow"))),
    "max": S.smooth_max,
    "min": S.smooth_min,
    "sq": lambda x: x * x,
}

CONSTANTS: dict[str, float] = {"pi": 3.141592653589793, "e": 2.718281828459045}


def _as_number(e: Any, fn: str) -> float:
    """상수여야 하는 인자(지수 등)를 숫자로 꺼낸다."""
    if isinstance(e, (int, float)):
        return float(e)
    if isinstance(e, S.Const):
        return e.value
    raise EquationSyntaxError(
        f"{fn}() 의 이 인자는 상수여야 합니다 (변수는 쓸 수 없습니다)", "", 0)


@dataclass
class FunctionTable:
    """파서가 쓸 수 있는 함수 모음. 계층별로 얹어서 쓴다."""

    functions: dict[str, Callable[..., Any]] = field(default_factory=lambda: dict(MATH_FUNCTIONS))

    def extend(self, extra: dict[str, Callable[..., Any]]) -> "FunctionTable":
        merged = dict(self.functions)
        merged.update(extra)
        return FunctionTable(merged)

    def __contains__(self, name: str) -> bool:
        return name in self.functions

    def __getitem__(self, name: str) -> Callable[..., Any]:
        return self.functions[name]

    @property
    def names(self) -> list[str]:
        return sorted(self.functions)


class _Parser:
    """우선순위 등반(precedence climbing) 파서."""

    def __init__(self, text: str, resolve, resolve_der=None,
                 functions: FunctionTable | None = None, context: str = ""):
        self.text = text
        self.tokens = tokenize(text)
        self.i = 0
        self.resolve = resolve
        self.resolve_der = resolve_der
        self.fn = functions or FunctionTable()
        self.context = context

    # --- 토큰 조작 ---
    @property
    def cur(self) -> Token:
        return self.tokens[self.i]

    def advance(self) -> Token:
        t = self.tokens[self.i]
        self.i += 1
        return t

    def expect_op(self, op: str) -> None:
        t = self.cur
        if t.kind != "op" or t.value != op:
            self.fail(f"{op!r} 가 필요합니다")
        self.advance()

    def fail(self, message: str):
        raise EquationSyntaxError(message, self.text, self.cur.pos, self.context)

    # --- 문법 ---
    def parse_equation(self) -> S.Expr:
        """``lhs = rhs`` 를 잔차 ``lhs - rhs`` 로 만든다. ``=`` 가 없으면 식 자체가 잔차."""
        lhs = self.parse_expr(0)
        if self.cur.kind == "op" and self.cur.value == "=":
            self.advance()
            rhs = self.parse_expr(0)
            if self.cur.kind == "op" and self.cur.value == "=":
                self.fail("등호는 하나만 쓸 수 있습니다")
            expr = lhs - rhs
        else:
            expr = lhs
        if self.cur.kind != "end":
            self.fail("식이 끝난 뒤에 남는 내용이 있습니다")
        return expr

    _BIN = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}

    def parse_expr(self, min_prec: int) -> S.Expr:
        left = self.parse_unary()
        while True:
            t = self.cur
            if t.kind != "op" or t.value not in self._BIN:
                break
            prec = self._BIN[t.value]
            if prec < min_prec:
                break
            op = t.value
            self.advance()
            # ^ 는 우결합, 나머지는 좌결합
            right = self.parse_expr(prec if op == "^" else prec + 1)
            left = self._apply(op, left, right)
        return left

    @staticmethod
    def _apply(op: str, a: S.Expr, b: S.Expr) -> S.Expr:
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b
        return a ** b

    def parse_unary(self) -> S.Expr:
        t = self.cur
        if t.kind == "op" and t.value in "+-":
            self.advance()
            v = self.parse_unary()
            return -v if t.value == "-" else v
        return self.parse_atom()

    def parse_atom(self) -> S.Expr:
        t = self.advance()
        if t.kind == "num":
            if t.unit:
                try:
                    return S.const(to_si(t.value, t.unit), dim_of(t.unit))
                except ValueError as exc:
                    raise EquationSyntaxError(str(exc), self.text, t.pos, self.context) from None
            return S.const(t.value)
        if t.kind == "op" and t.value == "(":
            e = self.parse_expr(0)
            if not (self.cur.kind == "op" and self.cur.value == ")"):
                self.fail("')' 가 필요합니다")
            self.advance()
            return e
        if t.kind == "name":
            if self.cur.kind == "op" and self.cur.value == "(":
                return self.parse_call(t)
            if t.value in CONSTANTS:
                return S.const(CONSTANTS[t.value])
            return self._resolve_name(t)
        self.i -= 1
        self.fail("식이 와야 할 자리입니다")

    def parse_call(self, name_tok: Token) -> S.Expr:
        name = name_tok.value
        self.expect_op("(")
        # der(x) 는 이름 자체가 필요하므로 특별 취급한다
        if name == "der":
            if self.cur.kind != "name":
                self.fail("der() 의 인자는 변수 이름이어야 합니다")
            vname = self.advance().value
            self.expect_op(")")
            if self.resolve_der is None:
                raise EquationSyntaxError("이 문맥에서는 der() 를 쓸 수 없습니다",
                                          self.text, name_tok.pos, self.context)
            try:
                return self.resolve_der(vname)
            except KeyError as exc:
                raise EquationSyntaxError(f"der() 의 대상 {vname!r} 는 선언된 변수가 아닙니다",
                                          self.text, name_tok.pos, self.context) from None
        args: list[Any] = []
        if not (self.cur.kind == "op" and self.cur.value == ")"):
            while True:
                args.append(self._parse_arg())
                if self.cur.kind == "op" and self.cur.value == ",":
                    self.advance()
                    continue
                break
        self.expect_op(")")
        if name not in self.fn:
            raise EquationSyntaxError(
                f"알 수 없는 함수 {name!r}. 쓸 수 있는 함수: {', '.join(self.fn.names)}",
                self.text, name_tok.pos, self.context)
        try:
            return self.fn[name](*args)
        except EquationSyntaxError:
            raise
        except TypeError as exc:
            raise EquationSyntaxError(f"{name}() 인자가 맞지 않습니다: {exc}",
                                      self.text, name_tok.pos, self.context) from None

    def _parse_arg(self):
        """함수 인자. 포트 이름 하나만 온 경우는 포트 객체 그대로 넘긴다."""
        if self.cur.kind == "name":
            nxt = self.tokens[self.i + 1]
            is_bare = nxt.kind == "op" and nxt.value in ",)"
            if is_bare:
                tok = self.advance()
                resolved = self.resolve(tok.value)
                if resolved is None:
                    raise EquationSyntaxError(
                        f"{tok.value!r} 를 찾을 수 없습니다", self.text, tok.pos, self.context)
                return resolved
        return self.parse_expr(0)

    def _resolve_name(self, tok: Token) -> S.Expr:
        try:
            resolved = self.resolve(tok.value)
        except (KeyError, AttributeError) as exc:
            raise EquationSyntaxError(str(exc).strip("'\""), self.text, tok.pos,
                                      self.context) from None
        if resolved is None:
            raise EquationSyntaxError(f"{tok.value!r} 를 찾을 수 없습니다",
                                      self.text, tok.pos, self.context)
        if not isinstance(resolved, S.Expr):
            raise EquationSyntaxError(
                f"{tok.value!r} 는 식에 직접 쓸 수 없습니다 (함수 인자로만 쓰입니다)",
                self.text, tok.pos, self.context)
        return resolved


def parse_equation(text: str, resolve, resolve_der=None,
                   functions: FunctionTable | None = None, context: str = "") -> S.Expr:
    """``"a.p - b.p = dp"`` 를 잔차 식으로 만든다."""
    return _Parser(text, resolve, resolve_der, functions, context).parse_equation()


def parse_expression(text: str, resolve, functions: FunctionTable | None = None,
                     context: str = "") -> S.Expr:
    """등호 없는 순수 식. outputs 정의 등에 쓴다."""
    p = _Parser(text, resolve, None, functions, context)
    e = p.parse_expr(0)
    if p.cur.kind != "end":
        p.fail("식이 끝난 뒤에 남는 내용이 있습니다")
    return e
