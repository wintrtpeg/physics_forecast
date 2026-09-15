"""구조 해석: 매칭, Dulmage-Mendelsohn 분해, BLT 블록 분할.

이 모듈이 툴의 사용성을 좌우한다. 물리 모델을 짜다 보면 방정식이 하나 모자라거나
중복되는 일이 다반사인데, 그냥 뉴턴법에 던지면 "특이 행렬" 한 줄만 나오고 끝난다.
여기서는 풀기 **전에** 다음을 알려준다.

* 미지수는 몇 개이고 방정식은 몇 개인가
* 부족하면 어느 변수 묶음이 결정되지 않는가 (부족결정 블록)
* 과하면 어느 방정식 묶음이 서로 싸우는가 (과결정 블록)
* 정방이면 어떤 순서로 몇 개짜리 블록을 푸는가 (BLT)

BLT 분할은 진단만이 아니라 성능에도 직결된다. 60x60 을 한 번에 뉴턴으로 푸는 대신
1x1 스칼라 20개 + 6x6 블록 하나로 쪼개 풀면 훨씬 빠르고 훨씬 잘 수렴한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Block:
    """BLT 한 덩어리. 방정식 인덱스와 미지수 인덱스가 같은 개수로 짝지어져 있다."""

    eqs: list[int]
    vars: list[int]

    @property
    def size(self) -> int:
        return len(self.eqs)

    @property
    def is_scalar(self) -> bool:
        return len(self.eqs) == 1


@dataclass
class StructuralReport:
    n_eqs: int
    n_vars: int
    matching: list[int]                 # eq -> var (없으면 -1)
    blocks: list[Block] = field(default_factory=list)
    underdetermined_vars: list[int] = field(default_factory=list)
    underdetermined_eqs: list[int] = field(default_factory=list)
    overdetermined_eqs: list[int] = field(default_factory=list)
    overdetermined_vars: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.n_eqs == self.n_vars
            and not self.underdetermined_vars
            and not self.overdetermined_eqs
        )

    @property
    def largest_block(self) -> int:
        return max((b.size for b in self.blocks), default=0)


def maximum_matching(incidence: list[set[int]], n_vars: int) -> list[int]:
    """방정식 -> 미지수 최대 매칭 (증대경로법). 모델 규모가 작아 이것으로 충분하다."""
    match_eq = [-1] * len(incidence)
    match_var = [-1] * n_vars

    def try_assign(eq: int, seen: list[bool]) -> bool:
        for v in incidence[eq]:
            if seen[v]:
                continue
            seen[v] = True
            if match_var[v] == -1 or try_assign(match_var[v], seen):
                match_var[v] = eq
                match_eq[eq] = v
                return True
        return False

    import sys
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 10 * (len(incidence) + n_vars) + 1000))
    try:
        for eq in range(len(incidence)):
            if match_eq[eq] == -1:
                try_assign(eq, [False] * n_vars)
    finally:
        sys.setrecursionlimit(limit)
    return match_eq


def _alternating_from_vars(incidence, var_to_eqs, match_var, seeds):
    """짝 없는 미지수에서 교대경로로 도달 가능한 (미지수, 방정식) 묶음 = 부족결정 블록."""
    seen_v, seen_e = set(seeds), set()
    stack = list(seeds)
    while stack:
        v = stack.pop()
        for eq in var_to_eqs[v]:
            if eq in seen_e:
                continue
            seen_e.add(eq)
            nv = _matched_var(eq, match_var)
            if nv is not None and nv not in seen_v:
                seen_v.add(nv)
                stack.append(nv)
    return sorted(seen_v), sorted(seen_e)


def _matched_var(eq: int, match_var: list[int]) -> int | None:
    for v, e in enumerate(match_var):
        if e == eq:
            return v
    return None


def _alternating_from_eqs(incidence, match_eq, seeds):
    """짝 없는 방정식에서 도달 가능한 묶음 = 과결정 블록."""
    seen_e, seen_v = set(seeds), set()
    stack = list(seeds)
    while stack:
        eq = stack.pop()
        for v in incidence[eq]:
            if v in seen_v:
                continue
            seen_v.add(v)
            for e2, mv in enumerate(match_eq):
                if mv == v and e2 not in seen_e:
                    seen_e.add(e2)
                    stack.append(e2)
    return sorted(seen_e), sorted(seen_v)


def _tarjan_scc(n: int, succ: list[list[int]]) -> list[list[int]]:
    """반복형 Tarjan. 위상 역순으로 SCC 를 돌려준다."""
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack: list[int] = []
    result: list[list[int]] = []
    counter = 0
    for root in range(n):
        if index[root] != -1:
            continue
        work = [(root, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on_stack[v] = True
            recurse = False
            for i in range(pi, len(succ[v])):
                w = succ[v][i]
                if index[w] == -1:
                    work[-1] = (v, i + 1)
                    work.append((w, 0))
                    recurse = True
                    break
                if on_stack[w]:
                    low[v] = min(low[v], index[w])
            if recurse:
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                result.append(comp)
            work.pop()
            if work:
                pv = work[-1][0]
                low[pv] = min(low[pv], low[v])
    return result


def analyze(incidence: list[set[int]], n_vars: int) -> StructuralReport:
    """희소 구조만 보고 시스템의 풀이 가능성과 블록 구조를 판정한다."""
    n_eqs = len(incidence)
    match_eq = maximum_matching(incidence, n_vars)
    match_var = [-1] * n_vars
    for eq, v in enumerate(match_eq):
        if v >= 0:
            match_var[v] = eq

    rep = StructuralReport(n_eqs=n_eqs, n_vars=n_vars, matching=match_eq)

    var_to_eqs: list[list[int]] = [[] for _ in range(n_vars)]
    for eq, vs in enumerate(incidence):
        for v in vs:
            var_to_eqs[v].append(eq)

    free_vars = [v for v in range(n_vars) if match_var[v] == -1]
    free_eqs = [e for e in range(n_eqs) if match_eq[e] == -1]

    if free_vars:
        rep.underdetermined_vars, rep.underdetermined_eqs = _alternating_from_vars(
            incidence, var_to_eqs, match_var, free_vars
        )
    if free_eqs:
        rep.overdetermined_eqs, rep.overdetermined_vars = _alternating_from_eqs(
            incidence, match_eq, free_eqs
        )

    if free_vars or free_eqs:
        return rep

    # 정방 & 완전매칭 -> BLT
    succ: list[list[int]] = [[] for _ in range(n_eqs)]
    for eq, vs in enumerate(incidence):
        for v in vs:
            owner = match_var[v]
            if owner != eq:
                succ[eq].append(owner)
    sccs = _tarjan_scc(n_eqs, succ)  # 이미 위상 순서(선행 블록이 먼저)
    for comp in sccs:
        eqs = sorted(comp)
        rep.blocks.append(Block(eqs=eqs, vars=[match_eq[e] for e in eqs]))
    return rep
