"""비선형 해석기.

두 가지를 제공한다.
* ``solve_steady`` : BLT 블록 순서대로 감쇠 뉴턴법. 큰 계를 작은 블록으로 쪼개 푼다.
* ``solve_transient`` : 음함수 오일러로 이산화한 DAE 를 매 스텝 ``solve_steady`` 로 푼다.

수렴 판정은 절대값이 아니라 **스케일 상대값**으로 한다. 압력 잔차는 1e5 Pa 스케일,
질량분율 잔차는 1e-6 스케일이라 절대 허용오차 하나로는 둘 다 만족시킬 수 없다.
각 방정식의 스케일은 야코비안 행 노름 x 변수 스케일로 추정한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BlockTrace:
    index: int
    size: int
    iterations: int
    converged: bool
    residual: float


@dataclass
class SolveResult:
    x: np.ndarray
    success: bool
    message: str = ""
    residual_inf: float = 0.0
    traces: list[BlockTrace] = field(default_factory=list)
    n_newton: int = 0

    def failed_blocks(self) -> list[BlockTrace]:
        return [t for t in self.traces if not t.converged]


def _clip_step(x: np.ndarray, dx: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """경계를 넘지 않는 최대 스텝 배율(<=1). 블록마다 매 반복 호출되므로 벡터화한다."""
    alpha = 1.0
    up = dx > 0
    dn = dx < 0
    if np.any(up):
        room = hi[up] - x[up]
        if np.any(room <= 0):
            return 0.0
        finite = np.isfinite(room)
        if np.any(finite):
            alpha = min(alpha, float(np.min(0.95 * room[finite] / dx[up][finite])))
    if np.any(dn):
        room = x[dn] - lo[dn]
        if np.any(room <= 0):
            return 0.0
        finite = np.isfinite(room)
        if np.any(finite):
            alpha = min(alpha, float(np.min(0.95 * room[finite] / (-dx[dn][finite]))))
    return max(alpha, 0.0)


def _newton_block(bf, x, p, lo, hi, xscale, rtol, atol, maxiter):
    """스케일링된 감쇠 뉴턴법.

    압력(1e5 Pa)과 질량분율(1e-4)이 한 블록에 섞이면 야코비안 조건수가 1e9 를 넘는다.
    열 스케일(변수 크기)과 행 스케일(방정식 크기)로 균등화한 뒤 풀어야 수렴한다.
    블록이 크면 희소 LU 를, 작으면 dense 를 쓴다.
    """
    var_idx = bf.vars
    n = bf.size
    sub_lo, sub_hi = lo[var_idx], hi[var_idx]
    sparse = bf.use_sparse
    if sparse:
        from scipy.sparse.linalg import splu
        from scipy.sparse import diags
    it = 0
    last_res = np.inf
    F = bf.residual(x, p)
    for it in range(1, maxiter + 1):
        if not np.all(np.isfinite(F)):
            return it, False, float("inf")
        cs = np.maximum(np.abs(x[var_idx]), xscale[var_idx])       # 열 스케일
        rs = np.maximum(bf.row_abs_sums(x, p, cs), 1e-300)         # 행 스케일
        raw = float(np.max(np.abs(F))) if n else 0.0
        last_res = raw
        if (float(np.max(np.abs(F) / (atol + rtol * rs))) if n else 0.0) <= 1.0:
            return it - 1, True, raw
        Fs = F / rs
        try:
            if sparse:
                A = diags(1.0 / rs) @ bf.sparse_jacobian(x, p) @ diags(cs)
                y = splu(A.tocsc()).solve(-Fs)
            else:
                y = np.linalg.solve((bf.dense_jacobian(x, p) * cs) / rs[:, None], -Fs)
        except Exception:
            A = (bf.dense_jacobian(x, p) * cs) / rs[:, None]
            lam = 1e-10 * max(1.0, float(np.max(np.abs(A))))
            y = None
            for _ in range(14):
                try:
                    y = np.linalg.solve(A + lam * np.eye(n), -Fs)
                    break
                except np.linalg.LinAlgError:
                    lam *= 10.0
            if y is None:
                y = np.linalg.lstsq(A, -Fs, rcond=None)[0]
        dx = cs * y
        if not np.all(np.isfinite(dx)):
            return it, False, last_res
        alpha = _clip_step(x[var_idx], dx, sub_lo, sub_hi)
        if alpha == 0.0:
            return it, False, last_res
        f0 = float(Fs @ Fs)
        base = x[var_idx].copy()
        ok = False
        for _ in range(40):
            x[var_idx] = base + alpha * dx
            Fn = bf.residual(x, p)
            if np.all(np.isfinite(Fn)) and float((Fn / rs) @ (Fn / rs)) <= (1.0 - 1e-4 * alpha) * f0:
                ok = True
                break
            alpha *= 0.5
        if ok:
            F = Fn                      # 선탐색에서 이미 구한 값을 재사용
        else:
            x[var_idx] = base + min(alpha, 1e-2) * dx
            F = bf.residual(x, p)
            if alpha < 1e-12:
                return it, False, last_res
    return it, False, last_res


def _least_squares_block(fb, x, var_idx, lo, hi, xscale, maxiter=300):
    """뉴턴이 실패했을 때의 최후 수단. 경계를 지키는 trust-region 최소제곱."""
    from scipy.optimize import least_squares

    cs = xscale[var_idx]
    base = x.copy()

    def resid(y):
        base[var_idx] = y * cs
        return fb(base)

    y0 = x[var_idx] / cs
    lo_s = np.where(np.isfinite(lo[var_idx]), lo[var_idx] / cs, -np.inf)
    hi_s = np.where(np.isfinite(hi[var_idx]), hi[var_idx] / cs, np.inf)
    y0 = np.clip(y0, lo_s + 1e-12, hi_s - 1e-12)
    try:
        sol = least_squares(resid, y0, bounds=(lo_s, hi_s), xtol=1e-14, ftol=1e-14,
                            gtol=1e-14, max_nfev=maxiter * max(1, len(y0)))
    except Exception:
        return False, float("inf")
    x[var_idx] = sol.x * cs
    F = fb(x)
    res = float(np.max(np.abs(F))) if len(F) else 0.0
    return bool(sol.success), res


def solve_steady(
    model,
    p: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-10,
    maxiter: int = 80,
    use_fallback: bool = True,
) -> SolveResult:
    """BLT 순서대로 블록을 순차적으로 푼다."""
    model.build()
    if not model.report.ok:
        return SolveResult(
            x=np.zeros(model.n_vars), success=False,
            message="구조 해석 실패. model.describe() 를 확인하세요.",
        )
    p = model.p0() if p is None else np.asarray(p, dtype=float)
    x = (model.x0() if x0 is None else np.asarray(x0, dtype=float)).copy()
    lo, hi = model.bounds()
    x = np.clip(x, lo + np.where(np.isfinite(lo), 1e-12, 0.0), hi)
    xscale = np.array([max(abs(v.start), 1e-12) for v in model.variables])

    traces: list[BlockTrace] = []
    total = 0
    ok_all = True
    for bi, bf in enumerate(model._block_fns):
        var_idx = bf.vars
        it, conv, res = _newton_block(bf, x, p, lo, hi, xscale, rtol, atol, maxiter)
        if not conv and use_fallback and len(var_idx) <= 25:
            _least_squares_block(lambda xx, _b=bf: _b.residual(xx, p), x, var_idx, lo, hi, xscale)
            # 최소제곱 해에서 뉴턴을 재시도하면 대개 마무리된다
            it2, conv, res = _newton_block(bf, x, p, lo, hi, xscale, rtol, atol, maxiter)
            it += it2
        total += it
        traces.append(BlockTrace(bi, len(var_idx), it, conv, res))
        if not conv:
            ok_all = False
            break

    F = model.residual(x, p)
    res_inf = float(np.max(np.abs(F))) if len(F) else 0.0
    if ok_all:
        msg = f"수렴 (블록 {len(traces)}개, 뉴턴 반복 {total}회)"
    else:
        bad = traces[-1]
        names = [model.variables[i].name for i in model.report.blocks[bad.index].vars]
        labels = [model.equations[i].label for i in model.report.blocks[bad.index].eqs]
        msg = (
            f"블록 #{bad.index} (크기 {bad.size}) 수렴 실패, 잔차 {bad.residual:.3e}\n"
            f"  미지수: {', '.join(names[:8])}{' ...' if len(names) > 8 else ''}\n"
            f"  방정식: {'; '.join(labels[:5])}{' ...' if len(labels) > 5 else ''}"
        )
    return SolveResult(x=x, success=ok_all, message=msg, residual_inf=res_inf,
                       traces=traces, n_newton=total)


def solve_transient(
    model,
    t_grid: np.ndarray,
    p_of_t,
    x0: np.ndarray | None = None,
    **kw,
) -> tuple[np.ndarray, np.ndarray, list[SolveResult]]:
    """음함수 오일러. ``model`` 은 transient 모드로 컴파일되어 있어야 한다.

    ``p_of_t(t) -> 파라미터 배열`` 을 받아 각 시점의 경계조건을 준다.
    반환: (t, X[len(t), n_vars], 스텝별 결과)
    """
    model.build()
    t_grid = np.asarray(t_grid, dtype=float)
    n = model.n_vars
    X = np.zeros((len(t_grid), n))
    x = (model.x0() if x0 is None else np.asarray(x0, dtype=float)).copy()
    results: list[SolveResult] = []
    for k, t in enumerate(t_grid):
        p = np.asarray(p_of_t(t), dtype=float).copy()
        if k == 0:
            dt0 = float(t_grid[1] - t_grid[0]) if len(t_grid) > 1 else 1.0
            model.set_discretization(p, x, dt=dt0, first_step=True)
        else:
            model.set_discretization(p, X[k - 1], dt=float(t - t_grid[k - 1]))
        r = solve_steady(model, p, x0=x, **kw)
        x = r.x.copy()
        X[k] = x
        results.append(r)
        if not r.success:
            break
    return t_grid, X, results


def quasi_steady_sweep(model, p_rows, x0=None, **kw):
    """5분 평균 데이터처럼 시점별 정상상태를 연속으로 푼다.

    직전 해를 다음 스텝의 초기값으로 쓰기 때문에(warm start) 수천 스텝도 빠르다.
    """
    model.build()
    xs = []
    results = []
    x = model.x0() if x0 is None else np.asarray(x0, dtype=float)
    for p in p_rows:
        r = solve_steady(model, np.asarray(p, dtype=float), x0=x, **kw)
        if r.success:
            x = r.x
        xs.append(r.x.copy())
        results.append(r)
    return np.array(xs), results
