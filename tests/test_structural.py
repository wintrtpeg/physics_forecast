"""구조 해석: 매칭, BLT, 과부족 결정 진단."""

from pforecast.core.structural import analyze


def test_square_system_is_block_triangular():
    # x0 -> x1 -> (x2, x3) 결합 블록
    rep = analyze([{0}, {0, 1}, {2, 3}, {2, 3}], 4)
    assert rep.ok
    assert [b.size for b in rep.blocks] == [1, 1, 2]
    assert rep.largest_block == 2


def test_blocks_come_in_solvable_order():
    rep = analyze([{0, 1}, {1}], 2)
    assert rep.ok
    first = rep.blocks[0]
    # 먼저 푸는 블록은 x1 (단독으로 결정 가능)
    assert first.vars == [1]


def test_underdetermined_reports_the_free_variables():
    rep = analyze([{0, 1}], 2)
    assert not rep.ok
    assert set(rep.underdetermined_vars) == {0, 1}
    assert rep.underdetermined_eqs == [0]


def test_overdetermined_reports_conflicting_equations():
    rep = analyze([{0}, {0}], 1)
    assert not rep.ok
    assert set(rep.overdetermined_eqs) == {0, 1}


def test_fully_coupled_system_is_one_block():
    n = 5
    rep = analyze([set(range(n)) for _ in range(n)], n)
    assert rep.ok
    assert len(rep.blocks) == 1
    assert rep.blocks[0].size == n
