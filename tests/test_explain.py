"""사람이 읽는 방정식(LAWS)이 실제 컴포넌트와 어긋나지 않는지.

LAWS 는 손으로 적으므로 잔차식과 자동 대조는 못 한다. 대신 어긋나면 결과 화면이
거짓말을 하게 되는 두 가지를 검사한다.

* 식에 적은 파라미터가 실제로 존재하는가 (이름을 바꾸고 LAWS 를 안 고친 경우)
* 보정 가능한 파라미터가 전부 어떤 식엔가 나오고, **구성방정식에만** 나오는가
  (보존법칙에 보정 파라미터가 들어가면 외삽 보증이 사라진다 — CLAUDE.md)
"""

import pytest

from pforecast.lib import Duct, Fan, Mixer, Stack, ToolGroupSource, WetScrubber
from pforecast.lib.explain import CLOSURE, DECLARED, laws_of, title_of
from pforecast.scenario import load_model

COMPONENTS = [ToolGroupSource("S"), Duct("D"), Mixer("M", n_inlets=3), WetScrubber("W"),
              Fan("F"), Stack("K")]


@pytest.mark.parametrize("comp", COMPONENTS, ids=lambda c: type(c).__name__)
def test_laws_name_real_parameters(comp):
    laws = laws_of(comp)
    assert laws, f"{type(comp).__name__} 에 LAWS 가 없습니다"
    params = comp.param_specs()
    for law in laws:
        for p in law.params:
            assert p in params, f"{type(comp).__name__} '{law.title}' 의 {p} 는 없는 파라미터"


@pytest.mark.parametrize("comp", COMPONENTS, ids=lambda c: type(c).__name__)
def test_tunable_parameters_live_only_in_closures(comp):
    laws = laws_of(comp)
    for name, spec in comp.param_specs().items():
        if not spec.tunable:
            continue
        kinds = {law.kind for law in laws if name in law.params}
        assert kinds, f"보정 파라미터 {type(comp).__name__}.{name} 가 어느 식에도 없습니다"
        assert kinds == {CLOSURE}, f"{type(comp).__name__}.{name} 가 {kinds} 에 들어 있습니다"


def test_declared_components_show_their_equations():
    system = load_model({"yaml": "examples/chiller_plant/system.yaml"})
    chiller = system.components["CHILLER"]
    laws = laws_of(chiller)
    assert {law.kind for law in laws} == {DECLARED}
    assert "eta_carnot" in {p for law in laws for p in law.params}
    assert title_of(chiller) == "원심식 칠러"


def test_titles_are_short_names():
    assert [title_of(c) for c in COMPONENTS] == [
        "동일 공정 장비군의 배기 발생원", "덕트 구간", "N개 지류를 합류시키는 헤더",
        "습식 스크러버", "원심 송풍기", "옥상 굴뚝"]


def test_physics_explanation_links_inputs_and_fitted_params():
    """결과 화면의 물리 설명: 흐름, 입력이 들어가는 식, 보정값이 붙는 컴포넌트."""
    from pforecast.easy import ColumnMap, EasyConfig, explain_physics

    system = load_model({"python": "examples/nox_stack/model.py"})
    model = system.compile()
    cfg = EasyConfig(csv="x.csv", target="NOX", features=["CNT", "AMB"],
                     train=["2025-01-01", "2025-01-10"], test=["2025-01-12", "2025-01-20"],
                     model="examples/nox_stack/model.py",
                     target_map=ColumnMap("NOX", ["STK.C_dry"], "mg/Nm3"),
                     feature_map={"CNT": ColumnMap("CNT", ["SRC_DRY.n_tools"]),
                                  "AMB": ColumnMap("AMB", ["STK.T_amb", "DCT_MAIN.T_amb"], "degC")})
    table = [{"parameter": "SCR.ntu_a", "unit": "1", "initial": 14.0, "fitted": 8.4,
              "change_pct": -40.0, "rel_stderr_pct": 12.0}]
    ex = explain_physics(system, model, cfg, table, rows=150, n_obs=5)
    assert ex["counts"]["equations"] == ex["counts"]["unknowns"] == 174
    assert ex["segments"][0][0].startswith("SRC_")          # 발생원부터
    assert ex["segments"][-1][-1] == "STK"
    uses = {i["column"]: i["uses"] for i in ex["inputs"]}
    assert "NOx 발생량 (배출계수)" in uses["CNT"][0]["laws"]
    assert {u["target"] for u in uses["AMB"]} == {"STK.T_amb", "DCT_MAIN.T_amb"}
    scr = next(g for g in ex["groups"] if "SCR" in g["instances"])
    assert [f["param"] for f in scr["fitted"]] == ["ntu_a"]
    assert ex["target"]["component"] == "옥상 굴뚝"
