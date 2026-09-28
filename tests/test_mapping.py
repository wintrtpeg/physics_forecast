"""컬럼 ↔ 모델 변수 자동 추천 (4단계). 계통이 달라도 같은 규칙으로 맞아야 한다.

컬럼 이름·설명 행·단위는 현장형 더미(`examples/nox_field`)의 것을 그대로 옮겼다
(그 CSV 는 커밋하지 않으므로). 정답은 사람이 맞춘 `examples/nox_field/tagmap.yaml`.
"""

import pytest

from pforecast.easy import model_catalog, suggest_mapping
from pforecast.scenario import load_model

NOX_COLS = [
    ("F2_UT_STK01_NOX_DRY", "굴뚝 NOx(건조)", "mg/Sm3"), ("F2_UT_STK01_FLOW", "굴뚝 배출유량", "Sm3/h"),
    ("F2_UT_STK01_TEMP", "굴뚝 온도", "degC"), ("F2_UT_STK01_TEMP_2", "굴뚝 온도(예비)", "degC"),
    ("F2_UT_SCR01_FAN_SP", "유인송풍기 정압", "mmAq"), ("F2_UT_SCR01_FAN_HZ", "유인송풍기 인버터", "Hz"),
    ("F2_UT_SCR01_DP", "스크러버 차압", "mmAq"), ("F2_UT_SCR01_CIRC_FLOW", "스크러버 순환수량", "m3/h"),
    ("F2_UT_SCR01_PH", "순환수 pH", "1"), ("F2_UT_SCR01_LEVEL", "순환조 레벨", "%"),
    ("F2_UT_SCR01_BR_DRY_FLOW", "DRY 지류 유량", "kg/s"), ("F2_FAB_DRY_UTIL", "건식식각 가동률", "%"),
    ("F2_FAB_CVD_UTIL", "증착 가동률", "%"), ("F2_FAB_WET_UTIL", "습식세정 가동률", "%"),
    ("F2_FAB_IMP_UTIL", "이온주입 가동률", "%"), ("F2_FAB_DRY_EQP_CNT", "건식식각 설치대수", "1"),
    ("F2_UT_AMB_TEMP", "외기온도", "degC"), ("F2_UT_AMB_HUMID", "외기습도", "%"),
]
NOX_X = ["F2_FAB_DRY_UTIL", "F2_FAB_CVD_UTIL", "F2_FAB_WET_UTIL", "F2_FAB_IMP_UTIL", "F2_FAB_DRY_EQP_CNT",
         "F2_UT_AMB_TEMP", "F2_UT_SCR01_FAN_HZ", "F2_UT_SCR01_CIRC_FLOW"]


@pytest.fixture(scope="module")
def nox_cat():
    m = load_model({"python": "examples/nox_stack/model.py"}).compile()
    m.build()
    return model_catalog(m)


def _cols(rows):
    return [{"name": n, "desc": d, "unit": u} for n, d, u in rows]


def test_nox_suggestions_match_the_hand_made_tagmap(nox_cat):
    from pforecast.data.tagmap import TagMap
    tm = TagMap.load("examples/nox_field/tagmap.yaml")
    r = suggest_mapping(nox_cat, _cols(NOX_COLS), "F2_UT_STK01_NOX_DRY", NOX_X)
    for e in tm.inputs:
        if e.tag in NOX_X:
            got = r["best"]["inputs"][e.tag]
            assert sorted(got["value"].split(",")) == sorted(e.targets), e.tag
            if e.scale != 1.0:                       # 인버터 Hz → 회전수비: 1/60
                assert got["scale"] == pytest.approx(e.scale) and got["unit"] == "1"
    obs = {e.tag: e.targets[0] for e in tm.observations}
    assert r["best"]["target"] == obs["F2_UT_STK01_NOX_DRY"]
    # 보조 관측을 자동으로 켜는 것은 모델이 '현장 계측값'으로 선언한 것뿐 — 정답과 같다
    assert r["best"]["extra"] == {k: v for k, v in obs.items() if k != "F2_UT_STK01_NOX_DRY" and k in dict(
        (c[0], 1) for c in NOX_COLS)}
    # 선언하지 않은 모델 출력(지류 유량 → 발생원 질량유량)은 추천으로만 보인다
    assert r["extra"]["F2_UT_SCR01_BR_DRY_FLOW"][0]["value"] == "SRC_DRY.mdot"
    assert "F2_UT_SCR01_BR_DRY_FLOW" not in r["best"]["extra"]


def test_location_alone_is_not_a_match(nox_cat):
    """DRY 가동률이 DRY 장비 대수(같은 위치, 다른 뜻)로 가면 안 된다."""
    r = suggest_mapping(nox_cat, _cols(NOX_COLS), None, ["F2_FAB_DRY_UTIL"])
    assert all(not c["value"].endswith("n_tools") for c in r["inputs"]["F2_FAB_DRY_UTIL"])
    # 단위 차원이 다르면 점수가 크게 깎인다 (pH 컬럼이 아무 데나 붙지 않는다)
    r = suggest_mapping(nox_cat, _cols(NOX_COLS), None, ["F2_UT_SCR01_PH"])
    assert "F2_UT_SCR01_PH" not in r["best"]["inputs"]


def test_one_model_input_gets_one_column(nox_cat):
    cols = _cols(NOX_COLS) + [{"name": "F2_FAB_DRY_UTIL_B", "desc": "건식식각 가동률(예비)", "unit": "%"}]
    r = suggest_mapping(nox_cat, cols, None, ["F2_FAB_DRY_UTIL", "F2_FAB_DRY_UTIL_B"])
    used = [v["value"] for v in r["best"]["inputs"].values()]
    assert len(used) == len(set(used))


def test_other_systems_use_the_same_rules():
    """냉수 플랜트: 코드도 규칙도 그대로. 모델 출력 설명(같은 이름 변수의 desc)도 쓴다."""
    m = load_model({"yaml": "examples/chiller_plant/system.yaml"}).compile()
    m.build()
    cat = model_catalog(m)
    assert any(o["label"] == "PLANT · 플랜트 총 전력" or "총 전력" in o["label"] for o in cat["observables"])
    cols = [{"name": "B2_CHL_TOTAL_KW", "desc": "냉동기동 총 전력", "unit": "kW"},
            {"name": "B2_CHL_LOAD", "desc": "냉방 부하", "unit": "kW"},
            {"name": "OA_WB_TEMP", "desc": "외기 습구온도", "unit": "degC"},
            {"name": "CHWS_TEMP_SP", "desc": "냉수 공급온도 설정", "unit": "degC"},
            {"name": "CH1_COP", "desc": "칠러 COP", "unit": "1"},
            {"name": "CT_CW_TEMP", "desc": "냉각수 온도", "unit": "degC"}]
    r = suggest_mapping(cat, cols, "B2_CHL_TOTAL_KW", ["B2_CHL_LOAD", "OA_WB_TEMP", "CHWS_TEMP_SP"])
    assert r["best"]["target"] == "PLANT.W_total"
    assert {k: v["value"] for k, v in r["best"]["inputs"].items()} == {
        "B2_CHL_LOAD": "LOAD.Q_load", "OA_WB_TEMP": "AMBIENT.T_wb", "CHWS_TEMP_SP": "LOAD.T_chws"}
    assert r["best"]["extra"] == {"CH1_COP": "CHILLER.COP", "CT_CW_TEMP": "TOWER.T_cw"}
