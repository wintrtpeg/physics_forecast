"""구성방정식 슬롯·후보와 후보 추천.

* 후보를 바꿔 끼워도 모델이 풀리고, 수치 버전(초기값용)이 심볼릭 식과 같은 값을 낸다.
* 선택은 모델 설정(``model.closures``)과 YAML 컴포넌트(``closures:``) 양쪽에서 된다.
* 추천은 **학습 기간만** 본다 — 예측 기간 값을 망가뜨려도 추천이 받는 데이터가 같다.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pforecast.core.solvers import solve_steady
from pforecast.easy import EasyConfig, closure_catalog, closure_params, config_from_body
from pforecast.lib.closures import apply_closures, closure_slots
from pforecast.scenario import load_model

MODEL = {"python": "examples/nox_stack/model.py"}


def _solve(choices):
    system = load_model(MODEL)
    apply_closures(system, choices)
    model = system.compile()
    model.build()
    r = solve_steady(model, model.p0(), model.x0())
    assert r.success
    return system, model, r.x


def _cases():
    system = load_model(MODEL)
    seen = set()
    for inst, sl in closure_slots(system):
        kind = (type(system.components[inst]).__name__, sl.key)
        if kind in seen:
            continue
        seen.add(kind)
        for opt in sl.options:
            yield pytest.param(inst, sl.key, opt.id, id=f"{inst}.{sl.key}={opt.id}")


@pytest.mark.parametrize("inst,key,cid", list(_cases()))
def test_numeric_closure_matches_symbolic(inst, key, cid):
    """초기값용 수치 식이 모델이 푸는 식과 같은 값을 낸다 (틀리면 뉴턴 출발점이 엉뚱해진다)."""
    system, model, x = _solve({f"{inst}.{key}": cid})
    comp = system.components[inst]
    v = lambda n: x[model.var_index(f"{inst}.{n}")]  # noqa: E731
    if key == "eta":
        num, sym = comp.closure_value("eta", LG=v("LG"), T=v("a.T")), v("eta")
    elif key in ("dp", "friction"):
        num, sym = comp.closure_value(key, m=v("a.mdot"), rho=v("rho")), v("dp")
    else:   # emission: 유출 NOx 질량유량
        num, sym = comp.closure_value("emission"), -v("outlet.mdot") * v("outlet.w_NOx")
    assert num == pytest.approx(sym, rel=1e-9)


def test_default_choices_change_nothing():
    """기본 후보를 명시해도 아무것도 안 바뀐다 (슬롯을 도입해도 기존 모델은 그대로)."""
    _, _, x0 = _solve({})
    system = load_model(MODEL)
    defaults = {f"{i}.{sl.key}": sl.default for i, sl in closure_slots(system)}
    _, _, x1 = _solve(defaults)
    assert np.array_equal(x0, x1)


def test_choice_through_model_spec_and_yaml_component(tmp_path):
    s1 = load_model({**MODEL, "closures": {"SCR.eta": "langmuir"}})
    assert s1.components["SCR"].closure_choices["eta"] == "langmuir"
    assert "k_LG" in s1.components["SCR"].param_specs()
    y = Path("examples/nox_stack/system.yaml").read_text(encoding="utf-8")
    y = y.replace("SCR:      {type: WetScrubber,", "SCR:      {type: WetScrubber, closures: {eta: constant},")
    p = tmp_path / "sys.yaml"
    p.write_text(y, encoding="utf-8")
    s2 = load_model({"yaml": str(p)})
    assert s2.components["SCR"].closure_choices["eta"] == "constant"
    assert "eta_max" not in s2.components["SCR"].param_specs()
    with pytest.raises(KeyError):
        load_model({**MODEL, "closures": {"SCR.eta": "no_such"}})


def test_closure_params_follow_the_choice():
    system = load_model(MODEL)
    user = ["SRC_DRY.ef_idle", "SRC_DRY.ef_process", "SCR.ntu_a", "SCR.K"]
    out = closure_params(system, user, {"SCR.eta": "langmuir", "SRC_DRY.emission": "util_only"})
    assert "SCR.ntu_a" not in out and "SCR.k_LG" in out            # 빠진 후보의 것은 빼고 새 것은 더한다
    assert "SRC_DRY.ef_idle" not in out and "SRC_DRY.ef_process" in out
    assert "SCR.K" in out
    out2 = closure_params(system, user, {"SCR.eta": "ntu_power_T"})
    assert {"SCR.ntu_a", "SCR.ntu_c"} <= set(out2)
    # 기본 후보를 명시한 것은 보정 목록을 바꾸지 않는다 (화면이 모든 슬롯을 보내도 안전)
    defaults = {f"{i}.{sl.key}": sl.default for i, sl in closure_slots(system)}
    assert closure_params(system, user, defaults) == user


def test_catalog_lists_slots_from_the_system():
    cat = closure_catalog(load_model(MODEL))
    keys = {c["key"] for c in cat}
    assert {"SCR.eta", "SCR.dp", "SRC_DRY.emission", "DCT_MAIN.friction"} <= keys
    eta = next(c for c in cat if c["key"] == "SCR.eta")
    assert eta["current"] == "ntu_power"
    roles = {o["id"]: o["role"] for o in eta["candidates"]}
    assert roles["constant"] == "baseline" and roles["ntu_power"] == "standard"


# ── 추천: 학습 기간만 ────────────────────────────────────────────────────

BODY = {"target": "F2_UT_STK01_NOX_DRY",
        "features": ["F2_FAB_DRY_UTIL", "F2_FAB_CVD_UTIL", "F2_FAB_WET_UTIL", "F2_FAB_IMP_UTIL",
                     "F2_UT_AMB_TEMP"],
        "train": ["2025-03-01", "2025-03-22"], "test": ["2025-03-24", "2025-03-30"],
        "model": "examples/nox_stack/model.py",
        "feature_map": {f"F2_FAB_{g}_UTIL": {"targets": [f"SRC_{g}.util"], "unit": "%"}
                        for g in ("DRY", "CVD", "WET", "IMP")},
        "target_map": {"targets": ["STK.C_dry"], "unit": "mg/Nm3"},
        "params": ["SRC_DRY.ef_process", "SCR.ntu_a"]}
BODY["feature_map"]["F2_UT_AMB_TEMP"] = {"targets": ["STK.T_amb"], "unit": "degC"}


def _cfg(root: Path, csv: str) -> EasyConfig:
    return config_from_body({**BODY, "csv": csv}, lambda p: (root / p).resolve()
                            if not Path(p).is_absolute() else Path(p))


def test_advice_never_sees_the_forecast_period(tmp_path, monkeypatch):
    """예측 기간 값을 망가뜨려도 추천 엔진이 받는 데이터는 한 비트도 안 바뀐다."""
    import pforecast.closure_advice as ca
    from pforecast.easy import advise_easy

    root = Path(".").resolve()
    src = root / "examples/nox_stack/data/plant_5min.csv"
    good, bad = tmp_path / "good.csv", tmp_path / "bad.csv"
    shutil.copy(src, good)
    df = pd.read_csv(src)
    t = pd.to_datetime(df["timestamp"])
    df.loc[t >= "2025-03-24", "F2_UT_STK01_NOX_DRY"] *= 10          # 예측 기간만 망가뜨린다
    df.loc[t >= "2025-03-24", "F2_FAB_DRY_UTIL"] = 999.0
    df.to_csv(bad, index=False)
    seen = []
    monkeypatch.setattr(ca, "advise_closures", lambda *a, **k: seen.append(a[2].copy()) or {"slots": []})
    advise_easy(_cfg(root, str(good)), tmp_path)
    advise_easy(_cfg(root, str(bad)), tmp_path)
    a, b = seen
    assert a.index.max() < pd.Timestamp("2025-03-23")               # 학습 끝(3/22) 안쪽만
    pd.testing.assert_frame_equal(a, b)


def test_advice_runs_and_explains_its_verdict(tmp_path):
    """작은 설정으로 끝까지: 슬롯마다 현재 식·하한선이 들어가고 근거가 붙는다."""
    from pforecast.easy import advise_easy

    root = Path(".").resolve()
    res = advise_easy(_cfg(root, "examples/nox_stack/data/plant_5min.csv"), tmp_path,
                      max_rows=12, slots=["SCR.eta"])
    json.dumps(res, default=str)                                   # 화면으로 보낼 수 있어야 한다
    assert res["split"]["validate"][1] < "2025-03-23"
    (slot,) = res["slots"]
    assert slot["key"] == "SCR.eta" and slot["recommended"]
    ids = {c["id"]: c for c in slot["candidates"]}
    assert ids["ntu_power"]["evaluated"]                           # 현재 식은 늘 채점한다
    assert "constant" in ids                                       # 하한선이 비교에 들어간다
    for c in slot["candidates"]:
        assert c["status"], c["id"]
        if c["status"] in ("제외", "구별 불가"):
            assert c["reasons"], c["id"]                           # 빼면 이유를 말한다
    assert slot["message"]
