# 새 컴포넌트 만들기

이 툴을 "범용"으로 만드는 것은 컴포넌트를 추가하는 비용입니다. 실제로 필요한 것은
`PARAMS` / `VARS` / `PORTS` 선언과 `equations()` 하나입니다.

---

## 최소 예제: 열교환기 (한 쪽만 모델링)

```python
from pforecast.core import symbolic as S
from pforecast.core.component import ParamSpec, PortSpec, Scope, VarSpec
from pforecast.lib.flow import GasComponent, _MIN_MDOT
from pforecast.lib.gas import gas_port, density, mixture_cp


class GasCooler(GasComponent):
    """외부 냉매로 배기를 식히는 열교환기 (NTU 방식)."""

    PARAMS = {
        "UA":     ParamSpec(5000.0, "W/K",  "총괄 열전달계수x면적", lo=0.0, tunable=True),
        "T_cool": ParamSpec(12.0,   "degC", "냉매 입구 온도"),
        "K":      ParamSpec(20.0,   "1/m4", "압력손실 계수", lo=0.0, tunable=True),
    }
    VARS = {
        "dp":   VarSpec("Pa",     start=200.0),
        "rho":  VarSpec("kg/m3",  start=1.15, lo=0.05, hi=10.0),
        "Q":    VarSpec("W",      start=1e4,  desc="제거 열량"),
    }

    def port_specs(self):
        return {"a": gas_port(self.medium, "in"), "b": gas_port(self.medium, "out")}

    def equations(self, s: Scope):
        a, b = s.port("a"), s.port("b")
        cp = mixture_cp(a, self.medium)
        C = S.smooth_abs(a.mdot, _MIN_MDOT) * cp          # 열용량 유량
        eff = S.const(1.0) - S.exp(-s.UA / C)             # NTU 유효도 (냉매측 무한대 가정)
        return [
            a.mdot + b.mdot,                              # 질량
            s.rho - density(a, self.medium),              # 상태
            s.dp - s.K * S.signed_pow(a.mdot, 2.0) / s.rho,
            a.p - b.p - s.dp,                             # 운동량
            s.Q - C * eff * (a.T - s.T_cool),             # 열전달
            C * (a.T - b.T) - s.Q,                        # 에너지
            *self._species_transport(a, b),               # 화학종 (반응 없음)
        ]

    def outputs(self, s: Scope):
        return {"Q": (s.Q, "kW"), "dp_mmAq": (s.dp, "mmAq"), "T_out": (s.port("b").T, "degC")}
```

방정식 개수를 세어 봅니다: 포트 2개, 내부변수 3개 → 필요 `3*2 + 3 = 9`.
실제 `1+1+1+1+1+1+3 = 9`. ✅ 세기 싫으면 그냥 `pf check` 를 돌리세요.

레지스트리에 등록하면 YAML 로도 쓸 수 있습니다:

```python
# src/pforecast/lib/__init__.py
REGISTRY["GasCooler"] = GasCooler
```

---

## 체크리스트

### 1. 보존법칙부터 쓴다

질량 → 운동량(압력) → 에너지 → 화학종 순서로 빠짐없이. 이게 외삽을 보증하는
유일한 근거입니다. 상관식은 **그 안의 계수**로만 들어가야 합니다.

### 2. 상관식 계수는 `tunable=True` 로

`ParamSpec(..., tunable=True)` 인 것만 보정 후보로 노출됩니다. 마찰계수, 물질전달
상수, 팬 곡선 계수처럼 "현장마다 다른 값"에만 붙이세요. 중력가속도나 몰질량에는
붙이지 않습니다.

### 3. 단위를 반드시 적는다

`ParamSpec(0.5, "1/m4", ...)` 의 `"1/m4"` 가 차원 검사의 근거가 됩니다.
수치 리터럴은 무차원으로 취급되므로, 차원이 있는 상수는 명시해야 합니다:

```python
S.const(9.80665, (1, 0, -2, 0, 0, 0, 0))     # m/s^2
S.const(273.15, (0, 0, 0, 1, 0, 0, 0))       # K
```

### 4. 0 으로 나누지 않게 정규화한다

에너지식에 `mdot` 이 분모로 들어가면 무유량에서 특이해집니다.

```python
C = S.smooth_abs(a.mdot, _MIN_MDOT) * cp     # |mdot| 의 매끄러운 하한
```

양방향 유동은 `S.signed_pow(mdot, 2.0)` 을 씁니다 (`mdot·|mdot|` 의 매끄러운 형태).
`abs()` 를 그대로 쓰면 원점에서 미분이 튀어 뉴턴법이 깨집니다.

### 5. 물리적 경계를 `VarSpec` 에 넣는다

```python
VarSpec("1", start=0.4, lo=0.0, hi=0.9999)   # 효율은 1 을 넘을 수 없다
```

단, **참 해가 경계에 걸리면 안 됩니다.** 실제로 그런 버그를 하나 겪었습니다 —
건조 기준 산소 몰분율의 상한을 0.209 로 걸었는데 실제 값이 0.2153 이라 수렴에
실패했습니다. 경계는 "물리적으로 불가능한 영역"을 막는 용도이지 "예상 범위"가
아닙니다.

### 6. `initial_guess()` 를 구현한다

없어도 돌아가지만, 있으면 수렴이 훨씬 안정적입니다. 상류에서 받은 입구 상태로
출구를 대충 계산해 돌려주면 됩니다. `GasComponent` 의 기본 구현(질량가중 평균)이
대부분의 경우 충분하고, 내부 변수만 `_var_guess()` 로 채우면 됩니다.

### 7. `outputs()` 로 사람이 볼 값을 낸다

미지수는 SI 단위입니다. 리포트에 mmAq, CMM, degC 로 나가려면 여기 등록합니다.
방정식 개수에는 영향이 없습니다.

---

## 포트를 새로 정의하기 (다른 유체 계통)

`gas_port()` 는 배기가스용 포트 팩토리입니다. 냉수 계통이라면 이렇게 만듭니다:

```python
from pforecast.core.component import PortSpec

def water_port(role: str) -> PortSpec:
    return PortSpec(
        across=(("p", "Pa"),),
        through=(("mdot", "kg/s"),),
        stream=(("T", "K"),),               # 조성이 없으니 온도만
        role=role,
        kind="water",                       # kind 가 다르면 연결이 거부된다
        starts={"p": 4.0e5, "mdot": 20.0, "T": 280.0},
        bounds={"T": (270.0, 400.0)},
    )
```

`kind` 문자열이 다르면 연결 시 오류가 납니다. 배기 포트에 냉수 포트를 잘못
꽂는 사고를 막아 줍니다.

---

## 과도 해석이 필요하면

상태변수의 시간미분은 `s.der("변수명")` 으로 씁니다.

```python
VARS = {"T_wall": VarSpec("K", start=300.0)}

def equations(self, s):
    ...
    return [
        s.C_wall * s.der("T_wall") - (Q_in - Q_out),   # 벽체 열용량
        ...
    ]
```

정상상태 컴파일(`mode="steady"`)에서는 `der` 가 0 으로 치환되고, 과도
컴파일(`mode="transient"`)에서는 음함수 오일러로 이산화됩니다. 컴포넌트 코드는
그대로입니다.

---

## 디버깅 순서

1. `pf check <model>` — 방정식/미지수 개수와 BLT 구조부터. 여기서 걸리면 방정식 문제.
2. 차원 오류가 나면 해당 방정식만 보면 됩니다 (어느 방정식인지 라벨로 알려줍니다).
3. `pf solve <model>` — 수렴 실패 시 **어느 블록의 어느 변수/방정식**인지 나옵니다.
4. 수렴은 하는데 값이 이상하면, 보통 부호 규약입니다. `through` 는 **컴포넌트로
   들어가는 방향이 +** 입니다. 출구 포트의 `mdot` 은 음수입니다.
