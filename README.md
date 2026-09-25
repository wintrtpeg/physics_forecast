# pforecast — 물리 지배방정식 기반 모델링·예측 툴

인프라 유틸리티 계통을 **보존법칙과 구성방정식으로 기술하고**, 현장 계측으로 물리
파라미터를 보정한 뒤, 학습 데이터에 없던 조건까지 예측하는 범용 툴입니다.
VS Code + 로컬 PC 에서 돌아가며, GPU 나 무거운 프레임워크가 필요 없습니다.

첫 적용 대상은 **배기 스크러버 계통 → 옥상 굴뚝 NOx 농도** 입니다.

---

## 왜 물리모델인가

ML 회귀모형은 학습 분포 안에서는 훌륭합니다. 문제는 우리가 정말 알고 싶은 질문이
거의 언제나 **분포 밖**에 있다는 것입니다.

| 질문 | ML 회귀 | 물리모델 |
|---|---|---|
| 지금 농도는? | 가능 | 가능 |
| 가동율 95%면? (과거 최대 70%) | 외삽, 근거 없음 | 가능 |
| 장비 30% 증설하면? | **불가** (대수가 상수였음) | 가능 |
| 송풍기 15% 감속하면? | **불가** (회전수가 거의 고정) | 가능 |
| 충전재 교체하면? | **불가** (설비 상태는 특징이 아님) | 가능 |

포함된 예제에서 실제로 측정한 결과입니다 (같은 저부하 데이터로 보정/학습,
고부하 구간에서 검증):

| 모델 | 학습 구간 RMSE | **외삽 구간 RMSE** | 외삽 편향 | 외삽 R² |
|---|---|---|---|---|
| 물리모델 | 2.56 mg/Sm³ | **2.52 mg/Sm³** | +0.17 | 0.920 |
| ML 기준모델(2차 다항) | 2.54 mg/Sm³ | **3.72 mg/Sm³** | −2.16 | 0.826 |

물리모델은 외삽 구간에서 오차가 **전혀 늘지 않습니다**. 학습 구간을 벗어난다는
개념 자체가 없기 때문입니다. ML 은 오차가 46% 늘고 계통적 편향이 생깁니다.

그리고 위 표의 아래 세 줄 — 증설·감속·설비교체 — 는 정도의 문제가 아니라
**종류의 문제**입니다. 과거 데이터에 변동이 없었던 변수는 어떤 회귀모형도
계수를 가질 수 없습니다.

---

## 빠른 시작

```bash
git clone <repo> && cd physics_forecast
pip install -e ".[plot,dev]"          # numpy, scipy, pandas, pyyaml, matplotlib

pf demo                                # 예제 전 과정 (구조검사 → 풀이 → 시나리오 → 보정)
```

개별 명령:

```bash
pf check     examples/nox_stack/model.py          # 구조 해석: 방정식/미지수/BLT 진단
pf solve     examples/nox_stack/model.py          # 설계점 정상상태 풀이
python examples/nox_stack/make_synthetic_data.py  # 가상 현장 데이터 30일치 생성
pf calibrate examples/nox_stack/calibration.yaml  # 보정 + 외삽 검증 + HTML 리포트
pf run       examples/nox_stack/scenarios.yaml -o out/scenario.html   # what-if
pf select    examples/nox_stack/selection.yaml    # 구성방정식 후보 비교
pf analyze   examples/nox_stack/analysis.yaml     # 데이터 주도 분석 + XAI 대시보드
pf equations examples/nox_stack/system.yaml --dims # 조립된 방정식과 차원 확인
```

VS Code 에서는 `Ctrl+Shift+B` 로 구조검사, `터미널 → 작업 실행`에 나머지가 등록되어
있습니다 (`.vscode/tasks.json`).

---

## 이 툴이 실제로 하는 일

### 1. 무인과(acausal) 모델링

컴포넌트는 "입력을 받아 출력을 계산"하지 않고 **만족해야 할 방정식만 선언**합니다.
누가 미지수인지는 계통을 연결한 뒤 구조 해석이 결정합니다.

```python
class Duct(GasComponent):
    PARAMS = {"K": ParamSpec(0.5, "1/m4", "덕트 저항계수", tunable=True), ...}
    VARS   = {"dp": VarSpec("Pa"), "rho": VarSpec("kg/m3")}

    def equations(self, s):
        a, b = s.port("a"), s.port("b")
        return [
            a.mdot + b.mdot,                                  # 질량 보존
            s.rho - density(a, self.medium),                  # 상태방정식
            s.dp - s.K * S.signed_pow(a.mdot, 2.0) / s.rho,   # 마찰 손실
            a.p - b.p - s.dp,                                 # 운동량
            ...                                               # 에너지, 화학종
        ]
```

덕분에 같은 스크러버 모델로 "유량을 주고 압력을 구할지 / 압력을 주고 유량을 구할지"를
바꿔 쓸 수 있습니다.

### 2. 풀기 전에 진단

```
$ pf check examples/nox_stack/model.py
모델: nox_stack
  컴포넌트 13개, 연결 12개
  방정식 174개, 미지수 174개, 파라미터 82개
  구조: 정상 (BLT 15블록, 크기분포 14x1, 1x160, 최대 160)
  야코비안 non-zero 499개
```

방정식이 모자라면 **어느 변수 묶음이 결정되지 않는지**, 과하면 **어느 방정식이
충돌하는지** 이름으로 알려줍니다. "특이 행렬" 한 줄 보고 끝나는 일이 없습니다.

### 3. 물리 파라미터 보정 + 식별성 진단

보정하는 것은 회귀계수가 아니라 덕트 저항계수, 후드 흡입압, 물질전달 상수 같은
**물리 파라미터**입니다. 그리고 반드시 함께 보고합니다 — *그 값이 데이터로부터
실제로 결정된 것인가?*

예제에서 스크러버 파라미터 7개를 한꺼번에 보정했을 때 실제로 나온 진단입니다:

```
[식별성 경고]
  ! SCR.eta_max 와 SCR.ntu_a 의 상관계수 -0.998
    -> 데이터가 둘을 구분하지 못합니다. 하나를 고정하거나
       두 값을 분리할 수 있는 운전 구간 데이터를 확보하세요.
  ! SCR.eta_max 의 상대표준오차가 14089% 입니다 -> 사실상 결정되지 않았습니다.
  ! SRC_DRY.ef_process 와 SRC_DRY.ef_idle 의 상관계수 +0.951
```

맞는 지적입니다. 제거효율이 `η = η_max·(1 − e^(−NTU))` 이고 이 계통의 NTU 가 작아
`η ≈ η_max·NTU` 이므로, 두 값은 **곱으로만** 결정됩니다. 배출계수들도 마찬가지로
굴뚝 농도에는 합으로만 들어옵니다.

반면 압력 계통 파라미터는 차압 계측이 따로 있어 잘 결정됩니다 — `DCT_MAIN.K` 는
상대표준오차 2.9%, `SCR.K` 는 0.4%. 이 값들은 설비 진단에 그대로 쓸 수 있습니다
(설계 대비 +39% → 덕트 오염/충전층 막힘).

예측 성능에는 문제가 없습니다(식별 가능한 조합은 정확히 맞았으므로). 하지만 개별
값을 물리적 해석에 쓰면 안 됩니다. **이런 진단은 ML 파이프라인이 구조적으로
줄 수 없습니다.**

### 4. what-if 시나리오

```yaml
cases:
  - name: 증설 +30% + 풀가동
    scale: {"SRC_*.n_tools": 1.30}     # glob 패턴으로 모든 장비군 한 번에
    set:   {"SRC_*.util": 1.00}
limits:
  STK.C_dry: {max: 100.0}              # 사내 관리기준 [mg/Sm3]
  SRC_DRY.q_per_tool: {min: 3.0}       # 대당 최소 배기 풍량 [CMM]
```

```
!! 관리기준 초과 11건
        label                target  value  limit  type   margin_%
증설 +30% + 풀가동          STK.C_dry  116.7    100  상한 초과     16.7
증설 +30% + 풀가동  SRC_DRY.q_per_tool  2.864      3  하한 미달     -4.5
```

대책까지 같은 표에서 평가됩니다 — 송풍기 10% 증속으로는 107.4, 순환수 2배로는
110.4. **둘 다 단독으로는 기준을 못 맞춥니다.**

---

## 데이터만 있을 때 — `pf analyze`

계통 구조를 모르는 상태에서도 타깃 컬럼 하나만 지정하면 **자동으로 갈 수 있는
데까지** 올라가고, 어디서 왜 막혔는지 알려줍니다.

| 단계 | 내용 | 필요한 것 | 자동? |
|---|---|---|---|
| L0 | 프로파일링, 운전 포락선, 준정상 구간 | 데이터만 | ✅ |
| L1 | 대리모델 + XAI (중요도·부분의존도·지연) | 타깃 지정 | ✅ |
| L2 | 무차원군 (Buckingham Π) | **컬럼 단위** | ✅ |
| L3 | **보존식 자동 탐지** | **컬럼 단위** | ✅ |
| L4 | 구조 지배방정식 모델 | **토폴로지 + 설비 제원** | ❌ |

**L4 는 자동화할 수 없습니다.** 컬럼 이름과 숫자만으로 "이건 스크러버이고 뒤에
팬이 달려 있다"를 유도할 수는 없습니다. 토폴로지는 물리적 사실이지 데이터의
통계적 성질이 아닙니다. 대신 L0~L3 이 그 앞까지를 전부 자동으로 끌어줍니다.

### L3 이 실제로 찾아낸 것

```
[확실]      BR_DRY + BR_CVD + BR_WET + BR_IMP − HDR = 0    잔차 0.31%, R² 0.986
[우연 의심] BR_DRY − BR_WET = 0                             잔차 2.00%, R² 0.479
```

첫 번째는 **질량 보존**입니다. 회귀계수가 아니라 물리 제약이라 외삽에서도 성립합니다.
두 번째는 크기가 비슷한 두 신호의 우연한 일치이고, 자동으로 걸러집니다.

### XAI 가 틀린 답을 낸 사례 — 그리고 고친 방법

정답을 아는 합성 데이터로 돌렸더니 개별 순열 중요도가 **완전히 뒤집힌 순위**를
냈습니다:

| | 진짜 기여 | 개별 순열 중요도 |
|---|---|---|
| DRY 장비군 | **60.4%** | 11.4% (4위) |
| WET 장비군 | 7.3% | **35.5%** (1위) |

원인은 네 가동율의 상호 상관 **0.9996**. 순열 중요도는 상관된 변수 사이에서 기여를
임의로 나눠 갖습니다. 그래서 상관 0.9 이상을 묶어 **함께 섞는** 방식으로 바꿨고,
결과가 정직해졌습니다:

> 영향의 **93.2%** 가 서로 구분되지 않는 한 덩어리(11개 변수)에 있습니다.
> 이 안에서 누가 원인인지는 **이 데이터로 알 수 없습니다.**

그리고 이게 물리 모델이 필요한 이유입니다 — 지배방정식에서는 각 장비군의 배출계수가
각자의 질량수지에 **구조적으로 다른 자리**로 들어가므로 분리 가능성이 생깁니다.

자세한 내용은 [`docs/data_driven.md`](docs/data_driven.md).

## 지배방정식을 직접 선언한다

컴포넌트를 파이썬으로 짤 필요가 없습니다. YAML 에 방정식을 그대로 적으면
**손으로 짠 것과 완전히 같은 식 그래프**가 나옵니다 (해석적 미분·차원 검사·구조
해석 모두 그대로 적용).

```yaml
ports:
  a: {kind: gas, role: in}
  b: {kind: gas, role: out}
params:
  K: {value: 30, unit: "1/m4", tunable: true}
vars:
  dp:  {unit: Pa, start: 100}
  rho: {unit: kg/m3, start: 1.15, lo: 0.05, hi: 10}
equations:
  mass:     "a.mdot + b.mdot = 0"
  state:    "rho = density(a)"
  friction: "dp = K * signed_pow(a.mdot, 2) / rho"
  momentum: "a.p - b.p = dp"
```

수치 리터럴에 단위를 달 수 있습니다 (`150[mmAq]`, `4186[J/(kg*K)]`). 달지 않은
숫자는 무차원으로 취급되므로, 차원이 있는 상수는 반드시 달아야 검사가 삽니다.

포트 종류는 `gas` / `liquid` / `thermal` 이 기본 제공되고 추가할 수 있습니다.
`tests/test_generic.py` 에 **액체 배관 계통을 파이썬 0줄로** 만들어 해석해와
대조하는 예가 있습니다.

## 구성방정식 후보를 데이터로 고른다

`extends` 로 기본 정의를 상속하고 **방정식 한 줄만** 갈아끼웁니다.

```yaml
extends: scrubber_base.yaml
params:
  k_LG: {value: 3.5e-3, unit: "1", lo: 1.0e-6, hi: 1.0, tunable: true}
equations:
  closure_eta: "eta = eta_max * LG / (k_LG + LG)"     # 이 줄만 다르다
```

```bash
pf select examples/nox_stack/selection.yaml
```

**후보로 둘 수 있는 것과 없는 것이 명확히 갈립니다.**

| | 후보? |
|---|---|
| 질량·운동량·에너지·화학종 보존, 상태방정식 | **아니오. 공리다.** |
| 구성방정식 — 마찰/물질전달/열전달 상관식, 성능곡선 | **예.** |

보존법칙을 후보로 돌리면 외삽 보증이 사라집니다. 그러면 물리모델을 쓸 이유가
없습니다. 자세한 내용은 [`docs/generalization.md`](docs/generalization.md).

## 구조

```
src/pforecast/
├─ core/                 # 도메인 비의존 엔진
│  ├─ units.py           # 단위 파서 + 7차원 벡터. 방정식 차원 동차성 검사
│  ├─ symbolic.py        # 식 그래프 → 해석적 미분 → numpy 코드 생성 (자체 구현)
│  ├─ parser.py          # 방정식 텍스트 → 식 그래프 (YAML 선언형의 토대)
│  ├─ component.py       # Component / Port(across·through·stream) / Scope
│  ├─ system.py          # 연결 → 방정식 조립 → 희소 야코비안 컴파일
│  ├─ structural.py      # 매칭, Dulmage-Mendelsohn, BLT 블록 분할
│  └─ solvers.py         # 블록 단위 스케일링 감쇠 뉴턴법, 음함수 오일러
├─ lib/                  # 물리 컴포넌트 라이브러리
│  ├─ gas.py             # 이상기체 혼합물 물성, 습공기, 표준상태 환산
│  ├─ flow.py            # Duct, Fan, Mixer
│  ├─ abatement.py       # ToolGroupSource, WetScrubber, Stack
│  └─ generic.py         # EquationComponent — YAML 선언형 컴포넌트
├─ data/                 # 태그맵(YAML) + CSV/SQL 어댑터 + 5분 평균 정렬
├─ calib/                # 파라미터 추정, 식별성 진단, ML 기준모델
├─ scenario/             # 모델 로더(py/yaml), what-if 실행기
├─ report/               # 자체 완결형 HTML 리포트
├─ analyze/              # 데이터 주도 분석 사다리
│  ├─ profile.py         # 프로파일링, 운전 포락선, 준정상 구간
│  ├─ dimensional.py     # 무차원군(Buckingham Π), 보존식 자동 탐지
│  ├─ surrogate.py       # 대리모델 + XAI (묶음 순열 중요도, PDP, 외삽 거리)
│  └─ pipeline.py        # 사다리 실행과 판정
├─ selection.py          # 구성방정식 후보 비교·판정
├─ params.py             # 보정값 저장/적용 (라인별로 파일만 교체)
├─ workflow.py           # 데이터 → 보정 → 검증 → 리포트
└─ cli.py                # pf 명령
```

### 왜 심볼릭 엔진을 직접 만들었나

사내 PC 에서 torch/jax/CasADi 설치를 장담할 수 없습니다. 그런데 물리 모델을 제대로
풀려면 정확한 야코비안이 필요합니다(수치미분은 느리고 강성 문제에서 수렴이 깨집니다).
그래서 **numpy 만으로 동작하는 작은 심볼릭 코어**를 넣었습니다. 해시콘싱으로 공통
부분식이 자동 공유되고, 해석적 미분 후 파이썬 소스를 생성해 `exec` 합니다.

희소성까지 활용합니다. 160×160 블록이라면 25,614 개가 아니라 **499 개**의 미분식만
생성합니다 (코드 생성 1.05초 → 0.07초).

---

## 성능

예제 모델(방정식 174개, 최대 블록 160×160) 기준, 일반 노트북:

| 작업 | 시간 |
|---|---|
| 모델 조립 + 코드 생성 | 0.3 초 |
| 정상상태 1회 (냉시동) | 0.02 초 |
| 정상상태 1회 (warm start) | 0.004 초 |
| 5분 데이터 30일치(8,640 스텝) | 약 40 초 |
| 파라미터 6개 보정 (120행 층화추출) | 약 75 초 |

---

## 다음 계통으로 확장하기

컴포넌트를 새로 만들 때 필요한 것은 `PARAMS` / `VARS` / `PORTS` 선언과
`equations()` 뿐입니다. 자세한 절차와 방정식 개수 맞추는 법은
[`docs/adding_components.md`](docs/adding_components.md) 를 보세요.

계획된 확장 순서:

1. **냉동기·냉수 플랜트** — 칠러 성능곡선, 냉각탑 Merkel/NTU, 펌프·배관망
2. **CDA 압축공기** — 컴프레서, 드라이어, 리시버, 배관망 압력 강하
3. **외조기(MAU)·공조** — 습공기 선도 기반 가열/냉각/감습/가습

`core/` 와 `lib/gas.py` 는 이미 도메인 비의존이므로 그대로 재사용됩니다.
`lib/flow.py` 의 Duct·Fan·Mixer 도 유체만 바꾸면 그대로 씁니다.

---

## 보안

* 현장 데이터는 `.gitignore` 로 커밋이 차단되어 있습니다 (`examples/**/data/*.csv` 예외 하나만 합성 데이터).
* 태그 이름은 코드가 아니라 `tagmap.yaml` 에만 존재합니다. 모델 코드에는 사내 태그가 들어가지 않습니다.
* 전 과정이 로컬에서 실행되며 외부 통신이 없습니다. HTML 리포트도 외부 리소스를 참조하지 않는 단일 파일입니다.

---

## 문서

* [`docs/architecture.md`](docs/architecture.md) — 계층 구조와 설계 판단의 근거
* [`docs/adding_components.md`](docs/adding_components.md) — 새 컴포넌트 만들기
* [`docs/nox_model.md`](docs/nox_model.md) — NOx 모델의 물리와 검증
* [`docs/generalization.md`](docs/generalization.md) — 방정식 직접 선언, 후보 비교, LLM 의 자리
* [`docs/data_driven.md`](docs/data_driven.md) — 타깃만 지정해서 어디까지 자동인가 (XAI)
