# 윈도우 PC 에서 쓰기 — 설치와 실행

필요한 것은 **파이썬 3.10 이상**(권장 3.11) 하나입니다. GPU, 관리자 권한, 웹 프레임워크,
외부 인터넷 요청은 필요 없습니다. 앱은 `127.0.0.1` 에만 붙고 데이터는 이 PC 밖으로
나가지 않습니다.

## 1. 파이썬 설치 (한 번)

python.org 또는 사내 소프트웨어 센터에서 **Python 3.11 (64-bit)** 을 설치합니다. 설치
첫 화면에서 **“Add python.exe to PATH”** 를 체크하세요. 관리자 권한이 없으면 “Install for
all users” 를 끄고 사용자 설치로 하면 됩니다.

## 2. 받기

| 상황 | 받을 것 |
|---|---|
| 사내 PC 가 인터넷(또는 사내 PyPI 미러)에 닿는다 | 저장소 zip (GitHub → Code → Download ZIP) 또는 `git clone` |
| 사내 PC 가 인터넷에 안 닿는다 | 인터넷 되는 PC 에서 만든 **오프라인 묶음** (아래 4번) |

아무 폴더에나 풀면 됩니다 (예: `D:\pforecast`). 경로에 한글이 있어도 됩니다.

## 3. 실행 — `run_pforecast.bat` 더블클릭

처음 한 번은 이 폴더에 가상환경(`.venv`)을 만들고 패키지를 설치합니다 (몇 분 걸립니다).
그다음부터는 바로 앱이 뜨고 브라우저가 열립니다. 검은 창을 닫으면
앱도 꺼집니다.

* 폴더 안에 `wheels\` 가 있으면 **인터넷 없이** 그 안의 파일로 설치합니다.
* `pyproject.toml` 이 바뀌면(새 버전을 받으면) 다시 설치합니다.
* 작업 폴더(CSV·모델·프로젝트가 있는 곳)를 따로 두려면: 바로가기를 만들고 대상 뒤에
  `D:\내작업폴더` 를 붙이세요 → `run_pforecast.bat D:\내작업폴더`.
* 포트 8765 를 다른 프로그램이 쓰면 파일 안의 `PORT` 값을 바꾸세요.

VS Code 에서 쓰려면 폴더를 열고 터미널에서:

```bat
.venv\Scripts\activate
pf serve                     :: 앱
pf project list              :: 앱에서 저장한 프로젝트
pf forecast <이름> --plan plans\계획.csv
```

## 4. 오프라인 묶음 만들기 (인터넷 되는 PC 에서)

```bash
python scripts/make_offline_bundle.py            # 윈도우 64비트 + 파이썬 3.11 용
python scripts/make_offline_bundle.py --python 3.12
```

`dist/pforecast_offline_win_py311.zip` 이 생깁니다. 사내 PC 에 복사해 풀고
`run_pforecast.bat` 을 더블클릭하면 됩니다.

* 소스는 **git 이 추적하는 파일만** 넣습니다. 현장 CSV·업로드·프로젝트·결과는 들어가지 않습니다.
* 의존 패키지는 **이 PC 에 깔린 버전으로 고정**해서 받습니다 — 테스트를 통과한 조합을 그대로
  옮기려고. 최신 버전을 받으려면 `--latest`.
* GPU 패키지(torch 등)는 넣지 않습니다. 필요 없습니다.
* 이 스크립트는 파이썬 3.11 이상에서 실행합니다 (묶음 대상 버전은 `--python` 으로 따로 정함).

이 저장소에서 실제로 만들어 본 묶음(`--latest`, 파이썬 3.11 · win_amd64)은 wheels 22개
(numpy, scipy, pandas, PyYAML, matplotlib, scikit-learn 과 그 의존성, pip·setuptools·wheel),
zip 90 MB 였습니다.

## 5. 폴더 구조 — 무엇이 어디에 생기나

```
pforecast\
  run_pforecast.bat        ← 더블클릭
  .venv\                   가상환경 (자동 생성)
  wheels\                  오프라인 설치 파일 (있으면)
  uploads\                 앱에서 올린 CSV
  projects\                저장한 프로젝트 (설정 + 마지막 결과)
  plans\                   만든 계획 CSV (엑셀에서 고쳐 다시 올림)
  models\                  ‘모델 만들기’로 저장한 계통 모델
  out\easy\  out\forecast\ 검증·예측 결과 CSV
```

`uploads\ projects\ plans\ models\ out\` 은 전부 `.gitignore` 로 커밋이 막혀 있습니다
(태그 이름·설계값·현장 값이 들어가므로).

## 6. 문제가 생기면

| 증상 | 할 일 |
|---|---|
| “파이썬 3.10 이상을 찾지 못했습니다” | 1번 다시. 설치 후 **새 창**에서 실행 |
| 설치 실패 (인터넷) | 사내 프록시가 필요하면 창에서 `set HTTPS_PROXY=http://프록시:포트` 후 bat 실행. 안 되면 4번 오프라인 묶음 |
| 설치 실패 (오프라인) | 묶음을 만든 파이썬 버전과 사내 PC 파이썬 버전(`py -3 --version`)이 같은지 |
| 브라우저가 안 열림 | 직접 `http://127.0.0.1:8765` |
| 한글이 깨져 보이는 CSV | 앱이 CP949·UTF-8 을 알아서 읽습니다. 그래도 이상하면 ‘데이터 정리 내역’을 보세요 |
| 설치를 처음부터 다시 | `.venv` 폴더를 지우고 bat 다시 실행 |

`run_pforecast.bat` 은 개발 환경(리눅스)에서 만들었고 **윈도우에서 직접 실행해 보지는
못했습니다.** 처음 실행에서 막히면 창에 뜬 메시지를 그대로 알려주세요. 오프라인 묶음 스크립트와
그 안에 들어가는 윈도우용 wheel 은 위처럼 실제로 받아 확인했습니다.
