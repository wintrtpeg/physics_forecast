@echo off
chcp 65001 >nul
rem pforecast 실행 — 이 파일을 더블클릭하세요.
rem  1) 처음 한 번: 이 폴더에 가상환경(.venv)을 만들고 패키지를 설치합니다.
rem     wheels\ 폴더가 있으면 인터넷 없이(오프라인) 설치합니다.
rem  2) 앱을 띄우고 브라우저를 엽니다. 창을 닫으면 앱도 꺼집니다.
rem  작업 폴더를 바꾸려면: run_pforecast.bat D:\내작업폴더
setlocal
cd /d "%~dp0"

rem 파이썬이 파일·콘솔을 UTF-8 로 다루게 한다 (한국어 윈도우 기본은 CP949)
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "WORK=%~1"
rem 끝의 \ 가 따옴표를 삼키지 않게 "." 을 붙인다 ("C:\pf\" 는 인자 파싱에서 C:\pf" 가 된다)
if "%WORK%"=="" set "WORK=%~dp0."
set "PORT=8765"
if defined PF_PORT set "PORT=%PF_PORT%"
rem PF_NO_BROWSER=1 이면 브라우저를 열지 않는다, PF_NO_PAUSE=1 이면 오류에서 멈추지 않는다 (자동 점검용)
set "OPENB="
if defined PF_NO_BROWSER set "OPENB=--no-browser"
set "PAUSE=pause"
rem (rem 으로 끄면 한 줄 블록의 뒷부분까지 주석이 된다 — 아무 일도 안 하는 명령으로)
if defined PF_NO_PAUSE set "PAUSE=ver>nul"

rem ── 파이썬 찾기 (3.10 이상) ─────────────────────────────────────────
set "PY="
where py >nul 2>nul && (py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=py -3")
if not defined PY (
  where python >nul 2>nul && (python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=python")
)
if not defined PY (
  echo [오류] 파이썬 3.10 이상을 찾지 못했습니다.
  echo        python.org 또는 사내 소프트웨어 센터에서 Python 3.11 을 설치하세요.
  echo        설치할 때 "Add python.exe to PATH" 를 체크하세요.
  %PAUSE%
  exit /b 1
)

rem ── 가상환경 + 설치 (처음 한 번, 또는 pyproject.toml 이 바뀌었을 때) ────────
if not exist ".venv\Scripts\python.exe" (
  echo 가상환경을 만드는 중...
  %PY% -m venv .venv || (echo [오류] 가상환경을 만들지 못했습니다. & %PAUSE% & exit /b 1)
)
set "VPY=.venv\Scripts\python.exe"
set "STAMP=.venv\pforecast.installed"
set "NEED=1"
if exist "%STAMP%" (
  fc /b pyproject.toml "%STAMP%" >nul 2>nul && set "NEED=0"
)
if "%NEED%"=="1" (
  if exist "wheels\" (
    echo 오프라인 설치 중 ^(wheels 폴더^)...
    "%VPY%" -m pip install --no-index --find-links wheels --upgrade pip >nul 2>nul
    "%VPY%" -m pip install --no-index --find-links wheels -e ".[plot,ml]" || goto :installfail
  ) else (
    echo 인터넷^(또는 사내 미러^)에서 설치 중... 처음 한 번만 몇 분 걸립니다
    "%VPY%" -m pip install --upgrade pip >nul 2>nul
    "%VPY%" -m pip install -e ".[plot,ml]" || goto :installfail
  )
  copy /y pyproject.toml "%STAMP%" >nul
)

rem ── 실행 ────────────────────────────────────────────────────────────
echo.
echo pforecast 를 띄웁니다: http://127.0.0.1:%PORT%
echo 작업 폴더: %WORK%
echo 끝내려면 이 창을 닫거나 Ctrl+C 를 누르세요.
echo.
"%VPY%" -m pforecast.cli serve --root "%WORK%" --port %PORT% %OPENB%
if errorlevel 1 (
  echo.
  echo [오류] 앱이 비정상 종료했습니다. 위 메시지를 확인하세요.
  echo        포트 %PORT% 를 다른 프로그램이 쓰고 있으면 이 파일의 PORT 값을 바꾸세요.
  %PAUSE%
)
exit /b 0

:installfail
echo.
echo [오류] 설치하지 못했습니다.
echo  - 사내망이면: 인터넷 되는 PC 에서 "python scripts\make_offline_bundle.py" 로 묶음을 만들어
echo    wheels 폴더째 이 폴더에 복사한 뒤 다시 실행하세요.
echo  - 프록시가 필요하면: set HTTPS_PROXY=http://프록시:포트 후 다시 실행하세요.
del /q "%STAMP%" >nul 2>nul
%PAUSE%
exit /b 1
