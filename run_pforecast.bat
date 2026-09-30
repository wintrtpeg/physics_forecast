@echo off
rem pforecast launcher - double-click this file.
rem   First run: creates .venv in this folder and installs packages
rem   (offline if a wheels\ folder is here). Then starts the app and opens the browser.
rem   Close this window (or Ctrl+C) to stop the app.
rem   Optional work folder:  run_pforecast.bat D:\my_work
rem
rem   This file is ASCII on purpose. cmd.exe mis-reads UTF-8 (Korean) batch files after
rem   "chcp 65001" and stops silently once an external program returns. All the work and
rem   the Korean messages are in scripts\launch.py.
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

rem Find a real Python (the Microsoft Store alias "python.exe" fails the check below).
set "PY=py -3"
py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 goto run
set "PY=python"
python -c "import sys" >nul 2>nul
if not errorlevel 1 goto run

echo.
echo [ERROR] Python was not found.
echo   Install Python 3.11 64-bit from python.org or the company software center,
echo   and check "Add python.exe to PATH" on the first installer screen.
echo   Korean guide: docs\windows.md
goto fail

:run
%PY% "%~dp0scripts\launch.py" %*
if errorlevel 1 goto fail
exit /b 0

:fail
if not defined PF_NO_PAUSE pause
exit /b 1
