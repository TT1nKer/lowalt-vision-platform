@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
set "CONSOLE_URL=http://127.0.0.1:7860"

if not exist "%PYTHON%" (
    echo [ERROR] Virtual environment Python was not found:
    echo         %PYTHON%
    echo.
    echo Create it with: py -3 -m venv .venv
    echo Then install:   .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

REM Do not launch a second server when the console is already online.
powershell.exe -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri '%CONSOLE_URL%' -TimeoutSec 2; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { exit 0 } } catch {}; exit 1" >nul 2>&1
if not errorlevel 1 (
    echo LOWALT Console is already running. Opening %CONSOLE_URL%
    start "" "%CONSOLE_URL%"
    exit /b 0
)

echo Starting LOWALT Console at %CONSOLE_URL%
echo Close this window to stop the service.
echo.
"%PYTHON%" -u "%~dp0app.py" --project-dir "%~dp0" --config "%~dp0config.yaml" --host 127.0.0.1 --port 7860 %*

if errorlevel 1 (
    echo.
    echo [ERROR] Console startup failed. Review the message above.
    pause
)
endlocal
