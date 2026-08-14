@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PLATFORM_PYTHON=%~dp0.venv\Scripts\python.exe"
set "PLATFORM_URL=http://127.0.0.1:7861"

if not exist "%PLATFORM_PYTHON%" (
    echo [ERROR] Virtual environment Python was not found:
    echo         %PLATFORM_PYTHON%
    echo.
    echo Create it with: py -3 -m venv .venv
    echo Then install:   .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

powershell.exe -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri '%PLATFORM_URL%' -TimeoutSec 2; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { exit 0 } } catch {}; exit 1" >nul 2>&1
if not errorlevel 1 (
    echo Low-altitude platform is already running. Opening %PLATFORM_URL%
    start "" "%PLATFORM_URL%"
    exit /b 0
)

echo Starting low-altitude platform at %PLATFORM_URL%
echo Close this window to stop the service.
echo.
"%PLATFORM_PYTHON%" -u "%~dp0platform_app.py" --project-dir "%~dp0" --host 127.0.0.1 --port 7861 %*

if errorlevel 1 (
    echo.
    echo [ERROR] Platform startup failed. Review the message above.
    pause
)
endlocal
