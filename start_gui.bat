@echo off
title safetensors2GGUF Workbench
cd /d "%~dp0"

rem Reuse a running Workbench; do not start a second server on its port.
powershell -NoProfile -Command "try { $config = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/config' -TimeoutSec 2; if ($config.formats.gguf -and $config.text_encoder_formats -and $config.platform) { exit 0 } } catch {}; exit 1"
if not errorlevel 1 (
    echo Opening the already running Workbench.
    start "" "http://127.0.0.1:8765"
    exit /b 0
)

where uv >nul 2>&1
if errorlevel 1 (
    echo ERROR: uv is not installed or not on PATH.
    echo Install it from https://docs.astral.sh/uv/
    pause
    exit /b 1
)

if not exist "frontend\dist\index.html" (
    where npm >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Install Node.js to build the frontend on first launch.
        pause
        exit /b 1
    )
    pushd frontend
    call npm ci
    if errorlevel 1 goto build_failed
    call npm run build
    if errorlevel 1 goto build_failed
    popd
)

echo Starting safetensors2GGUF Workbench at http://127.0.0.1:8765
echo The browser will open automatically.
echo Close this window to stop the server.
echo.

start "" "http://127.0.0.1:8765"
uv run python web_api.py

if errorlevel 1 (
    echo.
    echo The application exited with an error.
    pause
)
exit /b

:build_failed
popd
echo ERROR: Frontend build failed. See the output above.
pause
exit /b 1
