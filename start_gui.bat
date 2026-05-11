@echo off
title safetensors2GGUF
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo ERROR: uv is not installed or not on PATH.
    echo Install it from https://docs.astral.sh/uv/
    pause
    exit /b 1
)

echo Starting safetensors2GGUF Web UI...
echo The browser will open automatically.
echo Close this window to stop the server.
echo.
uv run python gui.py

if errorlevel 1 (
    echo.
    echo The application exited with an error.
    pause
)
