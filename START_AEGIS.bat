@echo off
setlocal EnableExtensions
title Aegis AI Launcher

pushd "%~dp0"

echo.
echo ============================================================
echo                         AEGIS AI
echo              On-device Digital Safety Copilot
echo ============================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv\Scripts\python.exe was not found.
    echo Make sure this file is inside the AegisAI project root.
    echo.
    pause
    popd
    exit /b 1
)

if not exist "backend\api.py" (
    echo [ERROR] backend\api.py was not found.
    echo.
    pause
    popd
    exit /b 1
)

if not exist "frontend\index.html" (
    echo [ERROR] frontend\index.html was not found.
    echo.
    pause
    popd
    exit /b 1
)

echo [1/3] Starting Aegis backend...
start "Aegis AI Backend" cmd /k ""%~dp0venv\Scripts\python.exe" -m uvicorn backend.api:app --host 127.0.0.1 --port 8000"

echo.
echo [2/3] Waiting for local backend...
echo Model loading may take a little while on the first launch.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ready=$false; for($i=0;$i -lt 240;$i++){ try { $r=Invoke-WebRequest -Uri 'http://127.0.0.1:8000/' -UseBasicParsing -TimeoutSec 2; if($r.StatusCode -eq 200){$ready=$true;break} } catch {}; Start-Sleep -Milliseconds 500 }; if(-not $ready){exit 1}"

if errorlevel 1 (
    echo.
    echo [ERROR] Backend did not become ready within 120 seconds.
    echo.
    echo Check the "Aegis AI Backend" window for the actual error.
    echo.
    pause
    popd
    exit /b 1
)

echo [OK] Backend is ready.
echo.
echo [3/3] Opening Aegis AI...

start "" "%~dp0frontend\index.html"

echo.
echo ============================================================
echo Aegis AI is running.
echo Keep the Aegis AI Backend window open.
echo Close it when the demo is finished.
echo ============================================================
echo.

popd
exit /b 0
