@echo off
REM =====================================================================
REM  start.bat -- one-click launcher for the MALE UAV Digital Twin stack
REM  - FastAPI physics + ML backend  -> http://127.0.0.1:8000 (and /docs)
REM  - Next.js GCS frontend           -> http://localhost:3000
REM  Double-click this file. Run stop.bat to shut everything down.
REM =====================================================================
setlocal
pushd "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] python not found on PATH. Install Python 3.9+ first.
  pause
  exit /b 1
)

REM ---- Backend :8000 ----
netstat -ano | findstr /R /C:":8000 .*LISTENING" >nul 2>nul
if not errorlevel 1 (
  echo [OK] Backend already listening on http://127.0.0.1:8000
) else (
  echo [..] Starting FastAPI backend on http://127.0.0.1:8000 ...
  start "UAV-Backend :8000" python -m uvicorn api:app --host 127.0.0.1 --port 8000
  echo [..] Waiting for backend /health ...
  python wait_for_backend.py
)

REM ---- Frontend :3000 ----
netstat -ano | findstr /R /C:":3000 .*LISTENING" >nul 2>nul
if not errorlevel 1 (
  echo [OK] Frontend already listening on http://localhost:3000
) else (
  echo [..] Starting Next.js GCS frontend on http://localhost:3000 ...
  pushd "gcs-app"
  start "UAV-GCS :3000" npm.cmd run dev -- --port 3000
  popd
  echo [..] Giving Next.js ~25 s to compile ...
  timeout /t 25 /nobreak >nul
)

echo.
echo =====================================================================
echo  Backend  : http://127.0.0.1:8000  ^(docs: /docs, stream: /stream^)
echo  Frontend : http://localhost:3000
echo  Streamlit: http://localhost:8501  ^(if running: streamlit run dashboard.py^)
echo  Run stop.bat to shut the stack down.
echo =====================================================================
pause
