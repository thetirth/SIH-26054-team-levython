@echo off
REM =====================================================================
REM  stop.bat -- shut down the MALE UAV Digital Twin stack (:8000 + :3000)
REM =====================================================================
setlocal

call :killport 8000
call :killport 3000
echo [OK] Stack stopped.
pause
exit /b 0

:killport
set "PORT=%~1"
set "PID="
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do set "PID=%%a"
if defined PID (
  echo [..] Stopping PID %PID% on port %PORT% ...
  taskkill /PID %PID% /F >nul 2>nul
) else (
  echo [..] Nothing listening on port %PORT%.
)
exit /b 0
