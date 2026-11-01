@echo off
REM ===========================================================================
REM  KBS Industrial Fault Diagnosis - convenience launcher for Windows
REM  Double-click this file, or run "npm run dev" from the project root.
REM
REM  This does NOT start Python directly. It delegates to the one-command
REM  launcher (scripts\dev.js) which creates/uses the virtual environment,
REM  installs missing dependencies, starts the Flask server and prints the URL.
REM ===========================================================================

setlocal
cd /d "%~dp0.."

where node >nul 2>nul
if errorlevel 1 (
    echo.
    echo   Node.js was not found on your PATH.
    echo   The one-command launcher requires Node.js:
    echo     https://nodejs.org/
    echo.
    echo   Alternative: start the app directly with
    echo     python app.py
    echo   from the submission folder, after installing Flask.
    echo.
    pause
    exit /b 1
)

call npm run dev
endlocal
