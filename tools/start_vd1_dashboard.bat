@echo off
setlocal
cd /d "%~dp0\.."

set "PYTHON=.venv\Scripts\python.exe"
set "VIDEO=data\input\videos\vd-1.mp4"
set "REPORT=reports\vd-1_dashboard_tracks_dense.json"

if not exist "%PYTHON%" (
    echo VantaSight virtual environment not found.
    echo Create it and install the project dependencies first.
    pause
    exit /b 1
)

if not exist "%VIDEO%" (
    echo %VIDEO% was not found. Add your local video before opening this review.
    pause
    exit /b 1
)

if not exist "%REPORT%" (
    echo %REPORT% was not found. Generate it with:
    echo .venv\Scripts\python.exe tools\track_video.py "data\input\videos\vd-1.mp4" --frame-step 30 --preserve-source-resolution --output reports\vd-1_dashboard_tracks_dense.csv --json-output reports\vd-1_dashboard_tracks_dense.json
    pause
    exit /b 1
)

"%PYTHON%" "tools\open_dashboard.py" "%REPORT%"

:finish
if errorlevel 1 (
    echo.
    echo Dashboard stopped with an error. Read the message above.
)
pause
