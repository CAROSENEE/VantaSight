@echo off
setlocal
cd /d "%~dp0\.."

set "PYTHON=.venv\Scripts\python.exe"
set "VIDEO=data\input\videos\vd-1.mp4"
set "REPORT=reports\m50_tracks.json"
set "SWEEP=reports\vd-1_provisional_threshold_sweep_fingerprint.csv"
set "AP=reports\vd-1_provisional_iou_ap_fingerprint.csv"

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
    echo %REPORT% was not found. Generate a tracking report first.
    echo See the Video tracking section in README.md.
    pause
    exit /b 1
)

if not exist "%SWEEP%" (
    echo %SWEEP% was not found. Generate the provisional threshold sweep first.
    echo See the evaluation commands in README.md.
    pause
    exit /b 1
)

if not exist "%AP%" (
    echo %AP% was not found. Generate the provisional IoU/AP report first.
    echo See the evaluation commands in README.md.
    pause
    exit /b 1
)

"%PYTHON%" "tools\open_dashboard.py" "%REPORT%" --threshold-sweep "%SWEEP%" --iou-ap "%AP%"

if errorlevel 1 (
    echo.
    echo Dashboard stopped with an error. Read the message above.
)
pause
