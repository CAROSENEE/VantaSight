@echo off
setlocal
cd /d "%~dp0\.."

set "PYTHON=.venv\Scripts\python.exe"
set "VIDEO=data\input\videos\vd-1.mp4"
set "REPORT=reports\vd-1_dashboard_tracks.json"
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
    echo %REPORT% was not found. Generate it with:
    echo .venv\Scripts\python.exe tools\track_video.py "data\input\videos\vd-1.mp4" --frame-step 120 --output reports\vd-1_dashboard_tracks.csv --json-output reports\vd-1_dashboard_tracks.json
    pause
    exit /b 1
)

if exist "%SWEEP%" if exist "%AP%" goto with_both_evaluations
if exist "%SWEEP%" goto with_threshold_sweep
if exist "%AP%" goto with_iou_ap
goto report_only

:with_both_evaluations
"%PYTHON%" "tools\open_dashboard.py" "%REPORT%" --threshold-sweep "%SWEEP%" --iou-ap "%AP%"
goto finish

:with_threshold_sweep
"%PYTHON%" "tools\open_dashboard.py" "%REPORT%" --threshold-sweep "%SWEEP%"
goto finish

:with_iou_ap
"%PYTHON%" "tools\open_dashboard.py" "%REPORT%" --iou-ap "%AP%"
goto finish

:report_only
"%PYTHON%" "tools\open_dashboard.py" "%REPORT%"

:finish
if errorlevel 1 (
    echo.
    echo Dashboard stopped with an error. Read the message above.
)
pause
