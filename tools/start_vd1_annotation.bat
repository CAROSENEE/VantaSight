@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
    echo VantaSight virtual environment not found.
    echo Create it and install the project dependencies first.
    pause
    exit /b 1
)

if not exist "data\input\videos\vd-1.mp4" (
    echo data\input\videos\vd-1.mp4 was not found.
    echo Put the local video in that folder, then run this launcher again.
    pause
    exit /b 1
)

if not exist "reports\vd-1_manual_review_shortlist.csv" (
    if not exist "reports\vd-1_dashboard_tracks_dense.json" (
        echo reports\vd-1_dashboard_tracks_dense.json was not found. Generate the dense tracking report first.
        pause
        exit /b 1
    )
    echo reports\vd-1_manual_review_shortlist.csv was not found. Generate it from the dense report:
    echo .venv\Scripts\python.exe tools\create_review_shortlist.py reports\vd-1_dashboard_tracks_dense.json --output reports\vd-1_manual_review_shortlist.csv
    pause
    exit /b 1
)

echo Starting a blank manual review. No provisional boxes will be loaded.
echo Look for a window titled "VantaSight annotation - vd-1.mp4 frame ...".
echo If the window does not appear, keep this console open and read any error below.
".venv\Scripts\python.exe" "tools\annotate_video_frames.py" "data\input\videos\vd-1.mp4" --shortlist-csv "reports\vd-1_manual_review_shortlist.csv" --output "data\annotations\vd-1_user_verified.jsonl" --keep-on-top

if errorlevel 1 (
    echo.
    echo Annotation stopped with an error. Read the message above.
)
pause
